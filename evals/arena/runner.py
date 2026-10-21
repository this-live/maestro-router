"""`arena.py run`: every (suite, task, seat, repeat) cell, then the open-suite ranking.

Files in evals/arena/runs/<run_id>/ (gitignored; nothing leaves this machine except each seat's own call to its provider):
    run.json          seats, judges, arguments, the harness argv notes
    tasks.jsonl       the tasks (job, difficulty, request text for the routing replay)
    results.jsonl     one row per attempt: {suite, task, seat, rep, status: ok|error|skipped, y, cost_usd, secs, ...}
    criteria.jsonl    open suite: the criteria written before any answer was seen
    judgments.jsonl   open suite: {task, rep, judge, pass, order, ranking, accepted, ...}
    diffs/            agentic: the patch of every cell

Resume: a rerun with the same --run-id skips cells that have an `ok` row, or `--max-attempts` error rows; `skipped`
rows (quota) do not count. Budget: the run's estimated spend so far (every row's cost_usd, earlier sessions included) is
compared with --budget-usd before each call starts; nothing new starts once it is reached.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import re
import shutil
import threading
import time

from . import rank
from . import seats as S
from .common import RUNS, SUITES, append_row, est_tokens, ek, globs, m, matches, read_json, read_jsonl, write_json
from .harness import EXE, HARNESSES, ask, cost_of, preflight
from .suites import grade_verifiable, load_agentic, load_open, load_verifiable, run_agentic

CHAT_OUT_TOKENS = 700              # dry-run estimates only
AGENTIC_TOKENS = (60000, 5000)


class Budget:
    def __init__(self, cap, spent=0.0):
        self.cap, self.spent, self.lock = cap, spent, threading.Lock()

    def add(self, usd):
        with self.lock:
            self.spent += usd or 0.0

    def reached(self):
        with self.lock:
            return self.cap is not None and self.spent >= self.cap


def run_paths(run_id, runs_dir=None):
    d = os.path.join(runs_dir or RUNS, run_id)
    return {"dir": d, "run": os.path.join(d, "run.json"), "tasks": os.path.join(d, "tasks.jsonl"),
            "results": os.path.join(d, "results.jsonl"), "criteria": os.path.join(d, "criteria.jsonl"),
            "judgments": os.path.join(d, "judgments.jsonl"), "diffs": os.path.join(d, "diffs")}


def cell_key(r):
    return (r["suite"], r["task"], r["seat"], int(r["rep"]))


def finished(rows, max_attempts):
    """Cell keys that need no more work: an ok row, or max_attempts error rows."""
    ok, errs = set(), {}
    for r in rows:
        k = cell_key(r)
        if r.get("status") == "ok":
            ok.add(k)
        elif r.get("status") == "error":
            errs[k] = errs.get(k, 0) + 1
    return ok | {k for k, n in errs.items() if n >= max_attempts}


def spent(paths):
    return sum(float(r.get("cost_usd") or 0.0) for p in ("results", "criteria", "judgments") for r in read_jsonl(paths[p]))


def load_tasks(a):
    suites = SUITES if a.suite == "all" else (a.suite,)
    out = {}
    if "verifiable" in suites:
        if a.items:
            out["verifiable"] = load_verifiable(a.items)
        elif a.suite == "verifiable":
            raise ValueError("--suite verifiable needs --items <file.jsonl>")
    if "agentic" in suites:
        fx = load_agentic(a.fixtures)
        if fx:
            out["agentic"] = fx
        elif a.suite == "agentic":
            raise ValueError("no fixtures under %s" % a.fixtures)
    if "open" in suites:
        if a.open_items:
            out["open"] = load_open(a.open_items)
        elif a.suite == "open":
            raise ValueError("--suite open needs --open-items <file.jsonl>")
    only = globs(a.only)
    for k in list(out):
        ts = [t for t in out[k] if not only or matches(t["id"], only)]
        if a.shuffle is not None:
            ts.sort(key=lambda t: ek.unit("arena-tasks", a.shuffle, t["id"]))
        out[k] = ts[:a.tasks] if a.tasks else ts
        if not out[k]:
            del out[k]
    if not out:
        raise ValueError("no tasks selected")
    return out


def estimate(seat, suite, task):
    if seat.get("price_in") is None:
        return None
    tin, tout = AGENTIC_TOKENS if suite == "agentic" else (est_tokens(task["prompt"]), CHAT_OUT_TOKENS)
    return (tin * seat["price_in"] + tout * (seat.get("price_out") or 0.0)) / 1e6


def quota_gate(seat, home, min_headroom):
    """(ok, reason) from Maestro's quota state (local files; no network)."""
    from maestro import quota
    scope = S.quota_scope(seat)
    if not scope:
        return True, ""
    model = seat.get("model") if seat["harness"] == "api" else None
    ok, why = quota.is_available(scope, model, home)
    if ok:
        hr = quota.headroom(scope, model, home)
        if hr is not None and hr < min_headroom:
            ok, why = False, "%s has %.0f%% left (under --min-headroom %.0f%%)" % (scope, hr * 100, min_headroom * 100)
    return ok, why


def wait_quota(seat, home, a):
    ok, why = quota_gate(seat, home, a.min_headroom)
    deadline = time.time() + (a.quota_wait or 0)
    while not ok and time.time() < deadline:
        time.sleep(min(30.0, max(0.1, deadline - time.time())))
        try:
            from maestro import quota
            quota.refresh_subscriptions(home)
        except Exception:
            pass
        ok, why = quota_gate(seat, home, a.min_headroom)
    return ok, why


def _safe(s):
    return re.sub(r"[^\w.@-]+", "_", s)[:120]


def execute(jobs, fn, conc, budget, key=None):
    """Run fn(job) for each job with at most `conc` at once; nothing new starts once the budget is reached.
    Jobs with the same key(job) never run at the same time (agentic cells on one fixture: a change to the fixture is
    then blamed on the one cell that ran, not on every cell that overlapped it); otherwise jobs start in order.
    Returns the number of jobs not started."""
    key = key or (lambda job: None)
    pending, running = list(jobs), {}
    with cf.ThreadPoolExecutor(max_workers=max(1, conc)) as ex:
        while pending:
            busy = {k for k in running.values() if k is not None}
            nxt = next((j for j in pending if key(j) is None or key(j) not in busy), None) \
                if len(running) < max(1, conc) else None
            if nxt is None:
                done, _ = cf.wait(set(running), return_when=cf.FIRST_COMPLETED)
                for f in done:
                    running.pop(f)
                continue
            if budget.reached():
                break
            pending.remove(nxt)
            running[ex.submit(fn, nxt)] = key(nxt)
        cf.wait(set(running))
    return len(pending)


def run(a, log=print):
    env = os.environ
    home = m.maestro_home(env)
    cfg = m.load_config(home)
    patterns = globs(a.seats)
    if not patterns and not a.dry_run:
        raise ValueError("--seats is required (globs, e.g. 'claude-code:haiku,local:*'); `arena.py seats` lists them")
    inv = m.ensure_inventory(home, env, cfg=cfg)
    every = S.discover(env, home, cfg, inv, seats_file=a.seats_file)
    for s in every:
        S.annotate(s)
    exes = {h: shutil.which(EXE[h], path=env.get("PATH")) for h in HARNESSES}
    every = [s for s in every if s["harness"] not in HARNESSES or exes[s["harness"]]]   # a CLI that is not installed cannot play
    chosen = S.select(every, patterns)
    if not chosen:
        raise ValueError("no runnable seat matches %s (`arena.py seats` lists them; a seat whose CLI is not on PATH is "
                         "left out)" % ", ".join(patterns))
    tasks = load_tasks(a)
    paths = run_paths(a.run_id, a.runs_dir)
    old = read_json(paths["run"]) or {}
    judges, note = [], ""
    if "open" in tasks:
        jp = globs(a.judges) or ([] if old.get("judges") else S.file_judges(a.seats_file))
        if jp:
            judges = S.select(every, jp)
            fams = {j.get("family") for j in judges}
            note = "" if len(fams) >= 2 else ("SINGLE-FAMILY JUDGING: every judge is %s; rankings may carry that family's "
                                              "self-preference." % (next(iter(fams)) or "unknown"))
        elif old.get("judges"):
            judges = [s for s in every if s["id"] in old["judges"]] or [old["seats"][j] for j in old["judges"] if j in old.get("seats", {})]
            note = old.get("judge_note", "")
        else:
            judges, note = S.default_judges(every)
        if not judges:
            raise ValueError("the open suite needs at least one judge seat")
    cells = []
    for rep in range(1, a.repeat + 1):
        for suite, ts in tasks.items():
            for t in ts:
                for s in chosen:
                    if suite != "agentic" or s["mode"] == "agentic":
                        cells.append((suite, t, s, rep))
    done = finished(read_jsonl(paths["results"]), a.max_attempts)
    todo = [c for c in cells if (c[0], c[1]["id"], c[2]["id"], c[3]) not in done]
    if a.dry_run:
        return dry_run(a, tasks, chosen, judges, note, cells, todo, log)
    os.makedirs(paths["diffs"], exist_ok=True)
    seats_all = dict(old.get("seats") or {}, **{s["id"]: s for s in chosen + judges})
    write_json(paths["run"], {"run_id": a.run_id, "created": old.get("created") or time.time(), "updated": time.time(),
                              "suites": sorted(set(old.get("suites") or []) | set(tasks)), "seats": seats_all,
                              "contestants": sorted(set(old.get("contestants") or []) | {s["id"] for s in chosen}),
                              "judges": [j["id"] for j in judges] or old.get("judges") or [], "judge_note": note,
                              "args": {k: v for k, v in vars(a).items() if k not in ("fn",)},
                              "harness_verified": {h: v["verified"] for h, v in HARNESSES.items()}})
    known = {(r["suite"], r["id"]): r for r in read_jsonl(paths["tasks"])}
    for suite, ts in tasks.items():
        for t in ts:
            known[(suite, t["id"])] = {"suite": suite, "id": t["id"], "job": t.get("job"), "difficulty": t.get("difficulty"),
                                       "text": t["text"], "tin": est_tokens(t["prompt"]), "lang": t.get("lang") or "en",
                                       "words": len(re.findall(r"\w+", t["text"]))}
    ek.write_jsonl(paths["tasks"], list(known.values()))
    try:
        from maestro import quota
        quota.refresh_subscriptions(home)
    except Exception:
        pass
    budget = Budget(a.budget_usd, spent(paths))
    counter = {"n": 0}
    lock = threading.Lock()
    log("arena run %s: %d cells (%d to do), %d seats, %s; budget %s, spent so far $%.4f" % (
        a.run_id, len(cells), len(todo), len(chosen), ", ".join("%s %d" % (k, len(v)) for k, v in tasks.items()),
        "none" if a.budget_usd is None else "$%.2f" % a.budget_usd, budget.spent))

    def one(cell):
        suite, t, s, rep = cell
        base = {"suite": suite, "task": t["id"], "seat": s["id"], "rep": rep, "ts": round(time.time(), 1)}
        ok, why = wait_quota(s, home, a)
        if not ok:
            append_row(paths["results"], dict(base, status="skipped", reason=why))
            return log_cell(base, "skipped (%s)" % why)
        try:
            if suite == "agentic":
                if not exes.get(s["harness"]):
                    raise RuntimeError("`%s` is not on PATH" % EXE.get(s["harness"], s["harness"]))
                why = preflight(s, "agentic", exes[s["harness"]])
                if why:   # the CLI's own sandbox cannot start here: not the model's failure, redone on resume
                    append_row(paths["results"], dict(base, status="skipped", reason=why))
                    return log_cell(base, "skipped (%s)" % why)
                row, patch = run_agentic(t, s, exes[s["harness"]], a.agentic_timeout, env)
                cost, src = cost_of(s, row.get("usage"), row.pop("reported_cost", None), t["prompt"], row.get("final") or "")
                if patch:
                    with open(os.path.join(paths["diffs"], "%s__%s__r%d.patch" % (_safe(t["id"]), _safe(s["id"]), rep)), "w",
                              encoding="utf-8") as f:
                        f.write(patch)
                row.update(cost_usd=cost, cost_source=src)
            else:
                res = ask(s, t["prompt"], a.timeout, env, home, cfg, inv)
                row = {"status": "ok" if res["ok"] else "error", "answer": res["text"], "chars": len(res["text"]),
                       "secs": res["secs"], "usage": res["usage"], "cost_usd": res["cost_usd"],
                       "cost_source": res["cost_source"], "error": res["error"]}
                if res.get("extra"):
                    row["extra"] = res["extra"]
                if suite == "verifiable" and res["ok"]:
                    row.update(grade_verifiable(t["checker_py"], res["text"]))
        except Exception as e:  # one broken cell never stops the run
            from .harness import m_scrub
            row = {"status": "error", "error": m_scrub("%s: %s" % (e.__class__.__name__, e)), "cost_usd": None}
        row.update(base)
        append_row(paths["results"], row)
        budget.add(row.get("cost_usd"))
        verdict = row.get("error") if row["status"] != "ok" else ("pass" if row.get("y") else "fail") if "y" in row else "answered"
        log_cell(base, "%s  %.1fs  %s" % (verdict, row.get("secs") or 0.0, "$%.4f" % row["cost_usd"] if row.get("cost_usd") is not None else "$?"))

    def log_cell(base, text):
        with lock:
            counter["n"] += 1
            log("[%d/%d] %s %s %s r%d: %s" % (counter["n"], len(todo), base["suite"], base["task"], base["seat"], base["rep"], text))

    left = execute(todo, one, a.concurrency, budget, key=lambda c: c[1]["id"] if c[0] == "agentic" else None)
    if left:
        log("budget reached ($%.4f of $%.2f): %d cells not started; rerun with the same --run-id and a higher --budget-usd "
            "to continue" % (budget.spent, a.budget_usd, left))
    if "open" in tasks and not left:
        left = rank_open(a, paths, tasks["open"], judges, budget, env, home, cfg, inv, log)
    log("spent (estimated) $%.4f. Report: python3 evals/arena.py report %s" % (budget.spent, a.run_id))
    return 0 if not left else 3


def latest_ok(rows, suite):
    out = {}
    for r in rows:
        if r.get("suite") == suite and r.get("status") == "ok":
            out[cell_key(r)] = r
    return out


def rank_open(a, paths, tasks, judges, budget, env, home, cfg, inv, log=print):
    """Criteria (lead judge, request only), then every judge twice per (task, repeat), order reversed the second time."""
    answers = {}
    for (_, t, s, rep), r in latest_ok(read_jsonl(paths["results"]), "open").items():
        if r.get("answer"):
            answers.setdefault((t, rep), {})[s] = r["answer"]
    by_id = {t["id"]: t for t in tasks}
    lead = judges[0]
    crit = {r["task"]: r for r in read_jsonl(paths["criteria"]) if r.get("ok")}
    tried = {}
    for r in read_jsonl(paths["criteria"]):
        tried[r["task"]] = tried.get(r["task"], 0) + 1
    need = sorted({t for (t, _), v in answers.items() if len(v) >= 2 and t in by_id and t not in crit
                   and tried.get(t, 0) < a.max_attempts})
    lock = threading.Lock()

    def criteria(tid):
        res = ask(lead, rank.criteria_prompt(by_id[tid]), a.timeout, env, home, cfg, inv)
        row = {"task": tid, "judge": lead["id"], "ok": False, "cost_usd": res["cost_usd"], "secs": res["secs"]}
        try:
            if not res["ok"]:
                raise ValueError(res["error"])
            row.update(rank.parse_criteria(res["text"]), ok=True)
        except ValueError as e:
            row["error"] = str(e)[:300]
        append_row(paths["criteria"], row)
        budget.add(row["cost_usd"])
        with lock:
            if row["ok"]:
                crit[tid] = row
        log("criteria %s by %s: %s" % (tid, lead["id"], "ok" if row["ok"] else row.get("error")))

    left = execute(need, criteria, a.concurrency, budget)
    if left:
        return left
    prev = read_jsonl(paths["judgments"])
    done = {(r["task"], r["rep"], r["judge"], r["pass"], tuple(sorted(r["order"]))) for r in prev if r.get("ok")}
    fails = {}
    for r in prev:
        if not r.get("ok"):
            k = (r["task"], r["rep"], r["judge"], r["pass"], tuple(sorted(r["order"])))
            fails[k] = fails.get(k, 0) + 1
    jobs = []
    for (tid, rep), ans in sorted(answers.items()):
        if tid not in crit or len(ans) < 2:
            continue
        for j in judges:
            for p, order in enumerate(rank.orders(tid, rep, j["id"], sorted(ans), a.seed)):
                k = (tid, rep, j["id"], p, tuple(sorted(order)))
                if k not in done and fails.get(k, 0) < a.max_attempts:
                    jobs.append((tid, rep, j, p, order))

    def judge(job):
        tid, rep, j, p, order = job
        ok, why = wait_quota(j, home, a)
        if not ok:
            log("judge %s skipped on %s: %s" % (j["id"], tid, why))
            return
        c = crit[tid]
        res = ask(j, rank.rank_prompt(by_id[tid], c, [answers[(tid, rep)][s] for s in order]), a.timeout, env, home, cfg, inv)
        row = {"task": tid, "rep": rep, "judge": j["id"], "judge_family": j.get("family"), "pass": p, "order": order,
               "ok": False, "cost_usd": res["cost_usd"], "secs": res["secs"]}
        try:
            if not res["ok"]:
                raise ValueError(res["error"])
            ranking, acc = rank.parse_judgment(res["text"], order, c)
            row.update(ranking=ranking, accepted=acc, ok=True)
        except ValueError as e:
            row["error"] = str(e)[:300]
        append_row(paths["judgments"], row)
        budget.add(row["cost_usd"])
        log("judge %s %s r%d pass %d: %s" % (j["id"], tid, rep, p, "ok" if row["ok"] else row.get("error")))

    left = execute(jobs, judge, a.concurrency, budget)
    if left:
        log("budget reached during judging: %d judge calls not started" % left)
    return left


def dry_run(a, tasks, chosen, judges, note, cells, todo, log):
    log("DRY RUN: nothing is sent anywhere.\n")
    log("seats (%d): %s" % (len(chosen), ", ".join(s["id"] for s in chosen)))
    for suite, ts in tasks.items():
        log("%s: %d tasks" % (suite, len(ts)))
    log("cells: %d (%d already done in run %s)" % (len(cells), len(cells) - len(todo), a.run_id))
    per, unknown = {}, set()
    for suite, t, s, rep in todo:
        e = estimate(s, suite, t)
        if e is None:
            unknown.add(s["id"])
        per[s["id"]] = per.get(s["id"], 0.0) + (e or 0.0)
    for sid in sorted(per):
        log("  %-40s ~$%.4f%s" % (sid, per[sid], "  (no catalog price)" if sid in unknown else ""))
    if judges:
        n_items = len(tasks.get("open", [])) * a.repeat
        log("judges: %s%s" % (", ".join("%s (%s)" % (j["id"], j.get("family")) for j in judges), "\n  " + note if note else ""))
        log("judge calls: %d criteria + %d rankings" % (len(tasks.get("open", [])), n_items * len(judges) * 2))
    log("\nestimated total for the cells: ~$%.2f (rough: chat %d output tokens per answer, agentic %d+%d tokens per run; "
        "subscription CLIs bill your plan's quota, not dollars)" % (sum(per.values()), CHAT_OUT_TOKENS, *AGENTIC_TOKENS))
    return 0
