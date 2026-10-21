"""`arena.py report <run_id> [--html]`: leaderboards, value frontier, routing replay, judging checks.

Writes report.md (and report.html: one self-contained file, inline CSS, no scripts, no external requests) to the run dir.
"""
from __future__ import annotations

import html
import os
import tempfile
import time

from . import rank
from .common import ci_text, fmt, m, mean, pct, rate_ci, read_json, read_jsonl, usd
from .runner import latest_ok, run_paths


# ── a tiny document model rendered to markdown and HTML ───────────────────────

class Doc:
    def __init__(self):
        self.blocks = []

    def h(self, level, text):
        self.blocks.append(("h", level, text))

    def p(self, text, warn=False):
        self.blocks.append(("p", text, warn))

    def table(self, head, rows):
        if rows:
            self.blocks.append(("t", head, rows))

    def md(self):
        out = []
        for b in self.blocks:
            if b[0] == "h":
                out.append("#" * b[1] + " " + b[2])
            elif b[0] == "p":
                out.append(("**%s**" % b[1]) if b[2] else b[1])
            else:
                cells = lambda r: "| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |"  # noqa: E731
                out.append("\n".join([cells(b[1]), "|" + "---|" * len(b[1])] + [cells(r) for r in b[2]]))
        return "\n\n".join(out) + "\n"

    def html(self, title):
        e = lambda s: html.escape(str(s))  # noqa: E731
        body = []
        for b in self.blocks:
            if b[0] == "h":
                body.append("<h%d>%s</h%d>" % (b[1], e(b[2]), b[1]))
            elif b[0] == "p":
                body.append('<p class="%s">%s</p>' % ("warn" if b[2] else "", e(b[1])))
            else:
                body.append("<div class=\"wrap\"><table><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>" % (
                    "".join("<th>%s</th>" % e(c) for c in b[1]),
                    "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % e(c) for c in r) for r in b[2])))
        css = (":root{--bg:#fff;--fg:#1a1d24;--mut:#5b6270;--line:#e3e6eb;--head:#f4f6f9;--warn:#fff4e5;--warnfg:#8a4b00}"
               "@media (prefers-color-scheme:dark){:root{--bg:#0f1218;--fg:#e6e9ef;--mut:#9aa3b2;--line:#2a2f3a;--head:#171b23;"
               "--warn:#3a2a10;--warnfg:#ffcf8a}}body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"
               "Segoe UI,sans-serif;max-width:1100px;margin:0 auto;padding:24px 16px}h1{font-size:24px}h2{font-size:19px;"
               "margin-top:32px;border-bottom:1px solid var(--line);padding-bottom:4px}h3{font-size:16px;margin-top:22px}"
               "p{color:var(--fg)}p.warn{background:var(--warn);color:var(--warnfg);padding:8px 12px;border-radius:6px;"
               "font-weight:600}.wrap{overflow-x:auto}table{border-collapse:collapse;font-size:13px;margin:8px 0}"
               "th,td{border:1px solid var(--line);padding:4px 8px;text-align:left;white-space:nowrap}"
               "th{background:var(--head)}tbody tr:nth-child(even){background:var(--head)}")
        return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
                "content=\"width=device-width,initial-scale=1\"><title>%s</title><style>%s</style></head><body>%s</body></html>\n"
                % (e(title), css, "\n".join(body)))


# ── data ──────────────────────────────────────────────────────────────────────

def load(run_id, runs_dir=None):
    paths = run_paths(run_id, runs_dir)
    meta = read_json(paths["run"])
    if not isinstance(meta, dict):
        raise ValueError("no run %s (%s)" % (run_id, paths["run"]))
    rows = read_jsonl(paths["results"])
    return {"paths": paths, "meta": meta, "rows": rows, "tasks": {(r["suite"], r["id"]): r for r in read_jsonl(paths["tasks"])},
            "criteria": read_jsonl(paths["criteria"]), "judgments": read_jsonl(paths["judgments"])}


def outcomes(run, suite):
    """{seat: {task: [(y, cost, secs, row)]}} from the latest ok row of each cell. Open: y = acceptance (y_soft >= 0.5)."""
    acc = rank.acceptance(run["judgments"]) if suite == "open" else {}
    out = {}
    for (_, task, seat, rep), r in latest_ok(run["rows"], suite).items():
        if suite == "open":
            ys = acc.get((task, rep, seat))
            y = None if ys is None else (1.0 if ys >= 0.5 else 0.0)
        else:
            y = float(r.get("y") or 0.0)
        out.setdefault(seat, {}).setdefault(task, []).append((y, r.get("cost_usd"), r.get("secs"), r))
    return out


def seat_stats(obs, n_boot=1000):
    """Leaderboard numbers of one seat from {task: [(y, cost, secs, row)]}."""
    ys = {t: [x[0] for x in v if x[0] is not None] for t, v in obs.items()}
    p, lo, hi = rate_ci(ys, n_boot)
    flat = [x for v in obs.values() for x in v]
    wins = sum(x[0] or 0.0 for x in flat if x[0] is not None)
    costs = [x[1] for x in flat if x[1] is not None]
    secs = [x[2] for x in flat if x[2] is not None]
    return {"n_tasks": len(obs), "n": sum(len(v) for v in ys.values()), "rate": (p, lo, hi), "wins": wins,
            "cost_task": mean(costs), "cost_succ": (sum(costs) / wins) if wins and len(costs) == len(flat) else None,
            "secs_task": mean(secs), "secs_succ": (sum(secs) / wins) if wins and secs else None,
            "cost_known": len(costs) == len(flat)}


def value_frontier(stats):
    """(best seat, cheapest seat whose 95% interval reaches the best seat's rate, Pareto list [(seat, rate, cost)])."""
    ranked = [(s, v) for s, v in stats.items() if v["rate"][0] is not None]
    if not ranked:
        return None, None, []
    best = max(ranked, key=lambda sv: (sv[1]["rate"][0], -(sv[1]["cost_task"] if sv[1]["cost_task"] is not None else 1e9)))
    bp = best[1]["rate"][0]
    match = [(s, v) for s, v in ranked if v["rate"][2] is not None and v["rate"][2] >= bp - 1e-9 and v["cost_task"] is not None]
    cheap = min(match, key=lambda sv: (sv[1]["cost_task"], -sv[1]["rate"][0]))[0] if match else None
    pareto, top = [], -1.0
    for s, v in sorted([sv for sv in ranked if sv[1]["cost_task"] is not None], key=lambda sv: (sv[1]["cost_task"], -sv[1]["rate"][0])):
        if v["rate"][0] > top:
            pareto.append((s, v["rate"][0], v["cost_task"]))
            top = v["rate"][0]
    return best[0], cheap, pareto


# ── routing replay ────────────────────────────────────────────────────────────

def _walk_pick(tier, pool):
    """The seat of `pool` ([(seat, category, cost)]) for a target tier: that tier, else up, else down; cheapest within."""
    for t in m.tier_walk(tier):
        c = sorted([x for x in pool if x[1] == t], key=lambda x: (x[2] if x[2] is not None else 1e9, x[0]))
        if c:
            return c[0][0]
    return None


def routing(run, suite, obs, cfg=None, decider=None):
    """Rows (policy, success, mean $, mean s, n) over the tasks every compared seat has an outcome for."""
    seats = sorted(s for s in obs if obs[s])
    common = sorted(set.intersection(*[{t for t, v in obs[s].items() if any(x[0] is not None for x in v)} for s in seats])) if seats else []
    if not common:
        return [], "no task has an outcome for every seat"
    meta = run["meta"]["seats"]
    cfg = dict(m.DEFAULT_CONFIG, **(cfg or {}))

    def cell(s, t):
        v = [x for x in obs[s][t] if x[0] is not None]
        return (mean([x[0] for x in v]), mean([x[1] for x in v]), mean([x[2] for x in v]))

    table = {(s, t): cell(s, t) for s in seats for t in common}
    avg_cost = {s: mean([table[(s, t)][1] for t in common]) for s in seats}
    pool = [(s, (meta.get(s) or {}).get("category"), avg_cost[s]) for s in seats]
    pool = [(s, "fast" if c == "local" else c, k) for s, c, k in pool if c in m.CELLS]
    picks = {}
    texts = {t: (run["tasks"].get((suite, t)) or {}).get("text") or "" for t in common}
    target = {t: m.tier_of(m.target_tier(m.classify_rules(texts[t]), cfg)) for t in common}
    if pool:
        picks["router: ladder (cheapest seat of the target category)"] = {t: _walk_pick(target[t], pool) for t in common}
    for h in sorted({(meta.get(s) or {}).get("harness") for s in seats}):
        hp = [x for x in pool if (meta.get(x[0]) or {}).get("harness") == h]
        if hp and len(hp) < len(pool):
            picks["router: ladder within %s" % h] = {t: _walk_pick(target[t], hp) for t in common}
    if decider == "utility" and pool:
        ut, why = _utility_picks(texts, pool, meta, cfg)
        if ut:
            picks["router: utility decider"] = ut
        else:
            picks["router: utility decider (unavailable: %s)" % why] = {}
    for s in seats:
        picks["always %s" % s] = {t: s for t in common}
    oracle = {}
    for t in common:
        good = [s for s in seats if (table[(s, t)][0] or 0) >= 0.5]
        cand = good or seats
        oracle[t] = min(cand, key=lambda s: (table[(s, t)][1] if table[(s, t)][1] is not None else 1e9, s))
    picks["hindsight best (cheapest seat that succeeded)"] = oracle
    rows = []
    for name, pk in picks.items():
        got = [table[(pk[t], t)] for t in common if pk.get(t)]
        if not got:
            rows.append((name, "-", "-", "-", 0))
            continue
        rows.append((name, pct(mean([g[0] for g in got])), usd(mean([g[1] for g in got])), fmt(mean([g[2] for g in got]), "%.1f"),
                     len(got)))
    return rows, ""


def _utility_picks(texts, pool, meta, cfg):
    """The utility decider through the real `route()` (skill mode) in a scratch home; picks map by model, else category."""
    try:
        by_h = {}
        for s, c, k in pool:
            by_h.setdefault((meta.get(s) or {}).get("harness"), []).append((s, c, k))
        h = max(by_h, key=lambda x: len(by_h[x]))
        inv = {"version": 2, "generated_at": time.time(), "harnesses": {h: {"cli": h, "available": "yes"}}, "api_keys": {},
               "local_endpoints": [], "local_models": [], "endpoints_sig": [], "hardware": {}, "warnings": []}
        out = {}
        with tempfile.TemporaryDirectory(prefix="arena-route-") as home:
            for t, text in texts.items():
                d = m.route(text, harness=h, inventory=inv, config=dict(cfg, decider="utility"), env={"MAESTRO_HOME": home},
                            home=home)
                if d.get("decider", "utility") != "utility":
                    return {}, "route() did not use the utility decider"
                hit = [s for s, c, k in by_h[h] if (meta.get(s) or {}).get("model") == d.get("model")]
                out[t] = hit[0] if hit else _walk_pick(m.tier_of(d.get("category") if d.get("category") in m.CELLS else "standard"), by_h[h])
        return out, ""
    except Exception as e:
        return {}, e.__class__.__name__


# ── the report ────────────────────────────────────────────────────────────────

def _fam(run, s):
    return (run["meta"]["seats"].get(s) or {}).get("family") or "?"


def build(run, n_boot=1000, decider=None, cfg=None):
    meta = run["meta"]
    doc = Doc()
    doc.h(1, "Arena report: %s" % meta["run_id"])
    rows = run["rows"]
    spend = sum(float(r.get("cost_usd") or 0) for r in rows + run["criteria"] + run["judgments"])
    ok_cells = set()
    for s in ("verifiable", "agentic", "open"):
        ok_cells |= set(latest_ok(rows, s))
    errs = len({(r["suite"], r["task"], r["seat"], int(r["rep"])) for r in rows if r.get("status") == "error"} - ok_cells)
    doc.p("Run started %s; suites: %s; contestants: %d; estimated spend $%.4f (Claude Code reports API-equivalent dollars "
          "also on a subscription; other CLIs are priced from tokens or characters x the catalog price; local models are "
          "$0). Cells that never succeeded: %d; quota skips: %d." % (
              time.strftime("%Y-%m-%d %H:%M", time.localtime(meta.get("created") or 0)), ", ".join(meta.get("suites") or []),
              len(meta.get("contestants") or []), spend, errs, sum(1 for r in rows if r.get("status") == "skipped")))
    for suite in ("verifiable", "agentic", "open"):
        obs = outcomes(run, suite)
        if not obs:
            continue
        doc.h(2, "%s suite" % suite)
        stats = {s: seat_stats(o, n_boot) for s, o in obs.items()}
        bt = {}
        if suite == "open":
            open_section(doc, run, stats, n_boot)
            ok = [j for j in run["judgments"] if j.get("ok")]
            bt, wp = rank.bt_with_ci(ok, n_boot=n_boot, players=list(obs)) if ok else ({}, {})
        head = ["seat", "family", "tasks", "n"] + (["BT score [95% CI]", "win prob"] if suite == "open" else []) + \
            ["%s [95%% CI]" % ("accepted" if suite == "open" else "pass rate"), "$/task", "$/success", "s/task", "s/success"]
        if suite == "verifiable":
            head.append("raw-answer pass")
        if suite == "agentic":
            head += ["diff lines", "timeouts", "escapes"]
        order = sorted(stats, key=lambda s: (-(bt.get(s, (0,))[0] if suite == "open" and s in bt else stats[s]["rate"][0] or 0), s))
        trows = []
        for s in order:
            v = stats[s]
            r = [s, _fam(run, s), v["n_tasks"], v["n"]]
            if suite == "open":
                b = bt.get(s)
                r += [ci_text(*b, as_pct=False) if b else "-", pct(wp.get(s)) if b else "-"]
            r += [ci_text(*v["rate"]), usd(v["cost_task"]), usd(v["cost_succ"]), fmt(v["secs_task"], "%.1f"), fmt(v["secs_succ"], "%.1f")]
            flat = [x[3] for t in obs[s].values() for x in t]
            if suite == "verifiable":
                r.append(pct(mean([float(x.get("y_strict") or 0) for x in flat])))
            if suite == "agentic":
                r += [fmt(mean([((x.get("diff") or {}).get("added", 0) + (x.get("diff") or {}).get("removed", 0)) for x in flat]), "%.0f"),
                      sum(1 for x in flat if x.get("timed_out")), sum(1 for x in flat if x.get("escapes"))]
            trows.append(r)
        doc.h(3, "Leaderboard")
        doc.table(head, trows)
        if suite == "verifiable":
            doc.p("Pass = the item's checker accepts the raw answer or one of the extraction variants (last function block, "
                  "last SQL block, last JSON value), tried identically for every seat; raw-answer pass counts the raw answer only.")
        breakdown(doc, run, suite, obs)
        best, cheap, pareto = value_frontier(stats)
        doc.h(3, "Value frontier")
        if best:
            b = stats[best]
            doc.p("Best: %s (%s). Cheapest seat whose 95%% interval reaches that rate: %s%s." % (
                best, pct(b["rate"][0]), cheap or "none with a known cost",
                " (%s per task vs %s)" % (usd(stats[cheap]["cost_task"]), usd(b["cost_task"])) if cheap else ""))
            doc.table(["Pareto seat (no cheaper seat does better)", "success", "$/task"], [(s, pct(r), usd(c)) for s, r, c in pareto])
        doc.h(3, "Routing replay")
        rrows, why = routing(run, suite, obs, cfg, decider)
        if rrows:
            doc.p("Maestro's router (rules classifier + ladder; the utility decider with --decider utility) picks a category per "
                  "task; the pick is the cheapest seat of that category in this run (walking up, then down, when none). Compared "
                  "with always using one seat and with hindsight (the cheapest seat that succeeded on each task).")
            doc.table(["policy", "success", "$/task", "s/task", "tasks"], rrows)
        else:
            doc.p("Routing replay skipped: %s." % why)
    return doc


def breakdown(doc, run, suite, obs):
    for key, label in (("job", "job"), ("difficulty", "difficulty")):
        vals = sorted({(run["tasks"].get((suite, t)) or {}).get(key) for o in obs.values() for t in o} - {None}, key=str)
        if len(vals) < 2:
            continue
        rows = []
        for s in sorted(obs):
            r = [s]
            for v in vals:
                ys = [x[0] for t, xs in obs[s].items() if (run["tasks"].get((suite, t)) or {}).get(key) == v
                      for x in xs if x[0] is not None]
                r.append("%s (%d)" % (pct(mean(ys)), len(ys)) if ys else "-")
            rows.append(r)
        doc.h(3, "By %s" % label)
        doc.table(["seat"] + [("d%s" % v) if key == "difficulty" else str(v) for v in vals], rows)


def open_section(doc, run, stats, n_boot):
    meta = run["meta"]
    judges = meta.get("judges") or []
    fams = sorted({_fam(run, j) for j in judges})
    doc.p("Judges: %s. Criteria were written from the request alone before any answer was seen (lead judge %s); every "
          "judge ranked the anonymized, shuffled answers twice, the second time in reverse order. Scores are Bradley-Terry "
          "log-strengths from the pairwise wins in all rankings (mean 0, 0.5 pseudo-wins each way per compared pair), 95%% "
          "intervals from a bootstrap over tasks. Accepted = every MUST criterion met and no disqualifier, averaged over "
          "judges and passes." % (", ".join("%s (%s)" % (j, _fam(run, j)) for j in judges) or "none", judges[0] if judges else "-"))
    if len(fams) < 2:
        doc.p("WARNING: SINGLE-FAMILY JUDGING. Every judge is %s, so the ranking may carry that family's self-preference and "
              "cannot be checked against another family. Add a judge from another family (--judges or seats.json)." %
              (fams[0] if fams else "unknown"), warn=True)
    elif meta.get("judge_note"):
        doc.p(meta["judge_note"], warn=True)
    js = run["judgments"]
    lengths = {(r["task"], r["rep"], r["seat"]): len(r.get("answer") or "") for r in latest_ok(run["rows"], "open").values()}
    family = {s: (v or {}).get("family") for s, v in meta["seats"].items()}
    d = rank.diagnostics(js, lengths, family)
    doc.h(3, "Judging checks")
    first, chance = d["position_first"]
    doc.p("Judgments: %d usable, %d failed to parse or answer. Position bias: the answer shown first was ranked first in %s "
          "of judgments (%s by chance); the reversed second pass cancels it in the scores. Forward vs reversed ranking by the "
          "same judge: mean Kendall tau %s (1 = identical). Length bias: mean within-item Spearman of score vs answer length "
          "%s over %d items (positive = longer answers ranked higher). Acceptance agreement between judges: %s." % (
              d["n_judgments"], d["n_failed"], pct(first), pct(chance), fmt(d["pass_tau"]), fmt(d["length_rho"][0], "%+.2f"),
              d["length_rho"][1], pct(d["accept_agreement"])))
    doc.table(["judge pair", "mean Kendall tau"], [("%s vs %s" % k, fmt(v)) for k, v in sorted(d["judge_tau"].items())])
    cf = sorted({k[1] for k in d["family_matrix"]})
    doc.table(["judge family \\ contestant family"] + cf,
              [[jf] + ["%s (%d)" % (fmt(d["family_matrix"][(jf, c)][0]), d["family_matrix"][(jf, c)][1]) if (jf, c) in d["family_matrix"] else "-"
                       for c in cf] for jf in sorted({k[0] for k in d["family_matrix"]})])
    sp = d["self_preference"]
    doc.p("Mean normalized rank score (1 = ranked best, 0 = worst) a judge family gave each contestant family. Self-preference "
          "(own-family answers, judge minus judges of other families): %s." % (
              "; ".join("%s %s (n=%d)" % (f, fmt(v[0], "%+.2f"), v[1]) for f, v in sorted(sp.items())) or
              "not measurable (needs a judge of another family scoring the same answers)"))


def write(run, md_path, html_path=None, **kw):
    doc = build(run, **kw)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(doc.md())
    if html_path:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(doc.html("Arena %s" % run["meta"]["run_id"]))
    return doc


def report(run_id, runs_dir=None, html_out=False, n_boot=1000, decider=None):
    run = load(run_id, runs_dir)
    d = run["paths"]["dir"]
    cfg = m.load_config(m.maestro_home())
    doc = write(run, os.path.join(d, "report.md"), os.path.join(d, "report.html") if html_out else None, n_boot=n_boot,
                decider=decider or (cfg.get("decider") if cfg.get("decider") == "utility" else None), cfg=cfg)
    return doc, d
