#!/usr/bin/env python3
"""The alpha set: outcomes per tier, and proving alpha with them (S1 spec §7, §8.2, §9).

Collection (subagent steps write manifests under evals/work/ and ingest the orchestrator's replies; see evalkit.py):
    alpha.py check                         validate verifiable items: reference passes, empty and copy-the-input fail, 3 runs
    alpha.py pilot [--n 200]               pick the pilot (stratified by job x panel difficulty, >= 50% verifiable)
    alpha.py answer --phase pilot|pilot-repeat|1|2|repeat [--ingest]
    alpha.py grade --stage check|criteria|llm|adjudicate|audit [--ingest]
    alpha.py fit [--set pilot|pool]        outcomes.jsonl (+ irt.json, validity.json, linking.jsonl)
    alpha.py pilot-report                  measurements, projection at rho 0.4-0.7, go/no-go (appends to the ledger)
    alpha.py select [--total 2400]         the pool: verifiable + open native self-contained + persona
Proving alpha:
    alpha.py replay --split dev --policy B4|R|... [--out preds.jsonl]
    alpha.py report --split dev|test|persona    every policy, CIs, slices (test/persona append to the ledger)
    alpha.py gate [--update]               the CI gate (stdlib, no network, < 60 s); "no outcome data yet" until it exists
    alpha.py release [--audit CSV] [--shadow JSON] [--apply]
    alpha.py calibrate-v                   V0 per objective on dev -> evals/alpha/v0.json
    alpha.py pacing [--synthetic N]        the quota pacing simulation
    alpha.py collect --seat SEAT [--n 150] run the linking set on one of your seats -> ~/.maestro/anchors.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import math
import os
import random
import re
import sys
import tempfile
import time

import alpha_eval as ev
import alpha_stats as st
import evalkit as ek
import label
import split

TIERS = ek.TIERS
V_JOB = {"code-fn": "code", "debug-fix": "debug", "data": "data", "edit": "edit", "extract": "summarize", "reason": "ask",
         "review": "review"}
ANSWER_PROMPT = ("Answer the request below as a helpful assistant in a chat. Do not use any tools and do not read or write "
                 "files. If the request needs information you do not have, say what you would need.{contract}\n\n"
                 "{history}USER REQUEST:\n{prompt}")
SELF_ID = re.compile(r"\b(as )?(claude|an ai (model|assistant) (made|created|developed) by anthropic|haiku|sonnet|opus|fable)\b", re.I)
PHASE2_RATE = 0.25


def paths(a=None):
    a = a or argparse.Namespace()
    al = getattr(a, "alpha_dir", None) or ek.ALPHA
    work = getattr(a, "work", None) or ek.WORK
    p = {k: os.path.join(al, k + ".jsonl") for k in ("pilot", "pool", "criteria", "outcomes", "pilot_outcomes", "linking")}
    p.update({k: os.path.join(al, k + ".json") for k in ("irt", "validity", "assets", "v0", "gates")})
    p.update(alpha=al, work=work, inventory=os.path.join(al, "inventory.json"), baseline=os.path.join(al, "baseline_preds.jsonl"),
             answers=os.path.join(work, "answers.jsonl"), grades=os.path.join(work, "grades.jsonl"),
             audit=os.path.join(work, "audit.csv"), audit_key=os.path.join(work, "audit_key.json"),
             verifiable=getattr(a, "verifiable", None) or os.path.join(al, "verifiable.jsonl"),
             corpus=getattr(a, "corpus_file", None) or os.path.join(ek.HERE, "corpus", "pool.jsonl"),
             persona=getattr(a, "persona_file", None) or os.path.join(ek.HERE, "corpus", "ood.jsonl"),
             aggregate=getattr(a, "aggregate", None) or os.path.join(ek.LABELS, "aggregate.jsonl"),
             manifest=getattr(a, "manifest", None) or split.MANIFEST, ledger=getattr(a, "ledger", None) or ek.LEDGER)
    return p


# ── items ──────────────────────────────────────────────────────────────────────

def argmax(xs):
    return max(range(len(xs)), key=lambda k: xs[k])


def candidates(p):
    """{id: item} for every possible alpha item: verifiable (V), native self-contained corpus items (open) and
    self-contained persona items. job / difficulty come from the panel when labeled (else the V type and its intended
    difficulty, used for stratification only)."""
    splits = split.load_splits(p["manifest"])
    agg = {r["id"]: r for r in ek.read_jsonl(p["aggregate"])}
    out = {}
    for source, path in (("verifiable", p["verifiable"]), ("open", p["corpus"]), ("persona", p["persona"])):
        for r in ek.read_jsonl(path):
            if source == "verifiable" and r.get("validated") is False:
                continue
            if source != "verifiable" and (not r.get("self_contained") or r.get("images")):
                continue
            rid, lab = r["id"], agg.get(r["id"])
            corpus, sp, cl = splits.get(rid, (None, None, None))
            if sp is None:
                cl = "k:" + rid
                sp = "persona" if source == "persona" else split.split_of(cl)
            out[rid] = {"id": rid, "source": source, "split": sp, "cluster": cl, "prompt": r.get("prompt") or "",
                        "history": r.get("history") or [], "context": r.get("context") or "chat", "lang": r.get("lang") or "en",
                        "contract": r.get("contract") or "", "checker_py": r.get("checker_py"), "type": r.get("type"),
                        "reference": r.get("reference_answer"),
                        "job": label.top_job(lab["job"]) if lab else V_JOB.get(r.get("type")) or (r.get("hint") or {}).get("job"),
                        "difficulty": argmax(lab["difficulty"]) if lab else r.get("difficulty_intent", (r.get("hint") or {}).get("difficulty")),
                        "panel": {k: lab[k] for k in ("difficulty", "min_tier")} if lab else None,
                        "words": len(re.findall(r"\w+", r.get("prompt") or ""))}
    return out


def stratified(items, n, key, cap=None, seed=0, salt=""):
    """Round-robin over strata (sorted), each shuffled by a seeded hash; at most `cap` per stratum."""
    by = {}
    for it in sorted(items, key=lambda it: ek.unit(salt, seed, it["id"])):
        by.setdefault(key(it), []).append(it)
    groups = [by[k][:cap] if cap else by[k] for k in sorted(by, key=str)]
    out = []
    while len(out) < n and any(groups):
        for g in groups:
            if g and len(out) < n:
                out.append(g.pop(0))
    return out


def pool_rows(items):
    keep = ("id", "source", "split", "cluster", "job", "difficulty", "lang", "context", "words")
    return [dict({k: it[k] for k in keep}, pilot=bool(it.get("pilot"))) for it in items]


def cmd_pilot(a, p):
    cands = list(candidates(p).values())
    key = lambda it: (it["job"], it["difficulty"])  # noqa: E731
    ver = stratified([c for c in cands if c["source"] == "verifiable"], int(math.ceil(a.n / 2.0)), key, seed=a.seed, salt="pilot")
    opn = stratified([c for c in cands if c["source"] == "open"], a.n - len(ver), key, seed=a.seed, salt="pilot")
    for it in ver + opn:
        it["pilot"] = True
    rows = pool_rows(ver + opn)
    ek.write_jsonl(p["pilot"], rows)
    print("pilot: %d items (%d verifiable, %d open) -> %s" % (len(rows), len(ver), len(opn), p["pilot"]))
    if len(ver) * 2 < len(rows):
        print("warning: fewer than 50% verifiable (only %d verifiable candidates)" % len(ver))
    return 0


def cmd_select(a, p):
    cands = candidates(p)
    pilot = [cands[r["id"]] for r in ek.read_jsonl(p["pilot"]) if r["id"] in cands]
    for it in pilot:
        it["pilot"] = True
    scale = a.total / 2400.0
    nv, no = int(round(1400 * scale)), int(round(1000 * scale))
    have = {it["id"] for it in pilot}
    key = lambda it: (it["job"], it["difficulty"])  # noqa: E731
    pv = sum(it["source"] == "verifiable" for it in pilot)
    ver = stratified([c for c in cands.values() if c["source"] == "verifiable" and c["id"] not in have], nv - pv, key,
                     seed=a.seed, salt="pool")
    opn = stratified([c for c in cands.values() if c["source"] == "open" and c["id"] not in have], no - (len(pilot) - pv), key,
                     cap=40, seed=a.seed, salt="pool")
    per = stratified([c for c in cands.values() if c["source"] == "persona"], a.persona, key, seed=a.seed, salt="pool")
    rows = pool_rows(pilot + ver + opn + per)
    ek.write_jsonl(p["pool"], rows)
    n_ver = sum(r["source"] == "verifiable" for r in rows if r["source"] != "persona")
    n_all = sum(r["source"] != "persona" for r in rows)
    print("pool: %d items (%d verifiable, %d open, %d persona) -> %s" % (len(rows), n_ver, n_all - n_ver, len(per), p["pool"]))
    if n_ver * 2 < n_all:
        print("warning: fewer than 50% verifiable")
    return 0


def load_set(p, which):
    cands = candidates(p)
    rows = ek.read_jsonl(p["pilot" if which == "pilot" else "pool"])
    return [dict(cands[r["id"]], pilot=which == "pilot" or bool(r.get("pilot"))) for r in rows if r["id"] in cands]


# ── answers (§7.3) ─────────────────────────────────────────────────────────────

def answer_prompt(it):
    hist = ek.render_history(it.get("history"))
    return ANSWER_PROMPT.format(contract=(" " + it["contract"]) if it.get("contract") else "",
                                history=hist + "\n\n" if hist else "", prompt=it["prompt"])


def answers(p):
    """{(item, tier, rep): latest answer row}."""
    out = {}
    for r in ek.read_jsonl(p["answers"]):
        out[(r["item"], r["tier"], r["rep"])] = r
    return out


def grades(p):
    """{(item, tier, rep): {grader: row}} (a later row for the same grader wins)."""
    out = {}
    for r in ek.read_jsonl(p["grades"]):
        out.setdefault((r["item"], r["tier"], r["rep"]), {})[r["by"]] = r
    return out


def verdicts(p, items):
    """{(item, tier, rep): y} from the checker (verifiable) or the combined LLM grade (open), for valid answers."""
    g, ans, val = grades(p), answers(p), ek.read_json(p["validity"]) or {}
    out = {}
    for key, a in ans.items():
        it = items.get(key[0])
        if not it or not a["valid"]:
            continue
        y = combine(it, key[1], g.get(key, {}), val)
        if y is not None:
            out[key] = y
    return out


def combine(it, tier, gs, val):
    """y for one cell: the checker on verifiable items; else the graders' accepted, with fable adjudicating a split,
    self-preference replacement and the false-pass correction (§7.5)."""
    if it["source"] == "verifiable":
        return float(gs["checker"]["y"]) if "checker" in gs else None
    votes = {}
    for g in ("sonnet", "opus"):
        if g in gs:
            selfpref = g == tier and (val.get("self_preference") or {}).get(g, 0.0) > 0.05
            votes[g] = gs["fable"]["y"] if selfpref and "fable" in gs else gs[g]["y"]
    if len(votes) < 2:
        return None
    y = sum(votes.values()) / 2.0
    if y == 0.5:
        if "fable" not in gs:
            return None
        y = float(gs["fable"]["y"])
    corr = val.get("fp_correction")
    if corr and tier in corr:
        y = y * (1 - corr[tier]["FDR"]) + (1 - y) * corr[tier]["FOR"]
    return y


def cmd_answer(a, p):
    d = ek.work_dir("answer", p["work"])
    if a.ingest:
        return ingest_answers(p, d)
    items = load_set(p, "pilot" if a.phase.startswith("pilot") else "pool")
    byid = {it["id"]: it for it in items}
    ans = answers(p)
    inflight = {(m["item"], m["tier"], m["rep"]) for m in ek.pending(d).values()}
    ys = verdicts(p, byid)
    want = []
    if a.phase in ("pilot", "1"):
        want = [(it, t, 1) for it in items for t in (TIERS if a.phase == "pilot" else ("haiku", "fable"))]
    elif a.phase == "2":
        for it in items:
            h, f = ys.get((it["id"], "haiku", 1)), ys.get((it["id"], "fable", 1))
            if h is None or f is None:
                continue
            if (h >= 0.5) != (f >= 0.5) or ek.unit("phase2", a.seed, it["id"]) < PHASE2_RATE:
                want += [(it, t, 1) for t in ("sonnet", "opus")]
    else:  # repeats: haiku and sonnet again on non-unanimous items
        nonu = [it for it in items if len({ys.get((it["id"], t, 1), -1) >= 0.5 for t in TIERS}) > 1
                and all((it["id"], t, 1) in ys for t in TIERS)]
        nonu.sort(key=lambda it: ek.unit("repeat", a.seed, it["id"]))
        want = [(it, t, 2) for it in (nonu if a.phase == "pilot-repeat" else nonu[:a.repeats]) for t in ("haiku", "sonnet")]
    calls = []
    for it, t, rep in want:
        key = (it["id"], t, rep)
        old = ans.get(key)
        if key in inflight or (old and (old["valid"] or old["attempt"] >= 2)):
            continue
        att = old["attempt"] + 1 if old else 1
        cid = "ans:%s:%s:r%d:a%d" % (it["id"], t, rep, att)
        calls.append(({"call_id": cid, "model": t, "prompt": answer_prompt(it), "expect": "text"},
                      {"kind": "answer", "item": it["id"], "tier": t, "rep": rep, "attempt": att}))
    files = ek.emit(d, "answer-%s" % a.phase, calls, a.per_file)
    print("%d answer calls in %d files under %s (send each prompt to its model in a fresh call with no tools; "
          "at most 8 at once)" % (len(calls), len(files), d))
    print(ek.quota_status()[1])
    return 0


def ingest_answers(p, d):
    replies = ek.read_replies(d)
    bad = replies.pop(None, [])
    rows, done = [], []
    for cid, meta in ek.pending(d).items():
        r = replies.get(cid)
        if r is None:
            continue
        text = r["reply"]
        why = r.get("error") and "error" or (r.get("tool_calls") and "tool use") or (r.get("invalid") and "invalid") or \
            (not text.strip() and "empty")
        rows.append({"call_id": cid, "item": meta["item"], "tier": meta["tier"], "rep": meta["rep"], "attempt": meta["attempt"],
                     "valid": not why, "reason": why or "", "sha256": ek.sha256_text(text), "tout": ek.est_tokens(text),
                     "chars": len(text), "text": text})
        done.append({"call_id": cid, "status": "invalid" if why else "ok"})
    ek.append_jsonl(p["answers"], rows)
    ek.mark_done(d, done)
    assets(p)
    print("ingested %d answers (%d invalid, retried once by the next `answer` run)%s" % (
        len(rows), sum(not r["valid"] for r in rows), "".join("\n  " + b for b in bad)))
    return 0


def assets(p):
    """sha256 of the out-of-git release assets (answer texts, grades) in evals/alpha/assets.json."""
    out = {}
    for k in ("answers", "grades"):
        if os.path.exists(p[k]):
            out[os.path.basename(p[k])] = {"sha256": ek.sha256_file(p[k]), "rows": len(ek.read_jsonl(p[k]))}
    ek.write_json(p["assets"], out)


# ── grading (§7.4, §7.5) ───────────────────────────────────────────────────────

def clean_answer(text, limit=6000):
    """Strip self-identifying phrases, normalize markdown outside code fences, truncate at `limit` characters."""
    parts = re.split(r"(```.*?(?:```|\Z))", text or "", flags=re.S)
    for k in range(0, len(parts), 2):
        s = SELF_ID.sub("", parts[k])
        s = re.sub(r"(?m)^(\s*)[*+] ", r"\1- ", s)
        s = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda mt: mt.group(1) or mt.group(2), s)
        s = re.sub(r"(?m)^#{1,6}[ \t]+", "", s)
        s = re.sub(r"[ \t]+$", "", s, flags=re.M)
        parts[k] = re.sub(r"\n{3,}", "\n\n", s)
    out = "".join(parts).strip()
    return out if len(out) <= limit else out[:limit] + "\n[answer truncated at %d characters]" % limit


def validity_ids(items, n=300, seed=0):
    """The grader validity subset: every pilot verifiable item plus random verifiable items up to n."""
    ver = sorted((it for it in items if it["source"] == "verifiable"), key=lambda it: (not it.get("pilot"), ek.unit("validity", seed, it["id"])))
    return {it["id"] for it in ver[:max(n, sum(1 for it in ver if it.get("pilot")))]}


CRITERIA_PROMPT = (
    "You write grading criteria for the request below before any answer exists. From the request alone, write:\n"
    "- 3 to 6 MUST criteria: each necessary for a correct answer, binary, and checkable from the answer text alone;\n"
    "- 0 to 3 DISQUALIFIERS: specific errors that make an answer unacceptable (for example a factual error about X, or "
    "ignoring constraint Y);\n- a reference sketch of 1 to 3 sentences describing a good answer.\n"
    "Reply with only JSON: {{\"must\": [\"...\"], \"disqualifiers\": [\"...\"], \"sketch\": \"...\"}}\n\n{body}")
GRADE_PROMPT = (
    "You grade answers to a request against fixed criteria. For each answer decide every MUST criterion (true = met) and "
    "every DISQUALIFIER (true = present). Judge only what the answer text shows; do not rank or score the answers.\n\n"
    "{body}\n\nMUST criteria:\n{must}\nDISQUALIFIERS:\n{disq}\nReference sketch: {sketch}\n\n{answers}\n\n"
    "Reply with only JSON: {{{shape}}}")
ADJ_PROMPT = (
    "Two graders disagreed on whether these answers are acceptable. An answer is acceptable when it meets every MUST "
    "criterion and shows no DISQUALIFIER. Their criterion-level judgments follow each answer. Decide each answer.\n\n"
    "{body}\n\nMUST criteria:\n{must}\nDISQUALIFIERS:\n{disq}\nReference sketch: {sketch}\n\n{answers}\n\n"
    "Reply with only JSON: {{{shape}}}")


def _body(it):
    hist = ek.render_history(it.get("history"))
    return (hist + "\n\n" if hist else "") + "USER REQUEST:\n" + it["prompt"]


def _numbered(xs):
    return "\n".join("%d. %s" % (k + 1, x) for k, x in enumerate(xs)) or "(none)"


def cmd_grade(a, p):
    d = ek.work_dir("grade", p["work"])
    if a.ingest:
        return ingest_grades(p, d)
    items = {it["id"]: it for it in load_set(p, a.set)}
    ans, g = answers(p), grades(p)
    inflight = {(m["kind"], m["item"]) for m in ek.pending(d).values()}
    if a.stage == "check":
        todo = [(k, r) for k, r in ans.items() if r["valid"] and k[0] in items and items[k[0]]["source"] == "verifiable"
                and "checker" not in g.get(k, {})]
        with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
            ys = list(pool.map(lambda kr: ek.run_checker(items[kr[0][0]]["checker_py"], kr[1]["text"]), todo))
        ek.append_jsonl(p["grades"], [{"item": k[0], "tier": k[1], "rep": k[2], "by": "checker", "y": int(y)} for (k, _), y in zip(todo, ys)])
        assets(p)
        print("checked %d verifiable answers: %d pass" % (len(todo), sum(ys)))
        return 0
    if a.stage == "audit":
        return write_audit(p, items, ans, g, a.seed)
    crit = {r["id"]: r for r in ek.read_jsonl(p["criteria"])}
    done_ids = ek.load_done(d)
    vids = validity_ids(items.values(), seed=a.seed)
    gradable = [it for it in items.values() if it["source"] != "verifiable" or it["id"] in vids]
    calls = []
    if a.stage == "criteria":
        for it in gradable:
            if it["id"] not in crit and ("criteria", it["id"]) not in inflight:
                calls.append(({"call_id": ek.attempt_id(done_ids, "crit:%s" % it["id"]), "model": "opus",
                               "prompt": CRITERIA_PROMPT.format(body=_body(it)),
                               "expect": "json"}, {"kind": "criteria", "item": it["id"]}))
    else:
        val = ek.read_json(p["validity"]) or {}
        for it in gradable:
            c = crit.get(it["id"])
            if not c or (a.stage, it["id"]) in inflight:
                continue
            keys = sorted(k for k, r in ans.items() if k[0] == it["id"] and r["valid"])
            if a.stage == "llm":
                for grader in ("sonnet", "opus"):
                    todo = [k for k in keys if grader not in g.get(k, {})]
                    if todo:
                        calls.append(grade_call(it, c, todo, ans, grader, a.seed, done=done_ids))
            else:
                selfp = [gr for gr, v in (val.get("self_preference") or {}).items() if v > 0.05]
                todo = [k for k in keys if "fable" not in g.get(k, {}) and (
                    ("sonnet" in g.get(k, {}) and "opus" in g.get(k, {}) and g[k]["sonnet"]["y"] != g[k]["opus"]["y"])
                    or (it["source"] != "verifiable" and k[1] in selfp))]
                if todo:
                    calls.append(grade_call(it, c, todo, ans, "fable", a.seed, g, done=done_ids))
    files = ek.emit(d, "grade-%s" % a.stage, calls, a.per_file)
    print("%d %s calls in %d files under %s" % (len(calls), a.stage, len(files), d))
    print(ek.quota_status()[1])
    return 0


def grade_call(it, c, keys, ans, grader, seed, g=None, done=()):
    """One call per (item, grader): the item's answers shuffled and anonymized as A, B, C, ..."""
    order = sorted(keys, key=lambda k: ek.unit("shuffle", seed, grader, *k))
    labels = {chr(65 + n): "%s:%s:%d" % k for n, k in enumerate(order)}
    blocks = []
    for lab, k in zip(labels, order):
        block = "Answer %s:\n<<<\n%s\n>>>" % (lab, clean_answer(ans[k]["text"]))
        if g is not None:
            for gr in ("sonnet", "opus"):
                det = (g.get(k, {}).get(gr) or {}).get("detail")
                if det:
                    block += "\nGrader %s: MUST %s; DISQUALIFIERS %s" % ("1" if gr == "sonnet" else "2", det["must"], det["disq"])
        blocks.append(block)
    if g is None:
        shape = ", ".join('"%s": {"must": [%d booleans], "disq": [%d booleans]}' % (lab, len(c["must"]), len(c["disq"])) for lab in labels)
        prompt = GRADE_PROMPT.format(body=_body(it), must=_numbered(c["must"]), disq=_numbered(c["disq"]), sketch=c["sketch"],
                                     answers="\n\n".join(blocks), shape=shape)
        kind = "llm"
    else:
        shape = ", ".join('"%s": true or false' % lab for lab in labels)
        prompt = ADJ_PROMPT.format(body=_body(it), must=_numbered(c["must"]), disq=_numbered(c["disq"]), sketch=c["sketch"],
                                   answers="\n\n".join(blocks), shape=shape)
        kind = "adjudicate"
    cid = ek.attempt_id(done, "%s:%s:%s:%s" % ("grd" if kind == "llm" else "adj", it["id"], grader,
                                               ek.sha256_text("|".join(labels.values()))[:8]))
    return ({"call_id": cid, "model": grader, "prompt": prompt, "expect": "json"},
            {"kind": kind, "item": it["id"], "grader": grader, "labels": labels, "n_must": len(c["must"]), "n_disq": len(c["disq"])})


def ingest_grades(p, d):
    replies = ek.read_replies(d)
    bad = replies.pop(None, [])
    crit_rows, grade_rows, done = [], [], []
    for cid, meta in ek.pending(d).items():
        r = replies.get(cid)
        if r is None:
            continue
        try:
            obj = ek.parse_json_reply(r["reply"])
            if meta["kind"] == "criteria":
                must, disq, sketch = obj.get("must"), obj.get("disqualifiers", obj.get("disq", [])), obj.get("sketch")
                if not (isinstance(must, list) and 3 <= len(must) <= 6 and isinstance(disq, list) and len(disq) <= 3
                        and isinstance(sketch, str) and all(isinstance(x, str) and x.strip() for x in must + disq)):
                    raise ValueError("criteria need 3-6 MUST strings, 0-3 disqualifier strings and a sketch")
                crit_rows.append({"id": meta["item"], "must": must, "disq": disq, "sketch": sketch})
            else:
                rows = []
                for lab, key in meta["labels"].items():
                    item, tier, rep = key.rsplit(":", 2)
                    v = obj[lab]
                    if meta["kind"] == "llm":
                        mu, dq = v.get("must"), v.get("disq", [])
                        if not (isinstance(mu, list) and len(mu) == meta["n_must"] and all(isinstance(x, bool) for x in mu)
                                and isinstance(dq, list) and len(dq) == meta["n_disq"] and all(isinstance(x, bool) for x in dq)):
                            raise ValueError("answer %s: need %d MUST and %d disqualifier booleans" % (lab, meta["n_must"], meta["n_disq"]))
                        y = int(all(mu) and not any(dq))
                        rows.append({"item": item, "tier": tier, "rep": int(rep), "by": meta["grader"], "y": y,
                                     "detail": {"must": mu, "disq": dq}})
                    else:
                        if not isinstance(v, bool):
                            raise ValueError("answer %s: need true or false" % lab)
                        rows.append({"item": item, "tier": tier, "rep": int(rep), "by": "fable", "y": int(v)})
                grade_rows += rows
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            done.append({"call_id": cid, "status": "invalid", "reason": str(e)[:200]})
            continue
        done.append({"call_id": cid, "status": "ok"})
    ek.append_jsonl(p["criteria"], crit_rows)
    ek.append_jsonl(p["grades"], grade_rows)
    ek.mark_done(d, done)
    assets(p)
    inv = [x for x in done if x["status"] == "invalid"]
    print("ingested %d criteria, %d grades; %d invalid replies (re-emitted by the next run)%s" % (
        len(crit_rows), len(grade_rows), len(inv), "".join("\n  %s: %s" % (x["call_id"], x["reason"]) for x in (inv[:10]))
        + "".join("\n  " + b for b in bad)))
    return 0


def write_audit(p, items, ans, g, seed):
    """150 cells for the user to grade blind: 50 where the graders disagreed, 50 from non-monotone items, 50 random."""
    ys = verdicts(p, items)
    open_keys = [k for k in ys if items[k[0]]["source"] != "verifiable" and k[2] == 1]
    rnd = lambda k: ek.unit("audit", seed, *k)  # noqa: E731
    dis = sorted((k for k in open_keys if "sonnet" in g.get(k, {}) and "opus" in g.get(k, {})
                  and g[k]["sonnet"]["y"] != g[k]["opus"]["y"]), key=rnd)[:50]
    nonmono = {i for i in {k[0] for k in open_keys} if is_nonmonotone({t: ys.get((i, t, 1)) for t in TIERS})}
    nm = sorted((k for k in open_keys if k[0] in nonmono and k not in dis), key=rnd)[:50]
    used = set(dis) | set(nm)
    rand = sorted((k for k in open_keys if k not in used), key=rnd)[:50]
    crit = {r["id"]: r for r in ek.read_jsonl(p["criteria"])}
    cells = [(k, "disagree") for k in dis] + [(k, "nonmonotone") for k in nm] + [(k, "random") for k in rand]
    cells.sort(key=lambda kc: rnd(kc[0]))
    os.makedirs(os.path.dirname(p["audit"]), exist_ok=True)
    with open(p["audit"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cell", "request", "criteria", "answer", "user_accepts (1/0)"])
        for n, (k, _) in enumerate(cells):
            c = crit.get(k[0]) or {}
            w.writerow(["c%03d" % n, _body(items[k[0]]), "MUST: " + "; ".join(c.get("must", [])) + " | NOT: " + "; ".join(c.get("disq", [])),
                        clean_answer(ans[k]["text"]), ""])
    ek.write_json(p["audit_key"], {"c%03d" % n: {"cell": "%s:%s:%d" % k, "stratum": s, "y": ys[k]} for n, (k, s) in enumerate(cells)})
    print("audit: %d cells -> %s (fill the last column; the key is kept in %s)" % (len(cells), p["audit"], p["audit_key"]))
    return 0


def audit_agreement(csv_path, key_path):
    key = ek.read_json(key_path) or {}
    per = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0] not in key or not row[-1].strip():
                continue
            u = 1 if row[-1].strip().lower() in ("1", "y", "yes", "true", "pass") else 0
            k = key[row[0]]
            per.setdefault(k["stratum"], []).append((u, int(k["y"] >= 0.5)))
    return {s: {"n": len(v), "raw": sum(a == b for a, b in v) / float(len(v)), "kappa": st.cohen_kappa(v)} for s, v in per.items()}


# ── fit: outcomes.jsonl (§7.8), irt.json, validity.json, linking.jsonl ──────────

def is_nonmonotone(y):
    ys = [y.get(t) for t in TIERS]
    return any(a is not None and a >= 0.5 and any(b is not None and b < 0.5 for b in ys[k + 1:]) for k, a in enumerate(ys))


def validity(p, items):
    """Grader validity on verifiable items graded by both the checker and LLM graders (§7.4, §7.5)."""
    g, ans = grades(p), answers(p)
    rows = [(k, gs) for k, gs in g.items() if "checker" in gs and k[0] in items and k[2] == 1]
    out = {"n_cells": len(rows), "graders": {}, "self_preference": {}, "fp_correction": None}
    for gr in ("sonnet", "opus"):
        per = {}
        for k, gs in rows:
            if gr in gs:
                per.setdefault(k[1], []).append((gs[gr]["y"], gs["checker"]["y"]))
        out["graders"][gr] = {t: {"n": len(v), "false_pass": _rate(v, 1, 0), "false_fail": _rate(v, 0, 1),
                                  "accept": sum(a for a, _ in v) / float(len(v)), "checker": sum(c for _, c in v) / float(len(v))}
                              for t, v in per.items() if v}
        own, other = per.get(gr, []), [x for t, v in per.items() if t != gr for x in v]
        if own and other:
            bias = lambda v: sum(a - c for a, c in v) / float(len(v))  # noqa: E731
            out["self_preference"][gr] = round(bias(own) - bias(other), 4)
    pairs = [(gs["sonnet"]["y"], gs["opus"]["y"]) for _, gs in g.items() if "sonnet" in gs and "opus" in gs]
    out["grader_kappa"] = st.cohen_kappa(pairs)
    comb = {}
    for k, gs in rows:
        if "sonnet" in gs and "opus" in gs:
            v = (gs["sonnet"]["y"] + gs["opus"]["y"]) / 2.0
            v = float(gs["fable"]["y"]) if v == 0.5 and "fable" in gs else v
            if v != 0.5:
                comb.setdefault(k[1], []).append((int(v), gs["checker"]["y"]))
    fp = {t: _rate(v, 1, 0) for t, v in comb.items()}
    out["combined_false_pass"] = fp
    if fp.get("haiku") is not None and fp.get("fable") is not None and fp["haiku"] - fp["fable"] > 0.03:
        out["fp_correction"] = {t: {"FDR": _cond(v, 1, 0), "FOR": _cond(v, 0, 1)} for t, v in comb.items()}
    xs, ys2, ws = [], [], []
    for i in {k[0] for k, _ in rows}:  # within-item length slope of accepted on log length
        cells = [(math.log(1 + ans[k]["chars"]), gs[gr]["y"]) for k, gs in rows if k[0] == i and k in ans
                 for gr in ("sonnet", "opus") if gr in gs]
        if len(cells) >= 2:
            mu = sum(x for x, _ in cells) / len(cells)
            for x, y in cells:
                xs.append(x - mu)
                ys2.append(y)
                ws.append(1.0)
    out["length_slope"] = st.logistic_fit(xs, ys2, ws)[1] if len(xs) >= 10 else None
    return out


def _rate(v, pred, truth):
    """P(grader says `pred` | checker says `truth`)."""
    den = [a for a, c in v if c == truth]
    return None if not den else sum(a == pred for a in den) / float(len(den))


def _cond(v, pred, truth):
    """P(checker says `truth` | grader says `pred`): FDR = P(fail | pass), FOR = P(pass | fail)."""
    den = [c for a, c in v if a == pred]
    return 0.0 if not den else sum(c == truth for c in den) / float(len(den))


def build_outcomes(p, items, seed=0):
    ans = answers(p)
    ys = verdicts(p, items)
    g = grades(p)
    rows = []
    for it in items.values():
        y = {t: ys[(it["id"], t, 1)] for t in TIERS if (it["id"], t, 1) in ys}
        if not y:
            continue
        y2 = {t: ys[(it["id"], t, 2)] for t in ("haiku", "sonnet") if (it["id"], t, 2) in ys}
        tout = {t: ans[(it["id"], t, 1)]["tout"] for t in y if (it["id"], t, 1) in ans}
        if all(t in y for t in TIERS) and it.get("pilot"):
            phase2, ipw = "full", 1.0
        elif (y.get("haiku", 0) >= 0.5) != (y.get("fable", 0) >= 0.5):
            phase2, ipw = "disagree", 1.0 if all(t in y for t in TIERS) else 0.0
        elif ek.unit("phase2", seed, it["id"]) < PHASE2_RATE:
            phase2, ipw = "sampled", 1.0 / PHASE2_RATE if all(t in y for t in TIERS) else 0.0
        else:
            phase2, ipw = "unsampled", 0.0
        graders = {}
        for gr in ("sonnet", "opus"):
            v = {t: g[(it["id"], t, 1)][gr]["y"] for t in TIERS if gr in g.get((it["id"], t, 1), {})}
            if v:
                graders[gr] = v
        ts = next((t for t in TIERS if y.get(t, 0) >= 0.5), "none")
        rows.append({"id": it["id"], "cluster": it["cluster"], "split": it["split"], "source": it["source"], "group": "chat",
                     "job": it["job"], "difficulty": it["difficulty"], "lang": it["lang"], "context": it["context"],
                     "words": it["words"], "phase2": phase2, "ipw": ipw, "y": {t: round(v, 4) for t, v in y.items()},
                     "y_soft": {t: (g[(it["id"], t, 1)]["sonnet"]["y"] + g[(it["id"], t, 1)]["opus"]["y"]) / 2.0 for t in y
                                if "sonnet" in g.get((it["id"], t, 1), {}) and "opus" in g.get((it["id"], t, 1), {})},
                     "y2": y2, "tin": ek.est_tokens(answer_prompt(it)), "tout": tout, "graders": graders,
                     "adjudicated": any("fable" in g.get((it["id"], t, 1), {}) for t in y), "t_star": ts,
                     "nonmonotone": is_nonmonotone(y), "panel": it.get("panel")})
    return rows


LAMBDA_D = ((0.5, 0.5), (0.3, 0.25), (-2.0, 0.1))     # validity >= 0.5 -> 0.5, 0.3-0.5 -> 0.25, < 0.3 -> 0.1


def panel_validity(rows, theta):
    """§8.2 on A-train: b from outcomes alone (anchors fixed); validity = Spearman(panel difficulty, t*); lambda_d from
    the table; AUC of panel difficulty for haiku failure; the ordinal thresholds tau_k, s_d and the min_tier link
    c_mt, s_mt fitted with b held at the outcome fit."""
    rows = [r for r in rows if r.get("split") == "train" and r.get("panel") and r["y"]]
    if len(rows) < 10:
        return {"n": len(rows), "note": "too few A-train items with panel labels"}
    _, b = st.fit_1pl([(r["id"], t, y, ev.cell_weight(r, t)) for r in rows for t, y in r["y"].items()], theta)
    tidx = {t: k for k, t in enumerate(TIERS)}
    ed = {r["id"]: sum(k * q for k, q in enumerate(r["panel"]["difficulty"])) for r in rows}
    validity = st.spearman([ed[r["id"]] for r in rows], [tidx.get(r["t_star"], 4) for r in rows])
    lam = next(l for lo, l in LAMBDA_D if (validity or 0.0) >= lo)
    hk = [r for r in rows if "haiku" in r["y"]]
    out = {"n": len(rows), "validity": validity, "lambda_d": lam,
           "auc_haiku_fail": st.auc([1 - r["y"]["haiku"] for r in hk], [ed[r["id"]] for r in hk])}

    def nll_ord(x):
        tau = [x[0], x[0] + math.exp(x[1]), x[0] + math.exp(x[1]) + math.exp(x[2])]
        s = math.exp(x[3])
        tot = 0.0
        for r in rows:
            ge = [1.0] + [st.sigmoid((b[r["id"]] - t) / s) for t in tau] + [0.0]
            tot -= sum(q * math.log(max(1e-12, ge[d] - ge[d + 1])) for d, q in enumerate(r["panel"]["difficulty"]))
        return tot
    x, _ = st.nelder_mead(nll_ord, [-1.5, 0.5, 0.5, 0.0])
    out.update(tau=[x[0], x[0] + math.exp(x[1]), x[0] + math.exp(x[1]) + math.exp(x[2])], s_d=math.exp(x[3]))
    mt = [r for r in rows if len(r["panel"].get("min_tier") or []) == 5]

    def nll_mt(x):
        tot = 0.0
        for r in mt:
            cdf = [0.0] + [st.sigmoid((theta[t] - b[r["id"]] - x[0]) / math.exp(x[1])) for t in TIERS]
            probs = [max(1e-12, cdf[k + 1] - cdf[k]) for k in range(4)] + [max(1e-12, 1 - cdf[4])]
            tot -= sum(q * math.log(pr) for q, pr in zip(r["panel"]["min_tier"], probs))
        return tot
    if mt:
        x, _ = st.nelder_mead(nll_mt, [0.0, 0.0])
        out.update(c_mt=x[0], s_mt=math.exp(x[1]))
    return out


def cmd_fit(a, p):
    items = {it["id"]: it for it in load_set(p, a.set)}
    if not items:
        print("no %s items yet" % a.set)
        return 1
    ek.write_json(p["validity"], validity(p, items))
    rows = build_outcomes(p, items, a.seed)
    out = p["pilot_outcomes" if a.set == "pilot" else "outcomes"]
    ek.write_jsonl(out, rows)
    theta, b = st.fit_1pl([(r["id"], t, y, ev.cell_weight(r, t)) for r in rows for t, y in r["y"].items()])
    if a.set == "pool":
        ek.write_json(p["irt"], {"model": "1PL, a = 1, theta_haiku := 0, b ~ N(0, 3^2)", "theta": theta, "a": {"chat": 1.0},
                                 "b": {k: round(v, 4) for k, v in sorted(b.items())}})
        ek.write_json(os.path.join(p["alpha"], "panel_validity.json"), panel_validity(rows, theta))
        link = sorted((it for it in items.values() if it["source"] == "verifiable" and it["split"] == "test" and it["id"] in b),
                      key=lambda it: ek.unit("linking", a.seed, it["id"]))[:150]
        ek.write_jsonl(p["linking"], [{"id": it["id"], "prompt": it["prompt"], "history": it["history"], "contract": it["contract"],
                                       "checker_py": it["checker_py"], "b": round(b[it["id"]], 4), "a": 1.0, "group": "chat"}
                                      for it in link])
    print("%d outcome rows -> %s; theta %s" % (len(rows), out, {t: round(v, 2) for t, v in theta.items()}))
    return 0


# ── pilot report (§7.1) ────────────────────────────────────────────────────────

def keyword_features(text):
    c = ek.m.classify_heuristic(text)
    sc = c.signals.get("scores", {})
    return [1.0, math.log(1 + c.signals.get("words", 0)), c.signals.get("difficulty_points", 0.0),
            c.signals.get("reasoning_demand", 0.0)] + [1.0 if c.difficulty == d else 0.0 for d in range(1, 4)] + \
        [min(4.0, sc.get(j, 0.0)) for j in ek.JOBS]


def keyword_irt(rows, texts, theta, folds=5, l2=1.0, iters=200, lr=0.5):
    """Keyword-only logistic regression: P(y_it) = sigmoid(theta_t - w.x_i) on standardized keyword features, fitted
    with item folds; returns out-of-fold b-hat {id: w.x}."""
    raw = {r["id"]: keyword_features(texts[r["id"]]) for r in rows if r["id"] in texts}
    if not raw:
        return {}
    cols = list(zip(*raw.values()))
    mu = [sum(c) / len(c) for c in cols]
    sd = [math.sqrt(sum((x - a) ** 2 for x in c) / len(c)) or 1.0 for c, a in zip(cols, mu)]
    feats = {i: [1.0] + [(x - a) / b for x, a, b in list(zip(v, mu, sd))[1:]] for i, v in raw.items()}
    fold = {i: int(ek.unit("fold", i) * folds) for i in feats}
    out = {}
    for f in range(folds):
        train = [r for r in rows if r["id"] in feats and fold[r["id"]] != f]
        w = [0.0] * len(next(iter(feats.values())))
        for _ in range(iters):
            grad = [l2 * x for x in w]
            for r in train:
                x = feats[r["id"]]
                bh = sum(a * b for a, b in zip(w, x))
                for t, y in r["y"].items():
                    e = y - st.sigmoid(theta[t] - bh)      # d(-loglik)/d(bh)
                    for k in range(len(w)):
                        grad[k] += e * x[k]
            n = max(1, len(train))
            w = [a - lr * g / n for a, g in zip(w, grad)]
        for i in feats:
            if fold[i] == f:
                out[i] = sum(a * b for a, b in zip(w, feats[i]))
    return out


def project(theta, bs, cost, rhos=(0.4, 0.5, 0.6, 0.7), n=1000, reps=4, seed=0):
    """§7.1 projection: resample items from the fitted b, simulate routers whose b-hat correlates rho with b, and
    report alpha capture against the constant hull plus the oracle headroom."""
    rng = random.Random(seed)
    mu = sum(bs) / len(bs)
    sd = math.sqrt(sum((x - mu) ** 2 for x in bs) / len(bs)) or 1.0
    vs = [cost[0] * 0.3 * (20000.0 ** (k / 29.0)) for k in range(30)]
    th = [theta[t] for t in TIERS]

    def curve(P, Y):
        pts = []
        for v in vs:
            c = s = 0.0
            for p_i, y_i in zip(P, Y):
                k, best = 0, v * p_i[0] - cost[0]
                for j in (1, 2, 3):
                    val = v * p_i[j] - cost[j]
                    if val > best:
                        k, best = j, val
                c += cost[k]
                s += y_i[k]
            pts.append((c / len(P), s / len(P)))
        return pts
    res, heads = {}, []
    for rho in rhos:
        caps = []
        for _ in range(reps):
            b = [rng.choice(bs) for _ in range(n)]
            P = [[st.sigmoid(t - x) for t in th] for x in b]
            Y = [[1.0 if rng.random() < q else 0.0 for q in row] for row in P]
            consts = [(cost[k], sum(Y[i][k] for i in range(n)) / n) for k in range(4)]
            ps = sd * math.sqrt(1 - rho ** 2)
            R = []
            for x in b:
                zh = rho * (x - mu) / sd + math.sqrt(1 - rho ** 2) * rng.gauss(0, 1)
                pm = mu + sd * rho * zh
                R.append([st.sigmoid((t - pm) / math.sqrt(1 + math.pi * ps ** 2 / 8)) for t in th])
            opts = curve(P, Y)
            h = st.envelope_auc(consts, cost[0], cost[3])
            r = st.envelope_auc(consts + curve(R, Y), cost[0], cost[3])
            o = st.envelope_auc(consts + opts, cost[0], cost[3])
            caps.append((r - h) / (o - h) if o - h > 1e-12 else 0.0)
            c_or = st.c_at(st.frontier(consts + opts), consts[3][1])
            heads.append(1 - c_or / cost[3] if c_or is not None else 0.0)
        res[rho] = sum(caps) / len(caps)
    return {"A": res, "oracle_headroom": sum(heads) / len(heads)}


def cmd_pilot_report(a, p):
    rows = ek.read_jsonl(p["pilot_outcomes"])
    if not rows:
        print("no pilot outcomes yet (alpha.py fit --set pilot)")
        return 1
    inv = ev.load_inventory(p["inventory"])
    items = ev.load_items(rows, inv)
    texts = {it["id"]: it["prompt"] for it in load_set(p, "pilot")}
    rep = {"n": len(rows)}
    rep["solve_rate"] = {t: sum(r["y"][t] for r in rows if t in r["y"]) / max(1, sum(t in r["y"] for r in rows)) for t in TIERS}
    full = [r for r in rows if all(t in r["y"] for t in TIERS)]
    rep["all_four"] = sum(all(r["y"][t] >= 0.5 for t in TIERS) for r in full) / max(1, len(full))
    rep["none"] = sum(all(r["y"][t] < 0.5 for t in TIERS) for r in full) / max(1, len(full))
    pairs = [(int(r["y"][t] >= 0.5), int(v >= 0.5)) for r in rows for t, v in (r.get("y2") or {}).items() if t in r["y"]]
    rep["repeat_kappa"] = st.cohen_kappa(pairs)
    val = ek.read_json(p["validity"]) or {}
    rep["grader_validity"] = val.get("graders")
    theta, b = st.fit_1pl([(r["id"], t, y, 1.0) for r in rows for t, y in r["y"].items()])
    tidx = {t: k for k, t in enumerate(TIERS)}
    ts = [(r, tidx.get(r["t_star"], 4)) for r in full if r.get("panel")]
    if len(ts) >= 3:
        rep["spearman_panel_difficulty_tstar"] = st.spearman([sum(k * q for k, q in enumerate(r["panel"]["difficulty"])) for r, _ in ts],
                                                             [x for _, x in ts])
        rep["spearman_panel_min_tier_tstar"] = st.spearman([sum(k * q for k, q in enumerate(r["panel"]["min_tier"])) for r, _ in ts],
                                                           [x for _, x in ts])
    ids = [r["id"] for r in rows if r["id"] in texts]
    nb = {i: sum(k * q for k, q in enumerate(ek.m.classify_rules(texts[i]).signals.get("difficulty_probs") or [0, 0, 0, 0]))
          for i in ids}
    rep["rho_nb"] = st.spearman([nb[i] for i in ids], [b[i] for i in ids])
    kw = keyword_irt([r for r in rows if r["id"] in texts], texts, theta)
    rep["rho_keyword_lr"] = st.spearman([kw[i] for i in ids if i in kw], [b[i] for i in ids if i in kw])
    cost = [sum(it["cost"][t] for it in items) / len(items) for t in TIERS]
    rep["theta"] = theta
    rep["mean_cost"] = dict(zip(TIERS, cost))
    pr = project(theta, list(b.values()), cost, n=a.sim_n, reps=a.reps, seed=a.seed)
    rep["projection"] = {"A": {str(k): round(v, 4) for k, v in pr["A"].items()}, "oracle_headroom": round(pr["oracle_headroom"], 4)}
    go = pr["A"].get(0.6, 0.0) >= 0.10 and pr["oracle_headroom"] >= 0.20
    rep["decision"] = "go" if go else "no-go"
    ek.write_json(os.path.join(p["alpha"], "pilot_report.json"), rep)
    for k, v in rep.items():
        print("%-34s %s" % (k, v))
    print("\nGO: full alpha set and outcome loss" if go else
          "\nNO-GO: alpha set at 900 items (select --total 900), panel-ordinal b, effort on sources 2-5")
    ek.ledger_append("pilot", rep["projection"], path=p["ledger"], decision=rep["decision"])
    return 0


# ── proving alpha (§9) ─────────────────────────────────────────────────────────

def texts_for(p, ids):
    cands = candidates(p)
    return {i: cands[i]["prompt"] for i in ids if i in cands}


ROUTE_CONFIGS = {"B4": {}, "R": {"decider": "utility", "objective": "balanced"},
                 "R-nb": {"decider": "utility", "objective": "balanced", "classifier": "rules"},
                 "R-lad": {"decider": "ladder", "classifier": "s1"}}
PROXY_CONFIGS = {"B4-proxy": {}, "R-proxy": {"decider": "utility", "objective": "balanced"}}


def policies(items, texts, inv, names=None, preds=()):
    """{name: (choices, P)}: constants, B0, B4/R variants through `route()`, B5, O_real, plus prediction files. R variants run
    at the s1.bin header's V0 (no wtp_usd), as production does."""
    out = {name: (ev.constant(items, t), {}) for name, t in ev.CONST.items()}
    out["B0"] = (ev.uniform(items), {})
    for name, cfg in ROUTE_CONFIGS.items():
        if names and name not in names:
            continue
        ch, ps, info = ev.route_policy(texts, inv, dict(cfg))
        if info["available"]:
            out[name] = (ch, ps)
    for name, cfg in PROXY_CONFIGS.items():   # proxy mode: server.App.plan with no upstream call
        if names and name not in names:
            continue
        ch, ps, info = ev.proxy_policy(texts, inv, cfg)
        if info["available"]:
            out[name] = (ch, ps)
    out["B5"] = (ev.ladder_gold(items), {})
    out["O_real"] = (ev.oracle_real(items), {})
    for spec in preds:
        name, _, path = spec.partition("=")
        out[name] = ev.preds_policy(ek.read_jsonl(path))
    return out


def cmd_replay(a, p):
    inv = ev.load_inventory(p["inventory"])
    rows = ek.read_jsonl(p["outcomes"])
    items = ev.load_items(rows, inv, a.split)
    if not items:
        print("no outcome data yet for split %s" % a.split)
        return 0
    texts = texts_for(p, [it["id"] for it in items])
    pols = policies(items, texts, inv, [a.policy], a.preds)
    if a.policy not in pols:
        print("policy %s is not available (decider or classifier not supported by route yet)" % a.policy)
        return 1
    ch, ps = pols[a.policy]
    s = ev.summarize(items, ch)
    print("%s on %s: n=%d success %.4f  $/100 %.4f  severe %.4f  under %.4f" % (a.policy, a.split, s["n"], s["S"], 100 * s["C"], s["severe"], s["under"]))
    if a.out:
        ek.write_jsonl(a.out, [dict({"id": i, "chosen": TIERS[v.index(max(v))]}, **({"p": ps[i]} if i in ps else {}))
                               for i, v in ch.items()])
    if a.split in ("test", "persona"):
        ek.ledger_append(a.split, {k: s[k] for k in ("S", "C", "severe")}, path=p["ledger"], policy=a.policy)
    return 0


def cmd_report(a, p):
    inv = ev.load_inventory(p["inventory"])
    items = ev.load_items(ek.read_jsonl(p["outcomes"]), inv, a.split)
    if not items:
        print("no outcome data yet for split %s" % a.split)
        return 0
    texts = texts_for(p, [it["id"] for it in items])
    pols = policies(items, texts, inv, preds=a.preds)
    res = ev.score(items, pols, a.boot, a.seed)
    res["split"], res["n_items"] = a.split, len(items)
    main = "R" if "R" in pols else "B4"
    res["main_policy"] = main
    res["slices"] = ev.slices(items, pols[main][0])
    if "B4" in pols and main != "B4":
        res["alpha_vs_B4"] = ev.alpha_ci(items, pols[main][0], a.boot, a.seed, extra_point=pols["B4"][0])
        res["mcnemar_vs_B4"] = ev.mcnemar_vs(items, pols[main][0], pols["B4"][0])
    res["mcnemar_vs_B2"] = ev.mcnemar_vs(items, pols[main][0], pols["B2"][0])
    cfg = dict(ek.m.DEFAULT_CONFIG, **(ROUTE_CONFIGS["R"] if main == "R" else {}))
    if main == "R":
        _, _, rinfo = ev.route_policy(texts, inv, ROUTE_CONFIGS["R"])
        res["heads"] = ev.heads(items, rinfo.get("posterior") or {})
    legacy = {"L%d" % k: r["prompt"] for k, r in enumerate(ek.read_jsonl(os.path.join(ek.HERE, "prompts.jsonl")))}
    res["latency_ms"] = ev.latency_profile(dict(legacy, **dict(texts, **ev.long_fixtures())), inv, cfg)
    irt = ek.read_json(p["irt"])
    if irt:
        theta, bfit = ev.b_fit(items, irt["theta"])
        rel = ev.reliability_split_half(items, irt["theta"])
        res["b_fit_reliability"] = rel
        if "R" in pols:  # the V-sweep through route() (40 values of wtp_usd on the V0 grid)
            _, _, rinfo = ev.route_policy(texts, inv, ROUTE_CONFIGS["R"])
            r_pts = []
            for v in ev.V_GRID[::3]:
                ch, _, _ = ev.route_policy(texts, inv, {"decider": "utility", "objective": "balanced", "wtp_usd": v})
                s = ev.summarize(items, ch)
                r_pts.append((s["C"], s["S"]))
            res["capture"] = ev.capture(items, r_pts, theta, bfit)
            mub = rinfo.get("mu_b") or {i: -st.logit(max(1e-6, min(1 - 1e-6, v["haiku"]))) for i, v in pols["R"][1].items()}
            mu = [(i, mub[i]) for i in bfit if i in mub]
            if len(mu) >= 3:
                rho = st.spearman([x for _, x in mu], [bfit[i] for i, _ in mu])
                res["discrimination"] = {"rho_b": rho, "rho_b_corrected": rho / math.sqrt(rel) if rho is not None and rel else None}
    print(ev.format_report(res))
    ek.write_json(os.path.join(p["alpha"], "report-%s.json" % a.split), res)
    if a.split in ("test", "persona"):
        ek.ledger_append(a.split, {k: res["policies"][main][k] for k in ("S", "C", "severe")}, path=p["ledger"],
                         policy=main, alpha=res["policies"][main]["alpha"])
    return 0


def cmd_gate(a, p):
    t0 = time.perf_counter()
    rows = ek.read_jsonl(p["outcomes"])
    if not any(r.get("split") == a.split for r in rows):
        print("alpha gate: no outcome data yet (%s has no %s rows); nothing to check" % (os.path.relpath(p["outcomes"], ek.ROOT), a.split))
        return 0
    inv = ev.load_inventory(p["inventory"])
    ids = [r["id"] for r in rows if r.get("split") == a.split]
    texts = texts_for(p, ids)
    gates = ek.read_json(p["gates"])
    base = ek.read_jsonl(p["baseline"])
    checks, cur, items = ev.gate(rows, inv, texts, gates, base, n_boot=a.boot, seed=a.seed, split=a.split)
    missing = [i for i in ids if i not in texts and any(r["id"] == i and r.get("ipw", 0) > 0 for r in rows)]
    if missing:
        checks.append(("item texts", "fail", "%d %s items have no text in the corpus files, e.g. %s" % (len(missing), a.split, missing[:3])))
    for name, status, detail in checks:
        print("%-24s %-5s %s" % (name, status.upper(), detail))
    print("decider %s; %.1f s" % (cur[2]["decider"], time.perf_counter() - t0))
    if a.update:
        n = ev.gate_update(p["gates"], p["baseline"], items, cur, texts, inv, a.split)
        print("updated %s and %s (%d rows); show the paired deltas above in the PR" % (p["gates"], p["baseline"], n))
        return 0
    return 1 if any(s == "fail" for _, s, _ in checks) else 0


def cmd_release(a, p):
    """§9.6 release gate (+ §12.4 tool-job gate). Every requirement must be verified; missing evidence fails."""
    inv = ev.load_inventory(p["inventory"])
    rows = ek.read_jsonl(p["outcomes"])
    res, fails = {}, []

    def need(name, ok, detail):
        res[name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            fails.append(name)
    test = ev.load_items(rows, inv, "test")
    texts = texts_for(p, [it["id"] for it in test])
    pols = policies(test, texts, inv, names=("B4", "R")) if test else {}
    if "R" not in pols:
        need("utility decider", False, "route() does not support decider=utility yet, or no test outcomes")
    else:
        ah = ev.alpha_ci(test, pols["R"][0], a.boot, a.seed)
        need("alpha_$ vs hull CI excludes 0", ah["lo"] is not None and ah["lo"] > 0, ah)
        ab = ev.alpha_ci(test, pols["R"][0], a.boot, a.seed, extra_point=pols["B4"][0])
        need("alpha_$ vs B4 CI excludes 0", ab["lo"] is not None and ab["lo"] > 0, ab)
        sl = ev.slices(test, pols["R"][0])
        worst = [(dim, k, v["severe"]) for dim, ks in sl.items() for k, v in ks.items() if isinstance(v, dict) and v["severe"] > 0.03]
        need("severe <= 3% in every slice n >= 30", not worst, worst[:10])
        per = ev.load_items(rows, inv, "persona")
        if per:
            pt = texts_for(p, [it["id"] for it in per])
            pch, _, _ = ev.route_policy(pt, inv, {"decider": "utility", "objective": "balanced"})
            ap = ev.alpha_ci(per, pch, 200, a.seed)
            need("persona alpha >= 0", (ap["alpha"] if ap["alpha"] is not None else ap["alpha_S"] or -1) >= 0, ap)
        else:
            need("persona alpha >= 0", False, "no persona outcomes")
        res["sensitivity"] = sensitivity(test, texts, inv)
        st_, detail = ev.v0_check(v0_path=p["v0"])
        need("s1.bin header carries the calibrated V0", st_ == "pass", detail)
    sub = subtask_accuracy(a)
    need("subtask job accuracy >= ladder - 2 pts", sub is not None and sub["s1"] >= sub["ladder"] - 0.02, sub or "no S-test labels")
    aud = audit_agreement(a.audit, p["audit_key"]) if a.audit and os.path.exists(a.audit) else {}
    rnd = aud.get("random")
    need("human audit raw agreement >= 80% (random stratum)", rnd is not None and rnd["raw"] >= 0.8, aud or "no audit CSV")
    sh = ek.read_json(a.shadow) if a.shadow else None
    need("proxy shadow >= 14 days without more bad/escalate", bool(sh) and sh.get("days", 0) >= 14
         and sh.get("bad_rate_utility", 1) <= sh.get("bad_rate_ladder", 0)
         and sh.get("escalate_rate_utility", 1) <= sh.get("escalate_rate_ladder", 0), sh or "no --shadow summary")
    res["agentic_ready"] = agentic_gate(rows, inv, p)
    for name, r in res.items():
        if isinstance(r, dict) and "ok" in r:
            print("%-52s %s  %s" % (name, "PASS" if r["ok"] else "FAIL", str(r["detail"])[:160]))
    print("agentic_ready: %s" % res["agentic_ready"])
    ek.ledger_append("release", {"fails": fails}, path=p["ledger"], detail={k: v for k, v in res.items() if k != "sensitivity"})
    if a.apply:
        dj = os.path.join(os.path.dirname(ek.S1_BIN), "decide.json")
        cfg = ek.read_json(dj)
        if cfg is not None and res["agentic_ready"].get("ok") is not None:
            cfg["agentic_ready"] = bool(res["agentic_ready"]["ok"])
            ek.write_json(dj, cfg)
            print("wrote agentic_ready=%s to %s" % (cfg["agentic_ready"], dj))
    print("RELEASE %s" % ("OK: decider utility may become the default" if not fails else "BLOCKED: %s" % "; ".join(fails)))
    return 1 if fails else 0


def sensitivity(items, texts, inv):
    """Share of decisions that flip, dS and dC for config knobs at x0.5 and x2 (printed, not gated), around the header's V0."""
    base_cfg, v0 = {"decider": "utility", "objective": "balanced"}, ev.header_v0()["balanced"]
    base, _, _ = ev.route_policy(texts, inv, base_cfg)
    b = ev.summarize(items, base)
    out = {}
    for knob, default in (("latency_usd_per_s", 0.002), ("risk", 1.0), ("wtp_usd", v0)):
        for f in (0.5, 2.0):
            ch, _, _ = ev.route_policy(texts, inv, dict(base_cfg, **{knob: default * f}))
            s = ev.summarize(items, ch)
            flips = sum(ch.get(i) != v for i, v in base.items()) / float(max(1, len(base)))
            out["%s x%g" % (knob, f)] = {"flips": round(flips, 4), "dS": round(s["S"] - b["S"], 4), "dC": s["C"] - b["C"]}
    for knob in ("stakes_loss (l)", "context_penalty (omega)", "g_max", "m_w"):
        out[knob] = "n/a: a decide.json constant, not a config key (ablate in train_s1 on alpha-dev)"
    return out


def subtask_accuracy(a):
    """Job accuracy of S1 (utility) vs the ladder on S-test items with panel labels."""
    sp = split.load_splits(getattr(a, "manifest", None) or split.MANIFEST)
    agg = {r["id"]: r for r in ek.read_jsonl(getattr(a, "aggregate", None) or os.path.join(ek.LABELS, "aggregate.jsonl"))}
    rows = [r for r in ek.read_jsonl(os.path.join(ek.HERE, "corpus", "subtasks.jsonl"))
            if sp.get(r.get("id"), (None, None))[1] == "test" and r.get("id") in agg]
    if len(rows) < 30:
        return None
    with tempfile.TemporaryDirectory() as home:
        cfg = dict(ek.m.DEFAULT_CONFIG, decider="utility")
        s1 = sum(ek.m.route(r["prompt"], harness="claude-code", inventory={}, config=cfg, env={"MAESTRO_HOME": home},
                            home=home)["job"] == label.top_job(agg[r["id"]]["job"]) for r in rows)
    lad = sum(ek.m.classify_rules(r["prompt"]).job == label.top_job(agg[r["id"]]["job"]) for r in rows)
    return {"n": len(rows), "s1": s1 / float(len(rows)), "ladder": lad / float(len(rows))}


def agentic_gate(rows, inv, p):
    """§12.4 on F-test (fixtures carry y per tier and the utility / ladder choices in preds files)."""
    fx = [r for r in rows if r.get("source") == "fixture" and r.get("split") == "test"]
    lad, utl = ek.read_jsonl(os.path.join(p["alpha"], "fixture_preds_ladder.jsonl")), \
        ek.read_jsonl(os.path.join(p["alpha"], "fixture_preds_utility.jsonl"))
    if not fx or not lad or not utl:
        return {"ok": None, "detail": "no F-test outcomes or fixture predictions yet; agentic_ready stays false"}
    items = ev.load_items(fx, inv)
    lc, _ = ev.preds_policy(lad)
    uc, up = ev.preds_policy(utl)
    sl, su = ev.summarize(items, lc), ev.summarize(items, uc)
    cells = ev.cell_lists(items, up)
    slope_ok = None
    if cells:
        vec = {}
        for pp, y, w, cl in cells:
            vec.setdefault(cl, []).append((pp, y, w))
        boots = []
        rng = random.Random(0)
        keys = list(vec)
        for _ in range(200):
            draw = [c for _ in keys for c in vec[keys[rng.randrange(len(keys))]]]
            boots.append(st.logistic_fit([st.logit(x[0]) for x in draw], [x[1] for x in draw], [x[2] for x in draw])[1])
        boots.sort()
        slope_ok = boots[4] <= 1.0 <= boots[194]
    ok = su["C"] <= sl["C"] and su["S"] >= sl["S"] - 0.05 and bool(slope_ok)
    return {"ok": ok, "detail": {"utility": {"S": su["S"], "C": su["C"]}, "ladder": {"S": sl["S"], "C": sl["C"]}, "slope_ci_has_1": slope_ok}}


def cmd_calibrate_v(a, p):
    inv = ev.load_inventory(p["inventory"])
    items = ev.load_items(ek.read_jsonl(p["outcomes"]), inv, "dev")
    if not items:
        print("no dev outcomes yet")
        return 0
    texts = texts_for(p, [it["id"] for it in items])
    _, _, info = ev.route_policy(dict(list(texts.items())[:1]), inv, {"decider": "utility"})
    if not info["available"]:
        print("route() has no utility decider yet: nothing to calibrate")
        return 0
    res = ev.calibrate_v(items, lambda obj, v: ev.route_policy(texts, inv, {"decider": "utility", "objective": obj, "wtp_usd": v})[0])
    ek.write_json(p["v0"], {"split": "dev", "V0": {k: r["V0"] for k, r in res.items()}, "V0_calibrated": True, "detail": res})
    for k, r in res.items():
        print("%-9s V0 %s  %s" % (k, r["V0"], r))
    print("wrote %s; ship it with: python evals/train_s1.py --v0 %s --patch --golden" % (p["v0"], os.path.relpath(p["v0"], ek.ROOT)))
    return 0


def cmd_pacing(a, p):
    inv = ev.load_inventory(p["inventory"])
    items = [it for it in ev.load_items(ek.read_jsonl(p["outcomes"]), inv, "test") if it["w"] > 0]
    if items and not a.synthetic:
        texts = texts_for(p, [it["id"] for it in items])
        ch, _, _ = ev.route_policy(texts, inv, {})
        tasks = [(it["y"][TIERS[ch[it["id"]].index(1.0)]], it["cost"][TIERS[ch[it["id"]].index(1.0)]] * a.usd_scale)
                 for it in items if it["id"] in ch]
        src = "alpha-test items routed by the current router"
    else:
        rng = random.Random(a.seed)
        tasks = [(1.0 if rng.random() < 0.8 else 0.0, rng.choice((0.004, 0.009, 0.02, 0.05)) * a.usd_scale)
                 for _ in range(a.synthetic or 500)]
        src = "synthetic tasks"
    res = ev.pacing(tasks, days=a.days, seed=a.seed)
    print("pacing simulation over %d days on %s (API $ x%g per task)" % (a.days, src, a.usd_scale))
    for pol, name in (("a", "subscription until exhausted, then API"), ("b", "pacing price (4.3)"), ("c", "API only")):
        r = res[pol]
        print("  (%s) %-40s success %.3f  API $%.2f  exhausted hours %d  waste 5h %.2f weekly %.2f" % (
            pol, name, r["success"], r["api_usd"], r["exhausted_hours"], r["waste_5h"], r["waste_weekly"]))
    for k, ok in res["checks"].items():
        print("  %-28s %s" % (k, "PASS" if ok else "FAIL"))
    return 0 if all(res["checks"].values()) else 1


# ── collect (§7.7): user-run anchors ───────────────────────────────────────────

def fit_theta(obs, prior_mu, prior_sd=0.5, iters=50):
    """1-D Newton MAP of a seat's theta from (y, b, a) observations with a N(prior_mu, prior_sd^2) prior; returns (theta, sd)."""
    th, h = prior_mu, 1.0 / prior_sd ** 2
    for _ in range(iters):
        g, h = -(th - prior_mu) / prior_sd ** 2, 1.0 / prior_sd ** 2
        for y, b, a in obs:
            pr = st.sigmoid(a * (th - b))
            g += a * (y - pr)
            h += a * a * pr * (1 - pr)
        step = g / h
        th += max(-2.0, min(2.0, step))
        if abs(step) < 1e-8:
            break
    return th, 1.0 / math.sqrt(h)


def ask_seat(seat, prompt, timeout=300.0):
    """One answer from a user seat: `local:<endpoint>/<model>` via `maestro-router run`, or `<provider>/<model>` (API or
    subscription CLI) through the v3 clients."""
    if seat.startswith("local:"):
        return ek.m.run_model(seat, prompt, timeout=timeout)
    from maestro import clients
    provider, _, model = seat.partition("/")
    status, _, body = clients.complete(provider, model, {"messages": [{"role": "user", "content": prompt}]}, timeout=timeout)
    if status != 200 or not isinstance(body, dict):
        raise RuntimeError("seat %s answered status %s" % (seat, status))
    return (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""


def collect(seat, link_rows, n=150, ask=ask_seat, prior=0.0, home=None, seed=0, log=print):
    rows = sorted(link_rows, key=lambda r: ek.unit("collect", seed, r["id"]))[:n]
    obs = []
    for k, r in enumerate(rows):
        try:
            text = ask(seat, answer_prompt(dict(r, contract=r.get("contract", ""))))
        except (RuntimeError, OSError) as e:
            log("  %s: no answer (%s)" % (r["id"], str(e)[:120]))
            continue
        obs.append((1.0 if ek.run_checker(r["checker_py"], text) else 0.0, r["b"], r.get("a", 1.0)))
        if (k + 1) % 25 == 0:
            log("  %d/%d answered" % (k + 1, len(rows)))
    if not obs:
        raise RuntimeError("no answers from %s" % seat)
    th, sd = fit_theta(obs, prior)
    home = home or ek.m.maestro_home()
    path = os.path.join(home, "anchors.json")
    data = ek.read_json(path) or {"version": 1, "seats": {}}
    data.setdefault("seats", {})[seat] = {"theta_chat": round(th, 4), "sd": round(max(sd, 0.25), 4), "n": len(obs),
                                          "solved": int(sum(y for y, _, _ in obs)), "at": int(time.time())}
    ek.write_json(path, data)
    return data["seats"][seat], path


def link_prior(seat, home):
    """The seat's catalog-linked chat theta from models.json (`theta: {job: [mean, sd]}`, written by discover), or None."""
    data = ek.read_json(os.path.join(home, "models.json")) or {}
    tail = seat.split("/", 1)[-1]
    for e in data.get("models") or []:
        th = e.get("theta") if isinstance(e, dict) else None
        if isinstance(th, dict) and th and (e.get("id") == seat or e.get("model") == tail):
            vals = [v[0] if isinstance(v, (list, tuple)) else v for v in th.values()]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if vals:
                return sum(vals) / len(vals)
    return None


def cmd_collect(a, p):
    rows = ek.read_jsonl(p["linking"])
    if not rows:
        print("no linking set yet (%s)" % p["linking"])
        return 1
    home = ek.m.maestro_home()
    mark = os.path.join(home, "collect_confirmed")
    if not a.yes and not os.path.exists(mark):
        ans = input("collect sends %d synthetic prompts to %s and runs the answers through the item checkers in a "
                    "sandboxed subprocess (python -I, empty env, CPU 10 s, 1 GB). Continue? [y/N] " % (min(a.n, len(rows)), a.seat))
        if ans.strip().lower() not in ("y", "yes"):
            return 1
        os.makedirs(home, exist_ok=True)
        open(mark, "w").close()
    prior = a.prior if a.prior is not None else link_prior(a.seat, home)
    if prior is None:
        print("note: no catalog-linked theta for %s in models.json (run `maestro-router discover`); prior mean 0.0" % a.seat)
        prior = 0.0
    rec, path = collect(a.seat, rows, a.n, prior=prior, seed=a.seed)
    print("%s: theta_chat %.2f (sd %.2f) from %d answers -> %s" % (a.seat, rec["theta_chat"], rec["sd"], rec["n"], path))
    return 0


# ── check (§7.2 validation of verifiable items) ────────────────────────────────

def check_item(r, runs=3):
    """[] when the item's checker passes its reference, fails an empty and a copy-the-input answer, and is stable."""
    probs = []
    if not r.get("checker_py") or not r.get("reference_answer"):
        return ["missing checker_py or reference_answer"]
    ref = [ek.run_checker(r["checker_py"], r["reference_answer"]) for _ in range(runs)]
    if not all(ref):
        probs.append("reference fails" if not any(ref) else "reference is nondeterministic")
    if ek.run_checker(r["checker_py"], ""):
        probs.append("empty answer passes")
    if ek.run_checker(r["checker_py"], r.get("prompt") or ""):
        probs.append("copy-the-input passes")
    return probs


def cmd_check(a, p):
    rows = ek.read_jsonl(p["verifiable"])
    if not rows:
        print("no verifiable items (%s)" % p["verifiable"])
        return 1
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
        res = list(pool.map(check_item, rows))
    bad = [(r["id"], pr) for r, pr in zip(rows, res) if pr]
    for rid, pr in bad:
        print("%s: %s" % (rid, "; ".join(pr)))
    print("%d items checked, %d fail validation (regenerate them)" % (len(rows), len(bad)))
    return 1 if bad else 0


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("pilot", "pilot-report", "select", "answer", "grade", "check", "fit", "replay", "report",
                                    "gate", "release", "calibrate-v", "pacing", "collect"))
    ap.add_argument("--n", type=int, default=None, help="pilot: items (200); collect: linking items (150)")
    ap.add_argument("--total", type=int, default=2400, help="select: pool size without persona (900 on no-go)")
    ap.add_argument("--persona", type=int, default=200)
    ap.add_argument("--phase", choices=("pilot", "pilot-repeat", "1", "2", "repeat"), default="1")
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--stage", choices=("check", "criteria", "llm", "adjudicate", "audit"), default="check")
    ap.add_argument("--set", choices=("pool", "pilot"), default="pool")
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--per-file", type=int, default=50, help="calls per batch file")
    ap.add_argument("--jobs", type=int, default=4, help="parallel checker processes")
    ap.add_argument("--split", default="test")
    ap.add_argument("--policy", default="B4")
    ap.add_argument("--preds", action="append", default=[], metavar="NAME=FILE", help="extra policy from a predictions file")
    ap.add_argument("--out")
    ap.add_argument("--update", action="store_true", help="gate: rewrite gates.json and baseline_preds.jsonl")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit", help="release: the user's filled audit CSV")
    ap.add_argument("--shadow", help="release: proxy shadow summary JSON")
    ap.add_argument("--apply", action="store_true", help="release: write agentic_ready to data/decide.json")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--usd-scale", type=float, default=10.0, help="pacing: API $ per simulated task = item $ x this")
    ap.add_argument("--sim-n", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--seat")
    ap.add_argument("--prior", type=float, help="collect: prior mean theta (default 0)")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--alpha-dir")
    ap.add_argument("--work")
    ap.add_argument("--verifiable")
    ap.add_argument("--corpus-file")
    ap.add_argument("--persona-file")
    ap.add_argument("--aggregate")
    ap.add_argument("--manifest")
    ap.add_argument("--ledger")
    a = ap.parse_args(argv)
    p = paths(a)
    if a.cmd == "pilot":
        a.n = a.n or 200
    if a.cmd == "collect":
        a.n = a.n or 150
        if not a.seat:
            ap.error("collect needs --seat")
    if a.cmd == "gate" and a.split != "test":
        ap.error("the CI gate runs on --split test")
    return globals()["cmd_" + a.cmd.replace("-", "_")](a, p)


if __name__ == "__main__":
    sys.exit(main())
