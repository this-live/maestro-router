#!/usr/bin/env python3
"""Measure "better Jev" (spec §11.4): typed questions answered by S1 natively, by the systemone delegate, and by Jev.

    python evals/typed_eval.py build --labels LABELS.jsonl --items ITEMS.jsonl [--outcomes evals/alpha/outcomes.jsonl]
                                     [--split test] [--out evals/typed_eval.jsonl]
    python evals/typed_eval.py [run] [--file evals/typed_eval.jsonl] [--delegate] [--jev] [--limit N] [--json]

build: one typed question per (item, field) from the aggregated panel labels of the test split (job choice, difficulty score,
needs_tools and self_contained noul; schema §6.3, soft or hard), plus one seat-choice question per alpha-test item whose answer
is the cheapest tier that passed (`t_star`). ITEMS rows: {"id", "prompt" (or "text"), "history"?}. Rows are
{"id", "kind", "name", "state", "question", "answer"}, the format `maestro-router decide --calibrate` reads.

run: accuracy, ECE (10 bins on the confidence of the predicted answer) and p50/p99 latency per system and question kind. S1 is
always run. --delegate asks the configured systemone_delegate (your own served model). --jev asks Jev when TYPESAFE_API_KEY is
set (answers cached in evals/.typed_eval_jev.json). Laya is run by the user elsewhere. Reported, not gated.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "skills", "maestro", "scripts"))
from maestro import router as R  # noqa: E402
from maestro import systemone as SO  # noqa: E402

LEVELS = ["trivial: any small model gets it right first try", "routine: a competent engineer gets it right without research",
          "hard: careful multi-step reasoning, several files or subtle constraints", "frontier: expert or novel work"]
SEATS = ["claude-code:haiku", "claude-code:sonnet", "claude-code:opus", "claude-code:fable"]
QUESTIONS = {
    "job": {"type": "choice", "instructions": "Which job is the task in state.task?", "criteria": dict(R._JEV_JOB_CRITERIA)},
    "difficulty": {"type": "score", "instructions": "How difficult is the task in state.task for a capable model?", "criteria": LEVELS},
    "needs_tools": {"type": "noul", "instructions": "Does doing it well require reading or writing files, running commands or browsing?"},
    "self_contained": {"type": "noul", "instructions": "Is everything needed to complete the task in the text?"},
    "model": {"type": "choice", "instructions": "Choose the cheapest model that will get the task right.",
              "criteria": {s: "%s tier, Claude Code subagent" % t for s, t in zip(SEATS, R.TIERS)}},
}


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def top(v):
    """The hard label of a soft one: a {label: p} dict, a list of level probabilities, or a number."""
    if isinstance(v, dict):
        return max(v, key=v.get)
    if isinstance(v, list):
        return max(range(len(v)), key=v.__getitem__)
    return int(round(v)) if isinstance(v, (int, float)) else None


def build(a):
    items = {r["id"]: r for r in jsonl(a.items)}
    out = []
    for lab in jsonl(a.labels):
        it = items.get(lab.get("id"))
        if not it or lab.get("split", a.split) != a.split:
            continue
        state = {"task": it.get("prompt") or it.get("text") or "", "history": it.get("history") or []}
        for kind, field in (("job", "job"), ("difficulty", "difficulty"), ("needs_tools", "needs_tools"), ("self_contained", "self_contained")):
            gold = top(lab.get(field))
            if gold is not None:
                out.append({"id": lab["id"], "kind": kind, "name": kind, "state": state, "question": QUESTIONS[kind], "answer": gold})
    for o in jsonl(a.outcomes) if a.outcomes and os.path.exists(a.outcomes) else []:
        it = items.get(o.get("id"))
        if it and o.get("split") == a.split and o.get("t_star") in ("haiku", "sonnet", "opus", "fable"):
            out.append({"id": o["id"], "kind": "model", "name": "model", "state": {"task": it.get("prompt") or it.get("text") or ""},
                        "question": QUESTIONS["model"], "answer": "claude-code:" + o["t_star"]})
    with open(a.out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in out)
    print("wrote %d typed questions to %s" % (len(out), a.out))
    return out


def predicted(q, ans):
    """(predicted answer, its confidence) of a Jev answer."""
    if q["type"] == "noul":
        return int(ans["noul"] >= 0.5), max(ans["noul"], 1 - ans["noul"])
    p = ans["probabilities"]
    k = max(p, key=p.get)
    return (k if q["type"] == "choice" else int(k)), p[k]


def ece(pairs, bins=10):
    n, tot = len(pairs), 0.0
    for b in range(bins):
        cell = [(c, ok) for c, ok in pairs if b / bins <= c < (b + 1) / bins or (b == bins - 1 and c == 1.0)]
        if cell:
            tot += len(cell) / float(n) * abs(sum(c for c, _ in cell) / len(cell) - sum(ok for _, ok in cell) / len(cell))
    return tot


def systems(a, ctx):
    """{name: ask(row) -> (status, answer)}."""
    def native(r):
        st, out = SO.handle({"model": "maestro-s1", "state": r["state"], "questions": {r["name"]: r["question"]}}, ctx)
        return st, out.get("answers", {}).get(r["name"])

    def delegate(r):     # a name with no alias and no intent is never native: the delegate answers it
        st, out = SO.handle({"model": "maestro-s1+llm", "state": r["state"], "questions": {"q_" + r["name"]: r["question"]}}, ctx)
        return st, out.get("answers", {}).get("q_" + r["name"])
    out = {"s1": native}
    if a.delegate:
        out["delegate"] = delegate
    key = os.environ.get("TYPESAFE_API_KEY")
    if a.jev and key:
        path = os.path.join(HERE, ".typed_eval_jev.json")
        cache = R._read_json(path) or {}

        def jev(r):
            k = json.dumps([r["state"], r["question"]], sort_keys=True)
            if k not in cache:
                body = {"model": R.JEV_MODEL, "state": r["state"], "questions": {r["name"]: r["question"]}}
                st, raw = R._http_fetch(R.JEV_URL, {"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                                        json.dumps(body).encode("utf-8"), 30)
                if st != 200:
                    return st, None
                cache[k] = json.loads(raw)["answers"][r["name"]]
                with open(path, "w") as f:
                    json.dump(cache, f)
            return 200, cache[k]
        out["jev"] = jev
    elif a.jev:
        print("jev: TYPESAFE_API_KEY is not set; skipped", file=sys.stderr)
    return out


def run(a, ctx=None):
    rows = jsonl(a.file)[:a.limit or None]
    report = {}
    for name, ask in systems(a, SO.context(ctx)).items():
        by = {}
        for r in rows:
            t0 = time.perf_counter()
            st, ans = ask(r)
            ms = (time.perf_counter() - t0) * 1000
            c = by.setdefault(r["kind"], {"n": 0, "errors": 0, "pairs": [], "ms": []})
            c["n"] += 1
            if st != 200 or not ans:
                c["errors"] += 1
                continue
            pred, conf = predicted(r["question"], ans)
            c["pairs"].append((conf, float(pred == r["answer"])))
            c["ms"].append(ms)
        report[name] = {k: {"n": c["n"], "errors": c["errors"], "accuracy": round(sum(ok for _, ok in c["pairs"]) / max(1, len(c["pairs"])), 4),
                            "ece": round(ece(c["pairs"]), 4) if c["pairs"] else None,
                            "p50_ms": round(sorted(c["ms"])[len(c["ms"]) // 2], 2) if c["ms"] else None,
                            "p99_ms": round(sorted(c["ms"])[min(len(c["ms"]) - 1, int(0.99 * len(c["ms"])))], 2) if c["ms"] else None}
                        for k, c in sorted(by.items())}
    if a.json:
        print(json.dumps(report, indent=1))
    else:
        for name, kinds in report.items():
            for k, m in kinds.items():
                print("%-9s %-15s n=%-5d err=%-4d acc=%.3f ece=%s p50=%s ms p99=%s ms" % (name, k, m["n"], m["errors"], m["accuracy"],
                                                                                          m["ece"], m["p50_ms"], m["p99_ms"]))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", nargs="?", default="run", choices=("build", "run"))
    ap.add_argument("--labels")
    ap.add_argument("--items")
    ap.add_argument("--outcomes", default=os.path.join(HERE, "alpha", "outcomes.jsonl"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=os.path.join(HERE, "typed_eval.jsonl"))
    ap.add_argument("--file", default=os.path.join(HERE, "typed_eval.jsonl"))
    ap.add_argument("--delegate", action="store_true")
    ap.add_argument("--jev", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        if not (a.labels and a.items):
            ap.error("build needs --labels and --items")
        build(a)
    else:
        run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
