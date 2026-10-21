#!/usr/bin/env python3
"""Live outcome evaluation of Maestro routing against the real discovered pool.

    python evals/outcome.py [--port 18941] [--tasks evals/outcome_tasks.jsonl]
                            [--out evals/outcome-results.jsonl] [--only id,...]

For each task: (1) ask serve `maestro/auto` (the routed choice), (2) ask every
pool model directly. A route is ADEQUATE when the chosen model succeeds and no
strictly cheaper tier also succeeds; SEVERE under-route when the chosen model
fails but a stronger tier succeeds; OVER-route when a cheaper tier succeeds but
was not chosen. Cost tiers: local=0, fast CLI=1, strong/frontier CLI=2.

Results append to the out file (JSONL); rerunning skips task/model pairs that
are already recorded, so an interrupted run resumes. Grading is programmatic
(contains/all_of/any_of/max_words/json_equals/python asserts) — no LLM judge.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request

POOL = [
    ("local:bob-llama/qwen35-9b", 0),
    ("local:sparky-vllm/nemotron-3.5-lightning", 0),
    ("cli:claude/haiku", 1),
    ("cli:codex/gpt-6-luna", 1),
    ("cli:claude/sonnet", 2),
    ("cli:codex/gpt-6.1-sol", 2),
]


def grade(check, text):
    if text is None:
        return False
    t = text.strip()
    kind = check["type"]
    if kind == "contains":
        return check["value"].lower() in t.lower()
    if kind == "all_of":
        return all(v.lower() in t.lower() for v in check["values"])
    if kind == "any_of":
        return any(v.lower() in t.lower() for v in check["values"])
    if kind == "max_words":
        return len(t.split()) <= check["value"] and len(t) > 0
    if kind == "json_equals":
        m = re.search(r"\[.*\]", t, re.S)
        try:
            return json.loads(m.group(0)) == check["value"]
        except Exception:
            return False
    if kind == "python":
        code = re.sub(r"^```(?:python)?\s*|\s*```$", "", t, flags=re.M).strip()
        prog = code + "\n" + "\n".join(check.get("asserts") or [])
        try:
            r = subprocess.run([sys.executable, "-c", prog], capture_output=True, timeout=20)
            return r.returncode == 0
        except Exception:
            return False
    return False


def chat(base, model, prompt, timeout=180):
    req = urllib.request.Request(base + "/v1/chat/completions",
        data=json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        d = json.loads(r.read())
        msg = (d.get("choices") or [{}])[0].get("message") or {}
        return d.get("model"), msg.get("content"), round(time.time() - t0, 1), None
    except Exception as e:
        return None, None, round(time.time() - t0, 1), str(e)[:120]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n)))
    return ((c - m) / d, (c + m) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18941)
    ap.add_argument("--tasks", default=os.path.join(os.path.dirname(__file__), "outcome_tasks.jsonl"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "outcome-results.jsonl"))
    ap.add_argument("--only", default="")
    ap.add_argument("--rerun-auto", action="store_true", help="drop recorded maestro/auto rows and re-ask them (after a tuning change)")
    args = ap.parse_args()
    if args.rerun_auto and os.path.exists(args.out):
        kept = [l for l in open(args.out) if '"maestro/auto"' not in l]
        open(args.out, "w").writelines(kept)
        print(f"rerun-auto: dropped {sum(1 for _ in open(args.out)) and ''}auto rows, kept {len(kept)}")

    tasks = [json.loads(l) for l in open(args.tasks) if l.strip()]
    if args.only:
        keep = set(args.only.split(","))
        tasks = [t for t in tasks if t["id"] in keep]
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                r = json.loads(l)
                done.add((r["task"], r["model"]))
            except Exception:
                pass
    base = f"http://127.0.0.1:{args.port}"
    out = open(args.out, "a")
    for task in tasks:
        for model, tier in [("maestro/auto", None)] + POOL:
            if (task["id"], model) in done:
                continue
            served, content, secs, err = chat(base, model, task["prompt"])
            rec = {"task": task["id"], "job": task["job"], "difficulty": task["difficulty"],
                   "model": model, "served_by": served, "tier": tier,
                   "ok": grade(task["check"], content) if content is not None else False,
                   "secs": secs, "err": err, "sample": (content or "")[:160]}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(rec["task"], model, "->", served, "ok" if rec["ok"] else "FAIL", secs, flush=True)
    out.close()

    rows = [json.loads(l) for l in open(args.out)]
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], {})[r["model"]] = r
    adequate = severe = over = failed_all = 0
    judged = 0
    tier_of = dict(POOL)
    tier_of.update({"cli:claude/opus": 3, "cli:codex/gpt-6-astra": 3, "cli:claude/fable": 2})
    for tid, res in by_task.items():
        routed = res.get("maestro/auto")
        if not routed or routed.get("served_by") not in tier_of:
            continue
        judged += 1
        rt = tier_of[routed["served_by"]]
        succ_tiers = [tier_of[m] for m, r in res.items() if m != "maestro/auto" and r["ok"]]
        if routed["ok"] and (not succ_tiers or min(succ_tiers) >= rt):
            adequate += 1
        elif routed["ok"]:
            over += 1
        elif any(t > rt for t in succ_tiers):
            severe += 1
        elif not succ_tiers:
            failed_all += 1
        else:
            severe += 1
    print(json.dumps({"judged": judged, "adequate_or_better": adequate + over, "adequate_rate": (adequate + over) / judged if judged else 0,
                      "wilson95": wilson(adequate + over, judged), "severe_under": severe,
                      "severe_rate": severe / judged if judged else 0, "over_route_waste": over,
                      "all_models_failed": failed_all}, indent=1))


if __name__ == "__main__":
    main()
