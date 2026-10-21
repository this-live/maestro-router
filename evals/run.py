#!/usr/bin/env python3
"""Score a Maestro classifier against evals/prompts.jsonl.

    python evals/run.py --classifier rules|s1 [--split dev|test|all] [--json]
                        [--min-job-acc 0.7] [--min-category-acc 0.7] [--min-category-within1 0.9]
                        [--max-under 0.2] [--max-severe-under 0.05]

Splits are by line index of the prompts file: dev = even indexes (0, 2, 4...), test = odd.
Tune on dev only; report test once. Default is all rows.

Each prompts.jsonl line: {"prompt": ..., "job": <one of maestro.JOBS>, "difficulty": 0-3}.
Category accuracy compares the category the ladder gives for the expected label with the
predicted one, on the cloud ladder (a `local` cell counts as fast). For rules that is the ladder
cell of the predicted job/difficulty (with confidence escalation). For jev it is the category of the
model Jev chose, routed through a synthetic inventory that has Claude Code at all four tiers.
Thresholds are fractions (0.7) or percents (70); the exit code is 1 when a gate fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "skills", "maestro", "scripts"))
from maestro import router as m  # noqa: E402


def pct(x):
    return x / 100.0 if x > 1 else x


def percentile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else 0.0


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("job") in m.JOBS and r.get("difficulty") in (0, 1, 2, 3):
                    rows.append(r)
    return rows


def category(cell):
    """The cloud-ladder category of a ladder cell: `local` counts as fast."""
    return m.tier_of(cell)


def make_predict_rules(variant="rules"):
    """rules = what ships (learned model + heuristics); heuristic = keyword rules only;
    model = Naive Bayes alone (no heuristic scores added)."""
    if variant == "heuristic":
        classify = m.classify_heuristic
    elif variant == "model":
        m.MODEL_CLASS_W = m.MODEL_DIFF_W = 0.0
        classify = m.classify_rules
    else:
        classify = m.classify_rules

    def predict(prompt):
        cls = classify(prompt)
        return cls, category(m.target_tier(cls, {})), bool(cls.note)
    return predict


def make_predict_s1():
    """S1's legacy decode (job and difficulty from the posterior, spec §3.6), the classifier route uses with `classifier: s1`."""
    from maestro import s1

    def predict(prompt):
        cls = m.classify_rules(prompt)
        cls = s1.legacy(s1.posterior(prompt, cls), cls)
        return cls, category(m.target_tier(cls, {})), bool(cls.note)
    return predict


def make_predict_jev(key):
    import tempfile
    home = tempfile.mkdtemp(prefix="maestro-eval-")  # keep the eval cache out of ~/.maestro
    env = {"TYPESAFE_API_KEY": key, "MAESTRO_HOME": home}
    cfg = m.load_config(home)
    inv = {"harnesses": {}, "local_models": [], "default_models": {}}

    def predict(prompt):
        d = m.route(prompt, harness="claude-code", classifier="jev", inventory=inv, config=cfg, env=env, home=home)
        cls = m.Classification(d["job"], d["difficulty"], d["confidence"])
        return cls, category(d["category"]), "jev" not in d
    return predict


def split_rows(rows, split):
    """dev = even line index, test = odd line index, all = everything."""
    if split == "dev":
        return rows[0::2]
    if split == "test":
        return rows[1::2]
    return rows


def evaluate(rows, predict):
    n = len(rows)
    job_ok = diff_ok = cat_ok = fallbacks = 0
    delta = Counter()  # predicted category index - expected category index
    abs_err = 0
    lat = []
    conf = defaultdict(Counter)
    support = Counter()
    for r in rows:
        t0 = time.perf_counter()
        pred, pred_cat, fell_back = predict(r["prompt"])
        lat.append((time.perf_counter() - t0) * 1000)
        fallbacks += 1 if fell_back else 0
        exp = m.Classification(r["job"], r["difficulty"], 1.0)
        support[exp.job] += 1
        conf[exp.job][pred.job] += 1
        job_ok += pred.job == exp.job
        diff_ok += pred.difficulty == exp.difficulty
        abs_err += abs(pred.difficulty - exp.difficulty)
        exp_cat = category(m.target_tier(exp, {}))
        cat_ok += pred_cat == exp_cat
        delta[m.TIERS.index(pred_cat) - m.TIERS.index(exp_cat)] += 1
    per_job = {}
    for c in m.JOBS:
        if support[c]:
            wrong = {p: k for p, k in conf[c].items() if p != c}
            per_job[c] = {"n": support[c], "recall": round(conf[c][c] / support[c], 3), "confused_with": wrong}
    return {
        "n": n,
        "job_accuracy": round(job_ok / n, 4),
        "difficulty_exact": round(diff_ok / n, 4),
        "difficulty_mae": round(abs_err / n, 4),
        "category_accuracy": round(cat_ok / n, 4),
        "category_within_1": round(sum(k for d, k in delta.items() if abs(d) <= 1) / n, 4),
        "under_routed": round(sum(k for d, k in delta.items() if d <= -1) / n, 4),
        "severe_under_routed": round(sum(k for d, k in delta.items() if d <= -2) / n, 4),
        "category_delta": {str(d): delta[d] for d in sorted(delta)},
        "latency_ms": {"p50": round(percentile(lat, 0.5), 3), "p95": round(percentile(lat, 0.95), 3)},
        "fallbacks": fallbacks,
        "per_job": per_job,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classifier", choices=("rules", "s1", "jev"), default="rules")
    ap.add_argument("--variant", choices=("rules", "heuristic", "model"), default="rules",
                    help="rules classifier only: shipped combo, heuristics alone, or the learned model alone")
    ap.add_argument("--prompts", default=os.path.join(HERE, "prompts.jsonl"))
    ap.add_argument("--split", choices=("dev", "test", "all"), default="all",
                    help="dev = even line index, test = odd (default: all)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--min-job-acc", type=float)
    ap.add_argument("--min-category-acc", type=float)
    ap.add_argument("--min-category-within1", type=float, help="gate: category within +-1 must be >= this")
    ap.add_argument("--max-under", type=float, help="gate: under-routed (<= -1) must be <= this")
    ap.add_argument("--max-severe-under", type=float, help="gate: severely under-routed (<= -2) must be <= this")
    a = ap.parse_args(argv)

    rows = split_rows(load(a.prompts), a.split)
    if not rows:
        print("no prompts in %s" % a.prompts, file=sys.stderr)
        return 2
    if a.classifier == "jev":
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            print("TYPESAFE_API_KEY is required for --classifier jev", file=sys.stderr)
            return 2
        predict = make_predict_jev(key)
    elif a.classifier == "s1":
        predict = make_predict_s1()
    else:
        predict = make_predict_rules(a.variant)
    res = evaluate(rows, predict)
    res["classifier"] = a.classifier
    res["split"] = a.split
    res["variant"] = a.variant if a.classifier == "rules" else a.classifier

    if a.json:
        print(json.dumps(res, indent=2))
    else:
        print("classifier=%s  split=%s  n=%d" % (a.classifier, a.split, res["n"]))
        print("job accuracy        %.1f%%" % (100 * res["job_accuracy"]))
        print("difficulty exact    %.1f%%   MAE %.2f" % (100 * res["difficulty_exact"], res["difficulty_mae"]))
        print("category accuracy   %.1f%%" % (100 * res["category_accuracy"]))
        print("category within +-1 %.1f%%" % (100 * res["category_within_1"]))
        print("under-routed (<=-1) %.1f%%" % (100 * res["under_routed"]))
        print("severe under (<=-2) %.1f%%" % (100 * res["severe_under_routed"]))
        print("category delta (pred-exp) " + "  ".join("%+d:%d" % (int(d), k) for d, k in res["category_delta"].items()))
        print("latency p50/p95     %.3f / %.3f ms" % (res["latency_ms"]["p50"], res["latency_ms"]["p95"]))
        if res["fallbacks"]:
            print("fell back to rules  %d" % res["fallbacks"])
        print("\nper job (recall, most common confusions):")
        for c, s in res["per_job"].items():
            worst = sorted(s["confused_with"].items(), key=lambda kv: -kv[1])[:3]
            print("  %-9s n=%-3d recall %.2f  %s" % (c, s["n"], s["recall"], ", ".join("%s x%d" % kv for kv in worst)))

    failed = []
    if a.min_job_acc is not None and res["job_accuracy"] < pct(a.min_job_acc):
        failed.append("job accuracy %.3f < %.3f" % (res["job_accuracy"], pct(a.min_job_acc)))
    if a.min_category_acc is not None and res["category_accuracy"] < pct(a.min_category_acc):
        failed.append("category accuracy %.3f < %.3f" % (res["category_accuracy"], pct(a.min_category_acc)))
    if a.min_category_within1 is not None and res["category_within_1"] < pct(a.min_category_within1):
        failed.append("category within 1 %.3f < %.3f" % (res["category_within_1"], pct(a.min_category_within1)))
    if a.max_under is not None and res["under_routed"] > pct(a.max_under):
        failed.append("under-routed %.3f > %.3f" % (res["under_routed"], pct(a.max_under)))
    if a.max_severe_under is not None and res["severe_under_routed"] > pct(a.max_severe_under):
        failed.append("severe under-routed %.3f > %.3f" % (res["severe_under_routed"], pct(a.max_severe_under)))
    for f in failed:
        print("FAIL: " + f, file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
