#!/usr/bin/env python3
"""Rebuild skills/maestro/scripts/maestro/maestro_model.json from evals/train.jsonl (deterministic, stdlib only).

    python evals/train.py [--train evals/train.jsonl] [--out skills/maestro/scripts/maestro/maestro_model.json]

The model is two multinomial Naive Bayes classifiers (job, difficulty; one shared feature table) over the binary features
from maestro.text_features(): lowercased word unigrams+bigrams of the instruction (fenced blocks removed)
plus structural flags. It never sees evals/prompts.jsonl, which is only for scoring.
Each feature stores one weight per job then 4 difficulty weights: log-probabilities centered per feature
(only differences between labels matter), quantized to integers of 0.5 nats. Rare features are dropped
to keep the file small. The JSON also has "jd", log P(difficulty | job) from the counts. It records the job list it was trained on ("jobs"); maestro ignores a model
whose list does not match maestro.JOBS, so retraining after changing the jobs is all that is needed.
Each train.jsonl line: {"prompt": ..., "job": <one of maestro.JOBS>, "difficulty": 0-3}.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "skills", "maestro", "scripts")
sys.path.insert(0, SCRIPTS)
from maestro import router as m  # noqa: E402

ALPHA = 0.1        # additive smoothing
MIN_COUNT = 3      # drop features seen in fewer than this many training prompts
SCALE = 0.5        # weights are stored as integers in units of SCALE nats


def fit(rows):
    """One feature table: for every feature, one weight per job followed by 4 difficulty weights."""
    labelings = ((lambda r: m.JOBS.index(r["job"]), len(m.JOBS)),
                 (lambda r: r["difficulty"], 4))
    count, total = {}, [[0] * n for _, n in labelings]
    for r in rows:
        ys = [key(r) for key, _ in labelings]
        for t, y in zip(total, ys):
            t[y] += 1
        for f in m.text_features(m._signals(r["prompt"])):
            c = count.setdefault(f, [[0] * n for _, n in labelings])
            for part, y in zip(c, ys):
                part[y] += 1
    weights = {}
    for f in sorted(count):
        if sum(count[f][0]) < MIN_COUNT:
            continue
        w = []
        for part, tot in zip(count[f], total):
            lp = [math.log((k + ALPHA) / (t + 2 * ALPHA)) for k, t in zip(part, tot)]
            mean = sum(lp) / len(lp)  # only differences between labels matter
            w += [int(round((x - mean) / SCALE)) for x in lp]
        if any(w):
            weights[f] = w
    prior = [round(math.log((t + ALPHA) / (len(rows) + ALPHA * len(tot))), 2) for tot in total for t in tot]  # smoothed: a job may have no rows yet
    return prior, weights


def job_difficulty(rows):
    """log P(difficulty | job), add-one smoothed: how hard each job usually is (a job may have no rows yet)."""
    jd = [[0] * 4 for _ in m.JOBS]
    for r in rows:
        jd[m.JOBS.index(r["job"])][r["difficulty"]] += 1
    return [[round(math.log((k + 1) / (sum(row) + 4)), 2) for k in row] for row in jd]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=os.path.join(HERE, "train.jsonl"))
    ap.add_argument("--out", default=os.path.join(SCRIPTS, "maestro", m.MODEL_FILE))
    a = ap.parse_args(argv)
    with open(a.train, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = [r for r in rows if r.get("job") in m.JOBS and r.get("difficulty") in (0, 1, 2, 3)]
    if not rows:
        print("no rows with a valid `job` label in %s (relabel the corpus first)" % a.train, file=sys.stderr)
        return 2
    prior, weights = fit(rows)
    model = {"version": 2, "jobs": list(m.JOBS), "n": len(rows), "scale": SCALE, "prior": prior, "jd": job_difficulty(rows), "w": weights}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(model, f, separators=(",", ":"), sort_keys=True)
        f.write("\n")
    print("trained on %d prompts; %d features; wrote %s (%d bytes)" % (
        len(rows), len(weights), a.out, os.path.getsize(a.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
