"""The shipped Naive Bayes classifier (maestro_model.json), moved out of router.py; `router` re-exports every name.

evals/train.py rebuilds the weights from evals/train.jsonl. If the file is missing, or was trained on a different job list, the
router falls back to its keyword heuristics. Job and difficulty are scored separately, then decoded together so the chosen
(job, difficulty) minimises expected tier error (under-routing costs UNDER_COST x).

The knobs (MODEL_FILE, MODEL_T, MODEL_TD, MODEL_CLASS_W, MODEL_DIFF_W, MODEL_TOOLS_B, UNDER_COST) stay in router.py and are read
from there at call time, so `router.MODEL_T = ...` (what evals and tests patch) still takes effect. Stdlib only, no I/O but the
one model file read.
"""
from __future__ import annotations

import json
import math
import os
import sys

from . import router as R

_MODEL = []          # lazily filled with [model or None]
_MODEL_WHY = []      # why a model file on disk was not used, if it was not


def model_problem(model):
    """Why these weights cannot be used with the current JOBS, or "" when they can."""
    if not isinstance(model, dict) or not isinstance(model.get("prior"), list) or not isinstance(model.get("w"), dict):
        return "malformed"
    if model.get("jobs") != list(R.JOBS) or len(model["prior"]) != len(R.JOBS) + 4:
        return "was trained on %s" % ("a different job list (%s)" % ", ".join(model["jobs"]) if model.get("jobs")
                                      else "the v2.0 task classes, not the current jobs")
    jd = model.get("jd")
    if jd is not None and (not isinstance(jd, list) or len(jd) != len(R.JOBS) or any(not isinstance(r, list) or len(r) != 4 for r in jd)):
        return "malformed"
    return ""


def load_model():
    """The weights next to router.py (repo, skill install) or in <prefix>/share/maestro (pip install).
    Weights trained on a different job list are ignored (reason in _MODEL_WHY) and the heuristics are used."""
    if not _MODEL:
        here = os.path.dirname(os.path.abspath(R.__file__))
        _MODEL.append(None)
        del _MODEL_WHY[:]
        for d in (here, os.path.join(sys.prefix, "share", "maestro"), os.path.join(_user_base(), "share", "maestro")):
            try:
                with open(os.path.join(d, R.MODEL_FILE), encoding="utf-8") as f:
                    model = json.load(f)
            except (OSError, ValueError):
                continue
            why = model_problem(model)
            if why:
                _MODEL_WHY.append("%s %s; using heuristics until it is retrained" % (R.MODEL_FILE, why))
            else:
                _MODEL[0] = model
            break
    return _MODEL[0]


def _user_base():
    return os.environ.get("PYTHONUSERBASE") or os.path.join(os.path.expanduser("~"), ".local")


def _softmax(xs):
    hi = max(xs)
    es = [math.exp(x - hi) for x in xs]
    z = sum(es)
    return [e / z for e in es]


def _nb_logits(model, feats):
    """Naive Bayes log-scores: one value per job, then 4 difficulty values (one shared feature table)."""
    s = [0] * len(model["prior"])
    for f in feats:
        w = model["w"].get(f)
        if w:
            s = [a + b for a, b in zip(s, w)]
    return [p + model["scale"] * a for p, a in zip(model["prior"], s)]


# risk is measured over the cloud ladder; a `local` cell counts as fast
_RISK_TIER = [[R.TIERS.index(R.tier_of(R.DEFAULT_LADDER[c][d])) for d in range(4)] for c in R.JOBS]


def classify_model(text, model):
    """Job + difficulty from Naive Bayes plus the heuristic signals; tier-risk-aware decoding.
    Weights that do not match the current JOBS are ignored (heuristics, with a signal saying why)."""
    text = text or ""
    why = model_problem(model)
    if why:
        c = R.classify_heuristic(text)
        c.signals["model_ignored"] = why
        return c
    sg = R._signals(text)
    feats = R.text_features(sg)
    h_job, _ = R._pick(sg["scores"], len(sg["toks"]))
    h_diff, pts = R._difficulty(sg["lower"], sg["demand"], h_job)
    logits = _nb_logits(model, feats)
    nj, jobs = len(R.JOBS), R.JOBS
    pc = _softmax([a / R.MODEL_T + R.MODEL_CLASS_W * sg["scores"][c] + (R.MODEL_TOOLS_B if c in R.TOOL_JOBS else 0.0)
                   for a, c in zip(logits, jobs)])
    # difficulty given the job: log P(d | job) from the training counts ("jd"), plus the word evidence
    # (the difficulty logits without their prior) and a bonus for the heuristic difficulty
    jd = model.get("jd") or [model["prior"][nj:]] * nj
    lik = [(a - p) / R.MODEL_TD for a, p in zip(logits[nj:], model["prior"][nj:])]
    lik[h_diff] += R.MODEL_DIFF_W
    pdj = [_softmax([a + b for a, b in zip(row, lik)]) for row in jd]
    tier = _RISK_TIER
    p_tier = [0.0] * 4
    for i in range(nj):
        for d in range(4):
            p_tier[tier[i][d]] += pc[i] * pdj[i][d]
    risk = [sum(p * (R.UNDER_COST * (k - t) if k > t else t - k) for k, p in enumerate(p_tier)) for t in range(4)]
    best = min(range(4), key=risk.__getitem__)
    _, i, d = max((pc[i] * pdj[i][d], i, d) for i in range(nj) for d in range(4) if tier[i][d] == best)
    job = jobs[i]
    return R.Classification(
        job, d, max(0.35, min(0.95, p_tier[best])), R._needs(text, sg["lower"], job, sg["scores"]), "rules",
        {"model": True, "job_probs": {c: round(p, 3) for c, p in zip(jobs, pc) if p >= 0.05},
         "difficulty_probs": [round(p, 3) for p in pdj[i]], "heuristic": [h_job, h_diff],
         "scores": {k: v for k, v in sg["scores"].items() if v},   # kept so S1 does not scan the text again (spec §2.7)
         "difficulty_points": round(pts, 2), "words": len(sg["toks"]), "material_chars": sg["mat"]})
