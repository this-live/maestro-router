#!/usr/bin/env python3
"""Train S1, Maestro's tiny decision model, and export skills/maestro/scripts/maestro/data/s1.bin (S1 spec §8, App. A).

    python evals/train_s1.py [--labels evals/train.jsonl] [--out <pkg>/data/s1.bin] [--gate] [--golden FILE]
    python evals/train_s1.py --check     # retrain with the committed header's config; int8 weights, metrics, parity
    python evals/train_s1.py --search    # 5-fold grouped CV over the §8.1 grid (slow), prints the table
    python evals/train_s1.py --v0 evals/alpha/v0.json --patch [--golden]   # V0 from `alpha.py calibrate-v` into the header

numpy only, and never shipped. Features come from maestro.s1 itself (normalize, features, the Bloom test), so training
and inference cannot drift. evals/prompts.jsonl is never trained on: rows whose prompt is in it are dropped, and only
--gate reads its dev half (even lines) to choose the legacy decode's U and to report the §8.4 replace-NB check.

Labels JSONL, one row per item; every head is optional (a head with fewer than --min-labels rows stays untrained,
its weights are zeroed and the header lists only the trained heads):
  {"prompt": str, "id"?, "split"?: train|calib|dev|test, "cluster"?: str, "group"?: chat|agentic,
   "mode"?, "harness"?, "tools"?, "images"?, "max_tokens"?, "prev_user"?, "n_user"?, "n_tool"?, "last_tool"?,
   "asst_len"?, "prev_job"?                                    (context keys, as maestro.s1.posterior takes them)
   "job": "debug" | {"debug": 0.8, "code": 0.2},  "difficulty": 0-3 | [p0, p1, p2, p3],
   "min_tier": "opus" | {"haiku": p, "sonnet": p, "opus": p, "fable": p, "none": p},
   "needs_tools", "self_contained", "detectable": 0/1 or a probability,  "stakes", "ambiguity": 0-2 | [p0, p1, p2],
   "output_tokens": "<100"|"100-300"|"300-1k"|"1k-3k"|">3k" | {bucket: p},  "agent_turns": "0"|"1-3"|"4-10"|"11-30"|">30" | {..},
   "outcomes": {"haiku": 0..1, ...},  "tokens": {"haiku": n, ...} (measured output tokens),  "turns": n (measured tool calls)}
Splits come from evals/split.py (S1 spec §6.2; see assign_splits): a row's own "split", else the frozen manifest, else
split.py's near-duplicate clusters hashed train < 75 <= calib < 90 <= dev (the bootstrap ranges). Test, legacy and persona
rows are never used. The early-stopping holdout is 10% of the train clusters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "skills", "maestro", "scripts")
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, HERE)
from maestro import router as R  # noqa: E402
from maestro import s1  # noqa: E402
import evalkit as ek  # noqa: E402
import split as SP  # noqa: E402

OUT = os.path.join(SCRIPTS, "maestro", "data", "s1.bin")
GOLDEN = os.path.join(HERE, "..", "tests", "data", "s1_golden.json")
TIERS = ("haiku", "sonnet", "opus", "fable")
GROUPS = ("chat", "agentic")
H, K, M_BLOOM, BIN_MAX = 64, s1.K, 1 << 20, 3 * 1024 * 1024
TOK_B = (("<100", -np.inf, math.log(100)), ("100-300", math.log(100), math.log(300)), ("300-1k", math.log(300), math.log(1000)),
         ("1k-3k", math.log(1000), math.log(3000)), (">3k", math.log(3000), np.inf))
TURN_B = (("1-3", -np.inf, math.log(3.5)), ("4-10", math.log(3.5), math.log(10.5)), ("11-30", math.log(10.5), math.log(30.5)),
          (">30", math.log(30.5), np.inf))
BIN_COLS = ("tools", "self_contained", "stakes1", "stakes2", "detectable", "amb1", "amb2")   # outputs 13..19
BIN_LAM = (0.5, 0.5, 0.3, 0.3, 0.3, 0.2, 0.2)
HEAD_COLS = {"job": range(0, 11), "b": [11], "b_sd": [12], "tools": [13], "self_contained": [14], "stakes": [15, 16],
             "detectable": [17], "ambiguity": [18, 19], "tok_out": [20, 21], "turns": [22, 23]}
THETA0 = (0.0, 1.1, 2.2, 3.3)      # placeholder anchor abilities (spec §3.5)
DEFAULTS = {"B": 1 << 15, "d": 32, "dropout": 0.2, "kw": True, "char3": False, "epochs": 30, "batch": 32,
            "lr_emb": 0.01, "lr": 0.003, "feat_drop": 0.1, "l2_emb": 1e-5, "l2_w": 1e-4, "patience": 4,
            "lambda_d": 0.5, "seed": 0, "min_labels": 20}
CTX_KEYS = ("mode", "harness", "tools", "images", "est_in", "max_tokens", "prev_user", "n_user", "n_tool", "last_tool",
            "asst_len", "prev_job")
# Golden prompts (tests/data/s1_golden.json): short, long, CJK, empty, a 60k-char paste, history, fences, URLs.
GOLDEN_TASKS = [
    "hi", "", "thanks!", "fix the failing test in auth.py", "what is a mutex?",
    "Design the architecture for a multi-tenant billing system and explain the tradeoffs",
    "Refactor parseConfigFile in src/config_loader.ts to drop the circular import; see https://example.com/issue/42",
    "请帮我修复这个函数中的错误，并解释原因", "このテストがCIでだけ失敗する理由を調べてください", "Почему этот тест падает только на CI?",
    "Why does this fail?\n```python\nimport os\nprint(os.environ['HOME'][5])\n```",
    "Traceback (most recent call last):\n  File \"app.py\", line 3, in <module>\n    main()\n  File \"app.py\", line 2, in main\n"
    "    1/0\nZeroDivisionError: division by zero\nfix it",
    {"repeat": "2024-05-01 12:00:01 ERROR worker 17 failed: connection reset by peer (attempt 3)\n", "chars": 60000,
     "prefix": "Why does the worker keep failing? Logs below.\n"},
    {"repeat": "word ", "chars": 1500, "prefix": "Write a migration plan for this: "},
    {"text": "and now do the same for the billing service", "prev_user": "add retries with backoff to the email sender",
     "mode": "proxy", "n_user": 3, "n_tool": 5, "last_tool": True, "asst_len": 900, "prev_job": "code", "tools": True,
     "harness": "claude-code"},
    {"text": "summarize the thread above in three bullets", "mode": "proxy", "n_user": 9, "n_tool": 0, "asst_len": 4000},
    "Write an ADR deciding between Kafka and SQS for our event bus",
    "sql for weekly active users by plan",
    "Deploy the build to staging, run smoke tests, and roll back if they fail",
    "Prove that this lock-free MPMC queue is linearizable and fix the ABA problem in pop()",
]


def expand(task):
    """A golden task spec as posterior() takes it (long pastes are stored as a repeat rule, not inline)."""
    if isinstance(task, dict) and "repeat" in task:
        body = (task["repeat"] * (task["chars"] // len(task["repeat"]) + 1))[:task["chars"] - len(task["prefix"])]
        return task["prefix"] + body
    return task


# ── labels ──────────────────────────────────────────────────────────────────────────────────────────────────────

def _soft(v, keys):
    """A label as a distribution over keys: a key, an index, a list of probabilities or a {key: p} dict."""
    if v is None:
        return None
    out = np.zeros(len(keys))
    if isinstance(v, dict):
        for k, p in v.items():
            if k in keys:
                out[keys.index(k)] += float(p)
    elif isinstance(v, (list, tuple)):
        out[:len(v)] = [float(x) for x in v[:len(keys)]]
    elif isinstance(v, str) and v in keys:
        out[keys.index(v)] = 1.0
    elif isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v < len(keys) and float(v).is_integer():
        out[int(v)] = 1.0
    return out / out.sum() if out.sum() > 0 else None


def _p(v):
    return None if v is None else float(v)


def assign_splits(rows, manifest=None, log=print):
    """([split], [cluster id]) per row through evals/split.py, so near-duplicates never straddle fit / calib / dev (spec §6.2):
    a row's own "split"; else the frozen manifest (by the row's id, or the id split.py gives an id-less row); else split.py's
    clusters over these rows (MinHash 5-shingle unions at Jaccard >= 0.5; the same scenario, persona or "cluster"; source_id)
    hashed with the bootstrap ranges train < 75 <= calib < 90 <= dev. Manifest test, legacy and persona rows come back "test"
    (never used)."""
    man = SP.load_splits(manifest or SP.MANIFEST)
    items, seen = [], set()
    for r in rows:
        text = ek.item_text(dict(r, prompt=r.get("prompt") or r.get("text") or ""))
        sha = ek.sha256_text(ek.norm_text(text))[:10]
        hit = man.get(r.get("id")) or next((man[k] for k in ("%s-%s" % (c, sha) for c in "TCVSXRFLP") if k in man), None)
        rid = str(r.get("id") or "T-" + sha)
        rid = rid if rid not in seen else "%s-%d" % (rid, len(items))
        seen.add(rid)
        items.append({"id": rid, "sig": SP.minhash(text), "scenario": r.get("scenario") or (r.get("cluster") and "cluster:%s" % r["cluster"]),
                      "persona": r.get("persona"), "source_id": r.get("source_id"), "man": hit})
    cl = SP.cluster(items)
    out, how = [], {}
    for r, it in zip(rows, items):
        if r.get("split") in ("train", "calib", "dev", "test"):
            sp, src = r["split"], "row"
        elif it["man"]:
            sp, src = it["man"][1] if it["man"][1] in ("train", "calib", "dev") else "test", "manifest"
        else:
            h = int(hashlib.sha256(cl[it["id"]].encode("utf-8")).hexdigest(), 16) % 100
            sp, src = "train" if h < 75 else "calib" if h < 90 else "dev", "cluster"
        out.append(sp)
        how[src] = how.get(src, 0) + 1
    if man and how.get("cluster"):
        log("warning: %d rows are not in %s; they were split by cluster" % (how["cluster"], manifest or SP.MANIFEST))
    log("splits by source: %s; %d clusters" % (how, len(set(cl.values()))))
    return out, [it["man"][2] if it["man"] else cl[it["id"]] for it in items]


def targets(rows):
    """Label arrays and masks for all rows (NaN-free; a mask of 0 means no label)."""
    n = len(rows)
    T = {"job": np.zeros((n, 11)), "m_job": np.zeros(n), "diff": np.zeros((n, 4)), "m_diff": np.zeros(n),
         "mt": np.zeros((n, 5)), "m_mt": np.zeros(n), "bin": np.zeros((n, 7)), "m_bin": np.zeros((n, 7)),
         "tokb": np.zeros((n, 5)), "m_tokb": np.zeros(n), "turnb": np.zeros((n, 4)), "m_turnb": np.zeros(n),
         "out": np.zeros((n, 4)), "m_out": np.zeros((n, 4)), "tok": np.zeros((n, 4)), "m_tok": np.zeros((n, 4)),
         "turn": np.zeros(n), "m_turn": np.zeros(n), "g": np.zeros(n, dtype=int)}
    for i, r in enumerate(rows):
        for key, src, keys in (("job", "job", list(R.JOBS)), ("diff", "difficulty", [0, 1, 2, 3]),
                               ("mt", "min_tier", list(TIERS) + ["none"]), ("tokb", "output_tokens", [b[0] for b in TOK_B])):
            v = r.get(src)
            q = _soft(v, keys)
            if q is not None:
                T[key][i], T["m_" + key][i] = q, 1.0
        q = _soft(r.get("agent_turns"), ["0"] + [b[0] for b in TURN_B])
        if q is not None and q[0] < 0.5:
            T["turnb"][i], T["m_turnb"][i] = q[1:] / q[1:].sum(), 1.0
        for c, v in ((0, _p(r.get("needs_tools"))), (1, _p(r.get("self_contained"))), (4, _p(r.get("detectable")))):
            if v is not None:
                T["bin"][i, c], T["m_bin"][i, c] = v, 1.0
        for c, src in ((2, "stakes"), (5, "ambiguity")):
            q = _soft(r.get(src), [0, 1, 2])
            if q is not None:
                T["bin"][i, c:c + 2], T["m_bin"][i, c:c + 2] = (q[1] + q[2], q[2]), 1.0
        for t, tier in enumerate(TIERS):
            if (r.get("outcomes") or {}).get(tier) is not None:
                T["out"][i, t], T["m_out"][i, t] = float(r["outcomes"][tier]), 1.0
            if (r.get("tokens") or {}).get(tier):
                T["tok"][i, t], T["m_tok"][i, t] = math.log(float(r["tokens"][tier])), 1.0
        if r.get("turns") is not None:
            T["turn"][i], T["m_turn"][i] = math.log(max(1.0, float(r["turns"]))), 1.0
        tools = r.get("needs_tools")
        T["g"][i] = 1 if r.get("group") == "agentic" or (r.get("group") is None and (
            r.get("turns") is not None or (tools is not None and float(tools) >= 0.5))) else 0
    return T


def trained_heads(T, idx, min_labels):
    """Heads with at least min_labels labeled rows among idx (the fit rows)."""
    cnt = lambda m: int((m[idx] > 0).any(axis=1).sum()) if m.ndim == 2 else int((m[idx] > 0).sum())  # noqa: E731
    have = {"job": cnt(T["m_job"]), "b": cnt(T["m_diff"]) + cnt(T["m_mt"]) + cnt(T["m_out"]), "b_sd": cnt(T["m_out"]),
            "tools": cnt(T["m_bin"][:, 0]), "self_contained": cnt(T["m_bin"][:, 1]), "stakes": cnt(T["m_bin"][:, 2]),
            "detectable": cnt(T["m_bin"][:, 4]), "ambiguity": cnt(T["m_bin"][:, 5]),
            "tok_out": cnt(T["m_tokb"]) + cnt(T["m_tok"]), "turns": cnt(T["m_turnb"]) + cnt(T["m_turn"])}
    return sorted(h for h, n in have.items() if n >= min_labels), have


# ── scalars (§3.4, §3.5): ordinal cut points, anchor abilities, slopes, verbosity, link ──────────────────────────

SNAMES = (["tau1", "lt2", "lt3", "ls_d", "c_mt", "ls_mt"] + ["th_%s_%s" % (g, t) for g in GROUPS for t in TIERS]
          + ["la_chat", "la_agentic"] + ["v_" + t for t in TIERS] + ["alpha_chat", "alpha_agentic", "beta"])
SI = {n: i for i, n in enumerate(SNAMES)}


def linked(link):
    """Does the --link file (evals/fit_link.py) carry z̄ for every anchor tier? Only then are α and β free (spec §3.4 step 3)."""
    return bool(link) and isinstance(link.get("zbar"), dict) and all(isinstance(link["zbar"].get(t), (int, float)) for t in TIERS)


def scalars_init(link):
    S = np.zeros(len(SNAMES))
    S[SI["tau1"]], S[SI["lt2"]], S[SI["lt3"]] = -1.5, math.log(1.7), math.log(1.7)
    S[SI["ls_d"]], S[SI["ls_mt"]] = math.log(0.9), math.log(0.8)
    S[SI["la_chat"]], S[SI["la_agentic"]] = math.log(1.2), math.log(1.5)
    for g in GROUPS:
        for t, th in zip(TIERS, THETA0):
            S[SI["th_%s_%s" % (g, t)]] = th
    al, beta = {"chat": -2.0, "agentic": -2.4}, 1.6            # placeholders (frozen, and ignored by decide, without a link)
    if linked(link):                                           # start at the least-squares line through the placeholder anchors
        beta = beta_prior(link)[0]
        al = {g: float(np.mean(THETA0) - beta * np.mean([link["zbar"][t] for t in TIERS])) for g in GROUPS}
    S[SI["alpha_chat"]], S[SI["alpha_agentic"]], S[SI["beta"]] = al["chat"], al["agentic"], beta
    return S


def beta_pub(link):
    """(β_pub, se_pub) of a fit_link.py file: `beta_pub` a number with `se_pub`, or a [b, se] pair; (1.0, 0.5) without one."""
    b = (link or {}).get("beta_pub")
    b, se = b if isinstance(b, (list, tuple)) else (b, (link or {}).get("se_pub", 0.5)) if b is not None else (1.0, 0.5)
    return float(b), float(se)


def beta_prior(link):
    """The prior N(β_pub / â_chat, max(se_pub, 0.3)²) of the success model's β (spec §3.4 step 2)."""
    b, se = beta_pub(link)
    return b / float((link or {}).get("a_chat_hat") or 1.2), max(se, 0.3)


def taus(S):
    t1 = S[SI["tau1"]]
    t2 = t1 + np.exp(S[SI["lt2"]])
    return np.array([t1, t2, t2 + np.exp(S[SI["lt3"]])])


def theta(S):
    return np.array([[S[SI["th_%s_%s" % (g, t)]] for t in TIERS] for g in GROUPS])


def prior_loss(S, link):
    """-log prior of the scalars (Gaussian priors of §3.4, §3.5; flat on alpha)."""
    lp = ((taus(S) - np.array([-1.5, 0.2, 1.9])) ** 2).sum() / (2 * 0.5 ** 2)
    lp += (S[SI["ls_d"]] - math.log(0.9)) ** 2 / (2 * 0.3 ** 2) + S[SI["c_mt"]] ** 2 / (2 * 0.5 ** 2)
    lp += (S[SI["ls_mt"]] - math.log(0.8)) ** 2 / (2 * 0.3 ** 2)
    lp += (S[SI["la_chat"]] - math.log(1.2)) ** 2 / (2 * 0.3 ** 2) + (S[SI["la_agentic"]] - math.log(1.5)) ** 2 / (2 * 0.4 ** 2)
    lp += sum(S[SI["v_" + t]] ** 2 for t in TIERS) / (2 * 0.5 ** 2)
    th = theta(S)
    if linked(link):
        z = np.array([link["zbar"][t] for t in TIERS])
        mean = np.array([S[SI["alpha_" + g]] + S[SI["beta"]] * z for g in GROUPS])
        b0, se = beta_prior(link)
        lp += (S[SI["beta"]] - b0) ** 2 / (2 * se ** 2)
    else:
        mean = np.array([THETA0, THETA0])
    return lp + ((th - mean) ** 2).sum() / (2 * 0.5 ** 2)


def frozen(T, idx, link):
    """Scalars that must not move: theta_{haiku,chat} (identifiability), agentic thetas with < 40 agentic outcome items
    (§3.5), and the link (alpha, beta) without a --link file."""
    f = np.zeros(len(SNAMES), bool)
    f[SI["th_chat_haiku"]] = True
    if int(((T["m_out"][idx].sum(1) > 0) & (T["g"][idx] == 1)).sum()) < 40:
        f[[SI["th_agentic_" + t] for t in TIERS] + [SI["la_agentic"]]] = True
    if not linked(link):
        f[[SI["alpha_chat"], SI["alpha_agentic"], SI["beta"]]] = True
    return f


# ── losses ──────────────────────────────────────────────────────────────────────────────────────────────────────

def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def ncdf(x):
    """Normal CDF from erf (Abramowitz-Stegun 7.1.26, |error| < 1.5e-7); numpy has no erf."""
    z = np.abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-z * z)
    return 0.5 * (1.0 + np.sign(x) * y)


def interval_nll(mu, sd, lab, bounds):
    """-log P(bucket) for a log-normal target observed only as a bucket (soft labels over the buckets)."""
    P = np.stack([ncdf((hi - mu) / sd) - ncdf((lo - mu) / sd) if np.isfinite(lo) or np.isfinite(hi) else np.ones_like(mu)
                  for _, lo, hi in bounds], 1)
    return -(lab * np.log(np.maximum(P, 1e-9))).sum(1)


def aux_rows(C, S, T, ix, lam_d):
    """Per-row loss of the heads whose gradients are taken numerically: b (ordinal, min_tier, outcome),
    tok_out and turns. C = raw outputs 11, 12, 20, 21, 22, 23 of the rows ix."""
    mu, sd = C[:, 0], np.exp(np.clip(C[:, 1], -10, 10))
    loss = np.zeros(len(ix))
    g = T["g"][ix]
    m = T["m_diff"][ix]
    if m.any():
        cum = sigmoid((mu[:, None] - taus(S)[None, :]) / np.exp(S[SI["ls_d"]]))
        P = np.concatenate([1 - cum[:, :1], cum[:, :-1] - cum[:, 1:], cum[:, -1:]], 1)
        loss += lam_d * m * -(T["diff"][ix] * np.log(np.maximum(P, 1e-9))).sum(1)
    th = theta(S)[g]
    a = np.exp(np.where(g == 1, S[SI["la_agentic"]], S[SI["la_chat"]]))
    m = T["m_mt"][ix]
    if m.any():
        F = sigmoid((th - mu[:, None] - S[SI["c_mt"]]) / np.exp(S[SI["ls_mt"]]))
        P = np.concatenate([F[:, :1], np.diff(F, axis=1), 1 - F[:, -1:]], 1)
        loss += lam_d * m * -(T["mt"][ix] * np.log(np.maximum(P, 1e-9))).sum(1)
    m = T["m_out"][ix]
    if m.any():
        z = a[:, None] * (th - mu[:, None]) / np.sqrt(1 + math.pi * (a * sd)[:, None] ** 2 / 8)
        y = T["out"][ix]
        ll = y * -np.logaddexp(0, -z) + (1 - y) * -np.logaddexp(0, z)
        loss += 2.0 * -(m * ll).sum(1) / np.maximum(1, m.sum(1))
    tmu, tsd = C[:, 2], np.exp(np.clip(C[:, 3], -10, 10))
    if T["m_tokb"][ix].any():
        loss += 0.2 * T["m_tokb"][ix] * interval_nll(tmu, tsd, T["tokb"][ix], TOK_B)
    m = T["m_tok"][ix]
    if m.any():
        v = np.array([S[SI["v_" + t]] for t in TIERS])
        nll = 0.5 * np.log(2 * math.pi * tsd[:, None] ** 2) + (T["tok"][ix] - tmu[:, None] - v) ** 2 / (2 * tsd[:, None] ** 2)
        loss += 0.5 * (m * nll).sum(1) / np.maximum(1, m.sum(1))
    rmu, rsd = C[:, 4], np.exp(np.clip(C[:, 5], -10, 10))
    if T["m_turnb"][ix].any():
        loss += 0.3 * T["m_turnb"][ix] * interval_nll(rmu, rsd, T["turnb"][ix], TURN_B)
    m = T["m_turn"][ix]
    if m.any():
        loss += 0.3 * m * (0.5 * np.log(2 * math.pi * rsd ** 2) + (T["turn"][ix] - rmu) ** 2 / (2 * rsd ** 2))
    return loss


AUX = [11, 12, 20, 21, 22, 23]


def head_loss(Y, S, T, ix, cfg, link, n_total, grad=True, active=None):
    """Mean loss over the rows ix (+ the scalar priors / n_total) and, if grad, (dL/dY, dL/dS).
    Job and the binary heads are analytic; the rest by central differences (exact enough, and hard to get wrong)."""
    n = len(ix)
    q = T["job"][ix] * 0.95 + 0.05 / 11
    z = Y[:, :11] - Y[:, :11].max(1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(1, keepdims=True))
    mj = T["m_job"][ix]
    rows = mj * -(q * lp).sum(1)
    u, t, mb = Y[:, 13:20], T["bin"][ix], T["m_bin"][ix] * np.array(BIN_LAM)
    rows += (mb * (np.logaddexp(0, u) - t * u)).sum(1)
    lam_d = cfg["lambda_d"]
    rows += aux_rows(Y[:, AUX], S, T, ix, lam_d)
    total = rows.sum() / n + prior_loss(S, link) / n_total
    if not grad:
        return total, rows
    gY = np.zeros_like(Y)
    gY[:, :11] = mj[:, None] * (np.exp(lp) - q)
    gY[:, 13:20] = mb * (sigmoid(u) - t)
    eps = 1e-5
    C = Y[:, AUX].copy()
    cols, scal = active if active is not None else (set(AUX), range(len(S)))
    for k, col in enumerate(AUX):
        if col not in cols:
            continue
        C[:, k] += eps
        up = aux_rows(C, S, T, ix, lam_d)
        C[:, k] -= 2 * eps
        dn = aux_rows(C, S, T, ix, lam_d)
        C[:, k] += eps
        gY[:, col] = (up - dn) / (2 * eps)
    gS = np.zeros_like(S)
    for j in scal:
        Sp, Sm = S.copy(), S.copy()
        Sp[j] += eps
        Sm[j] -= eps
        fp = aux_rows(C, Sp, T, ix, lam_d).sum() / n + prior_loss(Sp, link) / n_total
        fm = aux_rows(C, Sm, T, ix, lam_d).sum() / n + prior_loss(Sm, link) / n_total
        gS[j] = (fp - fm) / (2 * eps)
    return total, rows, gY / n, gS


# ── features and the Bloom filter ───────────────────────────────────────────────────────────────────────────────

def featurize(rows, cfg):
    out, nms = [], []
    for r in rows:
        text = r.get("prompt") or r.get("text") or ""
        ctx = {k: r[k] for k in CTX_KEYS if k in r}
        nm = s1.normalize(text, ctx.get("prev_user") or "")
        out.append(s1.features(nm, s1.keyword_info(None, text) if cfg["kw"] else None, ctx, cfg["char3"]))
        nms.append(nm)
    return out, nms


def bloom_of(feats, fit, B):
    """Bloom filter (M bits, k = 2) of the features seen in >= 2 fit rows, or in a closed namespace;
    and the live buckets (those a known feature maps to)."""
    count = {}
    for i in fit:
        for f in set(feats[i]):
            count[f] = count.get(f, 0) + 1
    bits, live = np.zeros(M_BLOOM, bool), np.zeros(B, bool)
    for f, c in count.items():
        if c >= 2 or f[0] in s1.CLOSED:
            fb = f.encode("utf-8")
            h1, h2 = s1.zlib.crc32(fb), s1.zlib.crc32(fb, s1.SEED2)
            bits[h1 & (M_BLOOM - 1)] = bits[(h1 + h2) & (M_BLOOM - 1)] = True
            live[h1 & (B - 1)] = True
    return np.packbits(bits, bitorder="little").tobytes(), live


# ── the network (§2.4, §8.1) ────────────────────────────────────────────────────────────────────────────────────

def init_params(cfg, rng):
    d = cfg["d"]
    lim1, lim2 = math.sqrt(6 / (d + H)), math.sqrt(6 / (H + K))
    b2 = np.zeros(K)
    b2[12], b2[20], b2[21], b2[22], b2[23] = math.log(s1.PRIORS["sd"]), math.log(500), 0.0, s1.PRIORS["turns"][0], math.log(0.8)
    return {"E": rng.normal(0, 0.1, (cfg["B"], d)), "W1": rng.uniform(-lim1, lim1, (d, H)), "b1": np.zeros(H),
            "W2": rng.uniform(-lim2, lim2, (K, H)), "b2": b2}


def bag_mean(E, ids_list):
    """Mean embedding per row (zeros for a row with no known features) and the flat (row, id) layout for backprop."""
    lens = np.array([len(x) for x in ids_list])
    flat = np.concatenate([np.asarray(x, dtype=np.int64) for x in ids_list]) if lens.sum() else np.zeros(0, np.int64)
    row = np.repeat(np.arange(len(ids_list)), lens)
    e = np.zeros((len(ids_list), E.shape[1]))
    np.add.at(e, row, E[flat])
    return e / np.maximum(lens, 1)[:, None], flat, row, lens


def forward(P, ids_list, rng=None, drop=0.0):
    e, flat, row, lens = bag_mean(P["E"], ids_list)
    a = e @ P["W1"] + P["b1"]
    h = np.maximum(a, 0)
    mask = (rng.random(h.shape) >= drop) / (1 - drop) if rng is not None and drop else None
    hd = h * mask if mask is not None else h
    return hd @ P["W2"].T + P["b2"], (e, a, h, hd, mask, flat, row, lens)


class Adam:
    def __init__(self, shape, lr):
        self.m, self.v, self.lr = np.zeros(shape), np.zeros(shape), lr
        self.t = np.zeros(shape[0], dtype=np.int64)   # per-row step counts (lazy Adam); t[0] for dense tensors

    def step(self, p, g, rows=None):
        """Adam; with rows, lazy: only those rows move and each keeps its own step count."""
        if rows is None:
            self.t[0] += 1
            t = self.t[0]
            self.m = 0.9 * self.m + 0.1 * g
            self.v = 0.999 * self.v + 0.001 * g * g
            p -= self.lr * (self.m / (1 - 0.9 ** t)) / (np.sqrt(self.v / (1 - 0.999 ** t)) + 1e-8)
            return
        self.t[rows] += 1
        t = self.t[rows][:, None]
        self.m[rows] = 0.9 * self.m[rows] + 0.1 * g
        self.v[rows] = 0.999 * self.v[rows] + 0.001 * g * g
        p[rows] -= self.lr * (self.m[rows] / (1 - 0.9 ** t)) / (np.sqrt(self.v[rows] / (1 - 0.999 ** t)) + 1e-8)


def active_set(T, fit, frz):
    """Which numeric-gradient outputs and scalars the labels can move (the rest have zero data gradient)."""
    has = lambda k: bool(T[k][fit].any())  # noqa: E731
    cols, sc = set(), []
    if has("m_diff"):
        cols.add(11)
        sc += ["tau1", "lt2", "lt3", "ls_d"]
    if has("m_mt"):
        cols.add(11)
        sc += ["c_mt", "ls_mt"] + ["th_%s_%s" % (g, t) for g in GROUPS for t in TIERS]
    if has("m_out"):
        cols.update((11, 12))
        sc += ["th_%s_%s" % (g, t) for g in GROUPS for t in TIERS] + ["la_chat", "la_agentic", "alpha_chat",
                                                                       "alpha_agentic", "beta"]
    if has("m_tokb") or has("m_tok"):
        cols.update((20, 21))
        sc += ["v_" + t for t in TIERS] if has("m_tok") else []
    if has("m_turnb") or has("m_turn"):
        cols.update((22, 23))
    if any(s.startswith("th_") for s in sc):   # the link is tied to the anchors by their prior: the joint MAP (§3.4 step 3)
        sc += ["alpha_chat", "alpha_agentic", "beta"]
    return cols, sorted({SI[s] for s in sc if not frz[SI[s]]})


def train(P, S, T, ids, fit, hold, cfg, link, live, log=print):
    """Multi-task training with early stopping on the holdout's joint loss; returns the best (P, S, epochs)."""
    rng = np.random.default_rng(cfg["seed"])
    frz = frozen(T, fit, link)
    act = active_set(T, fit, frz)
    opt_e = Adam(P["E"].shape, cfg["lr_emb"])
    opts = {k: Adam(P[k].shape, cfg["lr"]) for k in ("W1", "b1", "W2", "b2")}
    opt_s = Adam(S.shape, cfg["lr"])
    P["E"][~live] = 0.0
    best, best_loss, bad, n_total = None, np.inf, 0, len(fit)
    for ep in range(cfg["epochs"]):
        t0 = time.time()
        order = rng.permutation(fit)
        for s in range(0, len(order), cfg["batch"]):
            ix = order[s:s + cfg["batch"]]
            ids_b = [ids[r][rng.random(len(ids[r])) >= cfg["feat_drop"]] for r in ix]
            Y, (e, a, h, hd, mask, flat, row, lens) = forward(P, ids_b, rng, cfg["dropout"])
            _, _, gY, gS = head_loss(Y, S, T, ix, cfg, link, n_total, active=act)
            gW2 = gY.T @ hd + cfg["l2_w"] * P["W2"]
            gh = gY @ P["W2"]
            gh = gh * mask if mask is not None else gh
            ga = gh * (a > 0)
            gW1 = e.T @ ga + cfg["l2_w"] * P["W1"]
            ge = ga @ P["W1"].T
            if len(flat):
                G = ge[row] / lens[row][:, None]
                uq, inv = np.unique(flat, return_inverse=True)
                gE = np.zeros((len(uq), P["E"].shape[1]))
                np.add.at(gE, inv.ravel(), G)
                keep = live[uq]
                uq, gE = uq[keep], gE[keep]
                opt_e.step(P["E"], gE + cfg["l2_emb"] * P["E"][uq], uq)
            for k, g in (("W1", gW1), ("b1", ga.sum(0)), ("W2", gW2), ("b2", gY.sum(0))):
                opts[k].step(P[k], g)
            gS[frz] = 0.0
            opt_s.step(S, gS)
        Yh, _ = forward(P, [ids[r] for r in hold])
        hl = head_loss(Yh, S, T, hold, cfg, link, n_total, grad=False)[0] if len(hold) else 0.0
        log("epoch %2d  holdout loss %.4f  (%.1f s)" % (ep + 1, hl, time.time() - t0))
        if hl < best_loss - 1e-6:
            best, best_loss, bad = ({k: v.copy() for k, v in P.items()}, S.copy(), ep + 1), hl, 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    return best


# ── export (§8.3) and the s1.bin file (App. A) ──────────────────────────────────────────────────────────────────

def neutralize(P, trained):
    """Zero the outputs of untrained heads (b_sd keeps the prior log sd in its bias)."""
    W2, b2 = P["W2"].copy(), P["b2"].copy()
    for hname, cols in HEAD_COLS.items():
        if hname not in trained:
            W2[list(cols)] = 0.0
            b2[list(cols)] = math.log(s1.PRIORS["sd"]) if hname == "b_sd" else 0.0
    return W2, b2


def quantize(P, live, W2, b2, sample):
    """Fold E' = E W1; per-column scale = the 99.99th percentile of |E'| over live rows / 127 (clipping the rest), or
    the column max when clipping would break the parity bound on the sample rows; int8 stored as uint8 (+128)."""
    Ep = P["E"] @ P["W1"]
    Ep[~live] = 0.0
    for pct in (99.99, 100.0):
        scale = np.percentile(np.abs(Ep[live]), pct, axis=0) / 127.0 if live.any() else np.ones(H)
        scale = np.where(scale > 0, scale, 1e-8).astype(np.float32)
        Q = np.clip(np.round(Ep / scale.astype(np.float64)), -127, 127).astype(np.int16)
        Q[~live] = 0
        err = 0.0
        for ids in sample:
            hf = np.maximum((Ep[ids].mean(0) if len(ids) else 0.0) + P["b1"], 0)
            hq = np.maximum(scale.astype(np.float64) * Q[ids].sum(0) / max(len(ids), 1) + P["b1"], 0)
            err = max(err, float(np.abs(W2 @ (hf - hq)).max()))
        if err <= 0.04:
            break
    dense = np.concatenate([scale, P["b1"].astype(np.float32), W2.astype(np.float32).ravel(), b2.astype(np.float32)])
    return Q, (Q + 128).astype(np.uint8).tobytes(), dense.astype("<f4").tobytes(), {"pct": pct, "max_logit_err": round(err, 5)}


def write_bin(path, header, bloom, emb, dense):
    """MS1\\0, u32 header length, header JSON padded with spaces to 64 bytes, then 64-byte-aligned sections."""
    secs = (("bloom", bloom), ("emb", emb), ("dense", dense))
    header = dict(header, sha256={k: hashlib.sha256(v).hexdigest() for k, v in secs})
    for _ in range(3):   # offsets change the header length, which changes the offsets: settle
        raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        raw += b" " * (-len(raw) % 64)
        off, sections = (8 + len(raw) + 63) // 64 * 64, {}
        for k, v in secs:
            sections[k] = [off, len(v)]
            off = (off + len(v) + 63) // 64 * 64
        if header.get("sections") == sections:
            break
        header["sections"] = sections
    out = bytearray(s1.MAGIC + len(raw).to_bytes(4, "little") + raw)
    for k, v in secs:
        out += b"\0" * (sections[k][0] - len(out)) + v
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(out)
    os.replace(tmp, path)
    s1._S1.pop(path, None)
    return len(out)


# ── calibration (§2.6, §2.7), on the int8 model's outputs ───────────────────────────────────────────────────────

def int8_outputs(Q, dense_arrays, ids_list):
    scale, b1, W2, b2 = dense_arrays
    out = np.zeros((len(ids_list), K))
    for i, ids in enumerate(ids_list):
        n = len(ids)
        Ssum = Q[ids].sum(0).astype(np.float64) if n else np.zeros(H)
        h = np.maximum(scale * Ssum / max(n, 1) + b1, 0)
        out[i] = W2 @ h + b2
    return out


def platt(x, t, w=None, iters=50):
    """Logistic regression of soft targets t on x: (A, B) by Newton with a tiny ridge."""
    w = np.ones_like(t) if w is None else w
    A, B = 1.0, 0.0
    for _ in range(iters):
        p = sigmoid(A * x + B)
        g = np.array([(w * (p - t) * x).sum() + 1e-4 * (A - 1), (w * (p - t)).sum() + 1e-4 * B])
        r = w * p * (1 - p)
        Hm = np.array([[(r * x * x).sum() + 1e-4, (r * x).sum()], [(r * x).sum(), r.sum() + 1e-4]])
        step = np.linalg.solve(Hm, g)
        A, B = A - step[0], B - step[1]
        if np.abs(step).max() < 1e-9:
            break
    return [float(A), float(B)]


def golden_section(f, lo, hi, iters=60):
    """argmin of a unimodal f on [lo, hi]."""
    r = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - r * (b - a), a + r * (b - a)
    for _ in range(iters):
        if f(c) < f(d):
            b, d = d, c
            c = b - r * (b - a)
        else:
            a, c = c, d
            d = a + r * (b - a)
    return (a + b) / 2


def isotonic(x, y, knots=16):
    """Pool-adjacent-violators fit of y on x, reduced to <= knots points: [xs, ys], the form maestro.decide._cal reads (x = P_raw,
    so xs ascend in [0, 1]; bisect + linear interpolation)."""
    o = np.argsort(x)
    xs, ys = list(x[o]), list(y[o])
    blocks = []
    for xi, yi in zip(xs, ys):
        blocks.append([xi, xi, yi, 1.0])
        while len(blocks) > 1 and blocks[-2][2] >= blocks[-1][2]:
            b = blocks.pop()
            a = blocks[-1]
            a[1], a[2], a[3] = b[1], (a[2] * a[3] + b[2] * b[3]) / (a[3] + b[3]), a[3] + b[3]
    px = [(b[0] + b[1]) / 2 for b in blocks]
    py = [b[2] for b in blocks]
    keep = sorted(set(np.linspace(0, len(px) - 1, min(knots, len(px))).round().astype(int)))
    return [[float(px[i]) for i in keep], [float(py[i]) for i in keep]]


def p_success_raw(Y, S, T, ix, c_sigma, gamma, ood):
    g = T["g"][ix]
    a = np.exp(np.where(g == 1, S[SI["la_agentic"]], S[SI["la_chat"]]))
    sd = c_sigma * np.exp(np.clip(Y[:, 12], -10, 10)) * (1 + gamma * ood[ix])
    z = a[:, None] * (theta(S)[g] - Y[:, 11:12]) / np.sqrt(1 + math.pi * (a * sd)[:, None] ** 2 / 8)
    return sigmoid(z)


def nb_on_split(rows, fit, other):
    """The shipped NB algorithm (evals/train.py) refit on the fit rows only, so calib is out-of-sample for it."""
    import train as nbtrain
    fr = [{"prompt": rows[i].get("prompt") or "", "job": rows[i]["job"], "difficulty": rows[i]["difficulty"]}
          for i in fit if rows[i].get("job") in R.JOBS and rows[i].get("difficulty") in (0, 1, 2, 3)]
    if not fr:
        return None
    prior, weights = nbtrain.fit(fr)
    model = {"version": 2, "jobs": list(R.JOBS), "n": len(fr), "scale": nbtrain.SCALE, "prior": prior,
             "jd": nbtrain.job_difficulty(fr), "w": weights}
    return {i: s1.nb_job_p(R.classify_model(rows[i].get("prompt") or "", model)) for i in other}


def calibrate(Yc, T, cal, S, ood, trained, nbp, log=print):
    c = {"T_job": 1.0, "w_nb": 0.0, "platt": {k: [1.0, 0.0] for k in BIN_COLS}, "c_sigma_b": 1.0, "ood_gamma": 0.8,
         "tok_sd": 1.0, "turn_sd": 1.0, "final": {g: {"platt": [1.0, 0.0], "iso": None} for g in GROUPS}}
    fitted = []
    mj = T["m_job"][cal] > 0
    if "job" in trained and mj.sum() >= 10:
        q, z = T["job"][cal][mj], Yc[mj, :11]

        def nll(T_, w=0.0, nbq=None):
            lp = z / T_
            lp = lp - np.log(np.exp(lp - lp.max(1, keepdims=True)).sum(1, keepdims=True)) - lp.max(1, keepdims=True)
            if w:
                lp = (1 - w) * lp + w * np.log(np.maximum(nbq, 1e-12))
                lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
            return float(-(q * lp).sum(1).mean())
        grid = np.round(np.arange(0.5, 5.0001, 0.05), 2)
        c["T_job"] = float(grid[int(np.argmin([nll(t) for t in grid]))])
        fitted.append("T_job")
        if nbp:
            nbq = np.array([nbp[i] for i, ok in zip(cal, mj) if ok])
            scores = {w: nll(c["T_job"], w, nbq) for w in (0.0, 0.25, 0.5)}
            c["w_nb"] = min(scores, key=lambda w: (round(scores[w], 6), w))
            fitted.append("w_nb")
            log("job NLL on calib by w_nb: " + ", ".join("%.2f: %.4f" % kv for kv in scores.items()))
    for k, name in enumerate(BIN_COLS):
        head = {"tools": "tools", "self_contained": "self_contained", "detectable": "detectable"}.get(
            name, "stakes" if name.startswith("stakes") else "ambiguity")
        m = T["m_bin"][cal, k] > 0
        if head in trained and m.sum() >= 10:
            c["platt"][name] = platt(Yc[m, 13 + k], T["bin"][cal][m, k])
            fitted.append("platt." + name)
    md_ = T["m_diff"][cal] > 0
    if "b" in trained and md_.sum() >= 10:   # the legacy decode's thresholds P(d >= k): one Platt shared by k = 1..3
        u = (Yc[md_, 11:12] - taus(S)[None, :]) / math.exp(S[SI["ls_d"]])
        lab = T["diff"][cal][md_]
        c["platt"]["d"] = platt(u.ravel(), np.stack([lab[:, k + 1:].sum(1) for k in range(3)], 1).ravel())
        fitted.append("platt.d")
    mo = T["m_out"][cal]
    if "b_sd" in trained and mo.sum() >= 20:
        def out_nll(cs, gm):
            p = np.clip(p_success_raw(Yc, S, T, cal, cs, gm, ood), 1e-9, 1 - 1e-9)
            y = T["out"][cal]
            return float(-(mo * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / mo.sum())
        c["c_sigma_b"] = golden_section(lambda cs: out_nll(cs, 0.0), 0.5, 3.0)
        c["ood_gamma"] = golden_section(lambda gm: out_nll(c["c_sigma_b"], gm), 0.0, 2.0)
        fitted += ["c_sigma_b", "ood_gamma"]
        p = p_success_raw(Yc, S, T, cal, c["c_sigma_b"], c["ood_gamma"], ood)
        lg = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1))
        for gi, g in enumerate(GROUPS):
            cells = (mo > 0) & (T["g"][cal] == gi)[:, None]
            if cells.sum() >= 20:
                c["final"][g]["platt"] = platt(lg[cells], T["out"][cal][cells])
                fitted.append("final.%s.platt" % g)
                if cells.sum() >= 2000:
                    c["final"][g]["iso"] = isotonic(np.clip(p, 1e-9, 1 - 1e-9)[cells], T["out"][cal][cells])   # on P_raw
                    fitted.append("final.%s.iso" % g)
    for key, cols, lab in (("tok_sd", (20, 21), "tok"), ("turn_sd", (22, 23), "turn")):
        head = "tok_out" if lab == "tok" else "turns"
        m = T["m_" + lab][cal] > 0
        if head in trained and m.sum() >= 10:
            mu, sd = Yc[:, cols[0]], np.exp(np.clip(Yc[:, cols[1]], -10, 10))
            if lab == "tok":
                v = np.array([S[SI["v_" + t]] for t in TIERS])
                zs = ((T["tok"][cal] - mu[:, None] - v) / sd[:, None])[m]
            else:
                zs = ((T["turn"][cal] - mu) / sd)[m]
            c[key] = float(np.sqrt(np.mean(zs ** 2)))
            fitted.append(key)
    c["fitted"] = fitted
    return c


# ── metrics, parity (§8.3), the replace-NB gate (§8.4), golden file ─────────────────────────────────────────────

def ordinal_cum(mu, S, calib):
    """Calibrated P(d >= k), k = 1..3, as maestro.s1.legacy computes it (one Platt shared by the thresholds)."""
    a, b = (calib.get("platt") or {}).get("d") or (1.0, 0.0)
    return sigmoid(a * (mu[:, None] - taus(S)[None, :]) / math.exp(S[SI["ls_d"]]) + b)


def metrics(Y, T, ix, S, calib, nbp):
    out = {"n": int(len(ix))}
    mj = T["m_job"][ix] > 0
    if mj.any():
        z = Y[mj, :11] / calib["T_job"]
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        if calib["w_nb"] and nbp:
            q = np.array([nbp[i] for i, ok in zip(ix, mj) if ok])
            p = np.exp((1 - calib["w_nb"]) * np.log(np.maximum(p, 1e-12)) + calib["w_nb"] * np.log(np.maximum(q, 1e-12)))
            p /= p.sum(1, keepdims=True)
        lab = T["job"][ix][mj]
        out["job_acc"] = round(float((p.argmax(1) == lab.argmax(1)).mean()), 4)
        out["job_nll"] = round(float(-(lab * np.log(np.maximum(p, 1e-12))).sum(1).mean()), 4)
    md = T["m_diff"][ix] > 0
    if md.any():
        cum = ordinal_cum(Y[md, 11], S, calib)
        P = np.concatenate([1 - cum[:, :1], cum[:, :-1] - cum[:, 1:], cum[:, -1:]], 1)
        lab = T["diff"][ix][md]
        out["diff_acc"] = round(float((P.argmax(1) == lab.argmax(1)).mean()), 4)
        out["diff_mae"] = round(float(np.abs(P.argmax(1) - lab.argmax(1)).mean()), 4)
        out["diff_nll"] = round(float(-(lab * np.log(np.maximum(P, 1e-9))).sum(1).mean()), 4)
        a, b = Y[md, 11], lab @ np.arange(4)
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b + 1e-9 * np.arange(len(b))))
        out["diff_spearman"] = round(float(np.corrcoef(ra, rb)[0, 1]), 4) if len(a) > 2 else None
    return out


def parity(path, P, live, W2, b2, Q, ids_list, log=print):
    """stdlib int8 inference (maestro.s1) vs the numpy float model; and the integer bag sums, which must be identical."""
    md, why = s1.load(path)
    if md is None:
        raise SystemExit("parity: %s" % why)
    Ep = P["E"] @ P["W1"]
    Ep[~live] = 0.0
    max_d, agree, exact = 0.0, 0, 0
    for ids in ids_list:
        n = len(ids)
        ids = [int(i) for i in ids]
        y_py = np.array(s1.infer(md, ids))
        h = np.maximum((Ep[ids].mean(0) if n else np.zeros(H)) + P["b1"], 0)
        y_f = W2 @ h + b2
        max_d = max(max_d, float(np.abs(y_py - y_f).max()))
        agree += int(y_py[:11].argmax() == y_f[:11].argmax())
        S_np = (Q[ids].astype(np.int64) + 128).sum(0) if n else np.zeros(H, np.int64)
        exact += int(list(S_np) == s1.bag(md["emb"], ids, md["H"]))
    res = {"rows": len(ids_list), "max_abs_logit_diff": round(max_d, 5), "job_argmax_agree": agree / max(1, len(ids_list)),
           "bag_sums_bit_exact": exact == len(ids_list)}
    log("parity: %s" % json.dumps(res))
    if not (max_d <= 0.05 and res["job_argmax_agree"] >= 0.995 and res["bag_sums_bit_exact"]):
        raise SystemExit("parity check failed: %s" % res)
    return res


def gate(path, log=print):
    """§8.4 on the dev half of evals/prompts.jsonl: S1's legacy decode with U swept so under-routing <= NB's;
    S1 replaces NB in classify_rules only if category and job accuracy are >= NB - 1 pt."""
    import run as evalrun
    rows = evalrun.split_rows(evalrun.load(os.path.join(HERE, "prompts.jsonl")), "dev")
    nb_res = evalrun.evaluate(rows, evalrun.make_predict_rules("rules"))
    posts = {}

    def make(U):
        def predict(prompt):
            cls = R.classify_rules(prompt)
            if prompt not in posts:
                posts[prompt] = s1.posterior(prompt, cls, path=path)
            c = s1.legacy(posts[prompt], cls, path=path, U=U)
            return c, R.tier_of(R.target_tier(c, {})), False
        return predict
    sweep = []
    for U in np.round(np.arange(1.0, 8.01, 0.25), 2):
        r = evalrun.evaluate(rows, make(float(U)))
        sweep.append((float(U), r))
        log("gate U=%.2f  job %.3f  category %.3f  under %.3f  severe %.3f" % (
            U, r["job_accuracy"], r["category_accuracy"], r["under_routed"], r["severe_under_routed"]))
    ok = [(U, r) for U, r in sweep if r["under_routed"] <= nb_res["under_routed"]]
    U, r = max(ok, key=lambda ur: (ur[1]["category_accuracy"], -ur[0])) if ok else min(sweep, key=lambda ur: ur[1]["under_routed"])
    keys = ("job_accuracy", "category_accuracy", "under_routed", "severe_under_routed", "difficulty_exact")
    res = {"U": U, "split": "prompts dev (even lines)", "nb": {k: nb_res[k] for k in keys}, "s1": {k: r[k] for k in keys},
           "pass": bool(ok) and r["category_accuracy"] >= nb_res["category_accuracy"] - 0.01
           and r["job_accuracy"] >= nb_res["job_accuracy"] - 0.01}
    log("gate (§8.4): %s" % json.dumps(res))
    return res


def golden(path, out, log=print):
    rows = []
    for spec in GOLDEN_TASKS:
        task = expand(spec)
        post = s1.posterior(task, path=path)
        leg = s1.legacy(post, path=path)
        rows.append({"task": spec, "posterior": post, "legacy": [leg.job, leg.difficulty] if leg else None})
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"bin_sha256": file_sha(path), "tolerance": 1e-6, "rows": rows}, f, indent=1, ensure_ascii=False)
        f.write("\n")
    log("wrote %s (%d prompts)" % (out, len(rows)))


V0_KEYS = ("cost", "balanced", "quality")


def v0_of(path):
    """The header's "decide" section from `alpha.py calibrate-v`'s v0.json (spec §5.1). An objective without a value keeps the
    placeholder; V0_calibrated is true only when all three are calibrated."""
    with open(path, encoding="utf-8") as f:
        got = (json.load(f) or {}).get("V0") or {}
    got = {k: float(v) for k, v in got.items() if k in V0_KEYS and isinstance(v, (int, float)) and v > 0}
    if not got:
        raise SystemExit("%s has no V0 values" % path)
    return {"V0": dict({"cost": 0.05, "balanced": 0.5, "quality": 5.0}, **got), "V0_calibrated": len(got) == len(V0_KEYS)}


def patch(path, updates, log=print):
    """Rewrite the header of an existing s1.bin with `updates` (top-level keys), keeping its sections byte for byte."""
    md, why = s1.load(path)
    if md is None:
        raise SystemExit("--patch: %s" % why)
    with open(path, "rb") as f:
        raw = f.read()
    hd = md["header"]
    secs = {k: raw[o:o + n] for k, (o, n) in hd["sections"].items()}
    header = dict({k: v for k, v in hd.items() if k not in ("sections", "sha256")}, **updates)
    size = write_bin(path, header, secs["bloom"], secs["emb"], secs["dense"])
    log("patched %s (%d bytes): %s" % (path, size, json.dumps(updates)))
    return header


def file_sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ── driver ──────────────────────────────────────────────────────────────────────────────────────────────────────

def load_rows(path):
    with open(path, "rb") as f:
        raw = f.read()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    banned = set()
    try:
        with open(os.path.join(HERE, "prompts.jsonl"), encoding="utf-8") as f:
            banned = {json.loads(line)["prompt"] for line in f if line.strip()}
    except OSError:
        pass
    kept = [r for r in rows if (r.get("prompt") or r.get("text") or "") not in banned and r.get("split") != "test"]
    return kept, hashlib.sha256(raw).hexdigest(), len(rows) - len(kept)


def build(rows, cfg, out, link=None, data_sha="", log=print, extra=None):
    """Train, calibrate and export. Returns (report, state) where state has what parity/check need."""
    t_start = time.time()
    feats, _ = featurize(rows, cfg)
    T = targets(rows)
    split, clus = assign_splits(rows, log=log)
    idx = {k: np.array([i for i, s in enumerate(split) if s == k], dtype=np.int64) for k in ("train", "calib", "dev")}
    hold_mask = np.array([int(hashlib.sha256(("hold:" + clus[i]).encode()).hexdigest(), 16) % 10 == 0 for i in idx["train"]], bool)
    fit, hold = idx["train"][~hold_mask], idx["train"][hold_mask]
    trained, have = trained_heads(T, fit, cfg["min_labels"])
    log("rows: %d fit, %d holdout, %d calib, %d dev; labeled rows per head: %s; trained: %s" % (
        len(fit), len(hold), len(idx["calib"]), len(idx["dev"]), have, trained))
    bloom, live = bloom_of(feats, fit, cfg["B"])
    md_b = {"bloom": bloom, "M": M_BLOOM, "B": cfg["B"]}
    ids, ood = [], np.zeros(len(rows))
    for i, f in enumerate(feats):
        x, ood[i] = s1.known(md_b, f)
        ids.append(np.array(x, dtype=np.int64))
    fitted = linked(link) and SI["beta"] in active_set(T, fit, frozen(T, fit, link))[1]   # α, β free and moved by the data
    for k in range(2 if fitted else 1):   # β's prior uses â_chat, re-estimated once ("iterate twice", §3.4 step 2)
        P, S = init_params(cfg, np.random.default_rng(cfg["seed"])), scalars_init(link)
        P, S, epochs = train(P, S, T, ids, fit, hold, cfg, link, live, log)
        if fitted and not k:
            link = dict(link, a_chat_hat=round(math.exp(S[SI["la_chat"]]), 6))
            log("link: second pass with a_chat_hat %.3f" % link["a_chat_hat"])
    W2, b2 = neutralize(P, trained)
    ev = idx["dev"] if len(idx["dev"]) else idx["calib"]
    sample = [ids[i] for i in list(ev) + list(idx["calib"]) + list(fit)][:500]
    Q, emb, dense, quant = quantize(P, live, W2, b2, sample)
    d = np.frombuffer(dense, "<f4").astype(np.float64)
    arrays = (d[:H], d[H:2 * H], d[2 * H:2 * H + K * H].reshape(K, H), d[2 * H + K * H:])
    nbp = nb_on_split(rows, idx["train"], np.concatenate([idx["calib"], idx["dev"]])) if "job" in trained else None
    Yc = int8_outputs(Q, arrays, [ids[i] for i in idx["calib"]])
    calib = calibrate(Yc, T, idx["calib"], S, ood, trained, nbp, log)
    Yd = int8_outputs(Q, arrays, [ids[i] for i in ev])
    mets = {"dev": metrics(Yd, T, ev, S, calib, nbp), "calib": metrics(Yc, T, idx["calib"], S, calib, nbp)}
    Yt = int8_outputs(Q, arrays, [ids[i] for i in fit])
    bq = np.percentile(Yt[:, 11], np.arange(1, 100)) if "b" in trained else np.linspace(-3, 3, 99)
    th = theta(S)
    header = {
        "format": s1.FORMAT, "feat_version": s1.FEAT_VERSION, "B": cfg["B"], "H": H, "K": K,
        "bloom": {"M": M_BLOOM, "k": 2, "seed2": s1.SEED2}, "jobs": list(R.JOBS),
        "heads": {"job": [0, 11], "b": [11, 12], "tools": 13, "self_contained": 14, "stakes": [15, 16], "detectable": 17,
                  "ambiguity": [18, 19], "tok": [20, 21], "turns": [22, 23]},
        "quant": quant, "trained": trained, "b_scale": "outcome" if "b_sd" in trained else "ordinal" if "b" in trained else "none",
        "calib": calib,
        "ordinal": {"tau": [round(float(t), 6) for t in taus(S)], "s_d": round(math.exp(S[SI["ls_d"]]), 6),
                    "c_mt": round(float(S[SI["c_mt"]]), 6), "s_mt": round(math.exp(S[SI["ls_mt"]]), 6),
                    "b_quantiles": [round(float(v), 4) for v in bq]},
        "irt": {"theta": {g: {t: round(float(th[gi, ti]), 6) for ti, t in enumerate(TIERS)} for gi, g in enumerate(GROUPS)},
                "a": {g: round(math.exp(S[SI["la_" + g]]), 6) for g in GROUPS},
                "anchor_models": (link or {}).get("anchor_models") or {t: "anthropic/claude-%s" % t for t in TIERS},
                "verbosity": {t: round(float(S[SI["v_" + t]]), 6) for t in TIERS}},
        "link": {"alpha": {g: round(float(S[SI["alpha_" + g]]), 6) for g in GROUPS}, "beta": round(float(S[SI["beta"]]), 6),
                 "catalog_sha": (link or {}).get("catalog_sha", ""),
                 "sd_basis": {"anchor": 0.15, "bench": 0.35, "blend": 0.5, "prior": 0.75},
                 "beta_pub": list(beta_pub(link)), "a_chat_hat": (link or {}).get("a_chat_hat"),
                 "sources": (link or {}).get("sources") or [], "fitted": fitted},
        "decide": {"V0": {"cost": 0.05, "balanced": 0.5, "quality": 5.0}, "V0_calibrated": False},   # --v0 replaces it
        "train": {"data_sha256": data_sha, "numpy": np.__version__, "seed": cfg["seed"], "epochs": epochs,
                  "n": {"rows": len(rows), "fit": len(fit), "holdout": len(hold), "calib": len(idx["calib"]),
                        "dev": len(idx["dev"]), "labels": have},
                  "config": {k: cfg[k] for k in ("B", "d", "dropout", "kw", "char3", "lambda_d", "min_labels", "epochs",
                                                  "batch", "seed")},
                  "metrics": mets, "bootstrap": not any(h in trained for h in ("b_sd", "tools", "stakes")),
                  "secs": round(time.time() - t_start, 1)},
    }
    header.update(extra or {})
    size = write_bin(out, header, bloom, emb, dense)
    if size > BIN_MAX:
        raise SystemExit("s1.bin is %d bytes, over the %d cap" % (size, BIN_MAX))
    log("wrote %s (%d bytes); dev %s" % (out, size, json.dumps(mets["dev"])))
    state = {"P": P, "live": live, "W2": W2, "b2": b2, "Q": Q, "sample": sample, "header": header,
             "bloom": bloom, "emb": emb, "dense": dense}
    return header, state


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=os.path.join(HERE, "train.jsonl"))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--link", help="JSON from evals/fit_link.py: beta_pub, se_pub, a_chat_hat, zbar, anchor_models, catalog_sha")
    ap.add_argument("--gate", action="store_true", help="choose U and report §8.4 on the dev half of evals/prompts.jsonl")
    ap.add_argument("--golden", nargs="?", const=GOLDEN, help="write the golden file (default tests/data/s1_golden.json)")
    ap.add_argument("--check", action="store_true", help="retrain with the committed config and compare")
    ap.add_argument("--search", action="store_true", help="grouped 5-fold CV over the §8.1 grid")
    ap.add_argument("--v0", help="evals/alpha/v0.json from `alpha.py calibrate-v`: V0 per objective, written to the header")
    ap.add_argument("--patch", action="store_true", help="with --v0: patch the header of --out instead of retraining")
    ap.add_argument("--quiet", action="store_true")
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=(lambda s: s.lower() in ("1", "true", "yes", "on"))
                        if isinstance(v, bool) else type(v), default=None)
    a = ap.parse_args(argv)
    log = (lambda *x: None) if a.quiet else (lambda *x: print(*x, flush=True))
    cfg = {k: (getattr(a, k) if getattr(a, k) is not None else v) for k, v in DEFAULTS.items()}
    extra = {"decide": v0_of(a.v0)} if a.v0 else None
    if a.patch:
        if not extra:
            raise SystemExit("--patch needs --v0")
        patch(a.out, extra, log)
        if s1.verify(a.out):
            raise SystemExit("sha256 mismatch in %s" % a.out)
        if a.golden:
            golden(a.out, a.golden, log)
        return 0
    link = None
    if a.link:
        with open(a.link, encoding="utf-8") as f:
            link = json.load(f)
    if a.check:
        return check(a, cfg, link, log)
    rows, sha, dropped = load_rows(a.labels)
    if dropped:
        log("dropped %d rows (in evals/prompts.jsonl or split=test)" % dropped)
    if a.search:
        return search(rows, cfg, link, log)
    header, st = build(rows, cfg, a.out, link, sha, log, extra)
    parity(a.out, st["P"], st["live"], st["W2"], st["b2"], st["Q"], st["sample"], log)
    if a.gate:
        header["legacy"] = gate(a.out, log)
        write_bin(a.out, header, st["bloom"], st["emb"], st["dense"])
    bad = s1.verify(a.out)
    if bad:
        raise SystemExit("sha256 mismatch in %s" % bad)
    if a.golden:
        golden(a.out, a.golden, log)
    return 0


def check(a, cfg, link, log):
    """§8.1 --check: retrain with the committed header's config; >= 99.9% identical int8 weights, dev metrics within
    0.5 pt, and the parity check on the retrained file."""
    md, why = s1.load(a.out)
    if md is None:
        raise SystemExit("--check: %s" % why)
    old = md["header"]
    cfg = dict(cfg, **{k: v for k, v in old["train"]["config"].items() if k in cfg})
    rows, sha, _ = load_rows(a.labels)
    if sha != old["train"]["data_sha256"]:
        log("warning: %s differs from the data the committed s1.bin was trained on" % a.labels)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "s1.bin")
        header, st = build(rows, cfg, out, link, sha, log)
        par = parity(out, st["P"], st["live"], st["W2"], st["b2"], st["Q"], st["sample"], log)
        sec = old["sections"]["emb"]
        with open(a.out, "rb") as f:
            f.seek(sec[0])
            old_emb = np.frombuffer(f.read(sec[1]), np.uint8)
        same = float((old_emb == np.frombuffer(st["emb"], np.uint8)).mean())
    od, nd = old["train"]["metrics"]["dev"], header["train"]["metrics"]["dev"]
    drift = {k: round(abs(od[k] - nd[k]), 4) for k in od if k.endswith("_acc") and k in nd}
    ok = same >= 0.999 and all(v <= 0.005 for v in drift.values())
    print(json.dumps({"identical_int8": round(same, 6), "dev_drift": drift, "parity": par, "ok": ok}))
    return 0 if ok else 1


def search(rows, cfg, link, log):
    """5-fold grouped CV over B, d, dropout, kw, char3 (§8.1). Criterion: held-out outcome NLL + 0.5 job NLL when
    outcomes exist, else the held-out joint loss. Prints the table; adopt a config with the flags."""
    import itertools
    T = targets(rows)
    split, clus = assign_splits(rows, log=log)
    train_ix = [i for i, sp in enumerate(split) if sp == "train"]
    fold = {i: int(hashlib.sha256(("fold:" + clus[i]).encode()).hexdigest(), 16) % 5 for i in train_ix}
    results = []
    for B, d, dr, kw, c3 in itertools.product((1 << 14, 1 << 15), (16, 32), (0.1, 0.3), (True, False), (False, True)):
        c = dict(cfg, B=B, d=d, dropout=dr, kw=kw, char3=c3)
        feats, _ = featurize(rows, c)
        crit = []
        for k in range(5):
            fit = np.array([i for i in train_ix if fold[i] != k])
            held = np.array([i for i in train_ix if fold[i] == k])
            bloom, live = bloom_of(feats, fit, B)
            md_b = {"bloom": bloom, "M": M_BLOOM, "B": B}
            ids = [np.array(s1.known(md_b, f)[0], dtype=np.int64) for f in feats]
            P, S = init_params(c, np.random.default_rng(c["seed"])), scalars_init(link)
            P, S, _ = train(P, S, T, ids, fit, held, c, link, live, log=lambda *x: None)
            Y, _ = forward(P, [ids[i] for i in held])
            _, rws = head_loss(Y, S, T, held, c, link, len(fit), grad=False)
            if T["m_out"][held].any():
                aux = aux_rows(Y[:, AUX], S, T, held, 0.0)
                crit.append(float(aux.mean() / 2.0 + 0.5 * metrics(Y, T, held, S, {"T_job": 1.0, "w_nb": 0}, None).get("job_nll", 0)))
            else:
                crit.append(float(rws.mean()))
        results.append((float(np.mean(crit)), c))
        log("B=2^%d d=%d dropout=%.1f kw=%s char3=%s  criterion %.4f" % (B.bit_length() - 1, d, dr, kw, c3, results[-1][0]))
    results.sort(key=lambda r: r[0])
    best = results[0]
    base = next(r for r in results if not r[1]["char3"])
    if best[1]["char3"] and best[0] > base[0] * 0.99:
        best = base   # char 3-grams only when they win by >= 1% (and the latency gate holds; check it separately)
    print("best: " + json.dumps({k: best[1][k] for k in ("B", "d", "dropout", "kw", "char3")}) + "  criterion %.4f" % best[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
