"""Shared helpers for the arena: paths, files, small statistics. Stdlib only."""
from __future__ import annotations

import fnmatch
import math
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))          # evals/arena
EVALS = os.path.dirname(HERE)
if EVALS not in sys.path:
    sys.path.insert(0, EVALS)

import alpha_stats as st  # noqa: E402
import evalkit as ek  # noqa: E402  (also puts the maestro package on sys.path)

m = ek.m
RUNS = os.path.join(HERE, "runs")              # gitignored: answers, diffs and results stay on this machine
FIXTURES = os.path.join(HERE, "fixtures")
SEATS_FILE = os.path.join(HERE, "seats.json")
SUITES = ("verifiable", "agentic", "open")

read_jsonl, read_json, write_json, est_tokens = ek.read_jsonl, ek.read_json, ek.write_json, ek.est_tokens
_LOCK = threading.Lock()


def append_row(path, row):
    """Append one JSON row (thread safe; a crash loses at most the row being written)."""
    with _LOCK:
        ek.append_jsonl(path, [row])


def globs(values):
    """`--seats a,b --seats c` -> ["a", "b", "c"]."""
    out = []
    for v in values or []:
        out += [x.strip() for x in str(v).split(",") if x.strip()]
    return out


def matches(name, patterns):
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def wilson(k, n, z=1.96):
    """(point, lo, hi) of a binomial proportion (Wilson score interval); (None, None, None) when n == 0."""
    if not n:
        return None, None, None
    p = k / float(n)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def rate_ci(by_cluster, n_boot=1000, seed=0):
    """(point, lo, hi) of a success rate. `by_cluster`: {task: [y, ...]} (repeats of one task are one cluster).
    One observation per task: Wilson; repeats: a cluster bootstrap over tasks (alpha_stats.cluster_bootstrap)."""
    obs = [ys for ys in by_cluster.values() if ys]
    if not obs:
        return None, None, None
    if all(len(ys) == 1 for ys in obs):
        return wilson(sum(ys[0] for ys in obs), len(obs))
    vecs = [[float(sum(ys)), float(len(ys))] for ys in obs]
    return st.cluster_bootstrap(vecs, lambda v: v[0] / v[1] if v[1] else None, n=n_boot, seed=seed)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, spec="%.2f", none="-"):
    return none if x is None else spec % x


def pct(x, none="-"):
    return none if x is None else "%.0f%%" % (100 * x)


def usd(x, none="-"):
    if x is None:
        return none
    return "$%.4f" % x if abs(x) < 0.1 else "$%.2f" % x


def ci_text(p, lo, hi, as_pct=True):
    if p is None:
        return "-"
    f = (lambda v: "%.0f%%" % (100 * v)) if as_pct else (lambda v: "%+.2f" % v)
    return "%s [%s, %s]" % (f(p), "?" if lo is None else f(lo), "?" if hi is None else f(hi))
