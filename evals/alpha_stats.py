"""Statistics for the alpha set (S1 spec §7.1, §8.2, §9.2-§9.3). Pure functions, stdlib only: the CI gate imports this.

Cells are (item, tier, y, w) with y in [0, 1] (soft labels allowed) and w the inverse-probability weight.
"""
from __future__ import annotations

import bisect
import math
import random

TIER_ORDER = ("haiku", "sonnet", "opus", "fable")


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def logit(p, eps=1e-6):
    p = min(1 - eps, max(eps, p))
    return math.log(p / (1 - p))


def ranks(xs):
    """Average ranks (1-based), ties share the mean rank."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def pearson(x, y):
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx, sy = math.sqrt(sum((a - mx) ** 2 for a in x)), math.sqrt(sum((b - my) ** 2 for b in y))
    return None if sx == 0 or sy == 0 else sxy / (sx * sy)


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def auc(ys, ps, ws=None):
    """Weighted Mann-Whitney AUC of scores `ps` for binary-ish outcomes `ys` (y >= 0.5 is positive)."""
    ws = ws or [1.0] * len(ys)
    pos = sorted((p, w) for y, p, w in zip(ys, ps, ws) if y >= 0.5 and w > 0)
    neg = sorted((p, w) for y, p, w in zip(ys, ps, ws) if y < 0.5 and w > 0)
    wp, wn = sum(w for _, w in pos), sum(w for _, w in neg)
    if not wp or not wn:
        return None
    keys = [p for p, _ in neg]
    cum = [0.0]
    for _, w in neg:
        cum.append(cum[-1] + w)
    tot = 0.0
    for p, w in pos:
        lo, hi = bisect.bisect_left(keys, p), bisect.bisect_right(keys, p)
        tot += w * (cum[lo] + 0.5 * (cum[hi] - cum[lo]))
    return tot / (wp * wn)


def cohen_kappa(pairs):
    pairs = list(pairs)
    if not pairs:
        return None
    cats = sorted({x for p in pairs for x in p})
    n = float(len(pairs))
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in cats)
    return 1.0 if pe >= 1 else (po - pe) / (1 - pe)


def mcnemar(b, c):
    """Exact two-sided McNemar p-value for b and c discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    if n > 500:  # normal approximation with continuity correction
        return min(1.0, math.erfc(max(0.0, abs(b - c) - 1) / math.sqrt(n) / math.sqrt(2)))
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2 * tail)


# ── IRT (1PL, a = 1) ───────────────────────────────────────────────────────────

def fit_1pl(cells, theta=None, prior_sd=3.0, iters=60, tol=1e-6):
    """MAP fit of P(y=1) = sigmoid(theta_t - b_i). `theta` given = anchors fixed (only b is fitted); otherwise theta is
    fitted too with theta[haiku] := 0 (or the first tier present). b has a N(0, prior_sd^2) prior, so items solved by
    every tier or by none stay finite. Returns (theta, b)."""
    by_item, by_tier = {}, {}
    for i, t, y, w in cells:
        if w > 0:
            by_item.setdefault(i, []).append((t, y, w))
            by_tier.setdefault(t, []).append((i, y, w))
    fixed = theta is not None
    th = dict(theta) if fixed else {t: 0.0 for t in by_tier}
    if not fixed:  # a sensible start: logit of each tier's success
        for t, cs in by_tier.items():
            th[t] = logit(sum(y * w for _, y, w in cs) / sum(w for _, _, w in cs), 0.02)
    b = {i: 0.0 for i in by_item}
    pin = next((t for t in TIER_ORDER if t in th), None)
    for _ in range(iters):
        delta = 0.0
        for i, cs in by_item.items():  # Newton on the negative log posterior of b_i
            g, h = b[i] / prior_sd ** 2, 1.0 / prior_sd ** 2
            for t, y, w in cs:
                p = sigmoid(th.get(t, 0.0) - b[i])
                g += w * (y - p)
                h += w * p * (1 - p)
            step = max(-2.0, min(2.0, g / h))
            b[i] -= step
            delta = max(delta, abs(step))
        if not fixed:
            for t, cs in by_tier.items():
                g, h = 1e-4 * -th[t], 1e-4
                for i, y, w in cs:
                    p = sigmoid(th[t] - b[i])
                    g += w * (y - p)
                    h += w * p * (1 - p)
                step = max(-2.0, min(2.0, g / h))
                th[t] += step
                delta = max(delta, abs(step))
            if pin is not None and th[pin] != 0:
                s = th[pin]
                th = {t: v - s for t, v in th.items()}
                b = {i: v - s for i, v in b.items()}
        if delta < tol:
            break
    return th, b


# ── constant hull, envelopes ───────────────────────────────────────────────────

def frontier(points):
    """The lower convex hull of (cost, success) points as an increasing frontier: the cheapest way to reach each
    success level by mixing constant policies. Points not on it (dominated or inside) are dropped."""
    pts = sorted(set((float(c), float(s)) for c, s in points), key=lambda p: (p[0], -p[1]))
    best = []
    for c, s in pts:  # keep the best success at each cost, increasing in success
        if not best or s > best[-1][1]:
            if best and best[-1][0] == c:
                best[-1] = (c, s)
            else:
                best.append((c, s))
    hull = []
    for p in best:  # upper concave chain
        while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1]) >= (p[0] - hull[-2][0]) * (hull[-1][1] - hull[-2][1]):
            hull.pop()
        hull.append(p)
    return hull


def s_at(front, c):
    """Success of the frontier at cost c (None below the cheapest point; the top success beyond the last)."""
    if not front or c < front[0][0] - 1e-12:
        return None
    for (c1, s1), (c2, s2) in zip(front, front[1:]):
        if c <= c2:
            return s1 + (s2 - s1) * (c - c1) / (c2 - c1)
    return front[-1][1]


def c_at(front, s):
    """Cost of the frontier at success s (the cheapest point's cost below it; None above the top)."""
    if not front or s > front[-1][1] + 1e-12:
        return None
    if s <= front[0][1]:
        return front[0][0]
    for (c1, s1), (c2, s2) in zip(front, front[1:]):
        if s <= s2:
            return c1 + (c2 - c1) * (s - s1) / (s2 - s1)
    return front[-1][0]


def envelope_auc(points, lo, hi, grid=80):
    """Mean success of the upper concave envelope of `points` over a log-cost grid on [lo, hi] (alpha capture, §9.2)."""
    front = frontier(points)
    if not front or lo <= 0 or hi <= lo:
        return None
    vals = []
    for k in range(grid):
        c = math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * k / (grid - 1))
        s = s_at(front, c)
        vals.append(front[0][1] if s is None else s)
    return sum(vals) / len(vals)


# ── calibration ────────────────────────────────────────────────────────────────

def logistic_fit(xs, ys, ws, iters=50):
    """Weighted logistic regression y ~ sigmoid(a + b x) by Newton; returns (a, b)."""
    a, b = 0.0, 1.0
    for _ in range(iters):
        ga = gb = haa = hab = hbb = 0.0
        for x, y, w in zip(xs, ys, ws):
            p = sigmoid(a + b * x)
            r, v = w * (y - p), w * p * (1 - p)
            ga += r
            gb += r * x
            haa += v
            hab += v * x
            hbb += v * x * x
        haa += 1e-9
        hbb += 1e-9
        det = haa * hbb - hab * hab
        if det <= 0:
            break
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        a, b = a + da, b + db
        if abs(da) + abs(db) < 1e-9:
            break
    return a, b


def ece(ps, ys, ws=None, bins=10):
    """Expected calibration error with `bins` equal-mass (equal weight) bins."""
    ws = ws or [1.0] * len(ps)
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    total = sum(ws)
    if not total:
        return None
    out, acc, cur = 0.0, 0.0, [0.0, 0.0, 0.0]
    edge = total / bins
    for i in order:
        cur[0] += ws[i]
        cur[1] += ws[i] * ps[i]
        cur[2] += ws[i] * ys[i]
        acc += ws[i]
        if acc >= edge * (1 - 1e-9) or i == order[-1]:
            out += abs(cur[2] - cur[1])
            cur, acc = [0.0, 0.0, 0.0], 0.0
    return out / total


def ece_null_p95(ps, ws=None, sims=200, seed=0, bins=10):
    """The 95th percentile of the ECE a perfectly calibrated P would show on these cells (§9.3)."""
    rng = random.Random(seed)
    vals = sorted(ece(ps, [1.0 if rng.random() < p else 0.0 for p in ps], ws, bins) for _ in range(sims))
    return vals[min(len(vals) - 1, int(math.ceil(0.95 * len(vals))) - 1)]


def calibration(ps, ys, ws=None, null=True, seed=0):
    ws = ws or [1.0] * len(ps)
    tot = sum(ws)
    if not tot:
        return {"n": 0}
    ll = -sum(w * (y * math.log(max(1e-12, p)) + (1 - y) * math.log(max(1e-12, 1 - p))) for p, y, w in zip(ps, ys, ws)) / tot
    brier = sum(w * (p - y) ** 2 for p, y, w in zip(ps, ys, ws)) / tot
    a, b = logistic_fit([logit(p) for p in ps], ys, ws)
    out = {"n": len(ps), "logloss": ll, "brier": brier, "ece": ece(ps, ys, ws), "slope": b, "intercept": a,
           "auc": auc(ys, ps, ws)}
    if null:
        out["ece_null_p95"] = ece_null_p95(ps, ws, seed=seed)
    return out


def nelder_mead(f, x0, step=0.5, iters=500, tol=1e-9):
    """Minimize f over a few parameters (the panel-validity ordinal fits). Returns (x, f(x))."""
    n = len(x0)
    pts = [list(x0)] + [[x0[j] + (step if j == i else 0.0) for j in range(n)] for i in range(n)]
    vals = [f(x) for x in pts]
    for _ in range(iters):
        order = sorted(range(n + 1), key=vals.__getitem__)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        c = [sum(x[j] for x in pts[:-1]) / n for j in range(n)]
        xr = [2 * c[j] - pts[-1][j] for j in range(n)]
        fr = f(xr)
        if fr < vals[0]:
            xe = [3 * c[j] - 2 * pts[-1][j] for j in range(n)]
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = [0.5 * (c[j] + pts[-1][j]) for j in range(n)]
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [[0.5 * (pts[0][j] + x[j]) for j in range(n)] for x in pts[1:]]
                vals = [vals[0]] + [f(x) for x in pts[1:]]
    k = min(range(n + 1), key=vals.__getitem__)
    return pts[k], vals[k]


# ── bootstrap ──────────────────────────────────────────────────────────────────

def cluster_bootstrap(vectors, fn, n=2000, seed=0, level=0.95):
    """Cluster bootstrap. `vectors`: one list of floats per cluster (sums of that cluster's per-item quantities); `fn`
    maps the summed vector of a resample to a statistic (or None). Returns (point, lo, hi); lo/hi are None when too
    few resamples give a value."""
    if not vectors:
        return None, None, None
    q = len(vectors[0])
    total = [sum(v[k] for v in vectors) for k in range(q)]
    point = fn(total)
    rng = random.Random(seed)
    nc = len(vectors)
    stats = []
    for _ in range(n):
        counts = {}
        for _ in range(nc):
            j = rng.randrange(nc)
            counts[j] = counts.get(j, 0) + 1
        acc = [0.0] * q
        for j, m in counts.items():
            v = vectors[j]
            for k in range(q):
                acc[k] += m * v[k]
        s = fn(acc)
        if s is not None:
            stats.append(s)
    if len(stats) < 0.5 * n:
        return point, None, None
    stats.sort()
    lo = stats[int((1 - level) / 2 * (len(stats) - 1))]
    hi = stats[int((1 + level) / 2 * (len(stats) - 1))]
    return point, lo, hi
