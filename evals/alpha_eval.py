"""Proving alpha (S1 spec §5.1, §9): replay, metrics, the CI gate, the release gate, V0 calibration and the pacing
simulation. Stdlib only (the CI gate imports this); the CLI is evals/alpha.py.

Outcome rows (`evals/alpha/outcomes.jsonl`, §7.8) are scored offline: a policy picks a tier per item (or a mix of tiers),
and its success and dollars come from that tier's recorded outcome and tokens. Item weights are the phase-2 IPW weights
(disagree 1, sampled 1/rate, unsampled 0 for four-tier metrics); cell weights are 1 for haiku and fable and the item
weight for sonnet and opus.
"""
from __future__ import annotations

import json
import math
import os
import random
import struct
import tempfile
import time

import alpha_stats as st
import evalkit as ek

m = ek.m
TIERS = ek.TIERS
INVENTORY = os.path.join(ek.ALPHA, "inventory.json")
OUTCOMES = os.path.join(ek.ALPHA, "outcomes.jsonl")
GATES = os.path.join(ek.ALPHA, "gates.json")
BASELINE = os.path.join(ek.ALPHA, "baseline_preds.jsonl")
V0_FILE = os.path.join(ek.ALPHA, "v0.json")
TARGETS = {"quality": 0.99, "balanced": 0.97, "cost": 0.90}
V0_DEFAULT = {"cost": 0.05, "balanced": 0.5, "quality": 5.0}
V_GRID = [0.005 * (50 / 0.005) ** (k / 119.0) for k in range(120)]
BIN_MAX = 3 * 1024 * 1024
CONST = {"B1": "haiku", "B3": "sonnet", "B3'": "opus", "B2": "fable"}


def load_inventory(path=INVENTORY):
    return ek.read_json(path)


# ── items ──────────────────────────────────────────────────────────────────────

def cell_weight(row, tier):
    if row.get("phase2") in ("disagree", "sampled", "unsampled") and tier in ("sonnet", "opus"):
        return float(row.get("ipw", 0.0))
    return 1.0


def load_items(rows, inv, split=None):
    """Outcome rows -> scoring items with per-tier y, cost (USD), seconds and cell weights."""
    lat, start = inv["latency_prior"], float(inv.get("startup_s", 2.0))
    out = []
    for r in rows:
        if split and r.get("split") != split:
            continue
        y = {t: float(v) for t, v in (r.get("y") or {}).items() if t in TIERS and v is not None}
        tout = dict(r.get("tout") or {})
        known = [v for t, v in tout.items() if t in TIERS and v]
        for t in TIERS:
            tout.setdefault(t, sum(known) / len(known) if known else 500.0)
        cost, secs = {}, {}
        for t in TIERS:
            pin, pout = inv["tiers"][t]["price"]
            cost[t] = (float(r.get("tin") or 0) * pin + float(tout[t]) * pout) / 1e6
            ttft, tps = lat[inv["tiers"][t]["category"]]
            secs[t] = start + ttft + float(tout[t]) / tps
        w = float(r.get("ipw", 1.0)) if all(t in y for t in TIERS) else 0.0
        out.append({"id": r["id"], "cluster": r.get("cluster") or r["id"], "split": r.get("split"), "row": r, "y": y,
                    "w": w, "cw": {t: cell_weight(r, t) for t in y}, "cost": cost, "secs": secs})
    return out


def t_star(y):
    return next((k for k, t in enumerate(TIERS) if y.get(t, 0.0) >= 0.5), None)


# ── policies: {id: [p_haiku, p_sonnet, p_opus, p_fable]} plus optional per-tier P ──────────────────

def onehot(tier):
    return [1.0 if t == tier else 0.0 for t in TIERS]


def constant(items, tier):
    return {it["id"]: onehot(tier) for it in items}


def uniform(items):
    return {it["id"]: [0.25] * 4 for it in items}


def oracle_real(items):
    out = {}
    for it in items:
        k = t_star(it["y"])
        out[it["id"]] = onehot(TIERS[k if k is not None else 0])
    return out


def eu_choice(p, cost, v):
    """argmax_t V*P_t - cost_t (ties: cheaper)."""
    return max(range(4), key=lambda k: (round(v * p[TIERS[k]] - cost[TIERS[k]], 12), -k))


def sweep_points(items, pfn, grid=None):
    """(cost, success) points of a P-driven policy over a V grid (USD per success)."""
    grid = grid or [10 ** (x / 10.0) for x in range(-40, 21)]
    pts = []
    for v in grid:
        ch = {it["id"]: onehot(TIERS[eu_choice(pfn(it), it["cost"], v)]) for it in items if it["w"] > 0}
        s = summarize(items, ch)
        pts.append((s["C"], s["S"]))
    return pts


def ladder_gold(items, cfg=None):
    """B5: the ladder fed panel job and difficulty (rows carry `job` and `difficulty` from the panel)."""
    out = {}
    for it in items:
        r = it["row"]
        if r.get("job") in m.JOBS and r.get("difficulty") in (0, 1, 2, 3):
            cell = m.tier_of(m.target_tier(m.Classification(r["job"], r["difficulty"], 1.0), cfg or {}))
            out[it["id"]] = onehot(TIERS[m.TIERS.index(cell)])
    return out


def route_policy(texts, inv, config=None, inventory=None, home=None):
    """Run the real skill-mode `route()` on each text. Returns (choices, P per tier or {}, info). info["available"] is
    False when a requested decider did not take effect (the decision does not echo it). Quota windows are relaxed (spec §9.1,
    Q = 0): a temporary home gets fresh `relaxed` limits (rewritten every 10 min, so they never go stale), and a decision
    flagged quota_unknown raises."""
    cfg = dict(m.DEFAULT_CONFIG)
    cfg.update(config or {})
    seat_tier = {inv["tiers"][t]["seat_model"]: t for t in TIERS}
    cat_tier = {inv["tiers"][t]["category"]: t for t in TIERS}
    choices, ps, lat, info = {}, {}, [], {"available": True, "decider": "ladder"}
    tmp = None if home else tempfile.TemporaryDirectory(prefix="maestro-replay-")
    home, fresh = home or tmp.name, 0.0
    try:
        env = {"MAESTRO_HOME": home}
        for i, (rid, text) in enumerate(texts.items()):
            if tmp and time.time() - fresh > 600:
                fresh = time.time()
                write_limits(home, "relaxed", fresh, codex=True)
            t0 = time.perf_counter()
            d = m.route(text, harness=inv.get("harness", "claude-code"), inventory=inventory or inv["inventory"], config=cfg,
                        env=env, home=home)
            lat.append((time.perf_counter() - t0) * 1000)
            if "quota_unknown" in (d.get("flags") or ()):
                raise RuntimeError("replay decision for %s is flagged quota_unknown (spec §9.1 replays with Q = 0)" % rid)
            if i == 0:
                info["decider"] = d.get("decider", "ladder")
                want_cls = cfg.get("classifier")
                if cfg.get("decider", "ladder") != info["decider"] or (want_cls not in (None, "auto") and d.get("classifier") != want_cls):
                    info["available"] = False
                    break
            tier = seat_tier.get(d.get("model")) or cat_tier.get(d.get("category"))
            if tier:
                choices[rid] = onehot(tier)
            opts = {}
            for o in d.get("options") or []:
                t = seat_tier.get(str(o.get("seat", "")).split(":")[-1])
                if t and isinstance(o.get("p"), (int, float)):
                    opts[t] = float(o["p"])
            if len(opts) == 4:
                ps[rid] = opts
            post = d.get("posterior")
            if isinstance(post, dict):
                info.setdefault("posterior", {})[rid] = post
                if isinstance(post.get("b"), (list, tuple)) and post["b"]:
                    info.setdefault("mu_b", {})[rid] = float(post["b"][0])
            choices.setdefault(rid, None)
    finally:
        if tmp:
            tmp.cleanup()
    info["latency_ms"] = lat
    return {k: v for k, v in choices.items() if v}, ps, info


def proxy_policy(texts, inv, config=None):
    """Run the real proxy planner (`server.App.plan`, no upstream call) over the four tiers as API models with catalog
    scores. Returns (choices, P per tier or {}, info) like route_policy."""
    from maestro import catalog, score, server
    cat = catalog.load_snapshot()
    catalog._finish(cat)
    ctx = score.make_context(cat)
    by_model, models = {}, []
    for t in TIERS:
        rec = inv["tiers"][t]
        sc = score.score_lab(ctx, cat, rec["model"])
        by_model[rec["model"]] = t
        models.append({"id": rec["model"], "provider": "anthropic", "model": rec["model"].split("/", 1)[1], "access": "api",
                       "style": "anthropic", "tools_ok": True, "vision": True, "context": 200000, "cost_in": rec["price"][0],
                       "cost_out": rec["price"][1], "G": sc["G"], "jobs": sc["jobs"], "category": rec["category"],
                       "serves": rec["category"], "basis": sc["basis"], "benchmarks": [], "enabled": True, "override": None})
    choices, ps, info = {}, {}, {"available": True, "decider": "ladder", "latency_ms": []}
    with tempfile.TemporaryDirectory(prefix="maestro-proxy-replay-") as home:
        ek.write_json(os.path.join(home, "models.json"), {"version": 1, "generated_at": time.time(), "models": models})
        ek.write_json(os.path.join(home, "config.json"), dict(config or {}))
        app = server.App(home, {"ANTHROPIC_API_KEY": "replay-placeholder", "MAESTRO_HOME": home, "HOME": home})
        try:
            for i, (rid, text) in enumerate(texts.items()):
                t0 = time.perf_counter()
                d = app.plan({"model": "maestro/auto", "messages": [{"role": "user", "content": text}]}, {})
                info["latency_ms"].append((time.perf_counter() - t0) * 1000)
                if i == 0:
                    info["decider"] = d.get("decider", "ladder")
                    if (config or {}).get("decider", "ladder") != info["decider"]:
                        info["available"] = False
                        break
                tier = by_model.get(d["order"][0]["m"]["id"]) if d.get("order") else None
                if tier:
                    choices[rid] = onehot(tier)
                opts = {by_model[o["seat"]]: float(o["p"]) for o in d.get("options") or []
                        if o.get("seat") in by_model and isinstance(o.get("p"), (int, float))}
                if len(opts) == 4:
                    ps[rid] = opts
        finally:
            app.db.close()
    return choices, ps, info


def preds_policy(rows):
    """A predictions file ({"id", "chosen", "p"}) as a policy."""
    ch = {r["id"]: onehot(r["chosen"]) for r in rows if r.get("chosen") in TIERS}
    ps = {r["id"]: r["p"] for r in rows if isinstance(r.get("p"), dict) and all(t in r["p"] for t in TIERS)}
    return ch, ps


# ── metrics ────────────────────────────────────────────────────────────────────

def contrib(it, pi):
    """[w, w*S, w*C, w*secs, w*under, w*severe, w*over] for one item under a tier mix `pi`."""
    w, y = it["w"], it["y"]
    if w <= 0 or pi is None:
        return [0.0] * 7
    ys = [y[t] for t in TIERS]
    cs = [it["cost"][t] for t in TIERS]
    ts = t_star(y)
    s = c = secs = under = severe = over = 0.0
    for k, pk in enumerate(pi):
        if not pk:
            continue
        s += pk * ys[k]
        c += pk * cs[k]
        secs += pk * it["secs"][TIERS[k]]
        if ys[k] < 0.5 and any(v >= 0.5 for v in ys[k + 1:]):
            under += pk
        if ys[k] < 0.5 and any(v >= 0.5 for v in ys[k + 2:]):
            severe += pk
        if ts is not None and k > ts:
            over += pk * (cs[k] - cs[ts])
    return [w, w * s, w * c, w * secs, w * under, w * severe, w * over]


KAPPA = 0.002   # $/s of waiting (§5.3): time is reported as a second currency, $ + kappa * seconds


def _norm(tot):
    w = tot[0] or float("nan")
    return {"n_w": tot[0], "S": tot[1] / w, "C": tot[2] / w, "secs": tot[3] / w, "C_time": (tot[2] + KAPPA * tot[3]) / w,
            "under": tot[4] / w, "severe": tot[5] / w, "over": tot[6] / w}


def summarize(items, choices):
    tot = [0.0] * 7
    for it in items:
        if it["id"] in choices:
            for k, v in enumerate(contrib(it, choices[it["id"]])):
                tot[k] += v
    out = _norm(tot)
    out["n"] = sum(1 for it in items if it["w"] > 0 and it["id"] in choices)
    return out


def clusters(items):
    by = {}
    for it in items:
        by.setdefault(it["cluster"], []).append(it)
    return list(by.values())


def const_front(totals):
    """Constant-policy frontier from summed vectors laid out as [W, S_t x4, C_t x4]."""
    w = totals[0]
    return st.frontier([(totals[5 + k] / w, totals[1 + k] / w) for k in range(4)]) if w else []


def alpha_dollar(s, c, front):
    """(alpha_$, alpha_S): 1 - C/C_hull(S) when S is reachable by the hull, else S - S_hull(C)."""
    ch = st.c_at(front, s)
    if ch is not None and ch > 0:
        return 1 - c / ch, None
    sh = st.s_at(front, c)
    return None, (None if sh is None else s - sh)


def alpha_ci(items, choices, n=2000, seed=0, extra_point=None):
    """Cluster-bootstrap CI of alpha_$ of `choices` against the constant hull (plus `extra_point` choices mixed in, for
    alpha_$ against B4). Returns {"alpha", "lo", "hi", "alpha_S"}."""
    vecs = []
    for group in clusters(items):
        v = [0.0] * (11 if extra_point is None else 13)
        for it in group:
            if it["w"] <= 0 or it["id"] not in choices:
                continue
            w = it["w"]
            v[0] += w
            for k, t in enumerate(TIERS):
                v[1 + k] += w * it["y"][t]
                v[5 + k] += w * it["cost"][t]
            cr = contrib(it, choices[it["id"]])
            v[9] += cr[1]
            v[10] += cr[2]
            if extra_point is not None:
                ce = contrib(it, extra_point.get(it["id"]))
                v[11] += ce[1]
                v[12] += ce[2]
        vecs.append(v)

    def fn(tot):
        if not tot[0]:
            return None
        pts = [(tot[5 + k] / tot[0], tot[1 + k] / tot[0]) for k in range(4)]
        if extra_point is not None:
            pts.append((tot[12] / tot[0], tot[11] / tot[0]))
        return alpha_dollar(tot[9] / tot[0], tot[10] / tot[0], st.frontier(pts))[0]
    point, lo, hi = st.cluster_bootstrap(vecs, fn, n, seed)
    tot = [sum(v[k] for v in vecs) for k in range(len(vecs[0]))] if vecs else []
    a_s = None
    if tot and tot[0]:
        pts = [(tot[5 + k] / tot[0], tot[1 + k] / tot[0]) for k in range(4)]
        a_s = alpha_dollar(tot[9] / tot[0], tot[10] / tot[0], st.frontier(pts))[1]
    return {"alpha": point, "lo": lo, "hi": hi, "alpha_S": a_s}


def paired(items, cur, base, n=2000, seed=0):
    """Paired cluster bootstrap of current - baseline: success, cost ratio and severe under-routing."""
    vecs = []
    for group in clusters(items):
        v = [0.0] * 7
        for it in group:
            if it["w"] <= 0 or it["id"] not in cur or it["id"] not in base:
                continue
            a, b = contrib(it, cur[it["id"]]), contrib(it, base[it["id"]])
            v = [v[0] + a[0], v[1] + a[1], v[2] + b[1], v[3] + a[2], v[4] + b[2], v[5] + a[5], v[6] + b[5]]
        vecs.append(v)
    ds = st.cluster_bootstrap(vecs, lambda t: (t[1] - t[2]) / t[0] if t[0] else None, n, seed)
    dc = st.cluster_bootstrap(vecs, lambda t: (t[3] - t[4]) / t[4] if t[4] else None, n, seed)
    return {"dS": ds, "dC_rel": dc}


def cell_lists(items, ps):
    """(p, y, w, cluster) over every observed (item, tier) cell that has a prediction."""
    out = []
    for it in items:
        p = ps.get(it["id"])
        if not p:
            continue
        for t, y in it["y"].items():
            if t in p and it["cw"].get(t, 0) > 0:
                out.append((min(1 - 1e-6, max(1e-6, float(p[t]))), y, it["cw"][t], it["cluster"]))
    return out


def logloss_delta(items, pcur, pbase, n=2000, seed=0):
    by = {}
    for it in items:
        a, b = pcur.get(it["id"]), pbase.get(it["id"])
        if not a or not b:
            continue
        v = by.setdefault(it["cluster"], [0.0, 0.0, 0.0])
        for t, y in it["y"].items():
            w = it["cw"].get(t, 0.0)
            if w > 0 and t in a and t in b:
                ll = lambda p: -(y * math.log(min(1 - 1e-6, max(1e-6, p))) + (1 - y) * math.log(min(1 - 1e-6, max(1e-6, 1 - p))))  # noqa: E731
                v[0] += w
                v[1] += w * ll(float(a[t]))
                v[2] += w * ll(float(b[t]))
    return st.cluster_bootstrap(list(by.values()), lambda t: (t[1] - t[2]) / t[0] if t[0] else None, n, seed)


def slices(items, choices, min_n=30):
    keys = {"job": lambda r: r.get("job"), "difficulty": lambda r: r.get("difficulty"), "context": lambda r: r.get("context"),
            "lang": lambda r: r.get("lang"), "source": lambda r: r.get("source"), "group": lambda r: r.get("group"),
            "length": lambda r: next(b for lim, b in ((20, "<20w"), (60, "20-60w"), (200, "60-200w"), (1e18, "200w+"))
                                     if (r.get("words") or 0) < lim)}
    out = {}
    for name, fn in keys.items():
        groups = {}
        for it in items:
            groups.setdefault(str(fn(it["row"])), []).append(it)
        out[name] = {}
        for k, g in sorted(groups.items()):
            n = sum(1 for it in g if it["w"] > 0)
            out[name][k] = summarize(g, choices) if n >= min_n else "n/a (n=%d)" % n
            if isinstance(out[name][k], dict):
                out[name][k] = {x: round(out[name][k][x], 4) for x in ("S", "C", "severe", "n")}
    return out


def mcnemar_vs(items, a, b):
    """Exact McNemar on success (y >= 0.5) of two policies over items with all four tiers (unweighted)."""
    n01 = n10 = 0
    for it in items:
        if it["w"] <= 0 or it["id"] not in a or it["id"] not in b:
            continue
        sa = it["y"][TIERS[a[it["id"]].index(max(a[it["id"]]))]] >= 0.5
        sb = it["y"][TIERS[b[it["id"]].index(max(b[it["id"]]))]] >= 0.5
        n10 += sa and not sb
        n01 += sb and not sa
    return {"a_only": n10, "b_only": n01, "p": st.mcnemar(n10, n01)}


def heads(items, posteriors):
    """Job accuracy against the panel job and output-token error, from decisions that carry an S1 posterior."""
    rows = [(it, posteriors[it["id"]]) for it in items if it["id"] in posteriors]
    if not rows:
        return None
    job = [(max(p["job_p"], key=p["job_p"].get) == it["row"].get("job")) for it, p in rows if isinstance(p.get("job_p"), dict) and p["job_p"]]
    tok = []
    for it, p in rows:
        t = p.get("tok_out")
        touts = [v for v in (it["row"].get("tout") or {}).values() if v]
        if isinstance(t, (list, tuple)) and touts:
            tok.append(t[0] + t[1] ** 2 / 2 - math.log(sum(touts) / len(touts)))
    return {"n": len(rows), "job_accuracy": sum(job) / len(job) if job else None,
            "tok_log_rmse": math.sqrt(sum(x * x for x in tok) / len(tok)) if tok else None,
            "tok_within_x2": sum(abs(x) <= math.log(2) for x in tok) / len(tok) if tok else None}


def long_fixtures(n=50, chars=60000):
    """Deterministic 60k-character pastes for the latency profile: a log in one fence plus a question, and (every fifth) a
    markdown page of many small fenced snippets (each fence is typed separately, spec §2.2)."""
    out = {}
    for k in range(n):
        if k % 5 == 4:
            snips = ["Step %d: run this\n```%s\nx_%d = compute(%d)\n```\n" % (j, ("", "py", "sh")[j % 3], j, j * k % 97) for j in range(1500)]
            out["long-%02d" % k] = "".join(snips)[:chars - 40] + "\nwhich step is slow?"
            continue
        lines = ["2026-10-02 1%d:%02d:%02d INFO worker-%d batch %d done in %d ms" % (k % 10, j % 60, (j * 7) % 60, j % 9, j, (j * 37 + k) % 997)
                 for j in range(1400)]
        out["long-%02d" % k] = ("```\n" + "\n".join(lines))[:chars - 40] + "\n```\nwhy did batch latency jump?"
    return out


def latency_profile(texts, inv, cfg):
    """In-process route() p50 / p99 / max (ms) after 50 warm-up calls."""
    ts = list(texts.values())
    with tempfile.TemporaryDirectory(prefix="maestro-lat-") as home:
        env = {"MAESTRO_HOME": home}
        for x in (ts * (50 // max(1, len(ts)) + 1))[:50]:
            m.route(x, harness="claude-code", inventory=inv["inventory"], config=cfg, env=env, home=home)
        lat = []
        for x in ts:
            t0 = time.perf_counter()
            m.route(x, harness="claude-code", inventory=inv["inventory"], config=cfg, env=env, home=home)
            lat.append((time.perf_counter() - t0) * 1000)
    return {"n": len(lat), "p50": percentile(lat, 0.5), "p99": percentile(lat, 0.99), "max": max(lat) if lat else None}


# ── irt pieces used by reports ─────────────────────────────────────────────────

def cells_of(items):
    return [(it["id"], t, y, it["cw"].get(t, 0.0)) for it in items for t, y in it["y"].items()]


def b_fit(items, theta=None):
    """1PL on these items' outcomes (anchors fixed when `theta` is given)."""
    return st.fit_1pl(cells_of(items), theta)


def reliability_split_half(items, theta):
    """Spearman-Brown corrected correlation of b fitted from {haiku, opus} vs {sonnet, fable}."""
    a = [(i, t, y, w) for i, t, y, w in cells_of(items) if t in ("haiku", "opus")]
    b = [(i, t, y, w) for i, t, y, w in cells_of(items) if t in ("sonnet", "fable")]
    _, ba = st.fit_1pl(a, theta)
    _, bb = st.fit_1pl(b, theta)
    ids = sorted(set(ba) & set(bb))
    r = st.pearson([ba[i] for i in ids], [bb[i] for i in ids])
    return None if r is None or r <= 0 else 2 * r / (1 + r)


# ── report ─────────────────────────────────────────────────────────────────────

def score(items, policies, n_boot=2000, seed=0):
    """Metrics for every policy {name: (choices, P)}: S, C, secs, under/severe/over, alpha_$ vs the hull with a CI."""
    consts = {t: summarize(items, constant(items, t)) for t in TIERS}
    front = st.frontier([(consts[t]["C"], consts[t]["S"]) for t in TIERS])
    res = {"constants": {t: {k: consts[t][k] for k in ("S", "C", "secs")} for t in TIERS},
           "hull": [list(p) for p in front], "policies": {}}
    sfable = consts["fable"]["S"]
    best = next((t for t in TIERS if consts[t]["S"] >= TARGETS["balanced"] * sfable), "fable")
    res["Bbest"] = {"tier": best, "saving_vs_B2": 1 - consts[best]["C"] / consts["fable"]["C"] if consts["fable"]["C"] else None}
    for name, (ch, ps) in policies.items():
        s = summarize(items, ch)
        row = {k: s[k] for k in ("n", "S", "C", "secs", "C_time", "under", "severe", "over")}
        row["alpha"] = alpha_ci(items, ch, n_boot, seed)
        if ps:
            cells = cell_lists(items, ps)
            if cells:
                c = st.calibration([x[0] for x in cells], [x[1] for x in cells], [x[2] for x in cells], seed=seed)
                row["calibration"] = c
        res["policies"][name] = row
    return res


def capture(items, r_points, theta, b):
    """Alpha capture Α = (AUC_R - AUC_hull) / (AUC_Oirt - AUC_hull) over [C_B1, C_B2], plus APGR and CPT90/95."""
    consts = [(summarize(items, constant(items, t))["C"], summarize(items, constant(items, t))["S"]) for t in TIERS]
    lo, hi = consts[0][0], consts[3][0]
    o_pts = sweep_points(items, lambda it: {t: st.sigmoid(theta[t] - b[it["id"]]) for t in TIERS})
    h = st.envelope_auc(consts, lo, hi)
    r = st.envelope_auc(consts + r_points, lo, hi)
    o = st.envelope_auc(consts + o_pts, lo, hi)
    out = {"A": None if None in (h, r, o) or o - h <= 1e-12 else (r - h) / (o - h)}
    s1, s2 = consts[0][1], consts[3][1]
    for name, pts in (("R", consts + r_points), ("hull", consts), ("O_irt", consts + o_pts)):
        front = st.frontier(pts)
        grid = [lo + (hi - lo) * k / 100.0 for k in range(101)]
        if s2 > s1:
            out["APGR_" + name] = sum(((st.s_at(front, c) or s1) - s1) / (s2 - s1) for c in grid) / len(grid)
        for q in (0.90, 0.95):
            c = st.c_at(front, q * s2)
            out["CPT%d_%s" % (int(q * 100), name)] = None if c is None else c / hi
    return out


def format_report(res):
    lines = ["%-10s %5s %7s %9s %7s %7s %7s %8s  %s" % ("policy", "n", "success", "$/100", "secs", "under", "severe",
                                                        "over$", "alpha_$ vs hull [95% CI]")]
    for name, r in res["policies"].items():
        a = r["alpha"]
        ci = ("%+.3f [%s, %s]" % (a["alpha"], "n/a" if a["lo"] is None else "%+.3f" % a["lo"],
                                  "n/a" if a["hi"] is None else "%+.3f" % a["hi"])) if a["alpha"] is not None else \
            ("alpha_S %+.3f" % a["alpha_S"] if a["alpha_S"] is not None else "n/a")
        lines.append("%-10s %5d %7.3f %9.3f %7.1f %7.3f %7.3f %8.4f  %s" % (
            name, r["n"], r["S"], 100 * r["C"], r["secs"], r["under"], r["severe"], 100 * r["over"], ci))
    lines.append("Bbest (cheapest constant tier at %.0f%% of always-fable success): %s, saving vs B2 %s" % (
        100 * TARGETS["balanced"], res["Bbest"]["tier"],
        "n/a" if res["Bbest"]["saving_vs_B2"] is None else "%.1f%%" % (100 * res["Bbest"]["saving_vs_B2"])))
    return "\n".join(lines)


# ── s1.bin integrity (Appendix A) ──────────────────────────────────────────────

def bin_integrity(path):
    """(size, [problems]) for an s1.bin: magic, header JSON, section sha256 values."""
    with open(path, "rb") as f:
        data = f.read()
    probs = []
    if data[:4] != b"MS1\x00":
        return len(data), ["bad magic"]
    (hl,) = struct.unpack("<I", data[4:8])
    try:
        head = json.loads(data[8:8 + hl].decode("utf-8"))
    except ValueError:
        return len(data), ["header is not JSON"]
    for name, (off, size) in (head.get("sections") or {}).items():
        want = (head.get("sha256") or {}).get(name)
        if want and ek.sha256_bytes(data[off:off + size]) != want:
            probs.append("section %s sha256 mismatch" % name)
    return len(data), probs


def percentile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(math.ceil(q * len(xs))) - 1)] if xs else None


def latency_ratio(texts, inv, cfg, warm=50):
    """(p99 route ms, p99 classify_rules ms) in process after `warm` warm-up calls (§9.5)."""
    ts = list(texts.values())
    if not ts:
        return None, None
    with tempfile.TemporaryDirectory(prefix="maestro-lat-") as home:
        env = {"MAESTRO_HOME": home}
        for x in (ts * (warm // len(ts) + 1))[:warm]:
            m.route(x, harness="claude-code", inventory=inv["inventory"], config=cfg, env=env, home=home)
            m.classify_rules(x)
        r, c = [], []
        for x in ts:   # the best of 3 per text, interleaved: p99 over texts, not over scheduler noise
            rr, cc = [], []
            for _ in range(3):
                t0 = time.perf_counter()
                m.route(x, harness="claude-code", inventory=inv["inventory"], config=cfg, env=env, home=home)
                rr.append((time.perf_counter() - t0) * 1000)
                t0 = time.perf_counter()
                m.classify_rules(x)
                cc.append((time.perf_counter() - t0) * 1000)
            r.append(min(rr))
            c.append(min(cc))
    return percentile(r, 0.99), percentile(c, 0.99)


def write_limits(home, scenario, now=None, codex=False):
    """The statusline cache (and §4.3 history log) for a quota scenario: relaxed, pressured or fable-exhausted; codex=True
    also writes relaxed Codex windows (quota_hist.json)."""
    now = time.time() if now is None else now
    five = {"relaxed": 20, "pressured": 80, "fable-exhausted": 20}[scenario]
    rl = {"five_hour": {"used_percentage": five, "resets_at": now + 3 * 3600},
          "seven_day": {"used_percentage": 20, "resets_at": now + 4 * 86400}}
    if scenario == "fable-exhausted":
        rl["seven_day_fable"] = {"used_percentage": 100, "resets_at": now + 4 * 86400}
    ek.write_json(os.path.join(home, "claude_limits.json"), {"rate_limits": rl, "updated_at": now})
    log = [{"t": now - 3600 + 600 * k, "five_hour": max(0, five - 30 + 5 * k) if scenario == "pressured" else five,
            "seven_day": 20} for k in range(7)]
    ek.write_jsonl(os.path.join(home, "claude_limits.log"), log)
    if codex:
        ek.write_json(os.path.join(home, "quota_hist.json"), {"version": 1, "windows": {}, "current": {
            "codex/primary": {"used": 0.2, "reset_at": now + 3 * 3600, "length_s": 18000, "t": now},
            "codex/secondary": {"used": 0.2, "reset_at": now + 4 * 86400, "length_s": 604800, "t": now}}})


def quota_shares(texts, inv, cfg):
    """{scenario: (share to Claude seats, share to fable)} for the three fixed scenarios."""
    out, cfg = {}, dict(m.DEFAULT_CONFIG, **cfg)
    for sc in ("relaxed", "pressured", "fable-exhausted"):
        with tempfile.TemporaryDirectory(prefix="maestro-quota-") as home:
            write_limits(home, sc)
            n = claude = fable = 0
            for text in texts.values():
                d = m.route(text, harness="claude-code", inventory=inv["quota_inventory"], config=cfg,
                            env={"MAESTRO_HOME": home}, home=home)
                n += 1
                claude += d.get("seat") == "claude-code"
                fable += d.get("seat") == "claude-code" and d.get("model") == "fable"
            out[sc] = (claude / float(n or 1), fable / float(n or 1))
    return out


# ── the CI gate (§9.5) ─────────────────────────────────────────────────────────

def header_v0(bin_path=ek.S1_BIN):
    """V0 per objective as shipped: the s1.bin header's (train_s1.py --v0), else decide.json's placeholders."""
    from maestro import decide, s1
    md = s1.load(bin_path)[0]
    return dict(decide.params(md and md["header"])["decide"]["V0"])


def v0_check(bin_path=ek.S1_BIN, v0_path=V0_FILE):
    """(status, detail): does the shipped header carry the V0 that `alpha.py calibrate-v` wrote to evals/alpha/v0.json?"""
    want = {k: v for k, v in ((ek.read_json(v0_path) or {}).get("V0") or {}).items() if isinstance(v, (int, float))}
    if not want:
        return "skip", "no evals/alpha/v0.json yet (alpha.py calibrate-v); replays use the header's V0"
    have = header_v0(bin_path)
    bad = {k: (have.get(k), v) for k, v in want.items() if abs((have.get(k) or 0) - v) > 1e-9}
    return ("fail", "s1.bin header V0 %s differs from v0.json; run train_s1.py --v0 evals/alpha/v0.json --patch --golden" % bad) \
        if bad else ("pass", "header V0 %s" % have)


def current_policy(texts, inv, objective="balanced"):
    """The policy under test: S1 + decide when `route` supports the utility decider, else today's ladder. No wtp_usd: V0 comes
    from the s1.bin header, as in production."""
    cfg = {"decider": "utility", "objective": objective}
    ch, ps, info = route_policy(dict(list(texts.items())[:1]), inv, cfg)
    if not info["available"]:
        cfg = {"objective": objective}
    ch, ps, info = route_policy(texts, inv, cfg)
    return ch, ps, info, cfg


def gate(rows, inv, texts, gates=None, base_rows=None, bin_path=ek.S1_BIN, n_boot=2000, seed=0, split="test"):
    """Returns (checks [(name, status, detail)], current (choices, ps, info, cfg), items). status: pass | fail | skip."""
    items = [it for it in load_items(rows, inv, split) if it["id"] in texts or it["w"] <= 0]
    texts = {k: v for k, v in texts.items() if k in {it["id"] for it in items}}
    ch, ps, info, cfg = current_policy(texts, inv)
    checks = [("V0 in s1.bin",) + v0_check(bin_path)] if info["decider"] == "utility" else []
    gates = gates or {}
    if base_rows:
        bch, bps = preds_policy(base_rows)
        pr = paired(items, ch, bch, n_boot, seed)
        pt, lo, hi = pr["dS"]
        checks.append(("dS_bal (paired)", "fail" if lo is not None and lo < -0.02 else "pass",
                       "%+.4f [%s, %s]" % (pt or 0, _f(lo), _f(hi))))
        pt, lo, hi = pr["dC_rel"]
        checks.append(("dC_bal / C_base", "fail" if hi is not None and hi > 0.05 else "pass",
                       "%+.4f [%s, %s]" % (pt or 0, _f(lo), _f(hi))))
        if ps and bps:
            pt, lo, hi = logloss_delta(items, ps, bps, n_boot, seed)
            checks.append(("d logloss (all cells)", "fail" if hi is not None and hi > 0.01 else "pass",
                           "%+.4f [%s, %s] nats" % (pt or 0, _f(lo), _f(hi))))
        else:
            checks.append(("d logloss (all cells)", "skip", "no P from %s" % ("the current policy" if not ps else "the baseline")))
    else:
        checks.append(("paired deltas", "skip", "no baseline_preds.jsonl yet (alpha.py gate --update)"))
    s = summarize(items, ch)
    committed = (gates.get("baseline") or {}).get("severe")
    checks.append(("severe under-routing", "fail" if committed is not None and s["severe"] > committed + 0.01 else "pass",
                   "%.4f (committed %s)" % (s["severe"], _f(committed))))
    cells = cell_lists(items, ps)
    if cells:
        c = st.calibration([x[0] for x in cells], [x[1] for x in cells], [x[2] for x in cells], seed=seed)
        checks.append(("ECE (all cells)", "fail" if c["ece"] > c["ece_null_p95"] + 0.02 else "pass",
                       "%.4f vs null p95 %.4f + 0.02 (n=%d)" % (c["ece"], c["ece_null_p95"], c["n"])))
        checks.append(("calibration slope", "fail" if not 0.8 <= c["slope"] <= 1.25 else "pass", "%.3f" % c["slope"]))
    else:
        checks.append(("calibration", "skip", "the %s decider reports no P per tier" % info["decider"]))
    if info["decider"] == "utility":
        q = quota_shares(texts, inv, cfg)
        bad = q["pressured"][0] > q["relaxed"][0] or q["fable-exhausted"][1] > q["relaxed"][1]
        checks.append(("quota monotonicity", "fail" if bad else "pass",
                       "claude share relaxed %.3f pressured %.3f; fable share relaxed %.3f exhausted %.3f" % (
                           q["relaxed"][0], q["pressured"][0], q["relaxed"][1], q["fable-exhausted"][1])))
    else:
        checks.append(("quota monotonicity", "skip", "the ladder decider ignores quota in skill mode"))
    if os.path.exists(bin_path):
        size, probs = bin_integrity(bin_path)
        checks.append(("s1.bin size", "fail" if size > gates.get("bin_max_bytes", BIN_MAX) else "pass", "%d bytes" % size))
        checks.append(("s1.bin integrity", "fail" if probs else "pass", "; ".join(probs) or "section sha256 ok"))
    else:
        checks.append(("s1.bin", "skip", "no s1.bin yet"))
    p99, c99 = latency_ratio(texts, inv, dict(m.DEFAULT_CONFIG, **cfg))
    if p99 is not None:
        ratio, committed = p99 / max(c99, 1e-6), gates.get("latency_ratio_p99") or 0
        bad = p99 > gates.get("abs_p99_ms", 5.0) or (committed and ratio > committed * 1.5)
        checks.append(("latency", "fail" if bad else "pass", "p99 %.3f ms, ratio %.2f (committed %s)" % (p99, ratio, _f(committed))))
    return checks, (ch, ps, info, cfg), items


def _f(x):
    return "n/a" if x is None else "%.4f" % x


def gate_update(path_gates, path_base, items, current, texts, inv, split="test"):
    ch, ps, info, cfg = current
    s = summarize(items, ch)
    cells = cell_lists(items, ps)
    ll = st.calibration([x[0] for x in cells], [x[1] for x in cells], [x[2] for x in cells], null=False)["logloss"] if cells else 0.0
    p99, c99 = latency_ratio(texts, inv, dict(m.DEFAULT_CONFIG, **cfg))
    ek.write_json(path_gates, {"split": split, "n_items": s["n"], "n_cells": len(cells), "weights_sha": ek.weights_sha(),
                               "decider": info["decider"], "V0_balanced": header_v0()["balanced"],
                               "baseline": {"S_bal": round(s["S"], 6), "C_bal": round(s["C"], 8), "logloss": round(ll, 6),
                                            "severe": round(s["severe"], 6)},
                               "latency_ratio_p99": round(p99 / max(c99, 1e-6), 3) if p99 else 0.0,
                               "bin_max_bytes": BIN_MAX, "abs_p99_ms": 5.0})
    rows = []
    for it in items:
        if it["id"] in ch:
            row = {"id": it["id"], "chosen": TIERS[ch[it["id"]].index(1.0)]}
            if it["id"] in ps:
                row["p"] = {t: round(ps[it["id"]][t], 6) for t in TIERS}
            rows.append(row)
    ek.write_jsonl(path_base, rows)
    return len(rows)


# ── V0 calibration (§5.1) ──────────────────────────────────────────────────────

def calibrate_v(items, choose, grid=V_GRID, targets=None):
    """Smallest V0 on `grid` whose replay success is >= target x always-fable success, per objective. `choose(obj, V0)`
    returns choices {id: tier mix}. Bisection over the grid (success is non-decreasing in V0)."""
    sfable = summarize(items, constant(items, "fable"))["S"]
    out = {}
    for obj, share in (targets or TARGETS).items():
        want = share * sfable
        ok = lambda k: summarize(items, choose(obj, grid[k]))["S"] >= want - 1e-12  # noqa: E731
        lo, hi = 0, len(grid) - 1
        if not ok(hi):
            out[obj] = {"V0": None, "note": "never reaches %.3f of always-fable success" % share}
            continue
        while lo < hi:
            mid = (lo + hi) // 2
            if ok(mid):
                hi = mid
            else:
                lo = mid + 1
        s = summarize(items, choose(obj, grid[lo]))
        out[obj] = {"V0": round(grid[lo], 6), "S": round(s["S"], 4), "C": s["C"], "target": round(want, 4)}
    return out


# ── pacing simulation (§9.4) ───────────────────────────────────────────────────

def _g(used, tau, length, rate, reserve, du, gmax, now=1.8e9):
    """§4.3 pacing multiplier g of one window, computed by decide.quota_price (the code the router runs): `rate` (used per second)
    becomes a two-sample burn history, `du` the task's share through decide.json's k prior. A hard skip prices at g_max."""
    from maestro import decide as D
    short = length <= 6 * 3600
    hist = [[now - 1200, used - rate * 1200], [now, used]] if rate is not None else []
    win = {"name": "five_hour" if short else "seven_day", "used": used, "reset_at": now + tau, "length_s": length, "hist": hist}
    t = {"wcache": {}, "now": now, "reserve": reserve, "k": {}, "windows": {"sim": {"as_of": now, "windows": [win]}}}
    k0 = D.table()["pacing"]["k_prior"]["short" if short else "long"]
    q, why, _ = D.quota_price({"quota": "sim", "fam": ""}, du / k0, 1.0, gmax, D.windows_of("sim", t), t)
    return gmax if why else q


def pacing(tasks, days=14, seed=0, k_prior=(0.02, 0.003), own=(0.05, 0.004)):
    """Simulate (a) subscription until exhausted then API, (b) the §4.3 pacing price, (c) API only, over `days` of
    weekday bursts. `tasks`: [(y, api_usd)] drawn in order. Windows: Claude 5 h (opened by first use) and weekly,
    Codex 5 h; true k drawn in [0.5x, 2x] of the prior; the user's own session burns `own` per working hour."""
    rng = random.Random(seed)
    k_true = (k_prior[0] * rng.uniform(0.5, 2.0), k_prior[1] * rng.uniform(0.5, 2.0))
    stream = []
    for h in range(days * 24):
        day, hour = divmod(h, 24)
        busy = day % 7 < 5 and 9 <= hour < 18
        n = rng.randint(20, 60) if busy else rng.choice((0, 0, 0, 1, 2))
        stream += [(h * 3600 + rng.uniform(0, 3600), busy) for _ in range(n)]
    stream.sort()
    res = {}
    for pol in ("a", "b", "c"):
        st5 = {"claude": [0.0, None], "codex": [0.0, None]}  # used, window start
        wk, wk_start = 0.0, 0.0
        hist = []
        api = succ = 0.0
        exhausted_hours, waste5, wastewk = set(), [], []
        last_own = 0.0
        for n, (t, busy) in enumerate(stream):
            y, usd = tasks[n % len(tasks)]
            for name, s in st5.items():
                if s[1] is not None and t - s[1] >= 5 * 3600:
                    waste5.append(max(0.0, 1 - s[0]))
                    s[0], s[1] = 0.0, None
            if t - wk_start >= 7 * 86400:
                wastewk.append(max(0.0, 1 - wk))
                wk, wk_start = 0.0, wk_start + 7 * 86400
            if busy and t - last_own >= 600:  # the user's own session, every 10 minutes of working time
                if st5["claude"][1] is None:
                    st5["claude"][1] = t
                st5["claude"][0] += own[0] / 6
                wk += own[1] / 6
                last_own = t
            hist = [(ht, u) for ht, u in hist if t - ht <= 3600] + [(t, st5["claude"][0])]
            seat = "api"
            if pol == "a" and st5["claude"][0] < 1.0 and wk < 1.0:
                seat = "claude"
            elif pol == "b":
                du5, duw = k_prior[0] * usd, k_prior[1] * usd
                rate = None
                if len(hist) >= 2 and hist[-1][0] - hist[0][0] >= 600:
                    rate = (hist[-1][1] - hist[0][1]) / (hist[-1][0] - hist[0][0])
                tau5 = 5 * 3600 - (t - st5["claude"][1]) if st5["claude"][1] is not None else 5 * 3600
                tauw = 7 * 86400 - (t - wk_start)
                skip = st5["claude"][0] + du5 > 1 - 0.15 / 3 or wk + duw > 1 - 0.15 / 3
                g = max(_g(st5["claude"][0], tau5, 5 * 3600, rate, 0.15, du5, 1.0), _g(wk, tauw, 7 * 86400, None, 0.15, duw, 1.0))
                taux = 5 * 3600 - (t - st5["codex"][1]) if st5["codex"][1] is not None else 5 * 3600
                gx = _g(st5["codex"][0], taux, 5 * 3600, None, 0.15, du5, 1.0)
                skipx = st5["codex"][0] + du5 > 1 - 0.15 / 3
                opts = [(g * usd, 0, "claude")] if not skip else []
                opts += [(gx * usd, 1, "codex")] if not skipx else []
                opts.append((usd, 2, "api"))
                seat = min(opts)[2]
            if seat in ("claude", "codex"):
                s = st5[seat]
                if s[1] is None:
                    s[1] = t
                s[0] += k_true[0] * usd
                if seat == "claude":
                    wk += k_true[1] * usd
            else:
                api += usd
            succ += y
            if st5["claude"][0] >= 1.0 or wk >= 1.0:
                exhausted_hours.add(int(t // 3600))
        res[pol] = {"success": succ / len(stream), "api_usd": api, "exhausted_hours": len(exhausted_hours),
                    "waste_5h": sum(waste5) / len(waste5) if waste5 else 0.0,
                    "waste_weekly": sum(wastewk) / len(wastewk) if wastewk else 0.0, "tasks": len(stream)}
    a, b, c = res["a"], res["b"], res["c"]
    res["checks"] = {"b_exhausted_le_a": b["exhausted_hours"] <= a["exhausted_hours"], "b_api_le_c": b["api_usd"] <= c["api_usd"],
                     "b_success_within_1pt_of_a": abs(b["success"] - a["success"]) <= 0.01}
    return res
