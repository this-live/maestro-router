"""Maestro S1 decision layer: rank seats by expected utility in dollars (spec "S1 and expected-utility routing", §1, §3-§5).

    EU(s|x) = V·P̃ − C − Q − κ·T − (1 − P̃)·F      argmax subject to P̃ ≥ P_min(stakes); ties: EU (1e-6), then C + Q, then id

posterior (s1.posterior and the NB / Jev adapters), a plain dict, as spec §2.1:
    {"job_p": {job: p}, "b": [mu, sd], "group": "chat"|"agentic", "tools", "self_contained", "stakes" (E, 0..2), "detectable",
     "ambiguity", "tok_out": [mu, sd] (log tokens of one answer or turn), "turns": [mu, sd] (log turns), "ood", "source",
     "hints" (fused by s1.posterior, spec §5.6; decide reads only "context_tokens", the inline seat's context)}
seats: seat_of(router candidate | server candidate {"m": entry, ...} | models.json entry), inline_seat(), with_efforts().
ctx: {"mode": "skill"|"proxy", "cfg", "home", "now", "est_in", "prefix_in", "warm" (affinity seat id), "windows" (read_windows),
      "s1" (the s1.bin header; decide.json "s1_defaults" fills what it lacks), "text" (privacy check only), "private" (true: local only,
      e.g. a key-like string elsewhere in the conversation), "background",
      "catalog_sha", "link_stale", "difficulty", "decision_id"}.
Stdlib only. Never imports quota, providers or the router: `route` reads plain files (decide.json, anchors.json, personal.json,
the quota snapshots) through an mtime cache.
"""
from __future__ import annotations

import bisect
import json
import math
import os
import re
import time
import zlib

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "decide.json")
TIERS = ("fast", "standard", "strong", "frontier")
ANCHORS = ("haiku", "sonnet", "opus", "fable")          # the Claude anchor tier of each cloud category
QUOTA_OF = {"claude-code": "claude", "codex": "codex", "gemini-cli": "gemini"}
VENDOR_OF = {"claude": "anthropic", "codex": "openai", "gemini": "google"}
CONFIG_DEFAULTS = {"objective": "balanced", "wtp_usd": None, "latency_usd_per_s": 0.002, "background_latency_usd_per_s": 0.0002,
                   "risk": 1.0, "quota_reserve": 0.15, "local_usd_per_hour": 0.0, "inline_access": "subscription", "private": False,
                   "log_decisions": True, "personalize": True}
SHORT_S = 6 * 3600                                      # a window this long or shorter is "short" (5 h)
LOG_FILE, LOG_MAX = "decisions.jsonl", 10 * 1024 * 1024
# key-like strings (evals/mine.py:_REDACT) as (literal prefix, pattern), so each scan starts with a fast substring search. A match
# must start where the original's \b would put it: the lookbehind after the literal checks the character before it. Quantifiers with
# nothing after them are bounded ({16} finds what {16,} finds), so every attempt is O(1) except eyJ's (see is_private).
_SECRET_RX = [(lit, "%s%s%s" % (re.escape(lit), "" if lit[0] == "-" else r"(?<!\w%s)" % re.escape(lit), rest)) for lit, rest in (
    ("-----BEGIN ", r"[A-Z0-9 ]*PRIVATE KEY-----"), ("sk-", r"(?:ant-[A-Za-z0-9_-]{8}|(?:proj-|live-|test-)?[A-Za-z0-9_-]{16})"),
    ("gh", r"[pousr]_[A-Za-z0-9]{20}"), ("github_pat_", r"[A-Za-z0-9_]{20}"), ("AKIA", r"[0-9A-Z]{16}\b"),
    ("xox", r"[bapr]-[A-Za-z0-9-]{10}"), ("eyJ", ""))]
_JWT = r"eyJ[\w-]{5,}\.[\w-]{5,}\.[\w-]{5}"
_ISO_RX = (r"(\d{4})-(\d\d)-(\d\d)[T ](\d\d):(\d\d):(\d\d)(\.\d+)?\s*(Z|[+-]\d\d:?\d\d)?$")   # compiled on first use
_P32 = 4294967311                                       # a prime above 2^32, for MinHash permutations
_MH = [((0x9E3779B1 * (2 * i + 1)) % _P32, (0x85EBCA77 * (i + 1)) % _P32) for i in range(16)]
_CAL_ID = (1.0, 0.0, None, 1.0, 0.0)
_CACHE, _Z, _TABLE = {}, {}, []


class _LazyNormal:
    """statistics.NormalDist(), imported on first use (the import costs about 2 ms on every `maestro-router route`, which needs it only
    for a seat without a precomputed θ)."""

    def __getattr__(self, name):
        import statistics
        nd = statistics.NormalDist()
        self.inv_cdf, self.cdf = nd.inv_cdf, nd.cdf
        return getattr(nd, name)


_ND = _LazyNormal()


def _median(xs):
    xs = sorted(xs)
    return (xs[(len(xs) - 1) // 2] + xs[len(xs) // 2]) / 2.0


def _sig(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _f(d, k, default=0.0):
    """d[k] as a float when it is a number (personal.json and anchors.json are read defensively), else the default."""
    v = d.get(k) if isinstance(d, dict) else None
    return float(v) if _num(v) else default


def _interp3(x, ys):
    """Linear in E[stakes] through ys at 0, 1 and 2."""
    x = min(2.0, max(0.0, x))
    i = min(1, int(x))
    return ys[i] + (ys[i + 1] - ys[i]) * (x - i)


def _stat(path):
    try:
        st = os.stat(path)
        return st.st_mtime_ns, st.st_size
    except OSError:
        return None


def _json(path):
    """A JSON file, re-read only when its mtime or size changes; None when missing or invalid."""
    key = _stat(path)
    if key and _CACHE.get(path, (None,))[0] != key:
        try:
            with open(path, encoding="utf-8") as f:
                _CACHE[path] = (key, json.load(f))
        except (OSError, ValueError):
            _CACHE[path] = (key, None)
    return _CACHE[path][1] if key else None


def table():
    """data/decide.json (spec Appendix B), read once per process."""
    if not _TABLE:
        _TABLE.append(_json(DATA_FILE) or {})
    return _TABLE[0]


def params(header=None):
    """Model-side constants by section (irt, link, calib, ordinal, decide): the s1.bin header's, else decide.json's placeholders."""
    d, h = table()["s1_defaults"], header if isinstance(header, dict) else {}
    if (h.get("link") or {}).get("fitted") is False:     # train_s1's placeholders, not a fit: decide.json's interim link and anchors
        h = dict(h, link={}, irt={k: v for k, v in (h.get("irt") or {}).items() if k != "anchor_models"})
    out = {k: dict(v, **{a: b for a, b in (h.get(k) or {}).items() if b is not None}) for k, v in d.items() if k[0] != "_"}
    out["link"]["sha"] = link_sha(out["link"])
    return out


def link_sha(link):
    """A short hash of the link (α_g, β, sd by basis): discover stamps it next to each precomputed models.json θ (`theta_link`); a θ
    stamped with another link is ignored."""
    return "%08x" % zlib.crc32(json.dumps([link.get("alpha"), link.get("beta"), link.get("sd_basis")], sort_keys=True).encode())


def is_private(text, cfg=None):
    """True when `private` is set or the text holds a key-like string: only local seats may see it (spec §4.5). Linear in the text:
    an eyJ start whose JWT attempt fails skips the rest of its [\\w-] run (a later start there has the same tail and a shorter head)."""
    if (cfg or {}).get("private"):
        return True
    for lit, p in _SECRET_RX:
        if not text or lit not in text:
            continue
        if lit != "eyJ":
            if re.search(p, text):
                return True
            continue
        find, jwt, run = re.compile(p), re.compile(_JWT), re.compile(r"[\w-]*")
        m = find.search(text)
        while m:
            if jwt.match(text, m.start()):
                return True
            m = find.search(text, run.match(text, m.start()).end())
    return False


# ---- seats ----

def seat_of(c, models=None):
    """A seat from a router candidate (router._candidates), a server candidate ({"m": normalized models.json entry, "latency" ms,
    "headroom", optional "rl_tokens"}) or a bare models.json entry. `models` (entries by id) adds catalog data to router seats."""
    if isinstance(c.get("m"), dict) or "access" in c:
        e = c["m"] if isinstance(c.get("m"), dict) else c
        mid, acc = str(e.get("id")), {"subscription-cli": "subscription"}.get(e.get("access"), e.get("access") or "api")
        s = {"id": mid, "kind": "cli" if acc == "subscription" else "proxy", "harness": "proxy", "access": acc, "entry": e,
             "quota": (e.get("qp") or mid[4:].split("/", 1)[0]) if acc == "subscription" else None, "model": e.get("model") or mid,
             "effort": c.get("effort"), "eff_ok": bool(e.get("reasoning")), "tools": e.get("tools_ok") is not False,
             "ttft": c["latency"] / 1000.0 if _num(c.get("latency")) and c["latency"] > 0 else None,
             "headroom": c.get("headroom"), "rl_tokens": c.get("rl_tokens"),
             "cat": e.get("cat_override") or e.get("serves") or e.get("category")}
    else:
        seat, kind, model = c["seat"], c["kind"], c.get("model") or ""
        pid = {"claude-code": "claude", "gemini-cli": "gemini"}.get(seat, seat)
        e = (models or {}).get(model) or (models or {}).get("cli:%s/%s" % (pid, model)) or {}
        acc = "local" if c.get("local") else "inline" if kind == "inline" else "api" if seat == "opencode" else "subscription"
        s = {"id": c.get("name") or "%s:%s" % (seat, model or "default"), "kind": kind, "harness": seat, "access": acc, "entry": e,
             "quota": QUOTA_OF.get(seat) if acc in ("subscription", "inline") else None, "model": model, "effort": c.get("effort"),
             "eff_ok": seat == "codex" or bool(e.get("reasoning")), "tools": bool(c.get("tools", True)), "params_b": c.get("params"),
             "cat": c.get("serves") or e.get("serves") or e.get("category")}
    s["mname"] = s["model"].rsplit("/", 1)[-1].lower()
    s["anchor"] = s["mname"] if s["mname"] in ANCHORS and s["quota"] == "claude" else None
    s["vendor"] = VENDOR_OF.get(s["quota"]) or e.get("provider") or (s["model"].split("/", 1)[0] if "/" in s["model"] else "")
    for k, keys in (("context", ("context",)), ("tps", ("tps", "est_decode_tps")), ("prefill_tps", ("prefill_tps",)),
                    ("params_b", ("params_b",)), ("reasoning", ("reasoning",))):
        s[k] = next((e[x] for x in keys if e.get(x)), s.get(k))
    return s


def inline_seat(harness, tier, model=None, models=None):
    """The calling agent itself (skill mode with --current-tier): Claude Code's anchor for the tier, else the catalog θ of the
    harness's model for that tier (router.seat_models(harness)[tier])."""
    name = ANCHORS[TIERS.index(tier)] if harness == "claude-code" else model
    return seat_of({"name": "inline", "seat": harness, "kind": "inline", "model": name, "serves": tier}, models)


def with_efforts(seats, efforts=("low", "medium", "high")):
    """Effort is part of the seat (spec §1.1): each effort-capable seat with no effort set, once per effort ("<id>@<effort>")."""
    out = []
    for s in seats:
        out += [dict(s, id="%s@%s" % (s["id"], e), effort=e) for e in efforts] if s["eff_ok"] and not s.get("effort") else [s]
    return out


# ---- success model (spec §3) ----

def _zs(jobs):
    """({job: Φ⁻¹(clip(strength, 0.02, 0.98))}, mean), cached per jobs dict (held, so its id is not reused)."""
    hit = _Z.get(id(jobs))
    if hit is None or hit[0] is not jobs:
        z = {j: _ND.inv_cdf(min(0.98, max(0.02, float(v)))) for j, v in jobs.items() if _num(v)}
        if len(_Z) > 4096:
            _Z.clear()
        hit = _Z[id(jobs)] = (jobs, z, sum(z.values()) / len(z) if z else 0.0)
    return hit[1], hit[2]


def theta(s, job_p, prm, tier=None, user=None):
    """(Σ_j job_p_j·θ_{m,j}, sd_link, from_link) of a seat's model (spec §3.2). In order: a user-run anchor (anchors.json); an anchor
    tier's learned θ plus β times its job profile; models.json "theta"; the catalog link α_g + β·Φ⁻¹(strength); decide.json
    seat_prior; else its category's anchor tier (local: one tier step below fast) with the prior sd. job_p must sum to 1."""
    irt, link, D, e = prm["irt"], prm["link"], table(), s.get("entry") or {}
    ag = D["groups"]["agentic_jobs"]

    def g(j):
        return "agentic" if j in ag else "chat"
    if isinstance(user, dict) and _num(user.get("theta_chat")):
        return sum(p * _f(user, "theta_" + g(j), user["theta_chat"]) for j, p in job_p.items()), _f(user, "sd", 0.25), False
    jobs, G = e.get("jobs"), e.get("G")
    zs, zbar = _zs(jobs) if isinstance(jobs, dict) and jobs else ({}, _ND.inv_cdf(min(0.98, max(0.02, G))) if _num(G) else None)
    beta, sdb = link["beta"], link["sd_basis"]
    if tier:
        return sum(p * (irt["theta"][g(j)][tier] + beta * (zs[j] - zbar if j in zs else 0.0)) for j, p in job_p.items()), sdb["anchor"], False
    th = e.get("theta") if e.get("theta_link") in (None, link.get("sha")) else None      # computed under another link: stale
    if isinstance(th, dict) and all(isinstance(th.get(j), (list, tuple)) and len(th[j]) == 2 for j in job_p):
        return sum(p * th[j][0] for j, p in job_p.items()), sum(p * th[j][1] for j, p in job_p.items()), True
    if zbar is not None:
        return sum(p * (link["alpha"][g(j)] + beta * zs.get(j, zbar)) for j, p in job_p.items()), sdb.get(e.get("basis"), sdb["prior"]), True
    prior = D["seat_prior"].get(s["id"])
    if prior:
        return sum(p * prior[g(j)] for j, p in job_p.items()), sdb["anchor"], False
    th = irt["theta"]
    if s.get("cat") in TIERS:
        return sum(p * th[g(j)][ANCHORS[TIERS.index(s["cat"])]] for j, p in job_p.items()), sdb["prior"], True
    return sum(p * (2 * th[g(j)]["haiku"] - th[g(j)]["sonnet"]) for j, p in job_p.items()), sdb["prior"], True


def _cal(z, cal):
    """Cal_g of a raw logit (spec §2.6): Platt (A, B), or isotonic knots [xs, ys] (xs: P_raw ascending, ys: P; checked by _iso);
    then the personal Platt."""
    a, b, iso, pa, pb = cal
    if iso:
        xs, ys = iso
        p = _sig(z)
        i = bisect.bisect_right(xs, p)
        p = ys[0] if i == 0 else ys[-1] if i == len(xs) else ys[i - 1] + (ys[i] - ys[i - 1]) * (p - xs[i - 1]) / (xs[i] - xs[i - 1])
        p = min(1 - 1e-9, max(1e-9, p))
        z = math.log(p / (1 - p))
    else:
        z = a * z + b
    return _sig(pa * z + pb)


def _iso(v):
    """The header's isotonic map as ([xs], [ys]) when well formed (two equal-length lists of >= 2 numbers in [0, 1], xs ascending),
    else None: Platt is used. train_s1.calibrate writes it this way."""
    ok = isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(a, (list, tuple)) and len(a) >= 2 for a in v)
    ok = ok and len(v[0]) == len(v[1]) and all(_num(x) and 0 <= x <= 1 for a in v for x in a)
    return (list(v[0]), list(v[1])) if ok and all(a <= b for a, b in zip(v[0], v[0][1:])) else None


def p_success(th, sd_s, mu_b, sd_b, a, z=None, cal=_CAL_ID):
    """P (z None: the seat's sd widens the spread with σ_b) or the pessimistic P̃ (z = z_r: the mean drops by z·sd_s and only σ_b
    widens it), spec §3.1, through the calibration map."""
    if z is None:
        return _cal(a * (th - mu_b) / math.sqrt(1 + math.pi / 8 * a * a * (sd_b * sd_b + sd_s * sd_s)), cal)
    return _cal(a * (th - z * sd_s - mu_b) / math.sqrt(1 + math.pi / 8 * a * a * sd_b * sd_b), cal)


def update_b(mu, sd, th, a):
    """(μ', σ') of the difficulty after a failure on a seat of ability θ̄ (spec §3.7; probabilists' Gauss-Hermite, 7 points)."""
    gh, num, den, num2 = table()["gh7"], 0.0, 0.0, 0.0
    for x, w in zip(gh["x"], gh["w"]):
        b = mu + sd * x
        lk = w * (1 - _sig(a * (th - b)))
        den, num, num2 = den + lk, num + lk * b, num2 + lk * b * b
    m = num / den
    return m, math.sqrt(max(0.0, num2 / den - m * m))


# ---- tokens, cost, quota, latency (spec §4) ----

def tokens(s, t):
    """(tin, c_hit, tout_total, extra_in, fits) of the whole task on a seat (spec §4.1): input tokens, the cached share, output
    tokens, input tokens billed again as a cache write (proxy, switching seats), and whether the context fits."""
    D = table()
    eff = D["effort"].get(s.get("effort"), D["effort"]["medium"])[1] if s["eff_ok"] else 1.0
    v = t["prm"]["irt"]["verbosity"].get(s["tier"], 0.0) if s["tier"] else _f(t["verb"], s["vendor"])
    tout = t["tok_ref"] * math.exp(max(-1.0, min(1.0, v))) * eff * (1 + ((s.get("r") or 1.0) if s.get("reasoning") else 0.0))
    n, ctx = t["turns"], s.get("context")
    if t["mode"] == "proxy":
        pre, warm = min(t["prefix"], t["est_in"]), s["id"] == t["warm"]
        hit = pre / t["est_in"] if warm and t["est_in"] else 0.0
        extra = 0.0 if warm else pre * (D["cache_write_ratio"].get(s["vendor"], 1.0) - 1)
        return t["est_in"], hit, tout, extra, not ctx or t["est_in"] + tout * math.exp(0.84 * t["tsd"]) <= 0.9 * ctx
    oh = D["overhead_tokens"].get("inline" if s["access"] == "inline" else "local" if s["kind"] == "local" else s["harness"], 0)
    base = t["est_in"] + (t["ctx_tok"] if s["access"] == "inline" else D["reread_tokens"] * (1 - t["sc"]))
    step = D["turn_step_tokens"]
    fits = not ctx or base + oh + step * (t["turns80"] - 1) <= 0.9 * ctx
    return n * (base + oh) + step * n * (n - 1) / 2, (n - 1) / n, n * tout, 0.0, fits


def cost(tin, hit, tout, price, extra_in=0.0):
    """API dollars (spec §4.2): uncached input at p_in, cached at p_cache_read, output at p_out ($/M tokens), plus cache writes."""
    pin, pout, pcr = price
    return (tin * (1 - hit) * pin + tin * hit * pcr + tout * pout + extra_in * pin) / 1e6


def _price(s, med):
    """(p_in, p_out, p_cache_read) of a seat's model: its catalog price, else the median API price of its category among the seats
    being ranked, else decide.json category_price."""
    D, e = table(), s.get("entry") or {}
    if s["access"] == "local":
        return 0.0, 0.0, 0.0
    pin, pout = e.get("cost_in"), e.get("cost_out")
    if not (_num(pin) and _num(pout) and (s["access"] == "api" or pin + pout > 0)):
        pin, pout = med.get(s["cat"]) or D["category_price"].get(s["cat"]) or D["category_price"]["standard"]
    ratio = D["cache_read_ratio"]
    return pin, pout, e["cost_cache_read"] if _num(e.get("cost_cache_read")) else pin * ratio.get(s["vendor"], ratio["default"])


def latency(s, t, tin, hit, tout):
    """Wall-clock seconds (spec §4.6): turns × (ttft + startup) + decode, plus prefill of the uncached input on a local seat."""
    D = table()
    pri = D["latency_prior"]
    ttft, tps = pri.get(s["cat"]) or pri["fast"]
    ttft, tps = s.get("ttft") or ttft, s.get("tps") or tps
    pre = 0.0
    if s["access"] == "local":
        pre = tin * (1 - hit) / (s.get("prefill_tps") or D["prefill_tps"]["small" if (s.get("params_b") or 0) <= 30 else "large"])
    return t["turns"] * (ttft + pri["startup_s"].get(s["kind"], 0)) + tout / tps + pre


def _slope(hist, now, since):
    """Least-squares burn rate (used per second) over the samples of the last hour since the window opened; None with fewer than 2
    samples or under 10 minutes of span."""
    lo = max(now - 3600, since)
    pts = [(x, u) for x, u in hist[-128:] if lo <= x <= now]
    if len(pts) < 2 or pts[-1][0] - pts[0][0] < 600:
        return None
    mx, mu = sum(x for x, _ in pts) / len(pts), sum(u for _, u in pts) / len(pts)
    return max(0.0, sum((x - mx) * (u - mu) for x, u in pts) / sum((x - mx) ** 2 for x, _ in pts))


def windows_of(prov, t):
    """The windows of a subscription provider as [(name, used, U0 = used + r̂·τ, reserve, margin, length, family)], or None when
    unknown (no data, Gemini, or a Claude snapshot older than stale_s). Cached per decision."""
    if prov in t["wcache"]:
        return t["wcache"][prov]
    pc, now, info, out = table()["pacing"], t["now"], (t["windows"] or {}).get(prov), None
    if info and info.get("windows") and not (prov == "claude" and now - (info.get("as_of") or 0) > pc["stale_s"]):
        out = []
        for w in info["windows"]:
            name, used, reset = w.get("name") or "", w.get("used"), w.get("reset_at") if _num(w.get("reset_at")) else None
            ln = w.get("length_s") or (18000 if name in ("five_hour", "primary") else 604800)
            if not _num(used):
                continue
            if reset is not None and reset <= now:
                used, tau = 0.0, ln                         # it has refilled
            else:
                tau = min(ln, max(0.0, reset - now)) if reset is not None else ln / 2.0
            r = used / max(600.0, ln - tau)
            hist = list(w.get("hist") or ())[-127:] + [(min(now, info.get("as_of") or now), used)]   # no new sample since: no burn
            rr = _slope(hist, now, now - (ln - tau))
            if rr is not None:
                r = 0.5 * rr + 0.5 * r if ln <= SHORT_S else 0.15 * rr + 0.85 * r
            res = t["reserve"] if name in ("five_hour", "seven_day", "primary") else pc["reserve_other"]
            out.append((name, used, used + r * tau, res, pc["margin_short" if ln <= SHORT_S else "margin_long"], ln,
                        name[10:] if name.startswith("seven_day_") else ""))
    t["wcache"][prov] = out
    return out


def quota_price(s, api_usd, c_sub, g_max, wins, t):
    """(Q, hard-skip reason or None, binding window (g, slack, name, used, length, family) or None) of a subscription seat (spec
    §4.3): g of the most pressed window × the substitute's API cost; g_unknown when the windows are unknown."""
    pc, best = table()["pacing"], None
    if wins is None:
        return pc["g_unknown"] * c_sub, None, None
    for name, used, u0, res, m, ln, fam in wins:
        if fam and fam != s["fam"]:
            continue                                        # seven_day_<family> binds only that family
        k0 = pc["k_prior"]["short" if ln <= SHORT_S else "long"]
        k = t["k"].get("%s/%s|%s" % (s["quota"], name, s["fam"]))
        du = (min(4 * k0, max(0.25 * k0, k)) if _num(k) else k0) * api_usd
        if used + du > 1 - res / 3:
            return 0.0, "would exhaust the %s %s window" % (s["quota"], name), None
        slack = 1 - res - u0 - du
        g = 0.0 if slack >= m else g_max * (m - slack) / m if slack >= 0 else g_max
        if best is None or (g, -slack) > (best[0], -best[1]):
            best = (g, slack, name, used, ln, fam)
    return (best[0] if best else 0.0) * c_sub, None, best


# ---- utility and the decision (spec §5) ----

def utility(V, p, C, Q, kappa, secs, F):
    """EU in dollars (spec §1.3)."""
    return V * p - C - Q - kappa * secs - (1 - p) * F


def _task(post, ctx):
    """Everything about the task that every seat shares."""
    D, cfg = table(), dict(CONFIG_DEFAULTS, **{k: v for k, v in (ctx.get("cfg") or {}).items() if v is not None})
    prm, home, mode = params(ctx.get("s1")), ctx.get("home"), ctx.get("mode") or "skill"
    pers = _json(os.path.join(home, "personal.json")) if home and cfg["personalize"] else None
    pers = pers if isinstance(pers, dict) else {}
    user = (_json(os.path.join(home, "anchors.json")) or {}) if home else {}
    group = post.get("group") if post.get("group") in ("chat", "agentic") else "chat"
    jp = {j: float(p) for j, p in (post.get("job_p") or {}).items() if _num(p) and p > 0} or {"ask": 1.0}
    jp = {j: p / sum(jp.values()) for j, p in jp.items()}
    mu, sd = post["b"]
    db = sum(p * max(-0.5, min(0.5, _f(pers.get("job_b_shift"), j))) for j, p in jp.items())
    mu += db
    tmu, tsd = post.get("tok_out") or (math.log(800), 0.6)
    nmu, nsd = post.get("turns") or (0.0, 0.0)
    multi = group == "agentic" and mode == "skill"
    tok_ref, turns = math.exp(tmu + tsd * tsd / 2), max(1.0, math.exp(nmu + nsd * nsd / 2)) if multi else 1.0
    stakes, risk = min(2.0, max(0.0, post.get("stakes", 1.0))), cfg["risk"]
    v0 = prm["decide"]["V0"]
    V = (cfg["wtp_usd"] or v0.get(cfg["objective"], v0["balanced"])) * min(8.0, max(0.5, math.sqrt(tok_ref / 1000 * max(1.0, turns))))
    fin = (prm["calib"]["final"] or {}).get(group) or {}
    pl, pp = fin.get("platt") or (1.0, 0.0), pers.get("platt") or (1.0, 0.0)
    an = {str(v).rsplit("/", 1)[-1].lower(): k for k, v in prm["irt"]["anchor_models"].items() if k in ANCHORS}
    sha = prm["link"].get("catalog_sha")
    text = ctx.get("text") or ""
    return {"cfg": cfg, "prm": prm, "mode": mode, "group": group, "jp": jp, "mu": mu, "db": db, "sd": sd, "a": prm["irt"]["a"][group],
            "tok_ref": tok_ref, "tsd": tsd, "turns": turns, "turns80": max(1.0, math.exp(nmu + 0.84 * nsd)) if multi else 1.0, "V": V,
            "d": post.get("detectable", 0.5), "sc": post.get("self_contained", 0.5), "tools": post.get("tools", 0.0),
            "zr": risk * 0.5 * stakes, "ell": risk * _interp3(stakes, D["stakes_loss"]), "p_min": _interp3(stakes, D["p_min"]),
            "kappa": cfg["background_latency_usd_per_s" if ctx.get("background") else "latency_usd_per_s"],
            "cal": (pl[0], pl[1], _iso(fin.get("iso")), pp[0], pp[1]), "anchor_names": an, "ood": post.get("ood", 0.0),
            "stale": bool(ctx.get("link_stale") or (sha and ctx.get("catalog_sha") and sha != ctx["catalog_sha"])),
            "est_in": ctx.get("est_in") or (len(text) / 3.6 if text else 500.0), "prefix": ctx.get("prefix_in") or 0,
            "warm": ctx.get("warm"), "ctx_tok": min(1e7, max(0.0, float(_f(ctx.get("hints") or post.get("hints"), "context_tokens")
            or D["inline_context_tokens"]))), "now": ctx.get("now") or time.time(), "reserve": cfg["quota_reserve"],
            "windows": ctx["windows"] if "windows" in ctx else read_windows(home) if home else {}, "wcache": {},
            "k": pers.get("k") if isinstance(pers.get("k"), dict) else {}, "verb": pers.get("verbosity"), "offsets": pers.get("seat_offsets"),
            "user": user.get("seats") if isinstance(user, dict) and isinstance(user.get("seats"), dict) else {},
            "private": bool(ctx.get("private")) or is_private(text, cfg)}


def _evaluate(seats, post, ctx):
    """(options best first, skipped [{seat, reason}], flags). Each option keeps its terms for escalate, confidence and p_best."""
    t, D = _task(post, ctx), table()
    skipped, flags, opts = [], [], []
    if t["private"]:
        keep = [s for s in seats if s["access"] == "local"] or ([s for s in seats if s["access"] == "inline"] if t["mode"] == "skill" else [])
        skipped += [{"seat": s["id"], "reason": "private: local only"} for s in seats if s not in keep]
        seats, flags = keep, flags + (["private: local only"] if keep else ["private_no_local"])
    for s in seats:
        s = dict(s, tier=s.get("anchor") or t["anchor_names"].get(s["mname"])
                 or t["anchor_names"].get(str((s.get("entry") or {}).get("catalog_id") or "").rsplit("/", 1)[-1].lower()))
        s["fam"] = s["tier"] or s["mname"]
        th, sd, linked = theta(s, t["jp"], t["prm"], s["tier"], t["user"].get(s["id"]) or t["user"].get((s.get("entry") or {}).get("id")))
        if s["eff_ok"]:
            th += D["effort"].get(s.get("effort"), D["effort"]["medium"])[0]
        hz = D["harness"]
        eta = (0.0, 0.0) if t["mode"] == "proxy" or s["access"] == "inline" else (0.0, hz["chat_group_sd"]) if t["group"] == "chat" \
            else hz["subagent" if s["kind"] == "subagent" else "codex-cli" if s["harness"] == "codex" else "cli"]
        th += eta[0] - (D["context_penalty"] * (1 - t["sc"]) if t["mode"] == "skill" and s["access"] != "inline" else 0.0)
        off = max(-1.0, min(1.0, _f(t["offsets"], "%s|%s" % ("anthropic/" + s["tier"] if s["tier"] else s["id"], t["group"]))))
        th += off
        sd = math.sqrt(sd * sd + eta[1] ** 2 + (0.04 if linked and t["stale"] else 0.0))
        tin, hit, tout, extra, fits = tokens(s, t)
        if not fits or (_num(s.get("rl_tokens")) and tin + tout > s["rl_tokens"]):
            skipped.append({"seat": s["id"], "reason": "context too small" if not fits else "rate limit: too few tokens left"})
            continue
        opts.append({"seat": s["id"], "s": s, "th": th, "sd": sd, "tin": tin, "hit": hit, "tout": tout, "extra": extra, "linked": linked,
                     "off": off})
    med = {}
    for cat in {o["s"]["cat"] for o in opts}:
        pr = [(e["cost_in"], e["cost_out"]) for e in (o["s"].get("entry") or {} for o in opts if o["s"]["access"] == "api" and o["s"]["cat"] == cat)
              if _num(e.get("cost_in")) and _num(e.get("cost_out"))]
        if pr:
            med[cat] = (_median([p[0] for p in pr]), _median([p[1] for p in pr]))
    for o in opts:
        o["price"] = _price(o["s"], med)
        o["api"] = cost(o["tin"], o["hit"], o["tout"], o["price"], o["extra"])
    apis, out, pc = [o for o in opts if o["s"]["access"] == "api"], [], D["pacing"]
    for o in opts:
        s, cfg = o["s"], t["cfg"]
        acc = ("api" if cfg["inline_access"] == "api" else "subscription") if s["access"] == "inline" else s["access"]
        secs = latency(s, t, o["tin"], o["hit"], o["tout"])
        C = o["api"] if acc == "api" else cfg["local_usd_per_hour"] * secs / 3600 if acc == "local" else 0.0
        Q, why, bind = 0.0, None, None
        if acc == "subscription":
            subs = [cost(o["tin"], o["hit"], o["tout"], x["price"], o["extra"]) for x in apis if x["th"] >= o["th"] - 0.3 and x is not o]
            wins = windows_of(s["quota"], t) if s["quota"] in ("claude", "codex") else None
            Q, why, bind = quota_price(s, o["api"], min(subs) if subs else o["api"], pc["g_max_sub"] if subs else pc["g_max"], wins, t)
            o["unknown"] = wins is None
        elif acc == "api" and _num(s.get("headroom")) and s["headroom"] < 0.1:
            Q = 0.2 * t["V"]                                # under 10% of an API rate-limit window left
        if why:
            skipped.append({"seat": s["id"], "reason": why})
            continue
        tf = 1 - t["tools"] if 0.2 <= t["tools"] < 0.5 and not s["tools"] else 1.0
        p = p_success(o["th"], o["sd"], t["mu"], t["sd"], t["a"], None, t["cal"])
        p_lo = p_success(o["th"], o["sd"], t["mu"], t["sd"], t["a"], t["zr"], t["cal"]) * tf
        F = t["d"] * (t["kappa"] * (secs + D["retry_secs"]) + C) + (1 - t["d"]) * t["ell"] * t["V"]
        o.update(t=t, p=p, p_lo=p_lo, tf=tf, usd=C, quota_usd=Q, secs=secs, F=F, bind=bind, floor_ok=p_lo >= t["p_min"],
                 eu=utility(t["V"], p_lo, C, Q, t["kappa"], secs, F))
        out.append(o)
    out.sort(key=lambda o: (not o["floor_ok"], -round(o["eu"] if o["floor_ok"] else o["p_lo"], 6), round(o["usd"] + o["quota_usd"], 9), o["seat"]))
    if out and not out[0]["floor_ok"]:
        flags.append("below_floor")
    if t["stale"] and any(o["linked"] for o in out):
        flags.append("link_stale")
    if any(o.get("unknown") and o["s"]["quota"] in ("claude", "codex") for o in out) or (out and out[0].get("unknown")):
        flags.append("quota_unknown")
    if t["ood"] > 0.5:
        flags.append("unfamiliar task: routed conservatively")
    return out, skipped, flags


def rank(seats, post, ctx):
    """The eligible options, best first (spec §1.3, §5.4): seats under the floor P̃ < P_min(stakes) come after all others, ordered
    by P̃. Each option: {"seat", "p", "p_lo", "usd", "quota_usd", "secs", "eu", "floor_ok", ...internal terms}."""
    return _evaluate(seats, post, ctx)[0]


def escalate(ranked):
    """Where to go when the chosen seat fails (spec §3.7): the best EU under the difficulty posterior after that failure, among
    seats at least as able, or the same seat again with feedback (θ + φ_fb) when failures are detectable."""
    if not ranked:
        return None
    o1, t = ranked[0], ranked[0]["t"]
    mu, sd = update_b(t["mu"], t["sd"], o1["th"], t["a"])
    fb, fsd = table()["retry_feedback"]
    pool = [(o, o["th"], o["sd"]) for o in ranked[1:] if o["th"] >= o1["th"]] + ([(o1, o1["th"] + fb, math.hypot(o1["sd"], fsd))] if t["d"] >= 0.5 else [])
    best = max(pool, key=lambda x: utility(t["V"], p_success(x[1], x[2], mu, sd, t["a"], t["zr"], t["cal"]) * x[0]["tf"], x[0]["usd"],
                                           x[0]["quota_usd"], t["kappa"], x[0]["secs"], x[0]["F"]), default=None)
    return best[0]["seat"] if best else None


def confidence(ranked):
    """Analytic probability that the top option beats the runner-up (spec §5.8); 1.0 with a single option."""
    if len(ranked) < 2:
        return 1.0
    t, o1, o2 = ranked[0]["t"], ranked[0], ranked[1]

    def v(o):
        s2 = t["a"] ** 2 * (t["sd"] ** 2 + o["sd"] ** 2)
        return o["p_lo"] * (1 - o["p_lo"]) * s2 / (1 + math.pi / 8 * s2)
    den, gap = (t["V"] + (o1["F"] + o2["F"]) / 2) * math.sqrt(v(o1) + v(o2)), o1["eu"] - o2["eu"]
    return 0.5 * (1.0 + math.erf(gap / den / math.sqrt(2.0))) if den > 0 else 1.0 if gap > 0 else 0.5


def p_best(ranked, decision_id, n=64):
    """{seat: share of n draws of (b, θ_s) in which it is the EU argmax}, seeded by crc32(decision_id) (spec §5.8)."""
    if not ranked:
        return {}
    import random
    t, rng = ranked[0]["t"], random.Random(zlib.crc32(str(decision_id).encode()))
    wins = dict.fromkeys((o["seat"] for o in ranked), 0)
    for _ in range(n):
        b = rng.gauss(t["mu"], t["sd"])
        best = max(ranked, key=lambda o: utility(t["V"], _cal(t["a"] * (rng.gauss(o["th"], o["sd"]) - b), t["cal"]) * o["tf"], o["usd"],
                                                 o["quota_usd"], t["kappa"], o["secs"], o["F"]))
        wins[best["seat"]] += 1
    return {k: v / float(n) for k, v in wins.items()}


def reasons(ranked, flags=(), difficulty=None):
    """'{job} d{difficulty} (b {μ:+.1f}±{σ:.1f}): {seat} P {p:.2f} vs {alt} {p_alt:.2f}; {cost}{; quota}{; flags}' (spec §5.9)."""
    if not ranked:
        return "no seat qualifies" + "".join("; " + f for f in flags)
    o, t = ranked[0], ranked[0]["t"]
    d = difficulty if difficulty is not None else sum(1 for tau in t["prm"]["ordinal"]["tau"] if t["mu"] >= tau)
    head = "%s d%d (b %+.1f±%.1f): %s P %.2f" % (max(t["jp"], key=t["jp"].get), d, t["mu"], t["sd"], o["seat"], o["p"])
    parts = [head + (" vs %s %.2f" % (ranked[1]["seat"], ranked[1]["p"]) if len(ranked) > 1 else " (only option)")]
    if len(ranked) > 1:
        alt = ranked[1]
        dc, dq, dt = o["usd"] - alt["usd"], o["quota_usd"] - alt["quota_usd"], o["secs"] - alt["secs"]
        money, pts = dc + dq, round((o["p"] - alt["p"]) * 100)
        usd = "%s$%.2f%s" % ("+" if money > 0 else "-", abs(money), " of quota" if abs(dq) > abs(dc) else "") if abs(money) >= 0.005 else ""
        if usd and max(abs(dc), abs(dq)) >= t["kappa"] * abs(dt):
            parts.append("%s for %+d pts success" % (usd, pts))
        else:
            parts.append("%s, %d s %s" % (usd or "same price", abs(dt), "faster" if dt < 0 else "slower") if abs(dt) >= 1 else
                         "%s, same speed, %+d pts success" % (usd or "same price", pts))
    if o.get("bind"):
        g, _, name, used, ln, fam = o["bind"]
        label = "%s %s%s" % (o["s"]["quota"].capitalize(), "5h" if ln <= SHORT_S else "weekly", " " + fam if fam else "")
        parts.append("%s window has slack (quota ×0)" % label if g == 0 else "%s %d%% used: priced at API ×%.1f" % (label, used * 100, g))
    return "; ".join(parts + list(flags))


def public(o):
    """The JSON form of an option (spec §1.2)."""
    return {"seat": o["seat"], "p": round(o["p"], 3), "p_lo": round(o["p_lo"], 3), "usd": round(o["usd"], 4),
            "quota_usd": round(o["quota_usd"], 4), "secs": int(round(o["secs"])), "eu": round(o["eu"], 4)}


def decide(seats, post, ctx):
    """The decision keys of spec §1.2: {"decider", "decision_id", "chosen" (seat id or None), "p_success", "options", "skipped",
    "escalate_to", "confidence", "flags", "reason", "hints", "trace"}; "trace" holds what decisions.jsonl and learn.py need (with
    "o" and "db", the personal offset and b shift applied, which learn.py subtracts). ctx "chosen" (shadow mode: the seat the ladder
    picked) is reported as the choice, with its P, trace and escalation, and the EU argmax as "shadow"."""
    opts, skipped, flags = _evaluate(seats, post, ctx)
    did = ctx.get("decision_id") or os.urandom(4).hex()
    pick = ctx.get("chosen")
    lead = [o for o in opts if o["seat"] == pick][:1] + [o for o in opts if o["seat"] != pick] if pick else opts
    out = {"decider": "utility", "decision_id": did, "chosen": pick or (opts[0]["seat"] if opts else None), "options": [public(o) for o in opts],
           "skipped": skipped, "flags": flags, "reason": reasons(opts, flags, ctx.get("difficulty")),
           "hints": ctx.get("hints") or post.get("hints") or {},
           "escalate_to": escalate(lead), "confidence": round(confidence(opts), 3) if opts else 0.0}
    if pick:
        out["shadow"] = opts[0]["seat"] if opts else None
    if lead and lead[0]["seat"] == out["chosen"]:
        o, t = lead[0], lead[0]["t"]
        out["p_success"] = round(o["p"], 3)
        out["trace"] = {"mu_b": round(t["mu"], 4), "sd_b": round(t["sd"], 4), "theta": round(o["th"], 4), "sd_s": round(o["sd"], 4),
                        "a": t["a"], "group": t["group"], "api_usd": round(o["api"], 5), "quota": o["s"]["quota"], "fam": o["s"]["fam"],
                        "o": round(o["off"], 4), "db": round(t["db"], 4), "V": round(t["V"], 4), "eu": round(o["eu"], 4)}
    return out


# ---- quota snapshots and the decision log (plain files; spec §4.3, §10.1) ----

def _epoch(v):
    """Epoch seconds from a number (ms when huge) or an RFC 3339 string; None if unparseable."""
    if _num(v):
        return v / 1000.0 if v > 1e11 else float(v)
    m = re.match(_ISO_RX, str(v or "").strip())
    if not m:
        return None
    y, mo, d, hh, mi, ss = (int(x) for x in m.groups()[:6])
    y -= mo <= 2
    era, yoe = y // 400, y % 400
    days = era * 146097 + yoe * 365 + yoe // 4 - yoe // 100 + (153 * (mo + (-3 if mo > 2 else 9)) + 2) // 5 + d - 1 - 719468
    off = m.group(8) or "Z"
    off = 0 if off == "Z" else (1 if off[0] == "+" else -1) * (int(off[1:3]) * 3600 + int(off[-2:]) * 60)
    return days * 86400 + hh * 3600 + mi * 60 + ss + float(m.group(7) or 0) - off


def _tail(path, n=65536):
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - n))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and _num(row.get("t")):
            out.append(row)
    return out


def read_windows(home):
    """Subscription windows from plain files, mtime-cached (spec §4.3): claude_limits.json (the statusline cache) with the history in
    the last 64 KB of claude_limits.log, and quota_hist.json (Codex samples and latest values, written by quota.refresh_subscriptions).
    {provider: {"as_of": ts, "windows": [{"name", "used" 0..1, "reset_at", "length_s", "hist": [[t, used], ...]}]}}"""
    paths = [os.path.join(home, f) for f in ("claude_limits.json", "claude_limits.log", "quota_hist.json")]
    sig = [_stat(p) for p in paths]
    hit = _CACHE.get(("windows", home))
    if hit and hit[0] == sig:
        return hit[1]
    out, cl, qh = {}, _json(paths[0]), _json(paths[2])
    rl = cl.get("rate_limits") if isinstance(cl, dict) else None
    if isinstance(rl, dict):
        hist = {}
        for row in _tail(paths[1]):
            for k, v in row.items():
                if k != "t" and _num(v):
                    hist.setdefault(k, []).append([float(row["t"]), v / 100.0])
        wins = [{"name": k, "used": w["used_percentage"] / 100.0, "reset_at": _epoch(w.get("resets_at")), "hist": hist.get(k, []),
                 "length_s": 18000 if k == "five_hour" else 604800} for k, w in sorted(rl.items())
                if (k in ("five_hour", "seven_day") or k.startswith("seven_day_")) and isinstance(w, dict) and _num(w.get("used_percentage"))]
        out["claude"] = {"as_of": _epoch(cl.get("updated_at")) or (os.stat(paths[0]).st_mtime if sig[0] else None), "windows": wins}
    if isinstance(qh, dict) and isinstance(qh.get("current"), dict):
        for key, cur in sorted(qh["current"].items()):
            if "/" in key and isinstance(cur, dict) and _num(cur.get("used")):
                prov, name = key.split("/", 1)
                hist = [h for h in ((qh.get("windows") or {}).get(key) or []) if isinstance(h, list) and len(h) == 2 and _num(h[0]) and _num(h[1])]
                info = out.setdefault(prov, {"as_of": None, "windows": []})
                info["windows"].append({"name": name, "used": cur["used"], "reset_at": cur.get("reset_at"), "length_s": cur.get("length_s"), "hist": hist})
                info["as_of"] = max(info["as_of"] or 0, cur.get("t") or 0) or None
    _CACHE[("windows", home)] = (sig, out)
    return out


def learn_k(samples, tasks, prior, k=None):
    """k_{w,family} (spec §4.4): EWMA, half-life 20 intervals, of Δused / Σ predicted API$ over consecutive window samples [(t, used)]
    that bracket at least one routed task [(t, api_usd)]; clipped to [0.25, 4] × prior. Intervals with a reset are skipped."""
    k, a = prior if k is None else k, 1 - 0.5 ** (1 / 20.0)
    for (t0, u0), (t1, u1) in zip(samples, samples[1:]):
        usd = sum(c for x, c in tasks if t0 <= x < t1)
        if usd > 0 and u1 >= u0:
            k = min(4 * prior, max(0.25 * prior, (1 - a) * k + a * (u1 - u0) / usd))
    return k


def minhash(text, k=16):
    """k uint32 MinHash values over word 3-shingles of the text (spec §10.1); not invertible."""
    w = [m.group() for m, _ in zip(re.finditer(r"\w+", (text or "")[:65536].lower()), range(400))]   # bounded work
    sh = [zlib.crc32(" ".join(w[i:i + 3]).encode("utf-8")) for i in range(max(1, len(w) - 2))]
    return [min((a * h + b) % _P32 & 0xFFFFFFFF for h in sh) for a, b in _MH[:k]]


def jaccard(m1, m2):
    """Estimated Jaccard similarity of two MinHash signatures."""
    return sum(1 for x, y in zip(m1, m2) if x == y) / float(max(1, min(len(m1), len(m2))))


def log_decision(home, decision, post, text="", cfg=None, **extra):
    """Append one line to ~/.maestro/decisions.jsonl (spec §10.1; mode 0600, rotated at 10 MB): heads, options and a MinHash of the
    text, never the text. `extra` (harness, hints, shadow, policy, ...) is added as is. Off with log_decisions: false; errors ignored."""
    if (cfg or {}).get("log_decisions") is False or not decision.get("decision_id"):
        return False
    jp = sorted((post.get("job_p") or {}).items(), key=lambda x: -x[1])[:3]
    row = dict({"id": decision["decision_id"], "ts": round(time.time(), 1), "job_p": {j: round(p, 3) for j, p in jp},
                "b": [round(x, 3) for x in post.get("b") or ()], "group": post.get("group"), "chosen": decision.get("chosen"),
                "p": decision.get("p_success"), "options": [[o["seat"], o["p"], o["eu"]] for o in decision.get("options", [])[:6]],
                "decider": decision.get("decider"), "escalate_to": decision.get("escalate_to"), "trace": decision.get("trace"),
                "minhash": minhash(text)}, **extra)
    path = os.path.join(home, LOG_FILE)
    try:
        os.makedirs(home, mode=0o700, exist_ok=True)   # owner-only; `router.route` tightens an existing home (decide never imports the router)
        if os.path.exists(path) and os.path.getsize(path) > LOG_MAX:
            os.replace(path, path + ".1")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8"))
        finally:
            os.close(fd)
        return True
    except (OSError, TypeError, ValueError):
        return False
