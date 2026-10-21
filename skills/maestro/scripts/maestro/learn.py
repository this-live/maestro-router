"""Personalization (spec §10): outcome signals, the replayed fit of ~/.maestro/personal.json and its guards, the `outcome`,
`feedback` and `learn` commands and the SubagentStop hook. Skill mode: decisions.jsonl (decide.log_decision) and outcomes.jsonl;
proxy: maestro.db `decisions` (§10.1 columns; `s1` = {"job_p", "b", "group", "trace"}) and `outcomes`. Counts only, never text.
The fit replays 90 days (never incremental): weighted logistic MAP by Newton over seat offsets, per-job b shifts and a Platt.
"""
from __future__ import annotations

import importlib
import json
import math
import os
import random
import re
import sqlite3
import sys
import time

from . import router as R

SIGNALS = {"explicit": 1.0, "tests": 0.6, "escalate": 0.7, "reask": 0.0, "thanks": 0.0, "override": 0.0}   # v1 weights
IMPLICIT = ("reask", "thanks", "override")             # weight 0 until `learn --audit` measures precision >= 0.8
VERDICT = {"pass": 1.0, "good": 1.0, "partial": 0.5, "fail": 0.0, "bad": 0.0}
ANCHORS, LEVELS = ("haiku", "sonnet", "opus", "fable"), {"stakes": ("low", "normal", "high"), "size": ("s", "m", "l", "xl")}
WINDOW_S, HOOK_S, ESCALATE_S, MIN_W, SMALL, MAX_OFFSETS = 90 * 86400, 1800, 900, 30.0, 0.2, 27
SD, CAP, A0 = {"o": 0.5, "db": 0.4, "p": 0.2}, {"o": 1.0, "db": 0.5}, {"chat": 1.2, "agentic": 1.5}
PERSONAL, OUTCOMES, DECISIONS, AUDIT, DB = "personal.json", "outcomes.jsonl", "decisions.jsonl", "audit.json", "maestro.db"
COLUMNS = (("s1", "text"), ("p_success", "real"), ("eu", "real"), ("decider", "text"), ("shadow", "text"), ("explore", "integer"),
           ("propensity", "real"), ("policy", "text"))
_REASK = re.compile(r"^\s*(no\b|nope|wrong|that'?s not|still (fails|failing|broken)|doesn'?t work|didn'?t work|try again)", re.I)
_THANKS = re.compile(r"^\s*(thanks|thank you|thx|works|perfect|great,? (it|that) works)", re.I)


def _sig(x):
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, x))))


def _jsonl(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip().startswith("{")]
    except (OSError, ValueError):
        return []
    return [r for r in rows if isinstance(r, dict)]


# ---- outcome signals (spec §10.2) ----

def migrate(db):
    """The §10.1 `decisions` columns (added when missing) and the `outcomes` table, on the proxy's maestro.db."""
    have = {r[1] for r in db.execute("pragma table_info(decisions)")}
    for col, typ in COLUMNS:
        if have and col not in have:
            db.execute("alter table decisions add column %s %s" % (col, typ))
    db.execute("create table if not exists outcomes (id integer primary key autoincrement, decision_id integer, ts real, "
               "signal text, value real, weight real, detail text)")
    db.execute("create index if not exists outcomes_decision on outcomes (decision_id)")


def record(home, decision_id, signal, value, detail=None, db=None, ts=None):
    """Store one outcome: outcomes.jsonl (skill mode) or the `outcomes` table (db given). `detail` keeps numbers only."""
    det = {k: v for k, v in (detail or {}).items() if isinstance(v, (int, float)) or (isinstance(v, list) and all(isinstance(x, int) for x in v))}
    row = {"decision_id": decision_id, "ts": round(ts or time.time(), 1), "signal": signal, "value": max(0.0, min(1.0, float(value))),
           "weight": SIGNALS[signal], "detail": det}
    if db is not None:
        db.execute("insert into outcomes (decision_id, ts, signal, value, weight, detail) values (?,?,?,?,?,?)",
                   (int(decision_id), row["ts"], signal, row["value"], row["weight"], json.dumps(det)))
    else:
        R.ensure_home(home)
        with os.fdopen(os.open(os.path.join(home, OUTCOMES), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    return row


def parse_tests(text):
    """(passed, total) from the last test-runner summary in text (pytest, unittest, jest, cargo; else an exit code), or None.
    Counts are 1-9 digits not inside a longer number, so every pattern is linear (a 20k-digit tool result costs ~1 ms)."""
    ran, code = re.findall(r"Ran (\d{1,9}) tests?", text or ""), None
    for line in reversed((text or "").splitlines()):
        k = line.find("Tests:")
        tot = re.compile(r"(?<!\d)(\d{1,9}) total").search(line, k + 7) if k >= 0 and line[k + 6:k + 7].isspace() else None
        p = re.findall(r"(?<!\d)(\d{1,9}) passed", line)
        f = sum(int(x) for x in re.findall(r"(?<!\d)(\d{1,9}) (?:failed|errors?)\b", line))
        if p or f or tot:
            total = int(tot.group(1)) if tot else (int(p[-1]) if p else 0) + f
            return (int(p[-1]) if p else total - f), max(total, 1)
        m = re.search(r"FAILED \(([^)]*)\)", line)
        if ran and (m or re.match(r"\s*OK\b", line)):
            bad = sum(int(x) for x in re.findall(r"(?:failures|errors)=(\d{1,9})\b", m.group(1))) if m else 0
            return int(ran[-1]) - bad, max(int(ran[-1]), 1)
        m = re.search(r"exit(?:ed with)? code:? (\d{1,9})\b", line, re.I)
        code = int(m.group(1)) if m and code is None else code
    return (int(code == 0), 1) if code is not None else None


def from_hook(raw, home, now=None):
    """`maestro-router outcome --from-hook`: the SubagentStop hook JSON on stdin -> a `tests` outcome for the latest Claude Code subagent
    decision of the last 30 min, from the last 200 transcript lines (counts only). None when nothing applies; never raises."""
    try:
        path = json.loads(raw or "{}").get("transcript_path")
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - (1 << 20)))
            tail = f.read().decode("utf-8", "replace").splitlines()[-200:]
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    got, now = parse_tests("\n".join(x.replace("\\n", "\n") for x in tail)), now or time.time()
    decs = [d for d in decisions(home) if now - HOOK_S <= d.get("ts", 0) <= now and (d.get("kind") == "subagent" or (
        d.get("harness", "claude-code") == "claude-code" and str(d.get("chosen") or "").startswith("claude-code:")))]
    if not got or not decs:
        return None
    return record(home, decs[-1]["id"], "tests", got[0] / float(got[1]), {"tests": list(got)})


def feedback_value(v):
    """1, 0.5 or 0 from good / partial / bad (or pass / fail, or a number in [0, 1]); None otherwise."""
    if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1:
        return float(v)
    return VERDICT.get(str(v).strip().lower())


def api_feedback(db, body):
    """POST /api/feedback {id, value} and the X-Maestro-Feedback header (`<id>=good`): (status, JSON body)."""
    did, val = (body or {}).get("id"), feedback_value((body or {}).get("value"))
    if not isinstance(did, int) or isinstance(did, bool) or not 0 < did < 2 ** 63 or val is None:   # sqlite ids are int64
        return 400, {"error": {"message": "send {id: <decision id>, value: good|bad|partial}", "type": "invalid_request_error"}}
    if not db.execute("select 1 from decisions where id = ?", (did,)).fetchone():
        return 404, {"error": {"message": "no decision %d" % did, "type": "not_found"}}
    return 200, {"ok": True, "outcome": record(None, did, "explicit", val, db=db)}


def parse_feedback_header(h):
    """[(decision id, value)] from `X-Maestro-Feedback: 12=good, 13=bad`."""
    pairs = [(k.strip(), v) for k, v in (p.partition("=")[::2] for p in (h or "").split(","))]
    return [(int(k), feedback_value(v)) for k, v in pairs if re.match(r"[0-9]{1,18}$", k) and feedback_value(v) is not None]


def implicit(prev, cur):
    """Proxy signals from the next request of a conversation: [(signal, value)]. prev: {ts, sig (MinHash of the user message), rank
    (category rank of the routed model)}; cur: {ts, text, sig, rank (of a requested category or model, else None), model (a concrete
    requested model id, else None)}. escalate counts in v1; reask, thanks and override are logged with weight 0."""
    text, r0, r1 = cur.get("text") or "", prev.get("rank"), cur.get("rank")
    out = [("escalate", 0.0)] if None not in (r0, r1) and r1 > r0 else [("override", 1.0)] if None not in (r0, r1) and r1 < r0 and cur.get("model") else []
    if cur.get("ts", 0) - prev.get("ts", 0) <= 600 and (_REASK.match(text) or jaccard(prev.get("sig"), cur.get("sig")) >= 0.6):
        return out + [("reask", 0.0)]
    return out + ([("thanks", 1.0)] if _THANKS.match(text) else [])


def jaccard(a, b):
    return sum(1 for x, y in zip(a, b) if x == y) / float(min(len(a), len(b))) if a and b else 0.0


def explore(ranked, post, V, eps, rng=random):
    """(index, propensity) under ε-greedy (spec §10.4; config `explore`, default 0): the runner-up with probability ε, only when the
    EU gap is < 0.05·V, E[stakes] < 0.5 and detectable >= 0.5. The propensity is exact."""
    eps = max(0.0, min(0.1, float(eps or 0)))
    if eps <= 0 or len(ranked) < 2 or ranked[0]["eu"] - ranked[1]["eu"] >= 0.05 * V or post.get("stakes", 1) >= 0.5 \
            or post.get("detectable", 0) < 0.5:
        return 0, 1.0
    return (1, eps) if rng.random() < eps else (0, 1.0 - eps)


# ---- the replay (spec §10.3) ----

def decisions(home):
    """Skill-mode decisions (decisions.jsonl and its rotated .1), oldest first."""
    rows = _jsonl(os.path.join(home, DECISIONS + ".1")) + _jsonl(os.path.join(home, DECISIONS))
    return sorted((r for r in rows if r.get("id") and isinstance(r.get("ts"), (int, float))), key=lambda r: r["ts"])


def _db_rows(db):
    """(decisions, outcomes) of the proxy log in the skill-mode shapes; ids are prefixed "p" (proxy ids are integers)."""
    if db is None:
        return [], []
    try:
        dec = db.execute("select id, ts, chosen, s1, policy, prompt from decisions where s1 is not null").fetchall()
        out = db.execute("select decision_id, ts, signal, value, weight from outcomes").fetchall()
    except sqlite3.DatabaseError:
        return [], []
    rows = [dict(json.loads(s1), id="p%d" % i, ts=ts, chosen=chosen, policy=policy, prompt=prompt)
            for i, ts, chosen, s1, policy, prompt in dec if (s1 or "").startswith("{")]
    return rows, [{"decision_id": "p%d" % d, "ts": ts, "signal": s, "value": v, "weight": w} for d, ts, s, v, w in out]


def _escalations(decs):
    """Skill mode: a re-route of the same task (MinHash Jaccard >= 0.6, <= 15 min) to a stronger seat, or with a stronger hint,
    is an `escalate` outcome (value 0) for the earlier decision."""
    out = []
    for i, d1 in enumerate(decs):
        for d2 in decs[i + 1:]:
            if d2["ts"] - d1["ts"] > ESCALATE_S:
                break
            if jaccard(d1.get("minhash"), d2.get("minhash")) < 0.6:
                continue
            t1, t2, p1 = (d1.get("trace") or {}).get("theta"), (d2.get("trace") or {}).get("theta"), dict((o[0], o[1]) for o in d1.get("options") or [])
            up = (t1 is not None and t2 is not None and t2 > t1) or (d2.get("chosen") in p1 and p1[d2["chosen"]] > (d1.get("p") or 1))
            h1, h2 = d1.get("hints") or {}, d2.get("hints") or {}
            up = up or any(h2.get(k) in v and v.index(h2[k]) > (v.index(h1[k]) if h1.get(k) in v else 1) for k, v in LEVELS.items())
            if up:
                out.append({"decision_id": d1["id"], "ts": d2["ts"], "signal": "escalate", "value": 0.0, "weight": SIGNALS["escalate"]})
                break
    return out


def samples(home, db=None, now=None, aud=None):
    """(decisions, fit rows): one row per (decision, signal) with a positive weight; the latest outcome of each pair wins."""
    now = now or time.time()
    decs = [d for d in decisions(home) if d["ts"] >= now - WINDOW_S]
    pdec, pout = _db_rows(db)
    decs += [d for d in pdec if d["ts"] >= now - WINDOW_S]
    outs = _jsonl(os.path.join(home, OUTCOMES)) + pout + _escalations([d for d in decs if not d["id"].startswith("p")])
    aud = (aud if aud is not None else R._read_json(os.path.join(home, AUDIT)) or {}).get("signals") or {}
    latest = {}
    for o in sorted(outs, key=lambda o: o.get("ts") or 0):
        a = aud.get(o.get("signal")) or {}
        w = (a.get("precision", 0) if a.get("n", 0) >= 20 and a.get("precision", 0) >= 0.8 else 0.0) if o.get("signal") in IMPLICIT \
            else SIGNALS.get(o.get("signal"), 0.0)
        if w > 0:
            latest[(str(o["decision_id"]), o["signal"])] = (float(o["value"]), w)
    by_id, rows = {d["id"]: d for d in decs}, []
    for (did, _), (y, w) in sorted(latest.items()):
        d = by_id.get(did)
        tr = (d or {}).get("trace") or {}
        if not d or not all(isinstance(tr.get(k), (int, float)) for k in ("mu_b", "sd_b", "theta", "sd_s")):
            continue
        g = tr.get("group") or d.get("group") or "chat"
        jp = {j: float(p) for j, p in (d.get("job_p") or {}).items() if j in R.JOBS and isinstance(p, (int, float)) and p > 0}
        fam = tr.get("fam")
        rows.append({"ts": d["ts"], "y": y, "w": w, "g": g, "u": 1.0, "a": float(tr.get("a") or A0.get(g, 1.2)), "policy": d.get("policy"),
                     "mu": tr["mu_b"] - tr.get("db", 0.0), "sd": tr["sd_b"], "th": tr["theta"] - tr.get("o", 0.0), "sds": tr["sd_s"],
                     "jp": {j: p / sum(jp.values()) for j, p in jp.items()}, "top": max(jp, key=jp.get) if jp else "ask",
                     "key": "%s|%s" % ("anthropic/" + fam if fam in ANCHORS else d.get("chosen"), g)})
    return decs, sorted(rows, key=lambda r: r["ts"])


def _resets(rows, weights_sha=None):
    """A row informs its seat offset only under the current weights (policy) and while its seat's θ̄ (same top job) has not moved
    by more than 0.3 since: a change resets that seat's offset."""
    cur = weights_sha or (rows[-1]["policy"] if rows else None)
    ref = {}
    for r in reversed(rows):
        ref.setdefault((r["key"], r["top"]), r["th"])
        r["u"] = 1.0 if (cur is None or r["policy"] == cur) and abs(r["th"] - ref[(r["key"], r["top"])]) <= 0.3 else 0.0
    return rows


def _eta(r, P, cal):
    """(η = personal-Platt logit, h = global-calibrated logit, r = a/s) of one row under parameters P."""
    a = r["a"]
    k = a / math.sqrt(1 + math.pi / 8 * a * a * (r["sd"] ** 2 + r["sds"] ** 2))
    z = k * (r["th"] + r.get("u", 1.0) * P["o"].get(r["key"], 0.0) - r["mu"] - sum(q * P["db"].get(j, 0.0) for j, q in r["jp"].items()))
    ag, bg = cal.get(r["g"]) or (1.0, 0.0)
    h = ag * z + bg
    return P["A"] * h + P["B"], h, k * ag


def loss(rows, P, cal):
    """Weighted mean log loss (nats)."""
    tot = sum(r["w"] for r in rows) or 1.0
    return -sum(r["w"] * (r["y"] * math.log(max(1e-12, _sig(e))) + (1 - r["y"]) * math.log(max(1e-12, 1 - _sig(e))))
                for r, e in ((r, _eta(r, P, cal)[0]) for r in rows)) / tot


def _solve(H, g):
    """H x = g for a symmetric positive definite H (Gaussian elimination with partial pivoting)."""
    n, M = len(g), [row[:] + [g[i]] for i, row in enumerate(H)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - sum(M[r][c] * x[c] for c in range(r + 1, n))) / M[r][r]
    return x


def fit(rows, cal=None, iters=25):
    """MAP of {o: {key: offset}, db: {job: Δb}, A, B} with priors N(0, .5²), N(0, .4²) and N(shipped (1, 0), .2²), then the caps
    (|o| <= 1, |Δb| <= 0.5) and the small-data guard (no move over 0.2 before 30 weighted outcomes)."""
    cal, keys = cal or {}, {}
    for r in rows:
        keys[r["key"]] = keys.get(r["key"], 0.0) + r["w"] * r.get("u", 1.0)
    keys = sorted((k for k, v in keys.items() if v > 0), key=lambda k: -keys[k])[:MAX_OFFSETS]
    names = [("o", k) for k in keys] + [("db", j) for j in R.JOBS] + [("A", None), ("B", None)]
    prior = [(0.0, SD["o"]) for _ in keys] + [(0.0, SD["db"]) for _ in R.JOBS] + [(1.0, SD["p"]), (0.0, SD["p"])]
    x, idx = [m for m, _ in prior], {nm: i for i, nm in enumerate(names)}

    def unpack(v):
        return {"o": {k: v[idx[("o", k)]] for k in keys}, "db": {j: v[idx[("db", j)]] for j in R.JOBS}, "A": v[-2], "B": v[-1]}

    def obj(v):
        return -loss(rows, unpack(v), cal) * sum(r["w"] for r in rows) - sum((a - m) ** 2 / (2 * s * s) for a, (m, s) in zip(v, prior))
    for _ in range(iters):
        P, n = unpack(x), len(x)
        g = [-(a - m) / (s * s) for a, (m, s) in zip(x, prior)]
        H = [[(1 / (prior[i][1] ** 2) if i == j else 0.0) for j in range(n)] for i in range(n)]
        for r in rows:
            e, h, k = _eta(r, P, cal)
            p = _sig(e)
            J = {n - 2: h, n - 1: 1.0}
            if r["key"] in P["o"] and r.get("u", 1.0):
                J[idx[("o", r["key"])]] = P["A"] * k * r["u"]
            for j, q in r["jp"].items():
                J[idx[("db", j)]] = -P["A"] * k * q
            for i, di in J.items():
                g[i] += r["w"] * (r["y"] - p) * di
                for j2, dj in J.items():
                    H[i][j2] += r["w"] * p * (1 - p) * di * dj
        step, t, f0 = _solve(H, g), 1.0, obj(x)
        while t > 1e-3 and obj([a + t * s for a, s in zip(x, step)]) < f0 - 1e-12:
            t /= 2
        x = [a + t * s for a, s in zip(x, step)]
        if max(abs(t * s) for s in step) < 1e-6:
            break
    P, w = unpack(x), {"A": sum(r["w"] for r in rows)}
    for r in rows:
        w[("o", r["key"])] = w.get(("o", r["key"]), 0.0) + r["w"] * r.get("u", 1.0)
        for j, q in r["jp"].items():
            w[("db", j)] = w.get(("db", j), 0.0) + r["w"] * q
    clip = lambda v, m, cap, n: max(m - SMALL, min(m + SMALL, v)) if n < MIN_W else max(-cap, min(cap, v))  # noqa: E731
    P["o"] = {k: round(clip(v, 0.0, CAP["o"], w.get(("o", k), 0)), 4) for k, v in P["o"].items()}
    P["db"] = {j: round(clip(v, 0.0, CAP["db"], w.get(("db", j), 0)), 4) for j, v in P["db"].items() if abs(v) >= 1e-4}
    P["A"], P["B"] = round(clip(P["A"], 1.0, 1e9, w["A"]), 4), round(clip(P["B"], 0.0, 1e9, w["A"]), 4)
    return P


def replay_dll(A, B, cells=None):
    """Mean extra log loss of the personal Platt on alpha-dev cells [[logit, y, w]] (data/alpha_dev.json when shipped), else on a
    calibrated reference (logit ~ N(0, 1.5²), y ~ the global model): the replay guard rejects > 0.02 nats."""
    if cells is None:
        cells = R._read_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "alpha_dev.json"))
    if not isinstance(cells, list) or not cells:
        cells = [[z / 10.0, _sig(z / 10.0), math.exp(-(z / 10.0) ** 2 / 4.5)] for z in range(-60, 61)]
    tot = sum(c[2] for c in cells)
    ll = lambda a, b: -sum(w * (y * math.log(max(1e-12, _sig(a * z + b))) + (1 - y) * math.log(max(1e-12, 1 - _sig(a * z + b))))  # noqa: E731
                            for z, y, w in cells) / tot
    return ll(A, B) - ll(1.0, 0.0)


def _global_cal():
    """{group: (A_g, B_g)}: the final Platt of the s1.bin header (identity without one); isotonic maps are not replayed."""
    try:
        md = importlib.import_module("%s.s1" % (__package__ or "maestro")).load()[0]
        fin = md["header"]["calib"]["final"]
        return {g: tuple((fin.get(g) or {}).get("platt") or (1.0, 0.0)) for g in ("chat", "agentic")}
    except (ImportError, AttributeError, KeyError, TypeError, ValueError):
        return {}


def _k(home, decs):
    """{"<provider>/<window>|<family>": k} from the quota snapshots bracketing routed tasks (decide.learn_k, spec §4.4)."""
    try:
        D = importlib.import_module("%s.decide" % (__package__ or "maestro"))
        wins, pri = D.read_windows(home), D.table()["pacing"]["k_prior"]
    except (ImportError, AttributeError, KeyError, TypeError):
        return {}
    out = {}
    for prov, info in wins.items():
        for w in info.get("windows") or []:
            hist = sorted(tuple(h) for h in w.get("hist") or [])
            for fam in {(d.get("trace") or {}).get("fam") for d in decs if (d.get("trace") or {}).get("quota") == prov} - {None}:
                tasks = [(d["ts"], d["trace"]["api_usd"]) for d in decs if (d.get("trace") or {}).get("quota") == prov
                         and d["trace"].get("fam") == fam and isinstance(d["trace"].get("api_usd"), (int, float))]
                prior = pri["short" if (w.get("length_s") or 0) <= 6 * 3600 else "long"]
                k = D.learn_k(hist, tasks, prior)
                if len(hist) > 1 and tasks and k != prior:
                    out["%s/%s|%s" % (prov, w.get("name"), fam)] = round(k, 5)
    return out


def run(home, db=None, now=None, dry_run=False, weights_sha=None, cal=None, cfg=None):
    """Replay, fit, guard and (unless dry_run or a guard fails) write personal.json (0600; the last 3 kept as .1-.3). A report dict."""
    cfg = cfg if cfg is not None else R.load_config(home)
    if cfg.get("personalize") is False:
        return {"written": False, "why": "personalize is off"}
    decs, rows = samples(home, db, now)
    rows, cal = _resets(rows, weights_sha), cal if cal is not None else _global_cal()
    rep = {"written": False, "n_outcomes": len(rows), "weighted": round(sum(r["w"] for r in rows), 2)}
    cut = int(len(rows) * 0.8)
    if len(rows) < 10 or len(rows) - cut < 2:
        return dict(rep, why="not enough outcomes (%d; 10 needed)" % len(rows))
    zero = {"o": {}, "db": {}, "A": 1.0, "B": 0.0}
    held = rows[cut:]
    gl, pl = loss(held, zero, cal), loss(held, fit(rows[:cut], cal), cal)
    P = fit(rows, cal)
    dll = replay_dll(P["A"], P["B"])
    rep.update(guard={"global_ll": round(gl, 4), "personal_ll": round(pl, 4), "replay_dll": round(dll, 4)})
    if pl > gl:
        return dict(rep, why="time-split guard: the personal fit predicts the last 20% worse than the global model")
    if dll > 0.02:
        return dict(rep, why="replay guard: the personal Platt worsens alpha-dev log loss by %.3f nats" % dll)
    data = {"version": 1, "weights_sha": weights_sha or (rows[-1]["policy"] if rows else None), "fitted_at": round(now or time.time()),
            "n_outcomes": len(rows), "seat_offsets": P["o"], "job_b_shift": P["db"], "platt": [P["A"], P["B"]], "k": _k(home, decs),
            "guard": rep["guard"], "snapshots": 3}
    old = R._read_json(os.path.join(home, PERSONAL))
    data["verbosity"] = (old or {}).get("verbosity") or {}            # learned elsewhere (spec §4.1); kept as is
    if dry_run:
        return dict(rep, personal=data)
    path = os.path.join(home, PERSONAL)
    for src, dst in ((path + ".2", path + ".3"), (path + ".1", path + ".2"), (path, path + ".1")):
        if os.path.exists(src):
            os.replace(src, dst)
    R._write_json(path, data)   # created 0600
    return dict(rep, written=True, personal=data)


def audit(home, db=None, n=100, ask=input, out=print, seed=0):
    """`maestro-router learn --audit`: rate up to n decisions that carry an implicit signal; each signal's precision goes to audit.json."""
    pdec, pout = _db_rows(db)
    by_id = {d["id"]: d for d in decisions(home) + pdec}
    outs = [o for o in _jsonl(os.path.join(home, OUTCOMES)) + pout if o.get("signal") in IMPLICIT and str(o.get("decision_id")) in by_id]
    data = R._read_json(os.path.join(home, AUDIT)) or {"version": 1, "signals": {}}
    for o in random.Random(seed).sample(outs, min(n, len(outs))):
        d = by_id[str(o["decision_id"])]
        out("%s  %s -> %s%s; the %s signal says it %s" % (time.strftime("%Y-%m-%d %H:%M", time.localtime(d["ts"])),
            max(d.get("job_p") or {"?": 1}, key=(d.get("job_p") or {"?": 1}).get), d.get("chosen"),
            "  \"%s\"" % d["prompt"][:120] if d.get("prompt") else "", o["signal"], "worked" if o["value"] >= 0.5 else "failed"))
        a = (ask("Did the routed answer work? [y/n/s=skip/q=quit] ") or "s").strip().lower()[:1]
        if a == "q":
            break
        if a in ("y", "n"):
            s = data["signals"].setdefault(o["signal"], {"n": 0, "agree": 0})
            s["n"], s["agree"] = s["n"] + 1, s["agree"] + int((a == "y") == (o["value"] >= 0.5))
            s["precision"] = round(s["agree"] / float(s["n"]), 3)
    R._write_json(os.path.join(home, AUDIT), data)
    return data


# ---- commands (wired by commands.py) ----

def cmd_outcome(a):
    """`maestro-router outcome <id|last> pass|partial|fail [--tests P/T] [--from-hook]` (a.id, a.verdict, a.tests, a.from_hook)."""
    home = R.maestro_home()
    if getattr(a, "from_hook", False):
        from_hook(sys.stdin.read(), home)
        return 0                                         # a hook never fails the agent
    decs = decisions(home)
    d = decs[-1] if decs and a.id == "last" else next((x for x in decs if x["id"] == a.id), None)
    m = re.match(r"^(\d+)/(\d+)$", getattr(a, "tests", None) or "")
    if not d or not (a.verdict in VERDICT or m):
        print("outcome: %s" % ("no decision %r in %s" % (a.id, os.path.join(home, DECISIONS)) if not d else "say pass, partial or fail"),
              file=sys.stderr)
        return 1
    tests = {"tests": [int(m.group(1)), int(m.group(2))]} if m else None
    row = record(home, d["id"], "explicit", VERDICT[a.verdict], tests) if a.verdict in VERDICT else \
        record(home, d["id"], "tests", int(m.group(1)) / float(max(1, int(m.group(2)))), tests)
    print("recorded %s=%.1f for %s (%s)" % (row["signal"], row["value"], d["id"], d.get("chosen")))
    return 0


def cmd_feedback(a):
    """`maestro-router feedback <id> good|bad|partial` for a proxy decision (a.id, a.value)."""
    home, val = R.maestro_home(), feedback_value(a.value)
    path = os.path.join(home, DB)
    if not os.path.exists(path):   # no proxy decision exists yet; sqlite would create the file with the umask mode
        print("feedback: no decision %s in %s" % (a.id, path), file=sys.stderr)
        return 1
    db = sqlite3.connect(path, timeout=10, isolation_level=None)
    try:
        migrate(db)
        st, out = api_feedback(db, {"id": int(a.id) if re.match(r"[0-9]{1,18}$", str(a.id)) else None, "value": val})
    except sqlite3.DatabaseError as e:
        st, out = 500, {"error": {"message": "cannot use %s: %s" % (DB, e.__class__.__name__)}}
    finally:
        db.close()
    print("recorded" if st == 200 else "feedback: " + out["error"]["message"], file=sys.stdout if st == 200 else sys.stderr)
    return 0 if st == 200 else 1


def cmd_learn(a):
    """`maestro-router learn [--report|--audit|--reset|--dry-run]` (a.report, a.audit, a.reset, a.dry_run)."""
    home = R.maestro_home()
    if getattr(a, "reset", False):
        for p in (os.path.join(home, PERSONAL) + x for x in ("", ".1", ".2", ".3")):
            if os.path.exists(p):
                os.remove(p)
        print("personalization reset")
        return 0
    path = os.path.join(home, DB)
    db = sqlite3.connect(path, timeout=10, isolation_level=None) if os.path.exists(path) else None
    try:
        if getattr(a, "audit", False):
            out = audit(home, db)
        elif getattr(a, "report", False):
            out = {"personal": R._read_json(os.path.join(home, PERSONAL)), "outcomes": len(samples(home, db)[1]),
                   "audit": R._read_json(os.path.join(home, AUDIT))}
        else:
            out = run(home, db, dry_run=getattr(a, "dry_run", False))
    finally:
        if db is not None:
            db.close()
    print(json.dumps(out, indent=1))
    return 0
