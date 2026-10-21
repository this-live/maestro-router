"""The Maestro proxy: an OpenAI-compatible endpoint that routes every request, plus the admin API behind the web UI.

    maestro-router serve [--host 127.0.0.1] [--port 4545]

    GET  /v1/models              maestro/auto, maestro/<category>, maestro/<job> and every enabled <provider>/<model>
    POST /v1/chat/completions    classify -> category -> pick a model -> call it (streaming and tools included) -> fall back
    POST /v1/systemone           Jev's typed-decision wire format, answered by S1 and decide (systemone.py)
    GET  /health                 liveness, no auth
    GET  /                       the web UI (ui/index.html)
    /api/*                       admin JSON for the UI; needs the admin token (~/.maestro/admin_token) and a same-origin request

Routing (per request): candidates are the enabled models that can serve it (tools_ok when tools are present, vision, context, not on
cooldown or exhausted, `max_tier`); those whose per-job strength reaches the category's floor are ranked by expected cost (in log
buckets), then quota headroom, then observed latency, with `prefer_local` and the `objective` shifting the order. A tool turn keeps
the model chosen earlier in the conversation. With `decider: utility`, S1 and decide.py rank the same candidates by expected utility
in dollars (with a switching cost away from the conversation's warm seat) instead; with `s1_shadow` the other decider's pick is
logged next to each decision. On 429/5xx/timeout before the first byte the next candidate is tried (three at most);
once bytes have gone to the client an error is surfaced instead. Every request is one row in ~/.maestro/maestro.db (no prompt text
unless `log_prompts` is true). Keys are used only by `clients.complete` for their own provider and never appear in a response.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import socket
import sqlite3
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import clients as C
from . import providers as P
from . import quota as Q
from . import router as core
from . import score as S

DEFAULT_HOST, DEFAULT_PORT = "127.0.0.1", 4545
FLOORS = dict(S.FLOORS, local=0.0, fast=0.15)   # job strength a model needs per category (score.pick's thresholds; fast gets a small floor)
CELLS = core.CELLS
CAT_NAMES = ("local", "fast", "standard", "strong", "frontier")
NOT_LOADED = "installed but not loaded (Maestro never loads models)"
META_MODELS = ("auto",) + CAT_NAMES + tuple(core.JOBS)
AFFINITY_TTL_S = 30 * 60
MAX_ATTEMPTS = 3
REFRESH_S = 10 * 60
LEARN_S = 24 * 3600      # the personalization refit (learn.run) runs daily
DISCOVERY_MAX_AGE_S = 24 * 3600
MAX_BODY = 64 * 1024 * 1024
_HEX = re.compile(rb"[0-9A-Fa-f]+")   # a chunk size: no sign, no 0x, no underscores
RETRY_STATUS = frozenset((0, 401, 403, 404, 408, 429))   # plus every 5xx: the request is fine, this seat is not
ADMIN_TOKEN_FILE = "admin_token"
MODELS_FILE = "models.json"
DB_FILE = "maestro.db"
UI_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "index.html")
BUCKETS = {"balanced": 2.0, "cost": 6.0, "quality": 1.0}   # cost buckets per decade of price
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1", "[::1]")


class ApiError(Exception):
    def __init__(self, status, message, kind="invalid_request_error", code=None):
        Exception.__init__(self, message)
        self.status, self.kind, self.code = status, kind, code


# ════════════════════════════════════════════════════════════════════════════
# 1. the model list: models.json (stage 2a) with a fallback built from providers.json
# ════════════════════════════════════════════════════════════════════════════

def _override_of(m, cfg):
    """{"category"?, "enabled"?} the user set for this model: the config wins over what the loader left in the entry."""
    ov = ((cfg or {}).get("models_overrides") or {}).get(m["id"])
    ov = ov if isinstance(ov, dict) else {}
    old = m.get("override")
    old = {"category": old} if isinstance(old, str) else old if isinstance(old, dict) else {}
    return dict(old, **ov)


def _norm_model(m, cfg):
    """A models.json entry with every field the router reads present, plus `qp` (the provider name quota is kept under)."""
    mid = str(m["id"])
    access = m.get("access") or ("local" if mid.startswith("local:") else "subscription-cli" if mid.startswith("cli:") else "api")
    if access == "local":
        qp = "local:" + mid[6:].split("/", 1)[0]
    elif access == "subscription-cli":
        qp = mid[4:].split("/", 1)[0] if mid.startswith("cli:") else m.get("provider") or ""
    else:
        qp = m.get("provider") or mid.split("/", 1)[0]
    name = m.get("model") or (mid.split("/", 1)[1] if "/" in mid else mid)
    ov = _override_of(m, cfg)
    out = dict(m, id=mid, model=name, provider=m.get("provider") or qp, access=access, qp=qp,
               style=m.get("style") or {"local": "local", "subscription-cli": "cli"}.get(access, "openai"),
               tools_ok=m.get("tools_ok") if m.get("tools_ok") is not None else access != "subscription-cli",
               jobs=m.get("jobs") if isinstance(m.get("jobs"), dict) else {},
               enabled=ov.get("enabled") is not False and m.get("enabled") is not False,
               routable=access != "local" or m.get("loaded") is not False or bool((cfg or {}).get("local_allow_load")),
               category=m.get("category") or "standard", cat_override=ov.get("category") if ov.get("category") in CAT_NAMES else None)
    return out


def _tier_index(m):
    """The category rank (0 local .. 4 frontier) a model counts as: the user's override, else its category / `serves`, else its score."""
    for c in (m.get("cat_override"), m.get("category"), m.get("serves")):
        if c in CAT_NAMES[1:]:
            return CAT_NAMES.index(c)
    if m.get("access") == "local" and "local" in (m.get("category"), m.get("serves")):
        return 0   # a local model whose size only fits simple work
    g = m.get("G")
    if isinstance(g, (int, float)):
        return max([i for i, c in enumerate(CAT_NAMES) if i and g >= FLOORS[c]] or [1])
    return 1


def _fallback_models(home, cfg):
    """Entries built from providers.json when models.json does not exist: no scores, category from the suggested-model tables
    or the model's size. Good enough to route by category until `maestro-router discover` has run."""
    pv = P.load_providers(home) or {"providers": []}
    tier = {}
    for row in core.SUGGESTED_MODELS.values():
        for r in (row.values() if all(isinstance(v, dict) for v in row.values()) else [row]):
            for t, cell in r.items():
                c = core._cell(cell)
                if c:
                    name = c[0].split("/", 1)[-1]
                    tier[name] = max(tier.get(name, "fast"), t, key=core.TIERS.index)
    out = []
    for e in pv["providers"]:
        for name in e.get("models") or []:
            base = {"model": name, "cost_in": None, "cost_out": None, "G": None, "jobs": {}, "context": None, "vision": None,
                    "enabled": True, "basis": "prior", "benchmarks": []}
            if e["access"] == "local":
                rec = {"id": "%s/%s" % (e["id"], name), "name": name, "endpoint_name": e["id"][6:],
                       "params_b": (e.get("model_info") or {}).get(name, {}).get("params_b") or core._params_b(name)}
                serves, _ = core.rate_local(rec, cfg)
                out.append(dict(base, id=rec["id"], provider=e["id"], access="local", style="local", category="local", serves=serves,
                                loaded=(e.get("model_info") or {}).get(name, {}).get("loaded", True),
                                cost_in=0.0, cost_out=0.0, context=(e.get("model_info") or {}).get(name, {}).get("max_model_len")))
            elif e["access"] == "subscription-cli":
                out.append(dict(base, id="cli:%s/%s" % (e["id"], name), provider=e["id"], access="subscription-cli", style="cli",
                                tools_ok=False, cost_in=0.0, cost_out=0.0, category=tier.get(name, "standard")))
            elif e["status"] == "connected":
                out.append(dict(base, id="%s/%s" % (e["id"], name), provider=e["id"], access="api", style=e["style"],
                                category=tier.get(name) or "standard"))
    return {"version": 1, "generated_at": pv.get("generated_at"), "models": out, "hardware": {}, "local_fit": {},
            "source": "providers.json" if pv["providers"] else "none"}


def _load_models(home=None, cfg=None):
    """models.json as {"models": [...], "hardware", "local_fit", "generated_at", ...}, user overrides applied, or the providers.json
    fallback. `catalog.load_models(home, cfg)` has the same contract; this is only the fallback before the first discovery."""
    home = home or core.maestro_home()
    cfg = cfg if cfg is not None else core.load_config(home)
    data = core._read_json(os.path.join(home, MODELS_FILE))
    if not (isinstance(data, dict) and isinstance(data.get("models"), list)):
        data = _fallback_models(home, cfg)
    else:
        data = dict(data, source="models.json")
    data["models"] = [dict(m, enabled=_override_of(m, cfg).get("enabled", m.get("enabled", True)),
                           override=_override_of(m, cfg) or None) for m in data["models"] if isinstance(m, dict) and m.get("id")]
    return data


def models_loader():
    """`catalog.load_models` (models.json with the user's overrides); before the first discovery, the internal loader above."""
    from .catalog import load_models

    def load(home=None, cfg=None):
        raw = load_models(home, cfg)
        return dict(raw, source=raw.get("source") or "models.json") if raw.get("models") else _load_models(home, cfg)
    return load


def models_view(home=None, cfg=None):
    """(normalized entries, raw file dict) for the server, the UI and `maestro-router setup`."""
    home = home or core.maestro_home()
    cfg = cfg if cfg is not None else core.load_config(home)
    raw = models_loader()(home, cfg)
    return [_norm_model(m, cfg) for m in raw.get("models") or [] if isinstance(m, dict) and m.get("id")], raw


def run_discover(home=None, env=None, cfg=None, refresh_catalog=False):
    """Run `maestro-router discover` (detect, score, write models.json) -> (ok, message)."""
    from . import discover as D
    home = home or core.maestro_home(env)
    try:
        data, notes, _ = D.discover(home=home, env=env, cfg=cfg, refresh_catalog=refresh_catalog)
    except (OSError, ValueError, P.ProviderError) as e:
        return False, "discovery failed: %s" % P.scrub(str(e))
    n = len(data.get("models") or [])
    return True, "discovery finished: %d models%s" % (n, ("; " + "; ".join(notes[:3])) if notes else "")


# ════════════════════════════════════════════════════════════════════════════
# 2. request features, model names, ranking (pure)
# ════════════════════════════════════════════════════════════════════════════

def _parts(c):
    return c if isinstance(c, list) else []


def features(body):
    """What routing needs to know about a chat.completions request."""
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs or not all(isinstance(m, dict) for m in msgs):
        raise ApiError(400, "`messages` must be a non-empty list of message objects")
    system = next((C._text_of(m.get("content")) for m in msgs if m.get("role") in ("system", "developer")), "")
    users = [C._text_of(m.get("content")) for m in msgs if m.get("role") == "user"]
    first_user, last_user = (users[0], users[-1]) if users else ("", "")
    images = sum(1 for m in msgs for p in _parts(m.get("content")) if isinstance(p, dict) and p.get("type") in ("image_url", "input_image"))
    tools = bool(body.get("tools") or body.get("functions"))
    chars = sum(len(C._text_of(m.get("content"))) for m in msgs) + (len(json.dumps(body.get("tools"))) if body.get("tools") else 0)
    last = msgs[-1]
    used_tools = any(m.get("role") == "tool" or m.get("tool_calls") for m in msgs)
    out_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
    tail = []   # the tool results since the last assistant message (a test run there is an outcome signal)
    for m in reversed(msgs):
        if m.get("role") != "tool":
            break
        tail.append(C._text_of(m.get("content")))
    est_in = chars // 4 + 800 * images
    return {"system": system, "first_user": first_user, "user": last_user, "vision": images > 0, "tools": tools or used_tools,
            "tools_param": tools, "est_in": est_in, "max_out": out_tokens if isinstance(out_tokens, int) else None,
            "tool_turn": last.get("role") == "tool" or (last.get("role") == "assistant" and bool(last.get("tool_calls"))),
            "stream": bool(body.get("stream")), "images": images,
            # S1's conversation context (spec §12.2); prefix_in = what a warm seat has cached: everything but the last user message
            "prev_user": users[-2][-1024:] if len(users) > 1 else "", "n_user": len(users),
            "n_tool": sum(1 for m in msgs if m.get("role") == "tool"), "last_tool": last.get("role") == "tool",
            "asst_len": len(next((C._text_of(m.get("content")) for m in reversed(msgs) if m.get("role") == "assistant"), "")),
            "prefix_in": max(0, est_in - len(last_user) // 4), "tool_text": "\n".join(reversed(tail))[-20000:],
            "forwarded": [C._text_of(m.get("content")) for m in msgs] + [json.dumps(m["tool_calls"]) for m in msgs if m.get("tool_calls")]}


def keylike(f):
    """Does any message forwarded upstream (system, every turn, tool calls and results) hold a key-like string (spec §4.5)?"""
    if "keylike" not in f:
        from . import decide as D
        f["keylike"] = any(D.is_private(t) for t in f["forwarded"])
    return f["keylike"]


def classification_text(f):
    """The last user message; a very short one ("yes", "continue") borrows the start of the system prompt as weak context."""
    text = f["user"]
    if len(text.split()) < 4 and f["system"]:
        text = (text + "\n\n" + f["system"][:500]).strip()
    return text


def parse_name(name, by_id):
    """(kind, arg): auto | category | job | model | unknown."""
    n = (name or "").strip()
    if n in by_id:
        return "model", by_id[n]
    low = n.lower()
    rest = low[8:] if low.startswith("maestro/") else low if low in ("maestro", "auto") else None
    if rest is not None:
        if rest in ("", "auto", "maestro"):
            return "auto", None
        if rest in CAT_NAMES:
            return "category", rest
        if rest in core.JOBS:
            return "job", rest
        if n[8:] in by_id:
            return "model", by_id[n[8:]]
    same = [m for m in by_id.values() if m["model"] == n]
    return ("model", same[0]) if len(same) == 1 else ("unknown", n)


def strength(m, job):
    """Job strength 0..1, else the overall score G, else None (unscored)."""
    v = (m.get("jobs") or {}).get(job)
    return v if isinstance(v, (int, float)) else m["G"] if isinstance(m.get("G"), (int, float)) else None


def meets_floor(m, job, cell):
    """Does this model reach the category's floor? An override of the category counts as the user's word; unscored models
    are judged by their category."""
    if cell == "local":
        return m["access"] == "local"
    if m["cat_override"]:
        return CAT_NAMES.index(m["cat_override"]) >= CAT_NAMES.index(cell)
    s = strength(m, job)
    if m["access"] == "local" and m.get("category") != "specialist" and _tier_index(m) < CAT_NAMES.index(cell):
        return False   # a plain local model serves up to the category its size earns; a specialist is judged by its job strength
    if s is None:
        return _tier_index(m) >= CAT_NAMES.index(cell)
    # as in score.pick: the job strength reaches the floor and the model's own tier reaches the cell (a specialist needs only the first)
    return s >= FLOORS[cell] and (m.get("category") == "specialist" or _tier_index(m) >= CAT_NAMES.index(cell))


def est_cost(m, est_in, est_out):
    """Expected USD of one request. Local models are free; a subscription seat costs a token amount that grows with its strength, so
    the weakest seat that does the job goes first and any free local model beats it."""
    if m["access"] == "local":
        return 0.0
    if m["access"] == "subscription-cli":
        return 1e-5 * (1 + 10 * (strength(m, "ask") or 0.5))
    ci = m.get("cost_in") if isinstance(m.get("cost_in"), (int, float)) else 5.0
    co = m.get("cost_out") if isinstance(m.get("cost_out"), (int, float)) else 15.0
    return (est_in * ci + est_out * co) / 1e6


def rank_key(c, objective, prefer_local):
    """Sort key of a candidate, lowest first: near its limit last; then (balanced/cost) price in log buckets, or (quality) strength;
    then local first if preferred, quota headroom, observed latency, strength; then id."""
    lb = round(math.log10(c["cost"] + 1e-6) * BUCKETS.get(objective, 2.0))
    s = round((c["strength"] if c["strength"] is not None else 0.4) / 0.04)
    hr = round(c["headroom"] if c["headroom"] is not None else 0.5, 1)
    lat = round((c["latency"] or 0) / 500.0)
    loc = 0 if prefer_local and c["m"]["access"] == "local" else 1
    head = (c["penalty"] > 0, -s, lb) if objective == "quality" else (c["penalty"] > 0, lb)
    return head + (loc, -hr, lat, -s, c["m"]["id"])


def conv_key(f, hdrs, cfg):
    sid = hdrs.get("x-session-id") or (hdrs.get((cfg.get("session_affinity_header") or "").lower()) if cfg.get("session_affinity_header") else None)
    return "s:" + sid if sid else "h:" + hashlib.sha256((f["system"] + "\0" + f["first_user"]).encode("utf-8", "replace")).hexdigest()


def short_hash(s):
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()[:12]


# ════════════════════════════════════════════════════════════════════════════
# 3. the decision log (sqlite)
# ════════════════════════════════════════════════════════════════════════════

SCHEMA = """create table if not exists decisions (
  id integer primary key autoincrement, ts real, conv text, requested text, job text, difficulty integer, cell text, category text,
  chosen text, candidates text, reason text, latency_ms real, ttft_ms real, prompt_tokens integer, completion_tokens integer,
  usage_est integer, status integer, attempts text, stream integer, tools integer, cost real, baseline real, prompt text)"""
COLS = ("ts", "conv", "requested", "job", "difficulty", "cell", "category", "chosen", "candidates", "reason", "latency_ms", "ttft_ms",
        "prompt_tokens", "completion_tokens", "usage_est", "status", "attempts", "stream", "tools", "cost", "baseline", "prompt",
        "s1", "p_success", "eu", "decider", "shadow", "explore", "propensity", "policy")   # the last 8: S1 (learn.migrate adds them)


def open_db(home):
    core.ensure_home(home)
    path = os.path.join(home, DB_FILE)
    try:   # 0600 before WAL mode creates the -wal and -shm files, which SQLite gives the database file's mode
        os.close(os.open(path, os.O_RDWR | os.O_CREAT, 0o600))
        os.chmod(path, 0o600)
    except OSError:
        pass
    db = sqlite3.connect(path, check_same_thread=False, timeout=10, isolation_level=None)
    try:
        db.execute("pragma journal_mode=wal")
    except sqlite3.DatabaseError:
        pass
    db.execute(SCHEMA)
    from . import learn
    learn.migrate(db)          # the S1 columns and the outcomes table (spec §10.1)
    db.execute("create index if not exists decisions_ts on decisions (ts)")
    db.execute("delete from decisions where ts < ?", (time.time() - 90 * 86400,))   # the log keeps 90 days
    db.execute("delete from outcomes where ts < ?", (time.time() - 90 * 86400,))
    for p in (path, path + "-wal", path + "-shm"):   # also tightens files an older version left 0644
        try:
            os.chmod(p, 0o600)
        except OSError:
            pass
    return db


def _row(cur, r):
    d = dict(zip([c[0] for c in cur.description], r))
    for k in ("candidates", "attempts"):
        try:
            d[k] = json.loads(d[k]) if d.get(k) else []
        except ValueError:
            d[k] = []
    return d


# ════════════════════════════════════════════════════════════════════════════
# 4. the application: state shared by every request
# ════════════════════════════════════════════════════════════════════════════

class App:
    def __init__(self, home, env, jev_fetch=None):
        self.home, self.env, self.jev_fetch = home, env, jev_fetch
        self.db, self.db_lock, self.cfg_lock = open_db(home), threading.Lock(), threading.Lock()
        self.aff, self.aff_lock = {}, threading.Lock()
        self.lat = {}   # model id -> smoothed time to first byte, ms
        self.cache = (None, None)
        self.started = time.time()
        self.discovering, self.last_discovery = False, None
        self.host, self.port, self.loopback = DEFAULT_HOST, DEFAULT_PORT, True
        self.stop = threading.Event()
        self.token = load_admin_token(home)
        self.api_key = P.get_key(("MAESTRO_API_KEY",), env, home)
        self._load_latency()

    # ---- config and models
    def cfg(self):
        return core.load_config(self.home)

    def models(self, cfg=None):
        """{"list": [...], "by_id": {...}, "raw": file dict}, reloaded when models.json, config.json or providers.json change."""
        cfg = cfg if cfg is not None else self.cfg()
        sig = tuple(_mtime(os.path.join(self.home, f)) for f in (MODELS_FILE, "config.json", P.PROVIDERS_FILE))
        if self.cache[0] != sig:
            lst, raw = models_view(self.home, cfg)
            self.cache = (sig, {"list": lst, "by_id": {m["id"]: m for m in lst}, "raw": raw})
        return self.cache[1]

    def strongest(self, lst):
        pool = [m for m in lst if m["enabled"] and m["access"] == "api" and isinstance(m.get("G"), (int, float))]
        return max(pool, key=lambda m: (m["G"], -(m.get("cost_out") or 0))) if pool else None

    def write_config(self, mutate):
        """Read config.json as the user wrote it, let `mutate(dict)` change it, save it (indent 2). Returns the effective config."""
        with self.cfg_lock:
            raw = core._read_json(core.config_path(self.home))
            raw = raw if isinstance(raw, dict) else {}
            mutate(raw)
            if not core._write_json(core.config_path(self.home), raw, indent=2):
                raise ApiError(500, "cannot write %s" % core.config_path(self.home), "server_error")
        return self.cfg()

    # ---- session affinity
    def aff_get(self, key, now):
        with self.aff_lock:
            v = self.aff.get(key)
            return v if v and now - v["ts"] < AFFINITY_TTL_S else None

    def aff_put(self, key, val, now):
        with self.aff_lock:
            self.aff[key] = dict(val, ts=now)
            if len(self.aff) > 2000:
                for k in [k for k, v in self.aff.items() if now - v["ts"] >= AFFINITY_TTL_S] or list(self.aff)[:500]:
                    self.aff.pop(k, None)

    # ---- routing
    def candidates(self, ml, cfg, f, cls, cell, kind, now, requested=None):
        """(ranked, passing, skipped): ranked = passes the hard filters and the category floor, best first; passing = passes the
        filters only (strongest first); skipped = [{id, reason}]. `requested` is the concrete model id the caller named, if any."""
        prefer_local = cfg["prefer_local"]
        if cell == "local" and not prefer_local and kind not in ("category", "model"):
            cell = "fast"   # the ladder's local cell is a preference; only maestro/local or a concrete local id asks for it
        objective = cfg["objective"]
        cap = CAT_NAMES.index(cfg["max_tier"])
        est_out = min(f["max_out"] or 512, 4000)

        def evaluate(local_ok, local_only, floor_cell, allow_free=False):
            passing, skipped = [], []
            for m in ml["list"]:
                why = None
                if not m["enabled"]:
                    continue
                if not m["routable"]:
                    why = NOT_LOADED
                elif f["tools"] and m["access"] == "subscription-cli":
                    why = "subscription CLI seats cannot use tools"
                elif f["vision"] and m["access"] == "subscription-cli":
                    why = "subscription CLI seats cannot take images"
                elif f["tools"] and not m["tools_ok"]:
                    why = "no tool support"
                elif f["vision"] and m.get("vision") is False:
                    why = "no vision support"
                elif m.get("context") and f["est_in"] + est_out > m["context"]:
                    why = "context too small (%d tokens)" % m["context"]
                elif m["access"] != "local" and _tier_index(m) > cap and not m["cat_override"]:
                    why = "above max_tier %s" % cfg["max_tier"]
                elif m["access"] == "local" and not local_ok:
                    why = "prefer_local is off"
                elif (m["access"] == "api" and not m["cat_override"] and m["id"] != requested and not allow_free and (
                        m["model"].endswith(":free") or (m.get("cost_in") == 0 and m.get("cost_out") == 0))):
                    why = "free tier (price 0): rate-limited, not counted on unless requested"
                elif local_only and m["access"] != "local":
                    continue
                else:
                    ok, r = Q.is_available(m["qp"], m["model"], self.home, now)
                    why = None if ok else "unavailable: " + r
                if why:
                    skipped.append({"id": m["id"], "reason": why})
                    continue
                s = strength(m, cls.job)
                passing.append({"m": m, "strength": s, "cost": est_cost(m, f["est_in"], est_out),
                                "headroom": Q.headroom(m["qp"], m["model"], self.home, now),
                                "penalty": Q.penalty(m["qp"], m["model"], self.home, now), "latency": self.lat.get(m["id"])})
            ranked = sorted((c for c in passing if meets_floor(c["m"], cls.job, floor_cell)), key=lambda c: rank_key(c, objective, prefer_local))
            return ranked, passing, skipped

        local_ok = prefer_local or kind in ("category", "model")
        ranked, passing, skipped = evaluate(local_ok, cell == "local", cell)
        note = ""
        if not ranked and cell == "local":
            cell, note = "fast", "no local model can serve this; using fast"
            ranked, passing, skipped = evaluate(local_ok, False, cell)
        if not ranked and not local_ok:
            ranked, passing, skipped = evaluate(True, False, cell)
            note = "no cloud model available; local models allowed"
        if not ranked and not passing and any(s["reason"].startswith("free tier") for s in skipped):
            # Last resort: price-0 API models (free tiers, and the coding plans models.dev lists at price 0) when nothing else can
            # serve, so a first run with one plan key routes instead of answering 503.
            ranked, passing, skipped = evaluate(True, False, cell, allow_free=True)
            note = "only price-0 models (free tiers or plans) can serve this; using one"
        relaxed = False
        if not ranked and passing:
            relaxed = True
            ranked = sorted(passing, key=lambda c: (-(c["strength"] if c["strength"] is not None else _tier_index(c["m"]) / 5.0),
                                                     rank_key(c, objective, prefer_local)))
            note = "no model reaches the %s floor for %s; using the strongest available" % (cell, cls.job)
        passing.sort(key=lambda c: -(c["strength"] or 0))
        return {"ranked": ranked, "passing": passing, "skipped": skipped, "cell": cell, "note": note, "relaxed": relaxed}

    def plan(self, body, hdrs):
        """Everything decided before any upstream call: {decision..., "order": [candidate dicts], "skipped": [...]}. Raises ApiError."""
        cfg, now = self.cfg(), time.time()
        f = features(body)
        ml = self.models(cfg)
        name = body.get("model") if isinstance(body.get("model"), str) and body.get("model").strip() else "maestro/auto"
        kind, arg = parse_name(name, ml["by_id"])
        if kind == "unknown":
            raise ApiError(404, "the model %r does not exist. Use maestro/auto, maestro/<category or job> or one of GET /v1/models" % name,
                           "invalid_request_error", "model_not_found")
        if kind == "model" and not arg["enabled"]:
            raise ApiError(404, "the model %r is disabled in Maestro" % arg["id"], "invalid_request_error", "model_not_found")
        if kind == "model" and not arg["routable"]:
            raise ApiError(409, "%s is %s. Load it on its server yourself, or set local_allow_load in the config to let a request load it "
                           "(that can exhaust memory)" % (arg["id"], NOT_LOADED), "invalid_request_error", "model_not_loaded")
        akey = conv_key(f, hdrs, cfg) + "|" + name
        utility = cfg.get("decider") == "utility" and kind in ("auto", "job")
        warm = self.aff_get(akey, now)   # this conversation's last seat (every turn, 30 min): decide prices switching away from it
        aff = warm if f["tool_turn"] and not utility else None   # the ladder keeps a tool turn on its model
        classifier = "affinity" if aff else "rules"
        if aff:
            cls = core.Classification(aff["job"], aff["difficulty"], 1.0, classifier="affinity")
        else:
            cls = core.classify_rules(classification_text(f))
            if kind == "job":
                cls.job = arg
        cell = arg if kind == "category" else core.CELLS[max(1, _tier_index(arg))] if kind == "model" else core.target_tier(cls, cfg)
        if aff and kind in ("auto", "job"):
            cell = aff["cell"]
        r = self.candidates(ml, cfg, f, cls, cell, kind, now, arg["id"] if kind == "model" else None)
        ranked, notes = r["ranked"], [r["note"]] if r["note"] else []
        if kind == "model":
            why = next((s["reason"] for s in r["skipped"] if s["id"] == arg["id"]), None)
            hard = why and not why.startswith("unavailable")
            if hard:
                raise ApiError(400, "%s cannot serve this request: %s" % (arg["id"], why), "invalid_request_error", "model_unsuitable")
            first = next((c for c in r["passing"] if c["m"]["id"] == arg["id"]), None)
            ranked = ([first] if first else []) + [c for c in ranked if c["m"]["id"] != arg["id"]]
            if not first:
                notes.append("%s is %s; falling back" % (arg["id"], why))
        elif aff:
            hit = next((c for c in r["passing"] if c["m"]["id"] == aff["id"]), None)
            if hit:
                ranked = [hit] + [c for c in ranked if c["m"]["id"] != aff["id"]]
                notes.append("same model as earlier in this conversation")
        if not ranked:
            top = "; ".join("%s: %s" % (s["id"], s["reason"]) for s in r["skipped"][:4])
            raise ApiError(503, "no model can serve this request%s" % (" (%s)" % top if top else ""), "server_error", "no_route")
        jev_note = ""
        if kind == "auto" and not aff and len(ranked) > 1:
            ranked, cls, jev_note = self.jev(ranked, cls, f, cfg, ml)
        if cfg.get("private") or (utility and keylike(f)):   # local models only, whichever decider (its order stands if decide fails)
            ranked = [c for c in ranked if c["m"]["access"] == "local"] or _raise_private()
        s1d = None
        if utility or (cfg.get("s1_shadow", True) and kind in ("auto", "job")):
            try:
                s1d = self.s1_decide(f, cls, r["passing"], cfg, ml, warm, hdrs, kind, None if utility else ranked[0]["m"]["id"])
            except ApiError:
                raise
            except Exception as e:   # the decision layer never breaks a request: the ladder's order stands
                notes.append("the utility decider failed (%s); used the ladder" % e.__class__.__name__)
        ladder_pick = ranked[0]["m"]["id"]
        if not utility and kind in ("auto", "job") and len(ranked) > 1:
            demoted = self.outcome_demotions(cls.job)
            if demoted:
                bad = [c for c in ranked if c["m"]["id"] in demoted]
                if bad and len(bad) < len(ranked):
                    ranked = [c for c in ranked if c["m"]["id"] not in demoted] + bad
                    notes.append("demoted %s: recorded failures on %s" % (", ".join(sorted(demoted)), cls.job))
        if utility and s1d and s1d["order"]:
            ranked = s1d["order"]
        elif utility:
            utility = False
            notes += ["utility: no seat qualifies; used the ladder"] if s1d else []
        chosen = ranked[0]
        reason = "%s/d%d -> %s" % (cls.job, cls.difficulty, r["cell"]) if kind != "model" else "requested %s" % arg["id"]
        if kind == "category":
            reason = "%s requested; job %s/d%d" % (r["cell"], cls.job, cls.difficulty)
        s = chosen["strength"]
        price = {"local": "free", "subscription-cli": "subscription"}.get(chosen["m"]["access"]) or "~$%.4f/request" % chosen["cost"]
        reason += "; %s (%s, %s%s)" % (chosen["m"]["id"], "%s %.2f" % (cls.job, s) if s is not None else "unscored", price,
                                       ", quota %d%% left" % round(chosen["headroom"] * 100) if chosen["headroom"] is not None else "")
        if utility:
            reason = s1d["d"]["reason"] + ("; explored the runner-up" if s1d["explore"] else "")
        reason += "".join("; " + n for n in notes + ([jev_note] if jev_note else []))
        cat = chosen["m"]["cat_override"] or ("local" if chosen["m"]["access"] == "local" else chosen["m"]["category"])
        decision = {"requested": name, "kind": kind, "job": cls.job, "difficulty": cls.difficulty, "confidence": round(cls.confidence, 3),
                    "classifier": "jev" if jev_note else classifier, "cell": r["cell"], "category": cat,
                    "needs": [n for n, on in (("tools", f["tools"]), ("vision", f["vision"])) if on], "chosen": chosen["m"]["id"],
                    "reason": reason, "affinity": bool(aff) or bool(utility and warm and warm["id"] == chosen["m"]["id"]),
                    "candidates": [c["m"]["id"] for c in ranked[:20]], "decider": "utility" if utility else "ladder"}
        if s1d:
            d = s1d["d"]
            opt = next((o for o in d["options"] if o["seat"] == chosen["m"]["id"]), {})
            decision.update(p_success=opt.get("p"), eu=opt.get("eu"), escalate_to=d["escalate_to"], flags=d["flags"], options=d["options"][:12],
                            shadow=ladder_pick if utility else d.get("shadow"), explore=s1d["explore"], propensity=s1d["propensity"],
                            policy=s1d["policy"], posterior=s1d["post"], s1=s1d["log"])
        return dict(decision, order=ranked, skipped=r["skipped"], akey=akey, feats=f, cfg=cfg)

    def s1_decide(self, f, cls, passing, cfg, ml, warm, hdrs, kind, chosen=None):
        """S1's posterior and decide.py's expected-utility ranking of the candidates that pass the hard filters (spec §5, §12.2).
        With `chosen` (the ladder's pick, shadow mode) decide reports that seat and its own argmax as "shadow". Returns
        {"d": decide's keys, "order": candidates in EU order (explore may swap the top two), "post", "log" (the s1 column),
        "policy", "explore", "propensity"}. Raises ApiError 503 in private mode with no local model."""
        from . import decide as D, s1 as S1
        md = S1.load()[0]
        post = S1.posterior({"text": classification_text(f), "mode": "proxy", "tools": f["tools_param"], "images": f["images"],
                             "est_in": f["est_in"], "max_tokens": f["max_out"], "prev_user": f["prev_user"], "n_user": f["n_user"],
                             "n_tool": f["n_tool"], "last_tool": f["last_tool"], "asst_len": f["asst_len"],
                             "prev_job": (warm or {}).get("job")}, cls)
        bg = kind == "job" or (hdrs.get("x-maestro-background") or "").strip() in ("1", "true")
        d = D.decide([D.seat_of(c) for c in passing], post, {
            "mode": "proxy", "cfg": cfg, "home": self.home, "text": f["user"], "private": keylike(f), "est_in": f["est_in"], "prefix_in": f["prefix_in"],
            "warm": (warm or {}).get("id"), "background": bg, "s1": md and md["header"], "catalog_sha": ml["raw"].get("catalog_sha"),
            "chosen": chosen, "difficulty": cls.difficulty})
        if chosen is None and d["chosen"] is None and "private_no_local" in d["flags"]:
            _raise_private()
        by = {c["m"]["id"]: c for c in passing}
        order, ex, prop = [by[o["seat"]] for o in d["options"] if o["seat"] in by], 0, 1.0
        if chosen is None and cfg.get("explore") and len(order) > 1:
            from . import learn as L
            ex, prop = L.explore(d["options"], post, (d.get("trace") or {}).get("V") or 0.0, cfg["explore"])
            if ex:
                order[0], order[1] = order[1], order[0]
        jp = sorted(post["job_p"].items(), key=lambda kv: -kv[1])[:3]
        log = json.dumps({"job_p": {j: round(p, 3) for j, p in jp}, "b": [round(x, 3) for x in post["b"]], "group": post["group"],
                          "trace": d.get("trace")}, separators=(",", ":"))
        view = {"job_p": dict((j, round(p, 3)) for j, p in jp), "b": [round(x, 3) for x in post["b"]], "group": post["group"],
                "source": post.get("source"), "tools": round(post["tools"], 3), "stakes": round(post["stakes"], 3),
                "detectable": round(post["detectable"], 3), "ood": round(post["ood"], 3)}
        return {"d": d, "order": order, "post": view, "log": log, "explore": ex, "propensity": prop,
                "policy": core.s1_policy(md) if post.get("source") == "s1" else post.get("source")}

    def jev(self, ranked, cls, f, cfg, ml):
        """With TYPESAFE_API_KEY set (classifier auto or jev), Jev names the job and difficulty and may pick among the top five."""
        key = self.env.get("TYPESAFE_API_KEY")
        if not key or cfg["classifier"] == "rules" or self.env.get("MAESTRO_CLASSIFIER") == "rules" or cfg.get("private") or keylike(f):
            return ranked, cls, ""          # private tasks never go to Jev (a cloud service)
        top = ranked[:5]
        cands = [{"name": c["m"]["id"], "description": "%s model %s (%s, category %s%s)" % (
            "local" if c["m"]["access"] == "local" else "subscription" if c["m"]["access"] == "subscription-cli" else "cloud",
            c["m"]["model"], c["m"]["provider"], c["m"]["cat_override"] or c["m"]["category"],
            ", score %.2f" % c["m"]["G"] if isinstance(c["m"].get("G"), (int, float)) else "")} for c in top]
        cache = core.load_cache(self.home)
        j = core.JevClassifier(key, fetch=self.jev_fetch, cache=cache).classify(classification_text(f), cands, cfg["objective"])
        if not j.jev:
            return ranked, cls, ""
        core.save_cache(cache, self.home)
        pick = next((c for c in top if c["m"]["id"] == j.jev["choice"]), None)
        if pick and j.jev["confidence"] >= cfg["jev_min_confidence"]:
            ranked = [pick] + [c for c in ranked if c is not pick]
        return ranked, j, "jev chose %s (confidence %.2f)" % (j.jev["choice"], j.jev["confidence"])

    # ---- upstream
    def spec_for(self, m, cfg):
        """What `clients.complete` takes as its provider: a registry id, or a spec dict for a local endpoint."""
        if m["access"] != "local":
            return m["qp"]
        pid = m["qp"]
        for e in (P.load_providers(self.home) or {}).get("providers") or []:
            if e["id"] == pid:
                return dict(e, style="local", overrides={})
        for ep in core.local_endpoints(cfg, self.env):
            if "local:" + ep["name"] == pid:
                return {"id": pid, "name": "Local: " + ep["name"], "style": "local", "base_url": ep["url"] + "/v1",
                        "key_env": ep.get("api_key_env"), "overrides": {}}
        return pid

    def host_of(self, m, cfg):
        """host:port of the model's upstream, or None (unknown, or loopback, where one host serves unrelated providers)."""
        spec = self.spec_for(m, cfg)
        try:
            url = P.base_url(P.BY_ID[spec], self.env, self.home, cfg) if isinstance(spec, str) and spec in P.BY_ID else \
                spec.get("base_url") if isinstance(spec, dict) else None
            u = urllib.parse.urlsplit(url or "")
        except (ValueError, KeyError):
            return None
        return None if not u.hostname or u.hostname in ("localhost", "::1") or u.hostname.startswith("127.") else u.netloc.lower()

    def attempt(self, m, body, stream, cfg):
        """One upstream call. Returns (status, headers, out, first) where `first` is the first stream item for a stream. Streams are
        pulled once here so a failure before the first byte is an ordinary failed attempt."""
        t0 = time.time()
        try:
            status, hdrs, out = C.complete(self.spec_for(m, cfg), m["model"], body, stream=stream, env=self.env, home=self.home, cfg=cfg)
        except Exception as e:  # a bug in one client must not take the request down
            status, hdrs, out = 502, {}, C.error_body("%s while calling %s" % (e.__class__.__name__, m["id"]), code=502)
        first = None
        if 200 <= status < 300 and stream and isinstance(out, C.Stream):
            try:
                first = next(out)
            except StopIteration:
                status, out = 502, C.error_body("the upstream stream was empty", code=502)
            except (C.UpstreamError, OSError, ValueError) as e:
                status, out = 502, C.error_body(P.scrub(str(e)) or "the upstream stream failed", code=502)
            if first is None:
                hdrs = {}
        Q.record(m["qp"], m["model"], status, hdrs, self.home)
        return status, hdrs, out, first, (time.time() - t0) * 1000.0

    def run(self, plan, body):
        """Try the candidates in order (three at most) until one answers. Returns a turn dict; the caller relays it and calls finish()."""
        cfg, stream = plan["cfg"], plan["feats"]["stream"]
        self.signals(plan)
        attempts, last, bad_ids, bad_hosts = [], None, set(), set()
        for c in plan["order"]:
            m = c["m"]
            host = self.host_of(m, cfg)
            if len(attempts) >= MAX_ATTEMPTS:
                break
            if m["qp"] in bad_ids or (host and host in bad_hosts):   # this provider just failed as a whole: try another one
                continue
            status, hdrs, out, first, ms = self.attempt(m, body, stream, cfg)
            attempts.append({"model": m["id"], "status": status, "ms": round(ms), "error": None if 200 <= status < 300 else _err_text(out)})
            last = (m, status, hdrs, out, first, ms)
            if 200 <= status < 300 or not (status in RETRY_STATUS or status >= 500):
                break
            if status in (0, 401, 403) or status >= 500:   # a dead host, a bad key or an outage: the provider's other models share it
                bad_ids.add(m["qp"])
                bad_hosts.add(host)
            elif status == 429:                            # rate limits are per provider account here
                bad_ids.add(m["qp"])
        m, status, hdrs, out, first, ms = last
        ok = 200 <= status < 300
        plan.update(attempts=attempts, model=m, status=status, out=out, first=first, ttft=ms, t0=time.time() - ms / 1000.0)
        d = {k: plan[k] for k in ("requested", "job", "difficulty", "category", "reason")}
        d.update(model=m["id"] if ok else None, fallbacks=len(attempts) - 1, affinity=plan["affinity"], category=m["cat_override"] or ("local" if m["access"] == "local" else m["category"]))
        plan.update(category=d["category"], chosen=m["id"])
        plan["decision_id"] = self.log_row(plan, status)      # the row id: `maestro-router feedback <id>`, POST /api/feedback, X-Maestro-Feedback
        d.update({k: plan.get(k) for k in ("decider", "p_success", "escalate_to", "decision_id") if plan.get(k) is not None})
        plan["header"] = json.dumps(dict(d, reason=d["reason"][:240], attempted=[a["model"] for a in attempts]), separators=(",", ":"))
        now = time.time()
        if ok:
            b = (plan.get("posterior") or {}).get("b") or (None, None)
            self.aff_put(plan["akey"], {"id": m["id"], "job": plan["job"], "difficulty": plan["difficulty"], "cell": plan["cell"],
                                        "mu_b": b[0], "sd_b": b[1]}, now)
            self.lat[m["id"]] = ms if m["id"] not in self.lat else 0.7 * self.lat[m["id"]] + 0.3 * ms
        if plan["decision_id"] is not None:
            from . import decide as D
            self.aff_put("last|" + plan["akey"].split("|")[0], {"id": plan["decision_id"], "sig": D.minhash(plan["feats"]["user"]),
                                                                "rank": CAT_NAMES.index(d["category"]) if d["category"] in CAT_NAMES else None}, now)
        return plan

    def signals(self, plan):
        """Outcome signals for the conversation's previous decision (spec §10.2): a test-runner summary in the tool results of this
        request, or the implicit signals of a new user turn (escalate counts; reask, thanks and override are logged at weight 0)."""
        f, now = plan["feats"], time.time()
        prev = self.aff_get("last|" + plan["akey"].split("|")[0], now)
        if not prev:
            return []
        from . import decide as D, learn as L
        got = []
        if f["last_tool"]:
            pt = L.parse_tests(f["tool_text"])
            got = [("tests", pt[0] / float(pt[1]), {"tests": list(pt)})] if pt and pt[1] else []
        elif not f["tool_turn"]:
            kind, ml = plan["kind"], self.models(plan["cfg"])
            m = ml["by_id"].get(plan["chosen"]) if kind == "model" else None
            cat = plan["cell"] if kind == "category" else (m["cat_override"] or ("local" if m["access"] == "local" else m["category"])) if m else None
            cur = {"ts": now, "text": f["user"], "sig": D.minhash(f["user"]), "rank": CAT_NAMES.index(cat) if cat in CAT_NAMES else None,
                   "model": plan["chosen"] if kind == "model" else None}
            got = [(sg, v, None) for sg, v in L.implicit(prev, cur)]
        try:
            with self.db_lock:
                for sg, v, det in got:
                    L.record(None, prev["id"], sg, v, det, db=self.db)
        except sqlite3.DatabaseError:
            return []
        return got

    def outcome_demotions(self, job):
        """Models with >=2 explicit failure outcomes and no explicit success for this job.
        The ladder used to record outcomes but never act on them; this is the smallest
        ladder-visible learning step: such models sort after their peers (never removed,
        so a demoted model still answers when it is the only candidate)."""
        try:
            with self.db_lock:
                rows = self.db.execute(
                    "select d.chosen, sum(case when o.value = 0 then 1 else 0 end), sum(case when o.value > 0 then 1 else 0 end) "
                    "from outcomes o join decisions d on d.id = o.decision_id "
                    "where o.signal = 'explicit' and d.job = ? and d.chosen is not null group by d.chosen", (job,)).fetchall()
        except Exception:
            return set()
        return {m for m, fails, oks in rows if fails >= 2 and not oks}

    def feedback(self, body):
        """POST /api/feedback {id, value} and the X-Maestro-Feedback header: (status, JSON)."""
        from . import learn as L
        try:
            with self.db_lock:
                return L.api_feedback(self.db, body)
        except sqlite3.DatabaseError as e:
            return 500, {"error": {"message": "cannot record feedback (%s)" % e.__class__.__name__, "type": "server_error"}}

    # ---- log
    def _load_latency(self):
        try:
            for mid, ms in self.db.execute("select chosen, avg(ttft_ms) from (select chosen, ttft_ms from decisions where status < 400 "
                                           "and ttft_ms is not null order by id desc limit 500) group by chosen"):
                self.lat[mid] = ms
        except sqlite3.DatabaseError:
            pass

    def _row_values(self, turn, status, usage=None, est_out_chars=0, total_ms=None):
        cfg, f, m = turn["cfg"], turn["feats"], turn["model"]
        est = not (isinstance(usage, dict) and usage.get("prompt_tokens") is not None)
        pin = f["est_in"] if est else usage.get("prompt_tokens") or 0
        pout = est_out_chars // 4 if est else usage.get("completion_tokens") or 0
        cost = est_cost(m, pin, pout) if m["access"] == "api" else 0.0
        ml = self.models(cfg)
        top = self.strongest(ml["list"])
        base = (pin * (top.get("cost_in") or 0) + pout * (top.get("cost_out") or 0)) / 1e6 if top else cost
        return (turn["t0"], short_hash(turn["akey"].split("|")[0]), turn["requested"], turn["job"], turn["difficulty"], turn["cell"],
                turn["category"], m["id"] if 200 <= status < 300 else None, json.dumps(turn["candidates"]), turn["reason"][:500],
                total_ms if total_ms is not None else (time.time() - turn["t0"]) * 1000.0, turn["ttft"], pin, pout, int(est), status,
                json.dumps(turn["attempts"]), int(f["stream"]), int(f["tools"]), cost, base,
                f["user"][:120] if cfg.get("log_prompts") else None,
                turn.get("s1"), turn.get("p_success"), turn.get("eu"), turn.get("decider"), turn.get("shadow"),
                int(bool(turn.get("explore"))) if turn.get("s1") else None, turn.get("propensity"), turn.get("policy"))

    def log_row(self, turn, status):
        """Insert the decision row once a seat has answered (or all failed); finish() completes it. Returns its id, or None."""
        try:
            with self.db_lock:
                cur = self.db.execute("insert into decisions (%s) values (%s)" % (",".join(COLS), ",".join("?" * len(COLS))),
                                      self._row_values(turn, status or 502, total_ms=turn["ttft"]))
                return cur.lastrowid
        except sqlite3.DatabaseError:
            return None

    def finish(self, turn, status, usage=None, est_out_chars=0, total_ms=None):
        """Complete the decision row of a finished request. `usage` is the upstream usage dict, else tokens are estimated."""
        row = self._row_values(turn, status, usage, est_out_chars, total_ms)
        try:
            with self.db_lock:
                if turn.get("decision_id") is not None:
                    self.db.execute("update decisions set %s where id = ?" % ",".join("%s = ?" % c for c in COLS), row + (turn["decision_id"],))
                else:
                    self.db.execute("insert into decisions (%s) values (%s)" % (",".join(COLS), ",".join("?" * len(COLS))), row)
        except sqlite3.DatabaseError:
            pass

    def decisions(self, limit=50):
        try:
            with self.db_lock:
                cur = self.db.execute("select * from decisions order by id desc limit ?", (max(1, min(int(limit), 500)),))
                return [_row(cur, r) for r in cur.fetchall()]
        except (sqlite3.DatabaseError, ValueError):
            return []

    def stats(self):
        lt = time.localtime()
        since = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
        out = {"since": since, "requests": 0, "errors": 0, "fallbacks": 0, "cost": 0.0, "baseline": 0.0, "by_category": {}, "by_model": {}}
        try:
            with self.db_lock:
                rows = self.db.execute("select category, chosen, status, attempts, cost, baseline from decisions where ts >= ?", (since,)).fetchall()
        except sqlite3.DatabaseError:
            return out
        for cat, chosen, status, att, cost, base in rows:
            out["requests"] += 1
            out["errors"] += status >= 400
            out["cost"] += cost or 0.0
            out["baseline"] += base or 0.0
            try:
                out["fallbacks"] += max(0, len(json.loads(att or "[]")) - 1)
            except ValueError:
                pass
            if status < 400:
                out["by_category"][cat or "?"] = out["by_category"].get(cat or "?", 0) + 1
                out["by_model"][chosen] = out["by_model"].get(chosen, 0) + 1
        out["saved"] = max(0.0, out["baseline"] - out["cost"])
        return out

    # ---- periodic work
    def refresh_loop(self, stop, interval):
        while not stop.is_set():
            try:
                Q.refresh(self.env, self.home)
            except Exception:  # a bad probe must not end the thread
                pass
            stop.wait(interval)

    def learn_loop(self, stop, interval, first=600.0):
        """The personalization refit (spec §10.3: replay, fit, guards, personal.json), first after `first` seconds, then daily."""
        wait = min(first, interval)
        while not stop.wait(wait):
            try:
                self.learn_once()
            except Exception:  # a bad row must not end the thread
                pass
            wait = interval

    def learn_once(self):
        from . import learn, s1
        db = sqlite3.connect(os.path.join(self.home, DB_FILE), timeout=10, isolation_level=None)   # its own connection
        try:
            return learn.run(self.home, db, weights_sha=core.s1_policy(s1.load()[0]))
        finally:
            db.close()

    def systemone_ctx(self):
        """POST /v1/systemone's context: proxy mode, and the server's hard-filtered candidates as the seats of a `model` question."""
        cfg = self.cfg()

        def seats(names, task):
            text = task.get("text") or ""
            r = self.candidates(self.models(cfg), cfg, features({"messages": [{"role": "user", "content": text}]}),
                                core.classify_rules(text), "fast", "auto", time.time())
            by = {c["m"]["id"]: c for c in r["passing"]}
            return {n: by[n] for n in names if n in by}
        return {"home": self.home, "env": self.env, "cfg": cfg, "mode": "proxy", "seats": seats}

    def discover(self, refresh_catalog=False):
        if self.discovering:
            return False, "discovery is already running"
        self.discovering = True
        try:
            ok, msg = run_discover(self.home, self.env, self.cfg(), refresh_catalog)
            self.last_discovery = {"ok": ok, "message": msg, "at": time.time()}
            self.cache = (None, None)
            return ok, msg
        finally:
            self.discovering = False


def _mtime(path):
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def _err_text(out):
    e = out.get("error") if isinstance(out, dict) else None
    return (e.get("message") if isinstance(e, dict) else str(e or ""))[:200] if e else ""


def load_admin_token(home):
    """The admin token in ~/.maestro/admin_token (mode 0600), created on first use."""
    path = os.path.join(home, ADMIN_TOKEN_FILE)
    try:
        with open(path, "r", encoding="utf-8") as f:
            tok = f.read().strip()
        if tok:
            try:
                os.chmod(path, 0o600)   # a file an older version or a hand edit left readable by others
            except OSError:
                pass
            return tok
    except OSError:
        pass
    core.ensure_home(home)
    tok = secrets.token_urlsafe(24)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(tok + "\n")
    os.chmod(path, 0o600)
    return tok


# ════════════════════════════════════════════════════════════════════════════
# 5. admin API
# ════════════════════════════════════════════════════════════════════════════

def quota_view(app, qp, model=None):
    s = Q.status(qp, model, app.home)
    wins = [{"name": n, "fraction": w.get("fraction"), "used_percent": w.get("used_percent"), "reset_at": w.get("reset_at")}
            for _, n, w in Q._live_windows(app.home, qp, model, time.time())] if model is None else []
    return {"state": s["state"], "headroom": s["headroom"], "reset_at": s["reset_at"], "cooldown_until": s["cooldown_until"],
            "reason": s["reason"], "windows": wins}


def provider_rows(app, cfg):
    """The providers table: the saved detection with key presence refreshed, quota attached. Never a key value."""
    pv = P.load_providers(app.home)
    if pv is None:
        pv = P.detect_all(app.env, app.home, cfg, list_live=False)
    rows = []
    for e in pv["providers"]:
        r = {k: e.get(k) for k in ("id", "name", "access", "style", "status", "reason", "key_source", "key_env", "base_url")}
        r["models"] = len(e.get("models") or [])
        if e["access"] == "api" and e["id"] in P.BY_ID:
            name, src, missing = P.key_status(P.BY_ID[e["id"]], app.env, app.home)
            r.update(key_source=src, key_env=name or P.BY_ID[e["id"]]["env"][0], key_missing=missing)
            if e.get("held"):  # shares its key with another provider, or is opt-in only: detection's reason stands
                pass
            elif name and not missing and r["status"] == "missing":
                r.update(status="detected", reason="key set; run discovery to list models")
            elif not name and r["status"] in ("connected", "detected"):
                r.update(status="missing", reason="no key (%s)" % P.BY_ID[e["id"]]["env"][0], models=0)
        r["quota"] = quota_view(app, e["id"]) if r["status"] != "missing" else None
        rows.append(r)
    return rows


MODEL_FIELDS = ("id", "provider", "model", "access", "style", "tools_ok", "vision", "context", "cost_in", "cost_out", "G", "jobs", "category",
                "serves", "basis", "benchmarks", "family", "release_date", "enabled", "loaded", "routable", "box")


def model_row(app, m):
    q = Q.status(m["qp"], m["model"], app.home)
    return dict({k: m.get(k) for k in MODEL_FIELDS}, cat_override=m["cat_override"], latency_ms=app.lat.get(m["id"]),
                quota={"state": q["state"], "headroom": q["headroom"], "reset_at": q["reset_at"], "reason": q["reason"]})


def state(app, base):
    from . import onboard
    cfg = app.cfg()
    ml = app.models(cfg)
    prov = provider_rows(app, cfg)
    models = [model_row(app, m) for m in ml["list"]]
    return {"version": core.VERSION, "now": time.time(), "server": {"host": app.host, "port": app.port, "base_url": base, "auth_required": bool(app.api_key)},
            "models_source": ml["raw"].get("source") or "models.json", "generated_at": ml["raw"].get("generated_at"),
            "catalog_as_of": ml["raw"].get("catalog_as_of"), "discovering": app.discovering, "last_discovery": app.last_discovery,
            "counts": {"providers": sum(1 for p in prov if p["status"] in ("connected", "detected")), "models": len(models),
                       "enabled": sum(1 for m in models if m["enabled"]), "local": sum(1 for m in models if m["access"] == "local")},
            "providers": prov, "models": models, "hardware": ml["raw"].get("hardware") or {}, "local_fit": ml["raw"].get("local_fit") or {},
            "config": {k: cfg.get(k) for k in ("classifier", "objective", "prefer_local", "local_allow_load", "max_tier", "min_confidence", "ladder", "models", "log_prompts",
                                              "session_affinity_header", "job_seats", "prefer") + tuple(S1_KEYS)},
            "default_ladder": {j: list(v) for j, v in core.DEFAULT_LADDER.items()}, "jobs": list(core.JOBS), "cells": list(core.CELLS),
            "floors": FLOORS, "stats": app.stats(), "decisions": app.decisions(30),
            "snippets": {"hermes": onboard.hermes_yaml(base + "/v1", bool(app.api_key)), "openai": onboard.openai_snippet(base + "/v1"),
                         "curl": onboard.curl_snippet(base + "/v1"), "skill": onboard.SKILL_COMMAND}}


def public_plan(p):
    """A plan as JSON for the UI's dry run: the decision and the candidates with their numbers."""
    d = {k: v for k, v in p.items() if k in ("requested", "kind", "job", "difficulty", "confidence", "classifier", "cell", "category", "needs",
                                             "chosen", "reason", "affinity", "decider", "p_success", "escalate_to", "shadow", "flags",
                                             "options", "posterior")}
    d["candidates"] = [{"id": c["m"]["id"], "category": c["m"]["cat_override"] or c["m"]["category"], "access": c["m"]["access"],
                        "strength": c["strength"], "cost": c["cost"], "headroom": c["headroom"], "latency_ms": c["latency"]} for c in p["order"][:12]]
    d["skipped"] = p["skipped"][:40]
    d["skipped_count"] = len(p["skipped"])
    return d


def _synth_body(b):
    msgs = [{"role": "user", "content": str(b.get("prompt") or "")}]
    if b.get("system"):
        msgs.insert(0, {"role": "system", "content": str(b["system"])})
    body = {"model": b.get("model") or "maestro/auto", "messages": msgs}
    if b.get("tools"):
        body["tools"] = [{"type": "function", "function": {"name": "tool", "parameters": {"type": "object", "properties": {}}}}]
    if b.get("image"):
        body["messages"][-1]["content"] = [{"type": "text", "text": msgs[-1]["content"]}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}}]
    return body


def _check_ladder(v):
    if not isinstance(v, dict) or any(j not in core.JOBS or not isinstance(r, list) or len(r) != 4 or any(c not in core.CELLS for c in r) for j, r in v.items()):
        raise ApiError(400, "ladder must map a job to four cells (%s)" % ", ".join(core.CELLS))
    return v


def _bool(v):
    if not isinstance(v, bool):
        raise ApiError(400, "expected true or false")
    return v


CONFIG_KEYS = {
    "objective": lambda v: v if v in ("balanced", "cost", "quality") else _raise("objective must be balanced, cost or quality"),
    "prefer_local": _bool, "log_prompts": _bool,
    "max_tier": lambda v: v if v in core.TIERS else _raise("max_tier must be one of %s" % ", ".join(core.TIERS)),
    "classifier": lambda v: v if v in ("auto", "rules", "jev") else _raise("classifier must be auto, rules or jev"),
    "min_confidence": lambda v: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1 else _raise("min_confidence is 0 to 1"),
    "ladder": _check_ladder,
    "models": lambda v: v if isinstance(v, dict) else _raise("models must be an object"),
    "job_seats": lambda v: v if isinstance(v, dict) else _raise("job_seats must be an object"),
    "prefer": lambda v: v if isinstance(v, list) else _raise("prefer must be a list"),
    "session_affinity_header": lambda v: v if isinstance(v, str) and len(v) < 100 else _raise("session_affinity_header must be a header name"),
}


def _num_in(key, lo, hi):
    return lambda v: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi else _raise("%s is %g to %g" % (key, lo, hi))


S1_KEYS = {   # spec §12.3
    "decider": lambda v: v if v in ("ladder", "utility") else _raise("decider must be ladder or utility"),
    "wtp_usd": lambda v: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else _raise("wtp_usd must be a number > 0"),
    "latency_usd_per_s": _num_in("latency_usd_per_s", 0, 1e3), "background_latency_usd_per_s": _num_in("background_latency_usd_per_s", 0, 1e3),
    "risk": _num_in("risk", 0, 3), "quota_reserve": _num_in("quota_reserve", 0, 0.5), "local_usd_per_hour": _num_in("local_usd_per_hour", 0, 1e6),
    "inline_access": lambda v: v if v in ("subscription", "api") else _raise("inline_access must be subscription or api"),
    "private": _bool, "log_decisions": _bool, "s1_shadow": _bool, "personalize": _bool, "systemone_cloud": _bool,
    "explore": _num_in("explore", 0, 0.1),
    "systemone_delegate": lambda v: v if isinstance(v, str) and 0 < len(v) < 300 else _raise("systemone_delegate must be a model id"),
}
CONFIG_KEYS.update(S1_KEYS)
CONFIG_KEYS["classifier"] = lambda v: v if v in ("auto", "rules", "s1", "jev") else _raise("classifier must be auto, rules, s1 or jev")


def _raise(msg):
    raise ApiError(400, msg)


def _raise_private():
    raise ApiError(503, "private mode: no local model can serve this request (private is on, or the prompt holds a key-like string)",
                   "server_error", "private_no_local")


def allowed_key_names(app, cfg):
    names = set()
    for s in P.REGISTRY:
        names.update(s["env"])
        names.update(n for g in s["need"] for n in g)
    names.update(ep["api_key_env"] for ep in core.local_endpoints(cfg, app.env) if ep.get("api_key_env"))
    return names


def admin(app, method, path, query, body, base):
    """Dispatch /api/*. Returns (status, JSON-able). Raises ApiError."""
    cfg = app.cfg()
    if method == "GET" and path == "/api/state":
        return 200, state(app, base)
    if method == "GET" and path == "/api/decisions":
        return 200, {"decisions": app.decisions(int((query.get("limit") or ["50"])[0]))}
    if method == "GET" and path == "/api/stats":
        return 200, app.stats()
    if method == "GET" and path == "/api/alpha":
        return 200, alpha_view(app, cfg)
    if method != "POST":
        raise ApiError(404, "no such endpoint", "not_found")
    if path == "/api/keys":
        pid, value = body.get("provider") or body.get("name"), body.get("value")
        try:
            name = P.key_env_name(pid) if isinstance(pid, str) else _raise("provider is required")
        except ValueError as e:
            raise ApiError(400, str(e))
        if name not in allowed_key_names(app, cfg):
            raise ApiError(400, "%s is not a key any provider reads" % name)
        if value in (None, ""):
            spec = P.BY_ID.get(pid)
            gone = P.unset_key(list(spec["env"]) if spec else [name], app.home)
            return 200, {"ok": True, "removed": gone}
        try:
            P.set_key(name, value, app.home)
        except ValueError as e:
            raise ApiError(400, str(e))
        return 200, {"ok": True, "name": name, "source": "env" if app.env.get(name) else "keys.env",
                     "note": "a %s set in the server's environment takes precedence over keys.env" % name if app.env.get(name) else ""}
    if path.startswith("/api/models/") or path == "/api/models":
        mid = urllib.parse.unquote(path[len("/api/models/"):]) if path.startswith("/api/models/") else body.get("id")
        if mid not in app.models(cfg)["by_id"]:
            raise ApiError(404, "unknown model %r" % (mid,), "not_found")
        ch = {k: body[k] for k in ("enabled", "category") if k in body}
        if "enabled" in ch:
            _bool(ch["enabled"])
        if ch.get("category") not in (None,) + CAT_NAMES:
            raise ApiError(400, "category must be one of %s, or null" % ", ".join(CAT_NAMES))

        def mut(raw):
            ov = raw.setdefault("models_overrides", {})
            cur = dict(ov.get(mid) or {})
            for k, v in ch.items():
                if v is None or (k == "enabled" and v is True):
                    cur.pop(k, None)
                else:
                    cur[k] = v
            if cur:
                ov[mid] = cur
            else:
                ov.pop(mid, None)
        app.write_config(mut)
        return 200, {"ok": True, "model": model_row(app, app.models()["by_id"][mid])}
    if path == "/api/config":
        bad = [k for k in body if k not in CONFIG_KEYS]
        if bad:
            raise ApiError(400, "cannot change %s here" % ", ".join(bad))
        vals = {k: (None if v is None else CONFIG_KEYS[k](v)) for k, v in body.items()}

        def mut2(raw):
            for k, v in vals.items():
                if v is None:
                    raw.pop(k, None)
                else:
                    raw[k] = v
        new = app.write_config(mut2)
        return 200, {"ok": True, "config": {k: new.get(k) for k in CONFIG_KEYS}}
    if path == "/api/discover":
        ok, msg = app.discover(bool(body.get("refresh_catalog")))
        return 200, {"ok": ok, "message": msg, "models": len(app.models()["list"])}
    if path == "/api/route":
        return 200, public_plan(app.plan(_synth_body(body), {}))
    if path == "/api/send":
        plan = app.plan(_synth_body(body), {})
        turn = app.run(plan, _synth_body(body))
        st, out = turn["status"], turn["out"]
        text = ((out.get("choices") or [{}])[0].get("message") or {}).get("content") if 200 <= st < 300 and isinstance(out, dict) else None
        app.finish(turn, st, out.get("usage") if isinstance(out, dict) else None, len(text or ""))
        return 200, dict(public_plan(turn), status=st, answer=text, usage=out.get("usage") if isinstance(out, dict) else None,
                         attempts=turn["attempts"], error=None if 200 <= st < 300 else _err_text(out), model=turn["model"]["id"])
    if path == "/api/test":
        return 200, test_provider(app, cfg, body.get("provider"))
    if path == "/api/feedback":
        return app.feedback(body)
    raise ApiError(404, "no such endpoint", "not_found")


def alpha_view(app, cfg, days=14):
    """GET /api/alpha: how often the two deciders agree (shadow mode), proxy and skill mode, the outcome counts and the personal fit."""
    from . import learn as L
    since, out = time.time() - days * 86400, {"decider": cfg.get("decider"), "s1_shadow": cfg.get("s1_shadow"), "days": days}
    try:
        with app.db_lock:
            rows = app.db.execute("select chosen, shadow from decisions where shadow is not null and status < 400 and ts >= ?", (since,)).fetchall()
            outs = dict(app.db.execute("select signal, count(*) from outcomes group by signal").fetchall())
    except sqlite3.DatabaseError:
        rows, outs = [], {}
    skill = [d for d in L.decisions(app.home) if d["ts"] >= since and d.get("shadow")]
    for o in L._jsonl(os.path.join(app.home, L.OUTCOMES)):
        outs[o.get("signal")] = outs.get(o.get("signal"), 0) + 1
    for k, pairs in (("proxy", rows), ("skill", [(d.get("chosen"), d["shadow"]) for d in skill])):
        n, agree = len(pairs), sum(1 for a, b in pairs if a == b)
        out[k] = {"n": n, "agree": agree, "rate": round(agree / float(n), 3) if n else None}
    pers = core._read_json(os.path.join(app.home, L.PERSONAL)) or {}
    out.update(outcomes=outs, personal={k: pers.get(k) for k in ("fitted_at", "n_outcomes")} if pers else None)
    return out


def test_provider(app, cfg, pid):
    """The UI's Test button: list models live (API), re-check the login (CLI) or probe the endpoint (local). Never returns a key."""
    try:
        if pid in P.BY_ID:
            ids = P.list_models(P.BY_ID[pid], app.env, app.home, cfg=cfg)
            return {"ok": True, "message": "key works: %d models" % len(ids)}
        if pid in P.CLI_BY_ID:
            import shutil
            e = P.detect_cli(P.CLI_BY_ID[pid], app.env, core._user_home(app.env), lambda x: shutil.which(x, path=app.env.get("PATH")))
            return {"ok": e["status"] == "connected", "message": e["reason"]}
        for ep in core.local_endpoints(cfg, app.env):
            if "local:" + ep["name"] == pid:
                res = core.probe_endpoint(ep, env=app.env)
                return {"ok": bool(res), "message": "%d models, %.0f ms" % (len(res["models"]), res["latency_ms"]) if res else "no answer at " + ep["url"]}
    except P.ProviderError as e:
        return {"ok": False, "message": str(e)}
    return {"ok": False, "message": "unknown provider %r" % (pid,)}


# ════════════════════════════════════════════════════════════════════════════
# 6. HTTP
# ════════════════════════════════════════════════════════════════════════════

def _rewrite(chunk, mid, acc):
    """An SSE item with `model` set to the real upstream id; counts the text that passes for the token estimate."""
    if not chunk.startswith(b"data:") or chunk[5:].strip() == b"[DONE]":
        return chunk
    try:
        obj = json.loads(chunk[5:])
    except ValueError:
        return chunk
    if not isinstance(obj, dict):
        return chunk
    if "model" in obj:
        obj["model"] = mid
    for ch in obj.get("choices") or []:
        d = ch.get("delta") or {}
        acc[0] += len(d.get("content") or "") + sum(len((t.get("function") or {}).get("arguments") or "") for t in d.get("tool_calls") or [])
    return b"data: " + json.dumps(obj, separators=(",", ":")).encode("utf-8") + chunk[len(chunk.rstrip()):]


def model_list(app, cfg):
    ml = app.models(cfg)
    ctxs = [m["context"] for m in ml["list"] if m["enabled"] and m.get("context")]
    meta_ctx = min(128000, max(ctxs)) if ctxs else 128000
    now = int(app.started)
    data = [{"id": "maestro/" + n, "object": "model", "created": now, "owned_by": "maestro", "context_length": meta_ctx} for n in META_MODELS]
    for m in ml["list"]:
        if m["enabled"] and m["routable"]:
            e = {"id": m["id"], "object": "model", "created": now, "owned_by": m["provider"]}
            if m.get("context"):
                e["context_length"] = m["context"]
            data.append(e)
    from . import systemone
    return {"object": "list", "data": data, "models": systemone.MODEL_LIST}


def make_handler(app):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "maestro/" + core.VERSION
        timeout = 900

        body_read = False         # True once read_body consumed this request's body
        sent_connection = False   # a Connection header has already been written for this response

        def log_message(self, *a):
            pass

        # ---- keep-alive hygiene: a request body this handler did not read would be parsed as the next request on the connection
        def parse_request(self):
            self.body_read = True   # no bookkeeping while the headers are parsed (the 100-continue reply, a malformed request)
            ok = super().parse_request()
            self.body_read = self.sent_connection = False
            return ok

        def carries_body(self):
            h = getattr(self, "headers", None)
            return h is not None and (h.get("Transfer-Encoding") is not None or (h.get("Content-Length") or "0").strip() not in ("", "0"))

        def send_header(self, keyword, value):
            if keyword.lower() == "connection":
                self.sent_connection = True
            super().send_header(keyword, value)

        def end_headers(self):
            if self.carries_body() and not self.body_read and not self.sent_connection:
                self.send_header("Connection", "close")   # also sets self.close_connection
            super().end_headers()

        # ---- plumbing
        def base(self):
            return "http://" + (self.headers.get("Host") or "%s:%d" % (app.host, app.port))

        def send_json(self, status, obj, extra=None):
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(raw)

        def send_error_json(self, e, extra=None):
            self.send_json(e.status, {"error": {"message": str(e), "type": e.kind, "code": e.code}}, extra)

        def read_body(self):
            """The request body as a JSON object ({} when empty). Every limit is checked before the bytes are read; a request
            refused here has not consumed its body, so end_headers closes the connection."""
            te, cl = self.headers.get("Transfer-Encoding"), self.headers.get_all("Content-Length") or []
            if te is not None and cl:
                raise ApiError(400, "send either Transfer-Encoding or Content-Length, not both")
            if len(cl) > 1:
                raise ApiError(400, "more than one Content-Length header")
            if te is not None:
                if te.strip().lower() != "chunked":
                    raise ApiError(400, "Transfer-Encoding must be chunked")
                parts, total = [], 0
                while True:
                    line = self.rfile.readline(1025)
                    size = line.split(b";", 1)[0].strip()
                    if len(line) > 1024 or not _HEX.fullmatch(size):
                        raise ApiError(400, "bad chunk size")
                    n = int(size, 16)
                    if n > MAX_BODY - total:
                        raise ApiError(413, "request too large")
                    if n == 0:
                        for _ in range(64):   # trailers, then the blank line that ends the body
                            if self.rfile.readline(1025).strip() == b"":
                                break
                        else:
                            raise ApiError(400, "too many trailer lines")
                        break
                    data = self.rfile.read(n)
                    if len(data) < n or self.rfile.readline(3) not in (b"\r\n", b"\n"):
                        raise ApiError(400, "the request body ended early or a chunk is malformed")
                    parts.append(data)
                    total += n
                raw = b"".join(parts)
            else:
                text = (cl[0] if cl else "0").strip()
                if not text.isascii() or not text.isdigit():
                    raise ApiError(400, "Content-Length must be a non-negative integer")
                if len(text) > 18 or int(text) > MAX_BODY:
                    raise ApiError(413, "request too large")
                n = int(text)
                raw = self.rfile.read(n) if n else b""
                if len(raw) < n:
                    raise ApiError(400, "the request body ended early")
            self.body_read = True
            if not raw.strip():
                return {}
            try:
                obj = json.loads(raw)
            except ValueError:
                raise ApiError(400, "the request body is not valid JSON")
            if not isinstance(obj, dict):
                raise ApiError(400, "the request body must be a JSON object")
            return obj

        def check_origin(self):
            """Block DNS rebinding (a foreign Host on a loopback server) and cross-site requests (a foreign Origin)."""
            host = self.headers.get("Host") or ""
            if app.loopback and host.rsplit(":", 1)[0] not in LOOPBACK_HOSTS and host not in LOOPBACK_HOSTS:
                raise ApiError(403, "unexpected Host header %r" % host[:60], "forbidden")
            origin = self.headers.get("Origin")
            if origin and origin.rstrip("/") not in ("http://" + host, "https://" + host) and origin not in (app.cfg().get("allowed_origins") or []):
                raise ApiError(403, "cross-origin request refused", "forbidden")

        def check_bearer(self):
            if app.api_key:
                got = (self.headers.get("Authorization") or "")
                got = got[7:].strip() if got.lower().startswith("bearer ") else self.headers.get("x-api-key") or ""
                if not hmac.compare_digest(got.encode("utf-8"), app.api_key.encode("utf-8")):
                    raise ApiError(401, "missing or wrong API key (send Authorization: Bearer <MAESTRO_API_KEY>)", "authentication_error", "invalid_api_key")

        def check_admin(self):
            tok = self.headers.get("X-Maestro-Token") or ""
            if not tok or not hmac.compare_digest(tok.encode("utf-8"), app.token.encode("utf-8")):
                raise ApiError(401, "the admin token is missing or wrong (see ~/.maestro/admin_token)", "authentication_error", "invalid_token")

        # ---- routes
        def handle_any(self):
            url = urllib.parse.urlsplit(self.path)
            path = url.path.rstrip("/") or "/"
            try:
                if path in ("/", "/index.html") and self.command in ("GET", "HEAD"):
                    return self.ui()
                if path == "/favicon.ico":
                    self.send_response(204)
                    self.send_header("Content-Length", "0")
                    return self.end_headers()
                if path == "/health":
                    return self.send_json(200, {"status": "ok", "version": core.VERSION, "uptime_s": round(time.time() - app.started),
                                                "models": len(app.models()["list"])})
                if path.startswith("/api/"):
                    self.check_origin()
                    self.check_admin()
                    body = self.read_body() if self.command == "POST" else {}
                    status, obj = admin(app, self.command, path, urllib.parse.parse_qs(url.query), body, self.base())
                    return self.send_json(status, obj)
                if path.startswith("/v1/"):
                    self.check_origin()
                    self.check_bearer()
                    if path == "/v1/models" and self.command == "GET":
                        return self.send_json(200, model_list(app, app.cfg()))
                    if path == "/v1/chat/completions" and self.command == "POST":
                        return self.chat(self.read_body())
                    if path == "/v1/systemone" and self.command == "POST":
                        from . import systemone
                        return self.send_json(*systemone.handle(self.read_body(), app.systemone_ctx()))
                raise ApiError(404, "not found: %s %s" % (self.command, path), "invalid_request_error", "not_found")
            except ApiError as e:
                self.send_error_json(e)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                self.close_connection = True
            except Exception as e:  # never leak a traceback or a value from the environment
                try:
                    self.send_error_json(ApiError(500, "internal error (%s)" % e.__class__.__name__, "server_error"))
                except OSError:
                    self.close_connection = True

        do_GET = do_POST = do_HEAD = handle_any

        def do_OPTIONS(self):   # no CORS: browsers on other origins get nothing
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def ui(self):
            try:
                with open(UI_FILE, "rb") as f:
                    raw = f.read()
            except OSError:
                raise ApiError(500, "the web UI file is missing from this install", "server_error")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                                                        "connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(raw)

        def chat(self, body):
            if self.headers.get("X-Maestro-Feedback"):   # `<id>=good, <id>=bad`: explicit outcomes for earlier decisions
                from . import learn
                for did, v in learn.parse_feedback_header(self.headers.get("X-Maestro-Feedback")):
                    try:
                        app.feedback({"id": did, "value": v})
                    except Exception:  # noqa: BLE001  feedback never breaks the chat request
                        pass
            plan = app.plan(body, {k.lower(): v for k, v in self.headers.items()})
            turn = app.run(plan, body)
            status, out, header = turn["status"], turn["out"], {"x-maestro-decision": turn["header"]}
            mid = turn["model"]["id"]
            if not 200 <= status < 300:
                app.finish(turn, status if status else 502)
                err = out if isinstance(out, dict) and out.get("error") else C.error_body("upstream failed", code=status)
                return self.send_json(status if 400 <= status < 600 else 502, err, header)
            if not isinstance(out, C.Stream):
                if isinstance(out, dict):
                    out["model"] = mid
                app.finish(turn, 200, out.get("usage"), len(str(((out.get("choices") or [{}])[0].get("message") or {}).get("content") or "")))
                return self.send_json(200, out, header)
            self.stream(turn, out, header, mid)

        def stream(self, turn, st, header, mid):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            for k, v in header.items():
                self.send_header(k, v)
            self.end_headers()
            self.close_connection = True
            acc, status, t0 = [0], 200, time.time()
            try:
                self.wfile.write(_rewrite(turn["first"], mid, acc))
                self.wfile.flush()
                for chunk in st:
                    self.wfile.write(_rewrite(chunk, mid, acc))
                    self.wfile.flush()
            except C.UpstreamError as e:   # bytes have gone out: surface the error in the stream and stop
                status = 502
                try:
                    self.wfile.write(b"data: " + json.dumps(C.error_body(str(e), "upstream_error", 502)).encode("utf-8") + b"\n\n"
                                     + b"data: [DONE]\n\n")
                    self.wfile.flush()
                except OSError:
                    pass
            except OSError:   # the client went away
                status = 499
            finally:
                st.close()
                app.finish(turn, status, st.usage, acc[0], (time.time() - turn["t0"]) * 1000.0)

    return H


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128   # the default of 5 drops connections when a client fans out a burst of requests
    allow_reuse_address = True


class Server6(Server):
    address_family = socket.AF_INET6


def make_server(host=DEFAULT_HOST, port=DEFAULT_PORT, home=None, env=None, discover_on_start=False, refresh_s=REFRESH_S, jev_fetch=None,
                learn_s=LEARN_S):
    """Build (but do not start) the server: returns (httpd, app). Raises ValueError when the bind rules are broken.
    Background threads (quota refresh, discovery on start) start here and stop with `app.stop`."""
    env = os.environ if env is None else env
    home = home or core.maestro_home(env)
    loopback = host in LOOPBACK_HOSTS
    if not loopback and not P.get_key(("MAESTRO_API_KEY",), env, home):
        raise ValueError("binding to %s exposes the router to the network: set MAESTRO_API_KEY (clients then send it as a Bearer token)" % host)
    app = App(home, env, jev_fetch)
    httpd = (Server6 if ":" in host else Server)((host, port), make_handler(app))
    app.host, app.port, app.loopback = host, httpd.server_address[1], loopback
    if refresh_s:
        threading.Thread(target=app.refresh_loop, args=(app.stop, refresh_s), daemon=True, name="maestro-quota").start()
    if learn_s:
        threading.Thread(target=app.learn_loop, args=(app.stop, learn_s), daemon=True, name="maestro-learn").start()
    mj = os.path.join(home, MODELS_FILE)
    if discover_on_start and (not os.path.exists(mj) or time.time() - os.path.getmtime(mj) > DISCOVERY_MAX_AGE_S):
        threading.Thread(target=app.discover, daemon=True, name="maestro-discover").start()
    return httpd, app


def cmd_serve(a):
    from . import onboard
    env = os.environ
    try:
        httpd, app = make_server(a.host, a.port, env=env, discover_on_start=not a.no_discover)
    except ValueError as e:
        print("maestro: %s" % e, file=sys.stderr)
        return 2
    except OSError as e:
        print("maestro: cannot listen on %s:%s (%s)" % (a.host, a.port, e.strerror or e), file=sys.stderr)
        return 1
    shown = "127.0.0.1" if a.host in ("0.0.0.0", "::") else a.host
    base = "http://%s:%d" % (shown, app.port)
    ml = app.models()
    print("Maestro %s is serving on %s (models: %d from %s)" % (core.VERSION, base, len(ml["list"]), ml["raw"].get("source", "models.json")))
    print("  OpenAI-compatible endpoint: %s/v1   (model: maestro/auto)" % base)
    if sys.stdout.isatty():
        print("  Web UI:                     %s/#token=%s" % (base, app.token))
    else:   # stdout goes to a log or a pipe: never write the admin token there (--open still hands it to the browser)
        print("  Web UI:                     %s/" % base)
        print("  admin token: cat %s" % os.path.join(app.home, ADMIN_TOKEN_FILE))
    if app.api_key:
        print("  API clients must send Authorization: Bearer <MAESTRO_API_KEY>")
    if not os.path.exists(os.path.join(app.home, MODELS_FILE)):
        print("  No models.json yet: discovery runs in the background (or run `maestro-router discover`).")
    print("\nHermes Agent (add to ~/.hermes/config.yaml, or run `maestro-router hermes --install`):\n")
    print(onboard.hermes_yaml(base + "/v1", bool(app.api_key)))
    sys.stdout.flush()   # redirected to a file or a pipe, stdout is block-buffered and the banner would never appear
    if a.open:
        import webbrowser
        webbrowser.open("%s/#token=%s" % (base, app.token))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    finally:
        app.stop.set()
        httpd.server_close()
    return 0
