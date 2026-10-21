"""Rate limits, cooldowns and subscription usage (Maestro v3).

State is per scope, "provider" or "provider/model", kept in memory and saved to ~/.maestro/quota.json:
    {"windows": {name: {limit, remaining, used_percent, fraction, reset_at}},   fraction = share still available
     "cooldown_until": ts, "fails": n, "exhausted": bool, "note": str, "updated": ts}
A window whose reset time has passed is ignored (it has refilled). Unknown is neutral: headroom() returns None.
Sources: response headers (record), OpenRouter and DeepSeek probes (probe_all), the newest Codex rollout file and the
Claude Code statusline cache (refresh_subscriptions). Nothing here calls an undocumented endpoint or reads a token.

For routing: is_available() (cooldown or exhausted = ineligible), headroom() (0..1 or None), penalty() (0..1; strong
under LOW_HEADROOM left). Keys are used only to authenticate the two probes against their own provider.
"""
from __future__ import annotations

import json
import os
import re
import stat
import threading
import time
from datetime import datetime, timezone

from . import providers as P
from . import router as core

LOW_HEADROOM = 0.10          # under 10% left (over 90% used): strong penalty
COOLDOWN_BASE_S = 60.0       # default cooldown after a 429 / 5xx without retry-after
COOLDOWN_MAX_S = 15 * 60.0   # exponential up to 15 minutes
RETRY_AFTER_MAX_S = 24 * 3600.0
QUOTA_FILE = "quota.json"
CLAUDE_LIMITS_FILE = "claude_limits.json"
STATUSLINE_FILE = "statusline.sh"
HIST_FILE = "quota_hist.json"   # Codex usage samples for decide's burn-rate forecast (Claude's come from claude_limits.log)
HIST_EVERY_S = 300.0            # at most one sample per window per 5 minutes
HIST_KEEP = 2016                # a week of 5-minute samples per window
WINDOW_LENGTH_S = {"five_hour": 18000, "seven_day": 7 * 86400}

_LOCK = threading.RLock()
_STATE = {}   # home -> {"scopes": {...}}


# ════════════════════════════════════════════════════════════════════════════
# 1. header parsers (pure)
# ════════════════════════════════════════════════════════════════════════════

_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


def parse_duration(v):
    """Seconds from "1s", "6m0s", "2m59.56s", "20ms", "1h2m" or a plain number of seconds; None if unparseable."""
    v = (v or "").strip()
    try:
        return float(v)
    except ValueError:
        pass
    parts = _DURATION_RE.findall(v)
    if not parts or "".join(a + b for a, b in parts) != v:
        return None
    return sum(float(n) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[u] for n, u in parts)


_ISO_RE = re.compile(r"^(\d{4})-(\d\d)-(\d\d)[T ](\d\d):(\d\d):(\d\d)(\.\d+)?\s*(Z|[+-]\d\d(?::?\d\d)?)?$")


def parse_timestamp(v):
    """Epoch seconds from an RFC 3339 string (any fraction length; Python 3.9's fromisoformat is stricter) or a number
    (seconds, or milliseconds when huge); None if unparseable."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v / 1000.0 if v > 1e11 else float(v)
    m = _ISO_RE.match(str(v).strip())
    if not m:
        return None
    y, mo, d, h, mi, sec = (int(x) for x in m.groups()[:6])
    off = 0
    if m.group(8) and m.group(8) != "Z":
        digits = m.group(8)[1:].replace(":", "")
        off = (1 if m.group(8)[0] == "+" else -1) * (int(digits[:2]) * 3600 + int(digits[2:] or 0) * 60)
    try:
        base = datetime(y, mo, d, h, mi, sec, tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None
    return base - off + (float(m.group(7)) if m.group(7) else 0.0)


def parse_retry_after(v, now=None):
    """Seconds to wait from a Retry-After header (delta-seconds or an HTTP date); None if absent or bad."""
    if not v:
        return None
    try:
        return max(0.0, float(v))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime
        return max(0.0, parsedate_to_datetime(v).timestamp() - (time.time() if now is None else now))
    except (TypeError, ValueError):
        return None


def _window(limit, remaining, reset_at):
    try:
        limit, remaining = float(limit), float(remaining)
    except (TypeError, ValueError):
        return None
    return {"limit": limit, "remaining": remaining, "fraction": max(0.0, min(1.0, remaining / limit)) if limit > 0 else None,
            "reset_at": reset_at}


_OPENAI_RE = re.compile(r"^x-ratelimit-(limit|remaining|reset)-(requests|tokens)(?:-(day|minute|hour))?$")


def parse_openai_headers(headers, now=None):
    """OpenAI, Groq, Cerebras and other compatible servers: x-ratelimit-{limit,remaining,reset}-{requests,tokens}
    with optional -day / -minute variants. Reset is a duration ("6m0s") or seconds. {window name: window}."""
    now = time.time() if now is None else now
    got = {}
    for k, v in headers.items():
        m = _OPENAI_RE.match(k.lower())
        if m:
            name = m.group(2) + ("-" + m.group(3) if m.group(3) else "")
            got.setdefault(name, {})[m.group(1)] = v
    out = {}
    for name, f in got.items():
        secs = parse_duration(f.get("reset"))
        w = _window(f.get("limit"), f.get("remaining"), None if secs is None else now + secs)
        if w:
            out[name] = w
    return out


def parse_anthropic_headers(headers, now=None):
    """anthropic-ratelimit-{requests,tokens,input-tokens,output-tokens}-{limit,remaining,reset}; reset is RFC 3339."""
    out = {}
    low = {k.lower(): v for k, v in headers.items()}
    for name in ("requests", "tokens", "input-tokens", "output-tokens"):
        p = "anthropic-ratelimit-%s-" % name
        if p + "limit" in low:
            w = _window(low.get(p + "limit"), low.get(p + "remaining"), parse_timestamp(low.get(p + "reset")))
            if w:
                out[name] = w
    return out


def parse_headers(headers, now=None):
    """(windows, retry_after seconds or None) from any vendor's response headers."""
    low = {k.lower(): v for k, v in (headers or {}).items()}
    windows = dict(parse_openai_headers(low, now))
    windows.update(parse_anthropic_headers(low, now))
    return windows, parse_retry_after(low.get("retry-after"), now)


# ════════════════════════════════════════════════════════════════════════════
# 2. state: memory + ~/.maestro/quota.json
# ════════════════════════════════════════════════════════════════════════════

def _scopes(home):
    with _LOCK:
        if home not in _STATE:
            data = core._read_json(os.path.join(home, QUOTA_FILE))
            _STATE[home] = {"scopes": data["scopes"] if isinstance(data, dict) and isinstance(data.get("scopes"), dict) else {}}
        return _STATE[home]["scopes"]


def _save(home):
    core._write_json(os.path.join(home, QUOTA_FILE), {"version": 1, "scopes": _scopes(home)})


def reset_state(home=None):
    """Forget the in-memory state (and, with a home, the copy of it on disk). For tests and `discover`."""
    with _LOCK:
        _STATE.pop(home or core.maestro_home(), None)
        if home:
            try:
                os.remove(os.path.join(home, QUOTA_FILE))
            except OSError:
                pass


def _scope(home, key):
    return _scopes(home).setdefault(key, {"windows": {}, "cooldown_until": 0.0, "fails": 0, "exhausted": False,
                                          "note": "", "updated": 0.0})


AUTH_COOLDOWN_S = 600.0      # a 401/403 will not fix itself in a minute: leave the provider alone for 10 minutes


def _cool(sc, now, wait=None, why=""):
    """Start (or extend) a cooldown on a scope: `wait` seconds, else 60 s doubling with each failure up to 15 minutes."""
    sc["fails"] = int(sc.get("fails") or 0) + 1
    if wait is None:
        wait = min(COOLDOWN_BASE_S * 2 ** (sc["fails"] - 1), COOLDOWN_MAX_S)
    sc["cooldown_until"] = now + min(max(wait, 1.0), RETRY_AFTER_MAX_S)
    sc["cool_reason"] = why
    return sc["cooldown_until"]


def record(provider, model, status, headers, home=None, now=None):
    """Note the outcome of an upstream call. 2xx clears the failure counts; 429, 5xx and status 0 (no answer) start a cooldown
    on the model (retry-after, else the earliest reset of an emptied window, else 60 s doubling up to 15 min). A 401/403
    (`auth failed`, 10 min), status 0 and 5xx also put the whole provider on cooldown, and so does a 429 whose Anthropic
    rate-limit headers show an emptied window (those limits are per organization). Rate-limit headers update the windows.
    Returns the cooldown end (epoch) or None."""
    home = home or core.maestro_home()
    now = time.time() if now is None else now
    windows, retry = parse_headers(headers, now)
    key = "%s/%s" % (provider, model) if model else provider
    with _LOCK:
        sc = _scope(home, key)
        psc = _scope(home, provider) if model else sc
        sc["windows"].update(windows)
        sc["updated"] = now
        until = None
        if status in (401, 403):
            until = _cool(psc, now, AUTH_COOLDOWN_S, "auth failed")
        elif status == 429 or status >= 500 or status == 0:
            wait = retry
            empty = [w["reset_at"] - now for w in windows.values() if w["remaining"] <= 0 and w["reset_at"]]
            if wait is None:
                wait = min(empty) if empty else None
            until = _cool(sc, now, wait, "rate limited" if status == 429 else "no answer" if status == 0 else "server error")
            org = status == 429 and any(w["remaining"] <= 0 for w in parse_anthropic_headers(headers, now).values())
            if psc is not sc and (status != 429 or org):
                _cool(psc, now, wait if status == 429 else None, sc["cool_reason"])
        elif 200 <= status < 300:
            for x in (sc, psc):
                x["fails"], x["cooldown_until"] = 0, 0.0
        _save(home)
        return until


# ════════════════════════════════════════════════════════════════════════════
# 3. reading state for routing
# ════════════════════════════════════════════════════════════════════════════

def _keys(provider, model):
    return (provider, "%s/%s" % (provider, model)) if model else (provider,)


def _live_windows(home, provider, model, now):
    """The unexpired windows that apply to provider/model, as [(scope key, name, window)]."""
    sc = _scopes(home)
    out = []
    for key in _keys(provider, model):
        for name, w in ((sc.get(key) or {}).get("windows") or {}).items():
            if not w.get("reset_at") or w["reset_at"] > now:
                out.append((key, name, w))
    return out


def headroom(provider, model=None, home=None, now=None):
    """The smallest share still available (0..1) over the live windows of this provider and model, or None if unknown."""
    home = home or core.maestro_home()
    now = time.time() if now is None else now
    with _LOCK:
        fr = [w["fraction"] for _, _, w in _live_windows(home, provider, model, now) if w.get("fraction") is not None]
    return min(fr) if fr else None


def status(provider, model=None, home=None, now=None):
    """{"state": ok | low | cooldown | exhausted | unknown, "headroom", "reset_at", "cooldown_until", "reason"} for the UI and routing."""
    home = home or core.maestro_home()
    now = time.time() if now is None else now
    with _LOCK:
        sc = _scopes(home)
        flagged = [(sc.get(k) or {}) for k in _keys(provider, model)]
        cool = max(f.get("cooldown_until") or 0.0 for f in flagged)
        live = _live_windows(home, provider, model, now)
    fr = headroom(provider, model, home, now)
    empty = [w for _, _, w in live if w.get("remaining") is not None and w["remaining"] <= 0 or w.get("fraction") == 0.0]
    resets = [w["reset_at"] for _, _, w in live if w.get("reset_at")]
    out = {"state": "unknown", "headroom": fr, "reset_at": min(resets) if resets else None,
           "cooldown_until": cool if cool > now else None, "reason": ""}
    if cool > now:
        why = next((f.get("cool_reason") for f in flagged if (f.get("cooldown_until") or 0.0) == cool and f.get("cool_reason")), "")
        out.update(state="cooldown", reason="cooling down for %.0f s%s" % (cool - now, " (%s)" % why if why else ""))
    elif any(f.get("exhausted") for f in flagged) or empty:
        out.update(state="exhausted", reason=next((f.get("note") for f in flagged if f.get("exhausted") and f.get("note")), "limit reached"))
    elif fr is not None:
        out["state"] = "low" if fr < LOW_HEADROOM else "ok"
        out["reason"] = "%.0f%% left" % (fr * 100)
    return out


def is_available(provider, model=None, home=None, now=None):
    """(True, "") or (False, reason): on cooldown or exhausted means ineligible; unknown and low are eligible."""
    s = status(provider, model, home, now)
    return (s["state"] not in ("cooldown", "exhausted"), s["reason"] if s["state"] in ("cooldown", "exhausted") else "")


def penalty(provider, model=None, home=None, now=None):
    """Routing penalty 0..1: 0 when unknown or comfortable, 0.8 when under LOW_HEADROOM is left, 1.0 when ineligible."""
    s = status(provider, model, home, now)
    return 1.0 if s["state"] in ("cooldown", "exhausted") else 0.8 if s["state"] == "low" else 0.0


# ════════════════════════════════════════════════════════════════════════════
# 4. probes: OpenRouter and DeepSeek
# ════════════════════════════════════════════════════════════════════════════

def _get_json(spec, url, key, timeout):
    status, _, data = P.http_request("GET", url, P.auth_headers(spec, key), None, timeout, P.is_direct(spec, url))
    if status != 200:
        raise P.ProviderError("HTTP %d" % status)
    return json.loads(data)


def probe_openrouter(spec, key, timeout=8.0):
    """GET {base}/key -> the credit window ({limit, remaining}) when a limit is set; unlimited keys stay unknown."""
    d = (_get_json(spec, spec["base_url"] + "/key", key, timeout) or {}).get("data") or {}
    limit, left = d.get("limit"), d.get("limit_remaining")
    w = _window(limit, left, None) if limit is not None and left is not None else None
    return {"windows": {"credit": w} if w else {}, "exhausted": w is not None and w["remaining"] <= 0,
            "note": "free tier" if d.get("is_free_tier") else ""}


def probe_deepseek(spec, key, timeout=8.0):
    """GET /user/balance -> is_available (False = out of balance = exhausted) and the balance in the note."""
    origin = re.match(r"^(\w+://[^/]+)", spec["base_url"]).group(1)
    d = _get_json(spec, origin + "/user/balance", key, timeout) or {}
    bal = ", ".join("%s %s" % (b.get("total_balance"), b.get("currency")) for b in d.get("balance_infos") or [])
    return {"windows": {}, "exhausted": d.get("is_available") is False, "note": "out of balance" if d.get("is_available") is False
            else "balance " + bal if bal else ""}


PROBES = {"openrouter": probe_openrouter, "deepseek": probe_deepseek}


def probe_all(env=None, home=None, timeout=8.0, now=None):
    """Run the probes for every provider that has a key. Returns {provider id: "ok" | error message}."""
    env = os.environ if env is None else env
    home = home or core.maestro_home(env)
    now = time.time() if now is None else now
    cfg = core.load_config(home)
    out = {}
    for pid, fn in PROBES.items():
        spec = dict(P.BY_ID[pid], base_url=P.base_url(P.BY_ID[pid], env, home, cfg))
        key = P.get_key(spec["env"], env, home)
        if not key:
            continue
        try:
            res = fn(spec, key, timeout)
        except (OSError, ValueError, P.ProviderError) as e:
            out[pid] = P.scrub("%s: %s" % (e.__class__.__name__, e), key)
            continue
        with _LOCK:
            sc = _scope(home, pid)
            sc["windows"] = res["windows"]
            sc["exhausted"], sc["note"], sc["updated"] = res["exhausted"], res["note"], now
            _save(home)
        out[pid] = "ok"
    return out


# ════════════════════════════════════════════════════════════════════════════
# 5. subscriptions: Codex rollouts, Claude Code statusline cache
# ════════════════════════════════════════════════════════════════════════════

def _sorted_desc(path):
    try:
        return sorted(os.listdir(path), reverse=True)
    except OSError:
        return []


def _rollouts(root, limit=5):
    """The newest rollout-*.jsonl files under ~/.codex/sessions/YYYY/MM/DD/, newest first."""
    out = []
    for y in _sorted_desc(root):
        for m in _sorted_desc(os.path.join(root, y)):
            for d in _sorted_desc(os.path.join(root, y, m)):
                day = os.path.join(root, y, m, d)
                out += [os.path.join(day, f) for f in _sorted_desc(day) if f.startswith("rollout-") and f.endswith(".jsonl")]
                if len(out) >= limit:
                    return out[:limit]
    return out


def _rate_limits(obj):
    for o in (obj, obj.get("payload"), obj.get("msg")):
        if isinstance(o, dict) and isinstance(o.get("rate_limits"), dict):
            return o["rate_limits"]
    return None


def _codex_window(w, ts, now):
    """A Codex rate-limit window as {used_percent, window_minutes, reset_at, fraction}. Its reset is `resets_at` (epoch) or
    `resets_in_seconds` after the event's timestamp; a window that has reset counts as 0% used."""
    if not isinstance(w, dict) or not isinstance(w.get("used_percent"), (int, float)):
        return None
    reset = parse_timestamp(w.get("resets_at")) if w.get("resets_at") is not None else \
        ts + w["resets_in_seconds"] if isinstance(w.get("resets_in_seconds"), (int, float)) else \
        ts + w["window_minutes"] * 60 if isinstance(w.get("window_minutes"), (int, float)) else None
    used = 0.0 if reset is not None and reset <= now else float(w["used_percent"])
    return {"used_percent": used, "window_minutes": w.get("window_minutes"), "reset_at": reset,
            "fraction": max(0.0, min(1.0, 1 - used / 100.0))}


def codex_limits(user_home=None, now=None, env=None):
    """{"plan_type", "primary", "secondary", "as_of"} from the newest `token_count` event with rate_limits in the newest Codex
    rollout files, or None. Only the tail of each file is read."""
    now = time.time() if now is None else now
    root = os.path.join(user_home or core._user_home(env), ".codex", "sessions")
    for path in _rollouts(root):
        try:
            size = os.path.getsize(path)
            with open(path, "rb") as f:
                f.seek(max(0, size - 512 * 1024))
                lines = f.read().decode("utf-8", "replace").splitlines()
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        for line in reversed(lines):
            if "rate_limits" not in line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            rl = _rate_limits(ev) if isinstance(ev, dict) else None
            if not rl:
                continue
            ts = parse_timestamp(ev.get("timestamp")) or mtime
            win = {n: _codex_window(rl.get(n), ts, now) for n in ("primary", "secondary")}
            if any(win.values()):
                return dict(win, plan_type=rl.get("plan_type") or (ev.get("payload") or {}).get("plan_type"), as_of=ts)
    return None


def claude_limits(home=None, now=None):
    """{"five_hour": window, "seven_day": window, "updated_at"} from the statusline cache (~/.maestro/claude_limits.json), or None.
    A window that has reset counts as 0% used."""
    now = time.time() if now is None else now
    data = core._read_json(os.path.join(home or core.maestro_home(), CLAUDE_LIMITS_FILE))
    rl = (data or {}).get("rate_limits") if isinstance(data, dict) else None
    if not isinstance(rl, dict):
        return None
    out = {"updated_at": data.get("updated_at")}
    for name in ("five_hour", "seven_day"):
        w = rl.get(name)
        if isinstance(w, dict) and isinstance(w.get("used_percentage"), (int, float)):
            reset = parse_timestamp(w.get("resets_at")) if w.get("resets_at") is not None else None
            used = 0.0 if reset is not None and reset <= now else float(w["used_percentage"])
            out[name] = {"used_percent": used, "reset_at": reset, "fraction": max(0.0, min(1.0, 1 - used / 100.0))}
    return out if len(out) > 1 else None


def refresh_subscriptions(home=None, user_home=None, now=None, env=None):
    """Load Codex and Claude Code usage into the state (scopes "codex" and "claude") and append Codex samples to quota_hist.json.
    Gemini has no source: it stays unknown. Returns {"codex": limits or None, "claude": limits or None}."""
    home = home or core.maestro_home(env)
    now = time.time() if now is None else now
    found = {"codex": codex_limits(user_home, now, env), "claude": claude_limits(home, now)}
    with _LOCK:
        for pid, lim in found.items():
            sc = _scope(home, pid)
            sc["windows"] = {n: {"limit": 100.0, "remaining": 100.0 * w["fraction"], "used_percent": w["used_percent"],
                                 "fraction": w["fraction"], "reset_at": w["reset_at"], "length_s": _length_s(n, w)}
                             for n, w in (lim or {}).items() if isinstance(w, dict) and "fraction" in w}
            sc["updated"] = now
        _save(home)
        _append_history(home, found["codex"], now)
    return found


def _length_s(name, w):
    m = w.get("window_minutes")
    return m * 60.0 if isinstance(m, (int, float)) and not isinstance(m, bool) and m > 0 else WINDOW_LENGTH_S.get(name)


def _append_history(home, codex, now):
    """quota_hist.json: {"version": 1, "windows": {"codex/primary": [[t, used 0..1], ...]}, "current": {"codex/primary": {used,
    reset_at, length_s, t}}}. A sample is taken at most every HIST_EVERY_S per window and HIST_KEEP are kept. decide.read_windows
    reads it as a plain file (route never imports this module)."""
    wins = {n: w for n, w in (codex or {}).items() if n in ("primary", "secondary") and isinstance(w, dict)}
    if not wins:
        return
    path = os.path.join(home, HIST_FILE)
    data = core._read_json(path)
    data = data if isinstance(data, dict) and isinstance(data.get("windows"), dict) and isinstance(data.get("current"), dict) else {}
    hist, cur = data.get("windows") or {}, data.get("current") or {}
    for n, w in sorted(wins.items()):
        key, used = "codex/" + n, round(w["used_percent"] / 100.0, 4)
        cur[key] = {"used": used, "reset_at": w["reset_at"], "length_s": _length_s(n, w), "t": now}
        s = [x for x in hist.get(key) or [] if isinstance(x, list) and len(x) == 2 and all(isinstance(v, (int, float)) for v in x)]
        if not s or now - s[-1][0] >= HIST_EVERY_S:
            s.append([now, used])
        hist[key] = s[-HIST_KEEP:]
    core._write_json(path, {"version": 1, "windows": hist, "current": cur}, indent=None)


def window_view(provider, model=None, home=None, now=None):
    """[{name, used (0..1), reset_at, length_s}] of the live windows of a provider (and model) as plain dicts, for decide."""
    home = home or core.maestro_home()
    now = time.time() if now is None else now
    with _LOCK:
        live = _live_windows(home, provider, model, now)
    out = []
    for _, name, w in live:
        used = w["used_percent"] / 100.0 if isinstance(w.get("used_percent"), (int, float)) else \
            1.0 - w["fraction"] if isinstance(w.get("fraction"), (int, float)) else None
        out.append({"name": name, "used": used, "reset_at": w.get("reset_at"), "length_s": w.get("length_s") or WINDOW_LENGTH_S.get(name)})
    return out


def refresh(env=None, home=None, user_home=None, timeout=8.0, now=None):
    """Everything periodic: the provider probes and the subscription readers. Call at `discover` and every ~10 minutes."""
    return {"probes": probe_all(env, home, timeout, now), "subscriptions": refresh_subscriptions(home, user_home, now, env)}


# ════════════════════════════════════════════════════════════════════════════
# 6. Claude Code statusline hook (`maestro-router install --statusline`)
# ════════════════════════════════════════════════════════════════════════════

STATUSLINE_SCRIPT = r'''#!/bin/sh
# Written by `maestro-router install --statusline`. Claude Code runs this as its statusLine command and pipes a JSON document
# to stdin; the rate_limits part (five_hour / seven_day used_percentage and resets_at) is cached in
# ~/.maestro/claude_limits.json for Maestro's routing, and each change of a percentage is appended to
# ~/.maestro/claude_limits.log (kept under 4,000 lines) for its burn-rate forecast. It prints a short usage line.
python3 -c '
import json, os, sys, time
try:
    d = json.load(sys.stdin)
except ValueError:
    sys.exit(0)
r = d.get("rate_limits") if isinstance(d, dict) else None
if isinstance(r, dict):
    home = os.environ.get("MAESTRO_HOME") or os.path.join(os.path.expanduser("~"), ".maestro")
    pct = lambda x: {n: w.get("used_percentage") for n, w in x.items() if n.startswith(("five_hour", "seven_day")) and isinstance(w, dict)}
    try:
        os.umask(0o077)   # every file this writes is owner-only
        os.makedirs(home, mode=0o700, exist_ok=True)
        if os.stat(home).st_uid == os.getuid():
            os.chmod(home, 0o700)   # tighten a home an older Maestro left 0755
        p = os.path.join(home, "claude_limits.json")
        for q in (p, os.path.join(home, "claude_limits.log")):
            if os.path.exists(q):
                os.chmod(q, 0o600)
        try:
            with open(p) as f:
                old = pct(json.load(f)["rate_limits"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            old = None
        with open(p + ".tmp", "w") as f:
            json.dump({"rate_limits": r, "updated_at": time.time()}, f)
        os.replace(p + ".tmp", p)
        if pct(r) and pct(r) != old:
            lp = os.path.join(home, "claude_limits.log")
            with open(lp, "a") as f:
                f.write(json.dumps(dict(pct(r), t=round(time.time(), 1))) + "\n")
            if os.path.getsize(lp) > 4000 * 30:
                with open(lp) as f:
                    lines = f.readlines()
                if len(lines) > 4000:
                    with open(lp + ".tmp", "w") as f:
                        f.writelines(lines[-2000:])
                    os.replace(lp + ".tmp", lp)
    except OSError:
        pass
    print(" ".join("%s %.0f%%" % (k, (r.get(n) or {}).get("used_percentage") or 0)
                   for k, n in (("5h", "five_hour"), ("7d", "seven_day")) if isinstance(r.get(n), dict)))
'
'''


def install_statusline(home=None):
    """Write ~/.maestro/statusline.sh (mode 700: only its owner runs it). Returns (path, the settings.json snippet to add by hand). Never edits
    Claude Code's settings."""
    home = home or core.maestro_home()
    path = os.path.join(home, STATUSLINE_FILE)
    core.ensure_home(home)
    with open(path, "w", encoding="utf-8") as f:
        f.write(STATUSLINE_SCRIPT)
    os.chmod(path, stat.S_IRWXU)
    return path, json.dumps({"statusLine": {"type": "command", "command": path}}, indent=2)
