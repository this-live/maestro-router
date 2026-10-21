#!/usr/bin/env python3
"""Test Maestro on prompts you actually typed into your coding harnesses. Local only: nothing is sent anywhere.

    python3 evals/mine.py harvest [--home DIR] [--harness all|claude-code|codex|gemini-cli|opencode|hermes]
                                  [--since 90d] [--limit 80] [--project P] [--exclude-project P]
                                  [--include-automated] [--include-subagents] [--from-file FILE] [--out FILE]
    python3 evals/mine.py route   [--in prompts.jsonl] [--harness claude-code] [--classifier rules|jev] [--out routed.csv]
    python3 evals/mine.py score   [--csv routed.csv] [--prompts prompts.jsonl] [--export labeled.jsonl]

harvest reads the prompt history your harnesses keep on disk (best effort: an unknown format is skipped with a
one-line note), drops noise, dedupes, redacts likely secrets, samples, and writes JSONL rows
{"id", "source", "prompt", "ts", "project"} (default ~/.maestro/mine/prompts.jsonl, mode 0600).
`--home DIR` reads a copy of another machine's home directory; `--from-file FILE` reads prompts separated by
blank lines (Cursor, aider, anything else pasted by hand).

route runs Maestro's router (rules classifier, your real inventory and config) over them and writes a CSV.
Fill the your_* columns ONLY where Maestro is wrong (blank = agree), then run score.

score reports job / category accuracy, +-1 category, under-routing and confusions. --export writes the corrected
labels in the evals/prompts.jsonl format (prompt, job, difficulty). Your prompts are private: review that file
before you share it. Nothing here uploads it.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "skills", "maestro", "scripts"))
from maestro import router as m  # noqa: E402

HARNESSES = ("claude-code", "codex", "gemini-cli", "opencode")   # routing harnesses
SOURCES = HARNESSES + ("hermes",)                                 # what harvest reads (hermes routes as claude-code)
MAX_KEEP = 4000          # characters of each prompt written out (the head)
MAX_READ = 20000         # characters held in memory per prompt before redaction and truncation
SEED = 20260101
CSV_COLS = ("id", "source", "prompt", "maestro_job", "maestro_difficulty", "maestro_category", "maestro_model",
            "your_job", "your_difficulty", "your_category", "note")

# ── redaction ────────────────────────────────────────────────────────────────

_REDACT = [
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)", re.S),
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bsk-(?:proj-|live-|test-)?[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[bapr]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\beyJ[\w-]{5,}\.[\w-]{5,}\.[\w-]{5,}"),
]
_BEARER = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{8,}")
_ASSIGN = re.compile(r"""(?ix)\b((?:[\w-]*(?:password|passwd|pwd|api[_-]?key|apikey|secret|token|access[_-]?key))\s*=\s*)(?!\[REDACTED\])
                         (?:"[^"\n]{4,}"|'[^'\n]{4,}'|[^\s"'&,;]{4,})""")
_QUOTED = re.compile(r"""(?i)("(?:password|passwd|api[_-]?key|apikey|secret|[\w-]*token)"\s*:\s*)"[^"\n]{4,}\"""")
_HEX = re.compile(r"(?<![A-Za-z0-9])[0-9a-fA-F]{32,}(?![A-Za-z0-9])")
_B64 = re.compile(r"(?<![A-Za-z0-9+_=-])[A-Za-z0-9+_=-]{32,}(?![A-Za-z0-9+_=-])")


def _b64_sub(mo, hits):
    s = mo.group(0)
    if re.search(r"\d", s) and re.search(r"[a-z]", s) and re.search(r"[A-Z]", s):
        hits.append(s)
        return "[REDACTED]"
    return s


def redact(text):
    """Replace likely secrets with [REDACTED]. Returns (text, number of replacements)."""
    n = 0
    for rx in _REDACT:
        text, k = rx.subn("[REDACTED]", text)
        n += k
    text, k = _BEARER.subn(r"\1[REDACTED]", text)
    n += k
    text, k = _ASSIGN.subn(r"\1[REDACTED]", text)
    n += k
    text, k = _QUOTED.subn(r'\1"[REDACTED]"', text)
    n += k
    text, k = _HEX.subn("[REDACTED]", text)
    n += k
    hits = []
    text = _B64.sub(lambda mo: _b64_sub(mo, hits), text)
    return text, n + len(hits)


# ── filters ──────────────────────────────────────────────────────────────────

ACKS = frozenset(x.strip() for x in (
    "yes|y|yep|yeah|yup|no|n|nope|ok|okay|k|sure|continue|proceed|go on|go ahead|go for it|do it|yes please"
    "|please continue|keep going|next|retry|try again|thanks|thank you|thx|ty|looks good|lgtm|sounds good|great"
    "|perfect|nice|cool|done|yes go ahead|yes do it|ok go ahead|ok do it|looks good to me|sounds good to me"
    "|thank you so much|yes please proceed|keep going please|continue please|thanks a lot").split("|"))
_SLASH = re.compile(r"^/[\w:.-]+(?:\s|$)")     # /help, /compact foo; not /home/user/x.py
_PLACEHOLDER = re.compile(r"\[(?:Pasted text|Image|\.\.\.Truncated)[^\]]*\]")
CLAUDE_SKIP = ("Base directory for this skill", "<command-name>", "<command-message>", "<local-command-stdout>",
               "Caveat:", "<system-reminder>", "[Request interrupted", "This session is being continued")
CODEX_SKIP = ("<environment_context>", "<user_instructions>", "<permissions", "# AGENTS.md")


def norm_key(text):
    return re.sub(r"\s+", " ", text.lower()).strip().rstrip(".!? ")


def is_ack(text):
    return re.sub(r"[^\w\s]", "", norm_key(text)).strip() in ACKS


def clean(text):
    """Strip placeholders like [Pasted text #1 +20 lines] and surrounding whitespace."""
    return _PLACEHOLDER.sub("", text).strip()[:MAX_READ]


def parse_ts(v):
    """Epoch seconds from an epoch (s or ms), a numeric string or an ISO-8601 string; None if unknown."""
    try:
        if isinstance(v, bool) or v is None or v == "":
            return None
        if isinstance(v, str):
            try:
                v = float(v)
            except ValueError:
                s = re.sub(r"\.\d+", "", v.strip()).replace("Z", "+00:00")
                d = datetime.fromisoformat(s)
                return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp()
        v = float(v)
        return v / 1000.0 if v > 1e11 else v
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if ts else ""


def parse_since(s):
    """'90d' -> seconds; 'all' -> None."""
    if s in ("all", "0", ""):
        return None
    mo = re.fullmatch(r"(\d+)\s*([hdwmy])", s.strip().lower())
    if not mo:
        raise argparse.ArgumentTypeError("--since must look like 90d, 12w, 6m, 1y or all")
    return int(mo.group(1)) * {"h": 3600, "d": 86400, "w": 7 * 86400, "m": 30 * 86400, "y": 365 * 86400}[mo.group(2)]


def _project(path):
    return os.path.basename(str(path or "").rstrip("/\\"))


# ── readers: each returns rows {source, prompt, ts, project}; unreadable lines and files are skipped ──

def _lines(path):
    """Yield (parsed dict) for each JSON line of a file; bad lines and unreadable files are skipped."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("{"):
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(d, dict):
                        yield d
    except OSError:
        return


def _read_json(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _claude_text(entry):
    """User text of a Claude Code transcript entry, or None if it is not a typed prompt."""
    if entry.get("type") != "user" or entry.get("isMeta") or entry.get("isCompactSummary"):
        return None
    msg = entry.get("message")
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            return None
        content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                            and isinstance(b.get("text"), str))
    if not isinstance(content, str):
        return None
    text = content.strip()
    return None if not text or text.startswith(CLAUDE_SKIP) else text


def _claude_origin(entry):
    """'typed' if a person typed this message, 'automated' if a program sent it (claude -p, the Agent SDK,
    an orchestrator dispatching sessions). Claude Code records this per message as turnOrigin; older
    transcripts without it fall back to the entrypoint (sdk-* = automated)."""
    origin = entry.get("turnOrigin")
    if isinstance(origin, str) and origin:
        return "typed" if origin == "human" else "automated"
    return "automated" if str(entry.get("entrypoint") or "").startswith("sdk") else "typed"


def read_claude(home, subagents=False):
    base = os.path.join(home, ".claude")
    rows, notes = [], []
    files = sorted(glob.glob(os.path.join(base, "projects", "*", "*.jsonl")))
    for path in files:
        for e in _lines(path):
            if e.get("isSidechain"):
                continue
            text = _claude_text(e)
            if text:
                rows.append({"source": "claude-code", "prompt": text, "ts": parse_ts(e.get("timestamp")),
                             "project": _project(e.get("cwd")) or _project(os.path.dirname(path)),
                             "origin": _claude_origin(e)})
    if subagents:
        for path in sorted(glob.glob(os.path.join(base, "projects", "*", "*", "subagents", "**", "*.jsonl"), recursive=True)):
            for e in _lines(path):
                text = _claude_text(e)
                if text:   # the first typed-looking user message is the delegated subtask
                    rows.append({"source": "claude-code:subagent", "prompt": text, "ts": parse_ts(e.get("timestamp")),
                                 "project": _project(e.get("cwd")), "origin": "subagent"})
                    break
    hist = os.path.join(base, "history.jsonl")
    if os.path.exists(hist):
        for e in _lines(hist):
            text = e.get("display")
            if isinstance(text, str) and text.strip() and not text.startswith(CLAUDE_SKIP):
                rows.append({"source": "claude-code", "prompt": text.strip(), "ts": parse_ts(e.get("timestamp")),
                             "project": _project(e.get("project")), "origin": "typed"})  # interactive prompt history
    if not files and not os.path.exists(hist):
        notes.append("claude-code: no transcripts or history under %s" % base)
    return rows, notes


def read_codex(home):
    base = os.path.join(home, ".codex")
    rows, notes = [], []
    hist = os.path.join(base, "history.jsonl")
    for e in _lines(hist):
        text = e.get("text")
        if isinstance(text, str) and text.strip():
            rows.append({"source": "codex", "prompt": text.strip(), "ts": parse_ts(e.get("ts")), "project": ""})
    if not rows:
        for path in sorted(glob.glob(os.path.join(base, "sessions", "**", "rollout-*.jsonl"), recursive=True)):
            cwd, origin = "", "typed"
            for e in _lines(path):
                p = e.get("payload") if isinstance(e.get("payload"), dict) else e
                if e.get("type") == "session_meta" and isinstance(p, dict):
                    cwd = _project(p.get("cwd"))
                    if "exec" in str(p.get("originator") or ""):   # codex exec: scripted, not typed
                        origin = "automated"
                if p.get("type") != "message" or p.get("role") != "user" or not isinstance(p.get("content"), list):
                    continue
                text = "\n".join(b.get("text", "") for b in p["content"]
                                 if isinstance(b, dict) and b.get("type") == "input_text").strip()
                if text and not text.startswith(CODEX_SKIP):
                    rows.append({"source": "codex", "prompt": text, "ts": parse_ts(e.get("timestamp")), "project": cwd,
                                 "origin": origin})
    if not rows:
        notes.append("codex: no history.jsonl or rollout sessions under %s" % base)
    return rows, notes


def read_gemini(home):
    rows, notes = [], []
    files = sorted(glob.glob(os.path.join(home, ".gemini", "tmp", "*", "logs.json")))
    for path in files:
        data = _read_json(path)
        if not isinstance(data, list):
            notes.append("gemini-cli: %s is not a JSON array; skipped" % path)
            continue
        for e in data:
            if isinstance(e, dict) and e.get("type") == "user" and isinstance(e.get("message"), str):
                text = e["message"].strip()
                if text and not text.startswith("/"):
                    rows.append({"source": "gemini-cli", "prompt": text, "ts": parse_ts(e.get("timestamp")),
                                 "project": os.path.basename(os.path.dirname(path))[:12]})
    if not files:
        notes.append("gemini-cli: no ~/.gemini/tmp/*/logs.json under %s" % home)
    return rows, notes


def _opencode_files(root):
    rows, sessions = [], {}
    for p in glob.glob(os.path.join(root, "session", "*", "*.json")):
        s = _read_json(p)
        if isinstance(s, dict) and s.get("id"):
            sessions[s["id"]] = _project(s.get("directory"))
    for mp in sorted(glob.glob(os.path.join(root, "message", "*", "*.json"))):
        msg = _read_json(mp)
        if not isinstance(msg, dict) or msg.get("role") != "user" or not msg.get("id"):
            continue
        parts = []
        for pp in sorted(glob.glob(os.path.join(root, "part", str(msg["id"]), "*.json"))):
            part = _read_json(pp)
            if isinstance(part, dict) and part.get("type") == "text" and not part.get("synthetic") \
                    and isinstance(part.get("text"), str):
                parts.append(part["text"])
        text = "\n".join(parts).strip()
        if text:
            t = msg.get("time")
            rows.append({"source": "opencode", "prompt": text,
                         "ts": parse_ts(t.get("created") if isinstance(t, dict) else None),
                         "project": sessions.get(msg.get("sessionID"), "")})
    return rows


def _opencode_db(path):
    """Best effort: tables `message` and `part` whose rows carry the fields as columns or in a JSON `data` column."""
    con = sqlite3.connect("file:%s?mode=ro" % path.replace("?", "%3f"), uri=True)
    try:
        con.row_factory = sqlite3.Row
        tables = {r[0].lower(): r[0] for r in con.execute("select name from sqlite_master where type='table'")}
        if "message" not in tables or "part" not in tables:
            raise ValueError("no message/part tables")

        def load(table):
            for r in con.execute('select * from "%s"' % tables[table]):
                d = dict(r)
                for v in list(d.values()):
                    if isinstance(v, str) and v.startswith("{"):
                        try:
                            d.update({k: x for k, x in json.loads(v).items() if k not in d})
                        except ValueError:
                            pass
                yield d

        def pick(d, *names):
            for n in names:
                if d.get(n) not in (None, ""):
                    return d[n]

        texts = {}
        for p in load("part"):
            mid = pick(p, "message_id", "messageID", "messageId")
            if mid and p.get("type") == "text" and not p.get("synthetic") and isinstance(p.get("text"), str):
                texts.setdefault(mid, []).append(p["text"])
        rows = []
        for msg in load("message"):
            if msg.get("role") != "user" or msg.get("id") not in texts:
                continue
            t = msg.get("time")
            rows.append({"source": "opencode", "prompt": "\n".join(texts[msg["id"]]).strip(),
                         "ts": parse_ts(pick(msg, "time_created") or (t.get("created") if isinstance(t, dict) else None)),
                         "project": ""})
        return rows
    finally:
        con.close()


def read_opencode(home):
    root = os.path.join(home, ".local", "share", "opencode")
    rows, notes = _opencode_files(os.path.join(root, "storage")), []
    db = os.path.join(root, "opencode.db")
    if not rows and os.path.exists(db):
        try:
            rows = _opencode_db(db)
        except (sqlite3.Error, ValueError) as e:
            notes.append("opencode: %s exists but its schema was not recognised (%s); skipped" % (db, e))
    if not rows and not notes:
        notes.append("opencode: no messages under %s" % root)
    return rows, notes


HERMES_AUTOMATED = ("cron", "curator")   # Hermes session platforms that a program, not a person, started


def _hermes_text(content):
    if isinstance(content, list):
        content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                            and isinstance(b.get("text"), str))
    return content.strip() if isinstance(content, str) else ""


def read_hermes(home):
    """~/.hermes/sessions: session_*.json (platform, session_start, messages) and <date>_<id>.jsonl (a session_meta
    line with the platform, then one line per message). Cron and curator sessions are automated."""
    base = os.path.join(home, ".hermes", "sessions")
    rows = []
    for path in sorted(glob.glob(os.path.join(base, "session_*.json"))):
        d = _read_json(path)
        if not isinstance(d, dict) or not isinstance(d.get("messages"), list):
            continue
        plat = str(d.get("platform") or "")
        origin = "automated" if plat in HERMES_AUTOMATED or "_cron_" in os.path.basename(path) else "typed"
        for e in d["messages"]:
            text = _hermes_text(e.get("content")) if isinstance(e, dict) and e.get("role") == "user" else ""
            if text:
                rows.append({"source": "hermes", "prompt": text, "ts": parse_ts(d.get("session_start")),
                             "project": plat, "origin": origin})
    for path in sorted(glob.glob(os.path.join(base, "[0-9]*.jsonl"))):
        origin, plat = "typed", ""
        for e in _lines(path):
            if e.get("role") == "session_meta":
                plat = str(e.get("platform") or "")
                origin = "automated" if plat in HERMES_AUTOMATED else "typed"
            elif e.get("role") == "user":
                text = _hermes_text(e.get("content"))
                if text:
                    rows.append({"source": "hermes", "prompt": text, "ts": parse_ts(e.get("timestamp")), "project": plat,
                                 "origin": origin})
    return rows, ([] if rows else ["hermes: no sessions under %s" % base])


def read_file(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        chunks = re.split(r"\n\s*\n", f.read())
    return [{"source": "file", "prompt": c.strip(), "ts": None, "project": ""} for c in chunks if c.strip()]


# ── harvest ──────────────────────────────────────────────────────────────────

def filter_rows(rows, since_s=None, now=None):
    """Drop slash commands, tiny prompts, acknowledgements, old rows and duplicates (keeps the newest copy)."""
    now = now or time.time()
    drops = Counter()
    seen, kept = {}, []
    for r in sorted(rows, key=lambda r: -(r["ts"] or 0)):
        text = clean(r["prompt"])
        r = dict(r, prompt=text)
        if _SLASH.match(text):
            drops["slash command"] += 1
        elif len(text.split()) < 3:
            drops["under 3 words"] += 1
        elif is_ack(text):
            drops["acknowledgement"] += 1
        elif since_s and r["ts"] and r["ts"] < now - since_s:
            drops["older than --since"] += 1
        elif norm_key(text) in seen:
            drops["duplicate"] += 1
        else:
            seen[norm_key(text)] = True
            kept.append(r)
    return kept, drops


def sample(rows, limit, seed=SEED):
    """Stratify by source (equal shares, unused share redistributed), prefer recent, then random with a fixed seed."""
    if not limit or len(rows) <= limit:
        return list(rows)
    groups = {}
    for r in rows:
        groups.setdefault(r["source"], []).append(r)
    quota, left, todo = {}, limit, sorted(groups)
    while todo and left > 0:
        share = max(1, left // len(todo))
        small = [s for s in todo if len(groups[s]) <= share]
        if not small:
            for s in todo:
                quota[s] = share
            break
        for s in small:
            quota[s] = len(groups[s])
            left -= quota[s]
            todo.remove(s)
    rng = random.Random(seed)
    out = []
    for s, g in groups.items():
        g = sorted(g, key=lambda r: -(r["ts"] or 0))
        q = quota.get(s, 0)
        out += rng.sample(g[:q * 2], q) if q < len(g) else g
    return out[:limit]


def row_id(r):
    return hashlib.sha1((r["source"] + "\0" + norm_key(r["prompt"])).encode("utf-8")).hexdigest()[:10]


def write_private(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def mine_dir(env=None):
    return os.path.join(m.maestro_home(env), "mine")


def cmd_harvest(a):
    home = os.path.abspath(os.path.expanduser(a.home))
    which = a.harness or ("none" if a.from_file else "all")
    readers = {"claude-code": lambda: read_claude(home, a.include_subagents), "codex": lambda: read_codex(home),
               "gemini-cli": lambda: read_gemini(home), "opencode": lambda: read_opencode(home),
               "hermes": lambda: read_hermes(home)}
    rows, found = [], Counter()
    for name in SOURCES:
        if which not in ("all", name):
            continue
        try:
            got, notes = readers[name]()
        except Exception as e:  # a format we did not anticipate must not stop the others
            got, notes = [], ["%s: skipped (%s: %s)" % (name, type(e).__name__, e)]
        for n in notes:
            print("note: " + n, file=sys.stderr)
        rows += got
    if a.from_file:
        try:
            got = read_file(os.path.expanduser(a.from_file))
        except OSError as e:
            print("note: cannot read %s: %s" % (a.from_file, e), file=sys.stderr)
            got = []
        rows += got
    for r in rows:
        found[r["source"]] += 1
    automated = Counter(r.get("project") or "?" for r in rows if r.get("origin") == "automated")
    if not a.include_automated:
        rows = [r for r in rows if r.get("origin") != "automated"]
    excluded = 0
    if a.project or a.exclude_project:
        before = len(rows)
        rows = [r for r in rows if (not a.project or any(p.lower() in (r.get("project") or "").lower() for p in a.project))
                and not any(x.lower() in (r.get("project") or "").lower() for x in (a.exclude_project or []))]
        excluded = before - len(rows)
    kept, drops = filter_rows(rows, a.since)
    picked = sample(kept, a.limit)
    picked.sort(key=lambda r: (r["ts"] or 0, r["source"]))
    total_red, out_lines = 0, []
    for r in picked:
        text, n = redact(r["prompt"])
        total_red += n
        out_lines.append(json.dumps({"id": row_id(r), "source": r["source"], "prompt": text[:MAX_KEEP],
                                     "ts": iso(r["ts"]), "project": r["project"]}, ensure_ascii=False))
    out = os.path.expanduser(a.out or os.path.join(mine_dir(), "prompts.jsonl"))
    write_private(out, "".join(l + "\n" for l in out_lines))
    print("harvested from %s" % home)
    print("  found     " + (", ".join("%s %d" % kv for kv in sorted(found.items())) or "nothing"))
    if automated:
        tops = ", ".join("%s %d" % kv for kv in automated.most_common(5))
        print("  %s %d automated prompts (sent by a program, not typed: claude -p, the SDK, orchestrators) - top projects: %s"
              % ("included" if a.include_automated else "skipped ", sum(automated.values()), tops))
    if excluded:
        print("  excluded  %d by --project/--exclude-project" % excluded)
    print("  dropped   " + (", ".join("%s %d" % kv for kv in sorted(drops.items())) or "nothing"))
    print("  kept      %d after filters, sampled %d (limit %s)" % (len(kept), len(picked), a.limit or "none"))
    print("  by source " + (", ".join("%s %d" % kv for kv in sorted(Counter(r["source"] for r in picked).items())) or "-"))
    print("  by project " + (", ".join("%s %d" % kv for kv in Counter(r["project"] or "?" for r in picked).most_common(8)) or "-"))
    print("  redacted  %d likely secret%s" % (total_red, "" if total_red == 1 else "s"))
    print("wrote %d prompts to %s" % (len(picked), out))
    if picked:
        print("next: python3 evals/mine.py route --in %s" % out)
    return 0 if picked else 1


# ── route ────────────────────────────────────────────────────────────────────

def one_line(text, n=200):
    return re.sub(r"\s*[\r\n]+\s*", "⏎", text.replace("\t", " "))[:n]


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def source_harness(source):
    """Route each prompt as if it came from the harness it was typed into (claude-code:subagent -> claude-code;
    --from-file prompts have no harness and default to claude-code)."""
    base = str(source or "").split(":")[0]
    return base if base in HARNESSES else "claude-code"


def cmd_route(a):
    src = os.path.expanduser(a.in_path or os.path.join(mine_dir(), "prompts.jsonl"))
    try:
        prompts = [r for r in load_jsonl(src) if isinstance(r.get("prompt"), str) and r["prompt"].strip()]
    except (OSError, ValueError) as e:
        print("cannot read %s: %s (run `mine.py harvest` first)" % (src, e), file=sys.stderr)
        return 2
    if not prompts:
        print("no prompts in %s" % src, file=sys.stderr)
        return 2
    if a.classifier == "jev":
        if not os.environ.get("TYPESAFE_API_KEY"):
            print("--classifier jev needs TYPESAFE_API_KEY", file=sys.stderr)
            return 2
        print("warning: --classifier jev sends each prompt to api.typesafe.ai. Default rules run offline.", file=sys.stderr)
    home = m.maestro_home()
    cfg = m.load_config(home)
    inv = m.ensure_inventory(home, cfg=cfg)
    out, rows, skipped = os.path.expanduser(a.out or os.path.join(mine_dir(), "routed.csv")), [], 0
    for r in prompts:
        try:
            harness = a.harness or source_harness(r.get("source"))
            d = m.route(r["prompt"], harness=harness, classifier=a.classifier, config=cfg, inventory=inv, home=home)
        except Exception as e:
            skipped += 1
            print("note: could not route %s: %s" % (r.get("id", "?"), e), file=sys.stderr)
            continue
        rows.append((r, d))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for r, d in rows:
            w.writerow([r.get("id", ""), r.get("source", ""), one_line(r["prompt"]), d["job"], d["difficulty"],
                        d["category"], d["model"] or "", "", "", "", ""])
    try:
        os.chmod(out, 0o600)
    except OSError:
        pass
    print("routed %d prompts as %s (classifier %s, inventory %s)" % (
        len(rows), a.harness or "the harness each came from", a.classifier,
        "%d local model(s)" % len(inv.get("local_models", []))))
    if not inv.get("local_models"):
        print("note: no local or network models found, so only cloud models can be picked. Add your server to "
              "local_endpoints in ~/.maestro/config.json (see README) and re-run.")
    print("by harness:  " + ", ".join("%s %d" % kv for kv in Counter(d["harness"] for _, d in rows).most_common()))
    print("by job:      " + ", ".join("%s %d" % kv for kv in Counter(d["job"] for _, d in rows).most_common()))
    print("by category: " + ", ".join("%s %d" % kv for kv in Counter(d["category"] for _, d in rows).most_common()))
    print("by model:    " + ", ".join("%s %d" % kv for kv in Counter(d["model"] or "%s default" % d["seat"]
                                                                      for _, d in rows).most_common(8)))
    print("least confident:")
    for r, d in sorted(rows, key=lambda x: x[1]["confidence"])[:10]:
        print("  %.2f  %-9s d%d  %s" % (d["confidence"], d["job"], d["difficulty"], one_line(r["prompt"], 70)))
    print("\nwrote %s" % out)
    print("Open routed.csv; fill your_* ONLY where Maestro is wrong (blank = agree). Then run `mine.py score`.")
    print("  your_job: %s | your_difficulty: 0-3 | your_category: %s" % ("/".join(m.JOBS), "/".join(m.CELLS)))
    return 0 if rows else 1


# ── score ────────────────────────────────────────────────────────────────────

def effective_category(cat, job, diff):
    """A cloud tier for a decision's category: a local or specialist seat counts as its ladder cell (local = fast)."""
    if cat in m.TIERS:
        return cat
    return m.tier_of(m.target_tier(m.Classification(job, diff, 1.0), {}))


def read_labels(path):
    """Rows of the CSV with the label columns normalised; returns (rows, warnings)."""
    rows, warns = [], []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for i, r in enumerate(csv.DictReader(f), start=2):
            r = {k: (v or "").strip() for k, v in r.items() if k}
            try:
                r["maestro_difficulty"] = int(r["maestro_difficulty"])
                assert r["maestro_job"] in m.JOBS
            except (KeyError, ValueError, AssertionError):
                warns.append("line %d: not a Maestro row; skipped" % i)
                continue
            for col, ok in (("your_job", m.JOBS), ("your_category", m.CELLS + ("specialist",)), ("your_difficulty", "0123")):
                v = r.get(col, "").lower()
                if v and v not in ok:
                    warns.append("line %d: ignored %s=%r" % (i, col, r[col]))
                    v = ""
                r[col] = v
            rows.append(r)
    return rows, warns


def score_rows(rows):
    """Metrics and confusions; a blank your_* cell means Maestro was right."""
    n = len(rows)
    job_ok = diff_ok = cat_ok = fixed = 0
    delta, jobs_conf, cat_conf = Counter(), Counter(), Counter()
    for r in rows:
        mj, md = r["maestro_job"], r["maestro_difficulty"]
        yj = r["your_job"] or mj
        yd = int(r["your_difficulty"]) if r["your_difficulty"] else md
        fixed += bool(r["your_job"] or r["your_difficulty"] or r["your_category"])
        got = effective_category(r["maestro_category"], mj, md)
        want = effective_category(r["your_category"], yj, yd) if r["your_category"] else \
            m.tier_of(m.target_tier(m.Classification(yj, yd, 1.0), {})) if (yj, yd) != (mj, md) else got
        job_ok += yj == mj
        diff_ok += yd == md
        cat_ok += got == want
        delta[m.TIERS.index(got) - m.TIERS.index(want)] += 1
        if yj != mj:
            jobs_conf["%s -> %s" % (mj, yj)] += 1
        if got != want:
            cat_conf["%s -> %s" % (got, want)] += 1
    f = lambda k: sum(v for d, v in delta.items() if k(d)) / n
    return {"n": n, "corrected": fixed, "job_accuracy": job_ok / n, "difficulty_exact": diff_ok / n,
            "category_accuracy": cat_ok / n, "category_within_1": f(lambda d: abs(d) <= 1),
            "under_routed": f(lambda d: d <= -1), "severe_under_routed": f(lambda d: d <= -2),
            "over_routed": f(lambda d: d >= 1), "job_confusions": jobs_conf, "category_confusions": cat_conf}


def cmd_score(a):
    path = os.path.expanduser(a.csv or os.path.join(mine_dir(), "routed.csv"))
    try:
        rows, warns = read_labels(path)
    except OSError as e:
        print("cannot read %s: %s (run `mine.py route` first)" % (path, e), file=sys.stderr)
        return 2
    for w in warns:
        print("note: " + w, file=sys.stderr)
    if not rows:
        print("no rows in %s" % path, file=sys.stderr)
        return 2
    s = score_rows(rows)
    p = lambda x: "%.1f%%" % (100 * x)
    print("scored %d prompts; you corrected %d (blank = agree)" % (s["n"], s["corrected"]))
    if not s["corrected"]:
        print("warning: no your_* cells are filled in, so this only measures agreement with yourself. Review the CSV first.")
    print("job accuracy          " + p(s["job_accuracy"]))
    print("category accuracy     " + p(s["category_accuracy"]))
    print("category within +-1   " + p(s["category_within_1"]))
    print("under-routed          " + p(s["under_routed"]) + "   (Maestro picked a weaker category than you wanted)")
    print("severely under-routed " + p(s["severe_under_routed"]) + "   (two or more steps)")
    print("over-routed           " + p(s["over_routed"]))
    print("difficulty exact      " + p(s["difficulty_exact"]))
    for title, c in (("job confusions (maestro -> you)", s["job_confusions"]),
                     ("category confusions (maestro -> you)", s["category_confusions"])):
        if c:
            print("\n" + title)
            for k, v in c.most_common(12):
                print("  %3d  %s" % (v, k))
    if a.export:
        prompts_path = os.path.expanduser(a.prompts or os.path.join(mine_dir(), "prompts.jsonl"))
        try:
            full = {r["id"]: r["prompt"] for r in load_jsonl(prompts_path) if "id" in r}
        except (OSError, ValueError):
            full = {}
            print("note: %s not found; exporting the 200-character prompts from the CSV" % prompts_path, file=sys.stderr)
        out = []
        for r in rows:
            text = full.get(r["id"]) or r["prompt"].replace("⏎", "\n")
            out.append(json.dumps({"prompt": text, "job": r["your_job"] or r["maestro_job"],
                                   "difficulty": int(r["your_difficulty"]) if r["your_difficulty"] else r["maestro_difficulty"]},
                                  ensure_ascii=False))
        write_private(os.path.expanduser(a.export), "".join(l + "\n" for l in out))
        print("\nwrote %d labeled rows to %s (evals/prompts.jsonl format). These are your private prompts: review "
              "them before you share or contribute any." % (len(out), a.export))
    return 0


def build_parser():
    ap = argparse.ArgumentParser(prog="mine.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harvest", help="collect, filter, redact and sample your harness prompts")
    h.add_argument("--home", default="~", help="home directory to read (a copy of another machine's works)")
    h.add_argument("--harness", choices=("all",) + SOURCES, help="default: all (or none when --from-file is given)")
    h.add_argument("--since", type=parse_since, default=parse_since("90d"), help="e.g. 90d, 12w, 6m, all (default 90d)")
    h.add_argument("--limit", type=int, default=80, help="total prompts to keep, 0 = no limit (default 80)")
    h.add_argument("--include-subagents", action="store_true",
                   help="Claude Code: add the subtask each subagent was given (written by an AI, not by you)")
    h.add_argument("--include-automated", action="store_true",
                   help="keep prompts sent by programs (claude -p, Agent SDK, orchestrators, codex exec); default: typed only")
    h.add_argument("--project", action="append", help="only keep prompts from projects whose path contains this (repeatable)")
    h.add_argument("--exclude-project", action="append", help="drop prompts from projects whose path contains this (repeatable)")
    h.add_argument("--from-file", help="text file of prompts separated by blank lines")
    h.add_argument("--out", help="default ~/.maestro/mine/prompts.jsonl")
    r = sub.add_parser("route", help="run Maestro over the harvested prompts and write a CSV to review")
    r.add_argument("--in", dest="in_path")
    r.add_argument("--harness", choices=HARNESSES,
                   help="route every prompt as this harness (default: the harness each prompt came from)")
    r.add_argument("--classifier", choices=("rules", "jev"), default="rules")
    r.add_argument("--out", help="default ~/.maestro/mine/routed.csv")
    s = sub.add_parser("score", help="score Maestro against your corrections")
    s.add_argument("--csv", help="default ~/.maestro/mine/routed.csv")
    s.add_argument("--prompts", help="harvested prompts, to export full text (default ~/.maestro/mine/prompts.jsonl)")
    s.add_argument("--export", help="write labeled rows in evals/prompts.jsonl format")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    return {"harvest": cmd_harvest, "route": cmd_route, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
