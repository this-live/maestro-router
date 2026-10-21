"""Shared helpers for the S1 data and alpha tooling (spec §6, §7, §9). Evals only; stdlib only.

Work manifests. A step that needs a model (panel labels, answers, criteria, grading, adjudication) never calls a model
API. It writes JSONL batch files under `evals/work/<step>/`, one file per batch, one call per line:

    {"call_id": "ans:V-code-fn-001:haiku:r1:a1", "model": "haiku", "prompt": "...", "expect": "text" | "json"}

The orchestrating session sends `prompt` verbatim to `model` (one fresh call each; never batched, never with tools) and
appends one line per reply to `<batch>.replies.jsonl` next to the batch file:

    {"call_id": "...", "reply": "<the model's text>"}            optional: "error": "...", "tool_calls": 3

Then `ingest` validates the replies. Call ids are stable: the same work always gets the same id, so re-running a batch
command after a partial run re-emits only what is still missing. Per-call metadata (item ids, answer keys) lives in
`_meta.jsonl`, never in the batch file, so an orchestrator cannot leak it into a prompt.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "skills", "maestro", "scripts"))
from maestro import router as m  # noqa: E402

TIERS = ("haiku", "sonnet", "opus", "fable")                # the four Claude anchors, cheapest first
TIER_CATEGORY = dict(zip(TIERS, m.TIERS))                    # haiku -> fast ... fable -> frontier
JOBS = m.JOBS
ALPHA = os.path.join(HERE, "alpha")
LABELS = os.path.join(HERE, "labels")
WORK = os.path.join(HERE, "work")                            # gitignored: manifests, replies, answer texts
LEDGER = os.path.join(HERE, "ledger.jsonl")
S1_BIN = os.path.join(ROOT, "skills", "maestro", "scripts", "maestro", "data", "s1.bin")


# ── files ──────────────────────────────────────────────────────────────────────

def read_jsonl(path):
    """Rows of a JSONL file; [] when it does not exist."""
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _atomic(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def write_jsonl(path, rows):
    _atomic(path, "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def append_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    _atomic(path, json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_text(s):
    return sha256_bytes(s.encode("utf-8"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unit(*parts):
    """A deterministic uniform number in [0, 1) from string parts (seeded sampling without a global RNG)."""
    return int(sha256_text("\x1f".join(str(p) for p in parts))[:15], 16) / float(16 ** 15)


def norm_text(s):
    return " ".join((s or "").lower().split())


_FENCE = re.compile(r"```.*?(?:```|\Z)", re.S)


def code_ratio(text):
    """Share of characters inside fenced code blocks."""
    return sum(len(x) for x in _FENCE.findall(text or "")) / max(1, len(text or ""))


def est_tokens(text):
    """§7.3: round(bytes / (3.2 if code_ratio > 0.3 else 3.9))."""
    text = text or ""
    return int(round(len(text.encode("utf-8")) / (3.2 if code_ratio(text) > 0.3 else 3.9)))


def item_text(row):
    """The text a row is matched on: its prompt (or fixture task) plus earlier user turns."""
    hist = [h.get("content") or "" for h in row.get("history") or [] if isinstance(h, dict) and h.get("role") == "user"]
    return "\n".join(hist + [row.get("prompt") or row.get("task") or ""])


def render_history(history):
    out = []
    for h in history or []:
        if isinstance(h, dict) and h.get("content"):
            out.append("%s: %s" % ("Earlier user message" if h.get("role") == "user" else "Earlier assistant reply",
                                   h["content"]))
    return "\n\n".join(out)


def clip_middle(text, head=6000, tail=2000):
    if len(text) <= head + tail:
        return text
    return text[:head] + "\n[... %d characters omitted ...]\n" % (len(text) - head - tail) + text[-tail:]


# ── work manifests (see the module docstring) ──────────────────────────────────

def work_dir(step, root=None):
    return os.path.join(root or WORK, step)


def emit(dirpath, prefix, calls, per_file=50):
    """Write calls [(call, meta)] as batch files of `per_file` calls. A batch file is named by a hash of its call ids,
    so re-emitting the same work rewrites the same file. Returns the paths written."""
    os.makedirs(dirpath, exist_ok=True)
    old = load_meta(dirpath)
    new_meta = [dict(meta, call_id=c["call_id"]) for c, meta in calls if c["call_id"] not in old]
    if new_meta:
        append_jsonl(os.path.join(dirpath, "_meta.jsonl"), new_meta)
    paths = []
    for i in range(0, len(calls), per_file):
        chunk = [c for c, _ in calls[i:i + per_file]]
        name = "%s-%s.jsonl" % (prefix, sha256_text("\n".join(c["call_id"] for c in chunk))[:10])
        path = os.path.join(dirpath, name)
        write_jsonl(path, [{k: c[k] for k in ("call_id", "model", "prompt", "expect")} for c in chunk])
        paths.append(path)
    return paths


def load_meta(dirpath):
    return {r["call_id"]: r for r in read_jsonl(os.path.join(dirpath, "_meta.jsonl"))}


def read_replies(dirpath):
    """{call_id: reply row} from every *.replies.jsonl in the directory (a later line wins). Malformed lines are
    returned under the key None as a list of error strings."""
    out, bad = {}, []
    for path in sorted(glob.glob(os.path.join(dirpath, "*.replies.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    bad.append("%s:%d: not JSON" % (os.path.basename(path), n))
                    continue
                if not isinstance(r, dict) or not isinstance(r.get("call_id"), str) or not isinstance(r.get("reply"), str):
                    bad.append("%s:%d: needs string call_id and reply" % (os.path.basename(path), n))
                    continue
                out[r["call_id"]] = r
    if bad:
        out[None] = bad
    return out


def load_done(dirpath):
    return {r["call_id"]: r for r in read_jsonl(os.path.join(dirpath, "_done.jsonl"))}


def mark_done(dirpath, records):
    if records:
        append_jsonl(os.path.join(dirpath, "_done.jsonl"), records)


def attempt_id(done, base):
    """`base:aN` with N one more than the attempts already ingested (so a re-emitted call is never mistaken for one
    that was already answered, discarded or found invalid)."""
    return "%s:a%d" % (base, 1 + sum(1 for cid in done if cid.startswith(base + ":a")))


def pending(dirpath):
    """{call_id: meta} for emitted calls with no ingested reply yet."""
    done = load_done(dirpath)
    return {cid: meta for cid, meta in load_meta(dirpath).items() if cid not in done}


def parse_json_reply(text):
    """The JSON value in a model reply: the whole text, else a fenced block, else the outermost [...] or {...}."""
    text = (text or "").strip()
    cands = [text] + re.findall(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    for o, c in (("[", "]"), ("{", "}")):
        i, j = text.find(o), text.rfind(c)
        if 0 <= i < j:
            cands.append(text[i:j + 1])
    for c in cands:
        try:
            return json.loads(c)
        except ValueError:
            continue
    raise ValueError("no JSON value in the reply")


# ── checkers (§7.7: python -I, empty environment, temp cwd, CPU 10 s and 1 GB on POSIX) ──────────

_DRIVER = r'''
import sys
nonce = sys.argv[3]
src = open(sys.argv[1], encoding="utf-8").read()
ans = open(sys.argv[2], encoding="utf-8").read()
ok = False
try:
    ns = {"__name__": "__checker__"}
    exec(compile(src, "<checker>", "exec"), ns)
    ok = bool(ns["check"](ans))
except BaseException:
    ok = False
sys.stdout.write("\n" + nonce + ("PASS" if ok else "FAIL") + "\n")
sys.stdout.flush()
'''


def _limits():  # pragma: no cover (runs in the child)
    try:
        import resource
        for res, val in ((resource.RLIMIT_CPU, 10), (resource.RLIMIT_AS, 1 << 30)):
            try:
                resource.setrlimit(res, (val, val))
            except (ValueError, OSError):
                pass
    except ImportError:
        pass


def run_checker(checker_py, answer, timeout=30.0):
    """True when the item's `check(answer)` passes in a sandboxed subprocess; False on failure, crash or timeout."""
    nonce = secrets.token_hex(8)
    with tempfile.TemporaryDirectory(prefix="maestro-chk-") as d:
        paths = []
        for name, text in (("driver.py", _DRIVER), ("checker.py", checker_py), ("answer.txt", answer or "")):
            p = os.path.join(d, name)
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
            paths.append(p)
        try:
            p = subprocess.run([sys.executable, "-I"] + paths + [nonce], cwd=d, env={}, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout,
                               preexec_fn=_limits if os.name == "posix" else None)
        except (subprocess.TimeoutExpired, OSError):
            return False
    tail = p.stdout.decode("utf-8", "replace").rstrip().rsplit("\n", 1)[-1]
    return tail == nonce + "PASS"


# ── the orchestrator's quota rule (§7.3: pause while five_hour > 70% or seven_day > 60%) ──────────

def quota_status(home=None):
    """(pause, message) from the Claude statusline cache that Maestro already reads."""
    from maestro import quota
    lim = quota.claude_limits(home or m.maestro_home())
    if not lim or time.time() - float(lim.get("updated_at") or 0) > 1800:
        return False, "quota: unknown (no claude_limits.json fresher than 30 min; `maestro-router install --statusline`)"
    five = (lim.get("five_hour") or {}).get("used_percent")
    week = (lim.get("seven_day") or {}).get("used_percent")
    pause = (five or 0) > 70 or (week or 0) > 60
    return pause, "quota: five_hour %s%%, seven_day %s%% -> %s" % (
        "?" if five is None else "%.0f" % five, "?" if week is None else "%.0f" % week,
        "PAUSE before sending more calls (> 70% / > 60%)" if pause else "ok")


# ── ledger (§6.2) ──────────────────────────────────────────────────────────────

def weights_sha():
    """sha256 of the shipped S1 weights, else of the Naive Bayes model the router uses today."""
    if os.path.exists(S1_BIN):
        return "s1:" + sha256_file(S1_BIN)[:16]
    nb = os.path.join(os.path.dirname(S1_BIN), "..", m.MODEL_FILE)
    return "nb:" + (sha256_file(nb)[:16] if os.path.exists(nb) else "none")


def ledger_append(split, metrics, path=None, **extra):
    """One row each time a test or persona split is scored outside the CI gate: {ts, split, weights_sha, metrics}."""
    row = {"ts": int(time.time()), "split": split, "weights_sha": weights_sha(), "metrics": metrics}
    row.update(extra)
    append_jsonl(path or LEDGER, [row])
    return row
