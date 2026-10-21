"""Tasks per suite, grading verifiable answers, and one agentic fixture run.

verifiable  JSONL {id, prompt, checker_py, type?, job?, difficulty_intent?}: chat answers, graded by the item's checker in
            a sandboxed subprocess (evalkit.run_checker). Extraction variants: when the raw answer fails, the checker also
            sees the last fenced block that defines a function, the last SQL block and the last JSON value, each re-fenced;
            the same variants are tried for every seat, and the row records which one passed (`variant`, `y_strict`).
agentic     evals/arena/fixtures/<id>/ {meta.json, task.md, repo/, hidden_tests/}: the harness works on a fresh temp git
            copy of repo/; hidden_tests/ is copied in only afterwards and meta.test_cmd decides pass/fail.
open        JSONL {id, prompt, job?, difficulty?}: chat answers, ranked by judges (rank.py).
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

import alpha
from .common import FIXTURES, ek, read_jsonl
from .harness import HARNESSES, build_argv, harness_env, m_scrub, parse_output, run_cli

AGENT_PROMPT = ("You are working in a software repository: the current directory. Complete the task below by changing the "
                "files here. Work only inside this directory, and do not ask questions: nobody will answer them. When you "
                "are done, reply with a short summary of what you changed.\n\nTASK:\n{task}")
SKIP = ("__pycache__", ".pytest_cache", "*.pyc")
CELL_DIRS = ("repo", "scratch", "tmp")


# ── tasks ─────────────────────────────────────────────────────────────────────

def _chat_task(suite, r):
    it = {"prompt": r.get("prompt") or "", "history": r.get("history") or [], "contract": r.get("contract") or ""}
    job = r.get("job") or alpha.V_JOB.get(r.get("type"))
    diff = r.get("difficulty", r.get("difficulty_intent"))
    return {"suite": suite, "id": r["id"], "job": job, "difficulty": diff if diff in (0, 1, 2, 3) else None,
            "prompt": alpha.answer_prompt(it), "request": alpha._body(it), "text": ek.item_text(r), "checker_py": r.get("checker_py"),
            "lang": r.get("lang") or "en"}


def load_verifiable(path):
    rows = [r for r in read_jsonl(path) if r.get("validated") is not False]
    bad = [r.get("id") for r in rows if not (r.get("id") and r.get("prompt") and r.get("checker_py"))]
    if bad:
        raise ValueError("%s: %d rows lack id, prompt or checker_py (first: %s)" % (path, len(bad), bad[0]))
    return [_chat_task("verifiable", r) for r in rows]


def load_open(path):
    rows = read_jsonl(path)
    if any(not (r.get("id") and r.get("prompt")) for r in rows):
        raise ValueError("%s: every row needs id and prompt" % path)
    return [_chat_task("open", r) for r in rows]


def load_agentic(root=FIXTURES):
    out = []
    for fid in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        d = os.path.join(root, fid)
        meta = ek.read_json(os.path.join(d, "meta.json"))
        if not isinstance(meta, dict) or not os.path.isdir(os.path.join(d, "repo")) or not meta.get("test_cmd"):
            continue
        with open(os.path.join(d, "task.md"), encoding="utf-8") as f:
            task = f.read()
        diff = meta.get("difficulty_intent")
        out.append({"suite": "agentic", "id": meta.get("id") or fid, "job": meta.get("job"),
                    "difficulty": diff if diff in (0, 1, 2, 3) else None, "prompt": AGENT_PROMPT.format(task=task.strip()),
                    "text": task, "fixture_dir": d, "meta": meta, "lang": "en"})
    return out


# ── verifiable grading ────────────────────────────────────────────────────────

_FENCE = re.compile(r"```[ \t]*([\w+.-]*)[^\n]*\n(.*?)```", re.S)
_DEF = re.compile(r"^\s*(?:async\s+)?def\s+\w+|^\s*class\s+\w+", re.M)
_SQL = re.compile(r"\s*(?:select|with|insert|update|delete|create)\b", re.I)


def variants(answer):
    """[(name, text)]: the last block that defines a function, the last SQL block, the last JSON value (each re-fenced)."""
    blocks = _FENCE.findall(answer or "")
    out = []
    fn = [b for _, b in blocks if _DEF.search(b)]
    if fn:
        out.append(("function", "```python\n%s```" % fn[-1]))
    sql = [b for lang, b in blocks if lang.lower() == "sql" or _SQL.match(b)]
    if sql:
        out.append(("sql", "```sql\n%s```" % sql[-1]))
    js = None
    for _, b in reversed(blocks):
        try:
            json.loads(b)
            js = b
            break
        except ValueError:
            continue
    if js is None:
        try:
            js = json.dumps(ek.parse_json_reply(answer), ensure_ascii=False, indent=1) + "\n"
        except ValueError:
            js = None
    if js is not None:
        out.append(("json", "```json\n%s```" % js))
    return out


def grade_verifiable(checker_py, answer, run=None):
    """{"y", "y_strict", "variant"}: the raw answer first, then each extraction variant (the first that passes wins)."""
    run = run or ek.run_checker
    if run(checker_py, answer):
        return {"y": 1, "y_strict": 1, "variant": "raw"}
    for name, text in variants(answer):
        if run(checker_py, text):
            return {"y": 1, "y_strict": 0, "variant": name}
    return {"y": 0, "y_strict": 0, "variant": None}


# ── agentic ───────────────────────────────────────────────────────────────────

def _snapshot(path):
    out = []
    for base, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", ".pytest_cache"))
        for f in sorted(files):
            if f.endswith(".pyc"):
                continue
            p = os.path.join(base, f)
            try:
                st_ = os.stat(p)
                out.append((os.path.relpath(p, path), st_.st_size, st_.st_mtime_ns))
            except OSError:
                continue
    return out


GIT = ["git", "-c", "user.email=arena@localhost", "-c", "user.name=arena", "-c", "commit.gpgsign=false",
       "-c", "core.hooksPath=/dev/null"]


def _git(work, *args):
    try:
        p = subprocess.run(GIT + list(args), cwd=work, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=60)
        return p.returncode, p.stdout.decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError):
        return None, ""


def git_init(work):
    """Make `work` a git repo with one base commit; returns the base sha or None (no git)."""
    if not shutil.which("git"):
        return None
    _git(work, "init", "-q")
    with open(os.path.join(work, ".git", "info", "exclude"), "a", encoding="utf-8") as f:
        f.write("__pycache__/\n*.pyc\n.pytest_cache/\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "--allow-empty", "-m", "arena base")
    rc, sha = _git(work, "rev-parse", "HEAD")
    return sha.strip() if rc == 0 else None


def git_diff(work, base):
    """({"files", "added", "removed"}, patch text) of everything the harness changed since `base` (committed or not)."""
    if not base:
        return None, ""
    _git(work, "add", "-A")
    _, num = _git(work, "diff", "--cached", "--numstat", base)
    _, patch = _git(work, "diff", "--cached", base)
    files, add, rem = 0, 0, 0
    for line in num.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            files += 1
            add += int(parts[0]) if parts[0].isdigit() else 0
            rem += int(parts[1]) if parts[1].isdigit() else 0
    return {"files": files, "added": add, "removed": rem}, patch


def run_tests(cmd, work, timeout):
    argv = shlex.split(cmd)
    if argv and argv[0] in ("python3", "python"):
        argv[0] = sys.executable
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    r = run_cli(argv, work, env, None, timeout)
    return {"rc": r["rc"], "timed_out": r["timed_out"], "tail": (r["stdout"] + r["stderr"])[-600:]}


def run_agentic(task, seat, exe, timeout=None, env=None, test_timeout=180.0):
    """One fixture x seat. Returns (row fields, patch text). The harness runs with cwd = a fresh temp git copy of repo/,
    inside a cell directory that holds only repo/, scratch/ (generated harness config) and tmp/ (TMPDIR). Anything else
    that appears in the cell directory, or any change to the fixture itself, is an escape: the cell fails and is flagged."""
    meta, fdir = task["meta"], task["fixture_dir"]
    timeout = float(timeout or meta.get("timeout_s") or 600)
    before = _snapshot(fdir)
    root = tempfile.mkdtemp(prefix="arena-cell-")
    try:
        work, scratch, tmp = (os.path.join(root, d) for d in CELL_DIRS)
        shutil.copytree(os.path.join(fdir, "repo"), work, ignore=shutil.ignore_patterns(*SKIP))
        os.makedirs(scratch)
        os.makedirs(tmp)
        base = git_init(work)
        henv = harness_env(seat, scratch, env)
        henv["TMPDIR"] = tmp
        argv, stdin = build_argv(seat, "agentic", task["prompt"], exe)
        r = run_cli(argv, work, henv, stdin, timeout)
        text, usage, reported, extra = parse_output(HARNESSES[seat["harness"]]["out"], r["stdout"]) if r["stdout"].strip() \
            else ("", None, None, {})
        escapes = sorted(e for e in os.listdir(root) if e not in CELL_DIRS)
        if _snapshot(fdir) != before:
            escapes.append("<fixture changed>")
        diff, patch = git_diff(work, base)
        hidden = os.path.join(work, "hidden_tests")
        if os.path.exists(hidden):
            shutil.rmtree(hidden, ignore_errors=True)
        shutil.copytree(os.path.join(fdir, "hidden_tests"), hidden, ignore=shutil.ignore_patterns(*SKIP))
        tests = run_tests(meta["test_cmd"], work, test_timeout)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    passed = tests["rc"] == 0 and not tests["timed_out"]
    harness_failed = bool(r["error"]) or (r["rc"] not in (0, None) and not r["timed_out"] and not (diff or {}).get("files"))
    row = {"status": "error" if harness_failed else "ok", "y": 1 if passed and not escapes else 0, "tests_passed": passed,
           "rc": r["rc"], "timed_out": r["timed_out"], "secs": r["secs"], "diff": diff, "escapes": escapes,
           "final": (text or "")[-2000:], "usage": usage, "reported_cost": reported, "tests": tests,
           "error": r["error"] or (m_scrub("exit %s: %s" % (r["rc"], (r["stderr"] or "").strip()[-300:])) if harness_failed else "")}
    if extra:
        row["extra"] = extra
    return row, patch
