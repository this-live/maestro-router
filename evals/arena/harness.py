"""How each harness runs headless, in ONE table (HARNESSES): fix a flag here and nowhere else.

Template tokens: {exe} {model} {effort} {prompt}. An item ("effort", [...]) is spliced in only when the seat has an effort;
an item ("if", [subcommand...], [flags...]) only when `<exe> <subcommand> --help` mentions the first flag (checked once per
executable), for flags that newer CLI versions need and older ones reject.
`prompt` says how the task reaches the CLI: "stdin" (piped) or "arg" (the {prompt} token; a leading "-" gets a space so it
is not read as a flag). `out` names the stdout parser. `verified` records what was checked on the machine that wrote this
table (claude 2.1.289 `--help` and a live run, 2026-10-05; codex, gemini and hermes on a second machine, 2026-10-06); a
row marked "assumed" is taken from the CLI's documentation: run the one-line check in docs/ARENA-RUNBOOK.md and fix the row
if your version differs.

Agentic runs happen in a fresh temp copy of the fixture repo (cwd), never in the user's tree. Chat answers run in an empty
temp dir with tools off where the CLI allows it. Secrets: only the CLIs' own logins are used (never read here); a local
endpoint's key is passed by NAME (`{env:VAR}` in the OpenCode config, the Bearer header of a direct call to that endpoint).
"""
from __future__ import annotations

import functools
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time

from .common import est_tokens, m

ALLOWED_TOOLS = "Bash,Edit,Write,Read,Glob,Grep"

HARNESSES = {
    "claude-code": {
        "exe": "claude", "prompt": "stdin", "out": "claude-json", "quota": "claude", "family": "anthropic",
        "agentic": ["{exe}", "-p", ("if", [], ["--safe-mode"]), "--model", "{model}", ("effort", ["--effort", "{effort}"]),
                    "--permission-mode", "acceptEdits", "--allowedTools", ALLOWED_TOOLS, "--permission-prompts", "none",
                    "--no-session-persistence", "--output-format", "json"],
        "chat": ["{exe}", "-p", ("if", [], ["--safe-mode"]), "--model", "{model}", ("effort", ["--effort", "{effort}"]), "--tools", "",
                 "--no-session-persistence", "--output-format", "json"],
        "verified": "flags checked against `claude --help` (2.1.289); the chat row ran live in the smoke run; the agentic "
                    "row was not run live (it would spend the plan) - its flags are all in --help. claude 2.1.268, both rows "
                    "live, 2026-10-06: --safe-mode keeps the subscription login and built-in tools but drops the user's "
                    "CLAUDE.md, skills, plugins, hooks and MCP servers; without it a heavily customized install sent "
                    "~390k tokens for a one-line prompt and every cell failed ('Prompt is too long')",
    },
    "codex": {
        "exe": "codex", "prompt": "arg", "out": "codex-jsonl", "quota": "codex", "family": "openai",
        "agentic": ["{exe}", "exec", "-m", "{model}", ("effort", ["-c", "model_reasoning_effort={effort}"]),
                    "--sandbox", "workspace-write", "--skip-git-repo-check", "--json", "{prompt}"],
        "chat": ["{exe}", "exec", "-m", "{model}", ("effort", ["-c", "model_reasoning_effort={effort}"]),
                 "--sandbox", "read-only", "--skip-git-repo-check", "--json", "{prompt}"],
        # Agentic cells first check that Codex's own sandbox can start: on Ubuntu 24.04 (AppArmor restricts unprivileged
        # user namespaces) bubblewrap fails, every shell and file-write call fails, and the cell would count as a fail.
        "preflight": {"agentic": ["{exe}", "sandbox", "-c", 'sandbox_mode="workspace-write"', "--", "true"]},
        "verified": "codex-cli 0.146.0, chat row run live with gpt-5.5 and parsed, 2026-10-06; on a ChatGPT login that "
                    "version rejects gpt-6.x models (upgrade Codex for them); the -c effort key is the one Maestro's router "
                    "already uses",
    },
    "gemini-cli": {
        "exe": "gemini", "prompt": "arg", "out": "gemini-json", "quota": "gemini", "family": "google",
        "agentic": ["{exe}", "-m", "{model}", "--yolo", "--output-format", "json", "-p", "{prompt}"],
        "chat": ["{exe}", "-m", "{model}", "--output-format", "json", "-p", "{prompt}"],
        "verified": "gemini 0.38.0: -m, -p, --yolo, --output-format json all in --help (2026-10-06); not run live (the "
                    "login on that machine was refused by Google); plain-text output is accepted too",
    },
    "opencode": {
        "exe": "opencode", "prompt": "arg", "out": "text", "quota": None, "family": None,
        "agentic": ["{exe}", "run", ("if", ["run"], ["--standalone"]), "-m", "{model}", "{prompt}"],
        "chat": ["{exe}", "run", ("if", ["run"], ["--standalone"]), "-m", "{model}", "{prompt}"],
        "verified": "opencode 2.0.20, run live against a vLLM model through a generated OPENCODE_CONFIG, 2026-10-06: chat, a "
                    "file edit and a shell command all work headless without --auto. Without --standalone, `run` starts a "
                    "shared `opencode serve --service` that outlives the cell and keeps the first cell's config and env. "
                    "opencode 1.18.34 (the current release) has no --standalone and rejects it, and starts no service",
    },
    "hermes": {
        "exe": "hermes", "prompt": "arg", "out": "hermes-text", "quota": None, "family": None,
        "agentic": ["{exe}", "chat", "-Q", "-q", "{prompt}"],
        "chat": ["{exe}", "chat", "-Q", "-q", "{prompt}"],
        "verified": "hermes 0.21.3, chat row run live against a vLLM model through a temp HERMES_HOME (custom provider, "
                    "the seat's base_url), 2026-10-06: without -Q, stdout carries a banner, a boxed answer and resume "
                    "hints; with -Q it is the answer alone (session id on stderr). The agentic row was not run live. An upstream "
                    "0.21.3 build (no --format) still prints a tirith warning line first with -Q: the parser drops "
                    "leading warning lines",
    },
}
CLI_HARNESSES = tuple(HARNESSES)
EXE = {h: v["exe"] for h, v in HARNESSES.items()}


def build_argv(seat, mode, prompt, exe=None):
    """(argv, stdin text or None) for one seat in `mode` (agentic | chat)."""
    spec = HARNESSES[seat["harness"]]
    vals = {"exe": exe or spec["exe"], "model": seat.get("model") or "", "effort": seat.get("effort") or "",
            "prompt": (" " + prompt) if prompt.startswith("-") else prompt}
    argv = []
    for item in spec[mode]:
        if isinstance(item, tuple) and item[0] == "if":
            if supports(vals["exe"], tuple(item[1]), item[2][0]):
                argv += list(item[2])
        elif isinstance(item, tuple):
            if vals.get(item[0]):
                argv += [x.format(**vals) for x in item[1]]
        elif item == "{prompt}":
            argv.append(vals["prompt"])
        else:
            argv.append(item.format(**vals))
    if seat["harness"] == "opencode" and not seat.get("model"):
        argv = [a for k, a in enumerate(argv) if not (a == "-m" or (k and argv[k - 1] == "-m"))]
    return argv, (prompt if spec["prompt"] == "stdin" else None)


@functools.lru_cache(maxsize=None)
def _help(exe, sub):
    try:
        p = subprocess.run([exe] + list(sub) + ["--help"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return p.stdout.decode("utf-8", "replace")


def supports(exe, sub, flag):
    """True when `exe *sub --help` mentions `flag` (the help text is read once per executable and subcommand)."""
    return flag in _help(exe, tuple(sub))


@functools.lru_cache(maxsize=None)
def _preflight(exe, argv):
    with tempfile.TemporaryDirectory(prefix="arena-preflight-") as d:
        r = run_cli(list(argv), d, None, None, 60)
    if r["error"] or r["timed_out"] or r["rc"] != 0:
        tail = (r["error"] or r["stderr"] or r["stdout"] or "timed out").strip().splitlines()
        return m_scrub("%s sandbox unavailable: %s" % (os.path.basename(exe), tail[-1] if tail else "exit %s" % r["rc"]))
    return ""


def preflight(seat, mode, exe):
    """"" when the harness can run `mode` here, else the reason (its `preflight` command failed; checked once per CLI)."""
    cmd = (HARNESSES[seat["harness"]].get("preflight") or {}).get(mode)
    return _preflight(exe, tuple(x.format(exe=exe) for x in cmd)) if cmd else ""


def harness_env(seat, scratch, env=None):
    """The environment for one CLI call. A local-model seat behind OpenCode or Hermes gets a generated config in `scratch`
    (outside the workdir, so it never shows up in the diff); the endpoint key is referenced by name, never written."""
    env = dict(os.environ if env is None else env)
    ep = seat.get("endpoint")
    if not ep:
        return env
    if seat["harness"] == "opencode":
        opts = {"baseURL": ep["base_url"]}
        if ep.get("api_key_env"):
            opts["apiKey"] = "{env:%s}" % ep["api_key_env"]
        cfg = {"provider": {ep["name"]: {"npm": "@ai-sdk/openai-compatible", "name": ep["name"], "options": opts,
                                         "models": {ep["served"]: {"name": ep["served"]}}}}}
        path = os.path.join(scratch, "opencode.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=1)
        env["OPENCODE_CONFIG"] = path
    elif seat["harness"] == "hermes":
        home = os.path.join(scratch, "hermes")
        os.makedirs(home, exist_ok=True)
        with open(os.path.join(home, "config.yaml"), "w", encoding="utf-8") as f:
            f.write("model:\n  default: %s\n  provider: custom\n  base_url: %s\n  api_key: %s\n"
                    % (ep["served"], ep["base_url"], "arena-placeholder"))
        env["HERMES_HOME"] = home
    return env


def opencode_snippet(name, base_url, served, api_key_env=None):
    """The opencode.json block that points OpenCode at an OpenAI-compatible server (for the docs and `seats --json`)."""
    opts = {"baseURL": base_url}
    if api_key_env:
        opts["apiKey"] = "{env:%s}" % api_key_env
    return {"provider": {name: {"npm": "@ai-sdk/openai-compatible", "name": name, "options": opts,
                                "models": {served: {"name": served}}}}}


# ── processes ─────────────────────────────────────────────────────────────────

def _kill(p):
    try:
        if os.name == "posix":
            os.killpg(p.pid, signal.SIGKILL)
        else:
            p.kill()
    except OSError:
        pass


def run_cli(argv, cwd, env, stdin_text, timeout):
    """Run one CLI to completion or timeout (the whole process group is killed). Never raises. PWD is set to `cwd`:
    OpenCode 2.x works in $PWD rather than its real cwd, so an inherited PWD sent it into the directory the arena was
    started from (a live tree) instead of the cell's copy."""
    t0 = time.monotonic()
    env = dict(os.environ if env is None else env, PWD=os.path.abspath(cwd))
    try:
        p = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=os.name == "posix")
    except OSError as e:
        return {"rc": None, "stdout": "", "stderr": "", "secs": 0.0, "timed_out": False,
                "error": "cannot run %s: %s" % (os.path.basename(argv[0]), e.__class__.__name__)}
    timed_out = False
    try:
        out, err = p.communicate(stdin_text.encode("utf-8") if stdin_text is not None else None, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill(p)
        try:
            out, err = p.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            out, err = b"", b""
    return {"rc": p.returncode, "stdout": out.decode("utf-8", "replace"), "stderr": err.decode("utf-8", "replace"),
            "secs": round(time.monotonic() - t0, 2), "timed_out": timed_out, "error": ""}


# ── output parsers: (text, usage {"in", "out"} or None, cost USD or None, extra) ──────────────────

def _claude(stdout):
    j = json.loads(stdout)
    if isinstance(j, list):
        j = next((x for x in reversed(j) if isinstance(x, dict) and x.get("type") == "result"), {})
    if j.get("is_error"):
        raise RuntimeError(m_scrub(j.get("result") or j.get("subtype") or "claude reported an error"))
    u = j.get("usage") or {}
    usage = {"in": sum(int(u.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")),
             "out": int(u.get("output_tokens") or 0)} if u else None
    cost = j.get("total_cost_usd")
    return (j.get("result") or "").strip(), usage, float(cost) if isinstance(cost, (int, float)) else None, \
        {"turns": j.get("num_turns"), "models": sorted((j.get("modelUsage") or {}).keys())}


def _codex(stdout):
    text, usage, seen = "", None, False
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict):
            continue
        seen = True
        item = ev.get("item") or {}
        if ev.get("type") == "item.completed" and item.get("type") in ("agent_message", "assistant_message"):
            text = item.get("text") or text
        elif ev.get("type") == "turn.completed" and isinstance(ev.get("usage"), dict):
            u = ev["usage"]
            usage = usage or {"in": 0, "out": 0}
            usage["in"] += int(u.get("input_tokens") or 0)
            usage["out"] += int(u.get("output_tokens") or 0)
        elif ev.get("type") in ("turn.failed", "error"):
            msg = (ev.get("error") or {}).get("message") if isinstance(ev.get("error"), dict) else ev.get("message")
            raise RuntimeError(m_scrub(str(msg or "codex reported an error")))
    if not seen:
        raise ValueError("not JSONL")
    return text.strip(), usage, None, {}


def _gemini(stdout):
    j = json.loads(stdout)
    if not isinstance(j, dict):
        raise ValueError("not an object")
    if j.get("error"):
        e = j["error"]
        raise RuntimeError(m_scrub(str(e.get("message") if isinstance(e, dict) else e)))
    usage = None
    for mod in ((j.get("stats") or {}).get("models") or {}).values():
        t = (mod or {}).get("tokens") or {}
        usage = usage or {"in": 0, "out": 0}
        usage["in"] += int(t.get("prompt") or 0)
        usage["out"] += int(t.get("candidates") or 0)
    return str(j.get("response") or "").strip(), usage, None, {}


def _hermes(stdout):
    """Hermes -Q prints the answer, but some builds put startup warnings ("⚠ tirith security scanner ...") first."""
    lines = stdout.strip().splitlines()
    while lines and lines[0].lstrip().startswith("\u26a0"):
        lines.pop(0)
    return "\n".join(lines).strip(), None, None, {}


PARSERS = {"claude-json": _claude, "codex-jsonl": _codex, "gemini-json": _gemini, "hermes-text": _hermes,
           "text": lambda s: (s.strip(), None, None, {})}


def m_scrub(text):
    from maestro import providers as P
    return P.scrub(str(text))[:500]


def parse_output(kind, stdout):
    """Parse a CLI's stdout. Output that is not in the expected JSON shape falls back to plain text (noted in extra)."""
    try:
        return PARSERS[kind](stdout)
    except (ValueError, TypeError, AttributeError):
        return stdout.strip(), None, None, {"parse": "plain text (expected %s)" % kind}


# ── cost ──────────────────────────────────────────────────────────────────────

def cost_of(seat, usage, reported, prompt, text):
    """(USD, source). reported: the CLI's own number (Claude Code: API-equivalent dollars, also on a subscription);
    tokens: usage x catalog price; estimate: characters / 4 x catalog price; local: 0."""
    if reported is not None:
        return round(reported, 6), "reported"
    if seat.get("harness") == "local" or (seat.get("endpoint") and seat["endpoint"].get("local", True)):
        return 0.0, "local"
    pin, pout = seat.get("price_in"), seat.get("price_out")
    if pin is None or pout is None:
        return None, "unknown"
    tin, tout, src = (usage["in"], usage["out"], "tokens") if usage else (est_tokens(prompt), est_tokens(text), "estimate")
    return round((tin * pin + tout * pout) / 1e6, 6), src


# ── one chat answer from any seat ─────────────────────────────────────────────

def ask(seat, prompt, timeout=300.0, env=None, home=None, cfg=None, inventory=None, which=None):
    """One chat answer: {"ok", "text", "usage", "cost_usd", "cost_source", "secs", "error"}. Never raises.
    CLI seats run their `chat` row in an empty temp dir; `api` seats go through maestro.clients (their own provider only);
    `local` seats through `maestro-router run` (refuses models that are not loaded) or, with a base_url, one direct
    OpenAI-compatible call to that endpoint."""
    env = os.environ if env is None else env
    t0 = time.monotonic()
    usage, reported, extra = None, None, {}
    try:
        h = seat["harness"]
        if h in HARNESSES:
            exe = (which or (lambda e: shutil.which(e, path=env.get("PATH"))))(EXE[h])
            if not exe:
                raise RuntimeError("`%s` is not on PATH" % EXE[h])
            with tempfile.TemporaryDirectory(prefix="arena-chat-") as d:
                scratch = os.path.join(d, "scratch")
                work = os.path.join(d, "work")
                os.makedirs(scratch)
                os.makedirs(work)
                argv, stdin = build_argv(seat, "chat", prompt, exe)
                r = run_cli(argv, work, harness_env(seat, scratch, env), stdin, timeout)
            if r["error"]:
                raise RuntimeError(r["error"])
            if r["timed_out"]:
                raise RuntimeError("timed out after %.0f s" % timeout)
            if r["rc"] != 0:
                raise RuntimeError("exit %s: %s" % (r["rc"], m_scrub((r["stderr"] or r["stdout"]).strip()[-300:])))
            text, usage, reported, extra = parse_output(HARNESSES[h]["out"], r["stdout"])
        elif h == "api":
            text, usage = _ask_api(seat, prompt, timeout, env, home, cfg)
        elif h == "local":
            text, usage = _ask_local(seat, prompt, timeout, env, home, cfg, inventory)
        else:
            raise RuntimeError("unknown harness %r" % h)
        if not (text or "").strip():
            raise RuntimeError("empty reply")
    except RuntimeError as e:
        return {"ok": False, "text": "", "usage": None, "cost_usd": None, "cost_source": "", "secs": round(time.monotonic() - t0, 2),
                "error": m_scrub(e)}
    cost, src = cost_of(seat, usage, reported, prompt, text)
    out = {"ok": True, "text": text, "usage": usage, "cost_usd": cost, "cost_source": src,
           "secs": round(time.monotonic() - t0, 2), "error": ""}
    if extra:
        out["extra"] = extra
    return out


def _ask_api(seat, prompt, timeout, env, home, cfg):
    from maestro import clients, quota
    req = {"messages": [{"role": "user", "content": prompt}]}
    if seat.get("effort"):
        req["reasoning_effort"] = seat["effort"]
    status, headers, body = clients.complete(seat["provider"], seat["model"], req, env=env, home=home, timeout=timeout, cfg=cfg)
    try:
        quota.record(seat["provider"], seat["model"], status, headers, home)
    except Exception:  # quota bookkeeping never fails a cell
        pass
    if status != 200 or not isinstance(body, dict):
        err = (body or {}).get("error") if isinstance(body, dict) else None
        raise RuntimeError("HTTP %s: %s" % (status, (err or {}).get("message") if isinstance(err, dict) else err or "no answer"))
    u = body.get("usage") or {}
    usage = {"in": int(u.get("prompt_tokens") or 0), "out": int(u.get("completion_tokens") or 0)} if u else None
    return ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "", usage


def _ask_local(seat, prompt, timeout, env, home, cfg, inventory):
    if not seat.get("base_url"):
        try:
            return m.run_model(seat["model"], prompt, cfg=cfg, inv=inventory, env=env, timeout=timeout), None
        except RuntimeError as e:
            raise RuntimeError(str(e))
    headers = {"Content-Type": "application/json"}
    key_env = seat.get("api_key_env")
    if key_env and env.get(key_env):
        headers["Authorization"] = "Bearer " + env[key_env]   # only to this seat's own endpoint
    body = {"model": seat.get("served") or seat["model"], "messages": [{"role": "user", "content": prompt}]}
    try:
        status, raw = m._local_fetch(seat["base_url"].rstrip("/") + "/chat/completions", headers,
                                     json.dumps(body).encode("utf-8"), timeout)
    except Exception as e:
        raise RuntimeError("request to %s failed: %s" % (seat["base_url"], e.__class__.__name__))
    if status != 200:
        raise RuntimeError("HTTP %s from %s" % (status, seat["base_url"]))
    try:
        data = json.loads(raw)
        u = data.get("usage") or {}
        usage = {"in": int(u.get("prompt_tokens") or 0), "out": int(u.get("completion_tokens") or 0)} if u else None
        return data["choices"][0]["message"]["content"] or "", usage
    except (ValueError, KeyError, IndexError, TypeError):
        raise RuntimeError("unexpected reply from %s" % seat["base_url"])
