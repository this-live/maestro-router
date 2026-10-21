"""Clients for the proxy: one interface over three wire formats, all normalized to OpenAI chat.completions.

    complete(provider, model, request, stream=False, ...) -> (status, headers, body)

`request` is an OpenAI chat.completions request dict; `model` is the upstream model id (it replaces request["model"]).
  status   the upstream HTTP status; 0 means the request never got an answer (timeout, refused, DNS)
  headers  upstream response headers, lower-case names (feed them to quota.record)
  body     stream=False, or any status >= 400: a dict (the OpenAI response, or {"error": {...}}).
           stream=True and 2xx: a `Stream`, an iterator of bytes that concatenate to a valid SSE stream
           (`data: {...}\\n\\n` ... `data: [DONE]\\n\\n`). Nothing is read from the network until the first next(),
           so a caller can pull one item inside its try block to fall back before any byte reaches its own client.
           Stream.usage / Stream.finish_reason fill in as chunks pass. Failures after the first item raise UpstreamError.
Styles: openai (also Azure, Ollama and every OpenAI-compatible server), anthropic (Messages, translated both ways),
cli (subscription CLIs, chat only). Keys are used only for the auth headers of their own provider.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid

from . import providers as P
from . import router as core

DEFAULT_MAX_TOKENS = 16000
STOP_REASONS = {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls", "max_tokens": "length",
                "refusal": "content_filter", "pause_turn": "stop", "model_context_window_exceeded": "length"}
EFFORTS = ("low", "medium", "high", "xhigh", "max")


class UpstreamError(Exception):
    """A failure after the response started (mid-stream). The message is safe to show."""


class Stream:
    """An iterator of bytes with `usage` and `finish_reason` filled in as the stream is read."""

    def __init__(self, gen, close=None):
        self.usage = None
        self.finish_reason = None
        self._it = gen(self)
        self._close = close

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def close(self):
        if self._close:
            self._close()
            self._close = None


def error_body(message, kind="upstream_error", code=None):
    return {"error": {"message": message, "type": kind, "code": code}}


def _sse(obj):
    return b"data: " + json.dumps(obj).encode("utf-8") + b"\n\n"


def _upstream_error(status, data, *secrets):
    """The normalized error body of a non-2xx answer: the upstream JSON when it has an `error`, else a message.
    Any secret value is blanked out of the result, wherever the provider echoed it."""
    try:
        j = json.loads(data)
        if isinstance(j, dict) and j.get("error"):
            if isinstance(j["error"], str):
                j["error"] = {"message": j["error"], "type": "upstream_error"}
            text = json.dumps(j)
            for sec in secrets:
                text = text.replace(json.dumps(sec)[1:-1], "***") if sec else text
            return json.loads(text)
    except ValueError:
        pass
    return error_body(P.error_message(data, *secrets) or "HTTP %d" % status, code=status)


def _usage(inp, out):
    return {"prompt_tokens": inp, "completion_tokens": out, "total_tokens": inp + out}


def _text_of(content):
    """Plain text of a message `content` (a string, a list of parts, or None)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text") or "" for p in content if isinstance(p, dict) and p.get("type") in (None, "text"))
    return ""


# ════════════════════════════════════════════════════════════════════════════
# 1. openai-compatible
# ════════════════════════════════════════════════════════════════════════════

def _relay(resp, st):
    """Pass an SSE body through line by line, noting the usage block and the finish reason on the way. A stream that ends
    without `[DONE]` and without any chunk carrying a `finish_reason` was cut off: that raises UpstreamError."""
    ended = False
    try:
        for line in resp:
            if line.startswith(b"data:"):
                if line[5:].strip() == b"[DONE]":
                    ended = True
                elif b'"usage"' in line or b'"finish_reason"' in line:
                    try:
                        obj = json.loads(line[5:])
                        u = obj.get("usage")
                        st.usage = u if isinstance(u, dict) else st.usage
                        for ch in obj.get("choices") or []:
                            if ch.get("finish_reason"):
                                ended, st.finish_reason = True, ch["finish_reason"]
                    except (ValueError, AttributeError):
                        pass
            yield line
    except (OSError, ValueError) as e:
        raise UpstreamError("stream interrupted: %s" % e.__class__.__name__)
    finally:
        resp.close()
    if not ended:
        raise UpstreamError("the upstream stream ended early (no [DONE] and no finish_reason)")


def _post(spec, url, headers, payload, timeout, stream, retry_key=None, secrets=()):
    """POST JSON; on a 400 that names `retry_key` (a parameter some models reject), drop it and retry once.
    Returns (status, headers, data-or-open-response); status 0 (and an error dict) for a transport failure."""
    while True:
        try:
            status, hdrs, data = P.http_request("POST", url, headers, json.dumps(payload).encode("utf-8"), timeout,
                                                P.is_direct(spec, url), stream)
        except OSError as e:
            return 0, {}, error_body(P.scrub("%s: %s" % (e.__class__.__name__, e), *secrets), "connection_error")
        if status == 400 and retry_key in payload and retry_key.encode() in data:
            payload = {k: v for k, v in payload.items() if k != retry_key}  # the key is gone, so this retries once
            continue
        return status, hdrs, data


def _finish(status, hdrs, data, stream, secrets, relay=_relay, convert=None):
    """Turn a raw answer into the (status, headers, body) contract. `relay(resp, st)` yields the SSE bytes of a
    stream; `convert(body)` maps a non-streaming JSON body."""
    if not 200 <= status < 300:
        return status, hdrs, data if status == 0 else _upstream_error(status, data, *secrets)
    if stream:
        return status, hdrs, Stream(lambda st: relay(data, st), data.close)
    try:
        body = json.loads(data)
    except ValueError:
        return 502, hdrs, error_body("upstream sent a reply that is not JSON", code=502)
    return status, hdrs, convert(body) if convert else body


def complete_openai(spec, model, request, stream, base, key, timeout):
    payload = dict(request, model=model, stream=bool(stream))
    if not stream:
        payload.pop("stream_options", None)
    if not (spec["id"] == "openai" or spec["id"].startswith("azure")) and isinstance(payload.get("messages"), list):
        # the `developer` role is OpenAI's; other servers reject it or ignore the message
        payload["messages"] = [dict(m, role="system") if isinstance(m, dict) and m.get("role") == "developer" else m for m in payload["messages"]]
    headers = dict(P.auth_headers(spec, key), **{"Content-Type": "application/json"})
    status, hdrs, data = _post(spec, base + "/chat/completions", headers, payload, timeout, stream, "reasoning_effort", (key,))
    return _finish(status, hdrs, data, stream, (key,))


# ════════════════════════════════════════════════════════════════════════════
# 2. anthropic Messages <-> OpenAI chat
# ════════════════════════════════════════════════════════════════════════════

def _image_block(part):
    url = part.get("image_url")
    url = url.get("url") if isinstance(url, dict) else url
    m = re.match(r"data:([\w/+.-]+);base64,(.*)$", url or "", re.S)
    if m:
        return {"type": "image", "source": {"type": "base64", "media_type": m.group(1), "data": m.group(2)}}
    return {"type": "image", "source": {"type": "url", "url": url}} if url else None


def _user_blocks(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    out = []
    for p in content if isinstance(content, list) else []:
        if not isinstance(p, dict):
            continue
        b = {"type": "text", "text": p.get("text") or ""} if p.get("type") in (None, "text") else \
            _image_block(p) if p.get("type") == "image_url" else None
        if b and (b["type"] != "text" or b["text"]):
            out.append(b)
    return out


def _tool_input(arguments):
    try:
        v = json.loads(arguments) if isinstance(arguments, str) and arguments.strip() else arguments or {}
    except ValueError:
        v = {}
    return v if isinstance(v, dict) else {"value": v}


def _tool_choice(tc):
    if tc in ("auto", "required", "none"):
        return {"type": {"auto": "auto", "required": "any", "none": "none"}[tc]}
    if isinstance(tc, dict) and isinstance((tc.get("function") or {}).get("name"), str):
        return {"type": "tool", "name": tc["function"]["name"]}
    return None


def to_anthropic_request(req, model):
    """OpenAI chat.completions request -> Anthropic Messages request body."""
    system, msgs, after_tool = [], [], False
    for m in req.get("messages") or []:
        role = m.get("role")
        if role in ("system", "developer"):
            system.append(_text_of(m.get("content")))
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.get("tool_call_id"), "content": _text_of(m.get("content"))}
            if after_tool:
                msgs[-1]["content"].append(block)  # consecutive results share one user message
            else:
                msgs.append({"role": "user", "content": [block]})
            after_tool = True
            continue
        elif role == "assistant":
            blocks = _user_blocks(m.get("content"))
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                blocks.append({"type": "tool_use", "id": tc.get("id"), "name": fn.get("name"), "input": _tool_input(fn.get("arguments"))})
            if blocks:
                msgs.append({"role": "assistant", "content": blocks})
        else:
            blocks = _user_blocks(m.get("content"))
            if blocks:
                msgs.append({"role": "user", "content": blocks})
        after_tool = False
    body = {"model": model, "max_tokens": req.get("max_tokens") or req.get("max_completion_tokens") or DEFAULT_MAX_TOKENS,
            "messages": msgs}
    if any(system):
        body["system"] = "\n\n".join(s for s in system if s)
    tools = [{"name": t["function"]["name"], "description": t["function"].get("description") or "",
              "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}}}
             for t in req.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict)]
    if tools:
        body["tools"] = tools
        tc = _tool_choice(req.get("tool_choice"))
        if tc:
            body["tool_choice"] = tc
    if req.get("temperature") is not None:
        body["temperature"] = req["temperature"]
    elif req.get("top_p") is not None:
        body["top_p"] = req["top_p"]  # newer models take one of the two
    stop = req.get("stop")
    if stop:
        body["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    effort = {"minimal": "low"}.get(req.get("reasoning_effort"), req.get("reasoning_effort"))
    if effort in EFFORTS and "haiku" not in model:
        body["output_config"] = {"effort": effort}  # no `thinking` param: current models decide adaptively
    return body


def from_anthropic_response(resp, model):
    """Anthropic Messages response -> OpenAI chat.completion (thinking blocks are dropped)."""
    text, calls = [], []
    for b in resp.get("content") or []:
        if b.get("type") == "text":
            text.append(b.get("text") or "")
        elif b.get("type") == "tool_use":
            calls.append({"id": b.get("id"), "type": "function",
                          "function": {"name": b.get("name"), "arguments": json.dumps(b.get("input") or {})}})
    msg = {"role": "assistant", "content": "".join(text) if text or not calls else None}
    if calls:
        msg["tool_calls"] = calls
    u = resp.get("usage") or {}
    inp = sum(u.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    return {"id": "chatcmpl-" + str(resp.get("id") or uuid.uuid4().hex), "object": "chat.completion", "created": int(time.time()),
            "model": model, "choices": [{"index": 0, "message": msg,
                                         "finish_reason": STOP_REASONS.get(resp.get("stop_reason"), "stop")}],
            "usage": _usage(inp, u.get("output_tokens") or 0)}


def _anthropic_events(resp):
    """Parsed JSON events from an Anthropic SSE body (`event:` lines are redundant: every data has its `type`)."""
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("data:"):
            try:
                yield json.loads(line[5:])
            except ValueError:
                continue


def anthropic_stream_to_openai(events, model, include_usage=False, st=None):
    """Generator of OpenAI chat.completion.chunk SSE events (bytes) from Anthropic streaming events."""
    st = st or Stream(lambda s: iter(()))
    cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
    started, tools, inp, out, done = False, {}, 0, 0, False

    def chunk(delta, finish=None):
        nonlocal started
        if not started:
            delta = dict({"role": "assistant"}, **delta)
            started = True
        return _sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

    for ev in events:
        t = ev.get("type")
        if t == "message_start":
            msg = ev.get("message") or {}
            cid = "chatcmpl-" + str(msg.get("id") or cid[9:])
            u = msg.get("usage") or {}
            inp = sum(u.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
            out = u.get("output_tokens") or 0
        elif t == "content_block_start":
            cb = ev.get("content_block") or {}
            if cb.get("type") == "tool_use":
                tools[ev.get("index")] = len(tools)
                yield chunk({"tool_calls": [{"index": tools[ev.get("index")], "id": cb.get("id"), "type": "function",
                                             "function": {"name": cb.get("name"), "arguments": ""}}]})
            elif cb.get("type") == "text" and cb.get("text"):
                yield chunk({"content": cb["text"]})
        elif t == "content_block_delta":
            d = ev.get("delta") or {}
            if d.get("type") == "text_delta" and d.get("text"):
                yield chunk({"content": d["text"]})
            elif d.get("type") == "input_json_delta" and ev.get("index") in tools:
                yield chunk({"tool_calls": [{"index": tools[ev["index"]], "function": {"arguments": d.get("partial_json") or ""}}]})
            # thinking_delta / signature_delta: not exposed
        elif t == "message_delta":
            u = ev.get("usage") or {}
            out = u.get("output_tokens") or out
            inp = u.get("input_tokens") or inp
            st.finish_reason = STOP_REASONS.get((ev.get("delta") or {}).get("stop_reason"), "stop")
            yield chunk({}, st.finish_reason)
        elif t == "message_stop":
            st.usage = _usage(inp, out)
            if include_usage:
                yield _sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                            "choices": [], "usage": st.usage})
            yield b"data: [DONE]\n\n"
            done = True
        elif t == "error":
            raise UpstreamError(P.scrub((ev.get("error") or {}).get("message") or "upstream error"))
    if not done:
        raise UpstreamError("stream ended before message_stop")


def _anthropic_relay(resp, model, include_usage, st):
    try:
        yield from anthropic_stream_to_openai(_anthropic_events(resp), model, include_usage, st)
    except (OSError, ValueError) as e:
        raise UpstreamError("stream interrupted: %s" % e.__class__.__name__)
    finally:
        resp.close()


def complete_anthropic(spec, model, request, stream, base, key, timeout):
    payload = to_anthropic_request(request, model)
    if stream:
        payload["stream"] = True
    headers = dict(P.auth_headers(spec, key), **{"Content-Type": "application/json"})
    status, hdrs, data = _post(spec, base + "/messages", headers, payload, timeout, stream, "output_config", (key,))
    include = bool((request.get("stream_options") or {}).get("include_usage"))
    return _finish(status, hdrs, data, stream, (key,), lambda resp, st: _anthropic_relay(resp, model, include, st),
                   lambda body: from_anthropic_response(body, model))


# ════════════════════════════════════════════════════════════════════════════
# 3. subscription CLIs (chat only)
# ════════════════════════════════════════════════════════════════════════════

def cli_supports(request):
    """(True, "") when a subscription CLI can serve this request, else (False, reason). Tools and images need an API seat."""
    if request.get("tools") or request.get("functions"):
        return False, "subscription CLI seats cannot use tools"
    for m in request.get("messages") or []:
        if m.get("role") == "tool" or m.get("tool_calls"):
            return False, "subscription CLI seats cannot continue a tool conversation"
        if isinstance(m.get("content"), list) and any(isinstance(p, dict) and p.get("type") not in (None, "text") for p in m["content"]):
            return False, "subscription CLI seats cannot take images"
    return True, ""


def flatten_messages(messages):
    """One prompt for a CLI: a lone user message as is; otherwise the system prompt as a prefix section, then the turns."""
    msgs = [(m.get("role"), _text_of(m.get("content"))) for m in messages or []]
    system = "\n\n".join(t for r, t in msgs if r in ("system", "developer") and t)
    turns = [(r, t) for r, t in msgs if r not in ("system", "developer") and t]
    if len(turns) == 1 and turns[0][0] == "user" and not system:
        return turns[0][1]
    parts = ["[System instructions]\n" + system] if system else []
    parts += ["[%s]\n%s" % ("Assistant" if r == "assistant" else "User", t) for r, t in turns]
    return "\n\n".join(parts)


# The proxy runs subscription CLIs as chat models, so every one starts with its own tools, plugins, MCP servers and the
# user's permission settings switched off, as the arena's chat rows do (evals/arena/harness.py). The restricting flags
# are always passed: a CLI too old to know one fails, and the request falls back, instead of running unrestricted. Flags
# that only trim further are added when the installed CLI's --help lists them. The prompt always goes on stdin, never on
# the command line, where other local users could read it and long prompts would not fit.
CODEX_OFF = ("shell_tool", "unified_exec", "view_image", "apps", "plugins", "hooks", "memories", "multi_agent",
             "browser_use", "browser_use_external", "computer_use", "image_generation", "in_app_local_automation",
             "skill_mcp_dependency_install", "skill_search", "workspace_dependencies")
_CLI_CAPS = {}
# Gemini's restricting flags are not always available, so it runs only when its --help confirms all of them (Claude Code's
# and Codex's are always passed: a CLI that rejects one fails instead of running unrestricted).
GEMINI_REQUIRED = frozenset(("--approval-mode", "--extensions", "--allowed-mcp-server-names"))


def cli_caps(cli_id, exe, env=None, timeout=5.0):
    """The optional hardening flags this installed CLI understands (from its --help; cached per executable, a failed or
    slow probe too: the restricting flags never depend on it)."""
    try:
        key = (exe, os.stat(exe).st_mtime)
    except OSError:
        return set()
    if key not in _CLI_CAPS:
        sub = {"codex": ["exec"]}.get(cli_id, [])
        try:
            r = subprocess.run([exe] + sub + ["--help"], capture_output=True, text=True, timeout=min(5.0, timeout),
                               env=cli_env(cli_id, env),
                               stdin=subprocess.DEVNULL)
            text = (r.stdout or "") + (r.stderr or "")
        except (OSError, subprocess.SubprocessError):
            text = ""
        _CLI_CAPS[key] = {f for f in ("--safe-mode", "--no-session-persistence", "--ignore-user-config", "--ephemeral",
                                      "--approval-mode", "--extensions", "--allowed-mcp-server-names") if f in text}
    return _CLI_CAPS[key]


def _cli_argv(cli_id, exe, model, prompt, caps=frozenset()):
    """(argv, stdin text). The prompt is always stdin; `caps` are the optional flags the installed CLI supports."""
    if cli_id == "claude":
        return ([exe, "-p"] + (["--safe-mode"] if "--safe-mode" in caps else []) + ["--model", model, "--tools", "",
                "--strict-mcp-config"] + (["--no-session-persistence"] if "--no-session-persistence" in caps else [])
                + ["--output-format", "json"]), prompt
    if cli_id == "codex":
        off = [x for f in CODEX_OFF for x in ("-c", "features.%s=false" % f)]
        return ([exe, "exec", "-m", model, "--sandbox", "read-only"] + off
                + [f for f in ("--ignore-user-config", "--ephemeral") if f in caps] + ["--skip-git-repo-check"]), prompt
    return ([exe, "-m", model] + (["--approval-mode", "default"] if "--approval-mode" in caps else [])
            + (["-e", "none"] if "--extensions" in caps else [])
            + (["--allowed-mcp-server-names", "maestro-none"] if "--allowed-mcp-server-names" in caps else [])
            + ["-p", " "]), prompt   # -p makes it headless; the prompt itself comes on stdin


# What a CLI seat may see of the server's environment: the basics, proxy and certificate settings, and its own vendor's
# variables. Other providers' keys and MAESTRO_* (including MAESTRO_API_KEY) are dropped.
CLI_ENV_KEEP = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "TERM", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy",
                "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS", "REQUESTS_CA_BUNDLE",
                "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME", "XDG_RUNTIME_DIR")
CLI_ENV_PREFIXES = {"claude": ("CLAUDE_", "ANTHROPIC_"), "codex": ("CODEX_", "OPENAI_"), "gemini": ("GEMINI_", "GOOGLE_")}
# Version-manager / launcher variables. CLIs are commonly installed through mise/asdf/volta/nvm/pyenv/rbenv shims or
# wrapper scripts that need their manager's env to resolve the real binary; stripping it makes the shim re-resolve
# (or hang) instead of running. These carry no provider credentials, so every CLI seat keeps them.
CLI_ENV_LAUNCHER_PREFIXES = ("MISE_", "__MISE_", "ASDF_", "VOLTA_", "NVM_", "PYENV_", "RBENV_", "FNM_")


def cli_env(cli_id, env=None):
    env = os.environ if env is None else env
    prefixes = CLI_ENV_LAUNCHER_PREFIXES + CLI_ENV_PREFIXES.get(cli_id, ())
    return {k: v for k, v in env.items() if k in CLI_ENV_KEEP or k.startswith(prefixes)}


def _cli_text(cli_id, stdout):
    """(text, usage or None) from a CLI's stdout; raises ValueError with a message on an error result."""
    if cli_id != "claude":
        return stdout.strip(), None
    j = json.loads(stdout)
    if isinstance(j, list):
        j = next((x for x in reversed(j) if isinstance(x, dict) and x.get("type") == "result"), {})
    if j.get("is_error"):
        raise ValueError(P.scrub(j.get("result") or "claude reported an error"))
    u = j.get("usage") or {}
    return (j.get("result") or "").strip(), _usage(u.get("input_tokens") or 0, u.get("output_tokens") or 0) if u else None


def complete_cli(spec, model, request, stream, env, timeout):
    ok, why = cli_supports(request)
    if not ok:
        return 400, {}, error_body(why, "unsupported_request", "cli_no_tools")
    exe = next((shutil.which(e, path=env.get("PATH")) for e in spec["exes"] if shutil.which(e, path=env.get("PATH"))), None)
    if not exe:
        return 503, {}, error_body("`%s` is not on PATH" % spec["exes"][0], code=503)
    prompt = flatten_messages(request.get("messages"))
    caps = cli_caps(spec["id"], exe, env, timeout)
    if spec["id"] == "gemini" and not GEMINI_REQUIRED <= caps:   # fail closed: unconfirmed means unrestricted
        return 503, {}, error_body("gemini cannot be run with its tools restricted: its --help did not confirm %s "
                                   "(update Gemini CLI)" % ", ".join(sorted(GEMINI_REQUIRED - caps)), code=503)
    argv, stdin = _cli_argv(spec["id"], exe, model, prompt, caps)
    try:
        with tempfile.TemporaryDirectory() as cwd:  # not the user's repo: this seat is chat only
            r = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=timeout, env=cli_env(spec["id"], env),
                               cwd=cwd)
    except subprocess.TimeoutExpired:
        return 504, {}, error_body("%s timed out after %.0f s" % (spec["exes"][0], timeout), "timeout", 504)
    except OSError as e:
        return 503, {}, error_body("cannot run %s: %s" % (spec["exes"][0], e.__class__.__name__), code=503)
    try:
        if r.returncode != 0:
            raise ValueError(P.scrub(r.stderr or "exit code %d" % r.returncode))
        text, usage = _cli_text(spec["id"], r.stdout)
        if not text:
            raise ValueError("empty reply")
    except ValueError as e:
        return 502, {}, error_body("%s failed: %s" % (spec["exes"][0], e), code=502)
    usage = usage or _usage(len(prompt) // 4, len(text) // 4)  # an estimate: only Claude Code reports usage
    cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
    if not stream:
        return 200, {}, {"id": cid, "object": "chat.completion", "created": created, "model": model, "usage": usage,
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}

    def gen(st):  # faked streaming: the whole answer as one chunk
        base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}
        yield _sse(dict(base, choices=[{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]))
        st.finish_reason, st.usage = "stop", usage
        yield _sse(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}]))
        if (request.get("stream_options") or {}).get("include_usage"):
            yield _sse(dict(base, choices=[], usage=usage))
        yield b"data: [DONE]\n\n"
    return 200, {}, Stream(gen)


# ════════════════════════════════════════════════════════════════════════════
# 4. the one entry point
# ════════════════════════════════════════════════════════════════════════════

def complete(provider, model, request, stream=False, env=None, home=None, timeout=None, cfg=None):
    """Send `request` (OpenAI chat.completions) to `provider` as `model`. See the module docstring for the result."""
    env = os.environ if env is None else env
    home = home or core.maestro_home(env)
    cfg = cfg if cfg is not None else core.load_config(home)
    try:
        spec = P.model_spec(P.resolve(provider), model)  # a model may override its provider's style / base URL
    except P.ProviderError as e:
        return 404, {}, error_body(str(e), "unknown_provider", 404)
    if spec["style"] == "cli":
        return complete_cli(spec, model, request, stream, env, timeout or cfg["cli_timeout_s"])
    if not spec.get("callable", True):
        return 501, {}, error_body("%s is detected but not callable yet: %s" % (spec["name"], spec.get("notes") or "unsupported API"),
                                   "not_implemented", 501)
    timeout = timeout or cfg["request_timeout_s"]
    if spec["style"] == "local":
        key, base = (env.get(spec["key_env"]) if spec.get("key_env") else None), spec["base_url"]
    else:
        name, _, missing = P.key_status(spec, env, home)
        key, base = P.get_key(spec["env"], env, home), P.base_url(spec, env, home, cfg)
        if not key or missing:
            return 401, {}, error_body("no API key for %s%s" % (spec["name"], " (also needs %s)" % ", ".join(missing) if missing else ""),
                                       "missing_credentials", 401)
        if not base:
            return 400, {}, error_body("no base URL for %s (a setting in its URL is unset)" % spec["name"], "missing_settings", 400)
    fn = complete_anthropic if spec["style"] == "anthropic" else complete_openai
    return fn(spec, model, request, stream, base, key, timeout)
