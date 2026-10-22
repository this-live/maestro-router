"""Providers: the registry, credentials, detection and live model listing (Maestro v3).

Everything here is slow-path (`detect`, `discover`, `keys`, the proxy); `route` never imports this module.
Secrets: an API key is read only to authenticate to its own provider (an auth header) and is never printed,
logged, stored anywhere but ~/.maestro/keys.env, or returned in any structure this module builds. Subscription
logins (Claude Code, Codex, Gemini CLI, ...) are detected by presence only: their tokens are never read.
"""
from __future__ import annotations

import concurrent.futures as cf
import http.client
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request

from . import router as core
from . import safe_http

UA = "maestro-router/" + core.VERSION
LIST_TIMEOUT = 8.0
KEYS_FILE = "keys.env"
PROVIDERS_FILE = "providers.json"
ANTHROPIC_VERSION = "2023-06-01"


class ProviderError(Exception):
    """A provider call failed; the message is safe to show (it never contains a key). `status`: the HTTP status, if any."""

    def __init__(self, message, status=None):
        Exception.__init__(self, message)
        self.status = status


# ════════════════════════════════════════════════════════════════════════════
# 1. registry: every provider in models.dev (data/providers.json, built by evals/build_providers.py)
# ════════════════════════════════════════════════════════════════════════════

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "providers.json")

# The wire style follows the AI SDK package models.dev names. Anything else is "unsupported" (detected and listed, not
# callable) unless a base URL is known, which is taken to be OpenAI-compatible.
STYLE_BY_NPM = {"@ai-sdk/openai-compatible": "openai", "@ai-sdk/openai": "openai", "@ai-sdk/anthropic": "anthropic"}

# Base URLs models.dev leaves out (their AI SDK packages have them built in). Anthropic-style bases include /v1;
# the clients append /messages and /models.
API_OVERRIDES = {
    "openai": ("https://api.openai.com/v1", "openai"),
    "anthropic": ("https://api.anthropic.com/v1", "anthropic"),
    "google": ("https://generativelanguage.googleapis.com/v1beta/openai", "openai"),
    "xai": ("https://api.x.ai/v1", "openai"),
    "mistral": ("https://api.mistral.ai/v1", "openai"),
    "groq": ("https://api.groq.com/openai/v1", "openai"),
    "cerebras": ("https://api.cerebras.ai/v1", "openai"),
    "togetherai": ("https://api.together.xyz/v1", "openai"),
    "deepinfra": ("https://api.deepinfra.com/v1/openai", "openai"),
    "cohere": ("https://api.cohere.ai/compatibility/v1", "openai"),
    "perplexity": ("https://api.perplexity.ai", "openai"),
    "vercel": ("https://ai-gateway.vercel.sh/v1", "openai"),
    "azure": ("https://${AZURE_RESOURCE_NAME}.openai.azure.com/openai/v1", "openai"),
}
# Settings variables (not secrets) a provider needs besides its key, where models.dev does not put them in `api`.
EXTRA_PARAMS = {
    "azure": ("AZURE_RESOURCE_NAME",), "azure-cognitive-services": ("AZURE_COGNITIVE_SERVICES_RESOURCE_NAME",),
    "cloudflare-ai-gateway": ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_GATEWAY_ID"),
    "google-vertex": ("GOOGLE_VERTEX_PROJECT", "GOOGLE_VERTEX_LOCATION"),
    "google-vertex-anthropic": ("GOOGLE_VERTEX_PROJECT", "GOOGLE_VERTEX_LOCATION"),
    "watsonx": ("WATSONX_AI_PROJECT_ID",), "privatemode-ai": ("PRIVATEMODE_ENDPOINT",), "amazon-bedrock": ("AWS_REGION",),
}
# "Configured" is normally any one key variable (plus the params). These need other combinations: OR of ANDs.
NEEDS = {
    "amazon-bedrock": [["AWS_BEARER_TOKEN_BEDROCK"], ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]],
    "google-vertex": [["GOOGLE_VERTEX_PROJECT", "GOOGLE_VERTEX_LOCATION"]],  # credentials come from gcloud's ADC
    "google-vertex-anthropic": [["GOOGLE_VERTEX_PROJECT", "GOOGLE_VERTEX_LOCATION"]],
}
# Providers whose only credential is a generic platform variable users set for other reasons (GITHUB_TOKEN for git and gh,
# AWS_* for the AWS CLI, GitLab and Vertex project settings): they count as detected only when named under config `providers`.
OPT_IN_ONLY = ("github-copilot", "gitlab", "huggingface", "amazon-bedrock", "google-vertex", "google-vertex-anthropic")
NO_LISTING = {"azure": "set providers.azure.models in config.json (listing deployments needs ARM)"}
STATIC_MODELS = {"perplexity": ("sonar", "sonar-pro", "sonar-reasoning-pro", "sonar-deep-research")}  # no listing endpoint
API_KEY_HEADER = ("azure",)   # sent as `api-key: KEY` instead of a Bearer token
# Extra key variables harnesses already use for the same endpoint (Hermes sets KIMI_CODING_API_KEY for the Kimi
# coding endpoint), so a key exported for another harness enables the provider here with no re-entry.
ENV_ALIASES = {"kimi-code-plan-cn": ("KIMI_CODING_API_KEY",), "kimi-code-plan-global": ("KIMI_CODING_API_KEY",)}
QUOTA_PROBES = {"openrouter": "openrouter", "deepseek": "deepseek"}


def _style(npm, api):
    return STYLE_BY_NPM.get(npm) or ("openai" if api else "unsupported")


def _spec(pid, raw):
    api, style = API_OVERRIDES.get(pid, (raw.get("api"), None))
    style = style or _style(raw.get("npm"), api)
    params = tuple(dict.fromkeys(list(raw.get("params") or []) + list(EXTRA_PARAMS.get(pid, ()))))
    keys = tuple(dict.fromkeys([e for e in raw.get("env") or [] if e not in params] + list(ENV_ALIASES.get(pid, ()))))
    over = {}
    for ov, ids in raw.get("o") or []:
        over.update((mid, ov) for mid in ids)
    return {"id": pid, "name": raw["name"], "env": keys, "params": params,
            "need": NEEDS.get(pid) or [[k] + list(params) for k in keys] or [list(params)],
            "base_url": api, "style": style, "models_path": None if pid in NO_LISTING or pid in STATIC_MODELS else "/models",
            "auth": "api-key" if pid in API_KEY_HEADER else "bearer", "quota": QUOTA_PROBES.get(pid), "callable": style != "unsupported",
            "static_models": STATIC_MODELS.get(pid), "doc": raw.get("doc"), "overrides": over,
            "notes": "" if style != "unsupported" else "detected, not yet callable"}


def load_registry(path=DATA_FILE):
    """The provider registry: a list of spec dicts (id, name, env, params, need, base_url, style, models_path, auth, quota,
    callable, overrides, doc, notes). `env` holds the variables that carry the key (any one), `params` the settings that
    must also be present; `need` is the OR-of-ANDs of variable names that means "configured"."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [_spec(pid, raw) for pid, raw in sorted(data["providers"].items())]


REGISTRY = load_registry()
BY_ID = {p["id"]: p for p in REGISTRY}

# Subscription CLIs. Logged in = a credentials file / env var exists (or, for Codex, `login status` exits 0, and for
# Claude Code on macOS the keychain item exists). Token contents are never read.
CLI_PROVIDERS = [
    {"id": "claude", "name": "Claude Code", "seat": "claude-code", "exes": ("claude",), "files": (".claude/.credentials.json",),
     "env": ("CLAUDE_CODE_OAUTH_TOKEN",), "keychain": "Claude Code-credentials", "callable": True},
    {"id": "codex", "name": "Codex", "seat": "codex", "exes": ("codex",), "files": (".codex/auth.json",),
     "login_cmd": ("login", "status"), "callable": True},
    {"id": "gemini", "name": "Gemini CLI", "seat": "gemini-cli", "exes": ("gemini",), "files": (".gemini/oauth_creds.json",),
     "callable": True},
    {"id": "copilot", "name": "GitHub Copilot CLI", "seat": "copilot", "exes": ("copilot",), "files": (".copilot/config.json",),
     "env": ("COPILOT_GITHUB_TOKEN",), "callable": False},
    {"id": "cursor-agent", "name": "Cursor Agent", "seat": "cursor-agent", "exes": ("cursor-agent", "agent"), "files": (),
     "presence_only": True, "callable": False},
    {"id": "pi", "name": "Pi (harness providers)", "seat": "pi", "exes": ("pi",), "files": (".pi/agent/auth.json",),
     "callable": True},
]
for _c in CLI_PROVIDERS:
    _c.update(style="cli", access="subscription-cli")
CLI_BY_ID = {c["id"]: c for c in CLI_PROVIDERS}


def model_spec(spec, model):
    """The spec that applies to one model. models.dev lets a model override its provider's npm package, base URL and
    request shape (an Anthropic-style model behind an OpenAI-style gateway, a Responses-only model, ...)."""
    ov = (spec.get("overrides") or {}).get(model)
    if not ov:
        return spec
    s = dict(spec)
    if ov.get("api"):
        s["base_url"] = ov["api"]
    if ov.get("npm"):
        s["style"] = STYLE_BY_NPM.get(ov["npm"], "unsupported")
    if ov.get("shape") == "responses":  # chat completions is all the clients speak
        s["style"], s["notes"] = "unsupported", "needs the Responses API, not implemented"
    s["callable"] = s["style"] != "unsupported"
    return s


def cli_models(cli_id):
    """The built-in model table of a subscription CLI (CLIs cannot list models): the suggestions, without repeats."""
    seat = {"claude": "claude-code", "codex": "codex", "gemini": "gemini-cli"}.get(cli_id)
    out = []
    for cell in (core.SUGGESTED_MODELS.get(seat) or {}).values():
        c = core._cell(cell)
        if c and c[0] not in out:
            out.append(c[0])
    return out


def parse_pi_models(text):
    """Model names ("provider/model") from `pi --list-models` output: a header row, then one model per line with the
    provider first. Only models Pi itself lists as available are returned, so this reflects what Pi can call."""
    out = []
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0] == "provider" or "/" in parts[1]:
            continue
        name = "%s/%s" % (parts[0], parts[1])
        if name not in out:
            out.append(name)
    return out


def pi_models(exe, env):
    """The models Pi can call right now (`pi --list-models`); empty on any failure (Pi stays merely detected)."""
    try:
        r = subprocess.run([exe, "--list-models"], capture_output=True, text=True, timeout=20, env=env,
                           stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_pi_models(r.stdout or "") if r.returncode == 0 else []


def resolve(provider):
    """A registry entry (dict, API or CLI) from an id or from the entry itself."""
    if isinstance(provider, dict):
        return provider
    spec = BY_ID.get(provider) or CLI_BY_ID.get(provider)
    if not spec:
        raise ProviderError("unknown provider %r" % (provider,))
    return spec


# ════════════════════════════════════════════════════════════════════════════
# 2. credentials: environment first, then ~/.maestro/keys.env (mode 0600)
# ════════════════════════════════════════════════════════════════════════════

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def keys_path(home=None):
    return os.path.join(home or core.maestro_home(), KEYS_FILE)


def read_keys_file(home=None):
    """{NAME: value} from keys.env: `NAME=value` lines, `#` comments, an optional `export `, optional quotes."""
    out = {}
    try:
        with open(keys_path(home), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, _, val = line[7:].strip().partition("=") if line.startswith("export ") else line.partition("=")
                name, val = name.strip(), val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                    val = val[1:-1]
                if _NAME_RE.match(name) and val:
                    out[name] = val
    except OSError:
        pass
    return out


def find_key(names, env=None, home=None, file_keys=None):
    """(env var name, source) of the first of `names` that is set, source "env" or "keys.env"; (None, None) if none.
    Presence only: the value is not returned."""
    env = os.environ if env is None else env
    for n in names:
        if env.get(n):
            return n, "env"
    file_keys = read_keys_file(home) if file_keys is None else file_keys
    for n in names:
        if file_keys.get(n):
            return n, "keys.env"
    return None, None


def get_key(names, env=None, home=None):
    """The key value, for an auth header to its own provider only. None when unset."""
    env = os.environ if env is None else env
    name, src = find_key(names, env, home)
    return None if not name else env[name] if src == "env" else read_keys_file(home)[name]


def keys_file_problem(home=None):
    """A warning when keys.env exists and other users can read it (POSIX), else ""."""
    try:
        mode = stat.S_IMODE(os.stat(keys_path(home)).st_mode)
    except OSError:
        return ""
    return "%s is readable by other users; run chmod 600 on it" % keys_path(home) if mode & 0o077 and os.name == "posix" else ""


def _write_keys_lines(lines, home):
    path = keys_path(home)
    core.ensure_home(os.path.dirname(path))
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("".join(l + "\n" for l in lines))
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _keys_lines(home):
    try:
        with open(keys_path(home), "r", encoding="utf-8") as f:
            return f.read().splitlines()
    except OSError:
        return []


def _line_name(line):
    line = line.strip()
    line = line[7:].strip() if line.startswith("export ") else line
    return line.partition("=")[0].strip() if "=" in line and not line.startswith("#") else None


def set_key(name, value, home=None):
    """Write NAME=value to keys.env (0600), replacing an existing line. Raises ValueError on a bad name or value."""
    if not _NAME_RE.match(name or ""):
        raise ValueError("%r is not an environment variable name" % (name,))
    value = (value or "").strip()
    if not value or any(c in value for c in "\r\n\0"):
        raise ValueError("the key is empty or contains a newline")
    lines = [l for l in _keys_lines(home) if _line_name(l) != name]
    _write_keys_lines(lines + ["%s=%s" % (name, value)], home)


def unset_key(names, home=None):
    """Remove these names from keys.env. Returns the names that were there."""
    old = _keys_lines(home)
    gone = [n for n in names if any(_line_name(l) == n for l in old)]
    if gone:
        _write_keys_lines([l for l in old if _line_name(l) not in gone], home)
    return gone


def key_env_name(arg):
    """The env var `maestro-router keys set <arg>` writes: a provider id -> its first env var; an ALL_CAPS name -> itself."""
    if arg in BY_ID:
        return BY_ID[arg]["env"][0]
    if re.match(r"^[A-Z][A-Z0-9_]*$", arg or ""):
        return arg
    raise ValueError("unknown provider %r (`maestro-router keys list --all` shows the ids; or pass an env var name such as OPENAI_API_KEY)" % arg)


def key_status(spec, env=None, home=None):
    """(name of the key variable, source "env" | "keys.env", missing) for a registry entry. The name is None when nothing
    of it is set; `missing` lists the variables still needed when only part of a combination is there (Azure with a key
    but no resource name), and is empty when the provider is fully configured. Presence only: no value is returned."""
    env = os.environ if env is None else env
    file_keys = read_keys_file(home)

    def where(n):
        return "env" if env.get(n) else "keys.env" if file_keys.get(n) else None
    best = None
    for group in spec["need"]:
        have = [n for n in group if where(n)]
        if group and len(have) == len(group):
            name = next((n for n in group if n in spec["env"]), group[0])
            return name, where(name), []
        if have and (best is None or len(have) > len(best[0])):
            best = (have, group)
    if best is None:
        return None, None, []
    have, group = best
    name = next((n for n in have if n in spec["env"]), have[0])
    return name, where(name), [n for n in group if n not in have]


def key_names(env=None, home=None):
    """[(variable name, "env" | "keys.env")] for every API provider whose key is set, deduplicated. Presence only."""
    out = {}
    for spec in REGISTRY:
        name, src, missing = key_status(spec, env, home)
        if name and not missing:
            out.setdefault(name, src)
    return sorted(out.items())


# ════════════════════════════════════════════════════════════════════════════
# 3. HTTP (urllib only; shared by listing, the clients and the quota probes)
# ════════════════════════════════════════════════════════════════════════════

def auth_headers(spec, key):
    """Request headers for a provider: its auth scheme and the User-Agent (some CDNs reject urllib's default)."""
    h = {"User-Agent": UA, "Accept": "application/json"}
    if not key:
        return h
    if spec["style"] == "anthropic":
        h.update({"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION})
    elif spec.get("auth") == "api-key":
        h["api-key"] = key
    else:
        h["Authorization"] = "Bearer " + key
    return h


def is_direct(spec, url):
    """Local and loopback servers are never behind the proxy."""
    return spec.get("style") == "local" or core._host_of(url) in core.LOOPBACK


def http_request(method, url, headers=None, body=None, timeout=LIST_TIMEOUT, direct=False, stream=False):
    """(status, response headers with lower-case names, data). data is the bytes of the body, or with stream=True
    and a 2xx status the open response (the caller iterates and closes it). Transport failures raise OSError."""
    req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        resp = safe_http.opener(direct).open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        try:
            data = e.read(1 << 20)
        finally:
            e.close()
        return e.code, {k.lower(): v for k, v in e.headers.items()}, data
    except http.client.HTTPException as e:
        raise OSError("%s: %s" % (e.__class__.__name__, e))
    hdrs = {k.lower(): v for k, v in resp.headers.items()}
    if stream:
        return resp.status, hdrs, resp
    try:
        return resp.status, hdrs, resp.read()
    except http.client.HTTPException as e:
        raise OSError("%s: %s" % (e.__class__.__name__, e))
    finally:
        resp.close()


ERROR_TEXT_LIMIT = 16384
ERROR_ESCAPE_ROUNDS = 4
JSON_ESCAPE = re.compile(r'(?:\\u[0-9a-fA-F]{4}){1,2}|\\["\\/bfnrt]')
OMITTED_ERROR = "upstream error omitted (***)"


def scrub(text, *secrets):
    """Remove echoed credentials, including nested JSON escapes, before shortening an error."""
    guarded = any(secrets)
    if guarded and len(text) > ERROR_TEXT_LIMIT:
        return OMITTED_ERROR
    text = text if isinstance(text, str) else text.decode("utf-8", "replace")
    for depth in range(ERROR_ESCAPE_ROUNDS + 1):
        for secret in secrets:
            if secret:
                text = text.replace(secret, "***").replace(json.dumps(secret)[1:-1], "***")
        if not guarded or not JSON_ESCAPE.search(text):
            break
        if depth == ERROR_ESCAPE_ROUNDS:
            return OMITTED_ERROR  # fail closed rather than retain a deeper encoded credential
        text = JSON_ESCAPE.sub(lambda match: json.loads('"' + match.group() + '"'), text)
    return re.sub(r"\s+", " ", text).strip()[:160]


def error_message(data, *secrets):
    """The human message of an upstream error body (JSON `error.message` / `message` / `error`, else the text)."""
    if any(secrets) and isinstance(data, (str, bytes)) and len(data) > ERROR_TEXT_LIMIT:
        return OMITTED_ERROR
    try:
        j = json.loads(data)
        e = j.get("error", j) if isinstance(j, dict) else j
        msg = e.get("message") or e.get("error") or e.get("detail") if isinstance(e, dict) else e
        return scrub(msg if isinstance(msg, str) else json.dumps(msg), *secrets)
    except (ValueError, AttributeError, TypeError):
        return scrub(data, *secrets)


# ════════════════════════════════════════════════════════════════════════════
# 4. provider endpoint, live model listing
# ════════════════════════════════════════════════════════════════════════════

def fill(template, env=None, home=None):
    """A URL template with its ${VAR} parts taken from the environment / keys.env; None if one is unset."""
    if not template:
        return None
    missing = []

    def sub(m):
        v = get_key((m.group(1),), env, home)
        if not v:
            missing.append(m.group(1))
        return (v or "").rstrip("/") if m.group(1).endswith("BASE_URL") else v or ""
    out = re.sub(r"\$\{(\w+)\}", sub, template)
    return None if missing else out


def base_url(spec, env=None, home=None, cfg=None):
    """The chat base URL: config `providers.<id>.base_url` > the registry's, with ${VAR} parts filled in. None when a part is unset."""
    over = ((cfg or {}).get("providers") or {}).get(spec["id"])
    if isinstance(over, dict) and isinstance(over.get("base_url"), str) and over["base_url"]:
        return over["base_url"].rstrip("/")
    return fill(spec["base_url"], env, home)


_NON_CHAT = re.compile(r"embed|whisper|tts|dall-e|moderation|transcri|realtime|rerank|imagen|veo-|sora|lyria|guard-|/flux|stable-diffusion",
                       re.I)


def _ids(data):
    """Model ids from a /models body: {"data": [...]}, {"models": [...]} or a bare list (Together)."""
    items = data.get("data") or data.get("models") or [] if isinstance(data, dict) else data if isinstance(data, list) else []
    out = []
    for it in items:
        mid = it.get("id") or it.get("name") if isinstance(it, dict) else it
        if isinstance(mid, str) and mid:
            out.append(mid[7:] if mid.startswith("models/") else mid)  # Google prefixes its ids
    return out


def list_models(provider, env=None, home=None, timeout=LIST_TIMEOUT, cfg=None, chat_only=True):
    """Model ids this account can use, from the provider itself. Raises ProviderError with a safe message.
    API providers: GET {base_url}{models_path} with the key. Subscription CLIs: the built-in table. Local: the server's list.
    Azure: `providers.azure.models` in the config (listing deployments needs ARM)."""
    env = os.environ if env is None else env
    spec = resolve(provider)
    cfg = cfg if cfg is not None else core.load_config(home or core.maestro_home(env))
    conf = (cfg.get("providers") or {}).get(spec["id"])
    conf = conf if isinstance(conf, dict) else {}
    if spec["style"] == "cli":
        return cli_models(spec["id"])
    if spec["style"] == "local":
        ep = {"url": spec["endpoint_url"], "kind": spec.get("kind"), "api_key_env": spec.get("key_env")}
        res = core.probe_endpoint(ep, None, min(timeout, 2.0), env)
        if not res:
            raise ProviderError("cannot reach %s" % spec["endpoint_url"])
        return sorted(m["name"] for m in res["models"])
    if not spec["callable"]:
        raise ProviderError(spec["notes"] or "not callable")
    name, src, missing = key_status(spec, env, home)
    if not name:
        raise ProviderError("no API key (%s)" % " or ".join(spec["env"]))
    if missing:
        raise ProviderError("also needs %s" % ", ".join(missing))
    if isinstance(conf.get("models"), list) and conf["models"]:
        return [m for m in conf["models"] if isinstance(m, str)]
    if spec.get("static_models"):
        return list(spec["static_models"])
    if not spec["models_path"]:
        raise ProviderError(NO_LISTING.get(spec["id"]) or "set providers.%s.models in config.json (no listing endpoint)" % spec["id"])
    base = base_url(spec, env, home, cfg)
    if not base:
        raise ProviderError("the base URL needs %s" % ", ".join(v for v in re.findall(r"\$\{(\w+)\}", spec["base_url"] or "")))
    key = get_key(spec["env"], env, home)
    ids, url, pages = [], base + spec["models_path"], 0
    while url and pages < 5:
        pages += 1
        if spec["style"] == "anthropic":
            url += ("&" if "?" in url else "?") + "limit=1000"
        try:
            status, _, data = http_request("GET", url, auth_headers(spec, key), None, timeout, is_direct(spec, url))
        except OSError as e:
            raise ProviderError("request failed: %s" % scrub(str(e), key))
        if status != 200:
            raise ProviderError("HTTP %d: %s" % (status, error_message(data, key)), status)
        try:
            body = json.loads(data)
        except ValueError:
            raise ProviderError("unexpected reply (not JSON)")
        ids += _ids(body)
        more = isinstance(body, dict) and body.get("has_more") and body.get("last_id")
        url = (base + spec["models_path"] + "?after_id=" + str(body["last_id"])) if more else None
    ids = sorted(set(ids))
    return [m for m in ids if not _NON_CHAT.search(m)] if chat_only else ids


# ════════════════════════════════════════════════════════════════════════════
# 5. detection
# ════════════════════════════════════════════════════════════════════════════

def _rc(argv, env, timeout=4.0):
    """Exit code of a command with all output discarded, or None if it cannot run."""
    try:
        return subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=timeout, env=env).returncode
    except (OSError, subprocess.SubprocessError):
        return None


def detect_cli(spec, env, user_home, which, platform=None, run=None):
    """A provider entry for one subscription CLI. Presence only: nothing inside a credentials file is read."""
    run = run or (lambda argv: _rc(argv, env))
    platform = platform or sys.platform
    exe = next((which(e) for e in spec["exes"] if which(e)), None)
    why = ""
    if any(env.get(v) for v in spec.get("env", ())):
        why = "token in %s" % next(v for v in spec["env"] if env.get(v))
    elif any(os.path.exists(os.path.join(user_home, f)) for f in spec["files"]):
        why = "credentials file"
    elif spec.get("keychain") and platform == "darwin" and run(["security", "find-generic-password", "-s", spec["keychain"]]) == 0:
        why = "macOS keychain item"
    elif exe and spec.get("login_cmd") and run([exe] + list(spec["login_cmd"])) == 0:
        why = "`%s %s` succeeded" % (os.path.basename(exe), " ".join(spec["login_cmd"]))
    elif exe and spec.get("presence_only"):
        why = "installed"
    entry = {"id": spec["id"], "name": spec["name"], "access": "subscription-cli", "style": "cli", "status": "missing",
             "reason": "", "models": [], "key_source": None, "seat": spec["seat"], "cli_path": exe}
    if exe and why and spec["callable"]:
        models = pi_models(exe, env) if spec["id"] == "pi" else cli_models(spec["id"])
        entry.update(status="connected", reason="logged in (%s)" % why, models=models)
    elif exe and why:
        entry.update(status="detected", reason="found (%s); no non-interactive client yet" % why)
    elif why:
        entry.update(status="detected", reason="logged in (%s) but `%s` is not on PATH" % (why, spec["exes"][0]))
    elif exe:
        entry["reason"] = "installed, not logged in"
    else:
        entry["reason"] = "not installed"
    return entry


def _api_entry(spec, env, home, cfg):
    """The entry for an API provider before any network call: key presence, nothing else."""
    name, src, missing = key_status(spec, env, home)
    e = {"id": spec["id"], "name": spec["name"], "access": "api", "style": spec["style"], "status": "missing",
         "reason": "no key (%s)" % " or ".join(spec["env"][:2]), "models": [], "key_source": src, "key_env": name,
         "base_url": base_url(spec, env, home, cfg)}
    if name and missing:
        e["reason"] = "also needs %s" % ", ".join(missing)
    elif name:
        e.update(status="detected", reason="key set; not verified" if spec["callable"] else spec["notes"])
    return e


def _hold_shared(entries, cfg):
    """Detected providers that only look configured: opt-in-only ones (generic platform credentials) unless named under config
    `providers`, and, of several providers satisfied by the same key variable (MINIMAX_API_KEY -> minimax, minimax-cn, ...),
    all but one: the ones named in config, else the primary (the id that is a prefix of the others, else the shortest).
    Held entries become status "missing" with the reason, so nothing lists them or sends them requests."""
    named = {k for k, v in ((cfg or {}).get("providers") or {}).items() if isinstance(v, dict)}
    groups = {}
    for e in entries:
        if e["status"] != "detected":
            continue
        if e["id"] in OPT_IN_ONLY and e["id"] not in named:
            e.update(status="missing", held=True, reason="%s is a generic credential; name %s under config providers to use it"
                     % (e["key_env"], e["id"]))
        else:
            groups.setdefault(tuple(map(tuple, BY_ID[e["id"]]["need"])), []).append(e)
    for members in groups.values():
        if len(members) < 2:
            continue
        ids = sorted((m["id"] for m in members), key=lambda i: (len(i), i))
        keep = [i for i in ids if i in named] or [next((i for i in ids if all(o.startswith(i) for o in ids)), ids[0])]
        for m in members:
            if m["id"] not in keep:
                m.update(status="missing", held=True, shares_with=keep[0],
                         reason="shares %s with %s; name it under config providers to use it" % (m["key_env"], keep[0]))


def _local_entries(inv, cfg, env):
    keys = {e["url"]: e.get("api_key_env") for e in core.local_endpoints(cfg, env)}
    out = []
    for ep in inv.get("local_endpoints") or []:
        mods = [m for m in inv.get("local_models") or [] if m["url"] == ep["url"]]
        names = [m["name"] for m in mods]
        # raw ids as the server reports them (often Hugging Face repo ids), plus the context window vLLM/SGLang give
        info = {m["name"]: {k: m[k] for k in ("max_model_len", "root", "params_b", "loaded") if m.get(k) is not None} for m in mods}
        out.append({"id": "local:" + ep["name"], "name": "Local: " + ep["name"], "access": "local", "style": "local",
                    "status": "connected", "reason": "%s at %s" % (ep["kind"], ep["url"]), "models": sorted(names),
                    "key_source": None, "kind": ep["kind"], "endpoint_url": ep["url"], "key_env": keys.get(ep["url"]),
                    "base_url": ep["url"] + "/v1", "model_info": info})
    return out


def _verify(e, env, home, cfg, timeout):
    """(provider id, model ids or None, error message, HTTP status or None)."""
    try:
        return e["id"], list_models(BY_ID[e["id"]], env, home, timeout, cfg), "", None
    except ProviderError as err:
        return e["id"], None, str(err), err.status


def harness_providers(env=None, user_home=None):
    """Providers other harnesses already carry, by name only: Pi's authenticated providers (~/.pi/agent/auth.json
    keys and models.json provider names) and Hermes' providers/dispatch (~/.hermes/config.yaml). Credential values
    are never read: an `api_key:` line contributes only the ${ENV_VAR} it references, never a literal value."""
    env = os.environ if env is None else env
    user_home = user_home or core._user_home(env)
    out = []
    try:
        auth = json.load(open(os.path.join(user_home, ".pi", "agent", "auth.json")))
        if isinstance(auth, dict):
            for name in auth:
                out.append({"harness": "pi", "name": str(name), "via": "pi-auth"})
    except (OSError, ValueError):
        pass
    try:
        pi = json.load(open(os.path.join(user_home, ".pi", "agent", "models.json")))
        for name, prov in (pi.get("providers") or {}).items():
            if isinstance(prov, dict):
                row = {"harness": "pi", "name": str(name), "via": "pi-models"}
                if isinstance(prov.get("baseUrl"), str):
                    row["base_url"] = prov["baseUrl"]
                if isinstance(prov.get("api"), str):
                    row["transport"] = prov["api"]
                out.append(row)
    except (OSError, ValueError):
        pass
    try:
        text = open(os.path.join(user_home, ".hermes", "config.yaml")).read()
    except OSError:
        text = ""
    if text:
        block = re.search(r"^providers:\n((?:[ ].*\n?)+)", text, re.M)
        cur = None
        if block:
            for line in block.group(1).splitlines():
                m = re.match(r"^  ([A-Za-z0-9_.-]+):\s*$", line)
                if m:
                    cur = {"harness": "hermes", "name": m.group(1), "via": "hermes-providers"}
                    out.append(cur)
                    continue
                f = re.match(r"^    (api|base_url|transport):\s*(\S+)\s*$", line)
                if cur is not None and f:
                    cur["base_url" if f.group(1) in ("api", "base_url") else "transport"] = f.group(2)
                    continue
                k = re.match(r"^    api_key:\s*\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?\s*$", line)
                if cur is not None and k:
                    cur["key_env"] = k.group(1)
        for m in re.finditer(r"dispatch_fallback_providers:\n((?:\s*- .*\n?)+)", text):
            for item in re.findall(r"-\s*([A-Za-z0-9_.-]+)\s*$", m.group(1), re.M):
                out.append({"harness": "hermes", "name": item, "via": "hermes-dispatch"})
        m = re.search(r"^model:\n(?:[ ].*\n?)*?^  provider:\s*(\S+)\s*$", text, re.M)
        if m:
            out.append({"harness": "hermes", "name": m.group(1), "via": "hermes-default"})
    seen, dedup = set(), []
    for row in out:
        key = (row["harness"], row["name"], row["via"])
        if key not in seen:
            seen.add(key)
            dedup.append(row)
    return dedup


def detect_all(env=None, home=None, cfg=None, inventory=None, list_live=True, which=None, platform=None, run=None,
               timeout=LIST_TIMEOUT, write=True):
    """The provider inventory: API providers (key present -> listed live unless list_live=False), subscription CLIs and
    local endpoints. {"version", "generated_at", "providers": [{id, name, access, style, status, reason, models,
    key_source, ...}], "warnings"}; saved to ~/.maestro/providers.json. status: connected | detected | missing | error.
    `inventory` is the router's (local endpoints); it is detected when omitted."""
    env = os.environ if env is None else env
    home = home or core.maestro_home(env)
    cfg = cfg if cfg is not None else core.load_config(home)
    which = which or (lambda exe: shutil.which(exe, path=env.get("PATH")))
    user_home = core._user_home(env)
    inv = inventory if inventory is not None else core.ensure_inventory(home, env, cfg=cfg)
    entries = [_api_entry(s, env, home, cfg) for s in REGISTRY]
    _hold_shared(entries, cfg)
    todo = [e for e in entries if e["status"] == "detected" and BY_ID[e["id"]]["callable"]]
    if list_live and todo:
        with cf.ThreadPoolExecutor(max_workers=min(len(todo), 12)) as ex:
            results = list(ex.map(lambda e: _verify(e, env, home, cfg, timeout), todo))
        for pid, models, err, code in results:
            e = next(x for x in entries if x["id"] == pid)
            if code in (404, 405, 501):  # a working key on a provider without a model list: stays "detected"
                e.update(reason="key set; the provider has no model list endpoint (HTTP %d)" % code)
            elif err:
                e.update(status="error", reason=err)
            else:
                e.update(status="connected", models=models, reason="%d models" % len(models))
    entries += [detect_cli(c, env, user_home, which, platform, run) for c in CLI_PROVIDERS]
    entries += _local_entries(inv, cfg, env)
    out = {"version": 1, "generated_at": time.time(), "verified": bool(list_live), "providers": entries,
           "harness_providers": harness_providers(env, user_home),
           "warnings": [w for w in (keys_file_problem(home),) if w]}
    if write:
        core._write_json(os.path.join(home, PROVIDERS_FILE), out)
    return out


def load_providers(home=None):
    """The saved inventory (providers.json), or None."""
    data = core._read_json(os.path.join(home or core.maestro_home(), PROVIDERS_FILE))
    return data if isinstance(data, dict) and data.get("version") == 1 else None


def summary_lines(pv):
    """A compact text section for `maestro-router detect`: what is set up, then a count of what is not."""
    lines = ["providers%s:" % ("" if pv.get("verified") else " (keys not verified; add --models to list models)")]
    shown = [e for e in pv["providers"] if e["status"] != "missing" or e["reason"].startswith("installed")]
    for e in shown:
        n = " %d models" % len(e["models"]) if e["models"] and e["access"] == "api" else ""
        via = " [%s]" % e["key_source"] if e.get("key_source") else ""
        lines.append("  %-16s %-8s %-9s %s%s%s" % (e["id"], {"api": "api key", "subscription-cli": "login", "local": "local"}[e["access"]],
                                                    e["status"], e["reason"], n, via))
    absent = sum(1 for e in pv["providers"] if e["access"] == "api" and e["status"] == "missing")
    lines.append("  %d other API providers have no key (`maestro-router keys list --all`, `maestro-router keys set <provider>`)" % absent)
    hp = pv.get("harness_providers") or []
    if hp:
        lines.append("harness providers (carried by Pi/Hermes on this machine; Pi ones are callable via the pi seat):")
        for row in hp:
            extra = " ".join(x for x in (row.get("base_url") or "", row.get("transport") or "",
                                         ("$" + row["key_env"]) if row.get("key_env") else "") if x)
            lines.append("  %-8s %-24s %s%s" % (row["harness"], row["name"], row["via"], (" " + extra) if extra else ""))
    lines += ["warning: " + w for w in pv.get("warnings") or []]
    return lines


# ════════════════════════════════════════════════════════════════════════════
# 6. `maestro-router keys`
# ════════════════════════════════════════════════════════════════════════════

def cmd_keys(a):
    """keys set <provider|ENV_VAR> [value] | unset <provider|ENV_VAR> | list [--all | <provider>]. Values are never printed."""
    home = core.maestro_home()
    try:
        if a.action == "list":
            rows = [(s, key_status(s, None, home)) for s in REGISTRY if not a.name or a.name in (s["id"], s["name"])]
            if a.name and not rows:
                raise ValueError("unknown provider %r" % a.name)
            n_set = 0
            for s, (name, src, missing) in rows:
                ok = bool(name) and not missing
                n_set += ok
                if ok or a.all or a.name:
                    print("%-22s %-30s %s" % (s["id"], name or s["env"][0], "set (%s)" % src if ok else "missing" + (" " + ", ".join(missing) if missing else "")))
            if not a.all and not a.name:
                print("%d of %d providers have a key; --all lists the rest" % (n_set, len(rows)))
            print("keys file: %s" % keys_path(home))
            if keys_file_problem(home):
                print("warning: " + keys_file_problem(home))
            return 0
        if not a.name:
            raise ValueError("usage: maestro-router keys %s <provider|ENV_VAR>%s" % (a.action, " [value]" if a.action == "set" else ""))
        name = key_env_name(a.name)
        if a.action == "unset":
            names = list(BY_ID[a.name]["env"]) if a.name in BY_ID else [name]
            gone = unset_key(names, home)
            print("removed %s from %s" % (", ".join(gone), keys_path(home)) if gone else "%s is not in %s" % (a.name, keys_path(home)))
            return 0
        value = a.value
        if value is None:
            import getpass
            value = getpass.getpass("%s: " % name)
        else:
            print("note: a key typed on the command line stays in your shell history; omit it to be prompted", file=sys.stderr)
        set_key(name, value, home)
        print("saved %s to %s (mode 600)" % (name, keys_path(home)))
        return 0
    except (ValueError, OSError) as e:
        print("maestro: %s" % e, file=sys.stderr)
        return 2
