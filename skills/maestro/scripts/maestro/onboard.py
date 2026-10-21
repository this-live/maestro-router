"""Getting started: `maestro-router hermes` (Hermes Agent config), `maestro-router setup` (the interactive quick start) and the snippets the UI shows.

Hermes Agent talks to Maestro as a custom OpenAI-compatible provider: `model.provider: custom` with our `base_url`, and its
auxiliary tasks (context compression, titles, ...) and subagent delegation pointed at cheaper Maestro categories. PyYAML is not
available, so `--install` only appends top-level blocks that are missing from ~/.hermes/config.yaml and prints instructions for
the ones that exist. It backs the file up first and never rewrites a key it did not add.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import time

from . import providers as P
from . import router as core

SKILL_COMMAND = "npx skills add this-live/maestro-router"
CONTEXT_LENGTH = 128000          # what we tell Hermes maestro/auto can take; requests that do not fit skip smaller models
AUX_TASKS = (("compression", "maestro/fast"), ("title_generation", "maestro/fast"), ("session_search", "maestro/fast"),
             ("vision", "maestro/auto"))
BLOCKS = ("model", "auxiliary", "delegation")


def hermes_blocks(base_url, key_required=False):
    """{top-level key: YAML text} for the Hermes config."""
    key = "YOUR_MAESTRO_API_KEY" if key_required else "maestro"
    note = "  # the server requires MAESTRO_API_KEY: put it here" if key_required else "  # placeholder: Maestro ignores it unless MAESTRO_API_KEY is set"
    aux = "".join("  %s:\n    provider: custom\n    base_url: %s\n    api_key: %s\n    model: %s\n" % (t, base_url, key, m) for t, m in AUX_TASKS)
    return {
        "model": "model:\n  default: maestro/auto\n  provider: custom\n  base_url: %s\n  api_key: %s%s\n  context_length: %d\n"
                 % (base_url, key, note, CONTEXT_LENGTH),
        "auxiliary": "auxiliary:\n" + aux,
        "delegation": "delegation:\n  base_url: %s\n  api_key: %s\n  model: maestro/auto\n" % (base_url, key),
    }


def hermes_yaml(base_url, key_required=False):
    """The block `maestro-router hermes --print` shows: main model on maestro/auto, cheap auxiliary tasks on maestro/fast, delegation on auto."""
    head = "# Maestro as a custom provider for Hermes Agent (merge into ~/.hermes/config.yaml)\n"
    return head + "".join(hermes_blocks(base_url, key_required)[k] for k in BLOCKS)


def openai_snippet(base_url):
    return ("from openai import OpenAI\n\nclient = OpenAI(base_url=%r, api_key=\"maestro\")  # any string, unless MAESTRO_API_KEY is set\n"
            "r = client.chat.completions.create(model=\"maestro/auto\",  # or maestro/fast, maestro/code, provider/model\n"
            "                                   messages=[{\"role\": \"user\", \"content\": \"Why is my test flaky?\"}])\n"
            "print(r.choices[0].message.content, r.model)  # r.model is the model Maestro chose\n" % base_url)


def curl_snippet(base_url):
    return ("curl %s/chat/completions -H 'Content-Type: application/json' \\\n"
            "  -d '{\"model\": \"maestro/auto\", \"messages\": [{\"role\": \"user\", \"content\": \"hello\"}]}'\n" % base_url)


def hermes_path(env=None):
    env = os.environ if env is None else env
    return os.path.join(env.get("HERMES_HOME") or os.path.join(core._user_home(env), ".hermes"), "config.yaml")


def _top_keys(text):
    return {m.group(1) for m in re.finditer(r"(?m)^([A-Za-z_][\w.-]*)[ \t]*:", text)}


def _block(text, key):
    """The text of one top-level block (its line and the indented lines after it)."""
    m = re.search(r"(?m)^%s[ \t]*:.*\n?(?:(?:[ \t]+.*|#.*|)\n?)*" % re.escape(key), text)
    return m.group(0) if m else ""


def install_hermes(base_url, env=None, key_required=False, dry_run=False):
    """Add the missing top-level blocks to the Hermes config. Returns {"path", "created", "backup", "added": [...], "skipped": {key: why}}."""
    path = hermes_path(env)
    try:
        with open(path, "r", encoding="utf-8") as f:
            old = f.read()
    except OSError:
        old = None
    text = old or ""
    have = _top_keys(text)
    blocks = hermes_blocks(base_url, key_required)
    added = [k for k in BLOCKS if k not in have]
    skipped = {k: ("already points at Maestro" if base_url in _block(text, k) else "a `%s:` block already exists" % k)
               for k in BLOCKS if k in have}
    res = {"path": path, "created": old is None and bool(added), "backup": None, "added": added, "skipped": skipped, "dry_run": dry_run}
    if not added or dry_run:
        return res
    if old is not None:
        res["backup"] = "%s.maestro-bak-%s" % (path, time.strftime("%Y%m%d-%H%M%S"))
        shutil.copy2(path, res["backup"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + ("" if not text or text.endswith("\n") else "\n") + ("\n" if text else "")
                + "# added by `maestro-router hermes --install`\n" + "".join(blocks[k] for k in added))
    return res


def cmd_hermes(a):
    key_required = bool(P.get_key(("MAESTRO_API_KEY",), os.environ, core.maestro_home()))
    base = a.base_url or "http://%s:%d/v1" % (a.host, a.port)
    if not a.install:
        print(hermes_yaml(base, key_required), end="")
        return 0
    res = install_hermes(base, os.environ, key_required, a.dry_run)
    if res["added"]:
        print("%s %s in %s: %s" % ("would add" if a.dry_run else "added", ", ".join(res["added"]), res["path"], "created the file" if res["created"] else
                                   "backup: %s" % res["backup"] if res["backup"] else "dry run"))
    for k, why in res["skipped"].items():
        print("left `%s:` alone (%s)." % (k, why))
    todo = [k for k, w in res["skipped"].items() if not w.startswith("already")]
    if todo:
        blocks = hermes_blocks(base, key_required)
        print("\nMaestro does not edit existing blocks (no YAML library). Set these in %s yourself:\n" % res["path"])
        print("".join(blocks[k] for k in todo), end="")
    return 0


# ════════════════════════════════════════════════════════════════════════════
# maestro-router setup
# ════════════════════════════════════════════════════════════════════════════

def _ask(prompt, default, yes):
    if yes:
        return default
    try:
        ans = input("%s [%s] " % (prompt, "Y/n" if default else "y/N")).strip().lower()
    except EOFError:
        return default
    return default if not ans else ans.startswith("y")


def _add_keys(home):
    import getpass
    while True:
        try:
            pid = input("Add an API key for which provider? (id such as openai, anthropic, google, openrouter; Enter to continue) ").strip()
        except EOFError:
            return
        if not pid:
            return
        try:
            name = P.key_env_name(pid)
            P.set_key(name, getpass.getpass("%s (hidden): " % name), home)
            print("  saved %s to %s (mode 600)" % (name, P.keys_path(home)))
        except ValueError as e:
            print("  %s" % e)


def add_endpoint(url, home, name=None, probe=None):
    """Probe a model server and add it to local_endpoints -> a line for the user. Without a port, tries 8000 (vLLM) then 11434
    (Ollama). An unreachable server is saved anyway (it may be off right now)."""
    probe = probe or (lambda ep: core.probe_endpoint(ep, timeout=3.0))
    base = core._norm_url(url)
    host = core._host_of(base)
    name = name or ("this-machine" if host in core.LOOPBACK else host if re.match(r"^[\d.]+$", host) else host.split(".")[0])
    urls = [base] if re.search(r":\d+$", base) else [base + ":8000", base + ":11434"]
    found = None
    for u in urls:
        found = probe({"name": name, "url": u, "kind": None, "api_key_env": None})
        if found:
            url = u
            break
    else:
        url = urls[0]
    core.save_endpoints([{"name": name, "url": url, "kind": found["kind"] if found else None}], home)
    if not found:
        return "  saved %s as %s, but it did not answer (/v1/models or /api/tags); it is used once it does" % (url, name)
    return "  %s at %s (%s): %s" % (name, url, found["kind"], ", ".join(m["name"] + ("" if m.get("loaded", True) else " (installed, not loaded: not used)")
                                      for m in found["models"]) or "no models")


def _ask_endpoints(home):
    while True:
        try:
            url = input("Model server on your network? (vLLM, SGLang, Ollama, LM Studio, e.g. http://spark.local:8000; Enter to continue) ").strip()
        except EOFError:
            return
        if not url:
            return
        print(add_endpoint(url, home))


def local_report(home, cfg, env=None, inv=None, top=3):
    """Lines for setup's local step: the hardware Maestro gauged on each box and what each box serves (loaded models, which are
    the ones Maestro uses). Only when no box serves anything: the top `top` models that would fit the strongest box, as advice.
    Nothing is downloaded, loaded or started."""
    from . import catalog, score
    from . import discover as D
    env = os.environ if env is None else env
    inv = inv if inv is not None else core.ensure_inventory(home, env, cfg=cfg)
    hw = D.hardware_map(cfg, inv, env)
    boxes = {}
    for ep in inv.get("local_endpoints") or []:
        mods = [m for m in inv.get("local_models") or [] if m["url"] == ep["url"]]
        box = D._box_of({"id": "local:" + ep["name"], "endpoint_url": ep["url"]}, hw)
        b = boxes.setdefault(box, {"serving": [], "idle": []})
        for m in mods:
            b["serving" if m.get("loaded", True) or cfg.get("local_allow_load") else "idle"].append(m["name"])
    lines = []
    for box in list(hw) + [b for b in boxes if b not in hw]:
        h, sv = hw.get(box), boxes.get(box) or {"serving": [], "idle": []}
        gauged = "%s, %g GB%s%s" % (h["name"], h["memory_gb"], ", %g GB/s" % h["bandwidth_gbps"] if h.get("bandwidth_gbps") else ", memory speed unknown",
                                    " (assumed)" if h.get("assumed") else "") if h else "hardware unknown (set local_endpoints[].hardware)"
        lines.append("  %s: %s" % (box, gauged))
        lines.append("    serves: " + (", ".join(sv["serving"]) + " - using these" if sv["serving"] else "nothing") + (
            "; %d installed but not loaded, ignored (Maestro never loads models)" % len(sv["idle"]) if sv["idle"] else ""))
    if not any(b["serving"] for b in boxes.values()):
        if hw:
            best = max(hw, key=lambda k: (hw[k]["memory_gb"], hw[k].get("bandwidth_gbps") or 0))
            cat, _ = catalog.load(home)
            rows = D.local_fit(cat, score.make_context(cat), {best: hw[best]}, top=top)[best]
            lines.append("  Nothing is served. Models that would fit %s (advice only: Maestro will not download or start anything):" % best)
            for i, r in enumerate(rows, 1):
                lines += ["    %d. %s, %s, about %.0f GB   %s" % (i, r["model"], r["quant"], r["mem_gb"] + r["kv_gb"], r["serve_cmd"])]
        else:
            lines.append("  Nothing is served and the hardware could not be gauged, so there is nothing to suggest.")
    return lines


def category_summary(home, cfg):
    """Lines: how many enabled models each category holds, with a few names."""
    from . import server
    lst, raw = server.models_view(home, cfg)
    by = {}
    for m in lst:
        if m["enabled"]:
            by.setdefault(m["cat_override"] or ("local" if m["access"] == "local" else m["category"]), []).append(m["id"])
    lines = ["  %-11s %3d  %s" % (c, len(by[c]), ", ".join(by[c][:3]) + (", ..." if len(by[c]) > 3 else "")) for c in
             ("frontier", "strong", "standard", "fast", "local", "specialist") if c in by]
    return lines or ["  no models found yet: add a key or point Maestro at a model server you already run, then run `maestro-router discover`"], raw.get("source")


def cmd_setup(a):
    from . import server
    home, cfg, yes = core.maestro_home(), core.load_config(core.maestro_home()), a.yes
    print("Maestro setup (%s)\n\n1. What is here" % home)
    print("\n".join(P.summary_lines(P.detect_all(os.environ, home, cfg, list_live=False))))
    if not yes:
        print("\n2. API keys (typed without echo, stored only in %s)" % P.keys_path(home))
        _add_keys(home)
    else:
        print("\n2. API keys: skipped (--yes). Use `maestro-router keys set <provider>`.")
    print("\n3. Local and network model servers (a DGX Spark, a vLLM or Ollama box; this machine's ports are found on their own)")
    for url in getattr(a, "endpoint", None) or []:
        print(add_endpoint(url, home))
    if not yes:
        _ask_endpoints(home)
    cfg = core.load_config(home)
    print("\n   Hardware and what is served (Maestro uses exactly what is served; it never downloads, loads or starts a model)")
    print("\n".join(local_report(home, cfg)))
    print("\n4. Discovery")
    if _ask("Look up every model you can use and score them now?", True, yes):
        ok, msg = server.run_discover(home, os.environ, cfg)
        print("  " + msg)
    lines, source = category_summary(home, cfg)
    print("\nModels by category (from %s):" % source)
    print("\n".join(lines))
    print("\n5. Hermes Agent")
    hermes_home = os.path.dirname(hermes_path(os.environ))
    found = os.path.isdir(hermes_home) or shutil.which("hermes", path=os.environ.get("PATH")) is not None
    base = "http://%s:%d/v1" % (server.DEFAULT_HOST, server.DEFAULT_PORT)
    if _ask("Add Maestro to %s?" % hermes_path(os.environ), found, yes):
        res = install_hermes(base, os.environ, bool(P.get_key(("MAESTRO_API_KEY",), os.environ, home)))
        print("  added %s%s" % (", ".join(res["added"]) or "nothing", " (backup: %s)" % res["backup"] if res["backup"] else ""))
        for k, why in res["skipped"].items():
            print("  left `%s:` alone (%s); `maestro-router hermes --print` shows the block" % (k, why))
    else:
        print("  skipped (`maestro-router hermes --print` prints the config)")
    print("\n6. Skill for Claude Code, OpenCode and Gemini CLI")
    inv = core.ensure_inventory(home, cfg=cfg)
    seats = [h for h in core.AGENT_DIRS if h in (inv.get("harnesses") or {})]
    if seats and _ask("Install the maestro-<category> subagents for %s?" % ", ".join(seats), True, yes):
        for h in seats:
            for x in core.install(h, cfg=cfg, inv=inv):
                print("  %-28s %s" % (x["action"], x["path"]))
    else:
        print("  install it with `%s`" % SKILL_COMMAND)
    print("\nDone. Start the router and the web UI with:\n\n    maestro-router serve --open\n\nThen point any OpenAI client (or Hermes) at %s with model maestro/auto." % base)
    return 0
