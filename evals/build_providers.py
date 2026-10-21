#!/usr/bin/env python3
"""Regenerate skills/maestro/scripts/maestro/data/providers.json from a models.dev checkout (deterministic, stdlib only).

    python evals/build_providers.py --models-dev /path/to/models.dev [--out skills/maestro/scripts/maestro/data/providers.json]

For every providers/<id>/provider.toml it keeps: name, env (the variables that hold the key or settings), npm (the AI SDK
package, from which providers.py derives the wire style), api (base URL, may hold ${VAR}), doc, and `params` (the ${VAR}
names inside `api`). Model files with a `[provider]` table (their own npm / api / shape) are grouped under "o" as
[[override, [model ids]], ...], only for providers that have any. Python 3.11+ reads the TOML with tomllib; older versions
use the small subset parser below, enough for these files.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "skills", "maestro", "scripts", "maestro", "data", "providers.json")
KEEP = ("npm", "api", "shape")


def load_toml(path):
    with open(path, "rb") as f:
        raw = f.read()
    try:
        import tomllib
        return tomllib.loads(raw.decode("utf-8"))
    except ImportError:
        return mini_toml(raw.decode("utf-8"))


def mini_toml(text):
    """Top-level `key = "string" | ["a", "b"] | number` and one level of `[table]` sections; everything else is skipped."""
    out, cur = {}, None
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^\[([\w.-]+)\]$", line)
        if m:
            cur = out.setdefault(m.group(1), {}) if "." not in m.group(1) else None
            continue
        m = re.match(r'^([\w-]+)\s*=\s*(.+?)\s*$', line)
        if not m or line.startswith("#"):
            continue
        k, v = m.groups()
        try:
            val = json.loads(v) if v[0] in '"[' else v
        except ValueError:
            continue
        (out if cur is None else cur)[k] = val
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dev", default=os.environ.get("MODELS_DEV"), help="path of a models.dev checkout (or $MODELS_DEV)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    if not a.models_dev or not os.path.isdir(os.path.join(a.models_dev, "providers")):
        print("error: pass --models-dev <path of a models.dev checkout>", file=sys.stderr)
        return 2
    root = os.path.join(a.models_dev, "providers")
    providers = {}
    for pid in sorted(os.listdir(root)):
        tp = os.path.join(root, pid, "provider.toml")
        if not os.path.isfile(tp):
            continue
        t = load_toml(tp)
        p = {"name": t["name"], "env": list(t.get("env") or []), "npm": t.get("npm") or "", "doc": t.get("doc") or ""}
        if t.get("api"):
            p["api"] = t["api"].rstrip("/")
            p["params"] = sorted(set(re.findall(r"\$\{(\w+)\}", p["api"])))
        groups = {}
        mroot = os.path.join(root, pid, "models")
        for dirpath, _, files in os.walk(mroot):
            for fn in sorted(files):
                if not fn.endswith(".toml"):
                    continue
                try:
                    m = load_toml(os.path.join(dirpath, fn))
                except (OSError, ValueError):
                    continue
                prov = m.get("provider")
                if isinstance(prov, dict):
                    ov = {k: (v.rstrip("/") if k == "api" else v) for k, v in prov.items() if k in KEEP}
                    if ov:
                        mid = os.path.relpath(os.path.join(dirpath, fn), mroot)[:-5].replace(os.sep, "/")
                        groups.setdefault(json.dumps(ov, sort_keys=True), []).append(mid)
        if groups:
            p["o"] = [[json.loads(k), sorted(v)] for k, v in sorted(groups.items())]
        providers[pid] = p
    src = "models.dev"
    try:
        sha = subprocess.check_output(["git", "-C", a.models_dev, "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        src += " " + sha
    except (OSError, subprocess.CalledProcessError):
        pass
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "source": src, "providers": providers}, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")
    print("%d providers, %d with per-model overrides, %.1f KB -> %s" % (
        len(providers), sum(1 for p in providers.values() if "o" in p), os.path.getsize(a.out) / 1024, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
