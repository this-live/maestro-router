#!/usr/bin/env python3
"""Sandboxed provider matrix: every discovered model x discriminative tasks x rolls.

    python evals/matrix.py --serve [--rolls 2] [--port 18951]
                           [--tasks evals/matrix_tasks.jsonl] [--out evals/matrix-results.jsonl]

Sandbox: serve runs with MAESTRO_HOME=<sandbox dir> (config.json is copied in —
endpoints and overrides, never keys; decisions/outcomes/models state is fresh),
from a temp cwd. Rows are only credited to a provider when served_by matches
the requested model (strict_ok); fallback-served rows are visible, not hidden.
A run manifest (git SHA, config hash, hostname, providers, task count) is
written next to the results. Resumable: recorded (task, provider, roll) rows
are skipped on rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from outcome import grade as _grade, wilson  # noqa: E402


def grade(check, text):
    if check.get("type") == "sentence_count":
        if not text:
            return False
        import re
        parts = [s for s in re.split(r"[.!?]+", text.strip()) if s.strip()]
        return len(parts) == check["value"]
    return _grade(check, text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18951)
    ap.add_argument("--serve", action="store_true", help="start a sandboxed serve on --port first")
    ap.add_argument("--sandbox", default="")
    ap.add_argument("--rolls", type=int, default=2)
    ap.add_argument("--tasks", default=os.path.join(HERE, "matrix_tasks.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "matrix-results.jsonl"))
    ap.add_argument("--providers", default="", help="comma list; default: all /v1/models ids except maestro/*, plus maestro/auto")
    ap.add_argument("--max-workers", type=int, default=5)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    proc = None
    sandbox = args.sandbox
    if args.serve:
        sandbox = sandbox or tempfile.mkdtemp(prefix="maestro-matrix-")
        os.makedirs(os.path.join(sandbox, ".maestro"), exist_ok=True)
        src = os.path.expanduser("~/.maestro/config.json")
        cfg_bytes = b"{}"
        if os.path.exists(src):
            cfg_bytes = open(src, "rb").read()
            open(os.path.join(sandbox, ".maestro", "config.json"), "wb").write(cfg_bytes)
        env = dict(os.environ, MAESTRO_HOME=os.path.join(sandbox, ".maestro"))
        proc = subprocess.Popen(["maestro-router", "serve", "--port", str(args.port)],
                                env=env, cwd=tempfile.mkdtemp(prefix="maestro-matrix-cwd-"),
                                stdout=open(os.path.join(sandbox, "serve.log"), "w"), stderr=subprocess.STDOUT)
        for _ in range(60):
            try:
                urllib.request.urlopen(base + "/v1/models", timeout=3)
                break
            except Exception:
                time.sleep(1)
        manifest = {"sandbox": sandbox, "config_sha256": hashlib.sha256(cfg_bytes).hexdigest()[:16],
                    "hostname": platform.node(), "python": platform.python_version(),
                    "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            manifest["git_sha"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                                 cwd=os.path.join(HERE, "..")).stdout.strip()
        except Exception:
            manifest["git_sha"] = "unknown"
        open(os.path.join(sandbox, "manifest.json"), "w").write(json.dumps(manifest, indent=1))
        print("sandbox", sandbox, manifest, flush=True)

    models = json.loads(urllib.request.urlopen(base + "/v1/models", timeout=15).read())["data"]
    providers = [m["id"] for m in models if not m["id"].startswith("maestro/")] + ["maestro/auto"]
    if args.providers:
        keep = set(args.providers.split(","))
        providers = [p for p in providers if p in keep]
    print("providers", providers, flush=True)

    tasks = [json.loads(l) for l in open(args.tasks) if l.strip()]
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                r = json.loads(l)
                done.add((r["task"], r["provider"], r["roll"]))
            except Exception:
                pass
    out = open(args.out, "a")

    def chat(model, prompt, timeout=240):
        req = urllib.request.Request(base + "/v1/chat/completions",
            data=json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}]}).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            r = urllib.request.urlopen(req, timeout=timeout)
            d = json.loads(r.read())
            msg = (d.get("choices") or [{}])[0].get("message") or {}
            return d.get("model"), msg.get("content"), round(time.time() - t0, 1), None
        except Exception as e:
            return None, None, round(time.time() - t0, 1), str(e)[:120]

    def one(task, provider, roll):
        served, content, secs, err = chat(provider, task["prompt"])
        ok = grade(task["check"], content) if content is not None else False
        fail = None
        if err:
            fail = "timeout" if "timed out" in err else "http"
        elif not ok:
            fail = "empty" if not (content or "").strip() else "wrong"
        return {"task": task["id"], "job": task["job"], "difficulty": task["difficulty"],
                "provider": provider, "served_by": served, "roll": roll,
                "ok": ok, "strict_ok": bool(ok and served == provider),
                "secs": secs, "err": err, "fail_kind": fail, "sample": (content or "")[:140]}

    jobs = [(t, p, r) for t in tasks for p in providers for r in range(args.rolls)
            if (t["id"], p, r) not in done]
    print(f"{len(jobs)} calls to make", flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for rec in ex.map(lambda j: one(*j), jobs):
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(rec["task"], rec["provider"], "r%s" % rec["roll"], "->", rec["served_by"],
                  "ok" if rec["strict_ok"] else ("FALLBACK-OK" if rec["ok"] else "FAIL"), rec["secs"], flush=True)
    out.close()
    if proc:
        proc.terminate()

    rows = [json.loads(l) for l in open(args.out)]
    byp = {}
    for r in rows:
        byp.setdefault(r["provider"], []).append(r)
    print("\n== provider summary (strict attribution) ==")
    for p, rs in sorted(byp.items()):
        k = sum(1 for r in rs if r["strict_ok"])
        secs = sorted(r["secs"] for r in rs)
        p50 = secs[len(secs) // 2] if secs else 0
        p95 = secs[int(len(secs) * 0.95)] if secs else 0
        lo, hi = wilson(k, len(rs))
        fails = {}
        for r in rs:
            if not r["strict_ok"] and r["fail_kind"]:
                fails[r["fail_kind"]] = fails.get(r["fail_kind"], 0) + 1
        print(f"{p:48s} {k:3d}/{len(rs):3d} = {k/len(rs):.2f} Wilson[{lo:.2f},{hi:.2f}] p50={p50}s p95={p95}s fails={fails}")


if __name__ == "__main__":
    main()
