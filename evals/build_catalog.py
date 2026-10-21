#!/usr/bin/env python3
"""Regenerate skills/maestro/scripts/maestro/data/catalog.json.gz from a models.dev checkout (deterministic, stdlib only).

    python evals/build_catalog.py --models-dev /path/to/models.dev [--out ...] [--no-hf] [--as-of YYYY-MM-DD]

Needs Python 3.11+ (tomllib); this is a maintainer tool, the shipped package never reads TOML.

What goes in (see docs/DESIGN.md, "Catalog"):
  models  every text model in models/<lab>/*.toml: name, family, release date, reasoning, tool_call, context, modalities,
          status, open_weights, the [[benchmarks]] rows, a representative price (median over the providers that resolve
          to it), and for open-weight models the open-weight registry fields (Hugging Face repo, params_total,
          params_active, and where the numbers came from).
  prov    for every provider model: the canonical lab model it is (base_model), its price, context and status; models
          without a base_model carry their own attributes, so a provider id the lab files do not know still scores.
Benchmark gap-fill: evals/extra_benchmarks.json ({catalog id: [{name, score, metric, source, variant}]}, numbers read from the
official model cards) is merged into `b`; a row models.dev already has (same name and score) wins.
Dropped: embeddings, image / audio / video / speech models (no text output) and deprecated models older than 18 months.

Parameter counts for the registry: the Hugging Face API (`safetensors.total`) when huggingface.co is reachable, else the
hand-verified table evals/hf_params.json (read from the Hub pages), else parsed from the repo / model name (120b, 30b-a3b,
8x7b), else null (shown as "unknown"; nothing is guessed).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import datetime
import gzip
import json
import os
import statistics
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "skills", "maestro", "scripts"))
from maestro import catalog as C  # noqa: E402

DEFAULT_OUT = os.path.join(HERE, "..", "skills", "maestro", "scripts", "maestro", "data", "catalog.json.gz")
HF_TABLE = os.path.join(HERE, "hf_params.json")
EXTRA_BENCH = os.path.join(HERE, "extra_benchmarks.json")
DROP_RX = C.NON_CHAT_RX
OMIT_FROM_BASE = ("benchmarks", "license", "links", "weights")


def read_toml(path):
    import tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def omit(t, paths):
    for p in paths:
        parts = p.split(".")
        cur = t
        for x in parts[:-1]:
            cur = cur.get(x) if isinstance(cur, dict) else None
            if cur is None:
                break
        if isinstance(cur, dict):
            cur.pop(parts[-1], None)


def walk_toml(root):
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            p = os.path.join(dirpath, fn)
            if fn.endswith(".toml") and os.path.exists(p):  # models.dev has a few dangling symlinks
                yield os.path.relpath(p, root)[:-5].replace(os.sep, "/"), p


def full_date(d):
    d = str(d or "")
    return d + "-01" * (2 - d.count("-")) if d[:1].isdigit() else ""


def text_out(t):
    """Chat models answer in text only; image / audio / video generators (even ones that also print text) are out."""
    return C.text_only((t.get("modalities") or {}).get("output"))


def compact(t):
    """The attributes both lab and provider records share, without defaults."""
    m = t.get("modalities") or {}
    o = {"n": t.get("name") or "", "d": full_date(t.get("release_date"))}
    for k, key in (("f", "family"), ("st", "status")):
        if t.get(key):
            o[k] = t[key]
    for k, key in (("r", "reasoning"), ("t", "tool_call"), ("so", "structured_output"), ("ow", "open_weights")):
        if t.get(key):
            o[k] = 1
    ctx = (t.get("limit") or {}).get("context")
    if ctx:
        o["x"] = int(ctx)
    if (m.get("input") or ["text"]) != ["text"]:
        o["im"] = m["input"]
    if (m.get("output") or ["text"]) != ["text"]:
        o["om"] = m["output"]
    return o


def cutoff(as_of, months=18):
    y, mo = int(as_of[:4]), int(as_of[5:7])
    mo -= months
    while mo <= 0:
        mo, y = mo + 12, y - 1
    return "%04d-%02d-01" % (y, mo)


def keep(t, mid, old):
    if not text_out(t) or DROP_RX.search(mid) or DROP_RX.search(t.get("name") or ""):
        return False
    return not (t.get("status") == "deprecated" and full_date(t.get("release_date")) < old)


def hf_total(repo, timeout=8.0):
    """Total parameters (billions) from the Hub API, or None."""
    try:
        req = urllib.request.Request("https://huggingface.co/api/models/" + repo, headers={"User-Agent": "maestro-build-catalog"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            n = ((json.load(r).get("safetensors") or {}).get("total"))
        return round(n / 1e9, 2) if n else None
    except Exception:
        return None


def hf_reachable():
    return hf_total("openai/gpt-oss-20b", 6.0) is not None


def pick_repo(weights):
    repos = [w["url"].split("huggingface.co/", 1)[1].strip("/") for w in weights or [] if "huggingface.co/" in (w.get("url") or "")]
    plain = [r for r in repos if not C.QUANT_SUFFIX_RX.search(r)]
    return (plain or repos or [None])[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dev", default=os.environ.get("MODELS_DEV"), help="path of a models.dev checkout (or $MODELS_DEV)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--no-hf", action="store_true", help="do not call the Hugging Face API (use hf_params.json and names)")
    ap.add_argument("--as-of", default=None, help="snapshot date (default: today)")
    a = ap.parse_args(argv)
    if sys.version_info < (3, 11):
        print("error: build_catalog.py needs Python 3.11+ (tomllib)", file=sys.stderr)
        return 2
    if not a.models_dev or not os.path.isdir(os.path.join(a.models_dev, "models")):
        print("error: pass --models-dev <path of a models.dev checkout>", file=sys.stderr)
        return 2
    as_of = a.as_of or datetime.date.today().isoformat()
    old = cutoff(as_of)
    with open(HF_TABLE, encoding="utf-8") as f:
        table = json.load(f)
    repo_for = table.pop("_repos", {})
    table.pop("_note", None)

    raw, models = {}, {}
    for mid, p in walk_toml(os.path.join(a.models_dev, "models")):
        raw[mid] = read_toml(p)
    for mid, t in raw.items():
        if not keep(t, mid, old):
            continue
        rec = compact(t)
        rows = []
        for b in t.get("benchmarks") or []:
            if isinstance(b.get("score"), (int, float)) and not isinstance(b["score"], bool):
                rows.append([b["name"], b["score"]] + ([b["metric"]] if b.get("metric") else []))
        if rows:
            rec["b"] = rows
        if t.get("open_weights"):
            repo = pick_repo(t.get("weights")) or repo_for.get(mid)
            if repo:
                rec["hf"] = repo
        models[mid] = rec

    added, unknown = 0, []
    if os.path.exists(EXTRA_BENCH):
        with open(EXTRA_BENCH, encoding="utf-8") as f:
            extra = json.load(f)
        for mid, rows in sorted(extra.items()):
            if mid.startswith("_"):
                continue
            if mid not in models:
                unknown.append(mid)
                continue
            have = models[mid].setdefault("b", [])
            seen = {(r[0].lower(), r[1]) for r in have}
            for r in rows:
                if isinstance(r.get("score"), (int, float)) and (r["name"].lower(), r["score"]) not in seen:
                    have.append([r["name"], r["score"]] + ([r["metric"]] if r.get("metric") else []))
                    seen.add((r["name"].lower(), r["score"]))
                    added += 1
    # variants that share weights (`-highspeed`, `-turbo`, chat / reasoner modes of one repo) share what the card reports
    by_repo = {}
    for mid, m in models.items():
        if m.get("ow") and m.get("hf"):
            by_repo.setdefault(C.strip_quant(m["hf"]), []).append(mid)
    shared = 0
    for group in by_repo.values():
        if len(group) > 1:
            union = []
            for mid in group:
                for r in models[mid].get("b") or []:
                    if (r[0].lower(), r[1]) not in {(u[0].lower(), u[1]) for u in union}:
                        union.append(r)
            for mid in group:
                have = {(r[0].lower(), r[1]) for r in models[mid].get("b") or []}
                add = [r for r in union if (r[0].lower(), r[1]) not in have]
                if add and union:
                    models[mid]["b"] = (models[mid].get("b") or []) + add
                    shared += len(add)
    prov, prices = {}, {}
    for pid in sorted(os.listdir(os.path.join(a.models_dev, "providers"))):
        mroot = os.path.join(a.models_dev, "providers", pid, "models")
        if not os.path.isdir(mroot):
            continue
        out = {}
        for mid, p in walk_toml(mroot):
            t = read_toml(p)
            canon = None
            if "base_model" in t:
                canon = t.pop("base_model")
                skip = t.pop("base_model_omit", [])
                base = raw.get(canon)
                if base is not None:
                    merged = deep_merge({k: v for k, v in base.items() if k not in OMIT_FROM_BASE}, t)
                    omit(merged, skip)
                    t = merged
                else:
                    canon = None
            if not keep(t, mid, old):
                continue
            e = {}
            if canon and canon in models:
                e["c"] = canon
                own = compact(t)
                if own.get("st"):
                    e["st"] = own["st"]
                if own.get("x") and own["x"] != models[canon].get("x"):
                    e["x"] = own["x"]
            else:
                e.update(compact(t))
            c = t.get("cost") or {}
            for k, key in (("ci", "input"), ("co", "output")):
                if isinstance(c.get(key), (int, float)):
                    e[k] = c[key]
            if e.get("c") and e.get("co"):
                prices.setdefault(e["c"], []).append((e.get("ci", 0), e["co"]))
            out[mid] = e
        if out:
            prov[pid] = out
    for mid, ps in prices.items():
        models[mid]["p"] = [round(statistics.median(i for i, _ in ps), 4), round(statistics.median(o for _, o in ps), 4)]

    # open-weight registry: params from the HF API, else the verified table, else the name
    use_hf = not a.no_hf and hf_reachable()
    repos = sorted({m["hf"] for m in models.values() if m.get("hf")})
    live = {}
    if use_hf:
        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            live = {r: v for r, v in zip(repos, ex.map(hf_total, repos)) if v}
    stats = {"hf-api": 0, "table": 0, "name": 0, "unknown": 0}
    for mid, m in models.items():
        if not m.get("ow"):
            continue
        repo = m.get("hf")
        tn, an = C.parse_params(repo.split("/")[-1] if repo else "") if repo else (None, None)
        if not tn:
            tn, an2 = C.parse_params(mid.split("/")[-1])
            an = an or an2
        tab = table.get(repo) if repo else None
        total = live.get(repo)
        src = "hf-api" if total else None
        if not total and tab and tab[0]:
            total, src = tab[0], "table"
        if not total and tn:
            total, src = tn, "name"
        active = (tab[1] if tab else None) or an
        if total:
            m["pt"] = total
            m["ps"] = src
            if active and active < total:
                m["pa"] = active
            stats[src] += 1
        else:
            stats["unknown"] += 1

    snap = {"v": 1, "as_of": as_of, "source": "models.dev", "models": models, "prov": prov}
    try:
        sha = subprocess.check_output(["git", "-C", a.models_dev, "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        snap["source"] += " " + sha
    except (OSError, subprocess.CalledProcessError):
        pass
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    blob = json.dumps(snap, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with open(a.out, "wb") as f:  # mtime=0: the same input gives the same bytes
        with gzip.GzipFile(fileobj=f, mode="wb", mtime=0, compresslevel=9) as g:
            g.write(blob)
    print("benchmark gap-fill: %d rows added from extra_benchmarks.json, %d shared between variants of one repo%s" % (
        added, shared, "; unknown ids skipped: %s" % ", ".join(unknown) if unknown else ""))
    print("%d lab models (%d open-weight, params: %s), %d provider models in %d providers; %.1f KB json, %.1f KB gz -> %s%s" % (
        len(models), sum(1 for m in models.values() if m.get("ow")), ", ".join("%s %d" % kv for kv in stats.items()),
        sum(len(v) for v in prov.values()), len(prov), len(blob) / 1024, os.path.getsize(a.out) / 1024, a.out,
        "" if use_hf else "  (Hugging Face API not used)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
