#!/usr/bin/env python3
"""Cluster-hash splits, the frozen manifest and the leakage checks (S1 spec §6.2).

    python evals/split.py [--corpus C=path ...]            report clusters, splits and leakage (writes nothing)
    python evals/split.py --write [--force]                freeze evals/manifest.json (refuses on leakage)
    python evals/split.py --check [--deep]                 CI: frozen files unchanged, manifest consistent
                                                           (--deep re-clusters everything and compares)

Clusters are a union-find over (a) the same generator `scenario` or `persona` id, (b) MinHash over word 5-shingles
(64 permutations as zlib.crc32(shingle, seed_i)) joined at an estimated Jaccard >= 0.5, and (c) derived rows (S, X, R,
any row with `source_id`) joining their source. split = int(sha256(cluster_id), 16) % 100: 0-54 train, 55-64 calib,
65-74 dev, 75-99 test; agentic fixtures (F) split 50/50. A cluster holding a train.jsonl row (T) is train; a cluster
holding a legacy (L, prompts.jsonl) or persona (P) row is never train. Corpora (missing files are skipped):

    C corpus/pool.jsonl   P corpus/ood.jsonl   T train.jsonl   L prompts.jsonl   V alpha/verifiable.jsonl
    S corpus/subtasks.jsonl   X corpus/translations.jsonl   R corpus/paraphrases.jsonl   F alpha/fixtures.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import zlib

import evalkit as ek

MANIFEST = os.path.join(ek.HERE, "manifest.json")
DEFAULTS = (("C", "corpus/pool.jsonl"), ("P", "corpus/ood.jsonl"), ("T", "train.jsonl"), ("L", "prompts.jsonl"),
            ("V", "alpha/verifiable.jsonl"), ("S", "corpus/subtasks.jsonl"), ("X", "corpus/translations.jsonl"),
            ("R", "corpus/paraphrases.jsonl"), ("F", "alpha/fixtures.jsonl"))
RANGES = (("train", 55), ("calib", 65), ("dev", 75), ("test", 100))
FIXTURE_RANGES = (("train", 50), ("test", 100))
FIXED_ROLE = {"L": "legacy", "P": "persona"}          # these keep their own roles and are never trained on
N_PERM, BANDS, ROWS_PER_BAND = 64, 16, 4               # the spec's LSH (16 bands x 4 rows)
CAND_BANDS = 32                                        # candidate pairs from 32 bands x 2 rows: a superset of 16 x 4
UNION_J, LEAK_J = 0.5, 0.6
SEEDS = [zlib.crc32(b"maestro-minhash-%d" % i) for i in range(N_PERM)]
_WORD = re.compile(r"\w+")


def shingles(text, k=5):
    w = _WORD.findall((text or "").lower())
    if len(w) < k:
        return {" ".join(w).encode("utf-8")}
    return {" ".join(w[i:i + k]).encode("utf-8") for i in range(len(w) - k + 1)}


def minhash(text):
    sh = shingles(text)
    return tuple(min(zlib.crc32(s, seed) for s in sh) for seed in SEEDS)


def est_jaccard(a, b):
    return sum(x == y for x, y in zip(a, b)) / float(len(a))


def split_of(cluster_id, fixture=False):
    h = int(hashlib.sha256(cluster_id.encode("utf-8")).hexdigest(), 16) % 100
    return next(name for name, hi in (FIXTURE_RANGES if fixture else RANGES) if h < hi)


def read_items(corpora, notes=None):
    """[(letter, id, row)] from [(letter, path)]. Rows without an id (train.jsonl, prompts.jsonl) get
    "<letter>-" + sha256(normalized text)[:10]; an id already used by an earlier corpus is prefixed "<letter>:"."""
    out, seen = [], set()
    for letter, path in corpora:
        for n, r in enumerate(ek.read_jsonl(path)):
            rid = r.get("id") or "%s-%s" % (letter, ek.sha256_text(ek.norm_text(ek.item_text(r)))[:10])
            if rid in seen and not r.get("id"):
                rid = "%s-%d" % (rid, n)
            if rid in seen:
                if notes is not None:
                    notes.append("%s id %s already used; renamed %s:%s" % (letter, rid, letter, rid))
                rid = "%s:%s" % (letter, rid)
            seen.add(rid)
            out.append((letter, rid, r))
    return out


def load(corpora):
    """Rows {"id", "corpus", "sig", "sha", "scenario", "persona", "source_id"} and notes."""
    rows, notes = [], []
    for letter, rid, r in read_items(corpora, notes):
        text = ek.item_text(r)
        rows.append({"id": rid, "corpus": letter, "sig": minhash(text), "sha": ek.sha256_text(ek.norm_text(text)),
                     "scenario": r.get("scenario"), "persona": r.get("persona"), "source_id": r.get("source_id")})
    return rows, notes


def _candidates(rows):
    """Index pairs that share a 2-row band (every pair the spec's 16 x 4 bands would find, and more)."""
    width = N_PERM // CAND_BANDS
    pairs = set()
    for b in range(CAND_BANDS):
        buckets = {}
        for i, r in enumerate(rows):
            buckets.setdefault(r["sig"][b * width:(b + 1) * width], []).append(i)
        for members in buckets.values():
            if len(members) > 1:
                bysig = {}
                for i in members:
                    bysig.setdefault(rows[i]["sig"], []).append(i)
                reps = [v[0] for v in bysig.values()]
                for v in bysig.values():
                    pairs.update((v[0], j) for j in v[1:])
                pairs.update((reps[x], reps[y]) for x in range(len(reps)) for y in range(x + 1, len(reps)))
    return pairs


def cluster(rows):
    """{row id: cluster id}; the cluster id is "k:" + the smallest member id."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    index = {r["id"]: i for i, r in enumerate(rows)}
    groups = {}
    for i, r in enumerate(rows):
        for key in ("scenario", "persona"):
            if r.get(key):
                groups.setdefault((key, str(r[key])), []).append(i)
        if r.get("source_id") in index:
            union(i, index[r["source_id"]])
    for members in groups.values():
        for j in members[1:]:
            union(members[0], j)
    for i, j in _candidates(rows):
        if find(i) != find(j) and est_jaccard(rows[i]["sig"], rows[j]["sig"]) >= UNION_J:
            union(i, j)
    comp = {}
    for i, r in enumerate(rows):
        comp.setdefault(find(i), []).append(r["id"])
    out = {}
    for ids in comp.values():
        cid = "k:" + min(ids)
        for rid in ids:
            out[rid] = cid
    return out


def assign(rows, clusters):
    """({id: [corpus, split, cluster]}, errors). Role rules: L and P rows keep `legacy` / `persona`; the rest of a
    cluster that holds one is `test`; a cluster holding a T row is `train`; fixtures split 50/50; else the hash."""
    members = {}
    for r in rows:
        members.setdefault(clusters[r["id"]], []).append(r)
    out, errors = {}, []
    for cid, rs in members.items():
        corp = {r["corpus"] for r in rs}
        held = corp & set(FIXED_ROLE)
        if held and "T" in corp:
            errors.append("cluster %s holds train.jsonl rows and %s rows: %s" % (
                cid, "/".join(sorted(held)), ", ".join(sorted(r["id"] for r in rs))[:300]))
        if "T" in corp:
            base = "train"
        elif held:
            base = "test"
        else:
            base = split_of(cid, fixture=corp == {"F"})
        for r in rs:
            split = FIXED_ROLE.get(r["corpus"]) or ("train" if r["corpus"] == "T" else base)
            out[r["id"]] = [r["corpus"], split, cid]
    return out, errors


def leakage(rows, assigned):
    """Spec §6.2 disjointness: no train row with estimated Jaccard >= 0.6 to a legacy, persona, calib, dev or test row;
    disjoint normalized-prompt sha256 sets; no generator scenario in both train and test."""
    errors = []
    train = [i for i, r in enumerate(rows) if assigned[r["id"]][1] == "train"]
    is_train = set(train)
    for i, j in _candidates(rows):
        if (i in is_train) != (j in is_train) and est_jaccard(rows[i]["sig"], rows[j]["sig"]) >= LEAK_J:
            errors.append("near-duplicate across splits: %s (%s) ~ %s (%s)" % (
                rows[i]["id"], assigned[rows[i]["id"]][1], rows[j]["id"], assigned[rows[j]["id"]][1]))
    tsha = {rows[i]["sha"]: rows[i]["id"] for i in train}
    for i, r in enumerate(rows):
        if i not in is_train and r["sha"] in tsha:
            errors.append("same normalized prompt in train (%s) and %s (%s)" % (tsha[r["sha"]], assigned[r["id"]][1], r["id"]))
    scen = {}
    for r in rows:
        if r.get("scenario"):
            scen.setdefault(str(r["scenario"]), set()).add(assigned[r["id"]][1])
    errors += ["scenario %s is in train and test" % s for s, sp in scen.items() if {"train", "test"} <= sp]
    return errors


def corpora_paths(overrides=(), only=False, letters=None):
    """[(letter, path)] for the corpus files that exist: the defaults, replaced by `LETTER=PATH` overrides (only the
    overrides when `only`), restricted to `letters` when given."""
    paths = {} if only else dict((k, os.path.join(ek.HERE, v)) for k, v in DEFAULTS)
    for o in overrides:
        k, _, v = o.partition("=")
        if k not in dict(DEFAULTS) or not v:
            raise SystemExit("--corpus wants LETTER=PATH with LETTER in %s" % "".join(k for k, _ in DEFAULTS))
        paths[k] = v
    return [(k, paths[k]) for k, _ in DEFAULTS if k in paths and os.path.exists(paths[k]) and (not letters or k in letters)]


def build(corpora):
    rows, notes = load(corpora)
    clusters = cluster(rows)
    assigned, errors = assign(rows, clusters)
    return rows, assigned, notes, errors + leakage(rows, assigned)


def counts(assigned):
    out = {}
    for corpus, split, _ in assigned.values():
        out.setdefault(corpus, {}).setdefault(split, 0)
        out[corpus][split] += 1
    return out


def write_manifest(path, corpora, assigned):
    base = os.path.dirname(os.path.abspath(path))
    files = {os.path.relpath(p, base): {"corpus": k, "sha256": ek.sha256_file(p)} for k, p in corpora}
    head = {"version": 1, "created": time.strftime("%Y-%m-%d"), "ranges": dict(RANGES), "fixture_ranges": dict(FIXTURE_RANGES),
            "minhash": {"perm": N_PERM, "bands": BANDS, "rows": ROWS_PER_BAND, "cand_bands": CAND_BANDS,
                        "union_j": UNION_J, "leak_j": LEAK_J}, "files": files, "counts": counts(assigned)}
    lines = ["{"] + ['%s: %s,' % (json.dumps(k), json.dumps(v, sort_keys=True)) for k, v in head.items()]
    rows = sorted(assigned.items())
    lines.append('"rows": {')
    lines += ['%s: %s%s' % (json.dumps(k), json.dumps(v), "," if n < len(rows) - 1 else "") for n, (k, v) in enumerate(rows)]
    lines.append("}}")
    ek._atomic(path, "\n".join(lines) + "\n")


def load_splits(path=MANIFEST):
    """{id: (corpus, split, cluster)} from the frozen manifest ({} when there is none)."""
    man = ek.read_json(path) or {}
    return {k: tuple(v) for k, v in (man.get("rows") or {}).items()}


def check(path=MANIFEST, deep=False):
    """CI check. Returns (ok, messages). No manifest yet is ok (nothing frozen)."""
    man = ek.read_json(path)
    if man is None:
        return True, ["no manifest yet (%s): nothing frozen" % os.path.relpath(path, ek.ROOT)]
    base, msgs = os.path.dirname(os.path.abspath(path)), []
    for rel, f in sorted(man.get("files", {}).items()):
        p = os.path.join(base, rel)
        if not os.path.exists(p):
            msgs.append("frozen file missing: %s" % rel)
        elif ek.sha256_file(p) != f["sha256"]:
            msgs.append("frozen file changed: %s (re-run split.py --write and review the diff)" % rel)
    byc = {}
    for rid, (corpus, split, cid) in man.get("rows", {}).items():
        if split not in ("legacy", "persona"):
            byc.setdefault(cid, set()).add(split)
        if corpus == "T" and split != "train":
            msgs.append("train.jsonl row %s is not in train" % rid)
        if corpus in FIXED_ROLE and split != FIXED_ROLE[corpus]:
            msgs.append("%s row %s has split %s" % (corpus, rid, split))
    msgs += ["cluster %s crosses splits %s" % (c, sorted(s)) for c, s in byc.items() if len(s) > 1]
    if deep and not msgs:
        corpora = [(f["corpus"], os.path.join(base, rel)) for rel, f in man["files"].items()]
        order = [k for k, _ in DEFAULTS]
        _, assigned, _, errors = build(sorted(corpora, key=lambda kv: order.index(kv[0])))
        msgs += errors
        diff = [k for k in set(assigned) | set(man["rows"]) if assigned.get(k) != man["rows"].get(k)]
        msgs += ["%d rows differ from the manifest, e.g. %s" % (len(diff), sorted(diff)[:5])] if diff else []
    return not msgs, msgs or ["manifest ok: %d rows, %d files" % (len(man.get("rows", {})), len(man.get("files", {})))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append", default=[], metavar="LETTER=PATH")
    ap.add_argument("--only", action="store_true", help="use only the --corpus files, not the defaults")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--write", action="store_true", help="freeze the manifest")
    ap.add_argument("--force", action="store_true", help="--write even with leakage errors")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--deep", action="store_true", help="--check: also re-cluster and compare")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        ok, msgs = check(a.manifest, a.deep)
        print("\n".join(msgs))
        return 0 if ok else 1
    corpora = corpora_paths(a.corpus, a.only)
    if not corpora:
        print("no corpus files found", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    rows, assigned, notes, errors = build(corpora)
    res = {"rows": len(rows), "clusters": len({v[2] for v in assigned.values()}), "counts": counts(assigned),
           "notes": notes, "errors": errors, "seconds": round(time.perf_counter() - t0, 2)}
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        print("%d rows, %d clusters (%.1f s)" % (res["rows"], res["clusters"], res["seconds"]))
        for corpus, c in sorted(res["counts"].items()):
            print("  %s  %s" % (corpus, "  ".join("%s %d" % kv for kv in sorted(c.items()))))
        for line in notes + ["LEAK: " + e for e in errors]:
            print(line)
    if a.write:
        if errors and not a.force:
            print("refusing to write the manifest: %d leakage errors (--force to override)" % len(errors), file=sys.stderr)
            return 1
        write_manifest(a.manifest, corpora, assigned)
        print("wrote %s" % a.manifest)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
