#!/usr/bin/env python3
"""The teacher panel (S1 spec §6.3-§6.4): labeler manifests, ingest, Dawid-Skene aggregation and the agreement report.

    python evals/label.py batch --labeler haiku|sonnet|opus      20 items + 2 honeypots per call, mixed slices
    python evals/label.py batch --labeler fable --gold 100       the gold set (fable, with deliberation)
    python evals/label.py batch --labeler fable --adjudicate     items flagged by the last `aggregate`
    python evals/label.py ingest [--gold-review FILE]            validate replies; apply the user's gold CSV
    python evals/label.py aggregate                              -> labels/aggregate.jsonl, labels/adjudicate.jsonl
    python evals/label.py agreement                              -> labels/agreement.json

Batch files go to evals/work/label/ (format in evals/evalkit.py). Labelers never see the generator hint, Maestro's
prediction, other labels or the slice name: items are shown under opaque keys and the key -> id map stays in
_meta.jsonl. Labeled corpora: C, T, L, P, V, S (see split.py for paths).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys

import evalkit as ek
import split

LABELERS = ("haiku", "sonnet", "opus")
CORPORA = "CTLPVS"
SCHEMA_PATH = os.path.join(ek.LABELS, "schema.json")
RUBRIC_PATH = os.path.join(ek.LABELS, "rubric.md")
FIELDS = ("job", "difficulty", "min_tier", "needs_tools", "self_contained", "stakes", "detectable", "ambiguity",
          "output_tokens", "agent_turns", "rationale")
LEVELS = {"difficulty": (0, 1, 2, 3), "min_tier": ("haiku", "sonnet", "opus", "fable", "none"), "stakes": (0, 1, 2),
          "ambiguity": (0, 1, 2), "output_tokens": ("<100", "100-300", "300-1k", "1k-3k", ">3k"),
          "agent_turns": ("0", "1-3", "4-10", "11-30", ">30"), "needs_tools": (0, 1), "self_contained": (0, 1),
          "detectable": (0, 1)}
BINARY = ("needs_tools", "self_contained", "detectable")
THRESHOLDS = {"job": ("fleiss_kappa", 0.6), "difficulty": ("krippendorff_alpha_ordinal", 0.4),
              "stakes": ("krippendorff_alpha_ordinal", 0.4), "ambiguity": ("krippendorff_alpha_ordinal", 0.4),
              "needs_tools": ("cohen_kappa_mean", 0.6), "self_contained": ("cohen_kappa_mean", 0.6)}
REPORTED = {"min_tier": "krippendorff_alpha_ordinal", "detectable": "cohen_kappa_mean"}


def paths(labels_dir=None, work=None):
    d = labels_dir or ek.LABELS
    out = {k: os.path.join(d, k + ".jsonl") for k in ("panel", "gold", "honeypots", "aggregate", "adjudicate")}
    out.update(agreement=os.path.join(d, "agreement.json"), work=ek.work_dir("label", work))
    return out


# ── schema and rubric ──────────────────────────────────────────────────────────

def _check(schema, v, where):
    t = schema.get("type")
    kinds = {"object": dict, "string": str, "integer": int, "number": (int, float)}
    if t and (not isinstance(v, kinds[t]) or (t in ("integer", "number") and isinstance(v, bool))):
        return "%s: expected %s" % (where, t)
    if "const" in schema and v != schema["const"]:
        return "%s: must be %r" % (where, schema["const"])
    if "enum" in schema and v not in schema["enum"]:
        return "%s: %r not in %s" % (where, v, schema["enum"])
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if v < schema.get("minimum", -math.inf) or v > schema.get("maximum", math.inf):
            return "%s: out of range" % where
    if isinstance(v, str):
        if len(v) > schema.get("maxLength", 1 << 30) or len(v) < schema.get("minLength", 0):
            return "%s: bad length" % where
        if "pattern" in schema and not re.search(schema["pattern"], v):
            return "%s: bad format" % where
    if isinstance(v, dict):
        if not schema.get("minProperties", 0) <= len(v) <= schema.get("maxProperties", 1 << 30):
            return "%s: wrong number of entries" % where
        props = schema.get("properties", {})
        miss = [k for k in schema.get("required", ()) if k not in v]
        extra = [k for k in v if k not in props] if schema.get("additionalProperties") is False else []
        if miss or extra:
            return "%s: missing %s extra %s" % (where, miss, extra)
        for k, sub in props.items():
            if k in v:
                err = _check(sub, v[k], "%s.%s" % (where, k))
                if err:
                    return err
    return None


def load_schema():
    return ek.read_json(SCHEMA_PATH)


def rubric():
    with open(RUBRIC_PATH, encoding="utf-8") as f:
        return f.read()


def rubric_sha():
    return ek.sha256_file(RUBRIC_PATH)


def shuffled_rubric(seed):
    """The rubric with the job list order shuffled (position bias). The text is otherwise verbatim."""
    text = rubric()
    start, end = text.index("plan ("), text.index("conversational turn).") + len("conversational turn).")
    jobs = [j.strip() for j in re.split(r"\s+·\s+", " ".join(text[start:end].split()))]
    jobs[-1] = jobs[-1].rstrip(".")
    random.Random(seed).shuffle(jobs)
    return text[:start] + " · ".join(jobs) + "." + text[end:]


def coerce(obj):
    """Lenient on representation, strict on content: booleans and digit strings become ints, zero job entries are
    dropped, the rationale is cut to 25 words."""
    out = dict(obj)
    for k in ("difficulty", "needs_tools", "self_contained", "stakes", "detectable", "ambiguity"):
        v = out.get(k)
        if isinstance(v, bool):
            out[k] = int(v)
        elif isinstance(v, str) and v.strip().isdigit():
            out[k] = int(v.strip())
    if isinstance(out.get("agent_turns"), int) and not isinstance(out.get("agent_turns"), bool):
        out["agent_turns"] = str(out["agent_turns"])
    if isinstance(out.get("job"), dict):
        out["job"] = {k: v for k, v in out["job"].items() if not (isinstance(v, (int, float)) and v == 0)}
    if isinstance(out.get("rationale"), str):
        out["rationale"] = " ".join(out["rationale"].split()[:25])
    return out


def validate(row, schema=None):
    """None when `row` is a valid label row (schema.json plus the job rules), else the reason."""
    err = _check(schema or load_schema(), row, "label")
    if err:
        return err
    vals = list(row["job"].values())
    if any(abs(v * 10 - round(v * 10)) > 1e-6 for v in vals):
        return "label.job: values must be multiples of 0.1"
    if abs(sum(vals) - 1.0) > 1e-6:
        return "label.job: must sum to 1"
    return None


def top_job(dist):
    return max(ek.JOBS, key=lambda j: (dist.get(j, 0.0), -ek.JOBS.index(j)))


# ── items ──────────────────────────────────────────────────────────────────────

def load_items(corpus_args=(), only=False):
    return {rid: dict(r, _corpus=letter) for letter, rid, r in split.read_items(split.corpora_paths(corpus_args, only, CORPORA))}


def render_item(key, r):
    lines = ['<item key="%s">' % key]
    if r.get("context"):
        lines.append("Sent via: %s" % r["context"])
    if r.get("images"):
        lines.append("Attached images: %d (not shown)" % r["images"])
    hist = ek.render_history(r.get("history"))
    if hist:
        lines.append(ek.clip_middle(hist, 3000, 1000))
    lines += ["Request:", ek.clip_middle(r.get("prompt") or r.get("task") or ""), "</item>"]
    return "\n".join(lines)


_REPLY = ("Reply with only a JSON array of %d objects, one per item in the order shown. Each object has \"key\" (the "
          "item's key) plus exactly the fields above.")


def _labeled(p, labeler):
    return {r["id"] for r in ek.read_jsonl(p["panel"]) if r["labeler"] == labeler}


def _in_flight(p, kind, labeler):
    return {i for meta in ek.pending(p["work"]).values() if meta.get("kind") == kind and meta.get("labeler") == labeler
            for i in meta["keys"].values()}


def batch(labeler, items, p, size=20, per_file=10, seed=0, gold_n=0, adjudicate=False, limit=None):
    """Write labeling calls; returns (files, number of calls)."""
    sha = rubric_sha()
    gold = {r["id"]: r for r in ek.read_jsonl(p["gold"])}
    if gold_n:
        kind, have = "gold", set(gold) | _in_flight(p, "gold", labeler)
        picked = sorted(items, key=lambda i: ek.unit("gold", seed, i))
        by = {}
        for i in picked:
            by.setdefault(items[i].get("slice") or items[i]["_corpus"], []).append(i)
        todo, groups = [], [by[k] for k in sorted(by)]
        while len(todo) + len(have) < gold_n and any(groups):
            for g in groups:
                if g and len(todo) + len(have) < gold_n:
                    i = g.pop(0)
                    if i not in have:
                        todo.append(i)
    elif adjudicate:
        kind = "adjudicate"
        done = _labeled(p, "fable-adj") | _in_flight(p, "adjudicate", labeler)
        todo = [r["id"] for r in ek.read_jsonl(p["adjudicate"]) if r["id"] in items and r["id"] not in done]
        panel = {}
        for r in ek.read_jsonl(p["panel"]):
            if r["labeler"] in LABELERS:
                panel.setdefault(r["id"], []).append(r)
    else:
        kind = "panel"
        done = _labeled(p, labeler) | _in_flight(p, "panel", labeler)
        todo = sorted((i for i in items if i not in done and i not in gold), key=lambda i: ek.unit("mix", seed, labeler, i))
    todo = todo[:limit] if limit else todo
    calls, done_ids = [], ek.load_done(p["work"])
    for n in range(0, len(todo), size):
        group = todo[n:n + size]
        gid = ek.sha256_text("\x1f".join([kind, labeler] + sorted(group)))[:12]
        rng = random.Random(gid)
        ids, hp = list(group), []
        if kind == "panel" and len(gold) >= 2:
            hp = rng.sample(sorted(gold), 2)
            for h in hp:
                ids.insert(rng.randrange(len(ids) + 1), h)
        keyed = [("i%02d" % (k + 1), i) for k, i in enumerate(ids)]
        view = lambda i: items.get(i) or gold[i].get("_item") or {}  # noqa: E731
        if kind == "gold":
            extra = "Think carefully about every field of every item before you answer; these labels are the gold set.\n\n"
        elif kind == "adjudicate":
            extra = ("Three labelers disagreed on these items. Each item is followed by their labels (anonymized, in random "
                     "order) with rationales. Decide each field yourself and output your final label.\n\n")
        else:
            extra = ""
        parts = []
        for k, i in keyed:
            part = render_item(k, view(i))
            if kind == "adjudicate":
                labs = panel.get(i, [])[:3]
                rng.shuffle(labs)
                part += "".join("\nLabel %s: %s" % ("ABC"[x], json.dumps({f: r[f] for f in FIELDS}, ensure_ascii=False))
                                for x, r in enumerate(labs))
            parts.append(part)
        prompt = "%s\n\n%sItems (%d):\n\n%s\n\n%s" % (shuffled_rubric(gid).strip(), extra, len(keyed), "\n\n".join(parts),
                                                      _REPLY % len(keyed))
        cid = ek.attempt_id(done_ids, "lab:%s:%s:%s" % (kind, labeler, gid))
        call = {"call_id": cid, "model": labeler, "prompt": prompt, "expect": "json"}
        calls.append((call, {"kind": kind, "labeler": labeler, "keys": dict(keyed), "honeypots": [k for k, i in keyed if i in hp],
                             "rubric_sha": sha}))
    return ek.emit(p["work"], "label-%s-%s" % (kind, labeler), calls, per_file), len(calls)


# ── ingest ─────────────────────────────────────────────────────────────────────

def ingest(p, items=None):
    """Validate replies; append label rows. Returns a summary dict."""
    schema, done, out = load_schema(), [], {"ok": 0, "invalid_calls": 0, "invalid_items": 0, "discarded": 0, "errors": []}
    replies = ek.read_replies(p["work"])
    out["errors"] += replies.pop(None, [])
    gold = {r["id"]: r for r in ek.read_jsonl(p["gold"])}
    rows, hps = [], []
    for cid, meta in ek.pending(p["work"]).items():
        rep = replies.get(cid)
        if rep is None:
            continue
        try:
            arr = ek.parse_json_reply(rep["reply"])
            if not isinstance(arr, list):
                raise ValueError("expected a JSON array")
        except ValueError as e:
            out["invalid_calls"] += 1
            done.append({"call_id": cid, "status": "invalid", "reason": str(e)})
            continue
        got, bad = {}, []
        for obj in arr:
            if not isinstance(obj, dict) or obj.get("key") not in meta["keys"]:
                bad.append("unknown key")
                continue
            lab = "fable-adj" if meta["kind"] == "adjudicate" else "gold" if meta["kind"] == "gold" else meta["labeler"]
            row = dict(coerce({k: v for k, v in obj.items() if k != "key"}), id=meta["keys"][obj["key"]], labeler=lab,
                       schema=1, rubric_sha=meta["rubric_sha"])
            err = validate(row, schema)
            if err:
                bad.append("%s: %s" % (obj["key"], err))
            else:
                got[obj["key"]] = row
        hp_keys = [k for k in meta.get("honeypots", []) if k in got and meta["keys"][k] in gold]
        hp_ok = [top_job(got[k]["job"]) == top_job(gold[meta["keys"][k]]["job"]) for k in hp_keys]
        hps += [{"call_id": cid, "labeler": meta["labeler"], "id": meta["keys"][k], "ok": ok} for k, ok in zip(hp_keys, hp_ok)]
        if len(meta.get("honeypots", [])) == 2 and len(hp_ok) == 2 and not any(hp_ok):
            out["discarded"] += 1
            done.append({"call_id": cid, "status": "discarded", "reason": "both honeypots wrong on job"})
            continue
        keep = [r for k, r in got.items() if k not in meta.get("honeypots", [])]
        if meta["kind"] == "gold" and items:
            for r in keep:
                r["_item"] = {k: items[r["id"]].get(k) for k in ("prompt", "history", "context", "images", "task")}
        rows.append((meta["kind"], keep))
        out["ok"] += len(keep)
        out["invalid_items"] += len(bad)
        out["errors"] += ["%s %s" % (cid, b) for b in bad[:3]]
        done.append({"call_id": cid, "status": "ok", "n": len(keep), "bad": len(bad)})
    ek.append_jsonl(p["panel"], [r for kind, rs in rows if kind != "gold" for r in rs])
    ek.append_jsonl(p["gold"], [dict(r, verified=False) for kind, rs in rows if kind == "gold" for r in rs])
    ek.append_jsonl(p["honeypots"], hps)
    ek.mark_done(p["work"], done)
    return out


def gold_csv(p, path):
    """Write the gold set for the user to check: blank `<field>_fix` = agree."""
    gold = ek.read_jsonl(p["gold"])
    cols = [f for f in FIELDS if f != "rationale"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "text"] + [c for f2 in cols for c in (f2, f2 + "_fix")])
        for r in gold:
            text = ((r.get("_item") or {}).get("prompt") or "")[:1500]
            w.writerow([r["id"], text] + [c for f2 in cols for c in (_cell(r[f2]), "")])
    return len(gold)


def _cell(v):
    return " ".join("%s=%g" % kv for kv in sorted(v.items())) if isinstance(v, dict) else str(v)


def apply_gold_review(p, path):
    """Apply the user's CSV: a non-blank `<field>_fix` replaces the value; every row becomes verified."""
    gold = {r["id"]: r for r in ek.read_jsonl(p["gold"])}
    schema, n, errs = load_schema(), 0, []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g = gold.get(row.get("id"))
            if not g:
                continue
            new = dict(g)
            for k, v in row.items():
                if k.endswith("_fix") and v.strip():
                    field = k[:-4]
                    new[field] = ({a: float(b) for a, b in (t.split("=") for t in v.split())} if field == "job" else v.strip())
            new = coerce(new)
            err = validate({k: v for k, v in new.items() if k not in ("verified", "_item")}, schema)
            if err:
                errs.append("%s: %s" % (g["id"], err))
                continue
            gold[g["id"]] = dict(new, verified=True)
            n += 1
    ek.write_jsonl(p["gold"], list(gold.values()))
    return n, errs


# ── aggregation (Dawid-Skene) ──────────────────────────────────────────────────

def dawid_skene(obs, k, iters=30, alpha=0.5):
    """obs {item: {labeler: class index}} -> ({item: posterior[k]}, {labeler: k x k confusion}). One confusion matrix per
    labeler, Dirichlet(alpha) smoothing, majority initialization."""
    items = sorted(obs)
    labelers = sorted({j for o in obs.values() for j in o})
    post = {}
    for i in items:
        v = [0.0] * k
        for c in obs[i].values():
            v[c] += 1.0
        post[i] = [x / sum(v) for x in v]
    conf = {}
    for _ in range(iters):
        prior = [alpha + sum(post[i][c] for i in items) for c in range(k)]
        prior = [math.log(x / sum(prior)) for x in prior]
        conf = {j: [[alpha] * k for _ in range(k)] for j in labelers}
        for i in items:
            for j, lab in obs[i].items():
                for c in range(k):
                    conf[j][c][lab] += post[i][c]
        logc = {j: [[math.log(x / sum(row)) for x in row] for row in m] for j, m in conf.items()}
        for i in items:
            lp = [prior[c] + sum(logc[j][c][lab] for j, lab in obs[i].items()) for c in range(k)]
            mx = max(lp)
            e = [math.exp(x - mx) for x in lp]
            post[i] = [x / sum(e) for x in e]
    return post, conf


def _panel(p):
    by = {}
    for r in ek.read_jsonl(p["panel"]):
        by.setdefault(r["id"], {}).setdefault(r["labeler"], r)
    return by


def _value_index(field, r):
    return ek.JOBS.index(top_job(r["job"])) if field == "job" else LEVELS[field].index(r[field])


def aggregate(p, seed=0):
    """Soft labels for every item the panel labeled; returns (rows, adjudicate rows)."""
    by = _panel(p)
    soft = {}
    for field in ("job",) + tuple(LEVELS):
        k = len(ek.JOBS) if field == "job" else len(LEVELS[field])
        obs = {i: {j: _value_index(field, r) for j, r in labs.items() if j in LABELERS} for i, labs in by.items()}
        post, _ = dawid_skene({i: o for i, o in obs.items() if o}, k)
        soft[field] = post
    rows, adj = [], []
    for i, labs in sorted(by.items()):
        panel = [labs[j] for j in LABELERS if j in labs]
        if not panel:
            continue
        mean = {j: sum(r["job"].get(j, 0.0) for r in panel) / len(panel) for j in ek.JOBS}
        row = {"id": i, "n": len(panel), "job": {j: 0.5 * soft["job"][i][n] + 0.5 * mean[j] for n, j in enumerate(ek.JOBS)}}
        for f in LEVELS:
            row[f] = soft[f][i][1] if f in BINARY else soft[f][i]
        why = _triggers(i, panel, seed)
        if why:
            adj.append({"id": i, "why": why})
        fab = labs.get("fable-adj")
        if fab:
            row["job"] = {j: 0.7 * fab["job"].get(j, 0.0) + 0.3 * row["job"][j] for j in ek.JOBS}
            for f in LEVELS:
                hot = [1.0 if lv == fab[f] else 0.0 for lv in LEVELS[f]]
                row[f] = 0.7 * hot[1] + 0.3 * row[f] if f in BINARY else [0.7 * h + 0.3 * x for h, x in zip(hot, row[f])]
        row["job"] = {j: round(v, 4) for j, v in row["job"].items() if v >= 5e-5}
        for f in LEVELS:
            row[f] = round(row[f], 4) if f in BINARY else [round(x, 4) for x in row[f]]
        row["adjudicated"] = bool(fab)
        rows.append(row)
    for g in ek.read_jsonl(p["gold"]):
        if g["id"] not in by:
            row = {"id": g["id"], "n": 0, "gold": True, "verified": bool(g.get("verified")), "adjudicated": False,
                   "job": g["job"]}
            for f in LEVELS:
                hot = [1.0 if lv == g[f] else 0.0 for lv in LEVELS[f]]
                row[f] = hot[1] if f in BINARY else hot
            rows.append(row)
    ek.write_jsonl(p["aggregate"], rows)
    ek.write_jsonl(p["adjudicate"], adj)
    return rows, adj


def _triggers(i, panel, seed):
    why = []
    if len(panel) >= 2:
        tops = [top_job(r["job"]) for r in panel]
        if max(tops.count(t) for t in tops) < 2:
            why.append("no job majority")
        if max(r["difficulty"] for r in panel) - min(r["difficulty"] for r in panel) >= 2:
            why.append("difficulty spread")
        mt = [LEVELS["min_tier"].index(r["min_tier"]) for r in panel]
        if max(mt) - min(mt) >= 2:
            why.append("min_tier spread")
        for f in ("needs_tools", "self_contained"):
            if len({r[f] for r in panel}) > 1:
                why.append(f + " disagree")
    if ek.unit("adjudicate", seed, i) < 0.05:
        why.append("random 5%")
    return why


# ── agreement ──────────────────────────────────────────────────────────────────

def fleiss_kappa(table):
    """table: rows of category counts with the same number of raters per row."""
    table = [r for r in table if sum(r) >= 2]
    if not table:
        return None
    n = sum(table[0])
    table = [r for r in table if sum(r) == n]
    big_n = len(table)
    pbar = sum((sum(c * c for c in r) - n) / (n * (n - 1.0)) for r in table) / big_n
    pk = [sum(r[c] for r in table) / (big_n * float(n)) for c in range(len(table[0]))]
    pe = sum(x * x for x in pk)
    return None if pe >= 1 else (pbar - pe) / (1 - pe)


def krippendorff_ordinal(units, k):
    """units: lists of category indexes (0..k-1) given to the same item by different labelers."""
    o = [[0.0] * k for _ in range(k)]
    for vals in units:
        if len(vals) < 2:
            continue
        for a in range(len(vals)):
            for b in range(len(vals)):
                if a != b:
                    o[vals[a]][vals[b]] += 1.0 / (len(vals) - 1)
    nc = [sum(row) for row in o]
    n = sum(nc)
    if n <= 1:
        return None

    def d2(c, e):
        lo, hi = min(c, e), max(c, e)
        return (sum(nc[lo:hi + 1]) - (nc[c] + nc[e]) / 2.0) ** 2
    do = sum(o[c][e] * d2(c, e) for c in range(k) for e in range(k))
    de = sum(nc[c] * nc[e] * d2(c, e) for c in range(k) for e in range(k)) / (n - 1)
    return None if de == 0 else 1 - do / de


def cohen_kappa(pairs):
    pairs = list(pairs)
    if not pairs:
        return None
    cats = sorted({x for p in pairs for x in p})
    n = float(len(pairs))
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in cats)
    return 1.0 if pe >= 1 else (po - pe) / (1 - pe)


def agreement(p, items=None):
    by = _panel(p)
    full = {i: labs for i, labs in by.items() if all(j in labs for j in LABELERS)}
    rep = {"n_items": len(by), "n_full_panel": len(full), "fields": {}}
    for field in list(THRESHOLDS) + list(REPORTED):
        stat = THRESHOLDS.get(field, (REPORTED.get(field), None))[0]
        if stat == "fleiss_kappa":
            k = len(ek.JOBS)
            table = []
            for labs in full.values():
                row = [0] * k
                for j in LABELERS:
                    row[_value_index("job", labs[j])] += 1
                table.append(row)
            v = fleiss_kappa(table)
        elif stat == "krippendorff_alpha_ordinal":
            v = krippendorff_ordinal([[_value_index(field, labs[j]) for j in LABELERS if j in labs] for labs in by.values()],
                                     len(LEVELS[field]))
        else:
            ks = [cohen_kappa((labs[a][field], labs[b][field]) for labs in by.values() if a in labs and b in labs)
                  for x, a in enumerate(LABELERS) for b in LABELERS[x + 1:]]
            ks = [x for x in ks if x is not None]
            v = sum(ks) / len(ks) if ks else None
        thr = THRESHOLDS.get(field, (None, None))[1]
        rep["fields"][field] = {"statistic": stat, "value": None if v is None else round(v, 4), "threshold": thr,
                                "half_weight": bool(thr is not None and (v is None or v < thr))}
    hp = ek.read_jsonl(p["honeypots"])
    rep["honeypot_accuracy"] = {j: round(sum(r["ok"] for r in hp if r["labeler"] == j) / max(1, sum(r["labeler"] == j for r in hp)), 4)
                                for j in LABELERS if any(r["labeler"] == j for r in hp)}
    agg = {r["id"]: r for r in ek.read_jsonl(p["aggregate"])}
    if items:
        hint = [(items[i]["hint"], agg[i]) for i in agg if i in items and isinstance(items[i].get("hint"), dict)]
        if hint:
            jo = sum(h.get("job") == top_job(a["job"]) for h, a in hint) / float(len(hint))
            do = sum(h.get("difficulty") == max(range(4), key=lambda d: a["difficulty"][d]) for h, a in hint) / float(len(hint))
            rep["hint_check"] = {"n": len(hint), "job_agreement": round(jo, 4), "difficulty_agreement": round(do, 4),
                                 "panel_may_read_generator_cues": jo >= 0.95}
        legacy = [(items[i], agg[i]) for i in agg if i in items and items[i]["_corpus"] == "L"]
        if legacy:
            rep["legacy_noise"] = {
                "n": len(legacy),
                "job_disagree": round(sum(r.get("job") != top_job(a["job"]) for r, a in legacy) / float(len(legacy)), 4),
                "difficulty_disagree": round(sum(r.get("difficulty") != max(range(4), key=lambda d: a["difficulty"][d])
                                                 for r, a in legacy) / float(len(legacy)), 4)}
    gold = ek.read_jsonl(p["gold"])
    rep["gold"] = {"n": len(gold), "verified": bool(gold) and all(r.get("verified") for r in gold)}
    if gold and not rep["gold"]["verified"]:
        rep["gold"]["flag"] = "gold_unverified"
    ek.write_json(p["agreement"], rep)
    return rep


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("batch", "ingest", "aggregate", "agreement"))
    ap.add_argument("--labeler", choices=LABELERS + ("fable",))
    ap.add_argument("--gold", type=int, default=0, metavar="N", help="batch: build the N-item gold set (fable)")
    ap.add_argument("--adjudicate", action="store_true", help="batch: adjudication calls (fable)")
    ap.add_argument("--gold-review", metavar="CSV", help="ingest: apply the user's gold CSV")
    ap.add_argument("--gold-csv", metavar="CSV", help="aggregate: write the gold set as a CSV for the user")
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--per-file", type=int, default=10, help="calls per batch file")
    ap.add_argument("--limit", type=int, help="batch: at most this many items")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", action="append", default=[], metavar="LETTER=PATH")
    ap.add_argument("--only", action="store_true", help="only the --corpus files")
    ap.add_argument("--labels-dir", help="default evals/labels")
    ap.add_argument("--work", help="default evals/work")
    a = ap.parse_args(argv)
    p = paths(a.labels_dir, a.work)
    if a.cmd == "batch":
        if not a.labeler or (a.labeler == "fable") != bool(a.gold or a.adjudicate):
            ap.error("batch: --labeler haiku|sonnet|opus, or --labeler fable with --gold N or --adjudicate")
        items = load_items(a.corpus, a.only)
        if not ek.read_jsonl(p["gold"]) and not (a.gold or a.adjudicate):
            print("note: no gold set yet; batches carry no honeypots (label.py batch --labeler fable --gold 100)")
        files, n = batch(a.labeler, items, p, a.size, a.per_file, a.seed, a.gold, a.adjudicate, a.limit)
        print("%d calls in %d files under %s" % (n, len(files), p["work"]))
        print(ek.quota_status()[1])
        return 0
    if a.cmd == "ingest":
        if a.gold_review:
            n, errs = apply_gold_review(p, a.gold_review)
            print("gold: %d rows verified" % n + "".join("\n  " + e for e in errs))
            return 1 if errs else 0
        res = ingest(p, load_items(a.corpus, a.only))
        print("labels ok %(ok)d; invalid calls %(invalid_calls)d; invalid items %(invalid_items)d; discarded batches %(discarded)d" % res)
        for e in res["errors"][:20]:
            print("  " + e)
        return 0
    if a.cmd == "aggregate":
        rows, adj = aggregate(p, a.seed)
        print("aggregated %d items; %d flagged for adjudication -> %s" % (len(rows), len(adj), p["adjudicate"]))
        if a.gold_csv:
            print("gold CSV: %d rows -> %s" % (gold_csv(p, a.gold_csv), a.gold_csv))
        return 0
    rep = agreement(p, load_items(a.corpus, a.only))
    for f, s in rep["fields"].items():
        print("%-15s %-28s %s%s" % (f, s["statistic"], s["value"], "  (below %.2f: half weight)" % s["threshold"] if s["half_weight"] else ""))
    for k in ("hint_check", "legacy_noise", "honeypot_accuracy", "gold"):
        if k in rep:
            print("%s: %s" % (k, rep[k]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
