#!/usr/bin/env python3
"""beta_pub: the cross-vendor slope of the success link (S1 spec §3.4, step 1). Stdlib only (numpy is not needed).

    python evals/fit_link.py --swebench-dir pubdata/swev --leaderboards pubdata/swebench_leaderboards.json \\
                             --aider pubdata/polyglot_leaderboard.yml [--scaffold mini+vendor|mini|all] [--out FILE]

Public systems are matched to catalog models by model id (the leaderboard's "Model: ..." tag, Aider's `model`) and else
by display name. Each source's quantity is regressed on z = PhiInv(clip(catalog code strength, 0.02, 0.98)):

    SWE-bench Verified   the system's 1PL theta (fitted on all Verified systems x instances; fixed-scaffold
                         mini-SWE-agent runs plus vendor runs by default; "Attempts - 2+" systems dropped)
    Aider polyglot       logit(pass_rate_1), `diff` edit format only

The two slopes are pooled by inverse variance into (beta_pub, se_pub). Fewer than 15 distinct matched catalog models:
beta_pub = 1.0, se = 0.5. Validation (reported, not gated): leave-one-model-out AUC on the matched SWE-bench cells with
the catalog-linked theta vs the fitted theta (target: <= 0.02 AUC loss). Writes evals/alpha/link_pub.json, which is
`train_s1.py --link`'s input: beta_pub, se_pub, a_chat_hat (a_chat of the shipped s1.bin, else 1.2; train_s1 re-estimates it
once), anchor_models (decide.json's anchor tiers) and zbar (each anchor model's mean_j PhiInv(clip(catalog strength_j))).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys

import alpha_stats as st
import evalkit as ek

OUT = os.path.join(ek.ALPHA, "link_pub.json")
MIN_MATCHED = 15
_NORM = statistics.NormalDist()


def name_key(s):
    """Word tokens in any order, version numbers in order: "Claude 4.5 Opus (high)" == "claude-opus-4-5", but
    "gpt-4.5" != "gpt-5.4". None when there is nothing to match on."""
    s = re.sub(r"\(.*?\)", " ", (s or "").lower())
    s = re.sub(r"(?<![0-9])(19|20)\d{6}(?![0-9])|(19|20)\d\d-\d\d-\d\d", " ", s)
    toks = [t for t in re.findall(r"[a-z]+|\d+", s) if t not in ("latest", "preview", "exp", "instruct", "it")]
    words = tuple(sorted(t for t in toks if not t.isdigit()))
    return (words, tuple(t for t in toks if t.isdigit())) if words else None


def parse_yaml_list(text):
    """The Aider leaderboard's flat YAML: a list of `- key: value` mappings."""
    rows, cur = [], None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("- "):
            cur = {}
            rows.append(cur)
            line = line[2:]
        if cur is None or ":" not in line:
            continue
        k, _, v = line.strip().partition(":")
        v = v.strip().strip("'\"")
        try:
            cur[k.strip()] = float(v) if re.match(r"^-?\d+(\.\d+)?$", v) else v
        except ValueError:
            cur[k.strip()] = v
    return rows


def load_swebench(swe_dir, leaderboards, scaffold="mini+vendor"):
    """([system dicts], sorted instance ids). Every Verified system is kept for the 1PL; `use` marks the regression set."""
    meta = {}
    if leaderboards and os.path.exists(leaderboards):
        lb = ek.read_json(leaderboards) or {}
        for board in lb.get("leaderboards", []):
            if board.get("name") == "Verified":
                meta = {r["folder"]: r for r in board.get("results", [])}
    results = {}
    for folder, r in meta.items():  # per-instance details in the leaderboard (mini-SWE-agent runs have them)
        pid = r.get("per_instance_details")
        if isinstance(pid, dict) and pid:
            results[folder] = (set(pid), {k for k, v in pid.items() if isinstance(v, dict) and v.get("resolved")})
    for path in sorted(glob.glob(os.path.join(swe_dir, "*.json"))):
        folder = os.path.basename(path)[:-5]
        data = ek.read_json(path)
        if folder in results or not isinstance(data, dict) or not isinstance(data.get("resolved"), list):
            continue
        seen = {x for v in data.values() if isinstance(v, list) for x in v if isinstance(x, str)}
        results[folder] = (seen, set(data["resolved"]))
    systems, items = [], set()
    for folder in sorted(results):
        seen, resolved = results[folder]
        items.update(seen)
        r = meta.get(folder, {})
        tags = r.get("tags") or []
        model = next((t.split(":", 1)[1].strip().rstrip("/").split("/")[-1] for t in tags if t.startswith("Model:")), None)
        mini = r.get("agent") == "mini-SWE-agent" or "_mini-" in folder
        vendor = bool(r.get("agent_org")) and r.get("agent_org") == r.get("model_org")
        multi = any("Attempts - 2+" in t for t in tags)
        use = not multi and (scaffold == "all" or mini or (scaffold == "mini+vendor" and vendor))
        systems.append({"folder": folder, "model": model, "display": r.get("model_display") or r.get("name") or folder,
                        "resolved": resolved, "use": use})
    return systems, sorted(items)


def load_aider(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = parse_yaml_list(f.read())
    return [{"model": r.get("model"), "display": r.get("model"), "pass1": float(r["pass_rate_1"]) / 100.0}
            for r in rows if r.get("edit_format") == "diff" and isinstance(r.get("pass_rate_1"), float)]


def catalog_matcher():
    """(match(model_id, display) -> lab id or None, strength(lab id) -> catalog code strength, jobs(lab id) -> {job: strength}
    or None) from the bundled catalog."""
    from maestro import catalog, score
    cat = catalog.load_snapshot()
    catalog._finish(cat)
    ctx = score.make_context(cat)
    names = {}
    for lid, mrec in cat["models"].items():
        for key in (name_key(mrec.get("n")), name_key(lid.split("/")[-1])):
            if key:
                names.setdefault(key, lid)

    def match(model_id, display):
        if model_id:
            hit = cat["_idx"]["norm"].get(catalog.norm_id(model_id))
            if hit:
                return hit[0]
        for s in (model_id, display):
            lid = names.get(name_key(s))
            if lid:
                return lid
        return None

    def strength(lid):
        return score.score_lab(ctx, cat, lid)["jobs"]["code"]

    def jobs(lid):
        return score.score_lab(ctx, cat, lid)["jobs"] if lid in cat["models"] else None
    return match, strength, jobs


def z_of(s):
    return _NORM.inv_cdf(min(0.98, max(0.02, s)))


def anchors(jobs, anchor_models, a_chat=1.2):
    """What train_s1 needs besides beta_pub (S1 spec §3.2, §3.5): {"anchor_models", "zbar": {tier: mean_j z}, "a_chat_hat"}.
    A tier whose model is not in the catalog has no zbar (train_s1 then keeps the link frozen)."""
    zbar = {}
    for t, lid in sorted(anchor_models.items()):
        js = jobs(lid) or {}
        vals = [z_of(v) for v in js.values() if isinstance(v, (int, float))]
        if vals:
            zbar[t] = round(sum(vals) / len(vals), 6)
    return {"anchor_models": dict(anchor_models), "zbar": zbar, "a_chat_hat": a_chat}


def ols(xs, ys):
    """(intercept, slope, se of slope)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if n < 3 or sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    s2 = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys)) / (n - 2)
    return a, b, math.sqrt(s2 / sxx)


def fit(systems, items, aider, match, strength):
    cells = [(it, s["folder"], 1.0 if it in s["resolved"] else 0.0, 1.0) for s in systems for it in items]
    theta, b = st.fit_1pl(cells, iters=40, tol=1e-4) if cells else ({}, {})
    res = {"per_source": {}, "matches": {}}
    per_model = {}
    for s in systems:
        lid = match(s["model"], s["display"]) if s["use"] else None
        res["matches"][s["folder"]] = lid
        if lid:
            per_model.setdefault(lid, []).append(theta[s["folder"]])
    swe = {lid: sum(v) / len(v) for lid, v in per_model.items()}
    aid = {}
    for r in aider:
        lid = match(r["model"], r["display"])
        res["matches"]["aider:" + str(r["model"])] = lid
        if lid:
            aid.setdefault(lid, []).append(st.logit(r["pass1"], 0.005))
    aid = {lid: sum(v) / len(v) for lid, v in aid.items()}
    zs = {lid: z_of(strength(lid)) for lid in set(swe) | set(aid)}
    slopes = []
    for name, q in (("swebench", swe), ("aider", aid)):
        ids = sorted(q)
        fitres = ols([zs[i] for i in ids], [q[i] for i in ids]) if len(ids) >= 3 else None
        res["per_source"][name] = {"n": len(ids)} if not fitres else {"n": len(ids), "alpha": fitres[0], "beta": fitres[1], "se": fitres[2]}
        if fitres and fitres[2] > 0:
            slopes.append((fitres[1], fitres[2]))
    n_matched = len(zs)
    res["n_matched"] = n_matched
    if n_matched >= MIN_MATCHED and slopes:
        w = [1 / se ** 2 for _, se in slopes]
        res["beta_pub"] = sum(wi * bi for wi, (bi, _) in zip(w, slopes)) / sum(w)
        res["se_pub"] = math.sqrt(1 / sum(w))
        res["fallback"] = False
    else:
        res.update(beta_pub=1.0, se_pub=0.5, fallback=True)
    res["lomo"] = lomo(systems, items, b, theta, swe, zs, res["matches"])
    return res


def lomo(systems, items, b, theta, swe, zs, matches):
    """Leave-one-model-out on matched SWE-bench systems: AUC with theta = alpha + beta*z (fitted without that model)
    vs AUC with the system's fitted theta."""
    ids = sorted(swe)
    if len(ids) < 4:
        return None
    ys, p_fit, p_link = [], [], []
    for held in ids:
        rest = [i for i in ids if i != held]
        fr = ols([zs[i] for i in rest], [swe[i] for i in rest])
        if not fr:
            continue
        th_link = fr[0] + fr[1] * zs[held]
        for s in systems:
            if matches.get(s["folder"]) != held:
                continue
            for it in items:
                ys.append(1.0 if it in s["resolved"] else 0.0)
                p_fit.append(st.sigmoid(theta[s["folder"]] - b[it]))
                p_link.append(st.sigmoid(th_link - b[it]))
    a_fit, a_link = st.auc(ys, p_fit), st.auc(ys, p_link)
    return {"auc_fit": a_fit, "auc_link": a_link, "loss": None if None in (a_fit, a_link) else a_fit - a_link, "target": 0.02,
            "cells": len(ys)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swebench-dir", required=True)
    ap.add_argument("--leaderboards")
    ap.add_argument("--aider")
    ap.add_argument("--scaffold", choices=("mini+vendor", "mini", "all"), default="mini+vendor")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    systems, items = load_swebench(a.swebench_dir, a.leaderboards, a.scaffold)
    if not systems:
        print("no SWE-bench per-instance results in %s" % a.swebench_dir, file=sys.stderr)
        return 2
    match, strength, jobs = catalog_matcher()
    res = fit(systems, items, load_aider(a.aider), match, strength)
    from maestro import catalog, decide, s1
    md = s1.load()[0]
    prm = decide.params(md and md["header"])        # decide.json's anchor models; a_chat of the shipped s1.bin
    res.update(anchors(jobs, prm["irt"]["anchor_models"], prm["irt"]["a"]["chat"]))
    res["catalog_sha"] = catalog.snapshot_sha()          # decide flags link_stale when models.json was scored against another catalog
    res["sources"] = [os.path.basename(a.swebench_dir.rstrip("/")) + " (SWE-bench Verified, %d systems x %d instances)" % (len(systems), len(items))]
    if a.aider:
        res["sources"].append(os.path.basename(a.aider) + " (Aider polyglot, diff)")
    ek.write_json(a.out, res)
    print("zbar %s; a_chat_hat %.3f" % (json.dumps(res["zbar"]), res["a_chat_hat"]))
    print("beta_pub %.3f (se %.3f) from %d matched models%s" % (res["beta_pub"], res["se_pub"], res["n_matched"],
                                                               " [fallback: fewer than %d]" % MIN_MATCHED if res["fallback"] else ""))
    for name, r in res["per_source"].items():
        print("  %-9s %s" % (name, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})))
    if res["lomo"]:
        print("  LOMO AUC fitted %.3f vs linked %.3f (loss %.3f; target <= 0.02)" % (res["lomo"]["auc_fit"], res["lomo"]["auc_link"], res["lomo"]["loss"]))
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
