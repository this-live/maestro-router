"""`arena.py export <run_id>`: outcome rows in the evals/alpha format (S1 spec §7.8) so S1 can train on them.

Each seat stands for a tier: its category (the CLI's tier from Maestro's suggestions or your `models` config, else the
catalog's), a local model's from its strength G. `y` / `tout` are keyed by the anchor tier names the alpha tooling uses
(fast = haiku, standard = sonnet, strong = opus, frontier = fable); when several seats share a tier, the Claude seat
stands for it (the anchors are Claude tiers), else the first by id. Every seat's own outcome is kept in `y_seat` with its
tier, family and harness in `seat_meta`, so nothing is lost when two seats share a tier. Rows carry no answer text.
"""
from __future__ import annotations

import os

import alpha
import split
from . import rank
from .common import ek, m
from .report import load, outcomes

ANCHOR = dict(zip(m.TIERS, ek.TIERS))          # fast -> haiku ... frontier -> fable
SOURCE = {"verifiable": "verifiable", "open": "open", "agentic": "fixture"}


def seat_tier(seat):
    c = seat.get("category")
    if c in m.TIERS:
        return c
    if seat.get("strength") is not None:
        from maestro import score
        return score.tier_of_g(float(seat["strength"]))
    return "fast"


def rows(run):
    meta = run["meta"]["seats"]
    tiers = {s: seat_tier(v) for s, v in meta.items()}
    out = []
    for suite in ("verifiable", "open", "agentic"):
        obs = outcomes(run, suite)
        if not obs:
            continue
        soft = rank.acceptance(run["judgments"]) if suite == "open" else {}
        rep_of = {}
        for s, per in obs.items():
            for t, xs in per.items():
                for y, _, _, r in xs:
                    if y is not None:
                        rep_of.setdefault(t, {}).setdefault(s, {})[int(r["rep"])] = (y, r)
        for t, per in sorted(rep_of.items()):
            task = run["tasks"].get((suite, t)) or {}
            stand = {}
            for s in sorted(per, key=lambda s: ((meta.get(s) or {}).get("family") != "anthropic", s)):
                stand.setdefault(ANCHOR[tiers.get(s, "fast")], s)
            y = {tier: per[s][1][0] for tier, s in stand.items() if 1 in per[s]}
            y2 = {tier: per[s][2][0] for tier, s in stand.items() if 2 in per[s]}
            tout = {}
            for tier, s in stand.items():
                if 1 in per[s]:
                    r = per[s][1][1]
                    u = r.get("usage") or {}
                    tout[tier] = int(u.get("out") or 0) or ek.est_tokens(r.get("answer") or r.get("final") or "")
            cluster = "arena:%s:%s" % (suite, t)
            row = {"id": "arena:%s:%s" % (suite, t), "cluster": cluster, "split": split.split_of(cluster, fixture=suite == "agentic"),
                   "source": SOURCE[suite], "group": "agentic" if suite == "agentic" else "chat", "job": task.get("job"),
                   "difficulty": task.get("difficulty"), "lang": task.get("lang") or "en", "context": "chat" if suite != "agentic" else "repo",
                   "words": task.get("words"), "phase2": "full" if len(y) == 4 else "partial", "ipw": 1.0,
                   "y": y, "y2": y2, "tin": task.get("tin"), "tout": tout,
                   "t_star": next((tier for tier in ek.TIERS if y.get(tier, 0) >= 0.5), "none"), "nonmonotone": alpha.is_nonmonotone(y),
                   "stand_in": stand, "y_seat": {s: v[1][0] for s, v in per.items() if 1 in v},
                   "seat_meta": {s: {"tier": tiers.get(s), "family": (meta.get(s) or {}).get("family"),
                                     "harness": (meta.get(s) or {}).get("harness"), "model": (meta.get(s) or {}).get("model")}
                                 for s in per}}
            if suite == "open":
                row["y_soft"] = {tier: round(soft[(t, 1, s)], 4) for tier, s in stand.items() if (t, 1, s) in soft}
            out.append(row)
    return out


def export(run_id, runs_dir=None, out=None):
    run = load(run_id, runs_dir)
    data = rows(run)
    path = out or os.path.join(run["paths"]["dir"], "outcomes.jsonl")
    ek.write_jsonl(path, data)
    return path, data
