"""The TypeSafe Jev System One classifier (picks the model directly), moved out of router.py.

`router` re-exports every name lazily (module __getattr__), so this module is imported only on the Jev path: a set
TYPESAFE_API_KEY with classifier auto or jev, `doctor --jev`, the proxy's Jev pick and evals that ask for it.
One POST per task to api.typesafe.ai; any failure falls back to the rules classifier.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict

from . import router as R

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_MAX_CHARS = 4000
JEV_CACHE_SIZE = 512

_JEV_JOB_CRITERIA = {
    "plan": "architecture, system design, tradeoffs, roadmaps, migration plans",
    "code": "write or change code: features, refactors, tests",
    "debug": "find and fix what is broken: errors, stack traces, flaky tests, incidents, performance regressions",
    "review": "review code, PRs or diffs; security or quality audits of existing code",
    "ops": "run and set things up: commands, CI, deploys, infra, environments, dependency upgrades across a repo",
    "research": "look things up or investigate: docs, the web, how or where something works in the codebase, comparisons",
    "data": "SQL, data analysis, spreadsheets, metrics, number crunching",
    "write": "docs, specs, READMEs, emails, long-form prose",
    "edit": "small mechanical edits: rename, format, typo, commit message, version bump, add a comment",
    "summarize": "summarize, convert or pull facts out of text or data pasted into the prompt",
    "ask": "a quick question answerable from general knowledge, or a conversational turn",
}
_JEV_DIMENSIONS = {
    "reasoning": ("How deep is the reasoning the task in state.task requires?",
                  ["none: lookup or mechanical edit", "shallow: a few obvious steps",
                   "deep: careful multi-step reasoning or debugging", "expert: novel, proof-level or research-grade reasoning"]),
    "scope": ("How many files, systems or components does the task in state.task touch?",
              ["one snippet or file", "a few related files", "many files or one whole subsystem",
               "several systems or the whole codebase"]),
    "ambiguity": ("How underspecified is the task in state.task?",
                  ["fully specified", "minor gaps", "significant open questions",
                   "open-ended; the goal itself must be worked out"]),
    "risk": ("What is the blast radius or cost of getting the task in state.task wrong?",
             ["throwaway output", "easily reviewed and reverted", "affects shared or production code",
              "security, data loss or money at stake"]),
}
_JEV_OBJECTIVES = {
    "balanced": "Choose the model that should do this task. Prefer the cheapest/fastest model that will do it well; "
                "choose a stronger model only when the task's reasoning depth, scope, ambiguity, or risk requires it.",
    "cost": "Choose the model that should do this task. Pick the cheapest model that can do it acceptably; "
            "choose a stronger model only when a cheaper one would likely fail.",
    "quality": "Choose the model that should do this task. Pick the model most likely to get it right; "
               "cost is only a tie-breaker.",
}


class JevError(Exception):
    pass


def complexity(reasoning, scope, ambiguity, risk):
    """Overall difficulty 0-3 from Jev's four 0-3 dimension scores."""
    c = max(reasoning, scope) * 0.6 + (ambiguity + risk) / 2.0 * 0.4
    return max(0, min(3, int(c + 0.5)))


def load_cache(home=None):
    data = R._read_json(os.path.join(home or R.maestro_home(), "cache.json"))
    entries = data.get("entries") if isinstance(data, dict) and data.get("version") == 2 else None
    return OrderedDict(entries.items()) if isinstance(entries, dict) else OrderedDict()


def save_cache(cache, home=None):
    R._write_json(os.path.join(home or R.maestro_home(), "cache.json"), {"version": 2, "entries": dict(cache)})


class JevClassifier:
    """One POST to Jev System One. classify() never raises: any failure falls back to rules."""

    def __init__(self, api_key, fetch=None, timeout_s=2.5, cache=None, url=JEV_URL):
        self.api_key = api_key
        self.fetch = fetch or R._http_fetch
        self.timeout_s = timeout_s
        self.cache = OrderedDict() if cache is None else cache
        self.url = url
        self.last_answers = None

    def build_request(self, text, candidates=(), objective="balanced"):
        text = text[:JEV_MAX_CHARS]
        questions = {}
        if candidates:
            questions["model"] = {
                "type": "choice",
                "instructions": _JEV_OBJECTIVES.get(objective, _JEV_OBJECTIVES["balanced"]),
                "criteria": {c["name"]: c["description"] for c in candidates},
            }
        for name, (instructions, levels) in _JEV_DIMENSIONS.items():
            questions[name] = {"type": "score", "instructions": instructions, "criteria": list(levels)}
        questions["job"] = {
            "type": "choice",
            "instructions": "Which job is the task in state.task?",
            "criteria": dict(_JEV_JOB_CRITERIA),
        }
        signals = {"chars": len(text), "words": len(text.split()), "has_code": "```" in text,
                   "has_file_refs": bool(R._FILE_RE.search(text.lower()))}
        return {"model": JEV_MODEL, "state": {"task": text, "signals": signals}, "questions": questions}

    def _query(self, text, candidates, objective):
        body = json.dumps(self.build_request(text, candidates, objective)).encode("utf-8")
        headers = {"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json",
                   "Accept": "application/json", "User-Agent": "maestro-router/" + R.VERSION}
        status, raw = self.fetch(self.url, headers, body, self.timeout_s)
        if status != 200:
            raise JevError("HTTP %s" % status)
        try:
            answers = json.loads(raw)["answers"]
            klass = answers["job"]["choice"]
            scores = {k: max(0.0, min(3.0, float(answers[k]["score"]))) for k in _JEV_DIMENSIONS}
            res = {"job": klass, "job_confidence": float(answers["job"]["confidence"]),
                   "scores": scores, "complexity": complexity(**scores), "choice": None, "probabilities": {}}
            res["confidence"] = res["job_confidence"]
            if candidates:
                m = answers["model"]
                res.update(choice=m["choice"], confidence=float(m["confidence"]),
                           probabilities={k: float(v) for k, v in (m.get("probabilities") or {}).items()})
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise JevError("bad response: %s" % e.__class__.__name__)
        if klass not in R.JOBS:
            raise JevError("unknown job %r" % (klass,))
        if candidates and res["choice"] not in [c["name"] for c in candidates]:
            raise JevError("unknown model choice %r" % (res["choice"],))
        self.last_answers = answers
        return res

    def classify(self, text, candidates=(), objective="balanced"):
        """Returns a Classification; .jev holds the model choice and complexity when Jev answered."""
        text = (text or "")[:JEV_MAX_CHARS]
        rules = R.classify_rules(text)
        names = ",".join(sorted(c["name"] for c in candidates))
        key = hashlib.sha256(("%s\0%s\0%s" % (text, names, objective)).encode("utf-8")).hexdigest()
        res = self.cache.get(key)
        cached = bool(res)
        if cached:
            self.cache.move_to_end(key)
        else:
            try:
                res = self._query(text, candidates, objective)
            except Exception as e:  # network, timeout, HTTP, JSON, schema: all fall back
                rules.note = "jev failed (%s: %s); used rules" % (e.__class__.__name__, e)
                return rules
            self.cache[key] = res
            while len(self.cache) > JEV_CACHE_SIZE:
                self.cache.popitem(last=False)
        c = R.Classification(res["job"], res["complexity"], res["confidence"], rules.needs, "jev",
                             {"rules": rules.signals})
        c.jev = {"choice": res["choice"], "confidence": round(res["confidence"], 4),
                 "probabilities": res["probabilities"], "job": res["job"],
                 "complexity": dict(res["scores"], value=res["complexity"])}
        if cached:
            c.jev["cached"] = True
        return c


def choose_jev(cls, cands, inventory, harness, policy, current_tier=None):
    """Turn Jev's model pick into a seat. Confident: take it. Otherwise take the stronger of
    Jev's pick and the ladder cell for (jev job, complexity), via the preferred seat there."""
    j = cls.jev
    pick = next(c for c in cands if c["name"] == j["choice"])
    ladder = R.target_tier(R.Classification(cls.job, cls.difficulty, 1.0), policy)
    unfit = not R._executable(pick, cls.job, cls.needs)  # e.g. Jev picked a chat-only local model for a repo job
    minc = R._pol(policy, "jev_min_confidence")
    if j["confidence"] >= minc and not unfit:
        target, note = pick["serves"], "chose %s (confidence %.2f)" % (pick["name"], j["confidence"])
    else:
        target = max(pick["serves"], ladder, key=R.RANK.get)
        note = "picked %s at confidence %.2f (< %.2f); ladder says %s, using %s" % (
            pick["name"], j["confidence"], minc, ladder, target)
        if unfit:
            note = "picked %s but it cannot do %s work; ladder says %s, using %s" % (
                pick["name"], cls.job, ladder, target)
        if target != pick["serves"] or unfit:
            seat = R.select_seat(target, cls, inventory, harness, policy, current_tier)
            seat.update(target=target, how="ladder", note=note + "; " + seat["note"])
            return seat
    if pick["kind"] == "subagent" and not pick["local"] and current_tier == target:
        seat = R._inline(harness, target, note + "; current tier is already %s" % target, [])
    else:
        seat = R._seat_result(pick, cands, R.tier_walk(target, R._pol(policy, "max_tier")), [R.tier_of(target)],
                              R._rank_fn(policy), note, cls.job)
    seat.update(target=target, how="jev")
    return seat
