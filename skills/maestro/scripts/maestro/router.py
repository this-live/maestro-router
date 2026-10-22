#!/usr/bin/env python3
"""Maestro router: route each task to the right model you already have.

Skill-mode logic (classifier, ladder, seats, `route`, `install`, `models`), stdlib only, Python >= 3.9.
The package around it (providers, clients, quota) is documented in docs/DESIGN.md.

    maestro-router route "<task>" [--harness H] [--current-tier T] [--classifier auto|rules|s1|jev] [--decider ladder|utility] [hints]
    maestro-router detect [--catalog] [--scan [--save]] [--json]
    maestro-router run <model-id> "<task>" [--system S]     (chat with a local or network model)
    maestro-router models [--suggest | --save | --check] [set|unset <harness>.<category> [<model>] [--effort E]] [--json]
    maestro-router install [--harness claude-code|opencode|gemini-cli|all] [--dry-run]
    maestro-router doctor [--jev]
    maestro-router config
    maestro-router s1 | decide | outcome | feedback | learn    (S1: commands.py)

Sections: config/paths, classifier (rules), classifier (jev), policy,
inventory/detect, seats and models, route, install, run, CLI.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import re
import sys
import time
import zlib

VERSION = "3.4.1"
TIERS = ("fast", "standard", "strong", "frontier")        # the cloud ladder, weakest first
CELLS = ("local",) + TIERS                                # what a ladder cell can hold
CATEGORIES = CELLS + ("specialist",)                      # what a decision's `category` can be
RANK = {c: i for i, c in enumerate(CELLS)}
JOBS = ("plan", "code", "debug", "review", "ops", "research", "data", "write", "edit", "summarize", "ask")
# jobs that read or change the repo: a plain local model (chat only) cannot do them
TOOL_JOBS = frozenset(("code", "debug", "review", "ops", "research", "edit"))
MARKER = "maestro-generated"
EFFORTS = ("low", "medium", "high", "xhigh", "max")      # reasoning effort; only Codex takes it (`-c model_reasoning_effort=`)


def tier_of(cell):
    """The cloud tier a ladder cell falls back to: a `local` cell means "a local model, else fast"."""
    return "fast" if cell == "local" else cell


# ════════════════════════════════════════════════════════════════════════════
# 1. config / paths
# ════════════════════════════════════════════════════════════════════════════

# (job) -> ladder cell for difficulty 0..3; "local" = a local model if one is eligible, else fast
DEFAULT_LADDER = {
    # Retuned on measured outcomes (evals pilot, 100 checker-graded self-contained tasks): Sonnet-class models solved
    # 97-100% of hard and very hard items, so hard (d2) work goes to standard and strong is kept for very hard (d3) work.
    "ask":       ("local", "fast", "standard", "standard"),
    "summarize": ("local", "fast", "standard", "standard"),
    "edit":      ("local", "fast", "standard", "standard"),
    "write":     ("fast", "standard", "standard", "strong"),
    "data":      ("fast", "standard", "standard", "strong"),
    "research":  ("fast", "standard", "standard", "strong"),
    "code":      ("fast", "standard", "standard", "strong"),
    "ops":       ("fast", "standard", "standard", "strong"),
    "review":    ("standard", "standard", "standard", "strong"),
    "debug":     ("standard", "standard", "standard", "strong"),
    "plan":      ("standard", "standard", "strong", "frontier"),
}

DEFAULT_CONFIG = {
    "classifier": "auto",       # auto (jev if TYPESAFE_API_KEY is set, else rules) | rules | jev
    "objective": "balanced",    # jev: balanced | cost | quality
    "jev_min_confidence": 0.55, # jev: below this, take the stronger of Jev's pick and the ladder cell
    "min_confidence": 0.5,      # rules: below this, escalate one cell
    "max_tier": "frontier",     # never route to a cloud model above this tier
    "prefer_local": True,       # use local and network models for the jobs they can do
    "prefer": [],               # seat names to try first, e.g. ["codex"]
    "job_seats": {},            # pin a job to a seat, e.g. {"research": "gemini-cli"}
    "models": {},               # harness -> category -> model or {"model", "effort"}; "local" pins local models
    "harness_models": {},       # deprecated alias of `models`, read when `models` is empty
    "ladder": {},               # job -> [d0, d1, d2, d3] cells
    "model_notes": {},          # jev: override a candidate's description, e.g. {"claude-code:opus": "..."}
    "local_endpoints": [],      # URLs or {"name", "url", "kind": "ollama"|"openai", "api_key_env"}
    "local_models": {},         # model name -> {"serves": category, "specialist_for": [jobs]}
    "local_allow_load": False,  # route to models a server has installed but not loaded (Ollama, LM Studio load on first use; can exhaust memory)
    "specialists": {},          # model name -> [jobs] it is tuned for
    "run_timeout_s": 300.0,     # `maestro-router run` timeout
    "request_timeout_s": 120.0, # v3 proxy: upstream request timeout (per read when streaming)
    "cli_timeout_s": 300.0,     # v3 proxy: subscription-CLI call timeout
    "providers": {},            # v3: per-provider settings, e.g. {"azure": {"models": ["gpt-5"]}, "groq": {"base_url": "..."}}
    "models_overrides": {},     # v3: per-model user overrides applied on load, e.g. {"openai/gpt-5": {"category": "strong", "enabled": false}}
    "log_prompts": False,       # v3 proxy: keep the first 120 characters of each prompt in the decision log (off by default)
    "session_affinity_header": "",  # v3 proxy: request header that identifies a conversation (X-Session-Id is always honoured)
    "allowed_origins": [],      # v3 proxy: browser origins besides the server's own that may call it
    # S1 and expected-utility routing (docs/DESIGN.md, v3 Stage 3); the proxy validates the settable ones (server.CONFIG_KEYS)
    "decider": "ladder",        # ladder | utility (expected utility in dollars over every seat; the default flips after the release gate)
    "wtp_usd": None,            # utility: $ value of a success; null = the calibrated V0 of the objective
    "latency_usd_per_s": 0.002, # utility: $ per second of waiting
    "background_latency_usd_per_s": 0.0002,  # the same for background requests (X-Maestro-Background, maestro/<job>)
    "risk": 1.0,                # utility, 0-3: weight on an unnoticed failure and on how unsure a seat's ability is
    "quota_reserve": 0.15,      # utility, 0-0.5: share of a subscription window shared with your own session kept free
    "local_usd_per_hour": 0.0,  # utility: what an hour of a local model costs you
    "inline_access": "subscription",  # utility: how the calling agent itself is billed: subscription | api
    "private": False,           # only local models (or the calling agent) see tasks; on by itself for text with a key-like string
    "log_decisions": True,      # `maestro-router route` appends heads, options and a MinHash (never text) to decisions.jsonl
    "s1_shadow": True,          # also compute the other decider and log its pick (shadow mode)
    "personalize": True,        # learn personal offsets from outcomes (personal.json)
    "explore": 0.0,             # proxy, 0-0.1: sometimes take a safe runner-up and log the propensity
    "systemone_delegate": None, # /v1/systemone: a model you serve that answers the non-native questions
    "systemone_cloud": False,   # allow a cloud model as that delegate
}
_RANGES = {"latency_usd_per_s": (0, 1e3), "background_latency_usd_per_s": (0, 1e3), "risk": (0, 3), "quota_reserve": (0, 0.5),
           "local_usd_per_hour": (0, 1e6), "explore": (0, 0.1)}

SEAT_ALIASES = {"claude": "claude-code", "gemini": "gemini-cli", "cursor": "cursor-agent"}


def maestro_home(env=None):
    env = os.environ if env is None else env
    return env.get("MAESTRO_HOME") or os.path.join(_user_home(env), ".maestro")


def ensure_home(home):
    """Create the Maestro home owner-only (0700) and tighten an existing one this user owns. Every path that makes the home uses this;
    a failure to chmod (not our directory, a filesystem without modes) is ignored."""
    os.makedirs(home, mode=0o700, exist_ok=True)
    try:
        if os.name == "posix" and os.stat(home).st_uid == os.getuid():
            os.chmod(home, 0o700)
    except OSError:
        pass


def _user_home(env=None):
    env = os.environ if env is None else env
    return env.get("HOME") or env.get("USERPROFILE") or os.path.expanduser("~")


def config_path(home=None):
    return os.path.join(home or maestro_home(), "config.json")


def load_config(home=None):
    """Effective config: defaults overlaid with config.json (invalid parts ignored)."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        with open(config_path(home), "r", encoding="utf-8") as f:
            user = json.load(f)
    except (OSError, ValueError):
        return cfg
    if not isinstance(user, dict):
        return cfg
    for key, default in DEFAULT_CONFIG.items():
        v = user.get(key)
        if isinstance(default, float) and isinstance(v, int) and not isinstance(v, bool):
            v = float(v)
        if isinstance(v, type(default)):
            cfg[key] = v
    if cfg["max_tier"] not in TIERS:
        cfg["max_tier"] = "frontier"
    if cfg["classifier"] not in ("auto", "rules", "s1", "jev"):
        cfg["classifier"] = "auto"
    if cfg["objective"] not in OBJECTIVES:
        cfg["objective"] = "balanced"
    cfg["decider"] = cfg["decider"] if cfg["decider"] in ("ladder", "utility") else "ladder"
    cfg["inline_access"] = cfg["inline_access"] if cfg["inline_access"] in ("subscription", "api") else "subscription"
    cfg.update({k: DEFAULT_CONFIG[k] for k, (lo, hi) in _RANGES.items() if not lo <= cfg[k] <= hi})
    w, d = user.get("wtp_usd"), user.get("systemone_delegate")
    cfg["wtp_usd"] = float(w) if isinstance(w, (int, float)) and not isinstance(w, bool) and w > 0 else None
    cfg["systemone_delegate"] = d if isinstance(d, str) and d else None
    cfg["ladder"] = {
        c: v for c, v in cfg["ladder"].items()
        if c in JOBS and isinstance(v, list) and len(v) == 4 and all(t in CELLS for t in v)
    }
    return cfg


def _pol(policy, key):
    return (policy or {}).get(key, DEFAULT_CONFIG[key])


def user_models(policy):
    """The configured map {harness: {category: cell}}: `models`, else the deprecated `harness_models`."""
    return _pol(policy, "models") or _pol(policy, "harness_models") or {}


def _cell(v):
    """(model, effort) of a config cell, "id" or {"model": id, "effort": E}; None if unset or malformed."""
    v, e = (v.get("model"), v.get("effort")) if isinstance(v, dict) else (v, None)
    return (v, e if e in EFFORTS else None) if isinstance(v, str) and v else None


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_json(path, data, indent=1, private=True):
    """Atomic JSON write. private (Maestro's own state): the directory is the owner-only home and the temp file is created 0600, so
    the file is never readable by others, not even briefly. private=False is for a file that belongs to another tool."""
    try:
        d = os.path.dirname(path)
        ensure_home(d) if private else os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600 if private else 0o666)
        if private and hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)    # a stale temp file from a crashed write keeps its old mode otherwise
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def _http_fetch(url, headers, body, timeout, direct=False):
    """(status, bytes). POST when body is given, else GET. Network errors propagate.
    direct=True ignores proxy settings (local and LAN model servers are never behind the proxy)."""
    import urllib.error
    import urllib.request

    from .safe_http import opener   # keys never follow a redirect to another origin

    req = urllib.request.Request(url, data=body, headers=headers or {}, method="POST" if body is not None else "GET")
    try:
        with opener(direct).open(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _local_fetch(url, headers, body, timeout):
    return _http_fetch(url, headers, body, timeout, direct=True)


# ════════════════════════════════════════════════════════════════════════════
# 2. classifier: rules (default, offline)
#    Keyword tables and scoring salvaged from v1 taskclass.py; difficulty from
#    v1 reasoning.py demand + length + scope signals.
# ════════════════════════════════════════════════════════════════════════════


class Classification:
    def __init__(self, job, difficulty, confidence, needs=(), classifier="rules", signals=None, note=""):
        self.job = job
        self.difficulty = difficulty
        self.confidence = confidence
        self.needs = tuple(needs)
        self.classifier = classifier
        self.signals = signals if signals is not None else {}
        self.note = note  # why a requested classifier fell back, if it did
        self.jev = None   # Jev's model choice and complexity, when Jev answered

    def to_dict(self):
        return {
            "job": self.job,
            "difficulty": self.difficulty,
            "confidence": round(self.confidence, 3),
            "needs": list(self.needs),
            "classifier": self.classifier,
        }


# Cue words per job, matched against the instruction only (fenced blocks are pasted material,
# not intent). Single words match by token (plus crude stemming); anything with a space or symbol
# matches as a substring.
_PATTERNS = {
    "plan": (
        "architecture", "architect", "system design", "strategy", "strategic", "roadmap", "tradeoff", "trade-off",
        "decide between", "should we", "how should we", "prioritiz", "prioritis", "risk assessment", "pre-mortem",
        "postmortem", "post-mortem", "migration plan", "adr", "design doc", "rfc", "long-term",
        "evaluate options", "pros and cons", "proposal", "propose", "plan", "design", "weigh", "recommend",
        "phased", "rollout plan", "scaling", "multi-tenant", "multi-region", "best way to", "how to structure",
        "structure",
    ),
    "code": (
        "implement", "refactor", "unit test", "pytest", "jest", "test suite", "tests", "function", "class",
        "method", "module", "endpoint", "api route", "schema", "typescript", "python", "javascript", "rust",
        "golang", "compile", "regex", "react", "component", "algorithm", "optimiz", "middleware", "decorator",
        "hook", "handler", "feature", "add support", "add a flag", "add an option", "port this", "rewrite in",
        "write a test", "add tests", "script", "parser", "pagination", "helper", "utility",
    ),
    "debug": (
        "bug", "fix", "debug", "stack trace", "traceback", "exception", "error", "errors", "failing", "fails",
        "failed", "crash", "flaky", "broken", "not working", "doesn't work", "isn't working", "regression",
        "race condition", "deadlock", "memory leak", "segfault", "root cause", "incident", "outage", "throws",
        "raises", "panic", "null pointer", "nullpointer", "off by one", "intermittent", "hangs", "hanging",
        "slow", "latency", "p99", "p95", "spike", "timeout", "timeouts", "500s", "why is", "why does", "why did",
        "why's", "went wrong", "undefined is not", "keyerror", "typeerror", "valueerror", "oom", "leak",
        "diagnose", "culprit", "track down", "figure out why", "figure out what", "find out why", "find why",
        "what caused", "what's causing", "what is causing", "the cause", "cause of", "stopped working", "no longer",
        "silently", "wrong result", "stuck", "unstick", "double count", "nothing happens", "fails with", "failure",
        "failures", "unexpected", "make this failing", "make the failing", "make ci pass", "pass again",
    ),
    "review": (
        "review", "code review", "audit", "pull request", "pr", "diff", "look over", "sanity check",
        "vulnerability", "vulnerabilities", "security review", "lgtm", "code smell", "critique", "feedback on",
        "any issues", "spot the", "check this for", "looks ok", "look ok", "look right", "second pair", "nitpick",
        "threat model",
    ),
    "ops": (
        "browser", "click", "navigate", "ssh", "execute", "deploy", "restart", "multi-step", "workflow",
        "automate", "automation", "provision", "install", "screenshot", "end-to-end", "e2e", "scrape", "kubectl",
        "terraform", "set up", "rerun", "re-run", "roll back", "rollback", "pipeline", "cron", "docker", "helm",
        "make ci", "ci pass", "pass again", "green", "clone", "bisect", "failover", "staging", "on the server",
        "tail the", "curl", "npm test", "pytest -", "run the", "run `", "then run", "and run", "dependencies",
        "deps", "bump deps", "bump dependencies", "upgrade", "github actions", "makefile", "nginx", "systemd",
        "rebase", "cherry-pick", "force push", "env vars", "kubernetes", "k8s", "ci", "container",
    ),
    "research": (
        "research", "look up", "lookup", "find sources", "sources", "cite", "citation", "competitor",
        "competitive", "survey", "literature", "investigate", "investigation", "compare", "comparison",
        "alternatives", "state of the art", "papers", "best practices", "find where", "find out", "where is",
        "where's", "where are", "where do", "where in", "how does our", "how do we", "trace how", "explore",
        "look into", "what do the docs", "in the docs", "release notes say", "prior art", "search the web",
        "docs say", "who uses", "which library", "which libraries", "options for",
    ),
    "data": (
        "sql", "query", "queries", "postgres", "sqlite", "bigquery", "snowflake", "redshift", "group by", "join",
        "window function", "pandas", "dataframe", "spreadsheet", "excel", "pivot", "metrics", "dashboard",
        "weekly active", "wau", "dau", "mau", "retention", "cohort", "funnel", "conversion", "percentile",
        "median", "average", "aggregate", "forecast", "chart", "plot", "statistics", "stats", "analysis",
        "analyze the data", "revenue", "churn", "kpi", "a/b test", "p-value", "significance", "numbers",
        "by plan", "by month", "by week", "per user", "growth rate",
    ),
    "write": (
        "draft", "email", "readme", "blog post", "cover letter", "documentation", "docs", "spec", "tutorial",
        "guide", "announcement", "press release", "changelog", "release notes", "status update", "meeting notes",
        "agenda", "outline", "runbook", "prd", "user story", "essay", "newsletter", "reply to", "respond to",
        "rewrite", "polish", "tone", "wording", "pull request description", "contributing", "checklist", "todo",
        "to-do", "template", "jira", "write a short", "write up", "write-up", "explainer", "faq", "onboarding",
        "job description", "linkedin", "tweet", "speech", "pitch", "landing page copy",
    ),
    "edit": (
        "rename", "reformat", "format", "typo", "indentation", "whitespace", "spelling", "grammar",
        "commit message", "bump", "version", "comment", "docstring", "type hints", "annotations",
        "named constants", "gitignore", "license", "copyright", "env.example", "proofread", "unused import",
        "unused imports", "sort imports", "trailing", "one-liner", "find and replace", "capitalize", "lowercase",
        "uppercase", "prettier", "deprecation notice", "boilerplate", "replicas", "timeout to", "change the port",
    ),
    "summarize": (
        "summariz", "summaris", "summary", "tl;dr", "tldr", "extract", "pull out", "classify", "classification",
        "categoriz", "categoris", "label", "tag", "convert", "to json", "to csv", "to yaml", "to markdown",
        "translate", "transcribe", "key points", "list the", "list all", "list every", "parse", "normaliz",
        "normalis", "dedupe", "deduplicate", "identify", "what does this", "explain this", "explain what this",
        "action items", "timeline", "group these", "count how", "unique", "boil down", "condense", "shorten",
        "digest", "recap", "bullet points", "eli5",
    ),
    "ask": (
        "hi", "hello", "hey", "thanks", "thank you", "how are", "tell me", "explain", "what is", "what are",
        "what's", "who is", "how do i", "why is", "why do", "what does", "difference between", "can you help",
        "quick question", "is it", "should i", "worth", "meaning of", "does it support",
    ),
}
# Verbs that open an instruction ("please fix ...", "can you summarize ..."): worth extra.
_LEAD = {
    "edit": "rename reformat proofread comment",
    "write": "draft compose rewrite polish reply respond outline",
    "summarize": "summarize summarise extract parse convert classify categorize categorise tag translate count list identify",
    "ops": "run install deploy provision configure rotate restart bisect rollback clone tail set migrate upgrade",
    "research": "research investigate compare survey trace explore find look understand",
    "plan": "design plan architect propose prioritize prioritise recommend decide",
    "code": "implement refactor add build port",
    "debug": "debug fix troubleshoot diagnose",
    "review": "review audit critique",
    "data": "query analyze analyse chart plot",
    "ask": "hi hello hey thanks thank explain",
}
_FILLER = frozenset("please pls can could would will you kindly ok okay so and just also now then i need want to like help me us let's lets".split())
_QUESTION_LEAD = frozenset("what whats what's why how when where which who is are do does did can could should would".split())
_PROJECT_WORDS = frozenset("our we us repo codebase repository monorepo service services app pipeline production prod staging".split())
_EDIT_VERBS = frozenset("add update fix change write remove move create bump rename make correct replace delete set".split())
_EDIT_TARGETS = frozenset(
    "docstring docstrings comment comments indentation whitespace typo typos version license header copyright "
    "annotation annotations constant constants gitignore spelling grammar wording replicas port timeout import imports "
    "tabs spaces quotes indentation alphabetically".split())
_WRITE_TARGETS = frozenset("readme documentation docs changelog notice guide tutorial".split())
_INFRA = frozenset("kubernetes k8s pod pods ecs eks aws gcp s3 cluster nginx docker helm terraform ci prod production deployment deployments".split())
_DEIXIS = ("this ", "these ", "the following", " below", " above", "pasted", "attached", "here is", "here's")
_MATERIAL_NOUNS = frozenset("log logs csv json yaml xml text paragraph article thread transcript output trace diff table email notes report list dump stacktrace html spec snippet file files".split())

# on equal scores prefer the earlier job
_TIE_ORDER = ("plan", "ops", "debug", "review", "research", "data", "edit", "code", "summarize", "write", "ask")
_TOKEN_RE = re.compile(r"[a-z0-9_+#.'-]+")

_WORD_INDEX = {}   # word -> [jobs]
_PHRASES = {}      # job -> tuple of substrings
_LEAD_INDEX = {}   # verb -> job
for _c, _pats in _PATTERNS.items():
    _PHRASES[_c] = tuple(p for p in _pats if not re.fullmatch(r"[a-z0-9-]+", p))
    for _p in _pats:
        if re.fullmatch(r"[a-z0-9-]+", _p):
            _WORD_INDEX.setdefault(_p, []).append(_c)
for _c, _verbs in _LEAD.items():
    for _v in _verbs.split():
        _LEAD_INDEX.setdefault(_v, _c)


def _stems(tok):
    out = []
    for suf in ("ing", "ed", "es", "s", "e", "d"):
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            st = tok[:-len(suf)]
            out.append(st)
            if suf in ("ing", "ed") and st[-1] == st[-2]:
                out.append(st[:-1])
    return out


def _stem_or_self(tok):
    return tok if tok in _LEAD_INDEX else next((st for st in _stems(tok) if st in _LEAD_INDEX), tok)


def _keyword_scores(lower, toks):
    hits = set()
    for tok in set(toks):
        for c in _WORD_INDEX.get(tok, ()):
            hits.add((c, tok))
        if len(tok) > 3:
            for st in _stems(tok):
                if len(st) >= 3:
                    for c in _WORD_INDEX.get(st, ()):
                        hits.add((c, st))
    scores = dict.fromkeys(JOBS, 0.0)
    for c, _ in hits:
        scores[c] += 1
    for c, phrases in _PHRASES.items():
        scores[c] += sum(1 for p in phrases if p in lower)
    return scores


def _lead_verb(toks):
    """The first non-filler token, as (token, class of that verb or None)."""
    for t in toks[:6]:
        if t not in _FILLER:
            return t, _LEAD_INDEX.get(_stem_or_self(t))
    return "", None


_EXTRACT_CODE_RE = re.compile(r"\binto (?:a |an |the |named |separate )?(?:named |new |helper |shared )?(?:functions?|methods?|classes|class|modules?|constants?|variables?|components?|hooks?|helpers?|files?)\b")
_TO_CODE_RE = re.compile(
    r"\b(?:to|into|as) (?:a |an |the )?(?:typescript|python|rust|golang|go|java|javascript|kotlin|swift|c\+\+|c#|sql|pydantic|zod|"
    r"dataclass\w*|interfaces?|types?|classes|structs?)\b")
_CAUSE_RE = re.compile(r"\b(why|what (?:is |was |could be )?(?:causing|caused|the cause)|root[- ]cause|culprit)\b")
_FILE_RE = re.compile(r"(?<![\w/-])[\w/-]+\.(?:py|ts|tsx|js|jsx|mjs|go|rs|sql|java|rb|cpp|sh|ya?ml|toml|json|md)\b")
_REPO_RE = re.compile(r"\b(?:repo|codebase|the file|this file|pull request|branch|commit)\b")
_VISION_RE = re.compile(r"\b(?:image|images|screenshot|photo|picture|diagram|pdf)\b")

# reasoning-demand signals (from v1 reasoning.py)
_STEP_RE = re.compile(
    r"\b(step[- ]by[- ]step|derive|proof|prove|show that|therefore|hence|optimi[sz]e|"
    r"minimi[sz]e|maximi[sz]e|trade[- ]?off|constraint|simulate|monte carlo|"
    r"expected value|probability|bayesian|regression)\b", re.I)
_CODE_RE = re.compile(
    r"\b(debug|stack trace|traceback|segfault|race condition|deadlock|memory leak|"
    r"regression|refactor|architect|design (a|the) system|concurrency|async|"
    r"optimi[sz]e the|why does (this|my|the) code)\b", re.I)
_DELIBERATE_RE = re.compile(
    r"\b(think (hard|carefully|deeply)|reason (through|about)|walk me through|"
    r"explain your reasoning|chain of thought|analyze (in depth|carefully)|"
    r"consider (all|every) (option|angle|possibility))\b", re.I)
_EASY_RE = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|yes|no|sure|what time|what's the weather|translate this)\b",
    re.I)
# scope / hardness signals used for difficulty
_SCOPE_RE = re.compile(
    r"\b(across|entire|whole|every|multiple|several|end[- ]to[- ]end|cross[- ]\w+|codebase|"
    r"monorepo|distributed|microservices?|migrat\w+|from scratch|production|scal(?:e|ing|able)|"
    r"backwards? compat\w*|zero[- ]downtime|all (?:the )?(?:files|modules|services|endpoints|tests))\b")
_HARD_RE = re.compile(
    r"\b(design|architect\w*|prove|proof|algorithm|concurren\w+|race condition|deadlock|"
    r"consensus|threat model|invariant|complexity|security|vulnerabilit\w+|"
    r"performance|idempoten\w+|transaction\w*|lock[- ]free|multi[- ]region|real[- ]time)\b")
_EASY_WORD_RE = re.compile(r"\b(typo|rename|one[- ]liner|quick|quickly|simple|small|trivial|tiny|just|only|nothing else)\b")
_ANALYZE_RE = re.compile(
    r"\b(analy[sz]\w*|patterns?|predict\w*|root[- ]cause|correlat\w*|diagnos\w*|distinguish|anomal\w*|"
    r"reconstruct\w*|why|reason(?:ing)?|subtle|tricky|carefully|trade-?offs?|edge cases?)\b")
_STEPS_RE = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*])\s+|\b(?:first|then|finally|afterwards?|after that|next)\b")


def reasoning_demand(text):
    """Structural estimate of required test-time compute, 0..1 (from v1 reasoning.py)."""
    if not text.strip():
        return 0.0
    score = min(len(_STEP_RE.findall(text)) * 0.14, 0.34)
    score += min(len(_CODE_RE.findall(text)) * 0.13, 0.30)
    score += min(len(_DELIBERATE_RE.findall(text)) * 0.18, 0.36)
    words = len(text.split())
    sentences = len([s for s in re.split(r"[.!?]+", text) if s.strip()])
    score += min(words / 600, 0.12)
    score += min(sentences / 30, 0.08)
    score += min(text.count("?") * 0.04, 0.10)
    if _EASY_RE.search(text):
        score = min(score, 0.08)
    return max(0.0, min(1.0, score))


# how much work each job implies before any per-prompt signals
_JOB_PRIOR = {"plan": 1.4, "ops": 1.2, "research": 1.2, "debug": 1.1, "code": 0.9, "review": 0.9,
              "data": 0.8, "write": 0.4, "edit": 0.2, "summarize": 0.1, "ask": 0.0}


def _difficulty(lower, demand, job):
    """0 trivial, 1 routine, 2 hard, 3 frontier: class prior + reasoning demand + length + scope.
    `lower` is the instruction only, so a long pasted file does not make "fix the typo" hard."""
    words = len(lower.split())
    pts = _JOB_PRIOR[job] + demand * 3.0
    pts += 0.5 if words >= 40 else 0.0
    pts += 0.5 if words >= 100 else 0.0
    pts += 0.5 if words >= 250 else 0.0
    pts += min(len(set(_SCOPE_RE.findall(lower))) * 0.6, 1.2)
    pts += min(len(set(_HARD_RE.findall(lower))) * 0.6, 1.2)
    n_steps = len(_STEPS_RE.findall(lower))
    pts += 0.5 if n_steps >= 2 else 0.0
    pts += 0.5 if n_steps >= 4 else 0.0
    pts += 0.8 if _ANALYZE_RE.search(lower) else 0.0
    if _EASY_WORD_RE.search(lower):
        pts -= 0.5
    if pts < 0.5:
        return 0, pts
    if pts < 1.5:
        return 1, pts
    if pts < 2.5:
        return 2, pts
    return 3, pts


_NUM_RE = re.compile(r"^[0-9][0-9.,]*[a-z%+]{0,2}$")
_FILEISH_RE = re.compile(r"^[\w/-]+\.[a-z]{1,4}$")


def split_material(text):
    """(instruction, material_chars): text outside ``` fences is the instruction; fenced text is pasted material.
    A str.find loop (an unclosed fence runs to the end); the `re.S` regex it replaces took 2 ms on a 200 KB paste."""
    parts, i = [], 0
    while "```" in text:
        j = text.find("```", i)
        if j < 0:
            parts.append(text[i:])
            break
        parts.append(text[i:j] + " ")
        e = text.find("```", j + 3)
        if e < 0:
            break
        i = e + 3
    instr = "".join(parts) if parts else text
    mat = len(text) - len(instr) if parts else 0
    if len(instr) > 6000:  # the instruction is at the start or the end; keep classify fast on huge pastes
        instr = instr[:4000] + " " + instr[-2000:]
    return instr, mat


def text_features(sg):
    """Feature set for the learned model from _signals(): instruction unigrams+bigrams and structural flags."""
    toks = [("<n>" if _NUM_RE.match(t) else "<file>" if _FILEISH_RE.match(t) else t) for t in sg["toks"][:160]]
    feats = set(toks)
    feats.update(a + " " + b for a, b in zip(toks, toks[1:]))
    n, mat, lower = len(toks), sg["mat"], sg["lower"]
    feats.add("#len%d%d%d%d" % (n > 8, n > 16, n > 32, n > 64))
    feats.add("#mat%d" % (0 if not mat else 1 if mat < 600 else 2))
    feats.add("#scope%d" % min(len(set(_SCOPE_RE.findall(lower))), 2))
    feats.add("#hard%d" % min(len(set(_HARD_RE.findall(lower))), 2))
    feats.add("#steps%d" % min(len(_STEPS_RE.findall(lower)), 3))
    for flag in ("edit", "infra", "deictic", "project", "question"):
        if sg[flag]:
            feats.add("#" + flag)
    if sg["lead_cls"]:
        feats.add("#lead_" + sg["lead_cls"])
    if _ANALYZE_RE.search(lower):
        feats.add("#analyze")
    if _EASY_WORD_RE.search(lower):
        feats.add("#easy")
    return feats


def _signals(text):
    """Parse once: instruction tokens, pasted-material size, heuristic class scores and difficulty."""
    instr, mat = split_material(text)
    lower = instr.lower()
    toks = [t for t in (t.strip(".'-") for t in _TOKEN_RE.findall(lower)) if t]
    tset = set(toks)
    scores = _keyword_scores(lower, toks)
    lead, lead_cls = _lead_verb(toks)
    if lead_cls:
        scores[lead_cls] += 2
    deictic = any(d in lower for d in _DEIXIS)
    has_material = bool(mat) or (deictic and bool(_MATERIAL_NOUNS & tset))
    if has_material and (lead_cls == "summarize" or (lead_cls is None and scores["summarize"])):
        scores["summarize"] += 3            # a verb like summarize/convert/parse on given text
    if lead_cls in ("code", "debug", "review") and mat:
        scores[lead_cls] += 1
    edit = lead in _EDIT_VERBS and bool(_EDIT_TARGETS & tset)
    if edit:
        scores["edit"] += 4                 # small routine edit: docstring, typo, version bump, ...
    elif lead in _EDIT_VERBS and _WRITE_TARGETS & tset:
        scores["write"] += 3                # docs, README, changelog
    if lead_cls == "research" and _CAUSE_RE.search(lower):
        scores["debug"] += 3                # "investigate why ...", "find out what caused ..."
    if lead == "extract" and _EXTRACT_CODE_RE.search(lower):
        scores["summarize"] -= 4            # "extract this into a function", not "extract the emails from this"
        scores["edit"] += 4
    if lead_cls == "summarize" and _TO_CODE_RE.search(lower):
        scores["summarize"] -= 3            # "convert this schema to TypeScript interfaces" is code, not data wrangling
        scores["code"] += 4
    infra = bool(_INFRA & tset)
    if lead_cls in ("code", "debug", None) and infra and lead not in ("what", "why", "how", "is"):
        scores["ops"] += 1
    question = lead in _QUESTION_LEAD or lower.rstrip().endswith("?")
    if question and lead_cls is None:
        if _PROJECT_WORDS & tset and lead in ("how", "where", "why", "which"):
            scores["research"] += 2
        elif "should" in tset and ("we" in tset or "our" in tset):
            scores["plan"] += 2
        else:
            scores["ask"] += 1 if any(v for c, v in scores.items() if c != "ask") else 2
    if len(toks) <= 6 and not any(v for c, v in scores.items() if c != "ask"):
        scores["ask"] += 2
    if mat and not any(v for c, v in scores.items() if c != "ask"):
        scores["code"] += 1               # a bare code block with no instruction cue: assume code work
    demand = reasoning_demand(instr)
    if demand >= 0.75:
        scores["plan"] += 2
    elif demand >= 0.5:
        scores["plan"] += 1
    return {"instr": instr, "lower": lower, "toks": toks, "mat": mat, "scores": scores, "demand": demand,
            "lead_cls": lead_cls, "edit": edit, "infra": infra, "deictic": has_material,
            "project": bool(_PROJECT_WORDS & tset), "question": question}


def _pick(scores, words):
    """Top class and a 0..1 confidence from keyword scores."""
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], _TIE_ORDER.index(kv[0])))
    top_cls, top = ranked[0]
    if top == 0:
        return ("ask" if words < 30 else "write"), 0.4
    margin = top - ranked[1][1]
    conf = 0.55 + 0.08 * min(top, 4) - (0.12 if margin <= 0 else 0.0) - (0.05 if margin == 1 else 0.0)
    return top_cls, max(0.35, min(0.95, conf))


def _needs(text, lower, job, scores):
    needs = []
    if job in ("ops", "code", "debug", "review", "research") or scores["ops"] or _FILE_RE.search(lower) or _REPO_RE.search(lower):
        needs.append("tools")
    if _VISION_RE.search(lower):
        needs.append("vision")
    if len(text) > 60000:
        needs.append("long_context")
    return tuple(needs)


def classify_heuristic(text):
    """Deterministic keyword/structure classifier. No I/O."""
    text = text or ""
    sg = _signals(text)
    cls, conf = _pick(sg["scores"], len(sg["toks"]))
    difficulty, pts = _difficulty(sg["lower"], sg["demand"], cls)
    return Classification(
        cls, difficulty, conf, _needs(text, sg["lower"], cls, sg["scores"]), "rules",
        {"scores": {k: v for k, v in sg["scores"].items() if v}, "reasoning_demand": round(sg["demand"], 3),
         "difficulty_points": round(pts, 2), "words": len(sg["toks"]), "material_chars": sg["mat"]},
    )


# ── learned model: Naive Bayes over instruction words, shipped as maestro_model.json (the code is in nb.py) ──
# The knobs stay here: nb.py reads them from this module at call time, so patching router.MODEL_T still takes effect.
MODEL_FILE = "maestro_model.json"
MODEL_T = 4.0        # softmax temperature applied to the Naive Bayes job log-scores
MODEL_TD = 2.0       # the same for the difficulty log-likelihoods
MODEL_CLASS_W = 1.0  # weight of the heuristic keyword scores added to the class log-scores
MODEL_DIFF_W = 2.0   # bonus (nats) for the heuristic difficulty
MODEL_TOOLS_B = 1.0  # bonus (nats) for the jobs that need repo tools: when unsure, never pick a chat-only job over one
UNDER_COST = 1.5     # routing one tier too weak costs this many times routing one too strong
OBJECTIVES = ("balanced", "cost", "quality")
from .nb import _MODEL, _MODEL_WHY, _RISK_TIER, _nb_logits, _softmax, _user_base, classify_model, load_model, model_problem  # noqa: E402,F401


def classify_rules(text):
    """Offline classifier: the learned model when maestro_model.json fits the current jobs, else the heuristics."""
    model = load_model()
    if model:
        return classify_model(text, model)
    c = classify_heuristic(text)
    if _MODEL_WHY:
        c.signals["model_ignored"] = _MODEL_WHY[0]
    return c


# The Jev classifier lives in jev.py, imported on first use of one of these names (PEP 562), so `route` loads it only on
# the Jev path.
_JEV_NAMES = frozenset(("JEV_URL", "JEV_MODEL", "JEV_MAX_CHARS", "JEV_CACHE_SIZE", "_JEV_JOB_CRITERIA", "_JEV_DIMENSIONS",
                        "_JEV_OBJECTIVES", "JevError", "complexity", "load_cache", "save_cache", "JevClassifier", "choose_jev"))


def __getattr__(name):
    if name in _JEV_NAMES:
        from . import jev
        return getattr(jev, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


# ════════════════════════════════════════════════════════════════════════════
# 4. policy ladder
# ════════════════════════════════════════════════════════════════════════════


def ladder_tier(cls, policy):
    """The ladder cell for this job and difficulty: local, fast, standard, strong or frontier."""
    row = (_pol(policy, "ladder") or {}).get(cls.job) or DEFAULT_LADDER[cls.job]
    return row[max(0, min(3, int(cls.difficulty)))]


def target_tier(cls, policy):
    """Ladder lookup, one cell up if confidence < min_confidence, capped at max_tier.
    The learned model already decodes its tier by expected risk, so it is not escalated twice."""
    i = RANK[ladder_tier(cls, policy)]
    if cls.confidence < _pol(policy, "min_confidence") and not cls.signals.get("model"):
        i += 1
    return CELLS[min(i, RANK[_pol(policy, "max_tier")], len(CELLS) - 1)]


def tier_walk(tier, max_tier="frontier"):
    """Order to try tiers when a seat is missing: target, then up (to the cap), then down."""
    tier = tier_of(tier)
    i, cap = TIERS.index(tier), TIERS.index(max_tier)
    return [tier] + list(TIERS[i + 1:cap + 1]) + list(reversed(TIERS[:i]))


# ════════════════════════════════════════════════════════════════════════════
# 5. inventory / detect
# ════════════════════════════════════════════════════════════════════════════

# seat -> (executable, credential files relative to the user's home)
HARNESS_CLIS = {
    "claude-code": ("claude", (".claude/.credentials.json", ".claude.json")),
    "codex": ("codex", (".codex/auth.json",)),
    "gemini-cli": ("gemini", (".gemini/oauth_creds.json",)),
    "opencode": ("opencode", (".local/share/opencode/auth.json",)),
    "cursor-agent": ("cursor-agent", (".cursor/cli-config.json",)),
    "aider": ("aider", ()),
}
API_KEY_VARS = (
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "XAI_API_KEY", "GROQ_API_KEY",
    "MISTRAL_API_KEY", "MOONSHOT_API_KEY", "MINIMAX_API_KEY", "CEREBRAS_API_KEY",
    "TOGETHER_API_KEY", "FIREWORKS_API_KEY", "TYPESAFE_API_KEY",
)
INVENTORY_MAX_AGE_S = 24 * 3600
INVENTORY_VERSION = 3   # 3: local model and endpoint names are checked against SAFE_NAME
CATALOG_URL = "https://models.dev/api.json"


def _run(argv, timeout=1.0):
    import subprocess
    try:
        return subprocess.check_output(argv, timeout=timeout, stderr=subprocess.DEVNULL).decode("utf-8", "replace")
    except Exception:
        return None


def detect_hardware(which=None, path=None):
    """RAM / VRAM / Apple-silicon unified memory in GB (salvaged from v1 hardware.py)."""
    import platform
    import shutil
    which = which or shutil.which
    hw = {"ram_gb": None, "vram_gb": None, "apple_silicon": False, "memory_gb": None}
    system = platform.system()
    if system == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out and out.strip().isdigit():
            hw["ram_gb"] = round(int(out.strip()) / 1024 ** 3, 1)
        hw["apple_silicon"] = platform.machine() == "arm64"
    else:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        hw["ram_gb"] = round(int(line.split()[1]) / 1024 ** 2, 1)
                        break
        except (OSError, ValueError):
            pass
    if which("nvidia-smi", path=path):
        out = _run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        try:
            mib = sum(float(x) for x in (out or "").split() if x.strip())
            hw["vram_gb"] = round(mib / 1024, 1) or None
        except ValueError:
            pass
    # unified memory on Apple silicon; otherwise the GPU if there is one, else RAM
    hw["memory_gb"] = hw["vram_gb"] or hw["ram_gb"]
    return hw


def _params_b(name, size=None):
    """Billions of parameters: Ollama's details.parameter_size, else parsed from the model name
    (70b, 120b, 8x7b; MoE `30b-a3b` counts the total, 30). None when unknown."""
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*([bm])", (size or "").lower())
    if m:
        v = float(m.group(1))
        return round(v / 1000 if m.group(2) == "m" else v, 2)
    n = (name or "").lower()
    m = re.search(r"(?<![a-z\d.])(\d+)x(\d+(?:\.\d+)?)b(?![a-z])", n)
    if m:
        return round(int(m.group(1)) * float(m.group(2)), 2)
    m = re.search(r"(?<![a-z\d.])(\d+(?:\.\d+)?)b(?![a-z\d])", n)
    return round(float(m.group(1)), 2) if m else None


def _default_serves(b):
    """Strongest category a local model of b billion parameters can stand in for (never strong or frontier)."""
    return "local" if not b or b < 10 else "fast" if b < 40 else "standard"


def _specialist_jobs(name):
    n = (name or "").lower()
    return (["code", "debug", "review"] if re.search(r"code|devstral", n) else []) + (["data"] if re.search(r"math|sql", n) else [])


def _lookup(table, rec):
    for k in (rec["id"], "%s/%s" % (rec["endpoint_name"], rec["name"]), rec["name"]):
        if k in table:
            return table[k]
    return None


def local_pins(cfg, rec):
    """Cells this local model is pinned to in `models.local` ({"standard": "local:spark/gpt-oss:120b"}; an id,
    `endpoint/name` or bare name). A pin sets `serves` and makes the model first choice for that cell."""
    pins = user_models(cfg).get("local")
    names = (rec["id"], "%s/%s" % (rec["endpoint_name"], rec["name"]), rec["name"])
    return [c for c in CELLS if isinstance(pins, dict) and pins.get(c) in names]


@functools.lru_cache(maxsize=2)
def _models_json(path, mtime):   # models.json by id, re-read only when the file changes
    data = _read_json(path)
    data = data if isinstance(data, dict) else {}
    _MODELS_SHA[path] = data.get("catalog_sha")
    return {m["id"]: m for m in data.get("models") or [] if isinstance(m, dict) and m.get("id")}


_MODELS_SHA = {}   # models.json path -> its catalog_sha (the bundled catalog discover scored against)


def rate_local(rec, cfg):
    """(serves, specialist_for) of a local model: a `models.local` pin, else `local_models` overrides, else its scored entry in
    ~/.maestro/models.json (after `discover`; a plain file read), else defaults from its size and name."""
    ov = _lookup((cfg or {}).get("local_models") or {}, rec)
    ov = ov if isinstance(ov, dict) else {}
    sp = ov.get("specialist_for")
    if sp is None:
        sp = _lookup((cfg or {}).get("specialists") or {}, rec)
    if sp is None:
        sp = _specialist_jobs(rec["name"])
    sp = [j for j in sp if j in JOBS] if isinstance(sp, list) else []
    serves, path = _default_serves(rec.get("params_b")), os.path.join(maestro_home(), "models.json")
    try:
        sc = _models_json(path, os.stat(path).st_mtime_ns).get(rec["id"]) or {}
    except OSError:
        sc = {}
    serves = sc["serves"] if sc.get("serves") in CELLS else serves
    if sc.get("specialist") and not sp and isinstance(sc.get("G"), (int, float)):   # the jobs it is lifted for
        sp = [j for j, v in (sc.get("jobs") or {}).items() if j in JOBS and isinstance(v, (int, float)) and v >= sc["G"] + 0.12]
    if sp and serves == "local":
        serves = "fast"  # a specialist is at least fast-level for its jobs
    pins = local_pins(cfg or {}, rec)
    return (max(pins, key=RANK.get) if pins else ov["serves"] if ov.get("serves") in CELLS else serves), sp


# built-in local servers: (port, kind). The first is Ollama; the rest speak the OpenAI API.
BUILTIN_PORTS = ((11434, "ollama"), (1234, "openai"), (8000, "openai"), (30000, "openai"), (8080, "openai"))
LOOPBACK = ("localhost", "127.0.0.1", "::1", "0.0.0.0")
PROBE_BUILTIN_PORTS = True  # False, or MAESTRO_NO_BUILTIN_PORTS, skips them unless a fetch is injected (tests/helpers.py)


def _norm_url(u, default_port=None):
    u = u.strip().rstrip("/")
    u = u if "://" in u else "http://" + u
    u = u.replace("://0.0.0.0", "://127.0.0.1").replace("://localhost", "://127.0.0.1")
    if default_port and not re.search(r":\d+$", u):
        u += ":%d" % default_port
    return u


def _host_of(url):
    return re.sub(r"^\w+://", "", url).split("/")[0].split(":")[0]


def harness_endpoints(env=None):
    """Endpoints other harnesses on this machine already use, so a machine where a harness
    already works needs no separate Maestro setup: Pi (~/.pi/agent/models.json providers),
    Hermes (~/.hermes/config.yaml base_url lines), Codex (~/.codex/config.toml model_providers
    base_url), Claude Code (~/.claude/settings.json env.ANTHROPIC_BASE_URL). URLs only —
    keys are never read."""
    env = os.environ if env is None else env
    home = env.get("HOME")
    if not home:
        return []   # an explicit env without HOME names no machine's harnesses
    out = []

    def strip_v1(url):
        return re.sub(r"/v1$", "", url.strip().rstrip("/"))

    try:
        pi = json.load(open(os.path.join(home, ".pi", "agent", "models.json")))
        for name, prov in (pi.get("providers") or {}).items():
            if isinstance(prov, dict) and isinstance(prov.get("baseUrl"), str) and prov["baseUrl"].strip():
                out.append((strip_v1(prov["baseUrl"]), "pi-%s" % name, "openai"))
    except (OSError, ValueError):
        pass
    try:
        with open(os.path.join(home, ".hermes", "config.yaml")) as f:
            for url in re.findall(r"^\s*base_url:\s*(\S+)\s*$", f.read(), re.M):
                out.append((strip_v1(url), "hermes", "openai"))
    except OSError:
        pass
    try:
        with open(os.path.join(home, ".codex", "config.toml")) as f:
            for url in re.findall(r"^\s*base_url\s*=\s*\"([^\"]+)\"", f.read(), re.M):
                out.append((strip_v1(url), "codex", "openai"))
    except OSError:
        pass
    try:
        claude = json.load(open(os.path.join(home, ".claude", "settings.json")))
        url = (claude.get("env") or {}).get("ANTHROPIC_BASE_URL")
        if isinstance(url, str) and url.strip():
            out.append((url.strip().rstrip("/"), "claude-settings", None))
    except (OSError, ValueError):
        pass
    return out


def local_endpoints(cfg, env=None):
    """Every endpoint to probe, first-wins by URL: config `local_endpoints`, $OLLAMA_HOST,
    harness-derived endpoints (Pi/Hermes/Codex/Claude configs), then the built-in ports."""
    env = os.environ if env is None else env
    eps, seen = [], set()

    def add(url, name=None, kind=None, key_env=None, builtin=False):
        url = _norm_url(url)
        if url not in seen:
            seen.add(url)
            host = _host_of(url)
            name = name if name and SAFE_NAME.fullmatch(name) else None   # it becomes part of model ids in delegate.argv
            eps.append({"name": name or ("this-machine" if host in LOOPBACK else host), "url": url,
                        "kind": kind, "api_key_env": key_env, "builtin": builtin})

    for e in (cfg or {}).get("local_endpoints") or []:
        e = {"url": e} if isinstance(e, str) else e
        if isinstance(e, dict) and isinstance(e.get("url"), str) and e["url"].strip():
            add(e["url"], e.get("name") if isinstance(e.get("name"), str) else None,
                e.get("kind") if e.get("kind") in ("ollama", "openai") else None,
                e.get("api_key_env") if isinstance(e.get("api_key_env"), str) else None)
    if env.get("OLLAMA_HOST"):
        add(_norm_url(env["OLLAMA_HOST"], 11434), kind="ollama")
    for url, name, kind in harness_endpoints(env):
        add(url, name=name, kind=kind)
    for port, kind in BUILTIN_PORTS:
        add("http://127.0.0.1:%d" % port, kind=kind, builtin=True)
    return eps


def _is_this_machine(url):
    import socket
    host = _host_of(url)
    me = socket.gethostname().lower()
    return host in LOOPBACK or host.lower() == me or host.lower().startswith(me + ".")


def _loaded_names(url, kind, fetch, headers, timeout):
    """Names a server has loaded (GET only): Ollama /api/ps (an unanswered call counts as none loaded), LM Studio /api/v0/models
    (a "state" per model, "loaded" or "not-loaded"). None = the server does not say (vLLM, SGLang, llama.cpp): it serves what /v1/models lists."""
    try:
        status, raw = fetch(url + ("/api/ps" if kind == "ollama" else "/api/v0/models"), headers, None, timeout)
        rows = json.loads(raw)["models" if kind == "ollama" else "data"] if status == 200 else None
        rows = [x for x in rows if isinstance(x, dict)]
    except Exception:
        rows = None
    if kind == "ollama":
        return {x.get("name") or x.get("model") for x in rows or []}
    return {x.get("id") for x in rows if x.get("state") == "loaded"} if any("state" in x for x in rows or []) else None


# Names a model server reports end up in `delegate.argv`, which the calling agent runs. Only names made of these characters
# are used (no spaces, quotes or shell syntax), so a hostile server on the network cannot smuggle a command into it.
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+=-]{0,199}\Z")   # use fullmatch: `$` would allow a trailing newline


def probe_endpoint(ep, fetch=None, timeout=0.3, env=None):
    """{"kind", "latency_ms", "models": [{"name", "size", "loaded"}]} for a reachable model server, else None.
    Ollama: GET /api/tags (installed) and /api/ps (loaded). OpenAI-compatible: GET /v1/models. Unknown kind: try both."""
    env = os.environ if env is None else env
    if fetch is None and ep.get("builtin") and (not PROBE_BUILTIN_PORTS or env.get("MAESTRO_NO_BUILTIN_PORTS")):
        return None
    fetch = fetch or _local_fetch
    key = env.get(ep["api_key_env"]) if ep.get("api_key_env") else None  # sent as a Bearer token, never stored
    headers = {"Authorization": "Bearer " + key} if key else {}
    for kind in ([ep["kind"]] if ep.get("kind") else ["ollama", "openai"]):
        t0 = time.perf_counter()
        try:
            status, raw = fetch(ep["url"] + ("/api/tags" if kind == "ollama" else "/v1/models"), headers, None, timeout)
            if status != 200:
                continue
            data = json.loads(raw)
            if kind == "ollama":
                found = [(x.get("name") or x.get("model"), (x.get("details") or {}).get("parameter_size"), {}) for x in data["models"]]
            else:  # vLLM / SGLang report the context window (max_model_len) and the served path (root: often a Hugging Face repo id)
                found = [(x.get("id"), None, {k: x[k] for k in ("max_model_len", "root") if x.get(k) is not None}) for x in data["data"]]
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
        except Exception:  # refused, unreachable, timed out: the other kind would fail the same way
            return None
        loaded = _loaded_names(ep["url"], kind, fetch, headers, timeout)
        named = [(n, z, extra) for n, z, extra in found if isinstance(n, str) and n and "embed" not in n]
        safe = [(n, z, extra) for n, z, extra in named if SAFE_NAME.fullmatch(n)]
        return {"kind": kind, "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "ignored": len(named) - len(safe),
                "models": [dict(extra, name=n, size=z, loaded=loaded is None or n in loaded) for n, z, extra in safe]}
    return None


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None


def scan_network(env=None, probe=None, local_ip=None, budget=5.0, timeout=0.25, workers=64):
    """Opt-in (`detect --scan`): probe this machine's /24 for Ollama (11434) and OpenAI-compatible (8000)
    servers. Bounded by `budget` seconds. Returns endpoint dicts for what answered."""
    import concurrent.futures as cf
    env = os.environ if env is None else env
    if env.get("MAESTRO_NO_SCAN"):
        raise RuntimeError("network scan disabled by MAESTRO_NO_SCAN")
    if local_ip is None:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("10.255.255.255", 1))  # UDP connect sends nothing; it only picks the outgoing interface
            local_ip = s.getsockname()[0]
        except OSError:
            local_ip = ""
        finally:
            s.close()
    if not local_ip or local_ip.startswith("127."):
        return []
    base = local_ip.rsplit(".", 1)[0]
    targets = [{"name": "%s.%d" % (base, i), "url": "http://%s.%d:%d" % (base, i, port), "kind": kind,
                "api_key_env": None, "builtin": False}
               for i in range(1, 255) for port, kind in ((11434, "ollama"), (8000, "openai"))
               if "%s.%d" % (base, i) != local_ip]
    probe = probe or (lambda ep: probe_endpoint(ep, None, timeout, env))
    ex = cf.ThreadPoolExecutor(max_workers=workers)
    futs = {ex.submit(_safe, probe, ep): ep for ep in targets}
    done, _ = cf.wait(futs, timeout=budget)
    ex.shutdown(wait=False, cancel_futures=True)
    found = []
    for f in done:
        res = f.result()
        if res:
            found.append(dict(futs[f], kind=res["kind"], models=len(res["models"])))
    return sorted(found, key=lambda e: e["url"])


def save_endpoints(found, home=None):
    """Add scanned endpoints to config.json's local_endpoints (existing keys and entries are kept)."""
    path = config_path(home)
    raw = _read_json(path)
    raw = raw if isinstance(raw, dict) else {}
    have = [e for e in raw.get("local_endpoints", []) if isinstance(e, (str, dict))]
    urls = {_norm_url(e if isinstance(e, str) else str(e.get("url", ""))) for e in have}
    have += [{"name": e["name"], "url": e["url"], "kind": e["kind"]} for e in found if e["url"] not in urls]
    raw["local_endpoints"] = have
    return _write_json(path, raw)


def _price_tier(out_per_m):
    return "frontier" if out_per_m >= 30 else "strong" if out_per_m >= 15 else "standard" if out_per_m >= 6 else "fast"


def fetch_catalog(home, fetch=None, timeout=3.0):
    """models.dev catalog as {provider: {env, models: {id: [output $/1M, family, release_date, status, tool_call]}}},
    cached; $MAESTRO_CATALOG_URL points at a mirror."""
    path = os.path.join(home, "catalog.json")
    try:
        status, raw = (fetch or _http_fetch)(os.environ.get("MAESTRO_CATALOG_URL", CATALOG_URL),
                                             {"Accept": "application/json"}, None, timeout)
        if status != 200:
            raise ValueError(status)
        slim = {}
        for pid, p in json.loads(raw).items():
            models = {}
            for mid, m in (p.get("models") or {}).items():
                m = m or {}
                out = (m.get("cost") or {}).get("output")
                models[mid] = [out if isinstance(out, (int, float)) else None, m.get("family"), m.get("release_date"),
                               m.get("status"), m.get("tool_call")]
            if models:
                slim[pid] = {"env": p.get("env") or [], "models": models}
        _write_json(path, slim)
        return slim
    except Exception:
        return _read_json(path) or {}


def detect(home=None, env=None, which=None, probe=None, hw_probe=None, catalog=False, fetch=None,
           write=True, user_home=None, cfg=None, extra_endpoints=()):
    """Build the inventory: CLIs, keys, local and network models, hardware. Endpoints are probed in parallel."""
    import concurrent.futures as cf
    env = os.environ if env is None else env
    home = home or maestro_home(env)
    user_home = user_home or _user_home(env)
    cfg = cfg if cfg is not None else load_config(home)
    path = env.get("PATH")
    if which is None:
        import shutil
        which = shutil.which
    harnesses = {}
    for seat, (exe, creds) in HARNESS_CLIS.items():
        found = which(exe, path=path)
        if found:
            has_creds = any(os.path.exists(os.path.join(user_home, c)) for c in creds)
            harnesses[seat] = {"cli": exe, "path": found, "creds_file": has_creds,
                               "available": "yes" if has_creds else "likely"}
    hw = hw_probe() if hw_probe else detect_hardware(which, path)
    eps = local_endpoints(cfg, env)
    eps += [e for e in extra_endpoints if e["url"] not in {x["url"] for x in eps}]
    probe = probe or (lambda ep: probe_endpoint(ep, None, 0.3 if ep["builtin"] else 1.0, env))
    with cf.ThreadPoolExecutor(max_workers=min(len(eps), 16)) as ex:
        results = list(ex.map(lambda ep: _safe(probe, ep), eps))
    models, endpoints, warnings = [], [], []
    for ep, res in zip(eps, results):
        if not res:
            continue
        endpoints.append(dict({"name": ep["name"], "url": ep["url"], "kind": res["kind"], "models": len(res["models"])},
                              **({"ignored": res["ignored"]} if res.get("ignored") else {})))
        here = _is_this_machine(ep["url"])
        for mm in res["models"]:
            rec = {"id": "local:%s/%s" % (ep["name"], mm["name"]), "name": mm["name"], "endpoint_name": ep["name"],
                   "url": ep["url"], "kind": res["kind"], "params_b": _params_b(mm["name"], mm.get("size")),
                   "on_this_machine": here, "latency_ms": res["latency_ms"]}
            rec.update({k: mm[k] for k in ("max_model_len", "root") if k in mm}, loaded=mm.get("loaded", True))
            if any(x["id"] == rec["id"] for x in models):
                continue
            rec["serves"], rec["specialist_for"] = rate_local(rec, cfg)
            models.append(rec)
            need = (rec["params_b"] or 0) * 0.6 + 1  # rough 4-bit footprint in GB
            if here and hw.get("memory_gb") and need > hw["memory_gb"]:
                warnings.append("%s (~%.0f GB) may not fit in %.0f GB of memory" % (rec["name"], need, hw["memory_gb"]))
    inv = {
        "version": INVENTORY_VERSION,
        "generated_at": time.time(),
        "harnesses": harnesses,
        "api_keys": {k: k in env for k in API_KEY_VARS},  # presence only, values are never read
        "local_endpoints": endpoints,
        "local_models": models,
        "endpoints_sig": [e["url"] for e in eps],
        "hardware": hw,
        "warnings": warnings,
    }
    if catalog:
        cat = fetch_catalog(home, fetch)
        inv["catalog"] = {}
        for pid, p in cat.items():
            if any(e in env for e in p.get("env", [])):
                tiers = {}
                for mid, e in p["models"].items():
                    price = e[0] if isinstance(e, list) else e
                    if isinstance(price, (int, float)):
                        tiers.setdefault(_price_tier(price), []).append(mid)
                inv["catalog"][pid] = {t: sorted(v)[:25] for t, v in tiers.items()}
    if write:
        _write_json(os.path.join(home, "inventory.json"), inv)
    return inv


def ensure_inventory(home=None, env=None, max_age=INVENTORY_MAX_AGE_S, cfg=None):
    """Cached inventory; re-detected only when missing, older than 24 h, from an older Maestro,
    or when the configured local endpoints changed."""
    home = home or maestro_home(env)
    cfg = cfg if cfg is not None else load_config(home)
    inv = _read_json(os.path.join(home, "inventory.json"))
    if (isinstance(inv, dict) and inv.get("version") == INVENTORY_VERSION and time.time() - inv.get("generated_at", 0) < max_age
            and inv.get("endpoints_sig") == [e["url"] for e in local_endpoints(cfg, env)]):
        return inv
    return detect(home=home, env=env, cfg=cfg)


# ════════════════════════════════════════════════════════════════════════════
# 6. seats and delegation
# ════════════════════════════════════════════════════════════════════════════

KIND_RANK = {"subagent": 0, "model": 0, "local": 1, "cli": 2}

# Suggested model per harness and category: a snapshot of the models.dev catalog (`maestro-router models --check` compares
# it with today's). Claude Code's aliases track the latest release by themselves.
SUGGESTED_AS_OF = "2026-09-30"
SUGGESTED_SOURCE = "models.dev"
PROVIDER_KEYS = {"anthropic": ("ANTHROPIC_API_KEY",), "openai": ("OPENAI_API_KEY",), "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY")}


def _row(*models, **kw):
    return dict(zip(TIERS, [kw["prefix"] + "/" + m for m in models] if "prefix" in kw else models))


SUGGESTED_MODELS = {
    "claude-code": _row("haiku", "sonnet", "opus", "fable"),
    "codex": _row("gpt-6-luna", "gpt-6.1-sol", {"model": "gpt-6.1-sol", "effort": "high"}, "gpt-6-astra"),
    "gemini-cli": _row("gemini-flash-lite-latest", "gemini-flash-latest", "gemini-3.1-pro-preview", "gemini-3.1-pro-preview"),
    "opencode": {  # by provider: the first with an API key present is suggested (opencode_provider)
        "anthropic": _row("claude-haiku-4-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1", prefix="anthropic"),
        "openai": _row("gpt-6-luna", "gpt-6.1-sol", "gpt-6.1-sol", "gpt-6-astra", prefix="openai"),
        "google": _row("gemini-flash-lite-latest", "gemini-flash-latest", "gemini-3.1-pro-preview",
                       "gemini-3.1-pro-preview", prefix="google"),
    },
}


def norm_seat(name):
    return SEAT_ALIASES.get(name, name)


def detect_harness(env=None):
    env = os.environ if env is None else env
    if env.get("CLAUDECODE") == "1":
        return "claude-code"
    if any(k == "OPENCODE" or k.startswith("OPENCODE_") for k in env):
        return "opencode"
    if "GEMINI_CLI" in env:
        return "gemini-cli"
    if any(k.startswith("CODEX_") for k in env):
        return "codex"
    return "unknown"


def cli_argv(seat, model=None, effort=None):
    m = ["-m", model] if model else []
    if seat == "claude-code":
        return ["claude", "-p"] + (["--model", model] if model else []) + ["{task}"]
    if seat == "codex":  # the only CLI here with a reasoning-effort flag
        return ["codex", "exec"] + m + (["-c", "model_reasoning_effort=" + effort] if effort else []) + ["{task}"]
    if seat == "gemini-cli":
        return ["gemini"] + m + ["-p", "{task}"]
    if seat == "opencode":
        return ["opencode", "run"] + m + ["{task}"]
    return None


def opencode_provider(inventory):
    """The provider to suggest for OpenCode: the first of anthropic, openai, google with an API key present, else anthropic."""
    keys = (inventory or {}).get("api_keys") or {}
    return next((p for p, envs in PROVIDER_KEYS.items() if any(keys.get(e) for e in envs)), "anthropic")


def seat_models(seat, inventory, policy):
    """{tier: (model, effort, source)}: the user's `models` cell (config), else the suggestion (suggested), else, for
    strong only, the CLI's own default model (builtin: None, pass no -m)."""
    mine = user_models(policy).get(seat)
    mine = {"strong": mine} if isinstance(mine, str) else mine if isinstance(mine, dict) else {}
    sug = SUGGESTED_MODELS.get(seat) or {}
    sug = sug[opencode_provider(inventory)] if seat == "opencode" else sug
    out = {}
    for t in TIERS:
        for cell, src in ((_cell(mine.get(t)), "config"), (_cell(sug.get(t)), "suggested")):
            if cell:
                out[t] = cell + (src,)
                break
    out.setdefault("strong", (None, None, "builtin"))
    return out


SEAT_LABELS = {"claude-code": "Claude Code", "codex": "Codex", "gemini-cli": "Gemini CLI", "opencode": "OpenCode"}


def _local_match(model, locs):
    """The detected local model a `models` entry points at (`ollama/gpt-oss:120b`, an id, ...), if any."""
    for r in locs:
        if model and (model in (r["id"], r["name"], "%s/%s" % (r["endpoint_name"], r["name"])) or model.endswith("/" + r["name"])):
            return r
    return None


def _executable(c, job, needs):
    """Can this candidate do the job? A plain local model is chat only: no repo tools, so only the
    jobs that need none. A specialist is used only for its own jobs."""
    if c["specialist_for"] and job not in c["specialist_for"]:
        return False
    if c["tools"]:
        return True
    return job not in TOOL_JOBS and not {"tools", "vision", "long_context"} & set(needs)


def _candidates(cls, inventory, harness, policy):
    """Every seat+model that could run this task, after the hard filters."""
    out = []
    spec = _pol(policy, "specialists") or {}
    prefer_local = _pol(policy, "prefer_local")
    locs = []
    for r in inventory.get("local_models") or []:
        if not all(SAFE_NAME.fullmatch(str(r.get(k) or "")) for k in ("name", "endpoint_name")):
            continue   # a name that could carry shell syntax never reaches delegate.argv, whatever inventory says
        if r.get("loaded", True) is False and not _pol(policy, "local_allow_load"):
            continue   # installed but not loaded: Maestro never loads models
        r = dict(r, id="local:%s/%s" % (r["endpoint_name"], r["name"]))   # the id in delegate.argv: rebuilt from checked parts
        r["serves"], r["specialist_for"] = rate_local(r, policy)
        r["pins"] = local_pins(policy, r)
        locs.append(r)

    def add(seat, kind, model, serves, delegate, name=None, **extra):
        sp = spec.get(model) if isinstance(spec.get(model), list) else []
        name = name or "%s:%s" % (seat, model or "default")
        if any(x["name"] == name for x in out):  # one model can serve two categories (Codex standard and strong)
            name += "@" + serves
        out.append(dict({"local": False, "tools": True, "specialist_for": [j for j in sp if j in JOBS], "params": 0,
                         "on_machine": False, "latency": 0.0, "pins": (), "source": None, "effort": None},
                        name=name, seat=seat, kind=kind, model=model, serves=serves, delegate=delegate, **extra))

    # 1. native subagent in the current harness; a local model configured there can use the harness's tools
    native = set()
    if harness in AGENT_DIRS:
        for tier, (model, effort, src) in seat_models(harness, inventory, policy).items():
            d = {"kind": "subagent", "harness": harness, "model": model}
            if harness != "claude-code":
                d["agent"] = "maestro-" + tier
            r = _local_match(model, locs) if harness != "claude-code" else None
            if r and not prefer_local:
                continue
            if r:
                native.add(r["id"])
                add(harness, "subagent", model, tier, d, local=True, specialist_for=r["specialist_for"], source=src,
                    params=r["params_b"] or 0, on_machine=r["on_this_machine"], latency=r.get("latency_ms") or 0.0)
            else:
                add(harness, "subagent", model, tier, d, source=src)
    # 2. local and network models, run with `maestro-router run` (chat only: no repo tools)
    if prefer_local:
        for r in locs:
            if r["id"] not in native:
                argv = [sys.executable, os.path.dirname(os.path.abspath(__file__)), "run", r["id"], "{task}"]
                add("local", "local", r["id"], r["serves"], {"kind": "local", "model": r["id"], "argv": argv},
                    name=r["id"], local=True, tools=False, specialist_for=r["specialist_for"], pins=r["pins"],
                    source="config" if r["pins"] else "auto",
                    params=r["params_b"] or 0, on_machine=r["on_this_machine"], latency=r.get("latency_ms") or 0.0)
    # 1b. a named harness that is not one of the four CLIs. The caller applies the model itself.
    # Maestro does not spawn another harness. Other CLIs are not candidates for this call.
    caller_owned = harness not in SUGGESTED_MODELS and harness not in ("unknown", "", None)
    if caller_owned:
        mine = user_models(policy).get(harness) or {}
        if isinstance(mine, dict):
            for tier in TIERS:
                cell = _cell(mine.get(tier))
                if not cell:
                    continue
                model, effort = cell
                add(harness, "model", model, tier,
                    {"kind": "model", "harness": harness, "model": model}, source="config", effort=effort)
    # 3. installed harness CLIs, run headless, the current one first (Codex has no per-subagent model: it calls itself)
    for seat in [] if caller_owned else sorted(SUGGESTED_MODELS, key=lambda s: s != harness):
        if seat not in (inventory.get("harnesses") or {}) or (seat == harness and seat != "codex"):
            continue
        for tier, (model, effort, src) in seat_models(seat, inventory, policy).items():
            add(seat, "cli", model, tier, {"kind": "cli", "argv": cli_argv(seat, model, effort)}, source=src, effort=effort)
    cap = RANK[_pol(policy, "max_tier")]
    return [c for c in out if (c["local"] or RANK[c["serves"]] <= cap) and _executable(c, cls.job, cls.needs)]


def _category(c, job):
    return "specialist" if job in c["specialist_for"] else "local" if c["local"] else c["serves"]


def describe(c, inventory, policy):
    """Short description of a candidate for the Jev model question."""
    note = (_pol(policy, "model_notes") or {}).get(c["name"])
    if note:
        return str(note)
    if c["local"]:
        where = next((r["endpoint_name"] for r in inventory.get("local_models") or [] if r["id"] == c["name"]), c["seat"])
        size = "%sB" % ("%g" % c["params"]) if c["params"] else "size unknown"
        level = "simple work" if c["serves"] == "local" else "%s-level work" % c["serves"]
        text = "local model on %s (%s), free and private, serves up to %s, " % (where, size, level)
        text += "runs as a %s subagent with repo tools" % c["seat"] if c["tools"] else "no repo tools"
        return text + (", tuned for " + ", ".join(c["specialist_for"]) if c["specialist_for"] else "")
    creds = ((inventory.get("harnesses") or {}).get(c["seat"]) or {}).get("creds_file", True)
    cost = "subscription (flat cost)" if creds else "billing depends on your login"
    who = "%s (%s%s)" % (SEAT_LABELS.get(c["seat"], c["seat"]), c["model"] or "default model",
                         ", effort " + c["effort"] if c["effort"] else "")
    tuned = ", tuned for " + ", ".join(c["specialist_for"]) if c["specialist_for"] else ""
    if c["kind"] == "subagent":
        return "%s tier, %s subagent, %s, full tools/repo access%s" % (c["serves"], who, cost, tuned)
    return "%s tier, %s headless CLI, %s, tool access but no shared context%s" % (c["serves"], who, cost, tuned)


def build_candidates(cls, inventory, harness, policy):
    """Eligible executable candidates, each with a stable `name` (seat:model) and a description."""
    cands = _candidates(cls, inventory, harness, policy)
    for c in cands:
        c["description"] = describe(c, inventory, policy)
    return cands


def _rank_fn(policy):
    prefer = [norm_seat(p) for p in _pol(policy, "prefer")]

    def rank(c):
        return (prefer.index(c["seat"]) if c["seat"] in prefer else 99, KIND_RANK[c["kind"]], -c.get("params", 0))
    return rank


def _local_key(c, cell=None):
    """A model pinned to the cell first, then the smallest that serves it, local over cloud, this machine over the
    network, lower latency."""
    return (cell not in c["pins"], RANK[c["serves"]], not c["local"], c["params"], not c["on_machine"], c["latency"],
            KIND_RANK[c["kind"]])


def select_seat(cell, cls, inventory, harness, policy, current_tier=None, cands=None):
    """Pick who runs the task. `cell` is the ladder cell (local..frontier). Order: a specialist for the job,
    a local model that serves the cell, inline if already on the tier, then the cloud walk (target, up, down),
    then any local model as a last resort. Returns seat, kind, model, category, delegate, fallback, walk, note."""
    cands = _candidates(cls, inventory, harness, policy) if cands is None else cands
    tier = tier_of(cell)
    walk = tier_walk(tier, _pol(policy, "max_tier"))
    rank = _rank_fn(policy)

    def local_first(pool):
        for label, group in (("specialist for %s" % cls.job, [c for c in pool if cls.job in c["specialist_for"]]),
                             ("local model", [c for c in pool if c["local"] and not c["specialist_for"]])):
            ok = sorted((c for c in group if RANK[c["serves"]] >= RANK[cell]), key=lambda c: _local_key(c, cell))
            if ok:
                return _seat_result(ok[0], cands, walk, [tier], rank, "%s %s serves %s" % (label, ok[0]["model"], cell), cls.job)
        return None

    def cloud_walk(pool, note=""):
        tried = []
        for t in walk:
            tried.append(t)
            at = sorted((c for c in pool if not c["local"] and c["serves"] == t), key=rank)
            if at:
                return _seat_result(at[0], cands, walk, tried, rank, note, cls.job)
        return None

    pin = norm_seat((_pol(policy, "job_seats") or {}).get(cls.job, ""))
    note = ""
    if pin:
        pool = [c for c in cands if c["seat"] == pin]
        res = local_first(pool) or cloud_walk(pool, "pinned by job_seats")
        if res:
            return res
        note = "job_seats pin %r unavailable; " % pin
    res = local_first(cands)
    if res:
        res["note"] = note + res["note"]
        return res
    if current_tier == tier:
        return _inline(harness, tier, note + "current tier is already %s" % tier, [])
    res = cloud_walk(cands)
    if res:
        res["note"] = note + res["note"]
        return res
    locs = [c for c in cands if c["local"]]
    if locs:
        best = max(locs, key=lambda c: RANK[c["serves"]])
        return _seat_result(best, cands, walk, walk, rank, note + "no cloud seat available; using the local model", cls.job)
    return _inline(harness, tier, note + "no seat available at any tier", walk)


def _inline(harness, tier, note, walk):
    return {"seat": harness if harness != "unknown" else "inline", "kind": "inline", "model": None,
            "category": tier, "delegate": {"kind": "inline"}, "fallback": [], "walk": walk, "note": note, "source": None}


def _seat_result(chosen, cands, walk, tried, rank, note, job):
    fallback, seen = [], {chosen["seat"]}
    for c in [c for t in walk for c in sorted((c for c in cands if not c["local"] and c["serves"] == t), key=rank)] + \
            sorted((c for c in cands if c["local"]), key=_local_key):
        if c["seat"] not in seen and len(fallback) < 3:
            seen.add(c["seat"])
            fallback.append({"seat": c["seat"], "model": c["model"], "category": _category(c, job)})
    return {"seat": chosen["seat"], "kind": chosen["kind"], "model": chosen["model"], "source": chosen["source"],
            "category": _category(chosen, job), "delegate": chosen["delegate"], "fallback": fallback,
            "walk": tried, "note": note or "%s %s available" % (chosen["seat"], chosen["kind"])}


# ════════════════════════════════════════════════════════════════════════════
# 7. route
# ════════════════════════════════════════════════════════════════════════════


def route(text, harness=None, current_tier=None, classifier=None, explain=False,
          config=None, inventory=None, env=None, home=None, jev_fetch=None, decider=None, hints=None, log=False):
    """Classify the task, pick a ladder cell, pick a seat. Returns the decision dict.
    With decider "utility" (the config's, or this argument) S1 and decide.py rank every seat by expected utility in dollars
    (_decide). In S1 shadow mode the other decider runs when someone looks: log=True (`maestro-router route` appends decisions.jsonl)
    or explain. `hints` are the calling agent's: job, tools, self_contained, size, stakes, context_tokens."""
    t0 = time.perf_counter()
    env = os.environ if env is None else env
    home = home or maestro_home(env)
    cfg = config if config is not None else load_config(home)
    harness = norm_seat(harness) if harness else detect_harness(env)
    if current_tier is not None and current_tier not in TIERS:
        raise ValueError("current tier must be one of %s" % ", ".join(TIERS))
    inv = inventory if inventory is not None else ensure_inventory(home, env, cfg=cfg)

    which = classifier or env.get("MAESTRO_CLASSIFIER") or cfg.get("classifier", "auto")
    key = env.get("TYPESAFE_API_KEY")
    hints = {k: v for k, v in (hints or {}).items() if v is not None}
    decider = decider or cfg.get("decider") or "ladder"
    cls, note, cands = classify_rules(text), "", []
    if hints.get("job") in JOBS:
        cls.job = hints["job"]                 # the calling agent knows the job: the ladder uses it too
    if which == "jev" and not key:
        note = "jev requested but TYPESAFE_API_KEY is not set; used rules"
    elif which in ("jev", "auto") and key and _private(text, cfg):
        note = "jev skipped: the task is private (private mode or a key-like string)"   # Jev is a cloud service
    elif which in ("jev", "auto") and key:
        cands = build_candidates(cls, inv, harness, cfg)
        if len(cands) < 2:
            note = "jev skipped: only one eligible seat"
        else:
            from . import jev as J
            cache = J.load_cache(home)
            jcls = J.JevClassifier(key, fetch=jev_fetch, cache=cache).classify(text, cands, cfg["objective"])
            if jcls.jev and not jcls.jev.get("cached"):
                J.save_cache(cache, home)
            cls, note = (jcls, note) if jcls.jev else (cls, jcls.note)

    post, want = None, decider == "utility" or (cfg.get("s1_shadow", True) and (log or explain))
    use_s1 = not cls.jev and (which == "s1" or (which == "auto" and not key and _s1_passed()))
    if want or use_s1:
        try:
            from . import s1 as S1
            post = S1.posterior(text, cls, hints)
            cls = S1.legacy(post, cls) if use_s1 else cls   # S1's job and difficulty (unchanged unless S1 can decode them)
            if post.get("s1_ignored"):
                cls.signals["s1_ignored"] = post["s1_ignored"]
        except Exception as e:  # S1 never breaks routing: the rules classification and the ladder stand
            post, want, note = None, False, (note + "; " if note else "") + "S1 failed (%s); used the rules" % e.__class__.__name__
    if cls.jev:
        from . import jev as J
        seat = J.choose_jev(cls, cands, inv, harness, cfg, current_tier)
        cell = seat["target"]
        why = "jev %s/d%d: %s" % (cls.job, cls.difficulty, seat["note"])
    else:
        base, cell = ladder_tier(cls, cfg), target_tier(cls, cfg)
        seat = select_seat(cell, cls, inv, harness, cfg, current_tier)
        why = "%s/d%d \u2192 %s" % (cls.job, cls.difficulty, base)
        if cell != base:
            why += " \u2192 %s (%s)" % (cell, "low confidence" if RANK[cell] > RANK[base] else "max_tier cap")
        elif cls.confidence < cfg["min_confidence"] and not cls.signals.get("model"):
            why += " (low confidence, already at max_tier)"
        why += "; " + seat["note"]
    if seat["kind"] not in ("inline", "local") and seat["category"] not in (tier_of(cell), "specialist", "local"):
        why += " (no %s seat; used %s)" % (tier_of(cell), seat["category"])
    if note:
        why += "; " + note
    if seat["kind"] not in ("inline", "local") and (cfg.get("private") or decider == "utility" and _private(text, cfg)):
        seat = _inline(harness, seat["category"] if seat["category"] in TIERS else "standard", "private", seat["walk"])  # also if decide fails
        why += "; private: only local models or you see the task, so do it yourself"
    extra = {}
    if want:
        try:
            seat, why, extra = _decide(text, post, cls, seat, why, inv, harness, cfg, home, current_tier, decider, hints, log)
        except Exception as e:  # the decision layer never breaks routing: the ladder's seat stands
            why += "; the utility decider failed (%s); used the ladder" % e.__class__.__name__

    decision = dict(cls.to_dict(), category=seat["category"], harness=harness, seat=seat["seat"], model=seat["model"],
                    model_source=seat["source"], delegate=seat["delegate"], fallback=seat["fallback"], reason=why)
    if cls.jev:
        decision["jev"] = cls.jev
    decision.update(extra.get("top") or {})
    if explain:
        decision["explain"] = {"signals": cls.signals, "cell": cell, "walk": seat["walk"],
                               "policy": {k: cfg[k] for k in ("min_confidence", "max_tier", "prefer_local", "objective", "jev_min_confidence")}}
        decision["explain"]["candidates"] = [c["name"] for c in cands or _candidates(cls, inv, harness, cfg)]
        decision["explain"].update(extra.get("explain") or {})
    decision["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return decision


def _private(text, cfg):   # private mode, or a key-like string in the task (decide.is_private, spec §4.5)
    from . import decide
    return decide.is_private(text, cfg)


def _s1_passed():
    """Did the shipped s1.bin pass the replace-NB check (spec §8.4)? Then `classifier: auto` decodes job and difficulty with S1."""
    from . import s1
    md = s1.load()[0]
    return bool(md and (md["header"].get("legacy") or {}).get("pass"))


def _decide(text, post, cls, ladder, why, inv, harness, cfg, home, current_tier, decider, hints, log):
    """Expected-utility routing in skill mode (spec §5.7): every candidate seat (plus the calling agent with --current-tier) ranked
    by decide.py. Under "utility" the EU argmax runs the task, except a tool job before decide.json `agentic_ready` (the ladder
    keeps it; spec §12.4); otherwise decide reports the ladder's seat and the EU pick is the shadow. Returns (seat, reason, extra)."""
    from . import decide as D, s1 as S1
    md, cands, path = S1.load()[0], _candidates(cls, inv, harness, cfg), os.path.join(home, "models.json")
    try:
        models = _models_json(path, os.stat(path).st_mtime_ns)
    except OSError:
        models = {}
    seats = [D.seat_of(c, models) for c in cands]
    if current_tier:
        seats.append(D.inline_seat(harness, current_tier, seat_models(harness, inv, cfg).get(current_tier, (None,))[0], models))
    lid = "inline" if ladder["kind"] == "inline" else next((c["name"] for c in cands if (c["seat"], c["model"], c["kind"]) == (
        ladder["seat"], ladder["model"], ladder["kind"]) and _category(c, cls.job) == ladder["category"]), None)
    use = decider == "utility" and not (post["group"] == "agentic" and not D.table().get("agentic_ready"))
    ctx = {"mode": "skill", "cfg": cfg, "home": home, "text": text, "hints": hints, "difficulty": cls.difficulty,
           "est_in": len(text) / (3.0 if "```" in text else 3.6), "s1": md and md["header"], "catalog_sha": _MODELS_SHA.get(path)}
    d = D.decide(seats, post, dict(ctx, chosen=None if use else lid or "?"))
    view = {k: round(v, 3) if isinstance(v, float) else v for k, v in post.items() if k not in ("job_p", "hints")}
    view.update(job_p={j: round(p, 3) for j, p in sorted(post["job_p"].items(), key=lambda x: -x[1]) if p >= 0.01},
                b=[round(x, 3) for x in post["b"]], tok_out=[round(x, 3) for x in post["tok_out"]], turns=[round(x, 3) for x in post["turns"]])
    seat, top = ladder, {"decider": decider, "decision_id": d["decision_id"]}
    if decider == "utility":
        top.update(posterior=view, options=d["options"], escalate_to=d["escalate_to"], flags=d["flags"], p_success=d.get("p_success"))
    by = {s["id"]: c for s, c in zip(seats, cands)}
    if use and d["chosen"] in by:
        c = by[d["chosen"]]
        seat = _seat_result(c, cands, [], [], _rank_fn(cfg), d["reason"], cls.job)
        seat["fallback"], seen = [], {c["seat"]}
        for x in (by[o["seat"]] for o in d["options"] if o["seat"] in by):
            if x["seat"] not in seen and len(seat["fallback"]) < 3:
                seen.add(x["seat"])
                seat["fallback"].append({"seat": x["seat"], "model": x["model"], "category": _category(x, cls.job)})
    elif use:   # the calling agent itself, or no seat qualifies (private mode with no local model)
        seat = _inline(harness, current_tier or (tier_of(ladder["category"]) if ladder["category"] in CELLS else "standard"), d["reason"], [])
    if use:
        why = d["reason"]
        top["confidence"] = d["confidence"]
    elif decider == "utility":
        why += "; tool job: the ladder decides until agentic_ready (utility would pick %s)" % d.get("shadow")
    shadow = d.get("shadow") if not use else lid
    if log and cfg.get("log_decisions", True):
        try:
            ensure_home(home)   # tightens an existing home; decide.log_decision only creates it (and ignores errors)
        except OSError:
            pass
        D.log_decision(home, dict(d, chosen=d["chosen"] if use else lid), post, text, cfg, harness=harness, hints=hints, shadow=shadow,
                       decider="utility" if use else "ladder", kind=seat["kind"], policy=s1_policy(md))
    return seat, why, {"top": top, "explain": {"shadow": shadow, "options": d["options"], "skipped": d["skipped"], "flags": d["flags"],
                                               "posterior": view}}


def s1_policy(md):
    """The policy id logged with each decision ("s1:<crc32 of the s1.bin header>", or "nb"): learn.py resets a seat's personal
    offset when the weights change."""
    if not md:
        return "nb"
    if "policy" not in md:
        md["policy"] = "s1:%08x" % zlib.crc32(json.dumps(md["header"], sort_keys=True).encode("utf-8"))
    return md["policy"]


def format_text(d):
    model = d["model"] or "default"
    line = "%s %s/%s (%s) - %s" % (d["category"], d["seat"], model, d["delegate"]["kind"], d["reason"])
    return line.replace("\u2192", "->")  # safe on any console encoding


# ════════════════════════════════════════════════════════════════════════════
# 8. install (tier subagents for harnesses without a per-call model parameter)
# ════════════════════════════════════════════════════════════════════════════

AGENT_DIRS = {
    "claude-code": (".claude", "agents"),
    "opencode": (".config", "opencode", "agents"),
    "gemini-cli": (".gemini", "agents"),
}
TIER_BLURB = {
    "fast": "quick, low-risk subtasks: lookups, formatting, small edits",
    "standard": "routine implementation and everyday engineering work",
    "strong": "hard debugging, multi-file changes and careful reasoning",
    "frontier": "the hardest planning, architecture and open-ended judgment work",
}


def agent_file(harness, tier, cfg=None, inv=None):
    model = seat_models(harness, inv or {}, cfg or {}).get(tier, (None,))[0]
    lines = ["---", "name: maestro-" + tier,
             "description: Maestro %s-tier worker. Use for %s. Delegate here when `maestro-router route` picks tier %s." % (tier, TIER_BLURB[tier], tier)]
    if harness == "opencode":
        lines.append("mode: subagent")
    if model:
        lines.append("model: " + model)
    lines += ["---", "<!-- %s: written by `maestro-router install`; safe to regenerate, delete this marker to keep your edits -->" % MARKER, "",
              "You are a Maestro %s-tier worker. Do the delegated task completely and reply with a concise result." % tier, ""]
    return "\n".join(lines)


def install(harness="all", dry_run=False, user_home=None, cfg=None, env=None, inv=None):
    """Write maestro-<tier> subagent files. Never overwrites a file lacking the marker."""
    user_home = user_home or _user_home(env)
    cfg = cfg if cfg is not None else load_config(maestro_home(env))
    targets = list(AGENT_DIRS) if harness == "all" else [norm_seat(harness)]
    actions = []
    for h in targets:
        if h not in AGENT_DIRS:
            raise ValueError("unknown harness %r (choose from %s, all)" % (h, ", ".join(AGENT_DIRS)))
        for tier in TIERS:
            path = os.path.join(user_home, *AGENT_DIRS[h], "maestro-%s.md" % tier)
            content = agent_file(h, tier, cfg, inv)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    old = f.read()
            except OSError:
                old = None
            if old is None:
                act = "create"
            elif MARKER not in old:
                act = "skip (not maestro-generated)"
            elif old == content:
                act = "unchanged"
            else:
                act = "update"
            if not dry_run and act in ("create", "update"):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
            actions.append({"harness": h, "path": path, "action": act})
    return actions


# ════════════════════════════════════════════════════════════════════════════
# 9. run (one chat completion against a local or network model)
# ════════════════════════════════════════════════════════════════════════════


def run_model(model_id, task, system=None, cfg=None, inv=None, env=None, timeout=None, fetch=None, allow_load=False):
    """Send one chat completion to `local:<endpoint>/<model>` and return the reply text. Raises RuntimeError.
    A model the server has installed but not loaded is refused (the request would make it load), unless `allow_load` or `local_allow_load`.
    Ollama: POST /api/chat (stream false). OpenAI-compatible: POST /v1/chat/completions.
    An endpoint's `api_key_env` is the NAME of an env var; its value is sent as a Bearer token and never shown."""
    env = os.environ if env is None else env
    cfg = cfg if cfg is not None else load_config(maestro_home(env))
    m = re.match(r"local:([^/]+)/(.+)$", model_id or "")
    if not m:
        raise RuntimeError("model id must look like local:<endpoint>/<model>, e.g. local:spark/gpt-oss:120b")
    ep_name, model = m.groups()
    eps = local_endpoints(cfg, env)
    rec = next((r for r in (inv or {}).get("local_models") or [] if r["id"] == model_id), None)
    ep = next((e for e in eps if rec and e["url"] == rec["url"]), None) or next((e for e in eps if e["name"] == ep_name), None)
    url = rec["url"] if rec else ep["url"] if ep else None
    if not url:
        raise RuntimeError("unknown endpoint %r (add it to local_endpoints in the config, or run `maestro-router detect`)" % ep_name)
    kind = rec["kind"] if rec else (ep or {}).get("kind") or (probe_endpoint(ep, fetch, 1.0, env) or {}).get("kind")
    if not kind:
        raise RuntimeError("cannot reach %s" % url)
    if not (allow_load or cfg.get("local_allow_load")):   # live GETs: the loaded state changes (Ollama unloads after idling)
        live = probe_endpoint({"url": url, "kind": kind, "api_key_env": (ep or {}).get("api_key_env")}, fetch, 1.0, env)
        if any(x["name"] == model and not x["loaded"] for x in (live or {}).get("models", [])):
            raise RuntimeError("%s is installed on %s but not loaded, and Maestro never loads models (the request would). Load it "
                               "yourself, or pass --allow-load (config local_allow_load) and accept the memory risk" % (model, url))
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": task}]
    path, body = (("/api/chat", {"model": model, "messages": msgs, "stream": False}) if kind == "ollama"
                  else ("/v1/chat/completions", {"model": model, "messages": msgs}))
    headers = {"Content-Type": "application/json"}
    key = env.get(ep["api_key_env"]) if ep and ep.get("api_key_env") else None
    if key:
        headers["Authorization"] = "Bearer " + key
    try:
        status, raw = (fetch or _local_fetch)(url + path, headers, json.dumps(body).encode("utf-8"),
                                              timeout or cfg["run_timeout_s"])
    except Exception as e:
        raise RuntimeError("request to %s failed: %s" % (url, e.__class__.__name__))
    if status != 200:
        from .providers import error_message  # slow execution path only; route stays provider-free
        raise RuntimeError("HTTP %d from %s: %s" % (status, url, error_message(raw, key)))
    try:
        data = json.loads(raw)
        return data["message"]["content"] if kind == "ollama" else data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise RuntimeError("unexpected reply from %s" % url)


# ════════════════════════════════════════════════════════════════════════════
# 10. CLI
# ════════════════════════════════════════════════════════════════════════════


def _age(inv):
    s = time.time() - inv.get("generated_at", 0)
    return "%.0f min ago" % (s / 60) if s < 7200 else "%.1f h ago" % (s / 3600)


def summarize(inv, home=None, env=None):
    lines = ["harness CLIs:"]
    for seat, h in inv.get("harnesses", {}).items():
        lines.append("  %-13s %s (login: %s)" % (seat, h["path"], h["available"]))
    if not inv.get("harnesses"):
        lines.append("  none found")
    keys = [k for k, v in inv.get("api_keys", {}).items() if v]
    from . import providers   # detect and doctor only: `route` never calls summarize
    keys += [n if src == "env" else "%s (keys.env)" % n for n, src in providers.key_names(env, home) if n not in keys]
    lines.append("api keys present: " + (", ".join(keys) or "none"))
    lines.append("local and network models:" + ("" if inv.get("local_endpoints") else " none found"))
    for ep in inv.get("local_endpoints") or []:
        mods = [m for m in inv.get("local_models") or [] if m["url"] == ep["url"]]
        lines.append("  %s (%s, %s): %s" % (ep["name"], ep["kind"], ep["url"], ", ".join(
            "%s [%s%s%s]" % (m["name"], m["serves"], ", " + "/".join(m["specialist_for"]) if m["specialist_for"] else "",
                             "" if m.get("loaded", True) else ", installed, not loaded: not used")
            for m in mods) or "no models") + ("; %d ignored (names with characters Maestro does not pass on)" % ep["ignored"]
                                               if ep.get("ignored") else ""))
    hw = inv.get("hardware", {})
    lines.append("hardware: ram=%s GB vram=%s GB apple_silicon=%s" % (hw.get("ram_gb"), hw.get("vram_gb"), hw.get("apple_silicon")))
    for w in inv.get("warnings", []):
        lines.append("warning: " + w)
    return "\n".join(lines)


def cmd_route(a):
    text = " ".join(a.task)
    if text.strip() == "-":
        text = sys.stdin.read()
    if not text.strip():
        print("maestro: empty task", file=sys.stderr)
        return 2
    from . import commands
    try:
        d = route(text, harness=a.harness, current_tier=a.current_tier, classifier=a.classifier, explain=a.explain,
                  decider=getattr(a, "decider", None), hints=commands.hints_of(a), log=True)
    except ValueError as e:
        print("maestro: %s" % e, file=sys.stderr)
        return 2
    if a.format == "text":
        print("\n".join([format_text(d)] + (commands.explain_lines(d) if a.explain else [])))
    else:
        print(json.dumps(d, indent=None if not a.explain else 2))
    return 0


DELEGATE_CONTRACT = "Reply with only the result. No explanation, no preamble, no transcript."


def delegate_target(cand):
    """Map a route decision (or fallback entry) to a clients.complete(provider, model) pair."""
    seat, model = cand.get("seat"), cand.get("model") or ""
    if model.startswith("local:"):
        prov, _, name = model.partition("/")
        return prov, name
    return {"claude-code": "claude", "codex": "codex", "gemini-cli": "gemini"}.get(seat, seat), model


def build_capsule(text, state="", contract=DELEGATE_CONTRACT, max_chars=12000):
    """The context a worker gets: an optional state slice plus the subtask, never the
    main session's history. Oversized state is tail-truncated with a marker."""
    parts = []
    if state:
        if len(state) > max_chars:
            state = state[:max_chars] + "\n[maestro: state truncated to %d chars]" % max_chars
        parts.append("Context:\n" + state)
    parts.append("Task:\n" + text)
    if contract:
        parts.append(contract)
    return "\n\n".join(parts)


def cmd_delegate(a):
    """Spawn one stateless worker: classify the subtask, route it, run it, return only
    the distilled result. The main session keeps its state and its warm seat; the
    worker's whole context is the capsule, so switching models costs almost nothing."""
    text = " ".join(a.task)
    if text.strip() == "-":
        text = sys.stdin.read()
    if not text.strip():
        print("maestro: empty task", file=sys.stderr)
        return 2
    state = ""
    if a.state:
        try:
            state = open(a.state).read()
        except OSError as e:
            print("maestro: cannot read --state file: %s" % e, file=sys.stderr)
            return 2
    d = route(text, harness=a.harness, log=True)
    from . import clients as C
    prompt = build_capsule(text, state, "" if a.no_contract else DELEGATE_CONTRACT, a.max_capsule_chars)
    home = maestro_home()
    attempts = []
    for cand in [d] + (d.get("fallback") or []):
        provider, model = delegate_target(cand)
        if not provider or not model:
            continue
        if (cand.get("model") or "").startswith("local:"):
            # Local endpoints are not in the API provider registry; run_model speaks to them directly.
            try:
                content = run_model(cand["model"], prompt, None, load_config(home),
                                    _read_json(os.path.join(home, "inventory.json")))
                status = 200
            except RuntimeError as e:
                attempts.append({"provider": provider, "model": model, "status": str(e)[:80]})
                continue
        else:
            status, _hdrs, out = C.complete(provider, model, {"messages": [{"role": "user", "content": prompt}]})
            content = ((out.get("choices") or [{}])[0].get("message") or {}).get("content") or "" if status == 200 else ""
        attempts.append({"provider": provider, "model": model, "status": status})
        if status == 200:
            print(content)
            print(json.dumps({"worker": {"decision_id": d.get("decision_id"), "job": d.get("job"),
                                         "category": d.get("category"), "model": "%s/%s" % (provider, model),
                                         "seat": cand.get("seat"), "attempts": attempts,
                                         "capsule_chars": len(prompt)}}, indent=None), file=sys.stderr)
            return 0
    print("maestro: all worker candidates failed: %s" % json.dumps(attempts), file=sys.stderr)
    return 1


def cmd_detect(a):
    home = maestro_home()
    found = []
    if a.scan:
        try:
            found = scan_network()
        except RuntimeError as e:
            print("maestro: %s" % e, file=sys.stderr)
            return 2
        if a.save and found:
            save_endpoints(found, home)
    # scanned endpoints that were not saved are shown, but not kept in the cached inventory
    inv = detect(home=home, cfg=load_config(home), catalog=a.catalog, extra_endpoints=found, write=not a.scan or a.save)
    from . import providers  # slow path only: `route` never imports it
    pv = providers.detect_all(home=home, cfg=load_config(home), inventory=inv, list_live=a.models, write=a.models)
    if a.json:
        print(json.dumps(dict(inv, providers=pv["providers"], **({"scan": found} if a.scan else {})), indent=2))
        return 0
    print(summarize(inv, home=home))
    print("\n".join(providers.summary_lines(pv)))
    if a.scan:
        print("scan: %d endpoint(s) found on this network" % len(found))
        for e in found:
            print("  %s (%s, %d models)" % (e["url"], e["kind"], e["models"]))
        if found and not a.save:
            print("run `maestro-router detect --scan --save` to add them to %s" % config_path(home))
        elif found:
            print("saved to %s" % config_path(home))
    return 0


def cmd_run(a):
    task = " ".join(a.task)
    if not task.strip() or task.strip() == "-":
        task = sys.stdin.read()
    if not task.strip():
        print("maestro: empty task", file=sys.stderr)
        return 2
    home = maestro_home()
    try:
        print(run_model(a.model, task, a.system, load_config(home), _read_json(os.path.join(home, "inventory.json")),
                        timeout=a.timeout, allow_load=a.allow_load))
    except RuntimeError as e:
        print("maestro: %s" % e, file=sys.stderr)
        return 1
    return 0


ROW = ("harness", "category", "model", "effort", "source", "available")


def models_rows(inv, cfg):
    """The effective map as (harness, category, model, effort, source, available) rows: the harnesses in the inventory,
    then local models. Also the warnings about the config."""
    rows, warns, cap = [], [], RANK[cfg["max_tier"]]
    for seat in [s for s in SUGGESTED_MODELS if s in (inv.get("harnesses") or {})]:
        rows += [(seat, t, m, e, src, "no: above max_tier" if RANK[t] > cap else "yes")
                 for t, (m, e, src) in seat_models(seat, inv, cfg).items()]
    locs = [c for c in _candidates(Classification("ask", 0, 1.0), inv, "unknown", cfg) if c["kind"] == "local"]
    for cell in CELLS if locs else ():
        best = min((c for c in locs if RANK[c["serves"]] >= RANK[cell]), key=lambda c: _local_key(c, cell), default=None)
        rows.append(("local", cell, best["model"] if best else "-", None, best["source"] if best else "none", "yes" if best else "no"))
    pins = user_models(cfg).get("local")
    warns += ["models.local.%s = %s is not a detected local model (or not a category); ignored" % (c, v)
              for c, v in (pins.items() if isinstance(pins, dict) else ())
              if not any(c in local_pins(cfg, r) for r in inv.get("local_models") or [])]
    if _pol(cfg, "harness_models") and not _pol(cfg, "models"):
        warns.append("`harness_models` is deprecated: rename it to `models` in config.json")
    return rows, warns


def suggest_rows(inv):
    """The suggestion table as rows; OpenCode has a block per provider, the one picked from your API keys is marked."""
    pick, keys = opencode_provider(inv), inv.get("api_keys") or {}
    tag = ": picked, " + ("key set" if any(keys.get(e) for e in PROVIDER_KEYS[pick]) else "no key found")
    return [(seat + (" (%s)" % prov + tag * (prov == pick) if prov else ""), t) + _cell(v) + ("", "")
            for seat, sug in SUGGESTED_MODELS.items() for prov, row in (sug.items() if seat == "opencode" else [("", sug)])
            for t, v in row.items()]


def _show(seat, model, effort):  # a model with its effort, where the harness uses it
    return (model or "(CLI default)") + (" (effort %s)" % effort if effort and seat == "codex" else
                                         " (effort ignored by %s)" % seat.split()[0] if effort else "")


def row_lines(rows):
    out = []
    for i, (seat, cat, model, effort, source, avail) in enumerate(rows):
        out += [seat] * (i == 0 or seat != rows[i - 1][0])
        out.append(("  %-9s %-40s %-10s %s" % (cat, _show(seat, model, effort), source, avail)).rstrip())
    return out


def edit_models(home, changes):
    """Apply [(harness, category, cell, or None to remove)] to config.json's `models` (a deprecated harness_models moves
    in) and write it with indent 2, keeping every other key. Refuses a config that is not valid JSON."""
    path = config_path(home)
    raw = _read_json(path)
    if raw is None and os.path.exists(path):
        raise ValueError("%s is not valid JSON; fix it first" % path)
    raw = raw if isinstance(raw, dict) else {}
    models = raw.get("models") if isinstance(raw.get("models"), dict) else {}
    if not models and isinstance(raw.get("harness_models"), dict):
        models = {k: {"strong": v} if isinstance(v, str) else dict(v) for k, v in raw.pop("harness_models").items()
                  if isinstance(v, (str, dict))}
    for seat, cat, cell in changes:
        if cell is None:
            models.get(seat, {}).pop(cat, None)
        else:
            models.setdefault(seat, {})[cat] = cell
    raw.pop("models", None)
    raw.update({"models": {k: v for k, v in models.items() if v}} if any(models.values()) else {})
    if changes and not _write_json(path, raw, 2):
        raise ValueError("cannot write " + path)


def _models_set(home, verb, rest, effort):
    if len(rest) != (2 if verb == "set" else 1):
        raise ValueError("usage: maestro-router models set <harness>.<category> <model> [--effort E] | unset <harness>.<category>")
    seat, _, cat = rest[0].partition(".")
    seat = norm_seat(seat)
    known = bool(seat) and (seat in SUGGESTED_MODELS or seat == "local" or (seat.replace("-", "").isalnum() and seat[:1].isalpha()))
    if cat not in (CELLS if seat == "local" else TIERS) or not known:
        raise ValueError("%r: use <harness>.<category>, harness a name or one of %s, local; category one of %s (local also: local)" % (
            rest[0], ", ".join(SUGGESTED_MODELS), ", ".join(TIERS)))
    if effort and seat == "local":
        raise ValueError("--effort does not apply to local models")
    edit_models(home, [(seat, cat, None if verb == "unset" else {"model": rest[1], "effort": effort} if effort else rest[1])])
    return ["%s %s.%s%s" % (verb, seat, cat, " = " + _show(seat, rest[1], effort) if verb == "set" else ""),
            "run `maestro-router install` to update the subagent files"]


def check_models(cat, cells):
    """[(seat, category, model, verdict, detail)] for (seat, category, model) cells against the slim catalog. Verdicts:
    ok, missing, deprecated, newer (a newer non-deprecated tool-calling model of the same family and name prefix, so
    a Deep Research model is not a newer Pro; `-latest` aliases track the newest and are exempt) and skipped (unknown provider)."""
    out = []
    for seat, t, model in cells:
        prov, mid = model.split("/", 1) if seat == "opencode" and "/" in model else ({"codex": "openai", "gemini-cli": "google"}.get(seat), model)
        known = (cat.get(prov) or {}).get("models") or {}
        e = known.get(mid)
        v, why = (("skipped", "%s is not in the catalog" % prov) if prov not in cat else ("missing", "not in the catalog")
                  if not isinstance(e, list) else ("deprecated", "marked deprecated") if e[3] == "deprecated" else ("ok", ""))
        if v == "ok" and e[1] and e[2] and not mid.endswith("-latest"):
            new = max(((x[2], k) for k, x in known.items() if isinstance(x, list) and x[1] == e[1] and (x[2] or "") > e[2]
                       and x[3] != "deprecated" and x[4] is not False and k.split("-")[0] == mid.split("-")[0]
                       and not k.endswith("-latest")), default=None)
            v, why = ("newer", "%s (%s)" % (new[1], new[0])) if new else (v, why)
        out.append((seat, t, model, v, why))
    return out


def _models_check(home, inv, cfg):
    cat = fetch_catalog(home)
    if not any(isinstance(e, list) for p in cat.values() for e in p.get("models", {}).values()):
        raise RuntimeError("cannot reach models.dev and there is no cached catalog (run `maestro-router detect --catalog` online)")
    res = check_models(cat, [(s, t, m) for s in SUGGESTED_MODELS if s != "claude-code"
                             for t, (m, _, _) in seat_models(s, inv, cfg).items() if m])
    lines = ["checked against models.dev (the cached copy if it was unreachable); Claude Code aliases track the latest and are skipped"]
    lines += ["  %-10s %-20s %-34s %s" % (v.upper() if v != "ok" else v, "%s.%s" % (s, t), m, why) for s, t, m, v, why in res]
    lines.append("fix with `maestro-router models set <harness>.<category> <model>`" if any(r[3] not in ("ok", "skipped") for r in res)
                 else "all ids exist and none is deprecated")
    return ([dict(zip(("harness", "category", "model", "verdict", "detail"), r)) for r in res], lines,
            int(any(r[3] in ("missing", "deprecated") for r in res)))


def _map_view(home, cfg):
    """(models.json, table lines) once `maestro-router discover` has run, else None: `maestro-router models` then shows the harness table."""
    from . import discover
    return discover.models_view(home, cfg)


def cmd_models(a):
    home = maestro_home()
    cfg = load_config(home)
    verb, rest = (a.args[0], a.args[1:]) if a.args else ("", [])
    rc = 0
    try:
        if verb in ("set", "unset"):
            data = lines = _models_set(home, verb, rest, a.effort)
        elif verb:
            raise ValueError("unknown action %r (use set or unset)" % verb)
        else:
            inv = ensure_inventory(home, cfg=cfg)
            rows, warns = models_rows(inv, cfg)
            if a.save:
                new = [r for r in rows if r[4] == "suggested"]
                edit_models(home, [(h, c, {"model": m, "effort": e} if e else m) for h, c, m, e, _, _ in new])
                data = lines = ["+ %s.%s = %s" % (h, c, _show(h, m, e)) for h, c, m, e, _, _ in new] or ["nothing to add: every cell is already set"]
            elif a.check:
                data, lines, rc = _models_check(home, inv, cfg)
            elif a.suggest:
                rows = suggest_rows(inv)
                data = [dict(zip(ROW, r)) for r in rows]
                lines = ["suggested models as of %s (%s); they go stale, verify with `maestro-router models --check`" % (
                    SUGGESTED_AS_OF, SUGGESTED_SOURCE)] + row_lines(rows)
            elif not a.seats and _map_view(home, cfg):
                data, lines = _map_view(home, cfg)
            else:
                data = {"harness": detect_harness(), "suggested_as_of": SUGGESTED_AS_OF, "warnings": warns,
                        "models": [dict(zip(ROW, r)) for r in rows]}
                lines = row_lines(rows) + ["suggestions as of %s (%s): `maestro-router models --suggest` lists them, `--check` verifies them" % (
                    SUGGESTED_AS_OF, SUGGESTED_SOURCE)] + ["warning: " + w for w in warns] + ["harness detected now: " + data["harness"]]
    except (ValueError, RuntimeError) as e:
        print("maestro: %s" % e, file=sys.stderr)
        return 2 if isinstance(e, ValueError) else 1
    print(json.dumps(data, indent=2) if a.json else "\n".join(lines))
    return rc


def cmd_keys(a):
    from . import providers
    return providers.cmd_keys(a)


def cmd_serve(a):
    from . import server
    return server.cmd_serve(a)


def cmd_setup(a):
    from . import onboard
    return onboard.cmd_setup(a)


def cmd_hermes(a):
    from . import onboard
    return onboard.cmd_hermes(a)


def cmd_discover(a):
    from . import discover
    return discover.cmd_discover(a)


def cmd_install(a):
    if getattr(a, "hooks", False):
        from . import commands
        return commands.cmd_install_hooks(a)
    if a.statusline:
        from . import quota
        path, snippet = quota.install_statusline(maestro_home())
        print("wrote %s\nadd this to ~/.claude/settings.json (Maestro does not edit it):\n%s" % (path, snippet))
        return 0
    try:
        home = maestro_home()
        cfg = load_config(home)
        actions = install(a.harness, a.dry_run, cfg=cfg, inv=ensure_inventory(home, cfg=cfg))
    except ValueError as e:
        print("maestro: %s" % e, file=sys.stderr)
        return 2
    for x in actions:
        would = a.dry_run and x["action"] in ("create", "update")
        print("%-30s %s" % (("would " if would else "") + x["action"], x["path"]))
    return 0


def cmd_config(a):
    home = maestro_home()
    path = config_path(home)
    print(json.dumps({"path": path, "exists": os.path.exists(path), "config": load_config(home)}, indent=2))
    return 0


def cmd_doctor(a):
    home = maestro_home()
    print("maestro %s  python %s  home %s" % (VERSION, sys.version.split()[0], home))
    cfg = load_config(home)
    eff = cfg["classifier"]
    if eff == "auto":
        eff = "jev (TYPESAFE_API_KEY set)" if os.environ.get("TYPESAFE_API_KEY") else "rules (no TYPESAFE_API_KEY)"
    print("config: %s (%s), classifier=%s" % (config_path(home), "found" if os.path.exists(config_path(home)) else "defaults", eff))
    from . import s1
    md, why = s1.load()
    bad = s1.verify() if md else []
    print("s1: %s; decider=%s, shadow=%s" % (("%s (%s)" % (s1_policy(md), "sections ok" if not bad else "sha256 mismatch: " + ", ".join(bad)))
                                              if md else "not used: " + why, cfg["decider"], cfg["s1_shadow"]))
    inv = detect(home=home)
    print("inventory refreshed (%s)\n%s" % (_age(inv), summarize(inv, home=home)))
    print("current harness: %s" % detect_harness())
    d = route("fix the failing test in auth.py", inventory=inv, config=cfg)
    print("sample route: " + format_text(d))
    if a.jev:
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            print("jev: TYPESAFE_API_KEY is not set")
            return 1
        cands = build_candidates(classify_rules(""), {}, "claude-code", cfg)  # the four Claude Code categories
        from . import jev as J
        j = J.JevClassifier(key)
        t0 = time.perf_counter()
        c = j.classify("Design a multi-region failover architecture for our payments service", cands, cfg["objective"])
        ms = (time.perf_counter() - t0) * 1000
        if not c.jev:
            print("jev: FAILED after %.0f ms: %s" % (ms, c.note))
            return 1
        print("jev: ok in %.0f ms -> %s, %s/c%d, confidence %.2f" % (
            ms, c.jev["choice"], c.job, c.difficulty, c.jev["confidence"]))
        print(json.dumps(j.last_answers, indent=2))
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="maestro-router", description="Route each task to the right model you already have.")
    p.add_argument("--version", action="version", version="maestro-router " + VERSION)
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("route", help="pick a model and seat for a task")
    r.add_argument("task", nargs="+", help='task text, or "-" to read stdin')
    r.add_argument("--harness", help="claude-code | opencode | gemini-cli | codex | ... (default: auto)")
    r.add_argument("--current-tier", choices=TIERS, help="tier of the model you are running on")
    r.add_argument("--classifier", choices=("auto", "rules", "s1", "jev"), help="auto (default): jev if TYPESAFE_API_KEY is set")
    r.add_argument("--format", choices=("json", "text"), default="json")
    r.add_argument("--json", action="store_const", const="json", dest="format", help="JSON output (default)")
    r.add_argument("--explain", action="store_true", help="include classifier signals, the tier walk, S1's posterior and the options")
    r.set_defaults(fn=cmd_route)
    w = sub.add_parser("delegate", help="spawn one stateless worker for a subtask and print only its result")
    w.add_argument("task", nargs="+", help='subtask text, or "-" to read stdin')
    w.add_argument("--state", help="file with the state slice the worker needs (never your full history)")
    w.add_argument("--harness", help="harness the worker runs in (default: auto)")
    w.add_argument("--no-contract", action="store_true", help="do not append the reply-with-only-the-result contract")
    w.add_argument("--max-capsule-chars", type=int, default=12000, help="truncate --state beyond this many chars")
    w.set_defaults(fn=cmd_delegate)
    d = sub.add_parser("detect", help="find CLIs, keys, local and network models, and hardware")
    d.add_argument("--catalog", action="store_true", help="enrich with models.dev pricing tiers")
    d.add_argument("--scan", action="store_true", help="also probe this machine's /24 subnet for Ollama (11434) and OpenAI-compatible (8000) servers")
    d.add_argument("--save", action="store_true", help="with --scan: add what was found to local_endpoints in the config")
    d.add_argument("--models", action="store_true", help="also list each API provider's models live (network, 8 s timeout each) and save providers.json")
    d.add_argument("--json", action="store_true")
    d.set_defaults(fn=cmd_detect)
    k = sub.add_parser("keys", help="manage API keys stored in ~/.maestro/keys.env (mode 600)")
    k.add_argument("action", choices=("set", "unset", "list"))
    k.add_argument("name", nargs="?", help="provider id (openai, anthropic, ...) or an env var name")
    k.add_argument("--all", action="store_true", help="with list: show providers without a key too")
    k.add_argument("value", nargs="?", help="the key; omit to be prompted without echo")
    k.set_defaults(fn=cmd_keys)
    sv = sub.add_parser("serve", help="run the OpenAI-compatible router and the web UI")
    sv.add_argument("--host", default="127.0.0.1", help="bind address; anything but loopback needs MAESTRO_API_KEY (default 127.0.0.1)")
    sv.add_argument("--port", type=int, default=4545)
    sv.add_argument("--open", action="store_true", help="open the web UI in a browser")
    sv.add_argument("--no-discover", action="store_true", help="do not run discovery in the background when models.json is missing or stale")
    sv.set_defaults(fn=cmd_serve)
    se = sub.add_parser("setup", help="quick start: detect, add keys, discover, connect Hermes and the skill")
    se.add_argument("--yes", "-y", action="store_true", help="take the default answer to every question (no prompts)")
    se.add_argument("--endpoint", action="append", default=[], metavar="URL", help="a model server on your network (vLLM, Ollama, ...), e.g. http://spark.local:8000; repeatable")
    se.set_defaults(fn=cmd_setup)
    he = sub.add_parser("hermes", help="print (or install) the Hermes Agent config for Maestro")
    he.add_argument("--print", action="store_true", help="print the YAML (default)")
    he.add_argument("--install", action="store_true", help="add missing blocks to ~/.hermes/config.yaml (backs it up; never edits existing blocks)")
    he.add_argument("--dry-run", action="store_true")
    he.add_argument("--base-url", help="the /v1 URL of your Maestro server (default http://HOST:PORT/v1)")
    he.add_argument("--host", default="127.0.0.1")
    he.add_argument("--port", type=int, default=4545)
    he.set_defaults(fn=cmd_hermes)
    ru = sub.add_parser("run", help="send one chat completion to a local or network model")
    ru.add_argument("model", help="model id from a route decision, e.g. local:spark/gpt-oss:120b")
    ru.add_argument("--system", help="system prompt")
    ru.add_argument("--timeout", type=float, help="seconds (default: run_timeout_s in the config, 300)")
    ru.add_argument("--allow-load", action="store_true", help="run a model that is installed but not loaded (the server loads it; can exhaust memory)")
    ru.add_argument("task", nargs="*", help='task text, or "-" / nothing to read stdin')
    ru.set_defaults(fn=cmd_run)
    mo = sub.add_parser("models", help="show or change the model for each harness and category")
    mo.add_argument("args", nargs="*", help="set <harness>.<category> <model> | unset <harness>.<category>")
    mo.add_argument("--effort", choices=EFFORTS, help="with set: reasoning effort (only Codex uses it)")
    mo.add_argument("--suggest", action="store_true", help="print the suggested models and the OpenCode provider picked")
    mo.add_argument("--save", action="store_true", help="write the suggestions into the config for every cell you left unset")
    mo.add_argument("--check", action="store_true", help="compare the ids with the live models.dev catalog (network)")
    mo.add_argument("--seats", action="store_true", help="show the per-harness table even when `maestro-router discover` has written models.json")
    mo.add_argument("--json", action="store_true")
    mo.set_defaults(fn=cmd_models)
    di = sub.add_parser("discover", help="detect every usable model, score it against the catalog and write ~/.maestro/models.json")
    di.add_argument("--refresh-catalog", action="store_true", help="merge the live models.dev API (network; falls back to the bundled snapshot)")
    di.add_argument("--backtest", action="store_true", help="route the eval prompts over the discovered map (offline): category hit rate, cost, local share")
    di.add_argument("--live", type=int, metavar="N", default=0, help="send N tiny prompts to each model the backtest uses and record latency and failures (costs a few tokens)")
    di.add_argument("--local-fit", action="store_true", help="per local box: the best open-weight models that fit, with quantization and serve command")
    di.add_argument("--all", action="store_true", help="list every model, not the strongest 12 per provider")
    di.add_argument("--prompts", help="eval prompts (default: evals/prompts.jsonl)")
    di.add_argument("--json", action="store_true")
    di.set_defaults(fn=cmd_discover)
    i = sub.add_parser("install", help="write maestro-<tier> subagents")
    i.add_argument("--harness", default="all", help="claude-code | opencode | gemini-cli | all")
    i.add_argument("--dry-run", action="store_true")
    i.add_argument("--statusline", action="store_true",
                   help="instead: write ~/.maestro/statusline.sh (caches Claude Code's rate limits) and print the settings.json snippet")
    i.set_defaults(fn=cmd_install)
    dr = sub.add_parser("doctor", help="check the setup")
    dr.add_argument("--jev", action="store_true", help="make a live Jev call")
    dr.set_defaults(fn=cmd_doctor)
    c = sub.add_parser("config", help="print the effective config")
    c.set_defaults(fn=cmd_config)
    from . import commands
    commands.register(sub)   # route hints, install --hooks, s1, decide, outcome, feedback, learn
    return p


def main(argv=None):
    p = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["run"] and len(argv) > 2 and len(argv[2]) > 1 and argv[2].startswith("-") \
            and argv[2] not in ("--system", "--timeout", "-h", "--help"):
        argv.insert(2, "--")  # a task that starts with a dash is still a task
    a = p.parse_args(argv)
    if not getattr(a, "fn", None):
        p.print_help()
        return 2
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
