"""The S1 subcommands and `route` flags (spec §12.5), registered on the router's parser by `register(sub)`.

    maestro-router route ... [--job J] [--tools yes|no] [--self-contained yes|no] [--size s|m|l|xl] [--stakes low|normal|high]
                      [--context-tokens N] [--decider ladder|utility]
    maestro-router s1 "<text>" [--json]                      S1's heads for a task
    maestro-router decide [file|-] [--calibrate FILE]        one /v1/systemone request (systemone.py)
    maestro-router outcome <id|last> pass|partial|fail [--tests P/T] [--from-hook]
    maestro-router feedback <id> good|bad|partial            a proxy decision
    maestro-router learn [--report|--audit|--reset|--dry-run]
    maestro-router install --hooks                           the Claude Code SubagentStop hook (outcome --from-hook)

The handlers import their modules when they run, so registering costs nothing and `route` never loads learn or systemone.
"""
from __future__ import annotations

import importlib
import json
import math
import os
import sys

from . import router as R

HOOK_MARK = "outcome --from-hook"


def _call(module, fn):
    return lambda a: getattr(importlib.import_module("%s.%s" % (__package__ or "maestro", module)), fn)(a)


def token_count(v):   # --context-tokens: 0 to 10,000,000 (argparse reports anything else as an invalid token_count value)
    if 0 <= int(v) <= 10 ** 7:
        return int(v)
    raise ValueError(v)


def register(sub):
    """Add the hint flags to `route`, --hooks to `install`, and the s1, decide, outcome, feedback and learn subcommands."""
    r = sub.choices["route"]
    r.add_argument("--job", choices=R.JOBS, help="hint: the job, when you know it")
    r.add_argument("--tools", choices=("yes", "no"), help="hint: the task needs repo tools or commands")
    r.add_argument("--self-contained", dest="self_contained", choices=("yes", "no"), help="hint: everything needed is in the text")
    r.add_argument("--size", choices=("s", "m", "l", "xl"), help="hint: how big the answer or change is")
    r.add_argument("--stakes", choices=("low", "normal", "high"), help="hint: what getting it wrong costs")
    r.add_argument("--context-tokens", dest="context_tokens", type=token_count, metavar="N", help="hint: your conversation's size in tokens")
    r.add_argument("--decider", choices=("ladder", "utility"), help="override the config's decider for this task")
    sub.choices["install"].add_argument("--hooks", action="store_true", help="instead: add the Claude Code SubagentStop hook that "
                                        "records test results of delegated work (maestro-router outcome --from-hook)")
    s = sub.add_parser("s1", help="print S1's heads for a task (job, difficulty, tools, stakes, tokens, ...)")
    s.add_argument("task", nargs="+", help='task text, or "-" to read stdin')
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_s1)
    d = sub.add_parser("decide", help="answer one /v1/systemone request (Jev's wire format) from a file or stdin")
    d.add_argument("file", nargs="?", default="-")
    d.add_argument("--calibrate", metavar="FILE", help="fit the delegate's temperatures on typed-eval rows")
    d.set_defaults(fn=_call("systemone", "cmd_decide"))
    o = sub.add_parser("outcome", help="record how a routed task went (skill mode)")
    o.add_argument("id", nargs="?", default="last", help="decision_id from `maestro-router route`, or last")
    o.add_argument("verdict", nargs="?", help="pass | partial | fail")
    o.add_argument("--tests", metavar="P/T", help="tests passed / total")
    o.add_argument("--from-hook", dest="from_hook", action="store_true", help="read the Claude Code hook JSON on stdin")
    o.set_defaults(fn=_call("learn", "cmd_outcome"))
    f = sub.add_parser("feedback", help="rate a proxy decision (its id is in the x-maestro-decision header)")
    f.add_argument("id")
    f.add_argument("value", choices=("good", "bad", "partial"))
    f.set_defaults(fn=_call("learn", "cmd_feedback"))
    lr = sub.add_parser("learn", help="fit personal offsets from your outcomes (personal.json)")
    for flag, what in (("--report", "show what was learned"), ("--audit", "rate decisions that carry an implicit signal"),
                       ("--reset", "delete personal.json"), ("--dry-run", "fit, but do not write")):
        lr.add_argument(flag, action="store_true", help=what)
    lr.set_defaults(fn=_call("learn", "cmd_learn"))


def hints_of(a):
    """The route hints given on the command line."""
    return {k: getattr(a, k) for k in ("job", "tools", "self_contained", "size", "stakes", "context_tokens") if getattr(a, k, None) is not None}


def explain_lines(d):
    """`route --explain --format text`: the posterior and the options table."""
    ex, out = d.get("explain") or {}, []
    p = ex.get("posterior")
    if p:
        out.append("posterior (%s): %s; b %+.2f±%.2f; %s; tools %.2f, self-contained %.2f, stakes %.1f, detectable %.2f; ~%d tokens%s; ood %.2f"
                   % (p.get("source"), ", ".join("%s %.2f" % kv for kv in list(p["job_p"].items())[:3]), p["b"][0], p["b"][1], p.get("group"),
                      p["tools"], p["self_contained"], p["stakes"], p["detectable"], math.exp(p["tok_out"][0]),
                      " x %d turns" % round(math.exp(p["turns"][0])) if p.get("group") == "agentic" else "", p.get("ood", 0)))
    if ex.get("options"):
        out.append("  %-44s %6s %6s %8s %8s %6s %8s" % ("seat", "P", "P~", "$", "quota $", "secs", "EU"))
        out += ["  %-44s %6.2f %6.2f %8.4f %8.4f %6d %8.4f" % (o["seat"][:44], o["p"], o["p_lo"], o["usd"], o["quota_usd"], o["secs"], o["eu"])
                for o in ex["options"]]
        out += ["  skipped %s: %s" % (x["seat"], x["reason"]) for x in ex.get("skipped") or []]
        out.append("shadow (the other decider): %s" % ex.get("shadow"))
    return out


def cmd_s1(a):
    text = sys.stdin.read() if " ".join(a.task).strip() == "-" else " ".join(a.task)
    from . import s1
    post = s1.posterior(text, R.classify_rules(text))
    if a.json:
        print(json.dumps(post, indent=1))
        return 0
    md, why = s1.load()
    jobs = sorted(post["job_p"].items(), key=lambda kv: -kv[1])[:3]
    print("source %s%s" % (post["source"], " (trained: %s; the rest from job priors)" % ", ".join(post["trained"]) if md else " (%s)" % why))
    print("job: " + ", ".join("%s %.2f" % kv for kv in jobs))
    print("difficulty b: %+.2f ± %.2f   group: %s   ood: %.2f" % (post["b"][0], post["b"][1], post["group"], post["ood"]))
    print("tools %.2f  self-contained %.2f  stakes %.2f  detectable %.2f  ambiguity %.2f" % (
        post["tools"], post["self_contained"], post["stakes"], post["detectable"], post["ambiguity"]))
    print("output tokens ~%d (log sd %.2f)%s" % (math.exp(post["tok_out"][0]), post["tok_out"][1],
                                                 "   turns ~%d" % round(math.exp(post["turns"][0])) if post["group"] == "agentic" else ""))
    return 0


def hook_command():
    import shlex
    return "%s %s %s" % (shlex.quote(sys.executable), shlex.quote(os.path.dirname(os.path.abspath(__file__))), HOOK_MARK)


def install_hooks(user_home=None, dry_run=False):
    """Add the SubagentStop hook to ~/.claude/settings.json (kept as is otherwise; a backup is written first). (path, action)."""
    path = os.path.join(user_home or R._user_home(), ".claude", "settings.json")
    raw = R._read_json(path)
    if raw is None and os.path.exists(path):
        raise ValueError("%s is not valid JSON; fix it first" % path)
    raw = raw if isinstance(raw, dict) else {}
    hooks = raw.setdefault("hooks", {}) if isinstance(raw.get("hooks", {}), dict) else None
    if hooks is None or not isinstance(hooks.get("SubagentStop", []), list):
        raise ValueError("%s has an unexpected hooks section; add the hook by hand: %s" % (path, hook_command()))
    if any(HOOK_MARK in str(h.get("command", "")) for e in hooks.get("SubagentStop", []) if isinstance(e, dict) for h in e.get("hooks") or []):
        return path, "unchanged"
    hooks.setdefault("SubagentStop", []).append({"hooks": [{"type": "command", "command": hook_command()}]})
    if not dry_run:
        if os.path.exists(path):
            with open(path, "rb") as src, open(path + ".maestro-bak", "wb") as dst:
                dst.write(src.read())
        if not R._write_json(path, raw, indent=2, private=False):   # Claude Code's file and directory, not Maestro's
            raise ValueError("cannot write " + path)
    return path, "would add" if dry_run else "added"


def cmd_install_hooks(a):
    try:
        path, act = install_hooks(dry_run=a.dry_run)
    except ValueError as e:
        print("maestro: %s" % e, file=sys.stderr)
        return 2
    print("%s the SubagentStop hook in %s (%s)" % (act, path, hook_command()))
    return 0
