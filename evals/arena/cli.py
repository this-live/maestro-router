"""Command line of the arena (`python3 evals/arena.py ...`). docs/ARENA-RUNBOOK.md walks through a first run."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

from . import export as X
from . import report as R
from . import runner
from . import seats as S
from .common import FIXTURES, RUNS, SEATS_FILE, globs, m
from .harness import EXE, HARNESSES

USAGE = """examples:
  python3 evals/arena.py seats                                   # what can compete on this machine
  python3 evals/arena.py run --suite verifiable --items V.jsonl --seats 'claude-code:haiku,claude-code:sonnet' --tasks 5 --dry-run
  python3 evals/arena.py run --suite agentic --seats 'claude-code:*,codex:*,opencode:spark/*' --tasks 5 --budget-usd 5
  python3 evals/arena.py report <run_id> --html
  python3 evals/arena.py export <run_id>

Cost: every cell is a real call on your plan or API key. Start with --dry-run and small --tasks; --budget-usd caps the
estimated spend (default $5)."""


def cmd_seats(a):
    env = os.environ
    home = m.maestro_home(env)
    cfg = m.load_config(home)
    all_ = S.discover(env, home, cfg, seats_file=a.seats_file)
    for s in all_:
        S.annotate(s)
    for s in all_:
        if s["harness"] in HARNESSES and not shutil.which(EXE[s["harness"]]):
            s["source"] = (s.get("source") or "") + ", CLI not on PATH"
    shown = S.select(all_, globs(a.seats))
    fj = S.file_judges(a.seats_file)
    judges, note = (S.select(all_, fj), "from %s" % a.seats_file) if fj else S.default_judges(all_)
    if a.json:
        print(json.dumps({"seats": shown, "default_judges": [j["id"] for j in judges], "judge_note": note,
                          "harnesses": {h: {"exe": EXE[h], "path": shutil.which(EXE[h]), "verified": v["verified"]}
                                        for h, v in HARNESSES.items()}}, indent=1, sort_keys=True))
        return 0
    print(S.table(shown) if shown else "no seats found")
    missing = [h for h in HARNESSES if not shutil.which(EXE[h])]
    if missing:
        print("\nnot on PATH: %s" % ", ".join("%s (%s)" % (h, EXE[h]) for h in missing))
    print("\ndefault judges: %s" % (", ".join("%s (%s)" % (j["id"], j.get("family")) for j in judges) or "none"))
    if note:
        print("  " + note)
    return 0


def cmd_run(a):
    a.run_id = a.run_id or time.strftime("%Y%m%d-%H%M%S")
    try:
        return runner.run(a)
    except ValueError as e:
        print("arena: %s" % e, file=sys.stderr)
        return 2


def cmd_report(a):
    try:
        doc, d = R.report(a.run_id, a.runs_dir, a.html, a.boot, a.decider)
    except ValueError as e:
        print("arena: %s" % e, file=sys.stderr)
        return 2
    print(doc.md())
    print("written: %s%s" % (os.path.join(d, "report.md"), " and report.html" if a.html else ""))
    return 0


def cmd_export(a):
    try:
        path, rows = X.export(a.run_id, a.runs_dir, a.out)
    except ValueError as e:
        print("arena: %s" % e, file=sys.stderr)
        return 2
    print("%d outcome rows -> %s" % (len(rows), path))
    return 0


def parser():
    ap = argparse.ArgumentParser(prog="arena.py", description="Maestro's arena: harnesses and models head to head.",
                                 epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sp = sub.add_parser("seats", help="list the contestant seats on this machine")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--seats", action="append", help="glob filter (comma-separated or repeated)")
    sp.add_argument("--seats-file", default=SEATS_FILE)
    sp.set_defaults(fn=cmd_seats)
    rp = sub.add_parser("run", help="run suites over seats (resumable)")
    rp.add_argument("--suite", choices=("verifiable", "agentic", "open", "all"), default="verifiable")
    rp.add_argument("--seats", action="append", help="seat globs (required unless --dry-run)")
    rp.add_argument("--items", help="verifiable items JSONL {id, prompt, checker_py, ...}")
    rp.add_argument("--open-items", help="open items JSONL {id, prompt, job}")
    rp.add_argument("--fixtures", default=FIXTURES, help="agentic fixtures directory")
    rp.add_argument("--only", action="append", help="task id globs")
    rp.add_argument("--tasks", type=int, default=0, help="at most N tasks per suite (file order, or --shuffle SEED)")
    rp.add_argument("--shuffle", type=int, default=None, metavar="SEED")
    rp.add_argument("--repeat", type=int, default=1)
    rp.add_argument("--concurrency", type=int, default=2)
    rp.add_argument("--budget-usd", type=float, default=5.0, help="stop starting calls at this estimated spend (default 5)")
    rp.add_argument("--no-budget", dest="budget_usd", action="store_const", const=None)
    rp.add_argument("--judges", action="append", help="judge seat globs (default: strongest seat of each family)")
    rp.add_argument("--timeout", type=float, default=300.0, help="chat answer timeout, seconds")
    rp.add_argument("--agentic-timeout", type=float, default=None, help="override meta.timeout_s")
    rp.add_argument("--min-headroom", type=float, default=0.10, help="skip a seat whose quota window has less left")
    rp.add_argument("--quota-wait", type=float, default=0.0, help="seconds to wait for quota before skipping a cell")
    rp.add_argument("--max-attempts", type=int, default=2)
    rp.add_argument("--seed", type=int, default=0)
    rp.add_argument("--run-id")
    rp.add_argument("--runs-dir", default=RUNS)
    rp.add_argument("--seats-file", default=SEATS_FILE)
    rp.add_argument("--dry-run", action="store_true")
    rp.set_defaults(fn=cmd_run)
    rep = sub.add_parser("report", help="leaderboards and routing replay for a run")
    rep.add_argument("run_id")
    rep.add_argument("--html", action="store_true")
    rep.add_argument("--boot", type=int, default=1000)
    rep.add_argument("--decider", choices=("ladder", "utility"), default=None)
    rep.add_argument("--runs-dir", default=RUNS)
    rep.set_defaults(fn=cmd_report)
    ex = sub.add_parser("export", help="outcomes in the evals/alpha format")
    ex.add_argument("run_id")
    ex.add_argument("--out")
    ex.add_argument("--runs-dir", default=RUNS)
    ex.set_defaults(fn=cmd_export)
    return ap


def main(argv=None):
    ap = parser()
    a = ap.parse_args(argv)
    if not getattr(a, "fn", None):
        ap.print_help()
        return 2
    return a.fn(a)
