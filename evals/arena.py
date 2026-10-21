#!/usr/bin/env python3
"""Maestro's arena: run the same tasks through different harnesses and models, grade and rank the outputs, and produce a
leaderboard plus outcome data for routing. The code is the evals/arena/ package; docs/ARENA-RUNBOOK.md is the guide.

    python3 evals/arena.py seats [--json] [--seats GLOB]
    python3 evals/arena.py run --suite verifiable|agentic|open|all --seats GLOBS [--items F] [--open-items F]
                               [--tasks N] [--repeat K] [--concurrency C] [--budget-usd X] [--judges GLOBS] [--dry-run]
    python3 evals/arena.py report <run_id> [--html] [--decider utility]
    python3 evals/arena.py export <run_id>
"""
import sys

from arena.cli import main  # the package next to this file (a package wins over this module of the same name)

if __name__ == "__main__":
    sys.exit(main())
