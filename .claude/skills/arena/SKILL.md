---
name: arena
description: Run Maestro's arena to compare local harnesses and models and rank their outputs. Use when the user wants to test Claude Code, Codex, Gemini CLI, OpenCode, Hermes or local (Spark/vLLM) models head to head, build a leaderboard, check Maestro's routing against real outcomes, or export outcomes for S1.
---

# Arena

Follow `docs/ARENA-RUNBOOK.md` step by step. It is the source of truth for commands, flags and troubleshooting. Run
everything from the repository root with `python3 evals/arena.py`.

How to work:

1. **Check the machine first** (runbook sections 1-3): `maestro-router doctor`, the one-line headless check for each harness
   CLI the user wants to compare, then `python3 evals/arena.py seats`. If a CLI rejects a flag, fix its row in
   `HARNESSES` (`evals/arena/harness.py`), run `python -m unittest discover -s tests -p test_arena.py`, and tell the
   user what you changed.
2. **Ask before spending.** Every cell spends the user's plan quota, API money or Spark GPU time. Before any run larger
   than a smoke test (more than about 20 cells, any `--repeat`, or any judged open suite), show the user the
   `--dry-run` output and ask for:
   - the seats (which harnesses and models, which local models);
   - the suites and number of tasks;
   - `--budget-usd`;
   - the judges, if the default panel is single-family or not what they want.

   Never raise `--budget-usd`, add seats or rerun a large suite without asking.
3. **Run small first:** about 10 verifiable, 5 agentic and 5 open tasks, with 3-6 seats including at least one local
   model (runbook section 5). Use one `--run-id` and resume with the same id. Keep `--concurrency` at 1-2, and at 1
   when local seats share the Spark.
4. **Report the leaderboard.** Run `python3 evals/arena.py report <run_id> --html` and give the user:
   - the leaderboard per suite, with the 95% intervals;
   - the value-frontier sentence;
   - the routing-replay table (router vs. always-X vs. hindsight);
   - for the open suite, the judging checks: say plainly if judging was single-family or self-preference is large;
   - the estimated spend.

   Point to `evals/arena/runs/<run_id>/report.html`. Do not paste answers or diffs from the run directory into chat
   unless asked.
5. **Export on request:** `python3 evals/arena.py export <run_id>` writes `outcomes.jsonl` (no answer text) for S1.

Never send task text anywhere except to the seats being tested. Never read, print or store API keys or harness login
files. Do not commit anything under `evals/arena/runs/`.
