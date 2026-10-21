"""Maestro's arena: run the same tasks through different harnesses and models, grade and rank the outputs, and write a
leaderboard plus outcome rows S1 can train on. The CLI is `python3 evals/arena.py` (cli.py); docs/ARENA-RUNBOOK.md is
the step-by-step guide.

    common.py   paths, files, small statistics
    seats.py    which contestant seats this machine has (harness CLIs, API keys, served local models, seats.json)
    harness.py  THE argv table for every harness, process handling, output parsing, one chat answer from any seat
    suites.py   loading tasks; grading verifiable answers; one agentic fixture run in a temp git repo
    rank.py     open-ended ranking: criteria first, blind shuffled judging both ways, Bradley-Terry, bias checks
    runner.py   `run`: cells, concurrency, quota, budget, resume
    report.py   `report`: leaderboards, value frontier, routing replay, markdown and single-file HTML
    export.py   `export`: outcomes in the evals/alpha format
"""
