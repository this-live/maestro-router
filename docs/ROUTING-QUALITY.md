# Routing quality (Phase 2, 2026-10-21)

Measured live against the real pool on this machine: fleet locals
(sparky Nemotron-3.5-Lightning, bob Qwen3.5-9B), Claude Code seats
(haiku/sonnet/fable/opus), Codex seats (luna/sol/astra). No LLM judge:
every task in `evals/outcome_tasks.jsonl` has a programmatic check
(contains/all_of/any_of/max_words/JSON equality/Python asserts).
Reproduce: start `maestro-router serve`, then `python evals/outcome.py`
(resumable; `--rerun-auto` re-asks only the routed choice after tuning).

## Outcome gate (the ship gate) — earned at n=98

98 tasks, each run through `maestro/auto` and every pool model
(686 model calls per full sweep; raw rows in
`evals/outcome-results.jsonl`). A route is adequate when the chosen
model succeeds and no cheaper tier also succeeds; severe under-route
when the chosen model fails but a stronger tier succeeds;
adequate-or-better also counts stronger-than-needed successes (waste).

First pass at n=30 measured 29/30 with a Wilson lower bound of 83.3% —
not enough to claim anything. The suite was then expanded to 98 tasks
and run twice end-to-end (the second pass is a fresh routed roll, so
the two passes also measure run-to-run variance):

| Metric | Pass 1 | Pass 2 | Combined routed |
|---|---|---|---|
| Adequate-or-better | 98/98 | 97/98 | **195/196 (99.5%)** |
| Wilson 95% CI | [96.2%, 100%] | [94.4%, 99.8%] | **[97.2%, 99.9%]** |
| Severe under-routes | 0 | 1 (1.0%) | **1/196 (0.5%)** |
| Over-route waste | 14/98 | 14/98 | 14/98 per pass |

**Ship gate met:** adequate-or-better ≥95% holds even at the combined
Wilson lower bound (97.2%), and severe under-routes are 0.5% (≤2%).
The single severe case in pass 2 was a haiku task failing a lexical
grader on a stochastic roll — variance, not a ladder defect. The two
apparent severe cases in pass 1 at n=98 were grader bugs (correct
answers failing substring checks); the graders were fixed and the
tasks re-run, which is recorded here rather than silently edited.

What is *not* claimed: strictly-cheapest routing is 84/98 (85.7%) —
14% of tasks still get a stronger (paid) model than the cheapest that
would have passed, mostly review/plan/research at difficulty 2–3.
That is the remaining tuning frontier, and it is a cost question, not
a correctness one. Local latency is also real (Qwen-9B reasoning took
30–50 s on some tasks vs ~3 s for CLI seats); routing optimizes
cost/adequacy, not latency.

## The tuning that mattered

Local models were scored by catalog priors (Nemotron G 0.16, Qwen-9B
G 0.24), which capped them at the `fast` tier, so `standard` work
(palindrome code, SQLi review, backup plan) went to Opus/Sonnet while
the locals passed the same tasks for free. Fleet-local fix, in
`~/.maestro/config.json` (machine config, not a code default — strengths
are pool-specific):

```json
"models_overrides": {
  "local:sparky-vllm/nemotron-3.5-lightning": {"category": "standard"},
  "local:bob-llama/qwen35-9b": {"category": "standard"}
}
```

After it, most routed tasks are served by the free fleet locals,
including both hard coding tasks; subscription seats are kept for
frontier asks and a few strong reviews. (At n=98 the waste count is
14/98 — see the gate section; the 20→4 figure was the n=30 pass.)

## Classifier gate (secondary)

Rules classifier, LLM-written labels (partly hand-checked):
dev split job 84.8% / category 70.5%; held-out test split job 80.2% /
category 65.0%, within-one 99.6%, under-routed 8.0%, severe 0.0%.
The largest dev confusion (summarize→debug) is label ambiguity
("analyze this slow query log", "from this stack trace…"), not a rule
gap; forcing it would cost real debug recall. CI floors were tightened
to measured-minus-margin (job 0.78, category 0.62, within-one 0.96,
under 0.10, severe 0.02) so regressions now fail CI.

## Known limits

- Suite tasks skew short; long-horizon agentic work is unmeasured.
- Local latency is real: Qwen-9B reasoning answers took 30-50 s on
  several tasks vs ~3 s for CLI seats. Routing optimizes cost/adequacy,
  not latency; a latency-sensitive caller should say so (Hermes/Pi
  phases will surface this).
- Scores for fleet locals remain priors. The learning loop (Phase 4)
  is what should replace them with measured posteriors.
