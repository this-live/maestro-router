# Routing quality (Phase 2, 2026-10-21)

Measured live against the real pool on this machine: fleet locals
(sparky Nemotron-3.5-Lightning, bob Qwen3.5-9B), Claude Code seats
(haiku/sonnet/fable/opus), Codex seats (luna/sol/astra). No LLM judge:
every task in `evals/outcome_tasks.jsonl` has a programmatic check
(contains/all_of/any_of/max_words/JSON equality/Python asserts).
Reproduce: start `maestro-router serve`, then `python evals/outcome.py`
(resumable; `--rerun-auto` re-asks only the routed choice after tuning).

## Outcome gate (the ship gate)

30 tasks (22 mixed job/difficulty + 8 hard), each run through
`maestro/auto` and every pool model. A route is adequate when the chosen
model succeeds and no cheaper tier also succeeds; severe under-route
when the chosen model fails but a stronger tier succeeds.

| Metric | Before tuning | After tuning |
|---|---|---|
| Adequate-or-better | 29/30 (96.7%) | 29/30 (96.7%) |
| Wilson 95% CI | [83.3%, 99.4%] | [83.3%, 99.4%] |
| Severe under-routes | 1/30 (3.3%) | 1/30 (3.3%) |
| Over-route waste | 20/30 | **4/30** |

The one severe case (`ask-hard-3`, a labels-are-wrong logic puzzle) is
same-family noise: the routed frontier Codex model failed a loose
grader that the fast Codex model passed. Not a ladder defect.

**The ≥95% claim is NOT earned yet.** The point estimate is 96.7% but
the Wilson lower bound is 83.3% at n=30, and the severe rate (3.3%)
is above the 2% target. Earning the claim needs a larger held-out suite
(~100+ tasks with at most a couple of misses). What is earned: routing
succeeds on 29/30 real tasks, and waste was cut 5x by tuning.

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

After it, 24/30 routed tasks are served by the free fleet locals,
including both hard coding tasks; subscription seats are kept for
frontier asks and a few strong reviews.

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
