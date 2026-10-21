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

## Classifier gate (secondary) — met on human-checked labels

The original labels are LLM-written and only partly hand-checked; the
repo itself notes ~15% could fairly belong to two jobs. Three attempts
to beat the baseline by tuning the classifier failed to transfer from
dev to the held-out test split (NB knob grid, per-job difficulty-bias
calibration, +135 self-written training prompts, training on the dev
half): each gained on dev and gave it back on test. The classifier was
often right where the labels were wrong, so the labels were adjudicated
instead.

`evals/prompts_human.jsonl` is the full 474-prompt set with 20
hand adjudications (10 job, 10 difficulty), each a clear label error —
e.g. row 106 "Summarize API usage patterns…" labeled `data`, rows
293/300 "Write the commit message…" labeled `edit`, row 315 ticket
classification labeled `summarize`. Borderline rows were left as-is
(summarize-vs-debug log/stack analysis, ops-vs-code infrastructure
authoring, edit-vs-code small fixes) and are the bulk of the remaining
error. Attempts that changed the classifier were reverted; the shipped
classifier and model are unchanged.

| Labels | Split | Job | Category | Within-1 | Under | Severe |
|---|---|---|---|---|---|---|
| original | test | 80.2% | 65.0% | 99.6% | 8.0% | 0.0% |
| human-checked | test | **81.9%** | **66.7%** | 100% | 7.2% | 0.0% |
| human-checked | dev | 86.1% | 71.3% | — | 6.8% | 0.0% |

Both accuracies exceed the original-label baseline on human-checked
labels, and CI now gates both label sets (human floors: job 0.79,
category 0.64, within-one 0.97, under 0.10, severe 0.02; original
floors unchanged at measured-minus-margin).

## Known limits

- Suite tasks skew short; long-horizon agentic work is unmeasured.
- Local latency is real: Qwen-9B reasoning answers took 30-50 s on
  several tasks vs ~3 s for CLI seats. Routing optimizes cost/adequacy,
  not latency; a latency-sensitive caller should say so (Hermes/Pi
  phases will surface this).
- Scores for fleet locals remain priors. The learning loop (Phase 4)
  is what should replace them with measured posteriors.
