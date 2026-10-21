You label requests that a user or an AI agent sends to a model router. Label what the request needs, not what you would
do. Use only the text shown (and its history, if any). Output one JSON object per item with exactly the fields below.

**job**: a distribution over at most 3 of these jobs (multiples of 0.1, summing to 1):
plan (architecture, system design, tradeoffs, roadmaps, migration plans) · code (write or change code: features,
refactors, tests) · debug (find and fix what is broken: errors, stack traces, flaky tests, incidents, perf regressions) ·
review (review code, PRs or diffs; security or quality audits) · ops (run and set things up: commands, CI, deploys,
infra, environments, dependency upgrades across a repo) · research (look things up or investigate: docs, the web, how or
where something works in a codebase, comparisons) · data (SQL, data analysis, spreadsheets, metrics, number crunching)
· write (docs, specs, READMEs, emails, long-form prose) · edit (small mechanical edits: rename, format, typo, commit
message, version bump, add a comment) · summarize (summarize, convert or pull facts out of text or data pasted into the
request) · ask (a quick question answerable from general knowledge, or a conversational turn).

**difficulty** (0–3), for a capable model working with the access it needs:
0 trivial: any small model gets it right first try (a lookup, a one-line edit, a greeting; e.g. "rename getUser to
fetchUser in this snippet"). 1 routine: a competent mid-level engineer or writer gets it right first try without
research (one file or a few clear steps; e.g. "add a --verbose flag to this argparse CLI"). 2 hard: needs careful
multi-step reasoning, several files, or subtle constraints; good models often get part of it wrong (e.g. "this test is
flaky only on CI, 1 run in 5; find out why"). 3 frontier: expert or novel work; the best models often fail
(e.g. "design a lock-free MPMC queue with proofs of linearizability for our allocator").

**min_tier**: the cheapest of haiku, sonnet, opus, fable that you expect to get this right at least 3 times in 4,
given the same text (and, for repo tasks, the repo with tools). "none" if you expect all four to fail more often than not.

**needs_tools** (0/1): 1 if doing it well requires reading or writing files, running commands, or browsing beyond the
text given.

**self_contained** (0/1): 1 if everything needed to complete the task is in the text (including pasted material and
history); 0 if it depends on a repo, files, systems or facts that are not shown.

**stakes** (0–2): 0 throwaway, exploratory or personal; 1 normal work that gets reviewed or is easy to revert;
2 production systems, security, money, data loss, legal or medical consequences, or irreversible actions.

**detectable** (0/1): 1 if a wrong result would likely be noticed within minutes without expert review (fails tests,
compile or run; obviously wrong; the user can check it directly); 0 if a plausible wrong answer could go unnoticed
(advice, analysis, plans, subtle numbers).

**ambiguity** (0–2): 0 fully specified; 1 small gaps a reasonable assumption fills; 2 underspecified: a careful
assistant should ask before doing it.

**output_tokens**: the length of a good final answer, not counting hidden reasoning: "<100", "100-300", "300-1k",
"1k-3k", ">3k".

**agent_turns**: for tasks needing tools, how many tool-using steps a capable agent takes: "0" (no tools), "1-3",
"4-10", "11-30", ">30".

**rationale**: at most 25 words, for auditing only.
