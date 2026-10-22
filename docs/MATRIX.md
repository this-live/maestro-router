# Provider matrix (2026-10-21, pavilion)

Scientific routing measurement: sandboxed, strict attribution, repeated
rolls. Rig: `evals/matrix.py`; tasks: `evals/matrix_tasks.jsonl` (33
tasks, all 11 jobs, designed to separate providers); raw rows:
`evals/matrix-results.jsonl` (660 rows = 33 tasks x 10 providers x
2 rolls). Reproduce: `maestro-router serve` in an isolated
MAESTRO_HOME, then `python evals/matrix.py --port <p> --rolls 2`.

**Sandbox.** Serve ran with `MAESTRO_HOME` in a temp dir (config.json
copied in — endpoints and overrides only, never keys; all decisions,
outcomes and learned state fresh), from a temp cwd. Subscription seats
run locked down exactly as in production serve (no tools, temp cwd,
restricted env). A row only counts for a provider when `served_by`
equals the requested model; fallback-served rows are recorded, not
hidden. Graders are deterministic (substring/JSON/python asserts plus
one sentence-count check). No LLM judge anywhere.

**Burn.** One full sweep is ~660 model calls. Claude seats are the
marginal cost; Codex seats consumed the workspace's credits mid-run
(see below).

## Provider results (strict attribution, 66 rows each)

| Provider | Success | Wilson 95% | p50 | p95 | Notes |
|---|---|---|---|---|---|
| cli:claude/fable | 59/66 = 89% | [80, 95] | 4.7s | 9.2s | best overall |
| cli:claude/opus | 57/66 = 86% | [76, 93] | 4.7s | 7.1s | |
| cli:claude/sonnet | 54/66 = 82% | [71, 89] | 4.2s | 7.2s | |
| cli:claude/haiku | 51/66 = 77% | [66, 86] | 4.0s | 6.2s | |
| local nemotron-3.5-lightning (sparky vLLM) | 45/66 = 68% | [56, 78] | 10.7s | 93.6s | free; slow tail |
| local qwen35-9b (bob llama.cpp) | 35/66 = 53% | [41, 65] | 29.2s | 137.4s | free; slow, weak on plan/research |
| cli:codex/* (3 models) | not measurable | — | — | — | workspace ran out of credits mid-run |

Per-job (locals): Nemotron measured 1.00 on ask/code/data/debug/ops,
0.83 edit, 0.67 review/summarize, 0.17 research/write, 0.00 plan.
Qwen: 1.00 ask/code/debug, 0.83 ops, 0.50 data/write, 0.33
edit/review/summarize, 0.00 plan/research.

## Harness-native providers (2026-10-21, jarvis, v3.5.0)

The providers people actually run inside Pi/Hermes, measured through
the new Pi seat (same rig, same 33 tasks, 2 rolls; raw rows
`evals/matrix-pi-results.jsonl`, manifest `evals/matrix-pi-manifest.json`).
Burn: 66 short calls per provider (330 total) against the owner's Pi
subscriptions / Kimi coding plan; zero quota, HTTP or timeout failures
— every failure below is a wrong answer.

| Provider | Success | Wilson 95% | p50 | p95 | Notes |
|---|---|---|---|---|---|
| cli:pi/kimi-coding/k3 (via Pi) | 58/66 = 88% | [78, 94] | 6.5s | 12.6s | best of this set |
| cli:pi/minimax/MiniMax-M3 (via Pi) | 57/66 = 86% | [76, 93] | 3.9s | 17.2s | |
| kimi-code-plan-cn/k3 (direct API) | 57/66 = 86% | [76, 93] | 6.8s | 21.4s | same model as Pi row; consistent |
| cli:pi/xai/grok-4.7 (via Pi) | 56/66 = 85% | [74, 92] | 2.8s | 18.0s | |
| cli:pi/meta/muse-wonder-1.0 (via Pi) | 53/66 = 80% | [69, 88] | 2.8s | 11.4s | weakest here |

Reachability (jarvis): Pi carries xai, minimax, meta, kimi-coding,
anthropic and openai-codex (all callable via the seat); Hermes
dispatches xai-oauth/kimi-direct/minimax-oauth (surfaced by `detect`,
not separately callable — Maestro reaches the same subscriptions
through Pi or direct API). Direct API keys present: only Kimi
(`KIMI_CODING_API_KEY`); xAI/MiniMax/Meta/OpenRouter have no API key on
the fleet — owner-only additions, not needed while Pi seats work.
Note the catalog prior underrates Kimi K3 (G 0.34, "fast") against its
measured 86-88%: priors are a starting point, the matrix is the
evidence, and per-model overrides remain the correction path.

## Findings that changed the product

1. **Codex seats race under parallelism.** Concurrent Codex calls
   corrupt its models cache ("EOF while parsing a value"), producing
   502s and seat cooldowns that silently reroute work. Serve now
   serializes each subscription CLI seat (one in-flight call per seat;
   locals/APIs stay parallel). Shipped in v3.4.1.
2. **Codex quota is a hard wall.** The workspace ran out of credits
   during the sweep; every later Codex attempt 502'd and fell back.
   Maestro handled it correctly (fallback + cooldown), but Codex
   quality numbers do not exist until credits are restored.
3. **Explicit-model requests fall back too.** Asking serve for a
   specific model that then fails returns another model's answer
   (recorded in `attempted`). Callers who need a guaranteed model
   must check the response `model` field.
4. **One task is grader-edge, not signal:** mx-research-2 (CAP
   theorem phrasing) failed on every provider including Opus; it is
   counted in the raw rows and excluded from nothing — it is part of
   why "all models failed" is 3-4 rows.

## Routing verdict (maestro/auto vs the measured pool)

On this discriminative suite the router is **not** at the standard
the easy suite suggested (99.5% on outcome_tasks). Configuration
passes, all judged on 66 routed rolls against the same pool rows:

| Configuration | Adequate-or-better | Wilson | Severe |
|---|---|---|---|
| Both locals `standard` | 81.8% | — | 12.1% |
| Nemotron `standard` only (standing config) | 86.4% | [76.1, 92.7] | 9.1% |
| Locals `fast` only | 87.9% | — | 6.1% |
| Per-job score overrides from matrix | 77.3% | — | 16.7% (rejected) |

Roll-to-roll variance on the standing config is real (an earlier pass
of the same configuration measured 83.3% / 10.6% severe) because the
local models are stochastic reasoners. Conclusions that survive the
variance: (a) trusting locals with standard-tier work costs roughly
6-12 points of severe under-routing; (b) restricting locals to fast
work buys ~4-6 points of severe at the price of paid seats doing work
locals pass (waste 45/66); (c) per-job score overrides do not fix it —
within a category, free locals still win on cost. The real fix is
upstream strength scoring for locals from measured outcomes (the
learning loop now records them), not more config knobs. Until then,
the standing config on this machine is Nemotron-standard-only, chosen
as the measured cost/quality balance, and no >=95% routing claim is
made on discriminative work.

## Switching and orchestration (from the same cycle)

- Token accounting bug found by the switching track: Claude seat
  usage ignored cache tokens, understating processed context by up to
  ~100x on short warm replies. Fixed in v3.4.0; all cost and
  switching math now sees cache creation + reads.
- `maestro-router delegate` (v3.4.0) is the worker primitive for the
  main-session + workers pattern: main stays warm and stateful,
  workers get capsules and return distilled results.
- A first single-session vs main+workers comparison on tiny composites
  was inconclusive (both arms correct; token comparison was invalid
  until the cache fix). The honest statement: orchestration pays when
  the main session's history is long, which this battery did not
  stress. A long-history battery is future work, not a claim.
