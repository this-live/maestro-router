# Verified operation (2026-10-21, v3.1.1 scrutiny)

What was actually run, on a real machine (Linux, Python 3.14 venv, project-local
install from source), not inferred from docs.

## How Maestro operates

- **Skill mode** (`route`): classifies the task (job, difficulty, needs) with a
  rules classifier (learned S1 heads in server mode), picks a seat + model from
  the discovered inventory, and returns a delegate argv. The calling agent runs
  it. No daemon, no network on the decision path (~10-80 ms).
- **Server mode** (`serve`): a local OpenAI-compatible endpoint
  (`maestro/auto|local|fast|standard|strong|frontier`). It plans candidates,
  tries at most 3, falls back on 401/403/404/408/429/5xx/timeout-before-first-byte,
  keeps per-conversation seat affinity, logs every decision to `~/.maestro/maestro.db`,
  and learns from outcome signals (`feedback`, test results, implicit signals).
- **Discovery** (`detect`/`discover`): finds harness CLIs and their login state,
  API keys by provider (names only), local/network model servers (Ollama, vLLM,
  SGLang, llama.cpp, LM Studio ports or configured endpoints), scores models
  against the bundled models.dev catalog (as of 2026-09-30), and writes
  `~/.maestro/models.json`.

## Verified matrix

| Surface | Result |
|---|---|
| `route` (debug/write/plan prompts) | Correct job/category ladder; delegate argv + fallbacks returned |
| `detect` | Accurate on the test machine (claude/codex logged in, gemini installed-not-logged-in, no local servers) |
| `serve` → Claude Code seat (haiku) | 200 in ~4.5 s (after the 3.1.1 env fix; hung before it) |
| `serve` → Codex seat (`maestro/fast`) | 200, `x-maestro-decision` header present, content correct |
| `serve` → local OpenAI endpoint (`maestro/local`) | 200 against a stub server registered via `local_endpoints`; discovery + routing + relay all worked |
| Gemini CLI seat | Fail-closed design confirmed; without login it returns a clean 502 naming the missing `GEMINI_API_KEY`, never hangs |
| OpenCode | Not a serve seat; works as a harness: `maestro-router install` wrote `maestro-{fast,standard,strong,frontier}` subagents for Claude Code, OpenCode, and Gemini CLI config dirs |
| API providers (225 in registry) | Unit-tested with fake HTTP servers; not live-tested (no keys on the test machine) |
| Real Ollama/vLLM server | Not installed on the test machine; the identical OpenAI wire path is proven by the stub-endpoint run and the fake-server unit tests |

## Test evidence

- `python -m unittest discover -s tests`: 860 tests. One known environment flake:
  `test_decide.Decisions.test_rank_is_fast_on_40_seats` asserts a 2 ms p99 for the
  ranker and fails on a loaded/shared machine (fails identically on the clean
  tree); it passes on a quiet machine.
- CI eval gates (`evals/run.py --classifier rules --split test` at CI floors):
  job accuracy 80.2%, category accuracy 65.0%, under-routed 8.0%, severe 0.0% — pass.

## Defect found and fixed in 3.1.1

`cli_env` stripped version-manager variables (`MISE_*`, `ASDF_*`, …), so any CLI
installed through a shim/wrapper hung inside `serve` until the 300 s timeout.
See CHANGELOG 3.1.1.
