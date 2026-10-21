# Arena runbook

This is a step-by-step guide to running Maestro's arena on your own machine. It runs the same tasks through your
harnesses (Claude Code, Codex, Gemini CLI, OpenCode, Hermes) and your models, including models a DGX Spark or another
box serves with vLLM on your LAN. It grades and ranks the outputs and gives you:

- a leaderboard;
- the cheapest seat that is as good as the best one;
- a check of whether Maestro's router would have picked well;
- outcome rows that Maestro's decision model (S1) can learn from.

A person can follow this guide, and so can a local Claude Code session (the `arena` skill in `.claude/skills/arena/`
points here). Every command runs from the root of a clone of the repository.

**Cost warning.** Every cell is a real call:

- on your Claude, ChatGPT or Gemini plan, it spends the plan's 5-hour and weekly windows;
- on an API key, it spends money;
- on a local model, it takes GPU time on the Spark.

Always do a `--dry-run` first. Start small, and keep `--budget-usd` (default $5) low until you trust the numbers.

## 1. Prerequisites

1. Python 3.9 or newer, plus `git`. The arena uses only the standard library.
2. Clone the repository. The arena is a repo tool under `evals/`, so the pip package alone does not include it.

   ```bash
   git clone https://github.com/this-live/maestro-router && cd maestro-router
   pip install .   # optional: the `maestro-router` command, from this checkout
   ```

   Without pip, replace `maestro-router` below with `python3 skills/maestro/scripts/maestro`.
3. Tell Maestro about your model server, then check what it found:

   ```bash
   maestro-router setup --endpoint http://<spark>:8000      # e.g. http://spark.local:8000; repeat --endpoint for more boxes
   maestro-router doctor                                    # CLIs, logins, keys, local models (served vs. installed)
   ```

   The arena only uses models a server already has **loaded**. Maestro never loads or starts a model. If a model you
   want is missing, load it on the Spark yourself.
4. Log in to each harness CLI you want to compare: `claude`, `codex login`, `gemini`, `opencode auth login`, `hermes`.
   The arena never reads these logins; it only runs the CLIs.
5. Optional, so the arena can see your Claude quota and pause in time:

   ```bash
   maestro-router install --statusline
   ```

   Add the printed snippet to Claude Code's `settings.json`. Codex windows are read from Codex's own session files.
   Gemini reports no quota, so the arena cannot see it.

## 2. Check that each harness runs headless

Run one tiny command per CLI. Each should print a short answer and exit:

```bash
claude -p --safe-mode --model haiku --tools "" --output-format json "Reply with the word ok"
codex exec -m gpt-6-luna --sandbox read-only --skip-git-repo-check --json "Reply with the word ok"
gemini -m gemini-flash-latest --output-format json -p "Reply with the word ok"
opencode run --standalone -m anthropic/claude-haiku-4-5 "Reply with the word ok"
hermes chat -Q -q "Reply with the word ok"
maestro-router run local:spark/<model> "Reply with the word ok"      # a served model; `maestro-router doctor` lists the ids
```

If a CLI rejects a flag, fix that CLI's row in `HARNESSES` in `evals/arena/harness.py`. Every argv the arena uses is in
that one table, along with notes on what was verified and what was assumed. Then run
`python -m unittest discover -s tests -p test_arena.py`.

**OpenCode on the Spark.** The arena writes this OpenCode config into a scratch directory outside the work tree and
points `OPENCODE_CONFIG` at it, so you do not have to. To try the model by hand, put the same block in `opencode.json`
or `~/.config/opencode/opencode.json`:

```json
{
  "provider": {
    "spark": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "spark",
      "options": {"baseURL": "http://spark.local:8000/v1", "apiKey": "{env:SPARK_API_KEY}"},
      "models": {"openai/gpt-oss-120b": {"name": "openai/gpt-oss-120b"}}
    }
  }
}
```

Then run `opencode run --standalone -m spark/openai/gpt-oss-120b "Reply with the word ok"`. The arena passes
`--standalone` when your OpenCode lists it (2.x): without it, 2.x starts a shared background service that keeps the first
config it saw. The 1.18.x release line has no such flag and no such service. Leave out `apiKey` if vLLM runs
without `--api-key`.

**Hermes.** For a served local model, the arena gives Hermes a temporary `HERMES_HOME` whose `config.yaml` names the
model (`provider: custom`, `base_url`). The `hermes:maestro/auto` seat makes Hermes talk to Maestro's own router, so
start `maestro-router serve` first. Its cost shows as unknown, because Maestro decides what it spends.

## 3. List the seats

```bash
python3 evals/arena.py seats             # table: id, mode (agentic | chat), family, category, strength G, price
python3 evals/arena.py seats --json      # everything, including the argv notes per harness
```

Where seats come from:

| Seat | What it is |
|---|---|
| `claude-code:<model>`, `codex:<model>[@effort]`, `gemini-cli:<model>`, `opencode:<provider>/<model>` | Each CLI on your PATH, with your `maestro-router models` choices or Maestro's suggestions. These are **agentic**: they edit a repo. They can also answer chat suites, with tools off. |
| `local:<endpoint>/<model>` | Every model a server already serves. **Chat** only. |
| `opencode:<endpoint>/<model>`, `hermes:<endpoint>/<model>` | The same served models driven through an agent harness. **Agentic**. |
| `api:<provider>/<model>` | Suggested Anthropic, OpenAI and Google models when that key is set. **Chat**. |

To add, remove or pin seats, copy `evals/arena/seats.example.json` to `evals/arena/seats.json` and edit it:

- `add` takes extra seats (a Codex effort, a Spark model under a name you choose, any API model);
- `remove` drops seats by glob;
- `judges` names the default judge panel.

**Judges** rank the open-ended suite. The default panel is the strongest seat of each model family, for up to four
families: for example a GPT seat (Codex), a Gemini seat, a Claude seat and a strong Spark model. A local seat takes
its family's place when it is within 0.05 G of the strongest. If every available judge comes from one family, the
report says so in a warning. Add a judge from another family with `--judges` or `seats.json`.

## 4. Tasks

| Suite | Input | Graded by |
|---|---|---|
| `verifiable` | `--items file.jsonl`, rows `{id, prompt, checker_py, type?, difficulty_intent?}` (the alpha set's verifiable items) | the item's `check(answer)`, run in a sandboxed subprocess |
| `agentic` | `evals/arena/fixtures/<id>/` with `meta.json`, `task.md`, `repo/`, `hidden_tests/` (`--fixtures DIR` to point elsewhere) | `meta.test_cmd` after `hidden_tests/` is copied in |
| `open` | `--open-items file.jsonl`, rows `{id, prompt, job}` | the judge panel: criteria first, then blind rankings |

`--tasks N` keeps the first N tasks of each suite (`--shuffle SEED` for a seeded sample instead). `--only 'debug-*'`
filters by id.

## 5. A first run (small)

Pick 3 to 6 seats, at least one of them local, and keep the run small: 10 verifiable, 5 agentic and 5 open tasks.

```bash
SEATS='claude-code:sonnet,codex:gpt-6.1-sol,gemini-cli:gemini-flash-latest,local:spark/*,opencode:spark/*'
python3 evals/arena.py run --suite all --items V.jsonl --open-items open.jsonl --tasks 10 \
    --seats "$SEATS" --run-id first --concurrency 2 --budget-usd 5 --dry-run
```

The dry run prints the cells, an estimated cost per seat (from catalog prices) and the judge panel. It sends nothing.
When it looks right, run the same command without `--dry-run`.

`--tasks` applies to every suite. To get 10 verifiable but only 5 agentic and 5 open tasks, run the suites one at a
time under the same `--run-id`:

```bash
python3 evals/arena.py run --suite verifiable --items V.jsonl --tasks 10 --seats "$SEATS" --run-id first
python3 evals/arena.py run --suite agentic --tasks 5 --seats "$SEATS" --run-id first
python3 evals/arena.py run --suite open --open-items open.jsonl --tasks 5 --seats "$SEATS" --run-id first
python3 evals/arena.py report first --html
```

What happens in a run:

- **Agentic cells.** The fixture repo is copied to a fresh temp directory, which is then git-initialized. The harness
  runs inside that copy with the task text, then the hidden tests run. The report records:
  - pass/fail;
  - the diff (also saved under `diffs/`);
  - wall time, the CLI's exit status and timeouts;
  - usage and cost when the CLI prints them;
  - the final message;
  - any **escape**, meaning a file written next to the work tree or a change to the fixture itself. An escape fails
    the cell.

  Containment is the harness's own (Codex `workspace-write`, Claude Code `acceptEdits` with a tool allow-list,
  Gemini `--yolo` only inside the temp copy). The arena checks for escapes but is not a sandbox, so run it as a user
  that cannot harm anything important.
- **Chat cells** run each CLI in an empty temp directory with tools off.
- **The open suite.** After the answers, the lead judge writes criteria from the request alone. Each judge then ranks
  the anonymized, shuffled answers twice, the second time in reverse order.

A run can be stopped and resumed. Rerun the same command with the same `--run-id`. Finished cells are skipped, and a
failed cell is retried until it has failed `--max-attempts` times (default 2).

## 6. Reading the report

`python3 evals/arena.py report <run_id> [--html]` prints the report and writes `report.md`, plus `report.html` with
`--html` (a single file with no external requests), into `evals/arena/runs/<run_id>/`. For each suite it shows:

- **Leaderboard:** pass rate (open suite: Bradley-Terry score and accept rate) with a 95% interval, dollars per task
  and per success, and seconds per task and per success. Verifiable runs add raw-answer passes; agentic runs add diff
  size, timeouts and escapes. With 10 tasks the intervals are wide; trust them over the point estimates.
- **By job and by difficulty.**
- **Value frontier:** the best seat, and the cheapest seat whose interval still reaches the best seat's rate. That
  cheapest seat is the one to route to when you care about cost.
- **Routing replay:** what Maestro's router would have picked for each task (rules classifier plus ladder; add
  `--decider utility` for the expected-utility decider), compared with always using one seat and with hindsight (the
  cheapest seat that succeeded on each task). If the router lands close to hindsight, Maestro is routing well.
- **Judging checks (open suite):**
  - position bias (how often the first answer shown won);
  - forward-versus-reversed agreement;
  - agreement between judges;
  - length bias (positive means longer answers ranked higher);
  - a judge-family by contestant-family table with each family's self-preference: how much a family's judge favors its
    own family's answers compared with the other judges. A large positive self-preference means you should discount
    that judge, or add judges.

Claude Code reports API-equivalent dollars even on a subscription. Other CLIs are priced from tokens (or characters)
times the catalog price. Local models count as $0.

## 7. The full run

- Run more tasks with `--repeat 2`. Repeats give the bootstrap intervals something to work with and show how stable
  each seat is.
- Pace the run. One suite per sitting is plenty. `--concurrency 2` is the default; use 1 when the Spark is shared.
- Raise `--budget-usd` deliberately. The budget covers the whole run, including earlier sessions under the same
  `--run-id`, so a resume after "budget reached" needs a higher cap.
- **Quota:**
  - Cells for a subscription whose window is under `--min-headroom` (default 10% left), or that is cooling down after
    a 429, are skipped, logged as `skipped`, and redone by the next resume.
  - `--quota-wait SECONDS` waits that long instead of skipping.
  - Watch the windows in the web UI (`maestro-router serve --open`, Providers tab) or in Claude Code's status line.
  - Spread a big run over days instead of emptying a weekly window in one evening.
- **Spark contention.** OpenCode or Hermes on the Spark, a `local:` chat seat and anything else you run there share one
  GPU. vLLM batches requests, so they all slow down together and the measured seconds stop meaning much. Run local
  seats with `--concurrency 1`, don't use the Spark for other work during a run, and don't compare local wall times
  taken under different loads.

## 8. Export outcomes for S1

```bash
python3 evals/arena.py export <run_id>          # -> evals/arena/runs/<run_id>/outcomes.jsonl
```

The rows use the alpha outcome format (`docs/S1-SPEC.md` §7.8): `y` and `tout` keyed by tier (fast = haiku,
standard = sonnet, strong = opus, frontier = fable; when several seats share a tier, the Claude seat stands for it).
Each row also has every seat's own outcome in `y_seat`, with its tier, family and harness in `seat_meta`. Rows contain
no answer text. `evals/alpha_eval.load_items` reads them. Share the file, not the run directory: the run directory
holds the full answers and diffs.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `` `claude` is not on PATH `` (or another CLI) | Install it, or check `which <cli>` in the shell that runs the arena. Seats whose CLI is missing are left out of runs. |
| Claude Code: `is_error` or "not logged in" | Run `claude` once interactively. Running headless needs a finished login. |
| Claude Code: "Prompt is too long" on a one-line prompt | Your plugins, skills and MCP servers are being loaded. The arena passes `--safe-mode` (keeps the login and built-in tools, drops every customization); use it in the one-line check too. |
| Claude Code agentic cell does nothing | `--permission-prompts none` denies anything outside `acceptEdits` and the allowed tools. Check the cell's `final` message in `results.jsonl`. |
| Codex: unknown flag or model | Fix the `codex` row in `harness.py` (`codex exec --help`). Set models with `maestro-router models set codex.<category> <model>`. |
| Codex: "model is not supported when using Codex with a ChatGPT account" | Usually an old Codex CLI: newer models need a newer CLI (`npm i -g @openai/codex@latest`). Run the one-line check again. |
| Codex: no usage or cost | Older Codex versions print no `--json` events. The text is still graded and the cost is estimated. |
| Gemini: output is not JSON | Old Gemini CLI versions have no `--output-format`. The arena falls back to plain text; remove the flag from the row. |
| Gemini: `IneligibleTierError` ("no longer supported for Gemini Code Assist for individuals") | The Google login, not the CLI, is refused; upgrading does not help. Use a Gemini API key instead (`GEMINI_API_KEY` for the CLI, or `maestro-router keys set google` for `api:google/...` seats). |
| OpenCode: "provider not found" for a Spark seat | Check the seat's endpoint `base_url` (it must end in `/v1`) and that `served` is exactly what `curl <base_url>/models` lists. |
| Hermes: ignores the model | Check `hermes chat --help` on your version. If it takes `--model`, add it to the `hermes` row. |
| `local:` seat: "installed but not loaded" | Load the model on the server. Maestro will not. |
| `skipped (... cooling down ...)` | A 429 was recorded. Wait, or pass `--quota-wait 900`. |
| "budget reached" | Rerun the same `--run-id` with a higher `--budget-usd`. |
| A judge keeps failing to parse | Use a stronger judge (`--judges`). Failed judgments are retried on resume and excluded from scores. |

**Privacy.** Task text goes only to each seat's own provider (the CLI's vendor, your API provider or your own server).
Judges see answers with self-identifying phrases removed. Keys are never read, logged or written by the arena. A local
endpoint's key is referred to by its environment variable name.
