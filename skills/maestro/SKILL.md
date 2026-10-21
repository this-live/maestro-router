---
name: maestro
description: Route work to the right model. Use when the user selects maestro (/maestro, "use maestro"), or asks to "route this", "use the right model", "pick the best model for this", or to split a job across cheaper and stronger models. Classifies each subtask, picks a model category and seat from what the user actually has (cloud subagents, other CLIs, local and network models), and tells you how to delegate.
---

# Maestro

Maestro decides which model should do each piece of work. It is a local script:
no daemon, no network on the default path. It returns JSON that says who runs the task and how.

## Steps

1. Split the request into meaningful subtasks if it helps. Do not route trivia (one-line answers, a single read). One routing call per real unit of work.
2. For each subtask, run the router that ships in this skill's directory (`<skill-dir>` is the absolute path of the folder containing this SKILL.md; your working directory is usually the user's project, not the skill folder):

   ```
   python3 <skill-dir>/scripts/maestro route --harness <me> --current-tier <my tier> "<one-paragraph description of the subtask>"
   ```

   If `maestro-router` is on PATH (pip install), `maestro-router route ...` is equivalent.

   - `<me>` is `claude-code`, `opencode`, `gemini-cli` or `codex`. Omit `--harness` to auto-detect.
   - `<my tier>` is the cloud tier of the model you are running on, whatever the vendor: `fast` (cheap and quick: Haiku, GPT mini, Gemini Flash class), `standard` (the everyday workhorse: Sonnet, GPT, Gemini Pro class), `strong` (top reasoning models: Opus class) or `frontier` (the vendor's very best). Omit it if unsure.
   - Put enough context in the description to classify it: what, where (files), and how hard.
   - When you already know them, pass hints: `--job <job>` (one of the eleven below), `--tools yes|no` (needs repo tools or commands), `--size s|m|l|xl` (how big the answer or change is) and `--stakes low|normal|high` (what a wrong result costs). `--context-tokens N` is your conversation's size. Leave out what you are unsure of.
   - The router sorts the work into one of eleven jobs (plan, code, debug, review, ops, research, data, write, edit, summarize, ask) and picks a category: `local` (models on this machine or the user's network), `fast`, `standard`, `strong`, `frontier`, or `specialist` (a model tuned for that job).

3. Read `delegate` in the JSON and follow it exactly:

   | `delegate.kind` | What to do |
   |---|---|
   | `subagent` | Claude Code: spawn a subagent with the Agent tool and set its `model` parameter to `delegate.model` (e.g. `opus`). OpenCode / Gemini CLI: invoke the `maestro-<tier>` subagent named in `delegate.agent` (run `python3 <skill-dir>/scripts/maestro install` once to create them). |
   | `local` | A local or network model. Run `delegate.argv` as a command (it calls `maestro-router run` on the model), substituting the subtask text for `{task}` as one properly quoted argument, and use the printed reply as the subtask's output. The model has no repo tools and cannot read files: paste the relevant file contents, errors or data into the subtask text. If the subtask needs more context than you can reasonably paste (exploring the repo, many files), use the first `fallback` seat instead. |
   | `cli` | Run `delegate.argv` as a command, substituting the subtask text for `{task}` as a single, properly quoted argument (never splice it into a shell string unquoted). Use the result as the subtask's output. |
   | `model` | Your own harness runs the subtask on `delegate.model` (for a harness you named with `--harness` and gave models with `maestro-router models set <name>.<category> <model>`). Maestro starts nothing. |
   | `inline` | Do the subtask yourself; no delegation needed. |

   `delegate.argv` is a list. Pass each element as its own argument without a shell; if you must use a shell, quote every element. Maestro builds it only from its own path, your config and model names made of letters, digits and `. _ : / @ + = -`; a model server that reports any other name is ignored, so a server on your network cannot put a command into it.

   Give a delegated worker the full subtask, relevant file paths and what to return. It does not see this conversation.

4. Independent subtasks can run in parallel. Combine the results and check them before you report back.
5. After you check a delegated result, record how it went with the decision's `decision_id`: `python3 <skill-dir>/scripts/maestro outcome <decision_id> pass|partial|fail` (add `--tests P/T` when tests ran). Maestro learns from these, locally. If it failed, `escalate_to` names the seat to try next.
6. Tell the user briefly what was routed where, e.g. "Refactor: strong subagent. Docstrings: local model on spark. Tests: done inline." Use each decision's `reason` if the user asks why.

## Notes

- `confidence` under 0.5 already escalated the category by one; `fallback` lists other seats if the first fails (if a `local` run errors, use the first fallback).
- `python3 <skill-dir>/scripts/maestro route --explain ...` shows the classifier signals, the tier walk, S1's estimate of the task (difficulty, tools, stakes, tokens) and every seat's success chance, price and expected utility.
- With `decider: utility` in the config (`--decider utility` for one task), S1 and the expected-utility ranking pick the seat; the default `ladder` picks it from the job and difficulty table and logs what the utility decider would have chosen. Tool jobs stay on the ladder until the release gate sets `agentic_ready`.
- `python3 <skill-dir>/scripts/maestro doctor` checks what Maestro detected (CLIs, keys, local and network models, hardware).
- `python3 <skill-dir>/scripts/maestro models` prints which model each harness uses for each category and where it came from; `models --suggest` lists the suggested ids, `models set <harness>.<category> <model>` changes one, and `models --check` (network) verifies the ids against the live models.dev catalog. Suggestions are a dated snapshot, so if a delegated model errors as unknown, run `models --check`. After changing OpenCode or Gemini CLI models, run `install` again.
- Local and network models come from `local_endpoints` in the config and the usual localhost ports; `detect --scan` (only when the user asks) looks for more on the LAN. Only models a server has already loaded are used; Maestro never downloads, loads or starts a model (installed-but-not-loaded Ollama or LM Studio models are ignored unless `local_allow_load` is set). If the user has nothing served, `maestro-router discover --local-fit` lists models that would fit, as advice only.
- Config lives in `~/.maestro/config.json` (`python3 <skill-dir>/scripts/maestro config` prints it). If `TYPESAFE_API_KEY` is set, the Jev classifier picks the model (the task text is sent to TypeSafe); otherwise everything stays local.
