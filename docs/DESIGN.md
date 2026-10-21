# Maestro v2.1 — Design

> Maestro routes each task to the right model you already have. It is a skill
> plus a stdlib-only, zero-dependency CLI package. It has no daemon, no database and
> no required network access.

## Goals

1. **One job: model routing.** Maestro takes a task description and returns
   which model and seat should do it, and why.
2. **Lean.** The whole router is one Python file using only the stdlib
   (Python >= 3.9). It has no third-party dependencies.
3. **Fast.** Rule-based routing adds < 5 ms of in-process work and < 60 ms
   wall-clock including interpreter start. Detection is cached.
4. **Aware of what the user has.** It detects harness CLIs, subscriptions,
   API keys, local and network model servers (Ollama, vLLM, SGLang, LM Studio,
   llama.cpp) and hardware. It never routes to something the user can't run.
   **Maestro never downloads, loads, starts or installs a model.** It gauges
   the hardware, looks at what is already served (loaded) locally or on the
   network, and uses exactly those. Only when nothing is served does it suggest
   models that could run, as advice.
5. **Installs into any harness** the same way other skill libraries do:
   `npx skills add this-live/maestro-router`, or the Claude Code plugin
   marketplace.
6. **Measured.** An eval set in the repo scores every classifier, so claims
   about routing quality are backed by numbers.

Non-goals for v2.0:
- A proxy or gateway. That is Phase 4, as an optional `maestro[proxy]` extra
  built on a LiteLLM pre-call hook.
- Learned routers, consensus/quorum, fleets, dashboards and billing.

## Architecture

```
task text ──► classify ──► (job, difficulty 0-3, confidence, needs)
                 │            rules (offline) | jev (picks the model directly, used when TYPESAFE_API_KEY is set)
                 ▼
            policy ladder  ──► ladder cell: local | fast | standard | strong | frontier
                 ▼
     inventory (cached) ──► specialist / local model / cloud seat ──► pick ──► decision JSON
                            (category: local | fast | standard | strong | frontier | specialist)
```

### 1. Classifier (`classify`)

The output is `job`, `difficulty`, `confidence` and `needs`:
- `job` is one of eleven plain verbs (v2.1; they replace the seven v2.0 task
  classes):

  | job | meaning |
  |---|---|
  | `plan` | architecture, system design, tradeoffs, roadmaps, migration plans |
  | `code` | write or change code: features, refactors, tests |
  | `debug` | find and fix what's broken: errors, stack traces, flaky tests, incidents, perf regressions |
  | `review` | review code, PRs or diffs; security or quality audits of existing code |
  | `ops` | run and set things up: commands, CI, deploys, infra, environments, dependency upgrades across a repo |
  | `research` | look things up or investigate: docs, the web, "where/how does X work in our codebase", comparisons |
  | `data` | SQL, data analysis, spreadsheets, metrics, number crunching |
  | `write` | docs, specs, READMEs, emails, long-form prose |
  | `edit` | small mechanical edits: rename, format, typo, commit message, version bump, add a comment |
  | `summarize` | summarize, convert or pull facts out of text or data pasted into the prompt |
  | `ask` | quick question answerable from general knowledge; conversational turns |
- `difficulty` is 0 (trivial), 1 (routine), 2 (hard) or 3 (frontier).
- `confidence` is a number from 0 to 1.
- `needs` is a set of `{tools, vision, long_context}`.

There are two backends, selected by `--classifier auto|rules|jev`,
`MAESTRO_CLASSIFIER`, or `"classifier"` in the config. The default is
**`auto`**: Jev when `TYPESAFE_API_KEY` is set, otherwise rules. Setting the
key is the opt-in.

**`rules`** is offline and deterministic. It has two layers that ship together:

1. *Heuristics* (`classify_heuristic`): keyword and structure scoring salvaged
   from v1's `maestro/intel/taskclass.py` and `maestro/routing/reasoning.py`,
   with author-specific keywords stripped out. Only the **instruction** is
   scored: text inside ``` fences is pasted material, not intent, so a code
   block never decides the class and a long pasted file does not inflate
   difficulty. Signals: the leading verb ("summarize", "set up", "investigate",
   "rename"), small-edit targets (docstring, typo, version, license header),
   pasted-material plus extraction verbs, question shape, infra words, and the
   v1 reasoning-demand, scope and hard-topic markers for difficulty.
2. *Learned model* (`classify_model`, weights in `maestro_model.json`): two
   multinomial Naive Bayes classifiers (job, difficulty) over lowercased word
   unigrams and bigrams of the instruction plus structural flags (fence
   present, length bucket, the heuristic signals above). Log-scores are
   tempered (`MODEL_T` for the job, `MODEL_TD` for difficulty), the heuristic
   keyword scores are added to the job scores (`MODEL_CLASS_W`), and the jobs
   that need repo tools get a bonus (`MODEL_TOOLS_B`), so when the job is
   uncertain between a tools job and a chat-only one the tools job wins (a
   chat-only pick could send repo work to a tools-less local model; the
   reverse only costs a missed local saving). Difficulty is scored per job:
   `log P(difficulty | job)` from the training counts (`"jd"` in the JSON)
   plus the word evidence plus a bonus (`MODEL_DIFF_W`) for the heuristic
   difficulty. The joint (job, difficulty) is chosen to minimise *expected
   tier error* under the default ladder, where routing one tier too weak
   costs 1.5x one tier too strong (a `local` ladder cell counts as `fast`
   for this risk). `evals/train.py` rebuilds the weights deterministically from
   `evals/train.jsonl`, a separate corpus (it never contains eval-set prompts),
   and records the job list it was trained on in the JSON (`"jobs"`). If the
   file is missing, unreadable, or was trained on a different job list (the
   v2.0 file has no list), it is ignored and the heuristics alone are used;
   the decision's `--explain` signals say `model_ignored` and why. A model
   without `"jd"` still works (difficulty is then independent of the job).

It is also the offline fallback for Jev.

**`jev`** uses TypeSafe's Jev System One model to **choose the model
directly**, with a confidence, and to score the task's complexity.

1. **Candidates are built before the call.** From the inventory and the
   harness, Maestro lists the *eligible, executable* seat+model pairs, after
   hard filters: `needs` (tools, vision, long context; taken from the rules
   signals), harness executability, `max_tier`, and local-model eligibility
   (see §4: jobs that need no repo tools, `prefer_local`; a specialist only for
   its own jobs). Each candidate gets a stable name (`claude-code:sonnet`,
   `local:spark/gpt-oss:120b`, `codex:default`) and a short description
   generated from category, seat, model and cost, for example
   `"strong tier, Claude Code (opus) subagent, subscription (flat cost), full tools/repo access"`
   (a Codex seat reads `"strong tier, Codex (gpt-6.1-sol, effort high) headless CLI, ..."`). When one model serves two
   categories, the second candidate's name gets an `@<category>` suffix (`codex:gpt-6.1-sol@strong`).
   A local model reads `"local model on spark (120B), free and private, serves up
   to standard-level work, no repo tools"` (a specialist adds `", tuned for code,
   debug, review"`; a local model run as a native harness subagent says it has
   repo tools). `"model_notes": {"claude-code:opus": "..."}` in the config
   overrides any description.
2. **One request:** `POST https://api.typesafe.ai/v1/systemone` with a bearer
   key (verified against `typesafe-sdk` 0.7.2, whose models are generated from
   TypeSafe's OpenAPI spec):
   ```json
   {"model": "jev-latest",
    "state": {"task": "<task text, truncated to 4000 chars>", "signals": {...}},
    "questions": {
      "model":      {"type": "choice", "instructions": "Choose the model that should do this task. Prefer the cheapest/fastest model that will do it well; ...", "criteria": {"claude-code:sonnet": "standard tier, ...", ...}},
      "reasoning":  {"type": "score", "instructions": "...", "criteria": ["...", "...", "...", "..."]},
      "scope":      {"type": "score", ...},
      "ambiguity":  {"type": "score", ...},
      "risk":       {"type": "score", ...},
      "job":        {"type": "choice", "instructions": "Which job is the task in state.task?", "criteria": {"plan": "...", ...}}}}
   ```
   The `model` instruction changes with the configured `"objective"`:
   `balanced` (default), `cost` or `quality`. `job` is asked for
   explainability and evals; its criteria are the descriptions in the table
   above.
3. **Response:** `answers.model = {type, choice, confidence, probabilities}`,
   `answers.<dimension> = {type, score (float 0-3), confidence, legend, probabilities}`
   and `answers.job = {type, choice, confidence, probabilities}`.
4. **Decision.** `complexity = round(max(reasoning, scope) * 0.6 + mean(ambiguity, risk) * 0.4)`
   clamped to 0-3 (one function, `complexity()`).
   - If `answers.model.confidence >= jev_min_confidence` (default 0.55), take
     Jev's model.
   - Otherwise take the **stronger** of Jev's pick and the ladder tier for
     (Jev's `job`, complexity), choosing the preferred seat at that cell.
   - The decision records `jev: {choice, confidence, probabilities, complexity: {...}}`
     and `--explain` lists the candidates.
5. **One candidate means no call.** If only one eligible candidate exists,
   there is nothing to decide, so the Jev call is skipped.
6. **Failure handling.** The timeout is 2.5 s by default. **Any** failure
   (network, non-200, bad JSON, unknown choice) falls back to `rules`, and the
   decision's `reason` says why.
7. **Cache.** Results are cached in `~/.maestro/cache.json`, keyed by sha256 of
   the task text, the sorted candidate names and the objective. The cache is
   an LRU of 512 entries (file format version 2), so repeated routing is instant.
8. **Privacy.** With a key set, the truncated task text, a few structural
   signals (length, whether it has code or file paths) and the candidate
   names and descriptions are sent to `api.typesafe.ai`. Nothing is sent with
   `--classifier rules` or without a key. This is documented in the README.

### 2. Policy ladder and model categories

There are six **categories**. They are provider-neutral: a category names a
kind of model, and each seat maps it to whatever model that seat has.

| category | what it is | examples |
|---|---|---|
| `local` | Models you run yourself, on this machine or on your network. Free and private. | A 120B-class model on a DGX Spark served by Ollama or vLLM; an 8B model on a laptop; SGLang or llama.cpp on the LAN or over Tailscale |
| `fast` | Cheap, quick cloud models | Claude Haiku, GPT mini-class, Gemini Flash-class |
| `standard` | The everyday workhorse | Claude Sonnet, GPT-class mid-tier, Gemini Pro-class |
| `strong` | Hard reasoning | Claude Opus, the top reasoning models of other vendors |
| `frontier` | The very best | Claude Fable, each vendor's top model |
| `specialist` | A model tuned for particular jobs, used for those jobs when available | A coder model, a SQL or math model |

The cloud ladder order is `fast < standard < strong < frontier`. `local` and
`specialist` are not rungs of it: a **ladder cell** holds `local` or one of the
four cloud tiers, and a decision's `category` is `specialist` or `local` when
such a model was chosen, otherwise the cloud tier.

The policy maps `(job, difficulty)` to a cell. It is overridable in
`~/.maestro/config.json` under `"ladder"` (cells may be `local`).

| job       | d0       | d1       | d2       | d3       |
|-----------|----------|----------|----------|----------|
| ask       | local    | fast     | standard | standard |
| summarize | local    | fast     | standard | standard |
| edit      | local    | fast     | standard | standard |
| write     | fast     | standard | standard | strong   |
| data      | fast     | standard | standard | strong   |
| research  | fast     | standard | standard | strong   |
| code      | fast     | standard | standard | strong   |
| ops       | fast     | standard | standard | strong   |
| review    | standard | standard | standard | strong   |
| debug     | standard | standard | standard | strong   |
| plan      | standard | standard | strong   | frontier |

The table was retuned on measured outcomes (deviation 133): hard work (d2) goes to `standard`
(planning keeps `strong`), very hard work (d3) to `strong` (`frontier` only for planning).

A `local` cell means "a local model if one is eligible, otherwise `fast`".

- **Escalate on uncertainty.** With the rules classifier, if
  `confidence < 0.5` (configurable), go up one cell (`local` goes to `fast`).
  Jev uses the rule in §1 instead.
- **Cap.** `max_tier` defaults to `frontier` and is configurable, for
  example to `strong` to avoid frontier pricing. It caps cloud models; it does
  not apply to local models.
- **Fallback.** If no cloud seat exists at the target tier, walk *up* the
  tiers, then *down*. The decision records the walk.

**Local models: strength and preference.** Every local model has a `serves`
rating, the strongest cell it can stand in for:
- Default from parameter count: under 10B is `local` (only the `local` cells),
  10B up to 40B is `fast`, 40B and up is `standard`. Never `strong` or
  `frontier` by default. Unknown size is `local`.
- Parameter count comes from Ollama's `details.parameter_size`, otherwise from
  the name (`70b`, `120b`, `8x7b` = 56, MoE `30b-a3b` = 30, the total).
- `"local_models": {"<name>": {"serves": "strong"}}` overrides it (the key is
  the model name, `endpoint/name` or the full id). Ratings are computed at
  route time from the inventory's sizes, so a config change needs no re-detect.
- `"models": {"local": {"standard": "<id>", "local": "<id>"}}` pins a model to a
  category and wins over both. A model pinned under `standard` serves up to
  `standard` (the highest cell it is pinned under). A model pinned under a cell
  is also the first choice for that cell, ahead of the smallest-model rule
  below. Unpinned models keep their rating. A pin may name the model by full
  id, `endpoint/name` or bare name; one that matches no detected model is
  ignored in routing and reported by `maestro models`.

With `prefer_local` true (the default) a local model is chosen when its `serves`
is at least the cell **and** the job is executable by it. A plain local model is
chat only (it is reached with `maestro run`, no repo tools), so it can do only
the jobs that need none: `ask`, `summarize`, `write`, `plan` and `data`, and only
when `needs` has no `tools`, `vision` or `long_context`. The tool-needing jobs
(`code`, `debug`, `review`, `ops`, `research`, `edit`) can use a local model
only when the current harness runs it natively as a subagent, for example
OpenCode with a `models` entry that points at it. Among eligible models
the smallest that serves the cell wins, then a model on this machine over one
on the network, then lower probe latency.

**Specialists.** A model is a specialist for some jobs if config
`"specialists": {"<model-name>": ["code", "debug"]}` (or a per-model
`specialist_for` in `local_models`) names it, or if a local or network model's
name matches: `code` or `devstral` (also `coder`, `starcoder`, `codellama`)
gives `[code, debug, review]`, and `math` or `sql` gives `[data]`. A specialist
is rated like any model with a floor of `fast`, and is used only for its jobs.
When the cell is at most its `serves` and the job is in its list, it wins over
a generic local or cloud model at that cell, and the decision reports
`category: "specialist"`. The tool rules above still apply. A cloud model
named in `specialists` (say, a Codex model tuned for code) is handled the same
way.

### 3. Inventory (`detect`)

The inventory is written to `~/.maestro/inventory.json`, or
`$MAESTRO_HOME/inventory.json`. It is auto-refreshed when missing, older than
24 h, written by an older Maestro, or when the configured endpoint list
changed. Detection takes < 2 s worst case; `route` on a fresh inventory does no
I/O beyond reading it.

It detects:
- **Harness CLIs** via `shutil.which`: `claude`, `codex`, `gemini`,
  `opencode`, `cursor-agent`, `aider`.
- **Subscriptions.** Maestro checks only whether credential files exist and
  never reads them. Examples are `~/.claude/.credentials.json` or
  `~/.claude.json`, and `~/.codex/auth.json`. A missing file is not proof of
  no login, so a CLI present means the seat is `available: "likely"`.
- **API keys.** It checks only whether the variable is present and never
  reads the value. The variables are `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
  `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`,
  `DEEPSEEK_API_KEY`, `XAI_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`,
  `MOONSHOT_API_KEY` and `TYPESAFE_API_KEY`. The last one is for the
  classifier, not a seat.
- **Local and network models.** Endpoints are probed in parallel threads
  (`concurrent.futures`) with short timeouts (300 ms for the built-in ports, 1 s
  for endpoints you configured). In order (the first entry for a URL wins):
  1. config `local_endpoints`: URLs, or `{"name": "spark", "url": "http://spark.local:11434", "kind": "ollama"|"openai", "api_key_env": "SPARK_KEY"}`.
     `kind` is auto-detected when omitted (`/api/tags`, then `/v1/models`).
  2. `OLLAMA_HOST`, if set.
  3. The built-in localhost ports: Ollama `:11434` (`GET /api/tags`), and, over
     the OpenAI-compatible `GET /v1/models`, LM Studio `:1234`, vLLM `:8000`,
     SGLang `:30000`, llama.cpp `:8080`. `MAESTRO_NO_BUILTIN_PORTS=1` skips these
     (configured endpoints and `OLLAMA_HOST` are still probed); the test suite sets it.

  Local probes never go through an HTTP proxy. An endpoint's `api_key_env` is
  the *name* of an environment variable; its value is sent as a Bearer token
  when set, and never stored or printed.
  **Installed is not served.** Ollama lists every downloaded model in
  `/api/tags` but loads one on the first chat request (which can exhaust
  memory), and LM Studio JIT-loads whatever `/v1/models` lists. So the probe
  also reads the loaded state, with GET requests only: Ollama `GET /api/ps`
  (an unanswered call counts as nothing loaded), LM Studio `GET /api/v0/models`
  (`state` "loaded" / "not-loaded", used when the answer carries a `state`).
  OpenAI-compatible servers that do not say (vLLM, SGLang, llama.cpp) serve what
  `/v1/models` lists, so all of it counts as loaded. Each model gets
  `loaded: true|false`. A model with `loaded: false` is listed but not
  routable: skill-mode `route` never picks it, the proxy skips it with the
  reason "installed but not loaded (Maestro never loads models)", `maestro run`
  refuses it unless `--allow-load`, and `discover` records `routable: false`.
  The config key `local_allow_load` (default `false`) lifts this: installed
  models are then routable and the first request may make the server load one
  (the memory risk is the user's). An inventory from before the field counts
  as loaded.
  Each model is recorded as `{id, name, endpoint_name, url, kind, params_b,
  serves, specialist_for, on_this_machine, latency_ms, loaded}`. The canonical id is
  `local:<endpoint_name>/<model>`, for example `local:spark/gpt-oss:120b`. The
  endpoint name defaults to the hostname (`localhost` becomes `this-machine`).
- **`maestro detect --scan`** is opt-in and never automatic. It finds the
  machine's own /24 IPv4 subnet (a UDP-connect to pick the outgoing interface;
  no packet is sent) and probes ports 11434 (Ollama) and 8000 (OpenAI-compatible)
  with a 250 ms timeout on 64 threads, stopping after 5 s. It prints what it
  found; `--save` adds the endpoints to the config's `local_endpoints`. Without
  `--save` the found models are shown but not kept. `MAESTRO_NO_SCAN` refuses.
- **Hardware.** Salvaged from maestro-local `swarm/hardware.py`: RAM, the
  `nvidia-smi` VRAM sum, and Apple Silicon unified memory. It is reported
  for information and used to warn when a model on this machine is larger
  than the hardware fits (models on other machines are not checked).
- **Optional catalog enrichment.** `maestro detect --catalog` fetches
  `https://models.dev/api.json` (or `$MAESTRO_CATALOG_URL`) with a 3 s timeout
  and caches it (`maestro models --check` uses the same cache). For
  providers whose API key is present (any vendor, not only Claude), it adds
  the models, placed in categories by output price per 1M tokens: >= $30 is
  frontier, >= $15 strong, >= $6 standard, otherwise fast. These are
  informational because API-key models need the proxy (Phase 4) or an
  installed harness to execute. They never block anything.

### 4. Seats and delegation

A seat is something that can execute a task in a given category. The decision
tells the calling agent exactly how to delegate:

| seat kind | when | `delegate` field |
|---|---|---|
| `subagent` | The current harness supports per-subagent models | `{"kind":"subagent","harness":"claude-code","model":"sonnet"}` |
| `inline` | The chosen tier equals `--current-tier`, or no better seat exists | `{"kind":"inline"}`, meaning do it yourself without delegating |
| `cli` | Another installed harness CLI, run headless | `{"kind":"cli","argv":["codex","exec","-m","<model>","{task}"]}` |
| `local` | A local or network model, for jobs that need no repo tools (see §2) | `{"kind":"local","model":"local:spark/gpt-oss:120b","argv":["python3","<abs path to the maestro package dir>","run","local:spark/gpt-oss:120b","{task}"]}` |

The `local` argv uses `sys.executable` and the absolute path of the running
package directory (`python3 <dir> run ...` works because the package has a `__main__.py`). The calling agent runs it with the task as one quoted argument and
uses the printed reply. The old v2.0 `ollama` kind is gone.

**`maestro run <model-id> "<task>"`** (or the task on stdin) sends one chat
completion with stdlib `urllib`: Ollama `POST /api/chat` with `stream: false`,
OpenAI-compatible `POST /v1/chat/completions`. It prints the reply text and
takes an optional `--system` and `--timeout` (config `run_timeout_s`, default
300 s). It never prints headers or environment values. Before sending, it asks
the server which models are loaded (GET only, live, not from the cached
inventory) and refuses one that is installed but not loaded, because the chat
request would make the server load it; `--allow-load` (or the config
`local_allow_load`) sends it anyway.

**Models per harness and category.** Each harness has a model for every cloud
category, and `local` holds pinned local models. A cell is a model id, or
`{"model": id, "effort": "low|medium|high|xhigh|max"}`. It resolves in this
order (`model_source` in the decision):
1. the user's config, `"models": {"<harness>": {"<category>": <cell>}}`
   (`config`). The deprecated `"harness_models"` is read when `models` is
   empty; a bare string there means the `strong` cell;
2. the suggestion in `SUGGESTED_MODELS` (`suggested`);
3. for `strong` only, the CLI's own default model, i.e. no `-m` (`builtin`).
   Unreachable for the four harnesses below, since all have suggestions.

`SUGGESTED_MODELS` is a snapshot of the models.dev catalog, dated
`SUGGESTED_AS_OF` (2026-09-30):

| harness | fast | standard | strong | frontier |
|---|---|---|---|---|
| claude-code | haiku | sonnet | opus | fable |
| codex | gpt-6-luna | gpt-6.1-sol | gpt-6.1-sol (effort high) | gpt-6-astra |
| gemini-cli | gemini-flash-lite-latest | gemini-flash-latest | gemini-3.1-pro-preview | gemini-3.1-pro-preview |
| opencode | per provider, below | | | |

OpenCode uses `provider/model` ids. The provider is the first of anthropic,
openai, google whose API key variable is present in the inventory (`ANTHROPIC_API_KEY`;
`OPENAI_API_KEY`; `GEMINI_API_KEY` or `GOOGLE_API_KEY`), else anthropic:
anthropic `claude-haiku-4-5`, `claude-sonnet-5-5`, `claude-opus-5-5`,
`claude-fable-5-1`; openai `gpt-6-luna`, `gpt-6.1-sol`, `gpt-6.1-sol`,
`gpt-6-astra`; google the Gemini row above, prefixed `google/`.

Claude Code's names are its own aliases and track the latest release. The
others are dated ids, so `maestro models --check` compares every configured or
suggested Codex, Gemini CLI and OpenCode id with the live catalog (3 s
timeout, or the `detect --catalog` cache): does it exist, is it deprecated,
does a newer model of the same `family` exist (by `release_date`; only
non-deprecated tool-calling models with the same name prefix count, and
`*-latest` aliases are exempt because they track the newest). It reads only.

Seats built from the map:
- **claude-code, opencode, gemini-cli** as the current harness: a `subagent`
  seat per category (OpenCode and Gemini CLI also carry
  `"agent": "maestro-<category>"`, written by `maestro install` with the same
  resolved models, suggestions included). A model that points at a detected local model
  (`ollama/gpt-oss:120b`, an id, or a bare name) makes that seat a *local* model
  that has the harness's tools.
- **codex** has no per-subagent model, so it is a `cli` seat
  (`codex exec -m <model> [-c model_reasoning_effort=<effort>] {task}`),
  including when Codex is the current harness (`inline` when the category
  equals `--current-tier`). Effort is passed only for Codex; elsewhere it is
  ignored and `maestro models` says so.
- **Other installed CLIs** get the full map as `cli` seats, ranked after the
  current harness.
- The seat is available if its CLI is present (Claude Code also if
  `CLAUDECODE=1`).

**Headless CLI argv templates:**
- `claude -p --model <alias> {task}`
- `codex exec [-m <model>] {task}`
- `gemini [-m <model>] -p {task}`
- `opencode run [-m <provider/model>] {task}`

**Seat selection**, in order:
1. A `job_seats` pin (`{"research": "gemini"}`), if that seat can serve it.
2. A specialist for the job whose `serves` reaches the cell (§2).
3. A local model that serves the cell (§2), when `prefer_local`.
4. `inline`, when the cell's tier equals `--current-tier`.
5. The cloud walk: a native subagent in the current harness first (it keeps
   tools, repo context and permissions), then other CLI seats;
   `"prefer": ["codex"]` reorders these.
6. As a last resort, any local model that can do the job.
7. Otherwise `inline` with tier advice.

**Harness detection:** the `--harness` flag wins. After that it checks
`CLAUDECODE=1` for claude-code, then `OPENCODE`/`OPENCODE_*` env for
opencode, `GEMINI_CLI` for gemini-cli, and `CODEX_*` for codex. Otherwise it
is `unknown`, which gives CLI and local seats only, with tier advice.

### 5. Decision JSON (`maestro route`)

```json
{"job":"debug","difficulty":2,"confidence":0.78,"needs":["tools"],
 "classifier":"rules","category":"strong","harness":"claude-code",
 "seat":"claude-code","model":"opus","model_source":"suggested",
 "delegate":{"kind":"subagent","harness":"claude-code","model":"opus"},
 "fallback":[{"seat":"codex","model":null,"category":"strong"}],
 "reason":"debug/d2 → strong; claude-code subagent available",
 "latency_ms":1.4}
```

`category` is one of the six categories: the model that was chosen (`local`
or `specialist` for those models, otherwise its cloud tier). `model_source` says
where the model id came from: `config`, `suggested` or `builtin` for a cloud
model, `config` (pinned) or `auto` for a local one, `null` for `inline`.

With Jev the decision also carries `"jev": {"choice": "claude-code:opus", "confidence": 0.86, "probabilities": {...}, "complexity": {"reasoning": 2.1, "scope": 1.2, "ambiguity": 0.8, "risk": 1.0, "value": 2}}`.

`maestro route --explain` adds the classifier signals, the ladder cell, the
candidate names and the tier walk.
`--format text` prints one line for humans.

### 6. CLI surface

```
maestro route "<task>" [--harness H] [--current-tier T] [--classifier auto|rules|jev] [--json|--format text] [--explain]
maestro route -            (reads the task from stdin)
maestro detect [--catalog] [--scan [--save]] [--json]
maestro run <model-id> ["<task>"] [--system S] [--timeout SECONDS] [--allow-load]   (task from stdin if omitted)
maestro models [--suggest | --save | --check] [--json]
maestro models set <harness>.<category> <model> [--effort E]
maestro models unset <harness>.<category>
maestro install [--harness claude-code|opencode|gemini-cli|all] [--dry-run]
maestro doctor [--jev]     (inventory summary; with --jev, a live Jev call printing latency and answers)
maestro config             (prints the effective config and path)
```

`maestro models` prints the effective map for every harness in the inventory
(category, model, source, whether it is available, e.g. not above `max_tier`),
then for `local`, then the harness Maestro would detect right now.
`--suggest` prints the suggestion tables (OpenCode per provider, with the
picked one marked) and the as-of date; `--save` writes the suggestions for
every unset cell into `config.json` under `models`; `set` and `unset` edit one
cell (`<harness>.<category>`, validated; `--effort` only with a harness);
`--check` is described above; `--json` gives machine output. All edits keep
the other keys, write indent 2, move a deprecated `harness_models` into
`models`, and refuse a config that is not valid JSON. `local` pins that do not
match a detected model, and a `harness_models` key, produce warnings.

`maestro install` writes native tier subagents so harnesses without a
per-call model parameter can still delegate by tier:
- **Claude Code:** `~/.claude/agents/maestro-{fast,standard,strong,frontier}.md`,
  with `model:` frontmatter from the resolved map (haiku, sonnet, opus and fable
  by default). This is optional, since Claude Code's Agent tool takes `model`
  directly.
- **OpenCode:** `~/.config/opencode/agents/maestro-*.md`, with the resolved
  OpenCode models.
- **Gemini CLI:** `~/.gemini/agents/maestro-*.md`, with the resolved Gemini models.

It never overwrites a file that lacks the `maestro-generated` marker.

### 7. Skill

`skills/maestro/SKILL.md` follows the Agent Skills standard, with
frontmatter `name` and `description` and the script at
`skills/maestro/scripts/maestro`. It triggers when the user selects
Maestro (`/maestro`, "use maestro", "route this", "use the right model").
The instructions:
1. Split the work into subtasks when useful.
2. For each subtask, run
   `python3 <skill_dir>/scripts/maestro route --harness <me> --current-tier <my tier> "<one-paragraph subtask>"`.
3. Follow `delegate` exactly. `subagent` means the harness's subagent
   mechanism with that model. `cli` and `local` mean run the argv with
   `{task}` substituted as one quoted argument. `inline` means do it yourself.
4. Don't route trivia. One routing call per meaningful unit of work.
5. Report the routing choices briefly.

### 8. Packaging

- `skills/maestro/`: the skill and the `maestro` package (`scripts/maestro/`). This is
  all that `npx skills add` copies.
- `.claude-plugin/plugin.json` and `.claude-plugin/marketplace.json`: the
  Claude Code plugin marketplace. Install is
  `claude plugin marketplace add this-live/maestro-router`, then
  `claude plugin install maestro@maestro`.
- `pyproject.toml`: `pip install` / `uvx maestro-router` exposes a `maestro`
  console script from the same file. It has no dependencies.
- `evals/prompts.jsonl` and `evals/run.py`: the labeled set and the scorer
  (job accuracy, difficulty MAE, category accuracy, p50/p95 latency, per
  classifier).
- `evals/arena/fixtures/`: 30 agentic fixtures for the model/harness arena (a small
  repo, the user's request, hidden grading tests and a reference solution each);
  `evals/arena/validate_fixtures.py` checks that each fails as shipped and passes
  with its solution.
- `tests/`: stdlib `unittest` only. There is a Jev contract test that
  validates our request and parses a canned response using `typesafe-sdk`'s
  pydantic models when that package is installed, and is skipped otherwise.
- CI: GitHub Actions runs unit tests on Python 3.9–3.13, plus a rules-eval
  regression gate.
- License: Apache-2.0.

## v3 architecture (Stage 1: providers, clients, quota)

Maestro v3 grows from the skill router into a local model router with discovery, an OpenAI-compatible endpoint and a web UI.
Stage 1 builds the foundations; the catalog, scoring, proxy server and UI come in later stages. The single file became a
package, `skills/maestro/scripts/maestro/`, run as `python3 <skill-dir>/scripts/maestro <cmd>` (or `python -m maestro`):

| Module | Job |
|---|---|
| `router.py` | The v2.1 logic, moved unchanged: classifier, ladder, seats, `route`, `install`, `models`, and the CLI parser. |
| `__init__.py`, `__main__.py` | `import maestro` re-exports the router API; the directory itself is runnable. |
| `providers.py` | Registry, credentials, live model listing, detection, `maestro keys`. |
| `clients.py` | `complete(provider, model, request, stream)` over three wire formats, normalized to OpenAI chat.completions. |
| `quota.py` | Rate-limit headers, cooldowns, probes, subscription usage, the Claude Code statusline hook. |
| `data/providers.json` | The provider registry, generated from models.dev by `evals/build_providers.py` (about 51 KB). |

`route` stays fast: it never imports `providers`, `clients` or `quota`, and never touches the network. Only `detect`, `doctor`,
`run`, `keys`, `install --statusline` and (later) `discover` and `serve` do.

### Providers (`providers.py`)

- **Registry.** All 225 providers in models.dev, each with `id`, `name`, `env` (the variables that carry the key, any one),
  `params` (settings variables that must also be set: an Azure resource name, a Cloudflare account id, anything written as
  `${VAR}` in the base URL), `need` (the OR-of-ANDs that means "configured"), `base_url`, `style`, `models_path`, `quota` and
  per-model `overrides`. `style` comes from the AI SDK package: `@ai-sdk/openai-compatible` and `@ai-sdk/openai` are `openai`,
  `@ai-sdk/anthropic` is `anthropic`, a provider with a base URL and any other package is `openai`, and the rest (Bedrock, Vertex,
  gateways with a private SDK) are `unsupported`: detected and listed, not callable. `API_OVERRIDES` supplies the 12 base URLs
  models.dev leaves out; `NEEDS` and `EXTRA_PARAMS` describe the multi-variable providers (Azure, Bedrock, Vertex, Cloudflare).
- **Per-model overrides.** A model file in models.dev can carry its own `npm`, `api` and `shape`. `model_spec(spec, model)` applies
  them: an Anthropic-style model behind an OpenAI-style gateway, another base URL, or `shape = "responses"` (not callable: the
  clients only speak chat completions).
- **Credentials.** Environment variables first, then `~/.maestro/keys.env` (mode 0600, `NAME=value`). `maestro keys set|unset|list`
  manages the file; `set` prompts with `getpass` when the value is omitted; `list` shows only "set (env)", "set (keys.env)" or
  "missing". A key is read only to authenticate to its own provider, and it appears in no output, log or saved file.
- **Subscriptions** are detected by presence only: Claude Code (`~/.claude/.credentials.json`, the macOS keychain item via
  `security find-generic-password -s "Claude Code-credentials"` without `-w`, or `CLAUDE_CODE_OAUTH_TOKEN`), Codex
  (`codex login status` exits 0, or `~/.codex/auth.json`), Gemini CLI (`~/.gemini/oauth_creds.json`), Copilot (best effort) and Cursor
  Agent. Tokens are never read or forwarded. Claude, Codex and Gemini get a built-in model table; the others are detected only.
- **Listing.** `list_models(provider)` calls `GET {base_url}/models` (8 s) with the provider's own auth header (`Authorization: Bearer`,
  `x-api-key` + `anthropic-version` for Anthropic with pagination, `api-key` for Azure). Errors are captured with a message that
  cannot contain the key. A 404/405 leaves the provider "detected" (a working key, no list endpoint). Local endpoints reuse the
  router's discovery, and for OpenAI-compatible servers keep each model's raw id and `max_model_len`/`root` when vLLM or SGLang report them.
- **`detect_all()`** builds the inventory `{providers: [{id, name, access, style, status, reason, models, key_source, ...}]}`
  (status `connected | detected | missing | error`) and saves it to `~/.maestro/providers.json`. `maestro detect` shows a compact
  section (keys present, not verified); `detect --models` lists live and saves.

### Clients (`clients.py`)

One entry point, `complete(provider, model, request, stream=False)`, returning `(status, headers, body)` (status 0 = no answer).
`openai` posts to `{base}/chat/completions` (SSE relayed line by line, `reasoning_effort` dropped and retried once on a 400 that names
it). `anthropic` translates OpenAI chat to Messages and back, streaming included: system extraction, tools and tool calls, merged
`tool_result` turns, images, `max_tokens` (default 16000), `reasoning_effort` as `output_config.effort`, thinking blocks dropped,
stop reasons mapped (`refusal` becomes `content_filter`). `cli` (Claude Code, Codex, Gemini CLI) flattens the conversation to a prompt,
runs the binary in a scratch directory and fakes streaming as one chunk; it rejects requests with tools or images.

### Quota (`quota.py`)

Per provider and per `provider/model`, in memory and in `~/.maestro/quota.json`: rate-limit windows from response headers (OpenAI,
Groq, Cerebras `x-ratelimit-*`; Anthropic `anthropic-ratelimit-*`), `retry-after`, and cooldowns after 429, 5xx or no answer (retry-after,
else the reset of an emptied window, else 60 s doubling to 15 min). Probes: OpenRouter `/key`, DeepSeek `/user/balance`. Subscriptions:
the newest Codex rollout file (`rate_limits`, both `resets_at` and `resets_in_seconds` shapes; a reset window counts as 0% used) and the
Claude Code statusline cache written by `maestro install --statusline`, which prints the `settings.json` snippet and edits nothing.
For routing: `headroom()` (None = unknown), `is_available()` (cooldown or exhausted = no), `penalty()` (0.8 under 10% left).

## v3 architecture (Stage 2a: catalog, scoring, discovery)

`maestro discover` turns "what can this user call" into a scored map. Modules: `catalog.py` (the bundled models.dev snapshot, the
optional refresh, the join, `models.json` I/O), `score.py` (pure scoring functions) and `discover.py` (the command, the model map,
hardware and local fit, the backtest and the live probe). None of them is imported by `route`.

### Snapshot (`data/catalog.json.gz`, built by `evals/build_catalog.py`)

About 125 KB gzipped, deterministic bytes, regenerated from a models.dev checkout with Python 3.11+ (`tomllib`; the shipped package
never reads TOML). `models`: every text model of `models/<lab>/*.toml` (name, family, release date, reasoning, tool_call, context,
input modalities, status, open_weights, `[[benchmarks]]` as `[name, score, metric]`, a representative price = the median over the
provider entries that resolve to it). `prov`: for every provider model its canonical lab model (`base_model`), price, context and
status, or its own attributes when it has none. Dropped: embeddings, rerankers, guard / safety models, speech, image, audio and video
models (anything whose output is not exactly text), and deprecated models older than 18 months.

**Benchmark gap-fill.** models.dev has no benchmarks for many open-weight models, so `evals/extra_benchmarks.json`
(`{catalog id: [{name, score, metric, source, variant}]}`, 630 rows for 50 models) holds numbers read from the official model cards on
the Hugging Face Hub (and the gpt-oss model card, arXiv 2508.10925): Qwen3-Next, Qwen3.5 / 3.6, GLM-4.6 / 4.7 / 4.7-Flash, MiniMax M2 to
M2.7, Gemma 4, DeepSeek V3.1 / V3.2 / V4-Flash, Kimi K2 Thinking / K2.5, Step 3.5 Flash, MiMo V2 Flash, Nemotron 3 Nano / Super,
Mistral Small 4 / Medium 3.5 / Devstral 2, Inkling, Nex-N2-Pro, Trinity-Large-Thinking, gpt-oss and a few closed models that another vendor's
card tabulates (GPT-5.2, Claude Opus 4.5, Gemini 3 Pro). Only printed numbers are recorded; a number that another vendor measured
says so in `variant`, and where a card says "GPQA" the row is named GPQA Diamond with `card says GPQA`. `build_catalog.py` merges the rows
(a row models.dev already has, same name and score, wins) and shares them between variants of one repo (`-highspeed`, `-turbo`).
Cards that only show charts (Qwen3-Coder-Next, Hy3-preview) contribute nothing.

**Open-weight registry.** Every `open_weights = true` model records its Hugging Face repo (the `[[weights]]` URL; three models with no
URL get a verified repo from `evals/hf_params.json`), `params_total`, `params_active` (`pt`, `pa`) and where the number came from
(`ps`: `hf-api`, `table`, `name`). Order of preference: the Hub API (`safetensors.total`) when the build machine can reach it, the
hand-verified table `evals/hf_params.json` (totals read from the Hub model pages on 2026-09-30, 73 repos; active parameters only
where a model card or the repo name states them), then the name (`120b`, `30b-a3b`, `8x7b`, `2.4t`). Otherwise unknown (17 models,
mostly with no repo): nothing is guessed. Memory estimates are computed at discovery time, not stored.

### Join

For an available (provider, model id): the provider entry's canonical model, the exact id, `<provider>/<id>`, then the normalized id
(vendor prefix, region, dates, `-v1:0`, `-latest`, `-instruct`, `-preview` and separators removed; benchmarked models win ties).
Local servers add: the Hugging Face repo (vLLM / SGLang report repo ids in `id` and `root`; a quantization suffix such as `-FP8`,
`-AWQ`, `-NVFP4`, `-GGUF` is ignored, and a different org with the same repo name matches as `hf-name`), the served name normalized,
and Ollama tags (`gpt-oss:120b`, `qwen3-coder:30b`, `llama3.1:8b-instruct-q4_K_M`, `llama3.1:latest` using the reported size). CLI
model aliases (`sonnet`, `gemini-flash-latest`) resolve to the newest model whose id has those words plus only version numbers.
Models that match nothing are scored from size and price alone (`basis: prior`).

### Scoring (`score.py`)

1. Benchmarks are grouped into eight axes (code, term, reason, agent, browse, data, write, longctx) by name; variants of a benchmark
   are merged by the median.
2. A value becomes a percentile among the recent cohort: models released within 7 months of the newest catalog model, and at least 4
   reporters, else the benchmark is ignored (saturated or one-off). The reference date comes from the catalog, never the clock.
3. Prior (no benchmarks): size (`sqrt(active * total)`), log price, reasoning flag, name-tier words, recency and context, affinely
   calibrated to the benchmarked models. **Age cap:** a model older than 18 months has a prior of at most 0.40.
4. `G = w*bench + (1-w)*prior`, `w = n/(n+3)`. **Small-n damping:** axes are averaged with weight `c/(c+1.5)` (`c` = benchmarks in
   the axis), so one lone benchmark moves G little. **Unknown-lab distrust:** benchmarks reported by a lab with fewer than three
   benchmarked models in the catalog count 0.5.
5. Job strengths from axis weights (per axis blended with G by its count), with the small adjustments of the prototype (reasoning,
   tool use, context, coder names).
6. Categories: G >= .68 frontier, >= .52 strong, >= .35 standard, else fast (a weak-tier name such as mini / flash-lite needs .42 for
   standard). **Specialist:** a text model with coder / sql / computer-use / deep-research / translate naming, or one whose best job
   strength is at least 0.12 above G. Local models are `category: local` with `serves` = the tier of G capped at strong; a
   `models.local` pin or a `local_models.<name>.serves` in the config still wins.

### `models.json` (the contract with the proxy and the UI)

`~/.maestro/models.json`, one model per line: `{version: 1, generated_at, catalog_as_of, hardware, models, local_fit, ...}`. Each model has
`id` (`provider/model`, `local:<endpoint>/<model>`, `cli:<cli>/<model>`), `provider`, `model`, `access` (`api | subscription-cli | local`),
`style`, `tools_ok` (false for CLIs), `vision`, `context`, `cost_in`, `cost_out` (per 1M tokens), `G`, `jobs` (11 strengths), `category`,
`serves`, `basis` (`bench | blend | prior`), `benchmarks` (names used, best first), `catalog_id`, `join`, `family`, `release_date`, `quota`
(`{state, headroom}`), `enabled`, `override`; local models add `box`, `params_b`, `loaded` (the server reports it loaded; true when it
does not say) and `routable` (`loaded`, or `local_allow_load`; independent of `enabled`, and recomputed from the current config by
`catalog.load_models` and the proxy), `latency` appears after `--live`. A local model with `routable: false` is listed, never routed to,
probed or benchmarked. Additive extras:
`specialist`, `local_serving`, `notes`, `providers`, `catalog_source`. **Overrides** live in the config
(`models_overrides: {id: {category, enabled}}`) and are applied by `catalog.load_models(home=None, cfg=None)` on every read, so a
rediscovery never loses them: a valid category replaces the discovered one (kept as `category_auto`), `enabled` switches the model on or
off, `override` records what was applied. Without a file, `load_models` returns the same shape with `models: []` and `present: false`.
`maestro models` shows this map when it exists and has models (`--seats` forces the per-harness table of stage 0).

### `discover`, backtest, live probe

`maestro discover [--refresh-catalog] [--backtest] [--live N] [--local-fit] [--all] [--json]`: detect providers (`providers.detect_all`,
which lists models live), refresh quota probes, load the catalog, join, score, write `models.json` and print the table by provider.
`--refresh-catalog` fetches `https://models.dev/api.json` (or `$MAESTRO_CATALOG_URL`), merges prices, context, status and new models
(benchmarks stay as built) and caches the delta in `~/.maestro/catalog_refresh.json`; offline it says so and uses the snapshot.
`--backtest` routes the 474 eval prompts with `classify_rules` over the map, offline: per prompt the ladder cell, then `score.pick`
(tools when the job needs them, at or below `max_tier`, free tiers excluded, job strength and tier at the cell's level, local first when
`prefer_local`, then cheapest expected cost; see "Picking a model"). It reports the category hit rate
against the labelled expected category (with within-one, under- and severely under-routed), the requested tier served, cost per 100
requests against always using the strongest model (2,000 input tokens, 500 x (1 + difficulty) output tokens; subscription and local
models cost 0 at the margin) and the local share. `--live N` sends N tiny prompts (`max_tokens` 16, streamed) to each model the
backtest used and stores time to first byte, total time and failures under `latency` (kept across rediscoveries); it grades nothing.
It probes only routable models (a local model that is installed but not loaded is skipped: the request would load it) and sends at most
one request at a time to each local endpoint; the backtest ignores models that are not routable.

Plain `maestro discover` ends with one line per box: `spark: serving 2 models (gpt-oss-120b, ...) - using these` when the box serves
loaded models (no suggestions), or `spark: nothing served; `maestro discover --local-fit` lists models that would fit (advice only)`
(with "N installed but not loaded, ignored" when it has such models).

### Hardware and local fit

`hardware` maps boxes to `{name, memory_gb, kind, bandwidth_gbps}`: `this-machine` (DGX Spark / GB10 from `nvidia-smi` = 128 GB unified
memory at 273 GB/s, Apple silicon unified memory with the chip from `sysctl machdep.cpu.brand_string`, NVIDIA VRAM, else RAM) plus each
`local_endpoints[].hardware: {name, memory_gb, bandwidth_gbps?}`; an endpoint named or hosted "spark" with none gets
`{DGX Spark, 128, 273, assumed: true}`. Bandwidth comes from a table matched on the name (GB10 273, RTX 3090 936, 4090 1008, 5090 1792,
A100 2039, H100 3350, L40S 864, M1 to M4 by chip: M3 Max 400, M4 Max 546, M4 Pro 273, M3 Ultra 819, M2 Ultra 800 ...; Max and Pro chips
come in binned variants, so it is best effort) or from the config; unknown means speed unknown, not slow.

For each box, `local_fit` considers the open-weight text models with a known size, one per Hugging Face repo, at every precision
that fits (BF16 2.0, FP8 1.0, NVFP4 0.5625, Q4 0.6 bytes per parameter, +15% runtime, plus KV headroom for a 32K context, within 90% of
memory; Blackwell boxes may use NVFP4, Apple and CPU only Q4; gpt-oss is released in MXFP4 and is served as it is). Each precision
gets an estimated single-stream decode speed, `est_decode_tps = bandwidth * 0.6 / (active parameters * bytes per parameter)` (total
parameters when the active count is unknown, i.e. the model is assumed dense; the row says `active_known: false` and the table marks
the speed with `*`), and a fit score, `(mean job strength - quantization loss) * speed factor` with the factor 0.3 under 10 tok/s,
0.7 from 10 to 25 and 1 from 25 (or unknown). The best precision wins and rows are ranked by fit score. A dense 128B model on a
273 GB/s Spark decodes at about 2 tok/s and drops far down; a 5B-active mixture-of-experts model does 57. Rows carry the job strengths,
`tps`, `tps_est: true`, `active_params_b`, `mem_gb`, `kv_gb`, `fit_score` and a `vllm serve` / `ollama run` / `llama-server` command
(NVFP4 and Q4 commands say that such a build must exist). `local_serving` compares what a box already serves with where it would rank.
Advice only: nothing is downloaded or started.

Suggestions matter only when nothing is served. When a box already serves routable models, its memory counts as in use: one model server
usually claims most of it (vLLM reserves `--gpu-memory-utilization` of the memory, 0.9 by default), so a suggestion would **replace**
what is served, never add to it. `--local-fit` still lists them on demand, with the header "Alternatives on <box> (would REPLACE what it
serves now: ...)", sizes computed for the whole box as if the current server were stopped, and every command prefixed with
`# stop the current server first: this replaces what <box> serves, it does not add to it`. The rows in `models.json` carry the same
prefix in `serve_cmd` and `replaces: [model ids]`. The web UI's "Could run locally" panel shows "Using: <served models>" when any local
model is served and folds the suggestions behind "Alternatives (would replace what you serve)"; with nothing served it shows them under
"Maestro never downloads or starts models".

### Picking a model (`score.pick`)

`score.pick(candidates, job, category, objective="balanced", prefer_local=True, tools=False, vision=False, min_context=0,
max_tier="frontier", tokens=(2000, 500), difficulty=None)` ranks models.json entries for one job at one ladder cell and returns
`[{id, model, tier, strength, cost, reasons, fallback, below_floor}]`, best first. A model qualifies only if its strength for the job is
at least the lower threshold of the cell's category (fast 0, standard .35, strong .52, frontier .68) **and** its own tier is at least the
cell (a high job score cannot pull a standard-tier model into a strong cell). Out: disabled models, cooldown or exhausted quota, free
tiers (`:free` ids, API models priced 0), filtered-out capabilities (tools, vision, context), cloud models above `max_tier`, specialists
not lifted for the job. Ranking: `balanced` = quota state (low last), local first if `prefer_local`, expected cost, quota headroom (zero-cost
seats), strength; `cost` drops the local bonus; `quality` puts strength first. Qualification is monotone in the tier, so a cell with
no qualifier has none above it and "stepping up" never finds anyone; the walk goes down (with a `stepped down` reason) only for
difficulty below 2 or unknown, and if nothing qualifies at all the strongest model for the job comes back alone with `below_floor: true`.
The backtest and the proxy share this function.

### Proxy, web UI and Hermes (`server.py`, `ui/index.html`, `onboard.py`)

`maestro serve` is a `ThreadingHTTPServer` over the clients and quota layers (bind `127.0.0.1:4545`; anything beyond loopback needs `MAESTRO_API_KEY`).

- **Endpoints.** `GET /v1/models` (`maestro/auto`, `maestro/<category>`, `maestro/<job>`, every enabled `<provider>/<model>` with `context_length`), `POST /v1/chat/completions`, `GET /health` (no auth), `GET /` (the UI), `/api/*` (admin).
- **Model names.** `maestro/auto` classifies (rules, or Jev among the top five candidates when `TYPESAFE_API_KEY` is set); a category name fixes the cell, a job name fixes the job (difficulty still comes from the text); a concrete id is tried first and the next-best candidates of its category follow as fallbacks. Classification text is the last user message, plus the first 500 characters of the system prompt when the message has fewer than four words.
- **Candidates.** From the model list (`models.json`, user overrides applied; without it, entries built from `providers.json` with no scores and a category from the suggested-model tables): enabled; routable (a local model that is installed but not loaded is skipped with the reason "installed but not loaded (Maestro never loads models)", is left out of `GET /v1/models`, and a request naming it gets 409 `model_not_loaded`, all unless `local_allow_load`); `tools_ok` when tools (or earlier tool turns) are present, so subscription-CLI seats are skipped with the reason recorded; vision; `est_in + max_out` within the context; not on cooldown or exhausted (`quota.is_available`); not above `max_tier`; local models only when `prefer_local` (unless asked for by name or nothing else exists). Then the floor: the model's strength for the job must reach the cell's floor (fast .15, standard .35, strong .52, frontier .68; a category override counts as the user's word; a plain local model must also *serve* the cell; a specialist is judged by its job strength alone). Nothing reaches it: the strongest available is used and the reason says so. A `local` cell means local models, else `fast`.
- **Ranking.** Key: near its limit last, then expected cost in log buckets (two per decade for `balanced`, six for `cost`; `quality` puts strength first), then local first when preferred, quota headroom, observed time to first byte, strength, id. Expected cost is tokens times price; local is 0; a subscription seat is a token amount that grows with its strength (below any API price, above local).
- **Affinity.** Key: `X-Session-Id`, the header in `session_affinity_header`, else a hash of system prompt + first user message, plus the requested model name. A last message that is a tool result (or an assistant turn with pending tool calls) reuses the model and classification for 30 minutes if that model still passes the filters; a new user message routes afresh.
- **Execution.** Up to three candidates; 0 (no answer), 401/403/404/408/429 and 5xx fall back, other 4xx go to the caller. A stream is pulled once before any byte is sent so a failure there falls back; after that, an upstream error becomes an `error` event in the stream and the connection closes. Streaming responses are close-delimited (`Connection: close`). Chunks are rewritten only to set `model` to the real `<provider>/<model>`. `x-maestro-decision` carries `{requested, job, difficulty, category, model, fallbacks, affinity, attempted, reason}`.
- **Log.** `~/.maestro/maestro.db` (`decisions`, WAL, mode 0600, 90 days kept): time, conversation hash, requested name, job, difficulty, cell, category, chosen, candidate ids, reason, time to first byte, total time, tokens (usage, or estimated and flagged), status, attempts, stream/tools flags, estimated cost and the cost of the strongest API model for the same tokens (the UI's "saved"), and 120 prompt characters only with `log_prompts`.
- **Security.** Admin token in `~/.maestro/admin_token` (0600), sent as `X-Maestro-Token`; the UI takes it from the URL fragment and removes it from the address bar. `/api/*` and `/v1/*` check `Host` (loopback servers accept only loopback names, against DNS rebinding) and `Origin` (must equal the server's own, or be in `allowed_origins`). No CORS headers are sent. Keys are write-only through `POST /api/keys` (only variables some provider reads), stored in `keys.env`.
- **Background.** `quota.refresh` every 10 minutes in a daemon thread; discovery in a daemon thread at start when `models.json` is missing or older than 24 h (`run_discover()` calls stage 2a when present, else re-detects providers).
- **Admin API.** `GET /api/state|decisions|stats`; `POST /api/keys|models/<id>|config|discover|route|send|test`. `config` accepts only `objective`, `prefer_local` (`local_allow_load` is deliberately not settable from the UI), `log_prompts`, `max_tier`, `classifier`, `min_confidence`, `ladder`, `models`, `job_seats`, `prefer`, `session_affinity_header` (validated; `null` removes a key).
- **Hermes.** `maestro hermes --print` emits `model` (custom provider, `maestro/auto`, `context_length` 128000), `auxiliary` (`compression`, `title_generation`, `session_search` on `maestro/fast`; `vision` on `maestro/auto`) and `delegation` (`maestro/auto`). `--install` appends only the top-level blocks that are missing, after a backup, and prints the rest for the user to merge. `maestro setup [--yes]` walks detect, keys (hidden input), the local step (the hardware gauged for each box and what each box serves; only when no box serves anything, the top three models that would fit the strongest box, "advice only: Maestro will not download or start anything"), discovery, the category summary, Hermes and the skill.

## v3 Stage 3: S1 and expected-utility routing

The full specification is [docs/S1-SPEC.md](S1-SPEC.md); "spec §x" in the code and below refers to it. This section is the summary
of what was built; the deviations from that spec are items 89 onward.

**Shape.** One interface separates the model from the decision: S1 (or an adapter) turns a task into a `TaskPosterior`, and
`decide.py` ranks every seat that survives the existing hard filters by expected utility in dollars. Two deciders share the hard
filters: `ladder` (the v2/v3 table, the default) and `utility` (behind `decider: utility`). With `s1_shadow` (default on), the
decider not in use is computed too and its pick is logged as `shadow`, never acted on.

| module | lines | role |
|---|---|---|
| `s1.py` + `data/s1.bin` | 419 + 1.2 MB | normalize (str.find fence split, unfenced material runs, scan caps), crc32 features with a Bloom known-feature filter, int8 folded embedding bag (2^14 x 32 bootstrap) summed with SWAR, 24 outputs, calibrated heads, pooling with NB; the NB and Jev adapters; `legacy()` decodes job and difficulty with the risk-aware rule; `verify()` for `doctor` |
| `decide.py` + `data/decide.json` | ~725 + 4.7 KB | `seat_of` (router, server and models.json shapes), `inline_seat`, `theta`, `p_success`, `tokens`, `cost`, `latency`, `read_windows`, `quota_price`, `utility`, `rank`, `escalate`, `confidence`, `p_best`, `reasons`, `log_decision`, `minhash`, `learn_k`, `is_private` |
| `learn.py` | 487 | outcome signals (explicit, tests, escalate; reask/thanks/override at weight 0), the replayed MAP fit of `personal.json` with its guards, the `outcome`, `feedback`, `learn` commands and the SubagentStop hook handler |
| `systemone.py` | 495 | `POST /v1/systemone` and `maestro decide`: Jev's wire format, native intents, probed delegation (logprobs, prefill, votes), cache, calibration |
| `commands.py` | ~145 | `register(sub)`: route hint flags, `install --hooks`, `s1`, `decide`, `outcome`, `feedback`, `learn`; handlers import their modules lazily |
| `nb.py`, `jev.py` | 121, 192 | the Naive Bayes and Jev code moved out of `router.py`; `router` re-exports the same names (Jev lazily, via module `__getattr__`) |

**TaskPosterior** (a plain dict): `job_p` (11 jobs), `b = [mu, sd]` (difficulty on the anchor logit scale, OOD-inflated), `group`
(`agentic` when skill mode and tools >= 0.5, or proxy with a `tools` parameter; else `chat`), `tools`, `self_contained`,
`stakes` (E in [0, 2]), `detectable`, `ambiguity`, `tok_out` and `turns` (`[mu, sd]` of a log-normal), `ood`, `source` (`s1`, `nb`,
`jev`), `trained` (the heads `s1.bin` has weights for; the others come from job priors), optional `hints` and `s1_ignored`.

**Success and utility** (spec §3-§5). `P_s = Cal_g(σ(a_g(θ̄_s − μ_b)/√(1 + π/8·a_g²(σ_b² + sd_s²))))`; ranking uses the pessimistic
`P̃` (ability lowered by `z_r = risk·0.5·E[stakes]` sd). θ̄ mixes per-job abilities by `job_p`: Claude's haiku/sonnet/opus/fable
are anchors (θ from the header plus β times the model's job profile); other models come from `models.json` `theta` (precomputed by
`discover`) or the catalog link `α_g + β·Φ⁻¹(strength)`; `anchors.json` (user-run anchors) and `personal.json` offsets adjust it;
effort, harness (η) and the context penalty for delegated skill-mode seats are added. `EU = V·P̃ − C − Q − κ·T − (1 − P̃)·F` with
`V = V0[objective]·clip(√(tok/1000·turns), 0.5, 8)`, `F = d·(κ(T + 45) + C) + (1 − d)·ℓ(stakes)·V`. Seats under the floor
`P_min(stakes)` rank after all others. Tokens, cache reads, cache-write premiums and the switching cost (§4.2), the quota pacing
price (§4.3), API rate-limit pressure and local-seat latency (§4.5, §4.6) are as specified.

**Skill mode** (`router.route`, about 90 lines in `_decide` and the hook). `route` imports `s1`/`decide` lazily, only when the
decider is `utility`, or when `s1_shadow` is on and someone looks (`log=True`, which `maestro route` passes, or `--explain`); an
in-process `route()` without either runs exactly as before. Seats are `_candidates` (hard filters unchanged) plus the calling agent
as an `inline` seat when `--current-tier` is given. Under `utility` the EU argmax runs the task (`_seat_result` keeps the fallback
semantics, in EU order), except a tool job (`group == "agentic"`) while `decide.json:agentic_ready` is false: the ladder decides and
the EU pick is the shadow. The decision adds `decider`, `decision_id`, and under `utility` `posterior`, `options`, `p_success`,
`escalate_to`, `flags`, `confidence`; `--explain` adds `shadow`, `options`, `skipped`, `flags` and the posterior, and
`--format text --explain` prints the options table. `maestro route` appends one line to `~/.maestro/decisions.jsonl` (0600, 10 MB
rotation, `log_decisions: false` turns it off): heads, options, the trace learn needs, a 16-value MinHash, the policy id
(`s1:<crc32 of the header>`), never text. Any exception in the decision layer leaves the ladder's seat in place with a note.
`classifier: s1` uses `s1.legacy` for job and difficulty; `auto` does so only once the shipped header records the §8.4 pass.

**Proxy** (`server.App.plan`). `features()` adds `prev_user`, `n_user`, `n_tool`, `last_tool`, `asst_len`, `images`, `prefix_in`
and the trailing tool results. Affinity stores `{id, job, difficulty, cell, mu_b, sd_b}` on every successful turn (30 min). For
`maestro/auto` and `maestro/<job>`: under `utility`, `s1_decide` ranks `candidates()`'s `passing` list (hard filters, no floors)
with `decide.decide` in proxy mode (warm seat and prefix for the switching cost, background κ for `X-Maestro-Background: 1` and
`maestro/<job>`), optionally swaps in the runner-up under `explore` (learn.explore, propensity logged), and the EU order is the
fallback order; under `ladder`, the floor and `rank_key` order stands, a tool turn keeps its model, and `s1_decide` runs as shadow.
`maestro/<category>` and concrete ids are not touched. The decision row is now inserted when a seat has answered, so its id is the
`decision_id` in `x-maestro-decision` (which also gains `decider`, `p_success`, `escalate_to`); `finish()` completes the row. New
columns (`learn.migrate`): `s1` (≤ 400 B JSON: top-3 job_p, b, group, trace), `p_success`, `eu`, `decider`, `shadow`, `explore`,
`propensity`, `policy`; new table `outcomes`. Signals: a test-runner summary in the tool results of the next request (`tests`), the
implicit rules on a new user turn of the same conversation, `X-Maestro-Feedback: <id>=good|bad`, `POST /api/feedback {id, value}`.
New routes: `POST /v1/systemone` (same auth, Host and Origin checks as `/v1/*`; the `model` intent ranks the server's own
candidates), `GET /api/alpha` (shadow agreement in proxy and skill mode over 14 days, outcome counts, the personal fit), and
`GET /v1/models` adds `models` (systemone's model list). A daemon thread runs `learn.run` 10 minutes after start and then daily, on
its own database connection. `CONFIG_KEYS` validates the §12.3 keys.

**Quota history** (spec §4.3). The statusline script appends `{"t", "five_hour", "seven_day", "seven_day_<family>"...}` to
`claude_limits.log` on a change (cut to 2,000 lines past 4,000); `quota.refresh_subscriptions` appends Codex samples to
`quota_hist.json` (one per 300 s, 2,016 per window, plus a `current` map with reset time and window length); `decide.read_windows`
reads both with an mtime cache and never imports `quota`. `quota.window_view(provider)` gives `[{name, used, reset_at, length_s}]`.

**Discover and score.** `score_rec` also returns `axes8` (the blended per-axis strength) and `axes_n` (their weights), stored in
`models.json` for v1.1 and unused in v1. `discover` writes `theta: {job: [mean, sd]}` per scored model from the link in the
`s1.bin` header (else `decide.json`), `theta_link` (a hash of that link, so a θ computed under another link is ignored), and a
top-level `catalog_sha` (sha256 of the bundled catalog; `fit_link.py` records it too). `discover --live` stores
`systemone_strategy` for each loaded local model. All keys are additive; `MODELS_VERSION` is unchanged.

**Config keys** (§12.3, in `router.DEFAULT_CONFIG`, validated by `load_config` and `server.CONFIG_KEYS`): `decider` (`ladder`),
`classifier` gains `s1`, `wtp_usd` (null), `latency_usd_per_s` (0.002), `background_latency_usd_per_s` (0.0002), `risk` (1.0, 0-3),
`quota_reserve` (0.15, 0-0.5), `local_usd_per_hour` (0), `inline_access` (`subscription`), `private` (false), `log_decisions` (true),
`s1_shadow` (true), `personalize` (true), `explore` (0, 0-0.1), `systemone_delegate` (null), `systemone_cloud` (false).

**UI.** Routing: a Decider setting (ladder or expected utility), private mode, and a "Shadow agreement" card (`/api/alpha`).
Activity: Good / Bad buttons per row (`/api/feedback`) and the decider, P(success), shadow pick and decision id in the details.
Try it: the options table and the posterior for a dry run.

**Evals** (repo only). `split.py` (clusters, cluster-hash splits, `manifest.json`, leakage checks), `label.py` (teacher panel
batches, ingest, Dawid-Skene aggregation, agreement), `alpha.py` (pilot, select, two-phase answering, grading, fit, replay, report,
CI gate, release gate, calibrate-v, pacing, collect) over `alpha_eval.py` and `alpha_stats.py` (stdlib), `fit_link.py` (β_pub),
`train_s1.py` (numpy; `--check` reproduces `s1.bin` bit for bit), `typed_eval.py` (better-Jev), `run.py --classifier s1`. Subagent
steps write prompt manifests to `evals/work/` and ingest the orchestrator's replies; they never call a model API. CI runs
`split.py --check` and `alpha.py gate --split test` (both pass with a note until data exists).

**Measured** (this sandbox, CPython 3.11, 4 vCPU). S1 in process: p50 0.20 / p99 0.35 ms short, 1.7 / 3.2 ms at 300 words,
2.9-4.0 / 4.4-6.1 ms on 60k characters (the router's `classify_model` keeps its keyword scores, so S1 does not rescan the text).
`decide()` on 40 seats: p50 0.45-0.55 ms, p99 0.7-1.0 ms (3.9: p99 1.0-1.4 ms). 200 in-process `route()` calls with
`decider: utility`: 0.22 s. `maestro route` CLI wall time: 43.6 ms (base) vs 48.8 ms p50 (p10 40.7 vs 42.8) with shadow and
logging, 49.0 ms under `utility`. `/v1/systemone` native: 0.4 ms (3 questions), 1.4 ms (10 questions with a `model` question
over 9 seats); delegated 4.7 ms on a cache miss against a loopback server, 0.16 ms on a hit. `learn.run`: 0.07 s on 300 outcome
rows, 1.2 s on 10,000.

## Phases

1. **Core:** classify, policy, detect and route; tests; eval set; baseline
   numbers.
2. **Harnesses:** SKILL.md, `install`, the plugin marketplace, and a smoke
   test in Claude Code.
3. **Better classifiers:** a live Jev eval (needs a key and network), then
   an Arch-Router 1.5B option via Ollama. Pick the default by accuracy per
   dollar and latency.
4. **Proxy (optional):** `maestro[proxy]`, a LiteLLM pre-call hook exposing
   `maestro/auto` for OpenAI- and Anthropic-compatible clients, with LiteLLM
   pinned and hash-locked.

## Deviations and implementation notes

Where the implementation differs from, or fills a gap in, the design above.

1. **`target_tier` returns a ladder cell string** (`local` or a tier). Escalation and cap details (`ladder_tier`, low-confidence bump, `max_tier`) are reconstructed in `route` for the `reason` and `--explain`.
2. **Difficulty (rules)** is a points score: a per-job prior (plan 1.4 ... ask 0) plus reasoning demand x3, length, scope markers, hard-topic words and multi-step markers, minus "easy" words. Thresholds 0.5 / 1.5 / 2.5 map to 0-3.
3. **Rules matching** uses token lookup with crude stemming (fixes/fixed/fixing -> fix) for single words and substring checks for phrases, instead of the v1 exact-token sets. This avoids compiling large regexes at startup. Scores are still the number of distinct matched patterns, with the v1 structural boosts and confidence formula.
4. **`needs`** is `tools` for the ops/code/debug/review/research jobs, file paths, repo words or agentic keywords; `vision` for image/screenshot/pdf words; `long_context` above 60,000 characters. Jev does not return `needs`, so they always come from the rules signals.
5. **Seat tiers.** Superseded by item 33: every harness serves all four tiers from config or suggestions. A seat with neither serves only `strong` with no `-m` flag. OpenCode and Gemini CLI `delegate`s carry `"agent": "maestro-<tier>"`.
6. **`prefer_local`** gates local-model eligibility (v2.1: it also turns off native subagents that run a local model). See the v2.1 notes below for the order.
7. **Walk cap.** The upward walk stops at `max_tier`; when nothing is found the decision is `inline` with the requested tier. The current harness is never its own CLI seat, except Codex (item 33).
8. **Local models.** Embedding models are ignored; a model with no parseable size is rated `local` (v2.0 ignored it). `detect` warns when a model's rough 4-bit footprint (params x 0.6 + 1 GB) exceeds detected memory; it does not exclude the model.
9. **Catalog.** `detect --catalog` stores `{tier: [model ids]}` (max 25 per tier) only for providers whose key env var is present, and caches a slimmed copy of models.dev in `catalog.json`.
10. **Jev classifier (product-owner change).** Jev now picks the model, not just the class: see §1. `classifier` defaults to `auto`. `needs` always come from the rules signals. Local candidates are built from the rules classification (Jev has not answered yet); if Jev picks a candidate that cannot do its own `job` (a chat-only local model for a repo job, or a specialist outside its jobs), the pick is treated as low-confidence. The `category` in a Jev decision is the category of the chosen (or ladder-escalated) model. The cache stores the parsed answers. `MAESTRO_CLASSIFIER` is also honoured; precedence is flag, env, config. New config keys: `objective`, `jev_min_confidence`, `model_notes`.
11. **Inventory freshness** uses the `generated_at` field, not file mtime.
12. **`maestro install`** marks generated files with an HTML comment containing `maestro-generated`. The `model:` line is the resolved model (config or suggestion); the commented `# model:` template line is gone (item 33).
13. **Rules classifier (learned layer).** `classify_rules` is now the heuristics plus an embedded Naive Bayes model, not keywords alone; see §1. The model is loaded lazily from `maestro_model.json` next to `maestro.py` (or `<prefix>/share/maestro/`, where `pip install` puts it, since a single-module wheel has no package data). Difficulty in the heuristics is computed from the instruction only (fenced blocks removed and long text truncated to 4,000 + 2,000 chars), which also bounds classify time on huge pastes. `evals/run.py` gained `--split dev|test|all` (dev = even line index, test = odd), tier-delta metrics and `--variant`. The 1.3k-line target for `maestro.py` was raised to about 1.45k for the classifier.
14. **No double escalation.** The learned model already picks its tier by expected risk (under-routing weighted 1.5×), so the rules-path `min_confidence` +1 bump applies only to the heuristic fallback. Found in an end-to-end Claude Code run where a trivial docstring task was escalated to `opus`.

### v2.1 (plain-English jobs, six categories, local and network models)

Deviations from the v2.1 spec and notes on how it was implemented:

15. **Jobs replace task classes; categories replace tiers.** `task_class` is now `job` and `tier` is `category` in the decision, `Classification`, Jev's questions/cache (cache file format 2; old caches are ignored), the evals and the config (`class_seats` is now `job_seats`). There is no alias for the old names. `fallback` entries also use `category`.
16. **Cloud ladder constants.** `TIERS` stays the four cloud tiers, `CELLS = ("local",) + TIERS` is what a ladder cell holds and `CATEGORIES` adds `specialist`. `--current-tier` still takes a cloud tier.
17. **Retrained on the 11-job labels.** `evals/prompts.jsonl` (474 rows; the first 350 keep their original order, so the dev/test split by line parity is comparable to v2.0) and `evals/train.jsonl` carry native `job` labels, written by an LLM and partly hand-audited (about 15% of rows are genuinely ambiguous between two jobs). `evals/run.py` no longer maps the v2.0 `task_class` labels. `evals/train.py` trains on `job` labels, writes `jobs` and refuses a corpus without them. Three train rows that were near-duplicates of eval prompts (Jaccard above 0.6) were removed, leaving 1,536; the disjointness test in `tests/test_evals.py` enforces this. The keyword tables and the combination weights were tuned on the dev half only (with a 5-fold cross-validation on `train.jsonl` as a sanity check that a change was not noise); the test half was scored once per final variant. Numbers are in the README and the CI thresholds in `.github/workflows/ci.yml`.
18. **Eval metric names** follow the new vocabulary: `job_accuracy`, `category_accuracy`, `category_within_1`, `category_delta`, `per_job`; the CLI gates are `--min-job-acc`, `--min-category-acc` and `--min-category-within1`. Predicted and expected categories are compared on the cloud ladder (a `local` cell counts as `fast`).
19. **Decision `category` is the chosen model's category**, not always the target. For a cloud fallback (no seat at the target tier) it is the tier of the seat actually used, and the `reason` says so. For `inline` it is the target tier.
20. **Seat order changed.** Specialists, then local models, are considered before `inline` and before native subagents, because they are free and private. So `--current-tier standard` does not stop a `write` d1 task from going to a local model that serves standard. A `local` cell escalates to `fast` (not to `standard`) on low confidence.
21. **Specialists** are used only for their jobs: a `sqlcoder` model is never picked for `ask`. A plain local model that is also a specialist keeps its default `serves` from size, with a floor of `fast`. A cloud model named in `specialists` is filtered the same way.
22. **Parameter-count parsing** treats `NxMb` as N times M (`8x7b` is 56) and ignores `aNb` active-parameter tags. 40B exactly rates `standard`, 10B exactly rates `fast` (the spec's ranges overlap at 40).
23. **Native local subagents.** For OpenCode and Gemini CLI, a `models` entry that names a detected local model (by id, `endpoint/name`, name, or `<provider>/<name>`) is a `subagent` seat that is tools-capable and counts as `local` for preference and `serves` (the category key it is listed under is its `serves`). The model is then not offered again as a plain chat-only local model. The Claude Code seat never runs local models.
24. **Codex as its own CLI seat.** When Codex is the current harness, `codex exec -m <model>` is offered for the categories (item 33 made this unconditional). Otherwise the current harness is never its own CLI seat.
25. **Inventory** is version 2 (`local_endpoints`, `local_models`, `endpoints_sig`; the v2.0 `ollama` key is gone). `route` re-detects when the inventory is version 1, older than 24 h, or when the URLs in config `local_endpoints` / `OLLAMA_HOST` changed. That re-detect is the only time `route` touches the network, and it only probes local and configured endpoints.
26. **Local traffic bypasses proxies** (`ProxyHandler({})`), so a system HTTP proxy cannot break `detect` or `run`.
27. **`--scan`** finds the /24 from the UDP-connect address, skips its own address, and names hosts by IP (rename them in the config). The 5 s budget is enforced by waiting on the futures and abandoning the rest.
28. **`maestro run`** accepts a task that starts with a dash, and reads stdin when no task is given. It looks the model up in the cached inventory, then in the configured endpoints (probing the kind if unknown).
29. **Line count.** `maestro.py` is about 1,920 lines, over the 1,750 target (and CLAUDE.md's 1,450): eleven keyword tables and the local-model support (discovery, ratings, scan, run) are the growth.
30. **Classifier combination (v2.1 retrain).** The decode changed in three ways, each measured on the dev half and on the cross-validation of the training corpus: difficulty is conditioned on the job (`"jd"` table, about +1.5 points of category accuracy), the tools jobs get a bonus (`MODEL_TOOLS_B`; it halves the tools job read as a chat-only job on the training folds for about one point of category accuracy), and difficulty has its own temperature (`MODEL_TD`). Chosen constants: `MODEL_T` 4, `MODEL_TD` 2, `MODEL_CLASS_W` 1, `MODEL_DIFF_W` 2, `MODEL_TOOLS_B` 1. The dev grid is flat around them (about one point of category accuracy); raising `MODEL_DIFF_W` or `MODEL_TD` buys category accuracy only at the price of more under-routing, so the low-under-routing point was kept. Other things tried and not kept: Gaussian likelihood of the heuristic difficulty points, a direct tier head, a logistic-regression model (better exact accuracy but 15-20% under-routed at the same cost weighting), `ALPHA`/`MIN_COUNT`/`SCALE` changes.
31. **Keyword changes.** Debug cues ("diagnose", "culprit", "what caused", "figure out why", ...) and "investigate/find out why" leading to debug; a smaller infra bonus for ops; edit outranks code on a tie and gets a larger small-edit bonus; "extract X into a function/constants" is an edit, not a summary; "convert this to TypeScript/Python/types" is code, not a summary. Heuristics alone score 68% job accuracy on both the training corpus and the dev half.
32. **The eval prompts are longer than the training prompts.** The training corpus has no instruction longer than 64 words, while 17% of the dev half does, and the hardest (difficulty 3) rows are mostly long. The model under-reads them, and much of the remaining under-routing is a difficulty-3 debug, review, ops or research task read as difficulty 2. Adding long, hard, realistic prompts to `train.jsonl` is the most likely next gain.

### One configurable model map per harness, with dated suggestions

33. **`models` replaces `harness_models`; every harness has suggestions.** Routing used to pick only Claude models unless the user wrote `harness_models`. Now `seat_models()` returns `{category: (model, effort, source)}` from config, then `SUGGESTED_MODELS` (a snapshot of models.dev dated `SUGGESTED_AS_OF`), then, for `strong` only, the CLI's own default (see §4). `NATIVE_MODELS` is gone (its four aliases are the claude-code row). Codex, Gemini CLI and OpenCode now serve all four cloud categories with no config; OpenCode and Gemini CLI are current-harness `subagent` seats out of the box, Codex is a `cli` seat that also serves as the current harness. Other installed CLIs get the full map and rank after the current harness (the current harness's CLI seat is listed first, so inside Codex `codex exec` beats `claude -p`). A model that serves two categories keeps its name for the lower one and gets `@<category>` for the higher (Jev needs unique candidate names).
34. **`maestro models` and the decision.** New command (§6) and a `model_source` key in the decision JSON. `--check` needs the network (`--suggest`, `--save`, `set` and `unset` do not). `detect --catalog`'s cache (`catalog.json`) now keeps `[output price, family, release_date, status, tool_call]` per model, for every model with or without a price, so `--check` can reuse it; an older cache (a bare price) is treated as absent by `--check` and read as before by `detect`. `MAESTRO_CATALOG_URL` overrides the catalog URL.
35. **Effort** is passed only to Codex (`-c model_reasoning_effort=<effort>`); Claude Code, Gemini CLI and OpenCode ignore it. A value outside `low|medium|high|xhigh|max` is dropped.
36. **Local pins** (`models.local`) rate the model as the highest cell it is pinned under, and put it first for the cells it is pinned to. A pin beats `local_models[...].serves`. Pinning a large model under `local` therefore makes it serve only `local` cells; pin it under every cell you want it for.
37. **OpenCode with no provider key.** With none of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY` in the inventory (for instance a login stored by the OpenCode CLI, which Maestro never reads), the anthropic row is suggested. `maestro models --suggest` says when no key was found.
38. **`default_models` detection removed.** `detect` no longer reads `~/.codex/config.toml` or `~/.gemini/settings.json`, and the inventory has no `default_models`. With suggestions the value could never be reached, and a CLI run without `-m` uses its own default anyway. To use your own Codex or Gemini model, set it with `maestro models set`.
39. **`maestro install` reads the inventory** (cached, or detected when missing) to pick the OpenCode provider, so it can do the slow detect once.
40. **Newer-in-family** in `--check` ignores models that are deprecated, that have `tool_call: false`, that start with a different word than the checked id (`deep-research-*` shares the `gemini-pro` family but is not a newer Pro) and `*-latest` aliases. `gemini-flash-latest` and `gemini-flash-lite-latest` are aliases, so they pass even though `gemini-3.8-flash` is newer.
41. **Line count.** `maestro.py` is 2,124 lines, 200 over the 1,924 it had (the `models` command, `--check` and the editing commands are about 140 of that).

### v3, Stage 1 (package, providers, clients, quota)

42. **Package, not a file.** `maestro.py` became `maestro/router.py` with no logic change; the local-delegate argv is `[sys.executable, <package dir>, "run", ...]`. `maestro_model.json` moved into the package and ships as package data (the old `share/maestro` search path is kept for old wheels). Tests and evals import `maestro.router` as `m` because they patch module globals.
43. **Registry from models.dev.** The whole models.dev provider list (225) is the registry, so most providers have no hand-written code. `data/providers.json` is regenerated by `evals/build_providers.py`; `API_OVERRIDES`, `NEEDS` and `EXTRA_PARAMS` in `providers.py` cover what models.dev does not say. Anthropic-style base URLs include `/v1` (the clients append `/messages` and `/models`).
44. **Detection is presence.** A provider is "detected" when its `need` group is satisfied, and only a live listing makes it "connected". Providers that share a variable (`MINIMAX_API_KEY`) are all detected. `azure` lists models from `providers.azure.models` in config.json.
45. **`clients.py` is a separate module** from `providers.py` (the spec sketched one file). Anthropic `prompt_tokens` counts cached input tokens too. `tool_choice` is translated for Anthropic though the spec did not list it.
46. **New config keys:** `providers` (per-provider `base_url` and `models`), `request_timeout_s` (120), `cli_timeout_s` (300). New router commands: `keys`, `detect --models`, `install --statusline`.
47. **Line count.** `router.py` is 2,151 lines (mostly unchanged, plus the parser entries for the new commands); new code lives in `providers.py`, `clients.py` and `quota.py`.

### v3, Stage 2a (catalog, scoring, discovery)

48. **Three modules, not two.** The spec named `catalog.py` and `score.py`; the command, model map, backtest, live probe and hardware / fit
    code live in `discover.py` so that `router.py` only gains the parser entry and a five-line `models` hook.
49. **`models_overrides` is read from `config.json` directly** when the effective config lacks it (`load_config` only keeps known keys);
    `DEFAULT_CONFIG` is untouched so the parallel server stage can add its own keys without a conflict. `load_models` accepts a `cfg` that
    has the key.
50. **Prototype changes.** The prior is capped at 0.40 beyond 18 months; axes are averaged with per-axis reliability `c/(c+1.5)`
    instead of equally; benchmarks of labs with fewer than three benchmarked models count 0.5 (in `n` and in the axis means); the
    normalizer no longer strips `-thinking`, `-turbo` or `-chat` (they name different models). The specialist "lift" rule (0.12 above G)
    is implemented as specified but fires on no model of the current snapshot; specialists come from naming.
51. **Non-chat models are dropped, not scored.** The snapshot keeps only models whose output is exactly text (Gemini image models that
    also print text are out); `models.json` skips ids the provider lists that look like embeddings, speech or image models.
52. **No Hugging Face API from the build sandbox.** `huggingface.co` was unreachable; totals came from the Hub model pages (table
    `evals/hf_params.json`) or names. The build script uses the API first when it can reach it (`--no-hf` disables it), so a maintainer's
    rebuild records `ps: "hf-api"`. Active parameters are known for 65 of the 189 open-weight models (names and model cards); the rest stay unknown.
53. **Local `serves` may reach `strong`** when G says so (the v2.1 size rule stopped at `standard`); pins and `local_models` still override.
54. **`gpt-oss-120b` scores from its prior** (the catalog has no benchmarks for it), which puts it at the fast / standard border; the join is
    right, the data is thin. A model with real benchmarks that matches by Hugging Face repo gets them.
55. **Quantization advice is generic.** NVFP4 and Q4 serve commands assume a build exists (`<repo>-NVFP4`, `<repo>-GGUF`) and say so; the
    Ollama tag table has eight entries. KV headroom is a heuristic (1 GB + 0.09 GB per billion parameters up to 120B).
56. **Backtest picker is a simplified `route`.** The proxy adds quota, latency and Jev; the offline picker uses the rules above, so its
    numbers describe the map, not the server. `:free` tiers are excluded because they are rate-limited.
57. **Line count.** `router.py` grows by 23 lines (parser entries and the `models` hook); new code is in `catalog.py`, `score.py` and
    `discover.py`.
58. **Local fit is throughput-aware** (a follow-up to review): decode speed estimated from memory bandwidth and active parameters, a speed
    factor in the ranking, `tps` / `active_params_b` in the rows. Bandwidth is a name-matched table, so unknown devices have unknown speed;
    tok/s is a single-stream estimate at 60% of bandwidth and ignores prompt processing (prefill) and batching. Active parameters are
    known for 65 of the 189 open-weight models; the others are treated as dense, which underrates unlabelled mixtures of experts.
59. **Benchmark gap-fill from model cards** (`evals/extra_benchmarks.json`, 50 models, 630 rows) changed scores: `gpt-oss-120b` and 20b
    moved from prior to bench but stay at the fast / standard border, because a percentile among recent (mostly frontier) models puts
    mid-size open models near the bottom. Third-party rows for closed models (from the Kimi-K2.5 card) lifted GPT-5.2 and Claude
    Opus 4.5; they are marked in `variant`. Variants of one Hugging Face repo share their rows.
60. **`score.pick` replaces the backtest's own picker.** The task said to step up a category when nothing qualifies; because both tests
    (job floor, own tier) only get stricter upward, that never finds a model, so the function walks down for easy tasks and otherwise
    returns the strongest model flagged `below_floor`. Backtest, same map: under-routed 17.9% -> 8.9% (the classifier's own rate),
    severe 0, requested tier served 100%, cost per 100 requests $0.19 -> $0.39, local share unchanged at 16.5%.

### v3, Stage 2b (proxy, UI, Hermes)

61. **New modules.** `server.py` (proxy, admin API, log), `onboard.py` (`hermes`, `setup`, snippets), `ui/index.html`. The parser entries and wrappers for `serve`, `setup` and `hermes` are in `router.py` and import the modules lazily, so `route` never loads them (a test asserts it). New config keys: `models_overrides`, `log_prompts`, `session_affinity_header`, `allowed_origins`.
62. **Model list.** The server reads `models.json` through `catalog.load_models` (overrides from `models_overrides` applied); before the first discovery it builds entries from `providers.json` with the same contract. The admin "Re-run discovery" button and the start-up discovery call `discover.discover`.
63. **Floors.** The spec names the category thresholds (.68/.52/.35) but not a floor for `fast`; it is .15 so a near-useless model does not take fast work. The other thresholds come from `score.FLOORS`, and the qualification is `score.pick`'s: the job strength reaches the floor **and** the model's own tier (category, `serves`, or the user's override) reaches the cell, except for specialists. So a fast-tier model with a high score for one job does not take a standard cell, which was the main source of under-routing in the backtest. The proxy keeps its own ranking (cost buckets, quota headroom, observed latency, affinity), which `score.pick` does not model.
64. **Cost ranking uses buckets.** A strict sort on price would make quota headroom and latency irrelevant between two models a few percent apart, so price is compared in log buckets (about 2x wide for `balanced`).
65. **Subscription seats** are priced at a small amount that grows with strength, not 0, so a free local model beats a subscription and the weakest sufficient subscription seat goes first.
66. **Fallback statuses** include 401, 403, 404 and 408 (a broken key or a model the account lost) besides 0, 429 and 5xx; other 4xx go to the caller.
67. **Streaming framing.** Streams use `Connection: close` with no chunked encoding.
68. **Key display.** The UI shows where a key comes from (environment or keys.env), never its last characters, which the spec allowed.
69. **`hermes --install`** treats each of `model`, `auxiliary` and `delegation` separately: missing ones are appended, existing ones are never edited and are printed for manual merging (a fresh Hermes install usually has a `model:` block, so this is often the outcome). The `context_length` given to Hermes is a fixed 128000 because the meta-models route across contexts.
70. **Admin token in the URL.** The fragment never reaches the server; the UI removes it from the address bar after reading it into `sessionStorage`.
71. **Line count.** `server.py` is about 1,240 lines, `onboard.py` 200 and `index.html` about 570 (dense lines); `router.py` grew by 36 lines of parser entries and defaults.
72. **`setup` asks for network model servers** (step 3, or `--endpoint URL`, repeatable): the URL is probed (3 s; without a port, 8000 for vLLM then 11434 for Ollama) and added to `local_endpoints`, named after the host, even when it does not answer yet. The spec's setup only detected what was already configured, which left a DGX Spark on the LAN to a hand edit of `config.json`.

### v3, fixes from end-to-end testing

73. **Shared-key providers and opt-in providers.** Where several registry providers are satisfied by exactly the same key variable (`MINIMAX_API_KEY` -> `minimax`, `minimax-cn`, `minimax-coding-plan`, `minimax-cn-coding-plan`; likewise `ZHIPU_API_KEY`, `STEPFUN_API_KEY`, `XIAOMI_API_KEY`, `DASHSCOPE_API_KEY`, ...), `providers.detect_all` counts only one: those named under config `providers`, else the primary (the id that is a prefix of the others, else the shortest). The rest get status `missing`, `held: true`, `shares_with` and the reason "shares MINIMAX_API_KEY with minimax; name it under config providers to use it"; they are not listed live, get no requests and produce no models. `github-copilot`, `gitlab`, `huggingface` (`HF_TOKEN` is usually a download token), `amazon-bedrock`, `google-vertex` and `google-vertex-anthropic` (`providers.OPT_IN_ONLY`), whose only credential is a platform variable set for other reasons (`GITHUB_TOKEN`, `AWS_*`, `GOOGLE_VERTEX_*`), are held the same way until named under `providers`. Azure needs a resource name as well as a key, so it is not on the list.
74. **The proxy skips free-priced API models.** `server.candidates` drops `access == "api"` models whose input and output prices are both exactly 0 and ids ending in `:free` (the rule of `score.pick`), with the reason "free tier (price 0)" in `skipped`, unless the request names the model or the user pinned it with a `models_overrides` category. Before, `*-coding-plan` models at price 0 won every cost ranking.
75. **Fallback diversity.** `server.run` walks the whole candidate order and stops after three real attempts. After a 0, 401, 403 or 5xx it skips the later candidates of the same provider id or the same upstream host:port (loopback hosts are compared by provider id only); after a 429 it skips the same provider id. A provider whose models all fail costs one attempt, not three.
76. **Cooldowns for auth failures and provider outages.** `quota.record`: 401/403 start a provider-wide cooldown of 600 s (reason "auth failed"); status 0 and 5xx also cool the provider (60 s doubling to 15 min, as for the model); a 429 stays on the model unless the Anthropic rate-limit headers show an emptied window (those limits are per organization). Any 2xx clears the provider's counters. `is_available` and `status` already read the provider scope, and now name the reason.
77. **`prefer_local: false` keeps the ladder off local models.** For `maestro/auto` and `maestro/<job>`, a `local` ladder cell becomes `fast` before candidates are evaluated, and the cell no longer re-enables local models. `maestro/local`, the other category names and a concrete local id still reach them.
78. **A truncated upstream stream is an error.** `clients._relay` (OpenAI-compatible) raises when the stream ends without `[DONE]` and without a chunk carrying a `finish_reason`; the proxy writes the `error` event and then `data: [DONE]` (before, the error event ended the stream without `[DONE]`), and logs status 502. A stream that only lacks `[DONE]` but did finish is complete.
79. **Dated snapshot ids join the benchmarked base model.** `catalog.resolve`: a matched entry without benchmarks whose id ends in `-YYYYMMDD` or `-YYYY-MM-DD` uses the benchmarked entry of the date-less id (`claude-haiku-4-5-20251001` scores as `claude-haiku-4-5`); `how` is `dated`.
80. **`developer` messages** are sent as `system` to every OpenAI-compatible upstream except `openai` and `azure*`.
81. **Local tier from models.json.** `router.rate_local` takes `serves` (and, for a specialist, the jobs it is lifted for) from the model's entry in `~/.maestro/models.json` (id `local:<endpoint>/<model>`) when `discover` has run, so `detect`, `route` and the proxy rate a local model alike. It is a plain file read cached by mtime; `route` still imports nothing else. `models.local` pins, `local_models.<name>.serves` and `specialists` still win; the size heuristic is the fallback.
82. **Small fixes.** The listen queue is 128 (`request_queue_size`), `serve` flushes its banner (it never appeared when stdout was redirected) and `maestro ... | head` ends quietly on a closed pipe (`__main__`).

### v3, Maestro uses what is served

83. **The product rule.** Maestro never downloads, loads, starts or installs a model. It gauges the hardware, looks at what is already served locally or on the network, and uses exactly those; suggestions appear only when nothing is served. The spec's earlier text (local fit as an addition to what a box serves, setup "start a local server") did not say this; it is now in Goals, §3, the local-fit section, the README, and the UI.
84. **"Installed" is not "served".** `router.probe_endpoint` also reads the loaded state with GET only: Ollama `/api/ps`, LM Studio `/api/v0/models` (`state`), and treats other OpenAI-compatible servers as serving what `/v1/models` lists. Each model carries `loaded`. An Ollama whose `/api/ps` does not answer (before 0.1.38) counts as nothing loaded: the safe side. A `/api/v0/models` answer without a `state` field is ignored (some fake or proxy servers answer every path). The loaded state is as old as the inventory (24 h) or the last `discover`; Ollama unloads idle models after five minutes by default, so `maestro detect` or `discover` refreshes it, and `maestro run` always checks live.
85. **Not loaded means not routable.** `_candidates` skips such models (so the skill `route` and `models` never pick or list them as local), the proxy skips them with a reason (`NOT_LOADED`), leaves them out of `GET /v1/models` and answers a concrete request with 409 `model_not_loaded`; `discover` records `loaded` and `routable` (independent of `enabled`), the backtest and `--live` ignore them, and `--live` sends at most one request at a time per local endpoint. `maestro run` re-checks live and refuses unless `--allow-load`. The config key `local_allow_load` (default false; a file-only setting, not in the UI's `config` endpoint) makes installed models routable, with the memory risk documented.
86. **Suggestions replace, they do not add.** Plain `discover` prints one line per box (serving N models - using these, or nothing served and where the advice is). `--local-fit` on a box that serves models says "would REPLACE", counts the memory as in use, and prefixes each command with "stop the current server first" (in `models.json` too, with `replaces`). The `local_serving` block only lists routable models. This changes the v3 stage 2a text, which showed the fit next to "serving now" as if a second model could be added (a second vLLM server on a Spark that already serves one can run it out of memory).
87. **`setup`** prints the gauged hardware of each box (name, memory, bandwidth, "assumed" when guessed) and what it serves; with nothing served anywhere it prints the top three fits for the strongest box as advice. It starts nothing. Wording that told users to "start a local server" is now "point Maestro at a model server you already run".
88. **Line count.** `router.py` grew by about 27 lines (the loaded probe, the candidate and `run` guards, one CLI flag); the rest is in `discover.py`, `server.py`, `onboard.py`, `catalog.py` and the UI.

### v3, Stage 3 (S1 and expected-utility routing)

Spec references are to [docs/S1-SPEC.md](S1-SPEC.md). Items 89-96 are the integration; 97-131 fold in the module notes.

89. **Line targets.** `router.py` is 2,130 lines (2,253 before the stage; net −123). The spec's "about −300" assumed the route hook
    adds 30 lines; the hook, `_decide`, the policy id and the private rule add about 120, and moving NB (121 lines) and Jev (192)
    out removes 313. Over target: `s1.py` 419 (≤ 380; the proxy-context namespaces, material typing, hints, `verify`, decide.json
    overrides), `decide.py` about 725 (≤ 450; `seat_of` for three candidate shapes, the plain-file readers `route` needs because
    it cannot import `quota`, `learn_k`, defensive reads of user files; the readers, about 75 lines, could move to their own
    module), `systemone.py` 495 and `learn.py` 487 (≤ 300; proxy helpers, CLI handlers, `strategy_of`, k learning,
    pydantic-compatible validation). `commands.py` (about 145) dispatches to the handlers that stayed in those modules.
90. **`UNDER_COST` is superseded by F/V when `decider == "utility"`.** The implied penalty for under-routing is `1 + F/V` per task;
    the ladder and the legacy decode keep `UNDER_COST = 1.5`.
91. **Affinity on every turn, with a switching cost.** The proxy stores affinity on every successful turn (with μ_b, σ_b). Under
    `utility` a tool turn is not pinned any more: the warm seat pays cache reads on the prefix, others pay full input price plus any
    cache-write premium, so stickiness follows from the prices. Under the ladder the tool-turn rule stays as it was. `prefix_in` is
    `est_in − tokens(last user message)` (spec §4.2), not "tokens before the last user message" (§12.2): for a tool turn the tool
    results after that message are cached on the warm seat too.
92. **`route` writes `decisions.jsonl`**, but only from `maestro route` (`log=True`) or with `--explain`, which is also when shadow
    mode runs: an in-process `route()` call (evals, backtests, `mine.py`) pays nothing for S1 unless the decider is `utility`.
93. **The tool-job gate** (`decide.json:agentic_ready`, false) keeps agentic skill-mode tasks on the ladder with S1 as shadow; the
    reason says "tool job: the ladder decides until agentic_ready (utility would pick X)". The proxy has no such gate (the client
    harness runs the loop for every candidate, so η cancels; spec §3.3).
94. **The proxy's decision row is written early.** The header must carry the row id (`decision_id`, an integer: proxy outcomes key
    on it), so `run()` inserts the row as soon as a seat answered (or all failed) and `finish()` updates it with tokens, cost,
    status and total time. A request whose client disconnects mid-stream keeps the row with status 499 as before.
95. **Private mode under the ladder.** The spec states privacy for the decision layer; `private: true` is a user setting, so the
    ladder honours it too: in the proxy only local candidates remain (else 503 `private_no_local`), in skill mode a non-local seat
    becomes `inline`. The automatic key-like-string trigger applies only under `utility`, so the default routing does not change.
96. **Smaller integration choices.** `models.json` gains additive keys (`theta`, `theta_link`, `axes8`, `axes_n`,
    `systemone_strategy` per model; `catalog_sha` at the top); `MODELS_VERSION` is unchanged. Utility routing applies to `maestro/auto` and `maestro/<job>` only (a category or a concrete
    model keeps its meaning). With `TYPESAFE_API_KEY` the Jev classification still runs and feeds the posterior through the Jev
    adapter; the EU ranking decides. `decide`'s trace also carries `V` and the chosen option's `eu` (exploration needs V; the
    `eu` column). The Bayesian update after a failure (§3.7) is reported as `escalate_to`; the next route of the same task does not
    yet start from the updated (μ, σ). Any exception in the decision layer leaves the ladder's choice and adds a note (skill and
    proxy). `route --explain` shows the posterior even under the ladder. `docs/S1-SPEC.md` is the final spec, committed so the
    code's "spec §" references resolve. Regexes in `s1` (the CJK class, 1.4 ms) and `decide` (the key patterns) compile on first
    use: `maestro route` CLI wall time is +2 ms at p10 and +5 ms at p50 over the base (spec: +3 ms), most of it the two imports.
97. **Tokenizer** (spec §2.2). `https?://\S+|` is prepended to the spec's pattern (as written, `<u>` could never match). Tokens are
    taken from the original-case instruction and lowercased one by one, so camelCase identifier parts survive; only the kept tokens
    (first 192, last 64) are post-processed and `findall` counts the rest. Non-ASCII letters are kept (the fix of today's
    `_TOKEN_RE`, which the rules classifier still uses).
98. **Unfenced material.** JSON lines may start with `}` or `]`; indented lines continue a stack-trace or diff run (Python
    tracebacks interleave source lines, unified diffs have context lines); the detector runs only on texts of ≥ 5 lines. An
    untagged fence is typed by the majority line pattern of its first 20 lines, else `code` at ≥ 1% symbols, else `prose`.
99. **Feature details.** A feature is `ns + "\x1f" + content`; bucket ids are deduplicated; `ood` is 1.0 with no `w` and no `j`
    features; `s:steps=3` means three or more; `s:sym` buckets at 1, 3, 6 and 12% symbols; `x:in` from `len/3.6` (`/3.0` with
    fences) unless `est_in` is given; `h` features only with proxy context. The Bloom filter is built from the fit rows (closed
    namespaces after one occurrence); rows no inserted feature maps to stay zero, so Bloom false positives add 0 in training and
    inference alike.
100. **Quantization.** Scale = 99.99th percentile of |E'| per column / 127, falling back to the column max when clipping breaks the
     parity bound (Δlogit > 0.04 on sample rows); at bootstrap size clipping gave 0.08 against the 0.05 limit. The header's
     `quant` records the choice. SWAR lanes and the dense block are decoded with `struct` (endian-explicit), and `s1` imports
     `operator`, which is not on §12.1's list.
101. **Bootstrap `s1.bin`** is trained on `evals/train.jsonl` only (job and difficulty), so `trained = ["b", "job"]` and the other
     heads come from job priors (tools by job, detectable by job, stakes 1, self_contained 0.5, ambiguity 0.5, tok_out by job with
     sd 0.8, agentic turns log 6 ± 0.8). Rows without a split use sha256(cluster) % 100: train < 75 ≤ calib < 90 ≤ dev (the spec's
     55/10/10/25 assumes the full corpus). Grid search (5-fold grouped CV, held-out joint loss, since the spec's criterion needs
     outcomes) chose B = 2^14, d = 32, dropout 0.3, keyword features on, char 3-grams off. Placeholders until outcomes exist: the
     b sd prior (0.8), the OOD inflation γ (0.8), λ_d (0.5) and the final success calibration (Platt 1, 0).
102. **§8.4 fails for the bootstrap**, so `classifier: auto` stays on Naive Bayes and S1 feeds the decider only. On the dev half of
     `evals/prompts.jsonl` at matched under-routing (U = 4.5): S1 job 82.7% vs NB 84.4%, category 51.5% vs 58.7%, under-routed
     8.4% vs 8.9%. `train_s1.py --gate` reads only that dev half (to choose U and report); training never reads the file except
     to drop leaked prompts. The result is in the header's `legacy` key, which `route` reads.
103. **Legacy decode** uses one Platt pair shared by the three thresholds (`P(d ≥ k) = σ(A·(μ_b − τ_k)/s_d + B)`), fitted on calib
     (A = 0.73: the raw ordinal was overconfident); per-threshold pairs crossed and gave zero-probability classes. NB pooling's
     weight is chosen against an NB refit on the train split, because the shipped NB has seen calib. Gradients are analytic for
     job and the binary heads and central differences for the ordinal, token, turn and IRT scalars (Φ by Abramowitz-Stegun
     7.1.26); `--check` requires ≥ 99.9% identical int8 bytes (got 100%), ≤ 0.5 pt metric drift (0), max |Δlogit| ≤ 0.05 (0.022)
     and bit-exact integer bag sums. `ordinal.b_quantiles` holds 99 values, not 5 (systemone reads k/n quantiles).
104. **Default catalog link.** Without a fitted header link, decide uses α = 1.54 (both groups) and β = 2.58 (`decide.json:
     s1_defaults.link`), not Appendix A's α −2.0/−2.4 and β 1.6, which put every catalog model, Claude's own included, at
     θ −3.1 to −1.3 against anchors at 0 to 3.3, so utility routing could never pick a Codex, Gemini or local seat (§0.3 sources 3
     and 5 must work even on a no-go). The interim line is the least-squares fit of the catalog's own category thresholds
     Φ⁻¹(0.35, 0.52, 0.68) onto the anchor-tier midpoints 0.55, 1.65, 2.75; a test guards it. A header link marked
     `fitted: false` is ignored; `train_s1.py` / `fit_link.py` must write the fitted link.
105. **decide.json additions** beyond Appendix B: `_placeholders`, `category_price` ($/M at the Claude 1:2:4:10 ratio, for a seat
     with no price and no API seat of its category among the options), `s1_defaults` (Appendix A's header values for any key the
     header lacks).
106. **Reference implementation.** `decide_ref.py`'s `gh_update` scales the probabilists' nodes by √2 and gives (1.188, 1.145) for
     the §3.7 check; decide follows the spec formula (0.724, 0.884, which matches a numerical integral). Its `p_succ` at zero stakes
     returns P; decide's P̃ is continuous in z. Everything else matches the reference: `p_succ` on 300 random cases, `g_mult` on its
     four rows and a 400-point grid (m = 0.20 for weekly windows), `tokens_agentic`, `api_cost` and all 12 EU values of its relaxed
     and pressured scenarios.
107. **Hints are fused once,** in `s1.posterior` (§5.6); decide reads only `context_tokens` (the inline seat's context) and echoes
     the hints.
108. **θ sources, in order:** an `anchors.json` seat; an anchor tier (Claude seats named haiku/sonnet/opus/fable, a model or
     `catalog_id` matching `irt.anchor_models`); models.json `theta` stamped with the current link; the link from `jobs` or `G`;
     `decide.json:seat_prior`; the anchor tier of the seat's category with the prior sd (`local` one step below haiku). The job
     group of each job picks θ_{t,g} and α_g; the posterior's group picks a_g, Cal_g and η.
109. **Seat access and quota.** Claude Code, Codex and Gemini CLI seats are `subscription`, opencode `api`, router local candidates
     `local`, `inline` follows `inline_access`. The substitute for `C_sub` is the cheapest API seat with θ̄ ≥ θ̄_s − 0.3, priced on
     s's tokens (`g_max` 1.0 when one exists, else 3.0). Codex with no `quota_hist.json` is unknown like Claude with no data
     (g = 0.25); Codex snapshots are not aged (a rollout changes only when Codex runs). `quota_unknown` is flagged for an unknown
     Claude or Codex option, or an unknown chosen seat; Gemini (always unknown) does not flag every decision. Shared windows are
     five_hour, seven_day and Codex primary; `seven_day_<family>` binds only that family; an unknown reset uses τ = L/2.
110. **Personal files.** `personal.json` Platt composes after `Cal_g`; `seat_offsets` keys are `anthropic/<tier>|<group>` for
     anchors, else `<seat id>|<group>`; `k` keys `<prov>/<window>|<family>`; `verbosity` by vendor. Values are clipped
     (|o| ≤ 1, |Δb| ≤ 0.5, k in [0.25, 4] × prior) and malformed files ignored.
111. **Effort.** An unset effort on a capable seat (Codex, `reasoning: true` entries) counts as medium; the reasoning-token
     multiplier applies only when the entry says `reasoning` or the seat carries `r`. `with_efforts` expands seats per effort but
     is not used by `route` or the proxy in v1 (the configured effort is the seat's).
112. **Local latency** uses the proxy's TTFT EWMA (it already contains queueing, so it stands in for `κ·max(0, ttft − ttft_idle)`);
     an unknown tps uses the category prior of `serves`; prefill counts on the uncached input.
113. **Confidence** uses P̃ in the variance term and is 1.0 with one option; when the top seat is there by the floor rule it can be
     below 0.5. Isotonic knots are `[xs, ys]` on P_raw. Turns are clamped to ≥ 1; empty `job_p` falls back to `{"ask": 1}`.
114. **Snapshots and logs.** `read_windows` also reads `claude_limits.json` (the log lines carry only percentages; resets come
     from there; epoch numbers and RFC 3339 are parsed with a stdlib civil-date formula). `quota_hist.json` gains a `current` map
     (skill mode cannot import `quota` to learn the Codex reset and window length). decisions.jsonl rows add `escalate_to` and
     `trace` (μ_b, σ_b, θ̄, sd_s, a, group, API $, quota, family, the personal offset `o` and b shift `db` that were applied, V,
     eu): §10.3 refits from them and §4.4 needs the predicted API $.
115. **Key-like strings** (`_SECRET_RX`): `mine._REDACT` split into literal-prefixed patterns with the word-boundary check done per
     match (0.3 ms on 60 KB; the single alternation took 11.7 ms on 214 KB); a fuzz test checks they match exactly what the
     originals do. The whole text is scanned. `decide` does not import `router` (local `TIERS`, `ANCHORS`); a test checks its
     imports stay within §12.1's list.
116. **MinHash** is crc32 of each word 3-shingle under 16 fixed permutations `(a·h + b) mod p`: seeded crc32 would not work (CRC is
     affine, so seeds are not independent hashes). `split.py` uses its own seeded MinHash for clustering (seeds
     `crc32(b"maestro-minhash-%d" % i)`), which never meets the log's.
117. **systemone: the `model` intent.** Names resolve to the router's seats for the task, plus the seats a chat task would get,
     plus models.json ids (the proxy passes its own hard-filtered candidates). A seat that fails this task's hard filters still
     resolves (the question stays native) with P(best) 0; nothing eligible is 422 `no_seat_qualifies`. `probabilities` =
     `decide.p_best` over 64 draws, `choice` = the EU argmax, `confidence` = its P(best), `decision_id` = sha256(state, name)[:8]
     so answers are deterministic. `success` is P (not P̃) of the named seat, 0 when decide skips it.
118. **systemone: scores.** `difficulty` uses σ_b (already OOD-inflated) and the header's `b_quantiles` read at p = i/(m+1).
     `stakes` and `ambiguity` are Binomial(2, E/2) over their three levels unless the engine adds `stakes_p`/`ambiguity_p`; both take
     2-4 levels (the spec gives no range for ambiguity). `tokens` parses legends with `k` suffixes and open ends ("<100", ">3k"),
     cutting at geometric means, else default buckets; a `tok_out[0]` > 20 is a count, not a log. Confidence: the chosen option's
     probability (choice) or the most likely level's (score).
119. **systemone: delegation.** Before the first upstream request the delegate goes through `router.probe_endpoint`'s loaded check
     (the `maestro run` rule, `local_allow_load` included); requests go through `clients.complete` (the key only as its own
     provider's auth, upstream text scrubbed). Votes: mass = valid votes / 8 requested, first word matched case-insensitively; BPE
     `Ġ` and SentencePiece `▁` marks stripped before case and space variants are summed. Cache key: sha256(delegate, strategy,
     prompt messages including the question name; the state cut to 6,000 characters); entries keep raw probabilities and the
     calibration temperature is applied on read; `s1_cache.json` (0600) also holds the probed strategy (models.json
     `systemone_strategy` wins). `systemone_calib.json` holds temperatures on a 0.5-5.0 grid. The response `model` is `maestro-s1`
     or `maestro-s1+llm`; `usage` counts fresh delegate tokens only. Error codes: 422 `model_not_found`, `unknown_intent`,
     `intent_mismatch`, `question_not_supported`, `delegate_unsupported`, `too_many_options` (> 26 choices or 10 levels),
     `too_long` (> 64 questions); 503 `delegate_unavailable`, `engine_unavailable`; 502 `delegate_error`. Intents outside
     `maestro.*` are ignored.
120. **learn: the fit.** `p = σ(A·(A_g·z + B_g) + B)` with `z = a(θ̄ + u·o_key − μ_b − Σ_j q_j·Δb_j)/√(1 + π/8·a²(σ_b² + sd_s²))`
     from the stored trace (minus any `o`/`db` that were active, so a refit replays rather than increments); isotonic maps are not
     replayed. At most 27 offsets + 11 Δb + 2 Platt = 40 parameters, Fisher scoring with step halving (≤ 25 iterations).
121. **learn: guards.** Time split: fit on the first 80% by time, compare held-out log loss with the global model's, write nothing
     if worse, else refit on 100% (≥ 10 outcome rows, ≥ 2 held out). Replay guard: `data/alpha_dev.json` when the release ships
     it, until then a calibrated reference (logit ~ N(0, 1.5²), y = σ(logit)), in effect a 0.02-nat bound on the personal Platt.
     Small data: each parameter has its own weighted count; under 30 it moves at most 0.2 from its prior mean. Reset: a row stops
     informing its seat's offset when its policy differs from the current weights or its θ̄ moved > 0.3 for the same seat and top
     job. One term per (decision, signal), latest wins; weights explicit 1, tests 0.6, escalate 0.7, implicit 0 until an audit with
     n ≥ 20 and precision ≥ 0.8 (then that precision).
122. **learn: signals.** Skill-mode escalate is derived at replay time (route cannot import learn): a later decision within 15 min
     with MinHash Jaccard ≥ 0.6 and a higher θ̄, a higher p in the earlier options, or a stronger `stakes`/`size` hint. The hook
     reads the last 200 lines of a ≤ 1 MB transcript tail, the last summary wins, an exit code is the fallback, the target is the
     latest subagent (or `claude-code:`) decision of the last 30 min, and it always exits 0. `outcome --tests P/T` with a verdict
     is detail; without one it is a `tests` signal. `k` is learned per window and family from `read_windows` + `learn_k` and
     written only when it differs from the prior; `verbosity` is not learned (predicted tokens are not logged) and is carried
     over. `personal.json` (0600) rotates `.1`-`.3`; `--reset` removes all. `learn --audit` stores counts only (`audit.json`).
     Exploration clips ε to [0, 0.1]; the propensity is 1−ε or ε, 1 when the conditions fail. `learn.run` takes 1.2 s on 10,000
     rows (the estimate was < 50 ms); it runs daily in a thread, so it is not gated.
123. **Evals: splits.** LSH candidates come from 32 bands × 2 rows (a superset of the spec's 16 × 4; union threshold 0.5): with
     16 × 4 a pair at Jaccard 0.6 is missed 11% of the time. Cluster ids are `"k:" + smallest member id` (`"<letter>-" +
     sha256(text)[:10]` for rows without one); a cluster holding a T row is train, one holding an L or P row never is, and a T row
     clustered with an L/P row is a leakage error. The manifest holds `{id: [corpus, split, cluster]}` and file hashes;
     `split.py --check` (CI) verifies them in < 1 s.
124. **Evals: the panel.** 20 items + 2 honeypots per call under opaque keys, each clipped (head 6,000 + tail 2,000 characters);
     never shown: id, slice, hint, other labels (except to the adjudicator), Maestro's prediction. Ingest coerces booleans and digit
     strings, drops zero job entries, cuts rationales to 25 words, re-batches invalid items. Dawid-Skene on job top-1, difficulty,
     min_tier, stakes, ambiguity, token buckets and the binary fields; fable's adjudication enters one-hot at 0.7. Gold items appear
     only as honeypots.
125. **Evals: the alpha set.** Grading covers each grader's ungraded answers per phase; criteria are written once per item;
     self-preference replacement needs fable's vote; Markdown is normalized outside fences before grading. Outcomes use y ≥ 0.5 for
     `t_star`; phase-2 inverse-propensity weights are 1 (disagree), 4 (sampled), 0 (unsampled). `irt.json` is a 1PL (a = 1,
     θ_haiku := 0); the joint fit with a_g is `train_s1`'s. Release assets (answers, grades) live in `evals/work/` (ignored) with
     their hashes in `evals/alpha/assets.json`.
126. **Evals: policies and gates.** B6 (Jev), B7 and R-noσ enter as prediction files; R, R-nb and R-lad are `route()` configs and
     count only when the decision echoes the requested decider and classifier; proxy policies run `server.App.plan` with no
     upstream call. The CI gate uses the utility decider when `route()` supports it (it now does) with V0 from
     `evals/alpha/v0.json`; checks that need outcome data print SKIP until it exists; a test item without corpus text fails the gate.
     Quota scenarios write `claude_limits.json` and `.log` into a temp home. Release sensitivity covers the knobs that are config
     keys (`latency_usd_per_s`, `risk`, `wtp_usd`); ℓ, ω, g_max and m_w print n/a. `--apply` writes `agentic_ready`.
127. **Evals: pacing.** `alpha.py pacing` now prices through `decide.quota_price` and `windows_of` (the reference copy it started
     with is gone). Per-task API $ is scaled × 10 by default (`--usd-scale`; chat answers alone never pressure a $50 window); seats
     substitute at the same tier, so the "success within 1 pt" check holds by construction.
128. **Evals: scope.** The alpha report computes job accuracy and token error from decisions that carry a posterior; OOD AUROC,
     tools F1 and turns MAE are left to `train_s1`'s offline report. `evals/swe_text.py` (§7.1, optional) and
     `evals/fetch_public.py` are not built. `fit_link` on the public data: β_pub 0.92 (se 0.78) from 45 matched models; leave-one-model-
     out AUC loss 0.084, above the 0.02 target (reported, not gated).
129. **Orchestrator protocol** (subagent steps). Batch files `evals/work/<step>/<prefix>-<hash>.jsonl`, one call per line
     `{call_id, model, prompt, expect}`; replies appended to `<batch>.replies.jsonl` as `{call_id, reply}` (optional `error`,
     `tool_calls`, `invalid`); item ids and answer keys only in `_meta.jsonl`; an invalid call is re-issued as `<id>:a2`; at most 8
     at once, paused while the five-hour window is over 70% or the weekly over 60%.
130. **Corpus placement.** The generated corpus and the verifiable items still have to be copied to `evals/corpus/*.jsonl` and
     `evals/alpha/verifiable.jsonl` and frozen with `split.py --write`; the CI gate fails a test item with no text once outcome data
     is committed.
131. **Tests.** New: `test_s1.py` (golden outputs, hashing under different `PYTHONHASHSEED`, SWAR, caps, CJK, fallbacks),
     `test_decide.py` (reference formulas, pacing, escalation, privacy, p99 ≤ 2 ms on 40 seats), `test_learn.py`,
     `test_systemone.py` (fake logprob servers on 127.0.0.1; the typesafe-sdk shape test is skipped without the package),
     `test_split.py`, `test_label.py`, `test_alpha_replay.py`, `test_alpha_pipeline.py` and `test_s1_integration.py` (route under
     both deciders, the CLI commands and hook, the proxy's utility planning, switching cost, feedback, signals, `/api/alpha`,
     `/v1/systemone`, discover's θ). The forbidden-import tests now include `learn` and `systemone` (and a utility route imports
     none of `providers`, `clients`, `quota`, `catalog`, `score`, `discover`, `server`, `onboard`, `learn`, `systemone`);
     `test_latency.py` adds 200 utility routes in under 1 s.

132. **Review fixes (S1 build, three-lens adversarial review).** Choices made while fixing the 19 confirmed findings, kept verbatim below.

    in process on CPython 3.11 in this container.

    ## Trainer to runtime contracts

    1. **Isotonic map.** The contract is the one DESIGN.md already states: `calib.final.<group>.iso = [xs, ys]`, with xs = P_raw
       ascending in [0, 1] and ys = P.
       - `train_s1.isotonic` now returns that list. `calibrate` fits it on P_raw directly, not on the logit.
       - `decide._iso` checks the shape: two equal-length lists of at least 2 numbers in [0, 1], with xs ascending. Anything else,
         including the old `{"x", "y"}` dict, is ignored and Platt is used.
    2. **The link file.** `fit_link.py` still writes `beta_pub` as a number with a separate `se_pub`, and `train_s1` accepts that
       or a `[b, se]` pair.
       - fit_link now also writes `anchor_models` (decide.json's anchor tiers), `zbar` and `a_chat_hat`. `zbar` is, per tier, the
         mean over jobs of Φ⁻¹(clip(catalog strength)). `a_chat_hat` is the shipped header's a_chat, else 1.2.
       - `train_s1.linked(link)` is true only when `zbar` covers all four tiers. α and β are free only then.
       - Whenever any anchor θ is trainable, α and β join the active set, because the prior θ ~ N(α + β·z̄, 0.5²) ties them
         together (the joint MAP of §3.4 step 3).
       - The header's `link.fitted` is true only when α and β were free and the data moved them. That needs outcome or
         `min_tier` labels. A bootstrap trained on train.jsonl alone stays `fitted: false`, so decide keeps using decide.json's
         interim link.
       - "Iterate twice" is implemented as a full second training pass, with `a_chat_hat` set to the a_chat from the first
         pass. Training takes about 2 s.
       - The header also records `link.a_chat_hat`. Without a link, the header's `beta_pub` placeholder is now [1.0, 0.5], the
         fallback the spec states.
    3. **V0.**
       - `train_s1.py --v0 evals/alpha/v0.json` writes `decide.V0` into the header during a training run.
       - `--v0 FILE --patch [--golden]` rewrites only the header of an existing s1.bin. The sections are kept byte for byte and
         their sha256 values are recomputed.
       - An objective whose calibrated V0 is null keeps its placeholder. `V0_calibrated` is true only when all three objectives
         were calibrated.
       - Patching changes the file's sha256, so the golden file has to be regenerated (`--golden`).
       - The CI gate (`alpha_eval.current_policy`) and the R policies in `report` and `release` no longer inject `wtp_usd`. They
         replay at the header's V0, as production does. Sensitivity varies `wtp_usd` around the header's V0.
       - `calibrate-v` and the report's V sweep still set `wtp_usd`, because they sweep it.
       - New check "V0 in s1.bin": skip without v0.json, fail when v0.json and the header disagree. The CI gate runs it (under the
         utility decider), and the release gate requires it to pass.
    4. **Replay quota (§9.1, Q = 0).**
       - `route_policy` writes `relaxed` Claude limits and relaxed Codex windows (`quota_hist.json` `current`) into its temporary
         home. It rewrites them every 10 minutes so they never go stale on long sweeps.
       - A replayed decision flagged `quota_unknown` raises RuntimeError.
       - A caller-supplied `home` is used as is. `write_limits(..., codex=True)` is the new option. The quota-monotonicity
         scenarios are unchanged.
    5. **Splits.** `train_s1.assign_splits` replaces the trainer's own split function. It picks a row's split in this order:
       - the row's own `split`;
       - else the frozen manifest, looked up by the row's id or by the id split.py gives an id-less row (`<letter>-sha256[:10]`);
       - else split.py's clusters over the training rows (MinHash 5-shingle unions at J ≥ 0.5, plus scenario, persona,
         `cluster` and source_id), hashed with the bootstrap ranges train < 75 ≤ calib < 90 ≤ dev. These are deviation 101's
         ranges, now over split.py clusters instead of the exact instruction.

       Manifest rows marked test, legacy or persona are never used. Once a manifest exists, every train.jsonl row is `train`
       (split.py's role rule), so calib and dev must come from the panel corpora. Rows missing from the manifest are split by
       cluster, with a warning.

       On train.jsonl: 1,176 train / 219 calib / 141 dev rows, and no train vs calib/dev pair at estimated J ≥ 0.6 (there were
       153 pairs before). The shipped s1.bin was **not** retrained in this step, so its header metrics still come from the old
       split. A retrain with the new splits gives dev job accuracy 0.8085 (n = 141). `train_s1.py --check` will report drift
       against the committed file until it is retrained.
    6. **Burn rate.** `windows_of` appends `(min(now, as_of), used)` to the history before the slope. This carries the last known
       value forward to the snapshot time (Claude's `updated_at`, Codex's latest sample), not to "now". Without new samples, the
       slope reads no burn.

    ## Privacy

    7. **Jev never sees a private task.** It is skipped when `private` is on or the text holds a key-like string, under either
       decider.
       - Skill mode adds "jev skipped: the task is private (...)" to the reason.
       - The proxy skips Jev silently. Its key check covers every forwarded message (item 10).
    8. **The private filter no longer depends on the decider.**
       - Proxy: when `private` is on (or, under utility, when there is a key-like hit), `ranked` is filtered to local models before
         decide runs. A decide failure or an empty order then falls back to a local-only ladder, or to 503 `private_no_local`.
       - Skill mode: the private-to-inline rewrite of the ladder seat runs before `_decide` for both deciders. The seat that stands
         when decide fails, or when it keeps a tool job before `agentic_ready`, is therefore inline or local.
       - The automatic key-like trigger still applies only under utility, as in deviation 95.
    9. **/v1/systemone.** A cloud delegate is refused with 503 `private_no_local` when `private` is on or when the state's JSON
       text holds a key-like string. Local delegates are unaffected.
    10. **Whole-conversation scan in the proxy.** `features()` keeps the text of every forwarded message (system, all turns, tool
        results) plus the JSON of tool calls. `server.keylike(f)` scans it once per request (memoized), and decide gets it as
        ctx `"private"`.

        There is no per-conversation cache. Each request forwards the whole history, so the scan covers whatever is sent. A turn
        the client drops is no longer sent.

    ## Bounded work

    11. **`is_private`.**
        - Each pattern starts with its literal (so the fast prefix search is kept), followed by a fixed-width lookbehind for the
          \b check.
        - Quantifiers with nothing after them are bounded (`{16}` finds what `{16,}` finds).
        - eyJ is scanned one candidate per `[\w-]` run. A later start in the same run has the same tail and a shorter head, so it
          cannot match if the first one failed.
        - 60,000 random strings give exactly what `mine._REDACT` gives.
        - Timings: `'xeyJ'*15000` 0.7 ms (was 1.5 s), `'xeyJ'*30000` 0.9 ms (was 5.7 s), `'xsk-'*15000` 0.6 ms (was 0.67 s),
          `'-eyJ'*15000` 0.8 ms, `'eyJ'*20000` 0.4 ms (was 1.9 s).
    12. **`parse_tests`.**
        - Counts are 1 to 9 digits, not inside a longer number (`(?<!\d)`).
        - `Tests:` is located with str.find.
        - `int()` can no longer overflow.
        - Timings: `'1'*20000` 1 ms (was 2.6 s), `'Tests: 99999 '*2500` 3 ms (was 1.2 s).
    13. **Large inputs.**
        - systemone hashes the state once per request: 64 native questions on a 4 MB history state take 0.011 s (the review
          measured 5.5 s for a 20 MB state before).
        - `minhash` tokenizes at most the first 64 KB, lowercased, and stops at 400 words. 10 MB takes 8 ms (was 1.1 to 4.2 s).
          The signature is unchanged when the first 400 words lie in the first 64 KB.
    14. **Feedback ids.**
        - The header and CLI accept 1 to 18 ASCII digits, and the API accepts ids from 1 to 2^63 − 1. Anything else gets 400, or is
          ignored in the header.
        - A failure while recording header feedback never fails the chat request.
    16. **S1 robustness.**
        - `s1.known` encodes with `surrogatepass`, since a non-UTF-8 argv byte arrives as a lone surrogate.
        - `route` wraps S1's posterior and legacy decode like `_decide`. On failure it uses the rules classification and the
          ladder, and the reason says "S1 failed (...)".
        - `s1._open` validates the header fields that posterior, legacy and decide read: calib scalars, Platt pairs, calib.final,
          ordinal tau and s_d, train.config, and legacy.U. `load` also catches ZeroDivisionError and struct.error.
    17. **Fences.**
        - `normalize` splits fences with str.split.
        - Only the first 32 fences, and only those starting in the first 10 KB of material, are typed, each from its first 8 KB.
          Later fences still count and still feed the material text.
        - The output is identical on train.jsonl, on prompts.jsonl and on 30,000 random fence mixes.
        - normalize timings: `'```'*20000` 1.4 ms (was 25.7 ms); a 60k page of 1,500 small fences 0.9 ms (was 4.3 ms).
        - In-process `route` with `decider: utility`: 8.2 ms p50 on the many-fence page and 9.9 ms p50 on the fences-only text
          (was 10 to 11 ms and 30 to 32 ms). The ladder alone takes 4 to 6 ms on these inputs.
        - `long_fixtures` now makes every fifth fixture a many-fence page.
    19. **`--context-tokens`** must be between 0 and 10,000,000 (argparse). decide clamps the hint to [0, 1e7].

    ## Line counts

    - router.py is unchanged at 2,130 lines, because route() was compacted.
    - commands.py is at its target of 150.
    - decide (+19), s1 (+22), server (+12), systemone (+5) and learn (+3) grew for the fixes. They were already over the §12.1
      targets.

133. **The ladder is retuned on measured outcomes (hard work to `standard`, very hard work to `strong`).** `DEFAULT_LADDER` in
     `router.py` was a prior; the first measured outcomes say it over-spent. On 100 checker-graded, self-contained pilot tasks
     (`evals/alpha.py` pilot), Sonnet-class models solved 97-100% of the hard (d2) and very hard (d3) items, haiku solved 82%
     overall (95%, 86%, 83% and 62% at d0 to d3), opus 99% and fable 100%. With the old table the rules router sent 77% of those
     tasks to `strong` (opus) at an average price of 4.16 (haiku 1, sonnet 2, opus 4, fable 10) for 99 of 100 solved; with the new
     table it is 98 of 99 gradable (one answer was excluded because the model used a tool) at 2.26. Changes: d2 goes to `standard` for every job except `plan` (still `strong`); d3 goes to
     `strong` for most jobs and to `frontier` only for `plan`; `review` and `debug` are `standard` at d0 to d2; `ask` d3 is
     `standard`; `ask`, `summarize` and `edit` at d0 and d1 are unchanged. Only the table changed. The tests that encoded old
     cells were updated (hard cells now use a d3 prompt where a test needs a `strong` seat), and the golden file of
     `tests/data/s1_golden.json` was regenerated with `evals/train_s1.py`'s `golden()` because `s1.legacy` decodes by risk over
     the ladder's tiers (`_RISK_TIER`): `s1.bin` and every posterior are byte-identical and only two `legacy` pairs moved (one
     `debug`/1 to `review`/1, one `ops`/2 to `ops`/1). Nothing was retrained: `maestro_model.json` and `s1.bin` do not depend on
     the table (the risk decode reads it at route time), and S1's `legacy.U` was left at its fitted value.
     Eval on `evals/prompts.jsonl` (rules classifier; before -> after). Test half: job 79.8% -> 80.2%, category 53.2% -> 65.0%,
     within one 93.7% -> 99.6%, under-routed 8.9% -> 8.0%, severe 0.0% -> 0.0%. Dev half: job 84.4% -> 84.8%, category 58.7% ->
     70.5%, within one 94.1% -> 98.3%, under-routed 8.9% -> 7.2%, severe 0.0% -> 0.0%. The expected category in the eval is the
     ladder's cell for the labelled job and difficulty, so these numbers move with the table; they measure agreement with the
     labels, not outcomes. The CI floors in `.github/workflows/ci.yml` and the README benchmark table move with them. The pilot
     is small (100 tasks, one task style), so it supports not paying for `strong` on d2 work; it does not show that d3 work can
     go lower, and the `strong` cells at d3 were kept on purpose. A user who prefers the old behaviour sets `"ladder"` in config.
134. **A public website and a read-only demo of the web UI (`site/`, Railway).** Outside the package and outside `route`: nothing in
     `skills/` imports it, the wheel does not ship it, and the router is unchanged. `site/serve.py` (stdlib, Python 3.9+) serves the
     static pages (`/`, `/docs`, `/privacy`, `/terms`, `/static/*`) with a strict CSP (`script-src 'self'`, no inline script; inline
     styles allowed), `nosniff`, `no-referrer`, `frame-ancestors 'none'`, HSTS behind an HTTPS proxy, a day of caching for
     `/static/`, ETags, gzip, an extension allowlist, no hidden files and no directory listings; `/health` answers `ok` for Railway.
     It logs method, path and status only. `/demo/` is `ui/index.html` read at request time with `demo/demo.js` injected before the
     UI's own script (the CSP adds that script's sha256): the shim rewrites the UI's `/api/*` fetches to `/demo/api/*` and supplies a
     placeholder token, so the shipped UI needs no demo hook. `/demo/api/{state,decisions,stats,alpha}` serve
     `site/demo/api/*.json` with their timestamps shifted to now; the dry run answers only the UI's five example prompts (with
     `maestro/auto`, no tools, no image) from recorded plans; every other POST is 403 `demo_read_only`. The fixtures are recorded by
     `site/make_demo.py`, which runs the real `discover` (bundled catalog, so the scores are real) and `maestro serve` over fake
     providers on 127.0.0.1 with made-up keys, then scrubs paths and ports. Deploy: `railway.toml` builds `site/Dockerfile`
     (python:3.12-slim, non-root, only `site/` and the UI file; `.dockerignore` keeps the context to those) with `/health` as the
     health check. The web UI was restyled to the site's palette (navy `#0A0F2E`, cyan `#00D4FF`, the robot mark inline) with a
     light variant; ids, tabs and behaviour are unchanged. `tests/test_site.py` covers the server, the demo API and refusals,
     internal links, the absence of external scripts, and that the install commands and the Hermes block on the pages match
     README and `onboard.hermes_yaml`.
135. **The arena (`evals/arena.py`, `evals/arena/`): harnesses and models head to head.** A repo tool outside the package and
     outside `route`. Nothing in `skills/` imports it and the router is unchanged. It imports the maestro package for seats
     (`router.seat_models`, `providers`, `catalog`, `score`), calls (`clients.complete` for API seats, `router.run_model` for
     served local models, which never loads one), `quota.is_available` / `headroom`, and the replay (`classify_rules` +
     `target_tier`, and `route()` with the utility decider in a scratch home). Choices:
     - **One argv table** (`harness.HARNESSES`) holds every harness's agentic and chat command line, with what was verified
       on the authoring machine (Claude Code 2.1.289 `--help` and a live chat run) and what was taken from documentation
       (Codex, Gemini CLI, OpenCode, Hermes: not installed there).
       - Chat seats run with tools off (`--tools ""` for Claude Code, `--sandbox read-only` for Codex, no `--yolo` for Gemini)
         in an empty temp dir.
       - Agentic seats run in a fresh temp git copy of the fixture repo, inside a cell dir that holds only `repo/`,
         `scratch/` (generated OpenCode / Hermes config, outside the diff) and `tmp/` (TMPDIR).
       - A new file in the cell dir or a change to the fixture is an escape and fails the cell. Containment itself is the
         harness's (Codex workspace-write, Claude Code acceptEdits plus a tool allow-list); the arena detects escapes and
         does not sandbox. A timeout kills the whole process group.
     - **Local models are seats three ways:**
       - `local:` chat seats, from the inventory, loaded models only;
       - `opencode:<endpoint>/<model>`, through a generated OpenAI-compatible provider config named by `OPENCODE_CONFIG`
         (the key referenced as `{env:VAR}`, never written);
       - `hermes:<endpoint>/<model>`, through a temp `HERMES_HOME`.

       `hermes:maestro/auto` drives Hermes through `maestro serve`; its cost is unknown, since Maestro decides what it
       spends.
     - **Verifiable grading.** The checker runs in `evalkit.run_checker`'s sandbox on the raw answer first, then on the last
       function block, SQL block and JSON value, each re-fenced. The same variants are tried for every seat, and `y_strict`
       keeps the raw-only result.
     - **Open grading follows S1-SPEC §7.5, with two changes.**
       - The judges also return a full ranking, and Bradley-Terry log-strengths (MM algorithm, 0.5 pseudo-wins each way per
         compared pair, cluster bootstrap over tasks) replace the two-grader-plus-adjudicator vote.
       - Judges are a panel, not "two judges from a family other than the contestant's". The panel is the strongest seat
         of each family, up to four families; a local seat wins its family's slot within 0.05 G. A panel judges all
         answers of an item at once, so per-contestant judge exclusion would break the shared ranking. Instead the report
         shows a judge-family by contestant-family matrix and each family's self-preference: own-family answers, the judge
         minus the other families' judges. It warns loudly when only one family is available.

       Every judge runs twice, the second time on the reversed order, which cancels position bias. The report also shows
       the position bias, forward-versus-reversed Kendall tau, judge agreement and length bias (within-item Spearman of
       score and length).
     - **Budget and quota.**
       - The budget is the run's total estimated spend, earlier sessions of the same `--run-id` included, checked before
         each call starts.
       - Claude Code's `total_cost_usd` (API-equivalent) is used as reported. Other CLIs are priced as tokens (or
         characters / 4) times the catalog price; local seats are $0.
       - Quota: a seat whose scope (`claude`, `codex`, `gemini`, an API provider) is cooling down, exhausted, or under
         `--min-headroom` is skipped (a `skipped` row, redone on resume), or waited for with `--quota-wait`.
     - **Resume.** A cell is done once it has an `ok` row or `--max-attempts` error rows. An agentic harness that exits
       non-zero without a diff and without timing out is an error (a login problem, not a model failure).
     - **Routing replay.** The ladder pick is the cheapest seat of the target category in the run (walking up, then down),
       overall and within each harness. It is compared with always-X and hindsight (the cheapest seat that succeeded).
     - **Export.** Seats map to the anchor tiers by category (a local model by its G). When several seats share a tier,
       the Claude seat stands for it, and every seat's own outcome is kept in `y_seat` / `seat_meta`. The rows load with
       `alpha_eval.load_items`.
     - **Run files.** `evals/arena/runs/` (answers, diffs) is gitignored.

     Docs: README "Arena", `docs/ARENA-RUNBOOK.md`, and the Claude Code skill `.claude/skills/arena/SKILL.md`. Tests:
     `tests/test_arena.py`. These use fake harness CLIs on a temp PATH (solve, fail, time out, escape, crash, OpenCode on a
     local model) and a fake OpenAI-compatible server for chat seats and judges. They check:
     - Bradley-Terry on hand-computed cases;
     - that position bias cancels;
     - resume, the budget stop and quota skips;
     - the report's numbers;
     - the export format.
136. **Arena fixtures (`evals/arena/fixtures/`).** Outside the package and outside `route`: 30 small stdlib-only Python repos
     for comparing model + harness pairs on agentic work (8 debug, 8 code, 4 edit, 3 review, 3 data, 2 ops, 2 research;
     difficulty intent 6/9/9/6 for d0..d3). Each has `meta.json`, `task.md` (the request as a user would type it),
     `repo/`, `hidden_tests/` (copied in only for grading) and a `solution/` overlay; the format and grading protocol are in
     `evals/arena/fixtures/README.md`. `evals/arena/validate_fixtures.py` (stdlib) requires, in fresh temp copies with a
     60 s limit per run, that the starting repo fails the hidden tests, the solved repo passes them and its visible tests,
     and that repeated runs agree; `tests/test_arena_fixtures.py` runs it once per fixture. Nothing in Maestro reads the
     fixtures yet; no routing behaviour changes.
137. **The test suite never probes this machine's real model servers.** `router.PROBE_BUILTIN_PORTS` (default `True`) and
     the opt-in `MAESTRO_NO_BUILTIN_PORTS` env var make `probe_endpoint` return None, without a request, for the built-in
     localhost ports; an injected `fetch` and configured endpoints are unaffected, so production behaviour is unchanged.
     `tests/helpers.py` turns both on (the flag survives tests that clear `os.environ`; the env var reaches subprocesses)
     and drops `OLLAMA_HOST`. Before, on a machine serving Ollama or vLLM, detect, `mine route` and `setup` tests saw the
     real models. Two other host-dependent tests: the fake upstream in `tests/fake_http.py` now has a listen backlog of
     128 (the default 5 overflowed on the 40-request burst test, and macOS resets an overflowing connection where Linux
     retries), and the CLI client's "not on PATH" check no longer includes `/usr/bin`, where a host may have a real `codex`.
138. **Any harness can ask for a model (`delegate.kind: "model"`).** `maestro route --harness <name>` with a name that is
     not one of the four built-in CLIs (and not `unknown`) treats that harness as the caller: if `models.<name>` sets a
     model per category, the decision is `{"kind": "model", "harness": <name>, "model": ...}` and the harness applies it
     itself (ranked like a subagent). Maestro starts nothing, and the other installed CLIs are not candidates for that
     call; local models still are. Auto-detection only ever yields the four built-ins or `unknown`, so this applies only
     when a caller names itself. `maestro models set <name>.<category>` accepts any name that starts with a letter and is
     otherwise letters, digits and hyphens.
139. **Credentials never follow a redirect to another origin.** `safe_http.opener` (stdlib) replaces urllib's default
     opener in `providers.http_request` and `router._http_fetch`, the only two HTTP paths that carry keys (providers,
     clients, quota, local endpoints with `api_key_env`, Jev). It follows redirects as before but drops `Authorization`,
     `Proxy-Authorization`, `x-api-key` and `api-key` when the scheme, host or port changes; a same-origin redirect keeps
     them. Before, urllib copied them to any redirect target. `route` still imports nothing new on its rules path (the
     import is inside `_http_fetch`).
140. **Price-0 API models are the last resort, not a 503; the public release.** `candidates` skips API models priced
     at 0 (`:free` ids, and the coding plans models.dev lists at price 0) unless they are requested or pinned, because a
     free tier is rate-limited and cannot be counted on. When nothing else can serve a request, it now evaluates again
     with them allowed, and the decision's reason says "only price-0 models (free tiers or plans) can serve this; using
     one". Before, a first run whose only provider was a coding plan (found with one Kimi coding-plan key on a clean
     machine) answered every `maestro/auto` request with 503 although an explicit request to the same model worked.
     The order when other models exist is unchanged. Release packaging, outside the package and outside `route`: the
     distribution is `maestro-router` and installs the CLI as `maestro` and as `maestro-router` (mobile.dev's Maestro
     also installs `maestro`); the public repository is `this-live/maestro-router`; the website drops the
     sample-data demo of deviation 134 (no `/demo/`, no `site/demo/`, no `make_demo.py`) and the unpublished pilot
     tables, keeping only the reproducible router eval.
141. **Owner-only database files; every set key in "api keys present".** `server.open_db` creates `maestro.db` with mode
     0600 before connecting (SQLite gives the `-wal` and `-shm` files it creates the database file's mode) and chmods all
     three afterwards, which also fixes files an older version left 0644. `router.summarize` (used only by `detect` and
     `doctor`, never by `route`) adds `providers.key_names`: the key variable of every registry provider whose key is set
     in the environment or in `keys.env` (names only), so the line agrees with the providers section below it.
142. **Owner-only local state.** `router.ensure_home(home)` (makedirs mode 0700, then chmod 0700 when this user owns the
     directory; a chmod failure is ignored) is the only way Maestro creates `~/.maestro` / `$MAESTRO_HOME`: `_write_json`,
     `catalog.write_models`, `server.open_db`, `server.load_admin_token`, `providers` (keys.env), `quota.install_statusline`,
     `learn.record` and the statusline script's own makedirs; `decide.log_decision` (which never imports the router) creates
     it with mode 0700 and `router.route` tightens an existing one before logging. `router._write_json` creates its temp
     file with `os.open(..., 0o600)` (and fchmod for a stale temp file), so `inventory.json`, `config.json`, `providers.json`,
     the quota, cache and personal-model files are 0600 before `os.replace`; the chmod-after-replace calls in `learn` and
     `systemone` are gone. `_write_json(..., private=False)` keeps the old behavior for a file another tool owns
     (`install --hooks` writes `~/.claude/settings.json`; that directory is never chmodded). `load_admin_token` chmods an
     existing token file 0600. `maestro feedback` opens `maestro.db` only if it exists ("no decision N", exit 1, as for any
     unknown decision) instead of letting SQLite create it with the umask mode.
143. **Request-body limits and keep-alive hygiene.** `server.read_body` checks every limit before it reads: `Content-Length`
     must be ASCII digits (else 400; more than 18 digits or over `MAX_BODY` is 413; two `Content-Length` headers, or one
     together with `Transfer-Encoding`, is 400). A chunked body must be exactly `Transfer-Encoding: chunked`; each size line
     is read with a 1 KiB cap and must be bare hexadecimal (a minus sign, `0x`, `+` or `_` is 400), a chunk that would take
     the total past `MAX_BODY` is 413 before its bytes are read, chunks are collected in a list and joined once, trailers
     are consumed (at most 64 lines), and a body that ends early is 400. Both servers (`server.py`, `site/serve.py`) override
     `end_headers`: a response to a request that carries a body (`Content-Length` other than 0, or `Transfer-Encoding`) that
     was not read adds `Connection: close`, which also sets `close_connection`. The unread bytes therefore never become the
     next request (a pipelined request hidden in the body of a refused one used to be answered, which could be used to
     smuggle a request past the Origin check on a keep-alive connection). A request whose body `read_body` consumed keeps its connection.
     `site/serve.py` never reads a body, so every request that has one is answered and closed; `refuse()` always closes.
144. **The admin token stays out of logs.** `cmd_serve` prints `Web UI: <base>/#token=<token>` only when `sys.stdout.isatty()`.
     Otherwise it prints `Web UI: <base>/` and `admin token: cat <home>/admin_token` (the actual home, so a custom
     `MAESTRO_HOME` is shown correctly). `--open` passes the token URL to `webbrowser.open` as before. The token file and its
     0600 mode (deviation 142) are unchanged.
145. **Subscription CLI seats are locked down.** `clients._cli_argv` builds chat-only argv for the proxy: restricting flags
     are always passed (an older CLI that rejects one fails and the request falls back), and trimming flags are added
     when `cli_caps` finds them in the CLI's `--help` (cached per executable; the probe gets at most 5 s of the request's
     timeout). Claude Code: `--tools "" --strict-mcp-config` plus `--safe-mode --no-session-persistence`. Codex:
     `--sandbox read-only`, `-c features.<f>=false` for every tool-bearing feature (`CODEX_OFF`) plus
     `--ignore-user-config --ephemeral`. Gemini: `--approval-mode default -e none --allowed-mcp-server-names maestro-none`
     and `-p " "`. Every prompt goes on stdin. `clients.cli_env` passes basics, proxy and CA settings and the vendor's own
     prefixes only. Verified against Claude Code 2.1.268 and codex-cli 0.160.1 with a canary file (no tool event, no
     leak); Gemini CLI 0.62.0's flags were checked in `--help` but not run live (its login was refused).
146. **Only safe model and endpoint names reach `delegate.argv`.** `router.SAFE_NAME` (a letter or digit, then letters,
     digits and `. _ : / @ + = -`, at most 200 characters) filters the names `probe_endpoint` reads from a server (every
     consumer, from detect to the proxy, goes through it) and the endpoint names from config, which fall back to the host.
     A probe reports how many names it ignored, and `detect` prints the count. SKILL.md says the argv is a list to run
     without a shell.
147. **One command name; install lines pinned to v3.1.0.** Deviation 140 installed the command under two names. Only
     `maestro-router` is installed now (`[project.scripts]` has the one entry, argparse `prog` and `--version` say
     `maestro-router`), because mobile.dev's Maestro (mobile UI testing) already owns `maestro`; `python3 -m maestro` and
     `python3 <skill-dir>/scripts/maestro <cmd>` are unchanged, and a user who wants the short name adds
     `alias maestro=maestro-router`. Every command a user is told to type (help and error text, the README, the reference,
     the website, SKILL.md, the arena runbook and skill) says `maestro-router`; model names (`maestro/auto`), `~/.maestro`,
     the `maestro-<category>` subagents and the Python package name are not commands and keep their names. The published
     install lines are pinned to the v3.1.0 release: `pip install` of the release wheel
     (`.../releases/download/v3.1.0/maestro_router-3.1.0-py3-none-any.whl`), `pip install git+https://github.com/this-live/maestro-router@v3.1.0`
     from source, and `uvx --from git+https://github.com/this-live/maestro-router@v3.1.0 maestro-router ...`. `npx skills add`
     and the Claude Code plugin marketplace track the repository's default branch, because the skills installer has no
     option to pin a release. `tests/test_package.py` checks that `maestro` is not a console script, and
     `tests/test_site.py` that the pages and the README carry the same pinned lines.
148. **`SAFE_NAME` is a full match, and old inventories are not trusted.** The pattern ends in `\Z` and is applied with
     `fullmatch` (with `$`, `re.match` accepted a name ending in a newline). `INVENTORY_VERSION` is 3, so an
     `inventory.json` written before the check is re-detected rather than used for up to 24 h, and `select_seat` skips
     any local model whose name or endpoint name fails `SAFE_NAME`, whatever inventory it is given.
149. **The statusline script is owner-only.** The script `install --statusline` writes (run by Claude Code, outside
     Maestro's Python) sets `umask 077` before writing `claude_limits.json`, `claude_limits.log` and their `.tmp` files,
     chmods an existing home it owns to 0700 and existing limit files to 0600, and is itself installed 0700.
150. **Gemini CLI seats fail closed.** `clients.GEMINI_REQUIRED` (`--approval-mode`, `--extensions`,
     `--allowed-mcp-server-names`) must all be in the probed `--help`, or `complete_cli` answers 503 without running it.
     Unlike Claude Code's and Codex's restricting flags, Gemini's cannot simply always be passed (older versions lack
     them and Gemini's MCP tools marked trusted would run without approval). Not live-certified: the Gemini CLI login on
     the test machine is refused by Google, so only its flags were checked.
151. **The argv id comes from checked parts.** `select_seat` rebuilds each local model's id as
     `local:<endpoint_name>/<name>` after both pass `SAFE_NAME`, instead of trusting the inventory's `id` field, so a
     passed-in or stale inventory cannot pair a safe name with an unsafe id in `delegate.argv`.

152. **Local run errors use the provider error boundary.** `run_model` passes non-200 bodies to
     `providers.error_message(raw, endpoint_key)` before raising. JSON error messages are decoded,
     echoed endpoint credentials (including JSON-encoded structured values) removed, whitespace normalized and only then the message shortened.
     Both Ollama and OpenAI-compatible endpoints, direct callers and CLI stderr use this boundary;
     successful replies and authentication are unchanged. The helper is imported only on the slow
     execution error path, preserving the offline routing import boundary.

153. **Nested error encodings are bounded and fail closed.** Credential-bearing errors decode up to
     four rounds of JSON escape sequences before redaction and truncation. Errors over 16 KiB or
     retaining deeper escape sequences are omitted with a redaction marker, preserving HTTP status
     and endpoint context. Successful model replies are untouched.
