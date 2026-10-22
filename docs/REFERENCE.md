# Maestro reference

The full reference: every command, config key and design choice. Start with the [README](../README.md) for install and a quick start; the design spec is [DESIGN.md](DESIGN.md). The pictures in [architecture/](architecture/README.md) show how a task becomes a model choice.

## Install (30 seconds)

**Any harness that supports Agent Skills** (Claude Code, Codex, OpenCode, Gemini CLI, Cursor and others):

```bash
npx skills add this-live/maestro-router
```

**Claude Code plugin marketplace:**

```bash
claude plugin marketplace add this-live/maestro-router
claude plugin install maestro@maestro
```

The skills installer and the plugin marketplace track the repository's default branch (the skills installer has no option to pin a release).

**pip / uvx** (gives you the `maestro-router` command; the skill calls the same file). Not on PyPI yet, so install from GitHub, pinned to the v3.4.0 release:

```bash
pip install https://github.com/this-live/maestro-router/releases/download/v3.4.0/maestro_router-3.4.0-py3-none-any.whl
# or from source: pip install git+https://github.com/this-live/maestro-router@v3.4.0
# or, without installing: uvx --from git+https://github.com/this-live/maestro-router@v3.4.0 maestro-router route "fix the failing test"
```

The command is `maestro-router` because mobile.dev's Maestro (mobile UI testing) already uses `maestro`; add `alias maestro=maestro-router` to your shell profile if you want the short name.

Then tell your agent: "use maestro", "route this" or "use the right model". OpenCode and Gemini CLI delegate by category; run `maestro-router install` once to create their `maestro-<category>` subagents (and again after you change their models).

## Quick start (provider mode, web UI, Hermes)

```bash
maestro-router setup            # detect providers, add keys (hidden input), point it at the Spark / vLLM / Ollama boxes you already run, score every model, connect Hermes
maestro-router serve --open     # http://127.0.0.1:4545/v1 for clients, and the web UI
maestro-router hermes --install # or paste the output of `maestro-router hermes --print` into ~/.hermes/config.yaml
```

Already serving models on a DGX Spark (or any vLLM, SGLang, llama.cpp, LM Studio or Ollama box) on your network? Add it once and Maestro lists the models it serves, scores them and routes to them first when they are good enough. Maestro never downloads, loads or starts a model; it uses what is already served:

```bash
maestro-router setup --endpoint http://spark.local:8000   # or answer the question in `maestro-router setup`; saved to local_endpoints
maestro-router discover               # scores what the Spark serves ("spark: serving 1 model (gpt-oss-120b) - using these")
```

## How it works

```
task text ──► classify ──► (job, difficulty 0-3, confidence, needs)
                 │            rules (offline) | jev (picks the model; used when TYPESAFE_API_KEY is set)
                 ▼
            policy ladder  ──► local | fast | standard | strong | frontier
                 ▼
     inventory (cached) ──► a specialist, a local model or a cloud seat ──► decision JSON
```

**Jobs.** Maestro sorts every task into one of eleven jobs:

| job | what it covers | a dev might type |
|---|---|---|
| `plan` | architecture, system design, tradeoffs, roadmaps, migration plans | "should we split the monolith or keep it modular" |
| `code` | write or change code: features, refactors, tests | "add pagination to /users" |
| `debug` | find and fix what's broken: errors, stack traces, flaky tests, incidents, perf regressions | "why's the checkout test flaky" |
| `review` | review code, PRs or diffs; security or quality audits of existing code | "look over my diff before I merge" |
| `ops` | run and set things up: commands, CI, deploys, infra, environments, dependency upgrades across a repo | "bump deps and fix whatever breaks" |
| `research` | look things up or investigate: docs, the web, "where/how does X work in our codebase", comparisons | "where do we retry webhooks" |
| `data` | SQL, data analysis, spreadsheets, metrics, number crunching | "sql for weekly active users by plan" |
| `write` | docs, specs, READMEs, emails, long-form prose | "write a readme for the billing service" |
| `edit` | small mechanical edits: rename, format, typo, commit message, version bump, add a comment | "rename fooBar to foo_bar in utils.py" |
| `summarize` | summarize, convert or pull facts out of text or data pasted into the prompt | "tldr this thread" |
| `ask` | a quick question answerable from general knowledge; chat | "what's a mutex" |

Difficulty is 0 to 3: easy, medium, hard, very hard.

**Categories.** The ladder picks one of six kinds of model. They are provider-neutral: a category names a kind of model, and each of your seats maps it to a model it has.

| category | what it is | examples |
|---|---|---|
| `local` | Models you run yourself, on this machine or on your network. Free and private. | a 120B-class model on a DGX Spark served by Ollama or vLLM, an 8B model on your laptop |
| `fast` | Cheap, quick cloud models | Claude Haiku, GPT mini-class, Gemini Flash-class |
| `standard` | The everyday workhorse | Claude Sonnet, GPT-class mid-tier, Gemini Pro-class |
| `strong` | Hard reasoning | Claude Opus, the top reasoning models from other vendors |
| `frontier` | The very best | Claude Fable, each vendor's top model |
| `specialist` | A model tuned for particular jobs, used for those jobs when you have one | a coder model, a SQL model |

The cloud order is `fast < standard < strong < frontier`. The ladder maps a job and difficulty to a category. A `local` cell means "a local model if one is eligible, otherwise `fast`".

| job | d0 | d1 | d2 | d3 |
|---|---|---|---|---|
| ask | local | fast | standard | standard |
| summarize | local | fast | standard | standard |
| edit | local | fast | standard | standard |
| write | fast | standard | standard | strong |
| data | fast | standard | standard | strong |
| research | fast | standard | standard | strong |
| code | fast | standard | standard | strong |
| ops | fast | standard | standard | strong |
| review | standard | standard | standard | strong |
| debug | standard | standard | standard | strong |
| plan | standard | standard | strong | frontier |

The ladder was retuned after a small internal pilot (see deviation 133 in DESIGN.md for the method and its limits; the graded answers are not published in this repository): on 100 checker-graded self-contained pilot tasks, Sonnet-class models solved 97-100% of the hard and very hard ones, so hard (d2) work goes to `standard` (planning keeps `strong`), very hard (d3) work goes to `strong` (`frontier` only for planning), and `frontier` is no longer a default cell for code, research, ops, review or debug.

With the rules classifier, low confidence (< 0.5) escalates one cell; `max_tier` caps the cloud tier. With Jev, the ladder is the safety net when Jev is unsure (see Classifiers).

Maestro only routes to what you can run. `maestro-router detect` looks for harness CLIs (`claude`, `codex`, `gemini`, `opencode`), credential files and API key variables (presence only, values are never read), local and network model servers, and RAM/VRAM/Apple-silicon memory. The result is cached in `~/.maestro/inventory.json` for 24 hours (and refreshed if you change `local_endpoints`).

Each decision tells the calling agent how to delegate:

| `delegate.kind` | Meaning |
|---|---|
| `subagent` | Use the harness's subagent mechanism with that model (Claude Code: the Agent tool's `model` parameter) |
| `local` | Run `argv` (it calls `maestro-router run` on a local or network model), with `{task}` as one quoted argument, and use the printed reply |
| `cli` | Run `argv` headless, replacing `{task}` (`claude -p`, `codex exec`, `gemini -p`, `opencode run`) |
| `model` | Your own harness applies `model` itself: for a harness Maestro does not run (named with `--harness`, models set with `maestro-router models set <name>.<category> <model>`). Maestro starts nothing, and other CLIs are not offered. |
| `inline` | Do it yourself: you are already on the right tier, or no better seat exists |

Example:

```console
$ maestro-router route --harness claude-code "why's the checkout test flaky"
{"job": "debug", "difficulty": 2, "confidence": 0.59, "needs": ["tools"],
 "classifier": "rules", "category": "standard", "harness": "claude-code",
 "seat": "claude-code", "model": "sonnet", "model_source": "suggested",
 "delegate": {"kind": "subagent", "harness": "claude-code", "model": "sonnet"},
 "fallback": [], "reason": "debug/d2 → standard; claude-code subagent available",
 "latency_ms": 1.1}
```

`--explain` adds the classifier signals, the ladder cell and the tier walk; `--format text` prints one line.

## Local and network models

### Local models: Maestro uses what you already serve

Maestro never downloads, loads, starts or installs a model. It gauges your hardware, looks at what is already being served on this machine and on your network, and uses exactly those models.

- **Installed is not served.** Ollama lists every downloaded model but loads one on the first chat request, and LM Studio loads on demand too; either can run a machine out of memory. So Maestro asks (GET requests only) which models are *loaded* (Ollama `/api/ps`, LM Studio's model state; vLLM, SGLang and llama.cpp serve what they list). A model that is installed but not loaded is shown by `maestro-router detect` and `maestro-router discover` but is never routed to, benchmarked or sent a request, by the skill, the proxy or `maestro-router run` (which refuses it unless you pass `--allow-load`). Set `"local_allow_load": true` in the config to route to installed models anyway; the server then loads them on first use, which can exhaust memory. The loaded state is read when Maestro detects (cached up to 24 h) and again by `maestro-router run`; `maestro-router detect` refreshes it.
- **Suggestions only when nothing is served.** If a box serves nothing, `maestro-router discover` says so and `maestro-router discover --local-fit` lists the open-weight models that would fit its memory. That is advice: Maestro prints a command and stops. If a box already serves models, suggestions are not shown by default; `--local-fit` still lists them, marked as alternatives that would **replace** what is served, not add to it (one model server usually claims most of the memory; vLLM reserves `--gpu-memory-utilization`, 0.9 by default), with each command annotated "stop the current server first".

Models you already run can take the jobs that need no repo access, for free and in private. Maestro finds them for you:

- Ollama on `127.0.0.1:11434` (or `OLLAMA_HOST`), and LM Studio (`:1234`), vLLM (`:8000`), SGLang (`:30000`) and llama.cpp (`:8080`) on localhost, are probed on every `maestro-router detect` (`MAESTRO_NO_BUILTIN_PORTS=1` skips the localhost ports).
- For a box on your network or over Tailscale, list it in `~/.maestro/config.json`. A DGX Spark that already serves models with vLLM (it has to listen beyond localhost, `--host 0.0.0.0`, for other machines to reach it):

  ```json
  {"local_endpoints": [{"name": "spark", "url": "http://spark.local:8000", "kind": "openai"}]}
  ```

  vLLM serves one model per port, so add one entry per port. Model ids keep vLLM's names, slashes included (`local:spark/openai/gpt-oss-120b`). If you used `--served-model-name` and the name no longer says the size (e.g. `main`), set its strength with `local_models` (below). The same box running Ollama instead: `{"name": "spark", "url": "http://spark.local:11434"}`.

  An endpoint is a URL or `{"name", "url", "kind": "ollama" | "openai", "api_key_env"}`. `kind` is auto-detected when left out. `api_key_env` is the *name* of an environment variable; if it is set, its value is sent as a Bearer token (Maestro never prints or stores it).
- `maestro-router detect --scan` also probes your machine's own /24 subnet for Ollama (port 11434) and OpenAI-compatible servers (port 8000). It only runs when you ask, takes a few seconds at most, and prints what it finds; `--save` adds the endpoints to your config. Set `MAESTRO_NO_SCAN=1` to forbid it.

Each model gets an id like `local:spark/gpt-oss:120b` (`local:<endpoint>/<model>`; `this-machine` for localhost) and a `serves` rating, the strongest category it can stand in for. By default that comes from its size: under 10B serves only `local` cells, 10B to 40B serves up to `fast`, and 40B or more (a 120B model, say) serves up to `standard`. Local models are never `strong` or `frontier` unless you say so.

A local model is picked when its `serves` reaches the ladder cell (the smallest one that does, this machine before the network, lower latency first) **and** it can do the job. A plain local model is chat only, so it takes `ask`, `summarize`, `write`, `plan` and `data` (when the task needs no repo tools, images or huge context). `code`, `debug`, `review`, `ops`, `research` and `edit` work in your repo, so they use a local model only when your harness runs it natively as a subagent (for example OpenCode with the model listed in `models`).

Try it. With that config, on a Claude Code machine:

```console
$ maestro-router route --harness claude-code "write a readme for the billing service"
{"job": "write", "difficulty": 1, ..., "category": "local", "seat": "local",
 "model": "local:spark/gpt-oss:120b",
 "delegate": {"kind": "local", "model": "local:spark/gpt-oss:120b",
              "argv": ["python3", "/path/to/skills/maestro/scripts/maestro", "run", "local:spark/gpt-oss:120b", "{task}"]},
 "fallback": [{"seat": "claude-code", "model": "sonnet", "category": "standard"}], ...}
```

`maestro-router run local:spark/gpt-oss:120b "<task>"` sends one chat completion (Ollama `/api/chat`, or OpenAI-compatible `/v1/chat/completions`) and prints the reply. The task can also come on stdin; `--system` sets a system prompt and `--timeout` (default 300 s, config `run_timeout_s`) bounds the wait. A model that is installed but not loaded is refused (the request would make the server load it); `--allow-load` sends it anyway.

Tune it in the config (all optional):

| key | meaning |
|---|---|
| `prefer_local` | `true` (default). Set `false` to never use local models. |
| `local_endpoints` | endpoints to probe, as above |
| `local_allow_load` | `false` (default). `true` routes to models a server has installed but not loaded (Ollama, LM Studio); the server loads them on the first request, which can exhaust memory. |
| `local_models` | override a model's rating or jobs: `{"gpt-oss:120b": {"serves": "strong"}}`, `{"qwen2.5-coder:7b": {"specialist_for": ["code"]}}` (the key is a model name, `endpoint/name` or the full id) |
| `specialists` | models tuned for jobs: `{"my-sql-model": ["data"]}` |
| `models.local` | pin which local model serves which category, overriding the size rating (see Models per category) |

**Specialists.** A model can be a specialist for some jobs: named in `specialists`, or matching by name (`coder`, `code` and `devstral` models are specialists for `code`, `debug` and `review`; `math` and `sql` models for `data`). A specialist counts as at least `fast`, is used only for its jobs, and wins over a generic model when the ladder cell is within what it serves. Such a decision has `category: "specialist"`.

## Discovery: score every model you can use

`maestro-router discover` finds every model you can call, from every provider (API keys, `keys.env`, subscription logins, local and network servers), joins each one to the bundled [models.dev](https://models.dev) catalog (benchmarks, prices, context, release dates for every lab, about 120 KB), scores it and writes `~/.maestro/models.json`. `maestro-router models` then shows that map.

```console
$ maestro-router discover
discovered 464 models from 9 providers; wrote ~/.maestro/models.json
anthropic (api key, 16 models: 6 strong, 5 standard, 4 frontier, 1 fast)
  claude-fable-5      frontier   G 0.74  research 0.80 summarize 0.79 plan 0.77  bench
  claude-sonnet-5-5   strong     G 0.64  research 0.73 summarize 0.71 ask 0.70   bench
  ...
local:spark (local, 2 models: 2 local)
  openai/gpt-oss-120b  local>fast  G 0.33  data 0.36 edit 0.34 code 0.34         bench
```

- **How a model is scored.** Benchmarks (code, terminal, reasoning, agent, browse, data, write, long-context) become percentiles among recent models; they are blended with a prior from size, price, name and age (`G = w*benchmarks + (1-w)*prior`, `basis` says which dominated), then turned into a strength for each of the 11 jobs and a category: `frontier` (G >= .68), `strong` (.52), `standard` (.35), `fast`; `specialist` for coder-style models; `local` for models on your machines, with `serves` naming the tier they can stand in for. Pin a different category or switch a model off in `config.json`: `{"models_overrides": {"openai/gpt-5-mini": {"category": "standard", "enabled": false}}}` (rediscovery keeps it).
- `--refresh-catalog` merges the live models.dev API (new models, current prices); offline it falls back to the bundled snapshot. `--all` prints every model. `--json` prints the map.
- `--backtest` routes the 474 eval prompts over your map, offline, and reports how often the chosen model's category matches the expected one, the cost per 100 requests against always using your strongest model, and the share that goes to local models. `--live N` (costs a few tokens) sends N tiny prompts to each model the backtest used and records time to first byte, total time and failures.

### What could run on your hardware

`maestro-router discover` ends with one line per box: `spark: serving 1 model (gpt-oss-120b) - using these`, or, when nothing is loaded, ``spark: nothing served; `maestro-router discover --local-fit` lists models that would fit (advice only)``. `maestro-router discover --local-fit` lists, for each box you have (this machine, and each `local_endpoints` entry), the best open-weight models that fit in its memory, the quantization needed and a serve command. On a box that already serves models the list is headed "Alternatives ... (would REPLACE what it serves now)" and every command carries `# stop the current server first`; for a box that serves nothing it reads:

```console
Could run on spark: DGX Spark, 128 GB (assumed: set local_endpoints[].hardware to change); usable about 115 GB
  #   model                          params quant  mem GB  tok/s     G  category
  1   alibaba/qwen3.5-122b-a10b       122B NVFP4    90.7    ~29  0.43  standard
      vllm serve Qwen/Qwen3.5-122B-A10B-NVFP4 --max-model-len 32768   # needs an NVFP4 build of this model
```

Describe a remote box in the config: `{"local_endpoints": [{"name": "spark", "url": "http://spark.local:8000", "hardware": {"name": "DGX Spark", "memory_gb": 128, "bandwidth_gbps": 273}}]}` (an endpoint called or hosted "spark" is assumed to be a 128 GB, 273 GB/s DGX Spark; bandwidth is looked up by name when you leave it out). The estimate is weights at that precision plus 15%, plus room for a 32K context, within 90% of memory; BF16, FP8, NVFP4 (Blackwell only) and Q4 are considered. Models are ranked by how good they are *and how fast they would run*: decoding reads the active weights once per token, so `tok/s ~ bandwidth x 0.6 / (active parameters x bytes per parameter)`. A dense 128B model on a Spark manages about 2 tok/s and drops down the list; a mixture-of-experts model with 5B active parameters does about 57. Under 10 tok/s a model's score is cut to 30%, under 25 to 70%. Estimates marked `*` assume all parameters are active because the catalog does not know better. Parameter counts come from Hugging Face model pages, the model name, or are shown as unknown. This is advice only: Maestro never downloads, loads or starts anything, and when a box already serves models the suggestions would replace them.

## Commands

```
maestro-router route "<task>" [--harness H] [--current-tier T] [--classifier auto|rules|s1|jev] [--format json|text] [--explain]
              [--decider ladder|utility] [--job J] [--tools yes|no] [--self-contained yes|no] [--size s|m|l|xl] [--stakes low|normal|high] [--context-tokens N]
maestro-router route -                 read the task from stdin
maestro-router s1 "<task>" [--json]    S1's estimate of a task (job, difficulty, tools, stakes, tokens)
maestro-router decide [file|-] [--calibrate FILE]      answer one /v1/systemone request (Jev's typed-decision format)
maestro-router outcome <decision_id|last> pass|partial|fail [--tests P/T] [--from-hook]   record how a routed task went
maestro-router feedback <id> good|bad|partial          rate a proxy decision (the id is in the x-maestro-decision header)
maestro-router learn [--report | --audit | --reset | --dry-run]   fit (or inspect, or delete) your personal offsets
maestro-router install --hooks         add the Claude Code SubagentStop hook that records test results of delegated work
maestro-router detect [--catalog] [--models] [--scan [--save]] [--json]
maestro-router keys set <provider|ENV_VAR> [value]     store an API key in ~/.maestro/keys.env (mode 600; prompts without echo if no value)
maestro-router keys unset <provider|ENV_VAR>
maestro-router keys list [--all | <provider>]          which providers have a key, and from where (values are never shown)
maestro-router run <model-id> ["<task>"] [--system S] [--timeout SECONDS] [--allow-load]
maestro-router discover [--refresh-catalog] [--backtest] [--live N] [--local-fit] [--all] [--json]   score every usable model, write ~/.maestro/models.json (--live probes only loaded local models)
maestro-router models [--suggest | --save | --check | --seats] [--json]     the discovered model map (or the per-harness table)
maestro-router models set <harness>.<category> <model> [--effort E]
maestro-router models unset <harness>.<category>
maestro-router install [--harness claude-code|opencode|gemini-cli|all] [--dry-run]
maestro-router install --statusline    write ~/.maestro/statusline.sh (caches Claude Code's rate limits) and print the settings.json snippet
maestro-router doctor [--jev]
maestro-router config
maestro-router serve [--host 127.0.0.1] [--port 4545] [--open]   OpenAI-compatible router + web UI
maestro-router setup [--yes] [--endpoint URL]   quick start: detect, add keys and model servers, discover, connect Hermes and the skill
maestro-router hermes [--print | --install]   the Hermes Agent config for Maestro
```

Without pip, use `python3 skills/maestro/scripts/maestro <command>`.

`detect` also lists the API providers you have a key for (any of the 225 in [models.dev](https://models.dev), by environment variable or `keys.env`) and your subscription logins (Claude Code, Codex, Gemini CLI: presence only, tokens are never read); `detect --models` asks each provider for the models your key can use.

## Use it from Hermes / any OpenAI client

`maestro-router serve` runs a local OpenAI-compatible endpoint (default `http://127.0.0.1:4545/v1`). Point a client at it, use the model name `maestro/auto`, and Maestro picks the model per request from what you have: API keys, subscription logins (chat only) and local or network servers.

```bash
maestro-router serve --open            # prints the endpoint, the UI link with its admin token, and the Hermes config
maestro-router hermes --print          # just the Hermes Agent config
maestro-router hermes --install        # add it to ~/.hermes/config.yaml (backs the file up)
```

```yaml
# Hermes Agent: ~/.hermes/config.yaml
model:
  default: maestro/auto
  provider: custom
  base_url: http://127.0.0.1:4545/v1
  api_key: maestro
  context_length: 128000
auxiliary:            # compression, title_generation and session_search use maestro/fast; vision uses maestro/auto
  compression: {provider: custom, base_url: http://127.0.0.1:4545/v1, api_key: maestro, model: maestro/fast}
delegation: {base_url: http://127.0.0.1:4545/v1, api_key: maestro, model: maestro/auto}
```

`maestro-router hermes --print` prints the full block. `--install` only appends top-level blocks that are missing (`model`, `auxiliary`, `delegation`) after making a backup; if you already have one it prints the block for you to merge, because Maestro does not rewrite YAML it did not write. Any other client works the same way:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:4545/v1", api_key="anything")
r = client.chat.completions.create(model="maestro/auto", messages=[{"role": "user", "content": "why is my test flaky?"}])
print(r.model)   # the model Maestro chose, e.g. anthropic/claude-sonnet-5-5
```

- **Model names.** `maestro/auto` classifies the request and routes it. `maestro/fast|standard|strong|frontier|local` fixes the category, `maestro/code|debug|plan|...` fixes the job, and `<provider>/<model>` (see `GET /v1/models`) passes straight to that model, with fallback.
- **What it does per request.** Filters to the enabled models that can take the request (tools, images, context window, quota), keeps those whose score for the job reaches the category's floor and whose own category is at least the one the task needs, then ranks by expected cost, quota headroom and observed latency (`prefer_local` and `objective` shift this). Streaming, tool calls, `stream_options.include_usage` and images pass through; Anthropic models are translated both ways.
- **Agent loops stay on one model.** A request whose last message is a tool result reuses the model chosen for that conversation (keyed on the `X-Session-Id` header, the header named in `session_affinity_header`, or a hash of the system prompt and first user message; 30 minutes). A new user message is routed afresh.
- **Fallback.** On 429, 5xx, timeout or connection error before the first byte, the next candidate is tried (three at most) and the failing model cools down. Once bytes have streamed, the error is passed on instead. The `x-maestro-decision` response header (JSON) says what was chosen and why; the response's `model` is the real `<provider>/<model>`.
- **Subscription seats** (Claude Code, Codex, Gemini CLI) serve chat only: never a request with tools or images.
- **Binding.** `127.0.0.1` needs nothing. `--host 0.0.0.0` (to serve your whole fleet, say from a DGX Spark) requires `MAESTRO_API_KEY` in the server's environment, and clients send it as `Authorization: Bearer ...`. Browsers on other sites are refused (Origin and Host checks).
- **Privacy.** Every request is one row in `~/.maestro/maestro.db` (model, job, category, candidates, latency, tokens, status). Prompt text is stored (first 120 characters) only if you set `log_prompts: true`. API keys are used only against their own provider and are never logged or returned by any endpoint.

## Web UI

`maestro-router serve` in a terminal prints a link like `http://127.0.0.1:4545/#token=...`; with its output redirected to a file or pipe it prints the plain URL and `admin token: cat ~/.maestro/admin_token` instead, so the token never lands in a log (`--open` always opens the browser with the token). The admin token lives in `~/.maestro/admin_token`, mode 600. One page, no build step, no external requests, light and dark:

- **Overview**: status, counts, today's requests by category, estimated spend saved versus always using the strongest model, copy buttons for the Hermes config, an OpenAI client snippet and the skill install command, and a local-models panel: "Using: ..." lists what your servers already serve (nothing else is used) with the alternatives that would replace it folded away; only when nothing is served does it show "Could run locally" (the open-weight models your hardware could serve; advice only).
- **Providers**: every provider's status, quota bars with reset times, a write-only key field (saved to `keys.env`), and a Test button.
- **Models**: search and sort every model with its score, best jobs, cost, context, capabilities and quota; switch a model off or pin its category; re-run discovery.
- **Routing**: the decider (ladder or expected utility) with the shadow agreement and outcome counts, private mode, objective, prefer local, highest category, the job by difficulty ladder, and local pins.
- **Activity**: live decisions with the reasoning, candidates, fallbacks, latency and tokens, the other decider's pick, and Good / Bad buttons that feed personalization.
- **Try it**: a dry run that shows the decision with every candidate's numbers (and S1's success chance, price and expected utility per model), or send the prompt for real.

## Maestro S1: the decision model

S1 is Maestro's own tiny model plus an expected-utility decision layer, shipped behind a flag (`decider: utility`). The default stays the ladder; S1 runs next to it in shadow mode and logs what it would have picked, so you can compare before you switch.

- **S1** (`s1.py`, `data/s1.bin`, 1.2 MB, stdlib, no network) turns the task text into a posterior: the job, a difficulty `b` with its uncertainty, whether it needs repo tools, whether it is self-contained, the stakes, whether a failure would be noticed, output tokens and agent turns, and how unfamiliar the text is. It hashes words, identifiers, file types and structure into a 2^14-bucket int8 embedding bag and runs in about 0.2 ms on a short prompt (1-2 ms on a 60k-character paste). Without the weights it falls back to the Naive Bayes model and says so.
- **decide** (`decide.py`, `data/decide.json`) gives every seat you have a success probability from S1's difficulty and the model's ability (Claude's tiers are anchors; any other model is placed by its catalog job strengths through a fitted link), then an expected utility in dollars: the value of a success, minus API dollars, minus a quota price for subscription seats, minus waiting time, minus the cost of a failure (a retry if you would notice it, a stakes-weighted loss if you would not). The quota price is zero while a window has slack ("use it or lose it") and rises only when the window's burn rate would exhaust it before it resets. A conversation stays on its model when switching would lose the prompt cache. The calling agent itself is a seat (`--current-tier`), and `private` makes local models a hard constraint.
- **Hints** from the calling agent (`--job`, `--tools`, `--size`, `--stakes`, `--context-tokens`) sharpen the posterior. The decision adds `decider`, `decision_id`, `posterior`, `options` (P, P under pessimism, $, quota $, seconds, EU per seat), `p_success`, `escalate_to` (the seat to try if this one fails) and `confidence`; `--explain` prints the options table.
- **Tool jobs stay on the ladder** in skill mode until a release gate (`decide.json: agentic_ready`) shows the utility router is as cheap and as good on agentic fixtures; until then its pick is the shadow.
- **Outcomes and personalization.** `maestro-router outcome <decision_id> pass|fail`, the Claude Code SubagentStop hook (`install --hooks`, test summaries only), Good / Bad in the web UI, `maestro-router feedback`, the `X-Maestro-Feedback: <id>=good` header, and test results in tool messages are recorded locally. `maestro-router learn` (daily inside `serve`) refits a few per-seat offsets and a calibration from them, with guards (caps, a time-split check, at most 0.2 of movement before 30 outcomes); `learn --reset` undoes it. Logs hold heads, numbers and a MinHash, never the text unless `log_prompts` is on.
- **Our own Jev.** `POST /v1/systemone` (and `maestro-router decide`) answers TypeSafe Jev's typed-decision wire format: job, difficulty, tools, stakes and "which of these models" natively in about 1 ms; other questions only if you name a model you serve as `systemone_delegate`.

**Where it stands.** The bootstrap `s1.bin` is trained only on `evals/train.jsonl` (job and difficulty labels; the other heads come from job priors), so it does not yet replace the Naive Bayes classifier: on the dev half of `evals/prompts.jsonl` at matched under-routing it scores job 82.3% vs 84.8%, category 65.8% vs 70.5%, under-routed 7.6% vs 7.2% (`python evals/run.py --classifier s1`; test half: job 81.9%, category 58.2%, under-routed 8.0%, severe 0.4%). The utility decider does not depend on that: quota pacing, local seats, cache stickiness and inline-vs-delegate need the decision layer, not a better difficulty model. Its success probabilities are placeholders until the alpha set (`evals/alpha.py`: pilot, teacher-panel labels, graded outcomes for four Claude tiers, replay against the best constant policy, a paired bootstrap) is collected; the default flips only when `alpha.py release` passes. `route` with S1 adds about 2-5 ms to the 40-50 ms command; 200 in-process utility routes take about 0.2 s.

## Classifiers

- **rules**: offline, about 0.2 ms, deterministic. Keyword and structure scoring on the instruction (pasted code blocks and logs count as material, not intent), combined with a small embedded Naive Bayes model (`maestro_model.json`, trained by `evals/train.py`) that predicts job and difficulty and picks the category with the least expected under-routing. If the model file is missing, or was trained on a different list of jobs, the keyword rules run alone. Always the fallback for Jev.
- **jev**: TypeSafe's Jev System One model *chooses the model for you*. Maestro builds the list of models you can actually run (after filtering on tools, vision, context, your `max_tier` and local-model eligibility), describes each one, and asks Jev one question: which of these should do this task, preferring the cheapest that will do it well. It also scores reasoning depth, scope, ambiguity and risk, and names the job. A confident pick (>= `jev_min_confidence`, 0.55) is used as is; otherwise Maestro takes the stronger of Jev's pick and the ladder cell. If only one model is eligible there is nothing to decide and Jev is not called.
- **s1**: S1's job and difficulty, decoded with the same under-routing-aware rule as the Naive Bayes model (see Maestro S1). `auto` switches to it only once its weights pass the replace-NB check.
- **auto** (default): jev when `TYPESAFE_API_KEY` is set, otherwise rules (or s1 once it passes). Force one with `--classifier rules|s1|jev`, `MAESTRO_CLASSIFIER` or `"classifier"` in the config. Check the connection and latency with `maestro-router doctor --jev`.

**What is sent to TypeSafe.** Only when `TYPESAFE_API_KEY` is set and the classifier is `auto` or `jev`: one HTTPS request to `api.typesafe.ai` per routed task (unless cached) containing the task text you passed to `route` (truncated to 4,000 characters), a few structural signals (length, whether it contains code or file paths), and the names and descriptions of your candidate models (e.g. `claude-code:opus`: "strong tier, Claude Code subagent, subscription (flat cost)...", or `local:spark/gpt-oss:120b`: "local model on spark (120B), free and private..."). That includes your local endpoint names and model names. API keys, file contents, and your environment are never sent. To keep every task local, don't set the key or use `--classifier rules`. Failures (timeout 2.5 s, HTTP errors, malformed answers) fall back to rules and say so in the decision's `reason`; results are cached in `~/.maestro/cache.json`.

## Config

Optional `~/.maestro/config.json` (override the directory with `MAESTRO_HOME`). All keys are optional.

| key | default | meaning |
|---|---|---|
| `classifier` | `"auto"` | `"auto"` (jev if `TYPESAFE_API_KEY` is set, else rules), `"rules"`, `"s1"` or `"jev"` |
| `objective` | `"balanced"` | jev: `"balanced"`, `"cost"` (cheapest that works) or `"quality"` |
| `jev_min_confidence` | `0.55` | jev: below this, take the stronger of Jev's pick and the ladder cell |
| `min_confidence` | `0.5` | rules: below this, escalate one cell |
| `model_notes` | `{}` | jev: override a candidate's description, e.g. `{"claude-code:opus": "best for gnarly refactors"}` |
| `max_tier` | `"frontier"` | never route to a cloud model above this, e.g. `"strong"` to avoid frontier pricing |
| `prefer_local` | `true` | use local and network models for the jobs they can do |
| `local_endpoints` | `[]` | local or network model servers (see Local and network models) |
| `local_allow_load` | `false` | route to installed-but-not-loaded models (the server loads them; can exhaust memory) |
| `local_models` | `{}` | per-model `serves` and `specialist_for` overrides |
| `specialists` | `{}` | model name to the jobs it is tuned for |
| `run_timeout_s` | `300` | `maestro-router run` timeout |
| `prefer` | `[]` | seats to try first, e.g. `["codex"]` |
| `job_seats` | `{}` | pin a job to a seat, e.g. `{"research": "gemini-cli"}` |
| `models` | `{}` | the model per harness and category, `{"codex": {"fast": "...", "strong": {"model": "...", "effort": "high"}}}`, plus `local` pins; see Models per category |
| `harness_models` | `{}` | deprecated name for `models` (read when `models` is empty; `maestro-router models` tells you to rename it) |
| `ladder` | `{}` | override rows, e.g. `{"ask": ["local", "fast", "fast", "standard"]}` |
| `decider` | `"ladder"` | `"utility"`: S1 and the expected-utility ranking pick the seat (see Maestro S1); the default flips after the release gate |
| `s1_shadow` | `true` | also run the other decider and log its pick (no text) next to each decision |
| `wtp_usd` | `null` | utility: the dollar value of a success; null uses the calibrated value for your `objective` |
| `latency_usd_per_s` / `background_latency_usd_per_s` | `0.002` / `0.0002` | utility: what a second of waiting costs (background: `X-Maestro-Background: 1` or `maestro/<job>`) |
| `risk` | `1.0` | utility, 0-3: weight on an unnoticed failure and on how unsure a model's ability is |
| `quota_reserve` | `0.15` | utility, 0-0.5: share of a subscription window you share with your own session that routing keeps free |
| `local_usd_per_hour` | `0` | utility: what an hour of a local model costs you |
| `inline_access` | `"subscription"` | utility: how the agent that calls `route` is billed, `"subscription"` or `"api"` |
| `private` | `false` | only local models (or the calling agent) see tasks; utility also turns it on by itself for a task that holds a key-like string |
| `log_decisions` | `true` | `maestro-router route` appends heads, options and a MinHash (never text) to `~/.maestro/decisions.jsonl` |
| `personalize` / `explore` | `true` / `0` | learn personal offsets from outcomes; proxy only: take a near-tied runner-up with this probability (0-0.1) |
| `systemone_delegate` / `systemone_cloud` | `null` / `false` | a model you serve that answers non-native `/v1/systemone` questions; allow a cloud model there |

## Models per category

Every harness has a model for each cloud category (`fast`, `standard`, `strong`, `frontier`), and `local` covers the models you run yourself. Maestro ships suggestions, so it routes sensibly with no configuration, and you can see and change every cell.

| harness | fast | standard | strong | frontier |
|---|---|---|---|---|
| Claude Code | `haiku` | `sonnet` | `opus` | `fable` |
| Codex | `gpt-6-luna` | `gpt-6.1-sol` | `gpt-6.1-sol`, effort `high` | `gpt-6-astra` |
| Gemini CLI | `gemini-flash-lite-latest` | `gemini-flash-latest` | `gemini-3.1-pro-preview` | `gemini-3.1-pro-preview` |
| OpenCode | one row of the next table, chosen from your API keys | | | |

OpenCode names models `provider/model`. Maestro suggests the first provider you have an API key for, in the order anthropic, openai, google (anthropic when it finds none):

| provider | fast | standard | strong | frontier |
|---|---|---|---|---|
| anthropic | `anthropic/claude-haiku-4-5` | `anthropic/claude-sonnet-5-5` | `anthropic/claude-opus-5-5` | `anthropic/claude-fable-5-1` |
| openai | `openai/gpt-6-luna` | `openai/gpt-6.1-sol` | `openai/gpt-6.1-sol` | `openai/gpt-6-astra` |
| google | `google/gemini-flash-lite-latest` | `google/gemini-flash-latest` | `google/gemini-3.1-pro-preview` | `google/gemini-3.1-pro-preview` |

**These ids are a snapshot of the [models.dev](https://models.dev) catalog as of 2026-09-30.** Claude Code's names are its own aliases and follow the latest release by themselves; the others are dated ids and go stale as vendors ship new models. Run `maestro-router models --check` now and then (it needs the network): it reports every Codex, Gemini CLI and OpenCode id that no longer exists, is marked deprecated, or has a newer model in the same family. It only reads; it changes nothing.

A cell resolves in this order: your `models` config, then the suggestion, then (for `strong` only) the CLI's own default model, meaning no `-m` flag. Each decision says which one it used in `model_source` (`config`, `suggested` or `builtin` for a cloud model; `config` or `auto` for a local one).

**See it.** `maestro-router models` prints the effective map for every harness Maestro found: category, model, where the model came from (config, suggested, built-in, auto, none), whether it is usable (for example not above your `max_tier`), and the harness Maestro would detect right now. `maestro-router models --json` is the same for scripts, and `maestro-router models --suggest` prints the suggestion tables above, including which OpenCode provider was picked from your credentials.

**Change it.**

```bash
maestro-router models set codex.fast gpt-6-luna
maestro-router models set codex.strong gpt-6.1-sol --effort xhigh    # effort: low | medium | high | xhigh | max
maestro-router models unset codex.fast                               # back to the suggestion
maestro-router models --save                                         # write today's suggestions into the config for every cell you left unset
```

`set`, `unset` and `--save` edit only the `models` key of `~/.maestro/config.json` (indent 2, other keys untouched) and never overwrite a value you set. A `harness_models` key from an older config is moved into `models` when you do this. They refuse to touch a config file that is not valid JSON. The same thing by hand:

```json
{"models": {
  "codex":    {"fast": "gpt-6-luna", "strong": {"model": "gpt-6.1-sol", "effort": "high"}},
  "opencode": {"standard": "anthropic/claude-sonnet-5-5"},
  "local":    {"local": "local:spark/nvidia/Llama-3.1-Nemotron-Nano-8B-v1", "standard": "local:spark/openai/gpt-oss-120b"}
}}
```

A cell is a model id, or `{"model": id, "effort": ...}`. Effort is only passed where the CLI has a flag for it: Codex gets `-c model_reasoning_effort=<effort>`. Everywhere else it is ignored, and `maestro-router models` says so. Use the names that CLI accepts (`codex -m`, `gemini -m`, `opencode run -m`). OpenCode and Gemini CLI read their model from the `maestro-<category>` subagent files, so run `maestro-router install` again after changing them.

**How the models are used.** Inside Claude Code, OpenCode and Gemini CLI the delegate is a subagent on that category's model. Codex has no per-subagent model, so inside Codex a delegation to a different model is `codex exec -m <model>` (and `inline` when you are already on that category). Every other installed CLI gets the same full map, and is tried after the harness you are in.

**Local models.** `models.local` pins which local or network model serves which category and overrides the size rating. A model pinned under `standard` serves up to `standard`; a model pinned under `local` is the first choice for `local` cells; models you leave unpinned keep their rating from size. A pinned id must be a detected model (an id like `local:spark/gpt-oss:120b`, `spark/gpt-oss:120b` or the bare name); `maestro-router models` warns about any that is not, and routing ignores it.

Suggestions and the catalog check use [models.dev](https://models.dev) (`MAESTRO_CATALOG_URL` points `--check` and `detect --catalog` at a mirror). With `maestro-router detect --catalog`, API models from any vendor whose key you have set are also listed by price: about $30 per 1M output tokens or more is `frontier`, $15 `strong`, $6 `standard`, less is `fast`.

## Benchmarks

Offline rules classifier (the learned model plus keyword rules, no network) on the held-out test half of `evals/prompts.jsonl` (n = 237, every odd line; the model and the keyword tables were tuned on the other half only):

| metric | test | dev (tuning half) |
|---|---|---|
| job accuracy (11 jobs) | 80.2% | 84.8% |
| category accuracy | 65.0% | 70.5% |
| category within one step | 99.6% | 98.3% |
| under-routed (category too weak) | 8.0% | 7.2% |
| severely under-routed (two or more steps) | 0.0% | 0.0% |
| difficulty exact / mean absolute error | 49.4% / 0.59 | 48.1% / 0.60 |

Keyword rules alone score 67.1% job accuracy, 59.1% category accuracy, 14.3% under-routed and 0.4% severely under-routed on the test half, so the learned model is what keeps the router from under-routing. Category accuracy is deliberately not the goal: routing one category too weak costs 1.5 times routing one too strong, so most misses are too strong (27.0% of test rows, 26.6% by one step) rather than too weak. When the job is unsure between one that needs repo tools and one that does not, it picks the tools job, so repo work is not handed to a chat-only local model: 6 of 128 tool-job test rows were read as a chat-only job.

Read these numbers with the labels in mind. The prompts and their `job` and `difficulty` labels were written by an LLM and only partly hand-audited, and about 15% of the rows are genuinely ambiguous between two jobs (per the relabel log), so a perfect classifier would not score 100%. The set has 474 prompts, so one row is 0.4 points on a half.

What is also measured: `maestro-router route` takes about 60 ms wall clock including interpreter start (median of 20, cached inventory; a bare `python3 -c pass` is about 13 ms), and about 0.3 ms median (p95 under 1 ms) for the classification itself. The shipped model is 136 KB.

Reproduce with `python evals/run.py --classifier rules --split test` (`--split dev` is the tuning half; `--variant heuristic|model` scores the keyword rules or the learned model alone); `--classifier jev` scores Jev's model choice (needs a key). CI gates the test split.

## Test it on your own prompts

The benchmark above uses prompts written for it. `evals/mine.py` scores Maestro on what you actually type into Claude Code, Codex, Gemini CLI, OpenCode and Hermes, in three commands:

```bash
python3 evals/mine.py harvest                 # 1. collect your recent prompts (last 90 days, 80 of them)
python3 evals/mine.py route                   # 2. run Maestro on them; writes routed.csv to review
python3 evals/mine.py score                   # 3. score it against your corrections
```

1. **harvest** reads the prompt history your harnesses keep on disk, drops slash commands, one-word acknowledgements, tool output and duplicates, redacts likely secrets (API keys, tokens, `password=` values, private keys, long hex or base64 strings) as `[REDACTED]`, and samples across harnesses, preferring recent prompts. It writes `~/.maestro/mine/prompts.jsonl` (mode 0600) and prints where the prompts came from, by harness and by project. **By default it keeps only prompts a person typed.** Sessions a program started (`claude -p`, the Agent SDK, an orchestrator dispatching Claude Code sessions, `codex exec`) are skipped and counted, with their top projects, so you can see what was left out. Claude Code records this per message (`turnOrigin`); Codex per session (`originator`); Hermes per session (`cron` and `curator` sessions are automated; the session's platform, such as `cli` or `telegram`, is its project). Options: `--since 90d`, `--limit 80`, `--harness claude-code|codex|gemini-cli|opencode|hermes`, `--project P` / `--exclude-project P` (match on the project path, repeatable), `--include-automated` (also keep program-sent prompts), `--include-subagents` (Claude Code: also the subtask each subagent was given; an AI wrote those, not you).
2. **route** runs the router with the offline rules classifier and your real inventory and config, so categories reflect your actual local and network models. It prints counts by job and category plus the 10 prompts Maestro was least sure about, and writes `routed.csv`. Open it and fill `your_job`, `your_difficulty` (0-3) or `your_category` **only where Maestro is wrong**; blank means you agree.
3. **score** reports job accuracy, category accuracy, within one step, under-routed and severely under-routed rates, and which jobs and categories were confused. `--export labeled.jsonl` writes your corrected labels in the `evals/prompts.jsonl` format so you can contribute them later. That is never automatic, and your prompts are private: review the file before you share any of it.

**Privacy.** Everything runs on your machine and nothing is sent anywhere (the optional `route --classifier jev` is the one exception, warns first, and needs `TYPESAFE_API_KEY`). Secrets are redacted before anything is written, but redaction is pattern-based, so skim `prompts.jsonl` before you share it.

**Other machines.** To test prompts from the rest of your fleet, copy or mount that machine's home directory (only `.claude`, `.codex`, `.gemini`, `.local/share/opencode` and `.hermes/sessions` are read) and run `python3 evals/mine.py harvest --home /path/to/copy --out prompts-spark.jsonl`. **Cursor, aider and anything else:** paste prompts into a text file separated by blank lines and use `harvest --from-file prompts.txt`.

## Arena: test harnesses and models head to head

`evals/arena.py` runs the same tasks through each harness and model you have and grades the results. Harnesses are Claude Code, Codex, Gemini CLI, OpenCode and Hermes; models include API models and those a Spark or another box serves on your LAN. It ranks the outputs and writes a leaderboard plus outcome rows that Maestro's decision model can learn from. It is a repo tool, so run it from a clone. [ARENA-RUNBOOK.md](ARENA-RUNBOOK.md) is the step-by-step guide, and the Claude Code skill in `.claude/skills/arena/` follows it.

```bash
python3 evals/arena.py seats                          # contestant seats on this machine (id, agentic|chat, family, price)
python3 evals/arena.py run --suite verifiable --items V.jsonl --seats 'claude-code:haiku,claude-code:sonnet' \
    --tasks 5 --dry-run                               # what it would run and roughly cost; sends nothing
python3 evals/arena.py run ... (same without --dry-run) --run-id first
python3 evals/arena.py report first --html            # report.md and a self-contained report.html in evals/arena/runs/first/
python3 evals/arena.py export first                   # outcomes in the evals/alpha format (no answer text)
```

**Seats.** A seat is a harness, a model and an effort:

- each CLI on your PATH, with your `maestro-router models` choices or the suggestions (`claude-code:opus`, `codex:gpt-6.1-sol@high`, ...): agentic;
- every model a local server already serves (`local:spark/<model>`): chat;
- the same served models driven by OpenCode or Hermes (`opencode:spark/<model>`): agentic;
- API models with a key (`api:openai/gpt-6.1-sol`): chat.

`evals/arena/seats.json` adds, removes or pins seats and names the judges; see `seats.example.json`.

**Suites.**

- **verifiable**: checker-graded items. The raw answer is checked first, then the same extraction variants for every seat.
- **agentic**: fixtures in `evals/arena/fixtures/`. Each harness works headless in a fresh temp git copy, then the hidden tests decide pass or fail. The run records the diff, time, exit status, usage and cost, and any write outside the work tree, which fails the cell.
- **open**: free-form prompts, ranked by judges.

**How open answers are ranked.** The lead judge writes the criteria before it sees any answer. Then a panel ranks the anonymized, shuffled answers, every judge twice with the order reversed. The panel is the strongest seat of each model family, and the report warns when there is only one family. Bradley-Terry scores with bootstrap intervals turn the rankings into a score per seat. The report also checks position bias, judge agreement, length bias and each family's self-preference.

**The report** gives, for each suite:

- pass rate (or BT score) with 95% intervals, dollars and seconds per success, and breakdowns by job and difficulty;
- the value frontier: the cheapest seat whose interval reaches the best seat's rate;
- a routing replay: Maestro's router against always-X and the hindsight-best seat.

**Runs.** Runs are resumable: rerun the same `--run-id` and finished cells are skipped. Concurrency defaults to 2. Cells are skipped while a subscription window is nearly empty or cooling down. Nothing new starts once `--budget-usd` (default $5) of estimated spend is reached.

**Cost warning.** Every cell is a real call on your plan, your API key or your GPU. Dry-run first and start small.

Task text goes only to the seat being tested, and keys are never read or logged. Results stay in `evals/arena/runs/`, which is gitignored.

A 5-item smoke run of `claude-code:haiku` and `claude-code:sonnet` on the alpha pilot's verifiable items (all `code-fn`) cost $0.13 in API-equivalent dollars. Both seats passed 5/5 (95% interval 57-100%). Sonnet passed 4/5 on the raw answer and 5/5 with extraction variants. Sonnet was faster (5.8 s vs 12.4 s per task) and cheaper ($0.011 vs $0.015 per task, since haiku wrote more tokens).

## Development

```bash
python -m unittest discover -s tests -v
python evals/run.py --classifier rules --split test
python evals/train.py        # rebuild skills/maestro/scripts/maestro/maestro_model.json from evals/train.jsonl
python evals/train_s1.py [--check]   # retrain (or reproduce) data/s1.bin; numpy, never shipped
python evals/alpha.py gate   # the S1 alpha CI gate (passes with a note until outcome data exists)
python evals/mine.py --help  # score Maestro on your own harness prompts (see above)
python3 evals/arena.py --help   # harnesses and models head to head (docs/ARENA-RUNBOOK.md)
python evals/build_catalog.py --models-dev <models.dev checkout>   # rebuild data/catalog.json.gz (Python 3.11+)
python3 site/serve.py        # the website on http://127.0.0.1:8080 (PORT and HOST override; Railway runs it from site/Dockerfile)
```

Design notes: [DESIGN.md](DESIGN.md). Apache-2.0 licensed.
