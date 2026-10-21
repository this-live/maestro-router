# Contributing to Maestro

Maestro is one job: route a task to the right model the user already has.

- Maestro v3 is a package: `skills/maestro/scripts/maestro/` (`router.py` = the skill-mode router, `providers.py`, `clients.py`, `quota.py` = the v3 provider layer, `catalog.py`, `score.py`, `discover.py` = catalog, scoring and `maestro-router discover`, `server.py` + `ui/index.html` + `onboard.py` = `serve`, the web UI, `setup` and `hermes`, `__init__.py` re-exports the router API, `__main__.py` = CLI, `maestro_model.json`, `data/providers.json` and `data/catalog.json.gz` next to them). Run it as `python3 skills/maestro/scripts/maestro <cmd>`. Stdlib only, Python >= 3.9 (no `match`, no `tomllib`, no runtime `X | Y` types; keep `from __future__ import annotations`). Do not add dependencies.
- Keep it lean (`router.py` target under ~1,750 lines; it is over, do not grow it; new v3 code goes in its own module). Prefer a function over a class. No plugin systems.
- `route` must stay fast: no network on the rules path, inventory read from `~/.maestro/inventory.json` (re-detected only when missing or older than 24 h). `route` never imports `providers`, `clients`, `quota`, `catalog`, `score`, `discover`, `server` or `onboard`. Only `detect`, `doctor`, `run`, `keys`, `install --statusline`, `discover`, `serve`, `setup` and `hermes` do slow work; the network is scanned only with `detect --scan`.
- Secrets. Detection checks that API key env vars, `~/.maestro/keys.env` entries and credential files exist, nothing more; subscription logins (Claude Code, Codex, Gemini CLI, ...) are never read or forwarded. A key value may be read only to authenticate to its own provider (an auth header, or a `local_endpoints` entry's `api_key_env` Bearer token) and is never printed, logged, returned by an API, put in an error message, or stored anywhere but `~/.maestro/keys.env` (mode 0600, written by `maestro-router keys set`). Error text from upstream goes through `providers.scrub`. The server never returns a key from any endpoint (the UI is write-only), the admin token (`~/.maestro/admin_token`, mode 0600) is not in any API response, and `--host` beyond loopback requires `MAESTRO_API_KEY`. Tests use fake servers on 127.0.0.1 and never the real network; `tests/helpers.py` stops probes of this machine's own model ports (`PROBE_BUILTIN_PORTS`, `MAESTRO_NO_BUILTIN_PORTS`), so import it from every test module.
- `MAESTRO_HOME` overrides `~/.maestro`; tests and manual runs should point it at a temp dir.
- Public releases: the public repository is a fresh one-commit export, never this repository's history. Before any push to it, run `scripts/check-history.sh` in the checkout you push from; it refuses the private root commit and the old `platform/` tree. CI runs it in the public repository, and the export's pre-push hook runs it only if that checkout uses its own hooks (`git config core.hooksPath .git/hooks`; a global `core.hooksPath` otherwise wins).
- `docs/DESIGN.md` is the spec. Update it, and list the deviation at the end, when behavior changes.

Commands:

    python -m unittest discover -s tests -v                     # unit tests (stdlib unittest only)
    python evals/run.py --classifier rules --split dev          # tune here (even line index of evals/prompts.jsonl)
    python evals/run.py --classifier rules --split test         # report / CI gate (odd index); do not tune on it
    python evals/train.py                                       # rebuild skills/maestro/scripts/maestro/maestro_model.json
    python evals/build_providers.py --models-dev <models.dev>   # rebuild skills/maestro/scripts/maestro/data/providers.json
    python evals/build_catalog.py --models-dev <models.dev>     # rebuild data/catalog.json.gz (Python 3.11+; hand-verified sizes in evals/hf_params.json, model-card benchmarks in evals/extra_benchmarks.json)
    python skills/maestro/scripts/maestro doctor             # see what Maestro detected

Changing keywords, difficulty scoring or the model? Tune on `--split dev` only, then re-run `--split test` once and update the numbers in README.md and the CI thresholds in `.github/workflows/ci.yml` (test result minus 5 points; severe under-routing plus 3). `evals/train.jsonl` (the model's training corpus) must never contain prompts from `evals/prompts.jsonl`; retrain with `evals/train.py` after editing it and commit the regenerated `maestro_model.json`. The Jev contract test needs `pip install typesafe-sdk==0.7.2` and is skipped otherwise.
