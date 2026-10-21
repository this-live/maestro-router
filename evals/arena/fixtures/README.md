# Arena fixtures

Small, self-contained Python repositories used to compare model + harness pairs (Claude Code, Codex, Gemini CLI,
OpenCode, ...) on real agentic work: the harness gets the repo and the task, has to read files, edit code and
usually run tests, and is then graded objectively by tests it never saw.

## Layout

```
evals/arena/fixtures/<id>/
  meta.json        id, job, difficulty_intent, language, timeout_s, test_cmd, needs_tools (+ visible_test_cmd)
  task.md          the request, exactly as a user would type it to a coding agent in that repo
  repo/            the starting repository (3-25 files, stdlib-only Python, may have visible tests and noise)
  hidden_tests/    unittest files (+ __init__.py and any golden files) used for grading only
  solution/        reference fix as an overlay: full copies of the files that change or are added
                   (a fixture may use solution.patch instead, applied with `git apply` from the repo root)
```

`meta.json` fields:

| field               | meaning                                                                                  |
|---------------------|------------------------------------------------------------------------------------------|
| `id`                | directory name                                                                           |
| `job`               | `code`, `debug`, `review`, `ops`, `edit`, `research` or `data` (Maestro's job labels)    |
| `difficulty_intent` | 0 (any model) .. 3 (only strong models); the intended tier, before any measured results |
| `language`          | `python`                                                                                 |
| `timeout_s`         | wall-clock budget for the harness run (300-1200)                                         |
| `test_cmd`          | grading command, run from the repo root after `hidden_tests/` is copied in               |
| `visible_test_cmd`  | optional: the repo's own tests; they must also pass on the solved repo                   |
| `needs_tools`       | always `true`: the task cannot be solved from the prompt alone                           |

## Grading protocol

1. Copy `repo/` to a fresh directory and run the harness there with `task.md` as the prompt (time limit
   `timeout_s`). The harness never sees `hidden_tests/`, `solution/` or `meta.json`.
2. Copy `hidden_tests/` into the result as `<repo>/hidden_tests/` (replacing anything the agent created there).
3. Run `test_cmd` from the repo root (`python3 -m unittest discover -s hidden_tests -t . -q`). Exit code 0 = solved.

Hidden tests check the behaviour the task asks for and guard behaviour that must not change. They only rely on
names and formats stated in `task.md` or already present in the repo (README, existing code). They never touch
the network, use fake clocks instead of real time, and work on Python 3.9+.

## Mix

30 fixtures: 8 debug, 8 code, 4 edit, 3 review (the PR diff is in `task.md` and already applied in `repo/`),
3 data, 2 ops, 2 research (the answer goes into a file in a prescribed format). Difficulty: 6 x d0, 9 x d1,
9 x d2, 6 x d3.

| id | job | d | task |
|----|-----|---|------|
| debug-empty-report | debug | 0 | ZeroDivisionError on a header-only CSV; report `no data` |
| debug-retry-count | debug | 0 | retry decorator makes one attempt too few and swallows the final error |
| debug-slugify-unicode | debug | 1 | slugify mangles accented titles, double hyphens, trailing `-` after truncation |
| debug-pagination-offbyone | debug | 2 | overlapping pages, missing last page, `per_page=0` crash |
| debug-money-rounding | debug | 2 | float/banker's rounding off by a cent; installments must sum exactly |
| debug-cache-invalidation | debug | 2 | stale team/search caches after updates, limit missing from cache key, mutable cache entries |
| debug-dst-reminders | debug | 3 | daily reminders drift across DST; skipped/repeated local times; fold semantics |
| debug-queue-ordering | debug | 3 | heap compares Job objects, backoff head-of-line blocking, retries lose priority |
| code-todo-json-flag | code | 0 | add `--json` to `todo list` |
| code-markdown-table | code | 0 | add a markdown table formatter and `--format md` |
| code-ini-sections | code | 1 | config parser: sections, inline comments, quoted values, `section()` |
| code-rate-limiter | code | 1 | token bucket + per-key 429 with `Retry-After` |
| code-undo-redo | code | 2 | undo/redo with typing coalescing and history limit, wired into commands |
| code-http-router | code | 2 | path params with converters, static-first matching, 405 + `Allow` |
| code-expression-funcs | code | 3 | calculator: unary ops, right-assoc `^`, variables, functions, CLI vars |
| code-semver-ranges | code | 3 | npm-style range matching incl. caret/tilde/x/hyphen ranges and prerelease rule |
| edit-rename-camelcase | edit | 0 | rename camelCase helpers to snake_case everywhere, no aliases |
| edit-extract-validators | edit | 1 | move validation into a new module, keep re-exports |
| edit-print-to-logging | edit | 1 | replace prefixed prints with per-module loggers at the right levels |
| edit-dataclass-orders | edit | 2 | dicts to dataclasses across modules, JSON export byte-identical |
| review-pr-coupon | review | 1 | coupon-stacking PR: NameError, order dependence, negative totals |
| review-pr-lru | review | 2 | LRU PR: evicts the wrong end, `get` doesn't refresh, memoize ignores `None` |
| review-pr-search-sql | review | 3 | search PR: SQL injection, LIKE wildcards, sort collation, unvalidated limit |
| data-sales-by-region | data | 0 | revenue per region from a messy CSV, exact decimals, JSON report |
| data-merge-customers | data | 1 | merge/dedupe two customer exports by normalized email |
| data-sessionize | data | 2 | sessionize a dirty clickstream (offsets, dupes, 30-min gap) into CSV |
| ops-zipapp-entry | ops | 1 | zipapp build: wrong entry point, missing subpackage, `__main__`, version file |
| ops-config-precedence | ops | 2 | defaults < file < env < flags, env type coercion, config path resolution |
| research-env-vars | research | 1 | table of env vars the service reads and their defaults |
| research-audit-callgraph | research | 3 | which public functions reach `audit.record` (aliases, callbacks, methods) |

## Validation

```
python3 evals/arena/validate_fixtures.py                    # all fixtures, 2 runs each (determinism)
python3 evals/arena/validate_fixtures.py code-undo-redo --runs 1
python3 evals/arena/validate_fixtures.py --python python3.9
```

For each fixture the validator checks the structure, then in fresh temp copies: starting repo + hidden tests must
FAIL, repo + solution + hidden tests must PASS (and the visible tests must pass on the solved repo), every run
repeated with the same outcome, 60 s per run. `tests/test_arena_fixtures.py` runs it (one run each) in the unit
test suite.

## Adding a fixture

Keep it stdlib-only and small, make `task.md` sound like a real request (symptoms, tracebacks, file names; never a
reference to the hidden tests), state every name and format the hidden tests depend on, inject clocks instead of
sleeping, and run the validator on 3.9 and a current Python before committing.
