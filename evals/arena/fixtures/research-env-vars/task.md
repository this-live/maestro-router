Ops is writing the helm chart and asked which environment variables this service reads. Please go through the code under `service/` and write the answer to `ENV_VARS.md` at the repo root, as a single markdown table:

```
| name | default |
| --- | --- |
| SOME_VAR | 10 |
```

One row per environment variable that code under `service/` actually reads, sorted by name. `default` is the value used when the variable is unset, as written in the code (e.g. `4`, `False`, `info`); write `(required)` if the service can't run without it, or `(none)` if it's optional with no default. Don't change any code.
