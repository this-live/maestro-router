Config loading in `app/config.py` is a mess and it bit us in the k8s rollout:

- `APP_PORT=9000` is ignored whenever the config file sets `port`
- `APP_DEBUG=false` turns debug **on**
- if I don't pass `--port` on the command line, the port from the config file is ignored and we always get 8000
- `--config conf/prod.json` is looked up relative to the package directory instead of the current directory, and a `--config` that doesn't exist is silently ignored

What we want (it's in the README too): defaults < config file < `APP_*` env vars < command-line flags, only for values that are actually set at each level. Env values must be converted to the type of the default (int; bool accepting 1/0, true/false, yes/no, on/off in any case; list = comma-separated, items trimmed); a bad value raises `ConfigError` naming the variable. `--no-debug` should exist to force debug off. The config file is `--config`, else `$APP_CONFIG`, else `app.json` if it exists, relative paths resolved against the current directory; an explicitly requested file that's missing is a `ConfigError`. Keep `load_config(argv=None, environ=None)`'s signature.
