import argparse
import json
import os

DEFAULTS = {
    "host": "127.0.0.1",
    "port": 8000,
    "debug": False,
    "workers": 2,
    "allowed_hosts": ["localhost"],
    "log_level": "info",
}

TRUE = ("1", "true", "yes", "on")
FALSE = ("0", "false", "no", "off")


class ConfigError(Exception):
    pass


def _parser():
    # No defaults here: None means "not given on the command line", so lower layers survive.
    ap = argparse.ArgumentParser(prog="app")
    ap.add_argument("--config", default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--debug", dest="debug", action="store_true", default=None)
    ap.add_argument("--no-debug", dest="debug", action="store_false")
    ap.add_argument("--log-level", default=None)
    return ap


def _coerce(var, raw, default):
    if isinstance(default, bool):
        low = raw.strip().lower()
        if low in TRUE:
            return True
        if low in FALSE:
            return False
        raise ConfigError("%s: expected a boolean, got %r" % (var, raw))
    if isinstance(default, int):
        try:
            return int(raw.strip())
        except ValueError:
            raise ConfigError("%s: expected an integer, got %r" % (var, raw))
    if isinstance(default, list):
        return [part.strip() for part in raw.split(",") if part.strip()]
    return raw


def _config_path(args, environ):
    explicit = args.config or environ.get("APP_CONFIG")
    if explicit:
        path = os.path.abspath(explicit)
        if not os.path.isfile(path):
            raise ConfigError("config file not found: %s" % explicit)
        return path
    path = os.path.abspath("app.json")
    return path if os.path.isfile(path) else None


def load_config(argv=None, environ=None):
    environ = os.environ if environ is None else environ
    args = _parser().parse_args(argv)
    cfg = dict(DEFAULTS)
    path = _config_path(args, environ)
    if path:
        with open(path, encoding="utf-8") as f:
            try:
                cfg.update(json.load(f))
            except ValueError as e:
                raise ConfigError("%s: invalid JSON (%s)" % (path, e))
    for key, default in DEFAULTS.items():
        var = "APP_" + key.upper()
        if var in environ:
            cfg[key] = _coerce(var, environ[var], default)
    for key in ("host", "port", "workers", "log_level", "debug"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    return cfg
