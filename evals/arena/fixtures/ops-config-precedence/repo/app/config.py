import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = {
    "host": "127.0.0.1",
    "port": 8000,
    "debug": False,
    "workers": 2,
    "allowed_hosts": ["localhost"],
    "log_level": "info",
}


class ConfigError(Exception):
    pass


def _parser():
    ap = argparse.ArgumentParser(prog="app")
    ap.add_argument("--config", default=None)
    ap.add_argument("--host", default=DEFAULTS["host"])
    ap.add_argument("--port", type=int, default=DEFAULTS["port"])
    ap.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--log-level", default=DEFAULTS["log_level"])
    return ap


def load_config(argv=None, environ=None):
    environ = os.environ if environ is None else environ
    args = _parser().parse_args(argv)
    cfg = dict(DEFAULTS)
    for key in DEFAULTS:
        value = environ.get("APP_" + key.upper())
        if value is not None:
            cfg[key] = value
    path = os.path.join(HERE, args.config or environ.get("APP_CONFIG") or "app.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg.update(json.load(f))
    for key in ("host", "port", "workers", "log_level"):
        cfg[key] = getattr(args, key)
    if args.debug:
        cfg["debug"] = True
    return cfg
