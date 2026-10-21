import os


def env_int(name, default):
    raw = os.environ.get(name)
    return default if raw in (None, "") else int(raw)


def env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def env_list(name, default=""):
    return [p.strip() for p in os.environ.get(name, default).split(",") if p.strip()]
