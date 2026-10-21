import os

from .util import env_int

REDIS_URL = os.environ.get("REDIS_URL") or "redis://localhost:6379/0"


def cache_ttl():
    # ttl = int(os.environ.get("CACHE_TTL_SECONDS", 60))  -- removed in 2.3
    return env_int("CACHE_TTL", 300)
