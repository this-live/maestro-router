import math
import time

from . import config
from .http import json_response
from .ratelimit import TokenBucket

ITEMS = [{"id": 1, "name": "anvil"}, {"id": 2, "name": "rope"}]


class App:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.buckets = {}

    def _bucket(self, key):
        b = self.buckets.get(key)
        if b is None:
            b = self.buckets[key] = TokenBucket(config.RATE_LIMIT_BURST, config.RATE_LIMIT_PER_SEC, self.clock)
        return b

    def handle(self, method, path, headers=None):
        headers = headers or {}
        if method == "GET" and path == "/health":
            return json_response(200, {"ok": True})
        if method == "GET" and path == "/items":
            key = headers.get("X-Api-Key")
            if key not in config.API_KEYS:
                return json_response(401, {"error": "unauthorized"})
            bucket = self._bucket(key)
            if not bucket.allow():
                wait = max(1, math.ceil(bucket.retry_after() - 1e-9))
                return json_response(429, {"error": "rate_limited"}, Retry_After=str(wait))
            return json_response(200, {"items": ITEMS, "client": config.API_KEYS[key]})
        return json_response(404, {"error": "not_found"})
