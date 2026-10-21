import time

from . import config
from .http import json_response

ITEMS = [{"id": 1, "name": "anvil"}, {"id": 2, "name": "rope"}]


class App:
    def __init__(self, clock=time.monotonic):
        self.clock = clock

    def handle(self, method, path, headers=None):
        headers = headers or {}
        if method == "GET" and path == "/health":
            return json_response(200, {"ok": True})
        if method == "GET" and path == "/items":
            key = headers.get("X-Api-Key")
            if key not in config.API_KEYS:
                return json_response(401, {"error": "unauthorized"})
            return json_response(200, {"items": ITEMS, "client": config.API_KEYS[key]})
        return json_response(404, {"error": "not_found"})
