import time

_MISSING = object()


class TTLCache:
    """Tiny TTL cache. Keys are tuples whose first element is the kind ("user", "team", ...)."""

    def __init__(self, ttl=300.0, clock=time.monotonic):
        self.ttl = ttl
        self.clock = clock
        self._data = {}

    def get(self, key, default=None):
        hit = self._data.get(key, _MISSING)
        if hit is _MISSING:
            return default
        value, expires = hit
        if self.clock() >= expires:
            del self._data[key]
            return default
        return value

    def set(self, key, value):
        self._data[key] = (value, self.clock() + self.ttl)

    def delete(self, key):
        self._data.pop(key, None)

    def delete_where(self, predicate):
        """Drop every entry whose key satisfies `predicate(key)`."""
        for key in [k for k in self._data if predicate(k)]:
            del self._data[key]

    def clear(self):
        self._data.clear()

    def __len__(self):
        return len(self._data)
