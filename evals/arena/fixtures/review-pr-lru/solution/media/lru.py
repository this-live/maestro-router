import functools
from collections import OrderedDict


_MISSING = object()


class LRUCache:
    """Fixed-size cache that evicts the least recently used entry. Both get() and put() count as a use."""

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._data = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key, default=None):
        if key in self._data:
            self.hits += 1
            self._data.move_to_end(key)
            return self._data[key]
        self.misses += 1
        return default

    def put(self, key, value):
        if key in self._data:
            self._data.move_to_end(key)
        elif len(self._data) >= self.capacity:
            self._data.popitem(last=False)
        self._data[key] = value

    def __contains__(self, key):
        return key in self._data

    def __len__(self):
        return len(self._data)

    def keys(self):
        """Keys from least to most recently used."""
        return list(self._data)


def memoize(cache):
    """Cache a function's results in `cache`, keyed by its positional arguments."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args):
            result = cache.get(args, _MISSING)
            if result is _MISSING:
                result = fn(*args)
                cache.put(args, result)
            return result

        return wrapper

    return deco
