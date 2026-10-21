import logging

logger = logging.getLogger(__name__)


class Item:
    def __init__(self, key, value):
        self.key = key
        self.value = value


class MemoryStore:
    def __init__(self, name, data=None, readonly_keys=()):
        self.name = name
        self.data = dict(data or {})
        self.readonly_keys = set(readonly_keys)

    def list(self):
        items = [Item(k, v) for k, v in sorted(self.data.items())]
        logger.debug("%s lists %d keys", self.name, len(items))
        return items

    def exists(self, key):
        return key in self.data

    def _write(self, item):
        if item.key in self.readonly_keys:
            raise IOError("%s is read-only" % item.key)
        if item.value is None:
            logger.warning("writing empty value for %s", item.key)
        self.data[item.key] = item.value

    create = _write
    update = _write
