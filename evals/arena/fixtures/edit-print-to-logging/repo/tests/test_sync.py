import contextlib
import io
import unittest

from syncer.store import MemoryStore
from syncer.sync import sync


class SyncTest(unittest.TestCase):
    def test_counts(self):
        src = MemoryStore("src", {"a": 1, "b": 2, "c": 3})
        dst = MemoryStore("dst", {"b": 0}, readonly_keys={"c"})
        with contextlib.redirect_stdout(io.StringIO()):
            result = sync(src, dst)
        self.assertEqual(result, {"created": 1, "updated": 1, "failed": 1})
        self.assertEqual(dst.data, {"a": 1, "b": 2})


if __name__ == "__main__":
    unittest.main()
