import unittest

from media import thumbs
from media.lru import LRUCache, memoize


class LRU(unittest.TestCase):
    def test_evicts_least_recent(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        self.assertNotIn("a", c)
        self.assertEqual((c.get("b"), c.get("c")), (2, 3))

    def test_get_refreshes(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        c.get("a")
        c.put("c", 3)
        self.assertIn("a", c)
        self.assertNotIn("b", c)
        self.assertEqual(c.keys(), ["a", "c"])

    def test_update_existing_no_eviction(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("a", 10)
        self.assertEqual(len(c), 2)
        self.assertEqual(c.keys(), ["b", "a"])
        self.assertEqual(c.get("a"), 10)
        c.put("c", 3)
        self.assertEqual(c.keys(), ["a", "c"])

    def test_capacity_one(self):
        c = LRUCache(1)
        for i in range(5):
            c.put(i, i)
            self.assertEqual(c.keys(), [i])

    def test_counters(self):
        c = LRUCache(3)
        c.put("x", None)
        self.assertIsNone(c.get("x", "d"))
        self.assertEqual(c.get("y", "d"), "d")
        self.assertEqual((c.hits, c.misses), (1, 1))


class Memoize(unittest.TestCase):
    def test_none_results_cached(self):
        calls = []

        @memoize(LRUCache(8))
        def f(x):
            calls.append(x)
            return None if x % 2 else x * 10

        self.assertEqual([f(1), f(2), f(1), f(2), f(1)], [None, 20, None, 20, None])
        self.assertEqual(calls, [1, 2])

    def test_wraps(self):
        @memoize(LRUCache(2))
        def named(x):
            """doc"""
            return x

        self.assertEqual((named.__name__, named.__doc__), ("named", "doc"))

    def test_thumbnails(self):
        before = len(thumbs.RENDERS)
        for _ in range(3):
            thumbs.thumbnail_for("review-a.png")
            thumbs.thumbnail_for("review-notes.txt")
            thumbs.thumbnail_for("review-a.png", "large")
        self.assertEqual(len(thumbs.RENDERS) - before, 3)


if __name__ == "__main__":
    unittest.main()
