import unittest

from media.lru import LRUCache
from media.thumbs import thumbnail_for


class ThumbsTest(unittest.TestCase):
    def test_thumb(self):
        self.assertEqual(thumbnail_for("a.png"), "thumb:a.png@64x64")
        self.assertIsNone(thumbnail_for("notes.txt", "large"))

    def test_cache_basic(self):
        c = LRUCache(2)
        c.put("a", 1)
        self.assertEqual(c.get("a"), 1)
        self.assertEqual(c.get("zz", "dflt"), "dflt")
        self.assertEqual((c.hits, c.misses), (1, 1))

    def test_capacity(self):
        with self.assertRaises(ValueError):
            LRUCache(0)


if __name__ == "__main__":
    unittest.main()
