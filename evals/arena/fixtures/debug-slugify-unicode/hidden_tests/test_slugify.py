# -*- coding: utf-8 -*-
import unittest

from blog.posts import build_index, unique_slug
from blog.slug import slugify


class Slugify(unittest.TestCase):
    def test_accents(self):
        self.assertEqual(slugify("Café Déjà Vu"), "cafe-deja-vu")
        self.assertEqual(slugify("Straße nach Köln"), "strasse-nach-koln")
        self.assertEqual(slugify("Smørrebrød & Æbleskiver"), "smorrebrod-aebleskiver")
        self.assertEqual(slugify("naïve façade"), "naive-facade")

    def test_collapse_and_strip(self):
        self.assertEqual(slugify("  Hello,   World!  "), "hello-world")
        self.assertEqual(slugify("C++ / Rust: 10 tips"), "c-rust-10-tips")
        self.assertEqual(slugify("--already-slugged--"), "already-slugged")

    def test_truncation(self):
        self.assertEqual(slugify("a" * 30 + " " + "b" * 40, max_length=31), "a" * 30)
        self.assertEqual(slugify("x" * 100), "x" * 60)
        s = slugify("Ünïcödé — " * 20, max_length=25)
        self.assertLessEqual(len(s), 25)
        self.assertFalse(s.endswith("-"))
        self.assertTrue(s.startswith("unicode-unicode"))

    def test_non_latin_dropped(self):
        self.assertEqual(slugify("你好"), "")
        self.assertEqual(slugify("Tokyo 東京 2024"), "tokyo-2024")
        self.assertEqual(unique_slug("你好", set()), "post")

    def test_index(self):
        idx = build_index([{"title": "Café"}, {"title": "Cafe"}, {"title": "café!"}])
        self.assertEqual([s for s, _ in idx], ["cafe", "cafe-2", "cafe-3"])


if __name__ == "__main__":
    unittest.main()
