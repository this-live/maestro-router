import contextlib
import io
import os
import re
import tempfile
import unittest

import textkit
from textkit import cli, report, strings

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OLD = ("toTitleCase", "countWords", "truncateText", "normalizeSpace")


class Renamed(unittest.TestCase):
    def test_new_names(self):
        self.assertEqual(strings.to_title_case("the lord OF the rings"), "The Lord of the Rings")
        self.assertEqual(strings.count_words("it's a  dog's life"), 4)
        self.assertEqual(strings.truncate_text("one two three four", 12), "one two...")
        self.assertEqual(strings.normalize_space("  a \n\t b "), "a b")
        for name in ("to_title_case", "count_words", "truncate_text", "normalize_space"):
            self.assertTrue(callable(getattr(textkit, name)), name)
            self.assertIn(name, textkit.__all__)

    def test_old_names_gone(self):
        for mod in (textkit, strings, report):
            for name in OLD:
                self.assertFalse(hasattr(mod, name), "%s.%s still exists" % (mod.__name__, name))
        for base, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in ("hidden_tests", "__pycache__", ".git")]
            for f in files:
                if f.endswith(".py"):
                    with open(os.path.join(base, f), encoding="utf-8") as fh:
                        src = fh.read()
                    for name in OLD:
                        self.assertIsNone(re.search(r"\b%s\b" % name, src), "%s still mentions %s" % (f, name))

    def test_behaviour(self):
        s = report.summary("hello   world", "lorem ipsum dolor sit amet, consectetur", 20)
        self.assertEqual(s, {"title": "Hello World", "words": 6, "teaser": "lorem ipsum..."})
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write("a tale OF two cities\nIt was the best of times, it was the worst of times.")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(cli.main(["summary", p]), 0)
                self.assertEqual(cli.main(["title", "war and peace"]), 0)
        self.assertEqual(buf.getvalue(), "A Tale of Two Cities (12 words)\n"
                                         "It was the best of times, it was the worst of times.\nWar and Peace\n")


if __name__ == "__main__":
    unittest.main()
