import unittest

from textkit import countWords, normalizeSpace, toTitleCase, truncateText
from textkit.report import summary


class StringsTest(unittest.TestCase):
    def test_title(self):
        self.assertEqual(toTitleCase("the lord OF the rings"), "The Lord of the Rings")
        self.assertEqual(toTitleCase("what it is for"), "What It Is For")

    def test_count(self):
        self.assertEqual(countWords("it's a  dog's life"), 4)

    def test_truncate(self):
        self.assertEqual(truncateText("one two three four", 12), "one two...")
        self.assertEqual(truncateText("short", 12), "short")

    def test_space(self):
        self.assertEqual(normalizeSpace("  a \n\t b "), "a b")

    def test_summary(self):
        s = summary("hello   world", "lorem ipsum dolor sit amet, consectetur", 20)
        self.assertEqual(s, {"title": "Hello World", "words": 6, "teaser": "lorem ipsum..."})


if __name__ == "__main__":
    unittest.main()
