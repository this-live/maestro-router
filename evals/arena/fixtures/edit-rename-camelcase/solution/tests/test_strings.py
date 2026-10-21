import unittest

from textkit import count_words, normalize_space, to_title_case, truncate_text
from textkit.report import summary


class StringsTest(unittest.TestCase):
    def test_title(self):
        self.assertEqual(to_title_case("the lord OF the rings"), "The Lord of the Rings")
        self.assertEqual(to_title_case("what it is for"), "What It Is For")

    def test_count(self):
        self.assertEqual(count_words("it's a  dog's life"), 4)

    def test_truncate(self):
        self.assertEqual(truncate_text("one two three four", 12), "one two...")
        self.assertEqual(truncate_text("short", 12), "short")

    def test_space(self):
        self.assertEqual(normalize_space("  a \n\t b "), "a b")

    def test_summary(self):
        s = summary("hello   world", "lorem ipsum dolor sit amet, consectetur", 20)
        self.assertEqual(s, {"title": "Hello World", "words": 6, "teaser": "lorem ipsum..."})


if __name__ == "__main__":
    unittest.main()
