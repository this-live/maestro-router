import unittest

from conf.parser import ParseError, parse


class ParserTest(unittest.TestCase):
    def test_flat(self):
        self.assertEqual(parse("a = 1\n# c\n; c\n\nb=two words\n"), {"a": "1", "b": "two words"})

    def test_error_line(self):
        with self.assertRaisesRegex(ParseError, "line 2"):
            parse("a = 1\noops\n")


if __name__ == "__main__":
    unittest.main()
