import unittest

from mytool.commands import greet, maths


class CommandsTest(unittest.TestCase):
    def test_hello(self):
        self.assertEqual(greet.hello("Ada"), "hello, Ada")

    def test_sum(self):
        self.assertEqual(maths.fmt(maths.total([1, 2, 3])), "6")
        self.assertEqual(maths.fmt(maths.total([1, 2.5])), "3.5")


if __name__ == "__main__":
    unittest.main()
