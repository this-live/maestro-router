import unittest

from calc import EvalError, evaluate


class CalcTest(unittest.TestCase):
    def test_precedence(self):
        self.assertEqual(evaluate("1 + 2 * 3"), 7)
        self.assertEqual(evaluate("(1 + 2) * 3"), 9)
        self.assertEqual(evaluate("2 - 3 - 4"), -5)
        self.assertEqual(evaluate("8 / 4 / 2"), 1)
        self.assertEqual(evaluate("7 / 2"), 3.5)
        self.assertEqual(evaluate(" .5 + 1. "), 1.5)

    def test_errors(self):
        for bad in ("", "1 +", "(1", "1 2", ")", "2 $ 3", "1 / 0"):
            with self.assertRaises(EvalError, msg=bad):
                evaluate(bad)


if __name__ == "__main__":
    unittest.main()
