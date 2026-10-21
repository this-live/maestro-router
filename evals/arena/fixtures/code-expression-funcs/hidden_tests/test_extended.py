import os
import subprocess
import sys
import unittest

from calc import EvalError, evaluate

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class Unary(unittest.TestCase):
    def test_unary(self):
        cases = {"-3": -3, "2 * -3": -6, "--4": 4, "-(2+3)": -5, "+5": 5, "3 - -2": 5, "-+-1": 1, "4 / -2": -2}
        for text, want in cases.items():
            self.assertEqual(evaluate(text), want, text)


class Power(unittest.TestCase):
    def test_power(self):
        cases = {"2^3^2": 512, "-2^2": -4, "2^-1": 0.5, "(-2)^2": 4, "2*3^2": 18, "2^3*2": 16, "2^0.5^2": 2 ** 0.25,
                 "10 - 2^3": 2, "-2^-2": -0.25}
        for text, want in cases.items():
            self.assertAlmostEqual(evaluate(text), want, places=12, msg=text)


class Names(unittest.TestCase):
    def test_vars(self):
        self.assertEqual(evaluate("x * 2 + y", {"x": 3, "y": 1}), 7)
        self.assertEqual(evaluate("rate_2 * qty", {"rate_2": 0.5, "qty": 4}), 2)
        self.assertEqual(evaluate("-x^2", {"x": 3}), -9)
        self.assertEqual(evaluate("1 + 1"), 2)

    def test_unknown_name(self):
        with self.assertRaises(EvalError) as cm:
            evaluate("x + y", {"x": 1})
        self.assertIn("y", str(cm.exception))
        with self.assertRaises(EvalError):
            evaluate("x")


class Functions(unittest.TestCase):
    def test_funcs(self):
        env = {"x": 0}
        cases = {"max(1, 5, 3)": 5, "min(4, -2)": -2, "abs(-3.5)": 3.5, "round(2.567, 2)": 2.57, "round(3.7)": 4,
                 "max(1, min(2, x))": 1, "abs(min(-1, -5))": 5, "max(7)": 7, "2 * max(1, 2)^2": 8,
                 "-abs(-2)": -2, "min(3, 2^2, -x + 1)": 1}
        for text, want in cases.items():
            self.assertEqual(evaluate(text, env), want, text)

    def test_func_errors(self):
        for bad in ("max()", "abs(1, 2)", "round()", "round(1, 2, 3)", "max(1,)", "min 1", "max(1, 2", "abs()"):
            with self.assertRaises(EvalError, msg=bad):
                evaluate(bad)
        with self.assertRaises(EvalError) as cm:
            evaluate("foo(1)")
        self.assertIn("foo", str(cm.exception))


class Malformed(unittest.TestCase):
    def test_only_evalerror(self):
        bad = ["", "1 +", "(1", "1 2", ")", "2 $ 3", "1 / 0", "^2", "2^", "2 ^ ^ 3", "x y", "(,)", "1,2", "max(,1)",
               "abs(1)(2)", "3 x", "_(", "-", "()", "max((1)", "1.2.3", "x / (y - y)"]
        for text in bad:
            with self.assertRaises(EvalError, msg=text):
                evaluate(text, {"x": 1, "y": 2})


class Cli(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-m", "calc"] + list(args), cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True, timeout=30)

    def test_cli(self):
        p = self.run_cli("x^2 + y", "x=3", "y=0.5")
        self.assertEqual((p.returncode, p.stdout.strip()), (0, "9.5"))
        p = self.run_cli("2^10")
        self.assertEqual(p.stdout.strip(), "1024")
        p = self.run_cli("max(1,")
        self.assertEqual(p.returncode, 1)


if __name__ == "__main__":
    unittest.main()
