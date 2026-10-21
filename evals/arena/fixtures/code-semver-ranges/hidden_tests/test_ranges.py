import os
import subprocess
import sys
import unittest

from semverlite import max_satisfying, satisfies

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TRUE = [
    ("1.2.3", "1.2.3"), ("1.2.3", "=1.2.3"), ("1.2.3", ">=1.2.3"), ("1.2.4", ">1.2.3"), ("1.2.2", "<1.2.3"),
    ("1.2.3", "<=1.2.3"), ("1.2.3", ">= 1.2.3"), ("1.8.1", "^1.2.3"), ("1.2.3", "^1.2.3"), ("0.2.5", "^0.2.3"),
    ("0.0.3", "^0.0.3"), ("1.2.9", "~1.2.3"), ("1.2.0", "1.2"), ("1.2.99", "1.2.x"), ("1.2.7", "1.2.*"),
    ("1.9.9", "1.x"), ("1.0.0", "1"), ("5.0.0", "*"), ("0.0.1", ""), ("3.0.0", "x"), ("2.3.4", "1.2.3 - 2.3.4"),
    ("1.2.3", "1.2.3 - 2.3.4"), ("1.5.0", ">=1.2.0 <2.0.0"), ("3.1.0", "^1.2 || >=3.0.0"),
    ("1.4.0", "^1.2 || >=3.0.0"), ("1.9.0", "^1.2"), ("0.2.9", "^0.2"), ("1.2.5", "~1.2"), ("1.3.0", "~1"),
    ("1.2.3-alpha.7", ">1.2.3-alpha.3"), ("1.2.3-beta", "^1.2.3-alpha"), ("1.2.4", "^1.2.3-alpha"),
    ("1.2.3-alpha.10", ">1.2.3-alpha.9"), ("1.2.3-alpha.beta", ">1.2.3-alpha.9"), ("1.2.3+build.5", "1.2.3"),
    ("1.2.3", "  >1.0.0   <1.3.0  "), ("2.0.0", "<1.0.0 || 2.0.0 || 3.x"),
]
FALSE = [
    ("1.2.3", ">1.2.3"), ("2.0.0", "^1.2.3"), ("0.3.0", "^0.2.3"), ("0.0.4", "^0.0.3"), ("1.3.0", "~1.2.3"),
    ("1.2.2", "~1.2.3"), ("1.1.9", "1.2.x"), ("2.0.0", "1.x"), ("1.3.0", "1.2"), ("2.3.5", "1.2.3 - 2.3.4"),
    ("1.2.2", "1.2.3 - 2.3.4"), ("2.5.0", "^1.2 || >=3.0.0"), ("2.0.0", "^1.2"), ("0.3.0", "^0.2"),
    ("1.3.0", "~1.2"), ("2.0.0", "~1"), ("3.4.5-alpha.9", ">1.2.3-alpha.3"), ("1.2.3-alpha.7", ">=1.0.0"),
    ("2.0.0-rc.1", "^1.2.3"), ("1.2.3-alpha.1", "1.2.3"), ("1.2.3-alpha.3", ">1.2.3-alpha.3"),
    ("1.0.0-rc.1", "*"), ("1.2.3", "<1.2.3"), ("1.5.0", ">=1.2.0 <1.5.0"), ("1.2.3", "^1.2.4 || ~2.0.0"),
]


class Satisfies(unittest.TestCase):
    def test_true(self):
        for v, r in TRUE:
            self.assertTrue(satisfies(v, r), "%s should satisfy %r" % (v, r))

    def test_false(self):
        for v, r in FALSE:
            self.assertFalse(satisfies(v, r), "%s should not satisfy %r" % (v, r))

    def test_invalid(self):
        for r in (">>1.2.3", "^1.2.3.4", "abc", ">=", "1.2.3 -", "~", "1.2.3 - 2.0.0 - 3.0.0", "1.2.3 || 1.a"):
            with self.assertRaises(ValueError, msg=r):
                satisfies("1.2.3", r)
        with self.assertRaises(ValueError):
            satisfies("not-a-version", "*")


class MaxSatisfying(unittest.TestCase):
    def test_max(self):
        vs = ["1.2.3", "1.2.4", "1.3.0-beta", "2.0.0", "1.10.1", "v1.9.0"]
        self.assertEqual(max_satisfying(vs, "^1.2.0"), "1.10.1")
        self.assertEqual(max_satisfying(vs, "~1.2.0"), "1.2.4")
        self.assertEqual(max_satisfying(vs, "^1.3.0-alpha"), "1.10.1")
        self.assertEqual(max_satisfying(["1.3.0-beta", "1.2.9"], ">=1.3.0-alpha <1.4.0"), "1.3.0-beta")
        self.assertEqual(max_satisfying(vs, ">=1.9.0 <1.10.0"), "v1.9.0")
        self.assertIsNone(max_satisfying(vs, "^3"))
        self.assertIsNone(max_satisfying([], "*"))

    def test_cli(self):
        p = subprocess.run([sys.executable, "-m", "semverlite", "resolve", "^1.2", "1.2.0", "1.9.3", "2.0.0"],
                           cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                           timeout=30)
        self.assertEqual((p.returncode, p.stdout.strip()), (0, "1.9.3"))


if __name__ == "__main__":
    unittest.main()
