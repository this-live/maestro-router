import unittest

from semverlite.version import Version, compare


class VersionTest(unittest.TestCase):
    def test_order(self):
        chain = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2", "1.0.0-beta.11",
                 "1.0.0-rc.1", "1.0.0", "1.0.1", "1.10.0", "2.0.0"]
        vs = [Version.parse(v) for v in chain]
        self.assertEqual(sorted(reversed(vs)), vs)

    def test_build_ignored(self):
        self.assertEqual(compare("1.2.3+build.5", "1.2.3"), 0)

    def test_invalid(self):
        for bad in ("1.2", "01.2.3", "1.2.3-", "x.y.z"):
            with self.assertRaises(ValueError):
                Version.parse(bad)


if __name__ == "__main__":
    unittest.main()
