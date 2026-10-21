import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run(args, cwd):
    return subprocess.run([sys.executable] + args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True, timeout=60)


class Packaging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = run(["scripts/build.py"], ROOT)
        cls.tmp = tempfile.mkdtemp()
        cls.pyz = os.path.join(cls.tmp, "copied.pyz")
        if cls.build.returncode == 0 and os.path.exists(os.path.join(ROOT, "dist", "mytool.pyz")):
            shutil.copy(os.path.join(ROOT, "dist", "mytool.pyz"), cls.pyz)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_build_ok(self):
        self.assertEqual(self.build.returncode, 0, self.build.stderr)
        self.assertTrue(os.path.isfile(self.pyz))

    def test_pyz_anywhere(self):
        with open(os.path.join(ROOT, "VERSION")) as f:
            version = f.read().strip()
        p = run([self.pyz, "--version"], self.tmp)
        self.assertEqual((p.returncode, p.stdout.strip()), (0, "mytool %s" % version), p.stderr)
        p = run([self.pyz, "hello", "Ada"], self.tmp)
        self.assertEqual(p.stdout, "hello, Ada\n", p.stderr)
        p = run([self.pyz, "sum", "1", "2", "3.5"], self.tmp)
        self.assertEqual(p.stdout, "6.5\n", p.stderr)

    def test_python_m(self):
        p = run(["-m", "mytool", "--version"], ROOT)
        self.assertEqual((p.returncode, p.stdout.strip()), (0, "mytool 1.4.2"), p.stderr)
        p = run(["-m", "mytool", "sum", "1", "2"], ROOT)
        self.assertEqual(p.stdout, "3\n", p.stderr)

    def test_rebuild(self):
        out = os.path.join(self.tmp, "again.pyz")
        p = run(["scripts/build.py", out], ROOT)
        self.assertEqual(p.returncode, 0, p.stderr)
        p = run([out, "hello", "Bo"], self.tmp)
        self.assertEqual(p.stdout, "hello, Bo\n", p.stderr)


if __name__ == "__main__":
    unittest.main()
