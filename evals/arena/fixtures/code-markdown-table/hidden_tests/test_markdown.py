import os
import subprocess
import sys
import unittest

from report.formats import to_markdown, to_text

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

EXPECTED = """| team | region | requests | cost_usd |
| :--- | :--- | ---: | ---: |
| core | eu-west | 1200 | 34.50 |
| web | us-east | 98000 | 812.07 |
| data \\| ml | us-east | 450 |  |
| infra | ap-south | 7 | 0.21 |"""


class Markdown(unittest.TestCase):
    def test_example(self):
        self.assertEqual(to_markdown(["team", "region", "requests", "cost_usd"],
                                     [["core", "eu-west", "1200", "34.50"]]),
                         "| team | region | requests | cost_usd |\n| :--- | :--- | ---: | ---: |\n"
                         "| core | eu-west | 1200 | 34.50 |")

    def test_mixed_column_is_text(self):
        out = to_markdown(["id", "v"], [["1", "4"], ["2a", "-3.5"], ["3", "12"]])
        self.assertEqual(out.splitlines()[1], "| :--- | ---: |")

    def test_cli(self):
        p = subprocess.run([sys.executable, "-m", "report", "data/usage.csv", "--format", "md"], cwd=ROOT,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.rstrip("\n"), EXPECTED)

    def test_text_still_default(self):
        p = subprocess.run([sys.executable, "-m", "report", "data/usage.csv"], cwd=ROOT,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
        self.assertEqual(p.stdout.splitlines()[0].split(), ["team", "region", "requests", "cost_usd"])
        self.assertEqual(to_text(["a"], [["1"]]), "a\n-\n1")


if __name__ == "__main__":
    unittest.main()
