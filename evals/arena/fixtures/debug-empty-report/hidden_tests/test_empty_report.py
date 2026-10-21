import os
import subprocess
import sys
import tempfile
import unittest

from stats.report import report, summarize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class EmptyReport(unittest.TestCase):
    def test_summarize_empty(self):
        self.assertEqual(summarize([]), {"count": 0, "mean": None, "min": None, "max": None})

    def test_report_header_only(self):
        self.assertEqual(report(os.path.join(ROOT, "data", "empty.csv")), "no data")

    def test_report_all_blank(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "blank.csv")
            with open(p, "w") as f:
                f.write("ts,value\n2024-01-01,\n2024-01-02,  \n")
            self.assertEqual(report(p), "no data")

    def test_cli_exit_zero(self):
        p = subprocess.run([sys.executable, "-m", "stats", "report", "data/empty.csv"], cwd=ROOT,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), "no data")

    def test_normal_output_unchanged(self):
        self.assertEqual(report(os.path.join(ROOT, "data", "sample.csv")), "count=4 mean=10.688 min=3.25 max=20")
        self.assertEqual(report(os.path.join(ROOT, "data", "sample.csv"), "latency_ms"),
                         "count=5 mean=142.800 min=88 max=310")
        self.assertEqual(summarize([5]), {"count": 1, "mean": 5.0, "min": 5, "max": 5})


if __name__ == "__main__":
    unittest.main()
