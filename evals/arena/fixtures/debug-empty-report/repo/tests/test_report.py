import os
import unittest

from stats.report import report, summarize

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


class ReportTest(unittest.TestCase):
    def test_summarize(self):
        s = summarize([1, 2, 3, 4])
        self.assertEqual(s, {"count": 4, "mean": 2.5, "min": 1, "max": 4})

    def test_sample(self):
        self.assertEqual(report(os.path.join(DATA, "sample.csv")), "count=4 mean=10.688 min=3.25 max=20")


if __name__ == "__main__":
    unittest.main()
