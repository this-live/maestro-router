import json
import os
import subprocess
import sys
import tempfile
import unittest

from analysis.sales import summarize_sales

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPECTED = {
    "regions": [
        {"region": "APAC", "orders": 3, "revenue": "309.65"},
        {"region": "EMEA", "orders": 3, "revenue": "172.32"},
        {"region": "LATAM", "orders": 2, "revenue": "169.87"},
        {"region": "NA", "orders": 3, "revenue": "93.29"},
    ],
    "total_revenue": "745.13",
}


class Sales(unittest.TestCase):
    def test_repo_data(self):
        self.assertEqual(summarize_sales(os.path.join(ROOT, "data", "sales.csv")), EXPECTED)

    def test_ties_and_exact_sums(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.csv")
            with open(p, "w") as f:
                f.write("order_id,date,region,sku,qty,unit_price,status\n"
                        "1,2024-01-01,zz,A,3,0.10,paid\n"
                        "2,2024-01-01,AA,A,1,0.30,paid\n"
                        "2,2024-01-01,aa,B,1,0.00,paid\n"
                        "3,2024-01-01,mm,B,1,5.00,CANCELLED\n")
            self.assertEqual(summarize_sales(p), {
                "regions": [{"region": "AA", "orders": 1, "revenue": "0.30"},
                            {"region": "ZZ", "orders": 1, "revenue": "0.30"}],
                "total_revenue": "0.60",
            })

    def test_cli_writes_report(self):
        out = os.path.join(ROOT, "reports", "sales_by_region.json")
        if os.path.exists(out):
            os.remove(out)
        p = subprocess.run([sys.executable, "-m", "analysis.sales", "data/sales.csv"], cwd=ROOT,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
        self.assertEqual(p.returncode, 0, p.stderr)
        with open(out, encoding="utf-8") as f:
            self.assertEqual(json.load(f), EXPECTED)


if __name__ == "__main__":
    unittest.main()
