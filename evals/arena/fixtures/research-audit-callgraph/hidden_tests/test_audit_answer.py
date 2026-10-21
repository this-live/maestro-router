import os
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPECTED = [
    "billing.invoices.create_invoice",
    "billing.invoices.void_invoice",
    "billing.legacy.migrate",
    "billing.legacy.old_void",
    "billing.refunds.escalate",
    "billing.refunds.partial_refund",
    "billing.refunds.refund",
    "billing.reports.post_to_ledger",
    "billing.reports.rebill",
]


class Answer(unittest.TestCase):
    def test_answer(self):
        with open(os.path.join(ROOT, "AUDIT_WRITERS.txt"), encoding="utf-8") as f:
            got = [line.strip() for line in f if line.strip()]
        self.assertEqual(sorted(set(got)), sorted(got), "duplicates or unsorted lines")
        missing = sorted(set(EXPECTED) - set(got))
        extra = sorted(set(got) - set(EXPECTED))
        self.assertEqual((missing, extra), ([], []))
        self.assertEqual(got, EXPECTED)

    def test_code_untouched(self):
        with open(os.path.join(ROOT, "billing", "refunds.py"), encoding="utf-8") as f:
            self.assertIn("on_sent=_audit", f.read())


if __name__ == "__main__":
    unittest.main()
