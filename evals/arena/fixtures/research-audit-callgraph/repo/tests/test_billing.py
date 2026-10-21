import unittest

from billing import audit
from billing.invoices import create_invoice, record_payment, void_invoice


class BillingTest(unittest.TestCase):
    def test_lifecycle(self):
        inv = create_invoice("acme", "12.50")
        self.assertEqual(inv["cents"], 1250)
        record_payment(inv, "12.50")
        self.assertEqual(inv["status"], "paid")
        void_invoice(inv, "alice")
        self.assertEqual(audit.recent(1)[0]["action"], "invoice.void")


if __name__ == "__main__":
    unittest.main()
