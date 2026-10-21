import unittest

from invoicing import Invoice, split_payment


class InvoiceTest(unittest.TestCase):
    def test_simple(self):
        inv = Invoice(tax_rate="0.10")
        inv.add_line("Coffee", 2, "3.50")
        self.assertEqual(inv.line_totals(), ["7.00"])
        self.assertEqual(inv.totals(), {"subtotal": "7.00", "discount": "0.00", "tax": "0.70", "total": "7.70"})

    def test_even_split(self):
        self.assertEqual(split_payment("10.00", 4), ["2.50"] * 4)


if __name__ == "__main__":
    unittest.main()
