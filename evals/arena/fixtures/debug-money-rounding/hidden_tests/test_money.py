import unittest
from decimal import Decimal

from invoicing import Invoice, split_payment


class LineRounding(unittest.TestCase):
    def test_half_up_lines(self):
        inv = Invoice(tax_rate="0.0825")
        inv.add_line("Widget", 1, "2.675")
        inv.add_line("Bolt", 7, "0.145")
        inv.add_line("Nut", 3, "1.115")
        self.assertEqual(inv.line_totals(), ["2.68", "1.02", "3.35"])
        self.assertEqual(inv.totals(), {"subtotal": "7.05", "discount": "0.00", "tax": "0.58", "total": "7.63"})

    def test_single_cent_edge(self):
        inv = Invoice()
        inv.add_line("Clip", 1, "1.005")
        self.assertEqual(inv.line_totals(), ["1.01"])
        self.assertEqual(inv.totals()["total"], "1.01")

    def test_discount_then_tax(self):
        inv = Invoice(tax_rate="0.10", discount_pct="12.5")
        inv.add_line("A", 1, "10.10")
        self.assertEqual(inv.totals(), {"subtotal": "10.10", "discount": "1.26", "tax": "0.88", "total": "9.72"})

    def test_tax_half_up(self):
        inv = Invoice(tax_rate="0.15")
        inv.add_line("Gum", 1, "0.30")
        self.assertEqual(inv.totals()["tax"], "0.05")
        self.assertEqual(inv.totals()["total"], "0.35")

    def test_many_small_lines_add_up(self):
        inv = Invoice()
        for _ in range(10):
            inv.add_line("Penny candy", 1, "0.10")
        self.assertEqual(inv.totals()["subtotal"], "1.00")
        inv2 = Invoice(tax_rate="0.07")
        for _ in range(3):
            inv2.add_line("Item", 1, "0.1")
        t = inv2.totals()
        self.assertEqual(t, {"subtotal": "0.30", "discount": "0.00", "tax": "0.02", "total": "0.32"})

    def test_decimal_inputs_ok(self):
        inv = Invoice(tax_rate=Decimal("0.0825"))
        inv.add_line("Widget", "2", Decimal("2.675"))
        self.assertEqual(inv.line_totals(), ["5.35"])


class Split(unittest.TestCase):
    def test_split_100_by_3(self):
        self.assertEqual(split_payment("100.00", 3), ["33.34", "33.33", "33.33"])

    def test_split_small(self):
        self.assertEqual(split_payment("0.05", 3), ["0.02", "0.02", "0.01"])
        self.assertEqual(split_payment("7.63", 1), ["7.63"])

    def test_split_sums(self):
        for total in ("100.00", "99.99", "0.01", "1234.57", "7.63"):
            for n in range(1, 13):
                parts = split_payment(total, n)
                self.assertEqual(len(parts), n)
                self.assertEqual(sum(Decimal(p) for p in parts), Decimal(total), (total, n))
                self.assertTrue(all(len(p.split(".")[1]) == 2 for p in parts))
                cents = [int(Decimal(p) * 100) for p in parts]
                self.assertLessEqual(max(cents) - min(cents), 1)
                self.assertEqual(cents, sorted(cents, reverse=True))

    def test_bad_n(self):
        with self.assertRaises(ValueError):
            split_payment("1.00", 0)


if __name__ == "__main__":
    unittest.main()
