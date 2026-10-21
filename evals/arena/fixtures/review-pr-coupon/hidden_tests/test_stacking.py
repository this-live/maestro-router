import unittest

from checkout.api import quote
from checkout.pricing import Coupon, apply_coupon, total

CART = [("lamp", 1, 10000)]
P10, P25 = Coupon("P10", "percent", 10), Coupon("P25", "percent", 25)
F5, F60, F200 = Coupon("F5", "fixed", 500), Coupon("F60", "fixed", 6000), Coupon("F200", "fixed", 20000)


class Stacking(unittest.TestCase):
    def test_only_fixed(self):
        self.assertEqual(total(CART, [F5]), 9500)
        self.assertEqual(quote(CART, ["FIVEOFF", "SHIPFREE"]), {"subtotal": 10000, "total": 8701})

    def test_percent_first_regardless_of_order(self):
        self.assertEqual(total(CART, [F5, P10]), 8500)
        self.assertEqual(total(CART, [P10, F5]), 8500)
        self.assertEqual(quote(CART, ["FIVEOFF", "SPRING25"]), {"subtotal": 10000, "total": 7000})

    def test_best_percent_only(self):
        self.assertEqual(total(CART, [P10, P25]), 7500)
        self.assertEqual(total(CART, [P25, P10, F5]), 7000)

    def test_never_negative(self):
        self.assertEqual(total(CART, [F200]), 0)
        self.assertEqual(total(CART, [Coupon("P50", "percent", 50), F60]), 0)
        self.assertEqual(apply_coupon(300, F5), 0)

    def test_unchanged(self):
        self.assertEqual(total(CART), 10000)
        self.assertEqual(total(CART, []), 10000)
        self.assertEqual(total([("pen", 3, 333)], [P10]), 900)
        with self.assertRaises(ValueError):
            total(CART, [Coupon("Z", "bogus", 1)])


if __name__ == "__main__":
    unittest.main()
