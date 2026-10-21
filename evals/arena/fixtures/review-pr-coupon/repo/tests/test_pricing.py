import unittest

from checkout.api import quote
from checkout.pricing import Coupon, total

CART = [("mug", 2, 1250), ("tee", 1, 2500)]


class PricingTest(unittest.TestCase):
    def test_no_coupon(self):
        self.assertEqual(total(CART), 5000)

    def test_percent(self):
        self.assertEqual(total(CART, [Coupon("X", "percent", 10)]), 4500)

    def test_stacking(self):
        self.assertEqual(quote(CART, ["welcome10", "SPRING25", "FIVEOFF"]), {"subtotal": 5000, "total": 3250})

    def test_unknown(self):
        self.assertEqual(quote(CART, ["NOPE"]), {"error": "unknown coupon NOPE"})


if __name__ == "__main__":
    unittest.main()
