import os
import unittest

from orders.logic import apply_discount, mark_shipped, summarize
from orders.parse import parse_orders

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "orders.csv")


class OrdersTest(unittest.TestCase):
    def test_parse(self):
        orders = parse_orders(DATA)
        self.assertEqual([o["id"] for o in orders], ["A-100", "A-101", "A-102", "A-103", "A-104"])
        self.assertEqual(orders[0]["total_cents"], 2 * 1999 + 10 * 125)
        self.assertEqual(orders[3]["status"], "new")

    def test_discount_copy(self):
        o = parse_orders(DATA)[0]
        d = apply_discount(o, 10)
        self.assertEqual(d["items"][0]["price_cents"], 1799)
        self.assertEqual(o["items"][0]["price_cents"], 1999)

    def test_ship(self):
        orders = parse_orders(DATA)
        self.assertEqual(mark_shipped(orders[0])["status"], "shipped")
        with self.assertRaises(ValueError):
            mark_shipped(orders[2])

    def test_summary(self):
        self.assertEqual(summarize(parse_orders(DATA)), {"cancelled": 1999, "new": 5248 + 2125 + 4550, "shipped": 4550})


if __name__ == "__main__":
    unittest.main()
