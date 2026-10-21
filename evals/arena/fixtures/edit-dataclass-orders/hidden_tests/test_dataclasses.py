import dataclasses
import os
import unittest

from orders import logic
from orders.export import export_json
from orders.model import LineItem, Order
from orders.parse import parse_orders

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "orders.csv")


def golden(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return f.read()


class Model(unittest.TestCase):
    def test_dataclasses(self):
        self.assertTrue(dataclasses.is_dataclass(Order) and dataclasses.is_dataclass(LineItem))
        names = [f.name for f in dataclasses.fields(Order)]
        self.assertEqual(names, ["id", "customer", "items", "status"])
        self.assertEqual([f.name for f in dataclasses.fields(LineItem)], ["sku", "qty", "price_cents"])
        o = Order("X-1", "Bo", [LineItem("A", 2, 150), LineItem("B", 1, 5)])
        self.assertEqual(o.status, "new")
        self.assertEqual(o.total_cents, 305)
        with self.assertRaises(AttributeError):
            o.total_cents = 1

    def test_dict_round_trip(self):
        d = {"id": "X-1", "customer": "Bo", "items": [{"sku": "A", "qty": 2, "price_cents": 150}], "status": "new",
             "total_cents": 300}
        o = Order.from_dict(d)
        self.assertIsInstance(o.items[0], LineItem)
        self.assertEqual(o.to_dict(), d)
        d2 = dict(d, total_cents=999999)
        self.assertEqual(Order.from_dict(d2).total_cents, 300)


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.orders = parse_orders(DATA)

    def test_parse_returns_orders(self):
        self.assertTrue(all(isinstance(o, Order) for o in self.orders))
        self.assertTrue(all(isinstance(i, LineItem) for o in self.orders for i in o.items))
        self.assertEqual([o.id for o in self.orders], ["A-100", "A-101", "A-102", "A-103", "A-104"])
        self.assertEqual(self.orders[0].customer, "Ada Lovelace")
        self.assertEqual(self.orders[3].status, "new")

    def test_logic_no_mutation(self):
        o = self.orders[0]
        before = o.to_dict()
        d = logic.apply_discount(o, 10)
        s = logic.mark_shipped(o)
        self.assertIsInstance(d, Order)
        self.assertIsInstance(s, Order)
        self.assertEqual(o.to_dict(), before)
        self.assertEqual([i.price_cents for i in d.items], [1799, 112])
        self.assertEqual(d.total_cents, 2 * 1799 + 10 * 112)
        self.assertEqual(s.status, "shipped")
        self.assertIsNot(s.items, o.items)
        with self.assertRaises(ValueError):
            logic.mark_shipped(self.orders[2])

    def test_aggregates(self):
        self.assertEqual(logic.summarize(self.orders), {"cancelled": 1999, "new": 11923, "shipped": 4550})
        self.assertEqual(logic.top_customers(self.orders),
                         [("Ada Lovelace", 9798), ("Grace Hopper", 6675), ("Alan Turing", 1999)])
        self.assertEqual(logic.top_customers(self.orders, 1), [("Ada Lovelace", 9798)])

    def test_export_byte_identical(self):
        self.assertEqual(export_json(self.orders), golden("golden.json"))
        self.assertEqual(export_json([logic.apply_discount(o, 12.5) for o in self.orders]),
                         golden("golden_discount.json"))


if __name__ == "__main__":
    unittest.main()
