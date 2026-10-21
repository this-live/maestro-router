import unittest

from api.handlers import list_products
from api.pagination import paginate
from api.store import PRODUCTS


class Paginate(unittest.TestCase):
    def test_pages_of_25(self):
        items = list(range(25))
        p1, p2, p3 = (paginate(items, n, 10) for n in (1, 2, 3))
        self.assertEqual(p1["items"], list(range(10)))
        self.assertEqual(p2["items"], list(range(10, 20)))
        self.assertEqual(p3["items"], list(range(20, 25)))
        self.assertEqual([p["total_pages"] for p in (p1, p2, p3)], [3, 3, 3])
        self.assertEqual([p["has_next"] for p in (p1, p2, p3)], [True, True, False])
        self.assertEqual([p["next_page"] for p in (p1, p2, p3)], [2, 3, None])

    def test_every_item_exactly_once(self):
        for n in range(0, 24):
            for per in range(1, 8):
                items = ["i%d" % k for k in range(n)]
                seen, page = [], 1
                while True:
                    r = paginate(items, page, per)
                    self.assertLessEqual(len(r["items"]), per)
                    seen.extend(r["items"])
                    if not r["has_next"]:
                        break
                    page = r["next_page"]
                self.assertEqual(seen, items, (n, per))
                self.assertEqual(r["total_pages"], max(1, -(-n // per)), (n, per))

    def test_exact_multiple(self):
        r = paginate(list(range(20)), 2, 10)
        self.assertEqual(r["items"], list(range(10, 20)))
        self.assertEqual(r["total_pages"], 2)
        self.assertFalse(r["has_next"])

    def test_empty_and_past_end(self):
        r = paginate([], 1, 10)
        self.assertEqual((r["items"], r["total_pages"], r["has_next"], r["next_page"]), ([], 1, False, None))
        r = paginate(list(range(5)), 4, 2)
        self.assertEqual((r["items"], r["has_next"], r["next_page"]), ([], False, None))


class Handler(unittest.TestCase):
    def test_walk_all_products(self):
        seen, params = [], {"per_page": "20"}
        pages = 0
        while True:
            r = list_products(params)
            pages += 1
            seen.extend(r["items"])
            if not r["has_next"]:
                break
            params = {"per_page": "20", "page": str(r["next_page"])}
        self.assertEqual(pages, 3)
        self.assertEqual(seen, [p["sku"] for p in PRODUCTS])
        self.assertEqual(seen[-1], "SKU-047")

    def test_per_page_clamped(self):
        r = list_products({"per_page": "0"})
        self.assertEqual(r["per_page"], 1)
        self.assertEqual(r["items"], ["SKU-001"])
        self.assertEqual(r["total_pages"], 47)
        self.assertEqual(list_products({"per_page": "-5"})["per_page"], 1)
        self.assertEqual(list_products({"per_page": "500"})["per_page"], 100)
        self.assertEqual(list_products({"per_page": "x"})["per_page"], 20)

    def test_category(self):
        r = list_products({"category": "garden", "per_page": "10", "page": "2"})
        garden = [p["sku"] for p in PRODUCTS if p["category"] == "garden"]
        self.assertEqual(r["total"], len(garden))
        self.assertEqual(r["items"], garden[10:20])
        self.assertEqual(r["total_pages"], 2)


if __name__ == "__main__":
    unittest.main()
