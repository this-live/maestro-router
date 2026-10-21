import unittest

from api.handlers import list_products


class HandlersTest(unittest.TestCase):
    def test_first_page_starts_at_first_product(self):
        r = list_products({})
        self.assertEqual(r["items"][0], "SKU-001")
        self.assertEqual(r["page"], 1)
        self.assertEqual(r["total"], 47)

    def test_invalid_page_means_one(self):
        self.assertEqual(list_products({"page": "abc"})["page"], 1)
        self.assertEqual(list_products({"page": "-3"})["page"], 1)


if __name__ == "__main__":
    unittest.main()
