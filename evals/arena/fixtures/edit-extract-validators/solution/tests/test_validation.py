import unittest

from shop.api import create_order
from shop.validators import ValidationError, validate_email, validate_quantity, validate_sku


class ValidationTest(unittest.TestCase):
    def test_email(self):
        self.assertEqual(validate_email(" Bob@Example.COM "), "bob@example.com")
        with self.assertRaises(ValidationError):
            validate_email("bob@localhost")

    def test_sku_qty(self):
        self.assertEqual(validate_sku("abc-1234"), "ABC-1234")
        self.assertEqual(validate_quantity("3"), 3)
        with self.assertRaises(ValidationError):
            validate_quantity(1000)

    def test_api(self):
        status, body = create_order({"name": "Bo", "email": "bo@x.io", "lines": [{"sku": "abc-0001", "qty": 2},
                                                                                {"sku": "ABC-0001", "qty": "3"}]})
        self.assertEqual((status, body), (201, {"customer": "bo@x.io", "lines": [{"sku": "ABC-0001", "qty": 5}]}))
        self.assertEqual(create_order({"email": "nope"})[1]["field"], "email")


if __name__ == "__main__":
    unittest.main()
