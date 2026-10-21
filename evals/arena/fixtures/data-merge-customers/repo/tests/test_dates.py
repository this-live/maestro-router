import unittest
from datetime import datetime

from etl.dates import parse_ts


class DatesTest(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(parse_ts("2024-03-01T10:00:00Z"), datetime(2024, 3, 1, 10, 0))
        self.assertEqual(parse_ts("2024-03-05 12:00"), datetime(2024, 3, 5, 12, 0))
        self.assertEqual(parse_ts("2024-01-15"), datetime(2024, 1, 15))
        self.assertEqual(parse_ts("10/03/2024 08:30"), datetime(2024, 3, 10, 8, 30))

    def test_bad(self):
        with self.assertRaises(ValueError):
            parse_ts("yesterday")


if __name__ == "__main__":
    unittest.main()
