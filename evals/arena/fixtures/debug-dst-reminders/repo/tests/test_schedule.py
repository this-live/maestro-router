import unittest
from datetime import datetime

from reminders.schedule import describe, next_occurrences
from reminders.tz import UTC


class ScheduleTest(unittest.TestCase):
    def test_summer(self):
        start = datetime(2024, 7, 1, 12, 0, tzinfo=UTC)
        got = next_occurrences("09:00", start, 2)
        self.assertEqual(got, [datetime(2024, 7, 1, 13, 0, tzinfo=UTC), datetime(2024, 7, 2, 13, 0, tzinfo=UTC)])

    def test_describe(self):
        self.assertEqual(describe(datetime(2024, 1, 15, 14, 0, tzinfo=UTC)), "2024-01-15 09:00 EST")
        self.assertEqual(describe(datetime(2024, 7, 15, 13, 0, tzinfo=UTC)), "2024-07-15 09:00 EDT")

    def test_naive_rejected(self):
        with self.assertRaises(ValueError):
            next_occurrences("09:00", datetime(2024, 1, 1), 1)


if __name__ == "__main__":
    unittest.main()
