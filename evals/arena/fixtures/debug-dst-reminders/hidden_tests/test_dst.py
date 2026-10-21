import unittest
from datetime import datetime, timezone

from reminders.schedule import describe, next_occurrences
from reminders.tz import EASTERN, PACIFIC

UTC = timezone.utc


def z(*a):
    return datetime(*a, tzinfo=UTC)


class FallBack(unittest.TestCase):
    def test_nine_am_across_fall_back(self):
        got = next_occurrences("09:00", z(2024, 11, 1, 12, 0), 5)
        self.assertEqual(got, [z(2024, 11, 1, 13), z(2024, 11, 2, 13), z(2024, 11, 3, 14), z(2024, 11, 4, 14),
                               z(2024, 11, 5, 14)])
        self.assertEqual([describe(t) for t in got][2:],
                         ["2024-11-03 09:00 EST", "2024-11-04 09:00 EST", "2024-11-05 09:00 EST"])
        for t in got:
            self.assertEqual(t.utcoffset().total_seconds(), 0)

    def test_repeated_hour_fires_once(self):
        got = next_occurrences("01:30", z(2024, 11, 2, 0, 0), 3)
        self.assertEqual(got, [z(2024, 11, 2, 5, 30), z(2024, 11, 3, 5, 30), z(2024, 11, 4, 6, 30)])

    def test_inside_repeated_hour_no_second_fire(self):
        start = z(2024, 11, 3, 6, 15)  # 01:15 EST, the second pass through 01:xx
        self.assertEqual(next_occurrences("01:30", start, 2), [z(2024, 11, 4, 6, 30), z(2024, 11, 5, 6, 30)])
        start = z(2024, 11, 3, 5, 10)  # 01:10 EDT, first pass: today's 01:30 EDT still ahead
        self.assertEqual(next_occurrences("01:30", start, 2), [z(2024, 11, 3, 5, 30), z(2024, 11, 4, 6, 30)])


class SpringForward(unittest.TestCase):
    def test_nine_am_across_spring_forward(self):
        got = next_occurrences("09:00", z(2024, 3, 8, 15, 0), 4)
        self.assertEqual(got, [z(2024, 3, 9, 14), z(2024, 3, 10, 13), z(2024, 3, 11, 13), z(2024, 3, 12, 13)])

    def test_gap_time_shifts_forward(self):
        got = next_occurrences("02:30", z(2024, 3, 9, 0, 0), 3)
        self.assertEqual(got, [z(2024, 3, 9, 7, 30), z(2024, 3, 10, 7, 30), z(2024, 3, 11, 6, 30)])
        self.assertEqual(describe(got[1]), "2024-03-10 03:30 EDT")

    def test_long_run_keeps_wall_time(self):
        got = next_occurrences("07:45", z(2024, 1, 1, 0, 0), 400)
        self.assertEqual(len(got), 400)
        self.assertEqual({describe(t)[11:16] for t in got}, {"07:45"})
        days = [t.astimezone(EASTERN).date() for t in got]
        self.assertEqual(len(set(days)), 400)
        self.assertEqual(got, sorted(got))


class Other(unittest.TestCase):
    def test_strictly_after(self):
        got = next_occurrences("09:00", z(2024, 11, 2, 13, 0), 1)
        self.assertEqual(got, [z(2024, 11, 3, 14)])

    def test_weekdays_across_change(self):
        got = next_occurrences("09:00", z(2024, 11, 1, 14, 0), 3, weekdays={0, 1, 2, 3, 4})
        self.assertEqual(got, [z(2024, 11, 4, 14), z(2024, 11, 5, 14), z(2024, 11, 6, 14)])

    def test_other_zone(self):
        got = next_occurrences("18:00", z(2024, 11, 2, 12, 0), 2, tz=PACIFIC)
        self.assertEqual(got, [z(2024, 11, 3, 1), z(2024, 11, 4, 2)])

    def test_bad_time(self):
        with self.assertRaises(ValueError):
            next_occurrences("24:00", z(2024, 1, 1), 1)


if __name__ == "__main__":
    unittest.main()
