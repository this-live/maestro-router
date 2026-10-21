import unittest

from netutil.retry import retry


class RetryTest(unittest.TestCase):
    def test_success_first_try(self):
        calls = []

        @retry(times=3, sleep=lambda s: None)
        def f():
            calls.append(1)
            return "ok"

        self.assertEqual(f(), "ok")
        self.assertEqual(len(calls), 1)

    def test_bad_times(self):
        with self.assertRaises(ValueError):
            retry(times=0)


if __name__ == "__main__":
    unittest.main()
