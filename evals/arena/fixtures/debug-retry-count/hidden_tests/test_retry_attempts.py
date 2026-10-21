import unittest
from unittest import mock

from netutil import client
from netutil.retry import retry


class Attempts(unittest.TestCase):
    def test_total_attempts_and_reraise(self):
        calls, sleeps = [], []

        @retry(times=3, delay=0.5, backoff=2.0, sleep=sleeps.append)
        def f():
            calls.append(1)
            raise ConnectionError("down %d" % len(calls))

        with self.assertRaises(ConnectionError) as cm:
            f()
        self.assertEqual(len(calls), 3)
        self.assertEqual(str(cm.exception), "down 3")
        self.assertEqual(sleeps, [0.5, 1.0])

    def test_succeeds_on_last_attempt(self):
        calls, sleeps = [], []

        @retry(times=3, delay=1, sleep=sleeps.append)
        def f():
            calls.append(1)
            if len(calls) < 3:
                raise TimeoutError()
            return 42

        self.assertEqual(f(), 42)
        self.assertEqual(sleeps, [1, 2.0])

    def test_single_attempt(self):
        sleeps = []

        @retry(times=1, sleep=sleeps.append)
        def f():
            raise KeyError("x")

        with self.assertRaises(KeyError):
            f()
        self.assertEqual(sleeps, [])

    def test_other_exceptions_not_retried(self):
        calls = []

        @retry(times=5, exceptions=(ConnectionError,), sleep=lambda s: None)
        def f():
            calls.append(1)
            raise ValueError("bad")

        with self.assertRaises(ValueError):
            f()
        self.assertEqual(len(calls), 1)

    def test_wraps_metadata(self):
        @retry()
        def documented():
            """doc"""

        self.assertEqual(documented.__name__, "documented")
        self.assertEqual(documented.__doc__, "doc")

    def test_fetch_config(self):
        bodies = iter([ConnectionError("a"), ConnectionError("b"), ConnectionError("c"), '{"port": 8080}'])

        def transport():
            b = next(bodies)
            if isinstance(b, Exception):
                raise b
            return b

        with mock.patch.object(client, "_sleep") as sl:
            self.assertEqual(client.fetch_config(transport), {"port": 8080})
        self.assertEqual([c.args[0] for c in sl.call_args_list], [0.2, 0.4, 0.8])

    def test_fetch_config_gives_up(self):
        def transport():
            raise ConnectionError("refused")

        with mock.patch.object(client, "_sleep"):
            with self.assertRaises(ConnectionError):
                client.fetch_config(transport)


if __name__ == "__main__":
    unittest.main()
