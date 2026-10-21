import unittest

from svc import config
from svc.app import App
from svc.ratelimit import TokenBucket


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


class Bucket(unittest.TestCase):
    def test_burst_then_refill(self):
        c = Clock()
        b = TokenBucket(5, 1.0, c)
        self.assertEqual([b.allow() for _ in range(6)], [True] * 5 + [False])
        c.t += 0.5
        self.assertFalse(b.allow())
        self.assertAlmostEqual(b.retry_after(), 0.5, places=6)
        c.t += 0.5
        self.assertEqual(b.retry_after(), 0)
        self.assertTrue(b.allow())
        self.assertFalse(b.allow())

    def test_capacity_cap(self):
        c = Clock()
        b = TokenBucket(3, 2.0, c)
        for _ in range(3):
            b.allow()
        c.t += 100
        self.assertEqual(sum(b.allow() for _ in range(10)), 3)

    def test_denied_does_not_consume(self):
        c = Clock()
        b = TokenBucket(1, 0.25, c)
        self.assertTrue(b.allow())
        for _ in range(5):
            self.assertFalse(b.allow())
        self.assertAlmostEqual(b.retry_after(), 4.0, places=6)
        c.t += 4
        self.assertTrue(b.allow())


class AppLimits(unittest.TestCase):
    def get(self, app, key):
        return app.handle("GET", "/items", {"X-Api-Key": key} if key else {})

    def test_429(self):
        c = Clock()
        app = App(clock=c)
        statuses = [self.get(app, "key-alpha").status for _ in range(config.RATE_LIMIT_BURST + 1)]
        self.assertEqual(statuses, [200] * config.RATE_LIMIT_BURST + [429])
        r = self.get(app, "key-alpha")
        self.assertEqual(r.body, {"error": "rate_limited"})
        self.assertEqual(r.headers.get("Retry-After"), "1")
        c.t += 1
        self.assertEqual(self.get(app, "key-alpha").status, 200)

    def test_retry_after_rounds_up(self):
        c = Clock()
        app = App(clock=c)
        for _ in range(config.RATE_LIMIT_BURST):
            self.get(app, "key-beta")
        c.t += 0.1
        self.assertEqual(self.get(app, "key-beta").headers["Retry-After"], "1")

    def test_keys_independent(self):
        app = App(clock=Clock())
        for _ in range(config.RATE_LIMIT_BURST):
            self.get(app, "key-alpha")
        self.assertEqual(self.get(app, "key-alpha").status, 429)
        self.assertEqual(self.get(app, "key-beta").status, 200)

    def test_health_and_401_exempt(self):
        app = App(clock=Clock())
        for _ in range(20):
            self.assertEqual(app.handle("GET", "/health").status, 200)
            self.assertEqual(self.get(app, None).status, 401)
            self.assertEqual(self.get(app, "key-nope").status, 401)
        self.assertEqual(self.get(app, "key-alpha").status, 200)

    def test_uses_config(self):
        c = Clock()
        old = (config.RATE_LIMIT_BURST, config.RATE_LIMIT_PER_SEC)
        config.RATE_LIMIT_BURST, config.RATE_LIMIT_PER_SEC = 2, 0.1
        try:
            app = App(clock=c)
            self.assertEqual([self.get(app, "key-alpha").status for _ in range(3)], [200, 200, 429])
            self.assertEqual(self.get(app, "key-alpha").headers["Retry-After"], "10")
            c.t += 9.5
            r = self.get(app, "key-alpha")
            self.assertEqual((r.status, r.headers["Retry-After"]), (429, "1"))
        finally:
            config.RATE_LIMIT_BURST, config.RATE_LIMIT_PER_SEC = old


if __name__ == "__main__":
    unittest.main()
