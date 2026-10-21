import unittest

from web.app import router


class RouterTest(unittest.TestCase):
    def test_static(self):
        r = router.dispatch("GET", "/health")
        self.assertEqual((r.status, r.body), (200, {"ok": True}))
        self.assertEqual(router.dispatch("get", "/users").body, {"users": [1, 7]})

    def test_404(self):
        self.assertEqual(router.dispatch("GET", "/nope").status, 404)


if __name__ == "__main__":
    unittest.main()
