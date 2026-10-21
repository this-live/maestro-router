import unittest

from svc.app import App


class AppTest(unittest.TestCase):
    def test_health(self):
        self.assertEqual(App().handle("GET", "/health").status, 200)

    def test_items(self):
        r = App().handle("GET", "/items", {"X-Api-Key": "key-alpha"})
        self.assertEqual((r.status, r.body["client"]), (200, "alpha corp"))

    def test_unauthorized(self):
        self.assertEqual(App().handle("GET", "/items").status, 401)

    def test_404(self):
        self.assertEqual(App().handle("GET", "/nope").status, 404)


if __name__ == "__main__":
    unittest.main()
