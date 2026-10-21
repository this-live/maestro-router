import unittest

from web.app import router
from web.response import Response
from web.router import Router


def call(method, path):
    r = router.dispatch(method, path)
    return r.status, r.body


class Params(unittest.TestCase):
    def test_int_param(self):
        self.assertEqual(call("GET", "/users/7"), (200, {"id": 7, "name": "grace"}))
        self.assertEqual(call("GET", "/users/99"), (404, {"error": "no such user"}))
        self.assertEqual(call("PUT", "/users/1"), (200, {"updated": 1}))
        self.assertEqual(router.dispatch("DELETE", "/users/1").status, 204)

    def test_int_rejects_non_digits(self):
        for p in ("/users/abc", "/users/-3", "/users/1.5", "/users/"):
            self.assertEqual(router.dispatch("GET", p).status, 404, p)

    def test_str_param_one_segment(self):
        self.assertEqual(call("GET", "/files/report.pdf"), (200, {"file": "report.pdf"}))
        self.assertEqual(router.dispatch("GET", "/files/a/b").status, 404)

    def test_multiple_params(self):
        self.assertEqual(call("GET", "/teams/infra/members/42"), (200, {"team": "infra", "member": 42}))

    def test_static_wins(self):
        self.assertEqual(call("GET", "/users/me"), (200, {"id": 1, "name": "ada", "me": True}))
        self.assertEqual(call("GET", "/files/index"), (200, {"files": ["a.txt", "b.txt"]}))

    def test_trailing_slash(self):
        self.assertEqual(router.dispatch("GET", "/users/7/").status, 404)
        self.assertEqual(router.dispatch("GET", "/health/").status, 404)

    def test_405(self):
        r = router.dispatch("POST", "/users/7")
        self.assertEqual(r.status, 405)
        self.assertEqual(r.headers.get("Allow"), "DELETE, GET, PUT")
        r = router.dispatch("DELETE", "/health")
        self.assertEqual((r.status, r.headers.get("Allow")), (405, "GET"))
        r = router.dispatch("POST", "/users/me")  # "me" is not an int: only the static GET route matches
        self.assertEqual((r.status, r.headers.get("Allow")), (405, "GET"))

    def test_404_vs_405(self):
        self.assertEqual(router.dispatch("POST", "/nothing/here").status, 404)

    def test_unknown_converter(self):
        r = Router()
        with self.assertRaises(ValueError):
            r.route("GET", "/x/<float:v>")(lambda v: v)

    def test_fresh_router_order_independent(self):
        r = Router()
        r.route("GET", "/a/<name>")(lambda name: {"param": name})
        r.route("GET", "/a/b")(lambda: {"static": True})
        self.assertEqual(r.dispatch("GET", "/a/b").body, {"static": True})
        self.assertEqual(r.dispatch("GET", "/a/c").body, {"param": "c"})
        r.route("GET", "/resp")(lambda: Response(201, "made"))
        self.assertEqual(r.dispatch("GET", "/resp").status, 201)


if __name__ == "__main__":
    unittest.main()
