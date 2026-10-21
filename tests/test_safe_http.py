"""A provider key never follows a redirect to another origin (scheme, host or port)."""
import unittest

import helpers  # noqa: F401
from fake_http import FakeServer, json_reply

from maestro import providers as P
from maestro import router as core

KEY = "sk-test-NEVER-LEAK-0000"   # gitleaks:allow (fake test key)
CRED = {"Authorization": "Bearer " + KEY, "x-api-key": KEY, "api-key": KEY}


class Redirects(unittest.TestCase):
    def setUp(self):
        self.other = FakeServer(lambda r: json_reply({"ok": True}))
        self.addCleanup(self.other.close)

    def bounce(self, target):
        srv = FakeServer(lambda r: (302, {"Location": target + "/landed"}, b"") if r["path"] == "/v1/models" else json_reply({"ok": 1}))
        self.addCleanup(srv.close)
        return srv

    def assert_no_key(self, server):
        self.assertTrue(server.requests, "the redirect was not followed")
        for r in server.requests:
            for name in ("authorization", "x-api-key", "api-key"):
                self.assertNotIn(name, r["headers"])
            self.assertNotIn(KEY.encode(), r["body"])

    def test_providers_http_request_drops_credentials_across_origins(self):
        src = self.bounce(self.other.url)
        status, _, _ = P.http_request("GET", src.url + "/v1/models", dict(CRED), direct=True)
        self.assertEqual(status, 200)
        self.assert_no_key(self.other)

    def test_router_fetch_drops_credentials_across_origins(self):
        src = self.bounce(self.other.url)
        status, _ = core._http_fetch(src.url + "/v1/models", dict(CRED), None, 5, direct=True)
        self.assertEqual(status, 200)
        self.assert_no_key(self.other)

    def test_same_origin_redirect_keeps_the_key(self):
        srv = FakeServer(lambda r: (301, {"Location": "/v1/models/"}, b"") if r["path"] == "/v1/models" else json_reply({"ok": 1}))
        self.addCleanup(srv.close)
        self.assertEqual(P.http_request("GET", srv.url + "/v1/models", dict(CRED), direct=True)[0], 200)
        self.assertEqual(srv.requests[-1]["headers"].get("authorization"), "Bearer " + KEY)


if __name__ == "__main__":
    unittest.main()
