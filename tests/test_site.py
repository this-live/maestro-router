"""The website (site/): serve.py, links and the facts the pages repeat from README."""
from __future__ import annotations

import gzip
import html
import http.client
import importlib.util
import json
import os
import re
import threading
import unittest

import helpers  # noqa: F401  (puts the maestro package on sys.path)
from maestro import onboard

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SITE = os.path.join(ROOT, "site")
_spec = importlib.util.spec_from_file_location("maestro_site_serve", os.path.join(SITE, "serve.py"))
serve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(serve)
PAGES = sorted(f for f in os.listdir(SITE) if f.endswith(".html"))
INSTALL = ["npx skills add this-live/maestro-router", "claude plugin marketplace add this-live/maestro-router", "claude plugin install maestro@maestro",
           "pip install https://github.com/this-live/maestro-router/releases/download/v3.1.0/maestro_router-3.1.0-py3-none-any.whl",
           "pip install git+https://github.com/this-live/maestro-router@v3.1.0",
           'uvx --from git+https://github.com/this-live/maestro-router@v3.1.0 maestro-router route "fix the failing test"',
           "maestro-router setup", "maestro-router serve --open", "maestro-router hermes --install"]


def text_of(page):
    """The visible text of a page: tags removed, entities decoded."""
    with open(os.path.join(SITE, page), encoding="utf-8") as f:
        raw = f.read()
    return html.unescape(re.sub(r"<[^>]+>", "", raw)), raw


class SiteCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        serve.Handler.quiet = True
        cls.httpd = serve.make_server("127.0.0.1", 0)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def req(self, method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        data = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
        h = dict(headers or {})
        if data is not None:
            h.setdefault("Content-Type", "application/json")
        c.request(method, path, data, h)
        r = c.getresponse()
        raw = r.read()
        c.close()
        return r.status, {k.lower(): v for k, v in r.getheaders()}, raw

    def get_json(self, path):
        st, h, raw = self.req("GET", path)
        self.assertEqual(h["content-type"], "application/json")
        return st, json.loads(raw)


class Server(SiteCase):
    def test_health(self):
        st, h, raw = self.req("GET", "/health")
        self.assertEqual((st, raw, h["cache-control"]), (200, b"ok", "no-store"))

    def test_version_reports_the_deployed_commit(self):
        old = os.environ.get("RAILWAY_GIT_COMMIT_SHA")
        os.environ["RAILWAY_GIT_COMMIT_SHA"] = "abc123"
        try:
            self.assertEqual(self.get_json("/version"), (200, {"commit": "abc123"}))
        finally:
            if old is None:
                os.environ.pop("RAILWAY_GIT_COMMIT_SHA", None)
            else:
                os.environ["RAILWAY_GIT_COMMIT_SHA"] = old

    def test_pages_and_content_types(self):
        for path, ctype in (("/", "text/html; charset=utf-8"), ("/docs", "text/html; charset=utf-8"), ("/privacy", "text/html; charset=utf-8"),
                            ("/terms", "text/html; charset=utf-8"), ("/static/site.css", "text/css; charset=utf-8"),
                            ("/static/site.js", "text/javascript; charset=utf-8"), ("/static/favicon.svg", "image/svg+xml"),
                            ("/static/icon-64.png", "image/png"), ("/static/logo.jpg", "image/jpeg"), ("/static/manifest.json", "application/json")):
            st, h, raw = self.req("GET", path)
            self.assertEqual((st, h["content-type"]), (200, ctype), path)
            self.assertEqual(int(h["content-length"]), len(raw), path)
        self.assertEqual(self.req("GET", "/docs.html")[1].get("location"), "/docs")
        self.assertEqual(self.req("GET", "/index.html")[1].get("location"), "/")

    def test_security_headers(self):
        for path in ("/", "/static/site.css", "/nope", "/docs", "/health"):
            h = self.req("GET", path)[1]
            csp = h["content-security-policy"]
            self.assertIn("default-src 'none'", csp, path)
            self.assertIn("frame-ancestors 'none'", csp, path)
            self.assertNotIn("'unsafe-inline'", csp.split("script-src", 1)[1].split(";", 1)[0], path)
            self.assertNotRegex(csp, r"https?://", path)   # nothing loads from another origin
            self.assertEqual(h["x-content-type-options"], "nosniff")
            self.assertEqual(h["referrer-policy"], "no-referrer")
            self.assertEqual(h["x-frame-options"], "DENY")
            self.assertNotIn("set-cookie", h)
        self.assertNotIn("strict-transport-security", self.req("GET", "/")[1])
        self.assertIn("strict-transport-security", self.req("GET", "/", headers={"X-Forwarded-Proto": "https"})[1])

    def test_caching_etag_head_and_gzip(self):
        st, h, raw = self.req("GET", "/static/site.css")
        self.assertIn("max-age=86400", h["cache-control"])
        self.assertEqual(self.req("GET", "/static/site.css", headers={"If-None-Match": h["etag"]})[0], 304)
        self.assertEqual(self.req("GET", "/")[1]["cache-control"], "no-cache")
        st, hh, body = self.req("HEAD", "/")
        self.assertEqual((st, body), (200, b""))
        self.assertGreater(int(hh["content-length"]), 1000)
        st, hg, zipped = self.req("GET", "/", headers={"Accept-Encoding": "gzip"})
        self.assertEqual(hg["content-encoding"], "gzip")
        self.assertEqual(gzip.decompress(zipped), self.req("GET", "/")[2])
        self.assertNotIn("content-encoding", self.req("GET", "/static/icon-64.png", headers={"Accept-Encoding": "gzip"})[1])

    def test_traversal_hidden_files_and_sources_are_not_served(self):
        for path in ("/../README.md", "/static/../serve.py", "/%2e%2e/README.md", "/static/%2e%2e/serve.py", "/static/..%2fserve.py",
                     "/static/%2e%2e%2f%2e%2e%2fREADME.md", "/serve.py", "/Dockerfile", "/.dockerignore", "/static/.hidden",
                     "/static/site.css%00.png", "/..%5c..%5cREADME.md", "//etc/passwd", "/static/../../LICENSE"):
            st, h, raw = self.req("GET", path)
            self.assertEqual(st, 404, path)
            self.assertNotIn(b"Apache License\n                           Version 2.0", raw, path)
            self.assertNotIn(b"import ", raw, path)

    def test_404_page_and_no_directory_listing(self):
        st, h, raw = self.req("GET", "/no/such/page")
        self.assertEqual((st, h["content-type"]), (404, "text/html; charset=utf-8"))
        self.assertIn(b"404", raw)
        self.assertEqual(self.req("GET", "/static/")[0], 404)
        st, h, _ = self.req("GET", "/static")
        self.assertEqual((st, h["location"]), (301, "/static/"))

    def test_methods(self):
        for m in ("POST", "PUT", "DELETE", "PATCH"):
            for path in ("/", "/docs", "/api/route"):
                st, h, _ = self.req(m, path, body=b"")
                self.assertEqual((st, h.get("allow")), (405, "GET, HEAD"), (m, path))

    def test_there_is_no_demo(self):   # the site shows the real product only: no sample-data dashboard
        for path in ("/demo", "/demo/", "/demo/api/state"):
            self.assertEqual(self.req("GET", path)[0], 404, path)


class Pages(SiteCase):
    def test_every_internal_link_resolves(self):
        for page in PAGES:
            raw = text_of(page)[1]
            base = "/"
            ids = set(re.findall(r'\sid="([^"]+)"', raw))
            for ref in re.findall(r'(?:href|src)="([^"]+)"', raw):
                if ref.startswith(("https://", "mailto:", "data:")):
                    continue
                self.assertFalse(ref.startswith("http:"), (page, ref))
                if ref.startswith("#"):
                    self.assertIn(ref[1:], ids, (page, ref))
                    continue
                path, _, frag = ref.partition("#")
                path = path if path.startswith("/") else base + path
                st, h, body = self.req("GET", path)
                if st == 301:
                    st, h, body = self.req("GET", h["location"])
                self.assertEqual(st, 200, (page, ref))
                if frag:
                    self.assertIn('id="%s"' % frag, body.decode(), (page, ref))

    def test_no_external_scripts_styles_or_fonts(self):
        for page in PAGES:
            raw = text_of(page)[1]
            for src in re.findall(r"<script[^>]*\ssrc=\"([^\"]+)\"", raw):
                self.assertTrue(src.startswith("/static/"), (page, src))
            self.assertEqual(re.findall(r"<script>", raw), [], page)   # no inline scripts: the CSP allows none
            for href in re.findall(r"<link[^>]*\shref=\"([^\"]+)\"", raw):
                self.assertTrue(href.startswith("/static/"), (page, href))
            self.assertNotIn("@import", raw)
            self.assertNotRegex(raw, r"url\(\s*['\"]?https?:")
        with open(os.path.join(SITE, "static", "site.css"), encoding="utf-8") as f:
            css = f.read()
        self.assertNotRegex(css, r"url\(\s*['\"]?https?:|@import|@font-face")

    def test_install_commands_match_the_readme(self):
        with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
            readme = f.read()
        for cmd in INSTALL:
            self.assertIn(cmd, readme, cmd)
            for page in ("index.html", "docs.html"):
                self.assertIn(cmd, text_of(page)[0], (page, cmd))

    def test_name_clash_note_and_this_live_link(self):
        for page in ("index.html", "docs.html"):
            self.assertIn("maestro-router", text_of(page)[0], page)
        for page in PAGES:
            text, raw = text_of(page)
            self.assertIn('href="https://this.live/"', raw, page)
            self.assertNotIn("sample data", text.lower(), page)
            self.assertNotIn('href="/demo', raw, page)

    def test_docs_show_the_hermes_config_that_maestro_prints(self):
        text = text_of("docs.html")[0]
        self.assertIn(onboard.hermes_yaml("http://127.0.0.1:4545/v1", False).strip(), text)

    def test_pages_are_responsive_and_respect_reduced_motion(self):
        with open(os.path.join(SITE, "static", "site.css"), encoding="utf-8") as f:
            css = f.read()
        self.assertIn("prefers-reduced-motion: reduce", css)
        self.assertRegex(css, r"prefers-reduced-motion: reduce\)[^@]*#sparkle \{ display: none; \}")
        for page in PAGES:
            raw = text_of(page)[1]
            self.assertIn('name="viewport" content="width=device-width, initial-scale=1"', raw, page)
            self.assertIn('<html lang="en">', raw, page)
            self.assertEqual(raw.count("<h1"), 1, page)



if __name__ == "__main__":
    unittest.main()
