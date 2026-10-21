#!/usr/bin/env python3
"""The Maestro website: a small static server (stdlib only) for site/.

    python3 site/serve.py            # http://127.0.0.1:8080 (PORT and HOST override; Railway sets PORT)

- GET /health answers "ok"; GET /version answers {"commit": <deployed git sha or "unknown">}.
- Pages: / (index.html), /docs, /privacy, /terms; files under site/ with known types only; no directory listings.
- Security headers on every response; long-lived caching for /static/; gzip when the client accepts it.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import posixpath
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SITE = os.path.dirname(os.path.realpath(__file__))

TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".json": "application/json", ".webmanifest": "application/manifest+json", ".svg": "image/svg+xml", ".png": "image/png",
         ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".ico": "image/x-icon", ".txt": "text/plain; charset=utf-8",
         ".xml": "application/xml"}
COMPRESSIBLE = (".html", ".css", ".js", ".json", ".webmanifest", ".svg", ".txt", ".xml")
PAGES = {"/": "index.html", "/docs": "docs.html", "/privacy": "privacy.html", "/terms": "terms.html"}
ALIASES = {"/" + f: p for p, f in PAGES.items()}   # /docs.html -> /docs, /index.html -> /
CSP = ("default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; "
       "connect-src 'self'; manifest-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
SECURITY = {"X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY",
            "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Resource-Policy": "same-origin",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()"}

_cache = {}
_lock = threading.Lock()


def _read(path):
    """File bytes, cached by (mtime, size)."""
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)
    with _lock:
        hit = _cache.get(key)
    if hit is None:
        with open(path, "rb") as f:
            hit = f.read()
        with _lock:
            if len(_cache) > 256:
                _cache.clear()
            _cache[key] = hit
    return hit


def resolve(url_path):
    """The file under SITE for a URL path, or None. Refuses traversal, hidden files, NUL bytes and unknown types."""
    path = urllib.parse.unquote(url_path)
    if not path.startswith("/") or "\x00" in path or "\\" in path:
        return None
    parts = path.split("/")
    if any(p in ("..", ".") or p.startswith(".") for p in parts if p):
        return None
    full = os.path.realpath(os.path.join(SITE, posixpath.normpath(path).lstrip("/")))
    if not full.startswith(SITE + os.sep):
        return None
    if os.path.isdir(full):
        return full   # the caller redirects to the trailing-slash form or serves index.html
    if not os.path.isfile(full) or os.path.splitext(full)[1].lower() not in TYPES:
        return None
    return full


class Handler(BaseHTTPRequestHandler):
    server_version = "maestro-site"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    quiet = False   # tests turn the access log off

    def log_message(self, fmt, *args):   # method, path and status only: no IP addresses, no user agents
        pass

    def log_request(self, code="-", size="-"):
        if not self.quiet:
            sys.stderr.write("%s %s %s\n" % (self.command, self.path.split("?", 1)[0][:200], code))

    # ── keep-alive hygiene: this server never reads a request body, so one that arrives (a refused POST, a GET with a body) would be
    # parsed as the next request on the same connection. Answer it and close.
    sent_connection = False

    def parse_request(self):
        self.sent_connection = True   # no bookkeeping while the headers are parsed (a malformed request answers with its own close)
        ok = super().parse_request()
        self.sent_connection = False
        return ok

    def carries_body(self):
        h = getattr(self, "headers", None)
        return h is not None and (h.get("Transfer-Encoding") is not None or (h.get("Content-Length") or "0").strip() not in ("", "0"))

    def send_header(self, keyword, value):
        if keyword.lower() == "connection":
            self.sent_connection = True
        super().send_header(keyword, value)

    def end_headers(self):
        if self.carries_body() and not self.sent_connection:
            self.send_header("Connection", "close")   # also sets self.close_connection
        super().end_headers()

    # ── responses ──
    def send(self, status, body=b"", ctype="text/plain; charset=utf-8", cache="no-cache", headers=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        headers = dict(headers or {})
        if body and len(body) > 1024 and any(ctype.startswith(t) for t in ("text/", "application/json", "application/manifest", "image/svg", "application/xml")):
            headers["Vary"] = "Accept-Encoding"
            if "gzip" in (self.headers.get("Accept-Encoding") or "").lower():
                body = gzip.compress(body, 6, mtime=0)
                headers["Content-Encoding"] = "gzip"
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("Content-Security-Policy", CSP)
        for k, v in SECURITY.items():
            self.send_header(k, v)
        if (self.headers.get("X-Forwarded-Proto") or "").lower() == "https":
            self.send_header("Strict-Transport-Security", "max-age=31536000")
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_json(self, status, obj):
        self.send(status, json.dumps(obj, separators=(",", ":")), "application/json", "no-store")

    def not_found(self):
        try:
            page = _read(os.path.join(SITE, "404.html"))
        except OSError:
            page = b"Not found"
        self.send(404, page, "text/html; charset=utf-8", "no-cache")

    def redirect(self, location):
        self.send(301, "", headers={"Location": location})

    def send_file(self, full, cache):
        ext = os.path.splitext(full)[1].lower()
        body = _read(full)
        etag = '"%s"' % hashlib.sha1(body).hexdigest()[:20]
        if etag in (self.headers.get("If-None-Match") or ""):
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", cache)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send(200, body, TYPES[ext], cache, {"ETag": etag})

    # ── routing ──
    def do_GET(self):
        url = urllib.parse.urlsplit(self.path)
        path = url.path or "/"
        if path == "/health":
            return self.send(200, "ok", cache="no-store")
        if path == "/version":   # the deployed commit (Railway sets RAILWAY_GIT_COMMIT_SHA); used by the post-deploy smoke test
            return self.send_json(200, {"commit": os.environ.get("RAILWAY_GIT_COMMIT_SHA") or os.environ.get("MAESTRO_SITE_COMMIT") or "unknown"})
        if path in PAGES:
            return self.send_file(os.path.join(SITE, PAGES[path]), "no-cache")
        if path in ALIASES:
            return self.redirect(ALIASES[path])
        full = resolve(path)
        if full is None:
            return self.not_found()
        if os.path.isdir(full):
            if not path.endswith("/"):
                return self.redirect(path + "/")
            return self.not_found()   # no directory listings
        cache = "public, max-age=86400" if path.startswith("/static/") else "no-cache"
        return self.send_file(full, cache)

    do_HEAD = do_GET

    def refuse(self):
        self.close_connection = True   # whatever body came with the refused request stays unread
        self.send(405, "Method not allowed", headers={"Allow": "GET, HEAD", "Connection": "close"})

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_TRACE = do_CONNECT = refuse

class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128


def make_server(host="127.0.0.1", port=8080):
    return Server((host, port), Handler)


def main():
    host = os.environ.get("HOST") or ("0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
    port = int(os.environ.get("PORT") or 8080)
    httpd = make_server(host, port)
    print("Maestro site on http://%s:%d" % (host, httpd.server_address[1]), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
