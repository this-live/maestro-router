"""A tiny fake HTTP server on 127.0.0.1 for provider tests (stdlib http.server in a thread). No real network."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128   # the default of 5 overflows on a burst; macOS then resets the connection (Linux drops the SYN and the client retries)


class FakeServer:
    """handler(req) -> (status, headers, body). req: {"method", "path", "headers" (lower-case), "body" (bytes), "json"}.
    body: bytes, str, dict/list (sent as JSON), or an iterable of bytes (sent chunk by chunk, connection closed at the end).
    Every request is kept in .requests."""

    def __init__(self, handler):
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"  # close at the end of the body: streaming needs no chunked encoding

            def log_message(self, *a):
                pass

            def handle_any(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                try:
                    js = json.loads(raw) if raw else None
                except ValueError:
                    js = None
                req = {"method": self.command, "path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                       "body": raw, "json": js}
                outer.requests.append(req)
                status, headers, body = handler(req)
                headers = dict(headers or {})
                if isinstance(body, (dict, list)):
                    headers.setdefault("Content-Type", "application/json")
                    body = json.dumps(body).encode()
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                if isinstance(body, (bytes, str)):
                    body = body.encode() if isinstance(body, str) else body
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.end_headers()
                    for chunk in body:
                        if isinstance(chunk, (int, float)):
                            time.sleep(chunk)
                            continue
                        self.wfile.write(chunk)
                        self.wfile.flush()

            def handle(self):
                try:
                    super().handle()
                except (BrokenPipeError, ConnectionResetError):  # the client gave up (a timeout test)
                    pass

            do_GET = do_POST = handle_any

        self.httpd = _Server(("127.0.0.1", 0), H)
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def json_reply(obj, status=200, headers=None):
    return status, dict({"Content-Type": "application/json"}, **(headers or {})), json.dumps(obj).encode()


def fake_cli(bindir, name, body):
    """Write an executable /bin/sh script `name` into bindir (a fake CLI on a temp PATH)."""
    import os
    path = os.path.join(bindir, name)
    with open(path, "w") as f:
        f.write("#!/bin/sh\n" + body + "\n")
    os.chmod(path, 0o755)
    return path


def fake_spec(url, style="openai", **kw):
    """A registry entry for a fake provider at `url`, key in $FAKE_KEY."""
    from maestro import providers as P
    npm = {"openai": "@ai-sdk/openai-compatible", "anthropic": "@ai-sdk/anthropic"}[style]
    return dict(P._spec("fake", {"name": "Fake", "env": ["FAKE_KEY"], "npm": npm, "api": url}), **kw)
