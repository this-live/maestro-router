"""Request bodies on both servers: a bad Content-Length or chunk size is refused before it is read, the size limit holds for chunked
bodies, and a response sent without reading the body closes the connection (the unread bytes must never become the next request)."""
import socket
import unittest
from unittest import mock

import helpers  # noqa: F401
from test_server import E, ServerCase
from test_site import SiteCase

from maestro import server as S

CHAT = "/v1/chat/completions"
PIPED = b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"   # a request hiding in the body of another


def head(method, path, port, *extra):
    return ("%s %s HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nContent-Type: application/json\r\n%s\r\n" % (
        method, path, port, "".join(h + "\r\n" for h in extra))).encode("latin-1")


def read_response(sock):
    """(status, lower-cased headers, body) of the next response on an open socket; waits only as long as the socket timeout."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            raise AssertionError("connection closed before a full response: %r" % buf[:200])
        buf += chunk
    raw, _, rest = buf.partition(b"\r\n\r\n")
    lines = raw.decode("latin-1").split("\r\n")
    hdrs = {}
    for l in lines[1:]:
        k, _, v = l.partition(":")
        hdrs[k.strip().lower()] = v.strip()
    n = int(hdrs.get("content-length", "0"))
    while len(rest) < n:
        chunk = sock.recv(65536)
        if not chunk:
            break
        rest += chunk
    return int(lines[0].split()[1]), hdrs, rest[:n]


def read_all(sock):
    """Everything the server sends until it closes the connection; (bytes, closed). A server that keeps the connection open times out."""
    buf = b""
    try:
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                return buf, True
            buf += chunk
    except socket.timeout:
        return buf, False


class Wire:
    def connect(self, port, timeout=5):
        s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self.addCleanup(s.close)
        return s

    def only_one_response(self, port, payload):
        """Send `payload` in one write; the server must answer exactly once and close the connection."""
        s = self.connect(port)
        s.sendall(payload)
        data, closed = read_all(s)
        self.assertTrue(closed, "the server kept the connection open: %r" % data[:200])
        self.assertEqual(data.count(b"HTTP/1."), 1, data[:600])
        self.assertIn(b"onnection: close", data)
        return int(data.split(b" ", 2)[1])


class Bodies(Wire, ServerCase):
    def setUp(self):
        super().setUp()
        self.start([E("openai/gpt-x")])

    def refused(self, payload, status, hold_open=True):
        s = self.connect(self.port)
        s.sendall(payload)   # no shutdown: the server must answer without waiting for the end of the stream
        got, hdrs, body = read_response(s)
        self.assertEqual(got, status, body)
        self.assertEqual(hdrs.get("connection"), "close")
        data, closed = read_all(s)
        self.assertTrue(closed)
        return body

    def test_negative_or_malformed_content_length_is_a_400(self):
        for cl in ("-1", "-5", "abc", "1e3", "0x10", "+5", "5, 5", "٣"):
            try:
                payload = head("POST", CHAT, self.port, "Content-Length: " + cl) + b"x" * 5
            except UnicodeEncodeError:
                continue
            self.refused(payload, 400)

    def test_oversized_content_length_is_a_413_without_reading_the_body(self):
        self.refused(head("POST", CHAT, self.port, "Content-Length: %d" % (S.MAX_BODY + 1)), 413)
        self.refused(head("POST", CHAT, self.port, "Content-Length: " + "9" * 40), 413)

    def test_two_content_lengths_or_both_framings_are_a_400(self):
        self.refused(head("POST", CHAT, self.port, "Content-Length: 2", "Content-Length: 2") + b"{}", 400)
        self.refused(head("POST", CHAT, self.port, "Transfer-Encoding: chunked", "Content-Length: 2") + b"2\r\n{}\r\n0\r\n\r\n", 400)

    def test_a_negative_chunk_size_is_refused_without_reading_to_eof(self):
        self.refused(head("POST", CHAT, self.port, "Transfer-Encoding: chunked") + b"-1\r\n" + b"x" * 100, 400)
        self.refused(head("POST", CHAT, self.port, "Transfer-Encoding: chunked") + b"-1\r\n", 400)   # and nothing more is coming

    def test_unparsable_or_endless_chunk_size_lines_are_a_400(self):
        for line in (b"zz\r\n", b"0x10\r\n", b"+5\r\n", b"1_0\r\n", b"\r\n", b"A" * 3000):
            self.refused(head("POST", CHAT, self.port, "Transfer-Encoding: chunked") + line, 400)

    def test_a_chunk_larger_than_the_limit_is_a_413_before_it_is_read(self):
        self.refused(head("POST", CHAT, self.port, "Transfer-Encoding: chunked") + b"FFFFFFFF\r\n", 413)
        self.refused(head("POST", CHAT, self.port, "Transfer-Encoding: chunked") + b"FFFFFFFFFFFFFFFFFFFF\r\n", 413)

    def test_chunks_that_add_up_past_the_limit_are_a_413(self):
        with mock.patch.object(S, "MAX_BODY", 100):
            payload = head("POST", CHAT, self.port, "Transfer-Encoding: chunked") + b"3c\r\n" + b"a" * 60 + b"\r\n" + b"3c\r\n"
            self.refused(payload, 413)   # the second size line alone is enough; its 60 bytes are never awaited

    def test_a_truncated_body_is_a_400(self):
        s = self.connect(self.port)
        s.sendall(head("POST", CHAT, self.port, "Content-Length: 50") + b"{}")
        s.shutdown(socket.SHUT_WR)
        st, hdrs, body = read_response(s)
        self.assertEqual(st, 400)

    def test_a_chunked_body_with_trailers_is_read_and_the_connection_stays_open(self):
        s = self.connect(self.port)
        body = b'{"model": "maestro/auto", "messages": []}'
        s.sendall(head("POST", CHAT, self.port, "Transfer-Encoding: chunked")
                  + b"%x;ext=1\r\n" % 10 + body[:10] + b"\r\n" + b"%x\r\n" % (len(body) - 10) + body[10:] + b"\r\n"
                  + b"0\r\nX-Trailer: 1\r\n\r\n")
        st, hdrs, raw = read_response(s)
        self.assertEqual(st, 400)
        self.assertIn(b"messages", raw)                     # it reached the JSON checks: the framing was parsed correctly
        self.assertNotEqual(hdrs.get("connection"), "close")   # and the body was consumed, so the connection is reusable
        s.sendall(PIPED)
        self.assertEqual(read_response(s)[0], 200)

    def test_expect_100_continue_still_works_and_keeps_the_connection(self):
        s = self.connect(self.port)
        s.sendall(head("POST", CHAT, self.port, "Content-Length: 2", "Expect: 100-continue"))
        interim = s.recv(1024)
        self.assertTrue(interim.startswith(b"HTTP/1.1 100"), interim)
        self.assertNotIn(b"onnection", interim)
        s.sendall(b"{}")
        st, hdrs, _ = read_response(s)
        self.assertEqual(st, 400)
        self.assertNotEqual(hdrs.get("connection"), "close")

    def test_a_consumed_body_keeps_the_connection_alive(self):
        s = self.connect(self.port)
        s.sendall(head("POST", CHAT, self.port, "Content-Length: 2") + b"{}")
        st, hdrs, _ = read_response(s)
        self.assertEqual(st, 400)
        self.assertNotEqual(hdrs.get("connection"), "close")
        s.sendall(PIPED)
        self.assertEqual(read_response(s)[0], 200)

    def test_refusals_that_skip_the_body_close_the_connection(self):
        body = PIPED
        cl = "Content-Length: %d" % len(body)
        cases = {
            "404": (head("POST", "/nope", self.port, cl) + body, 404),
            "403 origin": (head("POST", CHAT, self.port, cl, "Origin: http://evil.example") + body, 403),
            "403 host": (head("POST", CHAT, self.port, cl).replace(b"Host: 127.0.0.1", b"Host: evil.example", 1) + body, 403),
            "401 admin": (head("POST", "/api/config", self.port, cl) + body, 401),
            "get with body": (head("GET", "/health", self.port, cl) + body, 200),
            "get api with body": (head("GET", "/api/status", self.port, cl, "X-Maestro-Token: " + self.token) + body, None),
            "chunked get": (head("GET", "/health", self.port, "Transfer-Encoding: chunked") + b"%x\r\n" % len(body) + body + b"\r\n0\r\n\r\n", 200),
        }
        for name, (payload, status) in cases.items():
            got = self.only_one_response(self.port, payload)
            if status:
                self.assertEqual(got, status, name)

    def test_a_wrong_api_key_closes_the_connection_too(self):
        self.stop()
        self.start([E("openai/gpt-x")], env_extra={"MAESTRO_API_KEY": "client-key"})
        body = PIPED
        self.assertEqual(self.only_one_response(self.port, head("POST", CHAT, self.port, "Content-Length: %d" % len(body)) + body), 401)


class Site(Wire, SiteCase):
    def test_a_refused_post_with_a_pipelined_request_gets_one_response_and_a_closed_connection(self):
        body = b"GET /privacy HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"
        payload = head("POST", "/", self.port, "Content-Length: %d" % len(body)) + body
        self.assertEqual(self.only_one_response(self.port, payload), 405)

    def test_other_refused_methods_and_requests_with_a_body_close_the_connection(self):
        body = b"GET /privacy HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"
        cl = "Content-Length: %d" % len(body)
        for method in ("PUT", "DELETE", "PATCH", "OPTIONS", "TRACE"):
            self.only_one_response(self.port, head(method, "/", self.port, cl) + body)
        self.assertEqual(self.only_one_response(self.port, head("GET", "/health", self.port, cl) + body), 200)
        self.assertEqual(self.only_one_response(self.port, head("GET", "/health", self.port, "Transfer-Encoding: chunked") + b"0\r\n\r\n" + body), 200)

    def test_ordinary_requests_stay_on_a_keep_alive_connection(self):
        s = self.connect(self.port)
        for _ in range(2):
            s.sendall(PIPED)
            st, hdrs, _ = read_response(s)
            self.assertEqual(st, 200)
            self.assertNotEqual(hdrs.get("connection"), "close")


if __name__ == "__main__":
    unittest.main()
