import os
import subprocess
import sys
import tempfile
import unittest

from clickstream.sessions import build_sessions, write_sessions_csv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data", "events.jsonl")

EXPECTED_CSV = """user,session_start,session_end,duration_s,events,pages,converted
u1,2024-05-01T10:00:00Z,2024-05-01T11:20:00Z,4800,6,3,1
u10,2024-05-01T08:00:00Z,2024-05-01T08:00:00Z,0,1,1,0
u2,2024-05-01T09:59:30Z,2024-05-01T10:29:30Z,1800,2,2,0
u2,2024-05-01T10:59:31Z,2024-05-01T11:15:00Z,929,3,1,1
u3,2024-05-01T23:45:00Z,2024-05-02T00:10:00Z,1500,3,2,0
"""


class Sessions(unittest.TestCase):
    def test_build(self):
        s = build_sessions(DATA)
        self.assertEqual(len(s), 5)
        self.assertEqual(s[3], {"user": "u2", "session_start": "2024-05-01T10:59:31Z",
                                "session_end": "2024-05-01T11:15:00Z", "duration_s": 929, "events": 3, "pages": 1,
                                "converted": 1})

    def test_csv(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "s.csv")
            write_sessions_csv(build_sessions(DATA), out)
            with open(out, newline="", encoding="utf-8") as f:
                self.assertEqual(f.read(), EXPECTED_CSV)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "cli.csv")
            p = subprocess.run([sys.executable, "-m", "clickstream.sessions", "data/events.jsonl", out], cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(out, newline="", encoding="utf-8") as f:
                self.assertEqual(f.read(), EXPECTED_CSV)

    def test_gap_boundary_and_offsets(self):
        lines = [
            '{"user": "a", "ts": "2024-01-01T00:00:00Z", "event": "page_view", "path": "/x"}',
            '{"user": "a", "ts": "2024-01-01T01:30:00+01:00", "event": "page_view", "path": "/y"}',
            '{"user": "a", "ts": "2024-01-01T01:00:01Z", "event": "page_view", "path": "/x"}',
            '{"user": "a", "ts": "2024-01-01T01:00:01Z", "event": "purchase", "path": "/x"}',
            '[1, 2]',
            '',
        ]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "e.jsonl")
            with open(p, "w") as f:
                f.write("\n".join(lines) + "\n")
            s = build_sessions(p)
        self.assertEqual([(x["session_start"], x["session_end"], x["events"], x["pages"], x["converted"]) for x in s], [
            ("2024-01-01T00:00:00Z", "2024-01-01T00:30:00Z", 2, 2, 0),
            ("2024-01-01T01:00:01Z", "2024-01-01T01:00:01Z", 2, 1, 1),
        ])


if __name__ == "__main__":
    unittest.main()
