import os
import unittest

from conf.parser import ParseError, load, parse, section
from conf.settings import Settings

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class Sections(unittest.TestCase):
    def test_sections_flatten(self):
        cfg = parse("top = 1\n[db]\nurl = x\n[ cache ]\nttl = 30\n")
        self.assertEqual(cfg, {"top": "1", "db.url": "x", "cache.ttl": "30"})

    def test_section_helper(self):
        cfg = parse("a = 0\n[db]\nurl = x\npool = 3\n[dbx]\nurl = y\n")
        self.assertEqual(section(cfg, "db"), {"url": "x", "pool": "3"})
        self.assertEqual(section(cfg, "dbx"), {"url": "y"})
        self.assertEqual(section(cfg, "nope"), {})

    def test_empty_header(self):
        with self.assertRaisesRegex(ParseError, "line 3"):
            parse("a = 1\n\n[]\nb = 2\n")
        with self.assertRaisesRegex(ParseError, "line 1"):
            parse("[   ]\n")

    def test_inline_comments(self):
        cfg = parse("port = 8080   # behind nginx\npool = 10 ; tuned\nurl = http://x/#top\nk = a;b\n")
        self.assertEqual(cfg, {"port": "8080", "pool": "10", "url": "http://x/#top", "k": "a;b"})
        self.assertEqual(parse("t = x\t# tab comment\n"), {"t": "x"})

    def test_quoted(self):
        cfg = parse('motd = "hello # world ; ok"\npad = "  padded  "  # trailing\nempty = ""\n')
        self.assertEqual(cfg, {"motd": "hello # world ; ok", "pad": "  padded  ", "empty": ""})

    def test_flat_files_unchanged(self):
        cfg = load(os.path.join(ROOT, "config", "legacy.conf"))
        self.assertEqual(cfg, {"retries": "3", "endpoint": "https://api.internal/v2/#status", "owner": "ops-team"})
        with self.assertRaisesRegex(ParseError, "line 2"):
            parse("[s]\nno equals here\n")

    def test_settings_from_sectioned_file(self):
        s = Settings.from_file(os.path.join(ROOT, "config", "app.ini"))
        self.assertEqual((s.name, s.host, s.port, s.debug, s.pool_size), ("billing", "0.0.0.0", 8080, False, 10))
        self.assertEqual(s.db_url, "postgres://app@db/billing?sslmode=require#primary ; ro")


if __name__ == "__main__":
    unittest.main()
