import os
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPECTED = [
    ("ALLOWED_ORIGINS", "*"),
    ("API_URL", "https://api.example.com"),
    ("CACHE_TTL", "300"),
    ("DATABASE_URL", "(required)"),
    ("ENABLE_BETA", "false"),
    ("LOG_LEVEL", "info"),
    ("QUEUE_NAME", "default"),
    ("REDIS_URL", "redis://localhost:6379/0"),
    ("SENTRY_DSN", "(none)"),
    ("SMTP_HOST", "localhost"),
    ("SMTP_PASSWORD_FILE", "(none)"),
    ("SMTP_PORT", "25"),
    ("WORKERS", "4"),
]


def norm(cell):
    return cell.strip().strip("`").strip("\"'").strip().lower()


def read_table():
    with open(os.path.join(ROOT, "ENV_VARS.md"), encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip().startswith("|")]
    rows = [[c.strip() for c in line.strip("|").split("|")] for line in lines]
    return rows


class Answer(unittest.TestCase):
    def test_table(self):
        rows = read_table()
        self.assertGreaterEqual(len(rows), 2, "no markdown table found")
        self.assertEqual([norm(c) for c in rows[0]], ["name", "default"])
        self.assertTrue(all(set(c.strip()) <= set("-: ") for c in rows[1]), "second row must be the separator")
        got = [(norm(r[0]).upper(), norm(r[1])) for r in rows[2:]]
        self.assertEqual([n for n, _ in got], [n for n, _ in EXPECTED], "variable names / order")
        self.assertEqual(got, EXPECTED)

    def test_code_untouched(self):
        with open(os.path.join(ROOT, "service", "settings.py"), encoding="utf-8") as f:
            self.assertIn('DATABASE_URL = os.environ["DATABASE_URL"]', f.read())


if __name__ == "__main__":
    unittest.main()
