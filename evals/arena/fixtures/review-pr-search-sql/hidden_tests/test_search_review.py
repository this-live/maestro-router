import unittest

from contacts.api import handle_search
from contacts.db import add_contact, connect, search_contacts

PEOPLE = [
    ("Ada Lovelace", "ada@example.com"),
    ("grace hopper", "grace@navy.mil"),
    ("Conan O'Brien", "conan@tbs.tv"),
    ("Zed Shaw", "zed@learncode.org"),
    ("Percent 100% Guy", "pct@example.com"),
    ("Under_Score", "us@example.com"),
    ("Underxscore", "ux@example.com"),
    ("bob", "bob@ADA.org"),
]


class Search(unittest.TestCase):
    def setUp(self):
        self.conn = connect()
        for name, email in PEOPLE:
            add_contact(self.conn, name, email)

    def names(self, term, **kw):
        return [r["name"] for r in search_contacts(self.conn, term, **kw)]

    def test_apostrophe(self):
        self.assertEqual(self.names("O'Brien"), ["Conan O'Brien"])

    def test_injection(self):
        self.assertEqual(self.names("x' OR '1'='1"), [])
        self.assertEqual(self.names("%' UNION SELECT 1, sqlite_version(), 3 --"), [])
        self.assertEqual(len(self.names("e")), 7)

    def test_wildcards_literal(self):
        self.assertEqual(self.names("100%"), ["Percent 100% Guy"])
        self.assertEqual(self.names("%"), ["Percent 100% Guy"])
        self.assertEqual(self.names("_"), ["Under_Score"])
        self.assertEqual(self.names("r_s"), ["Under_Score"])

    def test_case_insensitive_and_email(self):
        self.assertEqual(self.names("ADA"), ["Ada Lovelace", "bob"])
        self.assertEqual(self.names("NAVY"), ["grace hopper"])

    def test_sorted_case_insensitive(self):
        self.assertEqual(self.names("o"), ["Ada Lovelace", "bob", "Conan O'Brien", "grace hopper", "Percent 100% Guy", "Under_Score",
                                           "Underxscore", "Zed Shaw"])

    def test_result_shape(self):
        self.assertEqual(search_contacts(self.conn, "zed"), [{"id": 4, "name": "Zed Shaw", "email": "zed@learncode.org"}])

    def test_limit(self):
        self.assertEqual(self.names("o", limit=2), ["Ada Lovelace", "bob"])
        self.assertEqual(self.names("o", limit="3"), ["Ada Lovelace", "bob", "Conan O'Brien"])
        for bad in (0, 101, "5; DROP TABLE contacts", "abc", -1):
            with self.assertRaises(ValueError, msg=repr(bad)):
                search_contacts(self.conn, "o", limit=bad)
        self.assertEqual(len(self.names("e")), 7)

    def test_handler(self):
        status, body = handle_search(self.conn, {"q": "o'brien"})
        self.assertEqual((status, [r["name"] for r in body["results"]]), (200, ["Conan O'Brien"]))
        self.assertEqual(handle_search(self.conn, {"q": "o", "limit": "1; DROP TABLE contacts"})[0], 400)
        self.assertEqual(handle_search(self.conn, {"q": "o", "limit": "1000"})[0], 400)
        self.assertEqual(handle_search(self.conn, {"q": "o", "limit": "2"})[1]["results"][1]["name"], "bob")
        self.assertEqual(len(self.names("e")), 7)


if __name__ == "__main__":
    unittest.main()
