import unittest

from contacts.api import handle_get, handle_search
from contacts.db import add_contact, connect


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.conn = connect()
        add_contact(self.conn, "Ada Lovelace", "ada@example.com")
        add_contact(self.conn, "Grace Hopper", "grace@navy.mil")

    def test_get(self):
        self.assertEqual(handle_get(self.conn, {"id": "1"}), (200, {"id": 1, "name": "Ada Lovelace", "email": "ada@example.com"}))
        self.assertEqual(handle_get(self.conn, {"id": "x"})[0], 400)

    def test_search(self):
        status, body = handle_search(self.conn, {"q": "grace"})
        self.assertEqual(status, 200)
        self.assertEqual([r["name"] for r in body["results"]], ["Grace Hopper"])
        self.assertEqual(handle_search(self.conn, {"q": " "})[0], 400)


if __name__ == "__main__":
    unittest.main()
