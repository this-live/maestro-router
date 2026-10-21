import unittest

from users.db import Database
from users.seed import SAMPLE_USERS
from users.service import UserService


class ServiceTest(unittest.TestCase):
    def test_get_user_cached(self):
        db = Database(SAMPLE_USERS)
        svc = UserService(db)
        self.assertEqual(svc.get_user(2)["name"], "Bea")
        svc.get_user(2)
        self.assertEqual(db.queries, 1)

    def test_update_refreshes_user(self):
        svc = UserService(Database(SAMPLE_USERS))
        svc.get_user(7)
        svc.update_user(7, name="Alicia")
        self.assertEqual(svc.get_user(7)["name"], "Alicia")

    def test_missing(self):
        with self.assertRaises(KeyError):
            UserService(Database(SAMPLE_USERS)).get_user(99)


if __name__ == "__main__":
    unittest.main()
