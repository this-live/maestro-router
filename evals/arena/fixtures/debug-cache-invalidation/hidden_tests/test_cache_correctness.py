import unittest

from users.cache import TTLCache
from users.db import Database
from users.seed import SAMPLE_USERS
from users.service import UserService


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make():
    clock = Clock()
    db = Database(SAMPLE_USERS)
    return UserService(db, TTLCache(ttl=300, clock=clock)), db, clock


def names(rows):
    return [r["name"] for r in rows]


class Invalidation(unittest.TestCase):
    def test_rename_visible_in_team(self):
        svc, db, _ = make()
        self.assertEqual(names(svc.list_team("core")), ["Albert", "Alba", "Alice"])
        svc.update_user(7, name="Alicia")
        self.assertEqual(names(svc.list_team("core")), ["Albert", "Alba", "Alicia"])

    def test_move_team(self):
        svc, db, _ = make()
        svc.list_team("core")
        svc.list_team("infra")
        svc.get_user(7)
        svc.update_user(7, team="infra")
        self.assertEqual(names(svc.list_team("core")), ["Albert", "Alba"])
        self.assertEqual(names(svc.list_team("infra")), ["Bea", "Dana", "Alice"])
        self.assertEqual(svc.get_user(7)["team"], "infra")

    def test_rename_visible_in_search(self):
        svc, db, _ = make()
        self.assertEqual(names(svc.search("al")), ["Alba", "Albert", "Alfred", "Alice"])
        svc.update_user(4, name="Alan")
        self.assertEqual(names(svc.search("al")), ["Alan", "Alba", "Albert", "Alfred", "Alice"])

    def test_search_limit_in_key(self):
        svc, db, _ = make()
        self.assertEqual(names(svc.search("al", limit=2)), ["Alba", "Albert"])
        self.assertEqual(names(svc.search("al", limit=10)), ["Alba", "Albert", "Alfred", "Alice"])
        self.assertEqual(names(svc.search("AL", limit=2)), ["Alba", "Albert"])

    def test_callers_cannot_poison_cache(self):
        svc, db, _ = make()
        u = svc.get_user(1)
        u["display"] = "<b>Albert</b>"
        u["name"] = "MUTATED"
        self.assertEqual(svc.get_user(1), {"id": 1, "name": "Albert", "team": "core"})
        team = svc.list_team("web")
        team[0]["badge"] = "x"
        team.append({"id": 99, "name": "Ghost", "team": "web"})
        self.assertEqual(svc.list_team("web"), [{"id": 4, "name": "Carl", "team": "web"},
                                                {"id": 5, "name": "Alfred", "team": "web"}])
        hits = svc.search("da")
        hits[0]["name"] = "zzz"
        self.assertEqual(names(svc.search("da")), ["Dana"])


class StillCaches(unittest.TestCase):
    def test_hits_do_not_query(self):
        svc, db, _ = make()
        svc.get_user(1)
        svc.list_team("core")
        svc.search("al", limit=3)
        before = db.queries
        for _ in range(5):
            svc.get_user(1)
            svc.list_team("core")
            svc.search("al", limit=3)
        self.assertEqual(db.queries, before)

    def test_unrelated_entries_survive_update(self):
        svc, db, _ = make()
        svc.get_user(2)
        svc.list_team("web")
        before = db.queries
        svc.update_user(7, name="Alicia")
        q_after_update = db.queries
        svc.get_user(2)
        svc.list_team("web")
        self.assertEqual(db.queries, q_after_update)
        self.assertGreater(q_after_update, before)

    def test_ttl_expiry(self):
        svc, db, clock = make()
        svc.get_user(1)
        clock.t += 299
        svc.get_user(1)
        self.assertEqual(db.queries, 1)
        clock.t += 2
        svc.get_user(1)
        self.assertEqual(db.queries, 2)


if __name__ == "__main__":
    unittest.main()
