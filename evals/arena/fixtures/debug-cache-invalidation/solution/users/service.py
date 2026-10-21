import copy

from .cache import TTLCache


class UserService:
    def __init__(self, db, cache=None):
        self.db = db
        self.cache = cache if cache is not None else TTLCache()

    def _cached(self, key, load):
        value = self.cache.get(key)
        if value is None:
            value = load()
            self.cache.set(key, copy.deepcopy(value))
            return value
        return copy.deepcopy(value)  # callers may mutate what they get back

    def get_user(self, user_id):
        def load():
            user = self.db.fetch(user_id)
            if user is None:
                raise KeyError(user_id)
            return user

        return self._cached(("user", user_id), load)

    def list_team(self, team):
        return self._cached(("team", team), lambda: self.db.fetch_team(team))

    def search(self, prefix, limit=10):
        return self._cached(("search", prefix.lower(), limit), lambda: self.db.search_name(prefix, limit))

    def update_user(self, user_id, **fields):
        old = self.cache.get(("user", user_id)) or self.db.fetch(user_id) or {}
        self.db.update(user_id, **fields)
        teams = {old.get("team"), fields.get("team")}
        self.cache.delete(("user", user_id))
        self.cache.delete_where(lambda k: (k[0] == "team" and k[1] in teams) or k[0] == "search")
