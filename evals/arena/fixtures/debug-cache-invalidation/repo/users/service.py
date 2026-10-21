from .cache import TTLCache


class UserService:
    def __init__(self, db, cache=None):
        self.db = db
        self.cache = cache if cache is not None else TTLCache()

    def get_user(self, user_id):
        key = ("user", user_id)
        user = self.cache.get(key)
        if user is None:
            user = self.db.fetch(user_id)
            if user is None:
                raise KeyError(user_id)
            self.cache.set(key, user)
        return user

    def list_team(self, team):
        key = ("team", team)
        members = self.cache.get(key)
        if members is None:
            members = self.db.fetch_team(team)
            self.cache.set(key, members)
        return members

    def search(self, prefix, limit=10):
        key = ("search", prefix.lower())
        hits = self.cache.get(key)
        if hits is None:
            hits = self.db.search_name(prefix, limit)
            self.cache.set(key, hits)
        return hits

    def update_user(self, user_id, **fields):
        self.db.update(user_id, **fields)
        self.cache.delete(("user", user_id))
