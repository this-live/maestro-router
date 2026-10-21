class Database:
    """Stand-in for the real (slow) user DB. `queries` counts round trips."""

    def __init__(self, rows=()):
        self._rows = {r["id"]: dict(r) for r in rows}
        self.queries = 0

    def fetch(self, user_id):
        self.queries += 1
        row = self._rows.get(user_id)
        return dict(row) if row else None

    def fetch_team(self, team):
        self.queries += 1
        return [dict(r) for _, r in sorted(self._rows.items()) if r["team"] == team]

    def search_name(self, prefix, limit):
        self.queries += 1
        prefix = prefix.lower()
        hits = sorted((r for r in self._rows.values() if r["name"].lower().startswith(prefix)), key=lambda r: r["name"])
        return [dict(r) for r in hits[:limit]]

    def update(self, user_id, **fields):
        self.queries += 1
        if user_id not in self._rows:
            raise KeyError(user_id)
        self._rows[user_id].update(fields)
