We're seeing stale data from `UserService` (`users/service.py`) since the caching layer went in:

1. After `update_user(7, name="Alicia")`, `get_user(7)` is right but `list_team("core")` still shows "Alice" until the cache expires (5 min).
2. After moving someone with `update_user(7, team="infra")` they still show up in the old team's listing and don't appear in the new one.
3. `search("al", limit=2)` followed by `search("al", limit=10)` returns only 2 results the second time.
4. The admin page adds display-only keys to the dict it gets back from `get_user` / `list_team`, and those keys then show up in later calls for other requests.

Keep the caching (we need the hit rate, the DB is slow, so flushing the whole cache on every update is not OK), just make it correct.
