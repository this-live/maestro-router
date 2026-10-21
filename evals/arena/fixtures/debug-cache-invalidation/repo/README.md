# users

User directory service. `UserService` wraps the slow `Database` with a TTL cache (`users/cache.py`).
Run tests: `python3 -m unittest discover -s tests -t .`
