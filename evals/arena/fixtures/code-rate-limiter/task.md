We need per-API-key rate limiting in `svc/app.py` before we open the API to partners.

- Token bucket, implemented as `TokenBucket(capacity, rate, clock)` in a new `svc/ratelimit.py`: starts full, refills continuously at `rate` tokens/second up to `capacity`. `allow()` takes a token and returns True, or returns False without taking one. `retry_after()` returns the seconds (float) until a token will be available (0 if one is available now).
- `App` keeps one bucket per API key, sized from `RATE_LIMIT_BURST` / `RATE_LIMIT_PER_SEC` in `svc/config.py`, using the app's clock.
- Over the limit: status 429, body `{"error": "rate_limited"}`, and a `Retry-After` header with the wait in whole seconds, rounded up, at least 1.
- `/health` is never limited, and requests rejected with 401 (missing/unknown key) must not use up anyone's tokens.
