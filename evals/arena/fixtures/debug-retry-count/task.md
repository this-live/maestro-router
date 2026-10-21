Something's off with the `retry` decorator in `netutil/retry.py`. With `@retry(times=3)` the function only gets called twice, and when every attempt fails the wrapper quietly returns `None` instead of raising, so `fetch_config()` hands callers `None` and they crash later with `TypeError: 'NoneType' object is not subscriptable`.

`times` is meant to be the total number of attempts. After the last failed attempt the last exception should propagate, and there shouldn't be a sleep after the final attempt. Please fix it.
