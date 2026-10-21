import functools
import time


def retry(times=3, delay=0.0, backoff=2.0, exceptions=(Exception,), sleep=time.sleep):
    """Retry the wrapped function on `exceptions`, up to `times` attempts in total."""
    if times < 1:
        raise ValueError("times must be >= 1")

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            wait = delay
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions:
                    if attempt == times:
                        raise
                    sleep(wait)
                    wait *= backoff

        return wrapper

    return deco
