import json
import time

from .retry import retry

_sleep = time.sleep


def _pause(seconds):
    _sleep(seconds)


@retry(times=4, delay=0.2, exceptions=(ConnectionError,), sleep=lambda s: _pause(s))
def fetch_config(transport):
    """`transport()` returns the raw JSON body or raises ConnectionError."""
    return json.loads(transport())
