from datetime import timedelta

from .tz import EASTERN, UTC


def _parse_hhmm(at):
    hh, mm = at.split(":")
    hh, mm = int(hh), int(mm)
    if not (0 <= hh < 24 and 0 <= mm < 60):
        raise ValueError("bad time %r" % at)
    return hh, mm


def next_occurrences(at, start, count, tz=EASTERN, weekdays=None):
    """The next `count` times (aware UTC datetimes) strictly after `start` at which a reminder set for local
    time `at` ("HH:MM") in `tz` fires. `weekdays`: optional set of date.weekday() numbers (local)."""
    if start.tzinfo is None:
        raise ValueError("start must be timezone-aware")
    hh, mm = _parse_hhmm(at)
    local = start.astimezone(tz)
    first = local.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if first <= local:
        first += timedelta(days=1)
    t = first.astimezone(UTC)
    out = []
    while len(out) < count:
        if weekdays is None or t.astimezone(tz).weekday() in weekdays:
            out.append(t)
        t += timedelta(days=1)
    return out


def describe(when, tz=EASTERN):
    """'2024-11-04 09:00 EST' for an aware datetime."""
    local = when.astimezone(tz)
    return local.strftime("%Y-%m-%d %H:%M ") + local.tzname()
