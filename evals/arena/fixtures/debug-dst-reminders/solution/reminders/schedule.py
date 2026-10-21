from datetime import datetime, time, timedelta

from .tz import EASTERN, UTC


def _parse_hhmm(at):
    hh, mm = at.split(":")
    hh, mm = int(hh), int(mm)
    if not (0 <= hh < 24 and 0 <= mm < 60):
        raise ValueError("bad time %r" % at)
    return hh, mm


def next_occurrences(at, start, count, tz=EASTERN, weekdays=None):
    """The next `count` times (aware UTC datetimes) strictly after `start` at which a reminder set for local
    time `at` ("HH:MM") in `tz` fires. `weekdays`: optional set of date.weekday() numbers (local).

    Days are stepped in local calendar dates and each occurrence is built from the wall time (fold=0), so it
    stays at the same local time across DST changes; a skipped time maps to the shifted instant and a repeated
    time fires once, at its first (fold=0) instant. Comparisons are done in UTC."""
    if start.tzinfo is None:
        raise ValueError("start must be timezone-aware")
    hh, mm = _parse_hhmm(at)
    start_utc = start.astimezone(UTC)
    day = start_utc.astimezone(tz).date() - timedelta(days=1)
    out = []
    while len(out) < count:
        if weekdays is None or day.weekday() in weekdays:
            t = datetime.combine(day, time(hh, mm), tzinfo=tz).astimezone(UTC)
            if t > start_utc:
                out.append(t)
        day += timedelta(days=1)
    return out


def describe(when, tz=EASTERN):
    """'2024-11-04 09:00 EST' for an aware datetime."""
    local = when.astimezone(tz)
    return local.strftime("%Y-%m-%d %H:%M ") + local.tzname()
