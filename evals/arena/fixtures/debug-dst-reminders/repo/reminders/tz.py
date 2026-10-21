"""US time zones without tzdata (rules in force since 2007). PEP 495 aware: `fold` picks the side of a transition."""
from datetime import datetime, timedelta, timezone, tzinfo

UTC = timezone.utc
ZERO = timedelta(0)
HOUR = timedelta(hours=1)


def _first_sunday_on_or_after(dt):
    days_to_go = 6 - dt.weekday()
    if days_to_go:
        dt += timedelta(days_to_go)
    return dt


def us_dst_range(year):
    """DST runs from 2:00 on the second Sunday of March to 2:00 on the first Sunday of November (wall times)."""
    start = _first_sunday_on_or_after(datetime(year, 3, 8, 2))
    end = _first_sunday_on_or_after(datetime(year, 11, 1, 2))
    return start, end


class USTimeZone(tzinfo):
    def __init__(self, hours, reprname, stdname, dstname):
        self.stdoffset = timedelta(hours=hours)
        self.reprname = reprname
        self.stdname = stdname
        self.dstname = dstname

    def __repr__(self):
        return self.reprname

    def tzname(self, dt):
        return self.dstname if self.dst(dt) else self.stdname

    def utcoffset(self, dt):
        return self.stdoffset + self.dst(dt)

    def dst(self, dt):
        if dt is None or dt.tzinfo is None:
            return ZERO
        start, end = us_dst_range(dt.year)
        dt = dt.replace(tzinfo=None)
        if start + HOUR <= dt < end - HOUR:
            return HOUR
        if end - HOUR <= dt < end:  # repeated hour: fold=0 is the first (DST) pass
            return ZERO if dt.fold else HOUR
        if start <= dt < start + HOUR:  # skipped hour: fold=0 reads it with the old (standard) offset
            return HOUR if dt.fold else ZERO
        return ZERO

    def fromutc(self, dt):
        assert dt.tzinfo is self
        start, end = us_dst_range(dt.year)
        start = start.replace(tzinfo=self)
        end = end.replace(tzinfo=self)
        std_time = dt + self.stdoffset
        dst_time = std_time + HOUR
        if end <= dst_time < end + HOUR:
            return std_time.replace(fold=1)
        if std_time < start or dst_time >= end:
            return std_time
        return dst_time


EASTERN = USTimeZone(-5, "America/New_York", "EST", "EDT")
CENTRAL = USTimeZone(-6, "America/Chicago", "CST", "CDT")
PACIFIC = USTimeZone(-8, "America/Los_Angeles", "PST", "PDT")
ZONES = {repr(z): z for z in (EASTERN, CENTRAL, PACIFIC)}
