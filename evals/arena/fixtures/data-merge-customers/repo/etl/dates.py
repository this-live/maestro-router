from datetime import datetime

FORMATS = ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%d/%m/%Y %H:%M")


def parse_ts(text):
    """Parse the timestamp formats our exports use (all UTC) into a naive datetime."""
    text = (text or "").strip()
    for fmt in FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError("unrecognised timestamp %r" % text)
