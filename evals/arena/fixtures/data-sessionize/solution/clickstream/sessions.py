import csv
import json
import sys
from datetime import datetime, timedelta, timezone

GAP = timedelta(minutes=30)
FIELDS = ["user", "session_start", "session_end", "duration_s", "events", "pages", "converted"]


def parse_ts(text):
    text = text.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def fmt(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def load_events(path):
    seen = set()
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                user, ts, event = e["user"], parse_ts(e["ts"]), e["event"]
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
            key = (user, ts, event, e.get("path"))
            if key in seen:
                continue
            seen.add(key)
            out.append({"user": user, "ts": ts, "event": event, "path": e.get("path")})
    return out


def _session(user, evs):
    start, end = evs[0]["ts"], evs[-1]["ts"]
    return {
        "user": user,
        "session_start": fmt(start),
        "session_end": fmt(end),
        "duration_s": int((end - start).total_seconds()),
        "events": len(evs),
        "pages": len({e["path"] for e in evs if e["event"] == "page_view"}),
        "converted": int(any(e["event"] == "purchase" for e in evs)),
    }


def build_sessions(path):
    by_user = {}
    for e in load_events(path):
        by_user.setdefault(e["user"], []).append(e)
    sessions = []
    for user in sorted(by_user):
        evs = sorted(by_user[user], key=lambda e: e["ts"])
        current = [evs[0]]
        for e in evs[1:]:
            if e["ts"] - current[-1]["ts"] > GAP:
                sessions.append(_session(user, current))
                current = []
            current.append(e)
        sessions.append(_session(user, current))
    return sessions


def write_sessions_csv(sessions, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for s in sessions:
            w.writerow({k: s[k] for k in FIELDS})


if __name__ == "__main__":
    write_sessions_csv(build_sessions(sys.argv[1]), sys.argv[2])
