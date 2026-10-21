Product wants a sessions table from the raw clickstream in `data/events.jsonl`. Add `clickstream/sessions.py` with:

- `build_sessions(path)` -> list of dicts, one per session, keys: `user, session_start, session_end, duration_s, events, pages, converted`
- `write_sessions_csv(sessions, path)` writing those columns in that order (header row, `\n` line endings)
- `python3 -m clickstream.sessions data/events.jsonl out.csv`

Rules:
- the export is dirty: skip lines that aren't valid JSON or lack `user`/`ts`/`event`; drop exact duplicate events (same user, ts, event and path)
- timestamps are ISO 8601 with `Z` or a UTC offset; normalise to UTC. Lines aren't in order
- per user, a new session starts when the gap since that user's previous event is more than 30 minutes (exactly 30 min is the same session)
- `session_start`/`session_end` as `YYYY-MM-DDTHH:MM:SSZ` (UTC), `duration_s` = end - start in whole seconds, `events` = number of events, `pages` = number of distinct `path`s among `page_view` events, `converted` = 1 if the session has a `purchase` event else 0
- sorted by user, then session_start

Has to run on Python 3.9 (prod).
