Daily reminders are drifting around DST changes. A user with a 09:00 America/New_York reminder created in October got it at 08:00 every day after Nov 3; reminders created in February fire at 10:00 after the March change. A reminder should always fire at its wall-clock time in the user's zone.

Edge cases product signed off on:
- if the local time doesn't exist that day (spring forward, e.g. 02:30 on 2024-03-10) fire at the shifted time, i.e. 03:30 EDT
- if it happens twice (fall back, e.g. 01:30 on 2024-11-03) fire once, at the first (EDT) one, and never a second time that day, even if we compute the schedule while we're inside the repeated hour

The scheduling code is `reminders/schedule.py`. `reminders/tz.py` is hand-rolled on purpose (prod boxes have no tzdata), keep using it. Results must stay aware UTC datetimes.
