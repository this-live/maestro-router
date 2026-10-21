# reminders

Computes when recurring reminders fire.

```python
from reminders.schedule import next_occurrences, describe
next_occurrences("09:00", start_utc, count=5)               # every day at 09:00 New York time
next_occurrences("09:00", start_utc, 5, weekdays={0,1,2,3,4})  # Mon-Fri only
```

`start_utc` is an aware datetime; occurrences are strictly after it and returned as aware UTC datetimes.
Time zones come from `reminders/tz.py` (US rules, no tzdata needed).
