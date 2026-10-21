import json


def load_reminders(path):
    """[{"user": ..., "at": "HH:MM", "zone": "America/New_York", "weekdays": [0, 1, 2]}, ...]"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
