import json


def iter_events(path):
    """Yield the raw JSON objects from a .jsonl export (no validation)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
