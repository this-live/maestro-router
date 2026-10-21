import csv


def read_rows(path):
    """Yield CSV rows as dicts with stripped keys and values; blank lines are skipped."""
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            if any(row.values()):
                yield row
