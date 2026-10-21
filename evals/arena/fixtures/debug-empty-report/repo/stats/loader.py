import csv


def load_column(path, column="value"):
    """Return the non-blank values of `column` as floats."""
    values = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = (row.get(column) or "").strip()
            if raw:
                values.append(float(raw))
    return values
