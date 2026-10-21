import csv


def load(path):
    """Return (headers, rows) with every cell as a stripped string."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = [h.strip() for h in next(reader)]
        rows = [[c.strip() for c in row] for row in reader if any(c.strip() for c in row)]
    return headers, rows
