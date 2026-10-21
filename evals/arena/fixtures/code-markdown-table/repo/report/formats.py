import csv
import io


def to_text(headers, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    lines = ["  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip() for r in [headers] + rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def to_csv(headers, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(headers)
    w.writerows(rows)
    return buf.getvalue().rstrip("\n")


FORMATS = {"text": to_text, "csv": to_csv}
