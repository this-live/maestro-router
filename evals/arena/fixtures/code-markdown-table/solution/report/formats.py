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


def _is_number(s):
    try:
        float(s)
    except ValueError:
        return False
    return True


def to_markdown(headers, rows):
    def cell(c):
        return str(c).replace("|", "\\|")

    aligns = []
    for i in range(len(headers)):
        vals = [r[i] for r in rows if i < len(r) and str(r[i]) != ""]
        aligns.append("---:" if vals and all(_is_number(v) for v in vals) else ":---")
    lines = ["| " + " | ".join(cell(h) for h in headers) + " |", "| " + " | ".join(aligns) + " |"]
    lines += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


FORMATS = {"text": to_text, "csv": to_csv, "md": to_markdown}
