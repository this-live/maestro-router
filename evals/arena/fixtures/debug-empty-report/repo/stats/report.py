from .loader import load_column


def summarize(values):
    """count / mean / min / max of a list of numbers."""
    values = list(values)
    total = sum(values)
    mean = total / len(values)
    return {"count": len(values), "mean": round(mean, 3), "min": min(values), "max": max(values)}


def format_summary(s):
    return "count=%d mean=%.3f min=%g max=%g" % (s["count"], s["mean"], s["min"], s["max"])


def report(path, column="value"):
    return format_summary(summarize(load_column(path, column)))
