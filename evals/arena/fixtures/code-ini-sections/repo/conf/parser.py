class ParseError(ValueError):
    pass


def parse(text):
    """Parse `key = value` lines into a dict of strings. Blank lines and lines starting with # or ; are skipped."""
    out = {}
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if "=" not in line:
            raise ParseError("line %d: expected 'key = value'" % n)
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ParseError("line %d: empty key" % n)
        out[key] = value.strip()
    return out


def load(path):
    with open(path, encoding="utf-8") as f:
        return parse(f.read())
