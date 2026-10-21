import re


class ParseError(ValueError):
    pass


_COMMENT = re.compile(r"\s[#;]")


def _value(raw):
    raw = raw.strip()
    if raw.startswith('"'):
        end = raw.find('"', 1)
        if end != -1:
            return raw[1:end]
    m = _COMMENT.search(raw)
    if m:
        raw = raw[:m.start()]
    return raw.strip()


def parse(text):
    """Parse `key = value` lines into a flat dict of strings.

    Blank lines and lines starting with # or ; are skipped. `[name]` starts a section whose keys are stored as
    `name.key`. Whitespace + # or ; starts an inline comment; a double-quoted value is taken literally."""
    out = {}
    prefix = ""
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip()
            if not name:
                raise ParseError("line %d: empty section name" % n)
            prefix = name + "."
            continue
        if "=" not in line:
            raise ParseError("line %d: expected 'key = value'" % n)
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ParseError("line %d: empty key" % n)
        out[prefix + key] = _value(value)
    return out


def section(cfg, name):
    """{key: value} for the keys of section `name` (without the `name.` prefix)."""
    prefix = name + "."
    return {k[len(prefix):]: v for k, v in cfg.items() if k.startswith(prefix)}


def load(path):
    with open(path, encoding="utf-8") as f:
        return parse(f.read())
