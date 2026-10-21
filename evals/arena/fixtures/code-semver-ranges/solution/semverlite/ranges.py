"""npm-style version ranges. See README / docs for the supported syntax."""
import re

from .version import Version

_OP = re.compile(r"^(<=|>=|<|>|=|\^|~)?\s*(.*)$")
_PARTIAL = re.compile(r"^v?(\d+|[xX*])(?:\.(\d+|[xX*]))?(?:\.(\d+|[xX*]))?$")
_ANY = [(">=", Version(0, 0, 0))]


def _partial(text):
    """(major, minor, patch) with None for missing / x parts, or None if not a partial version."""
    m = _PARTIAL.match(text)
    if not m:
        return None
    parts = [None if (p is None or p in "xX*") else int(p) for p in m.groups()]
    for i in range(1, 3):  # 1.x.3 is not allowed
        if parts[i] is not None and parts[i - 1] is None:
            raise ValueError("invalid version part in %r" % text)
    return parts


def _full(text):
    return Version.parse(text)


def _xrange(text):
    if text in ("", "*", "x", "X"):
        return list(_ANY)
    parts = _partial(text)
    if parts is None:
        return [("=", _full(text))]
    ma, mi, pa = parts
    if ma is None:
        return list(_ANY)
    if mi is None:
        return [(">=", Version(ma, 0, 0)), ("<", Version(ma + 1, 0, 0))]
    if pa is None:
        return [(">=", Version(ma, mi, 0)), ("<", Version(ma, mi + 1, 0))]
    return [("=", Version(ma, mi, pa))]


def _caret(text):
    parts = _partial(text)
    if parts is None:
        v = _full(text)
        ma, mi, pa = v.release
        lo = v
    else:
        ma, mi, pa = parts
        if ma is None:
            return list(_ANY)
        lo = Version(ma, mi or 0, pa or 0)
    if ma > 0 or mi is None:
        hi = Version(ma + 1, 0, 0)
    elif mi > 0 or pa is None:
        hi = Version(0, mi + 1, 0)
    else:
        hi = Version(0, 0, pa + 1)
    return [(">=", lo), ("<", hi)]


def _tilde(text):
    parts = _partial(text)
    if parts is None:
        v = _full(text)
        ma, mi = v.major, v.minor
        lo = v
    else:
        ma, mi, pa = parts
        if ma is None:
            return list(_ANY)
        lo = Version(ma, mi or 0, pa or 0)
    hi = Version(ma + 1, 0, 0) if mi is None and parts is not None else Version(ma, mi + 1, 0)
    return [(">=", lo), ("<", hi)]


def _comparator(token):
    m = _OP.match(token)
    op, rest = m.group(1), m.group(2).strip()
    if op is None:
        return _xrange(rest)
    if not rest:
        raise ValueError("missing version after %r" % op)
    if op == "^":
        return _caret(rest)
    if op == "~":
        return _tilde(rest)
    return [(op, _full(rest))]


def _parse_set(text):
    text = text.strip()
    if " - " in " %s " % text and re.search(r"\s-\s", text):
        parts = re.split(r"\s+-\s+", text)
        if len(parts) != 2:
            raise ValueError("invalid hyphen range %r" % text)
        return [(">=", _full(parts[0])), ("<=", _full(parts[1]))]
    # glue operators to the version that follows them: ">= 1.2.3" -> ">=1.2.3"
    text = re.sub(r"(<=|>=|<|>|=|\^|~)\s+", r"\1", text)
    tokens = text.split()
    if not tokens:
        return list(_ANY)
    if "-" in tokens:
        raise ValueError("invalid range %r" % text)
    out = []
    for tok in tokens:
        out.extend(_comparator(tok))
    return out


def parse_range(range_text):
    """List of AND-sets of (op, Version) comparators."""
    return [_parse_set(part) for part in range_text.split("||")]


def _test(op, v, c):
    if op == "=":
        return v == c
    if op == ">":
        return v > c
    if op == ">=":
        return v >= c
    if op == "<":
        return v < c
    return v <= c


def _set_ok(v, comparators):
    if not all(_test(op, v, c) for op, c in comparators):
        return False
    if v.prerelease:
        return any(c.prerelease and c.release == v.release for _, c in comparators)
    return True


def satisfies(version, range_text):
    """True if the version string `version` is inside `range_text`."""
    v = Version.parse(version)
    return any(_set_ok(v, s) for s in parse_range(range_text))


def max_satisfying(versions, range_text):
    """The highest version string in `versions` that satisfies `range_text`, or None."""
    sets = parse_range(range_text)
    best = None
    for text in versions:
        v = Version.parse(text)
        if any(_set_ok(v, s) for s in sets) and (best is None or v > best[0]):
            best = (v, text)
    return best[1] if best else None
