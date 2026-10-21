import re

from .errors import EvalError

_TOKEN = re.compile(r"\s*(?:(\d+\.\d*|\.\d+|\d+)|([A-Za-z_][A-Za-z0-9_]*)|(\S))")
OPS = set("+-*/()^,")


def tokenize(text):
    """List of (kind, value, position) tuples ending with ("EOF", None, len). Kinds: NUM, NAME, OP, EOF."""
    tokens = []
    text = text.rstrip()
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        num, name, op = m.groups()
        if num:
            tokens.append(("NUM", float(num) if "." in num else int(num), m.start(1)))
        elif name:
            tokens.append(("NAME", name, m.start(2)))
        elif op in OPS:
            tokens.append(("OP", op, m.start(3)))
        else:
            raise EvalError("unexpected character %r at %d" % (op, m.start(3)))
        pos = m.end()
    tokens.append(("EOF", None, len(text)))
    return tokens
