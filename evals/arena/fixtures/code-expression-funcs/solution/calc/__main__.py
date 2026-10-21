import sys

from .errors import EvalError
from .parser import evaluate


def format_number(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _number(text):
    try:
        return int(text)
    except ValueError:
        return float(text)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: python3 -m calc EXPR [name=value ...]", file=sys.stderr)
        return 2
    env = {}
    for arg in argv[1:]:
        name, sep, value = arg.partition("=")
        try:
            if not sep or not name:
                raise ValueError(arg)
            env[name] = _number(value)
        except ValueError:
            print("error: bad variable %r (want name=number)" % arg, file=sys.stderr)
            return 2
    try:
        print(format_number(evaluate(argv[0], env)))
    except EvalError as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
