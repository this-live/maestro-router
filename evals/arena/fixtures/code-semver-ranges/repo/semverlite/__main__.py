import sys

from .ranges import max_satisfying
from .version import compare


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 3 and argv[0] == "compare":
        print(compare(argv[1], argv[2]))
        return 0
    if len(argv) >= 2 and argv[0] == "resolve":
        best = max_satisfying(argv[2:], argv[1])
        if best is None:
            print("no matching version", file=sys.stderr)
            return 1
        print(best)
        return 0
    print("usage: python3 -m semverlite compare A B | resolve RANGE V...", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
