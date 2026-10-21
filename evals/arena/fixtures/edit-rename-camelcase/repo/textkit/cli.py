import sys

from . import strings
from .report import render, summary


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "summary":
        with open(argv[1], encoding="utf-8") as f:
            title, _, body = f.read().partition("\n")
        print(render(summary(title, body)))
        return 0
    if len(argv) == 2 and argv[0] == "title":
        print(strings.toTitleCase(argv[1]))
        return 0
    print("usage: python3 -m textkit.cli summary FILE | title TEXT", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
