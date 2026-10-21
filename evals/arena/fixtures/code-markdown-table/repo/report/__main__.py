import argparse
import sys

from .formats import FORMATS
from .load import load


def main(argv=None):
    ap = argparse.ArgumentParser(prog="report")
    ap.add_argument("path")
    ap.add_argument("--format", choices=sorted(FORMATS), default="text")
    args = ap.parse_args(argv)
    headers, rows = load(args.path)
    print(FORMATS[args.format](headers, rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
