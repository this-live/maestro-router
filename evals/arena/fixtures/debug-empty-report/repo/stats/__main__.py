import argparse
import sys

from .report import report


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stats")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="summarize one column of a CSV file")
    r.add_argument("path")
    r.add_argument("--column", default="value")
    args = ap.parse_args(argv)
    if args.cmd == "report":
        print(report(args.path, args.column))
    return 0


if __name__ == "__main__":
    sys.exit(main())
