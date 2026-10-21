import argparse
import sys

from . import __version__
from .commands import greet, maths


def run(argv=None):
    ap = argparse.ArgumentParser(prog="mytool")
    ap.add_argument("--version", action="version", version="mytool %s" % __version__)
    sub = ap.add_subparsers(dest="cmd")
    h = sub.add_parser("hello")
    h.add_argument("name")
    s = sub.add_parser("sum")
    s.add_argument("numbers", nargs="+", type=float)
    args = ap.parse_args(argv)
    if args.cmd == "hello":
        print(greet.hello(args.name))
    elif args.cmd == "sum":
        print(maths.fmt(maths.total(args.numbers)))
    else:
        ap.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(run())
