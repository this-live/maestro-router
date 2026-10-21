import argparse
import sys

from .export import export_json
from .logic import apply_discount
from .parse import parse_orders


def main(argv=None):
    ap = argparse.ArgumentParser(prog="orders")
    ap.add_argument("csv")
    ap.add_argument("--discount", type=float, default=0)
    args = ap.parse_args(argv)
    orders = parse_orders(args.csv)
    if args.discount:
        orders = [apply_discount(o, args.discount) for o in orders]
    sys.stdout.write(export_json(orders))
    return 0


if __name__ == "__main__":
    sys.exit(main())
