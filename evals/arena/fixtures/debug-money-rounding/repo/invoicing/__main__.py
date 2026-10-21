import argparse
import csv

from .invoice import Invoice


def main(argv=None):
    ap = argparse.ArgumentParser(prog="invoicing")
    ap.add_argument("csv", help="description,qty,unit_price")
    ap.add_argument("--tax-rate", default="0")
    ap.add_argument("--discount-pct", default="0")
    a = ap.parse_args(argv)
    inv = Invoice(a.tax_rate, a.discount_pct)
    with open(a.csv, newline="") as f:
        for row in csv.DictReader(f):
            inv.add_line(row["description"], row["qty"], row["unit_price"])
    for line, amount in zip(inv.lines, inv.line_totals()):
        print("%-20s %4d x %8s = %10s" % (line["description"], line["qty"], line["unit_price"], amount))
    for k, v in inv.totals().items():
        print("%39s %10s" % (k, v))


if __name__ == "__main__":
    main()
