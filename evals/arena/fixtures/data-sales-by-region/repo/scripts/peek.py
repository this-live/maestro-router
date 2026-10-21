"""Quick look at an export: python3 scripts/peek.py data/sales.csv"""
import sys

sys.path.insert(0, ".")
from analysis.io import read_rows  # noqa: E402

for i, row in enumerate(read_rows(sys.argv[1])):
    print(row)
    if i >= 4:
        break
