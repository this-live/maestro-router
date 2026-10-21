import json
import os
import sys
from decimal import Decimal

from .io import read_rows


def summarize_sales(path):
    revenue, orders = {}, {}
    for row in read_rows(path):
        if row.get("status", "").lower() == "cancelled":
            continue
        region = row["region"].strip().upper()
        revenue[region] = revenue.get(region, Decimal("0")) + int(row["qty"]) * Decimal(row["unit_price"])
        orders.setdefault(region, set()).add(row["order_id"])
    q = Decimal("0.01")
    regions = [{"region": r, "orders": len(orders[r]), "revenue": str(revenue[r].quantize(q))}
               for r in sorted(revenue, key=lambda r: (-revenue[r], r))]
    return {"regions": regions, "total_revenue": str(sum(revenue.values(), Decimal("0")).quantize(q))}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    result = summarize_sales(argv[0] if argv else "data/sales.csv")
    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", "sales_by_region.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
