import csv

from .logic import order_total


def parse_orders(path):
    """CSV columns: order_id,customer,sku,qty,price_cents,status (one row per line item)."""
    orders = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            order = orders.get(row["order_id"])
            if order is None:
                order = orders[row["order_id"]] = {
                    "id": row["order_id"],
                    "customer": row["customer"].strip(),
                    "items": [],
                    "status": row.get("status") or "new",
                }
            order["items"].append({"sku": row["sku"], "qty": int(row["qty"]), "price_cents": int(row["price_cents"])})
    for order in orders.values():
        order["total_cents"] = order_total(order)
    return list(orders.values())
