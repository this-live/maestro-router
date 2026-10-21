import csv

from .model import LineItem, Order


def parse_orders(path):
    """CSV columns: order_id,customer,sku,qty,price_cents,status (one row per line item)."""
    orders = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            order = orders.get(row["order_id"])
            if order is None:
                order = orders[row["order_id"]] = Order(row["order_id"], row["customer"].strip(), [],
                                                        row.get("status") or "new")
            order.items.append(LineItem(row["sku"], int(row["qty"]), int(row["price_cents"])))
    return list(orders.values())
