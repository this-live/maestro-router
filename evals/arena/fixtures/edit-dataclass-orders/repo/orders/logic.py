import copy


def order_total(order):
    return sum(i["qty"] * i["price_cents"] for i in order["items"])


def apply_discount(order, pct):
    """New order with every unit price reduced by `pct` percent (rounded to the cent)."""
    new = copy.deepcopy(order)
    for item in new["items"]:
        item["price_cents"] = round(item["price_cents"] * (100 - pct) / 100)
    new["total_cents"] = order_total(new)
    return new


def mark_shipped(order):
    if order["status"] == "cancelled":
        raise ValueError("cannot ship cancelled order %s" % order["id"])
    new = copy.deepcopy(order)
    new["status"] = "shipped"
    return new


def summarize(orders):
    """{status: total_cents} sorted by status."""
    out = {}
    for o in orders:
        out[o["status"]] = out.get(o["status"], 0) + o["total_cents"]
    return dict(sorted(out.items()))


def top_customers(orders, n=3):
    """[(customer, total_cents)] biggest first, ties by name."""
    totals = {}
    for o in orders:
        totals[o["customer"]] = totals.get(o["customer"], 0) + o["total_cents"]
    return sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
