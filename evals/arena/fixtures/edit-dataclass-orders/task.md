The order pipeline passes raw dicts around and we keep getting `KeyError`s from typos. Please move it to dataclasses:

- new `orders/model.py` with `@dataclass LineItem(sku: str, qty: int, price_cents: int)` and `@dataclass Order(id: str, customer: str, items: list[LineItem], status: str = "new")`
- `Order.total_cents` is a read-only property (sum of qty x price), no longer a stored field
- `Order.to_dict()` returns exactly the dict shape we use today (including `total_cents`), and `Order.from_dict(d)` builds an Order from that shape (ignoring `total_cents`)
- `parse_orders`, everything in `orders/logic.py` and `export_json` take/return `Order` objects instead of dicts; the logic functions still must not mutate their input
- the JSON written by `export_json` must stay byte-for-byte identical

Update the tests too. Python 3.9 compatible please.
