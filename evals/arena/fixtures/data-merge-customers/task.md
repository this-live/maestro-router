We need one customer list out of `data/crm.csv` and `data/shop.csv`. Please implement `merge_customers(crm_path, shop_path)` and `write_merged(rows, path)` in `etl/merge.py` (stubs are there).

Rules:
- same customer = same normalized email: trimmed, lowercased, and any `+tag` removed from the part before the `@` (`Ann+promo@X.com` -> `ann@x.com`)
- for each customer take `name` and `phone` from the most recently updated record (`updated_at`, formats vary, see `etl/dates.py`; all UTC). On a tie, the CRM record wins. If that record has an empty phone, use the phone from the most recent record that has one
- `sources`: which files the customer appears in, `crm` and/or `shop`, sorted and joined with `;`
- rows without a usable email (empty or no `@`) are dropped
- result: list of dicts with keys `email, name, phone, sources`, sorted by email
- `write_merged` writes a CSV with exactly that header order, `\n` line endings
