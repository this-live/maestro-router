I need revenue per region from `data/sales.csv`. Please add `summarize_sales(path)` in a new `analysis/sales.py` (you can use the reader in `analysis/io.py`) that returns:

```json
{"regions": [{"region": "EMEA", "orders": 3, "revenue": "1234.50"}, ...], "total_revenue": "4567.89"}
```

- revenue = sum of `qty * unit_price` as an exact 2-decimal string (no float rounding noise)
- `orders` = number of distinct `order_id`s in that region
- skip rows whose `status` is `cancelled` (any case)
- region names are messy in the export: trim and uppercase them
- sort regions by revenue, highest first, ties by region name

Also make `python3 -m analysis.sales data/sales.csv` write that dict as JSON (indent 2) to `reports/sales_by_region.json`, creating `reports/` if needed.
