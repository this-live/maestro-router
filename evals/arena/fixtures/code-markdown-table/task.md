I want to paste these reports into PRs. Add a markdown output: `python3 -m report data/usage.csv --format md`, backed by a `to_markdown(headers, rows)` function in `report/formats.py` next to the other formatters.

Expected format (no column padding; numeric columns right-aligned with `---:`, everything else `:---`; a column is numeric when all its non-empty cells parse as numbers; a literal `|` inside a cell must be escaped as `\|`):

```
| team | region | requests | cost_usd |
| :--- | :--- | ---: | ---: |
| core | eu-west | 1200 | 34.50 |
```
