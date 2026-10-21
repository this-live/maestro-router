# catalog-api

Read-only product catalog service.

## GET /products

Query parameters (all optional, strings as they come off the query string):

| param      | meaning                                                         |
|------------|-----------------------------------------------------------------|
| `page`     | 1-based page number; missing/invalid/< 1 means 1                |
| `per_page` | items per page, default 20, clamped to 1..100; invalid means 20 |
| `category` | only products in this category                                   |

Response (`api.handlers.list_products(params)`):

```
{"items": [sku, ...], "page": 2, "per_page": 20, "total": 47, "total_pages": 3,
 "has_next": true, "next_page": 3}
```

- Every product appears on exactly one page, in catalog order.
- `total_pages` is at least 1 (an empty result has one empty page).
- A page past the end returns `"items": []`, `has_next: false`, `next_page: null`.
