Support ticket from the storefront team: when they walk `/products` page by page, page 2 starts with the same product page 1 ended on, and the last few products in the catalog never show up at all (with 47 products and `per_page=20` they only ever get 2 pages). Someone also hit a 500 with `?per_page=0`.

The pagination lives in `api/pagination.py` and the handler is `api/handlers.py`. Please fix it so it matches the contract in the README.
