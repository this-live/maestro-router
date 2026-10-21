def paginate(items, page=1, per_page=20):
    """Return one page of `items` plus paging info. Pages are numbered from 1."""
    items = list(items)
    total = len(items)
    total_pages = max(1, -(-total // per_page))  # ceil, at least one (possibly empty) page
    start = (page - 1) * per_page
    page_items = items[start:start + per_page]
    has_next = page < total_pages
    return {
        "items": page_items,
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
        "has_next": has_next,
        "next_page": page + 1 if has_next else None,
    }
