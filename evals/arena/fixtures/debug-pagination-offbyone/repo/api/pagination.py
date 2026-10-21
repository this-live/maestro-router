def paginate(items, page=1, per_page=20):
    """Return one page of `items` plus paging info. Pages are numbered from 1."""
    items = list(items)
    total = len(items)
    total_pages = total // per_page
    first = (page - 1) * per_page  # index of the first item on this page
    last = first + per_page  # index of the last item on this page
    page_items = items[first:last + 1]
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
