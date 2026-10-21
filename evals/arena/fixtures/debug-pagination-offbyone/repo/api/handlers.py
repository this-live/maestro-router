from .pagination import paginate
from .store import PRODUCTS

DEFAULT_PER_PAGE = 20
MAX_PER_PAGE = 100


def _int(params, key, default):
    try:
        return int(params.get(key, default))
    except (TypeError, ValueError):
        return default


def list_products(params):
    page = max(1, _int(params, "page", 1))
    per_page = min(_int(params, "per_page", DEFAULT_PER_PAGE), MAX_PER_PAGE)
    items = PRODUCTS
    if params.get("category"):
        items = [p for p in items if p["category"] == params["category"]]
    result = paginate(items, page, per_page)
    result["items"] = [p["sku"] for p in result["items"]]
    return result
