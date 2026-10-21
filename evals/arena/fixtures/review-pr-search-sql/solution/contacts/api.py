from .db import get_contact, search_contacts


def handle_get(conn, params):
    """GET /contacts/<id> -> (status, body)"""
    try:
        contact_id = int(params.get("id", ""))
    except ValueError:
        return 400, {"error": "bad id"}
    contact = get_contact(conn, contact_id)
    return (200, contact) if contact else (404, {"error": "not found"})


def handle_search(conn, params):
    """GET /contacts?q=...&limit=... -> (status, body)"""
    term = params.get("q", "").strip()
    if not term:
        return 400, {"error": "missing q"}
    try:
        results = search_contacts(conn, term, params.get("limit", "20"))
    except ValueError as e:
        return 400, {"error": str(e)}
    return 200, {"results": results}
