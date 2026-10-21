import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS contacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE
)
"""


def connect(path=":memory:"):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def add_contact(conn, name, email):
    cur = conn.execute("INSERT INTO contacts (name, email) VALUES (?, ?)", (name, email))
    conn.commit()
    return cur.lastrowid


def get_contact(conn, contact_id):
    row = conn.execute("SELECT id, name, email FROM contacts WHERE id = ?", (contact_id,)).fetchone()
    return dict(row) if row else None


def _like_escape(term):
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def search_contacts(conn, term, limit=20):
    """Contacts whose name or email contains `term` (case-insensitive, literal substring), sorted by name
    (case-insensitive), at most `limit` (1..100) results."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        raise ValueError("limit must be an integer")
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    pattern = "%" + _like_escape(term) + "%"
    sql = (
        "SELECT id, name, email FROM contacts "
        "WHERE name LIKE ? ESCAPE '\\' OR email LIKE ? ESCAPE '\\' "
        "ORDER BY name COLLATE NOCASE, id LIMIT ?"
    )
    return [dict(row) for row in conn.execute(sql, (pattern, pattern, limit))]
