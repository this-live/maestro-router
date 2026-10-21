Review request: CONTACTS-31 adds contact search. It's checked out in this repo. Please review it like you would before approving, and fix anything that's wrong directly in the code (the behaviour we want is what the docstring says; `handle_search` should answer 400 for a bad `limit` instead of blowing up).

```diff
diff --git a/contacts/api.py b/contacts/api.py
--- a/contacts/api.py
+++ b/contacts/api.py
@@ -1,4 +1,4 @@
-from .db import get_contact
+from .db import get_contact, search_contacts
 
 
 def handle_get(conn, params):
@@ -9,3 +9,11 @@
         return 400, {"error": "bad id"}
     contact = get_contact(conn, contact_id)
     return (200, contact) if contact else (404, {"error": "not found"})
+
+
+def handle_search(conn, params):
+    """GET /contacts?q=...&limit=... -> (status, body)"""
+    term = params.get("q", "").strip()
+    if not term:
+        return 400, {"error": "missing q"}
+    return 200, {"results": search_contacts(conn, term, params.get("limit", "20"))}
diff --git a/contacts/db.py b/contacts/db.py
--- a/contacts/db.py
+++ b/contacts/db.py
@@ -25,3 +25,14 @@
 def get_contact(conn, contact_id):
     row = conn.execute("SELECT id, name, email FROM contacts WHERE id = ?", (contact_id,)).fetchone()
     return dict(row) if row else None
+
+
+def search_contacts(conn, term, limit=20):
+    """Contacts whose name or email contains `term` (case-insensitive, literal substring), sorted by name
+    (case-insensitive), at most `limit` (1..100) results."""
+    sql = (
+        "SELECT id, name, email FROM contacts "
+        f"WHERE name LIKE '%{term}%' OR email LIKE '%{term}%' "
+        f"ORDER BY name LIMIT {limit}"
+    )
+    return [dict(row) for row in conn.execute(sql)]
```
