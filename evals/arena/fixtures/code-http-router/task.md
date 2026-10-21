The routes in `web/app.py` already use path parameters but `web/router.py` only does exact matching, so `GET /users/7` is a 404. Please teach the router:

- `<name>` matches one path segment (no `/`) and passes it as a string keyword argument; `<int:name>` matches digits only and passes an int. Any other converter should raise `ValueError` when the route is registered.
- Static routes beat parameterized ones when both match (`/users/me` vs `/users/<int:user_id>`... and `/files/index` vs `/files/<name>`), whatever the registration order.
- If the path matches some route but not for this method, respond 405 with an `Allow` header: the allowed methods, sorted, joined with `, `. Still 404 when no route matches the path at all.
- No trailing-slash magic: `/users/7/` is a 404.
