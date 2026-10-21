# web

Dependency-free request router for our internal tools. `web.app.router.dispatch(method, path)` returns a
`Response(status, body, headers)`; handlers return a body (dict/str) or a full `Response`.
