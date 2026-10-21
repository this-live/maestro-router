# svc

Partner API (framework-free so it can be embedded). `App.handle(method, path, headers)` returns a `Response`.

| route         | auth           |
|---------------|----------------|
| `GET /health` | none           |
| `GET /items`  | `X-Api-Key`    |
