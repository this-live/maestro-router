# Changelog

## 0.4.0
- `retry`: accept a `sleep` callable so tests do not have to wait.
- `client.fetch_config` retries `ConnectionError` up to 4 times.

## 0.3.0
- Initial `retry` decorator.
