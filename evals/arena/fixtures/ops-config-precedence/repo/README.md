# app

## Configuration

Precedence (lowest to highest): built-in defaults, config file (JSON), `APP_*` environment variables, CLI flags.

| key             | default         | env                | flag                       |
|-----------------|-----------------|--------------------|----------------------------|
| `host`          | `127.0.0.1`     | `APP_HOST`         | `--host`                   |
| `port`          | `8000`          | `APP_PORT`         | `--port`                   |
| `debug`         | `false`         | `APP_DEBUG`        | `--debug` / `--no-debug`   |
| `workers`       | `2`             | `APP_WORKERS`      | `--workers`                |
| `allowed_hosts` | `["localhost"]` | `APP_ALLOWED_HOSTS`| (none)                     |
| `log_level`     | `info`          | `APP_LOG_LEVEL`    | `--log-level`              |

Config file: `--config PATH`, else `$APP_CONFIG`, else `./app.json` when present.

    python3 -m app.server --config app.json --port 9000
