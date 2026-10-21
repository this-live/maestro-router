import os

from .util import env_flag, env_int, env_list

DATABASE_URL = os.environ["DATABASE_URL"]
API_URL = os.environ.get("API_URL", "https://api.example.com")
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
WORKERS = env_int("WORKERS", 4)
ENABLE_BETA = env_flag("ENABLE_BETA")
ALLOWED_ORIGINS = env_list("ALLOWED_ORIGINS", default="*")
SENTRY_DSN = os.getenv("SENTRY_DSN")
