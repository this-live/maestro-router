import os

_SMTP_DEFAULTS = {"SMTP_HOST": "localhost", "SMTP_PORT": "25"}


def smtp_settings():
    """SMTP connection settings. (LEGACY_SMTP_PASSWORD is no longer supported, use SMTP_PASSWORD_FILE.)"""
    return {name: os.environ.get(name, default) for name, default in _SMTP_DEFAULTS.items()}


def smtp_password():
    path = os.environ.get("SMTP_PASSWORD_FILE")
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return f.read().strip()
