import os


def _read_version():
    with open(os.path.join(os.path.dirname(__file__), "..", "VERSION"), encoding="utf-8") as f:
        return f.read().strip()


__version__ = _read_version()
