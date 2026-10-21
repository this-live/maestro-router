__version__ = "0+unknown"
try:
    from ._version import __version__  # noqa: F811  (written by scripts/build.py)
except ImportError:
    import os

    _path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "VERSION")
    if os.path.exists(_path):
        with open(_path, encoding="utf-8") as _f:
            __version__ = _f.read().strip()
