"""Build dist/mytool.pyz, a self-contained zipapp."""
import os
import shutil
import sys
import tempfile
import zipapp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(out=None):
    out = os.path.abspath(out or os.path.join(ROOT, "dist", "mytool.pyz"))
    staging = tempfile.mkdtemp()
    try:
        dest = os.path.join(staging, "mytool")
        shutil.copytree(os.path.join(ROOT, "mytool"), dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        with open(os.path.join(ROOT, "VERSION"), encoding="utf-8") as f:
            version = f.read().strip()
        with open(os.path.join(dest, "_version.py"), "w", encoding="utf-8") as f:
            f.write("__version__ = %r\n" % version)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        zipapp.create_archive(staging, out, interpreter="/usr/bin/env python3", main="mytool.cli:run")
    finally:
        shutil.rmtree(staging)
    return out


if __name__ == "__main__":
    print("built %s" % os.path.relpath(build(*sys.argv[1:]), ROOT))
