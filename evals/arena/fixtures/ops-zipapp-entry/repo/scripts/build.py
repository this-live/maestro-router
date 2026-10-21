"""Build dist/mytool.pyz, a self-contained zipapp."""
import os
import shutil
import sys
import tempfile
import zipapp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(out=None):
    out = out or os.path.join(ROOT, "dist", "mytool.pyz")
    staging = tempfile.mkdtemp()
    try:
        pkg = os.path.join(ROOT, "mytool")
        dest = os.path.join(staging, "mytool")
        os.makedirs(dest)
        for name in os.listdir(pkg):
            if name.endswith(".py"):
                shutil.copy2(os.path.join(pkg, name), dest)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        zipapp.create_archive(staging, out, interpreter="/usr/bin/env python3", main="mytool.cli:main")
    finally:
        shutil.rmtree(staging)
    return out


if __name__ == "__main__":
    print("built %s" % os.path.relpath(build(*sys.argv[1:]), ROOT))
