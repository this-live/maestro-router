import argparse
import json
import logging
import sys

from .store import MemoryStore
from .sync import sync


def main(argv=None):
    ap = argparse.ArgumentParser(prog="syncer")
    ap.add_argument("config", help='JSON: {"source": {...}, "dest": {...}, "readonly": [...]}')
    ap.add_argument("--verbose", action="store_true", help="debug logging")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    result = sync(MemoryStore("source", cfg["source"]), MemoryStore("dest", cfg["dest"], cfg.get("readonly", ())))
    print("created=%(created)d updated=%(updated)d failed=%(failed)d" % result)
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
