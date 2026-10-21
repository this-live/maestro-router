import argparse
import sys

from . import store
from .render import render_json, render_text


def main(argv=None):
    ap = argparse.ArgumentParser(prog="todo")
    ap.add_argument("--file", default=None, help="todo file (default $TODO_FILE or ~/.todo.json)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("title")
    ls = sub.add_parser("list")
    ls.add_argument("--all", action="store_true", help="include done items")
    ls.add_argument("--json", action="store_true", help="print the items as a JSON array")
    d = sub.add_parser("done")
    d.add_argument("id", type=int)
    args = ap.parse_args(argv)
    path = args.file or store.default_path()

    if args.cmd == "add":
        item = store.add(path, args.title)
        print("added %d" % item["id"])
    elif args.cmd == "list":
        items = store.visible(store.load(path), args.all)
        print(render_json(items) if args.json else render_text(items))
    elif args.cmd == "done":
        try:
            store.mark_done(path, args.id)
        except KeyError:
            print("no such item: %d" % args.id, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
