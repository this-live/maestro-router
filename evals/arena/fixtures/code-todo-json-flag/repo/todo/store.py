import json
import os


def default_path():
    return os.environ.get("TODO_FILE") or os.path.expanduser("~/.todo.json")


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(path, items):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
    os.replace(tmp, path)


def add(path, title):
    items = load(path)
    item = {"id": max((i["id"] for i in items), default=0) + 1, "title": title, "done": False}
    items.append(item)
    save(path, items)
    return item


def mark_done(path, item_id):
    items = load(path)
    for item in items:
        if item["id"] == item_id:
            item["done"] = True
            save(path, items)
            return item
    raise KeyError(item_id)


def visible(items, show_all=False):
    return [i for i in items if show_all or not i["done"]]
