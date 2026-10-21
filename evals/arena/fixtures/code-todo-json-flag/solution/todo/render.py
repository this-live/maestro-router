def render_text(items):
    if not items:
        return "nothing to do"
    return "\n".join("[%s] %d %s" % ("x" if i["done"] else " ", i["id"], i["title"]) for i in items)


def render_json(items):
    import json

    return json.dumps([{"id": i["id"], "title": i["title"], "done": i["done"]} for i in items], indent=2)
