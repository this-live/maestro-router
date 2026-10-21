def render_text(items):
    if not items:
        return "nothing to do"
    return "\n".join("[%s] %d %s" % ("x" if i["done"] else " ", i["id"], i["title"]) for i in items)
