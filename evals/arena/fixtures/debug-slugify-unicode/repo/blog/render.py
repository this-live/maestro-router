import html


def render_index(index):
    items = "\n".join('<li><a href="/%s/">%s</a></li>' % (slug, html.escape(title)) for slug, title in index)
    return "<ul>\n%s\n</ul>" % items
