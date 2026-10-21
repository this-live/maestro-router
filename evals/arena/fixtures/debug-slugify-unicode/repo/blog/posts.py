from .slug import slugify


def unique_slug(title, taken):
    """Slug for `title` that is not in `taken` (a set, updated in place): foo, foo-2, foo-3, ..."""
    base = slugify(title) or "post"
    slug, n = base, 2
    while slug in taken:
        slug = "%s-%d" % (base, n)
        n += 1
    taken.add(slug)
    return slug


def build_index(posts):
    """posts: list of dicts with a 'title'. Returns [(slug, title), ...] in input order."""
    taken = set()
    return [(unique_slug(p["title"], taken), p["title"]) for p in posts]
