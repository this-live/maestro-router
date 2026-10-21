from .lru import LRUCache, memoize

CACHE = LRUCache(512)
RENDERS = []  # instrumentation: every real render is recorded here


@memoize(CACHE)
def render_thumbnail(path, width, height):
    """Expensive in production (decodes the image); returns None for files we can't decode."""
    RENDERS.append((path, width, height))
    if not path.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
        return None
    return "thumb:%s@%dx%d" % (path, width, height)


def thumbnail_for(path, size="small"):
    w, h = {"small": (64, 64), "medium": (256, 256), "large": (1024, 768)}[size]
    return render_thumbnail(path, w, h)
