import re
import unicodedata

_SPECIAL = {"ß": "ss", "æ": "ae", "ø": "o"}


def slugify(text, max_length=60):
    """URL slug for a post title: lowercase ASCII letters, digits and single hyphens."""
    text = "".join(_SPECIAL.get(c, c) for c in text.lower())
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return slug[:max_length].rstrip("-")
