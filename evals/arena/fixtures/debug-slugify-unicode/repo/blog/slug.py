import re


def slugify(text, max_length=60):
    """URL slug for a post title: lowercase ASCII letters, digits and single hyphens."""
    slug = re.sub(r"[^a-z0-9]", "-", text.lower())
    slug = slug[:max_length]
    return slug.strip("-")
