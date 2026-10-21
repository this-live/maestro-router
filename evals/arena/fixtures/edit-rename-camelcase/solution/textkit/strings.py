import re

SMALL_WORDS = {"a", "an", "and", "as", "at", "but", "by", "for", "in", "of", "on", "or", "the", "to", "vs"}


def normalize_space(text):
    """Collapse runs of whitespace into single spaces and strip the ends."""
    return re.sub(r"\s+", " ", text).strip()


def to_title_case(text):
    """Title-case a headline, keeping small words lowercase except at the start and end."""
    words = normalize_space(text).split(" ")
    out = []
    for i, word in enumerate(words):
        lower = word.lower()
        if 0 < i < len(words) - 1 and lower in SMALL_WORDS:
            out.append(lower)
        else:
            out.append(lower[:1].upper() + lower[1:])
    return " ".join(out)


def count_words(text):
    return len(re.findall(r"[\w']+", text))


def truncate_text(text, limit=80, ellipsis="..."):
    """Cut `text` to at most `limit` characters on a word boundary, adding `ellipsis` if cut."""
    text = normalize_space(text)
    if len(text) <= limit:
        return text
    cut = text[: limit - len(ellipsis)]
    if " " in cut:
        cut = cut[: cut.rindex(" ")]
    return cut.rstrip(" ,;:") + ellipsis
