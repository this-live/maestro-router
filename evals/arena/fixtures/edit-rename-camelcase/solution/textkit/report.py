from .strings import count_words, to_title_case, truncate_text


def summary(title, body, limit=60):
    return {
        "title": to_title_case(title),
        "words": count_words(body),
        "teaser": truncate_text(body, limit),
    }


def render(s):
    return "%s (%d words)\n%s" % (s["title"], s["words"], s["teaser"])
