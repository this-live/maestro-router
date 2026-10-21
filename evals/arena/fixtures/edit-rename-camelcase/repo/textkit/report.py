from .strings import countWords, toTitleCase, truncateText


def summary(title, body, limit=60):
    return {
        "title": toTitleCase(title),
        "words": countWords(body),
        "teaser": truncateText(body, limit),
    }


def render(s):
    return "%s (%d words)\n%s" % (s["title"], s["words"], s["teaser"])
