"""npm-style version ranges. See README / docs for the supported syntax."""


def satisfies(version, range_text):
    """True if the version string `version` is inside `range_text`."""
    raise NotImplementedError


def max_satisfying(versions, range_text):
    """The highest version string in `versions` that satisfies `range_text`, or None."""
    raise NotImplementedError
