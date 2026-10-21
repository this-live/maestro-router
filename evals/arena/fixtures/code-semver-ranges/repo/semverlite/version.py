import functools
import re

_VERSION = re.compile(
    r"^v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)


@functools.total_ordering
class Version:
    """A SemVer 2.0.0 version. Ordering follows the spec (build metadata is ignored)."""

    def __init__(self, major, minor, patch, prerelease=(), build=()):
        self.major, self.minor, self.patch = major, minor, patch
        self.prerelease = tuple(prerelease)
        self.build = tuple(build)

    @classmethod
    def parse(cls, text):
        m = _VERSION.match(text.strip())
        if not m:
            raise ValueError("invalid version: %r" % text)
        pre = tuple(int(p) if p.isdigit() else p for p in m.group(4).split(".")) if m.group(4) else ()
        build = tuple(m.group(5).split(".")) if m.group(5) else ()
        return cls(int(m.group(1)), int(m.group(2)), int(m.group(3)), pre, build)

    @property
    def release(self):
        return (self.major, self.minor, self.patch)

    def _key(self):
        # A release sorts after all of its prereleases; numeric identifiers sort before alphanumeric ones.
        pre = tuple((0, p, "") if isinstance(p, int) else (1, 0, p) for p in self.prerelease)
        return self.release + ((1,) if not self.prerelease else (0,) + pre,)

    def __eq__(self, other):
        return isinstance(other, Version) and self._key() == other._key()

    def __lt__(self, other):
        return self._key() < other._key()

    def __hash__(self):
        return hash(self._key())

    def __str__(self):
        s = "%d.%d.%d" % self.release
        if self.prerelease:
            s += "-" + ".".join(str(p) for p in self.prerelease)
        if self.build:
            s += "+" + ".".join(self.build)
        return s

    def __repr__(self):
        return "Version(%r)" % str(self)


def compare(a, b):
    va, vb = Version.parse(a), Version.parse(b)
    return (va > vb) - (va < vb)
