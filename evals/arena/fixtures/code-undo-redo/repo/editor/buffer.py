class Buffer:
    def __init__(self, text="", max_undo=100):
        self._text = text
        self.max_undo = max_undo

    @property
    def text(self):
        return self._text

    def _check(self, pos, n=0):
        if not (0 <= pos <= len(self._text)) or n < 0 or pos + n > len(self._text):
            raise IndexError("position %d (+%d) out of range for length %d" % (pos, n, len(self._text)))

    def insert(self, pos, s):
        self._check(pos)
        self._text = self._text[:pos] + s + self._text[pos:]

    def delete(self, pos, n):
        """Delete `n` characters at `pos`; returns the deleted text."""
        self._check(pos, n)
        removed = self._text[pos:pos + n]
        self._text = self._text[:pos] + self._text[pos + n:]
        return removed

    def replace(self, pos, n, s):
        removed = self.delete(pos, n)
        self.insert(pos, s)
        return removed
