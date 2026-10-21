class Buffer:
    """Text buffer with undo/redo. History steps are (pos, removed, inserted)."""

    def __init__(self, text="", max_undo=100):
        self._text = text
        self.max_undo = max_undo
        self._undo = []
        self._redo = []
        self._typing = False  # True while the last step is a coalescable one-char insert

    @property
    def text(self):
        return self._text

    def _check(self, pos, n=0):
        if not (0 <= pos <= len(self._text)) or n < 0 or pos + n > len(self._text):
            raise IndexError("position %d (+%d) out of range for length %d" % (pos, n, len(self._text)))

    def _apply(self, pos, n, s):
        removed = self._text[pos:pos + n]
        self._text = self._text[:pos] + s + self._text[pos + n:]
        return removed

    def _record(self, pos, removed, inserted):
        typing = len(inserted) == 1 and not removed and not inserted.isspace()
        if typing and self._typing and self._undo:
            p, r, ins = self._undo[-1]
            if p + len(ins) == pos:
                self._undo[-1] = (p, r, ins + inserted)
                self._redo.clear()
                return
        self._undo.append((pos, removed, inserted))
        if len(self._undo) > self.max_undo:
            del self._undo[0]
        self._redo.clear()
        self._typing = typing

    def insert(self, pos, s):
        self._check(pos)
        self._apply(pos, 0, s)
        self._record(pos, "", s)

    def delete(self, pos, n):
        """Delete `n` characters at `pos`; returns the deleted text."""
        self._check(pos, n)
        removed = self._apply(pos, n, "")
        self._record(pos, removed, "")
        return removed

    def replace(self, pos, n, s):
        self._check(pos, n)
        removed = self._apply(pos, n, s)
        self._record(pos, removed, s)
        return removed

    def undo(self):
        self._typing = False
        if not self._undo:
            return False
        pos, removed, inserted = self._undo.pop()
        self._apply(pos, len(inserted), removed)
        self._redo.append((pos, removed, inserted))
        return True

    def redo(self):
        self._typing = False
        if not self._redo:
            return False
        pos, removed, inserted = self._redo.pop()
        self._apply(pos, len(removed), inserted)
        self._undo.append((pos, removed, inserted))
        return True
