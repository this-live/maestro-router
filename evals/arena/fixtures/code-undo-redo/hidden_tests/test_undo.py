import unittest

from editor.buffer import Buffer
from editor.commands import execute


def type_text(b, pos, s):
    for i, ch in enumerate(s):
        b.insert(pos + i, ch)


class Undo(unittest.TestCase):
    def test_each_op(self):
        b = Buffer("abc")
        b.insert(3, "def")
        b.delete(0, 2)
        b.replace(1, 2, "XYZ")
        self.assertEqual(b.text, "cXYZf")
        self.assertTrue(b.undo())
        self.assertEqual(b.text, "cdef")
        self.assertTrue(b.undo())
        self.assertEqual(b.text, "abcdef")
        self.assertTrue(b.undo())
        self.assertEqual(b.text, "abc")
        self.assertFalse(b.undo())
        self.assertEqual(b.text, "abc")

    def test_redo(self):
        b = Buffer()
        b.insert(0, "one")
        b.insert(3, " two")
        b.undo()
        b.undo()
        self.assertTrue(b.redo())
        self.assertEqual(b.text, "one")
        self.assertTrue(b.redo())
        self.assertEqual(b.text, "one two")
        self.assertFalse(b.redo())

    def test_new_edit_clears_redo(self):
        b = Buffer("x")
        b.insert(1, "yz")
        b.undo()
        b.insert(0, "w")
        self.assertFalse(b.redo())
        self.assertEqual(b.text, "wx")
        b.undo()
        self.assertEqual(b.text, "x")

    def test_failed_edit_records_nothing(self):
        b = Buffer("ab")
        b.insert(0, "Z")
        with self.assertRaises(IndexError):
            b.delete(5, 1)
        b.undo()
        self.assertEqual(b.text, "ab")
        self.assertFalse(b.undo())

    def test_typing_coalesces(self):
        b = Buffer()
        type_text(b, 0, "hi there")
        self.assertEqual(b.text, "hi there")
        b.undo()
        self.assertEqual(b.text, "hi ")
        b.undo()
        self.assertEqual(b.text, "hi")
        b.undo()
        self.assertEqual(b.text, "")
        self.assertFalse(b.undo())
        b.redo()
        self.assertEqual(b.text, "hi")
        b.redo()
        b.redo()
        self.assertEqual(b.text, "hi there")

    def test_typing_breaks(self):
        b = Buffer("0123456789")
        type_text(b, 2, "ab")
        type_text(b, 8, "cd")  # not where the last step ended
        b.insert(10, "efg")  # multi-char insert is its own step
        b.insert(13, "h")  # follows a multi-char insert: new step
        self.assertEqual(b.text, "01ab2345cdefgh6789")
        b.undo()
        self.assertEqual(b.text, "01ab2345cdefg6789")
        b.undo()
        self.assertEqual(b.text, "01ab2345cd6789")
        b.undo()
        self.assertEqual(b.text, "01ab23456789")
        b.undo()
        self.assertEqual(b.text, "0123456789")

    def test_undo_ends_group(self):
        b = Buffer()
        type_text(b, 0, "ab")
        b.undo()
        b.redo()
        b.insert(2, "c")
        b.undo()
        self.assertEqual(b.text, "ab")

    def test_delete_after_typing_not_merged(self):
        b = Buffer()
        type_text(b, 0, "abc")
        b.delete(2, 1)
        b.insert(2, "d")
        b.undo()
        self.assertEqual(b.text, "ab")
        b.undo()
        self.assertEqual(b.text, "abc")

    def test_max_undo(self):
        b = Buffer(max_undo=3)
        for i in range(5):
            b.insert(len(b.text), "%d " % i)
        for _ in range(3):
            self.assertTrue(b.undo())
        self.assertFalse(b.undo())
        self.assertEqual(b.text, "0 1 ")
        self.assertEqual(Buffer().max_undo, 100)


class Commands(unittest.TestCase):
    def test_u_r(self):
        b = Buffer()
        self.assertEqual(execute(b, "u"), "nothing to undo")
        self.assertEqual(execute(b, "r"), "nothing to redo")
        execute(b, "i 0 hello")
        execute(b, "s 0 1 J")
        self.assertEqual(execute(b, "u"), "")
        self.assertEqual(execute(b, "p"), "hello")
        self.assertEqual(execute(b, "r"), "")
        self.assertEqual(execute(b, "p"), "Jello")


if __name__ == "__main__":
    unittest.main()
