import unittest

from editor.buffer import Buffer
from editor.commands import execute


class BufferTest(unittest.TestCase):
    def test_edits(self):
        b = Buffer("hello")
        b.insert(5, " world")
        self.assertEqual(b.delete(0, 1), "h")
        self.assertEqual(b.replace(0, 4, "J"), "ello")
        self.assertEqual(b.text, "J world")

    def test_bounds(self):
        with self.assertRaises(IndexError):
            Buffer("ab").delete(1, 5)

    def test_commands(self):
        b = Buffer()
        self.assertEqual(execute(b, "i 0 hi there"), "")
        self.assertEqual(execute(b, "s 0 2 yo"), "")
        self.assertEqual(execute(b, "p"), "yo there")
        self.assertTrue(execute(b, "d 0 99").startswith("error"))


if __name__ == "__main__":
    unittest.main()
