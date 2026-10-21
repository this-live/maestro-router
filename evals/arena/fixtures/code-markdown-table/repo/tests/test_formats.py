import unittest

from report.formats import to_csv, to_text


class FormatsTest(unittest.TestCase):
    def test_text(self):
        self.assertEqual(to_text(["a", "bb"], [["1", "2"], ["333", "4"]]), "a    bb\n---  --\n1    2\n333  4")

    def test_csv(self):
        self.assertEqual(to_csv(["a", "b"], [["x,y", "2"]]), 'a,b\n"x,y",2')


if __name__ == "__main__":
    unittest.main()
