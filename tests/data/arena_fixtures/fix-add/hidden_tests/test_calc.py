import unittest

import calc


class Add(unittest.TestCase):
    def test_add(self):
        self.assertEqual(calc.add(2, 3), 5)
