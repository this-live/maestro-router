import unittest

from app.config import DEFAULTS
from app.server import describe


class ServerTest(unittest.TestCase):
    def test_describe(self):
        self.assertEqual(describe(DEFAULTS), "serving on 127.0.0.1:8000 (workers=2, debug=off, log=info)")


if __name__ == "__main__":
    unittest.main()
