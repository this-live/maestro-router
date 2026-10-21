import contextlib
import io
import os
import tempfile
import unittest

from todo.__main__ import main


def run(*argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(list(argv))
    return rc, buf.getvalue()


class CliTest(unittest.TestCase):
    def test_add_list_done(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "t.json")
            run("--file", f, "add", "Buy milk")
            run("--file", f, "add", "Call mom")
            run("--file", f, "done", "1")
            self.assertEqual(run("--file", f, "list")[1], "[ ] 2 Call mom\n")
            self.assertEqual(run("--file", f, "list", "--all")[1], "[x] 1 Buy milk\n[ ] 2 Call mom\n")


if __name__ == "__main__":
    unittest.main()
