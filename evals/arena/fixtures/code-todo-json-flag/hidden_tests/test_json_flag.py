import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def todo(f, *argv):
    p = subprocess.run([sys.executable, "-m", "todo", "--file", f] + list(argv), cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
    return p.returncode, p.stdout


class JsonFlag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.f = os.path.join(cls.tmp.name, "todo.json")
        for title in ("Buy milk", "Call mom", "Fix bike \"tyre\""):
            todo(cls.f, "add", title)
        todo(cls.f, "done", "2")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_json_pending(self):
        rc, out = todo(self.f, "list", "--json")
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out), [{"id": 1, "title": "Buy milk", "done": False},
                                           {"id": 3, "title": "Fix bike \"tyre\"", "done": False}])

    def test_json_all(self):
        rc, out = todo(self.f, "list", "--all", "--json")
        self.assertEqual([i["id"] for i in json.loads(out)], [1, 2, 3])
        self.assertEqual(json.loads(out)[1], {"id": 2, "title": "Call mom", "done": True})

    def test_json_empty(self):
        rc, out = todo(os.path.join(self.tmp.name, "none.json"), "list", "--json")
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out), [])

    def test_text_unchanged(self):
        rc, out = todo(self.f, "list")
        self.assertEqual(out, "[ ] 1 Buy milk\n[ ] 3 Fix bike \"tyre\"\n")
        rc, out = todo(os.path.join(self.tmp.name, "none.json"), "list")
        self.assertEqual(out, "nothing to do\n")


if __name__ == "__main__":
    unittest.main()
