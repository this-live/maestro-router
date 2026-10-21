import ast
import contextlib
import io
import os
import unittest

from syncer import cli
from syncer.store import MemoryStore
from syncer.sync import sync

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class Logging(unittest.TestCase):
    def run_sync(self, src, dst):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with self.assertLogs("syncer", level="DEBUG") as cm:
                result = sync(src, dst)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(err.getvalue(), "")
        return result, [(r.name, r.levelname, r.getMessage()) for r in cm.records]

    def test_records(self):
        src = MemoryStore("src", {"a": 1, "b": None, "c": 3})
        dst = MemoryStore("dst", {"b": 0}, readonly_keys={"c"})
        result, records = self.run_sync(src, dst)
        self.assertEqual(result, {"created": 1, "updated": 1, "failed": 1})
        self.assertEqual(records, [
            ("syncer.fetch", "DEBUG", "fetching from src"),
            ("syncer.store", "DEBUG", "src lists 3 keys"),
            ("syncer.fetch", "INFO", "fetched 3 items from src"),
            ("syncer.sync", "DEBUG", "created a"),
            ("syncer.store", "WARNING", "writing empty value for b"),
            ("syncer.sync", "DEBUG", "updated b"),
            ("syncer.sync", "ERROR", "could not write c: c is read-only"),
            ("syncer.sync", "INFO", "sync done: 1 created, 1 updated, 1 failed"),
        ])

    def test_empty_source(self):
        _, records = self.run_sync(MemoryStore("void"), MemoryStore("dst"))
        self.assertIn(("syncer.fetch", "WARNING", "void returned no items"), records)

    def test_no_prints_left(self):
        for name in ("store.py", "fetch.py", "sync.py"):
            with open(os.path.join(ROOT, "syncer", name), encoding="utf-8") as f:
                t = ast.parse(f.read())
            calls = [n for n in ast.walk(t) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "print"]
            self.assertEqual(calls, [], "print() left in %s" % name)

    def test_cli_summary(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            with self.assertLogs("syncer", level="DEBUG"):
                rc = cli.main(["--verbose", os.path.join(ROOT, "examples", "demo.json")])
        self.assertEqual(rc, 1)
        self.assertEqual(out.getvalue(), "created=2 updated=1 failed=1\n")


if __name__ == "__main__":
    unittest.main()
