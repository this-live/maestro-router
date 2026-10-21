"""evals/arena/fixtures: every agentic arena fixture is well formed, fails as shipped and passes with its reference
solution (evals/arena/validate_fixtures.py, one run each, fixtures validated in parallel)."""
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "evals", "arena"))
import validate_fixtures as vf  # noqa: E402


class Structure(unittest.TestCase):
    def test_all_well_formed(self):
        ids = vf.fixture_ids()
        self.assertGreaterEqual(len(ids), 30)
        problems = {i: vf.check_structure(os.path.join(vf.FIXTURES, i)) for i in ids}
        self.assertEqual({i: p for i, p in problems.items() if p}, {})

    def test_mix(self):
        metas = []
        for i in vf.fixture_ids():
            with open(os.path.join(vf.FIXTURES, i, "meta.json"), encoding="utf-8") as f:
                metas.append(json.load(f))
        self.assertEqual({m["job"] for m in metas}, set(vf.JOBS))
        self.assertEqual({m["difficulty_intent"] for m in metas}, {0, 1, 2, 3})
        self.assertEqual(len({m["id"] for m in metas}), len(metas))


class Validation(unittest.TestCase):
    def test_fail_then_pass(self):
        t0 = time.monotonic()
        results = vf.validate_all(runs=1, timeout=60)
        bad = {r["id"]: r["errors"] for r in results if r["errors"]}
        self.assertEqual(bad, {})
        self.assertLess(time.monotonic() - t0, 120)

    def test_detects_broken_fixture(self):
        src = os.path.join(vf.FIXTURES, "debug-empty-report")
        with tempfile.TemporaryDirectory() as d:
            # Reference "solution" that changes nothing: hidden tests must still fail -> validation error.
            dst = os.path.join(d, "debug-empty-report")
            shutil.copytree(src, dst)
            shutil.rmtree(os.path.join(dst, "solution"))
            shutil.copytree(os.path.join(dst, "repo"), os.path.join(dst, "solution"))
            res = vf.validate_fixture(dst, runs=1)
            self.assertTrue(any("FAIL on the solved repo" in e for e in res["errors"]), res)
            # Hidden tests that already pass on the starting repo are rejected too.
            shutil.rmtree(os.path.join(dst, "repo"))
            shutil.copytree(os.path.join(src, "repo"), os.path.join(dst, "repo"))
            shutil.copy(os.path.join(src, "solution", "stats", "report.py"), os.path.join(dst, "repo", "stats"))
            res = vf.validate_fixture(dst, runs=1)
            self.assertTrue(any("PASS on the starting repo" in e for e in res["errors"]), res)


if __name__ == "__main__":
    unittest.main()
