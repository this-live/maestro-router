"""Validate the arena fixtures in evals/arena/fixtures/ (stdlib only, Python >= 3.9).

For every fixture:
  1. meta.json / task.md / repo/ / hidden_tests/ / solution are well formed;
  2. the starting repo + hidden tests FAIL under meta.test_cmd;
  3. the repo + reference solution + hidden tests PASS;
  4. the visible tests (meta.visible_test_cmd, if any) pass on the solved repo;
  5. each run is repeated (--runs, default 2) and must give the same outcome every time.

Each run happens in a fresh temp copy with a time limit (--timeout, default 60 s). A leading `python3`/`python` in a
command is replaced by the interpreter under test (--python, default this one), so the same fixtures can be checked
on 3.9 and on newer versions.

    python3 evals/arena/validate_fixtures.py                       # all fixtures
    python3 evals/arena/validate_fixtures.py debug-pagination-offbyone --runs 1
    python3 evals/arena/validate_fixtures.py --python /path/to/python3.9
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
JOBS = ("code", "debug", "review", "ops", "edit", "research", "data")
REQUIRED = ("id", "job", "difficulty_intent", "language", "timeout_s", "test_cmd", "needs_tools")
SKIP_NAMES = ("__pycache__", ".pytest_cache")


def fixture_ids(root: str = FIXTURES) -> List[str]:
    return sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "meta.json")))


def _files(path: str) -> List[str]:
    out = []
    for base, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in SKIP_NAMES]
        out.extend(os.path.relpath(os.path.join(base, f), path) for f in files if not f.endswith(".pyc"))
    return sorted(out)


def check_structure(fdir: str) -> List[str]:
    """Static checks; returns a list of problems (empty when the fixture is well formed)."""
    errs = []
    fid = os.path.basename(fdir.rstrip(os.sep))
    try:
        with open(os.path.join(fdir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError) as e:
        return ["meta.json unreadable: %s" % e]
    for k in REQUIRED:
        if k not in meta:
            errs.append("meta.json missing %r" % k)
    if meta.get("id") != fid:
        errs.append("meta.id %r != directory name %r" % (meta.get("id"), fid))
    if meta.get("job") not in JOBS:
        errs.append("meta.job %r not in %s" % (meta.get("job"), JOBS))
    if meta.get("difficulty_intent") not in (0, 1, 2, 3):
        errs.append("meta.difficulty_intent must be 0-3")
    if meta.get("language") != "python":
        errs.append("meta.language must be 'python'")
    t = meta.get("timeout_s")
    if not isinstance(t, int) or not 300 <= t <= 1200:
        errs.append("meta.timeout_s must be an int in 300..1200")
    if meta.get("needs_tools") is not True:
        errs.append("meta.needs_tools must be true")
    task = os.path.join(fdir, "task.md")
    if not os.path.isfile(task) or os.path.getsize(task) < 20:
        errs.append("task.md missing or empty")
    else:
        with open(task, encoding="utf-8") as f:
            text = f.read().lower()
        if "hidden_tests" in text or "hidden test" in text:
            errs.append("task.md must not mention the hidden tests")
    repo = os.path.join(fdir, "repo")
    n = len(_files(repo)) if os.path.isdir(repo) else 0
    if not 3 <= n <= 25:
        errs.append("repo/ must hold 3-25 files (has %d)" % n)
    if os.path.exists(os.path.join(repo, "hidden_tests")):
        errs.append("repo/ must not contain hidden_tests/")
    hidden = os.path.join(fdir, "hidden_tests")
    if not any(f.startswith("test") and f.endswith(".py") for f in (os.listdir(hidden) if os.path.isdir(hidden) else [])):
        errs.append("hidden_tests/ has no test*.py")
    elif not os.path.isfile(os.path.join(hidden, "__init__.py")):
        errs.append("hidden_tests/__init__.py missing (needed by discover -t .)")
    sol_dir, sol_patch = os.path.join(fdir, "solution"), os.path.join(fdir, "solution.patch")
    if not (os.path.isdir(sol_dir) or os.path.isfile(sol_patch)):
        errs.append("no solution/ overlay or solution.patch")
    if os.path.isdir(sol_dir) and any(p.startswith("hidden_tests") for p in _files(sol_dir)):
        errs.append("solution/ must not touch hidden_tests/")
    return errs


def _argv(cmd: str, python: str) -> List[str]:
    argv = shlex.split(cmd)
    if argv and argv[0] in ("python3", "python"):
        argv[0] = python
    return argv


def _run(argv: List[str], cwd: str, timeout: float) -> Dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    t0 = time.monotonic()
    try:
        p = subprocess.run(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
        rc, out = p.returncode, p.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired as e:
        rc, out = None, (e.stdout or b"").decode("utf-8", "replace") + "\n[timeout after %ss]" % timeout
    return {"rc": rc, "out": out[-4000:], "secs": round(time.monotonic() - t0, 2)}


def _apply_solution(fdir: str, dest: str) -> Optional[str]:
    sol_dir, sol_patch = os.path.join(fdir, "solution"), os.path.join(fdir, "solution.patch")
    if os.path.isdir(sol_dir):
        for rel in _files(sol_dir):
            target = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(os.path.join(sol_dir, rel), target)
    if os.path.isfile(sol_patch):
        p = subprocess.run(["git", "apply", "--unsafe-paths", "--directory", ".", sol_patch], cwd=dest,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            return "solution.patch does not apply: " + p.stdout.decode("utf-8", "replace")[-500:]
    return None


def run_case(fdir: str, solved: bool, python: str, timeout: float) -> Dict:
    """One graded run in a fresh temp copy: repo (+ solution) + hidden_tests, then meta.test_cmd."""
    with open(os.path.join(fdir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    tmp = tempfile.mkdtemp(prefix="arena-")
    try:
        work = os.path.join(tmp, "repo")
        shutil.copytree(os.path.join(fdir, "repo"), work, ignore=shutil.ignore_patterns(*SKIP_NAMES))
        res = {}
        if solved:
            err = _apply_solution(fdir, work)
            if err:
                return {"rc": None, "out": err, "secs": 0.0}
            if meta.get("visible_test_cmd"):
                res["visible"] = _run(_argv(meta["visible_test_cmd"], python), work, timeout)
        shutil.copytree(os.path.join(fdir, "hidden_tests"), os.path.join(work, "hidden_tests"),
                        ignore=shutil.ignore_patterns(*SKIP_NAMES))
        res.update(_run(_argv(meta["test_cmd"], python), work, timeout))
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def validate_fixture(fdir: str, python: Optional[str] = None, runs: int = 2, timeout: float = 60.0) -> Dict:
    python = python or sys.executable
    fid = os.path.basename(fdir.rstrip(os.sep))
    errs = check_structure(fdir)
    out = {"id": fid, "errors": errs, "secs": 0.0}
    if errs:
        return out
    t0 = time.monotonic()
    for solved, want in ((False, "fail"), (True, "pass")):
        results = [run_case(fdir, solved, python, timeout) for _ in range(max(1, runs))]
        label = "solved" if solved else "unsolved"
        rcs = [r["rc"] for r in results]
        if any(rc is None for rc in rcs):
            errs.append("%s run timed out or could not run:\n%s" % (label, results[0]["out"][-1500:]))
            continue
        outcomes = {rc == 0 for rc in rcs}
        if len(outcomes) > 1:
            errs.append("%s runs are not deterministic: return codes %s" % (label, rcs))
        if want == "fail" and 0 in rcs:
            errs.append("hidden tests PASS on the starting repo (must fail)")
        if want == "pass" and any(rc != 0 for rc in rcs):
            errs.append("hidden tests FAIL on the solved repo:\n" + results[0]["out"][-1500:])
        if solved:
            for r in results:
                vis = r.get("visible")
                if vis is not None and vis["rc"] != 0:
                    errs.append("visible tests fail on the solved repo:\n" + vis["out"][-1500:])
                    break
    out["secs"] = round(time.monotonic() - t0, 2)
    return out


def validate_all(ids: Optional[List[str]] = None, python: Optional[str] = None, runs: int = 2,
                 timeout: float = 60.0, jobs: int = 0, root: str = FIXTURES) -> List[Dict]:
    ids = ids or fixture_ids(root)
    jobs = jobs or min(8, os.cpu_count() or 2)
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(lambda i: validate_fixture(os.path.join(root, i), python, runs, timeout), ids))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ids", nargs="*", help="fixture ids (default: all)")
    ap.add_argument("--python", default=sys.executable, help="interpreter that runs the tests")
    ap.add_argument("--runs", type=int, default=2, help="repeat each run N times (determinism check)")
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds per run")
    ap.add_argument("--jobs", type=int, default=0, help="parallel fixtures (default: min(8, cpus))")
    ap.add_argument("--json", action="store_true", help="print the results as JSON")
    a = ap.parse_args(argv)
    t0 = time.monotonic()
    results = validate_all(a.ids or None, a.python, a.runs, a.timeout, a.jobs)
    bad = [r for r in results if r["errors"]]
    if a.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            print("%-4s %-34s %6.1fs" % ("ok" if not r["errors"] else "FAIL", r["id"], r["secs"]))
            for e in r["errors"]:
                print("     - " + e.replace("\n", "\n       "))
        print("%d/%d fixtures valid (%s, %d run(s) each, %.1fs)" % (len(results) - len(bad), len(results),
                                                                    a.python, a.runs, time.monotonic() - t0))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
