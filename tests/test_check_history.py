"""scripts/check-history.sh refuses a history with the private root commit or the old platform/ tree."""
import os
import shutil
import subprocess
import tempfile
import unittest

import helpers  # noqa: F401
from helpers import ROOT

SCRIPT = os.path.join(ROOT, "scripts", "check-history.sh")


@unittest.skipUnless(shutil.which("git"), "git is not installed")
class CheckHistory(unittest.TestCase):
    def repo(self, files):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@example.com")
        git = lambda *a: subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false"] + list(a),
                                        cwd=d, env=env, check=True, capture_output=True, text=True).stdout.strip()
        git("init", "-q", "-b", "main")
        for path in files:
            os.makedirs(os.path.dirname(os.path.join(d, path)) or d, exist_ok=True)
            with open(os.path.join(d, path), "w") as f:
                f.write("x\n")
            git("add", path)
            git("commit", "-q", "-m", path)
        return d, git

    def run_check(self, d, **env):
        return subprocess.run([SCRIPT], cwd=d, capture_output=True, text=True, env=dict(os.environ, **env))

    def test_a_clean_export_passes(self):
        d, _ = self.repo(["README.md"])
        r = self.run_check(d)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("check-history: ok (1 commit", r.stdout)

    def test_the_private_root_commit_is_refused(self):
        d, git = self.repo(["README.md", "src/x.py"])
        root = git("rev-list", "--max-parents=0", "HEAD")
        r = self.run_check(d, MAESTRO_FORBIDDEN_ROOTS=root[:7])
        self.assertEqual(r.returncode, 1)
        self.assertIn("private repository's root", r.stderr)

    def test_the_old_platform_tree_is_refused_even_if_deleted_later(self):
        d, git = self.repo(["README.md", "platform/old.txt"])
        git("rm", "-q", "platform/old.txt")
        git("commit", "-q", "-m", "remove")
        r = self.run_check(d)
        self.assertEqual(r.returncode, 1)
        self.assertIn("platform/", r.stderr)


if __name__ == "__main__":
    unittest.main()
