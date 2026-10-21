"""Owner-only files and honest key reporting: maestro.db and its WAL sidecars are 0600, and `detect`/`doctor` list every
provider key that is set (presence only), not just a fixed list of common ones."""
import contextlib
import io
import os
import stat
import tempfile
import types
import unittest

import helpers  # noqa: F401
from helpers import maestro as core

from maestro import catalog as CAT
from maestro import commands as CMD
from maestro import decide as D
from maestro import learn as L
from maestro import providers as P
from maestro import quota as Q
from maestro import server as S


def mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


class DbFiles(unittest.TestCase):
    def test_db_and_wal_sidecars_are_owner_only(self):
        old = os.umask(0o022)   # the common default: new files 0644 unless Maestro says otherwise
        self.addCleanup(os.umask, old)
        with tempfile.TemporaryDirectory() as home:
            db = S.open_db(home)
            db.execute("insert into decisions (ts) values (1)")
            path = os.path.join(home, S.DB_FILE)
            for p in (path, path + "-wal", path + "-shm"):
                self.assertTrue(os.path.exists(p), p)
                self.assertEqual(mode(p), 0o600, p)
            db.close()

    def test_existing_world_readable_sidecars_are_tightened(self):
        with tempfile.TemporaryDirectory() as home:
            path = os.path.join(home, S.DB_FILE)
            S.open_db(home).close()
            for p in (path, path + "-wal", path + "-shm"):
                if os.path.exists(p):
                    os.chmod(p, 0o644)
            db = S.open_db(home)
            for p in (path, path + "-wal", path + "-shm"):
                if os.path.exists(p):
                    self.assertEqual(mode(p), 0o600, p)
            db.close()


class OwnerOnlyState(unittest.TestCase):
    """~/.maestro is created 0700 on every path that makes it, and the JSON state files are 0600 from the moment they exist."""

    def setUp(self):
        old = os.umask(0o022)   # the common default: new directories 0755 and files 0644 unless Maestro says otherwise
        self.addCleanup(os.umask, old)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def fresh(self, name="home"):
        return os.path.join(self.tmp.name, name, ".maestro")   # parent missing too, as on a first run

    def test_ensure_home_creates_0700_and_tightens_an_existing_directory(self):
        home = self.fresh()
        core.ensure_home(home)
        self.assertEqual(mode(home), 0o700)
        os.chmod(home, 0o755)
        core.ensure_home(home)
        self.assertEqual(mode(home), 0o700)
        core.ensure_home(home)   # idempotent

    def test_every_path_that_makes_the_home_makes_it_0700(self):
        ids = {"decision_id": "d1", "chosen": "x", "options": []}
        paths = {
            "_write_json": lambda h: core._write_json(os.path.join(h, "inventory.json"), {"a": 1}),
            "write_models": lambda h: CAT.write_models({"models": []}, h),
            "open_db": lambda h: S.open_db(h).close(),
            "load_admin_token": lambda h: S.load_admin_token(h),
            "keys": lambda h: P._write_keys_lines(["A_KEY=x"], h),
            "log_decision": lambda h: D.log_decision(h, ids, {"job_p": {}, "b": [], "group": None}),
            "learn.record": lambda h: L.record(h, "1", "explicit", 1.0),
            "statusline": lambda h: Q.install_statusline(h),
            "quota": lambda h: Q._save(h),
        }
        for name, make in paths.items():
            home = self.fresh(name)
            make(home)
            self.assertTrue(os.path.isdir(home), name)
            self.assertEqual(mode(home), 0o700, name)

    def test_route_with_logging_tightens_an_existing_home(self):
        home = self.fresh()
        os.makedirs(home, mode=0o755)
        os.chmod(home, 0o755)
        inv = helpers.inventory(clis=("codex",))
        core.route("what is a mutex? explain briefly", harness="claude-code", inventory=inv, config=core.load_config(home),
                   env={"PATH": ""}, home=home, log=True)
        self.assertTrue(os.path.exists(os.path.join(home, "decisions.jsonl")))
        self.assertEqual(mode(home), 0o700)
        self.assertEqual(mode(os.path.join(home, "decisions.jsonl")), 0o600)

    def test_json_state_is_0600_from_the_start(self):
        home = self.fresh()
        for name in ("inventory.json", "config.json", "models.json"):
            self.assertTrue(core._write_json(os.path.join(home, name), {"a": 1}), name)
            self.assertEqual(mode(os.path.join(home, name)), 0o600, name)
        CAT.write_models({"models": [{"id": "m"}]}, home)
        self.assertEqual(mode(CAT.models_path(home)), 0o600)
        # a stale 0644 temp file left by a crash must not leak its mode into the result
        stale = os.path.join(home, "inventory.json.tmp")
        with open(stale, "w") as f:
            f.write("{}")
        os.chmod(stale, 0o644)
        core._write_json(os.path.join(home, "inventory.json"), {"b": 2})
        self.assertEqual(mode(os.path.join(home, "inventory.json")), 0o600)
        core._write_json(os.path.join(home, "config.json"), {"c": 3})   # rewriting a 0644 file leaves it 0600
        os.chmod(os.path.join(home, "config.json"), 0o644)
        core._write_json(os.path.join(home, "config.json"), {"c": 4})
        self.assertEqual(mode(os.path.join(home, "config.json")), 0o600)

    def test_a_json_file_outside_the_home_keeps_its_directory_and_default_mode(self):
        # install --hooks writes ~/.claude/settings.json through _write_json: that is not Maestro's directory
        user = os.path.join(self.tmp.name, "user")
        os.makedirs(os.path.join(user, ".claude"), mode=0o755)
        CMD.install_hooks(user_home=user)
        self.assertEqual(mode(os.path.join(user, ".claude")), 0o755)
        self.assertEqual(mode(os.path.join(user, ".claude", "settings.json")), 0o644)

    def test_an_existing_world_readable_admin_token_is_tightened(self):
        home = self.fresh()
        os.makedirs(home)
        path = os.path.join(home, S.ADMIN_TOKEN_FILE)
        with open(path, "w") as f:
            f.write("tok-from-an-older-version\n")
        os.chmod(path, 0o644)
        self.assertEqual(S.load_admin_token(home), "tok-from-an-older-version")
        self.assertEqual(mode(path), 0o600)

    def test_feedback_does_not_create_the_database(self):
        home = self.fresh()
        old = os.environ.get("MAESTRO_HOME")
        os.environ["MAESTRO_HOME"] = home
        self.addCleanup(lambda: os.environ.__setitem__("MAESTRO_HOME", old) if old is not None else os.environ.pop("MAESTRO_HOME", None))
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = L.cmd_feedback(types.SimpleNamespace(id="1", value="good"))
        self.assertEqual(rc, 1)   # as for any unknown decision
        self.assertFalse(os.path.exists(os.path.join(home, L.DB)))
        self.assertFalse(os.path.exists(home))
        self.assertIn("no decision", err.getvalue())


class KeysLine(unittest.TestCase):
    def test_a_plan_only_key_is_listed(self):
        with tempfile.TemporaryDirectory() as home:
            env = {"HOME": home, "MAESTRO_HOME": home, "KIMI_API_KEY": "x" * 12}
            inv = {"harnesses": {}, "api_keys": {k: False for k in core.API_KEY_VARS}, "local_endpoints": []}
            text = core.summarize(inv, home=home, env=env)
            line = next(l for l in text.splitlines() if l.startswith("api keys present"))
            self.assertIn("KIMI_API_KEY", line)
            self.assertNotIn("none", line)
            self.assertNotIn("x" * 12, text)   # names only, never values

    def test_keys_env_names_are_listed_with_their_source(self):
        with tempfile.TemporaryDirectory() as home:
            with open(os.path.join(home, "keys.env"), "w") as f:
                f.write("GROQ_API_KEY=gsk-not-real\n")   # gitleaks:allow (fake test key)
            os.chmod(os.path.join(home, "keys.env"), 0o600)
            inv = {"harnesses": {}, "api_keys": {}, "local_endpoints": []}
            line = next(l for l in core.summarize(inv, home=home, env={"HOME": home}).splitlines() if l.startswith("api keys present"))
            self.assertIn("GROQ_API_KEY (keys.env)", line)
            self.assertNotIn("gsk-not-real", line)


if __name__ == "__main__":
    unittest.main()


class StatuslineFiles(unittest.TestCase):
    def test_statusline_writes_owner_only_files_and_tightens_the_home(self):
        import json, subprocess
        from maestro import quota as Q
        old = os.umask(0o022)
        self.addCleanup(os.umask, old)
        with tempfile.TemporaryDirectory() as tmp:
            home = os.path.join(tmp, ".maestro")
            os.makedirs(home, mode=0o755)
            os.chmod(home, 0o755)
            for name in ("claude_limits.json", "claude_limits.log"):
                with open(os.path.join(home, name), "w") as f:
                    f.write("{}\n" if name.endswith(".json") else "")
                os.chmod(os.path.join(home, name), 0o644)
            script, _ = Q.install_statusline(home)
            self.assertEqual(mode(script), 0o700)
            os.chmod(home, 0o755)   # as an older Maestro left it
            doc = {"rate_limits": {"five_hour": {"used_percentage": 12, "resets_at": 1}, "seven_day": {"used_percentage": 30}}}
            env = dict(os.environ, MAESTRO_HOME=home)
            r = subprocess.run([script], input=json.dumps(doc), capture_output=True, text=True, env=env, timeout=30)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(mode(home), 0o700)
            for name in ("claude_limits.json", "claude_limits.log"):
                self.assertEqual(mode(os.path.join(home, name)), 0o600, name)
