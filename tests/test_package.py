import os
import subprocess
import sys
import tempfile
import unittest

import helpers
from helpers import ROOT

PKG_DIR = os.path.join(ROOT, "skills", "maestro", "scripts", "maestro")


class Package(unittest.TestCase):
    def test_package_reexports_the_router_api(self):
        import maestro
        for name in ("route", "classify_rules", "detect", "load_config", "main", "JOBS", "TIERS", "Classification", "VERSION"):
            self.assertTrue(hasattr(maestro, name), name)
        self.assertIs(maestro.route, helpers.maestro.route)
        self.assertTrue(os.path.exists(os.path.join(PKG_DIR, "maestro_model.json")))

    def run_cli(self, *cmd):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, MAESTRO_HOME=tmp, PATH=os.environ.get("PATH", ""), HOME=tmp)
            env.pop("PYTHONPATH", None)
            return subprocess.run([sys.executable] + list(cmd), capture_output=True, text=True, timeout=60, env=env, cwd=tmp)

    def test_the_package_directory_is_runnable(self):
        res = self.run_cli(PKG_DIR, "route", "--classifier", "rules", "--format", "text", "fix the failing test in auth.py")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertTrue(res.stdout.strip())

    def test_dash_m_maestro_works_from_the_scripts_dir(self):
        res = self.run_cli("-c", "import sys; sys.path.insert(0, %r); sys.argv = ['maestro', '--version']; "
                                 "import runpy; runpy.run_module('maestro', run_name='__main__')" % os.path.dirname(PKG_DIR))
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("maestro-router ", res.stdout)

    def test_only_maestro_router_is_installed(self):   # `maestro` belongs to mobile.dev's Maestro (mobile UI testing)
        with open(os.path.join(ROOT, "pyproject.toml")) as f:
            text = f.read()
        scripts = text.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
        self.assertIn('\nmaestro-router = "maestro:main"', scripts)
        self.assertEqual([l.split("=")[0].strip() for l in scripts.splitlines() if "=" in l and not l.lstrip().startswith("#")],
                         ["maestro-router"])   # no console script named `maestro`
        self.assertNotIn('\nmaestro = ', scripts)
        self.assertIn('name = "maestro-router"', text)

    def test_a_closed_pipe_ends_quietly(self):   # `maestro-router models --json | head`
        res = self.run_cli("-c", "import os, sys, runpy; r, w = os.pipe(); os.close(r); os.dup2(w, 1); "
                                 "sys.argv = ['maestro', 'models', '--json']; runpy.run_path(%r, run_name='__main__')" % PKG_DIR)
        self.assertEqual((res.returncode, res.stderr), (0, ""))


if __name__ == "__main__":
    unittest.main()
