import json
import os
import stat
import tempfile
import unittest

from helpers import maestro as m


def fake_cli(dirpath, name):
    p = os.path.join(dirpath, name)
    with open(p, "w") as f:
        f.write("#!/bin/sh\nexit 0\n")
    os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR)


class Detect(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = os.path.join(self.tmp.name, "mhome")
        self.user = os.path.join(self.tmp.name, "user")
        self.bin = os.path.join(self.tmp.name, "bin")
        os.makedirs(self.bin)
        os.makedirs(os.path.join(self.user, ".codex"))
        os.makedirs(os.path.join(self.user, ".gemini"))
        fake_cli(self.bin, "codex")
        fake_cli(self.bin, "gemini")
        with open(os.path.join(self.user, ".codex", "auth.json"), "w") as f:
            f.write("{}")
        self.env = {"PATH": self.bin, "HOME": self.user, "MAESTRO_HOME": self.home,
                    "ANTHROPIC_API_KEY": "sk-secret-value", "TYPESAFE_API_KEY": ""}

    def tearDown(self):
        self.tmp.cleanup()

    def run_detect(self, **kw):
        kw.setdefault("probe", lambda ep: None)
        kw.setdefault("hw_probe", lambda: {"ram_gb": 16.0, "vram_gb": None, "apple_silicon": False, "memory_gb": 16.0})
        return m.detect(env=self.env, **kw)

    def test_finds_clis_creds_and_keys(self):
        inv = self.run_detect()
        self.assertEqual(set(inv["harnesses"]), {"codex", "gemini-cli"})
        self.assertEqual(inv["harnesses"]["codex"]["available"], "yes")
        self.assertEqual(inv["harnesses"]["gemini-cli"]["available"], "likely")
        self.assertTrue(inv["api_keys"]["ANTHROPIC_API_KEY"])
        self.assertFalse(inv["api_keys"]["OPENAI_API_KEY"])

    def test_writes_inventory_and_never_stores_secret_values(self):
        self.run_detect()
        with open(os.path.join(self.home, "inventory.json")) as f:
            raw = f.read()
        self.assertNotIn("sk-secret-value", raw)
        self.assertEqual(json.loads(raw)["version"], m.INVENTORY_VERSION)

    def test_write_false_writes_nothing(self):
        self.run_detect(write=False)
        self.assertFalse(os.path.exists(self.home))

    def test_endpoint_probe_is_injectable_and_models_are_rated_by_size(self):
        def probe(ep):
            if ep["url"] != "http://127.0.0.1:11434":
                return None
            return {"kind": "ollama", "latency_ms": 2.0, "models": [{"name": "a:3b", "size": "3.2B"}, {"name": "b:70b", "size": "70.6B"}]}
        inv = self.run_detect(probe=probe)
        self.assertEqual([(x["id"], x["serves"]) for x in inv["local_models"]],
                         [("local:this-machine/a:3b", "local"), ("local:this-machine/b:70b", "standard")])
        self.assertEqual(inv["local_endpoints"], [{"name": "this-machine", "url": "http://127.0.0.1:11434", "kind": "ollama", "models": 2}])
        self.assertTrue(inv["warnings"])  # 70B does not fit in 16 GB
        self.assertEqual(self.run_detect()["local_models"], [])

    def test_probe_endpoint_parses_ollama_tags_with_fake_fetch(self):
        tags = {"models": [{"name": "llama3.1:8b", "details": {"parameter_size": "8.0B"}},
                           {"name": "big:70b", "details": {"parameter_size": "70.6B"}},
                           {"name": "nomic-embed-text", "details": {"parameter_size": "137M"}}]}
        seen = []

        def fetch(url, headers, body, timeout):
            seen.append((url, timeout, body))
            return 200, json.dumps(tags).encode()

        out = m.probe_endpoint({"url": "http://127.0.0.1:11434", "kind": "ollama"}, fetch=fetch)
        self.assertEqual(seen[0], ("http://127.0.0.1:11434/api/tags", 0.3, None))
        self.assertEqual([(x["name"], x["size"]) for x in out["models"]], [("llama3.1:8b", "8.0B"), ("big:70b", "70.6B")])
        self.assertEqual(out["kind"], "ollama")
        self.assertIsNone(m.probe_endpoint({"url": "http://x:1", "kind": "ollama"}, fetch=lambda *a: (500, b"")))

        def boom(*a):
            raise OSError("refused")
        self.assertIsNone(m.probe_endpoint({"url": "http://x:1"}, fetch=boom))

    def test_probe_endpoint_openai_and_auto_kind(self):
        calls = []

        def fetch(url, headers, body, timeout):
            calls.append(url)
            return (404, b"") if url.endswith("/api/tags") else (200, json.dumps({"data": [{"id": "Llama-3.1-8B"}]}).encode())
        out = m.probe_endpoint({"url": "http://h:8000"}, fetch=fetch)
        self.assertEqual((out["kind"], out["models"]), ("openai", [{"name": "Llama-3.1-8B", "size": None, "loaded": True}]))
        self.assertEqual(calls, ["http://h:8000/api/tags", "http://h:8000/v1/models", "http://h:8000/api/v0/models"])

    def test_catalog_places_models_by_output_price_and_needs_key(self):
        cat = {"anthropic": {"env": ["ANTHROPIC_API_KEY"], "models": {
                   "a-big": {"cost": {"output": 75}}, "a-mid": {"cost": {"output": 15}},
                   "a-std": {"cost": {"output": 6}}, "a-cheap": {"cost": {"output": 1}}}},
               "openai": {"env": ["OPENAI_API_KEY"], "models": {"o1": {"cost": {"output": 60}}}},
               "google": {"env": ["GEMINI_API_KEY"], "models": {"g-flash": {"cost": {"output": 2.5}}, "g-pro": {"cost": {"output": 10}}}}}
        self.env["GEMINI_API_KEY"] = "g"
        inv = self.run_detect(catalog=True, fetch=lambda *a: (200, json.dumps(cat).encode()))
        self.assertEqual(set(inv["catalog"]), {"anthropic", "google"})  # no OPENAI key
        self.assertEqual(inv["catalog"]["anthropic"],
                         {"frontier": ["a-big"], "strong": ["a-mid"], "standard": ["a-std"], "fast": ["a-cheap"]})
        self.assertEqual(inv["catalog"]["google"], {"fast": ["g-flash"], "standard": ["g-pro"]})  # non-Claude models are placed by price too

    def test_catalog_failure_is_harmless(self):
        def boom(*a):
            raise OSError("offline")
        inv = self.run_detect(catalog=True, fetch=boom)
        self.assertEqual(inv["catalog"], {})

    def test_hardware_probe_returns_expected_keys(self):
        hw = m.detect_hardware()
        self.assertEqual(set(hw), {"ram_gb", "vram_gb", "apple_silicon", "memory_gb"})


class Install(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.user = self.tmp.name
        self.cfg = m.load_config(os.path.join(self.tmp.name, "nohome"))

    def tearDown(self):
        self.tmp.cleanup()

    def files(self):
        return [os.path.join(d, f) for d, _, fs in os.walk(self.user) for f in fs]

    def test_dry_run_writes_nothing(self):
        acts = m.install("all", dry_run=True, user_home=self.user, cfg=self.cfg)
        self.assertEqual(len(acts), 12)
        self.assertTrue(all(a["action"] == "create" for a in acts))
        self.assertEqual(self.files(), [])

    def test_real_install_writes_tier_agents(self):
        m.install("claude-code", user_home=self.user, cfg=self.cfg)
        path = os.path.join(self.user, ".claude", "agents", "maestro-strong.md")
        with open(path) as f:
            text = f.read()
        self.assertTrue(text.startswith("---\nname: maestro-strong\n"))
        self.assertIn("model: opus", text)
        self.assertIn("maestro-generated", text)
        self.assertEqual(len(self.files()), 4)

    def test_other_harness_paths_and_model_template(self):
        m.install("all", user_home=self.user, cfg=self.cfg)
        oc = os.path.join(self.user, ".config", "opencode", "agents", "maestro-fast.md")
        gm = os.path.join(self.user, ".gemini", "agents", "maestro-fast.md")
        with open(oc) as f:
            text = f.read()
        self.assertIn("mode: subagent", text)
        self.assertIn("model: anthropic/claude-haiku-4-5", text)  # the suggestion, no config needed
        self.assertTrue(os.path.exists(gm))

    def test_configured_model_is_written(self):
        cfg = dict(self.cfg, models={"opencode": {"fast": "anthropic/claude-x"}})
        m.install("opencode", user_home=self.user, cfg=cfg)
        with open(os.path.join(self.user, ".config", "opencode", "agents", "maestro-fast.md")) as f:
            self.assertIn("model: anthropic/claude-x", f.read())

    def test_never_overwrites_foreign_files_but_updates_generated_ones(self):
        m.install("claude-code", user_home=self.user, cfg=self.cfg)
        agents = os.path.join(self.user, ".claude", "agents")
        mine, generated = os.path.join(agents, "maestro-fast.md"), os.path.join(agents, "maestro-strong.md")
        with open(mine, "w") as f:
            f.write("my own agent\n")
        with open(generated, "a") as f:
            f.write("stale edit\n")
        acts = {os.path.basename(a["path"]): a["action"]
                for a in m.install("claude-code", user_home=self.user, cfg=self.cfg)}
        self.assertTrue(acts["maestro-fast.md"].startswith("skip"))
        self.assertEqual(acts["maestro-strong.md"], "update")
        self.assertEqual(acts["maestro-standard.md"], "unchanged")
        with open(mine) as f:
            self.assertEqual(f.read(), "my own agent\n")
        with open(generated) as f:
            self.assertNotIn("stale edit", f.read())

    def test_unknown_harness_errors(self):
        with self.assertRaises(ValueError):
            m.install("vim", user_home=self.user, cfg=self.cfg)


class Cli(unittest.TestCase):
    def test_main_route_and_install_dry_run_use_temp_home(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            old = {k: os.environ.get(k) for k in ("HOME", "MAESTRO_HOME", "CLAUDECODE", "MAESTRO_CLASSIFIER")}
            os.environ.update(HOME=tmp, MAESTRO_HOME=os.path.join(tmp, ".maestro"), CLAUDECODE="1",
                              MAESTRO_CLASSIFIER="rules")  # never hit the network in tests
            try:
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    rc = m.main(["route", "fix", "the", "failing", "test", "in", "auth.py"])
                self.assertEqual(rc, 0)
                self.assertEqual(json.loads(out.getvalue())["harness"], "claude-code")
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    rc = m.main(["install", "--dry-run", "--harness", "all"])
                self.assertEqual(rc, 0)
                self.assertIn("would create", out.getvalue())
                self.assertFalse(os.path.exists(os.path.join(tmp, ".claude")))
            finally:
                for k, v in old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
