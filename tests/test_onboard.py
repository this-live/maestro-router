"""`maestro-router hermes` (print and install), `maestro-router setup --yes`, and the snippets the UI shows."""
import argparse
import io
import json
import os
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import helpers  # noqa: F401
from helpers import maestro as core

from maestro import onboard as O

BASE = "http://127.0.0.1:4545/v1"


def top_level(text):
    return re.findall(r"(?m)^([A-Za-z_]\w*):", text)


def block(text, key):
    """The lines of one top-level block (a stand-in for a YAML parser, which the stdlib does not have)."""
    m = re.search(r"(?m)^%s:\n((?:[ \t]+.*\n)*)" % key, text)
    return m.group(1) if m else None


class HermesYaml(unittest.TestCase):
    def test_print_has_the_custom_provider_block_the_auxiliary_slots_and_delegation(self):
        y = O.hermes_yaml(BASE)
        self.assertEqual(top_level(y), ["model", "auxiliary", "delegation"])
        m = block(y, "model")
        for line in ("default: maestro/auto", "provider: custom", "base_url: " + BASE, "api_key: maestro", "context_length: 128000"):
            self.assertIn(line, m)
        aux = block(y, "auxiliary")
        for task, model in (("compression", "maestro/fast"), ("title_generation", "maestro/fast"), ("session_search", "maestro/fast"), ("vision", "maestro/auto")):
            self.assertRegex(aux, r"(?s)  %s:\n    provider: custom\n    base_url: %s\n    api_key: maestro\n    model: %s\n" % (task, re.escape(BASE), re.escape(model)))
        self.assertEqual(block(y, "delegation"), "  base_url: %s\n  api_key: maestro\n  model: maestro/auto\n" % BASE)

    def test_a_required_api_key_is_a_placeholder_never_the_key(self):
        y = O.hermes_yaml(BASE, key_required=True)
        self.assertIn("api_key: YOUR_MAESTRO_API_KEY", y)
        self.assertNotIn("api_key: maestro\n", y)

    def test_yaml_is_valid_when_a_parser_is_around(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML is not installed (Maestro does not need it)")
        d = yaml.safe_load(O.hermes_yaml(BASE))
        self.assertEqual(d["model"]["provider"], "custom")
        self.assertEqual(d["auxiliary"]["compression"]["model"], "maestro/fast")
        self.assertEqual(d["delegation"]["model"], "maestro/auto")

    def test_snippets(self):
        self.assertIn("base_url='%s'" % BASE, O.openai_snippet(BASE))
        self.assertIn(BASE + "/chat/completions", O.curl_snippet(BASE))
        self.assertIn("maestro/auto", O.curl_snippet(BASE))


class HermesInstall(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = self._t.name
        self.addCleanup(self._t.cleanup)
        self.env = {"HOME": self.tmp}
        self.path = os.path.join(self.tmp, ".hermes", "config.yaml")

    def read(self):
        with open(self.path) as f:
            return f.read()

    def write(self, text):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            f.write(text)

    def test_creates_the_file_when_there_is_none(self):
        res = O.install_hermes(BASE, self.env)
        self.assertEqual((res["created"], res["added"], res["skipped"], res["backup"]), (True, ["model", "auxiliary", "delegation"], {}, None))
        self.assertEqual(top_level(self.read()), ["model", "auxiliary", "delegation"])

    def test_hermes_home_env_var_moves_the_file(self):
        env = {"HOME": self.tmp, "HERMES_HOME": os.path.join(self.tmp, "elsewhere")}
        O.install_hermes(BASE, env)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "elsewhere", "config.yaml")))

    def test_appends_missing_blocks_after_a_backup_and_never_touches_other_keys(self):
        original = "# my hermes\ntoolsets:\n  - web\n  - terminal\nagent:\n  max_turns: 60\n"
        self.write(original)
        res = O.install_hermes(BASE, self.env)
        text = self.read()
        self.assertTrue(text.startswith(original))   # everything that was there is byte for byte the same
        self.assertEqual(top_level(text), ["toolsets", "agent", "model", "auxiliary", "delegation"])
        with open(res["backup"]) as f:
            self.assertEqual(f.read(), original)
        self.assertIn(".maestro-bak-", res["backup"])

    def test_an_existing_model_block_is_left_alone_and_instructions_are_printed(self):
        original = "model:\n  default: gpt-5\n  provider: openrouter\nagent:\n  x: 1\n"
        self.write(original)
        res = O.install_hermes(BASE, self.env, dry_run=True)
        self.assertEqual(res["skipped"], {"model": "a `model:` block already exists"})
        self.assertEqual(res["added"], ["auxiliary", "delegation"])
        args = argparse.Namespace(install=True, dry_run=False, base_url=BASE, host="127.0.0.1", port=4545, **{"print": False})
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"HOME": self.tmp}), redirect_stdout(buf):
            os.environ.pop("HERMES_HOME", None)
            os.environ.pop("MAESTRO_API_KEY", None)
            os.environ["MAESTRO_HOME"] = self.tmp
            self.assertEqual(O.cmd_hermes(args), 0)
        out = buf.getvalue()
        text = self.read()
        self.assertIn("left `model:` alone", out)
        self.assertIn("default: maestro/auto", out)   # the block to paste
        self.assertTrue(text.startswith(original))
        self.assertEqual(block(text, "model"), "  default: gpt-5\n  provider: openrouter\n")
        self.assertEqual(top_level(text), ["model", "agent", "auxiliary", "delegation"])

    def test_running_it_twice_changes_nothing_the_second_time(self):
        self.write("model:\n  default: maestro/auto\n  base_url: %s\n" % BASE)
        first = self.read()
        res = O.install_hermes(BASE, self.env)
        self.assertEqual(res["skipped"]["model"], "already points at Maestro")
        O.install_hermes(BASE, self.env)
        self.assertEqual(top_level(self.read()).count("auxiliary"), 1)
        self.assertTrue(self.read().startswith(first))
        again = O.install_hermes(BASE, self.env)
        self.assertEqual(again["added"], [])
        self.assertIsNone(again["backup"])

    def test_dry_run_writes_nothing(self):
        res = O.install_hermes(BASE, self.env, dry_run=True)
        self.assertEqual(res["added"], ["model", "auxiliary", "delegation"])
        self.assertFalse(os.path.exists(self.path))

    def test_file_without_a_final_newline(self):
        self.write("agent:\n  max_turns: 3")
        O.install_hermes(BASE, self.env)
        self.assertEqual(top_level(self.read()), ["agent", "model", "auxiliary", "delegation"])
        self.assertIn("max_turns: 3\n", self.read())


class Setup(unittest.TestCase):
    def test_setup_yes_runs_without_prompts_and_writes_the_hermes_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, ".hermes"))
            env = {"HOME": tmp, "MAESTRO_HOME": os.path.join(tmp, ".maestro"), "PATH": tmp}
            buf = io.StringIO()
            with mock.patch.dict(os.environ, env, clear=True), mock.patch("builtins.input", side_effect=AssertionError("must not prompt")), \
                    mock.patch("getpass.getpass", side_effect=AssertionError("must not prompt")), redirect_stdout(buf):
                rc = O.cmd_setup(argparse.Namespace(yes=True))
            out = buf.getvalue()
            self.assertEqual(rc, 0)
            for step in ("1. What is here", "3. Local and network", "4. Discovery", "5. Hermes Agent", "6. Skill", "maestro-router serve --open"):
                self.assertIn(step, out)
            with open(os.path.join(tmp, ".hermes", "config.yaml")) as f:
                self.assertEqual(top_level(f.read()), ["model", "auxiliary", "delegation"])

    def test_setup_without_yes_asks_and_stores_a_typed_key_privately(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"HOME": tmp, "MAESTRO_HOME": os.path.join(tmp, ".maestro"), "PATH": tmp}
            answers = iter(["mistral", "", "", "n", "n", "n"])   # a provider to add, then done; no model server; then no to discovery, hermes, skill
            buf = io.StringIO()
            with mock.patch.dict(os.environ, env, clear=True), mock.patch("builtins.input", lambda prompt="": next(answers)), \
                    mock.patch("getpass.getpass", return_value="mistral-secret-key"), redirect_stdout(buf):
                O.cmd_setup(argparse.Namespace(yes=False))
            with open(os.path.join(tmp, ".maestro", "keys.env")) as f:
                self.assertIn("MISTRAL_API_KEY=mistral-secret-key", f.read())
            self.assertNotIn("mistral-secret-key", buf.getvalue())
            self.assertEqual(oct(os.stat(os.path.join(tmp, ".maestro", "keys.env")).st_mode & 0o777), "0o600")
            self.assertFalse(os.path.exists(os.path.join(tmp, ".hermes")))

    def test_a_network_model_server_is_probed_and_saved_without_a_port_trying_vllm_then_ollama(self):
        with tempfile.TemporaryDirectory() as tmp:
            tried = []

            def probe(ep):
                tried.append(ep["url"])
                return {"kind": "ollama", "models": [{"name": "gpt-oss:120b"}]} if ep["url"].endswith(":11434") else None
            line = O.add_endpoint("spark.local", tmp, probe=probe)
            self.assertEqual(tried, ["http://spark.local:8000", "http://spark.local:11434"])
            self.assertIn("gpt-oss:120b", line)
            O.add_endpoint("http://10.0.0.7:8000", tmp, probe=lambda ep: None)   # off right now: saved anyway
            O.add_endpoint("spark.local:11434", tmp, probe=probe)                  # already there: not duplicated
            with open(os.path.join(tmp, "config.json")) as f:
                eps = json.load(f)["local_endpoints"]
            self.assertEqual(eps, [{"name": "spark", "url": "http://spark.local:11434", "kind": "ollama"},
                                   {"name": "10.0.0.7", "url": "http://10.0.0.7:8000", "kind": None}])


class SetupLocal(unittest.TestCase):
    """Setup's local step: the gauged hardware, what each box serves, and advice only when nothing is served."""
    SPARK = {"name": "spark", "url": None, "kind": "ollama",
             "hardware": {"name": "DGX Spark", "memory_gb": 128, "bandwidth_gbps": 273}}

    def setup(self, **server):
        from test_local import FakeModelServer
        srv = FakeModelServer(**server)
        self.addCleanup(srv.close)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        home = os.path.join(tmp.name, ".maestro")
        os.makedirs(home)
        with open(os.path.join(home, "config.json"), "w") as f:
            json.dump({"local_endpoints": [dict(self.SPARK, url=srv.url)]}, f)
        env = {"HOME": tmp.name, "MAESTRO_HOME": home, "PATH": tmp.name}
        buf = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("builtins.input", side_effect=AssertionError("must not prompt")), \
                redirect_stdout(buf):
            self.assertEqual(O.cmd_setup(argparse.Namespace(yes=True)), 0)
        return buf.getvalue(), srv

    def local_part(self, out):
        return out.split("Hardware and what is served")[1].split("4. Discovery")[0]

    def test_a_box_that_serves_models_is_used_and_nothing_is_suggested(self):
        out, srv = self.setup(models=("gpt-oss:120b", "tiny:3b"), loaded=("gpt-oss:120b",))
        part = self.local_part(out)
        self.assertIn("spark: DGX Spark, 128 GB, 273 GB/s", part)
        self.assertNotIn("(assumed)", part)   # described in the config
        self.assertIn("serves: gpt-oss:120b - using these", part)
        self.assertIn("1 installed but not loaded, ignored (Maestro never loads models)", part)
        self.assertNotIn("advice only", part)
        self.assertNotIn("would fit", part)
        self.assertNotIn("POST", [x[0] for x in srv.seen])   # setup never asks the server to load or run anything

    def test_nothing_served_prints_the_top_three_for_the_strongest_box_as_advice(self):
        out, srv = self.setup(models=("tiny:3b",), loaded=())
        part = self.local_part(out)
        self.assertIn("spark: DGX Spark, 128 GB, 273 GB/s", part)
        self.assertIn("serves: nothing; 1 installed but not loaded, ignored", part)
        self.assertIn("Models that would fit spark (advice only: Maestro will not download or start anything)", part)
        picks = [l for l in part.splitlines() if re.match(r"\s+[123]\. ", l)]
        self.assertEqual(len(picks), 3)
        self.assertFalse(any(re.match(r"\s+4\. ", l) for l in part.splitlines()))
        self.assertNotIn("POST", [x[0] for x in srv.seen])

    def test_an_assumed_spark_is_marked_assumed(self):
        cfg = dict(core.load_config("/nonexistent"), local_endpoints=[{"name": "spark", "url": "http://10.0.0.5:8000"}])
        inv = {"local_endpoints": [], "local_models": [], "hardware": {}}
        with tempfile.TemporaryDirectory() as tmp:
            lines = O.local_report(tmp, cfg, {"PATH": ""}, inv)
        self.assertIn("spark: DGX Spark, 128 GB, 273 GB/s (assumed)", "\n".join(lines))
        self.assertIn("advice only", "\n".join(lines))


class Cli(unittest.TestCase):
    def test_new_subcommands_are_registered(self):
        p = core.build_parser()
        self.assertEqual(p.parse_args(["serve", "--port", "0", "--open"]).port, 0)
        self.assertTrue(p.parse_args(["setup", "--yes"]).yes)
        a = p.parse_args(["hermes", "--install", "--dry-run"])
        self.assertTrue(a.install and a.dry_run)

    def test_hermes_print_via_main(self):
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"MAESTRO_HOME": tempfile.gettempdir()}), redirect_stdout(buf):
            os.environ.pop("MAESTRO_API_KEY", None)
            self.assertEqual(core.main(["hermes", "--print", "--port", "9999"]), 0)
        self.assertIn("base_url: http://127.0.0.1:9999/v1", buf.getvalue())
        json.dumps(buf.getvalue())


if __name__ == "__main__":
    unittest.main()
