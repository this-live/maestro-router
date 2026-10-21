import contextlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import helpers
from helpers import ROOT, inventory, local, maestro as core
from fake_http import FakeServer, fake_cli, fake_spec, json_reply

from maestro import providers as P

SECRET = "sk-test-SECRET-1234567890"   # gitleaks:allow (fake test key)


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.home = os.path.join(self.tmp, ".maestro")
        self.bin = os.path.join(self.tmp, "bin")
        os.makedirs(self.bin)
        self.addCleanup(self._tmp.cleanup)

    def env(self, **kw):
        return dict({"HOME": self.tmp, "MAESTRO_HOME": self.home, "PATH": self.bin}, **kw)


class Registry(unittest.TestCase):
    def test_the_whole_models_dev_landscape_is_in(self):
        self.assertGreaterEqual(len(P.REGISTRY), 220)
        self.assertEqual(len(P.BY_ID), len(P.REGISTRY))
        for s in P.REGISTRY:
            self.assertIn(s["style"], ("openai", "anthropic", "unsupported"), s["id"])
            self.assertTrue(s["need"] and all(isinstance(g, list) for g in s["need"]), s["id"])
            self.assertEqual(s["callable"], s["style"] != "unsupported", s["id"])
            if s["callable"]:
                self.assertTrue(s["base_url"], s["id"])
        size = os.path.getsize(P.DATA_FILE)
        self.assertLess(size, 150 * 1024)

    def test_providers_without_an_api_field_get_their_base_urls(self):
        want = {
            "openai": "https://api.openai.com/v1", "anthropic": "https://api.anthropic.com/v1",
            "google": "https://generativelanguage.googleapis.com/v1beta/openai", "xai": "https://api.x.ai/v1",
            "mistral": "https://api.mistral.ai/v1", "groq": "https://api.groq.com/openai/v1", "cerebras": "https://api.cerebras.ai/v1",
            "togetherai": "https://api.together.xyz/v1", "deepinfra": "https://api.deepinfra.com/v1/openai",
            "cohere": "https://api.cohere.ai/compatibility/v1", "perplexity": "https://api.perplexity.ai",
            "vercel": "https://ai-gateway.vercel.sh/v1", "openrouter": "https://openrouter.ai/api/v1",
            "fireworks-ai": "https://api.fireworks.ai/inference/v1", "moonshotai": "https://api.moonshot.ai/v1",
            "zai": "https://api.z.ai/api/paas/v4", "nvidia": "https://integrate.api.nvidia.com/v1",
            "huggingface": "https://router.huggingface.co/v1", "ollama-cloud": "https://ollama.com/v1",
            "alibaba": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"}
        for pid, url in want.items():
            self.assertEqual(P.BY_ID[pid]["base_url"], url, pid)
            self.assertTrue(P.BY_ID[pid]["callable"], pid)
        self.assertEqual(P.BY_ID["anthropic"]["style"], "anthropic")
        self.assertEqual(P.BY_ID["deepseek"]["env"], ("DEEPSEEK_API_KEY",))
        self.assertEqual((P.BY_ID["groq"]["quota"], P.BY_ID["openrouter"]["quota"], P.BY_ID["deepseek"]["quota"]), (None, "openrouter", "deepseek"))

    def test_style_follows_the_npm_package(self):
        mm = P.BY_ID["minimax"]
        self.assertEqual((mm["style"], mm["base_url"], mm["env"]), ("anthropic", "https://api.minimax.io/anthropic/v1", ("MINIMAX_API_KEY",)))
        self.assertEqual(P.BY_ID["302ai"]["style"], "openai")  # @ai-sdk/openai-compatible
        for pid in ("amazon-bedrock", "google-vertex", "google-vertex-anthropic", "cloudflare-ai-gateway", "watsonx", "gitlab"):
            self.assertEqual((P.BY_ID[pid]["style"], P.BY_ID[pid]["callable"]), ("unsupported", False), pid)
        self.assertIn("not yet callable", P.BY_ID["amazon-bedrock"]["notes"])

    def test_multi_variable_providers_need_all_their_parts(self):
        az = P.BY_ID["azure"]
        self.assertEqual((az["env"], az["params"], az["need"]), (("AZURE_API_KEY",), ("AZURE_RESOURCE_NAME",), [["AZURE_API_KEY", "AZURE_RESOURCE_NAME"]]))
        self.assertEqual(az["auth"], "api-key")
        cf = P.BY_ID["cloudflare-workers-ai"]  # ${CLOUDFLARE_ACCOUNT_ID} in the URL makes it a required setting
        self.assertEqual((cf["env"], cf["params"]), (("CLOUDFLARE_API_KEY",), ("CLOUDFLARE_ACCOUNT_ID",)))
        self.assertEqual(P.key_status(cf, {"CLOUDFLARE_API_KEY": "k"}, "/nonexistent"), ("CLOUDFLARE_API_KEY", "env", ["CLOUDFLARE_ACCOUNT_ID"]))
        self.assertEqual(P.key_status(cf, {"CLOUDFLARE_API_KEY": "k", "CLOUDFLARE_ACCOUNT_ID": "acct"}, "/nonexistent"), ("CLOUDFLARE_API_KEY", "env", []))
        self.assertEqual(P.key_status(cf, {"CLOUDFLARE_ACCOUNT_ID": "acct"}, "/nonexistent")[0], "CLOUDFLARE_ACCOUNT_ID")  # partial: not "configured"
        bed = P.BY_ID["amazon-bedrock"]
        self.assertEqual(P.key_status(bed, {"AWS_BEARER_TOKEN_BEDROCK": "t"}, "/nonexistent")[2], [])
        self.assertEqual(P.key_status(bed, {"AWS_ACCESS_KEY_ID": "a", "AWS_SECRET_ACCESS_KEY": "b"}, "/nonexistent")[2], [])
        self.assertEqual(P.key_status(bed, {"AWS_ACCESS_KEY_ID": "a"}, "/nonexistent")[2], ["AWS_SECRET_ACCESS_KEY"])
        self.assertEqual(P.key_status(P.BY_ID["google-vertex"], {"GOOGLE_VERTEX_PROJECT": "p", "GOOGLE_VERTEX_LOCATION": "l"}, "/nonexistent")[2], [])

    def test_any_one_env_var_of_a_plain_provider_is_enough(self):
        g = P.BY_ID["google"]
        for var in ("GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY", "GEMINI_API_KEY"):
            self.assertEqual(P.key_status(g, {var: "k"}, "/nonexistent"), (var, "env", []), var)
        self.assertEqual(P.key_status(g, {}, "/nonexistent"), (None, None, []))

    def test_templated_base_urls_fill_from_the_environment(self):
        cf = P.BY_ID["cloudflare-workers-ai"]
        self.assertIsNone(P.base_url(cf, {}, "/nonexistent"))
        self.assertEqual(P.base_url(cf, {"CLOUDFLARE_ACCOUNT_ID": "acct1"}, "/nonexistent"),
                         "https://api.cloudflare.com/client/v4/accounts/acct1/ai/v1")
        self.assertEqual(P.base_url(P.BY_ID["azure"], {"AZURE_RESOURCE_NAME": "myres"}, "/nonexistent"), "https://myres.openai.azure.com/openai/v1")

    def test_per_model_overrides_change_style_and_base(self):
        oc = P.BY_ID["opencode"]
        self.assertEqual(oc["style"], "openai")
        claude = P.model_spec(oc, "claude-fable-5")
        self.assertEqual(claude["style"], "anthropic")
        self.assertEqual(P.model_spec(oc, "some-other-model")["style"], "openai")
        self.assertIs(P.model_spec(oc, "some-other-model"), oc)
        gem = P.model_spec(oc, "gemini-3-flash")  # a native Google API: not something the clients speak
        self.assertEqual((gem["style"], gem["callable"]), ("unsupported", False))
        az = P.model_spec(P.BY_ID["azure"], "claude-fable-5")
        self.assertEqual((az["style"], az["base_url"]), ("anthropic", "https://${AZURE_RESOURCE_NAME}.services.ai.azure.com/anthropic/v1"))
        bed = P.model_spec(P.BY_ID["amazon-bedrock"], "openai.gpt-oss-120b")
        self.assertIn("Responses API", bed["notes"])
        zen = P.BY_ID["zenmux"]
        self.assertEqual(P.model_spec(zen, "anthropic/claude-3.5-haiku")["base_url"], "https://zenmux.ai/api/anthropic/v1")

    def test_cli_model_tables_come_from_the_suggestions(self):
        self.assertEqual(P.cli_models("claude"), ["haiku", "sonnet", "opus", "fable"])
        self.assertIn("gpt-6.1-sol", P.cli_models("codex"))
        self.assertEqual(len(P.cli_models("codex")), len(set(P.cli_models("codex"))))
        self.assertIn("gemini-3.1-pro-preview", P.cli_models("gemini"))
        self.assertEqual(P.cli_models("copilot"), [])


class Keys(Tmp):
    def test_env_beats_keys_file_and_values_stay_out_of_presence_checks(self):
        P.set_key("OPENAI_API_KEY", "from-file", self.home)
        self.assertEqual(P.find_key(("OPENAI_API_KEY",), {}, self.home), ("OPENAI_API_KEY", "keys.env"))
        self.assertEqual(P.find_key(("OPENAI_API_KEY",), {"OPENAI_API_KEY": "x"}, self.home), ("OPENAI_API_KEY", "env"))
        self.assertEqual(P.get_key(("OPENAI_API_KEY",), {"OPENAI_API_KEY": "from-env"}, self.home), "from-env")
        self.assertEqual(P.get_key(("OPENAI_API_KEY",), {}, self.home), "from-file")
        self.assertIsNone(P.get_key(("GROQ_API_KEY",), {}, self.home))
        self.assertEqual(P.find_key(("A", "B"), {"B": "1"}, self.home), ("B", "env"))
        self.assertEqual(P.find_key(("A",), {"A": ""}, self.home), (None, None))  # empty = unset

    def test_keys_file_is_0600_and_edits_keep_other_lines(self):
        os.makedirs(self.home)
        write(P.keys_path(self.home), "# mine\nexport GROQ_API_KEY='quoted'\nOTHER=1\n")
        P.set_key("OPENAI_API_KEY", SECRET, self.home)
        P.set_key("OPENAI_API_KEY", SECRET + "2", self.home)  # replaces, does not duplicate
        text = read(P.keys_path(self.home))
        self.assertEqual(text.count("OPENAI_API_KEY="), 1)
        self.assertIn("# mine", text)
        self.assertEqual(P.read_keys_file(self.home), {"GROQ_API_KEY": "quoted", "OTHER": "1", "OPENAI_API_KEY": SECRET + "2"})
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(os.stat(P.keys_path(self.home)).st_mode), 0o600)
        self.assertEqual(P.unset_key(["GROQ_API_KEY", "NOPE"], self.home), ["GROQ_API_KEY"])
        self.assertNotIn("GROQ_API_KEY", P.read_keys_file(self.home))
        self.assertIn("# mine", read(P.keys_path(self.home)))

    def test_bad_names_and_values_are_refused(self):
        for name, val in (("not a name", "x"), ("OPENAI_API_KEY", ""), ("OPENAI_API_KEY", "a\nb=c")):
            with self.assertRaises(ValueError):
                P.set_key(name, val, self.home)
        self.assertFalse(os.path.exists(P.keys_path(self.home)))

    @unittest.skipUnless(os.name == "posix", "POSIX permissions")
    def test_a_world_readable_keys_file_is_flagged(self):
        P.set_key("OPENAI_API_KEY", SECRET, self.home)
        self.assertEqual(P.keys_file_problem(self.home), "")
        os.chmod(P.keys_path(self.home), 0o644)
        self.assertIn("chmod 600", P.keys_file_problem(self.home))

    def test_key_env_name(self):
        self.assertEqual(P.key_env_name("openai"), "OPENAI_API_KEY")
        self.assertEqual(P.key_env_name("google"), "GOOGLE_API_KEY")
        self.assertEqual(P.key_env_name("azure"), "AZURE_API_KEY")
        self.assertEqual(P.key_env_name("MY_CUSTOM_KEY"), "MY_CUSTOM_KEY")
        with self.assertRaises(ValueError):
            P.key_env_name("nope")

    def cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, self.env(), clear=True), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = core.main(list(argv))
        return rc, out.getvalue(), err.getvalue()

    def test_keys_cli_set_list_unset_never_shows_a_value(self):
        rc, out, err = self.cli("keys", "set", "openai", SECRET)
        self.assertEqual(rc, 0)
        self.assertNotIn(SECRET, out + err)
        self.assertIn("shell history", err)
        rc, out, _ = self.cli("keys", "list")
        self.assertEqual(rc, 0)
        self.assertNotIn(SECRET, out)
        line = next(l for l in out.splitlines() if l.startswith("openai "))
        self.assertIn("set (keys.env)", line)
        self.assertNotIn("groq", out)  # only providers with a key are listed unless --all
        self.assertIn("providers have a key", out)
        rc, out, _ = self.cli("keys", "list", "--all")
        self.assertIn("missing", next(l for l in out.splitlines() if l.startswith("groq ")))
        rc, out, _ = self.cli("keys", "list", "groq")
        self.assertIn("missing", out)
        rc, out, _ = self.cli("keys", "unset", "openai")
        self.assertEqual(rc, 0)
        self.assertEqual(P.read_keys_file(self.home), {})

    def test_keys_set_prompts_without_echo_when_the_value_is_omitted(self):
        with mock.patch("getpass.getpass", return_value=SECRET) as gp:
            rc, out, err = self.cli("keys", "set", "groq")
        self.assertEqual(rc, 0)
        gp.assert_called_once()
        self.assertEqual(P.read_keys_file(self.home), {"GROQ_API_KEY": SECRET})
        self.assertNotIn(SECRET, out + err)

    def test_keys_cli_rejects_unknown_provider_and_missing_name(self):
        self.assertEqual(self.cli("keys", "set", "nope", "x")[0], 2)
        self.assertEqual(self.cli("keys", "unset")[0], 2)


class ListModels(Tmp):
    def spec(self, url, style="openai", **kw):
        return fake_spec(url, style, **kw)

    def test_openai_style_lists_chat_models_with_a_bearer_key(self):
        with FakeServer(lambda r: json_reply({"data": [{"id": "gpt-x"}, {"id": "text-embedding-3"}, {"id": "gpt-y"}]})) as srv:
            ids = P.list_models(self.spec(srv.url), self.env(FAKE_KEY=SECRET), self.home)
            self.assertEqual(ids, ["gpt-x", "gpt-y"])
            self.assertEqual(P.list_models(self.spec(srv.url), self.env(FAKE_KEY=SECRET), self.home, chat_only=False),
                             ["gpt-x", "gpt-y", "text-embedding-3"])
            req = srv.requests[0]
            self.assertEqual((req["method"], req["path"]), ("GET", "/models"))
            self.assertEqual(req["headers"]["authorization"], "Bearer " + SECRET)
            self.assertIn("maestro", req["headers"]["user-agent"])

    def test_key_from_keys_file_is_used(self):
        P.set_key("FAKE_KEY", SECRET, self.home)
        with FakeServer(lambda r: json_reply({"data": [{"id": "m"}]})) as srv:
            self.assertEqual(P.list_models(self.spec(srv.url), self.env(), self.home), ["m"])
            self.assertEqual(srv.requests[0]["headers"]["authorization"], "Bearer " + SECRET)

    def test_google_prefix_is_stripped_and_a_bare_list_is_accepted(self):
        with FakeServer(lambda r: json_reply({"data": [{"id": "models/gemini-x"}]})) as srv:
            self.assertEqual(P.list_models(self.spec(srv.url), self.env(FAKE_KEY="k"), self.home), ["gemini-x"])
        with FakeServer(lambda r: json_reply([{"id": "meta/llama"}, {"id": "other"}])) as srv:  # Together returns a bare list
            self.assertEqual(P.list_models(self.spec(srv.url), self.env(FAKE_KEY="k"), self.home), ["meta/llama", "other"])

    def test_anthropic_style_uses_x_api_key_version_and_pagination(self):
        pages = {0: {"data": [{"id": "claude-a"}, {"id": "claude-b"}], "has_more": True, "last_id": "claude-b"},
                 1: {"data": [{"id": "claude-c"}], "has_more": False, "last_id": "claude-c"}}

        def handler(r):
            return json_reply(pages[1 if "after_id=claude-b" in r["path"] else 0])
        with FakeServer(handler) as srv:
            ids = P.list_models(self.spec(srv.url, "anthropic"), self.env(FAKE_KEY=SECRET), self.home)
            self.assertEqual(ids, ["claude-a", "claude-b", "claude-c"])
            h = srv.requests[0]["headers"]
            self.assertEqual(h["x-api-key"], SECRET)
            self.assertEqual(h["anthropic-version"], "2023-06-01")
            self.assertNotIn("authorization", h)
            self.assertIn("limit=1000", srv.requests[0]["path"])
            self.assertEqual(len(srv.requests), 2)

    def test_azure_uses_the_api_key_header_and_config_models(self):
        spec = P.BY_ID["azure"]
        env = self.env(AZURE_API_KEY=SECRET, AZURE_RESOURCE_NAME="myres")
        self.assertEqual(P.base_url(spec, env, self.home), "https://myres.openai.azure.com/openai/v1")
        self.assertEqual(P.auth_headers(spec, SECRET)["api-key"], SECRET)
        with self.assertRaises(P.ProviderError) as cm:
            P.list_models("azure", env, self.home)
        self.assertIn("providers.azure.models", str(cm.exception))
        cfg = dict(core.load_config(self.home), providers={"azure": {"models": ["gpt-5", "o4"]}})
        self.assertEqual(P.list_models("azure", env, self.home, cfg=cfg), ["gpt-5", "o4"])
        with self.assertRaises(P.ProviderError) as cm:
            P.list_models("azure", self.env(AZURE_API_KEY=SECRET), self.home, cfg=cfg)
        self.assertIn("AZURE_RESOURCE_NAME", str(cm.exception))

    def test_errors_are_captured_and_never_contain_the_key(self):
        def handler(r):
            return json_reply({"error": {"message": "Incorrect API key provided: " + SECRET}}, 401)
        with FakeServer(handler) as srv:
            with self.assertRaises(P.ProviderError) as cm:
                P.list_models(self.spec(srv.url), self.env(FAKE_KEY=SECRET), self.home)
        self.assertIn("HTTP 401", str(cm.exception))
        self.assertNotIn(SECRET, str(cm.exception))

    def test_no_key_unreachable_slow_and_not_callable(self):
        with self.assertRaises(P.ProviderError) as cm:
            P.list_models(self.spec("http://127.0.0.1:9"), self.env(), self.home)
        self.assertIn("no API key", str(cm.exception))
        with self.assertRaises(P.ProviderError):
            P.list_models(self.spec("http://127.0.0.1:9"), self.env(FAKE_KEY="k"), self.home, timeout=1)
        with FakeServer(lambda r: (time.sleep(1.5), json_reply({"data": []}))[1]) as srv:
            t0 = time.time()
            with self.assertRaises(P.ProviderError):
                P.list_models(self.spec(srv.url), self.env(FAKE_KEY="k"), self.home, timeout=0.3)
            self.assertLess(time.time() - t0, 1.2)
        with self.assertRaises(P.ProviderError) as cm:
            P.list_models("amazon-bedrock", self.env(AWS_BEARER_TOKEN_BEDROCK="x"), self.home)
        self.assertIn("not yet callable", str(cm.exception))
        with self.assertRaises(P.ProviderError):
            P.list_models("no-such-provider", self.env(), self.home)

    def test_static_and_cli_and_local_tables(self):
        self.assertIn("sonar-pro", P.list_models("perplexity", self.env(PERPLEXITY_API_KEY="k"), self.home))
        self.assertEqual(P.list_models("claude", self.env(), self.home), ["haiku", "sonnet", "opus", "fable"])
        tags = {"models": [{"name": "qwen3:8b", "details": {"parameter_size": "8B"}}]}
        with FakeServer(lambda r: json_reply(tags)) as srv:
            spec = {"id": "local:x", "style": "local", "endpoint_url": srv.url, "kind": "ollama"}
            self.assertEqual(P.list_models(spec, self.env(), self.home), ["qwen3:8b"])


class Detect(Tmp):
    def detect(self, env=None, cfg=None, **kw):
        env = env or self.env()
        cfg = cfg or core.load_config(self.home)
        kw.setdefault("list_live", False)
        return P.detect_all(env=env, home=self.home, cfg=cfg, inventory=kw.pop("inventory", inventory()), **kw)

    def by_id(self, res):
        return {e["id"]: e for e in res["providers"]}

    def test_api_key_presence_without_the_network(self):
        res = self.detect(self.env(OPENAI_API_KEY=SECRET, AZURE_API_KEY="k"))
        e = self.by_id(res)
        self.assertEqual((e["openai"]["status"], e["openai"]["key_source"], e["openai"]["key_env"]), ("detected", "env", "OPENAI_API_KEY"))
        self.assertEqual(e["groq"]["status"], "missing")
        self.assertEqual(e["azure"]["status"], "missing")
        self.assertIn("AZURE_RESOURCE_NAME", e["azure"]["reason"])
        self.assertFalse(res["verified"])
        self.assertNotIn(SECRET, json.dumps(res))

    def test_live_listing_marks_connected_and_error(self):
        def handler(r):
            if r["headers"].get("authorization") == "Bearer " + SECRET:
                return json_reply({"data": [{"id": "gpt-a"}, {"id": "gpt-b"}]})
            return json_reply({"error": {"message": "bad key"}}, 401)
        with FakeServer(handler) as srv:
            cfg = dict(core.load_config(self.home), providers={"openai": {"base_url": srv.url}, "groq": {"base_url": srv.url}})
            P.set_key("GROQ_API_KEY", "wrong-key", self.home)
            res = self.detect(self.env(OPENAI_API_KEY=SECRET), cfg=cfg, list_live=True)
        e = self.by_id(res)
        self.assertEqual((e["openai"]["status"], e["openai"]["models"]), ("connected", ["gpt-a", "gpt-b"]))
        self.assertEqual((e["groq"]["status"], e["groq"]["key_source"]), ("error", "keys.env"))
        self.assertIn("HTTP 401", e["groq"]["reason"])
        saved = json.loads(read(os.path.join(self.home, "providers.json")))
        self.assertEqual({x["id"]: x["status"] for x in saved["providers"]}["openai"], "connected")
        blob = json.dumps(res) + read(os.path.join(self.home, "providers.json"))
        self.assertNotIn(SECRET, blob)
        self.assertNotIn("wrong-key", blob)
        self.assertEqual(P.load_providers(self.home)["version"], 1)

    def test_bedrock_and_vertex_are_detected_but_not_callable(self):
        named = dict(core.load_config(self.home), providers={"amazon-bedrock": {}, "google-vertex": {}})  # opt-in only providers
        e = self.by_id(self.detect(self.env(AWS_BEARER_TOKEN_BEDROCK="t", GOOGLE_VERTEX_PROJECT="p", GOOGLE_VERTEX_LOCATION="us-east5"), cfg=named))
        for pid in ("amazon-bedrock", "google-vertex"):
            self.assertEqual(e[pid]["status"], "detected")
            self.assertIn("not yet callable", e[pid]["reason"])
        self.assertEqual(e["amazon-bedrock"]["models"], [])
        e = self.by_id(self.detect(self.env(AWS_ACCESS_KEY_ID="a", GOOGLE_VERTEX_PROJECT="p")))  # half a combination is not configured
        self.assertEqual((e["amazon-bedrock"]["status"], e["google-vertex"]["status"]), ("missing", "missing"))
        self.assertIn("AWS_SECRET_ACCESS_KEY", e["amazon-bedrock"]["reason"])
        self.assertIn("GOOGLE_VERTEX_LOCATION", e["google-vertex"]["reason"])

    def test_every_provider_is_in_the_inventory_and_shared_vars_detect_each_provider(self):
        res = self.detect(self.env(MINIMAX_API_KEY="k"))
        e = self.by_id(res)
        self.assertGreaterEqual(len([x for x in res["providers"] if x["access"] == "api"]), 220)
        self.assertEqual((e["minimax"]["status"], e["minimax"]["style"]), ("detected", "anthropic"))
        self.assertEqual(e["minimax-cn"]["status"], "missing")  # same variable in models.dev: only one of the group counts
        self.assertIn("shares MINIMAX_API_KEY with minimax", e["minimax-cn"]["reason"])

    def test_of_providers_sharing_a_key_variable_only_one_counts(self):
        env = self.env(MINIMAX_API_KEY="k", ZHIPU_API_KEY="k", STEPFUN_API_KEY="k", DEEPSEEK_API_KEY="k")
        e = self.by_id(self.detect(env))
        self.assertEqual([p for p in e if p.startswith("minimax") and e[p]["status"] == "detected"], ["minimax"])
        self.assertEqual(e["minimax-coding-plan"]["shares_with"], "minimax")
        self.assertEqual([p for p in e if p.startswith("stepfun") and e[p]["status"] == "detected"], ["stepfun"])
        self.assertEqual([p for p in e if p in ("zai", "zai-coding-plan", "zhipuai", "zhipuai-coding-plan") and e[p]["status"] == "detected"], ["zai"])
        self.assertEqual(e["deepseek"]["status"], "detected")
        # a provider named under config `providers` counts instead of the primary
        cfg = dict(core.load_config(self.home), providers={"minimax-cn": {}})
        e = self.by_id(self.detect(env, cfg=cfg))
        self.assertEqual([p for p in e if p.startswith("minimax") and e[p]["status"] == "detected"], ["minimax-cn"])
        self.assertEqual(e["minimax"]["shares_with"], "minimax-cn")
        # held providers are not listed live: no request goes to them
        seen = []
        with FakeServer(lambda r: (seen.append(r["headers"].get("authorization")), json_reply({"data": [{"id": "m"}]}))[1]) as srv:
            cfg = dict(core.load_config(self.home), providers={k: {"base_url": srv.url} for k in ("minimax", "minimax-cn")})
            cfg["providers"].pop("minimax-cn")
            res = self.by_id(self.detect(self.env(MINIMAX_API_KEY="k"), cfg=cfg, list_live=True))
        self.assertEqual(len(seen), 1)
        self.assertEqual((res["minimax"]["status"], res["minimax-cn"]["status"], res["minimax-cn"]["models"]), ("connected", "missing", []))

    def test_generic_platform_credentials_do_not_detect_opt_in_providers(self):
        env = self.env(GITHUB_TOKEN="ghp_x", GH_TOKEN="ghp_y", AWS_ACCESS_KEY_ID="a", AWS_SECRET_ACCESS_KEY="b", AWS_BEARER_TOKEN_BEDROCK="t",
                       GOOGLE_VERTEX_PROJECT="p", GOOGLE_VERTEX_LOCATION="l")
        e = self.by_id(self.detect(env))
        for pid in ("github-copilot", "amazon-bedrock", "google-vertex", "google-vertex-anthropic"):
            self.assertEqual(e[pid]["status"], "missing", pid)
            self.assertIn("config providers", e[pid]["reason"])
        e = self.by_id(self.detect(env, cfg=dict(core.load_config(self.home), providers={"github-copilot": {}})))
        self.assertEqual((e["github-copilot"]["status"], e["amazon-bedrock"]["status"]), ("detected", "missing"))

    def test_claude_needs_the_cli_and_a_login(self):
        os.makedirs(os.path.join(self.tmp, ".claude"))
        write(os.path.join(self.tmp, ".claude", ".credentials.json"), '{"token": "SHOULD-NEVER-BE-READ"}')
        e = self.by_id(self.detect())["claude"]
        self.assertEqual(e["status"], "detected")
        self.assertIn("not on PATH", e["reason"])
        fake_cli(self.bin, "claude", "exit 0")
        res = self.detect()
        e = self.by_id(res)["claude"]
        self.assertEqual((e["status"], e["access"], e["style"]), ("connected", "subscription-cli", "cli"))
        self.assertEqual(e["models"], ["haiku", "sonnet", "opus", "fable"])
        self.assertNotIn("SHOULD-NEVER-BE-READ", json.dumps(res))

    def test_claude_oauth_env_counts_and_a_cli_alone_does_not(self):
        fake_cli(self.bin, "claude", "exit 0")
        self.assertEqual(self.by_id(self.detect())["claude"]["reason"], "installed, not logged in")
        self.assertEqual(self.by_id(self.detect(self.env(CLAUDE_CODE_OAUTH_TOKEN="tok")))["claude"]["status"], "connected")

    def test_macos_keychain_is_asked_without_dash_w_and_only_on_darwin(self):
        log = os.path.join(self.tmp, "security.log")
        fake_cli(self.bin, "security", 'echo "$@" >> %s; echo "secret-attributes"; exit 0' % log)
        fake_cli(self.bin, "claude", "exit 0")
        e = self.by_id(self.detect(platform="linux"))["claude"]
        self.assertEqual(e["status"], "missing")
        self.assertFalse(os.path.exists(log))
        e = self.by_id(self.detect(platform="darwin"))["claude"]
        self.assertEqual(e["status"], "connected")
        args = read(log).split()
        self.assertEqual(args, ["find-generic-password", "-s", "Claude", "Code-credentials"])
        self.assertNotIn("-w", args)

    def test_keychain_miss_means_not_logged_in(self):
        fake_cli(self.bin, "security", "exit 44")
        fake_cli(self.bin, "claude", "exit 0")
        self.assertEqual(self.by_id(self.detect(platform="darwin"))["claude"]["status"], "missing")

    def test_codex_login_status_or_auth_file(self):
        fake_cli(self.bin, "codex", '[ "$1 $2" = "login status" ] && exit 0; exit 2')
        e = self.by_id(self.detect())["codex"]
        self.assertEqual(e["status"], "connected")
        self.assertIn("login status", e["reason"])
        self.assertIn("gpt-6.1-sol", e["models"])
        fake_cli(self.bin, "codex", "exit 1")
        self.assertEqual(self.by_id(self.detect())["codex"]["reason"], "installed, not logged in")
        os.makedirs(os.path.join(self.tmp, ".codex"))
        write(os.path.join(self.tmp, ".codex", "auth.json"), "{}")
        self.assertEqual(self.by_id(self.detect())["codex"]["status"], "connected")

    def test_gemini_copilot_and_cursor_agent(self):
        fake_cli(self.bin, "gemini", "exit 0")
        fake_cli(self.bin, "copilot", "exit 0")
        fake_cli(self.bin, "agent", "exit 0")
        e = self.by_id(self.detect())
        self.assertEqual(e["gemini"]["reason"], "installed, not logged in")
        self.assertEqual(e["copilot"]["reason"], "installed, not logged in")
        self.assertEqual(e["cursor-agent"]["status"], "detected")  # `agent` counts as cursor-agent
        os.makedirs(os.path.join(self.tmp, ".gemini"))
        write(os.path.join(self.tmp, ".gemini", "oauth_creds.json"), "{}")
        self.assertEqual(self.by_id(self.detect())["gemini"]["status"], "connected")
        e = self.by_id(self.detect(self.env(COPILOT_GITHUB_TOKEN="t")))["copilot"]
        self.assertEqual(e["status"], "detected")
        self.assertIn("no non-interactive client", e["reason"])

    def test_missing_clis_are_missing(self):
        e = self.by_id(self.detect())
        for pid in ("claude", "codex", "gemini", "copilot", "cursor-agent"):
            self.assertEqual((e[pid]["status"], e[pid]["reason"]), ("missing", "not installed"))

    def test_local_endpoints_come_from_the_router_inventory(self):
        inv = inventory(local_models=[local("qwen3:8b", 8, endpoint="spark")])
        inv["local_endpoints"] = [{"name": "spark", "url": "http://spark:11434", "kind": "ollama", "models": 1}]
        e = self.by_id(self.detect(inventory=inv))["local:spark"]
        self.assertEqual((e["access"], e["style"], e["status"], e["models"]), ("local", "local", "connected", ["qwen3:8b"]))
        self.assertEqual(e["base_url"], "http://spark:11434/v1")

    def test_a_provider_without_a_model_list_endpoint_stays_detected(self):
        with FakeServer(lambda r: json_reply({"error": "not found"}, 404)) as srv:
            cfg = dict(core.load_config(self.home), providers={"groq": {"base_url": srv.url}})
            e = self.by_id(self.detect(self.env(GROQ_API_KEY="k"), cfg=cfg, list_live=True))["groq"]
        self.assertEqual(e["status"], "detected")
        self.assertIn("no model list endpoint", e["reason"])

    def test_vllm_style_locals_keep_raw_ids_and_the_context_window(self):
        body = {"object": "list", "data": [
            {"id": "openai/gpt-oss-120b", "object": "model", "root": "openai/gpt-oss-120b", "max_model_len": 131072},
            {"id": "MiniMaxAI/MiniMax-M3", "object": "model", "root": "/models/minimax-m3", "max_model_len": 196608},
            {"id": "plain-server-model", "object": "model"}]}
        with FakeServer(lambda r: json_reply(body)) as srv:
            cfg = dict(core.load_config(self.home), local_endpoints=[{"name": "spark", "url": srv.url, "kind": "openai"}])
            env = self.env()
            inv = core.detect(home=self.home, env=env, cfg=cfg, hw_probe=lambda: {}, write=False)
        mods = {m["name"]: m for m in inv["local_models"]}
        self.assertEqual(mods["openai/gpt-oss-120b"]["max_model_len"], 131072)
        self.assertEqual(mods["MiniMaxAI/MiniMax-M3"]["root"], "/models/minimax-m3")
        self.assertNotIn("max_model_len", mods["plain-server-model"])
        e = self.by_id(self.detect(inventory=inv))["local:spark"]
        self.assertEqual(e["models"], ["MiniMaxAI/MiniMax-M3", "openai/gpt-oss-120b", "plain-server-model"])
        self.assertEqual(e["model_info"]["openai/gpt-oss-120b"], {"max_model_len": 131072, "root": "openai/gpt-oss-120b", "params_b": 120.0, "loaded": True})
        self.assertNotIn("max_model_len", e["model_info"]["plain-server-model"])

    def test_summary_lines_are_compact(self):
        res = self.detect(self.env(OPENAI_API_KEY=SECRET))
        text = "\n".join(P.summary_lines(res))
        self.assertIn("openai", text)
        self.assertIn("other API providers have no key", text)
        self.assertNotIn(SECRET, text)


class Integration(Tmp):
    def test_route_never_imports_providers(self):
        code = ("import sys; sys.path.insert(0, %r); import maestro; "
                "d = maestro.route('fix the failing test', harness='claude-code', classifier='rules', "
                "inventory={'harnesses': {}, 'local_models': []}); "
                "assert not [m for m in sys.modules if m.startswith('maestro.') and m.split('.')[1] in "
                "('providers', 'clients', 'quota', 'learn', 'systemone')], sorted(sys.modules); print('ok')"
                % os.path.join(ROOT, "skills", "maestro", "scripts"))
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                             env=dict(os.environ, MAESTRO_HOME=self.home))
        self.assertEqual((res.returncode, res.stdout.strip()), (0, "ok"), res.stderr)

    def cli(self, *argv, env=None):
        out = io.StringIO()
        with mock.patch.dict(os.environ, env or self.env(), clear=True), contextlib.redirect_stdout(out):
            rc = core.main(list(argv))
        return rc, out.getvalue()

    def test_detect_prints_a_providers_section_and_json_has_it(self):
        rc, out = self.cli("detect", env=self.env(OPENAI_API_KEY=SECRET))
        self.assertEqual(rc, 0)
        self.assertIn("providers", out)
        self.assertIn("openai", out)
        self.assertNotIn(SECRET, out)
        rc, out = self.cli("detect", "--json", env=self.env(OPENAI_API_KEY=SECRET))
        data = json.loads(out)
        self.assertEqual({p["id"]: p["status"] for p in data["providers"]}["openai"], "detected")
        self.assertNotIn(SECRET, out)
        self.assertIn("harnesses", data)

    def test_detect_models_lists_live_and_saves_providers_json(self):
        with FakeServer(lambda r: json_reply({"data": [{"id": "gpt-a"}]})) as srv:
            os.makedirs(self.home)
            write(os.path.join(self.home, "config.json"), json.dumps({"providers": {"openai": {"base_url": srv.url}}}))
            rc, out = self.cli("detect", "--models", env=self.env(OPENAI_API_KEY=SECRET))
        self.assertEqual(rc, 0)
        self.assertIn("connected", out)
        self.assertEqual({p["id"]: p["models"] for p in P.load_providers(self.home)["providers"]}["openai"], ["gpt-a"])


if __name__ == "__main__":
    unittest.main()
