"""One model map per harness: resolution, suggestions, `maestro-router models`, local pins and the models.dev check."""
import contextlib
import http.server
import io
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

from helpers import cls, inventory, local, maestro as m

TIERS = m.TIERS


def as_dicts(rows):
    return [dict(zip(m.ROW, r)) for r in rows]


def catalog_json():
    """A small api.json in models.dev's shape (values copied from the real catalog, 2026-09-30)."""
    def mod(name, family, date, out, status=None, tool=True):
        d = {"id": name, "name": name, "family": family, "release_date": date, "tool_call": tool,
             "cost": {"input": out / 5.0, "output": out}}
        return dict(d, status=status) if status else d

    def prov(pid, env, *models):
        return {"id": pid, "env": [env], "models": {x["id"]: x for x in models}}
    return json.dumps({
        "anthropic": prov("anthropic", "ANTHROPIC_API_KEY",
                          mod("claude-haiku-4-5", "claude-haiku", "2025-10-15", 5),
                          mod("claude-sonnet-5-5", "claude-sonnet", "2026-09-28", 10),
                          mod("claude-opus-5-5", "claude-opus", "2026-09-22", 20),
                          mod("claude-fable-5-1", "claude-fable", "2026-09-01", 50)),
        "openai": prov("openai", "OPENAI_API_KEY",
                       mod("gpt-6-luna", "gpt-luna", "2026-09-22", 0.5),
                       mod("gpt-6-sol", "gpt-sol", "2026-09-22", 10),
                       mod("gpt-6.1-sol", "gpt-sol", "2026-09-29", 10),
                       mod("gpt-6-astra", "gpt-astra", "2026-09-04", 50),
                       mod("gpt-4.1-nano", "gpt-nano", "2025-04-14", 0.4, status="deprecated")),
        "google": prov("google", "GEMINI_API_KEY",
                       mod("gemini-flash-lite-latest", "gemini-flash-lite", "2026-07-21", 2.5),
                       mod("gemini-flash-latest", "gemini-flash", "2026-08-13", 3.75),
                       mod("gemini-3.7-flash", "gemini-flash", "2026-08-13", 3.75),
                       mod("gemini-3.8-flash", "gemini-flash", "2026-09-02", 3.75),
                       mod("gemini-3.1-pro-preview", "gemini-pro", "2026-02-19", 12),
                       # same family, newer, but a different product line or not a chat model: not "a newer Pro"
                       mod("deep-research-preview-04-2026", "gemini-pro", "2026-04-21", 12),
                       mod("gemini-3-pro-image", "gemini-pro", "2026-05-28", 120, tool=False)),
    }).encode("utf-8")


class Resolution(unittest.TestCase):
    def test_order_is_config_then_suggestion_then_builtin(self):
        sm = m.seat_models("codex", inventory(), {"models": {"codex": {"fast": "mine", "strong": {"model": "x", "effort": "low"}}}})
        self.assertEqual(sm["fast"], ("mine", None, "config"))
        self.assertEqual(sm["standard"], ("gpt-6.1-sol", None, "suggested"))
        self.assertEqual(sm["strong"], ("x", "low", "config"))
        self.assertEqual(sm["frontier"], ("gpt-6-astra", None, "suggested"))
        with mock.patch.dict(m.SUGGESTED_MODELS, {"codex": {}}):
            self.assertEqual(m.seat_models("codex", inventory(), {})["strong"], (None, None, "builtin"))
            self.assertEqual(set(m.seat_models("codex", inventory(), {})), {"strong"})

    def test_suggestions_are_the_dated_snapshot(self):
        self.assertEqual((m.SUGGESTED_AS_OF, m.SUGGESTED_SOURCE), ("2026-09-30", "models.dev"))
        got = {s: [m.seat_models(s, inventory(), {})[t][0] for t in TIERS] for s in ("claude-code", "codex", "gemini-cli")}
        self.assertEqual(got["claude-code"], ["haiku", "sonnet", "opus", "fable"])
        self.assertEqual(got["codex"], ["gpt-6-luna", "gpt-6.1-sol", "gpt-6.1-sol", "gpt-6-astra"])
        self.assertEqual(got["gemini-cli"], ["gemini-flash-lite-latest", "gemini-flash-latest",
                                             "gemini-3.1-pro-preview", "gemini-3.1-pro-preview"])
        self.assertEqual(m.seat_models("codex", inventory(), {})["strong"][1], "high")

    def test_harness_models_still_works_as_an_alias(self):
        pol = {"harness_models": {"codex": {"fast": "old-mini"}, "gemini-cli": "old-strong"}}
        self.assertEqual(m.seat_models("codex", inventory(), pol)["fast"], ("old-mini", None, "config"))
        self.assertEqual(m.seat_models("gemini-cli", inventory(), pol)["strong"], ("old-strong", None, "config"))  # a bare string = strong
        pol["models"] = {"codex": {"fast": "new-mini"}}
        self.assertEqual(m.seat_models("codex", inventory(), pol)["fast"][0], "new-mini")  # `models` wins when present
        self.assertEqual(m.seat_models("gemini-cli", inventory(), pol)["strong"][2], "suggested")

    def test_load_config_keeps_both_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump({"models": {"codex": {"fast": "a"}}, "harness_models": {"codex": {"fast": "b"}}}, f)
            cfg = m.load_config(tmp)
        self.assertEqual((cfg["models"], cfg["harness_models"]), ({"codex": {"fast": "a"}}, {"codex": {"fast": "b"}}))

    def test_bad_cells_fall_through_to_the_suggestion(self):
        pol = {"models": {"codex": {"fast": 3, "standard": {"effort": "high"}, "strong": {"model": "", "effort": "high"},
                                    "frontier": {"model": "z", "effort": "turbo"}}}}
        sm = m.seat_models("codex", inventory(), pol)
        self.assertEqual([sm[t][2] for t in TIERS], ["suggested", "suggested", "suggested", "config"])
        self.assertEqual(sm["frontier"], ("z", None, "config"))  # an unknown effort is dropped

    def test_decision_records_the_model_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"MAESTRO_HOME": tmp, "HOME": tmp, "PATH": ""}
            d = m.route("why does the auth test fail intermittently under load across the distributed cache layer and the retry middleware, find the root cause of the race condition", harness="claude-code", classifier="rules", inventory=inventory(),
                        env=env, home=tmp, config=dict(m.load_config(tmp), models={"claude-code": {"strong": "opus-x"}}))
            self.assertEqual((d["model"], d["model_source"]), ("opus-x", "config"))
            d = m.route("hi", harness="unknown", classifier="rules", inventory=inventory(), env=env, home=tmp)
            self.assertEqual((d["delegate"], d["model_source"]), ({"kind": "inline"}, None))


class Routing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = {"MAESTRO_HOME": self.tmp.name, "HOME": self.tmp.name, "PATH": ""}

    def tearDown(self):
        self.tmp.cleanup()

    def route(self, text, harness, inv, **kw):
        return m.route(text, harness=harness, classifier="rules", inventory=inv, env=self.env, home=self.tmp.name,
                       config=kw.pop("config", m.load_config(self.tmp.name)), **kw)

    def test_inside_codex_a_fast_task_runs_codex_exec_with_the_fast_model(self):
        d = self.route("rename tmp to userCount in stats.ts", "codex", inventory(clis=("codex",)))
        self.assertEqual((d["category"], d["seat"], d["model"], d["model_source"]), ("fast", "codex", "gpt-6-luna", "suggested"))
        self.assertEqual(d["delegate"], {"kind": "cli", "argv": ["codex", "exec", "-m", "gpt-6-luna", "{task}"]})
        d = self.route("rename tmp to userCount in stats.ts", "codex", inventory(clis=("codex",)), current_tier="fast")
        self.assertEqual(d["delegate"], {"kind": "inline"})  # already on that category: do it yourself

    def test_effort_reaches_only_the_codex_command(self):
        d = self.route("why does the auth test fail intermittently under load across the distributed cache layer and the retry middleware, find the root cause of the race condition", "codex", inventory(clis=("codex",)))
        self.assertEqual(d["category"], "strong")
        self.assertEqual(d["delegate"]["argv"], ["codex", "exec", "-m", "gpt-6.1-sol", "-c", "model_reasoning_effort=high", "{task}"])
        std = m.select_seat("standard", cls(), inventory(clis=("codex",)), "codex", {}, None)
        self.assertNotIn("-c", std["delegate"]["argv"])  # the standard cell has no effort
        for seat in ("claude-code", "gemini-cli", "opencode"):
            self.assertEqual(m.cli_argv(seat, "x", "high"), m.cli_argv(seat, "x"))
        pol = {"models": {"gemini-cli": {"strong": {"model": "g", "effort": "high"}}}}
        s = m.select_seat("strong", cls(), inventory(), "gemini-cli", pol, None)
        self.assertEqual(s["delegate"], {"kind": "subagent", "harness": "gemini-cli", "model": "g", "agent": "maestro-strong"})
        pol["job_seats"] = {"code": "gemini"}  # from another harness it is a headless CLI: no effort flag there either
        s = m.select_seat("strong", cls(), inventory(clis=("gemini-cli",)), "claude-code", pol, None)
        self.assertEqual(s["delegate"]["argv"], ["gemini", "-m", "g", "-p", "{task}"])

    def test_gemini_and_opencode_get_every_category_with_no_config(self):
        for harness, want in (("gemini-cli", ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-3.1-pro-preview", "gemini-3.1-pro-preview"]),
                              ("opencode", ["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-5-5",
                                            "anthropic/claude-opus-5-5", "anthropic/claude-fable-5-1"])):
            got = [m.select_seat(t, cls(), inventory(), harness, {}, None) for t in TIERS]
            self.assertEqual([s["model"] for s in got], want, harness)
            self.assertEqual([s["delegate"]["agent"] for s in got], ["maestro-" + t for t in TIERS])
            self.assertEqual({s["kind"] for s in got}, {"subagent"})

    def test_other_installed_clis_get_the_full_map_after_the_current_harness(self):
        inv = inventory(clis=("claude-code", "codex", "gemini-cli", "opencode"))
        cands = m.build_candidates(cls(), inv, "opencode", {})
        self.assertEqual([c["seat"] for c in cands][:4], ["opencode"] * 4)
        self.assertEqual({(c["seat"], c["serves"]) for c in cands if c["kind"] == "cli"},
                         {(s, t) for s in ("claude-code", "codex", "gemini-cli") for t in TIERS})
        self.assertEqual(self.route("plan the auth split across our services", "codex", inv)["seat"], "codex")

    def test_a_model_serving_two_categories_gets_distinct_candidate_names(self):
        cands = m.build_candidates(cls(), inventory(clis=("codex",)), "unknown", {})
        names = [c["name"] for c in cands]
        self.assertEqual(len(set(names)), len(names))
        self.assertEqual(names, ["codex:gpt-6-luna", "codex:gpt-6.1-sol", "codex:gpt-6.1-sol@strong", "codex:gpt-6-astra"])

    def test_jev_descriptions_name_the_model_and_effort(self):
        desc = {c["name"]: c["description"] for c in m.build_candidates(cls(), inventory(clis=("codex",)), "unknown", {})}
        self.assertEqual(desc["codex:gpt-6.1-sol"],
                         "standard tier, Codex (gpt-6.1-sol) headless CLI, subscription (flat cost), tool access but no shared context")
        self.assertIn("strong tier, Codex (gpt-6.1-sol, effort high) headless CLI", desc["codex:gpt-6.1-sol@strong"])
        d = {c["name"]: c["description"] for c in m.build_candidates(cls(), inventory(), "opencode", {})}
        self.assertIn("OpenCode (anthropic/claude-sonnet-5-5) subagent", d["opencode:anthropic/claude-sonnet-5-5"])

    def test_opencode_provider_follows_the_credentials_in_order(self):
        def pick(*keys):
            inv = inventory()
            inv["api_keys"] = {k: True for k in keys}
            return m.seat_models("opencode", inv, {})["standard"][0]
        self.assertEqual(pick(), "anthropic/claude-sonnet-5-5")  # nothing detected: the first provider
        self.assertEqual(pick("GEMINI_API_KEY"), "google/gemini-flash-latest")
        self.assertEqual(pick("GOOGLE_API_KEY", "OPENAI_API_KEY"), "openai/gpt-6.1-sol")
        self.assertEqual(pick("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"), "anthropic/claude-sonnet-5-5")
        self.assertEqual(m.opencode_provider({"api_keys": {"OPENAI_API_KEY": False}}), "anthropic")

    def test_install_writes_the_suggested_models(self):
        inv = inventory()
        inv["api_keys"] = {"OPENAI_API_KEY": True}
        m.install("all", user_home=self.tmp.name, cfg=m.load_config(self.tmp.name), inv=inv)
        def model(*parts):
            with open(os.path.join(self.tmp.name, *parts)) as f:
                return [l for l in f.read().splitlines() if l.startswith("model:")]
        self.assertEqual(model(".config", "opencode", "agents", "maestro-fast.md"), ["model: openai/gpt-6-luna"])
        self.assertEqual(model(".config", "opencode", "agents", "maestro-frontier.md"), ["model: openai/gpt-6-astra"])
        self.assertEqual(model(".gemini", "agents", "maestro-standard.md"), ["model: gemini-flash-latest"])
        self.assertEqual(model(".claude", "agents", "maestro-strong.md"), ["model: opus"])
        cfg = dict(m.load_config(self.tmp.name), models={"gemini-cli": {"standard": "mine"}})
        m.install("gemini-cli", user_home=self.tmp.name, cfg=cfg, inv=inv)
        self.assertEqual(model(".gemini", "agents", "maestro-standard.md"), ["model: mine"])


class LocalPins(unittest.TestCase):
    def setUp(self):
        self.inv = inventory(local_models=[local("gpt-oss:120b", 120, endpoint="spark"), local("qwen3:14b", 14)])

    def pick(self, cell, pol=None, job="ask"):
        s = m.select_seat(cell, cls(job, 1), self.inv, "claude-code", pol or {}, None)
        return s["model"], s["category"]

    def test_a_pin_overrides_the_size_rating(self):
        self.assertEqual(self.pick("fast"), ("local:this-machine/qwen3:14b", "local"))
        self.assertEqual(self.pick("standard"), ("local:spark/gpt-oss:120b", "local"))
        pol = {"models": {"local": {"fast": "local:spark/gpt-oss:120b"}}}
        self.assertEqual(self.pick("fast", pol), ("local:spark/gpt-oss:120b", "local"))  # pinned to fast: first choice there
        self.assertEqual(m.rate_local(self.inv["local_models"][0], pol)[0], "fast")      # ... and rated fast, not standard
        self.assertEqual(self.pick("standard", pol), ("sonnet", "standard"))              # so it no longer serves standard
        pol = {"models": {"local": {"standard": "spark/gpt-oss:120b", "local": "qwen3:14b"}}}  # endpoint/name and bare name work
        self.assertEqual(m.rate_local(self.inv["local_models"][1], pol)[0], "local")
        self.assertEqual(self.pick("local", pol)[0], "local:this-machine/qwen3:14b")

    def test_a_pin_under_local_is_the_preferred_model_for_local_cells(self):
        inv = inventory(local_models=[local("a:2b", 2), local("b:3b", 3)])
        pick = lambda pol: m.select_seat("local", cls("ask", 0), inv, "unknown", pol, None)["model"]
        self.assertEqual(pick({}), "local:this-machine/a:2b")  # smallest first without a pin
        self.assertEqual(pick({"models": {"local": {"local": "local:this-machine/b:3b"}}}), "local:this-machine/b:3b")

    def test_the_decision_says_where_a_local_choice_came_from(self):
        pol = {"models": {"local": {"standard": "local:spark/gpt-oss:120b"}}}
        self.assertEqual(m.select_seat("standard", cls("write", 1), self.inv, "unknown", pol, None)["source"], "config")
        self.assertEqual(m.select_seat("standard", cls("write", 1), self.inv, "unknown", {}, None)["source"], "auto")

    def test_an_invalid_pin_is_ignored_and_reported(self):
        pol = dict(m.load_config("/nonexistent"), models={"local": {"standard": "local:gone/nothing", "huge": "local:spark/gpt-oss:120b"}})
        self.assertEqual(self.pick("standard", pol), self.pick("standard"))
        rows, warns = m.models_rows(self.inv, pol)
        rows = as_dicts(rows)
        self.assertEqual(len(warns), 2)
        self.assertTrue(all("ignored" in w for w in warns))
        self.assertIn("local:gone/nothing", warns[0])
        self.assertEqual([r["source"] for r in rows if r["harness"] == "local"], ["auto", "auto", "auto", "none", "none"])

    def test_the_local_rows_of_the_map(self):
        rows, _ = m.models_rows(self.inv, dict(m.load_config("/nonexistent"), models={"local": {"local": "qwen3:14b"}}))
        rows = as_dicts(rows)
        got = {r["category"]: (r["model"], r["source"]) for r in rows if r["harness"] == "local"}
        self.assertEqual(got["local"], ("local:this-machine/qwen3:14b", "config"))
        self.assertEqual(got["fast"], ("local:spark/gpt-oss:120b", "auto"))  # qwen is pinned to local only, so it stops at local
        self.assertEqual(got["standard"], ("local:spark/gpt-oss:120b", "auto"))
        self.assertEqual(got["strong"], ("-", "none"))


class Commands(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = os.path.join(self.tmp.name, "mh")
        self.cfg_path = os.path.join(self.home, "config.json")
        os.makedirs(self.home)
        self.env = {"MAESTRO_HOME": self.home, "HOME": self.tmp.name, "PATH": "", "OPENCODE_CONFIG": "x"}
        self.write_inventory()

    def tearDown(self):
        self.tmp.cleanup()

    def write_inventory(self, clis=("codex", "opencode"), **kw):
        inv = inventory(clis=clis, **kw)
        inv["endpoints_sig"] = [e["url"] for e in m.local_endpoints(m.load_config(self.home), self.env)]
        with open(os.path.join(self.home, "inventory.json"), "w") as f:
            json.dump(inv, f)

    def config(self, data=None, raw=None):
        with open(self.cfg_path, "w") as f:
            f.write(raw if raw is not None else json.dumps(data))
        self.write_inventory()

    def read(self):
        with open(self.cfg_path) as f:
            return f.read()

    def cli(self, *argv, env=None):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, dict(self.env, **(env or {})), clear=True), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = m.main(list(argv))
        return rc, out.getvalue(), err.getvalue()

    def test_models_prints_the_effective_map_for_the_inventory(self):
        self.config({"models": {"codex": {"fast": "mine"}}})
        rc, out, _ = self.cli("models")
        self.assertEqual(rc, 0)
        self.assertIn("codex\n  fast      mine", out)
        self.assertIn("standard  gpt-6.1-sol", out)
        self.assertIn("strong    gpt-6.1-sol (effort high)", out)
        self.assertIn("opencode\n  fast      anthropic/claude-haiku-4-5", out)
        self.assertNotIn("gemini-cli", out)  # not in the inventory
        self.assertNotIn("claude-code\n", out)
        self.assertIn("2026-09-30", out)
        self.assertTrue(out.rstrip().endswith("harness detected now: opencode"))
        rows = {(l.split()[0], l.split()[2]) for l in out.splitlines() if l.startswith("  ")}
        self.assertIn(("fast", "config"), rows)
        self.assertIn(("fast", "suggested"), rows)

    def test_models_json_and_effort_note(self):
        self.config({"models": {"opencode": {"fast": {"model": "anthropic/x", "effort": "low"}}}, "max_tier": "strong"})
        rc, out, _ = self.cli("models", "--json")
        data = json.loads(out)
        self.assertEqual(data["harness"], "opencode")
        fast = next(r for r in data["models"] if r["harness"] == "opencode" and r["category"] == "fast")
        self.assertEqual((fast["model"], fast["effort"], fast["source"]), ("anthropic/x", "low", "config"))
        top = next(r for r in data["models"] if r["harness"] == "codex" and r["category"] == "frontier")
        self.assertEqual(top["available"], "no: above max_tier")
        self.assertIn("anthropic/x (effort ignored by opencode)", self.cli("models")[1])

    def test_models_warns_about_the_deprecated_key(self):
        self.config({"harness_models": {"codex": {"fast": "old"}}})
        _, out, _ = self.cli("models")
        self.assertIn("codex\n  fast      old", out)
        self.assertIn("warning: `harness_models` is deprecated", out)

    def test_suggest_lists_every_harness_and_the_opencode_pick(self):
        self.write_inventory()
        rc, out, _ = self.cli("models", "--suggest")
        self.assertEqual(rc, 0)
        for want in ("as of 2026-09-30", "claude-code\n  fast      haiku", "codex", "gemini-cli", "opencode (anthropic): picked, no key found",
                     "opencode (openai)\n  fast      openai/gpt-6-luna", "opencode (google)\n  fast      google/gemini-flash-lite-latest"):
            self.assertIn(want, out)
        inv = inventory(clis=("opencode",))
        inv["api_keys"] = {"OPENAI_API_KEY": True}
        inv["endpoints_sig"] = [e["url"] for e in m.local_endpoints(m.load_config(self.home), self.env)]
        with open(os.path.join(self.home, "inventory.json"), "w") as f:
            json.dump(inv, f)
        self.assertIn("opencode (openai): picked, key set", self.cli("models", "--suggest")[1])
        data = json.loads(self.cli("models", "--suggest", "--json")[1])
        self.assertEqual(len(data), 3 * 4 + 3 * 4)  # three harnesses, three OpenCode providers

    def test_set_and_unset_edit_only_the_models_key_with_indent_2(self):
        self.config({"classifier": "rules", "prefer": ["codex"], "models": {"codex": {"fast": "mine"}}})
        self.assertEqual(self.cli("models", "set", "codex.strong", "gpt-6.1-sol", "--effort", "xhigh")[0], 0)
        text = self.read()
        cfg = json.loads(text)
        self.assertEqual((cfg["classifier"], cfg["prefer"]), ("rules", ["codex"]))
        self.assertEqual(cfg["models"], {"codex": {"fast": "mine", "strong": {"model": "gpt-6.1-sol", "effort": "xhigh"}}})
        self.assertTrue(text.startswith('{\n  "classifier"'))
        rc, out, _ = self.cli("models", "set", "gemini", "x")  # a harness without a category
        self.assertEqual(rc, 2)
        rc, out, _ = self.cli("models", "set", "gemini.frontier", "g-x", "--effort", "high")
        self.assertIn("g-x (effort ignored by gemini-cli)", out)
        self.assertEqual(json.loads(self.read())["models"]["gemini-cli"], {"frontier": {"model": "g-x", "effort": "high"}})
        self.cli("models", "unset", "gemini-cli.frontier")
        self.cli("models", "unset", "codex.strong")
        self.assertEqual(json.loads(self.read())["models"], {"codex": {"fast": "mine"}})
        self.cli("models", "unset", "codex.fast")
        cfg = json.loads(self.read())
        self.assertNotIn("models", cfg)  # nothing left: the key goes, everything else stays
        self.assertEqual((cfg["classifier"], cfg["prefer"]), ("rules", ["codex"]))

    def test_set_creates_the_config_and_validates_names(self):
        # any harness name is allowed (it applies the model itself), but it must look like a name
        for bad in (("codex.huge", "x"), ("1nope.fast", "x"), ("no/pe.fast", "x"), ("nope.huge", "x"), ("codex", "x"),
                    ("local.strong.x", "x")):
            rc, _, err = self.cli("models", "set", *bad)
            self.assertEqual(rc, 2, bad)
            self.assertIn("maestro:", err)
        self.assertEqual(self.cli("models", "set", "codex.fast")[0], 2)  # no model given
        self.assertEqual(self.cli("models", "set", "local.fast", "local:a/b", "--effort", "low")[0], 2)
        self.assertEqual(self.cli("models", "frobnicate")[0], 2)
        self.assertFalse(os.path.exists(self.cfg_path))  # nothing written for any of them
        self.assertEqual(self.cli("models", "set", "codex.fast", "gpt-6-luna")[0], 0)
        self.assertEqual(json.loads(self.read()), {"models": {"codex": {"fast": "gpt-6-luna"}}})
        self.assertEqual(self.cli("models", "set", "local.standard", "local:spark/gpt-oss:120b")[0], 0)
        self.assertEqual(json.loads(self.read())["models"]["local"], {"standard": "local:spark/gpt-oss:120b"})
        self.assertEqual(self.cli("models", "set", "hermes.strong", "kimi-k3")[0], 0)   # a harness Maestro does not run
        self.assertEqual(json.loads(self.read())["models"]["hermes"], {"strong": "kimi-k3"})

    def test_set_refuses_to_overwrite_a_config_that_is_not_json(self):
        self.config(raw='{"classifier": "rules",')
        rc, _, err = self.cli("models", "set", "codex.fast", "x")
        self.assertEqual(rc, 2)
        self.assertIn("not valid JSON", err)
        self.assertEqual(self.read(), '{"classifier": "rules",')

    def test_set_moves_a_deprecated_harness_models_into_models(self):
        self.config({"harness_models": {"codex": {"fast": "old", "strong": "older"}, "opencode": "op-strong"}, "prefer": ["codex"]})
        self.assertEqual(self.cli("models", "set", "codex.standard", "mid")[0], 0)
        cfg = json.loads(self.read())
        self.assertNotIn("harness_models", cfg)
        self.assertEqual(cfg["models"], {"codex": {"fast": "old", "strong": "older", "standard": "mid"}, "opencode": {"strong": "op-strong"}})
        self.assertEqual(cfg["prefer"], ["codex"])

    def test_save_adds_only_unset_cells_and_keeps_everything_else(self):
        self.config({"classifier": "rules", "models": {"codex": {"fast": "mine", "strong": {"model": "s", "effort": "max"}}}})
        rc, out, _ = self.cli("models", "--save")
        self.assertEqual(rc, 0)
        self.assertIn("+ codex.standard = gpt-6.1-sol", out)
        self.assertIn("+ opencode.fast = anthropic/claude-haiku-4-5", out)
        self.assertNotIn("codex.fast", out)
        cfg = json.loads(self.read())
        self.assertEqual(cfg["classifier"], "rules")
        self.assertEqual(cfg["models"]["codex"], {"fast": "mine", "standard": "gpt-6.1-sol", "strong": {"model": "s", "effort": "max"},
                                                  "frontier": "gpt-6-astra"})
        self.assertEqual(cfg["models"]["opencode"]["frontier"], "anthropic/claude-fable-5-1")
        self.assertNotIn("claude-code", cfg["models"])  # not installed: nothing to save
        before = self.read()
        rc, out, _ = self.cli("models", "--save")
        self.assertIn("nothing to add", out)
        self.assertEqual(self.read(), before)

    def test_save_writes_the_codex_effort_as_a_cell_object(self):
        self.cli("models", "--save")
        cfg = json.loads(self.read())
        self.assertEqual(cfg["models"]["codex"]["strong"], {"model": "gpt-6.1-sol", "effort": "high"})
        self.assertEqual(cfg["models"]["codex"]["fast"], "gpt-6-luna")
        self.assertEqual(m.seat_models("codex", inventory(), m.load_config(self.home))["strong"], ("gpt-6.1-sol", "high", "config"))


class Check(unittest.TestCase):
    @classmethod
    def setUpClass(cls_):
        body = catalog_json()

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200 if self.path == "/api.json" else 404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body if self.path == "/api.json" else b"{}")

            def log_message(self, *a):
                pass
        cls_.server = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=cls_.server.serve_forever, daemon=True).start()
        cls_.url = "http://127.0.0.1:%d/api.json" % cls_.server.server_port

    @classmethod
    def tearDownClass(cls_):
        cls_.server.shutdown()
        cls_.server.server_close()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = os.path.join(self.tmp.name, "mh")
        os.makedirs(self.home)
        self.env = {"MAESTRO_HOME": self.home, "HOME": self.tmp.name, "PATH": "", "MAESTRO_CATALOG_URL": self.url,
                    "no_proxy": "127.0.0.1", "NO_PROXY": "127.0.0.1"}
        inv = inventory(clis=("codex", "opencode", "gemini-cli"))
        inv["endpoints_sig"] = [e["url"] for e in m.local_endpoints(m.load_config(self.home), self.env)]
        with open(os.path.join(self.home, "inventory.json"), "w") as f:
            json.dump(inv, f)

    def tearDown(self):
        self.tmp.cleanup()

    def cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, self.env, clear=True), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = m.main(list(argv))
        return rc, out.getvalue(), err.getvalue()

    def test_the_suggestions_pass_against_the_catalog(self):
        rc, out, _ = self.cli("models", "--check")
        self.assertEqual(rc, 0, out)
        self.assertIn("all ids exist and none is deprecated", out)
        self.assertNotIn("claude-code.", out)  # Claude Code aliases are not checked
        self.assertIn("gemini-flash-latest", out)  # an alias: fine although gemini-3.8-flash is newer
        self.assertIn("gemini-3.1-pro-preview", out)  # no newer Pro: a Deep Research model and an image model share the family
        self.assertNotIn("NEWER", out)

    def test_missing_deprecated_and_newer_in_family(self):
        cfg = {"models": {"codex": {"strong": "gpt-6-sol", "frontier": "gpt-4.1-nano"},
                          "gemini-cli": {"standard": "gemini-3.7-flash"},
                          "opencode": {"fast": "anthropic/claude-haiku-9", "standard": "spark/gpt-oss:120b"}}}
        with open(os.path.join(self.home, "config.json"), "w") as f:
            json.dump(cfg, f)
        rc, out, _ = self.cli("models", "--check", "--json")
        self.assertEqual(rc, 1)
        res = {(r["harness"], r["category"]): r for r in json.loads(out)}
        self.assertEqual(res[("codex", "strong")]["verdict"], "newer")
        self.assertIn("gpt-6.1-sol", res[("codex", "strong")]["detail"])
        self.assertEqual(res[("codex", "frontier")]["verdict"], "deprecated")
        self.assertEqual(res[("gemini-cli", "standard")]["verdict"], "newer")
        self.assertIn("gemini-3.8-flash", res[("gemini-cli", "standard")]["detail"])
        self.assertEqual(res[("opencode", "fast")]["verdict"], "missing")
        self.assertEqual(res[("opencode", "standard")]["verdict"], "skipped")  # a local provider is not in the catalog
        self.assertEqual(res[("codex", "fast")]["verdict"], "ok")
        self.assertNotIn(("claude-code", "fast"), res)
        rc, out, _ = self.cli("models", "--check")
        for want in ("NEWER", "DEPRECATED", "MISSING", "fix with `maestro-router models set"):
            self.assertIn(want, out)

    def test_check_only_reads_the_config(self):
        with open(os.path.join(self.home, "config.json"), "w") as f:
            f.write('{"models": {"codex": {"fast": "gone"}}}')
        self.cli("models", "--check")
        with open(os.path.join(self.home, "config.json")) as f:
            self.assertEqual(f.read(), '{"models": {"codex": {"fast": "gone"}}}')

    def test_falls_back_to_the_cached_catalog_and_says_when_there_is_none(self):
        self.assertEqual(self.cli("models", "--check")[0], 0)  # fetched and cached
        self.env["MAESTRO_CATALOG_URL"] = "http://127.0.0.1:1/nothing"
        rc, out, _ = self.cli("models", "--check")
        self.assertEqual(rc, 0)
        os.remove(os.path.join(self.home, "catalog.json"))
        rc, out, err = self.cli("models", "--check")
        self.assertEqual(rc, 1)
        self.assertIn("cannot reach models.dev", err)
        with open(os.path.join(self.home, "catalog.json"), "w") as f:
            json.dump({"anthropic": {"env": [], "models": {"old": 5}}}, f)  # a cache from before this format: not usable
        self.assertEqual(self.cli("models", "--check")[0], 1)

    def test_catalog_keeps_what_the_check_and_the_price_tiers_need(self):
        cat = m.fetch_catalog(self.home, fetch=lambda *a: (200, catalog_json()))
        self.assertEqual(cat["openai"]["models"]["gpt-4.1-nano"], [0.4, "gpt-nano", "2025-04-14", "deprecated", True])
        self.assertEqual(cat["google"]["models"]["gemini-3-pro-image"][4], False)
        self.assertEqual(cat["openai"]["env"], ["OPENAI_API_KEY"])


if __name__ == "__main__":
    unittest.main()
