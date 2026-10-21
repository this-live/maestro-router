import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from helpers import cls, inventory, local, maestro as m


class SeatSelection(unittest.TestCase):
    def test_claude_code_gets_a_native_subagent(self):
        s = m.select_seat("strong", cls("code", 2), inventory(), "claude-code", {}, None)
        self.assertEqual((s["kind"], s["model"]), ("subagent", "opus"))
        self.assertEqual(s["delegate"], {"kind": "subagent", "harness": "claude-code", "model": "opus"})

    def test_a_named_harness_gets_its_own_model_and_does_not_spawn_claude(self):
        policy = {"models": {"hermes": {"fast": "qwen36-35b-a3b", "standard": "MiniMax-M3",
                                        "strong": "kimi-k3", "frontier": "grok-4.7"}}}
        inv = inventory(clis=("claude-code",))
        s = m.select_seat("strong", cls("code", 2), inv, "hermes", policy, None)
        self.assertEqual(s["delegate"], {"kind": "model", "harness": "hermes", "model": "kimi-k3"})
        self.assertEqual(s["seat"], "hermes")
        self.assertNotIn("claude-code", [f["seat"] for f in s["fallback"]])

    def test_all_four_native_aliases(self):
        got = [m.select_seat(t, cls(), inventory(), "claude-code", {}, None)["model"] for t in m.TIERS]
        self.assertEqual(got, ["haiku", "sonnet", "opus", "fable"])

    def test_inline_when_tier_equals_current_tier(self):
        s = m.select_seat("standard", cls(), inventory(), "claude-code", {}, "standard")
        self.assertEqual(s["delegate"], {"kind": "inline"})
        self.assertEqual(s["kind"], "inline")

    def test_inline_when_no_seat_exists(self):
        s = m.select_seat("strong", cls(), inventory(), "unknown", {}, None)
        self.assertEqual(s["delegate"], {"kind": "inline"})

    def test_local_models_only_for_jobs_that_need_no_tools(self):
        inv = inventory(local_models=[local("llama3.2:3b", 3)])
        for name, ok in (("ask", True), ("summarize", True), ("write", True), ("plan", True), ("data", True),
                         ("edit", False), ("code", False), ("debug", False), ("review", False),
                         ("research", False), ("ops", False)):
            s = m.select_seat("local", cls(name, 0), inv, "unknown", {}, None)
            self.assertEqual(s["kind"] == "local", ok, name)

    def test_local_not_used_when_tools_needed_or_disabled(self):
        inv = inventory(local_models=[local("llama3.2:3b", 3)])
        s = m.select_seat("local", cls("write", 0, needs=("tools",)), inv, "unknown", {}, None)
        self.assertNotEqual(s["kind"], "local")
        s = m.select_seat("local", cls("ask", 0), inv, "unknown", {"prefer_local": False}, None)
        self.assertNotEqual(s["kind"], "local")
        for need in ("vision", "long_context"):
            s = m.select_seat("local", cls("ask", 0, needs=(need,)), inv, "unknown", {}, None)
            self.assertNotEqual(s["kind"], "local", need)

    def test_local_delegate_is_a_run_command(self):
        inv = inventory(local_models=[local("gpt-oss:120b", 120, endpoint="spark")])
        s = m.select_seat("standard", cls("write", 1), inv, "unknown", {}, None)
        self.assertEqual(s["delegate"], {"kind": "local", "model": "local:spark/gpt-oss:120b",
                                         "argv": [sys.executable, os.path.dirname(os.path.abspath(m.__file__)), "run",
                                                  "local:spark/gpt-oss:120b", "{task}"]})
        self.assertEqual((s["seat"], s["model"], s["category"]), ("local", "local:spark/gpt-oss:120b", "local"))

    def test_smallest_local_model_that_serves_the_cell_wins(self):
        inv = inventory(local_models=[local("small:3b", 3), local("mid:14b", 14), local("big:70b", 70)])
        self.assertEqual([x["serves"] for x in inv["local_models"]], ["local", "fast", "standard"])
        pick = lambda cell: m.select_seat(cell, cls("ask", 1), inv, "unknown", {}, None)["model"]
        self.assertEqual(pick("local"), "local:this-machine/small:3b")
        self.assertEqual(pick("fast"), "local:this-machine/mid:14b")
        self.assertEqual(pick("standard"), "local:this-machine/big:70b")
        self.assertEqual(m.select_seat("strong", cls("ask", 3), inv, "claude-code", {}, None)["model"], "opus")
        # ... but with no cloud seat at all, a local model is better than nothing
        s = m.select_seat("strong", cls("ask", 3), inv, "unknown", {}, None)
        self.assertEqual((s["kind"], s["model"]), ("local", "local:this-machine/big:70b"))

    def test_this_machine_beats_the_network_then_lower_latency(self):
        inv = inventory(local_models=[local("m:14b", 14, endpoint="spark", latency=2), local("m:14b", 14)])
        self.assertEqual(m.select_seat("fast", cls("ask", 1), inv, "unknown", {}, None)["model"], "local:this-machine/m:14b")
        inv = inventory(local_models=[local("m:14b", 14, endpoint="a", latency=9), local("m:14b", 14, endpoint="b", latency=2)])
        self.assertEqual(m.select_seat("fast", cls("ask", 1), inv, "unknown", {}, None)["model"], "local:b/m:14b")

    def test_native_subagent_does_not_outrank_local_when_prefer_local(self):
        inv = inventory(local_models=[local("small:3b", 3)])
        s = m.select_seat("local", cls("ask", 0), inv, "claude-code", {}, None)
        self.assertEqual(s["kind"], "local")
        s = m.select_seat("local", cls("ask", 0), inv, "claude-code", {"prefer_local": False}, None)
        self.assertEqual((s["kind"], s["model"]), ("subagent", "haiku"))

    def test_local_cell_without_a_local_model_is_fast(self):
        s = m.select_seat("local", cls("ask", 0), inventory(), "claude-code", {}, None)
        self.assertEqual((s["kind"], s["model"], s["category"]), ("subagent", "haiku", "fast"))

    def test_cli_fallback_with_argv_templates(self):
        inv = inventory(clis=("codex", "gemini-cli", "opencode", "claude-code"))
        pol = {"prefer": ["codex"], "models": {"codex": {"strong": "gpt-x"}}}
        s = m.select_seat("strong", cls(), inv, "unknown", pol, None)
        self.assertEqual(s["delegate"], {"kind": "cli", "argv": ["codex", "exec", "-m", "gpt-x", "{task}"]})
        self.assertNotIn("ollama", s["delegate"]["kind"])
        self.assertEqual(m.cli_argv("gemini-cli", "gem-y"), ["gemini", "-m", "gem-y", "-p", "{task}"])
        self.assertEqual(m.cli_argv("gemini-cli"), ["gemini", "-p", "{task}"])
        self.assertEqual(m.cli_argv("opencode", "p/m"), ["opencode", "run", "-m", "p/m", "{task}"])
        self.assertEqual(m.cli_argv("claude-code", "opus"), ["claude", "-p", "--model", "opus", "{task}"])

    def test_a_seat_without_any_model_uses_no_dash_m(self):
        with mock.patch.dict(m.SUGGESTED_MODELS, {"codex": {}}):  # nothing configured or suggested: the CLI's own default
            s = m.select_seat("strong", cls(), inventory(clis=("codex",)), "unknown", {}, None)
        self.assertEqual(s["delegate"]["argv"], ["codex", "exec", "{task}"])
        self.assertEqual((s["model"], s["source"]), (None, "builtin"))

    def test_current_harness_is_its_own_cli_seat_only_for_codex(self):
        inv = inventory(clis=("codex", "gemini-cli"))
        self.assertEqual(m.select_seat("strong", cls(), inv, "codex", {}, None)["delegate"]["kind"], "cli")
        names = {c["name"] for c in m.build_candidates(cls(), inv, "gemini-cli", {})}
        self.assertEqual({n for n in names if n.startswith("gemini-cli")}, {"gemini-cli:gemini-flash-lite-latest",
                         "gemini-cli:gemini-flash-latest", "gemini-cli:gemini-3.1-pro-preview",
                         "gemini-cli:gemini-3.1-pro-preview@frontier"})
        self.assertEqual(m.select_seat("strong", cls(), inv, "gemini-cli", {}, None)["kind"], "subagent")

    def test_codex_maps_every_category_through_models(self):
        inv = inventory(clis=("codex",))
        pol = {"models": {"codex": {"fast": "codex-mini", "standard": "codex-mid", "strong": "codex-top", "frontier": "codex-max"}}}
        got = {t: m.select_seat(t, cls(), inv, "codex", pol, None)["delegate"]["argv"] for t in m.TIERS}
        self.assertEqual(got["fast"], ["codex", "exec", "-m", "codex-mini", "{task}"])
        self.assertEqual(got["frontier"], ["codex", "exec", "-m", "codex-max", "{task}"])
        self.assertEqual(m.select_seat("standard", cls(), inv, "codex", pol, "standard")["kind"], "inline")

    def test_gemini_and_opencode_map_every_category_through_models(self):
        for harness in ("gemini-cli", "opencode"):
            pol = {"models": {harness: {t: "model-" + t for t in m.TIERS}}}
            for t in m.TIERS:
                s = m.select_seat(t, cls(), inventory(), harness, pol, None)
                self.assertEqual(s["delegate"], {"kind": "subagent", "harness": harness, "model": "model-" + t,
                                                 "agent": "maestro-" + t}, (harness, t))

    def test_walk_up_then_down_is_recorded(self):
        inv = inventory(clis=("codex",))
        with mock.patch.dict(m.SUGGESTED_MODELS, {"codex": {"strong": "gpt-x"}}):  # a seat that serves "strong" only
            s = m.select_seat("standard", cls(), inv, "unknown", {}, None)
            self.assertEqual((s["seat"], s["category"], s["walk"]), ("codex", "strong", ["standard", "strong"]))
            s = m.select_seat("frontier", cls(), inv, "unknown", {}, None)
            self.assertEqual((s["seat"], s["category"], s["walk"]), ("codex", "strong", ["frontier", "strong"]))

    def test_max_tier_blocks_walking_above_the_cap(self):
        inv = inventory(clis=("codex",))
        with mock.patch.dict(m.SUGGESTED_MODELS, {"codex": {}}):
            pol = {"max_tier": "standard", "models": {"codex": {"frontier": "z"}}}
            self.assertEqual(m.select_seat("fast", cls(), inv, "unknown", pol, None)["kind"], "inline")
            pol["max_tier"] = "frontier"
            s = m.select_seat("fast", cls(), inv, "unknown", pol, None)  # strong is the CLI's own default, frontier is "z"
            self.assertEqual((s["category"], s["walk"]), ("strong", ["fast", "standard", "strong"]))
            self.assertEqual(m.select_seat("frontier", cls(), inv, "unknown", pol, None)["model"], "z")

    def test_job_seats_pin(self):
        inv = inventory(clis=("gemini-cli", "codex"))
        pol = {"job_seats": {"research": "gemini"}}
        s = m.select_seat("standard", cls("research", 1), inv, "claude-code", pol, None)
        self.assertEqual(s["seat"], "gemini-cli")
        s = m.select_seat("standard", cls("code", 1), inv, "claude-code", pol, None)
        self.assertEqual(s["seat"], "claude-code")

    def test_job_seats_pin_unavailable_is_ignored(self):
        pol = {"job_seats": {"research": "gemini"}}
        s = m.select_seat("standard", cls("research", 1), inventory(), "claude-code", pol, None)
        self.assertEqual(s["seat"], "claude-code")
        self.assertIn("pin", s["note"])

    def test_opencode_subagent_works_from_suggestions_and_config_overrides_them(self):
        s = m.select_seat("strong", cls(), inventory(), "opencode", {}, "fast")
        self.assertEqual(s["delegate"], {"kind": "subagent", "harness": "opencode",
                                         "model": "anthropic/claude-opus-5-5", "agent": "maestro-strong"})
        pol = {"models": {"opencode": {"strong": "anthropic/claude-x"}}}
        s = m.select_seat("strong", cls(), inventory(), "opencode", pol, "fast")
        self.assertEqual((s["model"], s["source"]), ("anthropic/claude-x", "config"))

    def test_fallback_lists_other_seats(self):
        inv = inventory(clis=("codex",))
        s = m.select_seat("strong", cls(), inv, "claude-code", {}, None)
        self.assertEqual(s["fallback"], [{"seat": "codex", "model": "gpt-6.1-sol", "category": "strong"}])


class Route(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = {"MAESTRO_HOME": self.tmp.name, "HOME": self.tmp.name, "PATH": ""}

    def tearDown(self):
        self.tmp.cleanup()

    def test_decision_shape(self):
        d = m.route("fix the failing test in auth.py", harness="claude-code", inventory=inventory(),
                    env=self.env, home=self.tmp.name)
        for k in ("job", "difficulty", "confidence", "needs", "classifier", "category", "harness",
                  "seat", "model", "delegate", "fallback", "reason", "latency_ms"):
            self.assertIn(k, d)
        self.assertNotIn("task_class", d)
        self.assertNotIn("tier", d)
        self.assertIn(d["category"], m.CATEGORIES)
        self.assertEqual(d["classifier"], "rules")
        json.dumps(d)

    def test_explain_adds_signals_and_walk(self):
        d = m.route("fix the failing test in auth.py", harness="claude-code", inventory=inventory(),
                    env=self.env, home=self.tmp.name, explain=True)
        self.assertIn("signals", d["explain"])
        self.assertIn("walk", d["explain"])

    def test_route_writes_inventory_only_when_missing(self):
        m.route("hello", harness="claude-code", env=self.env, home=self.tmp.name)
        path = os.path.join(self.tmp.name, "inventory.json")
        self.assertTrue(os.path.exists(path))
        before = os.stat(path).st_mtime_ns
        m.route("hello", harness="claude-code", env=self.env, home=self.tmp.name)
        self.assertEqual(os.stat(path).st_mtime_ns, before)

    def test_stale_inventory_is_refreshed(self):
        path = os.path.join(self.tmp.name, "inventory.json")
        with open(path, "w") as f:
            json.dump(dict(inventory(), generated_at=1.0), f)
        m.route("hello", harness="claude-code", env=self.env, home=self.tmp.name)
        with open(path) as f:
            self.assertGreater(json.load(f)["generated_at"], 1e9)

    def test_harness_detection_from_env(self):
        self.assertEqual(m.detect_harness({"CLAUDECODE": "1"}), "claude-code")
        self.assertEqual(m.detect_harness({"OPENCODE_CONFIG": "x"}), "opencode")
        self.assertEqual(m.detect_harness({"GEMINI_CLI": "1"}), "gemini-cli")
        self.assertEqual(m.detect_harness({"CODEX_HOME": "x"}), "codex")
        self.assertEqual(m.detect_harness({}), "unknown")

    def test_jev_requested_without_key_falls_back_with_reason(self):
        d = m.route("hello", harness="claude-code", inventory=inventory(), env=self.env,
                    home=self.tmp.name, classifier="jev")
        self.assertEqual(d["classifier"], "rules")
        self.assertIn("TYPESAFE_API_KEY", d["reason"])

    def test_bad_current_tier_raises(self):
        with self.assertRaises(ValueError):
            m.route("hello", current_tier="huge", inventory=inventory(), env=self.env, home=self.tmp.name)


if __name__ == "__main__":
    unittest.main()
