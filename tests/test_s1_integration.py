"""S1 wired into Maestro (spec §12): `route` with the utility decider and shadow mode, the new CLI commands and hooks, the proxy's
utility planning, switching cost, feedback, outcome signals, /api/alpha and /v1/systemone, discover's θ and the moved modules."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import helpers
from helpers import inventory, local, maestro as m
from test_server import E, MSG, ServerCase, jcall

from maestro import commands, decide, discover, jev, nb, s1, score

ENV = {"PATH": ""}
CHAT = "what is a mutex? explain briefly"
TOOL = "fix the failing test in auth.py"


class Home(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = tmp.name
        self.inv = inventory(clis=("codex", "gemini-cli"), local_models=[local("llama3.2:3b", 3)])
        self.cfg = m.load_config(self.home)

    def route(self, text, **kw):
        cfg = dict(self.cfg, **kw.pop("cfg", {}))
        return m.route(text, harness="claude-code", inventory=self.inv, config=cfg, env=ENV, home=self.home, **kw)

    def logged(self):
        path = os.path.join(self.home, "decisions.jsonl")
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return [json.loads(line) for line in f]


class RouteUtility(Home):
    def test_utility_decides_a_chat_task_and_reports_the_decision_keys(self):
        d = self.route(CHAT, cfg={"decider": "utility"}, log=True)
        self.assertEqual(d["decider"], "utility")
        self.assertTrue(d["options"])
        top = d["options"][0]
        self.assertEqual(top["seat"], "%s:%s" % (d["seat"], d["model"]))       # the EU argmax runs the task
        self.assertAlmostEqual(d["p_success"], top["p"], places=3)
        self.assertTrue(0 <= d["confidence"] <= 1)
        for k in ("job_p", "b", "tools", "stakes", "detectable", "tok_out", "group", "source"):
            self.assertIn(k, d["posterior"])
        self.assertIn(" P ", d["reason"])
        row = self.logged()[-1]
        self.assertEqual((row["id"], row["decider"], row["chosen"]), (d["decision_id"], "utility", top["seat"]))
        self.assertEqual(len(row["minhash"]), 16)

    def test_a_tool_job_keeps_the_ladder_until_agentic_ready(self):
        ladder = self.route(TOOL)
        d = self.route(TOOL, cfg={"decider": "utility"}, log=True)
        self.assertEqual((d["seat"], d["model"]), (ladder["seat"], ladder["model"]))
        self.assertIn("agentic_ready", d["reason"])
        row = self.logged()[-1]
        self.assertEqual(row["decider"], "ladder")
        self.assertTrue(row["shadow"])

    def test_the_ladder_logs_its_pick_and_the_shadow_without_any_text(self):
        d = self.route("refactor the zanzibar billing module", log=True)
        self.assertEqual(d["decider"], "ladder")
        self.assertNotIn("options", d)                                         # shadow mode adds no keys beyond the id
        row = self.logged()[-1]
        self.assertEqual((row["id"], row["decider"]), (d["decision_id"], "ladder"))
        self.assertIn("shadow", row)
        with open(os.path.join(self.home, "decisions.jsonl")) as f:
            self.assertNotIn("zanzibar", f.read())

    def test_explain_shows_the_shadow_and_options_and_log_decisions_false_writes_nothing(self):
        d = self.route(CHAT, explain=True, cfg={"log_decisions": False}, log=True)
        self.assertTrue(d["explain"]["options"])
        self.assertIn("shadow", d["explain"])
        self.assertIn("posterior", d["explain"])
        self.assertEqual(self.logged(), [])
        self.assertNotIn("decision_id", self.route(CHAT))                     # no log, no explain: S1 does not run

    def test_hints_steer_the_posterior_and_the_ladder(self):
        d = self.route("tidy this up", cfg={"decider": "utility"}, hints={"job": "review", "stakes": "high", "size": "xl"})
        self.assertEqual(d["job"], "review")
        self.assertEqual(d["posterior"]["stakes"], 2.0)
        self.assertEqual(max(d["posterior"]["job_p"], key=d["posterior"]["job_p"].get), "review")

    def test_private_keeps_the_task_local_or_inline_under_both_deciders(self):
        for decider in ("ladder", "utility"):
            d = self.route(CHAT, cfg={"private": True, "decider": decider})
            self.assertIn(d["delegate"]["kind"], ("inline", "local"), decider)

    def test_a_key_in_the_task_makes_utility_local_only(self):
        d = self.route(CHAT + " with sk-ant-abcdefghijklmnop1234", cfg={"decider": "utility"})
        self.assertIn("private: local only", d["flags"])
        self.assertTrue(all(o["seat"].startswith("local:") for o in d["options"]))

    def test_missing_s1_weights_fall_back_to_the_nb_adapter(self):
        with mock.patch.object(s1, "load", lambda path=None: (None, "no s1.bin")):
            d = self.route(CHAT, cfg={"decider": "utility"}, explain=True)
        self.assertEqual(d["posterior"]["source"], "nb")
        self.assertEqual(d["explain"]["signals"]["s1_ignored"], "no s1.bin")
        self.assertTrue(d["options"])

    def test_a_broken_decision_layer_never_breaks_routing(self):
        with mock.patch.object(decide, "decide", side_effect=ZeroDivisionError):
            d = self.route(CHAT, cfg={"decider": "utility"})
        self.assertIn("utility decider failed (ZeroDivisionError)", d["reason"])
        self.assertTrue(d["seat"])

    def test_classifier_s1_decodes_job_and_difficulty(self):
        self.assertEqual(self.route(TOOL, classifier="s1")["classifier"], "s1")

    def test_the_moved_modules_are_re_exported_under_the_same_names(self):
        self.assertIs(m.classify_model, nb.classify_model)
        self.assertIs(m.load_model, nb.load_model)
        self.assertIs(m.JevClassifier, jev.JevClassifier)
        self.assertIs(m.choose_jev, jev.choose_jev)


class Commands(Home):
    def cli(self, *argv, stdin=None):
        out, err = io.StringIO(), io.StringIO()
        with open(os.path.join(self.home, "inventory.json"), "w") as f:
            json.dump(self.inv, f)
        env = {"MAESTRO_HOME": self.home, "HOME": self.home, "PATH": ""}
        with mock.patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                mock.patch.object(sys, "stdin", io.StringIO(stdin or "")):
            rc = m.main(list(argv))
        return rc, out.getvalue(), err.getvalue()

    def test_route_explain_text_prints_the_options_table_then_outcome_records(self):
        rc, out, _ = self.cli("route", CHAT, "--harness", "claude-code", "--decider", "utility", "--job", "ask", "--stakes", "low", "--explain", "--format", "text")
        self.assertEqual(rc, 0)
        self.assertIn("posterior (s1)", out)
        self.assertIn("EU", out)
        self.assertIn("shadow (the other decider)", out)
        rc, out, err = self.cli("outcome", "last", "fail", "--tests", "3/5")
        self.assertEqual(rc, 0, err)
        with open(os.path.join(self.home, "outcomes.jsonl")) as f:
            row = json.loads(f.readline())
        self.assertEqual((row["decision_id"], row["value"], row["detail"]), (self.logged()[-1]["id"], 0.0, {"tests": [3, 5]}))

    def test_s1_command_prints_the_heads(self):
        rc, out, _ = self.cli("s1", "--json", TOOL)
        self.assertEqual(rc, 0)
        post = json.loads(out)
        self.assertAlmostEqual(sum(post["job_p"].values()), 1.0, places=3)
        rc, out, _ = self.cli("s1", TOOL)
        self.assertIn("difficulty b:", out)

    def test_decide_answers_a_systemone_request_from_stdin(self):
        body = {"model": "maestro-s1", "state": TOOL, "questions": {"job": {"type": "choice", "criteria": {"debug": "d", "write": "w"}}}}
        rc, out, _ = self.cli("decide", "-", stdin=json.dumps(body))
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out)["answers"]["job"]["choice"], "debug")

    def test_install_hooks_adds_the_subagent_stop_hook_once_and_keeps_the_rest(self):
        path = os.path.join(self.home, ".claude", "settings.json")
        os.makedirs(os.path.dirname(path))
        with open(path, "w") as f:
            json.dump({"model": "opus", "hooks": {"PreToolUse": [{"hooks": [{"type": "command", "command": "x"}]}]}}, f)
        self.assertEqual(commands.install_hooks(self.home, dry_run=True)[1], "would add")
        self.assertEqual(commands.install_hooks(self.home)[1], "added")
        self.assertEqual(commands.install_hooks(self.home)[1], "unchanged")
        with open(path) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["model"], "opus")
        self.assertIn("PreToolUse", cfg["hooks"])
        cmd = cfg["hooks"]["SubagentStop"][0]["hooks"][0]["command"]
        self.assertTrue(cmd.endswith("outcome --from-hook"))
        self.assertTrue(os.path.exists(path + ".maestro-bak"))
        with open(path, "w") as f:
            f.write("{not json")
        with self.assertRaises(ValueError):
            commands.install_hooks(self.home)

    def test_the_hook_command_never_fails_the_agent(self):
        rc, _, _ = self.cli("outcome", "--from-hook", stdin="not json")
        self.assertEqual(rc, 0)


class Imports(unittest.TestCase):
    def test_route_with_the_utility_decider_imports_none_of_the_slow_or_learning_modules(self):
        with tempfile.TemporaryDirectory() as home:
            code = ("import sys; sys.path.insert(0, %r); import maestro; cfg = maestro.load_config(%r); cfg['decider'] = 'utility'; "
                    "inv = {'version': 2, 'generated_at': 9e12, 'harnesses': {}, 'api_keys': {}, 'local_endpoints': [], 'local_models': [], "
                    "'endpoints_sig': [], 'hardware': {}, 'warnings': []}; "
                    "maestro.route('what is a mutex', harness='claude-code', inventory=inv, config=cfg, explain=True, log=True); "
                    "maestro.route('fix the failing test', harness='claude-code', inventory=inv, config=cfg, current_tier='strong', log=True); "
                    "print(sorted(m for m in sys.modules if m.startswith('maestro.') and m.split('.')[1] in ('providers', 'clients', 'quota', "
                    "'catalog', 'score', 'discover', 'server', 'onboard', 'learn', 'systemone')))") % (
                os.path.join(helpers.ROOT, "skills", "maestro", "scripts"), home)
            res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env=dict(os.environ, MAESTRO_HOME=home))
        self.assertEqual(res.stdout.strip(), "[]", res.stderr)


class Proxy(ServerCase):
    def two(self, cfg=None, b_cost=(0.9, 3.6), **kw):
        up = self.up(name="openai", text="pong")
        models = [E("openai/a", "standard", 0.6, (1.0, 4.0)), E("openai/b", "standard", 0.6, b_cost)]
        self.start(models, dict({"providers": {"openai": {"base_url": up.url}}}, **(cfg or {})), **kw)
        return up

    def outcomes(self):
        import sqlite3
        db = sqlite3.connect(os.path.join(self.home, "maestro.db"))
        return db.execute("select decision_id, signal, value, weight, detail from outcomes order by id").fetchall()

    def test_utility_picks_by_expected_utility_and_the_header_carries_the_decision(self):
        self.two({"decider": "utility"})
        st, h, j = self.chat(MSG)
        self.assertEqual(st, 200)
        d = self.decision(h)
        self.assertEqual((d["decider"], d["model"]), ("utility", "openai/b"))      # same P, cheaper
        self.assertIsInstance(d["decision_id"], int)
        self.assertTrue(0 < d["p_success"] <= 1)
        self.assertIn("escalate_to", d)
        row = self.rows()[0]
        self.assertEqual((row["id"], row["decider"], row["chosen"]), (d["decision_id"], "utility", "openai/b"))
        self.assertLessEqual(len(row["s1"]), 400)
        self.assertEqual(set(json.loads(row["s1"])), {"job_p", "b", "group", "trace"})
        self.assertTrue(row["policy"].startswith("s1:"))
        self.assertEqual((row["explore"], row["propensity"]), (0, 1.0))
        self.assertEqual(row["status"], 200)
        self.assertEqual(row["prompt_tokens"], 5)                                 # finish() completed the early row

    def test_the_ladder_logs_the_utility_pick_as_its_shadow(self):
        self.two()
        st, h, _ = self.chat(MSG)
        d = self.decision(h)
        self.assertEqual(d["decider"], "ladder")
        row = self.rows()[0]
        self.assertEqual(row["decider"], "ladder")
        self.assertIn(row["shadow"], ("openai/a", "openai/b"))
        st, _, a = self.admin("GET", "/api/alpha")
        self.assertEqual((st, a["proxy"]["n"]), (200, 1))

    def test_a_warm_seat_keeps_a_long_conversation_because_switching_loses_the_cache(self):
        self.two({"decider": "utility"})
        long = {"messages": [{"role": "system", "content": "context " * 15000}, {"role": "user", "content": "first question"},
                             {"role": "assistant", "content": "an answer"}, {"role": "user", "content": "and the follow-up?"}]}
        st, h, _ = self.chat(long)
        self.assertEqual(self.decision(h)["model"], "openai/b")                   # cold: the cheaper seat
        self.app.aff_put("s:conv-1|maestro/auto", {"id": "openai/a", "job": "ask", "difficulty": 0, "cell": "fast"}, time.time())
        st, h, _ = self.chat(long, {"X-Session-Id": "conv-1"})
        d = self.decision(h)
        self.assertEqual(d["model"], "openai/a")                                 # warm: cache reads beat the cheaper list price
        self.assertTrue(d["affinity"])
        self.assertEqual(self.app.aff_get("s:conv-1|maestro/auto", time.time())["id"], "openai/a")   # stored on every turn

    def test_feedback_by_api_and_header_and_tests_in_tool_results(self):
        self.two()
        st, h, _ = self.chat(MSG, {"X-Session-Id": "c"})
        first = self.decision(h)["decision_id"]
        self.assertEqual(self.admin("POST", "/api/feedback", {"id": first, "value": "bad"})[0], 200)
        self.assertEqual(self.admin("POST", "/api/feedback", {"id": 999, "value": "bad"})[0], 404)
        self.assertEqual(self.admin("POST", "/api/feedback", {"id": first, "value": "meh"})[0], 400)
        tool = {"messages": MSG["messages"] + [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "sh", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "collected 6 items\n===== 5 passed, 1 failed in 0.4s ====="}]}
        st, h, _ = self.chat(dict(tool, tools=[{"type": "function", "function": {"name": "sh", "parameters": {"type": "object"}}}]),
                             {"X-Session-Id": "c", "X-Maestro-Feedback": "%d=good" % first})
        self.assertEqual(st, 200)
        got = [(d, s, round(v, 3)) for d, s, v, _, _ in self.outcomes()]
        self.assertIn((first, "explicit", 0.0), got)
        self.assertIn((first, "explicit", 1.0), got)
        self.assertIn((first, "tests", round(5 / 6.0, 3)), got)

    def test_private_mode_with_no_local_model_is_a_503(self):
        for decider in ("utility", "ladder"):
            self.stop()
            self.two({"decider": decider, "private": True})
            st, _, j = self.chat(MSG)
            self.assertEqual((st, j["error"]["code"]), (503, "private_no_local"), decider)

    def test_systemone_and_models_list(self):
        self.two()
        st, _, j = jcall(self.port, "POST", "/v1/systemone", {"model": "maestro-s1", "state": "fix the failing login test", "questions": {
            "job": {"type": "choice", "criteria": {"debug": "find a bug", "write": "prose"}},
            "seat": {"type": "choice", "intent": "maestro.model", "criteria": {"openai/a": "a", "openai/b": "b"}}}})
        self.assertEqual(st, 200, j)
        self.assertEqual(j["answers"]["job"]["choice"], "debug")
        self.assertIn(j["answers"]["seat"]["choice"], ("openai/a", "openai/b"))
        self.assertEqual(j["maestro"]["source"], {"job": "native", "seat": "native"})
        st, _, j = jcall(self.port, "POST", "/v1/systemone", {"model": "maestro-s1", "state": "x", "questions": {"q": {"type": "noul", "criteria": "?"}}})
        self.assertEqual(st, 422)
        st, _, j = jcall(self.port, "GET", "/v1/models")
        self.assertIn("maestro-s1", [x["name"] for x in j["models"]])

    def test_systemone_needs_the_api_key_like_the_rest_of_v1(self):
        self.two(env_extra={"MAESTRO_API_KEY": "proxy-key"})
        body = {"model": "maestro-s1", "state": "x", "questions": {"job": {"type": "choice", "criteria": {"debug": "d", "ask": "a"}}}}
        self.assertEqual(jcall(self.port, "POST", "/v1/systemone", body)[0], 401)
        self.assertEqual(jcall(self.port, "POST", "/v1/systemone", body, {"Authorization": "Bearer proxy-key"})[0], 200)

    def test_config_validates_the_s1_keys_and_learn_runs(self):
        self.two()
        st, _, j = self.admin("POST", "/api/config", {"decider": "utility", "explore": 0.05, "risk": 2})
        self.assertEqual((st, j["config"]["decider"], j["config"]["explore"]), (200, "utility", 0.05))
        for bad in ({"decider": "x"}, {"explore": 0.5}, {"quota_reserve": 0.9}, {"wtp_usd": 0}, {"private": "yes"}):
            self.assertEqual(self.admin("POST", "/api/config", bad)[0], 400, bad)
        self.assertEqual(self.admin("GET", "/api/state")[2]["config"]["decider"], "utility")
        rep = self.app.learn_once()
        self.assertFalse(rep["written"])
        self.assertIn("not enough outcomes", rep["why"])

    def test_dry_run_shows_the_options(self):
        self.two({"decider": "utility"})
        st, _, j = self.admin("POST", "/api/route", {"prompt": "what is a mutex?"})
        self.assertEqual((st, j["decider"]), (200, "utility"))
        self.assertEqual({o["seat"] for o in j["options"]}, {"openai/a", "openai/b"})
        self.assertIn("b", j["posterior"])


class DiscoverAndScore(unittest.TestCase):
    def test_discover_stamps_theta_under_the_current_link_and_decide_uses_it(self):
        e = {"id": "openai/gpt-x", "access": "api", "model": "gpt-x", "provider": "openai", "jobs": {j: 0.7 for j in m.JOBS}, "G": 0.7,
             "basis": "bench"}
        discover.stamp_theta([e])
        prm = decide.params()
        self.assertEqual(set(e["theta"]), set(m.JOBS))
        self.assertEqual(e["theta_link"], prm["link"]["sha"])
        self.assertEqual(e["theta"]["code"][1], prm["link"]["sd_basis"]["bench"])
        mu, sd, linked = decide.theta(decide.seat_of(dict(e, theta={j: [9.0, 0.1] for j in m.JOBS})), {"code": 1.0}, prm)
        self.assertEqual((mu, sd, linked), (9.0, 0.1, True))                      # the precomputed θ wins
        stale = dict(e, theta={j: [9.0, 0.1] for j in m.JOBS}, theta_link="00000000")
        self.assertNotEqual(decide.theta(decide.seat_of(stale), {"code": 1.0}, prm)[0], 9.0)   # another link: recomputed

    def test_score_rec_returns_the_blended_axes(self):
        from maestro import catalog
        cat = catalog.load_snapshot()
        catalog._finish(cat)
        ctx = score.make_context(cat)
        lab = next(iter(cat["models"]))
        res = score.score_rec(ctx, catalog.rec_of(cat, lab))
        self.assertEqual(set(res["axes8"]), set(score.AXES))
        self.assertTrue(set(res["axes_n"]) <= set(score.AXES))

    def test_catalog_sha_is_the_bundled_snapshot_hash(self):
        from maestro import catalog
        sha = catalog.snapshot_sha()
        self.assertEqual(len(sha), 64)
        self.assertIsNone(catalog.snapshot_sha("/nonexistent"))


if __name__ == "__main__":
    unittest.main()
