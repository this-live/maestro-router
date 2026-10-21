"""The arena (evals/arena.py, evals/arena/): seats, the argv table, grading, agentic runs in a temp git repo (fake harness
CLIs on a temp PATH), open-suite ranking against a fake OpenAI-compatible server, Bradley-Terry, resume, budget, quota,
report numbers and the export format. No network beyond 127.0.0.1; the real `claude` is never on the test PATH."""
import contextlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

import helpers  # noqa: F401
from fake_http import FakeServer

ROOT = helpers.ROOT
sys.path.insert(0, os.path.join(ROOT, "evals"))
import alpha_eval  # noqa: E402
from arena import cli, export, harness, rank, report, runner, suites  # noqa: E402
from arena import seats as S  # noqa: E402
from maestro import quota  # noqa: E402

m = helpers.maestro
FIXTURES = os.path.join(ROOT, "tests", "data", "arena_fixtures")

FAKE_CLAUDE = r'''#!%(py)s
import json, os, sys, time
args = sys.argv[1:]
if args == ["--help"]:
    print("  --safe-mode   Start with all customizations disabled")
    sys.exit(0)
model = args[args.index("--model") + 1]
prompt = sys.stdin.read()
def result(text, cost):
    print(json.dumps({"type": "result", "is_error": False, "result": text, "total_cost_usd": cost, "num_turns": 2,
                      "usage": {"input_tokens": 50, "cache_read_input_tokens": 5, "output_tokens": 10}}))
if "--tools" in args:                                    # chat mode
    if model == "erroring":
        print(json.dumps({"type": "result", "is_error": True, "result": "boom"}))
        sys.exit(1)
    if model == "solver":
        result("Here you go:\n```python\ndef add(a, b):\n    return a + b\n```\nDone.", 0.01)
    else:
        result("```python\ndef add(a, b):\n    return a - b\n```", 0.01)
    sys.exit(0)
assert "--permission-mode" in args and "acceptEdits" in args and "--allowedTools" in args
if model == "sleeper":
    time.sleep(60)
if model in ("solver", "escaper"):
    src = open("calc.py").read().replace("a - b", "a + b")
    open("calc.py", "w").write(src)
if model == "escaper":
    open(os.path.join("..", "escaped.txt"), "w").write("outside")
if model == "wanderer":                                  # like OpenCode 2.x: works in $PWD, not in its real cwd
    open(os.path.join(os.environ["PWD"], "wandered.txt"), "w").write("here")
if model == "toucher":                                   # edits the fixture source itself
    open(os.path.join(os.environ["ARENA_TEST_FIXTURE"], "touched.txt"), "w").write("x")
if model == "crasher":
    sys.stderr.write("not logged in\n")
    sys.exit(3)
result("changed calc.py" if model != "failer" else "looked around", 0.05)
'''

CHECK_STRICT = ('import re\ndef check(answer):\n    if not answer.lstrip().startswith("```python"):\n        return False\n'
                '    code = re.findall(r"```python\\n(.*?)```", answer, re.S)[-1]\n    ns = {}\n    exec(code, ns)\n'
                '    return ns["add"](2, 3) == 5\n')
CHECK_PLAIN = 'def check(answer):\n    return "a + b" in answer\n'


def blocks_of(prompt):
    return re.findall(r"Answer ([A-Z]):\n<<<\n(.*?)\n>>>", prompt, re.S)


def fake_llm(req):
    """A fake OpenAI-compatible server: contestants m-good / m-bad, judge-a ranks GOOD first, judge-b ranks in the order
    shown (pure position bias); both accept an answer iff it says GOOD."""
    if req["method"] == "GET":
        if req["path"] == "/v1/models":
            return 200, {}, {"data": [{"id": "qwen3-27b"}, {"id": "m-good"}]}
        return 404, {}, {"error": "no"}
    body = req["json"]
    model, prompt = body["model"], body["messages"][-1]["content"]
    if "You write grading criteria" in prompt:
        text = json.dumps({"must": ["says GOOD"], "disqualifiers": [], "sketch": "a GOOD answer"})
    elif "You compare anonymous answers" in prompt:
        bl = blocks_of(prompt)
        labels = [lab for lab, _ in bl]
        if model == "judge-a":
            labels = sorted(labels, key=lambda lab: ("GOOD" not in dict(bl)[lab], lab))
        text = json.dumps({"answers": {lab: {"must": ["GOOD" in t], "disq": []} for lab, t in bl}, "ranking": labels})
    elif model == "m-good":
        text = "GOOD answer, with a few more words than the other one."
    elif model == "qwen3-27b":
        text = "served by the spark"
    else:
        text = "bad"
    return 200, {}, {"choices": [{"message": {"role": "assistant", "content": text}}],
                     "usage": {"prompt_tokens": 20, "completion_tokens": 7}}


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="arena-test-")
        self.home = os.path.join(self.d, "home")
        self.bin = os.path.join(self.d, "bin")
        self.runs = os.path.join(self.d, "runs")
        os.makedirs(self.home)
        os.makedirs(self.bin)
        with open(os.path.join(self.bin, "claude"), "w") as f:
            f.write(FAKE_CLAUDE % {"py": sys.executable})
        os.chmod(os.path.join(self.bin, "claude"), 0o755)
        path = os.pathsep.join([self.bin, os.path.dirname(sys.executable), "/usr/bin", "/bin"])
        self.env = mock.patch.dict(os.environ, {"PATH": path, "MAESTRO_HOME": self.home})
        self.env.start()
        self.inv = mock.patch.object(m, "ensure_inventory", lambda *a, **k: helpers.inventory())
        self.inv.start()
        self.seats_file = os.path.join(self.d, "seats.json")
        self.write_seats([{"harness": "claude-code", "model": x} for x in
                          ("solver", "failer", "sleeper", "escaper", "crasher", "erroring")])

    def tearDown(self):
        self.inv.stop()
        self.env.stop()
        quota.reset_state(self.home)
        shutil.rmtree(self.d, ignore_errors=True)

    def write_seats(self, add, **extra):
        with open(self.seats_file, "w") as f:
            json.dump(dict({"add": add}, **extra), f)

    def items(self):
        p = os.path.join(self.d, "items.jsonl")
        with open(p, "w") as f:
            f.write(json.dumps({"id": "v-strict", "type": "code-fn", "difficulty_intent": 1, "prompt": "write add(a, b)",
                                "checker_py": CHECK_STRICT}) + "\n")
            f.write(json.dumps({"id": "v-plain", "type": "code-fn", "difficulty_intent": 0, "prompt": "write add again",
                                "checker_py": CHECK_PLAIN}) + "\n")
        return p

    def run_args(self, *extra, run_id="r1"):
        a = cli.parser().parse_args(["run", "--runs-dir", self.runs, "--seats-file", self.seats_file, "--run-id", run_id,
                                     "--fixtures", FIXTURES] + list(extra))
        logs = []
        rc = runner.run(a, log=logs.append)
        return rc, logs, runner.run_paths(run_id, self.runs)


class Bradley(unittest.TestCase):
    def test_hand_computed_three_players(self):
        # A beats B 2:1, B beats C 2:1, A beats C 4:1: strengths 4:2:1 satisfy the likelihood equations exactly
        # (A: 3*4/6 + 5*4/5 = 6 wins; B: 3*2/6 + 3*2/3 = 3; C: 3/3 + 5/5 = 2), so log-strengths are log 2, 0, -log 2.
        w = {("A", "B"): 2, ("B", "A"): 1, ("B", "C"): 2, ("C", "B"): 1, ("A", "C"): 4, ("C", "A"): 1}
        sc = rank.bradley_terry(w)
        self.assertAlmostEqual(sc["A"], math.log(2), places=6)
        self.assertAlmostEqual(sc["B"], 0.0, places=6)
        self.assertAlmostEqual(sc["C"], -math.log(2), places=6)

    def test_two_players_and_prior(self):
        sc = rank.bradley_terry({("A", "B"): 3, ("B", "A"): 1})
        self.assertAlmostEqual(sc["A"] - sc["B"], math.log(3), places=6)
        sc = rank.bradley_terry({("A", "B"): 2}, prior=0.5)          # unbeaten stays finite: 2.5 : 0.5
        self.assertAlmostEqual(sc["A"] - sc["B"], math.log(5), places=6)
        wp = rank.win_prob(rank.bradley_terry({("A", "B"): 3, ("B", "A"): 1}))
        self.assertAlmostEqual(wp["A"], 0.75, places=6)

    def test_position_bias_cancels_with_the_reversed_pass(self):
        seats = ["s1", "s2", "s3", "s4"]
        rows = []
        for t in ("t1", "t2", "t3"):
            for p, order in enumerate(rank.orders(t, 1, "judge", seats)):
                rows.append({"task": t, "rep": 1, "judge": "judge", "pass": p, "order": order, "ranking": list(order), "ok": True})
        self.assertEqual(rank.orders("t1", 1, "judge", seats)[1], list(reversed(rank.orders("t1", 1, "judge", seats)[0])))
        scores, _ = rank.bt_with_ci(rows, n_boot=50)
        for s in seats:
            self.assertAlmostEqual(scores[s][0], 0.0, places=6)
        d = rank.diagnostics(rows, {}, {})
        self.assertEqual(d["position_first"], (1.0, 0.25))           # the bias is visible in the checks
        self.assertAlmostEqual(d["pass_tau"], -1.0)

    def test_parse_judgment(self):
        crit = {"must": ["a", "b"], "disq": ["x"]}
        ranking, acc = rank.parse_judgment('{"answers": {"A": {"must": [true, true], "disq": [false]}, '
                                           '"B": {"must": [true, false], "disq": [false]}, "C": {"must": [true], "disq": []}},'
                                           ' "ranking": ["B", "A"]}', ["sa", "sb", "sc"], crit)
        self.assertEqual(ranking, ["sb", "sa", "sc"])               # one omitted answer goes last
        self.assertEqual(acc, {"sa": True, "sb": False, "sc": None})  # wrong lengths: unreadable, not a fail
        with self.assertRaises(ValueError):
            rank.parse_judgment('{"ranking": ["A"]}', ["sa", "sb", "sc"], crit)

    def test_anonymize_strips_every_family(self):
        t = rank.anonymize("As Claude, I think GPT-5 and Gemini Pro agree. Qwen3 too.\n```python\nimport openai\n```")
        for w in ("Claude", "GPT-5", "Gemini", "Qwen3"):
            self.assertNotIn(w, t)
        self.assertIn("import openai", t)                            # code is left alone


class Grading(unittest.TestCase):
    def test_extraction_variants(self):
        ans = "Sure:\n```python\ndef add(a, b):\n    return a + b\n```\nand\n```sql\nSELECT 1;\n```\n```json\n{\"k\": 1}\n```"
        v = dict(suites.variants(ans))
        self.assertTrue(v["function"].startswith("```python\ndef add"))
        self.assertIn("SELECT 1;", v["sql"])
        self.assertIn('{"k": 1}', v["json"])
        g = suites.grade_verifiable(CHECK_STRICT, ans)
        self.assertEqual((g["y"], g["y_strict"], g["variant"]), (1, 0, "function"))
        self.assertEqual(suites.grade_verifiable(CHECK_PLAIN, ans)["variant"], "raw")
        self.assertEqual(suites.grade_verifiable(CHECK_PLAIN, "nothing")["y"], 0)


class Argv(unittest.TestCase):
    def test_table(self):
        with mock.patch.object(harness, "supports", lambda exe, sub, flag: True):
            self._table()
        with mock.patch.object(harness, "supports", lambda exe, sub, flag: False):   # an older CLI without the flags
            self.assertEqual(harness.build_argv({"harness": "opencode", "model": "spark/q"}, "chat", "p", "/x/opencode")[0],
                             ["/x/opencode", "run", "-m", "spark/q", "p"])
            self.assertNotIn("--safe-mode", harness.build_argv({"harness": "claude-code", "model": "h"}, "chat", "p", "/x/c")[0])

    def test_supports_reads_the_cli_help_once(self):
        d = tempfile.mkdtemp()
        try:
            exe, log = os.path.join(d, "oc"), os.path.join(d, "calls")
            with open(exe, "w") as f:
                f.write("#!/bin/sh\necho x >> %s\n[ \"$1 $2\" = \"run --help\" ] && echo '  --standalone  private server'\nexit 0\n" % log)
            os.chmod(exe, 0o755)
            self.assertTrue(harness.supports(exe, ("run",), "--standalone"))
            self.assertTrue(harness.supports(exe, ("run",), "--standalone"))
            self.assertFalse(harness.supports(exe, ("run",), "--safe-mode"))
            self.assertFalse(harness.supports(os.path.join(d, "missing"), ("run",), "--standalone"))
            with open(log) as f:
                self.assertEqual(len(f.read().split()), 1)            # cached per executable and subcommand
        finally:
            shutil.rmtree(d)

    def _table(self):
        argv, stdin = harness.build_argv({"harness": "claude-code", "model": "opus", "effort": "high"}, "agentic", "do it", "/x/claude")
        self.assertEqual(argv[:5], ["/x/claude", "-p", "--safe-mode", "--model", "opus"])
        for flag in ("--effort", "high", "acceptEdits", harness.ALLOWED_TOOLS, "--output-format", "json"):
            self.assertIn(flag, argv)
        self.assertEqual(stdin, "do it")
        argv, _ = harness.build_argv({"harness": "claude-code", "model": "haiku"}, "chat", "q")
        self.assertEqual(argv[argv.index("--tools") + 1], "")
        self.assertNotIn("--effort", argv)
        for mode in ("agentic", "chat"):   # the user's plugins, skills, hooks and MCP servers stay out of every cell
            self.assertEqual(harness.build_argv({"harness": "claude-code", "model": "haiku"}, mode, "q")[0][1:3],
                             ["-p", "--safe-mode"])
        argv, stdin = harness.build_argv({"harness": "codex", "model": "gpt-6.1-sol", "effort": "high"}, "agentic", "-x")
        self.assertEqual(argv[-1], " -x")
        self.assertIn("model_reasoning_effort=high", argv)
        self.assertIn("workspace-write", argv)
        self.assertIsNone(stdin)
        self.assertIn("--yolo", harness.build_argv({"harness": "gemini-cli", "model": "g"}, "agentic", "p")[0])
        self.assertNotIn("--yolo", harness.build_argv({"harness": "gemini-cli", "model": "g"}, "chat", "p")[0])
        for mode in ("agentic", "chat"):   # without -Q, `hermes chat -q` prints a banner, a boxed answer and resume hints
            argv, stdin = harness.build_argv({"harness": "hermes", "model": "m"}, mode, "p", "/x/hermes")
            self.assertEqual(argv, ["/x/hermes", "chat", "-Q", "-q", "p"])
            self.assertIsNone(stdin)
            # OpenCode 2.x runs against a shared background service unless --standalone: that service keeps the first
            # cell's OPENCODE_CONFIG and env, so a later seat would silently reach the wrong provider
            argv, _ = harness.build_argv({"harness": "opencode", "model": "spark/q"}, mode, "p", "/x/opencode")
            self.assertEqual(argv, ["/x/opencode", "run", "--standalone", "-m", "spark/q", "p"])
            argv, _ = harness.build_argv({"harness": "opencode", "model": ""}, mode, "p", "/x/opencode")
            self.assertEqual(argv, ["/x/opencode", "run", "--standalone", "p"])

    def test_opencode_local_config_names_the_key_never_the_value(self):
        d = tempfile.mkdtemp()
        try:
            seat = {"harness": "opencode", "model": "spark/gpt-oss-120b",
                    "endpoint": {"name": "spark", "base_url": "http://spark:8000/v1", "served": "gpt-oss-120b", "api_key_env": "SPARK_KEY"}}
            env = harness.harness_env(seat, d, {"SPARK_KEY": "sk-secret-value"})
            with open(env["OPENCODE_CONFIG"]) as f:
                text = f.read()
            self.assertIn("{env:SPARK_KEY}", text)
            self.assertNotIn("sk-secret-value", text)
            self.assertEqual(json.loads(text)["provider"]["spark"]["options"]["baseURL"], "http://spark:8000/v1")
        finally:
            shutil.rmtree(d)

    def test_parsers(self):
        out = "\n".join(json.dumps(e) for e in (
            {"type": "item.completed", "item": {"type": "agent_message", "text": "hi"}},
            {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 3}}))
        self.assertEqual(harness.parse_output("codex-jsonl", out)[:2], ("hi", {"in": 10, "out": 3}))
        g = json.dumps({"response": "ok", "stats": {"models": {"m": {"tokens": {"prompt": 4, "candidates": 2}}}}})
        self.assertEqual(harness.parse_output("gemini-json", g)[:2], ("ok", {"in": 4, "out": 2}))
        self.assertEqual(harness.parse_output("claude-json", "plain")[0], "plain")   # falls back to text
        warn = "  \u26a0 tirith security scanner enabled but not available \u2014 pattern matching only\n"
        self.assertEqual(harness.parse_output("hermes-text", warn + "ok\n")[0], "ok")   # startup warning is not the answer
        self.assertEqual(harness.parse_output("hermes-text", "ok\n\u26a0 later line\n")[0], "ok\n\u26a0 later line")


class Seats(Base):
    def test_discover_cli_local_opencode_and_file(self):
        with open(os.path.join(self.bin, "opencode"), "w") as f:
            f.write("#!/bin/sh\nexit 0\n")
        os.chmod(os.path.join(self.bin, "opencode"), 0o755)
        inv = helpers.inventory(local_models=[dict(helpers.local("qwen3-27b", 27, endpoint="spark"), kind="openai"),
                                              helpers.local("cold:7b", 7, endpoint="spark", loaded=False)])
        self.write_seats([{"harness": "api", "provider": "openai", "model": "gpt-6.1-sol", "id": "api:openai/gpt-6.1-sol"}],
                         remove=["claude-code:fable"])
        got = {s["id"]: s for s in S.discover(inventory=inv, seats_file=self.seats_file, cfg={})}
        self.assertIn("claude-code:haiku", got)
        self.assertNotIn("claude-code:fable", got)                          # removed by seats.json
        self.assertEqual(got["local:spark/qwen3-27b"]["mode"], "chat")
        self.assertNotIn("local:spark/cold:7b", got)                        # installed, not loaded: never a seat
        oc = got["opencode:spark/qwen3-27b"]
        self.assertEqual((oc["mode"], oc["endpoint"]["served"]), ("agentic", "qwen3-27b"))
        self.assertIn("api:openai/gpt-6.1-sol", got)
        for s in got.values():
            S.annotate(s)
        judges, note = S.default_judges(list(got.values()))
        self.assertGreaterEqual(len({j["family"] for j in judges}), 2)
        self.assertEqual(note, "")
        self.assertEqual(S.quota_scope(got["claude-code:haiku"]), "claude")
        self.assertIsNone(S.quota_scope(got["local:spark/qwen3-27b"]))

    def test_hermes_seats_and_config(self):
        with open(os.path.join(self.bin, "hermes"), "w") as f:
            f.write("#!/bin/sh\nexit 0\n")
        os.chmod(os.path.join(self.bin, "hermes"), 0o755)
        inv = helpers.inventory(local_models=[dict(helpers.local("qwen3-27b", 27, endpoint="spark"), kind="openai")])
        got = {s["id"]: s for s in S.discover(inventory=inv, seats_file=None, cfg={})}
        h = got["hermes:spark/qwen3-27b"]
        self.assertEqual(h["endpoint"]["base_url"], "http://spark:11434/v1")
        mx = S.annotate(got["hermes:maestro/auto"])
        self.assertEqual((mx["family"], mx["price_in"]), ("maestro", None))   # spends what Maestro picks: cost unknown
        self.assertEqual(harness.cost_of(mx, {"in": 10, "out": 10}, None, "p", "t"), (None, "unknown"))
        d = tempfile.mkdtemp()
        try:
            env = harness.harness_env(h, d, {})
            with open(os.path.join(env["HERMES_HOME"], "config.yaml")) as f:
                y = f.read()
            self.assertIn("default: qwen3-27b", y)
            self.assertIn("base_url: http://spark:11434/v1", y)
        finally:
            shutil.rmtree(d)
        with self.assertRaises(ValueError):
            S.normalize({"harness": "opencode", "model": "x/y", "endpoint": {"name": "x"}})

    def test_single_family_is_flagged(self):
        seats = [S.annotate(S.normalize({"harness": "claude-code", "model": x})) for x in ("haiku", "opus")]
        judges, note = S.default_judges(seats)
        self.assertIn("SINGLE-FAMILY", note)
        self.assertEqual(judges[0]["id"], "claude-code:opus")

    def test_local_seat_through_maestro_run(self):
        with FakeServer(fake_llm) as srv:
            rec = dict(helpers.local("qwen3-27b", 27, endpoint="spark"), url=srv.url, kind="openai")
            cfg = dict(m.DEFAULT_CONFIG, local_endpoints=[{"url": srv.url, "name": "spark", "kind": "openai"}])
            seat = S.normalize({"harness": "local", "model": rec["id"], "id": rec["id"]})
            res = harness.ask(seat, "hello", 5, cfg=cfg, inventory=helpers.inventory(local_models=[rec]))
        self.assertTrue(res["ok"], res)
        self.assertEqual((res["text"], res["cost_usd"], res["cost_source"]), ("served by the spark", 0.0, "local"))


class Runs(Base):
    def test_agentic_solve_fail_timeout_escape(self):
        rc, logs, p = self.run_args("--suite", "agentic", "--seats", "claude-code:solver,claude-code:failer,claude-code:sleeper,"
                                    "claude-code:escaper,claude-code:crasher", "--agentic-timeout", "2", "--concurrency", "3")
        rows = {r["seat"].split(":")[1]: r for r in suites.read_jsonl(p["results"])}
        self.assertEqual(rows["solver"]["y"], 1)
        self.assertEqual(rows["solver"]["diff"], {"files": 1, "added": 1, "removed": 1})
        self.assertEqual((rows["solver"]["cost_usd"], rows["solver"]["cost_source"]), (0.05, "reported"))
        self.assertEqual(rows["solver"]["final"], "changed calc.py")
        self.assertEqual(rows["failer"]["y"], 0)
        self.assertTrue(rows["sleeper"]["timed_out"])
        self.assertEqual((rows["sleeper"]["y"], rows["sleeper"]["status"]), (0, "ok"))
        self.assertEqual(rows["escaper"]["escapes"], ["escaped.txt"])        # contained in the cell dir and flagged
        self.assertTrue(rows["escaper"]["tests_passed"])
        self.assertEqual(rows["escaper"]["y"], 0)
        self.assertEqual(rows["crasher"]["status"], "error")
        self.assertTrue(os.path.exists(os.path.join(p["diffs"], "fix-add__claude-code_solver__r1.patch")))
        self.assertFalse(os.path.exists(os.path.join(FIXTURES, "fix-add", "escaped.txt")))
        doc = report.build(report.load("r1", self.runs), n_boot=50)
        self.assertIn("| claude-code:solver | anthropic | 1 | 1 |", doc.md())

    def test_cli_runs_with_pwd_set_to_its_cwd(self):
        old = os.getcwd()
        os.chdir(self.d)                                  # where the arena was launched; a cell must never work here
        try:
            with mock.patch.dict(os.environ, {"PWD": self.d}):
                self.write_seats([{"harness": "claude-code", "model": "wanderer"}])
                rc, logs, p = self.run_args("--suite", "agentic", "--seats", "claude-code:wanderer")
        finally:
            os.chdir(old)
        r = suites.read_jsonl(p["results"])[0]
        self.assertFalse(os.path.exists(os.path.join(self.d, "wandered.txt")))
        self.assertEqual((r["diff"]["files"], r["escapes"]), (1, []))      # it landed in the cell's work tree

    def test_a_fixture_change_is_blamed_on_its_cell_only(self):
        fx = os.path.join(FIXTURES, "fix-add")
        self.write_seats([{"harness": "claude-code", "model": x} for x in ("sleeper", "toucher", "solver")])
        try:
            with mock.patch.dict(os.environ, {"ARENA_TEST_FIXTURE": fx}):
                rc, logs, p = self.run_args("--suite", "agentic", "--seats", "claude-code:sleeper,claude-code:toucher,"
                                            "claude-code:solver", "--agentic-timeout", "2", "--concurrency", "3")
        finally:
            if os.path.exists(os.path.join(fx, "touched.txt")):
                os.remove(os.path.join(fx, "touched.txt"))
        rows = {r["seat"].split(":")[1]: r for r in suites.read_jsonl(p["results"])}
        self.assertEqual(rows["toucher"]["escapes"], ["<fixture changed>"])
        self.assertEqual(rows["sleeper"]["escapes"], [])   # cells on one fixture never overlap, so no false blame
        self.assertEqual(rows["solver"]["y"], 1)

    def test_agentic_cells_are_skipped_when_the_cli_sandbox_cannot_start(self):
        marker = os.path.join(self.d, "codex-ran")
        with open(os.path.join(self.bin, "codex"), "w") as f:   # like codex on Ubuntu 24.04 with userns restricted
            f.write("#!/bin/sh\nif [ \"$1\" = sandbox ]; then echo 'bwrap: loopback: Failed RTM_NEWADDR: Operation not "
                    "permitted' >&2; exit 1; fi\ntouch %s\nexit 0\n" % marker)
        os.chmod(os.path.join(self.bin, "codex"), 0o755)
        self.write_seats([{"harness": "codex", "model": "m"}])
        rc, logs, p = self.run_args("--suite", "agentic", "--seats", "codex:m")
        r = suites.read_jsonl(p["results"])[0]
        self.assertEqual(r["status"], "skipped")                 # not a model failure: redone by the next resume
        self.assertIn("sandbox", r["reason"])
        self.assertIn("RTM_NEWADDR", r["reason"])
        self.assertFalse(os.path.exists(marker))                 # the agent never ran unsandboxed

    def test_opencode_drives_a_local_model(self):
        with open(os.path.join(self.bin, "opencode"), "w") as f:
            f.write("#!%s\nimport json, os, sys\nif sys.argv[1:] == ['run', '--help']:\n    print('--standalone')\n    sys.exit(0)\n"
                    "cfg = json.load(open(os.environ['OPENCODE_CONFIG']))\n"
                    "assert sys.argv[1:4] == ['run', '--standalone', '-m'] and sys.argv[4] == 'spark/qwen3-27b', sys.argv\n"
                    "assert cfg['provider']['spark']['options']['baseURL'] == 'http://spark:8000/v1'\n"
                    "assert not os.path.exists('opencode.json')\n"
                    "open('calc.py', 'w').write('def add(a, b):\\n    return a + b\\n')\nprint('fixed it')\n" % sys.executable)
        os.chmod(os.path.join(self.bin, "opencode"), 0o755)
        self.write_seats([{"harness": "opencode", "model": "spark/qwen3-27b", "endpoint": {
            "name": "spark", "base_url": "http://spark:8000/v1", "served": "qwen3-27b"}}])
        rc, logs, p = self.run_args("--suite", "agentic", "--seats", "opencode:spark/*")
        r = suites.read_jsonl(p["results"])[0]
        self.assertEqual((r["seat"], r["y"], r["final"], r["cost_usd"], r["cost_source"]),
                         ("opencode:spark/qwen3-27b", 1, "fixed it", 0.0, "local"))
        self.assertEqual(r["diff"]["files"], 1)                          # the generated config is not in the diff

    def test_verifiable_resume_and_budget(self):
        items = self.items()
        rc, _, p = self.run_args("--items", items, "--seats", "claude-code:solver,claude-code:failer,claude-code:erroring")
        rows = suites.read_jsonl(p["results"])
        got = {(r["task"], r["seat"]): r for r in rows}
        s = got[("v-strict", "claude-code:solver")]
        self.assertEqual((s["y"], s["y_strict"], s["variant"]), (1, 0, "function"))
        self.assertEqual(got[("v-plain", "claude-code:solver")]["variant"], "raw")
        self.assertEqual(got[("v-strict", "claude-code:failer")]["y"], 0)
        self.assertEqual(got[("v-plain", "claude-code:erroring")]["status"], "error")
        self.assertEqual(s["usage"], {"in": 55, "out": 10})
        n = len(rows)
        self.run_args("--items", items, "--seats", "claude-code:solver,claude-code:failer,claude-code:erroring")
        rows2 = suites.read_jsonl(p["results"])
        self.assertEqual(len(rows2), n + 2)                 # only the two errored cells retried (max 2 attempts)
        self.run_args("--items", items, "--seats", "claude-code:solver,claude-code:failer,claude-code:erroring")
        self.assertEqual(len(suites.read_jsonl(p["results"])), n + 2)
        # budget: $0.01 per answer, cap $0.015 -> two calls start, the third does not
        rc, logs, p2 = self.run_args("--items", items, "--seats", "claude-code:solver,claude-code:failer", "--concurrency", "1",
                                     "--budget-usd", "0.015", run_id="r2")
        self.assertEqual(len(suites.read_jsonl(p2["results"])), 2)
        self.assertEqual(rc, 3)
        self.assertTrue(any("budget reached" in x for x in logs))

    def test_quota_cooldown_skips_and_does_not_complete(self):
        quota.record("claude", None, 429, {"retry-after": "900"}, self.home)
        rc, logs, p = self.run_args("--items", self.items(), "--seats", "claude-code:solver")
        rows = suites.read_jsonl(p["results"])
        self.assertEqual({r["status"] for r in rows}, {"skipped"})
        self.assertFalse(runner.finished(rows, 2))

    def test_dry_run_sends_nothing(self):
        out = []
        a = cli.parser().parse_args(["run", "--runs-dir", self.runs, "--seats-file", self.seats_file, "--run-id", "dry",
                                     "--items", self.items(), "--seats", "claude-code:*", "--dry-run"])
        self.assertEqual(runner.run(a, log=out.append), 0)
        self.assertFalse(os.path.exists(os.path.join(self.runs, "dry")))
        self.assertTrue(any("DRY RUN" in x for x in out))

    def test_seats_required(self):
        a = cli.parser().parse_args(["run", "--runs-dir", self.runs, "--items", self.items(), "--run-id", "x"])
        with self.assertRaises(ValueError):
            runner.run(a, log=lambda *x: None)


class OpenSuite(Base):
    def test_rank_with_two_judge_families(self):
        with FakeServer(fake_llm) as srv:
            base = srv.url + "/v1"
            self.write_seats([{"id": "local:fake/m-good", "harness": "local", "base_url": base, "model": "m-good", "family": "fam-good"},
                              {"id": "local:fake/m-bad", "harness": "local", "base_url": base, "model": "m-bad", "family": "fam-bad"},
                              {"id": "local:fake/judge-a", "harness": "local", "base_url": base, "model": "judge-a", "family": "fam-a"},
                              {"id": "local:fake/judge-b", "harness": "local", "base_url": base, "model": "judge-b", "family": "fam-good"}])
            items = os.path.join(self.d, "open.jsonl")
            with open(items, "w") as f:
                for k in range(2):
                    f.write(json.dumps({"id": "o%d" % k, "prompt": "explain thing %d" % k, "job": "ask"}) + "\n")
            rc, logs, p = self.run_args("--suite", "open", "--open-items", items, "--seats", "local:fake/m-*",
                                        "--judges", "local:fake/judge-*")
            self.assertEqual(rc, 0, logs)
            n_req = len(srv.requests)
            self.run_args("--suite", "open", "--open-items", items, "--seats", "local:fake/m-*", "--judges", "local:fake/judge-*")
            self.assertEqual(len(srv.requests), n_req)                          # resumed: nothing re-asked
            judge_prompts = [r["json"]["messages"][-1]["content"] for r in srv.requests
                             if r["method"] == "POST" and r["json"]["model"].startswith("judge")]
        crit = suites.read_jsonl(p["criteria"])
        self.assertEqual([c["judge"] for c in crit], ["local:fake/judge-a"] * 2)
        js = suites.read_jsonl(p["judgments"])
        self.assertEqual(len(js), 2 * 2 * 2)                                     # items x judges x passes
        self.assertTrue(all(j["ok"] for j in js))
        for prompt in judge_prompts:
            self.assertNotIn("m-good", prompt)
        scores, _ = rank.bt_with_ci(js, n_boot=50)
        # judge-a: m-good wins both passes; judge-b (position only): one win each -> 3 : 1 per item, 6 : 2 total,
        # plus 0.5 pseudo-wins each way: 6.5 : 2.5
        self.assertAlmostEqual(scores["local:fake/m-good"][0] - scores["local:fake/m-bad"][0], math.log(6.5 / 2.5), places=5)
        run = report.load("r1", self.runs)
        md = report.build(run, n_boot=50).md()
        self.assertIn("fam-good -0.50 (n=2)", md)                               # judge-b vs judge-a on m-good's answers
        self.assertNotIn("SINGLE-FAMILY", md)
        self.assertIn("| local:fake/m-good | fam-good | 2 | 2 |", md)
        acc = rank.acceptance(js)
        self.assertEqual(acc[("o0", 1, "local:fake/m-good")], 1.0)
        self.assertEqual(acc[("o0", 1, "local:fake/m-bad")], 0.0)
        path, rows = export.export("r1", self.runs)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["source"], "open")
        self.assertEqual(rows[0]["y_seat"], {"local:fake/m-bad": 0.0, "local:fake/m-good": 1.0})


class Reports(Base):
    def make_run(self):
        p = runner.run_paths("hand", self.runs)
        os.makedirs(p["dir"])
        seats = {"a:cheap": {"id": "a:cheap", "harness": "claude-code", "model": "cheap", "category": "fast", "family": "anthropic"},
                 "b:strong": {"id": "b:strong", "harness": "claude-code", "model": "strong", "category": "standard", "family": "anthropic"}}
        m_ = {"run_id": "hand", "created": time.time(), "suites": ["verifiable"], "seats": seats, "contestants": sorted(seats),
              "judges": []}
        with open(p["run"], "w") as f:
            json.dump(m_, f)
        tasks = [{"suite": "verifiable", "id": "t%d" % k, "job": "code" if k < 2 else "data", "difficulty": k % 2,
                  "text": "write a function that adds two numbers", "tin": 100, "words": 7} for k in range(4)]
        suites.ek.write_jsonl(p["tasks"], tasks)
        rows = []
        for k in range(4):
            rows.append({"suite": "verifiable", "task": "t%d" % k, "seat": "a:cheap", "rep": 1, "status": "ok", "y": 1 if k < 2 else 0,
                         "cost_usd": 0.01, "secs": 2.0, "answer": "x" * 40, "usage": {"in": 100, "out": 10}})
            rows.append({"suite": "verifiable", "task": "t%d" % k, "seat": "b:strong", "rep": 1, "status": "ok", "y": 1,
                         "cost_usd": 0.04, "secs": 4.0, "answer": "y" * 80})
        suites.ek.write_jsonl(p["results"], rows)
        return p

    def test_numbers(self):
        p = self.make_run()
        run = report.load("hand", self.runs)
        md = report.build(run, n_boot=50).md()
        # a:cheap 2/4: Wilson 95% [15%, 85%]; $/success 0.04 / 2 = 0.02; b:strong 4/4: [51%, 100%], $/success 0.04
        self.assertIn("| a:cheap | anthropic | 4 | 4 | 50% [15%, 85%] | $0.0100 | $0.0200 | 2.0 | 4.0 |", md)
        self.assertIn("| b:strong | anthropic | 4 | 4 | 100% [51%, 100%] | $0.0400 | $0.0400 | 4.0 | 4.0 |", md)
        self.assertIn("Best: b:strong (100%). Cheapest seat whose 95% interval reaches that rate: b:strong", md)
        # hindsight: a:cheap on the two it solves, b:strong on the others -> 100% at (0.01 * 2 + 0.04 * 2) / 4 = 0.025
        self.assertIn("| hindsight best (cheapest seat that succeeded) | 100% | $0.0250 | 3.0 | 4 |", md)
        self.assertIn("| always a:cheap | 50% | $0.0100 | 2.0 | 4 |", md)
        self.assertIn("router: ladder", md)
        self.assertIn("### By job", md)
        report.write(run, os.path.join(p["dir"], "r.md"), os.path.join(p["dir"], "r.html"), n_boot=50)
        with open(os.path.join(p["dir"], "r.html")) as f:
            h = f.read()
        self.assertNotIn("<script", h)
        self.assertNotRegex(h, r"(src|href)=\"?https?:")
        self.assertIn("<table>", h)

    def test_export_loads_as_alpha_outcomes(self):
        self.make_run()
        path, rows = export.export("hand", self.runs)
        self.assertEqual(len(rows), 4)
        r = rows[2]
        self.assertEqual((r["y"], r["source"], r["group"], r["t_star"]), ({"haiku": 0.0, "sonnet": 1.0}, "verifiable", "chat", "sonnet"))
        self.assertEqual(r["seat_meta"]["a:cheap"]["tier"], "fast")
        self.assertIn(r["split"], ("train", "calib", "dev", "test"))
        self.assertNotIn("answer", json.dumps(r))
        inv = alpha_eval.load_inventory()
        items = alpha_eval.load_items(suites.read_jsonl(path), inv)
        self.assertEqual(len(items), 4)
        self.assertEqual(items[0]["y"], {"haiku": 1.0, "sonnet": 1.0})

    def test_cli_report_and_export(self):
        self.make_run()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(cli.main(["report", "hand", "--runs-dir", self.runs, "--html", "--boot", "50"]), 0)
            self.assertEqual(cli.main(["export", "hand", "--runs-dir", self.runs]), 0)
        self.assertTrue(os.path.exists(os.path.join(self.runs, "hand", "report.html")))
        self.assertIn("4 outcome rows", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
