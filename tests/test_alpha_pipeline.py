"""The alpha-set pipeline end to end with a fake orchestrator (S1 spec §7, §9.4-§9.5): manifests, ingest validation,
checkers, grading, outcomes, the pilot report, the CI gate, pacing, collect and fit_link."""
import argparse
import contextlib
import io
import json
import math
import os
import random
import re
import sys
import tempfile
import time
import unittest
from unittest import mock

import helpers  # noqa: F401

ROOT = helpers.ROOT
sys.path.insert(0, os.path.join(ROOT, "evals"))
import alpha  # noqa: E402
import alpha_eval as ev  # noqa: E402
import alpha_stats as st  # noqa: E402
import evalkit as ek  # noqa: E402
import fit_link  # noqa: E402
from test_label import reply_all  # noqa: E402

STRENGTH = {"haiku": 0, "sonnet": 1, "opus": 2, "fable": 3}
CHECKER = ('import re\n_EXPECTED = %r\n'
           'def check(answer):\n    hits = re.findall(r"ANSWER:\\s*(\\S+)", answer or "")\n    return bool(hits) and hits[-1] == _EXPECTED\n')


def difficulty(text):
    hit = re.search(r"D=(\d)", text)
    return int(hit.group(1)) if hit else 1


def fake_answer(call):
    """A tier solves an item when its strength reaches the item's D= marker."""
    ok = STRENGTH[call["model"]] >= difficulty(call["prompt"])
    m = re.search(r"EXPECT=(\w+)", call["prompt"])
    if m:
        return "Working it out.\n\nANSWER: %s" % (m.group(1) if ok else "nope")
    return "As Claude %s, here is my take.\n\n**RESULT %s**" % (call["model"].title(), "ok" if ok else "bad")


def fake_grade(call):
    if call["prompt"].startswith("You write grading criteria"):
        return json.dumps({"must": ["answers the question", "is correct", "is clear"], "disqualifiers": ["wrong result"],
                           "sketch": "a correct answer"})
    blocks = re.findall(r"Answer ([A-Z]):\n<<<\n(.*?)\n>>>", call["prompt"], re.S)
    out = {}
    for lab, text in blocks:
        ok = "RESULT ok" in text or ("ANSWER:" in text and "ANSWER: nope" not in text)
        if call["prompt"].startswith("Two graders disagreed"):
            out[lab] = ok
        else:
            if call["model"] == "opus" and "RESULT bad" in text:
                ok = True                                   # opus is lenient: creates splits for fable to settle
            out[lab] = {"must": [ok, True, True], "disq": [not ok and call["model"] == "sonnet"]}
    return json.dumps(out)


class Env:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.d = d
        ver, opn, per = [], [], []
        for i in range(16):
            ver.append({"id": "V-reason-%03d" % i, "type": ("reason", "code-fn", "data", "edit")[i % 4],
                        "prompt": "compute thing %d (D=%d) and end with ANSWER: <value> EXPECT=v%d" % (i, i % 4, i),
                        "difficulty_intent": i % 4, "checker_py": CHECKER % ("v%d" % i),
                        "reference_answer": "ANSWER: v%d" % i, "validated": True})
        for i in range(16):
            opn.append({"id": "write-%03d" % i, "slice": "write", "context": "chat", "lang": "en", "images": 0, "history": [],
                        "prompt": "write me a note about topic %d (D=%d)" % (i, (i + 1) % 4), "self_contained": True,
                        "hint": {"job": ("write", "plan", "ask", "summarize")[i % 4], "difficulty": (i + 1) % 4}})
        opn.append({"id": "cc-x", "prompt": "fix the repo (D=1)", "self_contained": False, "images": 0, "hint": {"job": "code"}})
        for i in range(4):
            per.append({"id": "persona-%d" % i, "prompt": "persona ask %d (D=%d)" % (i, i), "self_contained": True, "images": 0,
                        "hint": {"job": "ask", "difficulty": i}})
        self.files = {"verifiable": os.path.join(d, "V.jsonl"), "corpus": os.path.join(d, "pool.jsonl"),
                      "persona": os.path.join(d, "ood.jsonl")}
        ek.write_jsonl(self.files["verifiable"], ver)
        ek.write_jsonl(self.files["corpus"], opn)
        ek.write_jsonl(self.files["persona"], per)
        self.args = ["--alpha-dir", os.path.join(d, "alpha"), "--work", os.path.join(d, "work"),
                     "--verifiable", self.files["verifiable"], "--corpus-file", self.files["corpus"],
                     "--persona-file", self.files["persona"], "--aggregate", os.path.join(d, "agg.jsonl"),
                     "--manifest", os.path.join(d, "manifest.json"), "--ledger", os.path.join(d, "ledger.jsonl")]
        os.makedirs(os.path.join(d, "alpha"))
        with open(ek.ALPHA + "/inventory.json") as f, open(os.path.join(d, "alpha", "inventory.json"), "w") as g:
            g.write(f.read())

    def run(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch.dict(os.environ, {"MAESTRO_HOME": self.d}):
            code = alpha.main(list(argv) + self.args)
        return code, out.getvalue()

    def p(self):
        return alpha.paths(argparse.Namespace(
            alpha_dir=os.path.join(self.d, "alpha"), work=os.path.join(self.d, "work"), verifiable=self.files["verifiable"],
            corpus_file=self.files["corpus"], persona_file=self.files["persona"], aggregate=os.path.join(self.d, "agg.jsonl"),
            manifest=os.path.join(self.d, "manifest.json"), ledger=os.path.join(self.d, "ledger.jsonl")))

    def grade_all(self, which="pool"):
        work = os.path.join(self.d, "work")
        self.run("grade", "--stage", "check", "--set", which)
        for stage in ("criteria", "llm", "adjudicate"):
            self.run("grade", "--stage", stage, "--set", which)
            reply_all(os.path.join(work, "grade"), fake_grade)
            self.run("grade", "--ingest", "--set", which)


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.env = Env()

    def tearDown(self):
        self.env.tmp.cleanup()

    def test_check_validates_verifiable_items(self):
        code, out = self.env.run("check")
        self.assertEqual(code, 0, out)
        rows = ek.read_jsonl(self.env.files["verifiable"])
        rows[0]["reference_answer"] = "ANSWER: wrong"
        rows[1]["checker_py"] = "def check(a):\n    return True\n"
        ek.write_jsonl(self.env.files["verifiable"], rows)
        code, out = self.env.run("check")
        self.assertEqual(code, 1)
        self.assertIn("V-reason-000: reference fails", out)
        self.assertIn("empty answer passes", out)
        self.assertIn("copy-the-input passes", out)

    def test_pilot_to_pool_to_gate(self):
        env, p = self.env, self.env.p()
        work = os.path.join(env.d, "work")
        code, out = env.run("pilot", "--n", "12")
        self.assertEqual(code, 0, out)
        pilot = ek.read_jsonl(p["pilot"])
        self.assertEqual(len(pilot), 12)
        self.assertGreaterEqual(sum(r["source"] == "verifiable" for r in pilot), 6)
        self.assertNotIn("cc-x", {r["id"] for r in pilot})

        env.run("answer", "--phase", "pilot", "--per-file", "10")
        calls = [c for f in sorted(os.listdir(os.path.join(work, "answer"))) if f.startswith("answer-")
                 for c in ek.read_jsonl(os.path.join(work, "answer", f))]
        self.assertEqual(len(calls), 48)
        c0 = calls[0]
        self.assertTrue(c0["prompt"].startswith("Answer the request below as a helpful assistant in a chat. Do not use any tools"))
        self.assertIn("USER REQUEST:\n", c0["prompt"])
        self.assertRegex(c0["call_id"], r"^ans:[\w-]+:(haiku|sonnet|opus|fable):r1:a1$")
        first = {"n": 0}

        def answer(c):   # the first reply used a tool: invalid, retried once
            first["n"] += 1
            return fake_answer(c)
        reply_all(os.path.join(work, "answer"), answer)
        rep = os.path.join(work, "answer", [f for f in os.listdir(os.path.join(work, "answer")) if f.endswith(".replies.jsonl")][0])
        rows = ek.read_jsonl(rep)
        rows[0]["tool_calls"] = 2
        ek.write_jsonl(rep, rows)
        code, out = env.run("answer", "--ingest")
        self.assertIn("1 invalid", out)
        env.run("answer", "--phase", "pilot")
        self.assertEqual(len(ek.pending(os.path.join(work, "answer"))), 1)
        self.assertTrue(next(iter(ek.pending(os.path.join(work, "answer")))).endswith(":a2"))
        reply_all(os.path.join(work, "answer"), fake_answer)
        env.run("answer", "--ingest")
        ans = alpha.answers(p)
        self.assertEqual(sum(a["valid"] for a in ans.values()), 48)
        self.assertTrue(all("text" in a and len(a["sha256"]) == 64 and a["tout"] > 0 for a in ans.values()))

        env.run("grade", "--stage", "check", "--set", "pilot")
        env.run("grade", "--stage", "criteria", "--set", "pilot")
        reply_all(os.path.join(work, "grade"), lambda c: "no JSON here")       # invalid: re-emitted as attempt 2
        code, out = env.run("grade", "--ingest", "--set", "pilot")
        self.assertIn("invalid replies", out)
        self.assertFalse(ek.read_jsonl(p["criteria"]))
        env.run("grade", "--stage", "criteria", "--set", "pilot")
        self.assertTrue(all(cid.endswith(":a2") for cid in ek.pending(os.path.join(work, "grade"))))
        env.grade_all("pilot")
        crit = ek.read_jsonl(p["criteria"])
        self.assertTrue(crit)
        self.assertTrue(all(3 <= len(c["must"]) <= 6 for c in crit))
        g = alpha.grades(p)
        self.assertTrue(any("fable" in v for v in g.values()))     # opus's leniency split some cells
        code, out = env.run("fit", "--set", "pilot")
        self.assertEqual(code, 0, out)
        rows = ek.read_jsonl(p["pilot_outcomes"])
        self.assertEqual(len(rows), 12)
        cands = alpha.candidates(p)
        for r in rows:
            d = difficulty(cands[r["id"]]["prompt"])
            self.assertEqual(r["y"], {t: 1.0 if STRENGTH[t] >= d else 0.0 for t in ek.TIERS}, r["id"])
            self.assertEqual(r["phase2"], "full")
            self.assertEqual(r["t_star"], ek.TIERS[d])
            self.assertFalse(r["nonmonotone"])
            self.assertNotIn("text", json.dumps(r).replace("context", ""))
        val = ek.read_json(p["validity"])
        self.assertGreater(val["n_cells"], 0)
        self.assertIn("sonnet", val["graders"])
        assets = ek.read_json(p["assets"])
        self.assertIn("answers.jsonl", assets)

        env.run("answer", "--phase", "pilot-repeat")
        rep_calls = ek.pending(os.path.join(work, "answer"))
        self.assertTrue(rep_calls)
        self.assertTrue(all(m["rep"] == 2 and m["tier"] in ("haiku", "sonnet") for m in rep_calls.values()))
        reply_all(os.path.join(work, "answer"), fake_answer)
        env.run("answer", "--ingest")
        env.grade_all("pilot")
        env.run("fit", "--set", "pilot")
        code, out = env.run("pilot-report", "--sim-n", "200", "--reps", "1")
        self.assertEqual(code, 0, out)
        report = ek.read_json(os.path.join(env.d, "alpha", "pilot_report.json"))
        self.assertIn(report["decision"], ("go", "no-go"))
        self.assertEqual(report["repeat_kappa"], 1.0)
        self.assertEqual(set(report["projection"]["A"]), {"0.4", "0.5", "0.6", "0.7"})
        ledger = ek.read_jsonl(os.path.join(env.d, "ledger.jsonl"))
        self.assertEqual(ledger[-1]["split"], "pilot")
        self.assertEqual(ledger[-1]["decision"], report["decision"])

        code, out = env.run("select", "--total", "48", "--persona", "2")
        pool = ek.read_jsonl(p["pool"])
        self.assertEqual(sum(r["pilot"] for r in pool), 12)
        self.assertEqual(sum(r["source"] == "persona" for r in pool), 2)
        env.run("answer", "--phase", "1")
        new = ek.pending(os.path.join(work, "answer"))
        self.assertTrue(all(m["tier"] in ("haiku", "fable") and m["item"] not in {r["id"] for r in pilot} for m in new.values()))
        reply_all(os.path.join(work, "answer"), fake_answer)
        env.run("answer", "--ingest")
        env.grade_all()
        env.run("answer", "--phase", "2")
        ph2 = ek.pending(os.path.join(work, "answer"))
        self.assertTrue(all(m["tier"] in ("sonnet", "opus") for m in ph2.values()))
        reply_all(os.path.join(work, "answer"), fake_answer)
        env.run("answer", "--ingest")
        env.grade_all()
        code, out = env.run("fit")
        rows = ek.read_jsonl(p["outcomes"])
        kinds = {r["phase2"] for r in rows}
        self.assertIn("full", kinds)
        self.assertIn("disagree", kinds)
        for r in rows:
            if r["phase2"] == "disagree":
                self.assertEqual(set(r["y"]), set(ek.TIERS))
                self.assertEqual(r["ipw"], 1.0)
            if r["phase2"] == "unsampled":
                self.assertEqual(set(r["y"]), {"haiku", "fable"})
                self.assertEqual(r["ipw"], 0.0)
            if r["phase2"] == "sampled":
                self.assertEqual(r["ipw"], 4.0)
        irt = ek.read_json(p["irt"])
        self.assertLess(irt["theta"]["haiku"], irt["theta"]["fable"])

        # the CI gate on whatever landed in the test split (force a few rows there so the gate has data). The gate's plumbing is
        # checked with the ladder: these fake outcomes are nothing S1's P is calibrated against (UtilityHooks covers the
        # utility decider's checks).
        for r in rows[:8]:
            r["split"] = "test"
        ek.write_jsonl(p["outcomes"], rows)
        ladder = mock.patch.object(ev, "current_policy", lambda texts, inv, objective="balanced": ev.route_policy(
            texts, inv, {"objective": objective}) + ({"objective": objective},))
        ladder.start()
        self.addCleanup(ladder.stop)
        code, out = env.run("gate", "--boot", "200")
        self.assertEqual(code, 0, out)
        self.assertIn("SKIP", out)
        code, out = env.run("gate", "--update", "--boot", "200")
        self.assertTrue(os.path.exists(p["gates"]) and os.path.exists(p["baseline"]))
        base = ek.read_jsonl(p["baseline"])
        self.assertTrue(all(b["chosen"] in ek.TIERS for b in base))
        code, out = env.run("gate", "--boot", "200")
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"dS_bal \(paired\)\s+PASS")
        code, out = env.run("report", "--split", "test", "--boot", "100")
        self.assertEqual(code, 0, out)
        rep = ek.read_json(os.path.join(env.d, "alpha", "report-test.json"))
        for name in ("B1", "B2", "B3", "B3'", "B0", "B4", "B5", "O_real", "B4-proxy"):
            self.assertIn(name, rep["policies"])
        self.assertGreaterEqual(rep["policies"]["O_real"]["S"], rep["policies"]["B1"]["S"])
        self.assertIn("panel_validity.json", os.listdir(os.path.join(env.d, "alpha")))
        self.assertEqual(ek.read_jsonl(os.path.join(env.d, "ledger.jsonl"))[-1]["split"], "test")
        out_preds = os.path.join(env.d, "preds.jsonl")
        code, out = env.run("replay", "--split", "test", "--policy", "B4-proxy", "--out", out_preds)
        self.assertEqual(code, 0, out)
        self.assertTrue(all(r["chosen"] in ek.TIERS for r in ek.read_jsonl(out_preds)))
        code, out = env.run("release", "--boot", "100")
        self.assertEqual(code, 1)                            # no utility decider, audit or shadow evidence yet
        self.assertIn("RELEASE BLOCKED", out)
        g = ek.read_json(p["gates"])                       # a baseline that was much better must fail the gate
        g["baseline"]["severe"] = -0.5
        ek.write_json(p["gates"], g)
        code, out = env.run("gate", "--boot", "200")
        self.assertEqual(code, 1, out)
        self.assertRegex(out, r"severe under-routing\s+FAIL")

    def test_audit_csv_is_blind(self):
        env, p = self.env, self.env.p()
        env.run("pilot", "--n", "12")
        env.run("answer", "--phase", "pilot")
        reply_all(os.path.join(env.d, "work", "answer"), fake_answer)
        env.run("answer", "--ingest")
        env.grade_all("pilot")
        code, out = env.run("grade", "--stage", "audit", "--set", "pilot")
        with open(p["audit"]) as f:
            text = f.read()
        self.assertNotIn("stratum", text)
        self.assertNotIn("Claude", text)                   # self-identifying phrases stripped
        key = ek.read_json(p["audit_key"])
        self.assertTrue(key)
        import csv
        rows = list(csv.reader(io.StringIO(text)))
        for r in rows[1:]:
            r[-1] = "1" if key[r[0]]["y"] >= 0.5 else "0"
        with open(p["audit"], "w", newline="") as f:
            csv.writer(f).writerows(rows)
        agr = alpha.audit_agreement(p["audit"], p["audit_key"])
        self.assertTrue(agr)
        self.assertTrue(all(v["raw"] == 1.0 for v in agr.values()))


THETA = {"haiku": 0.0, "sonnet": 1.1, "opus": 2.2, "fable": 3.3}
REAL_ROUTE = ek.m.route


def fake_utility_route(text, harness=None, inventory=None, config=None, env=None, home=None, **kw):
    """The real router, plus a stand-in for the integrated utility decider (decider, options with P, posterior), so the
    replay, gate, V0 calibration and release hooks run before decide.py lands. Under quota pressure it moves to Codex."""
    d = REAL_ROUTE(text, harness=harness, inventory=inventory, config=config, env=env, home=home, **kw)
    cfg = config or {}
    if cfg.get("decider") != "utility":
        return d
    b = difficulty(text) - 0.5
    ps = {t: st.sigmoid(2 * (THETA[t] - b)) for t in ek.TIERS}
    cost = dict(zip(ek.TIERS, (0.001, 0.002, 0.004, 0.01)))
    v = cfg.get("wtp_usd") or 0.5
    tier = max(ek.TIERS, key=lambda t: v * ps[t] - cost[t])
    lim = ek.read_json(os.path.join(home, "claude_limits.json")) or {}
    rl = lim.get("rate_limits") or {}
    seat = "claude-code"
    if (rl.get("five_hour") or {}).get("used_percentage", 0) > 70 and tier in ("haiku", "sonnet"):
        seat = "codex"
    if tier == "fable" and (rl.get("seven_day_fable") or {}).get("used_percentage", 0) >= 100:
        tier = "opus"
    d.update(decider="utility", seat=seat, model=tier, options=[{"seat": "claude-code:" + t, "p": ps[t]} for t in ek.TIERS],
             posterior={"b": [b, 0.5], "job_p": {"write": 1.0}, "tok_out": [math.log(500), 0.3]})
    return d


class UtilityHooks(unittest.TestCase):
    def setUp(self):
        self.env = Env()
        p = self.env.p()
        ek.write_jsonl(self.env.files["verifiable"], [
            {"id": "V-x-%04d" % i, "type": "reason", "prompt": "compute thing %d (D=%d) EXPECT=v" % (i, i % 4),
             "checker_py": CHECKER % "v", "reference_answer": "ANSWER: v", "validated": True} for i in range(600)])
        rows, rng = [], random.Random(5)
        for k, it in enumerate(sorted(alpha.candidates(p).values(), key=lambda it: it["id"])):
            dd = difficulty(it["prompt"])
            y = {t: 1.0 if rng.random() < st.sigmoid(2 * (THETA[t] - dd + 0.5)) else 0.0 for t in ek.TIERS}
            rows.append({"id": it["id"], "cluster": "k:" + it["id"], "split": ("test", "dev", "persona")[k % 3]
                         if it["source"] != "persona" else "persona", "source": it["source"], "group": "chat",
                         "job": "write", "difficulty": dd, "lang": "en", "context": "chat", "words": 10, "phase2": "full",
                         "ipw": 1.0, "y": y, "tin": 300,
                         "tout": {t: 300 + 100 * n for n, t in enumerate(ek.TIERS)}})
        ek.write_jsonl(p["outcomes"], rows)
        ek.write_json(p["irt"], {"theta": THETA, "b": {}})
        self.patch = mock.patch.object(ek.m, "route", fake_utility_route)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.env.tmp.cleanup()

    def test_gate_report_calibrate_and_release_use_the_utility_decider(self):
        env, p = self.env, self.env.p()
        code, out = env.run("calibrate-v")
        self.assertEqual(code, 0, out)
        v0 = ek.read_json(p["v0"])
        self.assertTrue(v0["V0_calibrated"])
        self.assertLessEqual(v0["V0"]["cost"] or 0, v0["V0"]["quality"] or 1e9)
        code, out = env.run("gate", "--update", "--boot", "200")
        self.assertIn("decider utility", out)
        g = ek.read_json(p["gates"])
        self.assertEqual(g["decider"], "utility")
        g["latency_ratio_p99"] = 0          # tiny fixture prompts make the p99 ratio noise; the 5 ms ceiling still applies
        ek.write_json(p["gates"], g)
        self.assertTrue(all("p" in r for r in ek.read_jsonl(p["baseline"])))
        code, out = env.run("gate", "--boot", "200")
        self.assertEqual(code, 0, out)
        for check in ("d logloss", "ECE", "calibration slope", "quota monotonicity"):
            self.assertRegex(out, check + r".*\s(PASS)\s")
        code, out = env.run("report", "--split", "test", "--boot", "100")
        self.assertEqual(code, 0, out)
        rep = ek.read_json(os.path.join(env.d, "alpha", "report-test.json"))
        self.assertEqual(rep["main_policy"], "R")
        self.assertIn("capture", rep)
        self.assertEqual(rep["heads"]["job_accuracy"], 1.0)
        self.assertIn("calibration", rep["policies"]["R"])
        self.assertIn("alpha_vs_B4", rep)
        code, out = env.run("release", "--boot", "100")
        self.assertEqual(code, 1)
        self.assertIn("alpha_$ vs hull CI excludes 0", out)
        self.assertIn("human audit", out)
        self.assertEqual(ek.read_jsonl(os.path.join(env.d, "ledger.jsonl"))[-1]["split"], "release")


class Units(unittest.TestCase):
    def test_gate_without_outcomes(self):
        with tempfile.TemporaryDirectory() as d:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = alpha.main(["gate", "--alpha-dir", d, "--work", d])
            self.assertEqual(code, 0)
            self.assertIn("no outcome data yet", out.getvalue())

    def test_gate_is_fast_and_stdlib_only(self):
        code = ("import sys, time; t = time.time(); sys.argv = ['alpha.py', 'gate']; sys.path.insert(0, %r); import alpha; "
                "alpha.main(['gate']); bad = [m for m in sys.modules if m.split('.')[0] in ('numpy', 'sklearn', 'scipy')]; "
                "print(bad, time.time() - t)") % os.path.join(ROOT, "evals")
        import subprocess
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, cwd=ROOT,
                             env=dict(os.environ, MAESTRO_HOME=tempfile.gettempdir())).stdout
        self.assertIn("[]", out)

    def test_clean_answer(self):
        text = ("As Claude, I think **this** works.\n* one\n+ two\n\n\n\n## Heading\n```python\n# keep me\n* also code\n```\n"
                "An AI assistant made by Anthropic says hi. Opus would agree.")
        out = alpha.clean_answer(text)
        self.assertNotIn("Claude", out)
        self.assertNotIn("Anthropic", out)
        self.assertNotIn("Opus", out)
        self.assertIn("- one", out)
        self.assertIn("- two", out)
        self.assertIn("this works", out)
        self.assertIn("# keep me\n* also code", out)
        self.assertNotIn("\n\n\n", out)
        long = alpha.clean_answer("x" * 7000)
        self.assertTrue(long.startswith("x" * 6000))
        self.assertIn("truncated at 6000", long)

    def test_checker_sandbox(self):
        ok = 'def check(a):\n    return a == "yes"\n'
        self.assertTrue(ek.run_checker(ok, "yes"))
        self.assertFalse(ek.run_checker(ok, "no"))
        os.environ["MAESTRO_TEST_SECRET"] = "x"
        try:   # the checker gets an empty environment
            self.assertFalse(ek.run_checker("def check(a):\n    import os\n    return 'MAESTRO_TEST_SECRET' in os.environ or 'PATH' in os.environ\n", ""))
        finally:
            del os.environ["MAESTRO_TEST_SECRET"]
        self.assertFalse(ek.run_checker("def check(a):\n    while True: pass\n", "", timeout=2))
        spoof = 'import sys\ndef check(a):\n    print("PASS")\n    sys.stdout.flush()\n    import os; os._exit(0)\n'
        self.assertFalse(ek.run_checker(spoof, ""))
        self.assertFalse(ek.run_checker("syntax error here", "x"))

    def test_collect_fits_a_seat_and_writes_anchors(self):
        rng = random.Random(0)
        rows = [{"id": "L%03d" % i, "prompt": "q%d ANSWER: v" % i, "b": rng.uniform(-2, 3), "a": 1.0,
                 "checker_py": CHECKER % "v"} for i in range(40)]
        true = 1.0

        def ask(seat, prompt):
            b = next(r["b"] for r in rows if r["prompt"] in prompt)
            return "ANSWER: v" if rng.random() < st.sigmoid(true - b) else "ANSWER: x"
        with tempfile.TemporaryDirectory() as home:
            rec, path = alpha.collect("local:spark/qwen", rows, n=40, ask=ask, prior=0.0, home=home, log=lambda *_: None)
            data = ek.read_json(path)
        self.assertEqual(data["version"], 1)
        self.assertEqual(data["seats"]["local:spark/qwen"]["n"], 40)
        self.assertGreaterEqual(rec["sd"], 0.25)
        self.assertLess(abs(rec["theta_chat"] - true), 1.0)

    def test_fit_theta_newton(self):
        obs = [(1.0 if b < 1 else 0.0, b, 1.0) for b in [x / 10.0 for x in range(-30, 50)]]
        th, sd = alpha.fit_theta(obs, 0.0, prior_sd=10.0)
        self.assertAlmostEqual(th, 1.0, delta=0.3)
        self.assertLess(sd, 1.0)
        th0, _ = alpha.fit_theta([], 0.7)
        self.assertAlmostEqual(th0, 0.7)

    def test_pacing_simulation(self):
        rng = random.Random(0)
        tasks = [(1.0 if rng.random() < 0.8 else 0.0, rng.choice((0.04, 0.09, 0.2, 0.5))) for _ in range(300)]
        res = ev.pacing(tasks, days=7, seed=1)
        self.assertTrue(all(res["checks"].values()), res)
        self.assertGreater(res["a"]["exhausted_hours"], res["b"]["exhausted_hours"])   # pacing avoids exhausting the window
        self.assertAlmostEqual(res["c"]["api_usd"], sum(u for _, u in tasks[:res["c"]["tasks"] % len(tasks)]) +
                               (res["c"]["tasks"] // len(tasks)) * sum(u for _, u in tasks), places=6)
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()) as out:
            code = alpha.main(["pacing", "--synthetic", "200", "--days", "3", "--alpha-dir", ek.ALPHA, "--work", d])
        self.assertEqual(code, 0, out.getvalue())
        self.assertIn("b_exhausted_le_a", out.getvalue())

    def test_pacing_price_reference_behaviour(self):
        """The §4.3 table: fresh window 0; 42% used with 70 min left at 20%/h 0; weekly 80% with 4 days left g_max."""
        self.assertEqual(ev._g(0.0, 5 * 3600, 5 * 3600, None, 0.15, 0.01, 1.0), 0.0)
        self.assertEqual(ev._g(0.42, 70 * 60, 5 * 3600, 0.2 / 3600, 0.15, 0.01, 1.0), 0.0)
        self.assertEqual(ev._g(0.80, 4 * 86400, 7 * 86400, None, 0.15, 0.003, 1.0), 1.0)
        self.assertEqual(ev._g(0.30, 2 * 86400, 7 * 86400, None, 0.15, 0.003, 1.0), 0.0)

    def test_project_reports_capture_and_headroom(self):
        rng = random.Random(0)
        bs = [rng.gauss(0.5, 2.0) for _ in range(200)]
        pr = alpha.project({"haiku": 0.0, "sonnet": 1.1, "opus": 2.2, "fable": 3.3}, bs, [0.003, 0.006, 0.012, 0.03], n=300, reps=1)
        self.assertEqual(set(pr["A"]), {0.4, 0.5, 0.6, 0.7})
        self.assertLess(pr["A"][0.4], pr["A"][0.7] + 0.2)
        self.assertGreater(pr["oracle_headroom"], 0.0)

    def test_bin_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            body = b"x" * 64
            head = json.dumps({"format": 1, "sections": {"emb": [192, 64]}, "sha256": {"emb": ek.sha256_bytes(body)}}).encode()
            head = head + b" " * (184 - len(head))          # 8 + 184 = 192: the section's offset
            path = os.path.join(d, "s1.bin")
            with open(path, "wb") as f:
                f.write(b"MS1\x00" + len(head).to_bytes(4, "little") + head + body)
            self.assertEqual(ev.bin_integrity(path), (256, []))
            with open(path, "r+b") as f:
                f.seek(200)
                f.write(b"y")
            self.assertEqual(ev.bin_integrity(path)[1], ["section emb sha256 mismatch"])


class Pieces(unittest.TestCase):
    def test_nelder_mead(self):
        x, v = st.nelder_mead(lambda z: (z[0] - 1.5) ** 2 + 3 * (z[1] + 0.5) ** 2, [0.0, 0.0])
        self.assertAlmostEqual(x[0], 1.5, places=3)
        self.assertAlmostEqual(x[1], -0.5, places=3)

    def test_panel_validity_on_consistent_labels(self):
        rng = random.Random(0)
        theta = {"haiku": 0.0, "sonnet": 1.1, "opus": 2.2, "fable": 3.3}
        rows = []
        for i in range(120):
            b = rng.uniform(-2, 5)
            y = {t: 1.0 if rng.random() < st.sigmoid(2 * (theta[t] - b)) else 0.0 for t in ek.TIERS}
            d = min(3, max(0, int((b + 2) / 1.75)))
            mt = next((k for k, t in enumerate(ek.TIERS) if theta[t] - b > 0.8), 4)
            rows.append({"id": "i%d" % i, "split": "train", "y": y, "phase2": "full", "ipw": 1.0,
                         "t_star": next((t for t in ek.TIERS if y[t] >= 0.5), "none"),
                         "panel": {"difficulty": [1.0 if k == d else 0.0 for k in range(4)],
                                   "min_tier": [1.0 if k == mt else 0.0 for k in range(5)]}})
        v = alpha.panel_validity(rows, theta)
        self.assertGreater(v["validity"], 0.5)
        self.assertEqual(v["lambda_d"], 0.5)
        self.assertGreater(v["auc_haiku_fail"], 0.8)
        self.assertTrue(v["tau"][0] < v["tau"][1] < v["tau"][2])
        self.assertIn("c_mt", v)
        self.assertEqual(alpha.panel_validity(rows[:3], theta)["n"], 3)

    def test_quota_status_and_link_prior(self):
        with tempfile.TemporaryDirectory() as home:
            self.assertFalse(ek.quota_status(home)[0])
            ek.write_json(os.path.join(home, "claude_limits.json"), {"updated_at": time.time(), "rate_limits": {
                "five_hour": {"used_percentage": 75, "resets_at": time.time() + 3600},
                "seven_day": {"used_percentage": 10, "resets_at": time.time() + 86400}}})
            pause, msg = ek.quota_status(home)
            self.assertTrue(pause)
            self.assertIn("PAUSE", msg)
            self.assertIsNone(alpha.link_prior("local:spark/qwen", home))
            ek.write_json(os.path.join(home, "models.json"), {"models": [
                {"id": "local:spark/qwen", "model": "qwen", "theta": {"code": [1.0, 0.3], "ask": [2.0, 0.3]}}]})
            self.assertAlmostEqual(alpha.link_prior("local:spark/qwen", home), 1.5)

    def test_collect_cli_against_a_fake_local_server(self):
        """`alpha.py collect --seat local:<endpoint>/<model>` sends the linking prompts through `maestro-router run` to a fake
        OpenAI-compatible server on 127.0.0.1, grades them in the sandbox and writes anchors.json."""
        import fake_http

        def handler(req):
            if req["method"] == "GET":
                return fake_http.json_reply({"data": [{"id": "qwen"}]})
            prompt = req["json"]["messages"][-1]["content"]
            ok = int(re.search(r"q(\d+)", prompt).group(1)) % 3 != 0
            return fake_http.json_reply({"choices": [{"message": {"role": "assistant", "content": "ANSWER: %s" % ("v" if ok else "x")}}]})
        with fake_http.FakeServer(handler) as srv, tempfile.TemporaryDirectory() as d:
            home, al = os.path.join(d, "home"), os.path.join(d, "alpha")
            os.makedirs(home)
            ek.write_json(os.path.join(home, "config.json"), {"local_endpoints": [{"name": "spark", "url": srv.url, "kind": "openai"}]})
            ek.write_jsonl(os.path.join(al, "linking.jsonl"), [{"id": "L%d" % i, "prompt": "q%d" % i, "b": 0.0, "a": 1.0,
                                                                "checker_py": CHECKER % "v"} for i in range(12)])
            out = io.StringIO()
            with mock.patch.dict(os.environ, {"MAESTRO_HOME": home}), contextlib.redirect_stdout(out):
                code = alpha.main(["collect", "--seat", "local:spark/qwen", "--n", "12", "--yes", "--alpha-dir", al, "--work", d])
            self.assertEqual(code, 0, out.getvalue())
            seat = ek.read_json(os.path.join(home, "anchors.json"))["seats"]["local:spark/qwen"]
            self.assertEqual((seat["n"], seat["solved"]), (12, 8))
            self.assertGreater(seat["theta_chat"], 0.0)
            self.assertEqual(len([r for r in srv.requests if r["method"] == "POST"]), 12)

    def test_proxy_replay_uses_the_planner(self):
        ch, ps, info = ev.proxy_policy({"a": "design a lock-free MPMC queue with proofs of linearizability",
                                        "b": "what is 2+2"}, ev.load_inventory(), {})
        self.assertEqual(set(ch), {"a", "b"})
        self.assertEqual(ch["a"], ev.onehot("fable"))
        self.assertEqual(info["decider"], "ladder")


class FitLink(unittest.TestCase):
    def test_beta_recovered_from_synthetic_public_data(self):
        rng = random.Random(0)
        beta = 1.5
        models = ["vendor-m%02d" % k for k in range(18)]
        zs = {m: rng.uniform(-1.5, 1.5) for m in models}
        strength = {m: st_inv(zs[m]) for m in models}
        with tempfile.TemporaryDirectory() as d:
            items = ["inst-%03d" % i for i in range(150)]
            bs = {i: rng.uniform(-3, 3) for i in items}
            lb = {"leaderboards": [{"name": "Verified", "results": []}]}
            for k, mname in enumerate(models):
                th = 0.2 + beta * zs[mname]
                res = [i for i in items if rng.random() < st.sigmoid(th - bs[i])]
                folder = "2026_mini_%s" % mname
                ek.write_json(os.path.join(d, folder + ".json"), {"resolved": res, "no_generation": [i for i in items if i not in res]})
                lb["leaderboards"][0]["results"].append({"folder": folder, "agent": "mini-SWE-agent", "model_display": mname,
                                                         "tags": ["Model: %s" % mname, "System: Attempts - 1"]})
            lbp = os.path.join(d, "lb.json")
            ek.write_json(lbp, lb)
            yml = os.path.join(d, "aider.yml")
            with open(yml, "w") as f:
                for mname in models:
                    f.write("- dirname: x\n  model: %s\n  edit_format: diff\n  pass_rate_1: %.1f\n\n" % (
                        mname, 100 * st.sigmoid(-0.5 + beta * zs[mname] + rng.gauss(0, 0.2))))
                f.write("- dirname: y\n  model: vendor-m00\n  edit_format: whole\n  pass_rate_1: 99.0\n")
            systems, its = fit_link.load_swebench(d, lbp)
            self.assertEqual(len(systems), 18)
            self.assertEqual(len(fit_link.load_aider(yml)), 18)
            res = fit_link.fit(systems, its, fit_link.load_aider(yml), lambda mid, disp: mid or disp, lambda m: strength[m])
        self.assertFalse(res["fallback"])
        self.assertEqual(res["n_matched"], 18)
        self.assertAlmostEqual(res["per_source"]["aider"]["beta"], beta, delta=0.3)
        self.assertAlmostEqual(res["beta_pub"], beta, delta=0.5)
        self.assertLess(res["lomo"]["loss"], 0.05)

    def test_fallback_and_name_matching(self):
        res = fit_link.fit([], [], [{"model": "a", "display": "a", "pass1": 0.5}], lambda *a: None, lambda m: 0.5)
        self.assertEqual((res["beta_pub"], res["se_pub"], res["fallback"]), (1.0, 0.5, True))
        self.assertEqual(fit_link.name_key("Claude 4.5 Opus (high reasoning)"), fit_link.name_key("claude-opus-4-5"))
        self.assertNotEqual(fit_link.name_key("gpt-4.5-preview"), fit_link.name_key("gpt-5.4"))
        rows = fit_link.parse_yaml_list("- model: x\n  pass_rate_1: 12.5\n  edit_format: diff\n- model: 'y'\n")
        self.assertEqual(rows, [{"model": "x", "pass_rate_1": 12.5, "edit_format": "diff"}, {"model": "y"}])


def st_inv(z):
    return min(0.98, max(0.02, 0.5 * (1 + __import__("math").erf(z / 2 ** 0.5))))


if __name__ == "__main__":
    unittest.main()
