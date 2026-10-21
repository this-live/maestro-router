"""learn: outcome signals, the replayed personal fit, its guards and resets, the commands and the SubagentStop hook. No text is kept."""
import argparse
import io
import json
import math
import os
import random
import sqlite3
import stat
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import helpers  # noqa: F401
from helpers import maestro as R

from maestro import learn as L

NOW = 1790000000.0
SEAT, KEY = "codex:gpt-6.1-sol", "codex:gpt-6.1-sol|chat"


def sig(x):
    return 1 / (1 + math.exp(-x))


def decision(i, ts, seat=SEAT, job="debug", mu=0.5, th=1.0, policy="s1:aaaa", fam="gpt-6.1-sol", group="chat", **kw):
    row = {"id": "d%07x" % i, "ts": ts, "job_p": {job: 1.0}, "b": [mu, 0.7], "group": group, "chosen": seat, "p": 0.7,
           "options": [[seat, 0.7, 1.0]], "decider": "utility", "harness": "claude-code", "policy": policy, "minhash": [i] * 16,
           "trace": {"mu_b": mu, "sd_b": 0.7, "theta": th, "sd_s": 0.3, "a": 1.2, "group": group, "api_usd": 0.02, "quota": "codex", "fam": fam}}
    row.update(kw)
    return row


def write_log(home, rows, outcomes):
    with open(os.path.join(home, L.DECISIONS), "a") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(os.path.join(home, L.OUTCOMES), "a") as f:
        f.writelines(json.dumps(o) + "\n" for o in outcomes)


def simulate(home, n, true_o=0.8, seed=1, start=NOW - 30 * 86400, policy="s1:aaaa", seat=SEAT, fam="gpt-6.1-sol", job="debug", i0=0):
    """n decisions whose seat is true_o logits better than its logged θ̄, with explicit outcomes drawn from that truth."""
    rng, rows, outs = random.Random(seed), [], []
    for i in range(i0, i0 + n):
        mu, th = rng.uniform(-1.5, 2.5), 1.0
        p = sig(1.2 * (th + true_o - mu) / math.sqrt(1 + math.pi / 8 * 1.44 * (0.49 + 0.09)))
        d = decision(i, start + i * 600, seat=seat, mu=mu, th=th, policy=policy, fam=fam, job=job)
        rows.append(d)
        outs.append({"decision_id": d["id"], "ts": d["ts"] + 60, "signal": "explicit", "value": float(rng.random() < p), "weight": 1.0})
    write_log(home, rows, outs)
    return rows


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = self.tmp.name
        self.cfg = R.load_config(self.home)

    def tearDown(self):
        self.tmp.cleanup()

    def personal(self):
        with open(os.path.join(self.home, L.PERSONAL)) as f:
            return json.load(f)

    def run_(self, **kw):
        kw.setdefault("now", NOW)
        kw.setdefault("cfg", self.cfg)
        kw.setdefault("cal", {})
        return L.run(self.home, **kw)


class Signals(Case):
    def test_parse_tests(self):
        cases = {"===== 12 passed, 1 failed in 0.31s =====": (12, 13), "Tests:       1 failed, 12 passed, 13 total": (12, 13),
                 "test result: ok. 7 passed; 0 failed; 0 ignored": (7, 7), "Ran 13 tests in 0.1s\n\nFAILED (failures=1, errors=1)": (11, 13),
                 "Ran 5 tests in 0.0s\n\nOK (skipped=1)": (5, 5), "== 3 passed in 1s ==\n$ ls\nExit code: 1": (3, 3), "Exit code: 2": (0, 1),
                 "exited with code 0": (1, 1), "== 2 errors in 0.1s ==": (0, 2), "nothing to see": None}
        for text, want in cases.items():
            self.assertEqual(L.parse_tests(text), want, text)

    def test_record_keeps_numbers_only(self):
        row = L.record(self.home, "d1", "explicit", 1.0, {"tests": [3, 4], "note": "the secret prompt", "n": 2})
        self.assertEqual((row["weight"], row["detail"]), (1.0, {"tests": [3, 4], "n": 2}))
        path = os.path.join(self.home, L.OUTCOMES)
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        self.assertNotIn("secret", open(path).read())
        self.assertEqual(L.record(self.home, "d1", "reask", 0.0)["weight"], 0.0)          # implicit signals: weight 0 in v1

    def test_migrate_and_api_feedback(self):
        from maestro import server as S
        db = S.open_db(self.home)
        L.migrate(db)
        L.migrate(db)
        cols = {r[1] for r in db.execute("pragma table_info(decisions)")}
        self.assertTrue({c for c, _ in L.COLUMNS} <= cols)
        db.execute("insert into decisions (ts, chosen) values (?, ?)", (NOW, "openai/gpt-5"))
        self.assertEqual(L.api_feedback(db, {"id": 1, "value": "good"})[0], 200)
        self.assertEqual(L.api_feedback(db, {"id": 9, "value": "bad"})[0], 404)
        self.assertEqual(L.api_feedback(db, {"id": 1, "value": "meh"})[0], 400)
        self.assertEqual(db.execute("select decision_id, signal, value, weight from outcomes").fetchall(), [(1, "explicit", 1.0, 1.0)])
        self.assertEqual(L.parse_feedback_header("12=good, 13=bad,x=good, 14=partial"), [(12, 1.0), (13, 0.0), (14, 0.5)])
        db.close()

    def test_outcome_demotions_need_two_fails_and_no_success(self):
        import threading
        from maestro import server as S
        db = S.open_db(self.home)
        app = S.App.__new__(S.App)
        app.db, app.db_lock = db, threading.Lock()
        for i, chosen in enumerate(["local:x/a", "local:x/a", "local:x/b", "local:x/a"]):
            db.execute("insert into decisions (ts, job, chosen) values (?, 'code', ?)", (NOW + i, chosen))
        L.record(None, 1, "explicit", 0.0, db=db)
        L.record(None, 2, "explicit", 0.0, db=db)
        L.record(None, 3, "explicit", 0.0, db=db)
        self.assertEqual(app.outcome_demotions("code"), {"local:x/a"})
        L.record(None, 4, "explicit", 1.0, db=db)   # a later success on a lifts the demotion
        self.assertEqual(app.outcome_demotions("code"), set())
        self.assertEqual(app.outcome_demotions("write"), set())
        db.close()

    def test_implicit_proxy_signals(self):
        prev = {"ts": NOW, "sig": [1] * 16, "rank": 2}
        self.assertEqual(L.implicit(prev, {"ts": NOW + 60, "text": "use the frontier model", "rank": 4}), [("escalate", 0.0)])
        self.assertEqual(L.implicit(prev, {"ts": NOW + 60, "text": "x", "rank": 1, "model": "groq/b"}), [("override", 1.0)])
        self.assertEqual(L.implicit(prev, {"ts": NOW + 60, "text": "still failing after that", "sig": [2] * 16}), [("reask", 0.0)])
        self.assertEqual(L.implicit(prev, {"ts": NOW + 60, "text": "different words", "sig": [1] * 12 + [0] * 4}), [("reask", 0.0)])
        self.assertEqual(L.implicit(prev, {"ts": NOW + 3600, "text": "still failing", "sig": [1] * 16}), [])
        self.assertEqual(L.implicit(prev, {"ts": NOW + 60, "text": "Thanks, that works"}), [("thanks", 1.0)])

    def test_explore_is_off_by_default_and_exact_when_on(self):
        ranked, post = [{"eu": 1.0}, {"eu": 0.99}], {"stakes": 0.2, "detectable": 0.9}
        self.assertEqual(L.explore(ranked, post, 1.0, 0.0), (0, 1.0))
        lo, hi = types.SimpleNamespace(random=lambda: 0.01), types.SimpleNamespace(random=lambda: 0.5)
        self.assertEqual(L.explore(ranked, post, 1.0, 0.05, lo), (1, 0.05))
        self.assertEqual(L.explore(ranked, post, 1.0, 0.05, hi), (0, 0.95))
        self.assertEqual(L.explore(ranked, dict(post, stakes=1.0), 1.0, 0.05, lo), (0, 1.0))
        self.assertEqual(L.explore(ranked, dict(post, detectable=0.3), 1.0, 0.05, lo), (0, 1.0))
        self.assertEqual(L.explore([{"eu": 1.0}, {"eu": 0.5}], post, 1.0, 0.05, lo), (0, 1.0))
        self.assertEqual(L.explore(ranked, post, 1.0, 0.5, lo), (1, 0.1))                    # capped at 0.1

    def test_hook_records_tests_for_the_latest_subagent_decision(self):
        tr = os.path.join(self.home, "transcript.jsonl")
        with open(tr, "w") as f:
            f.write(json.dumps({"type": "user", "message": {"content": [{"type": "tool_result", "content":
                    "PRIVATE-CODE-TEXT\n===== 3 passed, 1 failed in 0.12s =====\n"}]}}) + "\n")
        write_log(self.home, [decision(1, NOW - 3000, seat="claude-code:opus"), decision(2, NOW - 300, seat="claude-code:sonnet"),
                              decision(3, NOW - 200, seat="local:spark/qwen3")], [])
        row = L.from_hook(json.dumps({"transcript_path": tr, "hook_event_name": "SubagentStop"}), self.home, now=NOW)
        self.assertEqual((row["decision_id"], row["signal"], row["value"], row["detail"]), ("d0000002", "tests", 0.75, {"tests": [3, 4]}))
        for name in os.listdir(self.home):
            if name != "transcript.jsonl":
                self.assertNotIn("PRIVATE", open(os.path.join(self.home, name)).read(), name)
        self.assertIsNone(L.from_hook(json.dumps({"transcript_path": tr}), self.home, now=NOW + 3600))
        self.assertIsNone(L.from_hook("not json", self.home, now=NOW))
        self.assertIsNone(L.from_hook(json.dumps({"transcript_path": "/nonexistent"}), self.home, now=NOW))

    def test_escalation_is_derived_from_the_decision_log(self):
        a = decision(1, NOW - 1000, th=1.0, minhash=[7] * 16)
        b = decision(2, NOW - 700, th=2.0, minhash=[7] * 16)                               # same task, stronger seat, 5 min later
        c = decision(3, NOW - 600, th=1.0, minhash=[9] * 16)
        d = decision(4, NOW - 400, th=1.0, minhash=[9] * 16, hints={"stakes": "high"})     # same seat, stronger hint
        e = decision(5, NOW - 300, th=0.5, minhash=[5] * 16)
        f = decision(6, NOW + 900, th=3.0, minhash=[5] * 16)                               # over 15 min later
        out = L._escalations([a, b, c, d, e, f])
        self.assertEqual([(o["decision_id"], o["value"], o["weight"]) for o in out], [("d0000001", 0.0, 0.7), ("d0000003", 0.0, 0.7)])


class Fit(Case):
    def test_offset_is_learned_and_the_replay_is_deterministic(self):
        simulate(self.home, 400, true_o=0.8)
        rep = self.run_()
        self.assertTrue(rep["written"], rep)
        p = self.personal()
        o, db = p["seat_offsets"][KEY], p["job_b_shift"].get("debug", 0.0)   # one seat, one job: the priors split o - Δb between them
        self.assertGreater(o, 0.25)
        self.assertGreater(o - db, 0.55)
        self.assertLessEqual(o, 1.0)
        self.assertEqual((p["n_outcomes"], p["weights_sha"], p["fitted_at"], p["snapshots"]), (400, "s1:aaaa", NOW, 3))
        self.assertEqual(stat.S_IMODE(os.stat(os.path.join(self.home, L.PERSONAL)).st_mode), 0o600)
        self.assertLess(os.path.getsize(os.path.join(self.home, L.PERSONAL)), 20 * 1024)
        again = self.run_(dry_run=True)["personal"]
        self.assertEqual(json.dumps(again, sort_keys=True), json.dumps(p, sort_keys=True))
        self.assertFalse(os.path.exists(os.path.join(self.home, L.PERSONAL + ".1")))
        self.run_()
        self.assertTrue(os.path.exists(os.path.join(self.home, L.PERSONAL + ".1")))

    def test_anchor_offsets_use_decides_key(self):
        simulate(self.home, 200, true_o=0.6, seat="claude-code:opus", fam="opus")
        self.assertIn("anthropic/opus|chat", self.run_(dry_run=True)["personal"]["seat_offsets"])

    def test_job_shift(self):
        simulate(self.home, 400, true_o=-0.6, job="debug")       # debug tasks are harder than predicted: Δb_debug > 0 (with the offset)
        p = self.run_(dry_run=True)["personal"]
        self.assertGreater(p["job_b_shift"].get("debug", 0) - p["seat_offsets"][KEY], 0.3)

    def test_caps(self):
        simulate(self.home, 600, true_o=4.0)
        p = self.run_(dry_run=True)["personal"]
        self.assertEqual(p["seat_offsets"][KEY], 1.0)
        self.assertLessEqual(max(abs(v) for v in p["job_b_shift"].values()), 0.5)

    def test_small_data_moves_at_most_point_two(self):
        simulate(self.home, 20, true_o=3.0)
        p = self.run_(dry_run=True)["personal"]
        self.assertLessEqual(abs(p["seat_offsets"][KEY]), 0.2)
        self.assertLessEqual(abs(p["platt"][0] - 1.0), 0.2)
        self.assertLessEqual(abs(p["platt"][1]), 0.2)

    def test_too_few_outcomes(self):
        simulate(self.home, 6)
        rep = self.run_()
        self.assertEqual(rep["written"], False)
        self.assertIn("not enough outcomes", rep["why"])

    def test_time_split_guard_keeps_the_previous_file(self):
        simulate(self.home, 240, true_o=1.5, seed=3)
        simulate(self.home, 60, true_o=-1.5, seed=4, start=NOW - 10 * 86400, i0=1000)   # the last 20% contradicts the first 80%
        with open(os.path.join(self.home, L.PERSONAL), "w") as f:
            json.dump({"version": 1, "previous": True}, f)
        rep = self.run_()
        self.assertFalse(rep["written"])
        self.assertIn("time-split", rep["why"])
        self.assertGreater(rep["guard"]["personal_ll"], rep["guard"]["global_ll"])
        self.assertTrue(self.personal()["previous"])

    def test_replay_guard(self):
        self.assertAlmostEqual(L.replay_dll(1.0, 0.0), 0.0)
        self.assertGreater(L.replay_dll(1.0, 0.8), 0.02)
        self.assertLess(L.replay_dll(1.05, 0.05), 0.02)
        self.assertAlmostEqual(L.replay_dll(1.0, 1.0, [[0.0, 0.5, 1.0]]), -0.5 * math.log(sig(1)) - 0.5 * math.log(1 - sig(1)) - math.log(2), places=6)
        simulate(self.home, 300)
        with mock.patch.object(L, "replay_dll", return_value=0.05):
            rep = self.run_()
        self.assertFalse(rep["written"])
        self.assertIn("replay guard", rep["why"])

    def test_reset_on_new_weights(self):
        simulate(self.home, 200, true_o=1.5, policy="s1:old", seed=5)
        simulate(self.home, 100, true_o=-0.3, policy="s1:new", seed=6, start=NOW - 5 * 86400, i0=1000)
        _, rows = L.samples(self.home, now=NOW)
        rows = L._resets(rows)
        self.assertEqual(sum(r["u"] for r in rows), 100)                     # the latest policy's rows only
        o_new = L.fit(rows)["o"][KEY]
        o_all = L.fit(L._resets(rows, "s1:old") and [dict(r, u=1.0) for r in rows])["o"][KEY]
        self.assertLess(o_new, o_all - 0.3)

    def test_reset_when_the_seats_theta_moves(self):
        rows = [{"key": KEY, "top": "debug", "th": th, "policy": "p"} for th in (1.0, 1.1, 1.6, 1.62)]
        self.assertEqual([r["u"] for r in L._resets(rows)], [0.0, 0.0, 1.0, 1.0])

    def test_personal_values_in_the_trace_are_backed_out(self):
        rows = simulate(self.home, 1)
        tr = dict(rows[0]["trace"], o=0.4, db=0.2, theta=1.4, mu_b=rows[0]["trace"]["mu_b"] + 0.2)
        write_log(self.home, [dict(rows[0], id="dx", trace=tr)], [{"decision_id": "dx", "ts": NOW, "signal": "explicit", "value": 1.0}])
        r = [x for x in L.samples(self.home, now=NOW)[1] if x["ts"] == rows[0]["ts"]]
        self.assertAlmostEqual(r[-1]["th"], 1.0)
        self.assertAlmostEqual(r[-1]["mu"], rows[0]["trace"]["mu_b"])

    def test_proxy_rows_from_the_db(self):
        from maestro import server as S
        db = S.open_db(self.home)
        L.migrate(db)
        rng = random.Random(2)
        for i in range(60):
            mu = rng.uniform(-1.5, 2.5)
            s1 = {"job_p": {"code": 1.0}, "b": [mu, 0.7], "group": "chat", "trace": {"mu_b": mu, "sd_b": 0.7, "theta": 1.0, "sd_s": 0.3, "a": 1.2,
                                                                                       "group": "chat", "fam": "gpt-5"}}
            cur = db.execute("insert into decisions (ts, chosen, s1, policy) values (?,?,?,?)", (NOW - 86400 + i, "openai/gpt-5", json.dumps(s1), "p"))
            L.record(None, cur.lastrowid, "explicit", float(rng.random() < sig(1.2 * (1.8 - mu))), db=db, ts=NOW - 86000 + i)
        decs, rows = L.samples(self.home, db, now=NOW)
        self.assertEqual((len(rows), rows[0]["key"]), (60, "openai/gpt-5|chat"))
        self.assertGreater(L.fit(L._resets(rows))["o"]["openai/gpt-5|chat"], 0.2)
        db.close()

    def test_k_from_quota_snapshots(self):
        D = types.ModuleType("maestro.decide")
        D.read_windows = lambda home: {"codex": {"windows": [{"name": "primary", "length_s": 18000, "hist": [[NOW - 9000, 0.1], [NOW - 600, 0.3]]}]}}
        D.table = lambda: {"pacing": {"k_prior": {"short": 0.02, "long": 0.003}}}
        D.learn_k = lambda samples, tasks, prior, k=None: 0.05 if tasks else prior
        simulate(self.home, 50, start=NOW - 8000, i0=0)
        with mock.patch.dict(sys.modules, {"maestro.decide": D}):
            p = self.run_(dry_run=True)
        self.assertEqual(p["personal"]["k"], {"codex/primary|gpt-6.1-sol": 0.05})

    def test_personalize_off(self):
        simulate(self.home, 100)
        self.assertEqual(self.run_(cfg=dict(self.cfg, personalize=False))["written"], False)

    def test_implicit_signals_count_only_after_an_audit(self):
        rows = simulate(self.home, 40)
        write_log(self.home, [], [{"decision_id": r["id"], "ts": r["ts"] + 90, "signal": "reask", "value": 0.0, "weight": 0.0} for r in rows])
        self.assertEqual(len(L.samples(self.home, now=NOW)[1]), 40)
        answers = iter(["y"] * 3 + ["n"] * 37)
        data = L.audit(self.home, ask=lambda prompt: next(answers), out=lambda *a: None)
        self.assertEqual((data["signals"]["reask"]["n"], data["signals"]["reask"]["precision"]), (40, 0.925))
        self.assertEqual(len(L.samples(self.home, now=NOW)[1]), 80)
        self.assertNotIn("debug", open(os.path.join(self.home, L.AUDIT)).read())


class Commands(Case):
    def call(self, fn, stdin="", **kw):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, {"MAESTRO_HOME": self.home}), mock.patch.object(sys, "stdin", io.StringIO(stdin)), \
                redirect_stdout(out), redirect_stderr(err):
            rc = fn(argparse.Namespace(**kw))
        return rc, out.getvalue() + err.getvalue()

    def test_outcome(self):
        write_log(self.home, [decision(1, NOW - 100), decision(2, NOW - 50)], [])
        self.assertEqual(self.call(L.cmd_outcome, id="last", verdict="pass", tests=None, from_hook=False)[0], 0)
        self.assertEqual(self.call(L.cmd_outcome, id="d0000001", verdict=None, tests="3/4", from_hook=False)[0], 0)
        self.assertEqual(self.call(L.cmd_outcome, id="nope", verdict="fail", tests=None, from_hook=False)[0], 1)
        self.assertEqual(self.call(L.cmd_outcome, id="last", verdict="maybe", tests=None, from_hook=False)[0], 1)
        rows = L._jsonl(os.path.join(self.home, L.OUTCOMES))
        self.assertEqual([(r["decision_id"], r["signal"], r["value"]) for r in rows], [("d0000002", "explicit", 1.0), ("d0000001", "tests", 0.75)])
        self.assertEqual(self.call(L.cmd_outcome, stdin="garbage", id=None, verdict=None, tests=None, from_hook=True)[0], 0)

    def test_feedback(self):
        from maestro import server as S
        db = S.open_db(self.home)
        db.execute("insert into decisions (ts, chosen) values (?, ?)", (NOW, "openai/gpt-5"))
        db.close()
        self.assertEqual(self.call(L.cmd_feedback, id="1", value="bad")[0], 0)
        self.assertEqual(self.call(L.cmd_feedback, id="7", value="bad")[0], 1)
        db = sqlite3.connect(os.path.join(self.home, L.DB))
        self.assertEqual(db.execute("select decision_id, value from outcomes").fetchall(), [(1, 0.0)])
        db.close()

    def test_learn_dry_run_report_and_reset(self):
        simulate(self.home, 100, start=NOW - 86400 * 5)
        with mock.patch.object(L.time, "time", return_value=NOW):
            rc, out = self.call(L.cmd_learn, reset=False, audit=False, report=False, dry_run=True)
        self.assertEqual((rc, json.loads(out)["written"]), (0, False))
        self.assertFalse(os.path.exists(os.path.join(self.home, L.PERSONAL)))
        with mock.patch.object(L.time, "time", return_value=NOW):
            self.call(L.cmd_learn, reset=False, audit=False, report=False, dry_run=False)
            self.call(L.cmd_learn, reset=False, audit=False, report=False, dry_run=False)
        self.assertTrue(os.path.exists(os.path.join(self.home, L.PERSONAL + ".1")))
        rc, out = self.call(L.cmd_learn, reset=False, audit=False, report=True, dry_run=False)
        self.assertEqual(json.loads(out)["personal"]["version"], 1)
        self.call(L.cmd_learn, reset=True, audit=False, report=False, dry_run=False)
        self.assertEqual([n for n in os.listdir(self.home) if n.startswith(L.PERSONAL)], [])


if __name__ == "__main__":
    unittest.main()
