"""Regression tests for the S1 review findings: the trainer-to-decide contracts (isotonic map, the fit_link file, V0, splits),
replay with relaxed quota, burn rate after idle time, privacy on every path (Jev, the fallback ladder, /v1/systemone, the whole
conversation) and bounded work on hostile inputs (key scan, test-summary parser, fences, MinHash, feedback ids)."""
import json
import math
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest import mock

import helpers
from helpers import inventory, maestro as m
from test_decide import NOW, E, S, ctx, post
from test_jev import Fake, canned
from test_server import MSG, ServerCase

from maestro import clients, decide as D, learn as L, s1, systemone as SO

EVALS = os.path.join(helpers.ROOT, "evals")
sys.path.insert(0, EVALS)
import alpha_eval as ev  # noqa: E402
import fit_link  # noqa: E402

try:
    import numpy  # noqa: F401  (the trainer needs it; training scripts only)
    import train_s1
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
KEY = "sk-ant-api03-" + "A" * 40
TIERS = ("haiku", "sonnet", "opus", "fable")


def timed(fn, *a):
    t0 = time.perf_counter()
    out = fn(*a)
    return out, time.perf_counter() - t0


def train_rows(n=160, seed=3):
    import random
    rnd = random.Random(seed)
    words = "fix deploy summarize design query review refactor explain migrate audit rename research".split()
    rows = []
    for i in range(n):
        w = rnd.sample(words, 3)
        rows.append({"prompt": "please %s the %s module %d and %s it" % (w[0], w[1], i % 9, w[2]), "split": "train" if i < 120 else "calib",
                     "job": rnd.choice(m.JOBS), "difficulty": rnd.randrange(4), "group": rnd.choice(["chat", "agentic"]),
                     "outcomes": {t: rnd.random() < 0.3 + 0.15 * k for k, t in enumerate(TIERS)}})
    return rows


class Tmp(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = tmp.name


# ---- 1. the isotonic map train_s1 writes is the one decide reads ----

@unittest.skipUnless(HAVE_NUMPY, "the trainer needs numpy")
class Isotonic(unittest.TestCase):
    def test_train_s1_isotonic_output_runs_through_decide(self):
        rng = numpy.random.default_rng(0)
        lg = rng.normal(0, 1.5, 3000)
        y = (rng.random(3000) < 1 / (1 + numpy.exp(-(0.6 * lg + 0.4)))).astype(float)
        iso = train_s1.isotonic(1 / (1 + numpy.exp(-lg)), y)       # what calibrate stores, on P_raw
        xs, ys = iso
        self.assertTrue(all(0 <= x <= 1 for x in xs) and xs == sorted(xs))
        cal = (1.0, 0.0, D._iso(iso), 1.0, 0.0)
        self.assertIsNotNone(cal[2])
        k = max(i for i, x in enumerate(xs) if x <= 0.5)
        want = ys[k] + (ys[k + 1] - ys[k]) * (0.5 - xs[k]) / (xs[k + 1] - xs[k])
        self.assertAlmostEqual(D._cal(0.0, cal), want, places=9)    # logit 0 is P_raw 0.5
        header = {"calib": {"final": {"chat": {"platt": [1.0, 0.0], "iso": iso}}}}
        out = D.decide([S(*("openai/a", "standard", (1, 5)))], post(), ctx(s1=header))
        self.assertTrue(out["options"] and 0 < out["options"][0]["p"] < 1)
        self.assertIsNone(D._iso({"x": xs, "y": ys}))                 # the old shape (and anything malformed) falls back to Platt
        self.assertIsNone(D._iso([[0.0, 2.0], [0.1, 0.9]]))


# ---- 2. fit_link.py -> train_s1.py --link ----

@unittest.skipUnless(HAVE_NUMPY, "the trainer needs numpy")
class LinkPipeline(Tmp):
    def test_fit_link_output_trains_a_fitted_link(self):
        import random
        rng = random.Random(1)
        zs = {"m%02d" % k: rng.uniform(-1.5, 1.5) for k in range(18)}
        strength = {k: min(0.98, max(0.02, 0.5 * (1 + math.erf(z / 2 ** 0.5)))) for k, z in zs.items()}
        aider = [{"model": k, "display": k, "pass1": 1 / (1 + math.exp(-(-0.5 + 1.5 * z + rng.gauss(0, 0.2))))} for k, z in zs.items()]
        res = fit_link.fit([], [], aider, lambda mid, disp: mid, lambda lid: strength[lid])
        jobs = {"lab/h": {"code": 0.35, "ask": 0.4}, "lab/s": {"code": 0.6, "ask": 0.7}, "lab/o": {"code": 0.7, "ask": 0.75},
                "lab/f": {"code": 0.8, "ask": 0.8}}
        res.update(fit_link.anchors(jobs.get, dict(zip(TIERS, sorted(jobs, key=lambda k: jobs[k]["code"]))), 1.2))
        self.assertEqual(set(res["zbar"]), set(TIERS))
        path = os.path.join(self.home, "link_pub.json")
        with open(path, "w") as f:
            json.dump(dict(res, catalog_sha="abc"), f)
        with open(path) as f:
            link = json.load(f)
        self.assertEqual(len(train_s1.scalars_init({"beta_pub": 1.9, "se_pub": 0.3, "catalog_sha": "x"})), len(train_s1.SNAMES))
        self.assertFalse(train_s1.linked({"beta_pub": 1.9, "se_pub": 0.3}))
        out = os.path.join(self.home, "s1.bin")
        cfg = dict(train_s1.DEFAULTS, epochs=2, min_labels=10)
        header, _ = train_s1.build(train_rows(), cfg, out, link, "", log=lambda *a: None)
        lk = header["link"]
        self.assertTrue(lk["fitted"])
        self.assertNotEqual((lk["alpha"]["chat"], lk["alpha"]["agentic"]), (-2.0, -2.4))
        self.assertGreater(abs(lk["beta"] - train_s1.scalars_init(link)[train_s1.SI["beta"]]), 1e-4)       # the data moved β
        self.assertEqual(lk["beta_pub"], [res["beta_pub"], res["se_pub"]])
        self.assertNotEqual(lk["a_chat_hat"], 1.2)                                           # the second pass used the first's a_chat
        self.assertAlmostEqual(lk["a_chat_hat"], header["irt"]["a"]["chat"], delta=0.1)
        self.assertEqual(header["irt"]["anchor_models"]["haiku"], "lab/h")
        self.assertEqual(D.params(header)["link"]["alpha"], lk["alpha"])                     # decide uses the fitted link
        h2, _ = train_s1.build(train_rows(), cfg, out, {"beta_pub": 1.9, "se_pub": 0.3}, "", log=lambda *a: None)
        self.assertFalse(h2["link"]["fitted"])                                               # no zbar: frozen placeholders, ignored
        bare = [{k: v for k, v in r.items() if k != "outcomes"} for r in train_rows()]
        h3, _ = train_s1.build(bare, cfg, out, link, "", log=lambda *a: None)
        self.assertFalse(h3["link"]["fitted"])                                               # no outcomes: nothing moves α and β


# ---- 3. V0: calibrate-v -> the header -> what the gates replay ----

@unittest.skipUnless(HAVE_NUMPY, "the trainer needs numpy")
class V0(Tmp):
    def test_v0_is_patched_into_the_header_and_checked(self):
        out = os.path.join(self.home, "s1.bin")
        shutil.copy(s1.BIN_PATH, out)
        v0 = os.path.join(self.home, "v0.json")
        with open(v0, "w") as f:
            json.dump({"V0": {"cost": 0.02, "balanced": 0.9, "quality": None}}, f)
        self.assertEqual(ev.v0_check(out, v0)[0], "fail")
        self.assertEqual(train_s1.main(["--v0", v0, "--patch", "--out", out, "--quiet"]), 0)
        md, _ = s1.load(out)
        self.assertEqual(md["header"]["decide"], {"V0": {"cost": 0.02, "balanced": 0.9, "quality": 5.0}, "V0_calibrated": False})
        self.assertEqual(s1.verify(out), [])
        self.assertEqual(D.params(md["header"])["decide"]["V0"]["balanced"], 0.9)
        self.assertEqual(ev.v0_check(out, v0)[0], "pass")
        self.assertEqual(ev.v0_check(out, os.path.join(self.home, "none.json"))[0], "skip")
        s1._S1.pop(out, None)

    def test_the_gate_replays_without_wtp_usd(self):
        inv = ev.load_inventory()
        _, _, _, cfg = ev.current_policy({"a": "what is a mutex?"}, inv)
        self.assertNotIn("wtp_usd", cfg)


# ---- 4. replay with relaxed quota (Q = 0) ----

class ReplayQuota(Tmp):
    def test_route_policy_relaxes_the_windows_and_refuses_quota_unknown(self):
        inv = ev.load_inventory()
        texts = {"a": "what is a mutex?", "b": "design a multi-region failover architecture"}
        ch, ps, info = ev.route_policy(texts, inv, {"decider": "utility"})
        self.assertEqual(info["decider"], "utility")
        ev.write_limits(self.home, "relaxed", codex=True)
        d = m.route(texts["a"], harness="claude-code", inventory=inv["quota_inventory"], config=dict(m.DEFAULT_CONFIG, decider="utility"),
                    env={"MAESTRO_HOME": self.home}, home=self.home)
        self.assertNotIn("quota_unknown", d["flags"])
        empty = tempfile.mkdtemp(dir=self.home)                     # a caller's home without limits: the replay refuses
        with self.assertRaises(RuntimeError):
            ev.route_policy(texts, inv, {"decider": "utility"}, home=empty)


# ---- 5. train_s1 splits through split.py ----

@unittest.skipUnless(HAVE_NUMPY, "the trainer needs numpy")
class Splits(Tmp):
    def test_near_duplicates_share_a_split_and_the_manifest_wins(self):
        paste = "Thanks for reaching out. I ordered the Pro plan last week and the invoice shows the wrong amount " * 6
        rows = [{"prompt": "Give me the TL;DR of this text\n```%s```" % paste, "job": "summarize"},
                {"prompt": "Group these customer feedback comments by theme\n```%s```" % paste, "job": "summarize"}]
        rows += [{"prompt": "unrelated task number %d about topic %d" % (k, k * 7), "job": "ask"} for k in range(40)]
        sp, cl = train_s1.assign_splits(rows, manifest=os.path.join(self.home, "none.json"), log=lambda *a: None)
        self.assertEqual(sp[0], sp[1])
        self.assertEqual(cl[0], cl[1])
        self.assertTrue(set(sp) <= {"train", "calib", "dev"})
        man = os.path.join(self.home, "manifest.json")
        with open(man, "w") as f:
            json.dump({"rows": {"C-1": ["C", "test", "k:C-1"], "C-2": ["C", "calib", "k:C-2"], "C-3": ["C", "legacy", "k:C-3"]}}, f)
        rows = [{"id": "C-1", "prompt": "a"}, {"id": "C-2", "prompt": "b"}, {"id": "C-3", "prompt": "c"}, {"id": "x", "prompt": "d", "split": "dev"}]
        sp, cl = train_s1.assign_splits(rows, manifest=man, log=lambda *a: None)
        self.assertEqual(sp, ["test", "calib", "test", "dev"])
        self.assertEqual(cl[:2], ["k:C-1", "k:C-2"])


# ---- 6. burn rate after idle time ----

class BurnRate(unittest.TestCase):
    def test_a_burst_then_idle_window_has_slack(self):
        hist = [[NOW - 59 * 60, 0.10], [NOW - 54 * 60, 0.20], [NOW - 49 * 60, 0.30]]
        t = {"now": NOW, "windows": {"claude": {"as_of": NOW - 60, "windows": [
            {"name": "five_hour", "used": 0.30, "reset_at": NOW + 7200, "length_s": 18000, "hist": hist}]}},
             "wcache": {}, "reserve": 0.15, "k": {}}
        wins = D.windows_of("claude", t)
        self.assertLess(wins[0][2], 0.6)                                        # U was 1.6 from the stale burst alone
        q, why, bind = D.quota_price({"quota": "claude", "fam": "sonnet"}, 0.01, 0.01, 1.0, wins, t)
        self.assertEqual((q, why, bind[0]), (0.0, None, 0.0))


# ---- 7-10. privacy on every path ----

class PrivateSkill(Tmp):
    def route(self, text, **cfg):
        fake = Fake(payload=canned("claude-code:opus", 0.9, "code", (2, 2, 1, 1)))
        d = m.route(text, harness="claude-code", inventory=inventory(), config=dict(m.load_config(self.home), **cfg),
                    env={"MAESTRO_HOME": self.home, "TYPESAFE_API_KEY": "k", "PATH": ""}, home=self.home, jev_fetch=fake)
        return d, fake

    def test_jev_never_sees_a_private_task(self):
        for decider in ("ladder", "utility"):
            d, fake = self.route("refactor the token module", private=True, decider=decider)
            self.assertEqual(fake.calls, [], decider)
            d, fake = self.route("refactor, token " + KEY, decider=decider)
            self.assertEqual(fake.calls, [], decider)
            self.assertIn("jev skipped", d["reason"])
        d, fake = self.route("refactor the token module across the services")
        self.assertEqual(len(fake.calls), 1)                                   # a plain task still goes to Jev

    def test_private_survives_a_failing_decider(self):
        for exc_at in ("decide", "s1"):
            target = mock.patch.object(D, "decide", side_effect=RuntimeError) if exc_at == "decide" else \
                mock.patch.object(s1, "posterior", side_effect=RuntimeError)
            with target:
                d = m.route("fix the failing login test", harness="claude-code", inventory=inventory(), env={"PATH": ""},
                            config=dict(m.load_config(self.home), private=True, decider="utility"), home=self.home)
            self.assertIn(d["delegate"]["kind"], ("inline", "local"), exc_at)
            self.assertIn("failed (RuntimeError)", d["reason"])

    def test_a_broken_s1_never_breaks_route(self):
        text = "fix this\n```\udcff\nx = 1\n```"
        d = m.route(text, harness="claude-code", inventory=inventory(), env={"PATH": ""}, config=m.load_config(self.home),
                    home=self.home, log=True)
        self.assertEqual(d["category"], "standard")   # debug/d1
        self.assertNotIn("failed", d["reason"])
        with mock.patch.object(s1, "posterior", side_effect=ZeroDivisionError):
            d = m.route("fix the failing login test", harness="claude-code", inventory=inventory(), env={"PATH": ""},
                        config=dict(m.load_config(self.home), decider="utility"), home=self.home)
        self.assertIn("S1 failed (ZeroDivisionError)", d["reason"])

    def test_a_malformed_header_is_refused_at_load(self):
        for edit in ({"calib": "x"}, {"train": "x"}, {"calib": {"T_job": 0}}, {"ordinal": {"tau": [0.0]}}, {"ordinal": {"tau": [0, 1, 2], "s_d": 0}},
                     {"calib": {"platt": {"tools": [1.0]}}}, {"legacy": {"U": "x"}}):
            self.assertIsNotNone(s1._bad_header(dict({"calib": {}}, **edit)), edit)
        self.assertIsNone(s1._bad_header(s1.load()[0]["header"]))


class PrivateProxy(ServerCase):
    def two(self, cfg=None, **kw):
        up = self.up(name="openai", text="pong")
        models = [E("openai/a", "standard", (1.0, 4.0), G=0.6), E("openai/b", "standard", (0.9, 3.6), G=0.6)]
        self.start(models, dict({"providers": {"openai": {"base_url": up.url}}}, **(cfg or {})), **kw)
        return up

    def test_a_key_anywhere_in_the_conversation_keeps_it_off_cloud_seats(self):
        self.two({"decider": "utility"})
        convs = [[{"role": "system", "content": "env: OPENAI_API_KEY=sk-proj-" + "A" * 48}, {"role": "user", "content": "why does my deploy fail?"}],
                 [{"role": "user", "content": "my key is " + KEY}, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "go on"}],
                 [{"role": "user", "content": "check my env"},
                  {"role": "assistant", "content": None, "tool_calls": [{"id": "t", "type": "function", "function": {"name": "sh", "arguments": "{}"}}]},
                  {"role": "tool", "tool_call_id": "t", "content": "GITHUB_TOKEN=ghp_" + "b" * 36}]]
        for msgs in convs:
            st, _, j = self.chat({"messages": msgs})
            self.assertEqual((st, j["error"]["code"]), (503, "private_no_local"), msgs[0]["content"][:20])
        st, _, _ = self.chat(MSG)
        self.assertEqual(st, 200)

    def test_private_mode_survives_a_failing_utility_decider(self):
        self.two({"decider": "utility", "private": True})
        with mock.patch.object(s1, "posterior", side_effect=RuntimeError):
            st, _, j = self.chat(MSG)
        self.assertEqual((st, j["error"]["code"]), (503, "private_no_local"))

    def test_jev_is_skipped_for_private_conversations(self):
        fake = Fake(payload=canned("openai/a", 0.9, "code", (2, 2, 1, 1)))
        self.two({"classifier": "auto"}, env_extra={"TYPESAFE_API_KEY": "k"}, jev_fetch=fake)
        st, _, _ = self.chat({"messages": [{"role": "system", "content": "token " + KEY}, {"role": "user", "content": "refactor the module"}]})
        self.assertEqual((st, fake.calls), (200, []))
        st, _, _ = self.chat(MSG)
        self.assertEqual((st, len(fake.calls)), (200, 1))

    def test_feedback_with_an_oversized_id_never_breaks_the_chat(self):
        self.two()
        st, _, _ = self.chat(MSG, {"X-Maestro-Feedback": "99999999999999999999999=good, ²=bad"})
        self.assertEqual(st, 200)
        self.assertEqual(self.admin("POST", "/api/feedback", {"id": 10 ** 30, "value": "good"})[0], 400)


class PrivateSystemone(Tmp):
    def test_private_states_never_reach_a_cloud_delegate(self):
        seen = []
        body = {"model": "maestro-s1+llm", "questions": {"tone": {"type": "choice", "criteria": {"x": "", "y": ""}}}}
        cfg = {"systemone_cloud": True, "systemone_delegate": "openai/gpt-5"}
        with mock.patch.object(clients, "complete", side_effect=lambda *a, **k: seen.append(a) or (500, {}, {})):
            for state, private in (("summarize the meeting", True), ("my key is " + KEY, False), ({"task": "x", "env": KEY}, False)):
                st, out = SO.handle(dict(body, state=state), {"cfg": dict(cfg, private=private), "inv": {}, "models": {}, "home": self.home})
                self.assertEqual((st, out["detail"][0]["type"]), (503, "private_no_local"), state)
        self.assertEqual(seen, [])

    def test_the_state_is_hashed_once_per_request(self):
        state = {"task": "fix it", "history": [{"role": "user", "content": "x" * 1000} for _ in range(4000)]}
        qs = {"q%d" % i: {"type": "score", "intent": "maestro.difficulty", "criteria": ["easy", "medium", "hard", "expert"]} for i in range(64)}
        (st, _), secs = timed(SO.handle, {"model": "maestro-s1", "state": state, "questions": qs}, {"home": self.home})
        self.assertEqual(st, 200)
        self.assertLess(secs, 0.5)


# ---- 11-19. bounded work on hostile inputs, and input ranges ----

class Bounded(Tmp):
    def test_the_key_scan_is_linear(self):
        for text in ("xeyJ" * 15000, "-eyJ" * 15000, "eyJ" * 20000, "xsk-" * 15000, "-sk-x" * 12000, "ghp_" * 15000, "x" * 60000):
            hit, secs = timed(D.is_private, text)
            self.assertLess(secs, 0.05, text[:8])
        self.assertTrue(D.is_private("-eyJ" * 1000 + ".abcdef.ghijk"))

    def test_the_test_summary_parser_is_linear_and_never_raises(self):
        for text in ("1" * 20000, "Tests: " + "1" * 20000, "Tests: 99999 " * 2500, "9" * 20000 + " passed", "exit code " + "1" * 9000):
            got, secs = timed(L.parse_tests, text)
            self.assertLess(secs, 0.05, text[:12])
            self.assertIsNone(got)
        self.assertEqual(L.parse_tests("Tests:       1 failed, 2 passed, 3 total"), (2, 3))
        self.assertEqual(L.parse_tests("===== 5 passed, 1 failed in 0.4s ====="), (5, 6))

    def test_feedback_ids_are_bounded(self):
        self.assertEqual(L.parse_feedback_header("99999999999999999999999=good, 12=bad, ²=good"), [(12, 0.0)])
        db = sqlite3.connect(":memory:")
        db.execute("create table decisions (id integer primary key)")
        for did in (10 ** 30, 0, -3, True):
            self.assertEqual(L.api_feedback(db, {"id": did, "value": "good"})[0], 400, did)

    def test_many_fences_and_huge_texts_stay_cheap(self):
        snips = "".join("Step %d\n```\nx_%d = f(%d)\n```\n" % (j, j, j) for j in range(1500))[:60000]
        for text in (snips, "```" * 20000, "```\n" + "a = {b: [c]}; " * 120000 + "\n```"):
            s1.normalize(text)
            nm, secs = timed(s1.normalize, text)
            self.assertLess(secs, 0.02)
            self.assertLessEqual(len(nm["types"]), 32 + 5)
        sig, secs = timed(D.minhash, "word " * 2000000)
        self.assertLess(secs, 0.1)
        self.assertEqual(sig, D.minhash("word " * 500))

    def test_context_tokens_are_a_nonnegative_size(self):
        inline = D.inline_seat("claude-code", "strong")
        tin = D.rank([inline], post(hints={"context_tokens": -10 ** 6}), ctx(mode="skill", est_in=1000))[0]["tin"]
        self.assertEqual(tin, 1000)
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            m.main(["route", "what is a mutex", "--context-tokens=-5"])


if __name__ == "__main__":
    unittest.main()
