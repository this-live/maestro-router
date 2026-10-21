"""The S1 decision layer (decide.py): success model, tokens and cost, quota pacing, utility, escalation, logs; plus the quota.py
snapshot history it reads. Numbers are checked against the spec's reference implementation (design/ref/decide_ref.py, copied below)."""
import ast
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import helpers
from helpers import inventory, local, maestro as core

from maestro import decide as D
from maestro import quota as Q

NOW = 1_790_000_000.0
H = 3600.0


# ---- the reference implementation's formulas (design/ref/decide_ref.py), verbatim ----
def SQ(x):
    return 1 / (1 + math.exp(-x))


def ref_p_succ(theta, sd_t, mu_b, sd_b, a, A=1.0, B=0.0, z=0.0):
    raw = a * (theta - z * sd_t - mu_b) / math.sqrt(1 + math.pi / 8 * a * a * (sd_b ** 2 + (0 if z else sd_t ** 2)))
    return SQ(A * raw + B)


def ref_g_mult(used, tau, L, hist_rate, reserve, du, gmax, m=0.10):
    r_win = used / max(600.0, L - tau)
    r = r_win if hist_rate is None else ((0.5 * hist_rate + 0.5 * r_win) if L <= 6 * 3600 else (0.15 * hist_rate + 0.85 * r_win))
    U = used + r * tau + du
    S = (1 - reserve) - U
    if S >= m:
        return 0.0, U, S
    if S >= 0:
        return gmax * (m - S) / m, U, S
    return gmax, U, S


def ref_tokens_agentic(T, base, Oh, step=1500):
    return T * (base + Oh) + step * T * (T - 1) / 2, (T - 1) / T


def ref_api_cost(tin, chit, tout, pin, pout, pcr):
    return (tin * (1 - chit) * pin + tin * chit * pcr + tout * pout) / 1e6


# ---- fixtures ----
def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def post(**kw):
    p = {"job_p": {"ask": 1.0}, "b": [-1.0, 0.5], "group": "chat", "tools": 0.0, "self_contained": 1.0, "stakes": 1.0,
         "detectable": 0.8, "ambiguity": 0.0, "tok_out": [math.log(500), 0.5], "turns": [0.0, 0.0], "ood": 0.0, "source": "s1"}
    p.update(kw)
    return p


def E(mid, cat="standard", cost=(1.0, 5.0), G=0.6, access=None, **kw):
    """A normalized models.json entry as server.App.candidates holds it under "m"."""
    access = access or ("local" if mid.startswith("local:") else "subscription-cli" if mid.startswith("cli:") else "api")
    prov, _, model = mid.partition("/")
    e = {"id": mid, "provider": prov, "model": model, "access": access, "tools_ok": access != "subscription-cli", "context": 200000,
         "cost_in": cost[0] if access == "api" else 0.0, "cost_out": cost[1] if access == "api" else 0.0, "G": G, "jobs": {},
         "category": cat, "serves": cat, "basis": "bench", "qp": prov[4:] if access == "subscription-cli" else prov}
    e.update(kw)
    return e


def S(mid, cat="standard", cost=(1.0, 5.0), **kw):
    return D.seat_of({"m": E(mid, cat, cost, **kw), "latency": None, "headroom": kw.pop("headroom", None)})


HAIKU, OPUS = ("anthropic/claude-haiku-4-5", "fast", (1, 5)), ("anthropic/claude-opus-5-5", "strong", (4, 20))


def router_seat(model, tier, seat="claude-code", kind="subagent", **kw):
    return D.seat_of(dict({"name": "%s:%s" % (seat, model), "seat": seat, "kind": kind, "model": model, "serves": tier, "local": False,
                           "tools": True, "effort": None}, **kw))


def claude_windows(used5=0.0, reset5=3 * H, used7=0.0, reset7=4 * 86400, as_of=NOW, hist5=(), hist7=()):
    return {"claude": {"as_of": as_of, "windows": [
        {"name": "five_hour", "used": used5, "reset_at": NOW + reset5, "length_s": 18000, "hist": [list(h) for h in hist5]},
        {"name": "seven_day", "used": used7, "reset_at": NOW + reset7, "length_s": 604800, "hist": [list(h) for h in hist7]}]}}


def ctx(**kw):
    c = {"mode": "proxy", "windows": {}, "now": NOW}
    c.update(kw)
    return c


class Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        Q.reset_state(self.home)
        self.addCleanup(Q.reset_state, self.home)


# ════════════════════════════════════════════════════════════════════════════

class SuccessModel(unittest.TestCase):
    def test_matches_the_reference_p_succ(self):
        rng = random.Random(1)
        for _ in range(300):
            th, sd, mu, sb, a = rng.uniform(-2, 4), rng.uniform(0.05, 0.9), rng.uniform(-2, 3), rng.uniform(0.2, 1.5), rng.uniform(0.8, 2)
            A, B, z = rng.uniform(0.7, 1.3), rng.uniform(-0.3, 0.3), rng.uniform(0.05, 1.5)
            cal = (A, B, None, 1.0, 0.0)
            self.assertAlmostEqual(D.p_success(th, sd, mu, sb, a, None, cal), ref_p_succ(th, sd, mu, sb, a, A, B, 0.0), places=12)
            self.assertAlmostEqual(D.p_success(th, sd, mu, sb, a, z, cal), ref_p_succ(th, sd, mu, sb, a, A, B, z), places=12)

    def test_reference_scenario_probabilities(self):
        """decide_ref.py's tool-group code task: P̃ per seat with ω·(1−sc) and the subagent harness sd for delegated seats."""
        want = {"inline:sonnet": 0.526, "claude-code:haiku": 0.174, "claude-code:sonnet": 0.394, "claude-code:opus": 0.669,
                "claude-code:fable": 0.862, "codex:gpt-6.1-sol": 0.686}
        seats = [("inline:sonnet", 1.1, 0.0, True), ("claude-code:haiku", 0.0, 0.15, False), ("claude-code:sonnet", 1.1, 0.15, False),
                 ("claude-code:opus", 2.2, 0.15, False), ("claude-code:fable", 3.3, 0.15, False), ("codex:gpt-6.1-sol", 2.4, 0.5, False)]
        for name, th, sd, inline in seats:
            th_eff = th - (0 if inline else 0.5 * (1 - 0.3))
            sd_eff = math.sqrt(sd ** 2 + (0 if inline else 0.3 ** 2))
            self.assertAlmostEqual(D.p_success(th_eff, sd_eff, 1.0, 0.8, 1.2, 0.5), want[name], places=3)

    def test_p_rises_with_theta_and_falls_with_b(self):
        ps = [D.p_success(th, 0.3, 0.5, 0.7, 1.2) for th in (-1, 0, 1, 2, 3)]
        self.assertEqual(ps, sorted(ps))
        ps = [D.p_success(1.0, 0.3, b, 0.7, 1.2, 0.5) for b in (-1, 0, 1, 2, 3)]
        self.assertEqual(ps, sorted(ps, reverse=True))

    def test_theta_sources(self):
        prm = D.params()
        jp = {"code": 0.5, "ask": 0.5}
        # an anchor: its tier's learned θ plus β times its job profile
        al, beta = prm["link"]["alpha"], prm["link"]["beta"]
        th, sd, linked = D.theta({"id": "x", "entry": {"jobs": {"code": 0.9, "ask": 0.2}}}, jp, prm, "opus")
        z = [D._ND.inv_cdf(0.9), D._ND.inv_cdf(0.2)]
        self.assertAlmostEqual(th, 2.2 + beta * 0.5 * ((z[0] - sum(z) / 2) + (z[1] - sum(z) / 2)))
        self.assertEqual((sd, linked), (0.15, False))
        # any other model: the catalog link α_g + β·Φ⁻¹(strength), α by the job's group, sd by the score's basis
        th, sd, linked = D.theta({"id": "y", "entry": {"jobs": {"code": 0.9, "ask": 0.2}, "basis": "blend"}}, jp, prm)
        self.assertAlmostEqual(th, 0.5 * (al["agentic"] + beta * z[0]) + 0.5 * (al["chat"] + beta * z[1]))
        hdr = D.params({"link": {"alpha": {"chat": -2.0, "agentic": -2.4}, "beta": 1.6}})      # a header overrides the defaults
        self.assertAlmostEqual(D.theta({"id": "y", "entry": {"jobs": {"code": 0.9, "ask": 0.2}}}, jp, hdr)[0],
                               0.5 * (-2.4 + 1.6 * z[0]) + 0.5 * (-2.0 + 1.6 * z[1]))
        self.assertEqual((sd, linked), (0.5, True))
        # discover's precomputed theta wins over the lazy computation
        th, sd, _ = D.theta({"id": "y", "entry": {"jobs": {"code": 0.9}, "theta": {"code": [1.0, 0.4], "ask": [0.0, 0.2]}}}, jp, prm)
        self.assertAlmostEqual(th, 0.5)
        self.assertAlmostEqual(sd, 0.3)
        # a user-run anchor (anchors.json) overrides the link
        self.assertEqual(D.theta({"id": "y", "entry": {"G": 0.9}}, jp, prm, None, {"theta_chat": 1.4, "sd": 0.22}), (1.4, 0.22, False))
        # no data: decide.json seat_prior, then the category's anchor tier with the prior sd
        self.assertEqual(D.theta({"id": "claude-code:opus", "entry": {}}, jp, prm)[:2], (2.2, 0.15))
        self.assertEqual(D.theta({"id": "z", "entry": {}, "cat": "standard"}, jp, prm)[:2], (1.1, 0.75))
        self.assertEqual(D.theta({"id": "z", "entry": {}, "cat": "local"}, jp, prm)[:2], (-1.1, 0.75))

    def test_escalation_posterior_matches_the_spec_check(self):
        mu, sd = D.update_b(0.0, 1.0, math.log(0.8 / 0.2) / 1.2, 1.2)
        self.assertAlmostEqual(mu, 0.724, places=3)
        self.assertAlmostEqual(sd, 0.884, places=3)
        self.assertGreater(mu, 0.0)

    def test_isotonic_and_personal_calibration(self):
        iso = ([0.0, 0.5, 1.0], [0.0, 0.8, 1.0])
        self.assertAlmostEqual(D._cal(0.0, (1.0, 0.0, iso, 1.0, 0.0)), 0.8)
        self.assertAlmostEqual(D._cal(math.log(3), (1.0, 0.0, iso, 1.0, 0.0)), 0.9)
        self.assertAlmostEqual(D._cal(0.3, (1.0, 0.0, None, 2.0, -0.1)), SQ(0.5))


class TokensCostQuota(unittest.TestCase):
    def test_agentic_tokens_and_cost_match_the_reference(self):
        p = post(group="agentic", job_p={"code": 1.0}, self_contained=0.3, tok_out=[math.log(1200), 0.6], turns=[math.log(8), 0.5])
        t = D._task(p, {"mode": "skill", "est_in": 1200, "windows": {}, "now": NOW})
        T = math.exp(math.log(8) + 0.125)
        self.assertAlmostEqual(t["turns"], T)
        tok = math.exp(math.log(1200) + 0.18)
        self.assertAlmostEqual(t["V"], 0.5 * min(8, max(0.5, math.sqrt(tok / 1000 * T))))
        for s, base, oh in ((router_seat("opus", "strong"), 1200 + 20000 * 0.7, 18000),
                            (router_seat("gpt-6.1-sol", "standard", "codex", "cli"), 1200 + 20000 * 0.7, 10000),
                            (D.inline_seat("claude-code", "standard"), 1200 + 30000, 0)):
            s = dict(s, tier=None)
            tin, hit, tout, extra, fits = D.tokens(s, t)
            rtin, rhit = ref_tokens_agentic(T, base, oh)
            self.assertAlmostEqual(tin, rtin, places=6)
            self.assertAlmostEqual(hit, rhit)
            self.assertTrue(fits)
            self.assertAlmostEqual(D.cost(tin, hit, 5000, (4, 20, 0.4)), ref_api_cost(tin, hit, 5000, 4, 20, 0.4))

    def pacing(self, windows, used=None, api=1.0, gmax=1.0, fam="opus", reserve=0.15, prov="claude", k=None):
        t = {"wcache": {}, "now": NOW, "windows": windows, "reserve": reserve, "k": k or {}}
        return D.quota_price({"quota": prov, "fam": fam}, api, 1.0, gmax, D.windows_of(prov, t), t)

    def test_pacing_matches_the_reference_table(self):
        # fresh 5 h window
        self.assertEqual(self.pacing(claude_windows(0.0, 5 * H, 0.0, 7 * 86400), api=0.5)[0], 0.0)
        self.assertEqual(ref_g_mult(0.0, 5 * H, 5 * H, None, 0.15, 0.01, 1.0)[0], 0.0)
        # 42% used, 70 min to reset, burning 20%/h (two samples 30 min apart): use it or lose it
        hist = ((NOW - 1800, 0.42 - 0.1), (NOW, 0.42))
        g, why, bind = self.pacing(claude_windows(0.42, 70 * 60, hist5=hist), api=0.5)
        self.assertEqual((g, why), (0.0, None))
        _, U, S = ref_g_mult(0.42, 70 * 60, 5 * H, 0.2 / 3600, 0.15, 0.01, 1.0)
        self.assertAlmostEqual(U, 0.6106, places=4)
        self.assertAlmostEqual(S, 0.2394, places=4)
        self.assertAlmostEqual(bind[1], S, places=9)   # the binding window's slack is the reference's S
        # weekly window 80% used with 4 days left: g_max; 30% used with 2 days left: 0
        self.assertEqual(self.pacing(claude_windows(0.0, 5 * H, 0.80, 4 * 86400), api=1.0, gmax=3.0)[0], 3.0)
        self.assertEqual(self.pacing(claude_windows(0.0, 5 * H, 0.30, 2 * 86400), api=1.0)[0], 0.0)

    def test_pacing_matches_the_reference_on_a_grid(self):
        rng = random.Random(7)
        for _ in range(400):
            L = rng.choice((5 * H, 7 * 86400))
            used, tau, api = rng.uniform(0, 0.8), rng.uniform(0, L), rng.uniform(0, 3)
            rate = rng.choice((None, rng.uniform(0, 0.4) / H))
            hist = () if rate is None else ((NOW - 1800, used - rate * 1800), (NOW, used))
            if rate is not None and tau > L - 1800:
                hist = ()   # the samples would predate the window
                rate = None
            short = L <= 6 * H
            w = claude_windows(used, tau, 0.0, 7 * 86400, hist5=hist) if short else claude_windows(0.0, 5 * H, used, tau, hist7=hist)
            k = 0.02 if short else 0.003
            g, why, _ = self.pacing(w, api=api)
            want = ref_g_mult(used, tau, L, rate, 0.15, k * api, 1.0, 0.10 if short else 0.20)[0]
            if used + k * api > 1 - 0.15 / 3:
                self.assertIsNotNone(why)
            else:
                self.assertAlmostEqual(g, want, places=9, msg=(L, used, tau, api, rate))

    def test_reference_eu_table(self):
        """decide_ref.py's relaxed and pressured scenarios through p_success, windows_of/quota_price, cost and utility."""
        want = {"relaxed": [0.460, -0.332, 0.084, 0.630, 1.045, 0.674], "pressured": [0.260, -0.436, -0.123, 0.216, 0.010, 0.674]}
        mu_b, sd_b, a, p_sc, d = 1.0, 0.8, 1.2, 0.3, 0.75
        tok, T = math.exp(math.log(1200) + 0.18), math.exp(math.log(8) + 0.125)
        V = 0.5 * min(8, max(0.5, math.sqrt(tok / 1000 * T)))
        seats = [("inline", 1.1, 0.0, (2, 10, 0.2), 0, (0.9, 80)), ("sub", 0.0, 0.15, (1, 5, 0.1), 18000, (0.6, 150)),
                 ("sub", 1.1, 0.15, (2, 10, 0.2), 18000, (0.9, 80)), ("sub", 2.2, 0.15, (4, 20, 0.4), 18000, (1.5, 50)),
                 ("sub", 3.3, 0.15, (10, 50, 1.0), 18000, (2.0, 40)), ("codex", 2.4, 0.5, (2.5, 15, 0.25), 10000, (1.5, 50))]
        for scen in ("relaxed", "pressured"):
            if scen == "relaxed":
                w = claude_windows(0.30, 3 * H, 0.25, 4 * 86400, hist5=((NOW - 1800, 0.30 - 0.05), (NOW, 0.30)))
            else:
                w = claude_windows(0.70, 2 * H, 0.55, 3 * 86400, hist5=((NOW - 1800, 0.70 - 0.1), (NOW, 0.70)))
            w["codex"] = {"as_of": NOW, "windows": [{"name": "primary", "used": 0.20, "reset_at": NOW + 3 * H, "length_s": 18000}]}
            got = []
            for kind, th, sd, pr, oh, (ttft, tps) in seats:
                inline = kind == "inline"
                P = D.p_success(th - (0 if inline else 0.5 * (1 - p_sc)), math.sqrt(sd ** 2 + (0 if inline else 0.09)), mu_b, sd_b, a, 0.5)
                tin, chit = ref_tokens_agentic(T, 1200 + (30000 if inline else 20000 * (1 - p_sc)), oh)
                tout = tok * T * 0.5
                api = D.cost(tin, chit, tout, pr)
                t = {"wcache": {}, "now": NOW, "windows": w, "reserve": 0.15, "k": {}}
                prov = "codex" if kind == "codex" else "claude"
                q, _, _ = D.quota_price({"quota": prov, "fam": ""}, api, api, 1.0, D.windows_of(prov, t), t)
                secs = T * (ttft + (0 if inline else 2)) + tout / tps
                F = d * (0.002 * (secs + 45)) + (1 - d) * 1.0 * V
                got.append(round(D.utility(V, P, 0.0, q, 0.002, secs, F), 3))
            self.assertEqual(got, want[scen])

    def test_fresh_window_is_free_and_a_pressured_one_is_priced(self):
        self.assertEqual(self.pacing(claude_windows(0.05, 4.5 * H))[0], 0.0)
        g, why, bind = self.pacing(claude_windows(0.70, 4 * H))
        self.assertGreater(g, 0.0)
        self.assertEqual((why, bind[2]), (None, "five_hour"))

    def test_hard_skip_on_exhaustion_and_quota_reserve(self):
        q, why, _ = self.pacing(claude_windows(0.95, 1 * H), api=1.0)
        self.assertEqual(why, "would exhaust the claude five_hour window")
        # a bigger reserve prices the same window earlier and skips it sooner
        w = claude_windows(0.50, 1 * H)                 # projected 0.625 + 0.02 at reset
        self.assertEqual(self.pacing(w, reserve=0.15)[0], 0.0)
        self.assertGreater(self.pacing(w, reserve=0.40)[0], 0.0)
        self.assertIsNone(self.pacing(claude_windows(0.85, 1 * H), api=0.1, reserve=0.15)[1])
        self.assertIsNotNone(self.pacing(claude_windows(0.85, 1 * H), api=0.1, reserve=0.5)[1])

    def test_family_windows_bind_only_their_family(self):
        w = claude_windows(0.1, 4 * H)
        w["claude"]["windows"].append({"name": "seven_day_opus", "used": 0.99, "reset_at": NOW + 86400, "length_s": 604800, "hist": []})
        self.assertIn("seven_day_opus", self.pacing(w, fam="opus")[1])
        self.assertIsNone(self.pacing(w, fam="sonnet")[1])

    def test_unknown_and_stale_windows(self):
        self.assertEqual(self.pacing({}, api=2.0), (0.25, None, None))
        self.assertEqual(self.pacing(claude_windows(0.0, 5 * H, as_of=NOW - 3600))[0], 0.25)   # older than 30 min
        self.assertEqual(self.pacing(claude_windows(0.9, 1 * H), prov="gemini")[0], 0.25)       # no source, ever

    def test_k_is_learned_from_window_samples_and_clipped(self):
        samples = [(0, 0.10), (300, 0.12), (600, 0.15), (900, 0.05)]   # the last interval has a reset: skipped
        tasks = [(100, 1.0), (450, 1.0), (700, 9.0)]
        a = 1 - 0.5 ** (1 / 20.0)
        k = (1 - a) * 0.02 + a * 0.02            # 0.02 used per $1
        k = (1 - a) * k + a * 0.03
        self.assertAlmostEqual(D.learn_k(samples, tasks, 0.02), k)
        self.assertAlmostEqual(D.learn_k([(0, 0.0), (1, 0.9)] * 1, [(0.5, 0.01)], 0.02, 0.07), 0.08)   # 4 × prior at most


class Decisions(Tmp):
    def test_cheaper_seat_with_equal_p_wins(self):
        a = S("openai/m-a", "strong", (4, 20), G=0.8)
        b = S("openai/m-b", "strong", (2, 10), G=0.8)
        opts = D.rank([a, b], post(), ctx())
        self.assertAlmostEqual(opts[0]["p"], opts[1]["p"])
        self.assertEqual(opts[0]["seat"], "openai/m-b")

    def test_higher_stakes_or_lower_detectability_picks_a_stronger_seat(self):
        seats = [S(*HAIKU), S(*OPUS)]

        def chosen(stakes, d):
            return D.decide(seats, post(b=[-2.5, 0.5], stakes=stakes, detectable=d), ctx(cfg={"wtp_usd": 0.4}))["chosen"]
        self.assertEqual(chosen(1.0, 1.0), HAIKU[0])
        self.assertEqual(chosen(1.0, 0.0), OPUS[0])     # lower detectability
        self.assertEqual(chosen(0.0, 0.0), HAIKU[0])
        self.assertEqual(chosen(2.0, 0.0), OPUS[0])     # higher stakes

    def test_floor_ranks_weak_seats_last_and_flags_below_floor(self):
        out = D.decide([S(*HAIKU)], post(b=[2.0, 0.5], stakes=2.0), ctx())
        self.assertEqual((out["chosen"], out["flags"]), (HAIKU[0], ["below_floor"]))
        self.assertIn("below_floor", out["reason"])
        opts = D.rank([S(*HAIKU), S("x/cheap-weak", "fast", (0.01, 0.01), G=0.3)], post(b=[0.5, 0.5], stakes=2.0), ctx())
        self.assertTrue(all(not o["floor_ok"] for o in opts))
        self.assertGreaterEqual(opts[0]["p_lo"], opts[1]["p_lo"])     # below the floor: the highest P̃ first

    def test_warm_seat_wins_on_a_long_prefix(self):
        a, b = S("anthropic/claude-sonnet-x", "standard", (3, 15)), S("anthropic/claude-sonnet-y", "standard", (3, 15))
        p = post(tok_out=[math.log(400), 0.5])
        cold = D.rank([a, b], p, ctx(est_in=120000, prefix_in=0))
        self.assertEqual(cold[0]["seat"], a["id"])                  # same price: the id breaks the tie
        warm = D.rank([a, b], p, ctx(est_in=120000, prefix_in=110000, warm=b["id"]))
        self.assertEqual(warm[0]["seat"], b["id"])
        self.assertLess(warm[0]["usd"] * 5, warm[1]["usd"])           # cache read 0.1 vs write 1.25

    def test_subscription_is_priced_against_the_cheapest_api_substitute(self):
        sub = router_seat("opus", "strong")
        api = S("anthropic/claude-opus-5-5", "strong", (4, 20))
        w = claude_windows(0.80, 4 * H)
        opts = {o["seat"]: o for o in D.rank([sub, api], post(), ctx(mode="skill", windows=w, est_in=500))}
        o = opts["claude-code:opus"]
        self.assertGreater(o["quota_usd"], 0.0)
        self.assertEqual(o["usd"], 0.0)
        sub_cost = D.cost(o["tin"], o["hit"], o["tout"], opts[api["id"]]["price"])
        self.assertAlmostEqual(o["quota_usd"], o["bind"][0] * sub_cost)
        self.assertEqual(o["bind"][0], 1.0)                          # S < 0 with an API substitute: g_max_sub
        relaxed = D.rank([sub, api], post(), ctx(mode="skill", windows=claude_windows(0.05, 4.5 * H), est_in=500))
        self.assertEqual(relaxed[0]["seat"], "claude-code:opus")       # slack: the subscription is free
        self.assertEqual(relaxed[0]["quota_usd"], 0.0)

    def test_stale_limits_flag_quota_unknown(self):
        out = D.decide([router_seat("opus", "strong")], post(), ctx(mode="skill", windows=claude_windows(0.1, 4 * H, as_of=NOW - 7200)))
        self.assertIn("quota_unknown", out["flags"])
        self.assertGreater(out["options"][0]["quota_usd"], 0.0)
        out = D.decide([router_seat("opus", "strong")], post(), ctx(mode="skill", windows=claude_windows(0.1, 4 * H)))
        self.assertNotIn("quota_unknown", out["flags"])
        self.assertEqual(out["options"][0]["quota_usd"], 0.0)

    def test_a_stale_catalog_link_widens_linked_seats_and_flags(self):
        seats = [S("openai/gpt-x", "strong", (2, 10), G=0.9), S(*OPUS)]
        hdr = {"link": {"catalog_sha": "abc"}}
        fresh = {o["seat"]: o for o in D.rank(seats, post(), ctx(s1=hdr, catalog_sha="abc"))}
        out = D.decide(seats, post(), ctx(s1=hdr, catalog_sha="def"))
        stale = {o["seat"]: o for o in D.rank(seats, post(), ctx(s1=hdr, catalog_sha="def"))}
        self.assertIn("link_stale", out["flags"])
        self.assertAlmostEqual(stale["openai/gpt-x"]["sd"] ** 2 - fresh["openai/gpt-x"]["sd"] ** 2, 0.04)
        self.assertEqual(stale[OPUS[0]]["sd"], fresh[OPUS[0]]["sd"])     # anchors are not linked
        self.assertNotIn("link_stale", D.decide(seats, post(), ctx(s1=hdr, catalog_sha="abc"))["flags"])

    def test_private_means_local_only(self):
        loc = D.seat_of({"name": "local:box/qwen", "seat": "local", "kind": "local", "model": "local:box/qwen", "serves": "standard",
                         "local": True, "tools": False, "params": 27})
        seats = [S(*OPUS), loc]
        out = D.decide(seats, post(), ctx(cfg={"private": True}))
        self.assertEqual(out["chosen"], "local:box/qwen")
        self.assertIn("private: local only", out["flags"])
        self.assertEqual(out["skipped"], [{"seat": OPUS[0], "reason": "private: local only"}])
        key = "here is my key sk-ant-" + "a" * 30
        self.assertEqual(D.decide(seats, post(), ctx(text=key))["chosen"], "local:box/qwen")
        self.assertEqual(D.decide(seats, post(), ctx(text="fix the tests in sk-learn"))["chosen"], OPUS[0])
        out = D.decide([S(*OPUS)], post(), ctx(text=key))                       # proxy: nothing qualifies
        self.assertEqual((out["chosen"], out["flags"]), (None, ["private_no_local"]))
        out = D.decide([router_seat("opus", "strong"), D.inline_seat("claude-code", "strong")], post(), ctx(mode="skill", text=key))
        self.assertEqual(out["chosen"], "inline")                              # skill mode: stay inline

    def test_secret_scan_matches_mine_redact_and_is_fast(self):
        sys.path.insert(0, os.path.join(helpers.ROOT, "evals"))
        import mine
        rng = random.Random(0)
        pieces = ["sk-", "sk-ant-", "sk-proj-", "ghp_", "github_pat_", "AKIA", "xoxb-", "eyJ", "-----BEGIN RSA PRIVATE KEY-----", "a", "_", " ",
                  "-", ".", "Z", "9", "\u00e9", "\n"]
        for _ in range(3000):
            s = "".join(rng.choice(pieces) + "".join(rng.choice("abcXYZ019_-.AKI") for _ in range(rng.randint(0, 25)))
                        for _ in range(rng.randint(1, 4)))
            self.assertEqual(D.is_private(s), any(rx.search(s) for rx in mine._REDACT), s)
        paste = "def task_runner(disk-usage): return x  # task-queue eyJ ok AKIA no ghost_ github\n" * 800    # 64 KB, many near misses
        t0 = time.perf_counter()
        for _ in range(10):
            self.assertFalse(D.is_private(paste))
        self.assertLess((time.perf_counter() - t0) / 10, 0.003)
        self.assertTrue(D.is_private(paste + "token=ghp_" + "x" * 30))

    def test_soft_tools_penalty_and_the_context_hint(self):
        chat_only = D.seat_of({"name": "local:box/q", "seat": "local", "kind": "local", "model": "local:box/q", "serves": "strong",
                               "local": True, "tools": False})
        o = D.rank([chat_only], post(tools=0.3), ctx(mode="skill"))[0]
        o2 = D.rank([chat_only], post(tools=0.1), ctx(mode="skill"))[0]
        self.assertAlmostEqual(o["p_lo"], o2["p_lo"] * 0.7)
        # the --context-tokens hint sizes the inline seat's context (the other hints are fused by s1.posterior)
        inline = D.inline_seat("claude-code", "strong")
        small = D.rank([inline], post(), ctx(mode="skill", est_in=1000))[0]["tin"]
        big = D.rank([inline], post(hints={"context_tokens": 90000}), ctx(mode="skill", est_in=1000))[0]["tin"]
        self.assertEqual((small, big), (1000 + 30000, 1000 + 90000))
        out = D.decide([inline], post(hints={"job": "debug"}), ctx(mode="skill"))
        self.assertEqual(out["hints"], {"job": "debug"})

    def test_rate_limit_and_context_hard_filters(self):
        tight = S("openai/small-ctx", "strong", (1, 5), context=10000)
        low = D.seat_of({"m": E("openai/low", "strong", (1, 5)), "headroom": 0.05, "rl_tokens": 100})
        ok = S("openai/ok", "strong", (1, 5))
        out = D.decide([tight, low, ok], post(), ctx(est_in=20000))
        self.assertEqual(out["chosen"], "openai/ok")
        self.assertEqual({s["seat"]: s["reason"] for s in out["skipped"]},
                         {"openai/small-ctx": "context too small", "openai/low": "rate limit: too few tokens left"})
        low = D.seat_of({"m": E("openai/low", "strong", (1, 5)), "headroom": 0.05})
        o = {o["seat"]: o for o in D.rank([low, ok], post(), ctx())}
        self.assertAlmostEqual(o["openai/low"]["quota_usd"], 0.2 * o["openai/low"]["t"]["V"])

    def test_effort_is_part_of_the_seat(self):
        codex = router_seat("gpt-6.1-sol", "standard", "codex", "cli")
        seats = D.with_efforts([codex, router_seat("opus", "strong")])
        self.assertEqual([s["id"] for s in seats], ["codex:gpt-6.1-sol@low", "codex:gpt-6.1-sol@medium", "codex:gpt-6.1-sol@high",
                                                     "claude-code:opus"])
        o = {x["seat"]: x for x in D.rank(seats, post(), ctx(mode="skill"))}
        self.assertAlmostEqual(o["codex:gpt-6.1-sol@high"]["th"] - o["codex:gpt-6.1-sol@medium"]["th"], 0.09)
        self.assertAlmostEqual(o["codex:gpt-6.1-sol@low"]["tout"] / o["codex:gpt-6.1-sol@medium"]["tout"], 0.6)

    def test_confidence_escalation_and_p_best(self):
        rng = random.Random(3)
        seats = [S("p/m%d" % i, rng.choice(("fast", "standard", "strong")), (rng.uniform(0.1, 10), rng.uniform(1, 40)),
                   G=rng.uniform(0.2, 0.95)) for i in range(12)]
        for _ in range(60):
            p = post(b=[rng.uniform(-2, 3), rng.uniform(0.2, 1.5)], stakes=rng.uniform(0, 2), detectable=rng.random())
            out = D.decide(seats, p, ctx())
            self.assertTrue(0.0 <= out["confidence"] <= 1.0)
            ranked = D.rank(seats, p, ctx())
            esc = D.escalate(ranked)
            if esc is None:     # no seat at least as able, and failures would go unnoticed
                self.assertTrue(p["detectable"] < 0.5 and all(o["th"] < ranked[0]["th"] for o in ranked[1:]))
            else:
                self.assertGreaterEqual(next(o for o in ranked if o["seat"] == esc)["th"], ranked[0]["th"])
        pb = D.p_best(ranked, "d7f3a2c1")
        self.assertAlmostEqual(sum(pb.values()), 1.0)
        self.assertEqual(pb, D.p_best(ranked, "d7f3a2c1"))
        self.assertEqual(D.confidence(ranked[:1]), 1.0)

    def test_decision_output_and_reason(self):
        sub, api = router_seat("opus", "strong"), S("anthropic/claude-haiku-4-5", "fast", (1, 5))
        p = post(job_p={"debug": 0.8, "code": 0.2}, b=[1.12, 0.71])
        out = D.decide([sub, api], p, ctx(mode="skill", windows=claude_windows(0.1, 4 * H), decision_id="d7f3a2c1", difficulty=2))
        self.assertEqual(set(out["options"][0]), {"seat", "p", "p_lo", "usd", "quota_usd", "secs", "eu"})
        self.assertEqual(out["decision_id"], "d7f3a2c1")
        self.assertTrue(out["reason"].startswith("debug d2 (b +1.1±0.7): claude-code:opus P "), out["reason"])
        self.assertIn("Claude 5h window has slack (quota ×0)", out["reason"])
        self.assertIn(out["escalate_to"], ("claude-code:opus", "anthropic/claude-haiku-4-5"))
        self.assertEqual(out["trace"]["quota"], "claude")
        self.assertEqual(D.decide([], p, ctx())["chosen"], None)

    def test_seat_shapes(self):
        cfg = core.load_config("/nonexistent")
        inv = inventory(clis=("codex", "gemini-cli", "opencode"), local_models=[local("gpt-oss:120b", 120, endpoint="spark")])
        cands = core._candidates(helpers.cls("ask", 1), inv, "claude-code", cfg)
        models = {"cli:codex/gpt-6.1-sol": E("cli:codex/gpt-6.1-sol", "standard", G=0.8, cost_in=1.25, cost_out=10.0)}
        seats = {s["id"]: s for s in (D.seat_of(c, models) for c in cands)}
        self.assertEqual(seats["claude-code:opus"]["access"], "subscription")
        self.assertEqual((seats["claude-code:opus"]["anchor"], seats["claude-code:opus"]["quota"]), ("opus", "claude"))
        self.assertEqual((seats["codex:gpt-6.1-sol"]["quota"], seats["codex:gpt-6.1-sol"]["entry"]["G"]), ("codex", 0.8))
        self.assertTrue(seats["codex:gpt-6.1-sol"]["eff_ok"])
        self.assertEqual(seats["opencode:anthropic/claude-opus-5-5"]["access"], "api")
        self.assertEqual((seats["local:spark/gpt-oss:120b"]["access"], seats["local:spark/gpt-oss:120b"]["tools"]), ("local", False))
        out = D.decide(list(seats.values()), post(), ctx(mode="skill"))
        self.assertTrue(out["chosen"])
        # the opencode API seat on the anchor model gets the anchor's θ
        o = {x["seat"]: x for x in D.rank(list(seats.values()), post(), ctx(mode="skill"))}
        self.assertEqual(o["opencode:anthropic/claude-opus-5-5"]["s"]["tier"], "opus")
        sub = D.seat_of({"m": E("cli:claude/sonnet", "standard")})
        self.assertEqual((sub["access"], sub["quota"], sub["anchor"], sub["kind"]), ("subscription", "claude", "sonnet", "cli"))
        bare = D.seat_of(E("local:spark/qwen3", "local", params_b=27, tps=40))
        self.assertEqual((bare["access"], bare["tps"], bare["params_b"]), ("local", 40, 27))

    def test_personal_offsets_and_user_anchors_are_read_from_home(self):
        seats = [router_seat("opus", "strong"), S("openai/gpt-x", "strong", (2, 10), G=0.7)]
        base = {o["seat"]: o["th"] for o in D.rank(seats, post(), ctx(mode="proxy", home=self.home))}
        with open(os.path.join(self.home, "personal.json"), "w") as f:
            json.dump({"seat_offsets": {"anthropic/opus|chat": 0.3, "openai/gpt-x|chat": -5}, "job_b_shift": {"ask": 9}}, f)
        with open(os.path.join(self.home, "anchors.json"), "w") as f:
            json.dump({"seats": {}}, f)
        o = D.rank(seats, post(), ctx(mode="proxy", home=self.home))
        th = {x["seat"]: x["th"] for x in o}
        self.assertAlmostEqual(th["claude-code:opus"] - base["claude-code:opus"], 0.3)
        self.assertAlmostEqual(th["openai/gpt-x"] - base["openai/gpt-x"], -1.0)      # |o| <= 1
        self.assertAlmostEqual(o[0]["t"]["mu"], -0.5)                               # |Δb| <= 0.5
        o = D.rank(seats, post(), ctx(mode="proxy", home=self.home, cfg={"personalize": False}))
        self.assertAlmostEqual({x["seat"]: x["th"] for x in o}["claude-code:opus"], base["claude-code:opus"])

    def test_malformed_files_and_posteriors_do_not_break_routing(self):
        seats = [router_seat("opus", "strong"), S("openai/gpt-x", "strong", (2, 10), G=0.7)]
        for name, body in (("personal.json", '[1, 2]'), ("personal.json", '{"seat_offsets": {"anthropic/opus|chat": "x"}, "k": 3}'),
                           ("anchors.json", '{"seats": {"openai/gpt-x": {"theta_chat": "high"}}}'), ("anchors.json", '"x"')):
            with open(os.path.join(self.home, name), "w") as f:
                f.write(body)
            self.assertTrue(D.decide(seats, post(), ctx(home=self.home))["chosen"])
        p = post(group="agentic", turns=[-2.0, 0.1], job_p={"code": 0.0, "nonsense": -1})    # under one turn: one turn
        o = D.rank([router_seat("opus", "strong")], p, ctx(mode="skill"))[0]
        self.assertEqual((o["t"]["turns"], o["hit"]), (1.0, 0.0))
        self.assertEqual(o["t"]["jp"], {"ask": 1.0})

    def test_rank_is_fast_on_40_seats(self):
        rng = random.Random(5)
        seats = [S("p%d/m%d" % (i % 5, i), rng.choice(("fast", "standard", "strong", "frontier")), (rng.uniform(0.1, 10), rng.uniform(1, 40)),
                   G=rng.uniform(0.2, 0.95), jobs={j: rng.uniform(0.2, 0.95) for j in core.JOBS}) for i in range(28)]
        seats += [router_seat(m, t) for m, t in zip(D.ANCHORS, D.TIERS)]
        seats += [router_seat("gpt-%d" % i, "standard", "codex", "cli") for i in range(4)]
        seats += [D.seat_of({"m": E("cli:claude/" + m, t)}) for m, t in zip(D.ANCHORS, D.TIERS)]
        self.assertEqual(len(seats), 40)
        p = post(job_p={"debug": 0.6, "code": 0.3, "ask": 0.1}, group="agentic", turns=[math.log(6), 0.5], tools=0.9)
        c = ctx(mode="skill", windows=claude_windows(0.4, 2 * H, 0.3, 3 * 86400, hist5=((NOW - 1800, 0.35), (NOW, 0.4))))
        for _ in range(50):
            D.decide(seats, p, c)
        p99s = []
        for _ in range(3):      # best of three rounds: a busy CI neighbour must not fail the build
            times = []
            for _ in range(200):
                t0 = time.perf_counter()
                D.decide(seats, p, c)           # rank plus escalation, confidence and the reason
                times.append(time.perf_counter() - t0)
            p99s.append(sorted(times)[197])
            if p99s[-1] <= 0.002:
                break
        self.assertLessEqual(min(p99s), 0.002, "p99 %s ms" % ["%.3f" % (x * 1000) for x in p99s])


class Files(Tmp):
    def write(self, name, obj):
        with open(os.path.join(self.home, name), "w", encoding="utf-8") as f:
            f.write(obj if isinstance(obj, str) else json.dumps(obj))

    def test_read_windows_parses_the_statusline_cache_log_and_codex_history(self):
        self.write("claude_limits.json", {"updated_at": NOW - 10, "rate_limits": {
            "five_hour": {"used_percentage": 42.0, "resets_at": NOW + 4200},
            "seven_day": {"used_percentage": 30, "resets_at": "2026-09-23T03:20:00Z"},
            "seven_day_opus": {"used_percentage": 50, "resets_at": "2026-09-23T05:20:00.5+02:00"}, "other": {"used_percentage": 1}}})
        self.write("claude_limits.log", "garbage\n" + "".join(json.dumps(r) + "\n" for r in (
            {"t": NOW - 1800, "five_hour": 32.0, "seven_day": 29}, {"t": NOW - 60, "five_hour": 42.0}, {"five_hour": 1})))
        self.write("quota_hist.json", {"version": 1, "windows": {"codex/primary": [[NOW - 600, 0.1], [NOW - 300, 0.15], ["x", 1]]},
                                       "current": {"codex/primary": {"used": 0.15, "reset_at": NOW + 600, "length_s": 18000, "t": NOW - 300}}})
        w = D.read_windows(self.home)
        cl = {x["name"]: x for x in w["claude"]["windows"]}
        self.assertEqual(sorted(cl), ["five_hour", "seven_day", "seven_day_opus"])
        self.assertEqual(w["claude"]["as_of"], NOW - 10)
        self.assertEqual((cl["five_hour"]["used"], cl["five_hour"]["reset_at"], cl["five_hour"]["length_s"]), (0.42, NOW + 4200, 18000))
        self.assertEqual(cl["five_hour"]["hist"], [[NOW - 1800, 0.32], [NOW - 60, 0.42]])
        self.assertEqual(cl["seven_day"]["reset_at"], 1790133600.0)
        self.assertEqual(cl["seven_day_opus"]["reset_at"], 1790133600.5)
        self.assertEqual(w["codex"]["windows"], [{"name": "primary", "used": 0.15, "reset_at": NOW + 600, "length_s": 18000,
                                                  "hist": [[NOW - 600, 0.1], [NOW - 300, 0.15]]}])
        self.assertIs(D.read_windows(self.home), w)                      # mtime cache
        self.write("claude_limits.json", {"rate_limits": {"five_hour": {"used_percentage": 99}}})
        self.assertEqual(D.read_windows(self.home)["claude"]["windows"][0]["used"], 0.99)
        self.assertEqual(D.read_windows(os.path.join(self.home, "none")), {})

    def test_route_side_reads_never_import_quota(self):
        code = ("import sys; sys.path.insert(0, %r); from maestro import decide; decide.read_windows(%r); "
                "decide.decide([], {'b': [0, 1]}, {'home': %r}); "
                "bad = [m for m in ('maestro.quota', 'maestro.providers', 'maestro.clients', 'maestro.server') if m in sys.modules]; "
                "assert not bad, bad; print('ok')") % (os.path.join(helpers.ROOT, "skills", "maestro", "scripts"), self.home, self.home)
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(res.stdout.strip(), "ok", res.stderr)

    def test_decide_imports_only_the_allowed_stdlib_modules(self):
        allowed = {"__future__", "array", "bisect", "hashlib", "json", "math", "mmap", "os", "random", "re", "statistics", "struct", "sys",
                   "time", "zlib"}
        tree = ast.parse(read(D.__file__))
        names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 0}
        self.assertLessEqual(names, allowed)
        self.assertFalse([n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level])   # no package-relative imports

    def test_log_decision_keeps_no_text_and_rotates(self):
        text = "refactor the secret billing module called zanzibar"
        out = D.decide([S(*HAIKU)], post(), ctx(decision_id="abcd1234"))
        self.assertTrue(D.log_decision(self.home, out, post(), text, harness="claude-code", hints={"job": "ask"}))
        path = os.path.join(self.home, "decisions.jsonl")
        with open(path) as f:
            raw = f.read()
        self.assertNotIn("zanzibar", raw)
        self.assertNotIn("billing", raw)
        row = json.loads(raw)
        self.assertEqual((row["id"], row["chosen"], row["harness"], len(row["minhash"])), ("abcd1234", HAIKU[0], "claude-code", 16))
        self.assertEqual(row["minhash"], D.minhash(text))
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
        self.assertFalse(D.log_decision(self.home, out, post(), text, cfg={"log_decisions": False}))
        with mock.patch.object(D, "LOG_MAX", 10):
            D.log_decision(self.home, out, post(), text)
        self.assertTrue(os.path.exists(path + ".1"))
        with open(path) as f:
            self.assertEqual(len(f.read().splitlines()), 1)

    def test_minhash_is_stable_across_processes_and_similar_texts_match(self):
        a = "fix the failing test in auth.py and update the changelog for the release"
        code = "import sys; sys.path.insert(0, %r); from maestro import decide; print(decide.minhash(%r))" % (
            os.path.join(helpers.ROOT, "skills", "maestro", "scripts"), a)
        outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                               env=dict(os.environ, PYTHONHASHSEED=seed)).stdout for seed in ("1", "2")}
        self.assertEqual(len(outs), 1)
        self.assertEqual(json.loads(outs.pop()), D.minhash(a))
        self.assertGreaterEqual(D.jaccard(D.minhash(a), D.minhash(a + " today")), 0.6)
        self.assertLess(D.jaccard(D.minhash(a), D.minhash("write a poem about the sea and the stars at night")), 0.3)

    def test_the_default_link_agrees_with_the_anchor_tiers(self):
        """Without a fitted header, a catalog model at a category threshold sits between the anchors of the tiers it separates."""
        from maestro import score
        prm = D.params()
        for g, lo, hi in ((score.STANDARD, "haiku", "sonnet"), (score.STRONG, "sonnet", "opus"), (score.FRONTIER, "opus", "fable")):
            th = D.theta({"id": "m", "entry": {"jobs": {"code": g}}}, {"code": 1.0}, prm)[0]
            self.assertTrue(prm["irt"]["theta"]["agentic"][lo] < th < prm["irt"]["theta"]["agentic"][hi], (g, th))

    def test_decide_json_shape(self):
        T = D.table()
        self.assertEqual(T["version"], 1)
        self.assertFalse(T["agentic_ready"])
        self.assertEqual(set(T["effort"]) - {"_source"}, set(core.EFFORTS))
        self.assertTrue(set(T["groups"]["agentic_jobs"]) <= set(core.JOBS))
        self.assertAlmostEqual(sum(T["gh7"]["w"]), 1.0, places=9)
        self.assertEqual(set(T["job_priors"]["tok_out"]), set(core.JOBS))
        self.assertEqual(set(D.params()), {"irt", "link", "calib", "ordinal", "decide"})
        self.assertEqual(D.params({"link": {"beta": 2.0}, "irt": None})["link"]["beta"], 2.0)


class QuotaSnapshots(Tmp):
    def rollout(self, tmp, primary, secondary, at):
        from test_quota import codex_event
        path = os.path.join(tmp, ".codex", "sessions", "2026/09/30", "rollout-x.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(codex_event(at, primary, secondary) + "\n")

    def test_refresh_appends_codex_history_and_window_view_reports_lengths(self):
        user = os.path.join(self.home, "user")
        self.rollout(user, {"used_percent": 20.0, "window_minutes": 300, "resets_at": NOW + 3600},
                     {"used_percent": 40.0, "window_minutes": 10080, "resets_at": NOW + 86400}, NOW - 5)
        Q.refresh_subscriptions(self.home, user, now=NOW)
        Q.refresh_subscriptions(self.home, user, now=NOW + 100)        # under 300 s: no new sample
        self.rollout(user, {"used_percent": 30.0, "window_minutes": 300, "resets_at": NOW + 3600},
                     {"used_percent": 41.0, "window_minutes": 10080, "resets_at": NOW + 86400}, NOW + 290)
        Q.refresh_subscriptions(self.home, user, now=NOW + 301)
        hist = core._read_json(os.path.join(self.home, Q.HIST_FILE))
        self.assertEqual(hist["windows"]["codex/primary"], [[NOW, 0.2], [NOW + 301, 0.3]])
        self.assertEqual(hist["current"]["codex/secondary"], {"used": 0.41, "reset_at": NOW + 86400, "length_s": 604800.0, "t": NOW + 301})
        self.assertEqual(Q.window_view("codex", home=self.home, now=NOW + 301), [
            {"name": "primary", "used": 0.3, "reset_at": NOW + 3600, "length_s": 18000.0},
            {"name": "secondary", "used": 0.41, "reset_at": NOW + 86400, "length_s": 604800.0}])
        w = D.read_windows(self.home)["codex"]
        self.assertEqual([x["name"] for x in w["windows"]], ["primary", "secondary"])
        self.assertEqual(w["as_of"], NOW + 301)
        with mock.patch.object(Q, "HIST_KEEP", 2):
            for i in range(4):
                Q.refresh_subscriptions(self.home, user, now=NOW + 1000 + 300 * i)
        self.assertEqual(len(core._read_json(os.path.join(self.home, Q.HIST_FILE))["windows"]["codex/primary"]), 2)
        with open(os.path.join(self.home, "claude_limits.json"), "w") as f:
            json.dump({"rate_limits": {"five_hour": {"used_percentage": 10, "resets_at": NOW + 600}}}, f)
        Q.refresh_subscriptions(self.home, os.path.join(self.home, "nobody"), now=NOW)
        self.assertEqual(Q.window_view("claude", "opus", home=self.home, now=NOW),
                         [{"name": "five_hour", "used": 0.1, "reset_at": NOW + 600, "length_s": 18000}])
        self.assertEqual(Q.window_view("gemini", home=self.home, now=NOW), [])

    def test_statusline_logs_changes_and_truncates(self):
        path, _ = Q.install_statusline(self.home)
        env = dict(os.environ, MAESTRO_HOME=self.home)

        def feed(five, seven=7):
            doc = {"rate_limits": {"five_hour": {"used_percentage": five, "resets_at": NOW + 100},
                                   "seven_day": {"used_percentage": seven, "resets_at": NOW + 200}}}
            res = subprocess.run([path], input=json.dumps(doc), capture_output=True, text=True, timeout=30, env=env)
            self.assertEqual(res.returncode, 0, res.stderr)
        log = os.path.join(self.home, "claude_limits.log")
        feed(10)
        feed(10)                     # unchanged: no line
        feed(12)
        rows = [json.loads(x) for x in read(log).splitlines()]
        self.assertEqual([(r["five_hour"], r["seven_day"]) for r in rows], [(10, 7), (12, 7)])
        self.assertTrue(all(abs(r["t"] - time.time()) < 60 for r in rows))
        with open(log, "w") as f:
            f.writelines('{"five_hour": 1, "seven_day": 1, "t": %d}\n' % (NOW + i) for i in range(4100))
        feed(13)
        lines = read(log).splitlines()
        self.assertEqual(len(lines), 2000)
        self.assertEqual(json.loads(lines[-1])["five_hour"], 13)
        hist = [h for x in D.read_windows(self.home)["claude"]["windows"] if x["name"] == "five_hour" for h in x["hist"]]
        self.assertEqual(hist[-1][1], 0.13)


if __name__ == "__main__":
    unittest.main()
