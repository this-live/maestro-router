"""The offline replay (S1 spec §9.1-§9.3, §12.6): a 24-item fixture with 4 tiers, the hull, IPW weights, statistics."""
import math
import os
import random
import sys
import unittest

import helpers  # noqa: F401

ROOT = helpers.ROOT
sys.path.insert(0, os.path.join(ROOT, "evals"))
import alpha_eval as ev  # noqa: E402
import alpha_stats as st  # noqa: E402

TIERS = ev.TIERS
THETA = {"haiku": 0.0, "sonnet": 1.1, "opus": 2.2, "fable": 3.3}
INV = ev.load_inventory()


def fixture(n=24, seed=3):
    """n items with b spread over [-2, 5]; y from the 1PL with seeded noise; tokens grow with tier."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        b = -2 + 7.0 * i / (n - 1)
        y = {t: 1.0 if rng.random() < st.sigmoid(2.5 * (THETA[t] - b)) else 0.0 for t in TIERS}
        rows.append({"id": "it%02d" % i, "cluster": "k%02d" % (i // 2), "split": "test", "source": "verifiable",
                     "job": "code", "difficulty": min(3, max(0, int((b + 2) / 2))), "lang": "en", "context": "chat",
                     "words": 30, "phase2": "full", "ipw": 1.0, "y": y, "tin": 600,
                     "tout": {"haiku": 400, "sonnet": 500, "opus": 600, "fable": 800}, "b_true": b})
    return rows


def eu_policy(items, bhat, v):
    return {it["id"]: ev.onehot(TIERS[ev.eu_choice({t: st.sigmoid(THETA[t] - bhat[it["id"]]) for t in TIERS}, it["cost"], v)])
            for it in items}


class Replay(unittest.TestCase):
    def setUp(self):
        self.rows = fixture()
        self.items = ev.load_items(self.rows, INV)

    def test_costs_use_recorded_tokens_and_catalog_prices(self):
        it = self.items[0]
        self.assertAlmostEqual(it["cost"]["haiku"], (600 * 1.0 + 400 * 5.0) / 1e6)
        self.assertAlmostEqual(it["cost"]["fable"], (600 * 10.0 + 800 * 50.0) / 1e6)
        self.assertAlmostEqual(it["secs"]["fable"], 2.0 + 2.0 + 800 / 40.0)

    def test_oracle_beats_router_beats_cheapest(self):
        rng = random.Random(0)
        bhat = {r["id"]: r["b_true"] + rng.gauss(0, 0.8) for r in self.rows}
        r = ev.summarize(self.items, eu_policy(self.items, bhat, 2.0))
        b1 = ev.summarize(self.items, ev.constant(self.items, "haiku"))
        o = ev.summarize(self.items, ev.oracle_real(self.items))
        self.assertEqual(r["n"], 24)
        self.assertGreaterEqual(o["S"], r["S"])
        self.assertGreaterEqual(r["S"], b1["S"])
        self.assertLessEqual(o["C"], ev.summarize(self.items, ev.constant(self.items, "fable"))["C"])
        self.assertEqual(o["under"], 0.0)
        self.assertEqual(o["over"], 0.0)

    def test_under_routing_and_over_spend_definitions(self):
        it = ev.load_items([dict(self.rows[0], y={"haiku": 0.0, "sonnet": 0.0, "opus": 1.0, "fable": 0.0})], INV)
        s = ev.summarize(it, ev.constant(it, "haiku"))
        self.assertEqual((s["under"], s["severe"]), (1.0, 1.0))
        s = ev.summarize(it, ev.constant(it, "sonnet"))
        self.assertEqual((s["under"], s["severe"]), (1.0, 0.0))
        s = ev.summarize(it, ev.constant(it, "fable"))
        self.assertAlmostEqual(s["over"], it[0]["cost"]["fable"] - it[0]["cost"]["opus"])

    def test_hull(self):
        pts = [(1, 0.5), (2, 0.55), (4, 0.8), (10, 0.85), (3, 0.4)]   # (2, .55) is under the chord, (3, .4) dominated
        front = st.frontier(pts)
        self.assertEqual(front, [(1.0, 0.5), (4.0, 0.8), (10.0, 0.85)])
        self.assertAlmostEqual(st.s_at(front, 2.5), 0.65)
        self.assertAlmostEqual(st.c_at(front, 0.65), 2.5)
        self.assertEqual(st.c_at(front, 0.3), 1.0)
        self.assertIsNone(st.c_at(front, 0.9))
        self.assertIsNone(st.s_at(front, 0.5))
        self.assertAlmostEqual(ev.alpha_dollar(0.8, 2.0, front)[0], 0.5)       # half the hull's cost at S = .8
        self.assertAlmostEqual(ev.alpha_dollar(0.9, 10.0, front)[1], 0.05)     # above the hull: alpha_S

    def test_constant_policies_have_zero_alpha_on_the_hull(self):
        for t in TIERS:
            a = ev.alpha_ci(self.items, ev.constant(self.items, t), n=200)
            if a["alpha"] is not None:
                self.assertLessEqual(a["alpha"], 1e-9)
        best = ev.alpha_ci(self.items, ev.oracle_real(self.items), n=200)
        self.assertGreater(best["alpha"] if best["alpha"] is not None else best["alpha_S"], 0)

    def test_ipw_weights(self):
        rows = []
        for i in range(6):   # 2 sampled agreeing items (weight 4) and 4 disagreeing items (weight 1)
            sampled = i < 2
            y = {"haiku": 1.0, "sonnet": 1.0, "opus": 1.0, "fable": 1.0} if sampled else {"haiku": 0.0, "sonnet": 0.0, "opus": 0.0, "fable": 1.0}
            rows.append(dict(fixture(6)[i], y=y, phase2="sampled" if sampled else "disagree", ipw=4.0 if sampled else 1.0))
        rows.append(dict(fixture(7)[6], y={"haiku": 1.0, "fable": 1.0}, phase2="unsampled", ipw=0.0))
        items = ev.load_items(rows, INV)
        s = ev.summarize(items, ev.constant(items, "haiku"))
        self.assertAlmostEqual(s["S"], 8.0 / 12.0)
        self.assertEqual(s["n"], 6)                           # the unsampled item has no sonnet/opus: excluded
        unsampled = items[-1]
        self.assertEqual(unsampled["w"], 0.0)
        self.assertEqual(unsampled["cw"], {"haiku": 1.0, "fable": 1.0})
        self.assertEqual(items[0]["cw"]["sonnet"], 4.0)
        self.assertEqual(items[0]["cw"]["haiku"], 1.0)

    def test_paired_bootstrap_and_logloss_delta(self):
        cur = ev.oracle_real(self.items)
        base = ev.constant(self.items, "haiku")
        pr = ev.paired(self.items, cur, base, n=300)
        pt, lo, hi = pr["dS"]
        self.assertGreater(pt, 0)
        self.assertLessEqual(lo, pt)
        self.assertGreaterEqual(hi, pt)
        good = {it["id"]: {t: st.sigmoid(2.5 * (THETA[t] - r["b_true"])) for t in TIERS} for it, r in zip(self.items, self.rows)}
        flat = {it["id"]: {t: 0.5 for t in TIERS} for it in self.items}
        pt, lo, hi = ev.logloss_delta(self.items, good, flat, n=300)
        self.assertLess(pt, 0)

    def test_route_policy_runs_the_real_router(self):
        texts = {"a": "fix the typo in README", "b": "design a lock-free MPMC queue with proofs of linearizability",
                 "c": "this test is flaky only on CI, 1 run in 5; find out why"}
        ch, ps, info = ev.route_policy(texts, INV, {})
        self.assertEqual(set(ch), set(texts))
        self.assertTrue(all(sum(v) == 1.0 for v in ch.values()))
        self.assertEqual(ch["a"], ev.onehot("haiku"))
        self.assertEqual(ch["b"], ev.onehot("fable"))
        self.assertEqual(info["decider"], "ladder")
        _, _, info = ev.route_policy(texts, INV, {"decider": "utility"})
        if "decider" not in ev.m.DEFAULT_CONFIG:
            self.assertFalse(info["available"])

    def test_calibrate_v_finds_the_smallest_grid_value(self):
        rng = random.Random(0)
        bhat = {r["id"]: r["b_true"] + rng.gauss(0, 0.5) for r in self.rows}
        res = ev.calibrate_v(self.items, lambda obj, v: eu_policy(self.items, bhat, v * 0.2))
        sfable = ev.summarize(self.items, ev.constant(self.items, "fable"))["S"]
        for obj, r in res.items():
            if r["V0"] is None:
                continue
            self.assertGreaterEqual(r["S"], ev.TARGETS[obj] * sfable - 1e-9)
            k = min(range(len(ev.V_GRID)), key=lambda j: abs(ev.V_GRID[j] - r["V0"]))
            if k:
                prev = ev.summarize(self.items, eu_policy(self.items, bhat, ev.V_GRID[k - 1] * 0.2))["S"]
                self.assertLess(prev, ev.TARGETS[obj] * sfable)
        self.assertLessEqual(res["cost"]["V0"] or 0, res["quality"]["V0"] or 1e9)

    def test_capture_is_between_hull_and_oracle(self):
        rng = random.Random(1)
        bhat = {r["id"]: r["b_true"] + rng.gauss(0, 0.6) for r in self.rows}
        theta, b = ev.b_fit(self.items, THETA)
        pts = ev.sweep_points(self.items, lambda it: {t: st.sigmoid(THETA[t] - bhat[it["id"]]) for t in TIERS})
        cap = ev.capture(self.items, pts, THETA, b)
        self.assertIsNotNone(cap["A"])
        self.assertLessEqual(cap["A"], 1.0 + 1e-9)
        self.assertIn("APGR_R", cap)
        self.assertIn("CPT95_hull", cap)


class Stats(unittest.TestCase):
    def test_fit_1pl_recovers_order(self):
        rng = random.Random(0)
        bs = {"i%d" % k: rng.uniform(-2, 4) for k in range(150)}
        cells = [(i, t, 1.0 if rng.random() < st.sigmoid(THETA[t] - b) else 0.0, 1.0) for i, b in bs.items() for t in TIERS for _ in range(3)]
        theta, b = st.fit_1pl(cells)
        self.assertEqual(theta["haiku"], 0.0)
        self.assertTrue(theta["haiku"] < theta["sonnet"] < theta["opus"] < theta["fable"])
        self.assertGreater(st.spearman([bs[i] for i in bs], [b[i] for i in bs]), 0.8)
        _, b2 = st.fit_1pl(cells, theta=THETA)                  # anchors fixed
        self.assertGreater(st.spearman([bs[i] for i in bs], [b2[i] for i in bs]), 0.8)

    def test_calibration_metrics(self):
        rng = random.Random(0)
        ps = [rng.random() for _ in range(3000)]
        ys = [1.0 if rng.random() < p else 0.0 for p in ps]
        c = st.calibration(ps, ys)
        self.assertLess(c["ece"], c["ece_null_p95"] + 0.02)
        self.assertAlmostEqual(c["slope"], 1.0, delta=0.15)
        self.assertAlmostEqual(c["intercept"], 0.0, delta=0.15)
        over = st.calibration([min(0.999, p * 1.6) for p in ps], ys, null=False)
        self.assertGreater(over["ece"], c["ece"])

    def test_ece_null_matches_the_spec_table(self):
        rng = random.Random(1)
        ps = [rng.random() for _ in range(600)]
        self.assertAlmostEqual(st.ece_null_p95(ps), 0.058, delta=0.02)

    def test_auc_spearman_mcnemar(self):
        self.assertEqual(st.auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]), 1.0)
        self.assertEqual(st.auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)
        self.assertAlmostEqual(st.spearman([1, 2, 3, 4], [10, 20, 30, 45]), 1.0)
        self.assertAlmostEqual(st.mcnemar(10, 10), 1.0)
        self.assertLess(st.mcnemar(30, 5), 0.001)
        self.assertLess(st.mcnemar(400, 300), 0.001)

    def test_cluster_bootstrap_is_seeded(self):
        vecs = [[1.0, float(k % 3)] for k in range(50)]
        a = st.cluster_bootstrap(vecs, lambda t: t[1] / t[0], n=200, seed=0)
        self.assertEqual(a, st.cluster_bootstrap(vecs, lambda t: t[1] / t[0], n=200, seed=0))
        self.assertLess(a[1], a[0])
        self.assertGreater(a[2], a[0])
        self.assertTrue(math.isclose(a[0], sum(k % 3 for k in range(50)) / 50.0))


if __name__ == "__main__":
    unittest.main()
