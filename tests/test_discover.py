"""Catalog snapshot, joins, scoring, categories, local fit, models.json, backtest and `discover` (no network)."""
import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import helpers
from helpers import ROOT, maestro as core

from maestro import catalog, discover, providers, score  # noqa: E402

BENCH = ("SWE-Bench Pro", "GPQA Diamond", "Terminal-Bench 2.0", "Humanity's Last Exam", "BrowseComp", "SciCode")


def read_json(path):
    with open(path) as f:
        return json.load(f)


def lab(name, date, bench=None, **kw):
    m = {"n": name, "d": date, "x": kw.pop("x", 200000), "r": kw.pop("r", 1), "t": kw.pop("t", 1)}
    m.update(kw)
    if bench is not None:
        m["b"] = [[b, v] for b, v in zip(BENCH, bench)]
    return m


def fixture():
    """A small catalog: six recent benchmarked models from three labs (so the cohort has 4+ reporters per benchmark), an old
    model, two open-weight models with Hugging Face repos and a provider table."""
    models = {
        "acme/titan": lab("Titan Pro", "2026-09-01", (80, 90, 85, 40, 70, 66), p=[5, 25]),
        "acme/atlas": lab("Atlas", "2026-08-01", (60, 70, 65, 25, 55, 50), p=[2, 10]),
        "acme/pebble-mini": lab("Pebble Mini", "2026-08-15", (20, 30, 25, 8, 15, 18), p=[0.1, 0.4]),
        "zeta/zed": lab("Zed", "2026-07-15", (50, 55, 45, 20, 40, 38), p=[1, 4]),
        "zeta/zed-flash": lab("Zed Flash", "2026-08-20", (35, 40, 30, 12, 25, 26), p=[0.3, 1.2]),
        "zeta/zed-lite": lab("Zed Lite", "2026-09-10", (15, 20, 10, 5, 10, 11), p=[0.1, 0.3]),
        "acme/relic-pro": lab("Relic Pro", "2024-01-15", p=[30, 120]),
        "openai/gpt-oss-120b": lab("GPT OSS 120B", "2026-04-01", (45, 50, 40, 15, 30, 32), ow=1, hf="openai/gpt-oss-120b", pt=116.83, pa=5.1,
                                   ps="table", p=[0.15, 0.6]),
        "alibaba/qwen3-coder-30b-a3b-instruct": lab("Qwen3 Coder 30B A3B", "2026-04-01", (40, 20, 35, 5, 10, 20), ow=1,
                                                    hf="Qwen/Qwen3-Coder-30B-A3B-Instruct", pt=30.0, pa=3.0, ps="name", p=[0.1, 0.4]),
        "meta/llama-3.1-8b-instruct": lab("Llama 3.1 8B", "2024-07-23", ow=1, hf="meta-llama/Llama-3.1-8B-Instruct", pt=8.0, ps="name", t=0, r=0),
        "moonshotai/kimi-big": lab("Kimi Big", "2026-07-16", (55, 60, 50, 20, 45, 41), ow=1, hf="moonshotai/Kimi-Big", pt=1000.0, ps="table"),
        "anthropic/claude-sonnet-4-5": lab("Claude Sonnet 4.5", "2025-09-29", x=1000000, p=[3, 15]),
    }
    prov = {"acme": {"titan": {"c": "acme/titan", "ci": 5, "co": 25}, "atlas": {"c": "acme/atlas", "ci": 2.5, "co": 12}},
            "router": {"acme/pebble-mini": {"c": "acme/pebble-mini", "ci": 0.1, "co": 0.4},
                       "mystery-model": {"n": "Mystery", "d": "2026-09-01", "r": 1, "t": 1, "x": 64000, "ci": 1, "co": 3}}}
    return catalog._finish({"v": 1, "as_of": "2026-09-30", "source": "test", "models": models, "prov": prov})


def api_provider(pid, models, style="openai"):
    return {"id": pid, "name": pid, "access": "api", "style": style, "status": "connected", "reason": "%d models" % len(models),
            "models": models, "key_source": "env"}


def local_provider(name, models, info=None, url="http://10.0.0.5:8000"):
    return {"id": "local:" + name, "name": "Local: " + name, "access": "local", "style": "local", "status": "connected",
            "reason": "openai at " + url, "models": models, "kind": "openai", "endpoint_url": url, "model_info": info or {},
            "base_url": url + "/v1"}


def cli_provider(cid, models):
    return {"id": cid, "name": cid, "access": "subscription-cli", "style": "cli", "status": "connected", "reason": "logged in",
            "models": models, "seat": cid}


def providers_doc(*entries):
    return {"version": 1, "generated_at": 1.0, "verified": True, "providers": list(entries), "warnings": []}


class Names(unittest.TestCase):
    def test_parse_params(self):
        p = catalog.parse_params
        self.assertEqual(p("gpt-oss-120b"), (120.0, None))
        self.assertEqual(p("Qwen3-Coder-30B-A3B-Instruct"), (30.0, 3.0))
        self.assertEqual(p("mixtral-8x7b"), (56.0, None))
        self.assertEqual(p("qwen3.8-2.4t-a95b"), (2400.0, 95.0))
        self.assertEqual(p("gemma-3-270m"), (0.27, None))
        self.assertEqual(p("llama-4-scout-17b-16e"), (None, 17.0))
        self.assertEqual(p("gemma-3n-e4b"), (None, None))
        self.assertEqual(p("gpt-5.4"), (None, None))

    def test_normalization_strips_dates_prefixes_and_quantization(self):
        self.assertEqual(catalog.norm_id("anthropic.claude-sonnet-4-5-20250929-v1:0"), catalog.norm_id("claude-sonnet-4-5"))
        self.assertEqual(catalog.norm_id("us.anthropic.claude-opus-5"), "claudeopus5")
        self.assertEqual(catalog.norm_id("gemini-flash-latest"), "geminiflash")
        self.assertEqual(catalog.strip_quant("RedHatAI/Qwen3-Coder-30B-A3B-Instruct-FP8-dynamic"), "redhatai/qwen3-coder-30b-a3b-instruct")
        self.assertEqual(catalog.strip_quant("org/Model-8B-AWQ-GGUF"), "org/model-8b")
        self.assertEqual(catalog.strip_quant("org/Model-Instruct"), "org/model-instruct")


class Snapshot(unittest.TestCase):
    def test_the_bundled_snapshot_loads_and_is_small(self):
        cat = catalog.load_snapshot()
        self.assertGreater(len(cat["models"]), 300)
        self.assertGreater(len(cat["prov"]), 100)
        self.assertRegex(cat["as_of"], r"^\d{4}-\d\d-\d\d$")
        self.assertLess(os.path.getsize(catalog.DATA_FILE), 300 * 1024)
        oss = cat["models"]["openai/gpt-oss-120b"]
        self.assertTrue(oss["ow"] and oss["hf"] == "openai/gpt-oss-120b" and oss["pt"] > 100)
        self.assertTrue(any(m.get("b") for m in cat["models"].values()))
        self.assertFalse(any(not catalog.text_only(m.get("om")) for m in cat["models"].values()))

    def test_every_open_weight_model_states_where_its_size_came_from(self):
        cat = catalog.load_snapshot()
        for lid, m in cat["models"].items():
            if m.get("ow") and m.get("pt"):
                self.assertIn(m["ps"], ("hf-api", "table", "name"), lid)

    def test_extra_benchmarks_are_sourced_and_reach_the_snapshot(self):
        with open(os.path.join(ROOT, "evals", "extra_benchmarks.json"), encoding="utf-8") as f:
            extra = json.load(f)
        ids = [k for k in extra if not k.startswith("_")]
        self.assertGreaterEqual(len(ids), 40)
        cat = catalog.load_snapshot()
        for mid in ids:
            self.assertIn(mid, cat["models"], mid)
            have = {(b[0].lower(), b[1]) for b in cat["models"][mid]["b"]}
            for r in extra[mid]:
                self.assertRegex(r["source"], r"^https://(huggingface\.co|arxiv\.org)/", (mid, r["name"]))
                self.assertIsInstance(r["score"], (int, float))
                self.assertIn((r["name"].lower(), r["score"]), have, (mid, r["name"]))
        oss = cat["models"]["openai/gpt-oss-120b"]["b"]
        self.assertIn(["SWE-Bench Verified", 62.4, "accuracy"], oss)      # the gpt-oss model card (arXiv 2508.10925)
        ctx = score.make_context(catalog._finish(cat))
        self.assertEqual(score.score_lab(ctx, catalog._finish(cat), "openai/gpt-oss-120b")["basis"], "bench")

    def test_snapshot_scores_the_frontier_above_the_small_models(self):
        cat = catalog._finish(catalog.load_snapshot())
        ctx = score.make_context(cat)
        top = max(cat["models"], key=lambda k: score.score_lab(ctx, cat, k)["G"])
        self.assertEqual(score.score_lab(ctx, cat, top)["category"], "frontier")
        self.assertLess(score.score_lab(ctx, cat, "openai/gpt-oss-20b")["G"], score.score_lab(ctx, cat, top)["G"])


class Join(unittest.TestCase):
    def setUp(self):
        self.cat = fixture()

    def test_provider_entry_canonical_id(self):
        self.assertEqual(catalog.resolve(self.cat, "acme", "titan")[:2], ("acme/titan", "canonical"))

    def test_exact_prefixed_and_normalized(self):
        r = catalog.resolve
        self.assertEqual(r(self.cat, "any", "acme/atlas")[:2], ("acme/atlas", "exact"))
        self.assertEqual(r(self.cat, "zeta", "zed")[:2], ("zeta/zed", "prefixed"))
        self.assertEqual(r(self.cat, "bedrock", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")[:2], ("anthropic/claude-sonnet-4-5", "normalized"))
        self.assertEqual(r(self.cat, "acme", "nothing-like-it")[:2], (None, "none"))

    def test_dated_snapshots_join_the_benchmarked_base_model(self):
        cat = copy.deepcopy(self.cat)
        cat["models"]["anthropic/claude-haiku-4-5"] = lab("Claude Haiku 4.5", "2025-10-15", (45, 60, 40, 12, 30, 35), p=[1, 5])
        cat["models"]["anthropic/claude-haiku-4-5-20251001"] = lab("Claude Haiku 4.5 (20251001)", "2025-10-01", p=[1, 5])
        cat["models"]["acme/pebble-mini-2026-08-01"] = lab("Pebble Mini (dated)", "2026-08-01", p=[0.1, 0.4])
        cat["prov"]["anthropic"] = {"claude-haiku-4-5-20251001": {"c": "anthropic/claude-haiku-4-5-20251001", "ci": 1, "co": 5}}
        cat = catalog._finish(cat)
        lid, how, entry = catalog.resolve(cat, "anthropic", "claude-haiku-4-5-20251001")   # canonical, but with no benchmarks of its own
        self.assertEqual((lid, how), ("anthropic/claude-haiku-4-5", "dated"))
        self.assertTrue(catalog.rec_of(cat, lid, entry, "claude-haiku-4-5-20251001")["bench"])
        self.assertEqual(catalog.resolve(cat, "any", "anthropic/claude-haiku-4-5-20251001")[:2], ("anthropic/claude-haiku-4-5", "dated"))
        self.assertEqual(catalog.resolve(cat, "any", "acme/pebble-mini-2026-08-01")[:2], ("acme/pebble-mini", "dated"))   # YYYY-MM-DD form
        self.assertEqual(catalog.resolve(cat, "anthropic", "claude-haiku-4-5")[:2], ("anthropic/claude-haiku-4-5", "prefixed"))
        # a dated model with benchmarks of its own, or with no benchmarked base, stays as it is
        cat["models"]["acme/atlas-20260801"] = lab("Atlas 0801", "2026-08-01", (60, 70, 65, 25, 55, 50))
        cat["models"]["acme/lonely-20260801"] = lab("Lonely", "2026-08-01")
        cat = catalog._finish(cat)
        self.assertEqual(catalog.resolve(cat, "any", "acme/atlas-20260801")[:2], ("acme/atlas-20260801", "exact"))
        self.assertEqual(catalog.resolve(cat, "any", "acme/lonely-20260801")[:2], ("acme/lonely-20260801", "exact"))

    def test_standalone_provider_entry_keeps_its_own_attributes(self):
        lid, how, entry = catalog.resolve(self.cat, "router", "mystery-model")
        self.assertIsNone(lid)
        rec = catalog.rec_of(self.cat, lid, entry, "mystery-model")
        self.assertEqual((rec["ctx"], rec["price_out"], rec["reasoning"]), (64000, 3, True))

    def test_hugging_face_repo_with_and_without_quant_suffix(self):
        r = catalog.resolve_local
        self.assertEqual(r(self.cat, "openai/gpt-oss-120b"), ("openai/gpt-oss-120b", "hf-repo"))
        self.assertEqual(r(self.cat, "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"), ("alibaba/qwen3-coder-30b-a3b-instruct", "hf-repo"))
        self.assertEqual(r(self.cat, "RedHatAI/Qwen3-Coder-30B-A3B-Instruct-AWQ")[1], "hf-name")   # another org, same repo name
        self.assertEqual(r(self.cat, "served-name", "/models/meta-llama/Llama-3.1-8B-Instruct-NVFP4")[0], "meta/llama-3.1-8b-instruct")
        self.assertEqual(r(self.cat, "gpt-oss-120b"), ("openai/gpt-oss-120b", "hf-name"))          # a served-model-name that is the repo name

    def test_ollama_tags(self):
        r = catalog.resolve_local
        self.assertEqual(r(self.cat, "gpt-oss:120b"), ("openai/gpt-oss-120b", "ollama"))
        self.assertEqual(r(self.cat, "qwen3-coder:30b"), ("alibaba/qwen3-coder-30b-a3b-instruct", "ollama"))
        self.assertEqual(r(self.cat, "llama3.1:8b-instruct-q4_K_M"), ("meta/llama-3.1-8b-instruct", "ollama"))
        self.assertEqual(r(self.cat, "llama3.1:latest", params_b=8.0), ("meta/llama-3.1-8b-instruct", "ollama"))
        self.assertEqual(r(self.cat, "gpt-oss:70b"), (None, "none"))
        self.assertEqual(r(self.cat, "mystery:7b"), (None, "none"))

    def test_aliases_pick_the_newest_matching_model(self):
        cat = catalog._finish({"v": 1, "as_of": "2026-09-30", "prov": {}, "models": {
            "anthropic/claude-sonnet-4-5": lab("a", "2025-09-29"), "anthropic/claude-sonnet-5": lab("b", "2026-09-01"),
            "anthropic/claude-sonnet-5-5": lab("c", "2026-09-28"), "anthropic/claude-haiku-4-5": lab("d", "2025-10-01"),
            "google/gemini-3.8-flash": lab("e", "2026-09-02"), "google/gemini-3.5-flash-lite": lab("f", "2026-07-01"),
            "google/gemini-3.9-flash-lite-preview": lab("g", "2026-09-10")}})
        self.assertEqual(catalog.resolve_alias(cat, "sonnet"), "anthropic/claude-sonnet-5-5")
        self.assertEqual(catalog.resolve_alias(cat, "haiku"), "anthropic/claude-haiku-4-5")
        self.assertEqual(catalog.resolve_alias(cat, "gemini-flash-latest"), "google/gemini-3.8-flash")
        self.assertEqual(catalog.resolve_alias(cat, "gemini-flash-lite-latest"), "google/gemini-3.9-flash-lite-preview")
        self.assertIsNone(catalog.resolve_alias(cat, "opus"))


class Scoring(unittest.TestCase):
    def setUp(self):
        self.cat = fixture()
        self.ctx = score.make_context(self.cat)

    def sc(self, lid):
        return score.score_lab(self.ctx, self.cat, lid)

    def test_deterministic_and_independent_of_order(self):
        a = {k: self.sc(k) for k in self.cat["models"]}
        cat2 = catalog._finish(dict(self.cat, models=dict(reversed(list(self.cat["models"].items())))))
        ctx2 = score.make_context(cat2)
        b = {k: score.score_lab(ctx2, cat2, k) for k in cat2["models"]}
        self.assertEqual(a, b)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps({k: self.sc(k) for k in self.cat["models"]}, sort_keys=True))

    def test_ordering_follows_the_benchmarks(self):
        g = lambda k: self.sc(k)["G"]  # noqa: E731
        self.assertGreater(g("acme/titan"), g("acme/atlas"))
        self.assertGreater(g("acme/atlas"), g("zeta/zed"))
        self.assertGreater(g("zeta/zed"), g("zeta/zed-flash"))
        self.assertGreater(g("zeta/zed-flash"), g("zeta/zed-lite"))
        self.assertEqual(self.sc("acme/titan")["basis"], "bench")
        self.assertEqual(self.sc("acme/relic-pro")["basis"], "prior")
        self.assertEqual(set(self.sc("acme/titan")["jobs"]), set(score.JOBS))
        self.assertTrue(all(0 <= v <= 1 for v in self.sc("acme/titan")["jobs"].values()))

    def test_category_thresholds(self):
        t = score.tier_of_g
        self.assertEqual([t(x) for x in (0.68, 0.679, 0.52, 0.519, 0.35, 0.349)], ["frontier", "strong", "strong", "standard", "standard", "fast"])
        self.assertEqual(t(0.40, weak=True), "fast")      # a weak-tier name needs .42 for standard
        self.assertEqual(t(0.42, weak=True), "standard")
        self.assertEqual(self.sc("acme/titan")["category"], "frontier")
        self.assertIn(self.sc("zeta/zed-lite")["category"], ("fast", "standard"))

    def test_old_models_have_a_capped_prior(self):
        rec = catalog.rec_of(self.cat, "acme/relic-pro")     # $120/M output, 32 months old
        self.assertLessEqual(score.prior(self.ctx, rec), score.OLD_PRIOR_CAP)
        young = dict(rec, release="2026-09-01")
        self.assertGreater(score.prior(self.ctx, young), score.OLD_PRIOR_CAP)

    def test_a_lone_benchmark_is_damped(self):
        rec = catalog.rec_of(self.cat, "zeta/zed-lite")
        rec = dict(rec, id="x/lone", lab="zeta", bench=[["SWE-Bench Pro", 99]])
        res = score.score_rec(self.ctx, rec)
        self.assertLess(res["G"], 0.8 * 1.0)                 # far below the 99th-percentile it would get undamped
        self.assertLess(res["G"] - res["prior"], 0.3)
        self.assertEqual(res["basis"], "blend")

    def test_benchmarks_of_unknown_labs_count_half(self):
        base = catalog.rec_of(self.cat, "acme/atlas")
        trusted = score.score_rec(self.ctx, base)
        unknown = score.score_rec(self.ctx, dict(base, lab="nobody"))
        self.assertEqual(trusted["n"], 6.0)
        self.assertEqual(unknown["n"], 3.0)

    def test_coder_names_are_specialists(self):
        rec = dict(catalog.rec_of(self.cat, "acme/atlas"), id="acme/atlas-coder")
        res = score.score_rec(self.ctx, rec)
        self.assertEqual((res["category"], res["serves"]), ("specialist", score.tier_of_g(res["G"])))
        self.assertEqual(self.sc("alibaba/qwen3-coder-30b-a3b-instruct")["category"], "specialist")
        self.assertNotEqual(self.sc("acme/atlas")["category"], "specialist")

    def test_a_job_far_above_g_makes_a_specialist(self):
        # strong on four coding benchmarks, at the bottom on two reasoning ones, against a large cohort
        rec = dict(catalog.rec_of(self.cat, "acme/atlas"), id="acme/atlas-x", bench=[
            ["SWE-Bench Pro", 88], ["SWE-bench Verified", 88], ["LiveCodeBench", 88], ["SciCode", 88],
            ["GPQA Diamond", 1], ["Humanity's Last Exam", 1]])
        cohort = [i * 3.0 for i in range(1, 31)]
        ctx = dict(self.ctx, bvals={k: cohort for k in ("swebenchpro", "swebenchverified", "livecodebench", "scicode", "gpqadiamond", "hle")})
        res = score.score_rec(ctx, rec)
        self.assertGreaterEqual(max(res["jobs"].values()) - res["G"], score.LIFT)
        self.assertEqual(res["category"], "specialist")
        self.assertEqual(res["serves"], score.tier_of_g(res["G"]))

    def test_local_serves_is_capped_at_strong(self):
        self.assertEqual(score.local_serves("frontier"), "strong")
        self.assertEqual(score.local_serves("standard"), "standard")


class LocalFit(unittest.TestCase):
    def test_memory_math(self):
        self.assertEqual(discover.mem_gb(120, "FP8"), 138.0)       # 120 * 1.0 * 1.15
        self.assertEqual(discover.mem_gb(120, "BF16"), 276.0)
        self.assertEqual(discover.mem_gb(100, "NVFP4"), 64.7)      # 100 * 0.5625 * 1.15
        self.assertEqual(discover.mem_gb(8, "Q4"), 5.5)
        self.assertEqual(discover.kv_gb(8), 1.7)
        self.assertEqual(discover.kv_gb(1000), 11.8)               # capped at 120B

    def test_fits_take_the_best_precision_that_fits_and_rank_by_fit_score(self):
        cat = fixture()
        ctx = score.make_context(cat)
        hw = {"spark": {"name": "DGX Spark", "memory_gb": 128, "kind": "blackwell", "assumed": True},
              "small": {"name": "GPU", "memory_gb": 16, "kind": "nvidia"}, "mac": {"name": "Apple Silicon", "memory_gb": 36, "kind": "apple"}}
        fits = discover.local_fit(cat, ctx, hw)
        spark = {r["model"]: r for r in fits["spark"]}
        self.assertEqual(spark["openai/gpt-oss-120b"]["quant"], "MXFP4")        # released in 4-bit: that is what to serve
        self.assertEqual(spark["alibaba/qwen3-coder-30b-a3b-instruct"]["quant"], "BF16")   # speed unknown: highest precision
        self.assertNotIn("moonshotai/kimi-big", spark)                          # 1T parameters: nothing fits
        g = [r["fit_score"] for r in fits["spark"]]
        self.assertEqual(g, sorted(g, reverse=True))
        small = {r["model"]: r for r in fits["small"]}
        self.assertEqual(small["meta/llama-3.1-8b-instruct"]["quant"], "FP8")   # BF16 needs 20+ GB
        self.assertNotIn("openai/gpt-oss-120b", small)
        self.assertEqual({r["quant"] for r in fits["mac"]}, {"Q4"})
        row = spark["openai/gpt-oss-120b"]
        self.assertIn("vllm serve", row["serve_cmd"])
        self.assertEqual(set(row["jobs"]), set(score.JOBS))
        self.assertTrue(all(r["mem_gb"] + r["kv_gb"] <= 128 * discover.USABLE for r in fits["spark"]))
        self.assertTrue(all(r["tps"] is None and r["tps_est"] for r in fits["spark"]))   # no bandwidth given

    def test_decode_speed_estimate_and_ranking_penalty(self):
        self.assertAlmostEqual(discover.decode_tps(273, 5.1, "MXFP4"), 273 * 0.6 / (5.1 * 0.5625))       # about 57 tok/s
        self.assertAlmostEqual(discover.decode_tps(273, 128, "NVFP4"), 273 * 0.6 / (128 * 0.5625))        # about 2.3 tok/s: useless
        self.assertIsNone(discover.decode_tps(None, 5, "FP8"))
        f = discover.speed_factor
        self.assertEqual([f(x) for x in (None, 9.9, 10, 24.9, 25, 500)], [1.0, 0.3, 0.7, 0.7, 1.0, 1.0])

    def test_fast_moe_beats_a_slightly_better_dense_model_on_a_bandwidth_bound_box(self):
        cat = fixture()
        cat["models"]["acme/dense-128b"] = lab("Dense 128B", "2026-09-01", (70, 80, 75, 30, 60, 60), ow=1, hf="acme/Dense-128B", pt=128.0, ps="name")
        ctx = score.make_context(cat)
        spark = {"name": "DGX Spark", "memory_gb": 128, "kind": "blackwell", "bandwidth_gbps": 273}
        rows = {r["model"]: r for r in discover.local_fit(cat, ctx, {"spark": spark})["spark"]}
        dense, moe = rows["acme/dense-128b"], rows["openai/gpt-oss-120b"]
        self.assertGreater(dense["G"], moe["G"])
        self.assertLess(dense["tps"], 10)
        self.assertFalse(dense["active_known"])
        self.assertEqual(dense["active_params_b"], 128.0)
        self.assertEqual((moe["active_params_b"], moe["active_known"]), (5.1, True))
        self.assertGreater(moe["tps"], 25)
        self.assertGreater(moe["fit_score"], dense["fit_score"])
        order = [r["model"] for r in discover.local_fit(cat, ctx, {"spark": spark})["spark"]]
        self.assertLess(order.index("openai/gpt-oss-120b"), order.index("acme/dense-128b"))
        slow = discover.local_fit(cat, ctx, {"slow": dict(spark, bandwidth_gbps=100)})["slow"]
        self.assertLess(next(r["tps"] for r in slow if r["model"] == "openai/gpt-oss-120b"), 25)   # a slower box slows everyone down

    def test_speed_can_pick_a_smaller_precision(self):
        cat = catalog._finish({"v": 1, "as_of": "2026-09-30", "prov": {}, "models": dict(fixture()["models"])})
        ctx = score.make_context(cat)
        box = {"name": "GPU", "memory_gb": 128, "kind": "blackwell", "bandwidth_gbps": 273}
        row = next(r for r in discover.local_fit(cat, ctx, {"b": box})["b"] if r["model"] == "alibaba/qwen3-coder-30b-a3b-instruct")
        self.assertEqual(row["quant"], "BF16")      # 3B active * 2 bytes: ~27 tok/s, already fast enough
        big = dict(cat["models"]["alibaba/qwen3-coder-30b-a3b-instruct"], pa=12.0)
        cat["models"]["alibaba/qwen3-coder-30b-a3b-instruct"] = big
        row = next(r for r in discover.local_fit(cat, ctx, {"b": box})["b"] if r["model"] == "alibaba/qwen3-coder-30b-a3b-instruct")
        self.assertIn(row["quant"], ("FP8", "NVFP4"))   # 12B active in BF16 would be ~7 tok/s; a 4 or 8-bit build is faster

    def test_serve_commands(self):
        self.assertEqual(discover.serve_cmd("openai/gpt-oss-120b", "Q4", "openai/gpt-oss-120b"), "ollama run gpt-oss:120b")
        self.assertIn("--quantization fp8", discover.serve_cmd("a/b", "FP8", "a/b"))
        self.assertTrue(discover.serve_cmd("a/b", "Q4", "a/b").startswith("llama-server -hf a/b-GGUF"))
        self.assertTrue(discover.serve_cmd("nvidia/X-120B-BF16", "NVFP4", "x").startswith("vllm serve nvidia/X-120B-NVFP4 "))
        self.assertIsNone(discover.ollama_tag("nobody/nothing"))

    def test_serving_now_is_compared_with_what_fits(self):
        cat = fixture()
        ctx = score.make_context(cat)
        hw = {"spark": {"name": "DGX Spark", "memory_gb": 128, "kind": "blackwell", "bandwidth_gbps": 273}}
        ranked = discover.local_fit(cat, ctx, hw)
        models = [{"id": "local:spark/x", "model": "x", "access": "local", "box": "spark", "catalog_id": None, "G": 0.0, "serves": "fast",
                   "basis": "prior", "jobs": {j: 0.0 for j in score.JOBS}},
                  {"id": "local:spark/oss", "model": "oss", "access": "local", "box": "spark", "catalog_id": "openai/gpt-oss-120b", "G": 0.3,
                   "serves": "fast", "basis": "bench", "jobs": {j: 0.3 for j in score.JOBS}}]
        s = discover.serving_vs_fit(models, ranked, hw)["spark"]
        self.assertEqual((s[0]["would_rank"], s[0]["of"]), (len(ranked["spark"]) + 1, len(ranked["spark"])))
        oss = next(r for r in ranked["spark"] if r["model"] == "openai/gpt-oss-120b")
        self.assertEqual((s[1]["tps"], s[1]["fit_score"]), (oss["tps"], oss["fit_score"]))

    def test_hardware_detection_and_assumptions(self):
        d = discover.describe_hardware
        self.assertEqual(d({"vram_gb": None, "ram_gb": 121.7}, "NVIDIA GB10")["name"], "DGX Spark")
        self.assertEqual(d({"vram_gb": None, "ram_gb": 121.7}, "NVIDIA GB10")["memory_gb"], 128)
        self.assertEqual(d({"apple_silicon": True, "ram_gb": 64.0}), {"name": "Apple Silicon", "memory_gb": 64.0, "kind": "apple", "bandwidth_gbps": None})
        self.assertEqual(d({"apple_silicon": True, "ram_gb": 64.0}, "", "Apple M4 Max")["bandwidth_gbps"], 546)
        self.assertEqual(d({"apple_silicon": True, "ram_gb": 192.0}, "", "Apple M3 Ultra")["bandwidth_gbps"], 819)
        self.assertEqual(d({"vram_gb": 24.0, "ram_gb": 64.0}, "NVIDIA GeForce RTX 4090")["bandwidth_gbps"], 1008)
        self.assertEqual(d({"vram_gb": None, "ram_gb": 121.7}, "NVIDIA GB10")["bandwidth_gbps"], 273)
        self.assertIsNone(d({"vram_gb": 8.0, "ram_gb": 32.0}, "Some Future GPU")["bandwidth_gbps"])
        self.assertEqual([discover.bandwidth_of(n) for n in ("NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-80GB", "NVIDIA GeForce RTX 5090", "Mac Studio")],
                         [3350, 2039, 1792, None])
        self.assertEqual(discover.bandwidth_of("Mac Studio", "Apple M2 Ultra"), 800)
        self.assertEqual(d({"vram_gb": 24.0, "ram_gb": 64.0}, "NVIDIA GeForce RTX 4090")["kind"], "nvidia")
        self.assertEqual(d({"ram_gb": 16.0})["kind"], "cpu")
        cfg = {"local_endpoints": [{"name": "spark", "url": "http://10.0.0.5:8000"},
                                   {"name": "studio", "url": "http://10.0.0.6:11434", "hardware": {"name": "Mac Studio", "memory_gb": 192}},
                                   {"name": "gpu", "url": "http://10.0.0.7:8000", "hardware": {"name": "RTX 4090", "memory_gb": 24}},
                                   {"name": "gpu5", "url": "http://10.0.0.9:8000", "hardware": {"name": "RTX 5090", "memory_gb": 32}},
                                   {"name": "plain", "url": "http://10.0.0.8:8000"}, "http://spark2:8000",
                                   {"name": "custom", "url": "http://10.0.0.10:8000", "hardware": {"name": "Rack", "memory_gb": 96, "bandwidth_gbps": 600}}]}
        hw = discover.hardware_map(cfg, {"hardware": {"ram_gb": 32.0}}, {"PATH": ""}, gpu="")
        self.assertEqual(hw["spark"], {"name": "DGX Spark", "memory_gb": 128, "kind": "blackwell", "bandwidth_gbps": 273, "assumed": True})
        self.assertEqual(hw["studio"]["bandwidth_gbps"], None)     # "Mac Studio" alone does not name a chip
        self.assertEqual(hw["gpu"]["bandwidth_gbps"], 1008)
        self.assertEqual((hw["studio"]["name"], hw["studio"]["memory_gb"], hw["studio"]["kind"]), ("Mac Studio", 192, "apple"))
        self.assertNotIn("assumed", hw["studio"])
        self.assertEqual((hw["gpu"]["kind"], hw["gpu5"]["kind"]), ("nvidia", "blackwell"))
        self.assertNotIn("plain", hw)
        self.assertEqual(hw["custom"]["bandwidth_gbps"], 600)
        self.assertEqual(hw["this-machine"]["kind"], "cpu")


class Home(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        self.cat = fixture()

    def write_config(self, cfg):
        with open(os.path.join(self.home, "config.json"), "w") as f:
            json.dump(cfg, f)

    def run_discover(self, doc, cfg=None, **kw):
        cfg = core.load_config(self.home) if cfg is None else cfg
        return discover.discover(home=self.home, env={"PATH": ""}, cfg=cfg, providers=doc, catalog_data=self.cat,
                                 inventory=helpers.inventory(), probe_quota=False, hardware=kw.pop("hardware", {}), **kw)


class ModelsJson(Home):
    def doc(self):
        return providers_doc(
            api_provider("acme", ["titan", "atlas", "brand-new-thing"]),
            api_provider("router", ["acme/pebble-mini", "mystery-model", "text-embedding-3-large"]),
            cli_provider("claude", ["sonnet"]),
            local_provider("spark", ["openai/gpt-oss-120b", "Qwen/Qwen3-Coder-30B-A3B-Instruct", "gpt-oss:120b", "unknown-7b"],
                           {"openai/gpt-oss-120b": {"max_model_len": 131072, "root": "openai/gpt-oss-120b"}}),
            {"id": "openai", "name": "OpenAI", "access": "api", "style": "openai", "status": "missing", "reason": "no key", "models": []})

    def test_shape_follows_the_contract(self):
        data, notes, specs = self.run_discover(self.doc(), hardware={"spark": {"name": "DGX Spark", "memory_gb": 128, "kind": "blackwell", "bandwidth_gbps": 273, "assumed": True}})
        on_disk = read_json(catalog.models_path(self.home))
        self.assertEqual(on_disk["version"], 1)
        self.assertEqual(on_disk["catalog_as_of"], "2026-09-30")
        self.assertIsInstance(on_disk["generated_at"], int)
        self.assertEqual(on_disk["hardware"]["spark"]["memory_gb"], 128)
        ids = {m["id"]: m for m in on_disk["models"]}
        for k in ("acme/titan", "router/acme/pebble-mini", "cli:claude/sonnet", "local:spark/openai/gpt-oss-120b", "acme/brand-new-thing"):
            self.assertIn(k, ids)
        self.assertNotIn("router/text-embedding-3-large", ids)                   # not a chat model
        want = {"id", "provider", "model", "access", "style", "tools_ok", "vision", "context", "cost_in", "cost_out", "G", "jobs", "category",
                "serves", "basis", "benchmarks", "catalog_id", "family", "release_date", "enabled", "override"}
        for m in on_disk["models"]:
            self.assertLessEqual(want, set(m), m["id"])
            self.assertEqual(set(m["jobs"]), set(score.JOBS))
            self.assertIn(m["category"], core.CATEGORIES)
            self.assertIn(m["basis"], ("bench", "blend", "prior"))
            self.assertTrue(m["enabled"] and m["override"] is None)
        t = ids["acme/titan"]
        self.assertEqual((t["access"], t["style"], t["catalog_id"], t["cost_in"], t["cost_out"], t["category"]), ("api", "openai", "acme/titan", 5, 25, "frontier"))
        self.assertEqual(ids["router/acme/pebble-mini"]["catalog_id"], "acme/pebble-mini")
        self.assertIsNone(ids["acme/brand-new-thing"]["catalog_id"])
        self.assertEqual(ids["acme/brand-new-thing"]["basis"], "prior")
        c = ids["cli:claude/sonnet"]
        self.assertEqual((c["access"], c["tools_ok"], c["style"]), ("subscription-cli", False, "cli"))
        self.assertIn("local_fit", on_disk)
        row = on_disk["local_fit"]["spark"][0]
        self.assertLessEqual({"tps", "tps_est", "active_params_b", "active_known", "fit_score", "kv_gb"}, set(row))
        self.assertEqual(on_disk["hardware"]["spark"]["bandwidth_gbps"], 273)
        self.assertIn("spark", on_disk["local_fit"])
        self.assertEqual(set(on_disk["local_fit"]["spark"][0]) & {"model", "hf_repo", "params_b", "quant", "mem_gb", "G", "category", "serve_cmd"},
                         {"model", "hf_repo", "params_b", "quant", "mem_gb", "G", "category", "serve_cmd"})
        self.assertEqual(sorted(specs), sorted(ids))

    def test_local_models_join_the_catalog_and_serve_a_tier(self):
        data, _, _ = self.run_discover(self.doc(), hardware={})
        ids = {m["id"]: m for m in data["models"]}
        oss = ids["local:spark/openai/gpt-oss-120b"]
        self.assertEqual((oss["access"], oss["category"], oss["catalog_id"], oss["join"], oss["context"], oss["cost_out"]),
                         ("local", "local", "openai/gpt-oss-120b", "hf-repo", 131072, 0.0))
        self.assertIn(oss["serves"], ("fast", "standard", "strong"))
        self.assertTrue(oss["tools_ok"])
        self.assertEqual(ids["local:spark/gpt-oss:120b"]["catalog_id"], "openai/gpt-oss-120b")
        self.assertEqual(ids["local:spark/gpt-oss:120b"]["join"], "ollama")
        self.assertEqual(ids["local:spark/Qwen/Qwen3-Coder-30B-A3B-Instruct"]["catalog_id"], "alibaba/qwen3-coder-30b-a3b-instruct")
        self.assertTrue(ids["local:spark/Qwen/Qwen3-Coder-30B-A3B-Instruct"]["specialist"])
        unk = ids["local:spark/unknown-7b"]
        self.assertIsNone(unk["catalog_id"])
        self.assertEqual((unk["basis"], unk["category"], unk["tools_ok"]), ("prior", "local", False))

    def test_local_pins_still_override_serves(self):
        self.write_config({"models": {"local": {"strong": "local:spark/unknown-7b"}}})
        data, _, _ = self.run_discover(self.doc(), hardware={})
        unk = next(m for m in data["models"] if m["id"] == "local:spark/unknown-7b")
        self.assertEqual(unk["serves"], "strong")

    def test_overrides_are_applied_on_load_and_survive_rediscovery(self):
        self.write_config({"models_overrides": {"acme/atlas": {"category": "fast", "enabled": False},
                                                "acme/titan": {"category": "nonsense", "enabled": "yes"}, "gone/model": {"enabled": False}}})
        self.run_discover(self.doc())
        for _ in range(2):
            loaded = catalog.load_models(self.home)
            ids = {m["id"]: m for m in loaded["models"]}
            self.assertTrue(loaded["present"])
            self.assertEqual((ids["acme/atlas"]["category"], ids["acme/atlas"]["enabled"]), ("fast", False))
            self.assertEqual(ids["acme/atlas"]["override"], {"category": "fast", "enabled": False})
            self.assertNotEqual(ids["acme/atlas"]["category_auto"], "fast")
            self.assertEqual(ids["acme/titan"]["override"], None)          # invalid values are ignored
            self.assertTrue(ids["acme/titan"]["enabled"])
            self.run_discover(self.doc())                                     # rediscovery
        raw = read_json(catalog.models_path(self.home))
        self.assertTrue(all(m["override"] is None and m["enabled"] for m in raw["models"]))   # the file never holds the overrides
        cfg = dict(core.load_config(self.home), models_overrides={"acme/titan": {"enabled": False}})
        self.assertFalse(next(m for m in catalog.load_models(self.home, cfg)["models"] if m["id"] == "acme/titan")["enabled"])

    def test_load_models_without_a_file(self):
        d = catalog.load_models(self.home)
        self.assertEqual((d["present"], d["models"], d["version"]), (False, [], 1))
        with open(catalog.models_path(self.home), "w") as f:
            f.write("{not json")
        self.assertFalse(catalog.load_models(self.home)["present"])

    def test_live_latency_is_kept_on_rediscovery(self):
        data, _, specs = self.run_discover(self.doc())
        data["models"][0]["latency"] = {"ttft_ms": 120, "total_ms": 400, "n": 2, "fail": 0}
        catalog.write_models(data, self.home)
        data2, _, _ = self.run_discover(self.doc())
        self.assertEqual(next(m for m in data2["models"] if m["id"] == data["models"][0]["id"])["latency"]["ttft_ms"], 120)

    def test_discover_uses_providers_detect_all(self):
        seen = {}

        def fake(env=None, home=None, cfg=None, inventory=None, **kw):
            seen["inventory"] = inventory
            return self.doc()
        with mock.patch.object(providers, "detect_all", fake):
            data, notes, _ = discover.discover(home=self.home, env={"PATH": ""}, cfg=core.load_config(self.home), catalog_data=self.cat,
                                               inventory=helpers.inventory(), probe_quota=False, hardware={})
        self.assertIsNotNone(seen["inventory"])
        self.assertIn("acme/titan", {m["id"] for m in data["models"]})
        self.assertEqual(data["providers"]["acme"]["status"], "connected")
        self.assertNotIn("openai", data["providers"])                          # missing providers are left out

    def test_models_cmd_prefers_models_json(self):
        def run(*args):
            out = io.StringIO()
            with contextlib.redirect_stdout(out), mock.patch.dict(os.environ, {"MAESTRO_HOME": self.home}):
                rc = core.main(["models"] + list(args))
            return rc, out.getvalue()
        with mock.patch.object(core, "ensure_inventory", lambda *a, **k: helpers.inventory(clis=["claude-code"])):
            rc, before = run()
            self.assertEqual(rc, 0)
            self.assertNotIn("discovered", before)
            self.run_discover(self.doc())
            rc, after = run()
            self.assertIn("discovered", after)
            self.assertIn("acme (api key", after)
            self.assertIn("titan", after)
            rc, seats = run("--seats")
            self.assertEqual(seats, before)
            rc, js = run("--json")
            self.assertIn("acme/titan", {m["id"] for m in json.loads(js)["models"]})


class Refresh(Home):
    def test_offline_degrades_to_the_snapshot(self):
        def boom(*a, **k):
            raise OSError("network is unreachable")
        cat, notes = catalog.load(self.home, refresh=True, fetch=boom)
        self.assertGreater(len(cat["models"]), 300)
        self.assertTrue(any("refresh failed" in n and "bundled snapshot" in n for n in notes), notes)
        self.assertFalse(os.path.exists(os.path.join(self.home, catalog.REFRESH_FILE)))

    def test_a_bad_status_or_body_is_not_fatal(self):
        cat, notes = catalog.load(self.home, refresh=True, fetch=lambda *a, **k: (503, b""))
        self.assertTrue(any("HTTP 503" in n for n in notes))
        cat, notes = catalog.load(self.home, refresh=True, fetch=lambda *a, **k: (200, b"<html>"))
        self.assertTrue(any("refresh failed" in n for n in notes))

    def test_refresh_merges_and_is_cached(self):
        api = {"openai": {"id": "openai", "models": {
            "gpt-brand-new": {"name": "GPT Brand New", "release_date": "2026-09-29", "reasoning": True, "tool_call": True,
                              "modalities": {"input": ["text", "image"], "output": ["text"]}, "cost": {"input": 2, "output": 8}, "limit": {"context": 500000}},
            "gpt-5.4": {"cost": {"input": 1.0, "output": 9.0}, "limit": {"context": 1050000}, "modalities": {"output": ["text"]}},
            "gpt-image-9": {"modalities": {"output": ["image"]}}, "text-embedding-9": {"modalities": {"output": ["text"]}}}}}
        cat, notes = catalog.load(self.home, refresh=True, fetch=lambda *a, **k: (200, json.dumps(api).encode()))
        self.assertTrue(any("refreshed from models.dev" in n for n in notes), notes)
        new = cat["prov"]["openai"]["gpt-brand-new"]
        self.assertEqual((new["co"], new["x"], new["r"]), (8, 500000, 1))
        self.assertEqual(cat["prov"]["openai"]["gpt-5.4"]["co"], 9.0)
        self.assertNotIn("gpt-image-9", cat["prov"]["openai"])
        self.assertNotIn("text-embedding-9", cat["prov"]["openai"])
        self.assertEqual(catalog.resolve(cat, "openai", "gpt-5.4")[0], "openai/gpt-5.4")      # the join still finds the lab model
        cat2, notes2 = catalog.load(self.home)                                                  # no refresh: the cached copy applies
        self.assertEqual(cat2["prov"]["openai"]["gpt-brand-new"]["co"], 8)
        self.assertTrue(any("refreshed copy" in n for n in notes2))
        self.assertLess(os.path.getsize(os.path.join(self.home, catalog.REFRESH_FILE)), 5000)


class Backtest(unittest.TestCase):
    def model(self, mid, access, cat, g, cin=0.0, cout=0.0, serves=None, tools=True, **kw):
        m = {"id": mid, "provider": mid.split("/")[0], "model": mid.split("/")[-1], "access": access, "category": cat, "serves": serves or cat,
             "G": g, "jobs": {j: g for j in score.JOBS}, "tools_ok": tools, "cost_in": cin, "cost_out": cout, "enabled": True}
        m.update(kw)
        return m

    def setUp(self):
        self.models = [
            self.model("p/small", "api", "fast", 0.30, 0.1, 0.4),
            self.model("p/mid", "api", "standard", 0.50, 1, 4),
            self.model("p/big", "api", "strong", 0.60, 3, 12),
            self.model("p/top", "api", "frontier", 0.80, 10, 30),
            self.model("local:box/oss", "local", "local", 0.32, serves="fast", tools=False),
        ]
        self.cfg = core.load_config(tempfile.mkdtemp())

    def classify(self, text):
        job, d = text.split(":")
        return core.Classification(job, int(d), 1.0)

    def ids(self, ranked):
        return [r["id"] for r in ranked]

    def test_pick_qualifies_by_job_strength_and_tier_then_cost(self):
        pick = score.pick
        self.assertEqual(self.ids(pick(self.models, "ask", "local"))[0], "local:box/oss")
        self.assertEqual(self.ids(pick(self.models, "ask", "fast"))[0], "local:box/oss")                     # local first when preferred
        self.assertEqual(self.ids(pick(self.models, "ask", "fast", prefer_local=False))[0], "local:box/oss")   # still free
        self.assertEqual(self.ids(pick(self.models, "ask", "fast", tools=True))[0], "p/small")                # cheapest that has tools
        self.assertEqual(self.ids(pick(self.models, "code", "standard", tools=True))[0], "p/mid")
        self.assertEqual(self.ids(pick(self.models, "code", "strong", tools=True))[0], "p/big")
        self.assertEqual(self.ids(pick(self.models, "code", "frontier", tools=True))[0], "p/top")
        self.assertEqual(self.ids(pick(self.models, "code", "frontier", tools=True, max_tier="strong"))[0], "p/big")   # above max_tier is out
        self.assertEqual(pick([], "code", "fast"), [])

    def test_a_job_score_alone_does_not_pull_a_lower_tier_model_up(self):
        lopsided = self.model("p/lopsided", "api", "standard", 0.50, 0.01, 0.01, jobs=dict({j: 0.5 for j in score.JOBS}, code=0.60))
        ranked = score.pick(self.models + [lopsided], "code", "strong", tools=True)
        self.assertNotIn("p/lopsided", self.ids(ranked)[:1])        # code .60 >= .52, but it is a standard-tier model
        self.assertEqual(ranked[0]["id"], "p/big")
        low_job = self.model("p/weakjob", "api", "strong", 0.60, 0.01, 0.01, jobs=dict({j: 0.6 for j in score.JOBS}, code=0.40))
        self.assertNotIn("p/weakjob", self.ids(score.pick(self.models + [low_job], "code", "strong", tools=True))[:1])   # strong tier, but code .40 < .52

    def test_easy_tasks_walk_down_hard_ones_fall_back_to_the_strongest(self):
        only_small = [self.models[0]]
        hard = score.pick(only_small, "plan", "frontier", difficulty=3)              # d>=2: never down, so the last resort
        self.assertEqual(self.ids(hard), ["p/small"])
        self.assertTrue(hard[0]["below_floor"])
        self.assertIn("nothing reaches plan >= 0.68 for frontier", hard[0]["reasons"][0])
        easy = score.pick(only_small, "plan", "frontier", difficulty=0)               # easy: walk down to the tier that has a model
        self.assertEqual((easy[0]["id"], easy[0]["tier"], easy[0]["fallback"], easy[0]["below_floor"]), ("p/small", "fast", None, False))
        self.assertIn("stepped down to fast", " ".join(easy[0]["reasons"]))
        order = score.pick(self.models, "code", "standard", tools=True, difficulty=0)
        self.assertEqual(self.ids(order), ["p/mid", "p/big", "p/top", "p/small"])      # standard qualifiers by cost, then the tier below
        self.assertEqual([r["fallback"] for r in order], [None, None, None, "fast"])
        hard_order = score.pick(self.models, "code", "standard", tools=True, difficulty=3)
        self.assertEqual(self.ids(hard_order), ["p/mid", "p/big", "p/top"])            # hard: never down
        self.assertEqual(self.ids(score.pick(self.models, "code", "strong", tools=True, difficulty=3)), ["p/big", "p/top"])   # a frontier model qualifies for a strong cell
        self.assertEqual(score.pick([], "plan", "strong"), [])

    def test_objectives(self):
        ranked = lambda **kw: self.ids(score.pick(self.models, "code", "standard", tools=True, **kw))   # noqa: E731
        self.assertEqual(ranked()[0], "p/mid")
        self.assertEqual(ranked(objective="cost")[0], "p/mid")
        self.assertEqual(ranked(objective="quality")[0], "p/top")            # the strongest qualifier, cost breaks ties
        self.assertEqual(ranked(objective="nonsense")[0], "p/mid")

    def test_zero_cost_seats_are_ordered_by_quota_headroom_and_filtered_by_state(self):
        a = self.model("cli:a/x", "subscription-cli", "standard", 0.5, quota={"state": "ok", "headroom": 0.2})
        b = self.model("cli:b/x", "subscription-cli", "standard", 0.5, quota={"state": "ok", "headroom": 0.9})
        c = self.model("cli:c/x", "subscription-cli", "standard", 0.5, quota={"state": "low", "headroom": 0.05})
        d = self.model("cli:d/x", "subscription-cli", "standard", 0.5, quota={"state": "cooldown", "headroom": 0.0})
        e = self.model("cli:e/x", "subscription-cli", "standard", 0.5, quota={"state": "exhausted", "headroom": 0.0})
        ranked = score.pick([a, b, c, d, e], "ask", "standard")
        self.assertEqual(self.ids(ranked), ["cli:b/x", "cli:a/x", "cli:c/x"])         # d and e are out, the low one goes last
        self.assertIn("subscription, no marginal cost", ranked[0]["reasons"][1])
        self.assertIn("quota low", ranked[2]["reasons"])

    def test_filters(self):
        free = self.model("r/x:free", "api", "frontier", 0.9)
        zero = self.model("r/zero", "api", "frontier", 0.9)
        off = dict(self.models[3], id="p/top2", enabled=False)
        vision = dict(self.models[1], id="p/vision", vision=True)
        small_ctx = dict(self.models[3], id="p/tiny", context=1000)
        pool = self.models + [free, zero, off, vision, small_ctx]
        self.assertEqual(self.ids(score.pick(pool, "code", "frontier", tools=True, min_context=5000))[0], "p/top")   # free, disabled, tiny-context out
        self.assertNotIn("p/top2", self.ids(score.pick(pool, "code", "frontier", tools=True)))
        self.assertNotIn("r/zero", self.ids(score.pick(pool, "code", "frontier", tools=True)))
        self.assertEqual(self.ids(score.pick(pool, "ask", "standard", vision=True)), ["p/vision"] + [i for i in self.ids(score.pick(pool, "ask", "standard", vision=True)) if i != "p/vision"])
        self.assertNotIn("p/tiny", self.ids(score.pick(pool, "code", "frontier", tools=True, min_context=5000)))
        spec = self.model("p/spec", "api", "specialist", 0.5, 0.01, 0.01, serves="standard", jobs=dict({j: 0.5 for j in score.JOBS}, code=0.60))
        self.assertIn("p/spec", self.ids(score.pick(pool + [spec], "code", "standard", tools=True)))    # lifted for code
        self.assertNotIn("p/spec", self.ids(score.pick(pool + [spec], "ask", "standard", tools=True)))   # not for ask

    def test_numbers_on_a_tiny_fixture(self):
        rows = [{"prompt": "ask:0", "job": "ask", "difficulty": 0},          # ladder: local   -> local/oss (fast): hit
                {"prompt": "code:1", "job": "code", "difficulty": 1},        # standard       -> p/mid: hit
                {"prompt": "code:3", "job": "code", "difficulty": 3},        # strong         -> p/big: hit
                {"prompt": "plan:3", "job": "plan", "difficulty": 3}]        # frontier       -> p/top: hit
        bt = discover.backtest(self.models, rows, self.cfg, classify=self.classify)
        self.assertEqual((bt["prompts"], bt["category_hit"], bt["within_1"], bt["under_routed"], bt["severe_under_routed"]), (4, 1.0, 1.0, 0.0, 0.0))
        self.assertEqual(bt["local_share"], 0.25)
        self.assertEqual(bt["strongest"], "p/top")
        # tokens: 2000 in, 500*(1+d) out.  mid d1: 2000*1+1000*4=6000 | big d3: 2000*3+2000*12=30000 | top d3: 2000*10+2000*30=80000; ask is local (0)
        self.assertAlmostEqual(bt["cost_per_100"], (6000 + 30000 + 80000) / 1e6 / 4 * 100, places=4)
        # always top: d0 1500? ask d0: 2000*10+500*30=35000 | d1: 2000*10+1000*30=50000 | d3: 80000 | d3: 80000
        self.assertAlmostEqual(bt["cost_per_100_strongest"], (35000 + 50000 + 80000 + 80000) / 1e6 / 4 * 100, places=4)
        self.assertAlmostEqual(bt["saving"], 1 - 116000 / 245000, places=3)

    def test_misses_are_counted(self):
        only_small = [self.models[0]]
        rows = [{"prompt": "plan:3", "job": "plan", "difficulty": 3}, {"prompt": "ask:0", "job": "ask", "difficulty": 0}]
        bt = discover.backtest(only_small, rows, self.cfg, classify=self.classify)
        self.assertEqual(bt["category_hit"], 0.5)
        self.assertEqual(bt["fallbacks"], 1)                                             # the frontier prompt is answered with what there is
        self.assertEqual((bt["under_routed"], bt["severe_under_routed"]), (0.5, 0.5))
        self.assertEqual(bt["requested_tier_served"], 0.5)

    def test_backtest_over_the_real_eval_prompts(self):
        rows = discover.load_prompts(os.path.join(ROOT, "evals", "prompts.jsonl"))
        self.assertEqual(len(rows), 474)
        bt = discover.backtest(self.models, rows, self.cfg)
        self.assertEqual(bt["prompts"], 474)
        self.assertGreater(bt["within_1"], 0.6)
        self.assertLess(bt["cost_per_100"], bt["cost_per_100_strongest"])
        self.assertEqual(bt["unroutable"], 0)

    def test_live_probe_records_latency_and_failures(self):
        data = {"models": [dict(self.models[0], model="small"), dict(self.models[1], model="mid")]}

        def request(spec, model, req, stream=False, **kw):
            self.assertEqual((req["max_tokens"], stream), (16, True))
            if model == "mid":
                return 500, {}, {"error": {"message": "boom"}}
            return 200, {}, iter([b"data: {}\n\n", b"data: [DONE]\n\n"])
        rows = [{"prompt": "hello %d" % i, "job": "ask", "difficulty": 0} for i in range(10)]
        res = discover.live_probe(data, {"p/small": "p", "p/mid": "p"}, ["p/small", "p/mid"], 3, rows, self.cfg, request_fn=request, workers=2)
        self.assertEqual((res["p/small"]["fail"], res["p/small"]["n"]), (0, 3))
        self.assertIsNotNone(res["p/small"]["ttft_ms"])
        self.assertEqual((res["p/mid"]["fail"], res["p/mid"]["ttft_ms"]), (3, None))
        self.assertIn("boom", res["p/mid"]["error"]) if False else self.assertIn("HTTP 500", res["p/mid"]["error"])
        self.assertEqual(data["models"][0]["latency"]["n"], 3)


class Served(Home):
    """Maestro uses what is served (loaded) and advises only when nothing is served."""
    SPARK = {"name": "DGX Spark", "memory_gb": 128, "kind": "blackwell", "bandwidth_gbps": 273, "assumed": True}

    def doc(self, models=("gpt-oss-120b",), loaded=None):
        info = {n: {"loaded": (loaded or {}).get(n, True)} for n in models}
        return providers_doc(api_provider("acme", ["titan"]), local_provider("spark", list(models), info))

    def cli(self, doc, *argv, cfg=None):
        self.write_config(dict({"local_endpoints": [{"name": "spark", "url": "http://10.0.0.5:8000", "kind": "openai"}]}, **(cfg or {})))
        out = io.StringIO()
        args = core.build_parser().parse_args(["discover"] + list(argv))
        with contextlib.redirect_stdout(out), mock.patch.dict(os.environ, {"MAESTRO_HOME": self.home, "PATH": ""}), \
                mock.patch.object(providers, "detect_all", lambda **kw: doc), \
                mock.patch.object(core, "ensure_inventory", lambda *a, **k: helpers.inventory()), \
                mock.patch.object(catalog, "load", lambda *a, **k: (self.cat, [])):
            self.assertEqual(args.fn(args), 0)
        return out.getvalue()

    def test_loaded_and_routable_are_recorded_and_local_allow_load_opens_routing(self):
        doc = self.doc(("on-8b", "off-120b"), {"off-120b": False})
        data, _, _ = self.run_discover(doc)
        by = {m["model"]: m for m in data["models"] if m["access"] == "local"}
        self.assertEqual((by["on-8b"]["loaded"], by["on-8b"]["routable"]), (True, True))
        self.assertEqual((by["off-120b"]["loaded"], by["off-120b"]["routable"], by["off-120b"]["enabled"]), (False, False, True))
        on_disk = {m["model"]: m for m in read_json(catalog.models_path(self.home))["models"] if m["access"] == "local"}
        self.assertFalse(on_disk["off-120b"]["routable"])
        data, _, _ = self.run_discover(doc, cfg=dict(core.load_config(self.home), local_allow_load=True))
        by = {m["model"]: m for m in data["models"] if m["access"] == "local"}
        self.assertEqual((by["off-120b"]["loaded"], by["off-120b"]["routable"]), (False, True))
        loaded = catalog.load_models(self.home, core.load_config(self.home))   # the config is read again on load
        self.assertFalse(next(m for m in loaded["models"] if m["model"] == "off-120b")["routable"])

    def test_backtest_and_live_probe_ignore_a_model_that_is_not_loaded(self):
        data, _, specs = self.run_discover(self.doc(("on-8b", "off-120b"), {"off-120b": False}))
        bt = discover.backtest(data["models"], [{"prompt": "hello there", "job": "ask", "difficulty": 0}] * 4, core.load_config(self.home))
        self.assertNotIn("local:spark/off-120b", bt["models_used"])
        sent = []

        def request(spec, model, req, stream=False, **kw):
            sent.append(model)
            return 200, {}, iter([b"data: {}\n\n"])
        rows = [{"prompt": "hi %d" % i, "job": "ask", "difficulty": 0} for i in range(4)]
        ids = [m["id"] for m in data["models"] if m["access"] == "local"]
        res = discover.live_probe(data, specs, ids, 2, rows, core.load_config(self.home), request_fn=request)
        self.assertEqual(list(res), ["local:spark/on-8b"])
        self.assertEqual(set(sent), {"on-8b"})

    def test_live_probe_sends_one_request_at_a_time_per_endpoint(self):
        import threading
        import time
        data, _, specs = self.run_discover(self.doc(("a-8b", "b-8b", "c-8b")))
        active, worst, lock = [0], [0], threading.Lock()

        def request(spec, model, req, stream=False, **kw):
            with lock:
                active[0] += 1
                worst[0] = max(worst[0], active[0])
            time.sleep(0.03)
            with lock:
                active[0] -= 1
            return 200, {}, iter([b"data: {}\n\n"])
        rows = [{"prompt": "hi %d" % i, "job": "ask", "difficulty": 0} for i in range(4)]
        ids = [m["id"] for m in data["models"] if m["access"] == "local"]
        res = discover.live_probe(data, specs, ids, 2, rows, core.load_config(self.home), request_fn=request, workers=6)
        self.assertEqual(len(res), 3)
        self.assertEqual(worst[0], 1)

    def test_cli_live_only_probes_routable_models(self):
        sent = []
        doc = self.doc(("on-8b", "off-120b"), {"off-120b": False})
        from maestro import clients
        with mock.patch.object(clients, "complete", lambda spec, model, req, **kw: (sent.append(model), (200, {}, iter([b"data: {}\n\n"])))[1]):
            out = self.cli(doc, "--live", "1", "--prompts", os.path.join(ROOT, "evals", "prompts.jsonl"))
        self.assertIn("on-8b", out)
        self.assertNotIn("local:spark/off-120b  ", out.split("live probe:")[1])
        self.assertNotIn("off-120b", sent)

    def test_plain_discover_with_a_served_box_says_it_uses_these_and_suggests_nothing(self):
        out = self.cli(self.doc(("gpt-oss-120b",)))
        self.assertIn("spark: serving 1 model (gpt-oss-120b) - using these", out)
        self.assertNotIn("would fit", out)
        self.assertNotIn("vllm serve", out)
        self.assertNotIn("Could run on", out)

    def test_plain_discover_with_nothing_served_points_at_local_fit(self):
        out = self.cli(providers_doc(api_provider("acme", ["titan"]), local_provider("spark", [])))
        self.assertIn("spark: nothing served; `maestro-router discover --local-fit` lists models that would fit (advice only)", out)
        self.assertNotIn("vllm serve", out)

    def test_installed_but_not_loaded_counts_as_nothing_served(self):
        out = self.cli(self.doc(("big-120b",), {"big-120b": False}))
        self.assertIn("spark: nothing served (1 installed but not loaded, ignored", out)
        self.assertIn("`maestro-router discover --local-fit` lists models that would fit (advice only)", out)
        self.assertIn("installed, not loaded: not used", out)

    def test_local_fit_on_a_serving_box_says_replace_and_annotates_every_command(self):
        out = self.cli(self.doc(("gpt-oss-120b",)), "--local-fit")
        self.assertIn("Alternatives on spark (would REPLACE what it serves now: gpt-oss-120b)", out)
        self.assertIn("vLLM reserves --gpu-memory-utilization, 0.9 by default", out)
        self.assertNotIn("Could run on", out)
        self.assertNotIn(" add ", out.split("Alternatives on")[1].lower().replace("does not add", ""))
        cmds = [l for l in out.splitlines() if l.strip().startswith(("vllm serve", "ollama run", "llama-server"))]
        self.assertTrue(cmds)
        lines = out.splitlines()
        for i, l in enumerate(lines):
            if l.strip().startswith(("vllm serve", "ollama run", "llama-server")):
                self.assertIn("# stop the current server first", lines[i - 1])

    def test_local_fit_json_rows_carry_replaces(self):
        data, _, _ = self.run_discover(self.doc(("gpt-oss-120b",)), hardware={"spark": self.SPARK})
        row = data["local_fit"]["spark"][0]
        self.assertEqual(row["replaces"], ["local:spark/gpt-oss-120b"])
        self.assertTrue(row["serve_cmd"].startswith("# stop the current server first"))
        data, _, _ = self.run_discover(providers_doc(api_provider("acme", ["titan"])), hardware={"spark": self.SPARK})
        self.assertNotIn("replaces", data["local_fit"]["spark"][0])
        self.assertFalse(data["local_fit"]["spark"][0]["serve_cmd"].startswith("#"))

    def test_local_fit_on_an_empty_box_keeps_the_could_run_header(self):
        out = self.cli(providers_doc(api_provider("acme", ["titan"]), local_provider("spark", [])), "--local-fit")
        self.assertIn("Could run on spark: DGX Spark, 128 GB", out)
        self.assertNotIn("stop the current server", out)
        self.assertIn("Advice only", out)


class Rendering(Home):
    def test_summary_and_fit_lines(self):
        data, _, _ = self.run_discover(providers_doc(api_provider("acme", ["titan", "atlas"]), local_provider("spark", ["gpt-oss:120b"])),
                                       hardware={"spark": {"name": "DGX Spark", "memory_gb": 128, "kind": "blackwell", "bandwidth_gbps": 273, "assumed": True}})
        text = "\n".join(discover.summary_lines(data))
        self.assertIn("acme (api key, 2 models", text)
        self.assertIn("titan", text)
        self.assertIn("frontier", text)
        self.assertIn("local>", text)
        fit = "\n".join(discover.fit_lines(data))
        self.assertIn("Alternatives on spark (would REPLACE what it serves now: gpt-oss:120b): DGX Spark, 128 GB, 273 GB/s (assumed", fit)
        self.assertNotIn("Could run on", fit)
        self.assertIn("tok/s", fit)
        self.assertIn("serving now: gpt-oss:120b", fit)
        self.assertIn("vllm serve", fit)
        self.assertIn("# stop the current server first", fit)

    def test_cli_discover_offline(self):
        out = io.StringIO()
        fake = providers_doc(api_provider("acme", ["titan"]))
        args = core.build_parser().parse_args(["discover", "--json", "--backtest", "--local-fit"])
        with contextlib.redirect_stdout(out), mock.patch.dict(os.environ, {"MAESTRO_HOME": self.home, "PATH": ""}), \
                mock.patch.object(providers, "detect_all", lambda **kw: fake), \
                mock.patch.object(core, "ensure_inventory", lambda *a, **k: helpers.inventory()), \
                mock.patch.object(catalog, "load", lambda *a, **k: (self.cat, [])):
            rc = args.fn(args)
        self.assertEqual(rc, 0)
        body = json.loads(out.getvalue())
        self.assertIn("backtest", body)
        self.assertEqual(body["models"][0]["id"], "acme/titan")
        self.assertTrue(os.path.exists(catalog.models_path(self.home)))


class RouteStaysLight(unittest.TestCase):
    def test_route_does_not_import_catalog_or_score(self):
        code = ("import sys, os, tempfile; sys.path.insert(0, %r); os.environ['MAESTRO_HOME'] = tempfile.mkdtemp(); import maestro; "
                "maestro.route('fix the failing test in auth.py', harness='claude-code', "
                "inventory={'harnesses': {}, 'local_models': [], 'default_models': {}}, classifier='rules'); "
                "bad = [m for m in ('maestro.catalog', 'maestro.score', 'maestro.discover', 'maestro.providers') if m in sys.modules]; "
                "print(bad)" % os.path.join(ROOT, "skills", "maestro", "scripts"))
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(res.stdout.strip(), "[]", res.stderr)


@unittest.skipUnless(sys.version_info >= (3, 11), "build_catalog.py needs tomllib (3.11+)")
class BuildCatalog(unittest.TestCase):
    def write(self, path, text):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)

    def test_builds_a_snapshot_from_a_models_dev_tree(self):
        sys.path.insert(0, os.path.join(ROOT, "evals"))
        import build_catalog
        with tempfile.TemporaryDirectory() as tmp:
            md = os.path.join(tmp, "md")
            body = 'name = "%s"\nrelease_date = "%s"\nreasoning = true\ntool_call = true\nopen_weights = %s\n[limit]\ncontext = 131_072\noutput = 8000\n' \
                   '[modalities]\ninput = ["text"]\noutput = %s\n'
            self.write(md + "/models/acme/big-120b.toml", body % ("Big 120B", "2026-08", "true", '["text"]') +
                       '[[weights]]\nlabel = "HF"\nurl = "https://huggingface.co/acme/Big-120B"\n'
                       '[[benchmarks]]\nname = "GPQA Diamond"\nscore = 80.5\nmetric = "accuracy"\n')
            self.write(md + "/models/acme/big-120b-turbo.toml", body % ("Big 120B Turbo", "2026-08", "true", '["text"]') +
                       '[[weights]]\nlabel = "HF"\nurl = "https://huggingface.co/acme/Big-120B"\n')
            self.write(md + "/models/acme/pixel.toml", body % ("Pixel", "2026-08", "false", '["image"]'))
            self.write(md + "/models/acme/old.toml", "status = \"deprecated\"\n" + body % ("Old", "2023-01", "false", '["text"]'))
            self.write(md + "/models/acme/mystery-moe.toml", body % ("Mystery", "2026-08", "true", '["text"]'))
            self.write(md + "/providers/cloud/provider.toml", 'name = "Cloud"\nenv = ["CLOUD_KEY"]\nnpm = "@ai-sdk/openai-compatible"\n')
            self.write(md + "/providers/cloud/models/big.toml", 'base_model = "acme/big-120b"\n[cost]\ninput = 1.0\noutput = 4.0\n')
            self.write(md + "/providers/cloud/models/own.toml", body % ("Own", "2026-07", "false", '["text"]') + "[cost]\ninput = 0.5\noutput = 2.0\n")
            out = os.path.join(tmp, "catalog.json.gz")
            extra = os.path.join(tmp, "extra.json")
            with open(extra, "w") as f:
                json.dump({"_note": "x", "acme/big-120b": [{"name": "GPQA Diamond", "score": 80.5, "metric": "accuracy", "source": "https://huggingface.co/x"},
                                                           {"name": "SWE-Bench Verified", "score": 61.0, "metric": "accuracy", "source": "https://huggingface.co/x"}],
                           "acme/mystery-moe": [{"name": "MMLU-Pro", "score": 70.0, "source": "https://huggingface.co/x"}],
                           "acme/not-there": [{"name": "GPQA Diamond", "score": 1.0}]}, f)
            self.addCleanup(setattr, build_catalog, "EXTRA_BENCH", build_catalog.EXTRA_BENCH)
            build_catalog.EXTRA_BENCH = extra
            with contextlib.redirect_stdout(io.StringIO()):
                rc = build_catalog.main(["--models-dev", md, "--out", out, "--no-hf", "--as-of", "2026-09-30"])
            self.assertEqual(rc, 0)
            cat = catalog.load_snapshot(out)
            self.assertEqual(sorted(cat["models"]), ["acme/big-120b", "acme/big-120b-turbo", "acme/mystery-moe"])     # image and old-deprecated dropped
            big = cat["models"]["acme/big-120b"]
            self.assertEqual((big["hf"], big["pt"], big["ps"], big["p"], big["d"]), ("acme/Big-120B", 120.0, "name", [1.0, 4.0], "2026-08-01"))
            self.assertEqual(big["b"], [["GPQA Diamond", 80.5, "accuracy"], ["SWE-Bench Verified", 61.0, "accuracy"]])   # the duplicate row is not added twice
            self.assertEqual(cat["models"]["acme/big-120b-turbo"]["b"], big["b"])       # same weights: the card's numbers are shared
            self.assertEqual(cat["models"]["acme/mystery-moe"]["b"], [["MMLU-Pro", 70.0]])
            self.assertNotIn("pt", cat["models"]["acme/mystery-moe"])                            # unknown stays unknown
            self.assertEqual(cat["prov"]["cloud"]["big"], {"c": "acme/big-120b", "ci": 1.0, "co": 4.0})
            self.assertEqual(cat["prov"]["cloud"]["own"]["n"], "Own")
            with open(out, "rb") as f:
                first = f.read()
            with contextlib.redirect_stdout(io.StringIO()):
                build_catalog.main(["--models-dev", md, "--out", out, "--no-hf", "--as-of", "2026-09-30"])
            with open(out, "rb") as f:
                self.assertEqual(f.read(), first)                                                # deterministic bytes


if __name__ == "__main__":
    unittest.main()
