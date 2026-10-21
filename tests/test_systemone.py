"""systemone: Jev's wire format, native answers from S1 (the engine and decide are stubbed in sys.modules) and delegation to a
served model through its logprobs (fake OpenAI-compatible servers on 127.0.0.1: plain logprobs, a thinking model, no logprobs)."""
import argparse
import io
import json
import math
import os
import stat
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

import helpers
from helpers import inventory, maestro as R
from fake_http import FakeServer

from maestro import decide as REAL_DECIDE, systemone as SO

ND = statistics.NormalDist()
POST = {"job_p": dict({j: 0.0 for j in R.JOBS}, debug=0.75, code=0.2, ask=0.05), "b": [0.4, 0.8], "group": "agentic", "tools": 0.9,
        "self_contained": 0.3, "stakes": 1.0, "detectable": 0.8, "ambiguity": 0.5, "tok_out": [math.log(900), 0.6], "turns": [2.0, 0.5],
        "ood": 0.1, "source": "s1"}
CLAUDE = ["claude-code:haiku", "claude-code:sonnet", "claude-code:opus", "claude-code:fable"]


def fake_s1(post=None, quantiles=None, calls=None):
    mod = types.ModuleType("maestro.s1")

    def posterior(task, cls=None, hints=None, path=None):
        if calls is not None:
            calls.append((task, cls))
        return json.loads(json.dumps(post or POST))
    mod.posterior = posterior
    mod.load = lambda path=None: ({"header": {"ordinal": {"b_quantiles": quantiles}}}, "") if quantiles else (None, "no s1.bin")
    return mod


def fake_decide(p, pbest, calls=None):
    """rank: the seats in the order of `p` (a dict, best first), dropping those not in it; p_best: `pbest`."""
    mod = types.ModuleType("maestro.decide")
    mod.seat_of = lambda c, models=None: {"id": c.get("name") or c.get("id")}

    def rank(seats, post, ctx):
        if calls is not None:
            calls.append((seats, ctx))
        ids = [s["id"] for s in seats]
        return [{"seat": k, "p": v, "eu": 1.0 - i * 0.1} for i, (k, v) in enumerate(p.items()) if k in ids]
    mod.rank = rank
    mod.p_best = lambda ranked, did, n=64: dict(pbest)
    mod.is_private = REAL_DECIDE.is_private
    return mod


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = self.tmp.name
        self.cfg = R.load_config(self.home)
        self.s1_calls, self.d_calls = [], []
        self.mods = {"maestro.s1": fake_s1(calls=self.s1_calls),
                     "maestro.decide": fake_decide({"claude-code:sonnet": 0.8, "claude-code:opus": 0.9, "claude-code:haiku": 0.5},
                                                   {"claude-code:opus": 0.55, "claude-code:sonnet": 0.45}, self.d_calls)}
        patcher = mock.patch.dict(sys.modules, self.mods)
        patcher.start()
        self.addCleanup(patcher.stop)
        SO._CACHE.clear()

    def tearDown(self):
        SO._CACHE.clear()
        self.tmp.cleanup()

    def ctx(self, **kw):
        return dict({"home": self.home, "env": {"PATH": ""}, "cfg": self.cfg, "inv": inventory(), "models": {}}, **kw)

    def ask(self, questions, state="fix the failing test in auth.py", model="maestro-s1", **kw):
        return SO.handle({"model": model, "state": state, "questions": questions}, self.ctx(**kw))

    def ok(self, questions, **kw):
        st, out = self.ask(questions, **kw)
        self.assertEqual(st, 200, out)
        return out


def read(path):
    with open(path) as f:
        return f.read()


def score(levels, **kw):
    return dict({"type": "score", "criteria": list(levels)}, **kw)


def choice(keys, **kw):
    return dict({"type": "choice", "criteria": {k: "" for k in keys}}, **kw)


class Validation(Base):
    def detail(self, body):
        st, out = SO.handle(body, self.ctx())
        self.assertEqual(st, 422)
        return [(d["loc"], d["type"]) for d in out["detail"]]

    def test_top_level_fields(self):
        self.assertEqual(self.detail([]), [(["body"], "dict_type")])
        self.assertEqual(self.detail({}), [(["body", "model"], "missing"), (["body", "state"], "missing"), (["body", "questions"], "missing")])
        self.assertIn((["body", "questions"], "too_short"), self.detail({"model": "maestro-s1", "state": "x", "questions": {}}))
        self.assertIn((["body", "state"], "type_error"), self.detail({"model": "maestro-s1", "state": 3, "questions": {"q": {"type": "noul"}}}))

    def test_questions(self):
        qs = {"a": {"criteria": {}}, "b": {"type": "rank"}, "c": {"type": "choice"}, "d": {"type": "score", "criteria": []},
              "e": {"type": "noul", "criteria": "yes"}, "f": {"type": "noul", "intent": 3}, "g": {"type": "choice", "criteria": {"x": 3}}}
        got = self.detail({"model": "maestro-s1", "state": "x", "questions": qs})
        self.assertEqual(got, [(["body", "questions", "a"], "union_tag_not_found"), (["body", "questions", "b"], "union_tag_invalid"),
                               (["body", "questions", "c", "choice", "criteria"], "missing"),
                               (["body", "questions", "d", "score", "criteria"], "value_error"),
                               (["body", "questions", "e", "noul", "criteria"], "value_error"),
                               (["body", "questions", "f", "noul", "intent"], "value_error"),
                               (["body", "questions", "g", "choice", "criteria"], "value_error")])

    def test_unknown_model(self):
        self.assertEqual(self.detail({"model": "gpt-9", "state": "x", "questions": {"q": {"type": "noul"}}}), [(["body", "model"], "model_not_found")])

    def test_question_cap(self):
        qs = {"q%d" % i: {"type": "noul"} for i in range(SO.MAX_QUESTIONS + 1)}
        self.assertEqual(self.detail({"model": "maestro-s1", "state": "x", "questions": qs}), [(["body", "questions"], "too_long")])


class Native(Base):
    def test_job_is_the_head_renormalized_over_the_keys(self):
        out = self.ok({"job": choice(["debug", "code"])})
        a = out["answers"]["job"]
        self.assertEqual((a["type"], a["choice"]), ("choice", "debug"))
        self.assertAlmostEqual(a["probabilities"]["debug"], 0.75 / 0.95, places=4)
        self.assertAlmostEqual(a["confidence"], a["probabilities"]["debug"])
        self.assertEqual((out["model"], out["maestro"]["source"], out["usage"]), ("maestro-s1", {"job": "native"},
                                                                                    {"input_tokens": 0, "output_tokens": 0}))

    def test_state_object_reaches_s1(self):
        state = {"task": "why does login fail", "harness": "codex", "history": [{"role": "user", "content": "earlier ask"},
                                                                                  {"role": "assistant", "content": "an answer"}]}
        self.ok({"job": choice(R.JOBS)}, state=state)
        task, cls = self.s1_calls[0]
        self.assertEqual((task["text"], task["prev_user"], task["n_user"], task["harness"], task["mode"]),
                         ("why does login fail", "earlier ask", 1, "codex", "skill"))
        self.assertIsInstance(cls, R.Classification)
        self.ok({"job": choice(R.JOBS)}, state={"prompt": "p"})
        self.assertEqual(self.s1_calls[-1][0]["text"], "p")
        self.ok({"job": choice(R.JOBS)}, state={"other": 1})
        self.assertEqual(self.s1_calls[-1][0]["text"], '{"other": 1}')

    def test_difficulty_bins_b_at_the_training_quantiles(self):
        out = self.ok({"difficulty": score(["trivial", "routine", "hard", "frontier"])})
        a, qs, (mu, sd) = out["answers"]["difficulty"], SO.B_QUANTILES, POST["b"]
        cuts = [qs[0] + (qs[1] - qs[0]) * 0.5, qs[2], qs[3] + (qs[4] - qs[3]) * 0.5]     # p = 1/4, 1/2, 3/4 of 5 points at i/6
        cdf = [0] + [ND.cdf((c - mu) / sd) for c in cuts] + [1]
        for k in range(4):
            self.assertAlmostEqual(a["probabilities"][str(k)], cdf[k + 1] - cdf[k], places=4)
        self.assertEqual(a["legend"], {"0": "trivial", "1": "routine", "2": "hard", "3": "frontier"})
        self.assertAlmostEqual(a["score"], sum(k * a["probabilities"][str(k)] for k in range(4)), places=3)
        for alias in ("complexity", "reasoning"):
            self.assertEqual(self.ok({alias: score("abcd")})["answers"][alias]["probabilities"], a["probabilities"])

    def test_difficulty_uses_header_quantiles_and_rises_with_b(self):
        self.mods["maestro.s1"].load = lambda path=None: ({"header": {"ordinal": {"b_quantiles": [-3 + 6 * i / 98.0 for i in range(99)]}}}, "")
        lo = self.ok({"difficulty": score("abc")})["answers"]["difficulty"]
        c1, c2 = (-3 + 6 * (100 / 3.0 - 1) / 98, -3 + 6 * (200 / 3.0 - 1) / 98)        # the 1/3 and 2/3 percentiles
        self.assertAlmostEqual(lo["probabilities"]["1"], ND.cdf((c2 - 0.4) / 0.8) - ND.cdf((c1 - 0.4) / 0.8), places=4)
        self.mods["maestro.s1"].posterior = lambda task, cls=None, **k: dict(POST, b=[2.0, 0.8])
        self.assertGreater(self.ok({"difficulty": score("abc")})["answers"]["difficulty"]["score"], lo["score"])

    def test_stakes_and_ambiguity_ordinals(self):
        a = self.ok({"stakes": score("abc"), "risk": score("ab"), "ambiguity": score("abcd")})["answers"]
        self.assertEqual(a["stakes"]["probabilities"], {"0": 0.25, "1": 0.5, "2": 0.25})            # E = 1: Binomial(2, 1/2)
        self.assertEqual(a["risk"]["probabilities"], {"0": 0.5, "1": 0.5})
        e = 0.25                                                                                    # ambiguity 0.5 / 2
        p3 = [(1 - e) ** 2, 2 * e * (1 - e), e * e]
        want = [p3[0] * 0.75, p3[0] * 0.25 + p3[1] * 0.5, p3[1] * 0.5 + p3[2] * 0.25, p3[2] * 0.75]
        self.assertEqual([a["ambiguity"]["probabilities"][str(k)] for k in range(4)], [round(x, 4) for x in want])
        self.mods["maestro.s1"].posterior = lambda task, cls=None, **k: dict(POST, stakes_p=[0.9, 0.6])
        self.assertEqual(self.ok({"stakes": score("abc")})["answers"]["stakes"]["probabilities"], {"0": 0.1, "1": 0.3, "2": 0.6})

    def test_binary_heads(self):
        a = self.ok({"needs_tools": {"type": "noul"}, "self_contained": {"type": "noul"}, "detectable": {"type": "noul", "criteria": {"true": "x"}},
                     "needs_vision": {"type": "noul"}})["answers"]
        self.assertEqual({k: v["noul"] for k, v in a.items()}, {"needs_tools": 0.9, "self_contained": 0.3, "detectable": 0.8, "needs_vision": 0.0})
        self.assertEqual(a["needs_tools"], {"type": "noul", "noul": 0.9})
        self.assertEqual(self.ok({"needs_vision": {"type": "noul"}}, state={"task": "t", "images": ["data:..."]})["answers"]["needs_vision"]["noul"], 1.0)

    def test_tokens_from_legend_ranges_or_default_buckets(self):
        legend = ["<100", "100-300", "300-1k", "1k-3k", ">3k"]
        a = self.ok({"tokens": score(legend), "length": score(["short", "medium", "long", "very long", "huge"])})["answers"]
        mu, sd = POST["tok_out"]
        cuts = [math.log(c) for c in (100, 300, 1000, 3000)]
        cdf = [0] + [ND.cdf((c - mu) / sd) for c in cuts] + [1]
        self.assertEqual([a["tokens"]["probabilities"][str(k)] for k in range(5)], [round(cdf[k + 1] - cdf[k], 4) for k in range(5)])
        self.assertEqual(a["tokens"]["probabilities"], a["length"]["probabilities"])   # the defaults are the panel's buckets
        self.assertEqual(a["tokens"]["legend"]["2"], "300-1k")
        self.assertEqual(SO._token_cuts(["0 - 500 tokens", "500 to 2,000", "over 2,000"]), [math.log(500), math.log(2000)])
        self.mods["maestro.s1"].posterior = lambda task, cls=None, **k: dict(POST, tok_out=[900, 0.6])   # a count, not a log
        self.assertEqual(self.ok({"tokens": score(legend)})["answers"]["tokens"]["probabilities"], a["tokens"]["probabilities"])

    def test_model_is_p_best_with_the_eu_argmax_as_choice(self):
        out = self.ok({"model": choice(CLAUDE[1:3])}, state={"task": "refactor auth", "harness": "claude-code"})
        a = out["answers"]["model"]
        self.assertEqual((a["choice"], a["confidence"]), ("claude-code:sonnet", 0.45))           # argmax EU, not argmax P(best)
        self.assertEqual(a["probabilities"], {"claude-code:sonnet": 0.45, "claude-code:opus": 0.55})
        seats, dctx = self.d_calls[-1]
        self.assertEqual([s["id"] for s in seats], CLAUDE[1:3])
        self.assertEqual((dctx["mode"], dctx["text"], len(dctx["decision_id"])), ("skill", "refactor auth", 8))
        self.assertEqual(self.ok({"route": choice(CLAUDE[1:3])}, state={"task": "refactor auth", "harness": "claude-code"})
                         ["answers"]["route"]["choice"], "claude-code:sonnet")

    def test_a_seat_failing_the_hard_filters_resolves_but_gets_zero(self):
        loc = "local:this-machine/qwen3:14b"
        self.mods["maestro.decide"].p_best = lambda ranked, did, n=64: {o["seat"]: 1.0 / len(ranked) for o in ranked}
        a = self.ok({"model": choice(CLAUDE[1:3] + [loc])}, inv=inventory(local_models=[helpers.local("qwen3:14b", 14)]))["answers"]["model"]
        self.assertEqual([s["id"] for s in self.d_calls[-1][0]], CLAUDE[1:3])                # a chat-only model for a repo job
        self.assertEqual(a["probabilities"][loc], 0.0)
        a = self.ok({"model": choice(CLAUDE[1:3] + [loc])}, state="what is a monad?", inv=inventory(local_models=[helpers.local("qwen3:14b", 14)]))
        self.assertIn(loc, [s["id"] for s in self.d_calls[-1][0]])

    def test_model_keys_must_all_resolve_to_seats(self):
        st, out = self.ask({"model": choice(["claude-code:opus", "gpt-9:ultra"])})
        self.assertEqual((st, out["detail"][0]["type"]), (422, "question_not_supported"))
        st, out = self.ask({"pick": choice(["claude-code:opus", "gpt-9:ultra"], intent="maestro.model")})
        self.assertEqual((st, out["detail"][0]["type"], out["detail"][0]["loc"]), (422, "intent_mismatch", ["body", "questions", "pick", "choice", "criteria"]))

    def test_models_json_ids_and_server_seats_resolve(self):
        self.mods["maestro.decide"].rank = lambda seats, post, ctx: [{"seat": s["id"], "p": 0.7, "eu": 1.0} for s in seats]
        self.mods["maestro.decide"].p_best = lambda ranked, did, n=64: {ranked[0]["seat"]: 1.0}
        a = self.ok({"model": choice(["openai/gpt-5"])}, models={"openai/gpt-5": {"id": "openai/gpt-5"}})["answers"]["model"]
        self.assertEqual((a["choice"], a["confidence"]), ("openai/gpt-5", 1.0))
        seen = []
        a = self.ok({"job": choice(["code", "debug"]), "model": choice(["groq/b"]), "needs_tools": {"type": "noul"}},
                    seats=lambda names, task: seen.append(names) or {"groq/b": {"m": {}, "id": "groq/b"}})
        self.assertEqual((seen, a["answers"]["model"]["choice"]), ([["groq/b"]], "groq/b"))      # only the model question asks for seats

    def test_no_seat_qualifies_is_a_422(self):
        self.mods["maestro.decide"].rank = lambda seats, post, ctx: []
        st, out = self.ask({"model": choice(CLAUDE)})
        self.assertEqual((st, out["detail"][0]["type"]), (422, "no_seat_qualifies"))

    def test_success_is_p_of_the_named_seat(self):
        a = self.ok({"success": {"type": "noul", "criteria": {"true": "claude-code:opus"}}})["answers"]["success"]
        self.assertEqual(a, {"type": "noul", "noul": 0.9})
        self.assertEqual(self.ok({"success": {"type": "noul", "criteria": {"true": "claude-code:fable"}}})["answers"]["success"]["noul"], 0.0)

    def test_explicit_intent_on_any_name(self):
        out = self.ok({"how_hard": score("abcd", intent="maestro.difficulty")})
        self.assertEqual(out["answers"]["how_hard"], self.ok({"difficulty": score("abcd")})["answers"]["difficulty"])
        st, out = self.ask({"q": {"type": "noul", "intent": "maestro.bogus"}})
        self.assertEqual((st, out["detail"][0]["type"]), (422, "unknown_intent"))
        st, out = self.ask({"q": score("abcdefg", intent="maestro.difficulty")})                 # 7 levels: not the shape
        self.assertEqual(out["detail"][0]["type"], "intent_mismatch")

    def test_no_similarity_matching(self):
        for name, q in (("how_hard", score(["trivial", "routine", "hard", "frontier"], instructions="How difficult is this task?")),
                        ("difficulty_level", score("abcd")), ("Difficulty", score("abcd")), ("scope", score("abcd")),
                        ("job", choice(["debug", "poetry"])), ("stakes", score("abcde")), ("needs_tools", score("ab")),
                        ("x", {"type": "noul", "intent": "jev.difficulty"})):
            st, out = self.ask({name: q})
            self.assertEqual((st, out["detail"][0]["type"]), (422, "question_not_supported"), name)

    def test_ood_tempers_toward_uniform(self):
        self.mods["maestro.s1"].posterior = lambda task, cls=None, **k: dict(POST, ood=0.9)
        a = self.ok({"job": choice(["debug", "code"]), "needs_tools": {"type": "noul"}})["answers"]
        w = [0.75 ** (1 / 1.9), 0.2 ** (1 / 1.9)]
        self.assertAlmostEqual(a["job"]["probabilities"]["debug"], w[0] / sum(w), places=4)
        y, n = 0.9 ** (1 / 1.9), 0.1 ** (1 / 1.9)
        self.assertAlmostEqual(a["needs_tools"]["noul"], y / (y + n), places=4)

    def test_missing_engine_is_a_503(self):
        with mock.patch.dict(sys.modules, {"maestro.s1": None}):
            st, out = self.ask({"job": choice(R.JOBS)})
        self.assertEqual((st, out["detail"][0]["type"]), (503, "engine_unavailable"))

    def test_native_latency(self):
        qs = {"job": choice(R.JOBS), "difficulty": score("abcd"), "stakes": score("abc"), "needs_tools": {"type": "noul"},
              "tokens": score(["<100", "100-300", "300-1k", "1k-3k", ">3k"]), "model": choice(CLAUDE)}
        self.ok(qs)
        ms = []
        for _ in range(50):
            t0 = time.perf_counter()
            self.ok(qs)
            ms.append((time.perf_counter() - t0) * 1000)
        ms.sort()
        self.assertLess(ms[len(ms) // 2], 25.0)   # stubbed engine: systemone's own overhead, rules classifier and seat resolution


# ---- delegation ----

def label_dist(text):
    """The fake model's answer logprobs by prompt: the probe is A; questions name their answer as 'ANSWER=<label>' in the state."""
    if text.startswith("Water boils"):
        return {"A": -0.02, " B": -4.5}
    want = text.split("ANSWER=")[1].split()[0] if "ANSWER=" in text else "A"
    return {want: -0.3, " " + want.lower(): -2.5, "The": -1.8, "B" if want != "B" else "A": -3.0}


def delegate_server(mode="plain", dist=label_dist, votes="A A B A A A B A", models=("qwen3",), loaded=True, status=200, err=None,
                    delay=0.0, track=None):
    def handler(req):
        if req["method"] == "GET":
            if req["path"] == "/v1/models":
                return 200, {}, {"object": "list", "data": [{"id": m} for m in models]}
            if req["path"] in ("/api/tags", "/api/ps"):
                return 200, {}, {"models": [{"name": m} for m in models] if loaded or req["path"] == "/api/tags" else []}
            return 404, {}, {"error": "not found"}
        if track is not None:
            with track["lock"]:
                track["now"] += 1
                track["max"] = max(track["max"], track["now"])
        try:
            time.sleep(delay)
            if status != 200:
                return status, {}, {"error": {"message": err or "boom"}}
            b = req["json"]
            text = b["messages"][0]["content"]
            if mode == "votes" or (mode == "thinking" and "n" in b):
                if "n" not in b:
                    return 200, {}, {"choices": [{"index": 0, "message": {"role": "assistant", "content": "A"}, "logprobs": None}]}
                vs = votes.split()
                return 200, {}, {"choices": [{"index": i, "message": {"role": "assistant", "content": vs[i % len(vs)] + "."}} for i in range(b["n"])],
                                 "usage": {"prompt_tokens": 40, "completion_tokens": 16}}
            d = dist(text) if mode == "plain" or b.get("continue_final_message") else {"<think>": -0.01, "Okay": -5.0}
            first = max(d, key=d.get)
            return 200, {}, {"choices": [{"index": 0, "message": {"role": "assistant", "content": first}, "finish_reason": "length",
                                          "logprobs": {"content": [{"token": first, "logprob": d[first],
                                                                    "top_logprobs": [{"token": t, "logprob": v} for t, v in d.items()]}]}}],
                             "usage": {"prompt_tokens": 50, "completion_tokens": 1}}
        finally:
            if track is not None:
                with track["lock"]:
                    track["now"] -= 1
    return FakeServer(handler)


class Delegation(Base):
    def setUp(self):
        Base.setUp(self)
        self.srv = None

    def tearDown(self):
        if self.srv:
            self.srv.close()
        Base.tearDown(self)

    def serve(self, kind="openai", key_env=None, **kw):
        self.srv = delegate_server(**kw)
        ep = {"name": "spark", "url": self.srv.url, "kind": kind}
        if key_env:
            ep["api_key_env"] = key_env
        self.cfg = dict(self.cfg, systemone_delegate="local:spark/qwen3", local_endpoints=[ep])
        return self.srv

    def posts(self):
        return [r["json"] for r in self.srv.requests if r["method"] == "POST"]

    def llm(self, questions, state="triage this ticket", **kw):
        return self.ask(questions, state=state, model="maestro-s1+llm", **kw)

    def test_plain_logprobs(self):
        self.serve()
        st, out = self.llm({"tone": choice(["calm", "angry", "excited"], instructions="What is the tone?"),
                            "urgency": score(["can wait", "this week", "today"]), "spam": {"type": "noul", "instructions": "Is it spam?"}},
                           state="ANSWER=B please help")
        self.assertEqual(st, 200, out)
        probe, q1 = self.posts()[0], self.posts()[1]
        self.assertEqual((probe["max_tokens"], probe["logprobs"], probe["top_logprobs"], probe["temperature"], probe["chat_template_kwargs"]),
                         (1, True, 20, 0, {"enable_thinking": False}))
        self.assertTrue(probe["messages"][0]["content"].endswith("Answer with the letter only."))
        self.assertIn("A. calm\nB. angry\nC. excited", q1["messages"][0]["content"])
        a = out["answers"]
        pb, pB, pa = math.exp(-0.3), math.exp(-2.5), math.exp(-3.0)                # B and " b" summed; A from its own entry
        floor = math.exp(-3.0 - 2)                                                 # C missing: min logprob - 2
        tot = pb + pB + pa + floor
        self.assertEqual(a["tone"]["choice"], "angry")
        self.assertAlmostEqual(a["tone"]["probabilities"]["angry"], (pb + pB) / tot, places=4)
        self.assertAlmostEqual(a["tone"]["probabilities"]["excited"], floor / tot, places=4)
        self.assertAlmostEqual(out["maestro"]["mass"]["tone"], pb + pB + pa, places=3)
        self.assertEqual(a["urgency"]["type"], "score")
        self.assertEqual(out["maestro"]["source"], {"tone": "delegate", "urgency": "delegate", "spam": "delegate"})
        self.assertEqual((out["maestro"]["strategy"], out["usage"]), ("logprobs", {"input_tokens": 150, "output_tokens": 3}))
        cache = os.path.join(self.home, SO.CACHE_FILE)
        self.assertEqual(stat.S_IMODE(os.stat(cache).st_mode), 0o600)
        self.assertNotIn("please help", read(cache))
        n = len(self.posts())
        SO._CACHE.clear()                                            # a new process: the strategy and answers come from s1_cache.json
        st, again = self.llm({"tone": choice(["calm", "angry", "excited"], instructions="What is the tone?")}, state="ANSWER=B please help")
        self.assertEqual((len(self.posts()), again["answers"]["tone"], again["usage"]["input_tokens"]), (n, a["tone"], 0))

    def test_low_mass_lowers_confidence(self):
        self.serve(dist=lambda text: {"A": -0.02} if text.startswith("Water") else {"The": -0.1, "A": -2.0, "B": -3.0})
        st, out = self.llm({"q": choice(["x", "y"])})
        mass = math.exp(-2.0) + math.exp(-3.0)
        self.assertAlmostEqual(out["maestro"]["mass"]["q"], mass, places=3)
        self.assertAlmostEqual(out["answers"]["q"]["confidence"], round(math.exp(-2.0) / mass, 4) * mass, places=3)

    def test_thinking_model_uses_the_prefill_strategy(self):
        self.serve(mode="thinking")
        st, out = self.llm({"tone": choice(["calm", "angry"])}, state="ANSWER=B hi")
        self.assertEqual((st, out["answers"]["tone"]["choice"], out["maestro"]["strategy"]), (200, "angry", "prefill"))
        p = self.posts()
        self.assertIn("chat_template_kwargs", p[0])
        self.assertEqual((p[1]["continue_final_message"], p[1]["add_generation_prompt"], p[1]["messages"][-1]),
                         (True, False, {"role": "assistant", "content": "Answer:"}))
        self.assertTrue(p[2]["continue_final_message"])
        self.assertEqual(len(p), 3)

    def test_server_without_logprobs_falls_back_to_votes(self):
        self.serve(mode="votes")
        st, out = self.llm({"tone": choice(["calm", "angry"]), "spam": {"type": "noul"}})
        self.assertEqual((st, out["maestro"]["strategy"]), (200, "votes"))
        self.assertEqual(out["answers"]["tone"]["probabilities"], {"calm": 0.75, "angry": 0.25})
        last = self.posts()[-1]
        self.assertEqual((last["n"], last["temperature"], last["max_tokens"]), (8, 1, 4))
        self.assertEqual(out["maestro"]["mass"]["spam"], 0.0)                  # votes "A"/"B" are not Yes/No

    def test_no_strategy_passes(self):
        self.serve(mode="votes", votes="Hmm")
        st, out = self.llm({"tone": choice(["calm", "angry"])})
        self.assertEqual((st, out["detail"][0]["type"]), (422, "delegate_unsupported"))
        self.assertIsNone(SO.strategy_of("local:spark/qwen3", self.ctx()))

    def test_strategy_of_for_discover(self):
        self.serve(mode="thinking")
        self.assertEqual(SO.strategy_of("local:spark/qwen3", self.ctx()), "prefill")
        self.assertIsNone(SO.strategy_of("local:nowhere/x", self.ctx()))

    def test_native_and_delegated_in_one_request(self):
        self.serve()
        out = self.llm({"job": choice(R.JOBS), "scope": score("abcd")})[1]
        self.assertEqual((out["model"], out["maestro"]["source"]), ("maestro-s1+llm", {"job": "native", "scope": "delegate"}))
        out = self.ask({"job": choice(R.JOBS), "scope": score("abcd")}, model="jev-latest")[1]
        self.assertEqual(out["model"], "maestro-s1+llm")

    def test_no_delegate_configured(self):
        st, out = self.llm({"tone": choice(["calm", "angry"])})
        self.assertEqual((st, out["detail"][0]["type"]), (422, "question_not_supported"))
        self.assertIn("systemone_delegate", out["detail"][0]["msg"])

    def test_cloud_delegate_needs_the_opt_in(self):
        srv = self.serve()
        self.cfg = dict(self.cfg, systemone_delegate="openai/gpt-5-mini", providers={"openai": {"base_url": srv.url + "/v1"}})
        env = {"PATH": "", "OPENAI_API_KEY": "sk-openai-SECRET-9999"}
        st, out = self.llm({"tone": choice(["calm", "angry"])}, env=env)
        self.assertEqual((st, out["detail"][0]["type"], self.posts()), (422, "question_not_supported", []))
        self.cfg["systemone_cloud"] = True
        st, out = self.llm({"tone": choice(["calm", "angry"])}, state="ANSWER=B", env=env)
        self.assertEqual((st, out["answers"]["tone"]["choice"]), (200, "angry"))
        self.assertEqual(self.srv.requests[-1]["headers"]["authorization"], "Bearer sk-openai-SECRET-9999")
        self.assertEqual(self.posts()[-1]["model"], "gpt-5-mini")
        self.assertNotIn("SECRET", json.dumps(out))

    def test_installed_but_not_loaded_is_refused(self):
        self.serve(kind="ollama", loaded=False)
        st, out = self.llm({"tone": choice(["calm", "angry"])})
        self.assertEqual((st, out["detail"][0]["type"], self.posts()), (503, "delegate_unavailable", []))
        self.assertIn("never loads", out["detail"][0]["msg"])
        with open(os.path.join(self.home, SO.CACHE_FILE), "w") as f:    # a cached strategy: still checked before the first request
            json.dump({"version": 1, "strategy": {"local:spark/qwen3": "logprobs"}, "entries": {}}, f)
        SO._CACHE.clear()
        st, out = self.llm({"tone": choice(["calm", "angry"])})
        self.assertEqual((st, out["detail"][0]["type"], self.posts()), (503, "delegate_unavailable", []))

    def test_cache_hits_need_no_network(self):
        self.serve()
        q = {"tone": choice(["calm", "angry"])}
        self.llm(q)
        n = len(self.srv.requests)
        self.assertEqual(self.llm(q)[0], 200)
        self.assertEqual(len(self.srv.requests), n)

    def test_unreachable_delegate(self):
        self.cfg = dict(self.cfg, systemone_delegate="local:spark/qwen3", local_endpoints=[{"name": "spark", "url": "http://127.0.0.1:9", "kind": "openai"}])
        st, out = self.llm({"tone": choice(["calm", "angry"])})
        self.assertEqual((st, out["detail"][0]["type"]), (503, "delegate_unavailable"))

    def test_upstream_error_is_scrubbed_and_the_key_stays_private(self):
        self.serve(key_env="SPARK_KEY", status=500, err="bad key sk-spark-SECRET-1234 rejected")
        st, out = self.llm({"tone": choice(["calm", "angry"])}, models={"local:spark/qwen3": {"id": "local:spark/qwen3", "systemone_strategy": "logprobs"}},
                           env={"PATH": "", "SPARK_KEY": "sk-spark-SECRET-1234"})
        self.assertEqual((st, out["detail"][0]["type"]), (502, "delegate_error"))
        self.assertNotIn("SECRET", json.dumps(out))
        self.assertEqual(self.srv.requests[-1]["headers"]["authorization"], "Bearer sk-spark-SECRET-1234")

    def test_too_many_options(self):
        self.serve()
        st, out = self.llm({"q": choice(["o%d" % i for i in range(27)])})
        self.assertEqual((st, out["detail"][0]["type"]), (422, "too_many_options"))

    def test_at_most_two_concurrent_requests(self):
        track = {"lock": threading.Lock(), "now": 0, "max": 0}
        self.serve(delay=0.15, track=track)
        self.llm({"warm": {"type": "noul"}})                                   # probe once
        res = []
        ts = [threading.Thread(target=lambda i=i: res.append(self.llm({"q%d" % i: {"type": "noul"}})[0])) for i in range(4)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual((res, track["max"]), ([200] * 4, 2))

    def test_calibration_fits_a_temperature_per_type(self):
        self.serve()
        rows = os.path.join(self.home, "typed.jsonl")
        with open(rows, "w") as f:
            for i in range(6):          # the model says B at ~0.8 but is right only half the time: T > 1
                f.write(json.dumps({"state": "ANSWER=B item %d" % i, "question": choice(["x", "y"]), "answer": "y" if i % 2 else "x"}) + "\n")
        temps = SO.calibrate(rows, self.ctx())
        self.assertGreater(temps["choice"], 1.5)
        saved = json.loads(read(os.path.join(self.home, SO.CALIB_FILE)))
        self.assertEqual(saved["temps"]["local:spark/qwen3"], temps)
        st, out = self.llm({"q": choice(["x", "y"])}, state="ANSWER=B item 0")
        self.assertLess(out["answers"]["q"]["probabilities"]["y"], 0.7)


class JevContract(Base):
    """Our endpoint against typesafe-sdk's generated models (skipped if not installed), and Maestro's own Jev client pointed at it."""

    @classmethod
    def setUpClass(cls):
        try:
            from typesafe_sdk._schemas import models
        except ImportError:
            raise unittest.SkipTest("typesafe-sdk not installed")
        cls.models = models

    def test_jev_request_and_response_validate(self):
        srv = delegate_server()
        self.addCleanup(srv.close)
        self.cfg = dict(self.cfg, systemone_delegate="local:spark/qwen3", local_endpoints=[{"name": "spark", "url": srv.url, "kind": "openai"}])
        cands = R.build_candidates(helpers.cls(), inventory(), "claude-code", {})
        body = dict(R.JevClassifier("k").build_request("refactor the payment module", cands), model="jev-latest")
        body["questions"]["spam"] = {"type": "noul", "instructions": "Is it spam?", "criteria": {"true": "spam", "false": "ham"}}
        self.models.SystemOneRequest.model_validate(body)
        st, out = SO.handle(body, self.ctx())
        self.assertEqual(st, 200, out)
        resp = self.models.SystemOneResponse.model_validate(out)
        self.assertEqual(set(resp.answers), set(body["questions"]))
        self.assertEqual(out["maestro"]["source"]["scope"], "delegate")
        self.assertEqual({k for k, v in out["maestro"]["source"].items() if v == "native"}, {"model", "job", "reasoning", "ambiguity", "risk"})
        for k, v in out["answers"].items():
            self.assertEqual(v["type"], body["questions"][k]["type"])
        st, err = SO.handle({"model": "maestro-s1", "state": "x", "questions": {"q": {"type": "rank"}}}, self.ctx())
        self.models.HTTPValidationError.model_validate(err)
        self.models.ModelMetadataList.model_validate({"models": SO.MODEL_LIST})

    def test_maestros_jev_client_works_against_our_endpoint(self):
        srv = delegate_server()
        self.addCleanup(srv.close)
        self.cfg = dict(self.cfg, systemone_delegate="local:spark/qwen3", local_endpoints=[{"name": "spark", "url": srv.url, "kind": "openai"}])
        ctx = self.ctx()

        def fetch(url, headers, body, timeout):
            st, out = SO.handle(json.loads(body), ctx)
            return st, json.dumps(out).encode()
        cands = R.build_candidates(helpers.cls(), inventory(), "claude-code", {})
        c = R.JevClassifier("k", fetch=fetch).classify("refactor the payment module", cands)
        self.assertEqual(c.classifier, "jev", c.note)
        self.assertEqual(c.jev["choice"], "claude-code:sonnet")
        self.assertIn(c.job, R.JOBS)


class TypedEval(Base):
    def test_build_and_run(self):
        sys.path.insert(0, os.path.join(helpers.ROOT, "evals"))
        import typed_eval as TE
        d = self.home
        with open(os.path.join(d, "items.jsonl"), "w") as f:
            f.writelines(json.dumps({"id": "c%d" % i, "prompt": "ANSWER=A fix the failing test %d" % i}) + "\n" for i in range(4))
        with open(os.path.join(d, "labels.jsonl"), "w") as f:
            f.writelines(json.dumps({"id": "c%d" % i, "split": "test" if i < 3 else "train", "job": {"debug": 0.8, "code": 0.2},
                                     "difficulty": [0.1, 0.6, 0.2, 0.1], "needs_tools": 0.9, "self_contained": 0}) + "\n" for i in range(4))
        with open(os.path.join(d, "outcomes.jsonl"), "w") as f:
            f.write(json.dumps({"id": "c0", "split": "test", "t_star": "opus"}) + "\n")
        out = os.path.join(d, "typed.jsonl")
        args = argparse.Namespace(labels=os.path.join(d, "labels.jsonl"), items=os.path.join(d, "items.jsonl"),
                                  outcomes=os.path.join(d, "outcomes.jsonl"), split="test", out=out)
        with redirect_stdout(io.StringIO()):
            rows = TE.build(args)
        self.assertEqual(len(rows), 13)
        self.assertEqual({(r["kind"], json.dumps(r["answer"])) for r in rows if r["id"] == "c0"},
                         {("job", '"debug"'), ("difficulty", "1"), ("needs_tools", "1"), ("self_contained", "0"), ("model", '"claude-code:opus"')})
        srv = delegate_server()
        self.addCleanup(srv.close)
        self.cfg = dict(self.cfg, systemone_delegate="local:spark/qwen3", local_endpoints=[{"name": "spark", "url": srv.url, "kind": "openai"}])
        with redirect_stdout(io.StringIO()):
            rep = TE.run(argparse.Namespace(file=out, limit=0, delegate=True, jev=False, json=True), self.ctx())
        self.assertEqual(set(rep), {"s1", "delegate"})
        self.assertEqual(rep["s1"]["job"]["accuracy"], 1.0)
        self.assertEqual((rep["s1"]["needs_tools"]["accuracy"], rep["s1"]["self_contained"]["accuracy"]), (1.0, 1.0))
        self.assertEqual(rep["delegate"]["job"]["n"], 3)
        self.assertEqual(rep["delegate"]["needs_tools"]["errors"], 0)


class Cli(Base):
    def test_decide_reads_a_file(self):
        path = os.path.join(self.home, "req.json")
        with open(path, "w") as f:
            json.dump({"model": "maestro-s1", "state": "x", "questions": {"needs_tools": {"type": "noul"}}}, f)
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"MAESTRO_HOME": self.home}), redirect_stdout(buf):
            rc = SO.cmd_decide(argparse.Namespace(file=path, calibrate=None))
        self.assertEqual((rc, json.loads(buf.getvalue())["answers"]["needs_tools"]["noul"]), (0, 0.9))
        with mock.patch.dict(os.environ, {"MAESTRO_HOME": self.home}), redirect_stdout(io.StringIO()):
            self.assertEqual(SO.cmd_decide(argparse.Namespace(file=path + ".missing", calibrate=None)), 2)

    def test_route_imports_neither_systemone_nor_learn(self):
        code = ("import sys; sys.path.insert(0, %r); import maestro; maestro.route('fix the failing test', harness='claude-code', "
                "inventory={'version': 2, 'generated_at': 9e12, 'harnesses': {}, 'api_keys': {}, 'local_endpoints': [], 'local_models': [], "
                "'endpoints_sig': [], 'hardware': {}, 'warnings': []}, config=maestro.load_config(%r)); "
                "print([m for m in ('maestro.systemone', 'maestro.learn') if m in sys.modules])") % (
            os.path.join(helpers.ROOT, "skills", "maestro", "scripts"), self.home)
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env=dict(os.environ, MAESTRO_HOME=self.home))
        self.assertEqual(res.stdout.strip(), "[]", res.stderr)


if __name__ == "__main__":
    unittest.main()
