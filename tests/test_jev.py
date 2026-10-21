import json
import os
import tempfile
import unittest
from collections import OrderedDict

from helpers import cls, inventory, local, maestro as m

DIMS = ("reasoning", "scope", "ambiguity", "risk")


def canned(choice="claude-code:opus", conf=0.9, klass="code", scores=(2.0, 1.0, 1.0, 1.0),
           probs=None):
    answers = {
        "model": {"type": "choice", "choice": choice, "confidence": conf,
                  "probabilities": probs or {choice: conf}},
        "job": {"type": "choice", "choice": klass, "confidence": 0.8, "probabilities": {klass: 0.8}},
    }
    for name, score in zip(DIMS, scores):
        answers[name] = {"type": "score", "score": score, "confidence": 0.7,
                         "legend": {"0": "a", "1": "b", "2": "c", "3": "d"}, "probabilities": {"2": 1.0}}
    return {"model": "jev-latest", "answers": answers, "usage": {"input_tokens": 120, "output_tokens": 8}}


class Fake:
    """Injectable fetch(url, headers, body, timeout) -> (status, bytes) that records calls."""

    def __init__(self, status=200, payload=None, exc=None, raw=None):
        self.status, self.exc, self.raw = status, exc, raw
        self.payload = payload if payload is not None else canned()
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append((url, headers, json.loads(body), timeout))
        if self.exc:
            raise self.exc
        return self.status, self.raw if self.raw is not None else json.dumps(self.payload).encode()


def claude_candidates(**kw):
    return m.build_candidates(cls(), inventory(), "claude-code", kw)


def jev(fake, cache=None):
    return m.JevClassifier("test-key", fetch=fake, cache=cache)


class JevClient(unittest.TestCase):
    def setUp(self):
        self.cands = claude_candidates()

    def test_request_shape(self):
        fake = Fake()
        c = jev(fake).classify("Design a multi-region failover architecture", self.cands)
        url, headers, body, timeout = fake.calls[0]
        self.assertEqual(url, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(timeout, 2.5)
        self.assertEqual(body["model"], "jev-latest")
        self.assertEqual(set(body["questions"]), {"model", "job", *DIMS})
        q = body["questions"]
        self.assertEqual(q["model"]["type"], "choice")
        self.assertEqual(list(q["model"]["criteria"]),
                         ["claude-code:haiku", "claude-code:sonnet", "claude-code:opus", "claude-code:fable"])
        self.assertEqual(q["model"]["criteria"]["claude-code:opus"],
                         "strong tier, Claude Code (opus) subagent, subscription (flat cost), full tools/repo access")
        self.assertIn("cheapest/fastest", q["model"]["instructions"])
        for d in DIMS:
            self.assertEqual((q[d]["type"], len(q[d]["criteria"])), ("score", 4))
        self.assertEqual(set(q["job"]["criteria"]), set(m.JOBS))
        self.assertEqual(len(q["job"]["criteria"]), 11)
        self.assertIn("SQL", q["job"]["criteria"]["data"])
        self.assertIn("Which job", q["job"]["instructions"])
        self.assertNotIn("task_class", q)
        self.assertEqual(set(body["state"]), {"task", "signals"})
        self.assertEqual((c.classifier, c.jev["choice"]), ("jev", "claude-code:opus"))

    def test_objective_changes_the_instruction(self):
        texts = set()
        for obj in ("balanced", "cost", "quality"):
            req = jev(Fake()).build_request("t", self.cands, obj)
            texts.add(req["questions"]["model"]["instructions"])
        self.assertEqual(len(texts), 3)

    def test_task_is_truncated_to_4000_chars(self):
        fake = Fake()
        jev(fake).classify("x" * 9000, self.cands)
        self.assertEqual(len(fake.calls[0][2]["state"]["task"]), 4000)

    def test_answers_map_to_classification(self):
        payload = canned(klass="plan", scores=(3.0, 1.0, 2.0, 2.0), probs={"claude-code:opus": 0.9, "claude-code:sonnet": 0.1})
        c = jev(Fake(payload=payload)).classify("t", self.cands)
        self.assertEqual((c.job, c.difficulty), ("plan", 3))  # 3*.6 + 2*.4 = 2.6
        self.assertEqual(c.jev["probabilities"]["claude-code:sonnet"], 0.1)
        self.assertEqual(c.jev["complexity"]["reasoning"], 3.0)
        self.assertAlmostEqual(c.confidence, 0.9)

    def test_complexity_formula(self):
        self.assertEqual(m.complexity(0, 0, 0, 0), 0)
        self.assertEqual(m.complexity(3, 3, 3, 3), 3)
        self.assertEqual(m.complexity(2, 1, 0, 0), 1)       # 1.2
        self.assertEqual(m.complexity(2, 1, 2, 2), 2)       # 1.2 + 0.8 = 2.0
        self.assertEqual(m.complexity(3, 0, 3, 3), 3)       # 1.8 + 1.2 = 3.0
        self.assertEqual(m.complexity(1, 1, 3, 3), 2)       # 0.6 + 1.2 = 1.8
        self.assertEqual(m.complexity(9, 9, 9, 9), 3)       # clamped

    def _assert_falls_back(self, fake, needle):
        c = jev(fake).classify("fix the failing test in auth.py", self.cands)
        self.assertEqual(c.classifier, "rules")
        self.assertIsNone(c.jev)
        self.assertEqual(c.job, "debug")
        self.assertIn(needle, c.note)

    def test_non_200_falls_back(self):
        self._assert_falls_back(Fake(status=503), "HTTP 503")

    def test_bad_json_falls_back(self):
        self._assert_falls_back(Fake(raw=b"<html>oops"), "bad response")

    def test_missing_answers_falls_back(self):
        self._assert_falls_back(Fake(payload={"model": "x"}), "bad response")

    def test_unknown_model_choice_falls_back(self):
        self._assert_falls_back(Fake(payload=canned(choice="gpt-9:ultra")), "unknown model choice")

    def test_unknown_job_falls_back(self):
        self._assert_falls_back(Fake(payload=canned(klass="poetry")), "unknown job")

    def test_old_task_class_answers_are_rejected(self):
        payload = canned()
        payload["answers"]["task_class"] = payload["answers"].pop("job")
        self._assert_falls_back(Fake(payload=payload), "bad response")

    def test_timeout_exception_falls_back(self):
        self._assert_falls_back(Fake(exc=TimeoutError("timed out")), "TimeoutError")

    def test_cache_hit_avoids_second_fetch(self):
        fake, cache = Fake(), OrderedDict()
        j = jev(fake, cache)
        first = j.classify("same task", self.cands)
        second = j.classify("same task", self.cands)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(first.jev["choice"], second.jev["choice"])
        self.assertTrue(second.jev["cached"])

    def test_cache_key_covers_candidates_and_objective(self):
        fake, cache = Fake(), OrderedDict()
        j = jev(fake, cache)
        j.classify("same task", self.cands)
        j.classify("same task", self.cands[:3])
        j.classify("same task", self.cands, "quality")
        self.assertEqual(len(fake.calls), 3)
        j.classify("same task", list(reversed(self.cands)))  # order does not matter
        self.assertEqual(len(fake.calls), 3)

    def test_failures_are_not_cached(self):
        cache = OrderedDict()
        jev(Fake(status=500), cache).classify("t", self.cands)
        self.assertEqual(len(cache), 0)

    def test_cache_is_an_lru_of_512(self):
        cache = OrderedDict()
        j = jev(Fake(), cache)
        for i in range(520):
            j.classify("task %d" % i, self.cands)
        self.assertEqual(len(cache), 512)


class Candidates(unittest.TestCase):
    def test_names_and_descriptions(self):
        inv = inventory(clis=("codex",), local_models=[local("qwen3:14b", 14)])
        c = m.build_candidates(cls("ask", 0), inv, "claude-code", {})
        names = [x["name"] for x in c]
        self.assertIn("claude-code:sonnet", names)
        self.assertIn("local:this-machine/qwen3:14b", names)
        self.assertIn("codex:gpt-6.1-sol", names)
        self.assertEqual(len(set(names)), len(names))
        desc = {x["name"]: x["description"] for x in c}
        self.assertEqual(desc["local:this-machine/qwen3:14b"],
                         "local model on this-machine (14B), free and private, serves up to fast-level work, no repo tools")

    def test_model_notes_override(self):
        c = claude_candidates(model_notes={"claude-code:opus": "my best model"})
        self.assertEqual({x["name"]: x["description"] for x in c}["claude-code:opus"], "my best model")

    def test_hard_filters(self):
        inv = inventory(local_models=[local("qwen3:14b", 14)])
        loc = "local:this-machine/qwen3:14b"
        names = lambda c, pol=None: {x["name"] for x in m.build_candidates(c, inv, "claude-code", pol or {})}
        self.assertIn(loc, names(cls("ask", 0)))
        self.assertNotIn(loc, names(cls("ask", 0, needs=("tools",))))
        self.assertNotIn(loc, names(cls("ask", 0, needs=("vision",))))
        self.assertNotIn(loc, names(cls("ask", 0, needs=("long_context",))))
        self.assertNotIn(loc, names(cls("code", 1)))
        self.assertNotIn("claude-code:fable", names(cls("ask", 0), {"max_tier": "strong"}))
        self.assertNotIn(loc, names(cls("ask", 0), {"prefer_local": False}))

    def test_current_harness_is_not_also_a_cli_candidate(self):
        inv = inventory(clis=("claude-code", "codex"))
        names = [x["name"] for x in m.build_candidates(cls(), inv, "claude-code", {})]
        self.assertEqual(names.count("claude-code:opus"), 1)
        self.assertEqual([x["kind"] for x in m.build_candidates(cls(), inv, "claude-code", {})
                          if x["name"] == "claude-code:opus"], ["subagent"])


class ChooseJev(unittest.TestCase):
    """Decision logic: trust a confident pick, otherwise the stronger of pick and ladder."""

    def setUp(self):
        self.inv = inventory()
        self.cands = m.build_candidates(cls(), self.inv, "claude-code", {})

    def decide(self, choice, conf, klass="code", c=1, policy=None, current_tier=None):
        k = m.Classification(klass, c, conf, ())
        k.jev = {"choice": choice, "confidence": conf}
        return m.choose_jev(k, self.cands, self.inv, "claude-code", policy or {}, current_tier)

    def test_confident_pick_is_taken_even_below_the_ladder(self):
        s = self.decide("claude-code:haiku", 0.9, "plan", 3)
        self.assertEqual((s["model"], s["target"], s["how"]), ("haiku", "fast", "jev"))

    def test_confident_pick_is_taken_even_above_the_ladder(self):
        s = self.decide("claude-code:fable", 0.9, "ask", 0)
        self.assertEqual((s["model"], s["target"]), ("fable", "frontier"))

    def test_low_confidence_escalates_to_the_ladder_tier(self):
        s = self.decide("claude-code:haiku", 0.3, "code", 3)  # ladder: strong
        self.assertEqual((s["model"], s["target"], s["how"]), ("opus", "strong", "ladder"))

    def test_low_confidence_keeps_a_pick_stronger_than_the_ladder(self):
        s = self.decide("claude-code:fable", 0.3, "ask", 0)  # ladder: fast
        self.assertEqual((s["model"], s["how"]), ("fable", "jev"))

    def test_threshold_is_configurable(self):
        s = self.decide("claude-code:haiku", 0.6, "code", 3, {"jev_min_confidence": 0.7})
        self.assertEqual(s["model"], "opus")
        s = self.decide("claude-code:haiku", 0.6, "code", 3, {"jev_min_confidence": 0.5})
        self.assertEqual(s["model"], "haiku")

    def test_max_tier_caps_the_ladder(self):
        s = self.decide("claude-code:haiku", 0.2, "plan", 3, {"max_tier": "strong"})
        self.assertEqual(s["model"], "opus")

    def test_pick_at_the_current_tier_is_inline(self):
        s = self.decide("claude-code:sonnet", 0.9, current_tier="standard")
        self.assertEqual(s["delegate"], {"kind": "inline"})

    def test_local_pick_for_a_repo_job_goes_to_the_ladder(self):
        inv = inventory(local_models=[local("qwen3:14b", 14)])
        cands = m.build_candidates(cls("ask", 0), inv, "claude-code", {})
        k = m.Classification("code", 3, 0.95, ())
        k.jev = {"choice": "local:this-machine/qwen3:14b", "confidence": 0.95}
        s = m.choose_jev(k, cands, inv, "claude-code", {}, None)
        self.assertEqual((s["kind"], s["model"]), ("subagent", "opus"))

    def test_confident_local_pick_for_a_chat_job_is_taken(self):
        inv = inventory(local_models=[local("qwen3:14b", 14)])
        cands = m.build_candidates(cls("ask", 0), inv, "claude-code", {})
        k = m.Classification("ask", 1, 0.95, ())
        k.jev = {"choice": "local:this-machine/qwen3:14b", "confidence": 0.95}
        s = m.choose_jev(k, cands, inv, "claude-code", {}, None)
        self.assertEqual((s["kind"], s["category"], s["how"]), ("local", "local", "jev"))
        self.assertEqual(s["delegate"]["kind"], "local")


class JevRoute(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = self.tmp.name
        self.env = {"MAESTRO_HOME": self.home, "TYPESAFE_API_KEY": "k", "PATH": ""}

    def tearDown(self):
        self.tmp.cleanup()

    def go(self, fake, text="refactor the payment module", **kw):
        kw.setdefault("harness", "claude-code")
        kw.setdefault("inventory", inventory())
        return m.route(text, env=self.env, home=self.home, jev_fetch=fake, **kw)

    def test_auto_uses_jev_when_key_is_set_and_records_it(self):
        fake = Fake(payload=canned("claude-code:opus", 0.9, "code", (2, 2, 1, 1)))
        d = self.go(fake, explain=True)
        self.assertEqual((d["classifier"], d["model"], d["category"]), ("jev", "opus", "strong"))
        self.assertEqual(d["job"], "code")
        self.assertEqual(d["jev"]["choice"], "claude-code:opus")
        self.assertEqual(d["jev"]["complexity"]["value"], d["difficulty"])
        self.assertIn("candidates", d["explain"])
        self.assertTrue(os.path.exists(os.path.join(self.home, "cache.json")))
        self.go(fake)
        self.assertEqual(len(fake.calls), 1)  # second route served from cache

    def test_auto_without_key_uses_rules_silently(self):
        del self.env["TYPESAFE_API_KEY"]
        fake = Fake()
        d = self.go(fake)
        self.assertEqual((d["classifier"], len(fake.calls)), ("rules", 0))
        self.assertNotIn("TYPESAFE", d["reason"])
        self.assertNotIn("jev", d)

    def test_explicit_rules_ignores_the_key(self):
        fake = Fake()
        d = self.go(fake, classifier="rules")
        self.assertEqual((d["classifier"], len(fake.calls)), ("rules", 0))

    def test_explicit_jev_without_key_says_why(self):
        del self.env["TYPESAFE_API_KEY"]
        d = self.go(Fake(), classifier="jev")
        self.assertEqual(d["classifier"], "rules")
        self.assertIn("TYPESAFE_API_KEY", d["reason"])

    def test_single_candidate_skips_the_call(self):
        fake = Fake()
        d = self.go(fake, harness="codex", inventory=inventory(clis=("codex",)), config=dict(m.load_config(self.home), max_tier="fast"))
        self.assertEqual(len(fake.calls), 0)
        self.assertEqual(d["classifier"], "rules")
        self.assertIn("one eligible", d["reason"])

    def test_env_classifier_and_config_classifier(self):
        fake = Fake()
        self.env["MAESTRO_CLASSIFIER"] = "rules"
        self.assertEqual(self.go(fake)["classifier"], "rules")
        del self.env["MAESTRO_CLASSIFIER"]
        self.assertEqual(self.go(fake, config=dict(m.load_config(self.home), classifier="rules"))["classifier"], "rules")
        self.assertEqual(len(fake.calls), 0)

    def test_failure_falls_back_to_rules_with_reason(self):
        d = self.go(Fake(exc=TimeoutError("slow")))
        self.assertEqual(d["classifier"], "rules")
        self.assertIn("jev failed", d["reason"])
        self.assertIn("TimeoutError", d["reason"])

    def test_low_confidence_uses_the_ladder(self):
        fake = Fake(payload=canned("claude-code:haiku", 0.2, "plan", (3, 3, 3, 3)))
        d = self.go(fake, text="plan the roadmap")
        self.assertEqual((d["model"], d["category"]), ("fable", "frontier"))
        self.assertIn("ladder", d["reason"])


class JevContract(unittest.TestCase):
    """Validate our wire format against typesafe-sdk's generated models (skipped if not installed)."""

    @classmethod
    def setUpClass(cls):
        try:
            from typesafe_sdk._schemas import models
        except ImportError:
            raise unittest.SkipTest("typesafe-sdk not installed")
        cls.models = models

    @staticmethod
    def inner(x):
        return getattr(x, "root", x)  # Question / Answer are union wrappers

    def test_request_body_validates(self):
        body = m.JevClassifier("k").build_request("fix the failing test in auth.py", claude_candidates())
        req = self.models.SystemOneRequest.model_validate(body)
        self.assertEqual(req.model, "jev-latest")
        kinds = {k: self.inner(v).type for k, v in req.questions.items()}
        self.assertEqual(kinds, {"model": "choice", "job": "choice", **{d: "score" for d in DIMS}})
        self.assertEqual(len(self.inner(req.questions["model"]).criteria), 4)

    def test_request_with_local_and_specialist_candidates_validates(self):
        inv = inventory(local_models=[local("gpt-oss:120b", 120, endpoint="spark"), local("sqlcoder:15b", 15)])
        cands = m.build_candidates(cls("data", 1), inv, "claude-code", {})
        body = m.JevClassifier("k").build_request("sql for weekly active users", cands)
        req = self.models.SystemOneRequest.model_validate(body)
        criteria = self.inner(req.questions["model"]).criteria
        self.assertIn("local:spark/gpt-oss:120b", criteria)
        self.assertEqual(len(self.inner(req.questions["job"]).criteria), 11)

    def test_request_without_candidates_validates(self):
        body = m.JevClassifier("k").build_request("hello")
        self.models.SystemOneRequest.model_validate(body)
        self.assertNotIn("model", body["questions"])

    def test_canned_response_parses_and_round_trips_through_our_client(self):
        resp = self.models.SystemOneResponse.model_validate(canned())
        model_ans = self.inner(resp.answers["model"])
        self.assertEqual(model_ans.choice, "claude-code:opus")
        self.assertAlmostEqual(self.inner(resp.answers["reasoning"]).score, 2.0)
        c = jev(Fake(payload=canned())).classify("t", claude_candidates())
        self.assertEqual(c.jev["choice"], model_ans.choice)


if __name__ == "__main__":
    unittest.main()
