import json
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

import helpers
from helpers import maestro as m

sys.path.insert(0, os.path.join(helpers.ROOT, "evals"))
import train as evaltrain  # noqa: E402

CASES = [
    ("hi there", "ask"),
    ("thanks!", "ask"),
    ("what is a mutex?", "ask"),
    ("what's the difference between a process and a thread", "ask"),
    ("Summarize this article in three bullet points: <article text>", "summarize"),
    ("Extract all URLs from the following text and return them as JSON", "summarize"),
    ("Translate this paragraph to Spanish", "summarize"),
    ("Draft a short reply to this email declining the meeting", "write"),
    ("write a readme for the billing service", "write"),
    ("Create a checklist for onboarding a new engineer", "write"),
    ("write the announcement for the new pricing page", "write"),
    ("write a commit message for these changes", "edit"),
    ("add a docstring to parse_config", "edit"),
    ("fix the typo in the readme title", "edit"),
    ("rename fooBar to foo_bar everywhere in utils.py", "edit"),
    ("Implement a function that returns the nth Fibonacci number in Python", "code"),
    ("Refactor the payment module to remove the circular import", "code"),
    ("add pagination to the /users endpoint", "code"),
    ("add unit tests for the date parser", "code"),
    ("fix the failing test in auth.py", "debug"),
    ("why's the checkout test flaky", "debug"),
    ("Debug this stack trace: TypeError: 'NoneType' object is not subscriptable at handlers.py line 88", "debug"),
    ("p99 doubled since last deploy, find the regression", "debug"),
    ("review this PR for security issues", "review"),
    ("can you look over my diff before I merge", "review"),
    ("audit the auth middleware for vulnerabilities", "review"),
    ("SSH into the server and restart the nginx service", "ops"),
    ("Deploy the build to staging, run smoke tests, and roll back if they fail", "ops"),
    ("bump deps and fix whatever breaks", "ops"),
    ("set up CI for this repo and make it pass", "ops"),
    ("Research the latest open-source options for vector databases and compare them with citations", "research"),
    ("Find sources on the effect of remote work on productivity", "research"),
    ("where's the retry logic for webhooks implemented in this codebase", "research"),
    ("sql for weekly active users by plan", "data"),
    ("pandas: median order value by month", "data"),
    ("build a retention cohort table from the events data", "data"),
    ("Design the architecture for a multi-tenant billing system and explain the tradeoffs", "plan"),
    ("12-month product roadmap, prioritize the initiatives", "plan"),
    ("should we move from postgres to dynamodb", "plan"),
    ("Write an ADR deciding between Kafka and SQS for our event bus", "plan"),
]


class ClassifierSanity(unittest.TestCase):
    def test_obvious_prompts(self):
        # the heuristics must get these right on their own; with a matching learned model the shipped
        # classifier does too, but the model is retrained separately (see CLAUDE.md)
        wrong = []
        for text, want in CASES:
            got = m.classify_heuristic(text).job
            if got != want:
                wrong.append((text[:50], want, got))
        self.assertEqual(wrong, [])

    def test_output_shape(self):
        c = m.classify_rules("fix the failing test in auth.py")
        self.assertIn(c.job, m.JOBS)
        self.assertEqual(set(c.to_dict()), {"job", "difficulty", "confidence", "needs", "classifier"})
        self.assertIn(c.difficulty, (0, 1, 2, 3))
        self.assertTrue(0.0 <= c.confidence <= 1.0)
        self.assertIn("tools", c.needs)
        self.assertEqual(c.classifier, "rules")

    def test_empty_text_is_ask(self):
        c = m.classify_rules("")
        self.assertEqual((c.job, c.difficulty), ("ask", 0))

    def test_hard_prompt_is_harder_than_trivial_one(self):
        easy = m.classify_rules("rename foo to bar in this snippet")
        hard = m.classify_rules(
            "Fix the race condition in the job queue across all workers. Add a regression "
            "test, prove the fix is correct under concurrency, and keep it backwards compatible.")
        self.assertGreater(hard.difficulty, easy.difficulty)
        self.assertGreaterEqual(hard.difficulty, 2)

    def test_vision_and_long_context_needs(self):
        self.assertIn("vision", m.classify_rules("what is in this screenshot?").needs)
        self.assertIn("long_context", m.classify_rules("summarize " + "word " * 15000).needs)

    def test_the_eleven_jobs(self):
        self.assertEqual(set(m.JOBS), {"plan", "code", "debug", "review", "ops", "research", "data",
                                       "write", "edit", "summarize", "ask"})
        self.assertEqual(set(m.DEFAULT_LADDER), set(m.JOBS))
        self.assertLessEqual(m.TOOL_JOBS, set(m.JOBS))

    def test_every_job_has_keywords_and_a_difficulty_prior(self):
        self.assertEqual(set(m._PATTERNS), set(m.JOBS))
        self.assertEqual(set(m._JOB_PRIOR), set(m.JOBS))


BIG_FILE = "```python\n" + "\n".join(
    "def f%d(x):\n    # recieve the value\n    return x + %d" % (i, i) for i in range(300)) + "\n```"
LOG = ("```\n2024-05-01 10:02:19 ERROR connection reset by peer host=db-2\n"
       "2024-05-01 10:02:20 ERROR retry 1/3 failed: timeout\n2024-05-01 10:02:25 INFO reconnected\n```")


class StructureNotKeywords(unittest.TestCase):
    """Behavior of the offline classifier that must hold with and without the learned model."""

    def setUp(self):
        self.impls = {"heuristic": m.classify_heuristic, "rules": m.classify_rules}

    def each(self, text):
        return {name: fn(text) for name, fn in self.impls.items()}

    def test_long_pasted_file_does_not_make_a_typo_fix_hard(self):
        for name, c in self.each("Fix the typo 'recieve' in the comment of f1 and nothing else.\n\n" + BIG_FILE).items():
            self.assertEqual((c.job, c.difficulty), ("edit", 0), name)

    def test_pasted_log_plus_summarize_is_summarize(self):
        for name, c in self.each("Summarize this log and tell me what went wrong:\n\n" + LOG).items():
            self.assertEqual(c.job, "summarize", name)
            self.assertLessEqual(c.difficulty, 1, name)

    def test_set_up_ci_and_make_it_pass_is_ops(self):
        for text in ("Set up CI and make it pass", "Set up CI for this repo and make it pass"):
            for name, c in self.each(text).items():
                self.assertEqual(c.job, "ops", name)

    def test_code_fence_is_material_not_intent(self):
        code = "```python\ndef f(x):\n    return x * 2\n```"
        design = self.each("Design the architecture for a multi-region ledger and explain the tradeoffs.\n\n" + code)
        rename = self.each("Rename the variable x to count in this function.\n\n" + code)
        for name in self.impls:
            self.assertEqual(design[name].job, "plan", name)
            self.assertEqual(rename[name].job, "edit", name)

    def test_fenced_block_is_stripped_from_the_instruction(self):
        instr, mat = m.split_material("fix the typo\n```\nsome long code\n```\nthanks")
        self.assertNotIn("long code", instr)
        self.assertGreater(mat, 10)
        self.assertEqual(m.split_material("no fences here")[1], 0)
        self.assertEqual(m.split_material("open fence\n```\nnever closed")[0].strip(), "open fence")

    def test_short_edits_research_and_ops_verbs(self):
        cases = [
            ("Add a docstring to this function", "edit"),
            ("Bump the version in package.json from 1.0.0 to 1.0.1", "edit"),
            ("Find where the retry logic for webhooks is implemented in this codebase", "research"),
            ("Deploy the build to staging and run the smoke tests", "ops"),
            ("add a section on rate limits to the readme", "write"),
        ]
        for text, want in cases:
            for name, c in self.each(text).items():
                self.assertEqual(c.job, want, "%s: %s" % (name, text))


def tiny_model():
    """A real (tiny) model trained on the current JOBS with evals/train.py's own fit()."""
    rows = [{"prompt": p, "job": j, "difficulty": d} for p, j, d in (
        ("fix the failing test in auth", "debug", 1), ("why is the build flaky", "debug", 2),
        ("write a readme for the service", "write", 1), ("draft an email to the team", "write", 1),
        ("sql for weekly active users", "data", 1), ("median order value by month", "data", 1),
        ("summarize this article", "summarize", 0), ("extract the emails from this text", "summarize", 0),
        ("deploy to staging", "ops", 1), ("set up ci", "ops", 1), ("hi there", "ask", 0), ("thanks", "ask", 0),
        ("design the architecture", "plan", 3), ("roadmap and tradeoffs", "plan", 3),
        ("implement a function", "code", 1), ("add an endpoint", "code", 1),
        ("review this pull request", "review", 1), ("look over my diff", "review", 1),
        ("rename the variable", "edit", 0), ("fix the typo", "edit", 0),
        ("look up the docs for this api", "research", 1), ("compare these libraries", "research", 2))]
    prior, weights = evaltrain.fit(rows * 3)
    return {"version": 2, "jobs": list(m.JOBS), "n": len(rows) * 3, "scale": evaltrain.SCALE, "prior": prior,
            "jd": evaltrain.job_difficulty(rows * 3), "w": weights}


class LearnedModel(unittest.TestCase):
    def setUp(self):
        self.saved = list(m._MODEL), list(m._MODEL_WHY)

    def tearDown(self):
        m._MODEL[:], m._MODEL_WHY[:] = self.saved

    def test_model_file_is_shipped_and_small(self):
        path = os.path.join(os.path.dirname(m.__file__), m.MODEL_FILE)
        self.assertTrue(os.path.exists(path))
        self.assertLess(os.path.getsize(path), 150 * 1024)

    def test_shipped_model_is_used_only_if_it_matches_the_jobs(self):
        m._MODEL[:] = []
        model = m.load_model()
        c = m.classify_rules("Deploy the build to staging and run the smoke tests")
        if model:
            self.assertEqual(model["jobs"], list(m.JOBS))
            self.assertTrue(c.signals.get("model"))
        else:  # not retrained yet for the current jobs
            self.assertIn("model_ignored", c.signals)
            self.assertNotIn("model", c.signals)

    def test_model_trained_on_other_classes_is_ignored_with_a_reason(self):
        old = {"version": 1, "n": 1, "scale": 0.5, "prior": [0.0] * 11, "w": {}}  # the v2.0 shape: no job list
        self.assertIn("v2.0 task classes", m.model_problem(old))
        c = m.classify_model("Deploy the build to staging and run the smoke tests", old)
        self.assertEqual(c.job, "ops")
        self.assertIn("model_ignored", c.signals)
        self.assertNotIn("model", c.signals)
        other = dict(old, jobs=["a", "b"], prior=[0.0] * 6)
        self.assertIn("different job list", m.model_problem(other))
        self.assertEqual(m.model_problem(tiny_model()), "")
        self.assertEqual(m.model_problem(None), "malformed")

    def test_load_model_rejects_a_stale_file(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "m.json"), "w") as f:
                json.dump({"version": 1, "prior": [0.0] * 11, "scale": 1, "w": {}}, f)
            m._MODEL[:] = []
            with mock.patch.object(m, "MODEL_FILE", "m.json"), \
                    mock.patch.object(m, "__file__", os.path.join(d, "maestro.py")):
                self.assertIsNone(m.load_model())
            self.assertTrue(m._MODEL_WHY and "retrained" in m._MODEL_WHY[0])

    def test_missing_or_corrupt_model_falls_back_to_heuristics(self):
        m._MODEL[:] = [None]
        c = m.classify_rules("Deploy the build to staging and run the smoke tests")
        self.assertEqual(c.job, "ops")
        self.assertNotIn("model", c.signals)
        m._MODEL[:] = []
        with mock.patch.object(m, "MODEL_FILE", "does-not-exist.json"):
            self.assertIsNone(m.load_model())
            self.assertEqual(m.classify_rules("hi there").job, "ask")

    def test_matching_model_path_is_used(self):
        m._MODEL[:] = [tiny_model()]
        c = m.classify_rules("why is the deploy flaky")
        self.assertTrue(c.signals.get("model"))
        self.assertIn(c.job, m.JOBS)
        self.assertIn("job_probs", c.signals)

    def test_decoding_prefers_over_to_under_routing(self):
        # a prompt whose difficulty is genuinely uncertain must not be routed to a weak tier when
        # the probability mass sits on stronger ones
        m._MODEL[:] = [tiny_model()]
        c = m.classify_rules("Design a zero-downtime migration of our payments database across regions")
        self.assertNotIn(m.tier_of(m.target_tier(c, {})), ("fast",))

    def test_a_malformed_difficulty_table_is_rejected_but_a_missing_one_is_fine(self):
        model = tiny_model()
        self.assertEqual(len(model["jd"]), len(m.JOBS))
        self.assertIn("malformed", m.model_problem(dict(model, jd=[[0.0] * 4])))
        no_jd = {k: v for k, v in model.items() if k != "jd"}
        self.assertEqual(m.model_problem(no_jd), "")
        self.assertIn(m.classify_model("fix the typo", no_jd).difficulty, (0, 1, 2, 3))

    def test_difficulty_is_conditioned_on_the_job(self):
        # a model with no word evidence at all: only the per-job difficulty table speaks
        n = len(m.JOBS)
        flat = {"jobs": list(m.JOBS), "scale": 0.5, "prior": [0.0] * (n + 4), "w": {}}
        hard_plan = [[-9.0, -9.0, -9.0, 0.0] if j == "plan" else [0.0, -9.0, -9.0, -9.0] for j in m.JOBS]
        with mock.patch.object(m, "MODEL_CLASS_W", 0.0), mock.patch.object(m, "MODEL_DIFF_W", 0.0), \
                mock.patch.object(m, "MODEL_TOOLS_B", 0.0):
            c = m.classify_model("zzz qqq", dict(flat, jd=hard_plan))
        probs = c.signals["difficulty_probs"]
        self.assertEqual(probs.index(max(probs)), 3 if c.job == "plan" else 0)

    def test_when_unsure_a_job_that_needs_repo_tools_wins_over_a_chat_only_one(self):
        n = len(m.JOBS)
        flat = {"jobs": list(m.JOBS), "scale": 0.5, "prior": [0.0] * (n + 4), "w": {}}  # every job equally likely
        with mock.patch.object(m, "MODEL_CLASS_W", 0.0):
            self.assertIn(m.classify_model("zzz qqq", flat).job, m.TOOL_JOBS)

    def test_unicode_and_odd_input_do_not_crash(self):
        for text in ("", "   ", "```", "```\nunclosed", "\u4f60\u597d\uff0c\u4e16\u754c", "?" * 500, "a" * 100000):
            c = m.classify_rules(text)
            self.assertIn(c.job, m.JOBS)
            self.assertIn(c.difficulty, (0, 1, 2, 3))

    def test_rules_classify_is_fast(self):
        m.classify_rules("warm up")
        t0 = time.perf_counter()
        for _ in range(500):
            m.classify_rules("Refactor the payment module to remove the circular import and add tests " * 3)
        self.assertLess((time.perf_counter() - t0) / 500, 0.002)


if __name__ == "__main__":
    unittest.main()
