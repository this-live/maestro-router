import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

import helpers
from helpers import maestro as m

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "evals"))
import run as evalrun  # noqa: E402
import split as evalsplit  # noqa: E402
import train as evaltrain  # noqa: E402


def _rows(name):
    with open(os.path.join(ROOT, "evals", name), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _toks(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


class Splits(unittest.TestCase):
    def test_dev_is_even_index_test_is_odd_and_they_partition(self):
        rows = list(range(10))
        self.assertEqual(evalrun.split_rows(rows, "dev"), [0, 2, 4, 6, 8])
        self.assertEqual(evalrun.split_rows(rows, "test"), [1, 3, 5, 7, 9])
        self.assertEqual(evalrun.split_rows(rows, "all"), rows)

    def test_metrics_include_category_delta_and_under_routing(self):
        rows = [{"prompt": "hi", "job": "plan", "difficulty": 3},   # expects frontier
                {"prompt": "hi", "job": "ask", "difficulty": 0}]    # expects a local cell = fast

        def predict(_prompt):
            return m.Classification("ask", 0, 0.9), "fast", False
        res = evalrun.evaluate(rows, predict)
        self.assertEqual(res["category_delta"], {"-3": 1, "0": 1})
        self.assertEqual(res["under_routed"], 0.5)
        self.assertEqual(res["severe_under_routed"], 0.5)
        self.assertEqual(res["category_within_1"], 0.5)
        self.assertEqual(res["job_accuracy"], 0.5)
        self.assertEqual(set(res["per_job"]), {"plan", "ask"})

    def test_local_cells_count_as_fast_in_the_scorer(self):
        self.assertEqual(evalrun.category("local"), "fast")
        rows = [{"prompt": "hi", "job": "ask", "difficulty": 0}]
        res = evalrun.evaluate(rows, lambda _p: (m.Classification("ask", 0, 0.9), "fast", False))
        self.assertEqual(res["category_accuracy"], 1.0)

    def test_load_keeps_only_rows_with_valid_labels(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"prompt": "a", "task_class": "chat", "difficulty": 0}) + "\n")
            f.write(json.dumps({"prompt": "b", "job": "debug", "difficulty": 1}) + "\n")
            f.write(json.dumps({"prompt": "c", "job": "nope", "difficulty": 1}) + "\n")
        try:
            rows = evalrun.load(f.name)
        finally:
            os.unlink(f.name)
        self.assertEqual([r["job"] for r in rows], ["debug"])


class TrainingCorpus(unittest.TestCase):
    def test_train_set_is_large_valid_and_disjoint_from_the_eval_set(self):
        train, ev = _rows("train.jsonl"), _rows("prompts.jsonl")
        self.assertGreaterEqual(len(train), 900)
        for r in train:
            self.assertIn(r["job"], m.JOBS)
            self.assertIn(r["difficulty"], (0, 1, 2, 3))
        evs = [(_toks(r["prompt"]), _toks(m.split_material(r["prompt"])[0])) for r in ev]
        for r in train:
            a, b = _toks(r["prompt"]), _toks(m.split_material(r["prompt"])[0])
            for x, y in evs:
                self.assertLessEqual(len(a & x) / max(1, len(a | x)), 0.6, r["prompt"][:60])
                self.assertLessEqual(len(b & y) / max(1, len(b | y)), 0.6, r["prompt"][:60])

    def test_train_set_covers_jobs_and_pasted_material(self):
        train = _rows("train.jsonl")
        self.assertEqual({r["job"] for r in train}, set(m.JOBS))
        fenced = sum("```" in r["prompt"] for r in train) / len(train)
        self.assertGreater(fenced, 0.25)

    def test_eval_set_uses_valid_labels_when_relabeled(self):
        ev = _rows("prompts.jsonl")
        self.assertEqual({r["job"] for r in ev}, set(m.JOBS))


class Training(unittest.TestCase):
    ROWS = [{"prompt": "fix the flaky test", "job": "debug", "difficulty": 1},
            {"prompt": "sql for weekly active users", "job": "data", "difficulty": 1},
            {"prompt": "hi", "job": "ask", "difficulty": 0}] * 4

    def test_train_writes_the_job_list_and_the_result_is_usable(self):
        with tempfile.TemporaryDirectory() as d:
            src, out = os.path.join(d, "train.jsonl"), os.path.join(d, "model.json")
            with open(src, "w") as f:
                f.write("\n".join(json.dumps(r) for r in self.ROWS))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaltrain.main(["--train", src, "--out", out]), 0)
            with open(out) as f:
                model = json.load(f)
        self.assertEqual(model["jobs"], list(m.JOBS))
        self.assertEqual(len(model["prior"]), len(m.JOBS) + 4)
        self.assertEqual(m.model_problem(model), "")
        self.assertTrue(all(len(w) == len(model["prior"]) for w in model["w"].values()))
        c = m.classify_model("sql for weekly active users", model)
        self.assertEqual(c.job, "data")

    def test_train_refuses_a_corpus_without_job_labels(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "old.jsonl")
            with open(src, "w") as f:
                f.write(json.dumps({"prompt": "hi", "task_class": "chat", "difficulty": 0}) + "\n")
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(evaltrain.main(["--train", src, "--out", os.path.join(d, "m.json")]), 2)
            self.assertIn("relabel", err.getvalue())
            self.assertFalse(os.path.exists(os.path.join(d, "m.json")))


class NewSplits(unittest.TestCase):
    """S1 §6.2: the frozen manifest (once the corpus is committed) keeps every cluster in one split, train.jsonl in train,
    the legacy and persona sets out of training, and the frozen files unchanged."""

    def test_frozen_manifest_is_consistent(self):
        if not os.path.exists(evalsplit.MANIFEST):
            self.skipTest("no evals/manifest.json yet")
        ok, msgs = evalsplit.check(evalsplit.MANIFEST)
        self.assertTrue(ok, msgs)

    def test_train_and_eval_sets_share_no_cluster(self):
        rows, assigned, _, errors = evalsplit.build([("T", os.path.join(ROOT, "evals", "train.jsonl")),
                                                     ("L", os.path.join(ROOT, "evals", "prompts.jsonl"))])
        self.assertEqual(errors, [])
        self.assertEqual({v[1] for v in assigned.values() if v[0] == "T"}, {"train"})


class ModelLookup(unittest.TestCase):
    def test_pip_install_location_is_searched(self):
        with tempfile.TemporaryDirectory() as prefix:
            share = os.path.join(prefix, "share", "maestro")
            os.makedirs(share)
            with open(os.path.join(share, "m.json"), "w") as f:
                json.dump({"jobs": list(m.JOBS), "prior": [0.0] * (len(m.JOBS) + 4), "scale": 1, "w": {}}, f)
            saved = list(m._MODEL)
            try:
                m._MODEL[:] = []
                with mock.patch.object(m, "MODEL_FILE", "m.json"), mock.patch.object(m.sys, "prefix", prefix):
                    self.assertEqual(m.load_model()["scale"], 1)
            finally:
                m._MODEL[:] = saved


if __name__ == "__main__":
    unittest.main()
