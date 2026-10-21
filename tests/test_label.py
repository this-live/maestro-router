"""evals/label.py: panel manifests, ingest validation, Dawid-Skene aggregation and agreement (S1 spec §6.3-§6.4)."""
import contextlib
import glob
import io
import json
import os
import random
import re
import sys
import tempfile
import unittest

import helpers  # noqa: F401

ROOT = helpers.ROOT
sys.path.insert(0, os.path.join(ROOT, "evals"))
import evalkit as ek  # noqa: E402
import label  # noqa: E402

JOBS = ek.JOBS


def good_row(**kw):
    row = {"id": "cc-debug-0004", "labeler": "sonnet", "schema": 1, "rubric_sha": label.rubric_sha(),
           "job": {"debug": 0.8, "research": 0.2}, "difficulty": 2, "min_tier": "opus", "needs_tools": 1, "self_contained": 0,
           "stakes": 1, "detectable": 1, "ambiguity": 1, "output_tokens": "300-1k", "agent_turns": "4-10",
           "rationale": "goroutine leak in fanout; needs reading code and pprof, a fix plus a test"}
    row.update(kw)
    return row


def fake_labels(prompt, labeler, wrong_jobs=False):
    """A fake labeler: reads each item's JOB= / D= markers from the item text; noisier for haiku."""
    out = []
    for key, body in re.findall(r'<item key="(i\d+)">(.*?)</item>', prompt, re.S):
        job = re.search(r"JOB=(\w+)", body).group(1)
        d = int(re.search(r"D=(\d)", body).group(1))
        if labeler == "haiku" and int(key[1:]) % 4 == 0:
            job, d = "ask", 0
        if wrong_jobs:
            job = "ops"
        out.append({"key": key, "job": {job: 1.0}, "difficulty": d, "min_tier": ("haiku", "sonnet", "opus", "fable")[d],
                    "needs_tools": int(job in ("code", "debug")), "self_contained": 1, "stakes": 1, "detectable": 1,
                    "ambiguity": 0, "output_tokens": "100-300", "agent_turns": "0", "rationale": "fake"})
    return json.dumps(out)


def reply_all(work, fn):
    """Act as the orchestrator: answer every call in every batch file of `work`."""
    n = 0
    for path in sorted(glob.glob(os.path.join(work, "*.jsonl"))):
        if path.endswith(".replies.jsonl") or os.path.basename(path).startswith("_"):
            continue
        rows = ek.read_jsonl(path)
        done = {r["call_id"] for r in ek.read_jsonl(path[:-6] + ".replies.jsonl")}
        new = [{"call_id": c["call_id"], "reply": fn(c)} for c in rows if c["call_id"] not in done]
        ek.append_jsonl(path[:-6] + ".replies.jsonl", new)
        n += len(new)
    return n


class Schema(unittest.TestCase):
    def test_spec_example_is_valid(self):
        self.assertIsNone(label.validate(good_row()))

    def test_job_rules(self):
        self.assertIn("multiples of 0.1", label.validate(good_row(job={"debug": 0.75, "research": 0.25})))
        self.assertIn("sum to 1", label.validate(good_row(job={"debug": 0.8})))
        self.assertIsNotNone(label.validate(good_row(job={"debug": 0.4, "code": 0.3, "ops": 0.2, "edit": 0.1})))
        self.assertIsNotNone(label.validate(good_row(job={"cooking": 1.0})))

    def test_enums_types_and_extra_fields(self):
        self.assertIsNotNone(label.validate(good_row(min_tier="standard")))
        self.assertIsNotNone(label.validate(good_row(difficulty=4)))
        self.assertIsNotNone(label.validate(good_row(needs_tools=True)))
        self.assertIsNotNone(label.validate(good_row(output_tokens="300-1000")))
        self.assertIsNotNone(label.validate(dict(good_row(), hint={"job": "debug"})))
        self.assertIsNone(label.validate(label.coerce(good_row(needs_tools=True, difficulty="2"))))

    def test_rubric_is_the_spec_text_and_shuffles_only_the_job_order(self):
        text = label.rubric()
        for f in label.FIELDS:
            self.assertIn("**%s**" % f, text)
        a, b = label.shuffled_rubric("x"), label.shuffled_rubric("y")
        self.assertNotEqual(a, b)
        norm = lambda t: sorted(" ".join(t.split()).split(" · "))  # noqa: E731
        self.assertEqual(len(norm(a)), len(norm(text)))
        for j in JOBS:
            self.assertIn("%s (" % j, a)
        self.assertEqual(a.split("**difficulty**")[1], text.split("**difficulty**")[1])


class Panel(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        rng = random.Random(1)
        self.items = []
        for i in range(44):
            job = JOBS[i % len(JOBS)]
            self.items.append({"id": "cc-%s-%04d" % (job, i), "slice": "cc-" + job, "context": "claude-code",
                               "prompt": "please handle task %d JOB=%s D=%d %s" % (i, job, i % 4, rng.random()),
                               "history": [], "images": 0, "self_contained": True, "hint": {"job": job, "difficulty": i % 4}})
        self.corpus = os.path.join(d, "pool.jsonl")
        ek.write_jsonl(self.corpus, self.items)
        self.p = label.paths(os.path.join(d, "labels"), os.path.join(d, "work"))
        self.byid = label.load_items(["C=" + self.corpus], only=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_batches_are_blind_mixed_and_stable(self):
        files, n = label.batch("sonnet", self.byid, self.p, size=20)
        self.assertEqual(n, 3)
        calls = [c for f in files for c in ek.read_jsonl(f)]
        self.assertEqual(set(calls[0]), {"call_id", "model", "prompt", "expect"})
        for c in calls:
            self.assertEqual(c["model"], "sonnet")
            self.assertNotIn("cc-", c["prompt"])          # no ids (they carry the slice name)
            self.assertNotIn("hint", c["prompt"])
            self.assertIn("Sent via: claude-code", c["prompt"])
        self.assertEqual(sum(len(re.findall(r"<item key=", c["prompt"])) for c in calls), 44)
        _, again = label.batch("sonnet", self.byid, self.p, size=20)
        self.assertEqual(again, 0)                         # everything is in flight

    def _label_all(self, labelers=label.LABELERS, wrong=False):
        for lab in labelers:
            label.batch(lab, self.byid, self.p, size=20)
        reply_all(self.p["work"], lambda c: fake_labels(c["prompt"], c["model"], wrong))
        return label.ingest(self.p, self.byid)

    def test_ingest_validates_and_aggregates(self):
        res = self._label_all()
        self.assertEqual(res["ok"], 44 * 3)
        rows = ek.read_jsonl(self.p["panel"])
        self.assertTrue(all(label.validate(r) is None for r in rows))
        agg, adj = label.aggregate(self.p)
        self.assertEqual(len(agg), 44)
        by = {r["id"]: r for r in agg}
        for it in self.items:
            r = by[it["id"]]
            self.assertEqual(label.top_job(r["job"]), it["hint"]["job"])
            self.assertAlmostEqual(sum(r["job"].values()), 1.0, places=3)
            self.assertEqual(len(r["difficulty"]), 4)
            self.assertAlmostEqual(sum(r["difficulty"]), 1.0, places=3)
            self.assertTrue(0.0 <= r["needs_tools"] <= 1.0)
        self.assertTrue(all(isinstance(a["why"], list) and a["why"] for a in adj))
        rep = label.agreement(self.p, self.byid)
        self.assertIn("hint_check", rep)
        self.assertEqual(rep["hint_check"]["job_agreement"], 1.0)
        self.assertTrue(rep["hint_check"]["panel_may_read_generator_cues"])
        self.assertGreater(rep["fields"]["job"]["value"], 0.6)
        self.assertFalse(rep["fields"]["job"]["half_weight"])

    def test_invalid_reply_and_items_are_reported_and_re_emitted(self):
        label.batch("opus", self.byid, self.p, size=20)
        bad = {"n": 0}

        def reply(c):
            bad["n"] += 1
            if bad["n"] == 1:
                return "I cannot do that"
            arr = json.loads(fake_labels(c["prompt"], "opus"))
            arr[0]["difficulty"] = 9
            return json.dumps(arr)
        reply_all(self.p["work"], reply)
        res = label.ingest(self.p, self.byid)
        self.assertEqual(res["invalid_calls"], 1)
        self.assertEqual(res["invalid_items"], 2)
        _, n = label.batch("opus", self.byid, self.p, size=20)
        self.assertEqual(n, 2)                             # the 20 of the failed call, plus the 2 invalid items

    def test_gold_honeypots_and_discard(self):
        label.batch("fable", self.byid, self.p, size=20, gold_n=10)
        reply_all(self.p["work"], lambda c: fake_labels(c["prompt"], "fable"))
        label.ingest(self.p, self.byid)
        gold = ek.read_jsonl(self.p["gold"])
        self.assertEqual(len(gold), 10)
        self.assertTrue(all(not g["verified"] for g in gold))
        files, n = label.batch("sonnet", self.byid, self.p, size=20)
        calls = [c for f in files for c in ek.read_jsonl(f)]
        self.assertEqual([len(re.findall(r"<item key=", c["prompt"])) for c in calls], [22, 16])
        meta = ek.load_meta(self.p["work"])
        self.assertTrue(all(len(meta[c["call_id"]]["honeypots"]) == 2 for c in calls))
        reply_all(self.p["work"], lambda c: fake_labels(c["prompt"], c["model"], wrong_jobs=True))
        res = label.ingest(self.p, self.byid)
        self.assertEqual(res["discarded"], 2)
        self.assertEqual(len(ek.read_jsonl(self.p["honeypots"])), 4)
        files, n = label.batch("sonnet", self.byid, self.p, size=20)
        self.assertEqual(n, 2)                             # discarded batches come back, under new call ids
        self.assertTrue(all(c["call_id"].endswith(":a2") for f in files for c in ek.read_jsonl(f)))
        reply_all(self.p["work"], lambda c: fake_labels(c["prompt"], c["model"]))
        self.assertEqual(label.ingest(self.p, self.byid)["ok"], 34)
        csv_path = os.path.join(self.tmp.name, "gold.csv")
        label.gold_csv(self.p, csv_path)
        with open(csv_path) as f:
            text = f.read().splitlines()
        cols = text[0].split(",")
        first = text[1].split(",")
        first[cols.index("difficulty_fix")] = "3"
        with open(csv_path, "w") as f:
            f.write("\n".join([text[0], ",".join(first)] + text[2:]) + "\n")
        n, errs = label.apply_gold_review(self.p, csv_path)
        self.assertEqual((n, errs), (10, []))
        gold = {g["id"]: g for g in ek.read_jsonl(self.p["gold"])}
        self.assertEqual(gold[first[0]]["difficulty"], 3)
        self.assertTrue(all(g["verified"] for g in gold.values()))

    def test_adjudication_combines_fable_with_the_panel(self):
        self._label_all()
        _, adj = label.aggregate(self.p)
        self.assertTrue(adj)
        files, n = label.batch("fable", self.byid, self.p, adjudicate=True)
        self.assertGreater(n, 0)
        prompt = ek.read_jsonl(files[0])[0]["prompt"]
        self.assertIn("Label A:", prompt)
        reply_all(self.p["work"], lambda c: json.dumps([dict(o, job={"plan": 1.0}) for o in json.loads(fake_labels(c["prompt"], "fable"))]))
        label.ingest(self.p, self.byid)
        agg, _ = label.aggregate(self.p)
        done = [r for r in agg if r["adjudicated"]]
        self.assertEqual(len(done), len(adj))
        self.assertTrue(all(r["job"]["plan"] >= 0.7 for r in done))           # 0.7 x fable + 0.3 x the panel


class Statistics(unittest.TestCase):
    def test_fleiss_kappa(self):
        self.assertAlmostEqual(label.fleiss_kappa([[3, 0], [0, 3], [3, 0], [0, 3]]), 1.0)
        # Fleiss (1971)-style hand example: 4 items, 3 raters, 2 categories
        k = label.fleiss_kappa([[2, 1], [1, 2], [3, 0], [0, 3]])
        self.assertAlmostEqual(k, (2 / 3.0 - 0.5) / 0.5, places=6)

    def test_krippendorff_ordinal(self):
        self.assertAlmostEqual(label.krippendorff_ordinal([[0, 0], [1, 1], [2, 2], [3, 3]], 4), 1.0)
        self.assertLess(label.krippendorff_ordinal([[0, 3], [3, 0], [1, 2], [2, 1]], 4), 0)
        near = label.krippendorff_ordinal([[0, 1], [1, 1], [2, 3], [3, 3], [0, 0], [2, 2]], 4)
        far = label.krippendorff_ordinal([[0, 3], [1, 1], [2, 0], [3, 3], [0, 0], [2, 2]], 4)
        self.assertGreater(near, far)

    def test_cohen_kappa(self):
        self.assertEqual(label.cohen_kappa([(1, 1), (0, 0)]), 1.0)
        self.assertAlmostEqual(label.cohen_kappa([(1, 1), (1, 0), (0, 1), (0, 0)]), 0.0)

    def test_dawid_skene_trusts_the_reliable_labeler(self):
        rng = random.Random(0)
        truth = {i: rng.randrange(3) for i in range(300)}
        obs = {}
        for i, t in truth.items():
            obs[i] = {"good": t if rng.random() < 0.9 else rng.randrange(3),
                      "bad1": t if rng.random() < 0.4 else rng.randrange(3),
                      "bad2": t if rng.random() < 0.4 else rng.randrange(3)}
        post, conf = label.dawid_skene(obs, 3)
        acc_ds = sum(max(range(3), key=lambda c: post[i][c]) == t for i, t in truth.items()) / 300.0
        acc_mv = sum(max(range(3), key=lambda c: (list(o.values()).count(c), o["good"] == c)) == truth[i] for i, o in obs.items()) / 300.0
        self.assertGreater(acc_ds, acc_mv)
        self.assertGreater(conf["good"][0][0] / sum(conf["good"][0]), conf["bad1"][0][0] / sum(conf["bad1"][0]))


class Cli(unittest.TestCase):
    def test_batch_requires_a_matching_labeler(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            label.main(["batch", "--labeler", "fable"])


if __name__ == "__main__":
    unittest.main()
