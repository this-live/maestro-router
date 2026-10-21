"""evals/split.py: cluster-hash splits, the frozen manifest and the leakage checks (S1 spec §6.2)."""
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

import helpers  # noqa: F401  (puts the package on sys.path)

ROOT = helpers.ROOT
sys.path.insert(0, os.path.join(ROOT, "evals"))
import split  # noqa: E402

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi "
         "psi omega apple river stone cloud forest ember quartz violet copper meadow harbor lantern").split()


def prompt(seed, n=24):
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) + str(rng.randint(0, 99)) for _ in range(n))


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class MinHash(unittest.TestCase):
    def test_signature_is_deterministic_and_estimates_jaccard(self):
        a = prompt(1, 60)
        b = a + " one two three four five"
        sa, sb = split.minhash(a), split.minhash(b)
        self.assertEqual(sa, split.minhash(a))
        self.assertEqual(len(sa), 64)
        self.assertGreater(split.est_jaccard(sa, sb), 0.7)
        self.assertLess(split.est_jaccard(sa, split.minhash(prompt(2, 60))), 0.2)

    def test_short_text_is_one_shingle(self):
        self.assertEqual(split.shingles("fix it"), {b"fix it"})
        self.assertEqual(split.minhash("Fix  it!"), split.minhash("fix it"))

    def test_split_function_ranges(self):
        counts = {}
        for i in range(4000):
            s = split.split_of("k:item-%d" % i)
            counts[s] = counts.get(s, 0) + 1
        self.assertAlmostEqual(counts["train"] / 4000.0, 0.55, delta=0.03)
        self.assertAlmostEqual(counts["calib"] / 4000.0, 0.10, delta=0.02)
        self.assertAlmostEqual(counts["dev"] / 4000.0, 0.10, delta=0.02)
        self.assertAlmostEqual(counts["test"] / 4000.0, 0.25, delta=0.03)
        fx = [split.split_of("k:f%d" % i, fixture=True) for i in range(1000)]
        self.assertEqual(set(fx), {"train", "test"})
        self.assertAlmostEqual(fx.count("train") / 1000.0, 0.5, delta=0.05)

    def test_hash_seed_does_not_change_signatures(self):
        code = "import sys; sys.path.insert(0, %r); import split; print(split.minhash('the quick brown fox jumps over it'))" % os.path.join(ROOT, "evals")
        outs = {subprocess.run([sys.executable, "-c", code], env=dict(os.environ, PYTHONHASHSEED=s), capture_output=True,
                               text=True, check=True).stdout for s in ("1", "2", "99")}
        self.assertEqual(len(outs), 1)


class Corpus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.c = [{"id": "c-%03d" % i, "slice": "s", "prompt": prompt(i), "history": [], "self_contained": True} for i in range(120)]
        self.c.append({"id": "c-dup", "prompt": self.c[3]["prompt"] + " thanks", "history": []})   # near-duplicate of c-003
        self.c.append({"id": "c-scn-a", "prompt": prompt(500), "scenario": "S1"})
        self.c.append({"id": "c-scn-b", "prompt": prompt(501), "scenario": "S1"})
        self.t = [{"prompt": prompt(1000 + i), "job": "code", "difficulty": 1} for i in range(30)]
        self.t.append({"prompt": self.c[7]["prompt"], "job": "code", "difficulty": 1})          # T pulls c-007 into train
        self.l = [{"prompt": prompt(2000 + i), "job": "ask", "difficulty": 0} for i in range(20)]
        self.l.append({"prompt": self.c[9]["prompt"], "job": "ask", "difficulty": 0})           # L keeps c-009 out of train
        self.s = [{"id": "s-1", "source_id": "c-011", "prompt": "a totally different rewrite " + prompt(3000)}]
        self.paths = {}
        for k, rows in (("C", self.c), ("T", self.t), ("L", self.l), ("S", self.s)):
            self.paths[k] = os.path.join(d, k + ".jsonl")
            write(self.paths[k], rows)
        self.corpora = [(k, self.paths[k]) for k in "CTLS"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_roles_clusters_and_no_leakage(self):
        rows, assigned, notes, errors = split.build(self.corpora)
        self.assertEqual(errors, [])
        self.assertEqual(assigned["c-dup"][2], assigned["c-003"][2])
        self.assertEqual(assigned["c-dup"][1], assigned["c-003"][1])
        self.assertEqual(assigned["c-scn-a"][2], assigned["c-scn-b"][2])
        self.assertEqual(assigned["s-1"][1:], assigned["c-011"][1:])           # derived rows inherit the source's cluster
        self.assertEqual(assigned["c-007"][1], "train")
        self.assertEqual(assigned["c-009"][1], "test")
        self.assertTrue(all(v[1] == "train" for k, v in assigned.items() if v[0] == "T"))
        self.assertTrue(all(v[1] == "legacy" for k, v in assigned.items() if v[0] == "L"))
        by = {}
        for corpus, sp, cid in assigned.values():
            if sp not in ("legacy", "persona"):
                by.setdefault(cid, set()).add(sp)
        self.assertTrue(all(len(v) == 1 for v in by.values()))

    def test_exhaustive_disjointness(self):
        """No train row within estimated Jaccard 0.6 of a held-out row, checked over every pair (not just LSH candidates)."""
        rows, assigned, _, _ = split.build(self.corpora)
        train = [r for r in rows if assigned[r["id"]][1] == "train"]
        held = [r for r in rows if assigned[r["id"]][1] != "train"]
        for a in train:
            for b in held:
                self.assertLess(split.est_jaccard(a["sig"], b["sig"]), split.LEAK_J, (a["id"], b["id"]))
        self.assertFalse({r["sha"] for r in train} & {r["sha"] for r in held})

    def test_train_row_duplicating_a_legacy_row_is_an_error(self):
        write(self.paths["T"], self.t + [{"prompt": self.l[0]["prompt"], "job": "ask", "difficulty": 0}])
        _, _, _, errors = split.build(self.corpora)
        self.assertTrue(any("train.jsonl" in e for e in errors))

    def test_manifest_write_check_and_drift(self):
        man = os.path.join(self.tmp.name, "manifest.json")
        self.assertEqual(self._write(man), 0)
        ok, msgs = split.check(man)
        self.assertTrue(ok, msgs)
        ok, msgs = split.check(man, deep=True)
        self.assertTrue(ok, msgs)
        loaded = split.load_splits(man)
        self.assertEqual(loaded["c-009"][1], "test")
        with open(self.paths["C"], "a") as f:
            f.write(json.dumps({"id": "c-new", "prompt": "brand new"}) + "\n")
        ok, msgs = split.check(man)
        self.assertFalse(ok)
        self.assertIn("changed", msgs[0])

    def _write(self, man):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            args = ["--only", "--manifest", man, "--write"]
            for k, p in self.corpora:
                args += ["--corpus", "%s=%s" % (k, p)]
            return split.main(args)

    def test_no_manifest_is_ok(self):
        ok, msgs = split.check(os.path.join(self.tmp.name, "none.json"))
        self.assertTrue(ok)
        self.assertIn("no manifest yet", msgs[0])

    def test_ids_for_rows_without_one_are_content_hashes(self):
        items = split.read_items([("T", self.paths["T"])])
        self.assertTrue(all(rid.startswith("T-") for _, rid, _ in items))
        self.assertEqual(len({rid for _, rid, _ in items}), len(items))
        self.assertEqual(items, split.read_items([("T", self.paths["T"])]))


if __name__ == "__main__":
    unittest.main()
