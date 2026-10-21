import ast
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from helpers import ROOT, maestro as m

from maestro import s1

GOLDEN = os.path.join(ROOT, "tests", "data", "s1_golden.json")
EVALS = os.path.join(ROOT, "evals")
try:
    import numpy  # noqa: F401  (only the trainer test needs it)
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


def expand(task):   # the golden file stores long pastes as a repeat rule (same as evals/train_s1.expand)
    if isinstance(task, dict) and "repeat" in task:
        body = (task["repeat"] * (task["chars"] // len(task["repeat"]) + 1))[:task["chars"] - len(task["prefix"])]
        return task["prefix"] + body
    return task


def close(a, b, tol, path="$"):
    """Recursive comparison of JSON-like values with a float tolerance; returns the first difference or ""."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return "" if abs(a - b) <= tol else "%s: %r != %r" % (path, a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return "%s: keys %s != %s" % (path, sorted(a), sorted(b))
        return next((d for d in (close(a[k], b[k], tol, path + "." + k) for k in a) if d), "")
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return "%s: length %d != %d" % (path, len(a), len(b))
        return next((d for d in (close(x, y, tol, "%s[%d]" % (path, i)) for i, (x, y) in enumerate(zip(a, b))) if d), "")
    return "" if a == b else "%s: %r != %r" % (path, a, b)


def rebuild(src, dst, edit):
    """Copy an s1.bin with its header JSON edited in place (padded to the same length, so offsets stay valid)."""
    with open(src, "rb") as f:
        raw = f.read()
    n = int.from_bytes(raw[4:8], "little")
    hd = json.loads(raw[8:8 + n])
    edit(hd)
    new = json.dumps(hd, sort_keys=True, separators=(",", ":")).encode()
    assert len(new) <= n
    with open(dst, "wb") as f:
        f.write(raw[:8] + new + b" " * (n - len(new)) + raw[8 + n:])


class Engine(unittest.TestCase):
    def test_the_shipped_weights_load_and_match_their_checksums(self):
        md, why = s1.load()
        self.assertIsNotNone(md, why)
        self.assertEqual(s1.verify(), [])
        self.assertLessEqual(os.path.getsize(s1.BIN_PATH), 3 * 1024 * 1024)
        self.assertEqual((md["H"], md["M"]), (64, 1 << 20))
        self.assertIn(md["B"], (1 << 14, 1 << 15))
        self.assertTrue({"job", "b"} <= md["trained"])

    def test_golden_outputs(self):
        with open(GOLDEN, encoding="utf-8") as f:
            gold = json.load(f)
        import hashlib
        with open(s1.BIN_PATH, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), gold["bin_sha256"],
                             "s1.bin changed: regenerate with python evals/train_s1.py --golden")
        self.assertEqual(len(gold["rows"]), 20)
        for row in gold["rows"]:
            post = s1.posterior(expand(row["task"]))
            diff = close(json.loads(json.dumps(post)), row["posterior"], gold["tolerance"])
            self.assertEqual(diff, "", str(row["task"])[:60])
            leg = s1.legacy(post)
            self.assertEqual([leg.job, leg.difficulty], row["legacy"])

    def test_swar_bag_equals_the_naive_sum(self):
        rnd = random.Random(7)
        for H in (64, 16):
            emb = bytes(rnd.randrange(256) for _ in range(1024 * H))
            for n in (0, 1, 255, 256, 257, 600):
                ids = rnd.sample(range(1024), n)
                naive = [sum(emb[i * H + c] for i in ids) for c in range(H)]
                self.assertEqual(s1.bag(memoryview(emb), ids, H), naive, (H, n))

    def test_hashing_is_stable_across_hash_seeds(self):
        code = ("import json, sys; sys.path.insert(0, %r); from maestro import s1; md, _ = s1.load(); out = []\n"
                "for t in ('fix the failing test in auth.py', 'Почему тест падает?', '请帮我修复这个错误', 'a b c d e f g'):\n"
                "    f = s1.features(s1.normalize(t), s1.keyword_info(None, t)); out.append([f, s1.known(md, f)])\n"
                "print(json.dumps(out))") % os.path.join(ROOT, "skills", "maestro", "scripts")
        outs = set()
        for seed in ("0", "1", "4242"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)
            self.assertEqual(res.returncode, 0, res.stderr)
            outs.add(res.stdout)
        self.assertEqual(len(outs), 1)


class Fallback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)
        for k in [k for k in s1._S1 if k.startswith(self.tmp)]:
            del s1._S1[k]

    def ignored(self, path):
        post = s1.posterior("fix the failing test in auth.py", path=path)
        self.assertEqual((post["source"], post["trained"]), ("nb", []))
        self.assertEqual(max(post["job_p"], key=post["job_p"].get), "debug")
        return post["s1_ignored"]

    def test_missing_corrupt_and_mismatched_files_fall_back_to_nb(self):
        p = os.path.join(self.tmp, "s1.bin")
        self.assertIn("not found", self.ignored(p))
        p = os.path.join(self.tmp, "junk.bin")
        with open(p, "wb") as f:
            f.write(b"not a weights file")
        self.assertIn("unknown format", self.ignored(p))
        cases = {"feat_version": lambda h: h.update(feat_version=s1.FEAT_VERSION + 1),
                 "job list": lambda h: h["jobs"].reverse(), "format": lambda h: h.update(format=9),
                 "section emb": lambda h: h["sections"]["emb"].__setitem__(1, 7), "sizes": lambda h: h.update(K=12)}
        for i, (want, edit) in enumerate(cases.items()):
            q = os.path.join(self.tmp, "s1-%d.bin" % i)
            rebuild(s1.BIN_PATH, q, edit)
            self.assertIn(want.split()[-1], self.ignored(q), want)
        q = os.path.join(self.tmp, "short.bin")
        with open(s1.BIN_PATH, "rb") as f, open(q, "wb") as g:
            g.write(f.read()[:200000])
        self.assertIn("section", self.ignored(q))

    def test_legacy_leaves_a_non_s1_posterior_alone(self):
        cls = m.classify_rules("fix the failing test in auth.py")
        post = s1.posterior("fix the failing test in auth.py", cls, path=os.path.join(self.tmp, "none.bin"))
        self.assertIs(s1.legacy(post, cls), cls)


class Features(unittest.TestCase):
    def test_caps_hold_on_a_60k_paste_and_a_long_prompt(self):
        paste = "Why does this fail?\n" + "".join("2024-05-01 12:00:%02d ERROR job %d: timeout talking to db-%d\n"
                                              % (i % 60, i, i % 7) for i in range(1200))
        prose = " ".join("w%dx" % i for i in range(3000)) + " " + "".join("词%d" % i for i in range(300))
        for text in (paste[:60000], prose, "```\n" + "x = 1\n" * 20000 + "```\nfix it"):
            feats = s1.features(s1.normalize(text, prev_user=text), s1.keyword_info(None, text),
                                {"mode": "proxy", "n_user": 9, "n_tool": 30, "last_tool": True, "asst_len": 9e5,
                                 "prev_job": "code", "tools": True, "images": 3, "max_tokens": 4096, "harness": "codex"})
            self.assertLessEqual(len(feats), 830)
            by_ns = {}
            for f in feats:
                by_ns[f[0]] = by_ns.get(f[0], 0) + 1
            for ns, n in by_ns.items():
                self.assertLessEqual(n, s1.CAPS[ns], ns)
            self.assertEqual(len(feats), len(set(feats)))

    def test_cjk_text_yields_j_features(self):
        nm = s1.normalize("请帮我修复这个函数中的错误")
        feats = s1.features(nm)
        self.assertIn("j\x1f错误", feats)
        self.assertIn("j\x1f修", feats)
        self.assertFalse([f for f in feats if f[0] == "w"])
        self.assertIn("g\x1fcjk", feats)
        bloom, M = bytearray(1 << 7), 1 << 10            # a Bloom filter that knows one j feature
        for f in ("j\x1f错误",):
            h1, h2 = s1.zlib.crc32(f.encode()), s1.zlib.crc32(f.encode(), s1.SEED2)
            for pos in (h1 & (M - 1), (h1 + h2) & (M - 1)):
                bloom[pos >> 3] |= 1 << (pos & 7)
        ids, ood = s1.known({"bloom": bytes(bloom), "M": M, "B": 16}, feats)
        n_j = len([f for f in feats if f[0] == "j"])
        self.assertAlmostEqual(ood, 1 - 1 / n_j)         # no w features: OOD is measured on the j namespace

    def test_the_tokenizer_keeps_non_ascii_words(self):
        toks = s1.normalize("Почему этот тест падает? naïve façade straße").get("toks")
        self.assertEqual(toks, ["почему", "этот", "тест", "падает", "naïve", "façade", "straße"])
        self.assertNotIn("почему", m._TOKEN_RE.findall("почему"))   # what the old tokenizer did
        self.assertEqual(s1.tokens("see https://x.io/a?b=1, src/app.py and 3.5ms")[0], ["see", "<u>", "<f>", "and", "<n>"])
        self.assertEqual(s1.tokens("parseConfigFile auth_utils")[1], ["parse", "config", "file", "auth", "utils"])

    def test_material_is_split_out(self):
        log = "\n".join("2024-01-02 10:11:%02d ERROR worker failed" % i for i in range(6))
        tb = 'Traceback (most recent call last):\n  File "a.py", line 3, in <module>\n    main()\n  File "a.py", line 2, in main\n    1/0'
        diff = "\n".join(["--- a/x.py", "+++ b/x.py", "@@ -1,3 +1,3 @@", "-a = 1", "+a = 2", " b = 3"])
        csv = "\n".join("%d,alice,admin,2024-01-0%d,ok" % (i, i) for i in range(1, 7))
        for body, kind in ((log, "log"), (tb, "stacktrace"), (diff, "diff"), (csv, "csv")):
            nm = s1.normalize("what went wrong here?\n" + body + "\nplease explain")
            self.assertEqual(nm["types"], [kind])
            self.assertEqual(nm["toks"], ["what", "went", "wrong", "here", "please", "explain"])
        nm = s1.normalize("Convert this:\n```json\n{\"a\": 1}\n```\nand this ```inline``` thing\n```python\nx = 1")
        self.assertEqual((nm["types"], nm["fences"]), (["json", "prose", "code:python"], 3))
        self.assertEqual(nm["toks"], ["convert", "this", "and", "this", "thing"])
        self.assertEqual(s1.normalize("short\nlines\nof\nprose")["types"], [])


class Posterior(unittest.TestCase):
    KEYS = {"job_p", "b", "group", "tools", "self_contained", "stakes", "detectable", "ambiguity", "tok_out", "turns",
            "ood", "source", "trained"}

    def test_shape_and_ranges(self):
        for text in ("hi", "", "fix the failing test in auth.py", "请帮我修复这个错误", "x" * 70000):
            post = s1.posterior(text)
            self.assertEqual(set(post), self.KEYS, text[:20])
            self.assertEqual(post["source"], "s1")
            self.assertAlmostEqual(sum(post["job_p"].values()), 1.0, places=9)
            self.assertEqual(list(post["job_p"]), list(m.JOBS))
            self.assertGreater(post["b"][1], 0)
            self.assertTrue(0 <= post["ood"] <= 1 and 0 <= post["stakes"] <= 2 and 0 <= post["tools"] <= 1)
            self.assertIn(post["group"], ("chat", "agentic"))

    def test_untrained_heads_come_from_job_priors(self):
        md, _ = s1.load()
        post = s1.posterior("Design the architecture for a multi-tenant billing system")
        self.assertNotIn("tools", md["trained"])
        expect = sum(p * s1.PRIORS["tools"].get(j, 0) for j, p in post["job_p"].items())
        self.assertAlmostEqual(post["tools"], expect)
        self.assertEqual((post["stakes"], post["self_contained"]), (s1.PRIORS["stakes"], s1.PRIORS["self_contained"]))

    def test_group_and_turns(self):
        chat = s1.posterior("what is a mutex?")
        self.assertEqual((chat["group"], chat["turns"]), ("chat", [0.0, 0.0]))
        agentic = s1.posterior("what is a mutex?", hints={"tools": "yes"})
        self.assertEqual(agentic["group"], "agentic")
        self.assertGreater(agentic["turns"][1], 0)
        proxy = s1.posterior({"text": "fix the failing test in auth.py", "mode": "proxy", "tools": False})
        self.assertEqual(proxy["group"], "chat")
        proxy = s1.posterior({"text": "what is a mutex?", "mode": "proxy", "tools": True, "n_user": 2, "n_tool": 1})
        self.assertEqual(proxy["group"], "agentic")

    def test_hints(self):
        base = s1.posterior("summarize the incident and fix the config")
        post = s1.posterior("summarize the incident and fix the config",
                            hints={"job": "review", "stakes": "high", "size": "xl", "self_contained": "no", "tools": None})
        self.assertGreaterEqual(post["job_p"]["review"], 0.7)
        self.assertAlmostEqual(sum(post["job_p"].values()), 1.0)
        self.assertEqual((post["stakes"], post["self_contained"]), (2.0, 0.05))
        self.assertAlmostEqual(post["tok_out"][0], (base["tok_out"][0] + math.log(8000)) / 2)
        self.assertEqual(post["hints"], {"job": "review", "stakes": "high", "size": "xl", "self_contained": "no"})

    def test_nb_and_jev_adapters(self):
        cls = m.Classification("debug", 2, 0.8, ("tools",), "rules", {"difficulty_probs": [0, 0, 1, 0],
                                                                       "job_probs": {"debug": 0.9, "code": 0.1}})
        post = s1.nb(cls)
        self.assertEqual(post["source"], "nb")
        self.assertAlmostEqual(post["b"][0], 1.0)
        self.assertAlmostEqual(post["b"][1], 0.8)
        mix = s1.nb(m.Classification("debug", 2, 0.8, (), "rules", {"difficulty_probs": [0, 0.5, 0.5, 0]}))
        self.assertAlmostEqual(mix["b"][0], 0.25)
        self.assertAlmostEqual(mix["b"][1], math.sqrt(0.64 + 0.5625))
        self.assertAlmostEqual(post["job_p"]["debug"], 0.9 / 1.18)   # rounded-away jobs are floored at 0.02
        self.assertEqual(post["tools"], sum(p * s1.PRIORS["tools"].get(j, 0) for j, p in post["job_p"].items()))
        jc = m.Classification("plan", 3, 0.7, (), "jev", {"rules": {}})
        jc.jev = {"choice": "x", "complexity": {"reasoning": 3, "scope": 2, "ambiguity": 1.5, "risk": 3, "value": 3}}
        jp = s1.posterior("design a thing", jc)
        self.assertEqual((jp["source"], jp["job_p"]["plan"], jp["b"]), ("jev", 1.0, [2.5, 0.8]))
        self.assertEqual((jp["stakes"], jp["ambiguity"]), (2.0, 1.0))

    def test_legacy_decodes_job_and_difficulty(self):
        for text in ("hi", "fix the failing test in auth.py",
                     "Design the architecture for a multi-tenant billing system and explain the tradeoffs"):
            cls = m.classify_rules(text)
            leg = s1.legacy(s1.posterior(text, cls), cls)
            self.assertEqual(leg.classifier, "s1")
            self.assertIn(leg.job, m.JOBS)
            self.assertIn(leg.difficulty, range(4))
            self.assertTrue(leg.signals["model"])          # decoded risk-aware: target_tier must not escalate again
            self.assertEqual(leg.needs, cls.needs)
        low = s1.legacy(s1.posterior("Design the architecture for a multi-tenant billing system"), U=1.0)
        high = s1.legacy(s1.posterior("Design the architecture for a multi-tenant billing system"), U=6.0)
        self.assertGreaterEqual(m.RANK[m.ladder_tier(high, {})], m.RANK[m.ladder_tier(low, {})])

    def test_latency(self):
        texts = ["fix the failing test in auth.py", "hi", "Research vector databases and compare them with citations",
                 " ".join("word%d" % i for i in range(300))]
        s1.posterior("warm up")
        lat = []
        for i in range(200):
            cls = m.classify_rules(texts[i % 4])
            t0 = time.perf_counter()
            s1.posterior(texts[i % 4], cls)
            lat.append(time.perf_counter() - t0)
        self.assertLess(sorted(lat)[100], 0.005)   # in-process p50 budget is 1 ms; loose for slow CI runners
        self.assertLess(sum(lat), 1.0)


class Imports(unittest.TestCase):
    def test_s1_imports_only_the_allowed_stdlib_modules(self):
        allowed = {"array", "bisect", "hashlib", "json", "math", "mmap", "os", "random", "re", "statistics", "struct",
                   "sys", "time", "zlib", "operator", "__future__"}
        with open(s1.__file__, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue({a.name for a in node.names} <= allowed, ast.dump(node))
            elif isinstance(node, ast.ImportFrom):
                self.assertTrue(node.level == 1 and node.module is None and [a.name for a in node.names] == ["router"]
                                or node.module in allowed, ast.dump(node))


@unittest.skipUnless(HAVE_NUMPY, "the trainer needs numpy (training scripts only)")
class Trainer(unittest.TestCase):
    def test_any_subset_of_heads_trains_exports_and_loads(self):
        sys.path.insert(0, EVALS)
        import train_s1
        rnd = random.Random(3)
        words = "fix deploy summarize design query review refactor explain migrate audit rename research".split()
        rows = []
        for i in range(160):
            w = rnd.sample(words, 3)
            r = {"prompt": "please %s the %s module %d and %s it" % (w[0], w[1], i % 9, w[2]), "split": "train" if i < 120 else "calib",
                 "job": rnd.choice(m.JOBS), "difficulty": rnd.randrange(4), "needs_tools": rnd.random() < 0.5,
                 "stakes": rnd.randrange(3), "output_tokens": rnd.choice(["<100", "300-1k", ">3k"]),
                 "outcomes": {t: rnd.random() < 0.6 for t in ("haiku", "fable")}, "group": rnd.choice(["chat", "agentic"])}
            if i % 2:
                r.update(min_tier=rnd.choice(["haiku", "opus", "none"]), detectable=1, ambiguity=[0.2, 0.5, 0.3],
                         self_contained=0, agent_turns="4-10", turns=7, tokens={"haiku": 300, "fable": 900})
            rows.append(r)
        cfg = dict(train_s1.DEFAULTS, epochs=2, min_labels=10)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "s1.bin")
            header, st = train_s1.build(rows, cfg, out, log=lambda *a: None)
            train_s1.parity(out, st["P"], st["live"], st["W2"], st["b2"], st["Q"], st["sample"], log=lambda *a: None)
            self.assertEqual(sorted(header["trained"]), sorted(s1.posterior("x", path=out)["trained"]))
            self.assertEqual(set(header["trained"]), {"job", "b", "b_sd", "tools", "self_contained", "stakes", "detectable",
                                                      "ambiguity", "tok_out", "turns"})
            self.assertEqual(s1.verify(out), [])
            post = s1.posterior("please fix the deploy module", path=out, hints={"tools": "yes"})
            self.assertEqual((post["source"], post["group"]), ("s1", "agentic"))
            self.assertIn("platt.tools", header["calib"]["fitted"])
            s1._S1.pop(out, None)
            partial = [{k: v for k, v in r.items() if k in ("prompt", "split", "job")} for r in rows]
            header, _ = train_s1.build(partial, cfg, out, log=lambda *a: None)
            self.assertEqual(header["trained"], ["job"])
            post = s1.posterior("please fix the deploy module", path=out)
            self.assertEqual(post["trained"], ["job"])
            self.assertEqual(post["b"], list(s1.nb(m.classify_rules("please fix the deploy module"))["b"]))
            s1._S1.pop(out, None)


if __name__ == "__main__":
    unittest.main()
