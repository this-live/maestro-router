"""NOTICE.md states the bundled third-party data correctly, and the skill directory carries the same LICENSE and NOTICE."""
import json
import os
import re
import unittest

import helpers  # noqa: F401
from helpers import ROOT


class Notice(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(ROOT, "NOTICE.md"), encoding="utf-8") as f:
            self.text = f.read()
        with open(os.path.join(ROOT, "evals", "extra_benchmarks.json"), encoding="utf-8") as f:
            data = json.load(f)
        self.rows = [(m, r) for m, v in data.items() if not m.startswith("_") for r in v]
        self.models = {m for m, _ in self.rows}

    def test_the_benchmark_counts_match_the_data(self):
        arxiv = [r for _, r in self.rows if "arxiv" in json.dumps(r).lower()]
        hf = [r for _, r in self.rows if "huggingface.co" in json.dumps(r).lower()]
        self.assertEqual(len(arxiv) + len(hf), len(self.rows))
        groups = {}
        for m, r in self.rows:
            x = re.search(r"as (evaluated|tabulated) in the ([^()]+?) card", r.get("variant") or "")
            if x:
                key = "tabulated" if x.group(1) == "tabulated" else x.group(2).strip()
                rows, models = groups.setdefault(key, [0, set()])
                groups[key][0] += 1
                models.add(m)
        self.assertEqual(sorted(groups), ["Kimi-K2.5", "Qwen3.6-35B-A3B", "tabulated"])
        expect = ["%d benchmark results" % len(self.rows), "for %d models" % len(self.models), "(%d rows)" % len(hf),
                  "(%d rows)" % len(arxiv),
                  "%d rows for %d open-weight models as tabulated" % (groups["tabulated"][0], len(groups["tabulated"][1])),
                  "%d rows for %d closed-model ids" % (groups["Kimi-K2.5"][0], len(groups["Kimi-K2.5"][1])),
                  "%d rows for %d open-weight Gemma 4 models" % (groups["Qwen3.6-35B-A3B"][0], len(groups["Qwen3.6-35B-A3B"][1]))]
        for phrase in expect:
            self.assertIn(phrase, self.text, phrase)

    def test_the_skill_directory_ships_the_same_license_and_notice(self):
        for name in ("LICENSE", "NOTICE.md"):
            with open(os.path.join(ROOT, name), encoding="utf-8") as a, open(os.path.join(ROOT, "skills", "maestro", name), encoding="utf-8") as b:
                self.assertEqual(a.read(), b.read(), name)
        self.assertTrue(re.search(r"models\.dev", self.text))


if __name__ == "__main__":
    unittest.main()
