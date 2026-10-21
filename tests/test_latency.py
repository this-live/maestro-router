import tempfile
import time
import unittest

from helpers import inventory, local, maestro as m

PROMPTS = [
    "fix the failing test in auth.py",
    "Design the architecture for a multi-tenant billing system and explain the tradeoffs",
    "Summarize this article in three bullet points",
    "hi",
    "Research the latest open-source vector databases and compare them",
]


class Latency(unittest.TestCase):
    def test_200_rules_routes_take_under_a_second(self):
        inv = inventory(clis=("codex", "gemini-cli"), local_models=[local("llama3.2:3b", 3), local("gpt-oss:120b", 120, endpoint="spark")])
        env = {"PATH": ""}
        cfg = m.load_config("/nonexistent")
        t0 = time.perf_counter()
        for i in range(200):
            m.route(PROMPTS[i % len(PROMPTS)], harness="claude-code", inventory=inv, config=cfg, env=env)
        self.assertLess(time.perf_counter() - t0, 1.0)

    def test_200_utility_routes_take_under_a_second(self):
        inv = inventory(clis=("codex", "gemini-cli"), local_models=[local("llama3.2:3b", 3), local("gpt-oss:120b", 120, endpoint="spark")])
        cfg = dict(m.load_config("/nonexistent"), decider="utility")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = {"PATH": "", "MAESTRO_HOME": tmp.name}
        m.route(PROMPTS[0], harness="claude-code", inventory=inv, config=cfg, env=env)   # load s1.bin and decide.json once
        t0 = time.perf_counter()
        for i in range(200):
            d = m.route(PROMPTS[i % len(PROMPTS)], harness="claude-code", inventory=inv, config=cfg, env=env)
        self.assertLess(time.perf_counter() - t0, 1.0)
        self.assertEqual(d["decider"], "utility")


if __name__ == "__main__":
    unittest.main()
