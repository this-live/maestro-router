"""Worker dispatch: decision->provider mapping and capsule assembly (no network)."""
import unittest

import helpers  # noqa: F401
from maestro import router as m


class DelegateTarget(unittest.TestCase):
    def test_local_model_maps_to_endpoint_provider(self):
        self.assertEqual(m.delegate_target({"seat": "local", "model": "local:sparky-vllm/nemotron-3.5-lightning"}),
                         ("local:sparky-vllm", "nemotron-3.5-lightning"))

    def test_cli_seats_map_to_providers(self):
        self.assertEqual(m.delegate_target({"seat": "claude-code", "model": "sonnet"}), ("claude", "sonnet"))
        self.assertEqual(m.delegate_target({"seat": "codex", "model": "gpt-6-luna"}), ("codex", "gpt-6-luna"))
        self.assertEqual(m.delegate_target({"seat": "gemini-cli", "model": "gemini-flash-latest"}), ("gemini", "gemini-flash-latest"))


class Capsule(unittest.TestCase):
    def test_capsule_has_state_task_and_contract(self):
        c = m.build_capsule("do the thing", state="fact A")
        self.assertIn("Context:\nfact A", c)
        self.assertIn("Task:\ndo the thing", c)
        self.assertIn("only the result", c)

    def test_capsule_without_state_or_contract(self):
        c = m.build_capsule("do the thing", contract="")
        self.assertNotIn("Context:", c)
        self.assertTrue(c.endswith("do the thing"))

    def test_oversized_state_is_truncated_with_marker(self):
        c = m.build_capsule("t", state="x" * 100, max_chars=10)
        self.assertIn("state truncated to 10 chars", c)
        self.assertLess(len(c), 200)


if __name__ == "__main__":
    unittest.main()
