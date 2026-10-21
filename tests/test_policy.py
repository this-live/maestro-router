import unittest

from helpers import cls, maestro as m


class Ladder(unittest.TestCase):
    def test_every_cell_matches_the_spec_table(self):
        spec = {
            "ask": "local fast standard standard",
            "summarize": "local fast standard standard",
            "edit": "local fast standard standard",
            "write": "fast standard standard strong",
            "data": "fast standard standard strong",
            "research": "fast standard standard strong",
            "code": "fast standard standard strong",
            "ops": "fast standard standard strong",
            "review": "standard standard standard strong",
            "debug": "standard standard standard strong",
            "plan": "standard standard strong frontier",
        }
        self.assertEqual(set(spec), set(m.JOBS))
        for name, row in spec.items():
            for d, want in enumerate(row.split()):
                self.assertEqual(m.target_tier(cls(name, d), {}), want, (name, d))

    def test_cells_and_categories(self):
        self.assertEqual(m.TIERS, ("fast", "standard", "strong", "frontier"))
        self.assertEqual(set(m.CATEGORIES), {"local", "fast", "standard", "strong", "frontier", "specialist"})
        self.assertEqual((m.tier_of("local"), m.tier_of("strong")), ("fast", "strong"))

    def test_low_confidence_escalates_one_cell(self):
        self.assertEqual(m.target_tier(cls("code", 1, 0.9), {}), "standard")
        self.assertEqual(m.target_tier(cls("code", 1, 0.4), {}), "strong")
        self.assertEqual(m.target_tier(cls("ask", 0, 0.4), {}), "fast")  # a local cell moves up to fast

    def test_min_confidence_is_configurable(self):
        c = cls("code", 1, 0.6)
        self.assertEqual(m.target_tier(c, {"min_confidence": 0.7}), "strong")
        self.assertEqual(m.target_tier(c, {"min_confidence": 0.5}), "standard")

    def test_escalation_stops_at_frontier(self):
        self.assertEqual(m.target_tier(cls("plan", 3, 0.1), {}), "frontier")

    def test_max_tier_caps_everything(self):
        pol = {"max_tier": "strong"}
        self.assertEqual(m.target_tier(cls("plan", 3), pol), "strong")
        self.assertEqual(m.target_tier(cls("code", 2, 0.1), pol), "strong")
        self.assertEqual(m.target_tier(cls("ask", 0), pol), "local")

    def test_ladder_override(self):
        pol = {"ladder": {"ask": ["strong", "strong", "strong", "strong"]}}
        self.assertEqual(m.target_tier(cls("ask", 0), pol), "strong")
        self.assertEqual(m.target_tier(cls("code", 0), pol), "fast")

    def test_walk_goes_up_then_down(self):
        self.assertEqual(m.tier_walk("standard"), ["standard", "strong", "frontier", "fast"])
        self.assertEqual(m.tier_walk("fast"), ["fast", "standard", "strong", "frontier"])
        self.assertEqual(m.tier_walk("frontier"), ["frontier", "strong", "standard", "fast"])
        self.assertEqual(m.tier_walk("local"), m.tier_walk("fast"))

    def test_walk_up_respects_max_tier(self):
        self.assertEqual(m.tier_walk("standard", "strong"), ["standard", "strong", "fast"])


class ConfigFile(unittest.TestCase):
    def test_defaults_when_missing_and_invalid_values_ignored(self):
        import json, os, tempfile
        with tempfile.TemporaryDirectory() as home:
            self.assertEqual(m.load_config(home)["max_tier"], "frontier")
            with open(os.path.join(home, "config.json"), "w") as f:
                json.dump({"max_tier": "bogus", "min_confidence": 0.7,
                           "ladder": {"ask": ["fast"], "code": ["fast"] * 4, "debug": ["local", "fast", "fast", "nope"],
                                      "edit": ["local", "fast", "fast", "fast"]}}, f)
            cfg = m.load_config(home)
            self.assertEqual(cfg["max_tier"], "frontier")
            self.assertEqual(cfg["min_confidence"], 0.7)
            self.assertEqual(sorted(cfg["ladder"]), ["code", "edit"])  # bad rows dropped, `local` cells allowed
            self.assertEqual(cfg["run_timeout_s"], 300.0)
            self.assertEqual(cfg["local_endpoints"], [])


if __name__ == "__main__":
    unittest.main()


class ModelTierNotEscalatedTwice(unittest.TestCase):
    """The learned model decodes its tier by expected risk; the ladder must not add +1 on top."""

    def test_model_tier_is_not_bumped(self):
        # a low-confidence result that came from the learned model keeps its ladder cell
        c = m.Classification("edit", 1, 0.3, signals={"model": True})
        self.assertEqual(m.target_tier(c, m.DEFAULT_CONFIG), m.ladder_tier(c, m.DEFAULT_CONFIG))
        self.assertNotIn(m.target_tier(c, m.DEFAULT_CONFIG), ("strong", "frontier"))

    def test_heuristic_low_confidence_still_escalates(self):
        c = m.Classification("code", 1, 0.3)
        self.assertEqual(m.target_tier(c, m.DEFAULT_CONFIG), "strong")
