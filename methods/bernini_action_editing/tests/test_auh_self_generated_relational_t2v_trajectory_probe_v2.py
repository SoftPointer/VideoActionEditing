from __future__ import annotations

import ast
from pathlib import Path
import unittest

from methods.bernini_action_editing import auh_self_generated_relational_t2v_trajectory_probe_v2 as probe


class AUHSelfGeneratedRelationalT2VProbeStaticTests(unittest.TestCase):
    def test_contract_has_explicit_frozen_base_and_no_target_training_or_decode(self) -> None:
        value = probe.probe_contract()
        self.assertEqual(value["frozen_base_arm"], "B0_FROZEN_BASE")
        self.assertTrue(value["frozen_base_per_appearance_sigma"])
        self.assertFalse(value["frozen_base_graph_observation_supplied"])
        self.assertFalse(value["target_inputs_consumed"])
        self.assertFalse(value["final_anchor_video_decode"])
        self.assertFalse(value["decoder_available_to_probe"])
        self.assertFalse(value["optimizer_created"])
        self.assertEqual(value["parameter_updates"], 0)
        self.assertFalse(value["route_or_injection_called"])
        self.assertFalse(value["scientific_claim_authorized"])

    def test_real_matrix_and_schedule_cells_are_exact(self) -> None:
        value = probe.probe_contract()
        registry = value["registry"]
        self.assertEqual(registry["appearance_ids"], ["appearance_0", "appearance_1", "appearance_2"])
        self.assertEqual(registry["arms"], ["action", "noop", "reverse", "static"])
        self.assertEqual(registry["sigma_cell_indices"], {"high": 18, "mid": 32, "mid_low": 38})
        self.assertEqual(value["capture_count"], 144)

    def test_source_contains_create_only_receipt_and_no_decode_or_optimizer_call(self) -> None:
        path = Path(probe.__file__)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertIn("open", calls)
        self.assertNotIn("decode", calls)
        self.assertNotIn("step", [name for name in calls if name == "optimizer"])
        self.assertIn('output.open("x"', source)
        self.assertNotIn("target_video=", source)


if __name__ == "__main__":
    unittest.main()
