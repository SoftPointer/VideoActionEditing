from __future__ import annotations

import ast
from pathlib import Path
import unittest

from methods.bernini_action_editing import auh_self_generated_branch_graph_trajectory_probe_v3 as probe


class AUHBranchGraphTrajectoryProbeV3StaticTests(unittest.TestCase):
    def test_contract_has_36_frozen_base_cells_and_strict_transfer_bar(self) -> None:
        value = probe.probe_contract()
        self.assertEqual(value["frozen_base_arm"], "B0_FROZEN_BASE_OBSERVER_ABSENT")
        self.assertEqual(value["frozen_base_cell_count"], 36)
        self.assertTrue(value["frozen_base_per_appearance_arm_sigma"])
        self.assertFalse(value["frozen_base_graph_observation_supplied"])
        self.assertEqual(
            value["appearance_transfer_thresholds_unchanged_from_v2"],
            {"cosine_min": 0.95, "distance_max": 0.15},
        )

    def test_contract_requires_independent_branches_without_target_or_training(self) -> None:
        value = probe.probe_contract()
        self.assertTrue(value["same_initial_gaussian_across_all_branches"])
        self.assertFalse(value["same_state_prompt_overlay_used"])
        self.assertFalse(value["branch_negative_prediction_shared_after_divergence"])
        self.assertFalse(value["target_inputs_consumed"])
        self.assertFalse(value["final_anchor_video_decode"])
        self.assertFalse(value["optimizer_created"])
        self.assertEqual(value["parameter_updates"], 0)
        self.assertFalse(value["route_or_injection_called"])
        self.assertFalse(value["scientific_claim_authorized"])

    def test_forward_counts_close_exact_real_matrix(self) -> None:
        self.assertEqual(probe.EXPECTED_TRAJECTORY_FORWARDS, 960)
        self.assertEqual(probe.EXPECTED_TRAJECTORY_STEPS, 480)
        self.assertEqual(probe.EXPECTED_CAPTURE_COUNT, 144)
        self.assertEqual(probe.EXPECTED_FROZEN_BASE_CELLS, 36)
        self.assertEqual(probe.EXPECTED_TOTAL_FORWARDS, 1032)

    def test_contract_binds_plain_source_manifest(self) -> None:
        manifest = probe.probe_contract()["source_manifest"]
        self.assertGreaterEqual(manifest["file_count"], 12)
        self.assertEqual(manifest["file_count"], len(manifest["files"]))
        self.assertTrue(manifest["all_plain_nonsymlink_files"])
        self.assertTrue(
            all(len(row["sha256"]) == 64 and row["bytes"] > 0 for row in manifest["files"])
        )

    def test_source_has_create_only_receipt_and_no_decode_call(self) -> None:
        source = Path(probe.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertNotIn("decode", calls)
        self.assertIn('output.open("x"', source)
        self.assertIn('branch=f"anchor_{arm}_trajectory"', source)
        self.assertNotIn("target_video=", source)


if __name__ == "__main__":
    unittest.main()
