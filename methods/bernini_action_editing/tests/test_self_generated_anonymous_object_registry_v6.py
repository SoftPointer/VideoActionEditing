import json
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import self_generated_anonymous_object_registry_v6 as registry


class AnonymousObjectRegistryV6Test(unittest.TestCase):
    def test_preregistration_is_frozen_and_launch_authorized(self):
        value = registry.load_preregistration()
        self.assertTrue(value["frozen_before_gpu_execution"])
        self.assertTrue(value["claims"]["gpu_launch_authorized"])
        self.assertFalse(value["claims"]["launch_blocked_pending_independent_audit"])
        self.assertTrue(value["claims"]["representation_admission_hard_false"])
        receipt = registry.registry_receipt()
        self.assertTrue(receipt["gpu_launch_authorized"])
        self.assertFalse(receipt["launch_blocked_pending_independent_audit"])
        ownership = value["ownership_and_failure_claims"]
        self.assertTrue(
            ownership["explicit_capture_ownership_boundary_exception_scrub_required"]
        )
        self.assertFalse(ownership["all_allocation_failure_zeroization_claimed"])

    def test_v4_formal_negative_is_bound(self):
        row = registry.load_preregistration()["v4_formal_negative"]
        self.assertEqual((row["shared_frame_valid_cells"], row["shared_frame_total_cells"]), (0, 36))
        self.assertEqual((row["critical_role_observed_rows"], row["critical_role_total_rows"]), (0, 48))
        self.assertFalse(row["edge_threshold_failure_claimed"])
        self.assertIn("absolute_gates", row["root_cause"])

    def test_registry_has_no_role_or_token_partition_interface(self):
        receipt = registry.registry_receipt()
        self.assertIsNone(receipt["semantic_role_ids"])
        self.assertIsNone(receipt["canonical_role_phrases"])
        self.assertIsNone(receipt["token_to_role"])
        self.assertFalse(hasattr(registry, "ROLE_IDS"))
        self.assertFalse(hasattr(registry, "ROLE_PHRASES"))

    def test_exact_eight_arms(self):
        self.assertEqual(
            (registry.PHASES, registry.PATCH_HEIGHT, registry.PATCH_WIDTH),
            (21, 37, 25),
        )
        self.assertEqual(
            registry.ARMS,
            (
                "action",
                "noop",
                "reverse",
                "static",
                "neutral",
                "paraphrase",
                "lexical_placebo",
                "source_swap",
            ),
        )

    def test_source_swap_is_exact_next_action_three_cycle(self):
        for index, appearance in enumerate(registry.APPEARANCES):
            expected = registry.APPEARANCES[(index + 1) % 3].captions["action"]
            self.assertEqual(appearance.captions["source_swap"], expected)
        self.assertEqual(
            registry.registry_receipt()["source_swap_cycle"],
            [
                ["appearance_0", "appearance_1"],
                ["appearance_1", "appearance_2"],
                ["appearance_2", "appearance_0"],
            ],
        )

    def test_crossfit_pairs_are_exact_one_to_one(self):
        row = registry.load_preregistration()["cross_fit"]
        a = row["A_to_B_phase_pairs"]
        b = row["B_to_A_phase_pairs"]
        self.assertEqual(a, [[phase, phase + 1] for phase in range(0, 20, 2)])
        self.assertEqual(b, [[phase, phase + 1] for phase in range(1, 20, 2)])
        for pairs in (a, b):
            self.assertEqual(len(pairs), 10)
            self.assertEqual(len({item[0] for item in pairs}), 10)
            self.assertEqual(len({item[1] for item in pairs}), 10)
            self.assertTrue(all(left != right for left, right in pairs))

    def test_crossfit_residual_is_proposer_only(self):
        row = registry.load_preregistration()["cross_fit"]
        self.assertTrue(row["proposal_tensors_stop_gradient"])
        self.assertFalse(row["evaluation_uses_action_noop_residual"])
        self.assertEqual(
            row["correspondence_descriptor"],
            "prompt_neutral_visual_query_hidden_sketch",
        )
        self.assertFalse(row["additive_R0_compensation_permitted"])

    def test_all_numeric_gates_are_preregistered(self):
        value = registry.load_preregistration()
        for section in (
            "discovery",
            "unbalanced_ot",
            "tracking",
            "dynamic_edges",
            "cross_fit",
            "branchwise_diagnostic_gates",
        ):
            self.assertIsInstance(value[section], dict)
        gate = value["branchwise_diagnostic_gates"]
        self.assertEqual(gate["static_zero_track_displacement_definition"], 0.0)
        self.assertGreater(gate["phase_shuffle_absolute_acceleration_floor"], 0.0)
        self.assertEqual(gate["source_swap_dynamic_edge_lifecycle_max"], 0)
        self.assertEqual(value["tracking"]["minimum_track_observed_phases"], 3)
        self.assertTrue(
            value["tracking"][
                "only_qualified_tracks_contribute_to_graph_or_control_metrics"
            ]
        )
        self.assertTrue(value["dynamic_edges"]["qualified_tracks_only"])

    def test_overall_admission_requires_all_nine(self):
        row = registry.load_preregistration()["overall_diagnostic_aggregation"]
        self.assertEqual(row["expected_cell_count"], 9)
        self.assertFalse(row["cell_selection_or_compensation_permitted"])
        self.assertTrue(row["missing_cell_fails"])

    def test_asset_is_canonical_json_object(self):
        raw = json.loads(registry.PREREG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(raw, registry.load_preregistration())
        self.assertEqual(len(registry.object_sha256(raw)), 64)


if __name__ == "__main__":
    unittest.main()
