import json
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import self_generated_anonymous_object_registry_v6 as v6_registry
import self_generated_factorial_compatibility_registry_v7 as registry


class FactorialCompatibilityRegistryV7Test(unittest.TestCase):
    def test_prereg_is_cpu_only_and_launch_blocked(self):
        value = registry.load_preregistration()
        self.assertTrue(value["frozen_before_gpu_execution"])
        self.assertTrue(value["implementation_scope"]["registry_and_cpu_reducer_only"])
        self.assertFalse(value["implementation_scope"]["gpu_runner_implemented"])
        self.assertFalse(
            value["implementation_scope"][
                "factorial_prompt_embedding_runtime_binding_implemented"
            ]
        )
        self.assertFalse(value["claims"]["gpu_launch_authorized"])
        self.assertTrue(value["claims"]["launch_blocked_pending_independent_audit"])

    def test_full_three_by_three_factorial_uses_exact_action_captions(self):
        self.assertEqual(len(registry.FACTORIAL_PROMPTS), 9)
        self.assertEqual(
            {(row.state_appearance_id, row.caption_appearance_id) for row in registry.FACTORIAL_PROMPTS},
            set(registry.FACTORIAL_KEYS),
        )
        for row in registry.FACTORIAL_PROMPTS:
            self.assertEqual(row.caption, registry.ACTION_CAPTIONS[row.caption_appearance_id])
        receipt = registry.registry_receipt()
        self.assertEqual(receipt["factorial_shape"], [3, 3])
        self.assertEqual(receipt["factorial_prompt_count"], 9)

    def test_factorial_relations_partition_exactly(self):
        matched = {
            key for key in registry.FACTORIAL_KEYS if registry.factorial_relation(*key) == "matched"
        }
        clockwise = set(registry.CLOCKWISE_OFF_DIAGONAL)
        anti = set(registry.ANTI_CLOCKWISE_OFF_DIAGONAL)
        self.assertEqual(len(matched), 3)
        self.assertEqual(len(clockwise), 3)
        self.assertEqual(len(anti), 3)
        self.assertFalse(clockwise & anti)
        self.assertEqual(matched | clockwise | anti, set(registry.FACTORIAL_KEYS))

    def test_branches_swap_nuisance_and_heldout_off_diagonals(self):
        a = registry.BRANCH_OFF_DIAGONAL_FOLDS["A_to_B"]
        b = registry.BRANCH_OFF_DIAGONAL_FOLDS["B_to_A"]
        self.assertEqual(a["nuisance"], registry.CLOCKWISE_OFF_DIAGONAL)
        self.assertEqual(a["heldout"], registry.ANTI_CLOCKWISE_OFF_DIAGONAL)
        self.assertEqual(b["nuisance"], registry.ANTI_CLOCKWISE_OFF_DIAGONAL)
        self.assertEqual(b["heldout"], registry.CLOCKWISE_OFF_DIAGONAL)
        for branch in registry.BRANCHES:
            nuisance = set(registry.BRANCH_OFF_DIAGONAL_FOLDS[branch]["nuisance"])
            heldout = set(registry.BRANCH_OFF_DIAGONAL_FOLDS[branch]["heldout"])
            self.assertFalse(nuisance & heldout)
            self.assertEqual(len(nuisance | heldout), 6)

    def test_caption_baseline_and_heldout_are_unique(self):
        for branch in registry.BRANCHES:
            for caption in registry.APPEARANCE_IDS:
                state = registry.nuisance_state_for_caption(branch, caption)
                self.assertIn((state, caption), registry.BRANCH_OFF_DIAGONAL_FOLDS[branch]["nuisance"])
            for state in registry.APPEARANCE_IDS:
                caption = registry.heldout_caption_for_state(branch, state)
                self.assertIn((state, caption), registry.BRANCH_OFF_DIAGONAL_FOLDS[branch]["heldout"])

    def test_geometry_and_v6_capture_abi_are_frozen(self):
        value = registry.load_preregistration()
        self.assertEqual(
            (registry.PHASES, registry.PATCH_HEIGHT, registry.PATCH_WIDTH),
            (21, 37, 25),
        )
        self.assertEqual(registry.BLOCKS, (6, 12, 18, 24))
        self.assertTrue(value["capture"]["reuses_v6_projected_capture_abi"])
        self.assertEqual(value["capture"]["factorial_action_state_caption_shape"], [3, 3])
        neutral = {
            registry.CONTROL_CAPTIONS[state]["neutral"]
            for state in registry.APPEARANCE_IDS
        }
        self.assertEqual(neutral, {registry.IDENTICAL_NEUTRAL_CAPTION})
        self.assertTrue(
            value["capture"]["identical_prompt_neutral_caption_required_across_states"]
        )

    def test_all_v6_numeric_diagnostic_thresholds_are_unchanged(self):
        current = registry.load_preregistration()
        previous = v6_registry.load_preregistration()
        keys = (
            "primary_minimum_track_count",
            "primary_minimum_track_coverage",
            "primary_minimum_dynamic_edge_lifecycle_events",
            "noop_maximum_component_count",
            "static_to_primary_displacement_ratio_max",
            "reverse_endpoint_direction_cosine_max",
            "phase_shuffle_to_primary_acceleration_ratio_min",
            "phase_shuffle_absolute_acceleration_floor",
            "paraphrase_support_iou_min",
            "paraphrase_endpoint_direction_cosine_min",
            "lexical_placebo_to_primary_component_ratio_max",
            "source_swap_to_primary_support_iou_max",
            "source_swap_evaluated_track_coverage_max",
            "source_swap_dynamic_edge_lifecycle_max",
        )
        for key in keys:
            self.assertEqual(
                current["branchwise_diagnostic_gates"][key],
                previous["branchwise_diagnostic_gates"][key],
            )
        for section, section_keys in {
            "cross_fit": (
                "correspondence_softmax_temperature",
                "correspondence_spatial_sigma",
                "neutral_visual_cosine_top_vs_median_margin_min",
                "correspondence_top_vs_median_margin_min",
                "correspondence_top10_mass_fraction_min",
            ),
            "tracking": (
                "maximum_occlusion_gap",
                "minimum_track_observed_phases",
                "minimum_primary_track_coverage",
            ),
            "dynamic_edges": (
                "soft_distance_temperature",
                "activation_affinity",
            ),
        }.items():
            for key in section_keys:
                self.assertEqual(current[section][key], previous[section][key])

    def test_joint_tube_contract_is_not_per_phase_slot_finalization(self):
        row = registry.load_preregistration()["space_time_tubes"]
        self.assertEqual(row["construction_domain"], [21, 37, 25])
        self.assertTrue(row["joint_space_time_connected_components"])
        self.assertFalse(row["independent_per_phase_slot_finalization_permitted"])
        self.assertTrue(row["unassigned_or_rejected_voxels_go_to_dustbin"])
        self.assertTrue(row["v6_unbalanced_ot_thresholds_reused_unchanged"])
        self.assertTrue(row["variable_cardinality"])
        self.assertTrue(row["no_forced_tube"])

    def test_all_scientific_training_renderer_route_and_future_controls_hard_false(self):
        claims = registry.load_preregistration()["claims"]
        self.assertTrue(claims["representation_admission_hard_false"])
        for key in (
            "stable_transferable_action_representation_claimed",
            "scientific_claim_authorized",
            "renderer_or_decoder_authorized",
            "training_or_parameter_updates_authorized",
            "route_or_injection_authorized",
            "prompt_shuffle_control_executed",
            "heldout_transfer_control_executed",
            "gpu_launch_authorized",
        ):
            self.assertFalse(claims[key])
        self.assertFalse(claims["renderer_called"])
        self.assertFalse(claims["decoder_called"])
        self.assertFalse(claims["optimizer_created"])
        self.assertEqual(claims["parameter_updates"], 0)
        self.assertFalse(claims["route_or_injection_called"])

    def test_asset_is_canonical_json_object(self):
        raw = json.loads(registry.PREREG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(raw, registry.load_preregistration())
        self.assertEqual(len(registry.object_sha256(raw)), 64)


if __name__ == "__main__":
    unittest.main()
