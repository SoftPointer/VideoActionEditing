from __future__ import annotations

import unittest

from methods.bernini_action_editing import self_generated_relational_t2v_probe_registry_v3 as registry


class RelationalT2VProbeRegistryV3Tests(unittest.TestCase):
    def test_canonical_roles_are_identical_and_appearance_is_null_context(self) -> None:
        for appearance in registry.APPEARANCES:
            self.assertEqual(dict(appearance.role_phrases), dict(registry.ROLE_PHRASES))
            self.assertTrue(appearance.receipt()["appearance_descriptors_owned_by_null_context"])
            for caption in appearance.captions.values():
                for phrase in registry.ROLE_PHRASES.values():
                    self.assertIn(phrase, caption.casefold())

    def test_branch_trajectory_and_frozen_base_are_explicit(self) -> None:
        value = registry.registry_receipt()
        self.assertTrue(
            value["trajectory_authority"][
                "one_complete_native_t2v_trajectory_per_appearance_arm"
            ]
        )
        self.assertFalse(value["trajectory_authority"]["same_state_prompt_overlay_used"])
        self.assertTrue(value["frozen_base_off_branch_required_per_appearance_arm_sigma"])
        self.assertFalse(value["frozen_base_can_supply_graph_success"])

    def test_support_frame_is_not_a_reward_edge_and_thresholds_remain_strict(self) -> None:
        value = registry.registry_receipt()
        self.assertEqual(
            value["interaction_graph"]["context_edge"],
            ["start_support", "end_support"],
        )
        self.assertFalse(value["interaction_graph"]["default_cartesian_product_used"])
        self.assertEqual(registry.ADMISSION_THRESHOLDS["appearance_cosine_min"], 0.95)
        self.assertEqual(registry.ADMISSION_THRESHOLDS["appearance_distance_max"], 0.15)
        self.assertEqual(registry.ADMISSION_THRESHOLDS["role_mass_min"], 1.0e-6)
        self.assertEqual(
            registry.ADMISSION_THRESHOLDS["support_signed_distance_change_min"],
            0.03,
        )
        self.assertEqual(registry.ADMISSION_THRESHOLDS["soft_edge_switch_min"], 0.02)
        self.assertEqual(
            registry.ADMISSION_THRESHOLDS["reverse_endpoint_topology_rms_max"],
            0.15,
        )
        self.assertEqual(
            registry.ADMISSION_THRESHOLDS["reverse_endpoint_topology_max_abs_max"],
            0.15,
        )


if __name__ == "__main__":
    unittest.main()
