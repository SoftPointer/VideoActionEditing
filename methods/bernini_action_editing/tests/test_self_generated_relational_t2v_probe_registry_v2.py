from __future__ import annotations

import unittest

from methods.bernini_action_editing import self_generated_relational_t2v_probe_registry_v2 as registry


class RelationalT2VProbeRegistryTests(unittest.TestCase):
    def test_exact_crossappearance_control_matrix_is_preregistered(self) -> None:
        value = registry.registry_receipt()
        self.assertEqual(value["appearance_ids"], list(registry.APPEARANCE_IDS))
        self.assertEqual(value["arms"], list(registry.ARMS))
        self.assertEqual(value["sigma_cell_indices"], {"high": 18, "mid": 32, "mid_low": 38})
        self.assertEqual(len(value["appearances"]), 3)
        self.assertEqual(len(value["typed_edges"]), 3)
        self.assertFalse(value["default_cartesian_graph_used"])

    def test_every_arm_keeps_all_semantic_role_phrases(self) -> None:
        for appearance in registry.APPEARANCES:
            for caption in appearance.captions.values():
                for phrase in appearance.role_phrases.values():
                    self.assertIn(phrase, caption)

    def test_frozen_base_target_and_training_boundaries_are_nonnegotiable(self) -> None:
        value = registry.registry_receipt()
        self.assertTrue(value["frozen_base_off_branch_required_per_appearance_sigma"])
        self.assertFalse(value["frozen_base_can_supply_graph_success"])
        self.assertFalse(value["target_inputs_authorized"])
        self.assertFalse(value["final_anchor_video_decode_authorized"])
        self.assertFalse(value["training_or_parameter_updates_authorized"])
        self.assertFalse(value["routing_or_injection_authorized"])
        self.assertFalse(value["scientific_claim_authorized"])


if __name__ == "__main__":
    unittest.main()
