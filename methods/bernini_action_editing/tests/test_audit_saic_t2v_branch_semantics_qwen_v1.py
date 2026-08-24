from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "audit_saic_t2v_branch_semantics_qwen_v1.py"
)
SPEC = importlib.util.spec_from_file_location("saic_qwen_audit", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def observation(**updates):
    value = {
        "schema_version": MODULE.MODEL_OUTPUT_SCHEMA,
        "start_state_match": "yes",
        "requested_branch_change_present": "yes",
        "requested_change_fidelity": "exact",
        "requested_attribute_already_present_at_start": "not_applicable",
        "target_action_progress": "none",
        "terminal_state_reached": "no",
        "temporal_order_coherent": "yes",
        "identity_geometry_stable": "yes",
        "protected_scene_stable": "yes",
        "camera_motion_level": "none",
        "appearance_change_level": "none",
        "observed_evidence": ["F0 to F8 shows one stable dog."],
    }
    value.update(updates)
    return MODULE.validate_model_output(value)


class SaicQwenBranchAuditTests(unittest.TestCase):
    def test_incomplete_requires_partial_without_terminal(self):
        row = observation(target_action_progress="partial")
        self.assertEqual(
            MODULE.deterministic_branch_gate("incomplete", row), (True, [])
        )
        row = observation(
            target_action_progress="full", terminal_state_reached="yes"
        )
        passed, failures = MODULE.deterministic_branch_gate("incomplete", row)
        self.assertFalse(passed)
        self.assertIn("terminal_state_leakage", failures)

    def test_camera_branch_requires_conspicuous_camera_and_no_action(self):
        row = observation(camera_motion_level="conspicuous")
        self.assertEqual(
            MODULE.deterministic_branch_gate("camera_only", row), (True, [])
        )
        passed, failures = MODULE.deterministic_branch_gate(
            "camera_only", observation(camera_motion_level="mild")
        )
        self.assertFalse(passed)
        self.assertIn("camera_change_missing", failures)

    def test_forward_and_reverse_require_full_action_and_terminal(self):
        complete = observation(
            target_action_progress="full", terminal_state_reached="yes"
        )
        for branch in ("forward", "reverse"):
            with self.subTest(branch=branch):
                self.assertEqual(
                    MODULE.deterministic_branch_gate(branch, complete), (True, [])
                )
                passed, failures = MODULE.deterministic_branch_gate(
                    branch, observation(target_action_progress="partial")
                )
                self.assertFalse(passed)
                self.assertIn("target_action_incomplete", failures)
                self.assertIn("terminal_state_missing", failures)

    def test_appearance_start_field_is_gated_only_for_appearance_branch(self):
        row = observation(
            target_action_progress="full",
            terminal_state_reached="yes",
            requested_attribute_already_present_at_start="yes",
        )
        self.assertEqual(
            MODULE.deterministic_branch_gate("forward", row), (True, [])
        )

    def test_noop_requires_held_start_without_action(self):
        self.assertEqual(
            MODULE.deterministic_branch_gate("noop", observation()), (True, [])
        )
        passed, failures = MODULE.deterministic_branch_gate(
            "noop", observation(target_action_progress="partial")
        )
        self.assertFalse(passed)
        self.assertIn("unexpected_target_action_progress", failures)

    def test_appearance_branch_requires_temporal_change(self):
        row = observation(
            appearance_change_level="localized",
            requested_attribute_already_present_at_start="no",
        )
        self.assertEqual(
            MODULE.deterministic_branch_gate("appearance_only", row), (True, [])
        )
        passed, failures = MODULE.deterministic_branch_gate(
            "appearance_only",
            observation(requested_attribute_already_present_at_start="no"),
        )
        self.assertFalse(passed)
        self.assertIn("appearance_change_missing", failures)

    def test_appearance_rejects_wrong_or_partial_target_change(self):
        for fidelity, code in (
            ("wrong", "requested_branch_change_wrong"),
            ("partial", "requested_branch_change_spatially_incomplete"),
        ):
            with self.subTest(fidelity=fidelity):
                passed, failures = MODULE.deterministic_branch_gate(
                    "appearance_only",
                    observation(
                        appearance_change_level="localized",
                        requested_change_fidelity=fidelity,
                        requested_attribute_already_present_at_start="no",
                    ),
                )
                self.assertFalse(passed)
                self.assertIn(code, failures)

    def test_appearance_rejects_target_attribute_present_at_start(self):
        passed, failures = MODULE.deterministic_branch_gate(
            "appearance_only",
            observation(
                appearance_change_level="localized",
                requested_attribute_already_present_at_start="yes",
            ),
        )
        self.assertFalse(passed)
        self.assertIn("requested_attribute_present_at_start", failures)

    def test_prompt_routes_action_by_instruction(self):
        self.assertNotIn("the target action is the dog's", MODULE.USER_TEMPLATE)
        self.assertIn("For dog stand-to-sit", MODULE.USER_TEMPLATE)
        self.assertIn("For human kneel-to-stand", MODULE.USER_TEMPLATE)
        self.assertIn("accurately realized partial-action", MODULE.USER_TEMPLATE)
        self.assertIn("camera intensity such as 'conspicuous'", MODULE.USER_TEMPLATE)
        self.assertIn("For noop, preserving", MODULE.USER_TEMPLATE)

    def test_prompt_frame_range_is_parameterized(self):
        self.assertIn("{frame_range}", MODULE.SYSTEM_PROMPT)
        prompt = MODULE.USER_TEMPLATE.format(
            frame_range="F0..F16",
            branch="forward",
            start_state="start",
            instruction="instruction",
        )
        self.assertIn("Chronological frames F0..F16", prompt)

    def test_uncertainty_never_passes(self):
        row = observation(
            camera_motion_level="conspicuous", start_state_match="uncertain"
        )
        passed, failures = MODULE.deterministic_branch_gate("camera_only", row)
        self.assertFalse(passed)
        self.assertIn("insufficient_visual_evidence", failures)

    def test_parser_rejects_extra_key(self):
        row = dict(observation(target_action_progress="partial"))
        row["extra"] = True
        with self.assertRaises(ValueError):
            MODULE.validate_model_output(row)


if __name__ == "__main__":
    unittest.main()
