#!/usr/bin/env python3

import copy
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import e00_three_vessel_clean_matched_route_probe_spec_v1 as subject


class ThreeVesselCleanMatchedRouteSpecTest(unittest.TestCase):
    def setUp(self):
        self.spec = subject.load_spec()

    def test_default_spec_is_complete_draft(self):
        receipt = subject.validate_spec(self.spec)
        self.assertEqual(receipt["schema_version"], subject.SCHEMA_VERSION)
        self.assertEqual(
            receipt["arm_difference_paths"], list(subject.ALLOWED_ARM_DIFFERENCES)
        )
        self.assertTrue(self.spec["status"]["draft_only"])
        self.assertFalse(self.spec["status"]["execution_authorized"])

    def test_three_source_objects_and_role_switch_are_explicit(self):
        event = self.spec["event"]
        self.assertEqual([row["object_id"] for row in event["source_objects"]], [1, 2, 3])
        self.assertIn("#1 -> #2", event["frame0_relation"])
        self.assertIn("#2 -> #3", event["desired_relation"])
        self.assertEqual(event["source_objects"][1]["material_color"], "transparent glass")

    def test_prompt_bytes_are_hash_bound(self):
        prompts = self.spec["prompt_contract"]
        for field in subject.PROMPT_FIELDS:
            self.assertEqual(
                prompts[f"{field}_utf8_sha256"], subject.text_sha256(prompts[field])
            )

    def test_clean_noise_has_no_anchor_gaussian_sga_or_anc(self):
        common = self.spec["common_arm_contract"]
        self.assertEqual(common["candidate_count_by_step"], [1] * 40)
        self.assertEqual(common["noise"]["mode"], "keyed_only")
        self.assertFalse(common["noise"]["anchor_generation_gaussian_path_read"])
        self.assertFalse(common["noise"]["anchor_generation_gaussian_used"])
        self.assertFalse(common["sga"]["enabled"])
        self.assertFalse(common["anc"]["enabled"])
        self.assertEqual(
            common["noise"]["keyed_bank_digest"],
            subject.keyed_noise_bank_digest(2027, 40),
        )

    def test_route_is_the_only_arm_difference(self):
        self.assertEqual(
            subject.arm_difference_paths(self.spec), subject.ALLOWED_ARM_DIFFERENCES
        )
        route_off = subject.materialize_arm(self.spec, "clean_route_off")
        route_on = subject.materialize_arm(self.spec, "clean_route_on")
        self.assertEqual(route_off["noise"], route_on["noise"])
        self.assertEqual(route_off["anchor_observer"], route_on["anchor_observer"])
        self.assertFalse(route_off["route"]["enabled"])
        self.assertTrue(route_on["route"]["enabled"])

    def test_rejects_anchor_generation_gaussian(self):
        broken = copy.deepcopy(self.spec)
        broken["common_arm_contract"]["noise"]["anchor_generation_gaussian_used"] = True
        with self.assertRaises(subject.ThreeVesselSpecError):
            subject.validate_spec(broken)

    def test_rejects_keyed_noise_bank_drift(self):
        broken = copy.deepcopy(self.spec)
        broken["common_arm_contract"]["noise"]["keyed_bank_digest"] = "0" * 64
        with self.assertRaises(subject.ThreeVesselSpecError):
            subject.validate_spec(broken)

    def test_rejects_sga_or_anc(self):
        for branch in ("sga", "anc"):
            with self.subTest(branch=branch):
                broken = copy.deepcopy(self.spec)
                broken["common_arm_contract"][branch]["enabled"] = True
                with self.assertRaises(subject.ThreeVesselSpecError):
                    subject.validate_spec(broken)

    def test_rejects_prompt_mutation(self):
        broken = copy.deepcopy(self.spec)
        broken["prompt_contract"]["editing_instruction"] += " Mutated."
        with self.assertRaises(subject.ThreeVesselSpecError):
            subject.validate_spec(broken)

    def test_rejects_object_two_material_drift(self):
        broken = copy.deepcopy(self.spec)
        broken["event"]["source_objects"][1]["material_color"] = "white ceramic"
        with self.assertRaises(subject.ThreeVesselSpecError):
            subject.validate_spec(broken)

    def test_rejects_nonmatched_route_override(self):
        broken = copy.deepcopy(self.spec)
        broken["arms"][1]["scheduler"] = "different"
        with self.assertRaises(subject.ThreeVesselSpecError):
            subject.validate_spec(broken)


if __name__ == "__main__":
    unittest.main()
