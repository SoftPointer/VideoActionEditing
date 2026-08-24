#!/usr/bin/env python3
"""CPU-only contract tests for the seen actual-target foundation canary."""

from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import unittest


RUNNER = Path(__file__).resolve().parents[1] / "actual_target_foundation_canary_v1.py"
SPEC = importlib.util.spec_from_file_location("actual_target_foundation_canary_v1", RUNNER)
assert SPEC is not None and SPEC.loader is not None
canary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = canary
SPEC.loader.exec_module(canary)


def passing_evidence(family: str) -> canary.CaseEvidenceV1:
    prereg = canary.load_preregistration()
    pair = next(row for row in prereg["pairs"] if row["family"] == family)
    margins = {name: 0.10 for name in canary.INPUT_CONTROLS}
    return canary.CaseEvidenceV1(
        family=family,
        pair_id=pair["pair_id"],
        branches={
            "frozen_base": {
                "all_models_eval_frozen": True,
                "source_and_weight_closure_unchanged": True,
                "parameter_updates": 0,
                "generator_forward_calls": 0,
            },
            "node": {
                "dustbin_used": True,
                "unbalanced_phase_pair_count": 7,
                "dustbin_unmatched_count": 2,
                "dustbin_transport_mass": 0.10,
                "forced_nonempty_slot_used": False,
                "anonymous_slot_relabel_invariant": True,
                "phase_cardinalities": [0, 2, 4, 3, 2, 1, 0, 5],
                "mechanically_valid_phases": 8,
                "positive_similarity": 0.80,
                "input_margins": dict(margins),
                "mask_descriptor_binding_break_margin": 0.10,
            },
            "track": {
                "assigned_track_count": 4,
                "visible_fraction": 0.90,
                "positive_similarity": 0.80,
                "input_margins": dict(margins),
                "cross_phase_track_identity_break_margin": 0.10,
            },
            "edge": {
                "dynamic_lifecycle_observed": True,
                "pairwise_lifecycle_count": 2,
                "evaluated_pairwise_edge_count": 8,
                "positive_similarity": 0.80,
                "input_margins": dict(margins),
                "drop_edge_margin": 0.10,
            },
            "ordered_phase": {"input_margins": dict(margins)},
        },
    )


class ActualTargetFoundationCanaryV1Test(unittest.TestCase):
    def test_preregistration_is_seen_development_only_and_exactly_four_family_bound(self):
        spec = canary.load_preregistration()
        self.assertTrue(spec["scope"]["development_only"])
        self.assertTrue(spec["scope"]["previously_seen_in_r1b"])
        self.assertFalse(spec["scope"]["locked_validation_claim_permitted"])
        self.assertFalse(spec["scope"]["scientific_representation_claim_permitted"])
        self.assertEqual({row["family"] for row in spec["pairs"]}, set(canary.FAMILIES))
        self.assertEqual({row["r1b_ordinal"] for row in spec["pairs"]}, {2, 6, 7, 11})

    def test_anonymous_variable_cardinality_and_dustbin_are_mandatory(self):
        representation = canary.load_preregistration()["proposal_and_representation"]
        self.assertFalse(representation["manual_boxes_permitted"])
        self.assertFalse(representation["semantic_role_names_permitted"])
        self.assertEqual(representation["variable_cardinality_range"], [0, 12])
        self.assertFalse(representation["forced_nonempty_slot_permitted"])
        self.assertTrue(representation["dustbin_required"])

    def test_controls_call_counts_and_foundation_availability_are_exact(self):
        spec = canary.load_preregistration()
        views = spec["views_and_controls"]
        self.assertEqual(set(views["counterfactual_controls"]), set(canary.COUNTERFACTUAL_CONTROLS))
        self.assertEqual(
            views["logical_calls_total_four_cases"],
            {
                "media_decode_calls": 8,
                "sam2_automatic_keyframe_calls": 96,
                "dinov2_keyframe_calls": 96,
                "cotracker_video_calls": 20,
                "vjepa2_video_calls": 20,
            },
        )
        availability = canary.load_availability()
        self.assertEqual(set(availability["foundations"]), {"sam2", "cotracker", "dinov2", "vjepa2"})
        self.assertFalse(availability["gpu_smoke_authorized"])
        self.assertFalse(availability["formal_gpu_execution_authorized"])

    def test_all_branches_pass_but_representation_remains_hard_false(self):
        row = canary.evaluate_case(passing_evidence(canary.FAMILIES[0]))
        self.assertTrue(row["case_pass"])
        self.assertTrue(all(row["branch_pass"].values()))
        self.assertFalse(row["representation_admitted"])

    def test_each_branch_is_independently_noncompensable(self):
        mutations = {
            "frozen_base": ("parameter_updates", 1),
            "node": ("mask_descriptor_binding_break_margin", 0.0),
            "track": ("cross_phase_track_identity_break_margin", 0.0),
            "edge": ("drop_edge_margin", 0.0),
            "ordered_phase": (
                "input_margins",
                {name: 0.0 for name in canary.INPUT_CONTROLS},
            ),
        }
        for branch, (key, value) in mutations.items():
            with self.subTest(branch=branch):
                evidence = passing_evidence(canary.FAMILIES[0])
                branches = copy.deepcopy(evidence.branches)
                branches[branch][key] = value
                row = canary.evaluate_case(
                    canary.CaseEvidenceV1(evidence.family, evidence.pair_id, branches)
                )
                self.assertFalse(row["branch_pass"][branch])
                self.assertFalse(row["case_pass"])

    def test_missing_extra_or_nonfinite_control_margin_fails(self):
        bad_margins = (
            {"target_reverse": 0.1, "target_deterministic_shuffle": 0.1},
            {**{name: 0.1 for name in canary.INPUT_CONTROLS}, "extra": 0.1},
            {**{name: 0.1 for name in canary.INPUT_CONTROLS}, "source_noop": math.nan},
        )
        for margins in bad_margins:
            evidence = passing_evidence(canary.FAMILIES[0])
            branches = copy.deepcopy(evidence.branches)
            branches["node"]["input_margins"] = margins
            row = canary.evaluate_case(
                canary.CaseEvidenceV1(evidence.family, evidence.pair_id, branches)
            )
            self.assertFalse(row["branch_pass"]["node"])
            self.assertFalse(row["case_pass"])

    def test_cardinality_requires_exact_eight_phases_and_allows_zero(self):
        evidence = passing_evidence(canary.FAMILIES[0])
        self.assertTrue(canary.evaluate_case(evidence)["branch_pass"]["node"])
        branches = copy.deepcopy(evidence.branches)
        branches["node"]["phase_cardinalities"] = [2] * 7
        row = canary.evaluate_case(
            canary.CaseEvidenceV1(evidence.family, evidence.pair_id, branches)
        )
        self.assertFalse(row["branch_pass"]["node"])

    def test_family_pair_binding_is_fail_closed(self):
        evidence = passing_evidence(canary.FAMILIES[0])
        with self.assertRaises(canary.CanaryError):
            canary.evaluate_case(
                canary.CaseEvidenceV1(canary.FAMILIES[1], evidence.pair_id, evidence.branches)
            )

    def test_aggregate_recomputes_branchwise_AND_and_never_admits_representation(self):
        rows = [canary.evaluate_case(passing_evidence(family)) for family in canary.FAMILIES]
        result = canary.aggregate_canary(rows)
        self.assertTrue(result["diagnostic_canary_pass"])
        self.assertEqual(result["passed_case_count"], 4)
        self.assertFalse(result["representation_admitted"])
        self.assertFalse(result["stable_transferable_action_representation_established"])
        self.assertFalse(result["generator_connection_authorized"])
        tampered = copy.deepcopy(rows)
        tampered[0]["branch_pass"]["edge"] = False
        tampered[0]["case_pass"] = True
        self.assertFalse(canary.aggregate_canary(tampered)["diagnostic_canary_pass"])

    def test_contract_is_zero_train_zero_generator_and_gpu_blocked(self):
        value = canary.contract()
        self.assertEqual(
            value["implementation_status"],
            "V2_CLOSURE_IMPLEMENTED_UNEXECUTED_PRE_FLIP_NO",
        )
        self.assertFalse(value["training_performed"])
        self.assertEqual(value["parameter_updates"], 0)
        self.assertFalse(value["generator_loaded"])
        self.assertEqual(value["generator_forward_calls"], 0)
        self.assertFalse(value["gpu_smoke_authorized"])
        self.assertFalse(value["formal_gpu_execution_authorized"])
        self.assertTrue(value["independent_audit_required_before_gpu"])
        self.assertTrue(value["representation_admission_hard_false"])

    def test_gpu_cli_exits_nonzero_before_any_execution(self):
        result = subprocess.run(
            [sys.executable, str(RUNNER), "--run-gpu-smoke"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("blocked pending a new independent audit", result.stderr)


if __name__ == "__main__":
    unittest.main()
