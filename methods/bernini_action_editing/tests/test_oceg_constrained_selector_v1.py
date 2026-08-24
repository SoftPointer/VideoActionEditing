#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import oceg_constrained_selector_v1 as selector  # noqa: E402


GATE_REGISTRY = {
    "quality": [
        "absolute_source_floor",
        "relative_b0_distribution",
        "temporal_cycle",
        "flicker_structure_guard",
        "policy_trust_region",
    ],
    "identity": [
        "required_node_persistence",
        "mutual_nn_identity",
        "no_id_switch",
        "authorized_topology_only",
    ],
    "noncollapse": [
        "participant_motion_floor",
        "motion_envelope_ceiling",
        "hard_negative_rejected",
    ],
}


def passing_gates():
    return {
        category: {gate_id: "pass" for gate_id in gate_ids}
        for category, gate_ids in GATE_REGISTRY.items()
    }


def candidate(candidate_id: str = "ours_0", *, margin: float = 0.8, deviation: float = 0.1):
    return {
        "candidate_id": candidate_id,
        "case_id": "case_0",
        "seed": 2028,
        "target_inputs_consumed": False,
        "real_target_decoded": False,
        "base_checkpoint_frozen": True,
        "selected_without_real_target": True,
        "hard_gates": passing_gates(),
        "graph_predicates": [
            {"predicate_id": "lift", "status": "pass"},
            {"predicate_id": "support", "status": "pass"},
        ],
        "action_margin_over_registered_floor": margin,
        "frozen_base_policy_deviation": deviation,
    }


def bundle():
    return {
        "schema_version": selector.INPUT_SCHEMA,
        "case_id": "case_0",
        "seed": 2028,
        "gate_registry": deepcopy(GATE_REGISTRY),
        "predicate_registry": [
            {"predicate_id": "lift", "applicability": "required", "preregistered": True},
            {"predicate_id": "support", "applicability": "required", "preregistered": True},
        ],
        "action_margin_min": 0.25,
        "frozen_base": {
            "record_id": "frozen_base_0",
            "case_id": "case_0",
            "seed": 2028,
            "target_inputs_consumed": False,
            "real_target_decoded": False,
            "base_frozen": True,
            "graph_observation_supplied": False,
            "graph_success": None,
            "used_as_graph_positive": False,
            "action_outcome": "fail",
            "hard_gates": passing_gates(),
        },
        "candidates": [candidate()],
    }


class OCEGConstrainedSelectorTests(unittest.TestCase):
    def test_legal_strict_net_gain_builds_only_offline_pair(self) -> None:
        result = selector.select_oceg_candidate(bundle())
        self.assertEqual(result["status"], "CANDIDATE_CHOSEN")
        self.assertEqual(result["selected_id"], "ours_0")
        self.assertEqual(result["preference_pair"]["rejected_id"], "frozen_base_0")
        self.assertTrue(result["preference_pair"]["offline_preference_pair_only"])
        self.assertFalse(result["training_authorized"])
        self.assertFalse(result["generator_injection_authorized"])
        self.assertFalse(result["scientific_claim_authorized"])

    def test_high_action_margin_cannot_compensate_for_blur(self) -> None:
        value = bundle()
        value["candidates"][0]["action_margin_over_registered_floor"] = 10_000.0
        value["candidates"][0]["hard_gates"]["quality"]["relative_b0_distribution"] = "fail"
        result = selector.select_oceg_candidate(value)
        self.assertEqual(result["status"], "FROZEN_BASE_FALLBACK")
        self.assertEqual(result["selected_id"], "frozen_base_0")
        self.assertIn(
            "quality:relative_b0_distribution:fail",
            result["candidate_evaluations"][0]["rejection_reasons"],
        )

    def test_wrong_identity_and_graph_failure_each_block_candidate(self) -> None:
        for mutate, expected in (
            (
                lambda row: row["hard_gates"]["identity"].__setitem__("no_id_switch", "fail"),
                "identity:no_id_switch:fail",
            ),
            (
                lambda row: row["graph_predicates"][0].__setitem__("status", "fail"),
                "graph:lift:fail",
            ),
        ):
            with self.subTest(expected=expected):
                value = bundle()
                mutate(value["candidates"][0])
                result = selector.select_oceg_candidate(value)
                self.assertEqual(result["status"], "FROZEN_BASE_FALLBACK")
                self.assertIn(expected, result["candidate_evaluations"][0]["rejection_reasons"])

    def test_unknown_required_support_is_not_success(self) -> None:
        value = bundle()
        value["candidates"][0]["graph_predicates"][1]["status"] = "uncertain"
        result = selector.select_oceg_candidate(value)
        self.assertEqual(result["status"], "FROZEN_BASE_FALLBACK")
        self.assertIn(
            "graph:support:uncertain",
            result["candidate_evaluations"][0]["rejection_reasons"],
        )

    def test_preregistered_not_applicable_edge_may_abstain_but_not_claim_pass(self) -> None:
        value = bundle()
        value["predicate_registry"][1]["applicability"] = "not_applicable"
        value["candidates"][0]["graph_predicates"][1]["status"] = "not_applicable"
        accepted = selector.select_oceg_candidate(value)
        self.assertEqual(accepted["status"], "CANDIDATE_CHOSEN")

        value["candidates"][0]["graph_predicates"][1]["status"] = "pass"
        rejected = selector.select_oceg_candidate(value)
        self.assertEqual(rejected["status"], "FROZEN_BASE_FALLBACK")
        self.assertIn(
            "graph:support:expected_not_applicable_got_pass",
            rejected["candidate_evaluations"][0]["rejection_reasons"],
        )

    def test_no_provable_base_failure_means_no_net_gain_claim(self) -> None:
        for outcome in ("pass", "uncertain"):
            with self.subTest(outcome=outcome):
                value = bundle()
                value["frozen_base"]["action_outcome"] = outcome
                result = selector.select_oceg_candidate(value)
                self.assertEqual(result["status"], "FROZEN_BASE_FALLBACK")
                self.assertFalse(
                    result["candidate_evaluations"][0]["strict_net_gain_over_frozen_base"]
                )

    def test_missing_or_target_leaking_frozen_base_fails_closed(self) -> None:
        value = bundle()
        del value["frozen_base"]
        with self.assertRaises(selector.OCEGConstrainedSelectionError):
            selector.select_oceg_candidate(value)

        value = bundle()
        value["frozen_base"]["target_inputs_consumed"] = True
        with self.assertRaises(selector.OCEGConstrainedSelectionError):
            selector.select_oceg_candidate(value)

    def test_unsafe_frozen_base_causes_abstention_not_fake_fallback(self) -> None:
        value = bundle()
        value["frozen_base"]["hard_gates"]["quality"]["absolute_source_floor"] = "fail"
        result = selector.select_oceg_candidate(value)
        self.assertEqual(result["status"], "ABSTAIN_NO_SAFE_OUTPUT")
        self.assertIsNone(result["selected_id"])
        self.assertIsNone(result["preference_pair"])

    def test_policy_deviation_only_breaks_tied_feasible_candidates(self) -> None:
        value = bundle()
        value["candidates"] = [
            candidate("higher_margin", margin=0.9, deviation=0.9),
            candidate("lower_deviation", margin=0.8, deviation=0.01),
        ]
        result = selector.select_oceg_candidate(value)
        self.assertEqual(result["selected_id"], "higher_margin")

        value["candidates"][1]["action_margin_over_registered_floor"] = 0.9
        result = selector.select_oceg_candidate(value)
        self.assertEqual(result["selected_id"], "lower_deviation")


if __name__ == "__main__":
    unittest.main()

