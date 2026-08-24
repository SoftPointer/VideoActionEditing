from __future__ import annotations

import unittest

import motive.r7_neighbor_audit_policy as audit_policy
from motive.r7_neighbor_audit_policy import (
    COHORT_DOUBLE_REVIEW_TARGETS,
    COHORT_PRIMARY_TARGETS,
    NEIGHBOR_AUDIT_POLICY,
    REASON_CODES,
    REQUIRED_GATE_IDS,
    VERDICTS,
    aggregate_gate_status,
    policy_payload,
    policy_sha256,
)


class R7NeighborAuditPolicyTests(unittest.TestCase):
    def test_policy_digest_and_design_are_frozen(self) -> None:
        self.assertEqual(
            policy_sha256(),
            "635ed3deab9726fb7d8f0297752b199d6f45ccd05f64462667153a5baca702de",
        )
        payload = policy_payload()
        sampling = payload["sampling_design"]
        self.assertEqual(sampling["primary_sampling_seed"], 260108832)
        self.assertEqual(sampling["double_review_sampling_seed"], 260108833)
        self.assertEqual(sum(COHORT_PRIMARY_TARGETS.values()), 800)
        self.assertEqual(sum(COHORT_DOUBLE_REVIEW_TARGETS.values()), 160)
        self.assertEqual(
            dict(COHORT_PRIMARY_TARGETS),
            {
                "hard": 240,
                "boundary": 240,
                "below_floor": 160,
                "far_negative": 80,
                "component_risk": 80,
            },
        )
        self.assertFalse(
            sampling["cohorts"]["component_risk"]["population_inference"]
        )
        self.assertEqual(
            payload["purpose"]["statistical_unit"],
            "unordered_base_component_pair",
        )
        self.assertEqual(
            payload["purpose"]["label_scope"],
            "split_threshold_audit_only",
        )
        self.assertFalse(payload["purpose"]["training_authorized"])
        self.assertFalse(
            payload["purpose"]["direct_training_supervision_allowed"]
        )
        population = payload["population_contract"]
        self.assertEqual(
            population["cohort_precedence"],
            [
                "component_risk",
                "hard",
                "boundary",
                "below_floor",
                "far_negative",
            ],
        )
        self.assertEqual(
            population["probability_sampling"]["design"],
            "SRSWOR",
        )
        self.assertIsNone(
            population["component_risk_sampling"]["design_weight"]
        )
        commits = payload["artifact_commit_contract"]
        self.assertEqual(commits["root_mode_octal"], "0555")
        self.assertEqual(commits["directory_mode_octal"], "0555")
        self.assertEqual(commits["file_mode_octal"], "0444")
        self.assertTrue(commits["merge_accepts_only_committed_label_roots"])
        self.assertFalse(commits["thresholds_human_calibrated"])
        threat = commits["threat_model"]
        self.assertFalse(
            threat[
                "same_uid_concurrent_mutation_after_validation_prevented"
            ]
        )
        self.assertIn(
            "external-create-only-receipts",
            threat["required_operational_controls"],
        )
        self.assertTrue(
            population["external_anchor_contract"][
                "expected_upstream_bindings_sha256_required"
            ]
        )

    def test_policy_is_deeply_immutable_and_copies_are_isolated(self) -> None:
        self.assertFalse(hasattr(audit_policy, "_POLICY_PAYLOAD"))
        canonical_bytes = audit_policy._POLICY_CANONICAL_BYTES
        self.assertIsInstance(canonical_bytes, bytes)
        with self.assertRaises(TypeError):
            canonical_bytes[0] = 0  # type: ignore[index]
        original_digest = policy_sha256()
        audit_policy._POLICY_CANONICAL_BYTES = b'{"attacker":true}'
        try:
            self.assertEqual(policy_sha256(), original_digest)
            self.assertEqual(
                policy_payload()["threshold_gates"]["hard_precision"][
                    "threshold"
                ],
                0.98,
            )
        finally:
            audit_policy._POLICY_CANONICAL_BYTES = canonical_bytes
        with self.assertRaises(TypeError):
            NEIGHBOR_AUDIT_POLICY["policy_version"] = 7  # type: ignore[index]
        with self.assertRaises(TypeError):
            sampling = NEIGHBOR_AUDIT_POLICY["sampling_design"]
            sampling["primary_review_target"] = 1  # type: ignore[index]
        payload = policy_payload()
        payload["sampling_design"]["primary_review_target"] = 1
        self.assertEqual(
            policy_payload()["sampling_design"]["primary_review_target"],
            800,
        )

    def test_verdict_reason_and_numerical_gates_match_preregistration(self) -> None:
        self.assertEqual(
            VERDICTS,
            (
                "must_same_split",
                "independent_content",
                "uncertain",
                "unreviewable",
            ),
        )
        self.assertEqual(
            REASON_CODES,
            (
                "same_clip_or_transcode",
                "temporal_overlap",
                "same_generation_lineage",
                "same_scene_different_action_edit",
                "same_subject_background_only",
                "same_action_only",
                "common_overlay_or_border",
                "unrelated",
                "media_failure",
            ),
        )
        payload = policy_payload()
        gates = payload["threshold_gates"]
        self.assertEqual(gates["hard_precision"]["threshold"], 0.98)
        self.assertEqual(
            gates["boundary_missed_link_rate"]["threshold"],
            0.02,
        )
        self.assertEqual(
            gates["below_floor_top_neighbor_missed_link_rate"]["threshold"],
            0.03,
        )
        self.assertEqual(
            gates["threshold_0_96_recall"]["threshold"],
            0.95,
        )
        self.assertEqual(gates["floor_0_92_recall"]["threshold"], 0.97)
        graph = payload["graph_safety_gates"]
        self.assertEqual(graph["priority_unresolved_count"], 0)
        self.assertEqual(
            graph["largest_component_fraction"]["threshold"],
            0.05,
        )
        self.assertEqual(
            graph["large_component_definition"]["iid_fraction"],
            0.01,
        )
        double = payload["double_review_gates"]
        self.assertEqual(double["minimum_conclusive_pairs"], 140)
        self.assertEqual(double["maximum_unresolved_fraction"], 0.10)
        self.assertEqual(
            double["minimum_raw_agreement_wilson_95_lcb"],
            0.85,
        )
        self.assertEqual(
            double["minimum_cohen_kappa_bootstrap_95_lcb"],
            0.70,
        )
        rectangle = payload["simultaneous_interval_contract"]
        self.assertEqual(
            rectangle["elementary_interval_method"],
            "SRSWOR-exact-hypergeometric-finite-population-inversion",
        )
        self.assertEqual(rectangle["family_size"], 5)
        self.assertAlmostEqual(
            rectangle["two_sided_tail_alpha"],
            0.05 / (2 * 5),
        )
        self.assertTrue(
            rectangle["unresolved_completion"][
                "formal_gate_uses_worst_case_bound"
            ]
        )
        agreement = double["raw_agreement_interval"]
        self.assertEqual(
            agreement["method"],
            "one-sided-Wilson-score-lower-bound",
        )
        self.assertAlmostEqual(agreement["z"], 1.6448536269514722)
        bootstrap = double["kappa_bootstrap"]
        self.assertEqual(bootstrap["seed"], 260108834)
        self.assertEqual(bootstrap["replicates"], 50000)
        self.assertEqual(bootstrap["lower_quantile"], 0.05)
        self.assertEqual(
            bootstrap["undefined_replicate"],
            "impute_kappa_minus_one_conservatively",
        )

    def test_caller_gate_statuses_cannot_impersonate_formal_report(self) -> None:
        statuses = {gate: "PASS" for gate in REQUIRED_GATE_IDS}
        with self.assertRaisesRegex(RuntimeError, "not a formal"):
            aggregate_gate_status(statuses)
        statuses[REQUIRED_GATE_IDS[0]] = "INSUFFICIENT"
        with self.assertRaisesRegex(RuntimeError, "cannot calibrate"):
            aggregate_gate_status(statuses)
        statuses[REQUIRED_GATE_IDS[-1]] = "FAIL"
        with self.assertRaisesRegex(RuntimeError, "cannot calibrate"):
            aggregate_gate_status(statuses)
        self.assertFalse(
            policy_payload()["purpose"]["thresholds_human_calibrated"]
        )

    def test_outcome_rejects_missing_extra_or_invalid_gate_statuses(self) -> None:
        statuses = {gate: "PASS" for gate in REQUIRED_GATE_IDS}
        missing = dict(statuses)
        missing.pop(REQUIRED_GATE_IDS[0])
        with self.assertRaisesRegex(ValueError, "gate set differs"):
            aggregate_gate_status(missing)
        extra = {**statuses, "invented_gate": "PASS"}
        with self.assertRaisesRegex(ValueError, "gate set differs"):
            aggregate_gate_status(extra)
        invalid = dict(statuses)
        invalid[REQUIRED_GATE_IDS[0]] = "SKIP"
        with self.assertRaisesRegex(ValueError, "gate statuses differ"):
            aggregate_gate_status(invalid)


if __name__ == "__main__":
    unittest.main()
