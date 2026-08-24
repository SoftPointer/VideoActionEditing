"""Immutable policy for the R7 neighbour-threshold human audit.

This policy is deliberately independent from the action-rate human audit.
It governs only the calibration of graph edges used to keep visually related
examples in the same data split.  Labels collected under this policy are
never training labels.
"""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping


POLICY_SCHEMA = "motive-r7-neighbor-threshold-audit-policy-v1"
POPULATION_ROW_SCHEMA = "motive-r7-neighbor-audit-population-row-v1"
POPULATION_CONTEXT_SCHEMA = (
    "motive-r7-neighbor-audit-population-context-v1"
)
POPULATION_COMMIT_SCHEMA = "motive-r7-neighbor-population-commit-v1"
SOURCE_MANIFEST_SCHEMA = "motive-r7-neighbor-audit-source-manifest-v1"
REVIEW_TEMPLATE_SCHEMA = "motive-r7-neighbor-blind-review-v1"
LABEL_COMMIT_SCHEMA = "motive-r7-neighbor-label-commit-v1"
MERGED_REVIEW_SCHEMA = "motive-r7-neighbor-audit-merged-review-v1"

VERDICTS: tuple[str, ...] = (
    "must_same_split",
    "independent_content",
    "uncertain",
    "unreviewable",
)

REASON_CODES: tuple[str, ...] = (
    "same_clip_or_transcode",
    "temporal_overlap",
    "same_generation_lineage",
    "same_scene_different_action_edit",
    "same_subject_background_only",
    "same_action_only",
    "common_overlay_or_border",
    "unrelated",
    "media_failure",
)

COHORT_PRIMARY_TARGETS: Mapping[str, int] = MappingProxyType(
    {
        "hard": 240,
        "boundary": 240,
        "below_floor": 160,
        "far_negative": 80,
        "component_risk": 80,
    }
)
COHORT_DOUBLE_REVIEW_TARGETS: Mapping[str, int] = MappingProxyType(
    {
        "hard": 48,
        "boundary": 48,
        "below_floor": 32,
        "far_negative": 16,
        "component_risk": 16,
    }
)

# Every entry must be reported.  The aggregate outcome helper below locks the
# PASS/FAIL/INSUFFICIENT precedence and rejects missing or invented gates.
REQUIRED_GATE_IDS: tuple[str, ...] = (
    "hard_precision_simultaneous_95_lcb",
    "boundary_missed_link_simultaneous_95_ucb",
    "below_floor_top_neighbor_missed_link_simultaneous_95_ucb",
    "threshold_096_recall_in_score_ge_092_domain_simultaneous_95_lcb",
    "floor_092_recall_in_top_neighbor_domain_simultaneous_95_lcb",
    "high_impact_hard_msf_witnesses_all_must_same_split",
    "high_impact_nonhard_cross_boundary_has_no_must_same_split",
    "priority_rows_unresolved_zero",
    "largest_component_fraction",
    "large_component_witness_coverage",
    "top_merge_witness_coverage",
    "probability_cohort_unresolved_fraction",
    "double_review_conclusive_count",
    "double_review_unresolved_fraction",
    "double_review_raw_agreement_wilson_95_lcb",
    "double_review_kappa_bootstrap_95_lcb",
)

GATE_STATUSES: tuple[str, ...] = ("PASS", "FAIL", "INSUFFICIENT")


_POLICY_PAYLOAD: dict[str, Any] = {
    "schema_version": POLICY_SCHEMA,
    "policy_version": 1,
    "purpose": {
        "statistical_unit": "unordered_base_component_pair",
        "decision_target":
            "visual-neighbor-thresholds-and-component-split-safety",
        "label_scope": "split_threshold_audit_only",
        "training_authorized": False,
        "direct_training_supervision_allowed": False,
        "generation_conditioning_allowed": False,
        "thresholds_human_calibrated": False,
        "formal_report_required_to_change_calibration_state": True,
    },
    "sampling_design": {
        # Different from both R7 action-rate sampling and its bootstrap seed.
        "primary_sampling_seed": 260108832,
        "double_review_sampling_seed": 260108833,
        "primary_review_target": 800,
        "double_review_target": 160,
        "double_review_fraction": 0.20,
        "cohorts": {
            "hard": {
                "primary_target": 240,
                "double_review_target": 48,
                "population_inference": True,
            },
            "boundary": {
                "primary_target": 240,
                "double_review_target": 48,
                "population_inference": True,
            },
            "below_floor": {
                "primary_target": 160,
                "double_review_target": 32,
                "population_inference": True,
            },
            "far_negative": {
                "primary_target": 80,
                "double_review_target": 16,
                "population_inference": True,
            },
            "component_risk": {
                "primary_target": 80,
                "double_review_target": 16,
                "population_inference": False,
                "role": "purposive_priority_casefinding_only",
            },
        },
        "formal_customization_allowed": False,
        "selection_unit_deduplication":
            "one-row-per-unordered-base-component-pair",
        "double_review_selection":
            "cohort-stratified-bottom-sha256-with-independent-seed",
        "population_selection": {
            "probability_cohorts":
                "global-bottom-sha256-within-complete-frozen-cohort",
            "component_risk":
                "frozen-priority-tier-then-global-bottom-sha256",
            "undersized_cohort_policy": "fail_closed",
            "population_row_membership_required": True,
            "population_digest_required": True,
        },
    },
    "population_contract": {
        "row_schema_version": POPULATION_ROW_SCHEMA,
        "context_schema_version": POPULATION_CONTEXT_SCHEMA,
        "commit_schema_version": POPULATION_COMMIT_SCHEMA,
        "row_fields": [
            "schema_version",
            "policy_sha256",
            "thresholds_human_calibrated",
            "statistical_unit_id",
            "base_component_pair",
            "source_bindings",
            "hidden_context",
            "witness",
            "media",
        ],
        "context_fields": [
            "schema_version",
            "policy_sha256",
            "thresholds_human_calibrated",
            "population_rows",
            "population_sha256",
            "canonical_order",
            "canonical_encoding",
            "statistical_unit",
            "cohort_precedence",
            "cohort_population_counts",
            "source_bindings",
        ],
        "canonical_order": "statistical_unit_id",
        "canonical_encoding":
            "canonical-json-one-row-per-line-with-terminal-newline",
        "population_sha256":
            "sha256-of-complete-canonical-ordered-population-jsonl",
        "external_anchor_contract": {
            "expected_population_sha256_required": True,
            "expected_population_context_sha256_required": True,
            "expected_upstream_bindings_sha256_required": True,
            "expected_population_commit_digest_required_after_commit": True,
            "anchors_must_be_external_to_population_and_context_files": True,
            "rebuilding_context_after_population_mutation_is_not_valid": True,
            "rebuilding_all_internal_anchors_after_upstream_binding_mutation_"
            "is_not_valid": True,
        },
        "required_upstream_binding_fields": [
            "indexed_graph_artifact_digest",
            "dino_edges_artifact_digest",
            "sampling_population_sha256",
            "validated_quotient_artifact_digest",
            "base_component_population_sha256",
        ],
        "one_row_per_unordered_base_component_pair": True,
        "cohort_precedence": [
            "component_risk",
            "hard",
            "boundary",
            "below_floor",
            "far_negative",
        ],
        "cohort_rules": {
            "component_risk":
                "priority_witness_true",
            "hard":
                "not_component_risk_and_score_greater_equal_0_96",
            "boundary":
                "not_component_risk_and_score_greater_equal_0_92_and_"
                "score_less_than_0_96",
            "below_floor":
                "not_component_risk_and_score_less_than_0_92_and_"
                "top_neighbor_true",
            "far_negative":
                "not_component_risk_and_score_less_than_0_92_and_"
                "top_neighbor_false",
        },
        "component_risk_priority_tiers": [
            "high_impact_hard_msf_witness",
            "high_impact_nonhard_cross_component_or_cross_split",
            "component_with_at_least_1_percent_of_iids",
            "top_merge_witness",
            "remaining_component_risk",
        ],
        "probability_sampling": {
            "design": "SRSWOR",
            "inclusion_probability_pi_h": "n_h/N_h",
            "design_weight": "N_h/n_h",
        },
        "component_risk_sampling": {
            "design": "nonprobability_purposive_priority",
            "inclusion_probability_pi_h": None,
            "design_weight": None,
            "population_inference": False,
        },
    },
    "artifact_commit_contract": {
        "population_commit_files": [
            "population.jsonl",
            "population_context.json",
            "population.done.json",
        ],
        "reviewer_bundle_files": [
            "review.jsonl",
            "review_bundle.done.json",
            "opaque_media_tree",
        ],
        "label_commit_files": [
            "labels.jsonl",
            "labels.done.json",
        ],
        "primary_and_secondary_reviewer_bundles_are_independent": True,
        "primary_and_secondary_label_commits_are_independent": True,
        "merge_accepts_only_committed_label_roots": True,
        "create_only": True,
        "exact_recursive_closure": True,
        "root_mode_octal": "0555",
        "directory_mode_octal": "0555",
        "file_mode_octal": "0444",
        "write_bits_allowed_after_commit": False,
        "thresholds_human_calibrated": False,
        "formal_report": False,
        "threat_model": {
            "formal_claim":
                "tamper-evident-at-validation-time-not-kernel-enforced-"
                "immutability",
            "detects": [
                "noncanonical-or-byte-mutated-files-at-validation-time",
                "missing-extra-symlink-hardlink-or-nonregular-paths",
                "root-directory-or-file-mode-drift",
                "file-identity-size-mtime-mode-or-link-count-change-during-"
                "read",
            ],
            "same_uid_concurrent_mutation_after_validation_prevented": False,
            "required_operational_controls": [
                "serialized-create-and-validate",
                "external-create-only-receipts",
                "read-only-or-WORM-storage-after-commit",
                "no-untrusted-same-uid-concurrent-writer",
            ],
        },
    },
    "blind_review_contract": {
        "verdicts": list(VERDICTS),
        "reason_codes": list(REASON_CODES),
        "reviewer_hidden_semantics": [
            "score",
            "score_bin",
            "cohort",
            "threshold_relation",
            "anchor",
            "qwen",
            "iid",
            "component",
            "provisional_split",
            "annotator_slot",
            "double_review_selection",
            "primary_review_result",
        ],
        "complete_two_video_review_required": True,
        "opaque_media_paths_required": True,
        "regular_non_symlink_media_required": True,
        "whole_file_byte_copy_required": True,
        "sha256_and_size_binding_required": True,
        "distinct_reviewer_ids_required": True,
        "secondary_blinded_to_primary_until_completion": True,
    },
    "simultaneous_interval_contract": {
        "familywise_confidence_level": 0.95,
        "familywise_alpha": 0.05,
        "interval_role": "formal_gate",
        "elementary_interval_method":
            "SRSWOR-exact-hypergeometric-finite-population-inversion",
        "elementary_family": [
            "hard_must_same_split_proportion",
            "boundary_must_same_split_proportion",
            "below_floor_top_neighbor_must_same_split_proportion",
            "hard_must_same_split_and_top_neighbor_proportion",
            "boundary_must_same_split_and_top_neighbor_proportion",
        ],
        "family_size": 5,
        "two_sided_tail_alpha": 0.005,
        "tail_alpha_formula": "0.05/(2*5)",
        "rectangle_construction":
            "cartesian-product-of-five-bonferroni-elementary-intervals",
        "finite_population_mass_arithmetic": {
            "stratum_mass_interval":
                "[N_h*proportion_lower,N_h*proportion_upper]",
            "sum_interval":
                "[sum_of_mass_lowers,sum_of_mass_uppers]",
        },
        "derived_ratio_interval_arithmetic": {
            "lower": "numerator_lower/denominator_upper",
            "upper": "min(1,numerator_upper/denominator_lower)",
            "zero_denominator_lower":
                "upper_is_1_and_gate_is_INSUFFICIENT_if_required",
        },
        "unresolved_completion": {
            "lower_bound": "treat_every_unresolved_as_failure_or_zero_mass",
            "upper_bound": "treat_every_unresolved_as_success_or_positive_mass",
            "formal_gate_uses_worst_case_bound": True,
        },
        "method_must_be_preregistered": True,
        "post_review_method_switching_allowed": False,
        "point_estimate_only_is_sufficient": False,
    },
    "estimand_definitions": {
        "positive_label": "must_same_split",
        "negative_label": "independent_content",
        "unresolved_labels": ["uncertain", "unreviewable", "missing"],
        "population_estimator":
            "base-component-pair-design-weighted-finite-population",
        "hard_precision":
            "P(must_same_split|hard_and_conclusive)",
        "boundary_missed_link_rate":
            "P(must_same_split|boundary_and_conclusive)",
        "below_floor_top_neighbor_missed_link_rate":
            "P(must_same_split|below_floor_top_neighbor_and_conclusive)",
        "threshold_0_96_recall":
            "N_hard*p_hard_divided_by_"
            "(N_hard*p_hard+N_boundary*p_boundary)",
        "floor_0_92_recall":
            "(N_hard*p_hard_must_and_top+"
            "N_boundary*p_boundary_must_and_top)_divided_by_"
            "(same_numerator+N_below_floor*p_below_floor_must)",
        "component_risk_role":
            "purposive_casefinding_and_graph_safety_gates_only",
    },
    "threshold_gates": {
        "hard_precision": {
            "domain": "hard_cohort",
            "interval": "simultaneous_95_lcb",
            "operator": ">=",
            "threshold": 0.98,
        },
        "boundary_missed_link_rate": {
            "domain": "boundary_cohort",
            "interval": "simultaneous_95_ucb",
            "operator": "<=",
            "threshold": 0.02,
        },
        "below_floor_top_neighbor_missed_link_rate": {
            "domain": "below_floor_top_neighbor_cohort",
            "interval": "simultaneous_95_ucb",
            "operator": "<=",
            "threshold": 0.03,
        },
        "threshold_0_96_recall": {
            "domain": "score_greater_equal_0_92_candidate_domain",
            "interval": "simultaneous_95_lcb",
            "operator": ">=",
            "threshold": 0.95,
        },
        "floor_0_92_recall": {
            "domain": "top_neighbor_candidate_domain",
            "interval": "simultaneous_95_lcb",
            "operator": ">=",
            "threshold": 0.97,
        },
        "maximum_unresolved_fraction_per_probability_cohort": 0.10,
        "probability_cohorts": [
            "hard",
            "boundary",
            "below_floor",
            "far_negative",
        ],
        "component_risk_excluded_from_overall_rate_inference": True,
    },
    "graph_safety_gates": {
        "high_impact_hard_msf_witness_rule":
            "all_labels_must_be_must_same_split",
        "high_impact_nonhard_cross_component_or_cross_split_rule":
            "zero_must_same_split_labels",
        "priority_unresolved_count": 0,
        "largest_component_fraction": {
            "operator": "<=",
            "threshold": 0.05,
            "denominator": "all_iids",
        },
        "large_component_definition": {
            "operator": ">=",
            "iid_fraction": 0.01,
        },
        "large_component_witness_coverage": 1.0,
        "top_merge_witness_coverage": 1.0,
        "priority_order": [
            "high_impact_hard_msf_witness",
            "high_impact_nonhard_cross_component_or_cross_split",
            "component_with_at_least_1_percent_of_iids",
            "top_merge_witness",
            "remaining_component_risk",
        ],
    },
    "double_review_gates": {
        "assigned_pairs": 160,
        "minimum_conclusive_pairs": 140,
        "maximum_unresolved_fraction": 0.10,
        "minimum_raw_agreement_wilson_95_lcb": 0.85,
        "minimum_cohen_kappa_bootstrap_95_lcb": 0.70,
        "same_reviewer_id_allowed": False,
        "secondary_may_observe_primary_result": False,
        "unresolved_definition":
            "missing_either_review-or-uncertain_either-or-unreviewable_either",
        "agreement_population":
            "double_review_pairs_with_two_conclusive_binary_labels",
        "agreement_categories": [
            "must_same_split",
            "independent_content",
        ],
        "raw_agreement_interval": {
            "method": "one-sided-Wilson-score-lower-bound",
            "confidence_level": 0.95,
            "z": 1.6448536269514722,
            "denominator": "pairs_with_two_conclusive_binary_labels",
            "zero_denominator": "INSUFFICIENT",
        },
        "kappa_bootstrap": {
            "method":
                "paired-cohort-stratified-iid-bootstrap-with-replacement",
            "seed": 260108834,
            "replicates": 50000,
            "interval": "one-sided-95-percentile-lower-bound",
            "lower_quantile": 0.05,
            "quantile_method": "linear_interpolation",
            "resampling_unit": "double_review_base_component_pair",
            "undefined_replicate":
                "impute_kappa_minus_one_conservatively",
            "undefined_point_estimate": "INSUFFICIENT",
            "all_replicates_degenerate": "INSUFFICIENT",
        },
    },
    "outcome_semantics": {
        "PASS":
            "all_required_gates_are_evaluable_and_pass_and_no_prerequisite_is_missing",
        "FAIL":
            "one_or_more_evaluable_required_gates_conclusively_violate_policy",
        "INSUFFICIENT":
            "no_conclusive_failure_but_one_or_more_required_gates_or_"
            "prerequisites_are_not_evaluable",
        "precedence": ["FAIL", "INSUFFICIENT", "PASS"],
        "training_authorized_for_any_outcome": False,
        "aggregate_gate_status_is_not_a_formal_report": True,
        "caller_supplied_gate_PASS_cannot_set_thresholds_human_calibrated":
            True,
    },
    "required_gate_ids": list(REQUIRED_GATE_IDS),
}


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def canonical_json(value: Any) -> str:
    """Return the sole canonical JSON encoding used by this audit."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


_POLICY_CANONICAL_BYTES = canonical_json(_POLICY_PAYLOAD).encode("utf-8")
del _POLICY_PAYLOAD


def _bind_immutable_policy(
    canonical_bytes: bytes,
) -> tuple[Mapping[str, Any], Any, Any]:
    """Bind every policy view to one immutable import-time byte string."""

    frozen = _freeze(json.loads(canonical_bytes.decode("utf-8")))
    digest = hashlib.sha256(canonical_bytes).hexdigest()

    def bound_payload() -> dict[str, Any]:
        """Return an isolated, JSON-compatible copy of the policy."""

        return json.loads(canonical_bytes.decode("utf-8"))

    def bound_sha256() -> str:
        """Return the digest of the sole canonical policy byte string."""

        return digest

    return frozen, bound_payload, bound_sha256


(
    NEIGHBOR_AUDIT_POLICY,
    policy_payload,
    policy_sha256,
) = _bind_immutable_policy(_POLICY_CANONICAL_BYTES)
del _bind_immutable_policy


def aggregate_gate_status(gate_statuses: Mapping[str, str]) -> str:
    """Reject attempts to turn caller-provided statuses into a report.

    The namespace is validated to give useful failures, but this scaffold has
    no authority to emit PASS/FAIL/INSUFFICIENT.  A separately frozen formal
    report must recompute every statistic from committed labels.
    """

    if set(gate_statuses) != set(REQUIRED_GATE_IDS):
        missing = sorted(set(REQUIRED_GATE_IDS) - set(gate_statuses))
        extra = sorted(set(gate_statuses) - set(REQUIRED_GATE_IDS))
        raise ValueError(
            "neighbor audit gate set differs: "
            f"missing={missing}, extra={extra}"
        )
    invalid = {
        gate: value
        for gate, value in gate_statuses.items()
        if type(value) is not str or value not in GATE_STATUSES
    }
    if invalid:
        raise ValueError(f"neighbor audit gate statuses differ: {invalid}")
    raise RuntimeError(
        "caller-supplied gate statuses are not a formal neighbor report and "
        "cannot calibrate thresholds"
    )
