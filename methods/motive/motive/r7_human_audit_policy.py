"""Immutable, pre-review policy for the formal R7 human audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


_POLICY_PAYLOAD: dict[str, Any] = {
    "schema_version": "motive-r7-human-audit-policy-v2",
    "policy_version": 2,
    "sampling_design": {
        "seed": 260108830,
        "positive_sample_target": 240,
        "pseudo_negative_sample_target": 200,
        "review_sample_target": 80,
        "double_review_fraction": 0.20,
        "formal_customization_allowed": False,
        "diagnostic_customization_allowed": True,
    },
    "population_gate": {
        "min_conclusive_per_probability_cohort": 100,
        "max_design_weighted_unresolved_fraction": 0.20,
        "min_pseudo_positive_precision_95_lcb": 0.80,
        "max_pseudo_negative_false_negative_rate_95_ucb": 0.20,
    },
    "finite_population_interval": {
        "method":
            "bonferroni_hypergeometric_finite_population_exact_inversion",
        "confidence_level": 0.95,
        "familywise_alpha": 0.05,
        "interval_convention": "two_sided_equal_tail",
        "per_noncensus_stratum_tail_alpha_formula":
            "familywise_alpha/(2*noncensus_strata)",
        "census_policy": "population_successes_equal_observed_successes",
        "lower_completion_unresolved_value": 0,
        "upper_completion_unresolved_value": 1,
        "wald_fpc_interval_role": "diagnostic_only",
    },
    "double_review_gate": {
        "min_conclusive_pairs": 50,
        "max_unresolved_pair_fraction": 0.20,
        "same_assigned_reviewer_id_pairs_allowed": False,
        "min_exact_raw_agreement_wilson_95_lcb": 0.80,
        "min_exact_cohen_kappa_bootstrap_95_lcb": 0.60,
        "uncertain_is_nominal_category": True,
        "missing_pair_policy": "exclude_from_reliability_estimators",
        "unresolved_pair_definition":
            "missing_either_review_or_uncertain_either_review",
    },
    "wilson_interval": {
        "method": "wilson_score_two_sided_95",
        "confidence_level": 0.95,
        "z": 1.959963984540054,
    },
    "kappa_bootstrap": {
        "method": "paired_iid_cohort_stratified_percentile_bootstrap",
        "confidence_level": 0.95,
        "interval": "two_sided_equal_tail_percentile",
        "lower_quantile": 0.025,
        "upper_quantile": 0.975,
        "quantile_method": "linear_interpolation",
        "seed": 260108831,
        "draws": 5000,
        "resampling_unit": "paired_iid",
        "resampling": "with_replacement_within_frozen_cohort",
        "undefined_draw_policy": "impute_kappa_minus_one_conservatively",
    },
    "formal_evidence": {
        "media_bytes_bound_required": True,
        "expected_sample_artifact_digest_required": True,
        "live_source_commit_revalidation_required": True,
        "external_source_artifact_digest_required": True,
        "external_source_fused_input_sha256_required": True,
        "current_sampler_implementation_required": True,
        "sampler_resume_full_recompute_required": True,
        "external_implementation_bundle_digest_required": True,
        "external_primary_labels_sha256_required": True,
        "external_secondary_labels_sha256_required_when_assigned": True,
        "external_independent_reviewer_process_attestation_required":
            True,
        "cryptographic_reviewer_identity_verified": False,
        "training_authorized": False,
        "label_scope": "rate_audit_only",
        "direct_training_supervision_allowed": False,
    },
}

IMPLEMENTATION_BUNDLE_SCHEMA = (
    "motive-r7-human-audit-implementation-bundle-v2"
)
IMPLEMENTATION_BUNDLE_FILES: tuple[tuple[str, str], ...] = (
    ("human_review", "human_review.py"),
    ("policy", "r7_human_audit_policy.py"),
    ("qwen_validator", "qwen_filter.py"),
    ("report", "r7_human_audit_report.py"),
    ("sampler", "r7_human_audit_sample.py"),
    (
        "source_manifest_validator",
        "r7_build_expansion_manifest.py",
    ),
    ("verdict_dependency", "train_action_repr.py"),
)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


HUMAN_AUDIT_POLICY: Mapping[str, Any] = _freeze(_POLICY_PAYLOAD)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def policy_payload() -> dict[str, Any]:
    """Return an isolated JSON-compatible copy of the immutable policy."""

    return json.loads(canonical_json(_POLICY_PAYLOAD))


def policy_sha256() -> str:
    return hashlib.sha256(
        canonical_json(_POLICY_PAYLOAD).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            f"implementation file must be regular and non-symlink: {path}"
        )
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def implementation_bundle_payload() -> dict[str, Any]:
    """Return the exact current source-file bundle and its canonical digest."""

    module_root = Path(__file__).resolve(strict=True).parent
    files = [
        {
            "logical_name": logical_name,
            "relative_path": relative_path,
            "sha256": _file_sha256(module_root / relative_path),
        }
        for logical_name, relative_path in IMPLEMENTATION_BUNDLE_FILES
    ]
    core = {
        "schema_version": IMPLEMENTATION_BUNDLE_SCHEMA,
        "digest_input": "canonical_json_of_schema_and_ordered_files",
        "files": files,
    }
    return {
        **core,
        "bundle_sha256": hashlib.sha256(
            canonical_json(core).encode("utf-8")
        ).hexdigest(),
    }


def implementation_bundle_sha256() -> str:
    return str(implementation_bundle_payload()["bundle_sha256"])
