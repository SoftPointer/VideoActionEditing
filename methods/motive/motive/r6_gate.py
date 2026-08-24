"""Fail-closed statistics gate for the R6 semantic/reference pilot.

R6 is a diagnostic representation experiment, not a generation gate.  This
module therefore has two deliberately separate conclusions:

* ``status`` is always ``INSUFFICIENT`` for structurally valid R6 pilots.
* ``pilot_diagnostic.status`` may be ``GO`` only when every pre-registered
  semantic/reference comparison, selector audit, and coverage check passes.

The query target may be used to construct an evaluation label, but never to
select a reference or as a predictor input.  An exact-target arm is allowed as
an explicitly tagged oracle diagnostic and is excluded from every criterion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


R6_GATE_SCHEMA = "motive-r6-semantic-reference-gate-v1"
R6_QUERY_SCHEMA = "motive-r6-per-query-v1"
R6_GATE_STATUSES = frozenset({"INSUFFICIENT", "INVALID"})
R6_PILOT_STATUSES = frozenset({"GO", "NO_GO", "INSUFFICIENT", "INVALID"})
R6_REQUIRED_ARMS = frozenset(
    {
        "semantic_only",
        "independent_ref",
        "wrong_ref",
        "matched_random",
        "centroid",
        "source_shuffle",
        "semantic_shuffle",
    }
)
R6_ORACLE_ARMS = frozenset({"exact_target_oracle"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class R6GateInputError(ValueError):
    """Raised when R6 evidence cannot be interpreted without guessing."""


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class R6PilotThresholds:
    """Pre-registered, diagnostic-only R6 thresholds."""

    minimum_model_seeds: int = 5
    minimum_positive_seed_directions: int = 4
    minimum_paired_test_groups: int = 5
    minimum_test_positive_groups: int = 20
    minimum_reference_any_fraction: float = 1.00
    minimum_reference_full_fraction: float = 1.00
    minimum_unique_train_references: int = 5
    maximum_reference_load_fraction: float = 0.35
    minimum_compatibility_auroc: float = 0.70
    minimum_compatible_recall: float = 0.80
    maximum_failed_outcome_fpr: float = 0.30
    minimum_synthetic_mismatch_gap: float = 0.05
    minimum_alternate_reference_prediction_cosine: float = 0.80
    minimum_actor_cosine_gain: float = 0.02
    minimum_macro_map_gain: float = 0.02
    maximum_signflip_p: float = 0.05
    confidence: float = 0.95
    bootstrap_samples: int = 5_000
    signflip_samples: int = 50_000

    def validate(self) -> None:
        for name in (
            "minimum_model_seeds",
            "minimum_positive_seed_directions",
            "minimum_paired_test_groups",
            "minimum_test_positive_groups",
            "minimum_unique_train_references",
            "bootstrap_samples",
            "signflip_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "minimum_reference_any_fraction",
            "minimum_reference_full_fraction",
            "maximum_reference_load_fraction",
            "maximum_signflip_p",
            "minimum_compatibility_auroc",
            "minimum_compatible_recall",
            "maximum_failed_outcome_fpr",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        for name in (
            "minimum_actor_cosine_gain",
            "minimum_macro_map_gain",
            "minimum_synthetic_mismatch_gap",
            "minimum_alternate_reference_prediction_cosine",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < float(self.confidence) < 1.0:
            raise ValueError("confidence must be in (0,1)")


def _stable_seed(base: int, label: str) -> int:
    digest = hashlib.sha256(f"{int(base)}\0{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _finite_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise R6GateInputError(f"{context} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise R6GateInputError(f"{context} must be a finite number") from error
    if not math.isfinite(result):
        raise R6GateInputError(f"{context} must be a finite number")
    return result


def _optional_metric(row: Mapping[str, Any], metric: str) -> float | None:
    if not bool(row.get("control_valid", True)):
        return None
    value = row.get(metric)
    if value is None:
        return None
    return _finite_float(value, context=f"{row.get('arm')}.{metric}")


def _validate_contract(
    contract: Mapping[str, Any],
    *,
    verified_pair_ledger_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise R6GateInputError("contract must be an object")
    if str(contract.get("schema_version")) != "motive-r6-training-v1":
        raise R6GateInputError("unsupported R6 training contract")
    model_seeds = contract.get("model_seeds")
    if (
        not isinstance(model_seeds, list)
        or not model_seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in model_seeds)
        or len(set(model_seeds)) != len(model_seeds)
    ):
        raise R6GateInputError("contract model_seeds must be unique integers")
    data_seed = contract.get("data_seed")
    if (
        isinstance(data_seed, bool)
        or not isinstance(data_seed, int)
        or data_seed < 0
    ):
        raise R6GateInputError("contract data_seed must be a non-negative integer")
    if contract.get("input_transform_fit_split") != "train":
        raise R6GateInputError(
            "R6 source/semantic input transforms must be fit on train only"
        )
    if contract.get("delta_transform_fit_split") != "train":
        raise R6GateInputError("R6 delta transform must be fit on train only")
    if contract.get("delta_transform_fit_role") != "positive_delta":
        raise R6GateInputError(
            "R6 delta transform must be fit on positive_delta rows only"
        )
    if contract.get("query_target_is_predictor_input") is not False:
        raise R6GateInputError("query target predictor input is forbidden")
    if contract.get("failed_outcomes_update_delta_predictor") is not False:
        raise R6GateInputError("failed outcomes may not update the delta predictor")
    if contract.get("compatibility_scales_conditioning_tokens") is not False:
        raise R6GateInputError("compatibility may not scale conditioning tokens")
    source_snapshot = contract.get("source_snapshot")
    if not isinstance(source_snapshot, Mapping):
        raise R6GateInputError("source_snapshot provenance is missing")
    for name in ("tree_sha256", "source_files_sha256"):
        if _SHA256_RE.fullmatch(
            str(source_snapshot.get(name) or "")
        ) is None:
            raise R6GateInputError(f"source_snapshot.{name} is invalid")
    for name in ("path", "source_files_manifest", "trainer_module_path"):
        if not str(source_snapshot.get(name) or "").strip():
            raise R6GateInputError(f"source_snapshot.{name} is empty")
    runtime = contract.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("deterministic_algorithms") is not True
        or not str(runtime.get("python_version") or "")
        or not str(runtime.get("numpy_version") or "")
        or not str(runtime.get("torch_version") or "")
        or not str(runtime.get("requested_device") or "")
    ):
        raise R6GateInputError("runtime/determinism provenance is incomplete")
    claim_scope = contract.get("claim_scope")
    if (
        not isinstance(claim_scope, Mapping)
        or claim_scope.get("generator_ready_tokens") is not False
        or claim_scope.get("motion_token_export_authorized") is not False
        or claim_scope.get("generation_authorized") is not False
    ):
        raise R6GateInputError(
            "R6 claim_scope must deny generator-ready/exported tokens"
        )
    dataset = contract.get("dataset")
    if not isinstance(dataset, Mapping) or not isinstance(
        dataset.get("action_family_source_verified"),
        bool,
    ):
        raise R6GateInputError(
            "dataset.action_family_source_verified must be boolean"
        )
    expected_row_count = dataset.get("row_count")
    expected_iid_set_digest = str(dataset.get("iid_set_digest") or "")
    expected_split_role_counts = dataset.get("split_role_counts")
    if (
        isinstance(expected_row_count, bool)
        or not isinstance(expected_row_count, int)
        or expected_row_count < 1
        or _SHA256_RE.fullmatch(expected_iid_set_digest) is None
        or not isinstance(expected_split_role_counts, Mapping)
    ):
        raise R6GateInputError(
            "dataset expected IID/count coverage contract is invalid"
        )
    normalized_split_role_counts: dict[str, int] = {}
    for key, value in expected_split_role_counts.items():
        if (
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise R6GateInputError(
                "dataset split_role_counts is invalid"
            )
        normalized_split_role_counts[key] = value
    evaluation = contract.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise R6GateInputError("evaluation contract is missing")
    active_threshold = _finite_float(
        evaluation.get("active_log_magnitude_threshold"),
        context="evaluation.active_log_magnitude_threshold",
    )
    if (
        abs(active_threshold - 1e-4) > 1e-12
        or evaluation.get("active_threshold_origin")
        != "fixed-pre-registered-1e-4"
        or evaluation.get("active_threshold_fit_scope") != "none"
    ):
        raise R6GateInputError(
            "active threshold must be the fixed pre-registered 1e-4"
        )
    if evaluation.get("family_retrieval_gate_eligible") is not False:
        raise R6GateInputError(
            "unverified family retrieval must be gate-ineligible"
        )

    semantic = contract.get("semantic_artifact")
    if not isinstance(semantic, Mapping):
        raise R6GateInputError("semantic_artifact is missing")
    if semantic.get("source_field") != "instruction":
        raise R6GateInputError("semantic artifact must be instruction-only")
    if semantic.get("frozen_encoder") is not True:
        raise R6GateInputError("semantic encoder must be frozen")
    if semantic.get("target_derived_input") is not False:
        raise R6GateInputError("target-derived semantic input is forbidden")
    if semantic.get("label_derived_input") is not False:
        raise R6GateInputError("label-derived semantic input is forbidden")
    semantic_digest = str(semantic.get("provenance_digest") or "")
    if _SHA256_RE.fullmatch(semantic_digest) is None:
        raise R6GateInputError("semantic provenance digest is invalid")

    selector = contract.get("reference_selector")
    if not isinstance(selector, Mapping):
        raise R6GateInputError("reference_selector is missing")
    exact = {
        "selector_kind": "prompt-to-observed-action-semantic-train-bank",
        "candidate_bank_split": "train",
        "candidate_bank_label_role": "positive_delta",
        "threshold_fit_split": "train",
        "threshold_fit_role": "positive_delta",
        "threshold_origin": "train-positive-self-alignment-q10",
        "query_target_used": False,
        "different_iid_enforced": True,
        "different_content_group_enforced": True,
        "oracle_action_family_used": False,
        "gate_eligible": True,
    }
    for name, expected in exact.items():
        if selector.get(name) != expected:
            raise R6GateInputError(
                f"reference_selector.{name} must equal {expected!r}"
            )
    if not isinstance(
        selector.get("different_subject_cluster_enforced"),
        bool,
    ):
        raise R6GateInputError(
            "reference_selector.different_subject_cluster_enforced "
            "must be boolean"
        )
    quantile = _finite_float(
        selector.get("threshold_quantile"),
        context="reference_selector.threshold_quantile",
    )
    if abs(quantile - 0.10) > 1e-12:
        raise R6GateInputError("reference threshold quantile must be 0.10")
    threshold = _finite_float(
        selector.get("similarity_threshold"),
        context="reference_selector.similarity_threshold",
    )
    if not -1.0 <= threshold <= 1.0:
        raise R6GateInputError("reference similarity threshold must be in [-1,1]")
    for name in (
        "pair_ledger_sha256",
        "pair_digest",
        "reference_bank_provenance_digest",
    ):
        if _SHA256_RE.fullmatch(str(selector.get(name) or "")) is None:
            raise R6GateInputError(f"reference_selector.{name} is invalid")
    if (
        verified_pair_ledger_sha256 is not None
        and selector["pair_ledger_sha256"]
        != verified_pair_ledger_sha256
    ):
        raise R6GateInputError(
            "reference pair ledger bytes disagree with contract SHA-256"
        )

    coverage = selector.get("test_positive_coverage")
    load = selector.get("reference_load")
    if not isinstance(coverage, Mapping) or not isinstance(load, Mapping):
        raise R6GateInputError("reference coverage/load audit is missing")
    normalized = {
        "data_seed": int(data_seed),
        "model_seeds": tuple(int(seed) for seed in model_seeds),
        "action_family_source_verified": bool(
            dataset["action_family_source_verified"]
        ),
        "expected_row_count": expected_row_count,
        "expected_iid_set_digest": expected_iid_set_digest,
        "expected_split_role_counts": normalized_split_role_counts,
        "selector": dict(selector),
        "subject_cluster_enforced": bool(
            selector["different_subject_cluster_enforced"]
        ),
        "coverage_any": _finite_float(
            coverage.get("any_reference_fraction"),
            context="test_positive_coverage.any_reference_fraction",
        ),
        "coverage_full": _finite_float(
            coverage.get("full_reference_fraction"),
            context="test_positive_coverage.full_reference_fraction",
        ),
        "coverage_eligible": int(coverage.get("eligible_queries", 0)),
        "unique_references": int(load.get("unique_reference_count", 0)),
        "maximum_load_fraction": _finite_float(
            load.get("maximum_reference_fraction"),
            context="reference_load.maximum_reference_fraction",
        ),
        "pair_ledger_integrity_verified": (
            verified_pair_ledger_sha256 is not None
        ),
    }
    if normalized["coverage_eligible"] < 0 or normalized["unique_references"] < 0:
        raise R6GateInputError("reference audit counts must be non-negative")
    for value in (
        normalized["coverage_any"],
        normalized["coverage_full"],
        normalized["maximum_load_fraction"],
    ):
        if not 0.0 <= value <= 1.0:
            raise R6GateInputError("reference audit fractions must be in [0,1]")
    return normalized


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_seeds: Sequence[int],
    expected_row_count: int,
    expected_iid_set_digest: str,
    expected_split_role_counts: Mapping[str, int],
) -> dict[str, Any]:
    if not rows:
        raise R6GateInputError("per-query evidence is empty")
    allowed = R6_REQUIRED_ARMS | R6_ORACLE_ARMS
    coverage: dict[str, set[tuple[int, str]]] = {
        arm: set() for arm in allowed
    }
    metadata: dict[tuple[int, str], tuple[str, str, str]] = {}
    iid_metadata: dict[str, tuple[str, str, str]] = {}
    oracle_count = 0
    for position, row in enumerate(rows):
        if row.get("schema_version") != R6_QUERY_SCHEMA:
            raise R6GateInputError(f"row {position} has unsupported schema")
        arm = str(row.get("arm") or "")
        if arm not in allowed:
            raise R6GateInputError(f"row {position} has unexpected arm={arm!r}")
        seed = row.get("model_seed")
        iid = str(row.get("iid") or "")
        split = str(row.get("split") or "")
        group = str(row.get("content_group_id") or "")
        role = str(row.get("label_role") or "")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed not in model_seeds
            or not iid
            or split not in {"train", "validation", "test"}
            or not group
            or role not in {"positive_delta", "failed_outcome_compatibility"}
        ):
            raise R6GateInputError(f"row {position} metadata is invalid")
        key = (seed, iid)
        if key in coverage[arm]:
            raise R6GateInputError(f"duplicate row for arm/seed/iid={arm}/{seed}/{iid}")
        coverage[arm].add(key)
        invariant = (split, group, role)
        if key in metadata and metadata[key] != invariant:
            raise R6GateInputError(f"row metadata changes across arms for {key}")
        metadata[key] = invariant
        if iid in iid_metadata and iid_metadata[iid] != invariant:
            raise R6GateInputError(
                f"row metadata changes across seeds for iid={iid}"
            )
        iid_metadata[iid] = invariant
        compatibility_target = _finite_float(
            row.get("compatibility_target"),
            context="compatibility_target",
        )
        expected_compatibility = (
            1.0 if role == "positive_delta" else 0.0
        )
        if compatibility_target != expected_compatibility:
            raise R6GateInputError(
                f"compatibility target/label role mismatch for iid={iid}"
            )
        compatibility_probability = _finite_float(
            row.get("compatibility_probability"),
            context="compatibility_probability",
        )
        if not 0.0 <= compatibility_probability <= 1.0:
            raise R6GateInputError(
                f"compatibility probability is outside [0,1] for iid={iid}"
            )
        synthetic_probability = row.get(
            "synthetic_mismatched_positive_probability"
        )
        if synthetic_probability is not None:
            synthetic_value = _finite_float(
                synthetic_probability,
                context="synthetic_mismatched_positive_probability",
            )
            if not 0.0 <= synthetic_value <= 1.0:
                raise R6GateInputError(
                    "synthetic mismatch probability is outside [0,1]"
                )
        if arm in R6_ORACLE_ARMS:
            oracle_count += 1
            if row.get("oracle_diagnostic") is not True:
                raise R6GateInputError("exact target rows must be tagged oracle_diagnostic")
            if row.get("gate_eligible") is not False:
                raise R6GateInputError("exact target rows must be gate-ineligible")
        else:
            if row.get("oracle_diagnostic") is not False:
                raise R6GateInputError(f"gate arm {arm} cannot be oracle-tagged")
            if row.get("gate_eligible") is not True:
                raise R6GateInputError(f"gate arm {arm} must be gate-eligible")
            safety_flags = {
                "query_target_used_as_predictor_input": False,
                "failed_outcome_used_as_noop": False,
                "compatibility_scales_conditioning_tokens": False,
            }
            for name, expected in safety_flags.items():
                if row.get(name) is not expected:
                    raise R6GateInputError(
                        f"gate row {arm}/{seed}/{iid} {name} "
                        f"must be {expected}"
                    )

    reference = coverage["semantic_only"]
    for arm in sorted(R6_REQUIRED_ARMS):
        if coverage[arm] != reference:
            raise R6GateInputError(
                f"{arm} coverage differs from semantic_only "
                f"(missing={len(reference - coverage[arm])}, "
                f"extra={len(coverage[arm] - reference)})"
            )
    iid_values = sorted(iid_metadata)
    if len(iid_values) != int(expected_row_count):
        raise R6GateInputError(
            "per-query unique IID count differs from dataset contract"
        )
    if _canonical_digest(iid_values) != expected_iid_set_digest:
        raise R6GateInputError(
            "per-query IID set digest differs from dataset contract"
        )
    observed_split_role: dict[str, int] = {}
    for split, _, role in iid_metadata.values():
        name = f"{split}:{role}"
        observed_split_role[name] = observed_split_role.get(name, 0) + 1
    if observed_split_role != dict(expected_split_role_counts):
        raise R6GateInputError(
            "per-query split/role counts differ from dataset contract"
        )
    expected_iids = set(iid_values)
    for seed in model_seeds:
        seed_iids = {
            iid for observed_seed, iid in reference if observed_seed == seed
        }
        if seed_iids != expected_iids:
            raise R6GateInputError(
                f"semantic_only seed={seed} IID coverage is incomplete"
            )
    return {
        "query_seed_count": len(reference),
        "oracle_row_count": oracle_count,
        "oracle_excluded_from_gate": True,
        "unique_iid_count": len(iid_values),
        "iid_set_digest": expected_iid_set_digest,
        "split_role_counts": observed_split_role,
    }


def hierarchical_paired_group_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    treatment_arm: str,
    control_arm: str,
    metric: str,
    split: str = "test",
    bootstrap_samples: int = 5_000,
    signflip_samples: int = 50_000,
    confidence: float = 0.95,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Crossed seed/group bootstrap plus paired group-level sign flip.

    The same content groups recur across model seeds, so groups are resampled
    once per draw (shared across sampled seeds), rather than independently
    within each seed.  Significance is computed by sign-flipping each
    group-level mean after averaging its paired effects across seeds.
    """

    if treatment_arm not in R6_REQUIRED_ARMS or control_arm not in R6_REQUIRED_ARMS:
        raise ValueError("oracle or unknown arm cannot enter an R6 comparison")
    if split not in {"validation", "test"}:
        raise ValueError("R6 gate comparisons are validation/test only")
    by_key: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        if (
            str(row.get("arm")) not in {treatment_arm, control_arm}
            or str(row.get("split")) != split
            or str(row.get("label_role")) != "positive_delta"
        ):
            continue
        key = (int(row["model_seed"]), str(row["iid"]), str(row["arm"]))
        by_key[key] = row
    seed_group_values: dict[int, dict[str, list[float]]] = {}
    paired_queries = 0
    for seed, iid, arm in sorted(by_key):
        if arm != treatment_arm:
            continue
        treatment = by_key[(seed, iid, treatment_arm)]
        control = by_key.get((seed, iid, control_arm))
        if control is None:
            continue
        lhs = _optional_metric(treatment, metric)
        rhs = _optional_metric(control, metric)
        if lhs is None or rhs is None:
            continue
        lhs_group = str(treatment["content_group_id"])
        if lhs_group != str(control["content_group_id"]):
            raise R6GateInputError("paired rows disagree on content group")
        seed_group_values.setdefault(seed, {}).setdefault(lhs_group, []).append(lhs - rhs)
        paired_queries += 1
    collapsed: dict[int, dict[str, float]] = {
        seed: {group: float(np.mean(values)) for group, values in groups.items()}
        for seed, groups in seed_group_values.items()
    }
    seed_means = {
        str(seed): float(np.mean(list(groups.values())))
        for seed, groups in collapsed.items()
        if groups
    }
    point = float(np.mean(list(seed_means.values()))) if seed_means else None
    result: dict[str, Any] = {
        "treatment_arm": treatment_arm,
        "control_arm": control_arm,
        "metric": metric,
        "split": split,
        "paired_queries": paired_queries,
        "paired_seeds": len(seed_means),
        "paired_groups": len(
            {group for groups in collapsed.values() for group in groups}
        ),
        "seed_means": seed_means,
        "positive_seed_count": sum(value > 0.0 for value in seed_means.values()),
        "mean_gain": point,
        "confidence_interval": None,
        "bootstrap_tail_fraction_le_zero_diagnostic": None,
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_design": (
            "crossed-two-way-resample-shared-groups-and-model-seeds"
        ),
        "signflip_p": None,
        "signflip_samples": int(signflip_samples),
        "signflip_method": None,
    }
    if point is None:
        return result
    rng = np.random.default_rng(
        _stable_seed(
            random_seed,
            f"{treatment_arm}:{control_arm}:{metric}:{split}",
        )
    )
    seeds = np.asarray(sorted(collapsed), dtype=np.int64)
    groups = np.asarray(
        sorted({group for values in collapsed.values() for group in values}),
        dtype=object,
    )
    if bootstrap_samples:
        samples = np.empty(int(bootstrap_samples), dtype=np.float64)
        for draw in range(int(bootstrap_samples)):
            sampled_seeds = rng.choice(
                seeds,
                size=len(seeds),
                replace=True,
            )
            sampled_groups = rng.choice(
                groups,
                size=len(groups),
                replace=True,
            )
            cells = [
                collapsed[int(seed)][str(group)]
                for seed in sampled_seeds
                for group in sampled_groups
                if str(group) in collapsed[int(seed)]
            ]
            samples[draw] = float(np.mean(cells))
        alpha = 1.0 - float(confidence)
        result["confidence_interval"] = [
            float(np.quantile(samples, alpha / 2.0)),
            float(np.quantile(samples, 1.0 - alpha / 2.0)),
        ]
        result["bootstrap_tail_fraction_le_zero_diagnostic"] = float(
            (1 + np.count_nonzero(samples <= 0.0)) / (len(samples) + 1)
        )

    group_means = np.asarray(
        [
            np.mean(
                [
                    collapsed[seed][str(group)]
                    for seed in collapsed
                    if str(group) in collapsed[seed]
                ]
            )
            for group in groups
        ],
        dtype=np.float64,
    )
    observed = float(np.mean(group_means))
    if len(group_means) <= 20:
        total = 1 << len(group_means)
        extreme = 0
        for pattern in range(total):
            signed = np.asarray(
                [
                    value if (pattern >> index) & 1 else -value
                    for index, value in enumerate(group_means)
                ],
                dtype=np.float64,
            )
            if float(np.mean(signed)) >= observed - 1e-15:
                extreme += 1
        result["signflip_p"] = float(extreme / total)
        result["signflip_samples"] = total
        result["signflip_method"] = "exact-group-level"
    elif signflip_samples:
        sign_rng = np.random.default_rng(
            _stable_seed(
                random_seed,
                f"signflip:{treatment_arm}:{control_arm}:{metric}:{split}",
            )
        )
        extreme = 0
        for _ in range(int(signflip_samples)):
            signs = sign_rng.choice(
                np.asarray([-1.0, 1.0]),
                size=len(group_means),
                replace=True,
            )
            if float(np.mean(group_means * signs)) >= observed - 1e-15:
                extreme += 1
        result["signflip_p"] = float(
            (extreme + 1) / (int(signflip_samples) + 1)
        )
        result["signflip_method"] = "monte-carlo-group-level"
    return result


def _comparison_passes(
    comparison: Mapping[str, Any],
    *,
    minimum_gain: float,
    thresholds: R6PilotThresholds,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if int(comparison["paired_seeds"]) < thresholds.minimum_model_seeds:
        failures.append("insufficient paired model seeds")
    if int(comparison["paired_groups"]) < thresholds.minimum_paired_test_groups:
        failures.append("insufficient paired test groups")
    if int(comparison["positive_seed_count"]) < thresholds.minimum_positive_seed_directions:
        failures.append("insufficient positive seed directions")
    gain = comparison.get("mean_gain")
    if gain is None or float(gain) < float(minimum_gain):
        failures.append(f"mean gain below {minimum_gain:.6g}")
    interval = comparison.get("confidence_interval")
    if not isinstance(interval, list) or len(interval) != 2 or float(interval[0]) <= 0.0:
        failures.append("bootstrap confidence interval crosses zero")
    p_value = comparison.get("signflip_p")
    if p_value is None or float(p_value) > thresholds.maximum_signflip_p:
        failures.append("paired group-level sign-flip p-value too large")
    return not failures, failures


def _binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels == 1
    negative = labels == 0
    if not bool(positive.any()) or not bool(negative.any()):
        raise R6GateInputError(
            "compatibility AUROC requires positive and failed-outcome rows"
        )
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        average_rank = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = average_rank
        start = stop
    positives = int(np.count_nonzero(positive))
    negatives = int(np.count_nonzero(negative))
    rank_sum = float(np.sum(ranks[positive]))
    return (
        rank_sum - positives * (positives + 1) / 2.0
    ) / float(positives * negatives)


def compatibility_diagnostic(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str = "test",
    minimum_recall: float = 0.80,
) -> dict[str, Any]:
    """Lock an operating threshold on validation and evaluate it on test."""

    per_seed: dict[str, Any] = {}
    test_rows = [
        row
        for row in rows
        if row.get("arm") == "semantic_only"
        and row.get("split") == split
    ]
    validation_rows = [
        row
        for row in rows
        if row.get("arm") == "semantic_only"
        and row.get("split") == "validation"
    ]
    seeds = sorted(
        {
            int(row["model_seed"])
            for row in test_rows + validation_rows
        }
    )

    def arrays(
        selected: Sequence[Mapping[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        label_values: list[int] = []
        for row in selected:
            target = _finite_float(
                row.get("compatibility_target"),
                context="compatibility_target",
            )
            if target not in {0.0, 1.0}:
                raise R6GateInputError(
                    "compatibility_target must be exactly binary"
                )
            expected = (
                1.0
                if row.get("label_role") == "positive_delta"
                else 0.0
                if row.get("label_role")
                == "failed_outcome_compatibility"
                else None
            )
            if expected is None or target != expected:
                raise R6GateInputError(
                    "compatibility target/label role mismatch"
                )
            label_values.append(int(target))
        labels = np.asarray(label_values, dtype=np.int64)
        scores = np.asarray(
            [
                _finite_float(
                    row.get("compatibility_probability"),
                    context="compatibility_probability",
                )
                for row in selected
            ],
            dtype=np.float64,
        )
        if bool(((labels != 0) & (labels != 1)).any()):
            raise R6GateInputError("compatibility_target must be binary")
        if bool(((scores < 0.0) | (scores > 1.0)).any()):
            raise R6GateInputError(
                "compatibility_probability must be in [0,1]"
            )
        return labels, scores

    for seed in seeds:
        validation_seed_rows = [
            row
            for row in validation_rows
            if int(row["model_seed"]) == seed
        ]
        test_seed_rows = [
            row for row in test_rows if int(row["model_seed"]) == seed
        ]
        validation_labels, validation_scores = arrays(
            validation_seed_rows
        )
        test_labels, test_scores = arrays(test_seed_rows)
        validation_positive_count = int(
            np.count_nonzero(validation_labels == 1)
        )
        validation_negative_count = int(
            np.count_nonzero(validation_labels == 0)
        )
        test_positive_count = int(np.count_nonzero(test_labels == 1))
        test_negative_count = int(np.count_nonzero(test_labels == 0))
        if (
            not validation_positive_count
            or not validation_negative_count
            or not test_positive_count
            or not test_negative_count
        ):
            per_seed[str(seed)] = {
                "validation_positive_count": validation_positive_count,
                "validation_failed_outcome_count": (
                    validation_negative_count
                ),
                "test_positive_count": test_positive_count,
                "test_failed_outcome_count": test_negative_count,
                "auroc": None,
                "compatible_recall": None,
                "failed_outcome_fpr": None,
                "operating_threshold": None,
                "incomplete_reason": (
                    "validation and test must each contain both real classes"
                ),
            }
            continue
        thresholds = np.unique(validation_scores)[::-1]
        operating: tuple[float, float, float] | None = None
        for threshold in thresholds:
            predicted = validation_scores >= threshold
            recall = float(
                np.count_nonzero(predicted & (validation_labels == 1))
                / validation_positive_count
            )
            fpr = float(
                np.count_nonzero(predicted & (validation_labels == 0))
                / validation_negative_count
            )
            if recall + 1e-12 >= float(minimum_recall):
                candidate = (fpr, -float(threshold), recall)
                if operating is None or candidate < operating:
                    operating = candidate
        assert operating is not None
        locked_threshold = float(-operating[1])
        test_predicted = test_scores >= locked_threshold
        test_recall = float(
            np.count_nonzero(test_predicted & (test_labels == 1))
            / test_positive_count
        )
        test_fpr = float(
            np.count_nonzero(test_predicted & (test_labels == 0))
            / test_negative_count
        )
        per_seed[str(seed)] = {
            "validation_positive_count": validation_positive_count,
            "validation_failed_outcome_count": validation_negative_count,
            "test_positive_count": test_positive_count,
            "test_failed_outcome_count": test_negative_count,
            "validation_compatible_recall": float(operating[2]),
            "validation_failed_outcome_fpr": float(operating[0]),
            "auroc": _binary_auroc(test_labels, test_scores),
            "compatible_recall": test_recall,
            "failed_outcome_fpr": test_fpr,
            "operating_threshold": locked_threshold,
            "operating_point_origin": (
                f"validation-roc-min-fpr-at-recall-ge-{minimum_recall:.2f}"
            ),
        }
    complete = [
        value for value in per_seed.values() if value["auroc"] is not None
    ]

    def mean(name: str) -> float | None:
        return (
            float(np.mean([float(value[name]) for value in complete]))
            if complete
            else None
        )

    mismatch_by_seed: dict[int, list[float]] = {}
    for row in test_rows:
        if (
            row.get("label_role") == "positive_delta"
            and row.get("synthetic_mismatched_positive_probability")
            is not None
        ):
            observed_probability = _finite_float(
                row.get("compatibility_probability"),
                context="compatibility_probability",
            )
            mismatch_probability = _finite_float(
                row.get("synthetic_mismatched_positive_probability"),
                context="synthetic_mismatched_positive_probability",
            )
            if (
                not 0.0 <= observed_probability <= 1.0
                or not 0.0 <= mismatch_probability <= 1.0
            ):
                raise R6GateInputError(
                    "compatibility probabilities must be in [0,1]"
                )
            mismatch_by_seed.setdefault(
                int(row["model_seed"]),
                [],
            ).append(
                observed_probability - mismatch_probability
            )
    mismatch_seed_means = {
        str(seed): float(np.mean(values))
        for seed, values in sorted(mismatch_by_seed.items())
        if values
    }
    return {
        "split": split,
        "operating_threshold_fit_split": "validation",
        "per_seed": per_seed,
        "complete_seed_count": len(complete),
        "both_classes_present_every_seed": (
            bool(per_seed) and len(complete) == len(per_seed)
        ),
        "mean_auroc": mean("auroc"),
        "mean_compatible_recall": mean("compatible_recall"),
        "mean_failed_outcome_fpr": mean("failed_outcome_fpr"),
        "synthetic_mismatch": {
            "definition": (
                "deterministic within-split prompt/other-positive-motion pair; "
                "not a real failed outcome and excluded from AUROC/FPR"
            ),
            "pair_count": sum(
                len(values) for values in mismatch_by_seed.values()
            ),
            "seed_means": mismatch_seed_means,
            "positive_seed_count": sum(
                value > 0.0 for value in mismatch_seed_means.values()
            ),
            "mean_observed_minus_mismatch_probability": (
                float(np.mean(list(mismatch_seed_means.values())))
                if mismatch_seed_means
                else None
            ),
        },
    }


def alternate_reference_stability(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize rank-0 versus rank-1 independent-reference predictions."""

    selected = [
        row
        for row in rows
        if row.get("arm") == "independent_ref"
        and row.get("split") == "test"
        and row.get("label_role") == "positive_delta"
    ]
    available = [
        row
        for row in selected
        if row.get("alternate_reference_available") is True
        and row.get("alternate_reference_prediction_cosine") is not None
    ]
    seed_group: dict[int, dict[str, list[float]]] = {}
    for row in available:
        seed_group.setdefault(
            int(row["model_seed"]),
            {},
        ).setdefault(
            str(row["content_group_id"]),
            [],
        ).append(
            _finite_float(
                row["alternate_reference_prediction_cosine"],
                context="alternate_reference_prediction_cosine",
            )
        )
    seed_means = {
        str(seed): float(
            np.mean(
                [
                    np.mean(values)
                    for values in groups.values()
                ]
            )
        )
        for seed, groups in sorted(seed_group.items())
        if groups
    }
    return {
        "definition": (
            "actor prediction cosine using rank-0 versus rank-1 primary "
            "semantic-selector references; neither uses query target"
        ),
        "eligible_rows": len(selected),
        "available_rows": len(available),
        "coverage_fraction": (
            float(len(available)) / float(len(selected))
            if selected
            else 0.0
        ),
        "seed_means": seed_means,
        "complete_seed_count": len(seed_means),
        "mean_prediction_cosine": (
            float(np.mean(list(seed_means.values())))
            if seed_means
            else None
        ),
    }


def evaluate_r6_gate(
    *,
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    thresholds: R6PilotThresholds | None = None,
    random_seed: int = 0,
    verified_pair_ledger_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate R6 evidence while keeping the formal decision fail-closed."""

    limits = thresholds or R6PilotThresholds()
    limits.validate()
    normalized = _validate_contract(
        contract,
        verified_pair_ledger_sha256=verified_pair_ledger_sha256,
    )
    row_audit = _validate_rows(
        rows,
        model_seeds=normalized["model_seeds"],
        expected_row_count=normalized["expected_row_count"],
        expected_iid_set_digest=normalized["expected_iid_set_digest"],
        expected_split_role_counts=normalized[
            "expected_split_role_counts"
        ],
    )

    family_metric_required = normalized["action_family_source_verified"]
    comparisons_spec = (
        ("independent_ref", "semantic_only", "actor_cosine", limits.minimum_actor_cosine_gain, True),
        ("independent_ref", "wrong_ref", "actor_cosine", limits.minimum_actor_cosine_gain, True),
        ("semantic_only", "semantic_shuffle", "actor_cosine", limits.minimum_actor_cosine_gain, True),
        ("semantic_only", "centroid", "actor_cosine", limits.minimum_actor_cosine_gain, True),
        ("semantic_only", "matched_random", "actor_cosine", limits.minimum_actor_cosine_gain, True),
        ("semantic_only", "source_shuffle", "actor_cosine", limits.minimum_actor_cosine_gain, True),
        (
            "independent_ref",
            "semantic_only",
            "actor_cross_content_ap",
            limits.minimum_macro_map_gain,
            family_metric_required,
        ),
        (
            "semantic_only",
            "semantic_shuffle",
            "actor_cross_content_ap",
            limits.minimum_macro_map_gain,
            family_metric_required,
        ),
    )
    comparisons: dict[str, Any] = {}
    criteria: list[dict[str, Any]] = []
    for treatment, control, metric, minimum, required in comparisons_spec:
        key = f"{treatment}_vs_{control}:{metric}"
        comparison = hierarchical_paired_group_bootstrap(
            rows,
            treatment_arm=treatment,
            control_arm=control,
            metric=metric,
            bootstrap_samples=limits.bootstrap_samples,
            signflip_samples=limits.signflip_samples,
            confidence=limits.confidence,
            random_seed=random_seed,
        )
        passed, failures = _comparison_passes(
            comparison,
            minimum_gain=minimum,
            thresholds=limits,
        )
        comparisons[key] = comparison
        criteria.append(
            {
                "name": key,
                "passed": passed,
                "required": bool(required),
                "minimum_gain": float(minimum),
                "failures": failures,
                "descriptive_only_reason": (
                    None
                    if required
                    else (
                        "auto/Qwen action families are unverified; "
                        "family retrieval cannot affect the pilot decision"
                    )
                ),
            }
        )

    coverage_checks = (
        (
            "reference_any_coverage",
            normalized["coverage_any"] >= limits.minimum_reference_any_fraction,
            normalized["coverage_any"],
            limits.minimum_reference_any_fraction,
            ">=",
        ),
        (
            "reference_full_coverage",
            normalized["coverage_full"] >= limits.minimum_reference_full_fraction,
            normalized["coverage_full"],
            limits.minimum_reference_full_fraction,
            ">=",
        ),
        (
            "unique_train_references",
            normalized["unique_references"] >= limits.minimum_unique_train_references,
            normalized["unique_references"],
            limits.minimum_unique_train_references,
            ">=",
        ),
        (
            "maximum_reference_load",
            normalized["maximum_load_fraction"] <= limits.maximum_reference_load_fraction,
            normalized["maximum_load_fraction"],
            limits.maximum_reference_load_fraction,
            "<=",
        ),
    )
    for name, passed, observed, required, operator in coverage_checks:
        criteria.append(
            {
                "name": name,
                "passed": bool(passed),
                "required": True,
                "observed": observed,
                "threshold": required,
                "operator": operator,
                "failures": [] if passed else ["reference coverage/load threshold failed"],
            }
        )

    compatibility = compatibility_diagnostic(
        rows,
        split="test",
        minimum_recall=limits.minimum_compatible_recall,
    )
    compatibility_complete = (
        compatibility["both_classes_present_every_seed"]
        and compatibility["complete_seed_count"]
        >= limits.minimum_model_seeds
    )
    compatibility_checks = (
        (
            "compatibility_test_both_classes",
            compatibility_complete,
            compatibility["complete_seed_count"],
            limits.minimum_model_seeds,
            ">=",
        ),
        (
            "compatibility_test_auroc",
            (
                compatibility["mean_auroc"] is not None
                and compatibility["mean_auroc"]
                >= limits.minimum_compatibility_auroc
            ),
            compatibility["mean_auroc"],
            limits.minimum_compatibility_auroc,
            ">=",
        ),
        (
            "compatibility_test_compatible_recall",
            (
                compatibility["mean_compatible_recall"] is not None
                and compatibility["mean_compatible_recall"]
                >= limits.minimum_compatible_recall
            ),
            compatibility["mean_compatible_recall"],
            limits.minimum_compatible_recall,
            ">=",
        ),
        (
            "compatibility_test_failed_outcome_fpr",
            (
                compatibility["mean_failed_outcome_fpr"] is not None
                and compatibility["mean_failed_outcome_fpr"]
                <= limits.maximum_failed_outcome_fpr
            ),
            compatibility["mean_failed_outcome_fpr"],
            limits.maximum_failed_outcome_fpr,
            "<=",
        ),
        (
            "compatibility_synthetic_mismatch_specificity",
            (
                compatibility["synthetic_mismatch"]["positive_seed_count"]
                >= limits.minimum_positive_seed_directions
                and compatibility["synthetic_mismatch"][
                    "mean_observed_minus_mismatch_probability"
                ]
                is not None
                and compatibility["synthetic_mismatch"][
                    "mean_observed_minus_mismatch_probability"
                ]
                >= limits.minimum_synthetic_mismatch_gap
            ),
            compatibility["synthetic_mismatch"][
                "mean_observed_minus_mismatch_probability"
            ],
            limits.minimum_synthetic_mismatch_gap,
            ">=",
        ),
    )
    for name, passed, observed, required, operator in compatibility_checks:
        criteria.append(
            {
                "name": name,
                "passed": bool(passed),
                "required": True,
                "observed": observed,
                "threshold": required,
                "operator": operator,
                "failures": [] if passed else ["compatibility diagnostic failed"],
            }
        )

    alternate_stability = alternate_reference_stability(rows)
    alternate_complete = (
        alternate_stability["eligible_rows"] > 0
        and alternate_stability["coverage_fraction"] == 1.0
        and alternate_stability["complete_seed_count"]
        >= limits.minimum_model_seeds
    )
    alternate_passed = (
        alternate_complete
        and alternate_stability["mean_prediction_cosine"] is not None
        and alternate_stability["mean_prediction_cosine"]
        >= limits.minimum_alternate_reference_prediction_cosine
    )
    criteria.append(
        {
            "name": "alternate_retrieved_reference_stability",
            "passed": bool(alternate_passed),
            "required": True,
            "observed": alternate_stability["mean_prediction_cosine"],
            "threshold": (
                limits.minimum_alternate_reference_prediction_cosine
            ),
            "operator": ">=",
            "coverage_fraction": alternate_stability[
                "coverage_fraction"
            ],
            "failures": (
                []
                if alternate_passed
                else [
                    "rank-1 reference coverage is incomplete or rank-0/rank-1 "
                    "predictions are unstable"
                ]
            ),
        }
    )

    test_groups = {
        str(row["content_group_id"])
        for row in rows
        if row["arm"] == "semantic_only"
        and row["split"] == "test"
        and row["label_role"] == "positive_delta"
    }
    enough_groups = len(test_groups) >= limits.minimum_test_positive_groups
    criteria.append(
        {
            "name": "minimum_test_positive_groups",
            "passed": enough_groups,
            "required": True,
            "observed": len(test_groups),
            "threshold": limits.minimum_test_positive_groups,
            "operator": ">=",
            "failures": [] if enough_groups else ["too few test positive groups"],
        }
    )
    enough_seeds = len(normalized["model_seeds"]) >= limits.minimum_model_seeds
    criteria.append(
        {
            "name": "minimum_model_seeds",
            "passed": enough_seeds,
            "required": True,
            "observed": len(normalized["model_seeds"]),
            "threshold": limits.minimum_model_seeds,
            "operator": ">=",
            "failures": [] if enough_seeds else ["too few model seeds"],
        }
    )
    subject_verified = normalized["subject_cluster_enforced"]
    criteria.append(
        {
            "name": "verified_cross_subject_reference_exclusion",
            "passed": subject_verified,
            "required": True,
            "observed": subject_verified,
            "threshold": True,
            "operator": "==",
            "failures": (
                []
                if subject_verified
                else [
                    "subject clusters are unavailable; content-group "
                    "surrogates do not prove cross-subject exclusion"
                ]
            ),
        }
    )
    pilot_complete = (
        enough_seeds
        and enough_groups
        and compatibility_complete
        and alternate_complete
        and subject_verified
    )
    pilot_status = (
        "GO"
        if pilot_complete
        and all(item["passed"] for item in criteria if item["required"])
        else "NO_GO"
        if pilot_complete
        else "INSUFFICIENT"
    )
    return {
        "schema_version": R6_GATE_SCHEMA,
        "status": "INSUFFICIENT",
        "production_decision": False,
        "generation_authorized": False,
        "formal_gate": {
            "status": "INSUFFICIENT",
            "reason": (
                "R6 is a representation diagnostic; generation/editing requires "
                "human intent labels and a separate causal intervention gate"
            ),
        },
        "pilot_diagnostic": {
            "status": pilot_status,
            "decision_is_production_eligible": False,
            "all_required_criteria_passed": all(
                item["passed"] for item in criteria if item["required"]
            ),
            "criteria": criteria,
        },
        "thresholds": asdict(limits),
        "comparisons": comparisons,
        "reference_selector_audit": {
            "selector_kind": normalized["selector"]["selector_kind"],
            "pair_digest": normalized["selector"]["pair_digest"],
            "pair_ledger_sha256": normalized["selector"]["pair_ledger_sha256"],
            "coverage_any": normalized["coverage_any"],
            "coverage_full": normalized["coverage_full"],
            "unique_reference_count": normalized["unique_references"],
            "maximum_reference_fraction": normalized["maximum_load_fraction"],
            "query_target_used": False,
            "oracle_action_family_used": False,
            "different_subject_cluster_enforced": subject_verified,
            "pair_ledger_integrity_verified": normalized[
                "pair_ledger_integrity_verified"
            ],
            "integrity_scope": (
                "pair-ledger-bytes-verified"
                if normalized["pair_ledger_integrity_verified"]
                else "self-attested-contract-only"
            ),
        },
        "compatibility_diagnostic": compatibility,
        "alternate_reference_stability": alternate_stability,
        "row_audit": row_audit,
        "test_positive_group_count": len(test_groups),
        "oracle_diagnostic": {
            "row_count": row_audit["oracle_row_count"],
            "excluded_from_all_gate_criteria": True,
        },
    }


def invalid_r6_gate_summary(error: BaseException | str) -> dict[str, Any]:
    return {
        "schema_version": R6_GATE_SCHEMA,
        "status": "INVALID",
        "production_decision": False,
        "generation_authorized": False,
        "formal_gate": {"status": "INVALID"},
        "pilot_diagnostic": {
            "status": "INVALID",
            "decision_is_production_eligible": False,
        },
        "error": str(error),
    }


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-query", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--pair-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help=(
            "bootstrap/sign-flip RNG seed; defaults to the immutable "
            "training contract data_seed"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        contract = _load_json(
            args.contract.expanduser().resolve(strict=True)
        )
        random_seed = (
            contract.get("data_seed")
            if args.random_seed is None
            else args.random_seed
        )
        thresholds = R6PilotThresholds(
            bootstrap_samples=int(args.bootstrap_samples)
        )
        summary = evaluate_r6_gate(
            rows=_load_jsonl(args.per_query.expanduser().resolve(strict=True)),
            contract=contract,
            thresholds=thresholds,
            random_seed=int(random_seed),
            verified_pair_ledger_sha256=_file_digest(
                args.pair_ledger.expanduser().resolve(strict=True)
            ),
        )
    except Exception as error:
        summary = invalid_r6_gate_summary(error)
    _atomic_json(args.output.expanduser(), summary)
    print(f"[r6-gate] status={summary['status']} output={args.output.expanduser()}")
    return 0 if summary["status"] != "INVALID" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
