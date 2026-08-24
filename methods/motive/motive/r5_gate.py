"""Statistics and fail-closed decision gate for R5-lite.

The gate intentionally separates *diagnostic evidence* from a production
decision.  A legacy-Qwen pilot or a perceptual-hash-only split may produce all
metrics, but can never return ``PASS``.  Production eligibility requires human
labels and a source/subject/scene visual-cluster split in addition to the
pre-registered sample-size and statistical criteria.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


R5_GATE_SCHEMA = "motive-r5-representation-gate-v1"
R5_QUERY_SCHEMA = "motive-r5-per-query-v1"
R5_PRODUCTION_SPLIT_VERSION = "source-visual-cluster-v1"
R5_GATE_STATUSES = frozenset({"PASS", "NO_GO", "INSUFFICIENT", "INVALID"})
REQUIRED_ARMS = frozenset(
    {
        "full",
        "text_only",
        "pairshuffle",
        "matched_random",
        "centroid",
        "source_shuffle",
        "prompt_shuffle",
    }
)


class GateInputError(ValueError):
    """Raised when an R5 result cannot be interpreted without guessing."""


@dataclass(frozen=True)
class R5GateThresholds:
    minimum_human_positives: int = 2_000
    minimum_positive_groups: int = 1_000
    minimum_action_families: int = 8
    minimum_test_positive_groups: int = 200
    minimum_model_seeds: int = 5
    minimum_positive_seed_directions: int = 4
    minimum_actor_cosine_gain: float = 0.05
    minimum_macro_map_gain: float = 0.05
    maximum_signflip_p: float = 0.01
    confidence: float = 0.95
    bootstrap_samples: int = 20_000
    signflip_samples: int = 50_000

    def validate(self) -> None:
        integer_names = (
            "minimum_human_positives",
            "minimum_positive_groups",
            "minimum_action_families",
            "minimum_test_positive_groups",
            "minimum_model_seeds",
            "minimum_positive_seed_directions",
            "bootstrap_samples",
            "signflip_samples",
        )
        for name in integer_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0,1)")
        if not 0.0 <= self.maximum_signflip_p <= 1.0:
            raise ValueError("maximum_signflip_p must be in [0,1]")
        for name in ("minimum_actor_cosine_gain", "minimum_macro_map_gain"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")


def _stable_seed(base: int, label: str) -> int:
    digest = hashlib.sha256(
        f"{int(base)}\0{label}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _finite_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise GateInputError(f"{context} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise GateInputError(f"{context} must be a finite number") from error
    if not math.isfinite(result):
        raise GateInputError(f"{context} must be a finite number")
    return result


def _unit_rows(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must have shape [N,D]")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(
        matrix,
        np.maximum(norms, 1e-12),
        out=np.zeros_like(matrix),
        where=norms > 1e-12,
    )


def cross_content_retrieval(
    *,
    predicted_actor_direction: np.ndarray,
    target_actor_direction: np.ndarray,
    action_families: Sequence[str],
    content_group_ids: Sequence[str],
    iids: Sequence[str],
    valid_mask: Sequence[bool] | None = None,
    active_mask: Sequence[bool] | None = None,
) -> list[dict[str, Any]]:
    """Compute query-level cross-content AP/R@1/R@5.

    The gallery is the same evaluation split supplied by the caller.  A
    candidate is relevant when it has the same action family and belongs to a
    different content group.  Queries without a cross-content positive are
    retained with ``retrieval_valid=false`` and never enter macro averages.
    """

    prediction = _unit_rows(
        predicted_actor_direction,
        name="predicted_actor_direction",
    )
    target = _unit_rows(
        target_actor_direction,
        name="target_actor_direction",
    )
    if prediction.shape != target.shape:
        raise ValueError("retrieval prediction/target shapes differ")
    rows = len(prediction)
    families = tuple(str(value).strip() for value in action_families)
    groups = tuple(str(value).strip() for value in content_group_ids)
    iid_values = tuple(str(value) for value in iids)
    if any(len(values) != rows for values in (families, groups, iid_values)):
        raise ValueError("retrieval metadata length mismatch")
    if any(not value for value in families) or any(not value for value in groups):
        raise ValueError("retrieval families/groups must be non-empty")
    valid = (
        np.ones(rows, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    active = (
        np.ones(rows, dtype=bool)
        if active_mask is None
        else np.asarray(active_mask, dtype=bool)
    )
    if valid.shape != (rows,) or active.shape != (rows,):
        raise ValueError("retrieval masks must have shape [N]")

    output: list[dict[str, Any]] = []
    for query in range(rows):
        query_valid = bool(valid[query] and active[query])
        candidate_indices = np.flatnonzero(
            valid
            & active
            & (np.asarray(groups, dtype=str) != groups[query])
        )
        relevant = np.asarray(
            [families[int(index)] == families[query] for index in candidate_indices],
            dtype=bool,
        )
        if not query_valid or not len(candidate_indices) or not bool(relevant.any()):
            output.append(
                {
                    "retrieval_valid": False,
                    "retrieval_candidates": int(len(candidate_indices)),
                    "retrieval_positives": int(np.count_nonzero(relevant)),
                    "actor_cross_content_ap": None,
                    "actor_cross_content_r1": None,
                    "actor_cross_content_r5": None,
                }
            )
            continue
        scores = prediction[query] @ target[candidate_indices].T
        # Stable iid tie-breaking prevents archive order from changing metrics.
        order = sorted(
            range(len(candidate_indices)),
            key=lambda position: (
                -float(scores[position]),
                iid_values[int(candidate_indices[position])],
            ),
        )
        ranked_relevant = relevant[np.asarray(order, dtype=np.int64)]
        relevant_ranks = np.flatnonzero(ranked_relevant)
        precisions = [
            float(np.count_nonzero(ranked_relevant[: rank + 1])) / float(rank + 1)
            for rank in relevant_ranks
        ]
        output.append(
            {
                "retrieval_valid": True,
                "retrieval_candidates": int(len(candidate_indices)),
                "retrieval_positives": int(np.count_nonzero(relevant)),
                "actor_cross_content_ap": float(np.mean(precisions)),
                "actor_cross_content_r1": float(bool(ranked_relevant[:1].any())),
                "actor_cross_content_r5": float(bool(ranked_relevant[:5].any())),
            }
        )
    return output


def macro_retrieval(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Macro-average retrieval metrics by action family."""

    valid = [
        row
        for row in rows
        if bool(row.get("retrieval_valid"))
        and bool(row.get("control_valid", True))
        and row.get("label_role") == "positive_delta"
    ]
    families = sorted({str(row.get("action_family") or "") for row in valid})
    per_family: dict[str, dict[str, float]] = {}
    for family in families:
        family_rows = [
            row for row in valid if str(row.get("action_family") or "") == family
        ]
        if not family or not family_rows:
            continue
        per_family[family] = {
            "mAP": float(
                np.mean(
                    [
                        _finite_float(
                            row["actor_cross_content_ap"],
                            context="actor_cross_content_ap",
                        )
                        for row in family_rows
                    ]
                )
            ),
            "R1": float(
                np.mean(
                    [
                        _finite_float(
                            row["actor_cross_content_r1"],
                            context="actor_cross_content_r1",
                        )
                        for row in family_rows
                    ]
                )
            ),
            "R5": float(
                np.mean(
                    [
                        _finite_float(
                            row["actor_cross_content_r5"],
                            context="actor_cross_content_r5",
                        )
                        for row in family_rows
                    ]
                )
            ),
            "queries": len(family_rows),
        }
    return {
        "valid_queries": len(valid),
        "families": len(per_family),
        "macro_mAP": (
            float(np.mean([value["mAP"] for value in per_family.values()]))
            if per_family
            else None
        ),
        "macro_R1": (
            float(np.mean([value["R1"] for value in per_family.values()]))
            if per_family
            else None
        ),
        "macro_R5": (
            float(np.mean([value["R5"] for value in per_family.values()]))
            if per_family
            else None
        ),
        "per_family": per_family,
    }


def false_activation_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    activation_threshold: float,
) -> dict[str, Any]:
    """Report predicted actor magnitude and activation by clean negative type."""

    threshold = _finite_float(
        activation_threshold,
        context="activation_threshold",
    )
    if threshold < 0.0:
        raise ValueError("activation_threshold must be non-negative")
    negative_rows = [
        row
        for row in rows
        if row.get("label_role") == "negative_audit"
        and bool(row.get("control_valid", True))
    ]
    result: dict[str, Any] = {}
    for label_type in sorted(
        {str(row.get("label_type") or "") for row in negative_rows}
    ):
        selected = [
            row
            for row in negative_rows
            if str(row.get("label_type") or "") == label_type
        ]
        magnitudes = np.asarray(
            [
                _finite_float(
                    row.get("actor_predicted_log_magnitude"),
                    context="actor_predicted_log_magnitude",
                )
                for row in selected
            ],
            dtype=np.float64,
        )
        result[label_type] = {
            "rows": len(selected),
            "mean_predicted_log_magnitude": float(np.mean(magnitudes)),
            "median_predicted_log_magnitude": float(np.median(magnitudes)),
            "p90_predicted_log_magnitude": float(
                np.quantile(magnitudes, 0.9)
            ),
            "activation_threshold": threshold,
            "activation_rate": float(np.mean(magnitudes >= threshold)),
        }
    return {
        "negative_rows": len(negative_rows),
        "activation_threshold": threshold,
        "by_type": result,
    }


def summarize_arm_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    activation_threshold: float,
) -> dict[str, Any]:
    """Aggregate direction, magnitude, camera and retrieval diagnostics."""

    positive = [
        row
        for row in rows
        if row.get("label_role") == "positive_delta"
        and bool(row.get("control_valid", True))
    ]

    def mean_field(name: str) -> float | None:
        values = [
            _finite_float(row[name], context=name)
            for row in positive
            if row.get(name) is not None
        ]
        return float(np.mean(values)) if values else None

    actor_active = [
        row
        for row in positive
        if bool(row.get("actor_target_active"))
        and row.get("actor_direction_cosine") is not None
    ]
    camera_active = [
        row
        for row in positive
        if bool(row.get("camera_target_active"))
        and row.get("camera_direction_cosine") is not None
    ]
    return {
        "rows": len(rows),
        "positive_rows": len(positive),
        "actor_active_rows": len(actor_active),
        "camera_active_rows": len(camera_active),
        "actor_mean_direction_cosine": (
            float(
                np.mean(
                    [
                        _finite_float(
                            row["actor_direction_cosine"],
                            context="actor_direction_cosine",
                        )
                        for row in actor_active
                    ]
                )
            )
            if actor_active
            else None
        ),
        "actor_log_magnitude_mae": mean_field(
            "actor_log_magnitude_absolute_error"
        ),
        "camera_mean_direction_cosine": (
            float(
                np.mean(
                    [
                        _finite_float(
                            row["camera_direction_cosine"],
                            context="camera_direction_cosine",
                        )
                        for row in camera_active
                    ]
                )
            )
            if camera_active
            else None
        ),
        "camera_log_magnitude_mae": mean_field(
            "camera_log_magnitude_absolute_error"
        ),
        "retrieval": macro_retrieval(positive),
        "negative_false_activation": false_activation_summary(
            rows,
            activation_threshold=activation_threshold,
        ),
    }


def _paired_group_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    treatment_arm: str,
    control_arm: str,
    metric: str,
    split: str,
    higher_is_better: bool,
) -> tuple[np.ndarray, dict[int, float], int]:
    by_key: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for position, row in enumerate(rows):
        if str(row.get("split")) != split:
            continue
        if row.get("label_role") != "positive_delta":
            continue
        arm = str(row.get("arm"))
        if arm not in {treatment_arm, control_arm}:
            continue
        if not bool(row.get("control_valid", True)):
            continue
        if row.get(metric) is None:
            continue
        seed = row.get("model_seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise GateInputError(f"row {position} has invalid model_seed")
        iid = str(row.get("iid") or "")
        group = str(row.get("content_group_id") or "")
        if not iid or not group:
            raise GateInputError(f"row {position} lacks iid/content_group_id")
        key = (seed, iid, arm)
        if key in by_key:
            raise GateInputError(f"duplicate paired result key={key}")
        by_key[key] = row

    paired: dict[tuple[int, str], float] = {}
    group_for_pair: dict[tuple[int, str], str] = {}
    seeds = sorted({key[0] for key in by_key})
    for seed in seeds:
        treatment_iids = {
            key[1]
            for key in by_key
            if key[0] == seed and key[2] == treatment_arm
        }
        control_iids = {
            key[1]
            for key in by_key
            if key[0] == seed and key[2] == control_arm
        }
        for iid in sorted(treatment_iids & control_iids):
            treatment = by_key[(seed, iid, treatment_arm)]
            control = by_key[(seed, iid, control_arm)]
            treatment_group = str(treatment["content_group_id"])
            if treatment_group != str(control["content_group_id"]):
                raise GateInputError(
                    f"paired arms disagree on content group for seed={seed} iid={iid}"
                )
            difference = _finite_float(
                treatment[metric],
                context=f"{treatment_arm}.{metric}",
            ) - _finite_float(
                control[metric],
                context=f"{control_arm}.{metric}",
            )
            if not higher_is_better:
                difference = -difference
            paired[(seed, iid)] = difference
            group_for_pair[(seed, iid)] = treatment_group

    group_seed_values: dict[tuple[str, int], list[float]] = {}
    for key, difference in paired.items():
        group_seed_values.setdefault(
            (group_for_pair[key], key[0]),
            [],
        ).append(difference)
    group_values: dict[str, list[float]] = {}
    seed_values: dict[int, list[float]] = {}
    for (group, seed), values in group_seed_values.items():
        mean = float(np.mean(values))
        group_values.setdefault(group, []).append(mean)
        seed_values.setdefault(seed, []).append(mean)
    collapsed_groups = np.asarray(
        [
            float(np.mean(group_values[group]))
            for group in sorted(group_values)
        ],
        dtype=np.float64,
    )
    collapsed_seeds = {
        seed: float(np.mean(values))
        for seed, values in sorted(seed_values.items())
    }
    return collapsed_groups, collapsed_seeds, len(paired)


def paired_group_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    treatment_arm: str,
    control_arm: str,
    metric: str,
    split: str = "test",
    higher_is_better: bool = True,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    signflip_samples: int = 20_000,
    random_seed: int = 260108828,
) -> dict[str, Any]:
    """Paired comparison with content-group bootstrap and sign-flip test."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0,1)")
    if bootstrap_samples < 0 or signflip_samples < 0:
        raise ValueError("resampling counts must be non-negative")
    groups, seed_means, pairs = _paired_group_values(
        rows,
        treatment_arm=treatment_arm,
        control_arm=control_arm,
        metric=metric,
        split=split,
        higher_is_better=higher_is_better,
    )
    result: dict[str, Any] = {
        "treatment_arm": treatment_arm,
        "control_arm": control_arm,
        "metric": metric,
        "split": split,
        "higher_is_better": bool(higher_is_better),
        "paired_queries_across_seeds": pairs,
        "content_groups": int(len(groups)),
        "model_seeds": len(seed_means),
        "per_seed_gain": {str(key): value for key, value in seed_means.items()},
        "positive_seed_count": int(
            sum(value > 0.0 for value in seed_means.values())
        ),
        "mean_gain": float(np.mean(groups)) if len(groups) else None,
        "group_bootstrap_ci": None,
        "group_signflip_one_sided_p": None,
        "confidence": confidence,
        "bootstrap_samples": bootstrap_samples,
        "signflip_samples": signflip_samples,
    }
    if not len(groups):
        return result
    rng = np.random.default_rng(
        _stable_seed(
            random_seed,
            f"{treatment_arm}:{control_arm}:{metric}:{split}",
        )
    )
    if bootstrap_samples:
        indices = rng.integers(
            0,
            len(groups),
            size=(bootstrap_samples, len(groups)),
        )
        means = np.mean(groups[indices], axis=1)
        alpha = (1.0 - confidence) / 2.0
        result["group_bootstrap_ci"] = [
            float(np.quantile(means, alpha)),
            float(np.quantile(means, 1.0 - alpha)),
        ]
    if signflip_samples:
        observed = float(np.mean(groups))
        exceed = 0
        remaining = signflip_samples
        # Chunking keeps the test bounded when the formal set has many groups.
        while remaining:
            count = min(remaining, 2_000)
            signs = rng.integers(
                0,
                2,
                size=(count, len(groups)),
                dtype=np.int8,
            )
            signs = signs.astype(np.float64) * 2.0 - 1.0
            null_means = np.mean(signs * groups[None, :], axis=1)
            exceed += int(np.count_nonzero(null_means >= observed - 1e-15))
            remaining -= count
        result["group_signflip_one_sided_p"] = float(
            (exceed + 1) / (signflip_samples + 1)
        )
    return result


def _comparison_passes(
    comparison: Mapping[str, Any],
    *,
    minimum_gain: float,
    minimum_positive_seeds: int,
    maximum_p: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    gain = comparison.get("mean_gain")
    interval = comparison.get("group_bootstrap_ci")
    p_value = comparison.get("group_signflip_one_sided_p")
    if gain is None or _finite_float(gain, context="mean_gain") < minimum_gain:
        reasons.append(f"mean_gain<{minimum_gain:g}")
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or _finite_float(interval[0], context="bootstrap lower") <= 0.0
    ):
        reasons.append("bootstrap_ci_lower<=0")
    if (
        p_value is None
        or _finite_float(p_value, context="signflip p") > maximum_p
    ):
        reasons.append(f"signflip_p>{maximum_p:g}")
    if int(comparison.get("positive_seed_count", 0)) < minimum_positive_seeds:
        reasons.append(f"positive_seeds<{minimum_positive_seeds}")
    return not reasons, reasons


def _dataset_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    dataset = contract.get("dataset")
    if not isinstance(dataset, Mapping):
        raise GateInputError("contract.dataset must be an object")
    required = {
        "label_mode",
        "production_eligible",
        "split_version",
        "positive_count",
        "positive_group_count",
        "action_family_count",
        "test_positive_group_count",
        "negative_audit_count",
    }
    missing = sorted(required - set(dataset))
    if missing:
        raise GateInputError(f"contract.dataset is missing {missing}")
    for name in (
        "positive_count",
        "positive_group_count",
        "action_family_count",
        "test_positive_group_count",
        "negative_audit_count",
    ):
        value = dataset[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GateInputError(f"contract.dataset.{name} is invalid")
    if dataset["label_mode"] not in {"human", "strict_legacy_qwen"}:
        raise GateInputError("unsupported label_mode in contract")
    if not isinstance(dataset["production_eligible"], bool):
        raise GateInputError("contract.dataset.production_eligible must be boolean")
    return dict(dataset)


def _validate_result_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    declared_seeds: Sequence[int],
    dataset: Mapping[str, Any],
) -> None:
    """Bind contract counts/seeds to the actual per-query evidence."""

    coverage: dict[str, set[tuple[int, str]]] = {
        arm: set() for arm in REQUIRED_ARMS
    }
    iid_metadata: dict[str, tuple[str, str, str, str]] = {}
    observed_seeds: set[int] = set()
    for position, row in enumerate(rows):
        arm = str(row.get("arm"))
        if arm not in REQUIRED_ARMS:
            raise GateInputError(f"row {position} has unexpected arm={arm!r}")
        seed = row.get("model_seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise GateInputError(f"row {position} has invalid model_seed")
        iid = str(row.get("iid") or "")
        if not iid:
            raise GateInputError(f"row {position} has empty iid")
        key = (seed, iid)
        if key in coverage[arm]:
            raise GateInputError(
                f"duplicate arm/seed/iid result: {arm}/{seed}/{iid}"
            )
        coverage[arm].add(key)
        observed_seeds.add(seed)
        metadata = (
            str(row.get("label_role") or ""),
            str(row.get("content_group_id") or ""),
            str(row.get("action_family") or ""),
            str(row.get("split") or ""),
        )
        previous = iid_metadata.setdefault(iid, metadata)
        if previous != metadata:
            raise GateInputError(
                f"per-query metadata changes across arms/seeds for iid={iid}"
            )
    if observed_seeds != set(declared_seeds):
        raise GateInputError(
            "contract.model_seeds disagree with per-query rows: "
            f"{sorted(declared_seeds)} != {sorted(observed_seeds)}"
        )
    reference = coverage["full"]
    for arm in sorted(REQUIRED_ARMS):
        if coverage[arm] != reference:
            raise GateInputError(
                f"{arm} coverage differs from full "
                f"(missing={len(reference - coverage[arm])}, "
                f"extra={len(coverage[arm] - reference)})"
            )
    positive_iids = {
        iid
        for iid, metadata in iid_metadata.items()
        if metadata[0] == "positive_delta"
    }
    negative_iids = {
        iid
        for iid, metadata in iid_metadata.items()
        if metadata[0] == "negative_audit"
    }
    calculated = {
        "positive_count": len(positive_iids),
        "positive_group_count": len(
            {iid_metadata[iid][1] for iid in positive_iids}
        ),
        "action_family_count": len(
            {
                iid_metadata[iid][2]
                for iid in positive_iids
                if iid_metadata[iid][2] != "unknown"
            }
        ),
        "test_positive_group_count": len(
            {
                iid_metadata[iid][1]
                for iid in positive_iids
                if iid_metadata[iid][3] == "test"
            }
        ),
        "negative_audit_count": len(negative_iids),
    }
    for name, actual in calculated.items():
        if int(dataset[name]) != actual:
            raise GateInputError(
                f"contract.dataset.{name}={dataset[name]} "
                f"disagrees with per-query rows={actual}"
            )


def evaluate_r5_gate(
    *,
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    thresholds: R5GateThresholds | None = None,
) -> dict[str, Any]:
    """Return ``PASS|NO_GO|INSUFFICIENT`` or raise on invalid evidence."""

    selected_thresholds = thresholds or R5GateThresholds()
    selected_thresholds.validate()
    if not rows:
        raise GateInputError("per-query result is empty")
    dataset = _dataset_contract(contract)
    declared_seeds = contract.get("model_seeds")
    if (
        not isinstance(declared_seeds, list)
        or not declared_seeds
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in declared_seeds
        )
        or len(set(declared_seeds)) != len(declared_seeds)
    ):
        raise GateInputError("contract.model_seeds must be unique integers")
    observed_arms = {str(row.get("arm")) for row in rows}
    missing_arms = sorted(REQUIRED_ARMS - observed_arms)
    if missing_arms:
        raise GateInputError(f"per-query result is missing arms {missing_arms}")
    _validate_result_coverage(
        rows,
        declared_seeds=declared_seeds,
        dataset=dataset,
    )

    insufficiencies: list[str] = []
    if dataset["label_mode"] != "human":
        insufficiencies.append("formal gate requires human labels")
    if not dataset["production_eligible"]:
        insufficiencies.append("dataset declares production_eligible=false")
    if dataset["split_version"] != R5_PRODUCTION_SPLIT_VERSION:
        insufficiencies.append(
            "formal gate requires source/subject/scene visual clusters"
        )
    scale_requirements = (
        (
            "positive_count",
            selected_thresholds.minimum_human_positives,
            "human positives",
        ),
        (
            "positive_group_count",
            selected_thresholds.minimum_positive_groups,
            "positive content groups",
        ),
        (
            "action_family_count",
            selected_thresholds.minimum_action_families,
            "action families",
        ),
        (
            "test_positive_group_count",
            selected_thresholds.minimum_test_positive_groups,
            "test positive groups",
        ),
    )
    for field, minimum, label in scale_requirements:
        if int(dataset[field]) < minimum:
            insufficiencies.append(
                f"{label} {dataset[field]}<{minimum}"
            )
    if len(declared_seeds) < selected_thresholds.minimum_model_seeds:
        insufficiencies.append(
            f"model seeds {len(declared_seeds)}<"
            f"{selected_thresholds.minimum_model_seeds}"
        )
    auxiliary = contract.get("formal_auxiliary_checks")
    if not isinstance(auxiliary, Mapping):
        insufficiencies.append("formal auxiliary checks are undeclared")
    else:
        required_auxiliary = (
            "direction_probe",
            "speed_probe",
            "phase_probe",
            "camera_leakage",
            "stability",
            "pair_specificity",
        )
        incomplete = [
            name for name in required_auxiliary if auxiliary.get(name) is not True
        ]
        if incomplete:
            insufficiencies.append(
                "formal auxiliary checks incomplete: " + ",".join(incomplete)
            )
    if contract.get("formal_auxiliary_checks_complete") is not True:
        insufficiencies.append("formal_auxiliary_checks_complete is not true")

    comparison_specs = [
        ("full", "centroid", "actor_direction_cosine", 0.05),
        ("full", "pairshuffle", "actor_direction_cosine", 0.0),
        ("full", "matched_random", "actor_direction_cosine", 0.0),
        ("full", "text_only", "actor_direction_cosine", 0.0),
        ("full", "source_shuffle", "actor_direction_cosine", 0.0),
        ("full", "prompt_shuffle", "actor_direction_cosine", 0.0),
        ("full", "centroid", "actor_cross_content_ap", 0.05),
        ("full", "pairshuffle", "actor_cross_content_ap", 0.0),
        ("full", "matched_random", "actor_cross_content_ap", 0.0),
        ("full", "text_only", "actor_cross_content_ap", 0.0),
        ("full", "source_shuffle", "actor_cross_content_ap", 0.0),
        ("full", "prompt_shuffle", "actor_cross_content_ap", 0.0),
    ]
    comparisons: dict[str, Any] = {}
    criteria: list[dict[str, Any]] = []
    for treatment, control, metric, default_minimum in comparison_specs:
        minimum = (
            selected_thresholds.minimum_actor_cosine_gain
            if metric == "actor_direction_cosine" and control == "centroid"
            else selected_thresholds.minimum_macro_map_gain
            if metric == "actor_cross_content_ap" and control == "centroid"
            else default_minimum
        )
        key = f"{treatment}_vs_{control}:{metric}"
        comparison = paired_group_comparison(
            rows,
            treatment_arm=treatment,
            control_arm=control,
            metric=metric,
            confidence=selected_thresholds.confidence,
            bootstrap_samples=selected_thresholds.bootstrap_samples,
            signflip_samples=selected_thresholds.signflip_samples,
            random_seed=int(contract.get("data_seed", 260108828)),
        )
        comparisons[key] = comparison
        passed, reasons = _comparison_passes(
            comparison,
            minimum_gain=minimum,
            minimum_positive_seeds=(
                selected_thresholds.minimum_positive_seed_directions
            ),
            maximum_p=selected_thresholds.maximum_signflip_p,
        )
        criteria.append(
            {
                "name": key,
                "minimum_gain": minimum,
                "passed": passed,
                "failure_reasons": reasons,
            }
        )

    # Magnitude and camera are intentionally visible in the decision artifact.
    # They are not pooled into direction/retrieval and no unregistered tolerance
    # is invented here.
    diagnostic_comparisons: dict[str, Any] = {}
    for metric in (
        "actor_log_magnitude_absolute_error",
        "camera_log_magnitude_absolute_error",
        "camera_direction_cosine",
    ):
        higher = metric == "camera_direction_cosine"
        for arm in ("full", "text_only"):
            key = f"{arm}_vs_centroid:{metric}"
            diagnostic_comparisons[key] = paired_group_comparison(
                rows,
                treatment_arm=arm,
                control_arm="centroid",
                metric=metric,
                higher_is_better=higher,
                confidence=selected_thresholds.confidence,
                bootstrap_samples=selected_thresholds.bootstrap_samples,
                signflip_samples=selected_thresholds.signflip_samples,
                random_seed=int(contract.get("data_seed", 260108828)),
            )

    status = (
        "INSUFFICIENT"
        if insufficiencies
        else ("PASS" if all(item["passed"] for item in criteria) else "NO_GO")
    )
    return {
        "schema_version": R5_GATE_SCHEMA,
        "status": status,
        "production_decision": status == "PASS",
        "dataset": dataset,
        "thresholds": asdict(selected_thresholds),
        "insufficiency_reasons": insufficiencies,
        "criteria": criteria,
        "comparisons": comparisons,
        "magnitude_camera_diagnostics": diagnostic_comparisons,
        "interpretation": (
            "diagnostic_only"
            if status == "INSUFFICIENT"
            else "formal_gate"
        ),
    }


def invalid_gate_summary(error: BaseException | str) -> dict[str, Any]:
    return {
        "schema_version": R5_GATE_SCHEMA,
        "status": "INVALID",
        "production_decision": False,
        "error": str(error),
    }


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise GateInputError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-query", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--signflip-samples", type=int, default=50_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = _iter_jsonl(args.per_query.expanduser())
        contract = json.loads(
            args.contract.expanduser().read_text(encoding="utf-8")
        )
        if not isinstance(contract, dict):
            raise GateInputError("contract is not a JSON object")
        thresholds = R5GateThresholds(
            bootstrap_samples=int(args.bootstrap_samples),
            signflip_samples=int(args.signflip_samples),
        )
        summary = evaluate_r5_gate(
            rows=rows,
            contract=contract,
            thresholds=thresholds,
        )
    except Exception as error:
        summary = invalid_gate_summary(error)
        _atomic_json(args.output.expanduser(), summary)
        raise
    _atomic_json(args.output.expanduser(), summary)
    print(
        f"[r5-gate] status={summary['status']} output={args.output.expanduser()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
