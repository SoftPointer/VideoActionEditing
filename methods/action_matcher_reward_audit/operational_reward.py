#!/usr/bin/env python3
"""Operational group-relative action reward built from frozen video features.

This module intentionally does *not* implement a candidate-independent
``same action`` classifier.  It ranks a Best-of-N candidate pool under the
explicit prior that the pool is likely to contain at least one valid action.
It abstains and drops a low-confidence group when that prior is not supported
by the available evidence; no slow-model fallback is part of this contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from run_sequence_audit import (
    EPS,
    cosine,
    score_endpoint,
    score_frame_diagonal,
)


SCHEMA_VERSION = "action-editing-operational-reward-v1"
AXES = (
    "motion_set",
    "ordered_alignment",
    "reverse_contrast",
    "endpoint",
    "activity_match",
)
CONTRACT_AXES = {
    "directional_endpoint": AXES,
    "generic_ordered": (
        "motion_set",
        "ordered_alignment",
        "reverse_contrast",
        "activity_match",
    ),
    "cyclic": ("motion_set", "ordered_alignment", "activity_match"),
}


@dataclass(frozen=True)
class RewardConfig:
    """Pre-registered policy knobs; thresholds remain calibration targets."""

    contract: str = "generic_ordered"
    minimum_activity_ratio: float = 0.10
    minimum_reverse_contrast: float = 0.0
    minimum_event_score: float = 0.25
    minimum_top_gap: float = 0.20
    tie_tolerance: float = 1.0e-8

    def validate(self) -> None:
        if self.contract not in CONTRACT_AXES:
            raise ValueError(f"unknown action contract: {self.contract}")
        if not 0.0 <= self.minimum_activity_ratio <= 1.0:
            raise ValueError("minimum_activity_ratio must be in [0,1]")
        if self.minimum_top_gap < 0.0:
            raise ValueError("minimum_top_gap must be non-negative")
        if not 0.0 <= self.minimum_event_score <= 1.0:
            raise ValueError("minimum_event_score must be in [0,1]")
        if self.tie_tolerance < 0.0:
            raise ValueError("tie_tolerance must be non-negative")


def _finite_tensor(name: str, value: torch.Tensor, ndim: int) -> torch.Tensor:
    tensor = value.detach().float().cpu()
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {tensor.shape}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def temporal_activity(sequence: torch.Tensor) -> float:
    """RMS radius of unit frame descriptors around their temporal mean."""

    values = _finite_tensor("frame_sequence", sequence, 2)
    if len(values) < 2:
        return 0.0
    values = F.normalize(values, dim=1, eps=EPS)
    centered = values - values.mean(dim=0, keepdim=True)
    return float(centered.square().sum(dim=1).mean().sqrt())


def _candidate_raw_scores(
    reference_m3: torch.Tensor,
    reference_sequence: torch.Tensor,
    candidate_m3: torch.Tensor,
    candidate_sequence: torch.Tensor,
) -> dict[str, float]:
    reference_m3 = _finite_tensor("reference_m3", reference_m3, 1)
    candidate_m3 = _finite_tensor("candidate_m3", candidate_m3, 1)
    reference_sequence = _finite_tensor("reference_sequence", reference_sequence, 2)
    candidate_sequence = _finite_tensor("candidate_sequence", candidate_sequence, 2)
    if reference_m3.shape != candidate_m3.shape:
        raise ValueError("reference and candidate M3 dimensions differ")
    if reference_sequence.shape[1] != candidate_sequence.shape[1]:
        raise ValueError("reference and candidate frame dimensions differ")

    reference_activity = temporal_activity(reference_sequence)
    candidate_activity = temporal_activity(candidate_sequence)
    ratio = candidate_activity / max(reference_activity, EPS)
    activity_match = math.exp(-abs(math.log(max(ratio, EPS))))
    ordered = score_frame_diagonal(reference_sequence, candidate_sequence)
    reversed_ordered = score_frame_diagonal(
        reference_sequence, torch.flip(candidate_sequence, dims=(0,))
    )
    return {
        # SemanticMoments M3 is deliberately only a coarse motion-set score.
        "motion_set": cosine(reference_m3, candidate_m3),
        "ordered_alignment": ordered,
        "reverse_contrast": ordered - reversed_ordered,
        "endpoint": score_endpoint(reference_sequence, candidate_sequence),
        "activity_match": activity_match,
        "activity_ratio": ratio,
    }


def average_tie_percentiles(values: Sequence[float]) -> list[float]:
    """Return [0,1] average-rank percentiles where larger values are better."""

    if not values:
        raise ValueError("cannot rank an empty score list")
    numeric = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("percentile inputs must be finite")
    if len(numeric) == 1:
        return [1.0]
    output = []
    for value in numeric:
        less = sum(other < value for other in numeric)
        equal_others = sum(other == value for other in numeric) - 1
        output.append((less + 0.5 * equal_others) / (len(numeric) - 1))
    return output


def _gate_reasons(raw: Mapping[str, float], config: RewardConfig) -> list[str]:
    reasons = []
    if raw["activity_ratio"] < config.minimum_activity_ratio:
        reasons.append("activity_below_reference_ratio_floor")
    if (
        config.contract in {"directional_endpoint", "generic_ordered"}
        and raw["reverse_contrast"] < config.minimum_reverse_contrast
    ):
        reasons.append("reverse_explains_candidate_at_least_as_well")
    return reasons


def score_candidate_pool(
    *,
    reference_id: str,
    reference_m3: torch.Tensor,
    reference_sequence: torch.Tensor,
    candidates: Sequence[Mapping[str, Any]],
    config: RewardConfig | None = None,
    valid_candidate_prior: bool = False,
) -> dict[str, Any]:
    """Score one anchor-aligned Best-of-N pool and expose abstention state.

    Each candidate mapping must contain ``candidate_id``, ``m3`` and
    ``frame_sequence``.  ``event_score`` is the minimum percentile over all
    contract-required axes; one strong signal therefore cannot compensate for
    a failed axis.  Hard-gated candidates receive zero utility.
    """

    policy = config or RewardConfig()
    policy.validate()
    if not candidates:
        raise ValueError("candidate pool is empty")
    identifiers = [str(row["candidate_id"]) for row in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate_id must be unique inside a pool")

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        raw = _candidate_raw_scores(
            reference_m3,
            reference_sequence,
            candidate["m3"],
            candidate["frame_sequence"],
        )
        rows.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "raw_scores": raw,
                "gate_reasons": _gate_reasons(raw, policy),
            }
        )

    required_axes = CONTRACT_AXES[policy.contract]
    for axis in required_axes:
        percentiles = average_tie_percentiles(
            [row["raw_scores"][axis] for row in rows]
        )
        for row, percentile in zip(rows, percentiles):
            row.setdefault("pool_percentiles", {})[axis] = percentile

    for row in rows:
        row["eligible"] = not row["gate_reasons"]
        row["event_score"] = (
            min(row["pool_percentiles"].values()) if row["eligible"] else 0.0
        )

    eligible = [row for row in rows if row["eligible"]]
    ranked = sorted(eligible, key=lambda row: (-row["event_score"], row["candidate_id"]))
    selected = ranked[0] if ranked else None
    abstain_reasons = []
    if not valid_candidate_prior:
        abstain_reasons.append("valid_candidate_prior_not_asserted")
    if selected is None:
        abstain_reasons.append("no_eligible_candidate")
    else:
        if selected["event_score"] < policy.minimum_event_score:
            abstain_reasons.append("top_event_score_below_floor")
        if len(ranked) > 1:
            top_gap = selected["event_score"] - ranked[1]["event_score"]
            if top_gap + policy.tie_tolerance < policy.minimum_top_gap:
                abstain_reasons.append("top_two_gap_too_small")
        else:
            top_gap = None

    return {
        "schema_version": SCHEMA_VERSION,
        "reference_id": str(reference_id),
        "contract": policy.contract,
        "required_axes": list(required_axes),
        "config": asdict(policy),
        "assumptions": {
            "requires_valid_candidate_prior": True,
            "valid_candidate_prior_asserted_by_caller": bool(valid_candidate_prior),
            "absolute_same_action_calibrated": False,
            "candidate_pool_scores_are_cross_pool_comparable": False,
        },
        "diagnostic_top_candidate_id": selected["candidate_id"] if selected else None,
        "selected_candidate_id": (
            selected["candidate_id"] if selected and not abstain_reasons else None
        ),
        "top_event_score": selected["event_score"] if selected else None,
        "top_gap": top_gap if selected is not None else None,
        "abstain_required": bool(abstain_reasons),
        "abstain_reasons": abstain_reasons,
        "selection_authorized": not abstain_reasons,
        "candidates": rows,
    }


def pareto_preference_pairs(pool_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return raw-axis Pareto pairs with an eligible chosen endpoint.

    A hard-gated candidate is allowed as the rejected endpoint.  This is how a
    clear reverse/no-op failure becomes useful preference supervision without
    ever being admitted as a positive.
    """

    axes = list(pool_result["required_axes"])
    rows = list(pool_result["candidates"])
    chosen_rows = [row for row in rows if row["eligible"]]
    pairs = []
    for chosen in chosen_rows:
        for rejected in rows:
            if chosen["candidate_id"] == rejected["candidate_id"]:
                continue
            differences = {
                axis: chosen["raw_scores"][axis] - rejected["raw_scores"][axis]
                for axis in axes
            }
            if all(value >= -EPS for value in differences.values()) and any(
                value > EPS for value in differences.values()
            ):
                pairs.append(
                    {
                        "chosen_candidate_id": chosen["candidate_id"],
                        "rejected_candidate_id": rejected["candidate_id"],
                        "raw_axis_margins": differences,
                        "event_score_margin": (
                            chosen["event_score"] - rejected["event_score"]
                        ),
                    }
                )
    return sorted(
        pairs,
        key=lambda row: (
            -row["event_score_margin"],
            row["chosen_candidate_id"],
            row["rejected_candidate_id"],
        ),
    )


def select_training_pair(
    pool_result: Mapping[str, Any], *, minimum_event_gain: float = 0.20
) -> dict[str, Any] | None:
    """Choose one non-compensating preference pair, or authorize zero update."""

    if pool_result.get("abstain_required", True):
        return None
    if not 0.0 < minimum_event_gain <= 1.0:
        raise ValueError("minimum_event_gain must lie in (0,1]")
    pairs = [
        row
        for row in pareto_preference_pairs(pool_result)
        if row["event_score_margin"] + EPS >= minimum_event_gain
    ]
    if not pairs:
        return None
    selected_id = pool_result.get("selected_candidate_id")
    selected_pairs = [
        row for row in pairs if row["chosen_candidate_id"] == selected_id
    ]
    return (selected_pairs or pairs)[0]
