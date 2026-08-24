#!/usr/bin/env python3
"""Fail-closed source-object role assignment on frozen SAM2 proposal tracks.

This module is deliberately observer-only.  SAM2 supplies source-instance
boundaries and tracks.  Frozen Bernini r6 attn2 affinities supply role evidence
*only after it has been reduced over each SAM2 proposal region*.  The module
never edits a renderer tensor, never calls a model, and never forces a role to
take a proposal.

The three vessel roles are mutually exclusive.  A role is emitted only when it
beats the preregistered 64-token null bank, its cyclic-token permutation, the
temporal consistency gates, and every competing proposal.  Ties, duplicate
proposal families, role conflicts, degenerate controls, or incomplete tracks
remain unassigned.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "bernini-source-object-proposal-role-probe-v15c-r3"
TRACK_SCHEMA_VERSION = "bernini-source-sam2-proposal-tracks-v15c-r3"
ROLE_NAMES = ("old_actor", "new_actor", "recipient")
FULL_R6_ROLE_NAMES = ("agent", *ROLE_NAMES, "support")
BLOCK_INDICES = (4, 9, 14, 19, 24)
PHASE_FRAMES = tuple(range(0, 81, 4))
PHASE_COUNT = 21
GRID_HEIGHT = 37
GRID_WIDTH = 25
NULL_COUNT = 64
MAXIMUM_PROPOSAL_COUNT = 64
PROPOSAL_ID_PATTERN = re.compile(r"^sam2-f000-[0-9a-f]{64}$")


class SourceProposalRoleProbeV15CError(RuntimeError):
    """A v15c observer input or fail-closed contract differs."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, np.ndarray):
        raise SourceProposalRoleProbeV15CError("expected numpy array")
    return tuple(int(item) for item in value.shape)


def _finite_float_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or _shape(value) != shape
        or value.dtype.kind not in "fc"
        or not bool(np.isfinite(value).all())
    ):
        raise SourceProposalRoleProbeV15CError(f"{label} tensor contract differs")
    return np.asarray(value, dtype=np.float32)


@dataclass(frozen=True)
class ProbeThresholdsV15C:
    """Preregistered, ROI-free observer gates.

    The thresholds operate on controls and proposal tracks, not on a hand box
    or a semantic segmentation label.  They are intentionally conservative:
    failure means "unassigned", never "pick the least bad proposal".
    """

    familywise_alpha: float = 0.05
    familywise_role_count: int = 3
    phase_proposal_max_null_percentile: float = 0.90
    minimum_consistent_phases: int = 13
    minimum_longest_consistent_run: int = 4
    minimum_real_over_permutation_phases: int = 14
    minimum_proposal_dominance_phases: int = 16
    minimum_distinct_null_track_scores: int = 16
    null_track_score_epsilon: float = 1.0e-6
    duplicate_median_iou: float = 0.55
    duplicate_median_containment: float = 0.80
    family_overlap_median_iou: float = 0.10
    family_nesting_median_containment: float = 0.50
    minimum_phase_coverage_mass: float = 0.20

    def __post_init__(self) -> None:
        fractions = (
            self.familywise_alpha,
            self.phase_proposal_max_null_percentile,
            self.duplicate_median_iou,
            self.duplicate_median_containment,
            self.family_overlap_median_iou,
            self.family_nesting_median_containment,
        )
        if (
            any(not 0.0 < float(value) < 1.0 for value in fractions)
            or self.familywise_role_count != len(ROLE_NAMES)
            or not 1 <= self.minimum_consistent_phases <= PHASE_COUNT
            or not 1 <= self.minimum_longest_consistent_run <= PHASE_COUNT
            or not 1 <= self.minimum_real_over_permutation_phases <= PHASE_COUNT
            or not 1 <= self.minimum_proposal_dominance_phases <= PHASE_COUNT
            or not 2 <= self.minimum_distinct_null_track_scores <= NULL_COUNT
            or not math.isfinite(self.null_track_score_epsilon)
            or self.null_track_score_epsilon <= 0.0
            or not math.isfinite(self.minimum_phase_coverage_mass)
            or self.minimum_phase_coverage_mass <= 0.0
        ):
            raise SourceProposalRoleProbeV15CError("threshold contract differs")

    @property
    def required_track_null_percentile(self) -> float:
        return 1.0 - self.familywise_alpha / float(self.familywise_role_count)


@dataclass(frozen=True)
class ProposalTrackInputV15C:
    proposal_ids: tuple[str, ...]
    phase_coverage: np.ndarray
    track_gate_pass: tuple[bool, ...]

    def __post_init__(self) -> None:
        proposal_count = len(self.proposal_ids)
        if (
            not 1 <= proposal_count <= MAXIMUM_PROPOSAL_COUNT
            or len(set(self.proposal_ids)) != proposal_count
            or any(
                not isinstance(item, str)
                or PROPOSAL_ID_PATTERN.fullmatch(item) is None
                for item in self.proposal_ids
            )
            or len(self.track_gate_pass) != proposal_count
            or any(type(item) is not bool for item in self.track_gate_pass)
        ):
            raise SourceProposalRoleProbeV15CError("proposal registry differs")
        value = _finite_float_array(
            self.phase_coverage,
            (proposal_count, PHASE_COUNT, GRID_HEIGHT, GRID_WIDTH),
            "phase coverage",
        )
        if bool((value < 0.0).any()) or bool((value > 1.0).any()):
            raise SourceProposalRoleProbeV15CError("phase coverage is outside [0,1]")


@dataclass(frozen=True)
class R6AffinityInputV15C:
    real: np.ndarray
    shuffled: np.ndarray
    null_bank: np.ndarray

    def __post_init__(self) -> None:
        _finite_float_array(
            self.real,
            (len(BLOCK_INDICES), len(ROLE_NAMES), PHASE_COUNT, GRID_HEIGHT, GRID_WIDTH),
            "real affinity",
        )
        _finite_float_array(
            self.shuffled,
            (len(BLOCK_INDICES), len(ROLE_NAMES), PHASE_COUNT, GRID_HEIGHT, GRID_WIDTH),
            "shuffled affinity",
        )
        _finite_float_array(
            self.null_bank,
            (len(BLOCK_INDICES), NULL_COUNT, PHASE_COUNT, GRID_HEIGHT, GRID_WIDTH),
            "null affinity",
        )


def spatial_median_mad(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize each final HxW map without any spatial supervision."""

    array = np.asarray(value, dtype=np.float32)
    if array.ndim < 2 or tuple(array.shape[-2:]) != (GRID_HEIGHT, GRID_WIDTH):
        raise SourceProposalRoleProbeV15CError("spatial map geometry differs")
    median = np.median(array, axis=(-2, -1), keepdims=True)
    mad = np.median(np.abs(array - median), axis=(-2, -1), keepdims=True)
    scale = np.float32(1.4826) * mad
    standardized = np.divide(
        array - median,
        scale,
        out=np.zeros_like(array, dtype=np.float32),
        where=scale > np.float32(1.0e-12),
    )
    return standardized, np.squeeze(scale, axis=(-2, -1)).astype(np.float32)


def region_mean(value: np.ndarray, coverage: np.ndarray, minimum_mass: float) -> float:
    weights = np.asarray(coverage, dtype=np.float32)
    if weights.shape != (GRID_HEIGHT, GRID_WIDTH):
        raise SourceProposalRoleProbeV15CError("proposal coverage geometry differs")
    mass = float(weights.sum())
    if not math.isfinite(mass) or mass < minimum_mass:
        return float("nan")
    result = float(np.sum(np.asarray(value, dtype=np.float32) * weights) / mass)
    return result if math.isfinite(result) else float("nan")


def midrank_percentile(value: float, controls: np.ndarray) -> float:
    array = np.asarray(controls, dtype=np.float64)
    if array.shape != (NULL_COUNT,) or not np.isfinite(array).all() or not math.isfinite(value):
        return float("nan")
    below = float(np.sum(array < value))
    equal = float(np.sum(array == value))
    return (below + 0.5 * equal) / float(NULL_COUNT)


def empirical_upper_p(value: float, controls: np.ndarray) -> float:
    """Finite-null upper-tail p-value with the +1 correction."""

    array = np.asarray(controls, dtype=np.float64)
    if array.shape != (NULL_COUNT,) or not np.isfinite(array).all() or not math.isfinite(value):
        return float("nan")
    return (1.0 + float(np.sum(array >= value))) / float(NULL_COUNT + 1)


def longest_true_run(values: Sequence[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _distinct_float_count(values: np.ndarray, epsilon: float) -> int:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.shape != (NULL_COUNT,) or not np.isfinite(ordered).all():
        return 0
    count = 1
    last = float(ordered[0])
    for value in ordered[1:]:
        if abs(float(value) - last) > epsilon:
            count += 1
            last = float(value)
    return count


def _phase_region_scores(
    affinity: R6AffinityInputV15C,
    tracks: ProposalTrackInputV15C,
    thresholds: ProbeThresholdsV15C,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Return block-median region statistics, never pointwise role masks."""

    real_z, real_scale = spatial_median_mad(affinity.real)
    shuffled_z, shuffled_scale = spatial_median_mad(affinity.shuffled)
    null_z, null_scale = spatial_median_mad(affinity.null_bank)
    proposal_count = len(tracks.proposal_ids)
    real_scores = np.full(
        (len(ROLE_NAMES), proposal_count, PHASE_COUNT), np.nan, dtype=np.float32
    )
    shuffled_scores = np.full_like(real_scores, np.nan)
    null_scores = np.full(
        (proposal_count, NULL_COUNT, PHASE_COUNT), np.nan, dtype=np.float32
    )
    for proposal in range(proposal_count):
        for phase in range(PHASE_COUNT):
            coverage = tracks.phase_coverage[proposal, phase]
            for role in range(len(ROLE_NAMES)):
                block_real = [
                    region_mean(
                        real_z[block, role, phase],
                        coverage,
                        thresholds.minimum_phase_coverage_mass,
                    )
                    for block in range(len(BLOCK_INDICES))
                ]
                block_shuffled = [
                    region_mean(
                        shuffled_z[block, role, phase],
                        coverage,
                        thresholds.minimum_phase_coverage_mass,
                    )
                    for block in range(len(BLOCK_INDICES))
                ]
                if np.isfinite(block_real).all():
                    real_scores[role, proposal, phase] = np.float32(
                        np.median(block_real)
                    )
                if np.isfinite(block_shuffled).all():
                    shuffled_scores[role, proposal, phase] = np.float32(
                        np.median(block_shuffled)
                    )
            for null_index in range(NULL_COUNT):
                block_null = [
                    region_mean(
                        null_z[block, null_index, phase],
                        coverage,
                        thresholds.minimum_phase_coverage_mass,
                    )
                    for block in range(len(BLOCK_INDICES))
                ]
                if np.isfinite(block_null).all():
                    null_scores[proposal, null_index, phase] = np.float32(
                        np.median(block_null)
                    )
    scale_receipt = {
        "real_zero_scale_maps": int(np.sum(real_scale <= np.float32(1.0e-12))),
        "shuffled_zero_scale_maps": int(
            np.sum(shuffled_scale <= np.float32(1.0e-12))
        ),
        "null_zero_scale_maps": int(np.sum(null_scale <= np.float32(1.0e-12))),
        "aggregation": "proposal_coverage_weighted_region_mean_then_equal_block_median",
        "pointwise_mask_from_affinity_created": False,
    }
    return real_scores, shuffled_scores, null_scores, scale_receipt


def _tube_pair_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    ious: list[float] = []
    containments: list[float] = []
    for phase in range(PHASE_COUNT):
        a = np.asarray(left[phase], dtype=np.float32)
        b = np.asarray(right[phase], dtype=np.float32)
        intersection = float(np.minimum(a, b).sum())
        union = float(np.maximum(a, b).sum())
        smaller = min(float(a.sum()), float(b.sum()))
        if union > 0.0 and smaller > 0.0:
            ious.append(intersection / union)
            containments.append(intersection / smaller)
    if not ious:
        return 1.0, 1.0
    return float(np.median(ious)), float(np.median(containments))


def duplicate_proposal_adjacency(
    tracks: ProposalTrackInputV15C, thresholds: ProbeThresholdsV15C
) -> Mapping[str, tuple[str, ...]]:
    result: dict[str, list[str]] = {item: [] for item in tracks.proposal_ids}
    for first in range(len(tracks.proposal_ids)):
        for second in range(first + 1, len(tracks.proposal_ids)):
            iou, containment = _tube_pair_metrics(
                tracks.phase_coverage[first], tracks.phase_coverage[second]
            )
            # Any duplicate, broader overlap family, or nesting family is an
            # ambiguity.  Every member is excluded; no representative is picked.
            if (
                iou >= thresholds.family_overlap_median_iou
                or containment >= thresholds.family_nesting_median_containment
            ):
                result[tracks.proposal_ids[first]].append(tracks.proposal_ids[second])
                result[tracks.proposal_ids[second]].append(tracks.proposal_ids[first])
    return {key: tuple(sorted(value)) for key, value in result.items()}


def _candidate_evidence(
    *,
    role_index: int,
    proposal_index: int,
    real_scores: np.ndarray,
    shuffled_scores: np.ndarray,
    null_scores: np.ndarray,
    proposal_max_null_phase: np.ndarray,
    proposal_max_null_track: np.ndarray,
    track_valid: bool,
    duplicate_neighbors: Sequence[str],
    thresholds: ProbeThresholdsV15C,
) -> dict[str, Any]:
    real_phase = real_scores[role_index, proposal_index].astype(np.float64)
    shuffled_phase = shuffled_scores[role_index, proposal_index].astype(np.float64)
    own_null_phase = null_scores[proposal_index].astype(np.float64)
    null_phase = np.asarray(proposal_max_null_phase, dtype=np.float64)
    finite = (
        np.isfinite(real_phase)
        & np.isfinite(shuffled_phase)
        & np.isfinite(null_phase).all(axis=0)
    )
    if bool(finite.any()):
        track_real = float(np.median(real_phase[finite]))
        track_shuffled = float(np.median(shuffled_phase[finite]))
        own_track_null = np.median(own_null_phase[:, finite], axis=1)
        track_null = np.asarray(proposal_max_null_track, dtype=np.float64)
    else:
        track_real = float("nan")
        track_shuffled = float("nan")
        own_track_null = np.full((NULL_COUNT,), np.nan, dtype=np.float64)
        track_null = np.full((NULL_COUNT,), np.nan, dtype=np.float64)
    null_percentile = midrank_percentile(track_real, track_null)
    raw_upper_p = empirical_upper_p(track_real, track_null)
    fwer_upper_p = min(1.0, thresholds.familywise_role_count * raw_upper_p)
    distinct_nulls = _distinct_float_count(
        track_null, thresholds.null_track_score_epsilon
    )
    phase_percentiles = np.asarray(
        [
            midrank_percentile(float(real_phase[phase]), null_phase[:, phase])
            if finite[phase]
            else float("nan")
            for phase in range(PHASE_COUNT)
        ],
        dtype=np.float64,
    )
    consistent = [
        bool(
            finite[phase]
            and phase_percentiles[phase]
            >= thresholds.phase_proposal_max_null_percentile
            and real_phase[phase] > shuffled_phase[phase]
        )
        for phase in range(PHASE_COUNT)
    ]
    real_over_permutation_count = int(
        np.sum(finite & (real_phase > shuffled_phase))
    )
    null_quantile = (
        float(np.quantile(track_null, thresholds.required_track_null_percentile))
        if bool(np.isfinite(track_null).all())
        else float("nan")
    )
    gates = {
        "track_geometry": track_valid is True,
        "all_21_phase_regions_present": int(np.sum(finite)) == PHASE_COUNT,
        "null_bank_non_degenerate": (
            distinct_nulls >= thresholds.minimum_distinct_null_track_scores
            and float(np.std(track_null)) > thresholds.null_track_score_epsilon
        ),
        "track_above_proposal_max_null_with_three_role_fwer": (
            math.isfinite(null_percentile)
            and null_percentile >= thresholds.required_track_null_percentile
            and math.isfinite(null_quantile)
            and track_real > null_quantile
            and math.isfinite(fwer_upper_p)
            and fwer_upper_p <= thresholds.familywise_alpha
        ),
        "track_above_token_permutation": (
            math.isfinite(track_real)
            and math.isfinite(track_shuffled)
            and track_real > track_shuffled
            and real_over_permutation_count
            >= thresholds.minimum_real_over_permutation_phases
        ),
        "temporal_consistency": (
            sum(consistent) >= thresholds.minimum_consistent_phases
            and longest_true_run(consistent)
            >= thresholds.minimum_longest_consistent_run
        ),
        "no_source_family_overlap_or_nesting_conflict": (
            len(duplicate_neighbors) == 0
        ),
    }
    eligible = all(gates.values())
    return {
        "role": ROLE_NAMES[role_index],
        "track_real": track_real,
        "track_shuffled": track_shuffled,
        "proposal_max_null_required_quantile": null_quantile,
        "track_null_percentile": null_percentile,
        "required_track_null_percentile": thresholds.required_track_null_percentile,
        "proposal_max_null_raw_upper_p": raw_upper_p,
        "three_role_bonferroni_fwer_upper_p": fwer_upper_p,
        "own_proposal_null_median": (
            float(np.median(own_track_null))
            if bool(np.isfinite(own_track_null).all())
            else float("nan")
        ),
        "distinct_null_track_scores": distinct_nulls,
        "consistent_phase_count": int(sum(consistent)),
        "longest_consistent_run": int(longest_true_run(consistent)),
        "real_over_permutation_phase_count": real_over_permutation_count,
        "consistent_phases": [
            phase for phase, passed in enumerate(consistent) if passed
        ],
        "duplicate_neighbors": list(duplicate_neighbors),
        "source_family_overlap_or_nesting_neighbors": list(duplicate_neighbors),
        "gates": gates,
        "eligible_before_proposal_competition": eligible,
        "evidence_margin": (
            track_real - null_quantile
            if math.isfinite(track_real) and math.isfinite(null_quantile)
            else float("nan")
        ),
    }


def _choose_without_forcing(
    *,
    role_index: int,
    evidence: Sequence[Mapping[str, Any]],
    proposal_ids: Sequence[str],
    real_scores: np.ndarray,
    thresholds: ProbeThresholdsV15C,
) -> tuple[int | None, Mapping[str, Any]]:
    eligible = [
        index
        for index, row in enumerate(evidence)
        if row["eligible_before_proposal_competition"] is True
    ]
    if not eligible:
        return None, {
            "status": "unassigned_no_eligible_proposal",
            "eligible_proposal_indices": [],
        }
    ordered = sorted(
        eligible,
        key=lambda index: (
            -float(evidence[index]["evidence_margin"]),
            proposal_ids[index],
        ),
    )
    winner = ordered[0]
    if len(ordered) == 1:
        return winner, {
            "status": "unique_eligible_proposal",
            "eligible_proposal_indices": ordered,
        }
    runner = ordered[1]
    winner_margin = float(evidence[winner]["evidence_margin"])
    runner_margin = float(evidence[runner]["evidence_margin"])
    if not winner_margin > runner_margin:
        return None, {
            "status": "unassigned_non_unique_top_evidence_margin",
            "eligible_proposal_indices": ordered,
            "eligible_proposal_ids": [proposal_ids[index] for index in ordered],
            "winner_index_if_forced": winner,
            "winner_id_if_forced": proposal_ids[winner],
            "top_evidence_margin": winner_margin,
            "runner_up_evidence_margin": runner_margin,
        }
    winner_phase = real_scores[role_index, winner]
    dominance_by_competitor: dict[str, int] = {}
    all_dominated = True
    for competitor in ordered[1:]:
        competitor_phase = real_scores[role_index, competitor]
        finite = np.isfinite(winner_phase) & np.isfinite(competitor_phase)
        dominance_count = int(np.sum(finite & (winner_phase > competitor_phase)))
        dominance_by_competitor[proposal_ids[competitor]] = dominance_count
        all_dominated = (
            all_dominated
            and dominance_count >= thresholds.minimum_proposal_dominance_phases
        )
    if not all_dominated:
        return None, {
            "status": "unassigned_winner_failed_all_eligible_temporal_dominance",
            "eligible_proposal_indices": ordered,
            "eligible_proposal_ids": [proposal_ids[index] for index in ordered],
            "winner_index_if_forced": winner,
            "winner_id_if_forced": proposal_ids[winner],
            "dominance_phase_count_by_competitor_id": dominance_by_competitor,
        }
    return winner, {
        "status": "unique_top_winner_dominated_every_eligible_proposal",
        "eligible_proposal_indices": ordered,
        "eligible_proposal_ids": [proposal_ids[index] for index in ordered],
        "winner_id": proposal_ids[winner],
        "runner_up_id": proposal_ids[runner],
        "dominance_phase_count_by_competitor_id": dominance_by_competitor,
    }


def run_source_object_proposal_role_probe_v15c(
    *,
    tracks: ProposalTrackInputV15C,
    affinity: R6AffinityInputV15C,
    thresholds: ProbeThresholdsV15C = ProbeThresholdsV15C(),
) -> Mapping[str, Any]:
    """Run the observer-only role probe and return a JSON-safe receipt payload."""

    real_scores, shuffled_scores, null_scores, scale_receipt = _phase_region_scores(
        affinity, tracks, thresholds
    )
    # The null family is chosen before looking at any role score.  For each of
    # the 64 preregistered null tokens, maximize over every geometrically valid
    # source proposal.  This prevents a proposal from being judged only against
    # its own (easier) null distribution.
    null_search_indices = [
        index
        for index, gate in enumerate(tracks.track_gate_pass)
        if gate is True and bool(np.isfinite(null_scores[index]).all())
    ]
    if null_search_indices:
        proposal_null_track = np.median(
            null_scores[null_search_indices].astype(np.float64), axis=2
        )
        proposal_max_null_track = np.max(proposal_null_track, axis=0)
        proposal_max_null_phase = np.max(
            null_scores[null_search_indices].astype(np.float64), axis=0
        )
    else:
        proposal_max_null_track = np.full(
            (NULL_COUNT,), np.nan, dtype=np.float64
        )
        proposal_max_null_phase = np.full(
            (NULL_COUNT, PHASE_COUNT), np.nan, dtype=np.float64
        )
    duplicates = duplicate_proposal_adjacency(tracks, thresholds)
    evidence: dict[str, list[dict[str, Any]]] = {}
    choices: dict[str, int | None] = {}
    competition: dict[str, Mapping[str, Any]] = {}
    for role_index, role in enumerate(ROLE_NAMES):
        rows: list[dict[str, Any]] = []
        for proposal_index, proposal_id in enumerate(tracks.proposal_ids):
            row = _candidate_evidence(
                role_index=role_index,
                proposal_index=proposal_index,
                real_scores=real_scores,
                shuffled_scores=shuffled_scores,
                null_scores=null_scores,
                proposal_max_null_phase=proposal_max_null_phase,
                proposal_max_null_track=proposal_max_null_track,
                track_valid=tracks.track_gate_pass[proposal_index],
                duplicate_neighbors=duplicates[proposal_id],
                thresholds=thresholds,
            )
            rows.append({"proposal_id": proposal_id, **row})
        evidence[role] = rows
        choice, detail = _choose_without_forcing(
            role_index=role_index,
            evidence=rows,
            proposal_ids=tracks.proposal_ids,
            real_scores=real_scores,
            thresholds=thresholds,
        )
        choices[role] = choice
        competition[role] = detail

    reverse: dict[int, list[str]] = {}
    for role, proposal_index in choices.items():
        if proposal_index is not None:
            reverse.setdefault(proposal_index, []).append(role)
    conflicts = {
        tracks.proposal_ids[index]: list(roles)
        for index, roles in reverse.items()
        if len(roles) > 1
    }
    for roles in conflicts.values():
        for role in roles:
            choices[role] = None
            competition[role] = {
                **competition[role],
                "status": "unassigned_cross_role_proposal_conflict",
            }

    assignments = {
        role: (tracks.proposal_ids[index] if index is not None else None)
        for role, index in choices.items()
    }
    complete = (
        all(value is not None for value in assignments.values())
        and len(set(assignments.values())) == len(ROLE_NAMES)
        and not conflicts
    )
    threshold_payload = {
        key: getattr(thresholds, key)
        for key in thresholds.__dataclass_fields__
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "OBSERVER_ROLE_TRACK_CANDIDATE_REQUIRES_OVERLAY_AUDIT"
            if complete
            else "NO_GO_FAIL_CLOSED_ROLE_TRACK_ASSIGNMENT"
        ),
        "role_names": list(ROLE_NAMES),
        "proposal_ids": list(tracks.proposal_ids),
        "block_indices": list(BLOCK_INDICES),
        "phase_frames": list(PHASE_FRAMES),
        "assignments": assignments,
        "cross_role_conflicts": conflicts,
        "competition": competition,
        "evidence": evidence,
        "source_proposal_family_overlap_nesting_adjacency": {
            key: list(value) for key, value in duplicates.items()
        },
        "duplicate_proposal_adjacency": {
            key: list(value) for key, value in duplicates.items()
        },
        "standardization": scale_receipt,
        "multiple_comparison_control": {
            "null_family": "per_null_token_max_over_every_geometry_valid_source_proposal",
            "null_search_proposal_ids": [
                tracks.proposal_ids[index] for index in null_search_indices
            ],
            "null_search_proposal_count": len(null_search_indices),
            "finite_null_count": (
                int(np.sum(np.isfinite(proposal_max_null_track)))
                if null_search_indices
                else 0
            ),
            "finite_null_method": "plus_one_empirical_upper_tail",
            "role_family": list(ROLE_NAMES),
            "role_family_count": thresholds.familywise_role_count,
            "familywise_alpha": thresholds.familywise_alpha,
            "correction": "Bonferroni_over_three_roles_after_proposal_max_null",
            "minimum_attainable_raw_p": 1.0 / float(NULL_COUNT + 1),
            "minimum_attainable_fwer_p": (
                thresholds.familywise_role_count / float(NULL_COUNT + 1)
            ),
        },
        "thresholds": threshold_payload,
        "mechanical_candidate_qualified": complete,
        "manual_full_track_overlay_audit_required": True,
        "localization_semantically_certified": False,
        "action_success_certified": False,
        "route_authorized": False,
        "decode_authorized": False,
        "training_authorized": False,
        "optimizer_updates": 0,
        "renderer_forward_calls": 0,
        "anchor_consumed": False,
        "target_instruction_consumed": False,
        "material_or_transparency_classification_consumed": False,
        "whole_object_semantically_certified": False,
        "source_pixels_modified": False,
        "affinity_used_pointwise_as_mask": False,
        "forced_assignment": False,
        "roi_or_manual_box_consumed": False,
    }
    payload = _json_safe(payload)
    payload["receipt_sha256"] = object_sha256(payload)
    return payload


def load_r6_affinity_for_v15c(path: Path) -> R6AffinityInputV15C:
    """Load only the sealed r6 raw role/null/permutation maps."""

    try:
        from safetensors.numpy import load_file
    except ImportError as error:  # pragma: no cover
        raise SourceProposalRoleProbeV15CError("safetensors is unavailable") from error
    tensors = load_file(str(path))
    real = np.stack(
        [tensors[f"block_{block:02d}_affinity"] for block in BLOCK_INDICES], axis=0
    )[:, 1:4]
    shuffled = np.stack(
        [
            tensors[f"block_{block:02d}_shuffled_affinity"]
            for block in BLOCK_INDICES
        ],
        axis=0,
    )[:, 1:4]
    null_bank = np.stack(
        [
            tensors[f"block_{block:02d}_null_span_affinity"]
            for block in BLOCK_INDICES
        ],
        axis=0,
    )
    return R6AffinityInputV15C(
        real=np.ascontiguousarray(real, dtype=np.float32),
        shuffled=np.ascontiguousarray(shuffled, dtype=np.float32),
        null_bank=np.ascontiguousarray(null_bank, dtype=np.float32),
    )


def load_tracks_for_v15c(
    metadata_path: Path, tensor_path: Path
) -> tuple[ProposalTrackInputV15C, Mapping[str, Any]]:
    try:
        from safetensors.numpy import load_file
    except ImportError as error:  # pragma: no cover
        raise SourceProposalRoleProbeV15CError("safetensors is unavailable") from error
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema_version") != TRACK_SCHEMA_VERSION
        or not isinstance(metadata.get("proposals"), list)
    ):
        raise SourceProposalRoleProbeV15CError("track metadata differs")
    tensors = load_file(str(tensor_path))
    if set(tensors) != {"phase_coverage"}:
        raise SourceProposalRoleProbeV15CError("track tensor registry differs")
    rows = metadata["proposals"]
    if not 1 <= len(rows) <= MAXIMUM_PROPOSAL_COUNT or any(
        not isinstance(row, Mapping)
        or type(row.get("automatic_track_geometry_gate_pass")) is not bool
        for row in rows
    ):
        raise SourceProposalRoleProbeV15CError("track gate registry differs")
    ids = tuple(row.get("proposal_id") for row in rows)
    gates = tuple(row["automatic_track_geometry_gate_pass"] for row in rows)
    return (
        ProposalTrackInputV15C(
            proposal_ids=ids,
            phase_coverage=np.ascontiguousarray(
                tensors["phase_coverage"], dtype=np.float32
            ),
            track_gate_pass=gates,
        ),
        metadata,
    )


__all__ = [
    "BLOCK_INDICES",
    "FULL_R6_ROLE_NAMES",
    "GRID_HEIGHT",
    "GRID_WIDTH",
    "NULL_COUNT",
    "PHASE_COUNT",
    "PHASE_FRAMES",
    "ProbeThresholdsV15C",
    "ProposalTrackInputV15C",
    "R6AffinityInputV15C",
    "ROLE_NAMES",
    "SCHEMA_VERSION",
    "SourceProposalRoleProbeV15CError",
    "TRACK_SCHEMA_VERSION",
    "duplicate_proposal_adjacency",
    "load_r6_affinity_for_v15c",
    "load_tracks_for_v15c",
    "run_source_object_proposal_role_probe_v15c",
]
