#!/usr/bin/env python3
"""ROI-blind source-role mask calibration for E00 v15b/r5 and r6.

The r5 artifact contains role maps but only one averaged ``null_affinity``.
That tensor is not a null localization-strength distribution, so this module
may expose standardized exploratory tracks for r5 but MUST return empty strict
masks.  A strict candidate is possible only when an authenticated bank of 64
individually observed non-special token/span maps is supplied.

No source ROI, anchor map, target video, route, trainer, or decoder is accepted
by this API.  Manual ROIs belong solely to the separate review renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "bernini-source-role-mask-calibration-v15b-r5"
ROLE_NAMES = ("agent", "old_actor", "new_actor", "recipient", "support")
VESSEL_ROLES = ("old_actor", "new_actor", "recipient")
BLOCKS = (4, 9, 14, 19, 24)
PHASES = 21
HEIGHT = 37
WIDTH = 25
NULL_BANK_SIZE = 64
ROBUST_SCALE = 1.4826
EXPLORATORY_Z_THRESHOLD = 3.0
NULL_STRENGTH_PERCENTILE = 0.95
MINIMUM_TRACK_PHASES = 3
MAX_TRACK_STEP_FRACTION = 0.28


class V15BR5CalibrationError(RuntimeError):
    """Fail-closed malformed observer/calibration input."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise V15BR5CalibrationError("calibration receipt is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _finite_float32(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.shape != shape or not np.isfinite(array).all():
        raise V15BR5CalibrationError(f"{label} must be finite float32 {shape}")
    return np.ascontiguousarray(array)


def spatial_median_mad_standardize(value: np.ndarray) -> np.ndarray:
    """Standardize the final HxW axes; invariant to positive affine scale."""

    array = np.asarray(value)
    if array.dtype != np.float32 or array.ndim < 2 or not np.isfinite(array).all():
        raise V15BR5CalibrationError("spatial standardization needs finite float32 maps")
    median = np.median(array, axis=(-2, -1), keepdims=True)
    mad = np.median(np.abs(array - median), axis=(-2, -1), keepdims=True)
    scale = ROBUST_SCALE * mad
    # Constant maps contain no localization evidence and remain exactly zero.
    result = np.divide(
        array - median,
        scale,
        out=np.zeros_like(array, dtype=np.float32),
        where=scale > np.float32(1e-12),
    )
    return np.ascontiguousarray(result.astype(np.float32, copy=False))


def vessel_standardized_winners(
    standardized: np.ndarray,
    *,
    role_names: Sequence[str] = ROLE_NAMES,
) -> np.ndarray:
    """Return peer winner indices using only standardized vessel maps."""

    names = tuple(role_names)
    if len(names) != len(set(names)) or any(name not in names for name in VESSEL_ROLES):
        raise V15BR5CalibrationError("role registry differs")
    scores = np.asarray(standardized)
    if scores.ndim != 4 or scores.shape[0] != len(names):
        raise V15BR5CalibrationError("winner input geometry differs")
    vessel_indices = [names.index(name) for name in VESSEL_ROLES]
    return np.argmax(scores[vessel_indices], axis=0).astype(np.int8)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    if mask.dtype != np.bool_ or mask.shape != (HEIGHT, WIDTH):
        raise V15BR5CalibrationError("component mask geometry differs")
    visited = np.zeros_like(mask)
    output: list[np.ndarray] = []
    for row in range(HEIGHT):
        for col in range(WIDTH):
            if not mask[row, col] or visited[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            points: list[tuple[int, int]] = []
            while stack:
                current_row, current_col = stack.pop()
                points.append((current_row, current_col))
                for next_row, next_col in (
                    (current_row - 1, current_col),
                    (current_row + 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                ):
                    if (
                        0 <= next_row < HEIGHT
                        and 0 <= next_col < WIDTH
                        and mask[next_row, next_col]
                        and not visited[next_row, next_col]
                    ):
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))
            component = np.zeros_like(mask)
            rows, cols = zip(*points)
            component[np.asarray(rows), np.asarray(cols)] = True
            output.append(component)
    return output


@dataclass(frozen=True)
class _TrackNode:
    phase: int
    mask: np.ndarray
    peak_z: float
    peak_row: int
    peak_col: int
    centroid_row: float
    centroid_col: float

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def _phase_nodes(value: np.ndarray, *, phase: int, threshold: float) -> list[_TrackNode]:
    if value.shape != (HEIGHT, WIDTH) or value.dtype != np.float32:
        raise V15BR5CalibrationError("track map geometry differs")
    if not math.isfinite(threshold):
        raise V15BR5CalibrationError("track threshold is not finite")
    nodes: list[_TrackNode] = []
    for component in _components(value >= np.float32(threshold)):
        flat_indices = np.flatnonzero(component.reshape(-1))
        if not len(flat_indices):
            continue
        values = value.reshape(-1)[flat_indices]
        peak_flat = int(flat_indices[int(values.argmax())])
        rows, cols = np.nonzero(component)
        nodes.append(
            _TrackNode(
                phase=phase,
                mask=component,
                peak_z=float(value.reshape(-1)[peak_flat]),
                peak_row=peak_flat // WIDTH,
                peak_col=peak_flat % WIDTH,
                centroid_row=float(rows.mean()),
                centroid_col=float(cols.mean()),
            )
        )
    return nodes


def _normalized_distance(left: _TrackNode, right: _TrackNode) -> float:
    row = (left.centroid_row - right.centroid_row) / max(1, HEIGHT - 1)
    col = (left.centroid_col - right.centroid_col) / max(1, WIDTH - 1)
    return float(math.sqrt(row * row + col * col))


def _best_consecutive_track(nodes_by_phase: Sequence[Sequence[_TrackNode]]) -> list[_TrackNode]:
    """Dynamic-program one consecutive path; an empty result is valid."""

    if len(nodes_by_phase) != PHASES:
        raise V15BR5CalibrationError("track phase registry differs")
    all_nodes: list[_TrackNode] = []
    scores: list[float] = []
    lengths: list[int] = []
    previous: list[int | None] = []
    indices_by_phase: list[list[int]] = []
    for phase, phase_nodes in enumerate(nodes_by_phase):
        phase_indices: list[int] = []
        prior_indices = indices_by_phase[phase - 1] if phase else []
        for node in phase_nodes:
            best_score = node.peak_z
            best_length = 1
            best_previous: int | None = None
            for prior_index in prior_indices:
                prior = all_nodes[prior_index]
                distance = _normalized_distance(prior, node)
                if distance > MAX_TRACK_STEP_FRACTION:
                    continue
                candidate_score = scores[prior_index] + node.peak_z - 2.0 * distance
                candidate_length = lengths[prior_index] + 1
                if (candidate_score, candidate_length, -prior_index) > (
                    best_score,
                    best_length,
                    -(best_previous if best_previous is not None else len(all_nodes) + 1),
                ):
                    best_score = candidate_score
                    best_length = candidate_length
                    best_previous = prior_index
            index = len(all_nodes)
            all_nodes.append(node)
            scores.append(best_score)
            lengths.append(best_length)
            previous.append(best_previous)
            phase_indices.append(index)
        indices_by_phase.append(phase_indices)
    if not all_nodes:
        return []
    end = max(range(len(all_nodes)), key=lambda i: (lengths[i], scores[i], -i))
    track: list[_TrackNode] = []
    while end is not None:
        track.append(all_nodes[end])
        end = previous[end]
    track.reverse()
    return track


def _track_receipt(track: Sequence[_TrackNode]) -> Mapping[str, Any]:
    distances = [
        _normalized_distance(left, right) for left, right in zip(track, track[1:])
    ]
    coherence = math.exp(-float(np.mean(distances)) / 0.15) if distances else 0.0
    return {
        "phase_count": len(track),
        "phases": [item.phase for item in track],
        "areas": [item.area for item in track],
        "peaks": [
            [item.phase, item.peak_row, item.peak_col, item.peak_z] for item in track
        ],
        "mean_normalized_step": float(np.mean(distances)) if distances else None,
        "temporal_coherence": coherence,
    }


def _null_strength_percentile(real_strength: float, null_strengths: np.ndarray) -> float:
    values = np.asarray(null_strengths, dtype=np.float64)
    if values.shape != (NULL_BANK_SIZE,) or not np.isfinite(values).all():
        raise V15BR5CalibrationError("null strength distribution differs")
    below = float(np.sum(values < real_strength))
    equal = float(np.sum(values == real_strength))
    return (below + 0.5 * equal) / NULL_BANK_SIZE


@dataclass(frozen=True)
class CalibrationResult:
    standardized_role_maps: np.ndarray
    exploratory_track_masks: np.ndarray
    strict_block_masks: np.ndarray
    strict_aggregate_masks: np.ndarray
    receipt: Mapping[str, Any]


def calibrate_source_role_maps(
    role_maps: np.ndarray,
    *,
    null_span_maps: np.ndarray | None,
    null_registry_sha256: str | None,
    role_names: Sequence[str] = ROLE_NAMES,
    block_indices: Sequence[int] = BLOCKS,
) -> CalibrationResult:
    """Calibrate maps without any spatial supervision.

    ``role_maps`` is ``[B,R,21,37,25]``.  A valid r6 null bank is
    ``[B,64,21,37,25]``.  Missing or malformed bank evidence never falls back
    to the legacy averaged null map; it yields empty strict masks.
    """

    names = tuple(role_names)
    blocks = tuple(block_indices)
    if names != ROLE_NAMES or blocks != BLOCKS:
        raise V15BR5CalibrationError("registered role/block order differs")
    real = _finite_float32(
        role_maps,
        (len(BLOCKS), len(ROLE_NAMES), PHASES, HEIGHT, WIDTH),
        "role maps",
    )
    standardized = spatial_median_mad_standardize(real)
    exploratory = np.zeros_like(standardized, dtype=np.bool_)
    exploratory_receipts: dict[str, Any] = {}
    for block_offset, block in enumerate(BLOCKS):
        exploratory_receipts[str(block)] = {}
        winners = vessel_standardized_winners(standardized[block_offset])
        for role_index, role in enumerate(ROLE_NAMES):
            phase_nodes = []
            for phase in range(PHASES):
                allowed = standardized[block_offset, role_index, phase] >= np.float32(
                    EXPLORATORY_Z_THRESHOLD
                )
                if role in VESSEL_ROLES:
                    allowed &= winners[phase] == VESSEL_ROLES.index(role)
                masked_value = np.where(
                    allowed,
                    standardized[block_offset, role_index, phase],
                    np.float32(-np.inf),
                ).astype(np.float32)
                phase_nodes.append(
                    _phase_nodes(
                        masked_value,
                        phase=phase,
                        threshold=EXPLORATORY_Z_THRESHOLD,
                    )
                )
            track = _best_consecutive_track(phase_nodes)
            if len(track) >= MINIMUM_TRACK_PHASES:
                for node in track:
                    exploratory[block_offset, role_index, node.phase] = node.mask
            exploratory_receipts[str(block)][role] = _track_receipt(track)

    strict_blocks = np.zeros_like(exploratory)
    strict_aggregate = np.zeros(
        (len(ROLE_NAMES), PHASES, HEIGHT, WIDTH), dtype=np.bool_
    )
    bank_status = "present"
    if null_span_maps is None:
        bank_status = "null_token_bank_absent"
    else:
        nulls = _finite_float32(
            null_span_maps,
            (len(BLOCKS), NULL_BANK_SIZE, PHASES, HEIGHT, WIDTH),
            "null span maps",
        )
        if (
            not isinstance(null_registry_sha256, str)
            or len(null_registry_sha256) != 64
            or any(item not in "0123456789abcdef" for item in null_registry_sha256)
        ):
            raise V15BR5CalibrationError("null registry SHA is missing/invalid")
        null_z = spatial_median_mad_standardize(nulls)
        # Peak/track statistics are scalar distributions across 64 controls;
        # there is intentionally no pointwise real-minus-null operation.
        weights: dict[str, dict[str, float]] = {role: {} for role in ROLE_NAMES}
        candidate_tracks: dict[tuple[int, int], list[_TrackNode]] = {}
        candidate_percentiles: dict[tuple[int, int], list[float]] = {}
        for block_offset, block in enumerate(BLOCKS):
            winners = vessel_standardized_winners(standardized[block_offset])
            null_strength = np.max(null_z[block_offset], axis=(-2, -1))
            for role_index, role in enumerate(ROLE_NAMES):
                phase_nodes = []
                percentiles = []
                for phase in range(PHASES):
                    real_strength = float(
                        standardized[block_offset, role_index, phase].max()
                    )
                    percentile = _null_strength_percentile(
                        real_strength, null_strength[:, phase]
                    )
                    percentiles.append(percentile)
                    null_site_threshold = float(
                        np.quantile(null_z[block_offset, :, phase], 0.995)
                    )
                    threshold = max(EXPLORATORY_Z_THRESHOLD, null_site_threshold)
                    allowed = standardized[block_offset, role_index, phase] >= threshold
                    if percentile < NULL_STRENGTH_PERCENTILE:
                        allowed[:] = False
                    if role in VESSEL_ROLES:
                        allowed &= winners[phase] == VESSEL_ROLES.index(role)
                    phase_nodes.append(
                        _phase_nodes(
                            np.where(
                                allowed,
                                standardized[block_offset, role_index, phase],
                                np.float32(-np.inf),
                            ).astype(np.float32),
                            phase=phase,
                            threshold=threshold,
                        )
                    )
                track = _best_consecutive_track(phase_nodes)
                candidate_tracks[(block_offset, role_index)] = track
                candidate_percentiles[(block_offset, role_index)] = percentiles
                track_info = _track_receipt(track)
                track_percentiles = [percentiles[node.phase] for node in track]
                median_percentile = (
                    float(np.median(track_percentiles)) if track_percentiles else 0.0
                )
                weights[role][str(block)] = (
                    median_percentile * float(track_info["temporal_coherence"])
                    if len(track) >= MINIMUM_TRACK_PHASES
                    else 0.0
                )
        # Role-wise weights use only null percentile and temporal coherence.
        for role_index, role in enumerate(ROLE_NAMES):
            values = np.asarray(
                [weights[role][str(block)] for block in BLOCKS], dtype=np.float64
            )
            total = float(values.sum())
            if total <= 0.0:
                continue
            normalized = values / total
            for block_offset, weight in enumerate(normalized):
                if weight <= 0.0:
                    continue
                track = candidate_tracks[(block_offset, role_index)]
                for node in track:
                    strict_blocks[block_offset, role_index, node.phase] = node.mask
            aggregate_z = np.sum(
                standardized[:, role_index] * normalized[:, None, None, None], axis=0
            ).astype(np.float32)
            aggregate_null_z = np.sum(
                null_z * normalized[:, None, None, None, None], axis=0
            ).astype(np.float32)
            null_strength = np.max(aggregate_null_z, axis=(-2, -1))
            phase_nodes = []
            for phase in range(PHASES):
                percentile = _null_strength_percentile(
                    float(aggregate_z[phase].max()), null_strength[:, phase]
                )
                threshold = max(
                    EXPLORATORY_Z_THRESHOLD,
                    float(np.quantile(aggregate_null_z[:, phase], 0.995)),
                )
                allowed = aggregate_z[phase] >= threshold
                if percentile < NULL_STRENGTH_PERCENTILE:
                    allowed[:] = False
                # Aggregate vessel peer competition is deferred until every
                # role map exists below; no independent role enters it.
                phase_nodes.append(
                    _phase_nodes(
                        np.where(allowed, aggregate_z[phase], np.float32(-np.inf)).astype(
                            np.float32
                        ),
                        phase=phase,
                        threshold=threshold,
                    )
                )
            track = _best_consecutive_track(phase_nodes)
            if len(track) >= MINIMUM_TRACK_PHASES:
                for node in track:
                    strict_aggregate[role_index, node.phase] = node.mask
        # Enforce peer competition only among vessels, using weighted
        # standardized role evidence, never an appearance/part/group layer.
        weighted_roles = np.zeros_like(strict_aggregate, dtype=np.float32)
        for role_index, role in enumerate(ROLE_NAMES):
            values = np.asarray(
                [weights[role][str(block)] for block in BLOCKS], dtype=np.float64
            )
            if float(values.sum()) > 0.0:
                values /= float(values.sum())
                weighted_roles[role_index] = np.sum(
                    standardized[:, role_index] * values[:, None, None, None], axis=0
                )
        vessel_winners = vessel_standardized_winners(weighted_roles)
        for vessel_offset, role in enumerate(VESSEL_ROLES):
            role_index = ROLE_NAMES.index(role)
            strict_aggregate[role_index] &= vessel_winners == vessel_offset
        bank_status = "null_token_bank_present_calibration_candidate_only"

    strict_count = int(strict_aggregate.sum())
    receipt_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "role_names": list(ROLE_NAMES),
        "block_indices": list(BLOCKS),
        "geometry": [PHASES, HEIGHT, WIDTH],
        "standardization": "per_block_role_phase_spatial_median_1.4826MAD",
        "vessel_competition": "standardized_maps_only_old_new_recipient",
        "component_connectivity": 4,
        "track_policy": {
            "consecutive_phases_only": True,
            "minimum_track_phases": MINIMUM_TRACK_PHASES,
            "maximum_normalized_step": MAX_TRACK_STEP_FRACTION,
            "fixed_quota": False,
            "forced_nonempty": False,
        },
        "block_role_weight_inputs": [
            "null_peak_strength_percentile",
            "temporal_track_coherence",
        ],
        "spatial_supervision_inputs": [],
        "legacy_averaged_null_consumed": False,
        "null_bank_status": bank_status,
        "null_bank_size_required": NULL_BANK_SIZE,
        "null_registry_sha256": null_registry_sha256,
        "exploratory_tracks": exploratory_receipts,
        "strict_mask_pixel_count": strict_count,
        "localization_semantically_certified": False,
        "action_success_certified": False,
        "route_authorized": False,
        "training_authorized": False,
        "decode_authorized": False,
    }
    if null_span_maps is None:
        if strict_count != 0:
            raise V15BR5CalibrationError("missing null bank produced a strict mask")
        receipt_payload["status"] = "strict_fail_null_token_bank_absent"
        receipt_payload["mechanical_candidate_qualified"] = False
    else:
        receipt_payload["status"] = "candidate_requires_manual_overlay_audit"
        receipt_payload["mechanical_candidate_qualified"] = bool(strict_count)
    receipt = {
        **receipt_payload,
        "receipt_sha256": object_sha256(receipt_payload),
    }
    return CalibrationResult(
        standardized_role_maps=standardized,
        exploratory_track_masks=np.ascontiguousarray(exploratory),
        strict_block_masks=np.ascontiguousarray(strict_blocks),
        strict_aggregate_masks=np.ascontiguousarray(strict_aggregate),
        receipt=receipt,
    )


__all__ = [
    "BLOCKS",
    "CalibrationResult",
    "NULL_BANK_SIZE",
    "ROLE_NAMES",
    "V15BR5CalibrationError",
    "VESSEL_ROLES",
    "calibrate_source_role_maps",
    "object_sha256",
    "spatial_median_mad_standardize",
    "vessel_standardized_winners",
]
