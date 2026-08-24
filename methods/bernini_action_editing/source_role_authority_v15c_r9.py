#!/usr/bin/env python3
"""Four-role, source-only authority and ownership partition for v15c-r9.

This module extends the v15c-r8 LOCAL replay contract without changing it.
SAM2 supplies generic proposals, tracks, and replayed raw signed-valued logits;
the frozen r6 attn2 observer supplies text affinity for ``agent`` plus the three
vessel roles.  No anchor, target editing instruction, appearance classifier,
renderer, decoder, optimizer, or training path is present here.

Raw proposal masks are allowed to overlap at hand/object contact and occlusion.
They are never passed directly to the v15b ownership interface.  The ownership
partition uses a strict, deterministic argmax of replayed SAM2 logits.  A tie or
insufficient margin becomes unassigned.  If the subtraction caused by that
partition changes 4-connectivity, hole topology, area support, or temporal
continuity, the entire affected role is unassigned; masks are never repaired.

The future certifying ABI requires 64 aligned joint null replicates with an
explicit four-role axis.  For every replicate, the null statistic is maximized
over all geometry-valid proposals and all four separately observed role-null
statistics; every real role then uses that global max-T distribution and the
+1 empirical upper-tail p-value.  The sealed r6 artifact has only one common
64-null axis.  Repeating it over four slots would not control four-role FWER,
so r6 is retained as diagnostic evidence but its assignment result is NO-GO.
The three vessel roles keep their historical Bonferroni gate as an additional
gate if a future exact joint-null artifact becomes available.
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

try:
    from . import source_object_proposal_role_probe_v15c as r8_core
except ImportError:  # pragma: no cover - flat sealed snapshot
    import source_object_proposal_role_probe_v15c as r8_core


SCHEMA_VERSION = "bernini-source-four-role-authority-v15c-r9-local"
OWNERSHIP_SCHEMA_VERSION = "bernini-source-four-role-ownership-v15c-r9-local"
V15B_ADAPTER_SCHEMA_VERSION = (
    "bernini-source-four-role-ownership-v15c-r9-to-v15b-r8-local"
)
TRACK_SCHEMA_VERSION = r8_core.TRACK_SCHEMA_VERSION
ROLE_NAMES = ("human_agent", "old_actor", "new_actor", "recipient")
VESSEL_ROLE_NAMES = ("old_actor", "new_actor", "recipient")
FULL_R6_ROLE_NAMES = r8_core.FULL_R6_ROLE_NAMES
R6_ROLE_INDEX = {
    "human_agent": 0,
    "old_actor": 1,
    "new_actor": 2,
    "recipient": 3,
}
BLOCK_INDICES = r8_core.BLOCK_INDICES
PHASE_FRAMES = r8_core.PHASE_FRAMES
PHASE_COUNT = r8_core.PHASE_COUNT
GRID_HEIGHT = r8_core.GRID_HEIGHT
GRID_WIDTH = r8_core.GRID_WIDTH
NULL_COUNT = r8_core.NULL_COUNT
MAXIMUM_PROPOSAL_COUNT = r8_core.MAXIMUM_PROPOSAL_COUNT
PROPOSAL_ID_PATTERN = r8_core.PROPOSAL_ID_PATTERN
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
V15C_R10_JOINT_NULL_REGISTRY_SHA256 = (
    "b184cef1a7e797c24226b7615057cfb284b7ac9cf8a40657871f705cc696d79e"
)
V15C_R10_ROLE_NULL_REGISTRY_SHA256 = (
    "4744c8165efb3710c76e25ed1a2e796fe82dc72a8aa12f7dc1ddbbb60dec85fb",
    "4a8949a10e837a7821daf4e77852924df31e908954de020d16b82a6b4f15b023",
    "1ca1bfa1e683a00cc5fc209665d92fdc544798cfa87ca00db48806b75a104c0b",
    "2fd677f9bdce917ff7ed5a5df6ec0236c77777ad6dacbea86fd68e82b0c27e75",
)
V15C_R10_BINDING_SCHEMA_VERSION = "bernini-v15c-r10-joint-null-to-r9-affinity-v1"


class SourceRoleAuthorityV15CR9Error(RuntimeError):
    """An r9 source authority, statistic, or ownership gate differs."""


def canonical_bytes(value: Any) -> bytes:
    return r8_core.canonical_bytes(value)


def object_sha256(value: Any) -> str:
    return r8_core.object_sha256(value)


def file_sha256(path: Path) -> str:
    return r8_core.file_sha256(path)


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value)
    descriptor = {
        "dtype": array.dtype.str,
        "shape": [int(item) for item in array.shape],
        "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }
    return object_sha256(descriptor)


def joint_null_binding_payload_v15c_r9(
    *,
    real: np.ndarray,
    shuffled: np.ndarray,
    null_bank: np.ndarray,
    null_registry_sha256: str,
    role_null_registry_sha256: Sequence[str],
    role_null_tensor_sha256: Sequence[str],
    upstream_validation: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Rebuild a local r10-to-r9 mechanical-unit binding payload.

    This makes ``four_role_joint_null_available`` dependent on the pinned r10
    registry, four distinct role registries, four recomputed role tensor
    digests, and an upstream strict-loader validation receipt.  It also rejects
    the former common-null broadcast construction before any max-T statistic.
    This payload is not external observer authority; the public r9 runner still
    refuses every available=True input until a fresh r10 runner exists.
    """

    expected_real_shape = (
        len(BLOCK_INDICES), len(ROLE_NAMES), PHASE_COUNT,
        GRID_HEIGHT, GRID_WIDTH,
    )
    expected_null_shape = (
        len(BLOCK_INDICES), len(ROLE_NAMES), NULL_COUNT,
        PHASE_COUNT, GRID_HEIGHT, GRID_WIDTH,
    )
    if (
        not isinstance(real, np.ndarray)
        or real.shape != expected_real_shape
        or real.dtype != np.float32
        or not real.flags.c_contiguous
        or not bool(np.isfinite(real).all())
        or not isinstance(shuffled, np.ndarray)
        or shuffled.shape != expected_real_shape
        or shuffled.dtype != np.float32
        or not shuffled.flags.c_contiguous
        or not bool(np.isfinite(shuffled).all())
        or not isinstance(null_bank, np.ndarray)
        or null_bank.shape != expected_null_shape
        or null_bank.dtype != np.float32
        or not null_bank.flags.c_contiguous
        or not bool(np.isfinite(null_bank).all())
        or null_registry_sha256 != V15C_R10_JOINT_NULL_REGISTRY_SHA256
        or tuple(role_null_registry_sha256)
        != V15C_R10_ROLE_NULL_REGISTRY_SHA256
        or len(set(role_null_registry_sha256)) != len(ROLE_NAMES)
        or type(upstream_validation) is not dict
        or set(upstream_validation) != {
            "schema_version", "validation_sha256",
            "capture_channel_registry_sha256",
            "capture_channel_value_binding_sha256",
            "independent_capture_channel_value_binding_pinned",
            "actual_sp4_rank_shard_files_replayed", "official_r10_runner_present",
            "role_assignment_mechanical_candidate_qualified",
            "route_authorized", "decode_authorized", "training_authorized",
        }
        or upstream_validation["schema_version"]
        != "bernini-four-role-joint-null-observer-v15c-r10-local"
        or type(upstream_validation["validation_sha256"]) is not str
        or SHA256_PATTERN.fullmatch(upstream_validation["validation_sha256"])
        is None
        or type(upstream_validation["capture_channel_registry_sha256"]) is not str
        or SHA256_PATTERN.fullmatch(
            upstream_validation["capture_channel_registry_sha256"]
        ) is None
        or type(upstream_validation["capture_channel_value_binding_sha256"])
        is not str
        or SHA256_PATTERN.fullmatch(
            upstream_validation["capture_channel_value_binding_sha256"]
        ) is None
        or any(
            type(upstream_validation[name]) is not bool
            for name in (
                "independent_capture_channel_value_binding_pinned",
                "actual_sp4_rank_shard_files_replayed", "official_r10_runner_present",
                "role_assignment_mechanical_candidate_qualified",
                "route_authorized", "decode_authorized", "training_authorized",
            )
        )
    ):
        raise SourceRoleAuthorityV15CR9Error("r10 joint-null binding differs")
    recomputed = tuple(
        array_sha256(null_bank[:, role]) for role in range(len(ROLE_NAMES))
    )
    if (
        tuple(role_null_tensor_sha256) != recomputed
        or len(set(recomputed)) != len(ROLE_NAMES)
        or any(
            np.array_equal(null_bank[:, left], null_bank[:, right])
            for left in range(len(ROLE_NAMES))
            for right in range(left + 1, len(ROLE_NAMES))
        )
    ):
        raise SourceRoleAuthorityV15CR9Error(
            "common/broadcast/byte-identical role-null tensor differs"
        )
    return {
        "schema_version": V15C_R10_BINDING_SCHEMA_VERSION,
        "joint_null_registry_sha256": null_registry_sha256,
        "role_names": list(ROLE_NAMES),
        "real_tensor_shape": list(expected_real_shape),
        "null_tensor_shape": list(expected_null_shape),
        "real_tensor_sha256": array_sha256(real),
        "shuffled_tensor_sha256": array_sha256(shuffled),
        "role_null_registry_sha256": list(role_null_registry_sha256),
        "role_null_tensor_sha256": list(recomputed),
        "upstream_validation": dict(upstream_validation),
        "null_index_alignment": (
            "same_joint_index_j_with_distinct_preregistered_role_controls"
        ),
        "common_null_broadcast_used": False,
    }


def _finite_number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise SourceRoleAuthorityV15CR9Error(f"{label} differs")
    return float(value)


@dataclass(frozen=True)
class WholeTrackGeometryV15CR9:
    """Source-mask geometry already recomputed by the r8 byte replay."""

    all_81_frames_visible: bool
    area_p95_to_p05_ratio: float
    median_adjacent_iou: float
    p10_area_pixels: float
    median_largest_component_fraction: float
    median_bbox_fill_fraction: float
    p10_bbox_diagonal_frame_fraction: float

    def __post_init__(self) -> None:
        if type(self.all_81_frames_visible) is not bool:
            raise SourceRoleAuthorityV15CR9Error("visibility gate differs")
        values = (
            self.area_p95_to_p05_ratio,
            self.median_adjacent_iou,
            self.p10_area_pixels,
            self.median_largest_component_fraction,
            self.median_bbox_fill_fraction,
            self.p10_bbox_diagonal_frame_fraction,
        )
        if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in values):
            raise SourceRoleAuthorityV15CR9Error("track geometry differs")
        if (
            self.area_p95_to_p05_ratio < 1.0
            or not 0.0 <= self.median_adjacent_iou <= 1.0
            or self.p10_area_pixels < 0.0
            or not 0.0 <= self.median_largest_component_fraction <= 1.0
            or not 0.0 <= self.median_bbox_fill_fraction <= 1.0
            or not 0.0 <= self.p10_bbox_diagonal_frame_fraction <= math.sqrt(2.0)
        ):
            raise SourceRoleAuthorityV15CR9Error("track geometry range differs")


@dataclass(frozen=True)
class ProposalTrackInputV15CR9:
    proposal_ids: tuple[str, ...]
    phase_coverage: np.ndarray
    track_gate_pass: tuple[bool, ...]
    geometry: tuple[WholeTrackGeometryV15CR9, ...]

    def __post_init__(self) -> None:
        count = len(self.proposal_ids)
        if (
            not 1 <= count <= MAXIMUM_PROPOSAL_COUNT
            or len(set(self.proposal_ids)) != count
            or any(type(item) is not str or PROPOSAL_ID_PATTERN.fullmatch(item) is None for item in self.proposal_ids)
            or len(self.track_gate_pass) != count
            or any(type(item) is not bool for item in self.track_gate_pass)
            or len(self.geometry) != count
            or any(type(item) is not WholeTrackGeometryV15CR9 for item in self.geometry)
            or not isinstance(self.phase_coverage, np.ndarray)
            or self.phase_coverage.shape != (count, PHASE_COUNT, GRID_HEIGHT, GRID_WIDTH)
            or self.phase_coverage.dtype.kind not in "fc"
            or not bool(np.isfinite(self.phase_coverage).all())
            or bool((self.phase_coverage < 0.0).any())
            or bool((self.phase_coverage > 1.0).any())
        ):
            raise SourceRoleAuthorityV15CR9Error("proposal track registry differs")


@dataclass(frozen=True)
class R6AffinityInputV15CR9:
    real: np.ndarray
    shuffled: np.ndarray
    null_bank: np.ndarray
    null_registry_sha256: str
    null_index_alignment_verified: bool
    four_role_joint_null_available: bool
    role_null_registry_sha256: tuple[str, ...] = ()
    role_null_tensor_sha256: tuple[str, ...] = ()
    joint_null_upstream_validation: Mapping[str, Any] | None = None
    joint_null_binding_sha256: str | None = None

    def __post_init__(self) -> None:
        contracts = (
            (self.real, (len(BLOCK_INDICES), len(ROLE_NAMES), PHASE_COUNT, GRID_HEIGHT, GRID_WIDTH)),
            (self.shuffled, (len(BLOCK_INDICES), len(ROLE_NAMES), PHASE_COUNT, GRID_HEIGHT, GRID_WIDTH)),
            (
                self.null_bank,
                (
                    len(BLOCK_INDICES),
                    len(ROLE_NAMES),
                    NULL_COUNT,
                    PHASE_COUNT,
                    GRID_HEIGHT,
                    GRID_WIDTH,
                ),
            ),
        )
        if any(
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype != np.float32
            or not value.flags.c_contiguous
            or not bool(np.isfinite(value).all())
            for value, shape in contracts
        ):
            raise SourceRoleAuthorityV15CR9Error("r6 affinity tensor contract differs")
        if (
            type(self.null_registry_sha256) is not str
            or SHA256_PATTERN.fullmatch(self.null_registry_sha256) is None
            or type(self.null_index_alignment_verified) is not bool
            or type(self.four_role_joint_null_available) is not bool
        ):
            raise SourceRoleAuthorityV15CR9Error("null registry alignment differs")
        if self.four_role_joint_null_available:
            if self.null_index_alignment_verified is not True:
                raise SourceRoleAuthorityV15CR9Error(
                    "joint-null availability lacks index alignment"
                )
            payload = joint_null_binding_payload_v15c_r9(
                real=self.real,
                shuffled=self.shuffled,
                null_bank=self.null_bank,
                null_registry_sha256=self.null_registry_sha256,
                role_null_registry_sha256=self.role_null_registry_sha256,
                role_null_tensor_sha256=self.role_null_tensor_sha256,
                upstream_validation=self.joint_null_upstream_validation,
            )
            if (
                type(self.joint_null_binding_sha256) is not str
                or self.joint_null_binding_sha256 != object_sha256(payload)
            ):
                raise SourceRoleAuthorityV15CR9Error(
                    "joint-null adapter binding differs"
                )
        elif (
            self.role_null_registry_sha256 != ()
            or self.role_null_tensor_sha256 != ()
            or self.joint_null_upstream_validation is not None
            or self.joint_null_binding_sha256 is not None
        ):
            raise SourceRoleAuthorityV15CR9Error(
                "diagnostic common-null input claims r10 joint provenance"
            )


@dataclass(frozen=True)
class RoleThresholdsV15CR9:
    familywise_alpha: float = 0.05
    phase_global_max_null_percentile: float = 0.90
    minimum_consistent_phases: int = 13
    minimum_longest_consistent_run: int = 4
    minimum_real_over_permutation_phases: int = 14
    minimum_proposal_dominance_phases: int = 16
    minimum_distinct_null_track_scores: int = 16
    null_track_score_epsilon: float = 1.0e-6
    duplicate_median_iou: float = 0.55
    duplicate_median_containment: float = 0.80
    vessel_conflict_median_iou: float = 0.10
    vessel_conflict_median_containment: float = 0.50
    human_vessel_max_median_iou: float = 0.12
    human_vessel_max_median_human_containment: float = 0.08
    human_vessel_max_median_vessel_containment: float = 0.65
    human_vessel_max_p95_vessel_containment: float = 0.95
    minimum_phase_coverage_mass: float = 0.20
    human_maximum_area_p95_to_p05_ratio: float = 4.0
    human_minimum_median_adjacent_iou: float = 0.35
    human_minimum_p10_area_pixels: float = 7434.0
    human_minimum_median_largest_component_fraction: float = 0.90
    human_minimum_median_bbox_fill_fraction: float = 0.18
    human_minimum_p10_bbox_diagonal_frame_fraction: float = 0.18
    ownership_logit_margin: float = 1.0e-4
    ownership_minimum_area_retention: float = 0.85
    ownership_minimum_median_area_retention: float = 0.97
    ownership_minimum_median_adjacent_iou: float = 0.20

    def __post_init__(self) -> None:
        fraction_names = (
            "familywise_alpha",
            "phase_global_max_null_percentile",
            "duplicate_median_iou",
            "duplicate_median_containment",
            "vessel_conflict_median_iou",
            "vessel_conflict_median_containment",
            "human_vessel_max_median_iou",
            "human_vessel_max_median_human_containment",
            "human_vessel_max_median_vessel_containment",
            "human_vessel_max_p95_vessel_containment",
            "minimum_phase_coverage_mass",
            "human_minimum_median_adjacent_iou",
            "human_minimum_median_largest_component_fraction",
            "human_minimum_median_bbox_fill_fraction",
            "human_minimum_p10_bbox_diagonal_frame_fraction",
            "ownership_minimum_area_retention",
            "ownership_minimum_median_area_retention",
            "ownership_minimum_median_adjacent_iou",
        )
        if any(not 0.0 < _finite_number(getattr(self, name), name) < 1.0 for name in fraction_names):
            raise SourceRoleAuthorityV15CR9Error("fraction threshold differs")
        if (
            not 1 <= self.minimum_consistent_phases <= PHASE_COUNT
            or not 1 <= self.minimum_longest_consistent_run <= PHASE_COUNT
            or not 1 <= self.minimum_real_over_permutation_phases <= PHASE_COUNT
            or not 1 <= self.minimum_proposal_dominance_phases <= PHASE_COUNT
            or not 2 <= self.minimum_distinct_null_track_scores <= NULL_COUNT
            or _finite_number(self.null_track_score_epsilon, "null epsilon") <= 0.0
            or _finite_number(self.human_maximum_area_p95_to_p05_ratio, "human area ratio") < 1.0
            or _finite_number(self.human_minimum_p10_area_pixels, "human area") <= 0.0
            or _finite_number(self.ownership_logit_margin, "ownership margin") <= 0.0
        ):
            raise SourceRoleAuthorityV15CR9Error("threshold contract differs")

    @property
    def global_max_t_required_percentile(self) -> float:
        return 1.0 - self.familywise_alpha

    @property
    def vessel_required_percentile(self) -> float:
        return 1.0 - self.familywise_alpha / float(len(VESSEL_ROLE_NAMES))


def _human_geometry_gates(
    geometry: WholeTrackGeometryV15CR9,
    thresholds: RoleThresholdsV15CR9,
) -> Mapping[str, bool]:
    return {
        "human_all_81_frames_visible": geometry.all_81_frames_visible is True,
        "human_area_stability": geometry.area_p95_to_p05_ratio <= thresholds.human_maximum_area_p95_to_p05_ratio,
        "human_temporal_continuity": geometry.median_adjacent_iou >= thresholds.human_minimum_median_adjacent_iou,
        "human_whole_person_area_support": geometry.p10_area_pixels >= thresholds.human_minimum_p10_area_pixels,
        "human_four_connected_component_support": geometry.median_largest_component_fraction >= thresholds.human_minimum_median_largest_component_fraction,
        "human_bbox_fill_support": geometry.median_bbox_fill_fraction >= thresholds.human_minimum_median_bbox_fill_fraction,
        "human_whole_person_extent": geometry.p10_bbox_diagonal_frame_fraction >= thresholds.human_minimum_p10_bbox_diagonal_frame_fraction,
    }


def _phase_region_scores(
    affinity: R6AffinityInputV15CR9,
    tracks: ProposalTrackInputV15CR9,
    thresholds: RoleThresholdsV15CR9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, Any]]:
    real_z, real_scale = r8_core.spatial_median_mad(affinity.real)
    shuffled_z, shuffled_scale = r8_core.spatial_median_mad(affinity.shuffled)
    null_z, null_scale = r8_core.spatial_median_mad(affinity.null_bank)
    proposal_count = len(tracks.proposal_ids)
    real_scores = np.full((len(ROLE_NAMES), proposal_count, PHASE_COUNT), np.nan, dtype=np.float32)
    shuffled_scores = np.full_like(real_scores, np.nan)
    null_scores = np.full(
        (len(ROLE_NAMES), proposal_count, NULL_COUNT, PHASE_COUNT),
        np.nan,
        dtype=np.float32,
    )
    for proposal in range(proposal_count):
        for phase in range(PHASE_COUNT):
            coverage = tracks.phase_coverage[proposal, phase]
            for role in range(len(ROLE_NAMES)):
                block_real = [
                    r8_core.region_mean(real_z[block, role, phase], coverage, thresholds.minimum_phase_coverage_mass)
                    for block in range(len(BLOCK_INDICES))
                ]
                block_shuffled = [
                    r8_core.region_mean(shuffled_z[block, role, phase], coverage, thresholds.minimum_phase_coverage_mass)
                    for block in range(len(BLOCK_INDICES))
                ]
                if bool(np.isfinite(block_real).all()):
                    real_scores[role, proposal, phase] = np.float32(np.median(block_real))
                if bool(np.isfinite(block_shuffled).all()):
                    shuffled_scores[role, proposal, phase] = np.float32(np.median(block_shuffled))
            for role in range(len(ROLE_NAMES)):
                for null_index in range(NULL_COUNT):
                    values = [
                        r8_core.region_mean(
                            null_z[block, role, null_index, phase],
                            coverage,
                            thresholds.minimum_phase_coverage_mass,
                        )
                        for block in range(len(BLOCK_INDICES))
                    ]
                    if bool(np.isfinite(values).all()):
                        null_scores[role, proposal, null_index, phase] = np.float32(
                            np.median(values)
                        )
    return real_scores, shuffled_scores, null_scores, {
        "real_zero_scale_maps": int(np.sum(real_scale <= np.float32(1.0e-12))),
        "shuffled_zero_scale_maps": int(np.sum(shuffled_scale <= np.float32(1.0e-12))),
        "null_zero_scale_maps": int(np.sum(null_scale <= np.float32(1.0e-12))),
        "aggregation": "proposal_coverage_weighted_region_mean_then_equal_block_median",
        "pointwise_mask_from_affinity_created": False,
    }


def tube_pair_metrics(left: np.ndarray, right: np.ndarray) -> Mapping[str, Any]:
    if left.shape != right.shape or left.ndim != 3:
        raise SourceRoleAuthorityV15CR9Error("tube geometry differs")
    ious: list[float] = []
    small_containments: list[float] = []
    left_containments: list[float] = []
    right_containments: list[float] = []
    overlap_phases = 0
    for phase in range(left.shape[0]):
        a = np.asarray(left[phase], dtype=np.float64)
        b = np.asarray(right[phase], dtype=np.float64)
        intersection = float(np.minimum(a, b).sum())
        union = float(np.maximum(a, b).sum())
        a_mass = float(a.sum())
        b_mass = float(b.sum())
        if intersection > 0.0:
            overlap_phases += 1
        if union > 0.0 and a_mass > 0.0 and b_mass > 0.0:
            ious.append(intersection / union)
            left_containments.append(intersection / a_mass)
            right_containments.append(intersection / b_mass)
            small_containments.append(intersection / min(a_mass, b_mass))
    if not ious:
        ious = [0.0]
        left_containments = [0.0]
        right_containments = [0.0]
        small_containments = [0.0]
    return {
        "median_iou": float(np.median(ious)),
        "p95_iou": float(np.percentile(ious, 95)),
        "median_smaller_containment": float(np.median(small_containments)),
        "p95_smaller_containment": float(np.percentile(small_containments, 95)),
        "median_left_containment": float(np.median(left_containments)),
        "median_right_containment": float(np.median(right_containments)),
        "p95_right_containment": float(np.percentile(right_containments, 95)),
        "overlap_phase_count": overlap_phases,
    }


def _distinct_count(values: np.ndarray, epsilon: float) -> int:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.shape != (NULL_COUNT,) or not bool(np.isfinite(ordered).all()):
        return 0
    count = 1
    last = float(ordered[0])
    for value in ordered[1:]:
        if abs(float(value) - last) > epsilon:
            count += 1
            last = float(value)
    return count


def _candidate_evidence(
    *,
    role_index: int,
    proposal_index: int,
    real_scores: np.ndarray,
    shuffled_scores: np.ndarray,
    global_max_null_phase: np.ndarray,
    global_max_null_track: np.ndarray,
    tracks: ProposalTrackInputV15CR9,
    affinity: R6AffinityInputV15CR9,
    thresholds: RoleThresholdsV15CR9,
) -> dict[str, Any]:
    role = ROLE_NAMES[role_index]
    real = real_scores[role_index, proposal_index].astype(np.float64)
    shuffled = shuffled_scores[role_index, proposal_index].astype(np.float64)
    finite = np.isfinite(real) & np.isfinite(shuffled) & np.isfinite(global_max_null_phase).all(axis=0)
    track_real = float(np.median(real[finite])) if bool(finite.any()) else float("nan")
    track_shuffled = float(np.median(shuffled[finite])) if bool(finite.any()) else float("nan")
    raw_p = r8_core.empirical_upper_p(track_real, global_max_null_track)
    global_percentile = r8_core.midrank_percentile(track_real, global_max_null_track)
    global_quantile = (
        float(np.quantile(global_max_null_track, thresholds.global_max_t_required_percentile))
        if bool(np.isfinite(global_max_null_track).all()) else float("nan")
    )
    vessel_fwer_p = min(1.0, len(VESSEL_ROLE_NAMES) * raw_p)
    vessel_quantile = (
        float(np.quantile(global_max_null_track, thresholds.vessel_required_percentile))
        if bool(np.isfinite(global_max_null_track).all()) else float("nan")
    )
    phase_percentiles = np.asarray([
        r8_core.midrank_percentile(float(real[phase]), global_max_null_phase[:, phase]) if finite[phase] else float("nan")
        for phase in range(PHASE_COUNT)
    ])
    consistent = [
        bool(finite[phase] and phase_percentiles[phase] >= thresholds.phase_global_max_null_percentile and real[phase] > shuffled[phase])
        for phase in range(PHASE_COUNT)
    ]
    real_over_shuffle = int(np.sum(finite & (real > shuffled)))
    global_gate = bool(
        affinity.null_index_alignment_verified
        and affinity.four_role_joint_null_available
        and math.isfinite(raw_p)
        and raw_p <= thresholds.familywise_alpha
        and math.isfinite(global_percentile)
        and global_percentile >= thresholds.global_max_t_required_percentile
        and math.isfinite(global_quantile)
        and track_real > global_quantile
    )
    vessel_extra_gate = bool(
        role not in VESSEL_ROLE_NAMES
        or (
            math.isfinite(vessel_fwer_p)
            and vessel_fwer_p <= thresholds.familywise_alpha
            and math.isfinite(vessel_quantile)
            and track_real > vessel_quantile
        )
    )
    human_gates = (
        _human_geometry_gates(tracks.geometry[proposal_index], thresholds)
        if role == "human_agent" else {}
    )
    gates: dict[str, bool] = {
        "generic_source_track_geometry": tracks.track_gate_pass[proposal_index] is True,
        "all_21_phase_regions_present": int(np.sum(finite)) == PHASE_COUNT,
        "global_null_index_alignment": affinity.null_index_alignment_verified is True,
        "four_role_joint_null_axis_available": (
            affinity.four_role_joint_null_available is True
        ),
        "global_max_t_null_non_degenerate": (
            _distinct_count(global_max_null_track, thresholds.null_track_score_epsilon) >= thresholds.minimum_distinct_null_track_scores
            and float(np.std(global_max_null_track)) > thresholds.null_track_score_epsilon
        ),
        "global_four_role_max_t_fwer": global_gate,
        "vessel_three_role_bonferroni_extra_gate": vessel_extra_gate,
        "track_above_token_permutation": bool(
            math.isfinite(track_real) and math.isfinite(track_shuffled)
            and track_real > track_shuffled
            and real_over_shuffle >= thresholds.minimum_real_over_permutation_phases
        ),
        "temporal_consistency": bool(
            sum(consistent) >= thresholds.minimum_consistent_phases
            and r8_core.longest_true_run(consistent) >= thresholds.minimum_longest_consistent_run
        ),
        **human_gates,
        "no_same_role_duplicate_or_nesting_candidate": True,
    }
    return {
        "role": role,
        "proposal_id": tracks.proposal_ids[proposal_index],
        "track_real": track_real,
        "track_shuffled": track_shuffled,
        "global_max_t_required_quantile": global_quantile,
        "global_max_t_percentile": global_percentile,
        "global_max_t_empirical_upper_p": raw_p,
        "vessel_three_role_bonferroni_fwer_upper_p": vessel_fwer_p if role in VESSEL_ROLE_NAMES else None,
        "consistent_phase_count": int(sum(consistent)),
        "longest_consistent_run": int(r8_core.longest_true_run(consistent)),
        "real_over_permutation_phase_count": real_over_shuffle,
        "same_role_duplicate_or_nesting_neighbors": [],
        "gates": gates,
        "eligible_before_same_role_family_and_proposal_competition": all(gates.values()),
        "eligible_before_proposal_competition": all(gates.values()),
        "evidence_margin": track_real - global_quantile if math.isfinite(track_real) and math.isfinite(global_quantile) else float("nan"),
    }


def _choose_without_forcing(
    role_index: int,
    evidence: Sequence[Mapping[str, Any]],
    proposal_ids: Sequence[str],
    real_scores: np.ndarray,
    thresholds: RoleThresholdsV15CR9,
) -> tuple[int | None, Mapping[str, Any]]:
    eligible = [index for index, row in enumerate(evidence) if row["eligible_before_proposal_competition"] is True]
    if not eligible:
        return None, {"status": "unassigned_no_eligible_proposal", "eligible_proposal_ids": []}
    ordered = sorted(eligible, key=lambda index: (-float(evidence[index]["evidence_margin"]), proposal_ids[index]))
    winner = ordered[0]
    if len(ordered) == 1:
        return winner, {"status": "unique_eligible_proposal", "eligible_proposal_ids": [proposal_ids[winner]]}
    runner = ordered[1]
    if not float(evidence[winner]["evidence_margin"]) > float(evidence[runner]["evidence_margin"]):
        return None, {
            "status": "unassigned_non_unique_top_evidence_margin",
            "eligible_proposal_ids": [proposal_ids[index] for index in ordered],
            "winner_id_if_forced": proposal_ids[winner],
            "runner_up_id": proposal_ids[runner],
        }
    dominance: dict[str, int] = {}
    for competitor in ordered[1:]:
        left = real_scores[role_index, winner]
        right = real_scores[role_index, competitor]
        finite = np.isfinite(left) & np.isfinite(right)
        dominance[proposal_ids[competitor]] = int(np.sum(finite & (left > right)))
    if any(value < thresholds.minimum_proposal_dominance_phases for value in dominance.values()):
        return None, {
            "status": "unassigned_winner_failed_all_eligible_temporal_dominance",
            "eligible_proposal_ids": [proposal_ids[index] for index in ordered],
            "winner_id_if_forced": proposal_ids[winner],
            "runner_up_id": proposal_ids[runner],
            "dominance_phase_count_by_competitor_id": dominance,
        }
    return winner, {
        "status": "unique_top_winner_dominated_every_eligible_proposal",
        "eligible_proposal_ids": [proposal_ids[index] for index in ordered],
        "winner_id": proposal_ids[winner],
        "runner_up_id": proposal_ids[runner],
        "dominance_phase_count_by_competitor_id": dominance,
    }


def _run_source_four_role_statistic_mechanical_unit_v15c_r9(
    *,
    tracks: ProposalTrackInputV15CR9,
    affinity: R6AffinityInputV15CR9,
    thresholds: RoleThresholdsV15CR9 = RoleThresholdsV15CR9(),
) -> Mapping[str, Any]:
    real_scores, shuffled_scores, null_scores, standardization = _phase_region_scores(affinity, tracks, thresholds)
    valid = [
        index
        for index, gate in enumerate(tracks.track_gate_pass)
        if gate is True and bool(np.isfinite(null_scores[:, index]).all())
    ]
    global_fwer_certified = bool(
        valid
        and affinity.null_index_alignment_verified
        and affinity.four_role_joint_null_available
    )
    if global_fwer_certified:
        # A valid replicate j must contain four separately observed null role
        # statistics.  Repeating one common r6 null map over the role axis is
        # explicitly insufficient and is never used for certification.
        role_proposal_phase = null_scores[:, valid].astype(np.float64)
        global_max_null_phase = np.max(role_proposal_phase, axis=(0, 1))
        role_proposal_track = np.median(role_proposal_phase, axis=3)
        global_max_null_track = np.max(role_proposal_track, axis=(0, 1))
    else:
        global_max_null_phase = np.full((NULL_COUNT, PHASE_COUNT), np.nan)
        global_max_null_track = np.full((NULL_COUNT,), np.nan)

    evidence: dict[str, list[dict[str, Any]]] = {}
    competition: dict[str, Mapping[str, Any]] = {}
    choices: dict[str, int | None] = {}
    same_role_families: dict[str, list[Mapping[str, Any]]] = {}
    for role_index, role in enumerate(ROLE_NAMES):
        rows = [
            _candidate_evidence(
                role_index=role_index,
                proposal_index=proposal_index,
                real_scores=real_scores,
                shuffled_scores=shuffled_scores,
                global_max_null_phase=global_max_null_phase,
                global_max_null_track=global_max_null_track,
                tracks=tracks,
                affinity=affinity,
                thresholds=thresholds,
            )
            for proposal_index in range(len(tracks.proposal_ids))
        ]
        prelim = [index for index, row in enumerate(rows) if row["eligible_before_same_role_family_and_proposal_competition"] is True]
        family_pairs = []
        duplicate_indices: set[int] = set()
        for position, left in enumerate(prelim):
            for right in prelim[position + 1 :]:
                metrics = tube_pair_metrics(tracks.phase_coverage[left], tracks.phase_coverage[right])
                if metrics["median_iou"] >= thresholds.duplicate_median_iou or metrics["median_smaller_containment"] >= thresholds.duplicate_median_containment:
                    duplicate_indices.update((left, right))
                    family_pairs.append({"left": tracks.proposal_ids[left], "right": tracks.proposal_ids[right], "metrics": metrics})
        for index in duplicate_indices:
            neighbors = sorted({
                pair["right"] if pair["left"] == tracks.proposal_ids[index] else pair["left"]
                for pair in family_pairs
                if tracks.proposal_ids[index] in (pair["left"], pair["right"])
            })
            rows[index]["same_role_duplicate_or_nesting_neighbors"] = neighbors
            rows[index]["gates"]["no_same_role_duplicate_or_nesting_candidate"] = False
            rows[index]["eligible_before_proposal_competition"] = False
        same_role_families[role] = family_pairs
        evidence[role] = rows
        choice, detail = _choose_without_forcing(role_index, rows, tracks.proposal_ids, real_scores, thresholds)
        choices[role] = choice
        competition[role] = detail

    cross_role_conflicts: list[Mapping[str, Any]] = []
    contact_relations: list[Mapping[str, Any]] = []
    # Vessels remain a strict mutually-exclusive role family at the tube level.
    for position, left_role in enumerate(VESSEL_ROLE_NAMES):
        left = choices[left_role]
        if left is None:
            continue
        for right_role in VESSEL_ROLE_NAMES[position + 1 :]:
            right = choices[right_role]
            if right is None:
                continue
            metrics = tube_pair_metrics(tracks.phase_coverage[left], tracks.phase_coverage[right])
            if left == right or metrics["median_iou"] >= thresholds.vessel_conflict_median_iou or metrics["median_smaller_containment"] >= thresholds.vessel_conflict_median_containment:
                cross_role_conflicts.append({"roles": [left_role, right_role], "kind": "vessel_role_tube_conflict", "metrics": metrics})

    human = choices["human_agent"]
    if human is not None:
        for vessel_role in VESSEL_ROLE_NAMES:
            vessel = choices[vessel_role]
            if vessel is None:
                continue
            metrics = tube_pair_metrics(tracks.phase_coverage[human], tracks.phase_coverage[vessel])
            limited = bool(
                human != vessel
                and metrics["median_iou"] <= thresholds.human_vessel_max_median_iou
                and metrics["median_left_containment"] <= thresholds.human_vessel_max_median_human_containment
                and metrics["median_right_containment"] <= thresholds.human_vessel_max_median_vessel_containment
                and metrics["p95_right_containment"] <= thresholds.human_vessel_max_p95_vessel_containment
            )
            relation = {"roles": ["human_agent", vessel_role], "kind": "source_contact_or_occlusion_evidence", "limited_overlap_gate": limited, "metrics": metrics}
            contact_relations.append(relation)
            if not limited:
                cross_role_conflicts.append({**relation, "kind": "human_vessel_overlap_not_safely_limited"})

    conflicted_roles = {role for row in cross_role_conflicts for role in row["roles"]}
    for role in conflicted_roles:
        choices[role] = None
        competition[role] = {**competition[role], "status": "unassigned_cross_role_overlap_or_identity_conflict"}
    assignments = {role: (tracks.proposal_ids[index] if index is not None else None) for role, index in choices.items()}
    complete = all(value is not None for value in assignments.values()) and len(set(assignments.values())) == len(ROLE_NAMES) and not cross_role_conflicts
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "LOCAL_FOUR_ROLE_CANDIDATE_REQUIRES_OWNERSHIP_AND_REJECT_ONLY_AUDIT" if complete else "NO_GO_FAIL_CLOSED_FOUR_ROLE_ASSIGNMENT",
        "role_names": list(ROLE_NAMES),
        "r6_role_channel_index": dict(R6_ROLE_INDEX),
        "proposal_ids": list(tracks.proposal_ids),
        "assignments": assignments,
        "evidence": evidence,
        "competition": competition,
        "same_role_duplicate_nesting_families": same_role_families,
        "cross_role_conflicts": cross_role_conflicts,
        "human_vessel_contact_or_occlusion_evidence": contact_relations,
        "multiple_comparison_control": {
            "method": (
                "aligned_64_null_global_max_T_over_four_roles_and_all_geometry_valid_proposals"
                if global_fwer_certified
                else "NO_GO_existing_r6_common_null_lacks_four_role_joint_axis"
            ),
            "null_registry_sha256": affinity.null_registry_sha256,
            "null_index_alignment_verified": affinity.null_index_alignment_verified,
            "four_role_joint_null_available": (
                affinity.four_role_joint_null_available
            ),
            "global_four_role_fwer_certified": global_fwer_certified,
            "common_null_broadcast_used_for_certification": False,
            "global_null_search_proposal_ids": [tracks.proposal_ids[index] for index in valid],
            "global_null_role_slots": list(ROLE_NAMES),
            "finite_null_method": "plus_one_empirical_upper_tail",
            "global_familywise_alpha": thresholds.familywise_alpha,
            "minimum_attainable_global_fwer_p": 1.0 / float(NULL_COUNT + 1),
            "vessel_three_role_bonferroni_is_additional_gate": True,
        },
        "standardization": standardization,
        "thresholds": {name: getattr(thresholds, name) for name in thresholds.__dataclass_fields__},
        "role_assignment_mechanical_candidate_qualified": complete,
        "ownership_partition_required": True,
        "ownership_partition_mechanical_candidate_qualified": False,
        "mechanical_candidate_qualified": False,
        "manual_full_track_overlay_audit_required": True,
        "local_schema_replay_only": True,
        "remote_worker_execution_verified": False,
        "observer_execution_authorized": False,
        "localization_semantically_certified": False,
        "scientific_claim_authorized": False,
        "action_success_certified": False,
        "route_authorized": False,
        "decode_authorized": False,
        "training_authorized": False,
        "optimizer_updates": 0,
        "renderer_forward_calls": 0,
        "anchor_consumed": False,
        "target_instruction_consumed": False,
        "material_or_transparency_classification_consumed": False,
        "affinity_used_pointwise_as_mask": False,
        "forced_assignment": False,
        "roi_or_manual_box_consumed": False,
    }
    payload = r8_core._json_safe(payload)
    payload["receipt_sha256"] = object_sha256(payload)
    return payload


def run_source_four_role_authority_v15c_r9(
    *,
    tracks: ProposalTrackInputV15CR9,
    affinity: R6AffinityInputV15CR9,
    thresholds: RoleThresholdsV15CR9 = RoleThresholdsV15CR9(),
) -> Mapping[str, Any]:
    """Official r9 entry point; r10 future availability is deliberately shut.

    The checked-in r9 runner only owns the historical common-null diagnostic.
    A future true joint-null tensor must be consumed by a fresh r10 runner and
    postflight that also bind actual SP4 rank shards and a proposal registry
    fixed before affinity inspection.  Until those exist, available=True must
    not escape through this legacy public seam.
    """

    if not isinstance(affinity, R6AffinityInputV15CR9):
        raise SourceRoleAuthorityV15CR9Error("r9 affinity type differs")
    if affinity.four_role_joint_null_available is True:
        raise SourceRoleAuthorityV15CR9Error(
            "r9 future joint-null path is NO-GO; fresh r10 runner/postflight absent"
        )
    return _run_source_four_role_statistic_mechanical_unit_v15c_r9(
        tracks=tracks,
        affinity=affinity,
        thresholds=thresholds,
    )


def _components_and_holes(mask: np.ndarray) -> tuple[int, int]:
    import cv2

    binary = np.ascontiguousarray(mask, dtype=np.uint8)
    if binary.ndim != 2 or not bool(np.isin(binary, [0, 1]).all()):
        raise SourceRoleAuthorityV15CR9Error("binary topology mask differs")
    count, _labels = cv2.connectedComponents(binary, connectivity=4)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None:
        holes = sum(1 for row in hierarchy[0] if int(row[3]) >= 0)
    return max(0, int(count) - 1), int(holes)


def partition_source_role_ownership_v15c_r9(
    *,
    proposal_masks: Mapping[str, np.ndarray],
    replayed_raw_signed_valued_logits: Mapping[str, np.ndarray],
    thresholds: RoleThresholdsV15CR9 = RoleThresholdsV15CR9(),
) -> Mapping[str, Any]:
    """Partition overlapping source proposals into v15b-safe ownership masks.

    ``replayed_raw_signed_valued_logits`` are ordinary floating-point SAM2
    outputs recovered from the r8 safetensor evidence.  The word "replayed"
    is provenance, not a cryptographic signature; no external signer or TEE is
    claimed.
    """

    if set(proposal_masks) != set(ROLE_NAMES) or set(replayed_raw_signed_valued_logits) != set(ROLE_NAMES):
        raise SourceRoleAuthorityV15CR9Error("ownership role registry differs")
    masks = []
    logits = []
    shape: tuple[int, ...] | None = None
    for role in ROLE_NAMES:
        mask = proposal_masks[role]
        logit = replayed_raw_signed_valued_logits[role]
        if (
            not isinstance(mask, np.ndarray)
            or mask.dtype != np.bool_
            or mask.ndim != 3
            or not isinstance(logit, np.ndarray)
            or logit.shape != mask.shape
            or logit.dtype.kind not in "fc"
            or not bool(np.isfinite(logit).all())
            or not np.array_equal(mask, logit > 0.0)
            or (shape is not None and mask.shape != shape)
        ):
            raise SourceRoleAuthorityV15CR9Error("mask/logit replay contract differs")
        shape = mask.shape
        masks.append(mask)
        logits.append(np.asarray(logit, dtype=np.float32))
    if shape is None or shape[0] != 81:
        raise SourceRoleAuthorityV15CR9Error("ownership frame count differs")
    raw = np.stack(masks, axis=0)
    score = np.stack(logits, axis=0)
    active_count = np.sum(raw, axis=0)
    masked_score = np.where(raw, score, -np.inf)
    order = np.argsort(masked_score, axis=0, kind="stable")
    winner = order[-1]
    top = np.take_along_axis(masked_score, winner[None], axis=0)[0]
    runner = np.take_along_axis(masked_score, order[-2][None], axis=0)[0]
    margin = np.full(top.shape, -np.inf, dtype=np.float32)
    np.subtract(
        top,
        runner,
        out=margin,
        where=np.isfinite(top) & np.isfinite(runner),
    )
    strict = (active_count == 1) | (
        (active_count > 1)
        & np.isfinite(runner)
        & (margin > thresholds.ownership_logit_margin)
    )
    tentative = np.stack([(active_count > 0) & strict & (winner == index) for index in range(len(ROLE_NAMES))], axis=0)
    ambiguous = (active_count > 1) & ~strict

    contact_masks = {
        role: np.ascontiguousarray(raw[0] & raw[ROLE_NAMES.index(role)], dtype=np.bool_)
        for role in VESSEL_ROLE_NAMES
    }
    role_gates: dict[str, Mapping[str, bool]] = {}
    role_metrics: dict[str, Mapping[str, Any]] = {}
    failed_roles: list[str] = []
    for index, role in enumerate(ROLE_NAMES):
        raw_area = raw[index].reshape(81, -1).sum(axis=1).astype(np.float64)
        owned_area = tentative[index].reshape(81, -1).sum(axis=1).astype(np.float64)
        retention = np.divide(owned_area, raw_area, out=np.zeros_like(owned_area), where=raw_area > 0.0)
        adjacency = []
        topology_equal = []
        owned_connected = []
        for frame in range(81):
            raw_components, raw_holes = _components_and_holes(raw[index, frame])
            owned_components, owned_holes = _components_and_holes(tentative[index, frame])
            topology_equal.append(raw_components == owned_components and raw_holes == owned_holes)
            owned_connected.append(owned_components == 1)
            if frame:
                intersection = float(np.logical_and(tentative[index, frame - 1], tentative[index, frame]).sum())
                union = float(np.logical_or(tentative[index, frame - 1], tentative[index, frame]).sum())
                adjacency.append(intersection / union if union > 0.0 else 0.0)
        median_adjacent = float(np.median(adjacency))
        gates = {
            "all_81_ownership_frames_visible": bool((owned_area > 0.0).all()),
            "ownership_is_single_4_connected_component_every_frame": all(owned_connected),
            "ownership_hole_and_component_topology_matches_proposal_every_frame": all(topology_equal),
            "ownership_minimum_area_retention": bool(float(np.min(retention)) >= thresholds.ownership_minimum_area_retention),
            "ownership_median_area_retention": bool(float(np.median(retention)) >= thresholds.ownership_minimum_median_area_retention),
            "ownership_temporal_continuity": median_adjacent >= thresholds.ownership_minimum_median_adjacent_iou,
        }
        role_gates[role] = gates
        role_metrics[role] = {
            "minimum_area_retention": float(np.min(retention)),
            "median_area_retention": float(np.median(retention)),
            "median_adjacent_iou": median_adjacent,
            "tentative_ownership_pixel_count": int(tentative[index].sum()),
            "raw_proposal_pixel_count": int(raw[index].sum()),
        }
        if not all(gates.values()):
            failed_roles.append(role)

    final = tentative.copy()
    removed = np.zeros(shape, dtype=np.bool_)
    for role in failed_roles:
        index = ROLE_NAMES.index(role)
        removed |= final[index]
        final[index] = False
    unassigned = np.ascontiguousarray(ambiguous | removed, dtype=np.bool_)
    if bool((np.sum(final, axis=0) > 1).any()):
        raise SourceRoleAuthorityV15CR9Error("ownership partition is not exclusive")
    complete = len(failed_roles) == 0
    receipt: dict[str, Any] = {
        "schema_version": OWNERSHIP_SCHEMA_VERSION,
        "status": "LOCAL_OWNERSHIP_PARTITION_REQUIRES_REJECT_ONLY_AUDIT" if complete else "NO_GO_OWNERSHIP_ROLE_UNASSIGNED",
        "role_names": list(ROLE_NAMES),
        "arbitration": "strict_argmax_of_replayed_raw_signed_valued_sam2_logits_else_unassigned",
        "no_external_signature_or_tee_claimed": True,
        "raw_proposal_overlap_preserved_as_evidence": True,
        "contact_relation_is_not_ownership": True,
        "morphological_repair_applied": False,
        "failed_roles": failed_roles,
        "role_gates": role_gates,
        "role_metrics": role_metrics,
        "tensor_sha256": {
            "raw_proposal_masks": array_sha256(raw),
            "tentative_ownership_masks": array_sha256(tentative),
            "final_ownership_masks": array_sha256(final),
            "unassigned_occlusion_mask": array_sha256(unassigned),
            "human_vessel_contact_masks": {role: array_sha256(contact_masks[role]) for role in VESSEL_ROLE_NAMES},
        },
        "pairwise_exclusive_final_ownership": True,
        "all_four_roles_ownership_qualified": complete,
        "mechanical_candidate_qualified": complete,
        "local_schema_replay_only": True,
        "remote_worker_execution_verified": False,
        "observer_execution_authorized": False,
        "localization_semantically_certified": False,
        "scientific_claim_authorized": False,
        "route_authorized": False,
        "decode_authorized": False,
        "training_authorized": False,
    }
    receipt["receipt_sha256"] = object_sha256(receipt)
    return {
        "raw_proposal_masks": raw,
        "tentative_ownership_masks": tentative,
        "final_ownership_masks": final,
        "unassigned_occlusion_mask": unassigned,
        "human_vessel_contact_masks": contact_masks,
        "receipt": receipt,
    }


def adapt_qualified_ownership_to_v15b_v15c_r9(
    *,
    final_ownership_masks: np.ndarray,
    human_vessel_contact_masks: Mapping[str, np.ndarray],
    target_height: int = GRID_HEIGHT,
    target_width: int = GRID_WIDTH,
    strict_coverage_margin: float = 1.0e-3,
) -> Mapping[str, Any]:
    """Create a mutually-exclusive 21-phase mask candidate for v15b.

    This is an ABI adapter only.  It does not instantiate
    ``SourceRoleMaskSetV15B`` and does not authorize routing.  A spatial cell
    with multiple role coverage is owned only by a strict coverage winner;
    otherwise it stays unassigned.  Any v15b area/connectivity/centroid gate
    failure clears the whole role and keeps the adapter NO-GO.
    """

    import cv2

    value = final_ownership_masks
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.bool_
        or value.ndim != 4
        or value.shape[0] != len(ROLE_NAMES)
        or value.shape[1] != 81
        or bool((value.sum(axis=0) > 1).any())
        or type(target_height) is not int
        or type(target_width) is not int
        or target_height <= 0
        or target_width <= 0
        or type(strict_coverage_margin) not in (int, float)
        or not math.isfinite(float(strict_coverage_margin))
        or not 0.0 < float(strict_coverage_margin) < 1.0
        or set(human_vessel_contact_masks) != set(VESSEL_ROLE_NAMES)
    ):
        raise SourceRoleAuthorityV15CR9Error("v15b adapter input differs")
    for role, mask in human_vessel_contact_masks.items():
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != tuple(value.shape[1:]):
            raise SourceRoleAuthorityV15CR9Error(f"{role} contact mask differs")

    coverage = np.zeros(
        (len(ROLE_NAMES), PHASE_COUNT, target_height, target_width),
        dtype=np.float32,
    )
    for role_index in range(len(ROLE_NAMES)):
        for phase, frame in enumerate(PHASE_FRAMES):
            coverage[role_index, phase] = cv2.resize(
                value[role_index, frame].astype(np.float32),
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
    order = np.argsort(coverage, axis=0, kind="stable")
    winner = order[-1]
    top = np.take_along_axis(coverage, winner[None], axis=0)[0]
    runner = np.take_along_axis(coverage, order[-2][None], axis=0)[0]
    decisive = (top > 0.0) & ((top - runner) > float(strict_coverage_margin))
    tentative = np.stack(
        [decisive & (winner == role_index) for role_index in range(len(ROLE_NAMES))],
        axis=0,
    )
    ambiguous = (top > 0.0) & ~decisive
    role_gates: dict[str, Mapping[str, bool]] = {}
    failed: list[str] = []
    spatial_area = target_height * target_width
    for role_index, role in enumerate(ROLE_NAMES):
        masks = tentative[role_index]
        areas = masks.reshape(PHASE_COUNT, -1).sum(axis=1).astype(np.float64)
        centroids = []
        connected = []
        hole_topology_equal = []
        for phase in range(PHASE_COUNT):
            components, holes = _components_and_holes(masks[phase])
            full_components, full_holes = _components_and_holes(
                value[role_index, PHASE_FRAMES[phase]]
            )
            connected.append(components == 1)
            hole_topology_equal.append(
                components == full_components and holes == full_holes
            )
            ys, xs = np.nonzero(masks[phase])
            centroids.append(
                (float(np.mean(ys)), float(np.mean(xs)))
                if len(xs) else (float("nan"), float("nan"))
            )
        jumps = [
            math.hypot(right[0] - left[0], right[1] - left[1])
            if all(math.isfinite(item) for item in (*left, *right)) else float("inf")
            for left, right in zip(centroids, centroids[1:])
        ]
        gates = {
            "all_21_phases_visible": bool((areas > 0).all()),
            "single_4_connected_component_every_phase": all(connected),
            "hole_and_component_topology_matches_full_resolution_every_phase": all(
                hole_topology_equal
            ),
            "minimum_area_two_cells": bool(float(np.min(areas)) >= 2.0),
            "maximum_area_fraction_0p20": bool(float(np.max(areas)) <= max(2, int(spatial_area * 0.20))),
            "maximum_area_ratio_4": bool(float(np.min(areas)) > 0.0 and float(np.max(areas) / np.min(areas)) <= 4.0),
            "maximum_centroid_jump_3p5": bool(jumps and max(jumps) <= 3.5),
        }
        role_gates[role] = gates
        if not all(gates.values()):
            failed.append(role)
    final = tentative.copy()
    removed = np.zeros((PHASE_COUNT, target_height, target_width), dtype=np.bool_)
    for role in failed:
        index = ROLE_NAMES.index(role)
        removed |= final[index]
        final[index] = False
    unassigned = np.ascontiguousarray(ambiguous | removed, dtype=np.bool_)
    if bool((final.sum(axis=0) > 1).any()):
        raise SourceRoleAuthorityV15CR9Error("v15b adapter emitted overlap")

    contact_coverage = np.zeros(
        (len(VESSEL_ROLE_NAMES), PHASE_COUNT, target_height, target_width),
        dtype=np.float32,
    )
    for vessel_index, vessel_role in enumerate(VESSEL_ROLE_NAMES):
        for phase, frame in enumerate(PHASE_FRAMES):
            contact_coverage[vessel_index, phase] = cv2.resize(
                human_vessel_contact_masks[vessel_role][frame].astype(np.float32),
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
    contact_relation = np.any(contact_coverage > 0.0, axis=0)
    contact_relation[0] = False
    complete = not failed
    receipt: dict[str, Any] = {
        "schema_version": V15B_ADAPTER_SCHEMA_VERSION,
        "status": "LOCAL_V15B_MASK_ABI_CANDIDATE_UNAUTHORIZED" if complete else "NO_GO_V15B_MASK_ABI_ROLE_UNASSIGNED",
        "r9_to_v15b_role_semantics": {
            "human_agent": "human_agent",
            "old_actor": "old_actor",
            "new_actor": "moving_object",
            "recipient": "recipient",
        },
        "phase_frames": list(PHASE_FRAMES),
        "height": target_height,
        "width": target_width,
        "strict_coverage_margin": float(strict_coverage_margin),
        "failed_roles": failed,
        "role_gates": role_gates,
        "pairwise_exclusive_role_masks": True,
        "contact_relation_mask_is_independent": True,
        "raw_overlapping_proposals_passed_to_v15b": False,
        "morphological_repair_applied": False,
        "tensor_sha256": {
            "role_masks": array_sha256(final),
            "contact_relation_mask": array_sha256(contact_relation),
            "unassigned_mask": array_sha256(unassigned),
        },
        "v15b_source_role_mask_set_creation_authorized": False,
        "mechanical_candidate_qualified": complete,
        "observer_execution_authorized": False,
        "localization_semantically_certified": False,
        "scientific_claim_authorized": False,
        "route_authorized": False,
        "decode_authorized": False,
        "training_authorized": False,
    }
    receipt["receipt_sha256"] = object_sha256(receipt)
    return {
        "role_masks": final,
        "contact_relation_mask": np.ascontiguousarray(contact_relation, dtype=np.bool_),
        "unassigned_mask": unassigned,
        "receipt": receipt,
    }


def load_tracks_for_v15c_r9(metadata_path: Path, tensor_path: Path) -> tuple[ProposalTrackInputV15CR9, Mapping[str, Any]]:
    base, metadata = r8_core.load_tracks_for_v15c(metadata_path, tensor_path)
    rows = metadata.get("proposals")
    if type(rows) is not list or len(rows) != len(base.proposal_ids):
        raise SourceRoleAuthorityV15CR9Error("track metadata registry differs")
    geometry = []
    for row in rows:
        if type(row) is not dict:
            raise SourceRoleAuthorityV15CR9Error("track geometry row differs")
        gates = row.get("automatic_track_geometry_gates")
        geometry.append(WholeTrackGeometryV15CR9(
            all_81_frames_visible=(type(gates) is dict and gates.get("all_81_frames_visible") is True),
            area_p95_to_p05_ratio=_finite_number(row.get("area_p95_to_p05_ratio"), "area ratio"),
            median_adjacent_iou=_finite_number(row.get("median_adjacent_iou"), "adjacent IoU"),
            p10_area_pixels=_finite_number(row.get("p10_area_pixels"), "p10 area"),
            median_largest_component_fraction=_finite_number(row.get("median_largest_component_fraction"), "component fraction"),
            median_bbox_fill_fraction=_finite_number(row.get("median_bbox_fill_fraction"), "bbox fill"),
            p10_bbox_diagonal_frame_fraction=_finite_number(row.get("p10_bbox_diagonal_frame_fraction"), "bbox diagonal"),
        ))
    return ProposalTrackInputV15CR9(
        proposal_ids=base.proposal_ids,
        phase_coverage=base.phase_coverage,
        track_gate_pass=base.track_gate_pass,
        geometry=tuple(geometry),
    ), metadata


def load_r6_affinity_for_v15c_r9(path: Path, *, null_registry_sha256: str, null_index_alignment_verified: bool) -> R6AffinityInputV15CR9:
    try:
        from safetensors.numpy import load_file
    except ImportError as error:  # pragma: no cover
        raise SourceRoleAuthorityV15CR9Error("safetensors is unavailable") from error
    tensors = load_file(str(path))
    real = np.stack([tensors[f"block_{block:02d}_affinity"] for block in BLOCK_INDICES], axis=0)[:, :4]
    shuffled = np.stack([tensors[f"block_{block:02d}_shuffled_affinity"] for block in BLOCK_INDICES], axis=0)[:, :4]
    common_null = np.stack(
        [tensors[f"block_{block:02d}_null_span_affinity"] for block in BLOCK_INDICES],
        axis=0,
    )
    # The sealed r6 artifact has one common 64-null axis, not four observed
    # role-indexed null axes.  Replication below is diagnostic storage only;
    # ``four_role_joint_null_available=False`` prevents any FWER claim.
    null = np.broadcast_to(
        common_null[:, None],
        (len(BLOCK_INDICES), len(ROLE_NAMES)) + tuple(common_null.shape[1:]),
    )
    return R6AffinityInputV15CR9(
        real=np.ascontiguousarray(real, dtype=np.float32),
        shuffled=np.ascontiguousarray(shuffled, dtype=np.float32),
        null_bank=np.ascontiguousarray(null, dtype=np.float32),
        null_registry_sha256=null_registry_sha256,
        null_index_alignment_verified=null_index_alignment_verified,
        four_role_joint_null_available=False,
    )


__all__ = [
    "BLOCK_INDICES", "FULL_R6_ROLE_NAMES", "GRID_HEIGHT", "GRID_WIDTH",
    "NULL_COUNT", "OWNERSHIP_SCHEMA_VERSION", "PHASE_COUNT", "PHASE_FRAMES",
    "ProposalTrackInputV15CR9", "R6AffinityInputV15CR9", "R6_ROLE_INDEX",
    "ROLE_NAMES", "RoleThresholdsV15CR9", "SCHEMA_VERSION",
    "SourceRoleAuthorityV15CR9Error", "TRACK_SCHEMA_VERSION",
    "V15B_ADAPTER_SCHEMA_VERSION", "VESSEL_ROLE_NAMES",
    "WholeTrackGeometryV15CR9", "adapt_qualified_ownership_to_v15b_v15c_r9",
    "array_sha256",
    "load_r6_affinity_for_v15c_r9", "load_tracks_for_v15c_r9",
    "partition_source_role_ownership_v15c_r9",
    "run_source_four_role_authority_v15c_r9", "tube_pair_metrics",
]
