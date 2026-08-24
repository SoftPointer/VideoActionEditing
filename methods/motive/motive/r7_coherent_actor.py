"""Development-only coherent-motion actor selector for R7-P1.

This module is an independent NumPy prototype.  It consumes tracks that have
*already* been camera compensated and asks a narrower question than the R7-P0
teacher: is there a spatially local group of tracks whose motion is coherent
over time?

The selector is deliberately fail-closed.  Expected data-quality failures are
returned as a :class:`CoherentActorSelection` with ``diagnostic_ready=False``
and empty actor outputs.  Geometry alone cannot identify a semantic actor:
dynamic backgrounds and local parallax can remain indistinguishable from an
action.  This is not a production teacher and its defaults have not been
calibrated on R7-P0.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


R7_COHERENT_ACTOR_SCHEMA = "motive-r7-p1-coherent-actor-dev-v2"
R7_COHERENT_ACTOR_PHASE_STEPS = 32
R7_COHERENT_ACTOR_SCOPE = (
    "development-only geometric coherent-motion proposal; does not identify "
    "a semantic actor and must not be used as a production teacher"
)


class _SelectionFailure(ValueError):
    """Internal control flow for an expected fail-closed outcome."""

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = str(reason)
        self.detail = str(detail)
        super().__init__(f"{self.reason}: {self.detail}")


def _fail(reason: str, detail: str) -> None:
    raise _SelectionFailure(reason, detail)


@dataclass(frozen=True)
class CoherentActorConfig:
    """Uncalibrated, development-only selector thresholds.

    Coordinates are expected to be normalized by frame width and height, and
    times are in seconds.  When ``frame_size`` is supplied to
    :func:`select_coherent_actor`, coordinates are converted to max-side
    isotropic units before these thresholds are applied.
    """

    phase_steps: int = R7_COHERENT_ACTOR_PHASE_STEPS
    minimum_frames: int = 8
    minimum_tracks: int = 8
    minimum_component_tracks: int = 3
    visibility_threshold: float = 0.5
    minimum_track_visibility: float = 0.60
    minimum_transition_visibility: float = 0.50
    minimum_active_transitions: int = 3
    minimum_pair_active_transitions: int = 2
    minimum_transition_speed: float = 0.012
    minimum_track_path_length: float = 0.025
    minimum_track_excursion: float = 0.010
    minimum_track_rms_speed: float = 0.015
    minimum_component_path_length: float = 0.030
    minimum_component_excursion: float = 0.012
    minimum_component_rms_speed: float = 0.018
    spatial_neighbor_radius: float = 0.22
    articulated_neighbor_radius: float = 0.11
    strong_direction_cosine: float = 0.70
    soft_direction_cosine: float = -0.25
    minimum_amplitude_similarity: float = 0.42
    minimum_soft_amplitude_similarity: float = 0.52
    minimum_activity_iou: float = 0.25
    minimum_soft_activity_iou: float = 0.45
    minimum_soft_speed_profile_cosine: float = 0.72
    minimum_component_edge_density: float = 0.20
    minimum_component_degree: int = 2
    minimum_component_degree_fraction: float = 0.30
    minimum_component_core_order: int = 2
    maximum_component_graph_diameter: int = 3
    minimum_distinct_track_distance: float = 0.004
    minimum_component_unique_fraction: float = 0.75
    maximum_component_fraction: float = 0.50
    maximum_global_moving_fraction: float = 0.65
    minimum_global_comoving_fraction: float = 0.35
    maximum_component_spatial_coverage: float = 0.70
    maximum_component_axis_coverage: float = 0.65
    maximum_component_spatial_occupancy: float = 0.60
    spatial_occupancy_bins: int = 6
    maximum_interpolation_gap_frames: int = 0
    minimum_phase_track_fraction: float = 0.50
    eps: float = 1e-8

    def validate(self) -> None:
        integer_minima = {
            "phase_steps": (self.phase_steps, 8),
            "minimum_frames": (self.minimum_frames, 4),
            "minimum_tracks": (self.minimum_tracks, 3),
            "minimum_component_tracks": (
                self.minimum_component_tracks,
                2,
            ),
            "minimum_active_transitions": (
                self.minimum_active_transitions,
                2,
            ),
            "minimum_pair_active_transitions": (
                self.minimum_pair_active_transitions,
                1,
            ),
            "minimum_component_degree": (
                self.minimum_component_degree,
                1,
            ),
            "minimum_component_core_order": (
                self.minimum_component_core_order,
                1,
            ),
            "maximum_component_graph_diameter": (
                self.maximum_component_graph_diameter,
                1,
            ),
            "spatial_occupancy_bins": (
                self.spatial_occupancy_bins,
                2,
            ),
        }
        for name, (value, minimum) in integer_minima.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                _fail("invalid_config", f"{name} must be >= {minimum}")
        if (
            isinstance(self.maximum_interpolation_gap_frames, bool)
            or not isinstance(self.maximum_interpolation_gap_frames, int)
            or self.maximum_interpolation_gap_frames < 0
        ):
            _fail(
                "invalid_config",
                "maximum_interpolation_gap_frames must be a nonnegative integer",
            )
        if self.minimum_component_tracks > self.minimum_tracks:
            _fail(
                "invalid_config",
                "minimum_component_tracks cannot exceed minimum_tracks",
            )
        unit_values = (
            "visibility_threshold",
            "minimum_track_visibility",
            "minimum_transition_visibility",
            "minimum_amplitude_similarity",
            "minimum_soft_amplitude_similarity",
            "minimum_activity_iou",
            "minimum_soft_activity_iou",
            "minimum_soft_speed_profile_cosine",
            "minimum_component_edge_density",
            "minimum_component_degree_fraction",
            "minimum_component_unique_fraction",
            "maximum_component_fraction",
            "maximum_global_moving_fraction",
            "minimum_global_comoving_fraction",
            "maximum_component_spatial_coverage",
            "maximum_component_axis_coverage",
            "maximum_component_spatial_occupancy",
            "minimum_phase_track_fraction",
        )
        for name in unit_values:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                _fail("invalid_config", f"{name} must be in [0,1]")
        if not 0.0 < self.maximum_component_fraction < 1.0:
            _fail(
                "invalid_config",
                "maximum_component_fraction must be in (0,1)",
            )
        if not 0.0 < self.maximum_global_moving_fraction < 1.0:
            _fail(
                "invalid_config",
                "maximum_global_moving_fraction must be in (0,1)",
            )
        if (
            self.maximum_component_fraction
            > self.maximum_global_moving_fraction
        ):
            _fail(
                "invalid_config",
                "component fraction cannot exceed global moving fraction",
            )
        if (
            self.minimum_global_comoving_fraction
            > self.maximum_global_moving_fraction
        ):
            _fail(
                "invalid_config",
                "global co-motion threshold cannot exceed moving threshold",
            )
        if (
            not -1.0 <= float(self.soft_direction_cosine) <= 1.0
            or not -1.0 <= float(self.strong_direction_cosine) <= 1.0
            or self.soft_direction_cosine > self.strong_direction_cosine
        ):
            _fail(
                "invalid_config",
                "direction cosine thresholds are inconsistent",
            )
        positive_values = (
            "minimum_transition_speed",
            "minimum_track_path_length",
            "minimum_track_excursion",
            "minimum_track_rms_speed",
            "minimum_component_path_length",
            "minimum_component_excursion",
            "minimum_component_rms_speed",
            "spatial_neighbor_radius",
            "articulated_neighbor_radius",
            "minimum_distinct_track_distance",
            "eps",
        )
        for name in positive_values:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                _fail(
                    "invalid_config",
                    f"{name} must be finite and positive",
                )
        if self.articulated_neighbor_radius > self.spatial_neighbor_radius:
            _fail(
                "invalid_config",
                "articulated radius cannot exceed spatial neighbor radius",
            )


@dataclass(frozen=True)
class CoherentMotionComponent:
    """Auditable measurements for one graph connected component."""

    track_indices: tuple[int, ...]
    track_count: int
    unique_track_count: int
    unique_track_fraction: float
    edge_count: int
    edge_density: float
    minimum_degree: int
    required_minimum_degree: int
    core_order: int
    graph_diameter: int
    mean_edge_coherence: float
    median_path_length: float
    median_excursion: float
    rms_speed: float
    active_transition_count: int
    spatial_coverage: float
    spatial_coverage_x: float
    spatial_coverage_y: float
    spatial_bbox_area_coverage: float
    spatial_occupancy: float
    component_fraction: float
    selection_score: float
    accepted: bool
    rejection_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoherentActorSelection:
    """Structured fail-closed result of coherent actor selection."""

    diagnostic_ready: bool
    failure_reason: str | None
    failure_detail: str | None
    components: tuple[CoherentMotionComponent, ...]
    selected_component: int | None
    actor_track_mask: np.ndarray
    actor_track_indices: np.ndarray
    actor_trajectory: np.ndarray
    actor_track_trajectories: np.ndarray
    actor_track_phase_mask: np.ndarray
    phase_times: np.ndarray
    phase_energy: np.ndarray
    phase_visibility: np.ndarray
    eligible_track_count: int
    moving_track_count: int
    global_moving_fraction: float
    moving_spatial_coverage: float
    moving_spatial_coverage_x: float
    moving_spatial_coverage_y: float
    moving_spatial_bbox_area_coverage: float
    moving_spatial_occupancy: float
    global_direction_coherence: float
    global_translation_coherence: float
    global_radial_coherence: float
    global_rotation_coherence: float
    coordinate_space: str
    isotropic_scale: tuple[float, float]
    frame_size: tuple[int, int] | None
    semantic_actor_identified: bool = False
    scope: str = R7_COHERENT_ACTOR_SCOPE
    schema_version: str = R7_COHERENT_ACTOR_SCHEMA

    def to_summary(self) -> dict[str, Any]:
        """Return JSON-safe scalar/component metadata, excluding dense arrays."""

        return {
            "schema_version": self.schema_version,
            "diagnostic_ready": self.diagnostic_ready,
            "failure_reason": self.failure_reason,
            "failure_detail": self.failure_detail,
            "components": [component.to_dict() for component in self.components],
            "selected_component": self.selected_component,
            "actor_track_indices": self.actor_track_indices.tolist(),
            "eligible_track_count": self.eligible_track_count,
            "moving_track_count": self.moving_track_count,
            "global_moving_fraction": self.global_moving_fraction,
            "moving_spatial_coverage": self.moving_spatial_coverage,
            "moving_spatial_coverage_x": self.moving_spatial_coverage_x,
            "moving_spatial_coverage_y": self.moving_spatial_coverage_y,
            "moving_spatial_bbox_area_coverage": (
                self.moving_spatial_bbox_area_coverage
            ),
            "moving_spatial_occupancy": self.moving_spatial_occupancy,
            "global_direction_coherence": self.global_direction_coherence,
            "global_translation_coherence": (
                self.global_translation_coherence
            ),
            "global_radial_coherence": self.global_radial_coherence,
            "global_rotation_coherence": self.global_rotation_coherence,
            "coordinate_space": self.coordinate_space,
            "isotropic_scale": list(self.isotropic_scale),
            "frame_size": (
                list(self.frame_size) if self.frame_size is not None else None
            ),
            "semantic_actor_identified": self.semantic_actor_identified,
            "scope": self.scope,
        }


@dataclass(frozen=True)
class _PairEdge:
    first: int
    second: int
    coherence: float


@dataclass(frozen=True)
class _SpatialMetrics:
    diagonal: float
    x_extent: float
    y_extent: float
    bbox_area: float
    occupancy: float


def _infer_track_count(tracks: Any) -> int:
    try:
        array = np.asarray(tracks)
    except Exception:
        return 0
    if array.ndim == 3 and array.shape[-1] == 2:
        return int(array.shape[1])
    return 0


def _safe_phase_steps(config: CoherentActorConfig) -> int:
    value = getattr(config, "phase_steps", R7_COHERENT_ACTOR_PHASE_STEPS)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    return R7_COHERENT_ACTOR_PHASE_STEPS


def _coordinate_contract(
    frame_size: Any,
) -> tuple[tuple[int, int] | None, tuple[float, float], str]:
    """Validate ``(height,width)`` and return a max-side isotropic scale."""

    if frame_size is None:
        return None, (1.0, 1.0), "normalized-unit-square"
    try:
        values = tuple(frame_size)
    except TypeError:
        _fail("invalid_frame_size", "frame_size must be (height,width)")
    if len(values) != 2:
        _fail("invalid_frame_size", "frame_size must be (height,width)")
    dimensions: list[int] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
        ):
            _fail(
                "invalid_frame_size",
                "frame_size values must be positive integers",
            )
        dimensions.append(int(value))
    height, width = dimensions
    maximum = float(max(height, width))
    scale = (float(width / maximum), float(height / maximum))
    return (
        (height, width),
        scale,
        "normalized-max-side-isotropic",
    )


def _failure_result(
    *,
    reason: str,
    detail: str,
    track_count: int,
    phase_steps: int,
    components: tuple[CoherentMotionComponent, ...] = (),
    eligible_track_count: int = 0,
    moving_track_count: int = 0,
    global_moving_fraction: float = 0.0,
    moving_spatial_coverage: float = 0.0,
    moving_spatial_coverage_x: float = 0.0,
    moving_spatial_coverage_y: float = 0.0,
    moving_spatial_bbox_area_coverage: float = 0.0,
    moving_spatial_occupancy: float = 0.0,
    global_direction_coherence: float = 0.0,
    global_translation_coherence: float = 0.0,
    global_radial_coherence: float = 0.0,
    global_rotation_coherence: float = 0.0,
    coordinate_space: str = "normalized-unit-square",
    isotropic_scale: tuple[float, float] = (1.0, 1.0),
    frame_size: tuple[int, int] | None = None,
) -> CoherentActorSelection:
    return CoherentActorSelection(
        diagnostic_ready=False,
        failure_reason=str(reason),
        failure_detail=str(detail),
        components=components,
        selected_component=None,
        actor_track_mask=np.zeros(max(track_count, 0), dtype=bool),
        actor_track_indices=np.zeros(0, dtype=np.int64),
        actor_trajectory=np.zeros((phase_steps, 2), dtype=np.float32),
        actor_track_trajectories=np.zeros(
            (0, phase_steps, 2),
            dtype=np.float32,
        ),
        actor_track_phase_mask=np.zeros((0, phase_steps), dtype=bool),
        phase_times=np.zeros(phase_steps, dtype=np.float64),
        phase_energy=np.zeros(phase_steps, dtype=np.float32),
        phase_visibility=np.zeros(phase_steps, dtype=np.float32),
        eligible_track_count=int(eligible_track_count),
        moving_track_count=int(moving_track_count),
        global_moving_fraction=float(global_moving_fraction),
        moving_spatial_coverage=float(moving_spatial_coverage),
        moving_spatial_coverage_x=float(moving_spatial_coverage_x),
        moving_spatial_coverage_y=float(moving_spatial_coverage_y),
        moving_spatial_bbox_area_coverage=float(
            moving_spatial_bbox_area_coverage
        ),
        moving_spatial_occupancy=float(moving_spatial_occupancy),
        global_direction_coherence=float(global_direction_coherence),
        global_translation_coherence=float(global_translation_coherence),
        global_radial_coherence=float(global_radial_coherence),
        global_rotation_coherence=float(global_rotation_coherence),
        coordinate_space=str(coordinate_space),
        isotropic_scale=(
            float(isotropic_scale[0]),
            float(isotropic_scale[1]),
        ),
        frame_size=frame_size,
    )


def _validated_inputs(
    tracks: Any,
    visibility: Any,
    frame_times: Any,
    config: CoherentActorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config.validate()
    track_array = np.asarray(tracks)
    if track_array.ndim != 3 or track_array.shape[-1] != 2:
        _fail("invalid_shape", "tracks must have shape [T,N,2]")
    if not np.issubdtype(track_array.dtype, np.number):
        _fail("invalid_dtype", "tracks must be numeric")
    track_array = track_array.astype(np.float64, copy=False)
    if not np.isfinite(track_array).all():
        _fail("non_finite_input", "tracks contain NaN or infinity")

    visibility_array = np.asarray(visibility)
    if visibility_array.ndim != 2:
        _fail("invalid_shape", "visibility must have shape [T,N]")
    if visibility_array.shape != track_array.shape[:2]:
        _fail(
            "invalid_shape",
            "visibility shape must equal tracks.shape[:2]",
        )
    if not np.issubdtype(visibility_array.dtype, np.number):
        _fail("invalid_dtype", "visibility must be numeric")
    visibility_array = visibility_array.astype(np.float64, copy=False)
    if not np.isfinite(visibility_array).all():
        _fail("non_finite_input", "visibility contains NaN or infinity")
    if bool(
        ((visibility_array < 0.0) | (visibility_array > 1.0)).any()
    ):
        _fail("invalid_visibility", "visibility must be in [0,1]")

    time_array = np.asarray(frame_times)
    if time_array.ndim != 1:
        _fail("invalid_shape", "frame_times must have shape [T]")
    if len(time_array) != len(track_array):
        _fail(
            "invalid_shape",
            "frame_times length must equal the track frame count",
        )
    if not np.issubdtype(time_array.dtype, np.number):
        _fail("invalid_dtype", "frame_times must be numeric")
    time_array = time_array.astype(np.float64, copy=False)
    if not np.isfinite(time_array).all():
        _fail("non_finite_input", "frame_times contain NaN or infinity")
    if len(time_array) < 2 or bool((np.diff(time_array) <= 0.0).any()):
        _fail(
            "invalid_frame_times",
            "frame_times must be strictly increasing",
        )
    if len(track_array) < config.minimum_frames:
        _fail(
            "insufficient_frames",
            f"need at least {config.minimum_frames} frames",
        )
    if track_array.shape[1] < config.minimum_tracks:
        _fail(
            "insufficient_tracks",
            f"need at least {config.minimum_tracks} tracks",
        )
    coordinate_span = np.ptp(track_array, axis=(0, 1))
    if (
        not np.isfinite(coordinate_span).all()
        or float(np.max(coordinate_span)) > 10.0
        or float(np.max(np.abs(track_array))) > 10.0
    ):
        _fail(
            "coordinate_system_mismatch",
            "tracks do not resemble normalized frame coordinates",
        )
    return (
        np.ascontiguousarray(track_array),
        np.ascontiguousarray(visibility_array),
        np.ascontiguousarray(time_array),
    )


def _robust_track_motion(
    tracks: np.ndarray,
    visible: np.ndarray,
    step_length: np.ndarray,
    speed: np.ndarray,
    valid_transition: np.ndarray,
    *,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return spike-resistant path, absolute excursion, and RMS speed."""

    track_count = step_length.shape[1]
    paths = np.zeros(track_count, dtype=np.float64)
    excursions = np.zeros(track_count, dtype=np.float64)
    rms = np.zeros(track_count, dtype=np.float64)
    for index in range(track_count):
        visible_values = tracks[visible[:, index], index]
        if len(visible_values):
            excursions[index] = float(
                np.max(
                    np.linalg.norm(
                        visible_values - visible_values[0],
                        axis=-1,
                    )
                )
            )
        keep = valid_transition[:, index]
        lengths = step_length[keep, index]
        speeds = speed[keep, index]
        if len(lengths) == 0:
            continue
        # A lone tracker jump is capped almost to zero when the remaining
        # transitions are static, while sustained motion is left unchanged.
        length_cap = float(np.quantile(lengths, 0.90))
        speed_cap = float(np.quantile(speeds, 0.90))
        if length_cap > eps:
            paths[index] = float(np.sum(np.minimum(lengths, length_cap)))
        if speed_cap > eps:
            clipped = np.minimum(speeds, speed_cap)
            rms[index] = float(np.sqrt(np.mean(clipped * clipped)))
    return paths, excursions, rms


def _pair_statistics(
    first: int,
    second: int,
    *,
    tracks: np.ndarray,
    visible: np.ndarray,
    velocity: np.ndarray,
    speed: np.ndarray,
    transition_visible: np.ndarray,
    active: np.ndarray,
    config: CoherentActorConfig,
) -> tuple[float, float, float, float, float, int]:
    common_frames = visible[:, first] & visible[:, second]
    if not bool(common_frames.any()):
        return math.inf, -1.0, 0.0, 0.0, 0.0, 0
    spatial_distance = float(
        np.median(
            np.linalg.norm(
                tracks[common_frames, first]
                - tracks[common_frames, second],
                axis=-1,
            )
        )
    )
    common = transition_visible[:, first] & transition_visible[:, second]
    both_active = common & active[:, first] & active[:, second]
    active_count = int(np.sum(both_active))
    active_union = common & (active[:, first] | active[:, second])
    union_count = int(np.sum(active_union))
    activity_iou = (
        float(active_count / union_count) if union_count > 0 else 0.0
    )
    if active_count == 0:
        return spatial_distance, -1.0, 0.0, activity_iou, 0.0, 0

    first_velocity = velocity[both_active, first]
    second_velocity = velocity[both_active, second]
    first_speed = speed[both_active, first]
    second_speed = speed[both_active, second]
    denominator = np.maximum(
        first_speed * second_speed,
        config.eps,
    )
    cosines = np.sum(first_velocity * second_velocity, axis=-1) / denominator
    pair_weights = np.minimum(first_speed, second_speed)
    direction = float(
        np.sum(pair_weights * cosines)
        / max(float(np.sum(pair_weights)), config.eps)
    )
    log_ratio = np.abs(
        np.log(
            (first_speed + config.eps)
            / (second_speed + config.eps)
        )
    )
    amplitude_similarity = float(np.exp(-float(np.median(log_ratio))))

    common_first_speed = speed[common, first]
    common_second_speed = speed[common, second]
    profile_denominator = float(
        np.linalg.norm(common_first_speed)
        * np.linalg.norm(common_second_speed)
    )
    profile_cosine = (
        float(
            np.dot(common_first_speed, common_second_speed)
            / profile_denominator
        )
        if profile_denominator > config.eps
        else 0.0
    )
    return (
        spatial_distance,
        float(np.clip(direction, -1.0, 1.0)),
        float(np.clip(amplitude_similarity, 0.0, 1.0)),
        float(np.clip(activity_iou, 0.0, 1.0)),
        float(np.clip(profile_cosine, 0.0, 1.0)),
        active_count,
    )


def _build_edges(
    candidates: np.ndarray,
    *,
    tracks: np.ndarray,
    visible: np.ndarray,
    velocity: np.ndarray,
    speed: np.ndarray,
    transition_visible: np.ndarray,
    active: np.ndarray,
    config: CoherentActorConfig,
) -> tuple[_PairEdge, ...]:
    edges: list[_PairEdge] = []
    candidate_values = sorted(int(index) for index in candidates)
    for offset, first in enumerate(candidate_values):
        for second in candidate_values[offset + 1 :]:
            (
                distance,
                direction,
                amplitude,
                activity_iou,
                profile_cosine,
                active_count,
            ) = _pair_statistics(
                first,
                second,
                tracks=tracks,
                visible=visible,
                velocity=velocity,
                speed=speed,
                transition_visible=transition_visible,
                active=active,
                config=config,
            )
            if active_count < config.minimum_pair_active_transitions:
                continue
            strong = (
                distance <= config.spatial_neighbor_radius
                and direction >= config.strong_direction_cosine
                and amplitude >= config.minimum_amplitude_similarity
                and activity_iou >= config.minimum_activity_iou
            )
            # The soft edge is intentionally local.  It connects jointly
            # activated articulated parts whose directions differ, without
            # allowing distant objects with coincident speed envelopes to
            # collapse into one actor.
            soft = (
                distance <= config.articulated_neighbor_radius
                and direction >= config.soft_direction_cosine
                and amplitude >= config.minimum_soft_amplitude_similarity
                and activity_iou >= config.minimum_soft_activity_iou
                and profile_cosine
                >= config.minimum_soft_speed_profile_cosine
            )
            if not strong and not soft:
                continue
            direction_unit = 0.5 * (direction + 1.0)
            coherence = (
                0.35 * direction_unit
                + 0.25 * amplitude
                + 0.20 * activity_iou
                + 0.20 * profile_cosine
            )
            edges.append(
                _PairEdge(
                    first=first,
                    second=second,
                    coherence=float(np.clip(coherence, 0.0, 1.0)),
                )
            )
    return tuple(edges)


def _connected_components(
    candidates: np.ndarray,
    edges: tuple[_PairEdge, ...],
) -> tuple[tuple[int, ...], ...]:
    adjacency = {
        int(index): set() for index in np.asarray(candidates).tolist()
    }
    for edge in edges:
        adjacency[edge.first].add(edge.second)
        adjacency[edge.second].add(edge.first)
    remaining = set(adjacency)
    output: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        stack = [start]
        members: list[int] = []
        remaining.remove(start)
        while stack:
            current = stack.pop()
            members.append(current)
            neighbors = sorted(adjacency[current] & remaining, reverse=True)
            for neighbor in neighbors:
                remaining.remove(neighbor)
                stack.append(neighbor)
        output.append(tuple(sorted(members)))
    return tuple(output)


def _spatial_metrics(
    component: tuple[int, ...],
    eligible: np.ndarray,
    tracks: np.ndarray,
    visible: np.ndarray,
    *,
    bins: int,
    eps: float,
) -> _SpatialMetrics:
    track_centers = np.zeros((tracks.shape[1], 2), dtype=np.float64)
    for index in np.flatnonzero(eligible):
        track_centers[index] = np.median(
            tracks[visible[:, index], index],
            axis=0,
        )
    eligible_centers = track_centers[eligible]
    component_centers = track_centers[np.asarray(component, dtype=np.int64)]
    global_extent = np.ptp(eligible_centers, axis=0)
    component_extent = np.ptp(component_centers, axis=0)
    global_diagonal = float(np.linalg.norm(global_extent))
    component_diagonal = float(np.linalg.norm(component_extent))
    if global_diagonal <= eps:
        diagonal = 1.0
    else:
        diagonal = float(
            np.clip(component_diagonal / global_diagonal, 0.0, 1.0)
        )
    axis_ratios = np.ones(2, dtype=np.float64)
    for dimension in range(2):
        if global_extent[dimension] > eps:
            axis_ratios[dimension] = float(
                np.clip(
                    component_extent[dimension]
                    / global_extent[dimension],
                    0.0,
                    1.0,
                )
            )
    global_area = float(np.prod(global_extent))
    component_area = float(np.prod(component_extent))
    bbox_area = (
        float(np.clip(component_area / global_area, 0.0, 1.0))
        if global_area > eps
        else diagonal
    )

    global_minimum = np.min(eligible_centers, axis=0)
    normalized = np.zeros_like(eligible_centers)
    for dimension in range(2):
        if global_extent[dimension] > eps:
            normalized[:, dimension] = (
                eligible_centers[:, dimension] - global_minimum[dimension]
            ) / global_extent[dimension]
    cell_coordinates = np.minimum(
        np.floor(normalized * bins).astype(np.int64),
        bins - 1,
    )
    eligible_indices = np.flatnonzero(eligible)
    component_lookup = {
        int(value) for value in np.asarray(component, dtype=np.int64)
    }
    eligible_cells = {
        (int(cell[0]), int(cell[1])) for cell in cell_coordinates
    }
    component_cells = {
        (int(cell_coordinates[offset, 0]), int(cell_coordinates[offset, 1]))
        for offset, track_index in enumerate(eligible_indices)
        if int(track_index) in component_lookup
    }
    occupancy = float(
        len(component_cells) / max(len(eligible_cells), 1)
    )
    return _SpatialMetrics(
        diagonal=diagonal,
        x_extent=float(axis_ratios[0]),
        y_extent=float(axis_ratios[1]),
        bbox_area=bbox_area,
        occupancy=float(np.clip(occupancy, 0.0, 1.0)),
    )


def _global_motion_diagnostics(
    candidates: np.ndarray,
    *,
    eligible: np.ndarray,
    tracks: np.ndarray,
    visible: np.ndarray,
    step: np.ndarray,
    active: np.ndarray,
    config: CoherentActorConfig,
) -> tuple[_SpatialMetrics, float, float, float, float]:
    """Measure spatial extent and translation/radial/rotation flow modes."""

    members = tuple(int(index) for index in candidates)
    spatial = _spatial_metrics(
        members,
        eligible,
        tracks,
        visible,
        bins=config.spatial_occupancy_bins,
        eps=config.eps,
    )
    directions: list[np.ndarray] = []
    for index in candidates:
        keep = active[:, index]
        displacement = np.sum(step[keep, index], axis=0)
        norm = float(np.linalg.norm(displacement))
        if norm > config.eps:
            directions.append(displacement / norm)
    if not directions:
        direction_coherence = 0.0
    else:
        direction_coherence = float(
            np.linalg.norm(np.sum(np.stack(directions), axis=0))
            / len(directions)
        )

    translation_values: list[float] = []
    radial_values: list[float] = []
    rotation_values: list[float] = []
    transition_weights: list[float] = []
    for transition in range(step.shape[0]):
        selected = candidates[active[transition, candidates]]
        if len(selected) < config.minimum_component_tracks:
            continue
        flow = step[transition, selected]
        magnitude = np.linalg.norm(flow, axis=1)
        valid_flow = magnitude > config.eps
        selected = selected[valid_flow]
        flow = flow[valid_flow]
        magnitude = magnitude[valid_flow]
        if len(selected) < config.minimum_component_tracks:
            continue
        total = float(np.sum(magnitude))
        translation_values.append(
            float(np.linalg.norm(np.sum(flow, axis=0)) / max(total, config.eps))
        )

        # Candidate-centered radial/tangential diagnostics are invariant to
        # where a local component lies in the frame.  These are descriptive
        # flow modes, not semantic actor evidence.
        center = np.median(tracks[transition, selected], axis=0)
        relative = tracks[transition, selected] - center
        radius = np.linalg.norm(relative, axis=1)
        valid_radius = radius > config.eps
        if int(np.sum(valid_radius)) < config.minimum_component_tracks:
            radial_values.append(0.0)
            rotation_values.append(0.0)
        else:
            unit_flow = flow[valid_radius] / magnitude[valid_radius, None]
            radial_unit = (
                relative[valid_radius] / radius[valid_radius, None]
            )
            tangent_unit = np.stack(
                (-radial_unit[:, 1], radial_unit[:, 0]),
                axis=1,
            )
            model_weights = magnitude[valid_radius]
            weight_sum = max(
                float(np.sum(model_weights)),
                config.eps,
            )
            radial_values.append(
                abs(
                    float(
                        np.sum(
                            model_weights
                            * np.sum(unit_flow * radial_unit, axis=1)
                        )
                        / weight_sum
                    )
                )
            )
            rotation_values.append(
                abs(
                    float(
                        np.sum(
                            model_weights
                            * np.sum(unit_flow * tangent_unit, axis=1)
                        )
                        / weight_sum
                    )
                )
            )
        transition_weights.append(total)

    def weighted(values: list[float]) -> float:
        if not values:
            return 0.0
        weights = np.asarray(
            transition_weights[: len(values)],
            dtype=np.float64,
        )
        return float(
            np.clip(
                np.average(np.asarray(values), weights=weights),
                0.0,
                1.0,
            )
        )

    return (
        spatial,
        float(np.clip(direction_coherence, 0.0, 1.0)),
        weighted(translation_values),
        weighted(radial_values),
        weighted(rotation_values),
    )


def _graph_structure(
    members: tuple[int, ...],
    edges: tuple[_PairEdge, ...],
    *,
    degree_fraction: float,
    minimum_degree: int,
) -> tuple[int, int, int, int]:
    """Return minimum degree, required degree, core order, and diameter."""

    member_set = set(members)
    adjacency = {index: set() for index in members}
    for edge in edges:
        if edge.first in member_set and edge.second in member_set:
            adjacency[edge.first].add(edge.second)
            adjacency[edge.second].add(edge.first)
    degrees = [len(adjacency[index]) for index in members]
    observed_minimum = min(degrees) if degrees else 0
    required = max(
        int(minimum_degree),
        int(math.ceil(degree_fraction * max(len(members) - 1, 0))),
    )

    remaining = {index: set(neighbors) for index, neighbors in adjacency.items()}
    core_order = 0
    while remaining:
        current = min(
            remaining,
            key=lambda index: (len(remaining[index]), index),
        )
        degree = len(remaining[current])
        core_order = max(core_order, degree)
        for neighbor in tuple(remaining[current]):
            remaining[neighbor].discard(current)
        del remaining[current]

    diameter = 0
    for start in members:
        distances = {start: 0}
        queue = [start]
        while queue:
            current = queue.pop(0)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        if len(distances) != len(members):
            return observed_minimum, required, core_order, len(members)
        diameter = max(diameter, max(distances.values(), default=0))
    return observed_minimum, required, core_order, diameter


def _distinct_track_count(
    members: tuple[int, ...],
    *,
    tracks: np.ndarray,
    visible: np.ndarray,
    minimum_distance: float,
) -> int:
    """Count spatially distinct trajectories without depending on input order."""

    records: list[tuple[tuple[float, ...], np.ndarray]] = []
    for index in members:
        values = tracks[visible[:, index], index]
        center = np.median(values, axis=0)
        trajectory = np.where(
            visible[:, index, None],
            tracks[:, index],
            0.0,
        )
        signature = tuple(
            float(value)
            for value in np.round(
                np.concatenate(
                    (
                        center,
                        trajectory.reshape(-1),
                        visible[:, index].astype(np.float64),
                    )
                ),
                decimals=12,
            )
        )
        records.append((signature, center))
    accepted: list[np.ndarray] = []
    for _, center in sorted(records, key=lambda value: value[0]):
        if all(
            float(np.linalg.norm(center - previous)) >= minimum_distance
            for previous in accepted
        ):
            accepted.append(center)
    return len(accepted)


def _component_metrics(
    members: tuple[int, ...],
    *,
    edges: tuple[_PairEdge, ...],
    paths: np.ndarray,
    excursions: np.ndarray,
    track_rms: np.ndarray,
    speed: np.ndarray,
    transition_visible: np.ndarray,
    eligible: np.ndarray,
    tracks: np.ndarray,
    visible: np.ndarray,
    config: CoherentActorConfig,
) -> CoherentMotionComponent:
    member_set = set(members)
    component_edges = tuple(
        edge
        for edge in edges
        if edge.first in member_set and edge.second in member_set
    )
    count = len(members)
    maximum_edges = count * (count - 1) // 2
    edge_density = (
        float(len(component_edges) / maximum_edges)
        if maximum_edges > 0
        else 0.0
    )
    mean_coherence = (
        float(np.mean([edge.coherence for edge in component_edges]))
        if component_edges
        else 0.0
    )
    (
        minimum_degree,
        required_minimum_degree,
        core_order,
        graph_diameter,
    ) = _graph_structure(
        members,
        component_edges,
        degree_fraction=config.minimum_component_degree_fraction,
        minimum_degree=config.minimum_component_degree,
    )
    unique_track_count = _distinct_track_count(
        members,
        tracks=tracks,
        visible=visible,
        minimum_distance=config.minimum_distinct_track_distance,
    )
    unique_track_fraction = float(unique_track_count / max(count, 1))
    member_array = np.asarray(members, dtype=np.int64)
    median_path = float(np.median(paths[member_array]))
    median_excursion = float(np.median(excursions[member_array]))
    rms_speed = float(np.median(track_rms[member_array]))
    aggregate_speed = np.zeros(speed.shape[0], dtype=np.float64)
    for transition in range(speed.shape[0]):
        supported = member_array[
            transition_visible[transition, member_array]
        ]
        if len(supported) >= config.minimum_component_tracks:
            aggregate_speed[transition] = float(
                np.median(speed[transition, supported])
            )
    active_count = int(
        np.sum(aggregate_speed >= config.minimum_transition_speed)
    )
    eligible_count = max(int(np.sum(eligible)), 1)
    component_fraction = float(count / eligible_count)
    spatial = _spatial_metrics(
        members,
        eligible,
        tracks,
        visible,
        bins=config.spatial_occupancy_bins,
        eps=config.eps,
    )

    rejection: str | None = None
    if count < config.minimum_component_tracks:
        rejection = "too_few_component_tracks"
    elif (
        unique_track_count < config.minimum_component_tracks
        or unique_track_fraction < config.minimum_component_unique_fraction
    ):
        rejection = "insufficient_distinct_component_tracks"
    elif edge_density < config.minimum_component_edge_density:
        rejection = "low_component_edge_density"
    elif minimum_degree < required_minimum_degree:
        rejection = "low_component_minimum_degree"
    elif core_order < config.minimum_component_core_order:
        rejection = "low_component_core_order"
    elif graph_diameter > config.maximum_component_graph_diameter:
        rejection = "excessive_component_graph_diameter"
    elif active_count < config.minimum_active_transitions:
        rejection = "insufficient_component_activity"
    elif median_path < config.minimum_component_path_length:
        rejection = "insufficient_component_path"
    elif median_excursion < config.minimum_component_excursion:
        rejection = "insufficient_component_excursion"
    elif rms_speed < config.minimum_component_rms_speed:
        rejection = "insufficient_component_energy"
    elif component_fraction > config.maximum_component_fraction:
        rejection = "global_residual_motion"
    elif spatial.diagonal > config.maximum_component_spatial_coverage:
        rejection = "global_residual_motion"
    elif (
        max(spatial.x_extent, spatial.y_extent)
        > config.maximum_component_axis_coverage
    ):
        rejection = "global_residual_motion"
    elif (
        spatial.occupancy
        > config.maximum_component_spatial_occupancy
    ):
        rejection = "global_residual_motion"

    selection_score = float(
        median_path
        * max(median_excursion, config.eps)
        * rms_speed
        * math.sqrt(max(count, 1))
        * (0.50 + 0.50 * mean_coherence)
        * (0.50 + 0.50 * min(active_count / speed.shape[0], 1.0))
    )
    return CoherentMotionComponent(
        track_indices=members,
        track_count=count,
        unique_track_count=unique_track_count,
        unique_track_fraction=unique_track_fraction,
        edge_count=len(component_edges),
        edge_density=edge_density,
        minimum_degree=minimum_degree,
        required_minimum_degree=required_minimum_degree,
        core_order=core_order,
        graph_diameter=graph_diameter,
        mean_edge_coherence=mean_coherence,
        median_path_length=median_path,
        median_excursion=median_excursion,
        rms_speed=rms_speed,
        active_transition_count=active_count,
        spatial_coverage=spatial.diagonal,
        spatial_coverage_x=spatial.x_extent,
        spatial_coverage_y=spatial.y_extent,
        spatial_bbox_area_coverage=spatial.bbox_area,
        spatial_occupancy=spatial.occupancy,
        component_fraction=component_fraction,
        selection_score=selection_score,
        accepted=rejection is None,
        rejection_reason=rejection,
    )


def _track_signature(
    index: int,
    tracks: np.ndarray,
    visible: np.ndarray,
) -> tuple[float, ...]:
    """Physical per-track signature independent of tracker array position."""

    values = np.where(
        visible[:, index, None],
        tracks[:, index],
        0.0,
    )
    signature = np.concatenate(
        (
            values.reshape(-1),
            visible[:, index].astype(np.float64),
        )
    )
    return tuple(
        float(value) for value in np.round(signature, decimals=12)
    )


def _canonical_track_indices(
    indices: tuple[int, ...],
    tracks: np.ndarray,
    visible: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        sorted(
            (int(index) for index in indices),
            key=lambda index: _track_signature(index, tracks, visible),
        ),
        dtype=np.int64,
    )


def _component_signature(
    component: CoherentMotionComponent,
    tracks: np.ndarray,
    visible: np.ndarray,
) -> tuple[tuple[float, ...], ...]:
    """Content-based tie break that does not depend on input track order."""

    return tuple(
        _track_signature(int(index), tracks, visible)
        for index in _canonical_track_indices(
            component.track_indices,
            tracks,
            visible,
        )
    )


def _visible_runs(
    mask: np.ndarray,
    *,
    maximum_gap: int,
) -> tuple[tuple[int, int], ...]:
    """Return half-open visible runs, optionally bridging a bounded gap."""

    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    if not len(indices):
        return ()
    runs: list[tuple[int, int]] = []
    start = int(indices[0])
    previous = int(indices[0])
    for value in indices[1:]:
        current = int(value)
        if current - previous - 1 > maximum_gap:
            runs.append((start, previous + 1))
            start = current
        previous = current
    runs.append((start, previous + 1))
    return tuple(runs)


def _phase_outputs(
    component: CoherentMotionComponent,
    *,
    tracks: np.ndarray,
    visibility: np.ndarray,
    visible: np.ndarray,
    frame_times: np.ndarray,
    speed: np.ndarray,
    transition_visible: np.ndarray,
    config: CoherentActorConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    members = _canonical_track_indices(
        component.track_indices,
        tracks,
        visible,
    )
    frame_support = np.sum(visible[:, members], axis=1)
    minimum_support = max(
        config.minimum_component_tracks,
        int(math.ceil(config.minimum_phase_track_fraction * len(members))),
    )
    supported = frame_support >= minimum_support
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index, value in enumerate(supported.tolist() + [False]):
        if value and run_start is None:
            run_start = index
        elif not value and run_start is not None:
            runs.append((run_start, index))
            run_start = None
    if not runs:
        _fail(
            "insufficient_component_visibility",
            "selected component has no supported time interval",
        )
    # Longest run, then earliest run: deterministic and fail-closed.
    start, stop = min(runs, key=lambda item: (-(item[1] - item[0]), item[0]))
    if stop - start < config.minimum_frames:
        _fail(
            "insufficient_component_visibility",
            "selected component lacks a long continuous visible interval",
        )
    phase_times = np.linspace(
        frame_times[start],
        frame_times[stop - 1],
        config.phase_steps,
        dtype=np.float64,
    )

    absolute_trajectories = np.zeros(
        (len(members), config.phase_steps, 2),
        dtype=np.float64,
    )
    phase_mask = np.zeros(
        (len(members), config.phase_steps),
        dtype=bool,
    )
    interpolated_visibility = np.zeros(
        (len(members), config.phase_steps),
        dtype=np.float64,
    )
    for output_index, track_index in enumerate(members):
        local_visible = visible[start:stop, track_index]
        for local_start, local_stop in _visible_runs(
            local_visible,
            maximum_gap=config.maximum_interpolation_gap_frames,
        ):
            frame_indices = np.arange(
                start + local_start,
                start + local_stop,
                dtype=np.int64,
            )
            keep = visible[frame_indices, track_index]
            frame_indices = frame_indices[keep]
            if len(frame_indices) < 2:
                continue
            source_times = frame_times[frame_indices]
            source_values = tracks[frame_indices, track_index]
            in_support = (
                (phase_times >= source_times[0])
                & (phase_times <= source_times[-1])
            )
            if not bool(in_support.any()):
                continue
            for dimension in range(2):
                absolute_trajectories[
                    output_index, in_support, dimension
                ] = np.interp(
                    phase_times[in_support],
                    source_times,
                    source_values[:, dimension],
                )
            interpolated_visibility[output_index, in_support] = np.interp(
                phase_times[in_support],
                source_times,
                visibility[frame_indices, track_index],
            )
            phase_mask[output_index, in_support] = True

    per_phase_support = np.sum(phase_mask, axis=0)
    if bool((per_phase_support < minimum_support).any()):
        _fail(
            "insufficient_component_visibility",
            "actor trajectories cannot be interpolated across all phases",
        )
    aggregate = np.zeros((config.phase_steps, 2), dtype=np.float64)
    for phase in range(config.phase_steps):
        aggregate[phase] = np.median(
            absolute_trajectories[phase_mask[:, phase], phase],
            axis=0,
        )
    aggregate -= aggregate[0]

    relative_trajectories = absolute_trajectories.copy()
    for index in range(len(relative_trajectories)):
        valid_phases = np.flatnonzero(phase_mask[index])
        if len(valid_phases):
            anchor = relative_trajectories[index, valid_phases[0]].copy()
            relative_trajectories[index, valid_phases] -= anchor

    transition_energy = np.zeros(len(frame_times) - 1, dtype=np.float64)
    for transition in range(start, stop - 1):
        keep = members[transition_visible[transition, members]]
        if len(keep) >= minimum_support:
            transition_energy[transition] = float(
                np.median(speed[transition, keep])
            )
    event_slice = slice(start, stop - 1)
    event_energy = transition_energy[event_slice]
    if (
        len(event_energy) < config.minimum_active_transitions
        or int(
            np.sum(event_energy >= config.minimum_transition_speed)
        )
        < config.minimum_active_transitions
    ):
        _fail(
            "insufficient_component_activity",
            "phase interval lacks sustained component motion",
        )
    transition_midtimes = (
        frame_times[start : stop - 1] + frame_times[start + 1 : stop]
    ) / 2.0
    phase_energy = np.interp(
        phase_times,
        transition_midtimes,
        event_energy,
        left=float(event_energy[0]),
        right=float(event_energy[-1]),
    )
    phase_visibility = np.zeros(config.phase_steps, dtype=np.float64)
    for phase in range(config.phase_steps):
        phase_visibility[phase] = float(
            np.mean(
                interpolated_visibility[phase_mask[:, phase], phase]
            )
        )
    arrays = (
        aggregate,
        relative_trajectories,
        phase_mask,
        phase_times,
        phase_energy,
        phase_visibility,
    )
    if not all(np.isfinite(value).all() for value in arrays):
        _fail("non_finite_output", "coherent actor output is non-finite")
    return arrays


def select_coherent_actor(
    stabilized_tracks: Any,
    visibility: Any,
    frame_times: Any,
    *,
    config: CoherentActorConfig | None = None,
    frame_size: tuple[int, int] | None = None,
) -> CoherentActorSelection:
    """Select a local coherent actor component from stabilized point tracks.

    ``frame_size`` follows the upstream ``(height,width)`` convention.  When
    supplied, equal pixel motion has equal length in both axes; omitting it
    preserves the legacy normalized-unit-square coordinate system.

    This function never turns an invalid or underdetermined input into an
    actor representation.  Expected failures are encoded in the result, and
    callers should consume dense outputs only when ``diagnostic_ready`` is
    true.  Even a ready result is only a geometric coherent-motion proposal,
    not evidence that the component is a semantic actor.
    """

    cfg = config or CoherentActorConfig()
    inferred_track_count = _infer_track_count(stabilized_tracks)
    phase_steps = _safe_phase_steps(cfg)
    validated_frame_size: tuple[int, int] | None = None
    isotropic_scale = (1.0, 1.0)
    coordinate_space = "normalized-unit-square"
    try:
        (
            validated_frame_size,
            isotropic_scale,
            coordinate_space,
        ) = _coordinate_contract(frame_size)
        tracks, visibility_array, times = _validated_inputs(
            stabilized_tracks,
            visibility,
            frame_times,
            cfg,
        )
        tracks = np.ascontiguousarray(
            tracks
            * np.asarray(isotropic_scale, dtype=np.float64)[None, None]
        )
        frame_count, track_count = tracks.shape[:2]
        visible = visibility_array >= cfg.visibility_threshold
        transition_visible = visible[:-1] & visible[1:]
        dt = np.diff(times)
        step = np.diff(tracks, axis=0)
        velocity = step / dt[:, None, None]
        speed = np.linalg.norm(velocity, axis=-1)
        step_length = np.linalg.norm(step, axis=-1)
        speed[~transition_visible] = 0.0
        step_length[~transition_visible] = 0.0

        frame_visibility = np.mean(visible, axis=0)
        transition_visibility = np.mean(transition_visible, axis=0)
        eligible = (
            (frame_visibility >= cfg.minimum_track_visibility)
            & (
                transition_visibility
                >= cfg.minimum_transition_visibility
            )
        )
        eligible_count = int(np.sum(eligible))
        if eligible_count < cfg.minimum_tracks:
            _fail(
                "insufficient_visible_tracks",
                (
                    f"only {eligible_count} tracks meet visibility "
                    "requirements"
                ),
            )

        paths, excursions, track_rms = _robust_track_motion(
            tracks,
            visible,
            step_length,
            speed,
            transition_visible,
            eps=cfg.eps,
        )
        active = (
            transition_visible
            & (speed >= cfg.minimum_transition_speed)
        )
        active_counts = np.sum(active, axis=0)
        moving = (
            eligible
            & (active_counts >= cfg.minimum_active_transitions)
            & (paths >= cfg.minimum_track_path_length)
            & (excursions >= cfg.minimum_track_excursion)
            & (track_rms >= cfg.minimum_track_rms_speed)
        )
        candidates = np.flatnonzero(moving)
        moving_count = len(candidates)
        global_moving_fraction = float(
            moving_count / max(eligible_count, 1)
        )
        if moving_count == 0:
            return _failure_result(
                reason="no_moving_tracks",
                detail=(
                    "no visible track exceeds absolute motion requirements"
                ),
                track_count=track_count,
                phase_steps=cfg.phase_steps,
                eligible_track_count=eligible_count,
                coordinate_space=coordinate_space,
                isotropic_scale=isotropic_scale,
                frame_size=validated_frame_size,
            )
        (
            moving_spatial,
            global_direction_coherence,
            global_translation_coherence,
            global_radial_coherence,
            global_rotation_coherence,
        ) = _global_motion_diagnostics(
            candidates,
            eligible=eligible,
            tracks=tracks,
            visible=visible,
            step=step,
            active=active,
            config=cfg,
        )
        if global_moving_fraction > cfg.maximum_global_moving_fraction:
            return _failure_result(
                reason="global_residual_motion",
                detail=(
                    f"{moving_count}/{eligible_count} eligible tracks move; "
                    "camera residual is not actor-local"
                ),
                track_count=track_count,
                phase_steps=cfg.phase_steps,
                eligible_track_count=eligible_count,
                moving_track_count=moving_count,
                global_moving_fraction=global_moving_fraction,
                moving_spatial_coverage=moving_spatial.diagonal,
                moving_spatial_coverage_x=moving_spatial.x_extent,
                moving_spatial_coverage_y=moving_spatial.y_extent,
                moving_spatial_bbox_area_coverage=moving_spatial.bbox_area,
                moving_spatial_occupancy=moving_spatial.occupancy,
                global_direction_coherence=global_direction_coherence,
                global_translation_coherence=(
                    global_translation_coherence
                ),
                global_radial_coherence=global_radial_coherence,
                global_rotation_coherence=global_rotation_coherence,
                coordinate_space=coordinate_space,
                isotropic_scale=isotropic_scale,
                frame_size=validated_frame_size,
            )
        spatially_global = bool(
            moving_spatial.diagonal
            > cfg.maximum_component_spatial_coverage
            or max(
                moving_spatial.x_extent,
                moving_spatial.y_extent,
            )
            > cfg.maximum_component_axis_coverage
            or moving_spatial.occupancy
            > cfg.maximum_component_spatial_occupancy
        )
        if (
            global_moving_fraction
            >= cfg.minimum_global_comoving_fraction
            and spatially_global
        ):
            return _failure_result(
                reason="global_residual_motion",
                detail=(
                    "a high fraction of moving tracks is spatially global; "
                    "direction cancellation cannot make it actor-local"
                ),
                track_count=track_count,
                phase_steps=cfg.phase_steps,
                eligible_track_count=eligible_count,
                moving_track_count=moving_count,
                global_moving_fraction=global_moving_fraction,
                moving_spatial_coverage=moving_spatial.diagonal,
                moving_spatial_coverage_x=moving_spatial.x_extent,
                moving_spatial_coverage_y=moving_spatial.y_extent,
                moving_spatial_bbox_area_coverage=moving_spatial.bbox_area,
                moving_spatial_occupancy=moving_spatial.occupancy,
                global_direction_coherence=global_direction_coherence,
                global_translation_coherence=(
                    global_translation_coherence
                ),
                global_radial_coherence=global_radial_coherence,
                global_rotation_coherence=global_rotation_coherence,
                coordinate_space=coordinate_space,
                isotropic_scale=isotropic_scale,
                frame_size=validated_frame_size,
            )

        edges = _build_edges(
            candidates,
            tracks=tracks,
            visible=visible,
            velocity=velocity,
            speed=speed,
            transition_visible=transition_visible,
            active=active,
            config=cfg,
        )
        member_components = _connected_components(candidates, edges)
        measured = tuple(
            _component_metrics(
                members,
                edges=edges,
                paths=paths,
                excursions=excursions,
                track_rms=track_rms,
                speed=speed,
                transition_visible=transition_visible,
                eligible=eligible,
                tracks=tracks,
                visible=visible,
                config=cfg,
            )
            for members in member_components
        )
        # Canonical component order is based on physical content rather than
        # tracker indexing, so a permutation of input tracks cannot alter the
        # selected physical component.
        component_order = sorted(
            range(len(measured)),
            key=lambda index: _component_signature(
                measured[index],
                tracks,
                visible,
            ),
        )
        components = tuple(measured[index] for index in component_order)
        accepted = [
            index
            for index, component in enumerate(components)
            if component.accepted
        ]
        if not accepted:
            reasons = {
                component.rejection_reason for component in components
            }
            failure_reason = (
                "global_residual_motion"
                if "global_residual_motion" in reasons
                else "no_coherent_component"
            )
            return _failure_result(
                reason=failure_reason,
                detail="no graph component passes coherence and locality gates",
                track_count=track_count,
                phase_steps=cfg.phase_steps,
                components=components,
                eligible_track_count=eligible_count,
                moving_track_count=moving_count,
                global_moving_fraction=global_moving_fraction,
                moving_spatial_coverage=moving_spatial.diagonal,
                moving_spatial_coverage_x=moving_spatial.x_extent,
                moving_spatial_coverage_y=moving_spatial.y_extent,
                moving_spatial_bbox_area_coverage=moving_spatial.bbox_area,
                moving_spatial_occupancy=moving_spatial.occupancy,
                global_direction_coherence=global_direction_coherence,
                global_translation_coherence=(
                    global_translation_coherence
                ),
                global_radial_coherence=global_radial_coherence,
                global_rotation_coherence=global_rotation_coherence,
                coordinate_space=coordinate_space,
                isotropic_scale=isotropic_scale,
                frame_size=validated_frame_size,
            )
        selected_index = min(
            accepted,
            key=lambda index: (
                -components[index].selection_score,
                -components[index].track_count,
                -components[index].median_path_length,
                _component_signature(components[index], tracks, visible),
            ),
        )
        selected = components[selected_index]
        try:
            (
                actor_trajectory,
                track_trajectories,
                track_phase_mask,
                phase_times,
                phase_energy,
                phase_visibility,
            ) = _phase_outputs(
                selected,
                tracks=tracks,
                visibility=visibility_array,
                visible=visible,
                frame_times=times,
                speed=speed,
                transition_visible=transition_visible,
                config=cfg,
            )
        except _SelectionFailure as error:
            return _failure_result(
                reason=error.reason,
                detail=error.detail,
                track_count=track_count,
                phase_steps=cfg.phase_steps,
                components=components,
                eligible_track_count=eligible_count,
                moving_track_count=moving_count,
                global_moving_fraction=global_moving_fraction,
                moving_spatial_coverage=moving_spatial.diagonal,
                moving_spatial_coverage_x=moving_spatial.x_extent,
                moving_spatial_coverage_y=moving_spatial.y_extent,
                moving_spatial_bbox_area_coverage=moving_spatial.bbox_area,
                moving_spatial_occupancy=moving_spatial.occupancy,
                global_direction_coherence=global_direction_coherence,
                global_translation_coherence=(
                    global_translation_coherence
                ),
                global_radial_coherence=global_radial_coherence,
                global_rotation_coherence=global_rotation_coherence,
                coordinate_space=coordinate_space,
                isotropic_scale=isotropic_scale,
                frame_size=validated_frame_size,
            )
        actor_indices = _canonical_track_indices(
            selected.track_indices,
            tracks,
            visible,
        )
        actor_mask = np.zeros(track_count, dtype=bool)
        actor_mask[actor_indices] = True
        return CoherentActorSelection(
            diagnostic_ready=True,
            failure_reason=None,
            failure_detail=None,
            components=components,
            selected_component=selected_index,
            actor_track_mask=actor_mask,
            actor_track_indices=actor_indices,
            actor_trajectory=actor_trajectory.astype(np.float32),
            actor_track_trajectories=track_trajectories.astype(np.float32),
            actor_track_phase_mask=track_phase_mask,
            phase_times=phase_times,
            phase_energy=phase_energy.astype(np.float32),
            phase_visibility=phase_visibility.astype(np.float32),
            eligible_track_count=eligible_count,
            moving_track_count=moving_count,
            global_moving_fraction=global_moving_fraction,
            moving_spatial_coverage=moving_spatial.diagonal,
            moving_spatial_coverage_x=moving_spatial.x_extent,
            moving_spatial_coverage_y=moving_spatial.y_extent,
            moving_spatial_bbox_area_coverage=moving_spatial.bbox_area,
            moving_spatial_occupancy=moving_spatial.occupancy,
            global_direction_coherence=global_direction_coherence,
            global_translation_coherence=global_translation_coherence,
            global_radial_coherence=global_radial_coherence,
            global_rotation_coherence=global_rotation_coherence,
            coordinate_space=coordinate_space,
            isotropic_scale=isotropic_scale,
            frame_size=validated_frame_size,
        )
    except _SelectionFailure as error:
        return _failure_result(
            reason=error.reason,
            detail=error.detail,
            track_count=inferred_track_count,
            phase_steps=phase_steps,
            coordinate_space=coordinate_space,
            isotropic_scale=isotropic_scale,
            frame_size=validated_frame_size,
        )


__all__ = [
    "CoherentActorConfig",
    "CoherentActorSelection",
    "CoherentMotionComponent",
    "R7_COHERENT_ACTOR_PHASE_STEPS",
    "R7_COHERENT_ACTOR_SCHEMA",
    "R7_COHERENT_ACTOR_SCOPE",
    "select_coherent_actor",
]
