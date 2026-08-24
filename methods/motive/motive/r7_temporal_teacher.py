"""R7 event-window temporal teacher primitives.

The numerical core in this module operates only on NumPy arrays.  In
particular, importing it does not import PyTorch or CoTracker.  The optional
``LazyCoTrackerAdapter`` imports those packages only when ``track`` is called.

The teacher is deliberately conservative.  It raises
``TemporalTeacherError`` instead of emitting a representation when tracking,
camera estimation, actor separation, or event localization is underdetermined.
This is a diagnostic geometry teacher, not an inference-time motion token.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


R7_TEMPORAL_TEACHER_SCHEMA = "motive-r7-event-temporal-teacher-v2"
R7_TRACK_SCHEMA = "motive-cotracker-observations-v1"
R7_PHASE_STEPS = 32


class TemporalTeacherError(ValueError):
    """A fail-closed data-quality or contract error."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = str(reason)
        super().__init__(f"{self.reason}: {message}")


def _fail(reason: str, message: str) -> None:
    raise TemporalTeacherError(reason, message)


def _finite_array(
    values: Any,
    *,
    name: str,
    ndim: int | None = None,
) -> np.ndarray:
    array = np.asarray(values)
    if ndim is not None and array.ndim != ndim:
        _fail("invalid_shape", f"{name} must have {ndim} dimensions")
    if not np.issubdtype(array.dtype, np.number):
        _fail("invalid_dtype", f"{name} must be numeric")
    if not np.isfinite(array).all():
        _fail("non_finite_input", f"{name} contains NaN or infinity")
    return array


@dataclass(frozen=True)
class TemporalTeacherConfig:
    """Fixed-schema and quality thresholds for the R7-P0 teacher."""

    phase_steps: int = R7_PHASE_STEPS
    minimum_frames: int = 8
    minimum_tracks: int = 16
    minimum_camera_tracks: int = 8
    minimum_actor_tracks: int = 2
    output_actor_tracks: int = 8
    visibility_threshold: float = 0.5
    minimum_track_visibility: float = 0.55
    minimum_event_track_visibility: float = 0.50
    camera_trim_fraction: float = 0.70
    camera_irls_iterations: int = 6
    camera_inlier_threshold: float = 0.004
    minimum_camera_inlier_fraction: float = 0.50
    minimum_actor_speed: float = 0.008
    actor_mad_multiplier: float = 2.5
    maximum_actor_fraction: float = 0.50
    minimum_event_transitions: int = 3
    event_energy_fraction: float = 0.85
    event_padding_transitions: int = 1
    perturb_track_drop_fraction: float = 0.10
    perturb_visibility_drop_fraction: float = 0.03
    perturb_time_jitter_fraction: float = 0.08
    perturb_coordinate_jitter: float = 0.00025
    stability_event_iou_threshold: float = 0.70
    stability_embedding_cosine_threshold: float = 0.85
    stability_duration_relative_error_threshold: float = 0.10
    stability_embedding_norm_relative_error_threshold: float = 0.10
    stability_trajectory_rmse_threshold: float = 0.01
    eps: float = 1e-8

    def validate(self) -> None:
        integer_minima = {
            "minimum_frames": (self.minimum_frames, 4),
            "minimum_tracks": (self.minimum_tracks, 3),
            "minimum_camera_tracks": (self.minimum_camera_tracks, 3),
            "minimum_actor_tracks": (self.minimum_actor_tracks, 1),
            "output_actor_tracks": (self.output_actor_tracks, 1),
            "minimum_event_transitions": (
                self.minimum_event_transitions,
                2,
            ),
        }
        if self.phase_steps != R7_PHASE_STEPS:
            _fail(
                "invalid_config",
                f"phase_steps is schema-fixed at {R7_PHASE_STEPS}",
            )
        for name, (value, minimum) in integer_minima.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                _fail("invalid_config", f"{name} must be >= {minimum}")
        unit_intervals = (
            "visibility_threshold",
            "minimum_track_visibility",
            "minimum_event_track_visibility",
            "camera_trim_fraction",
            "minimum_camera_inlier_fraction",
            "maximum_actor_fraction",
            "event_energy_fraction",
            "perturb_track_drop_fraction",
            "perturb_visibility_drop_fraction",
            "stability_event_iou_threshold",
            "stability_embedding_cosine_threshold",
            "stability_duration_relative_error_threshold",
            "stability_embedding_norm_relative_error_threshold",
        )
        for name in unit_intervals:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                _fail("invalid_config", f"{name} must be in [0,1]")
        if self.camera_trim_fraction <= 0.5:
            _fail(
                "invalid_config",
                "camera_trim_fraction must preserve a background majority",
            )
        if not 0.0 < self.maximum_actor_fraction < 1.0:
            _fail("invalid_config", "maximum_actor_fraction must be in (0,1)")
        positive_values = (
            "camera_inlier_threshold",
            "minimum_actor_speed",
            "actor_mad_multiplier",
            "eps",
        )
        for name in positive_values:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                _fail("invalid_config", f"{name} must be finite and positive")
        if (
            not math.isfinite(self.perturb_time_jitter_fraction)
            or not 0.0 <= self.perturb_time_jitter_fraction < 0.5
        ):
            _fail(
                "invalid_config",
                "perturb_time_jitter_fraction must be in [0,0.5)",
            )
        if (
            not math.isfinite(self.perturb_coordinate_jitter)
            or self.perturb_coordinate_jitter < 0.0
        ):
            _fail(
                "invalid_config",
                "perturb_coordinate_jitter must be finite and nonnegative",
            )
        if (
            not math.isfinite(self.stability_trajectory_rmse_threshold)
            or self.stability_trajectory_rmse_threshold <= 0.0
        ):
            _fail(
                "invalid_config",
                "stability_trajectory_rmse_threshold must be positive",
            )


@dataclass(frozen=True)
class TrackObservations:
    """Validated point tracks returned by a tracker.

    ``tracks`` are pixel coordinates with shape ``[T,N,2]``.  Visibility is a
    probability in ``[0,1]`` with shape ``[T,N]``.
    """

    tracks: np.ndarray
    visibility: np.ndarray
    frame_times: np.ndarray
    frame_size: tuple[int, int]
    backend: str
    provenance: Mapping[str, Any]
    schema_version: str = R7_TRACK_SCHEMA

    @classmethod
    def create(
        cls,
        *,
        tracks: Any,
        visibility: Any,
        frame_times: Any,
        frame_size: tuple[int, int],
        backend: str,
        provenance: Mapping[str, Any] | None = None,
    ) -> "TrackObservations":
        track_array = _finite_array(tracks, name="tracks", ndim=3).astype(
            np.float32,
            copy=False,
        )
        if track_array.shape[-1] != 2:
            _fail("invalid_shape", "tracks must have shape [T,N,2]")
        visibility_array = _finite_array(
            visibility,
            name="visibility",
            ndim=2,
        ).astype(np.float32, copy=False)
        if visibility_array.shape != track_array.shape[:2]:
            _fail(
                "invalid_shape",
                "visibility shape must equal tracks.shape[:2]",
            )
        if bool(
            ((visibility_array < 0.0) | (visibility_array > 1.0)).any()
        ):
            _fail("invalid_visibility", "visibility must be in [0,1]")
        time_array = _finite_array(
            frame_times,
            name="frame_times",
            ndim=1,
        ).astype(np.float64, copy=False)
        if len(time_array) != len(track_array):
            _fail(
                "invalid_shape",
                "frame_times length must equal the track frame count",
            )
        if len(time_array) < 2 or bool((np.diff(time_array) <= 0.0).any()):
            _fail(
                "invalid_frame_times",
                "frame_times must be strictly increasing",
            )
        if (
            not isinstance(frame_size, tuple)
            or len(frame_size) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in frame_size
            )
        ):
            _fail(
                "invalid_frame_size",
                "frame_size must be a positive integer (height,width) tuple",
            )
        backend_value = str(backend).strip()
        if not backend_value:
            _fail("missing_provenance", "tracker backend is empty")
        provenance_value = dict(provenance or {})
        return cls(
            tracks=np.ascontiguousarray(track_array),
            visibility=np.ascontiguousarray(visibility_array),
            frame_times=np.ascontiguousarray(time_array),
            frame_size=frame_size,
            backend=backend_value,
            provenance=provenance_value,
        )


@dataclass(frozen=True)
class CameraCompensation:
    """Robust per-transition camera estimate and stabilized point tracks."""

    normalized_tracks: np.ndarray
    stabilized_tracks: np.ndarray
    transition_affines: np.ndarray
    cumulative_affines: np.ndarray
    transition_inlier_fraction: np.ndarray
    transition_valid_counts: np.ndarray
    raw_background_median: float
    residual_background_median: float
    background_residual_reduction: float
    camera_explained_ratio: float
    crossfit_valid: bool
    crossfit_raw_median: float
    crossfit_residual_median: float
    crossfit_residual_reduction: float


@dataclass(frozen=True)
class EventWindow:
    """A half-open transition interval and its corresponding frame interval."""

    transition_start: int
    transition_stop: int
    frame_start: int
    frame_stop: int
    start_time: float
    end_time: float
    duration: float
    normalized_start: float
    normalized_end: float
    captured_energy_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalTeacher:
    """Fixed-shape event representation emitted by the array core."""

    event_window: EventWindow
    actor_trajectory: np.ndarray
    actor_velocity: np.ndarray
    actor_acceleration: np.ndarray
    actor_track_trajectories: np.ndarray
    actor_track_mask: np.ndarray
    camera_trajectory: np.ndarray
    phase_visibility: np.ndarray
    phase_uncertainty: np.ndarray
    phase_energy: np.ndarray
    active_track_indices: np.ndarray
    active_track_scores: np.ndarray
    event_duration: float
    mean_visibility: float
    background_residual_reduction: float
    camera_explained_ratio: float
    camera_inlier_fraction: float
    camera_crossfit_valid: bool
    camera_crossfit_raw_median: float
    camera_crossfit_residual_median: float
    camera_crossfit_residual_reduction: float
    schema_version: str = R7_TEMPORAL_TEACHER_SCHEMA

    def embedding(self) -> np.ndarray:
        """Return the actor/event stability embedding (camera kept separate)."""

        duration = max(float(self.event_duration), 1e-8)
        # Position, phase-normalized velocity/acceleration, and normalized
        # energy retain trajectory shape without making small time jitter look
        # like a different action.  Finite differences amplify sub-pixel
        # tracker jitter, so derivative blocks have fixed schema weights.  The
        # unscaled arrays remain available for speed/acceleration probes.
        energy = np.asarray(self.phase_energy, dtype=np.float64)
        energy_scale = float(np.linalg.norm(energy))
        if energy_scale > 1e-12:
            energy = energy / energy_scale
        vector = np.concatenate(
            (
                np.asarray(self.actor_trajectory, dtype=np.float64).reshape(-1),
                (
                    0.25
                    * np.asarray(self.actor_velocity, dtype=np.float64)
                    * duration
                ).reshape(-1),
                (
                    0.02
                    * np.asarray(self.actor_acceleration, dtype=np.float64)
                    * duration
                    * duration
                ).reshape(-1),
                (0.10 * energy).reshape(-1),
            )
        )
        if not np.isfinite(vector).all() or float(np.linalg.norm(vector)) <= 1e-12:
            _fail(
                "degenerate_embedding",
                "temporal teacher embedding is zero or non-finite",
            )
        return vector.astype(np.float32)


@dataclass(frozen=True)
class PerturbationStability:
    event_window_iou: float
    embedding_cosine: float
    event_duration_relative_error: float
    embedding_norm_relative_error: float
    trajectory_rmse: float
    base_active_tracks: int
    perturbed_active_tracks: int
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StableTemporalTeacher:
    base: TemporalTeacher
    perturbed: TemporalTeacher | None
    stability: PerturbationStability | None
    diagnostic_ready: bool
    failure_reason: str | None
    audit_perturbed: TemporalTeacher | None = None
    audit_stability: PerturbationStability | None = None
    audit_available: bool = False
    audit_passed: bool = False
    audit_failure_reason: str | None = None


def _normalize_tracks(
    tracks: np.ndarray,
    frame_size: tuple[int, int],
) -> np.ndarray:
    height, width = frame_size
    scale = np.asarray([width, height], dtype=np.float64)
    normalized = np.asarray(tracks, dtype=np.float64) / scale
    if not np.isfinite(normalized).all():
        _fail("non_finite_input", "normalized tracks are non-finite")
    # Tracker extrapolation slightly outside the image is expected.  Extreme
    # coordinates normally indicate a coordinate-system mismatch.
    extreme = np.any((normalized < -2.0) | (normalized > 3.0), axis=-1)
    if float(np.mean(extreme)) > 0.01:
        _fail(
            "coordinate_system_mismatch",
            "more than 1% of tracks are far outside the frame",
        )
    return normalized


def _apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def _solve_affine(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    rows = len(source)
    design = np.zeros((rows * 2, 6), dtype=np.float64)
    outcome = np.empty(rows * 2, dtype=np.float64)
    design[0::2, 0] = source[:, 0]
    design[0::2, 1] = source[:, 1]
    design[0::2, 2] = 1.0
    design[1::2, 3] = source[:, 0]
    design[1::2, 4] = source[:, 1]
    design[1::2, 5] = 1.0
    outcome[0::2] = target[:, 0]
    outcome[1::2] = target[:, 1]
    root_weight = np.repeat(np.sqrt(np.maximum(weights, 0.0)), 2)
    weighted_design = design * root_weight[:, None]
    weighted_outcome = outcome * root_weight
    solution, _, rank, singular = np.linalg.lstsq(
        weighted_design,
        weighted_outcome,
        rcond=None,
    )
    if rank < 6 or len(singular) < 6 or singular[-1] <= eps:
        _fail(
            "degenerate_camera_geometry",
            "visible tracks do not constrain an affine camera model",
        )
    matrix = np.asarray(
        [
            [solution[0], solution[1], solution[2]],
            [solution[3], solution[4], solution[5]],
        ],
        dtype=np.float64,
    )
    determinant = float(np.linalg.det(matrix[:, :2]))
    if (
        not np.isfinite(matrix).all()
        or abs(determinant) <= eps
        or abs(determinant) > 4.0
    ):
        _fail(
            "invalid_camera_model",
            "estimated affine camera transform is singular or implausible",
        )
    return matrix


def _robust_affine(
    source: np.ndarray,
    target: np.ndarray,
    config: TemporalTeacherConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(source)
    if count < config.minimum_camera_tracks:
        _fail(
            "insufficient_camera_tracks",
            f"need {config.minimum_camera_tracks}, received {count}",
        )
    weights = np.ones(count, dtype=np.float64)
    matrix = _solve_affine(source, target, weights, eps=config.eps)
    keep_count = max(
        config.minimum_camera_tracks,
        int(math.ceil(config.camera_trim_fraction * count)),
    )
    for _ in range(config.camera_irls_iterations):
        residual = np.linalg.norm(
            target - _apply_affine(matrix, source),
            axis=1,
        )
        order = np.argsort(residual, kind="stable")
        keep = order[:keep_count]
        core = residual[keep]
        median = float(np.median(core))
        mad = float(np.median(np.abs(core - median)))
        scale = max(1.4826 * mad, config.camera_inlier_threshold / 3.0)
        robust = 1.0 / (1.0 + (residual / (2.5 * scale)) ** 2)
        weights = np.zeros(count, dtype=np.float64)
        weights[keep] = robust[keep]
        matrix = _solve_affine(
            source,
            target,
            weights,
            eps=config.eps,
        )
    residual = np.linalg.norm(
        target - _apply_affine(matrix, source),
        axis=1,
    )
    order = np.argsort(residual, kind="stable")
    core = residual[order[:keep_count]]
    median = float(np.median(core))
    mad = float(np.median(np.abs(core - median)))
    threshold = max(
        config.camera_inlier_threshold,
        median + 3.0 * max(1.4826 * mad, config.eps),
    )
    inliers = residual <= threshold
    if int(np.sum(inliers)) < config.minimum_camera_tracks:
        _fail(
            "insufficient_camera_inliers",
            "robust affine camera model has too few inliers",
        )
    # A final inlier-only fit removes the small leverage retained by IRLS.
    matrix = _solve_affine(
        source,
        target,
        inliers.astype(np.float64),
        eps=config.eps,
    )
    residual = np.linalg.norm(
        target - _apply_affine(matrix, source),
        axis=1,
    )
    return matrix, inliers, residual


def _crossfit_camera_residuals(
    normalized_tracks: np.ndarray,
    visibility: np.ndarray,
    config: TemporalTeacherConfig,
) -> tuple[float, float, float]:
    """Evaluate camera residuals on tracks excluded from affine fitting.

    Global track index parity defines two deterministic folds.  For each
    transition and fold, the affine is fitted only on the opposite fold and
    evaluated only on the held-out fold.  No held-out coordinate or residual
    participates in fitting, robust trimming, or inlier selection.
    """

    frame_count, track_count, _ = normalized_tracks.shape
    track_indices = np.arange(track_count, dtype=np.int64)
    raw_values: list[np.ndarray] = []
    residual_values: list[np.ndarray] = []
    for transition in range(frame_count - 1):
        visible = (
            (visibility[transition] >= config.visibility_threshold)
            & (visibility[transition + 1] >= config.visibility_threshold)
        )
        for eval_fold in (0, 1):
            evaluate = visible & ((track_indices % 2) == eval_fold)
            fit = visible & ~((track_indices % 2) == eval_fold)
            fit_indices = np.flatnonzero(fit)
            eval_indices = np.flatnonzero(evaluate)
            if (
                len(fit_indices) < config.minimum_camera_tracks
                or len(eval_indices) < config.minimum_camera_tracks
            ):
                _fail(
                    "insufficient_crossfit_camera_tracks",
                    (
                        f"transition {transition} fold {eval_fold} has "
                        f"fit={len(fit_indices)}, eval={len(eval_indices)}"
                    ),
                )
            fit_source = normalized_tracks[transition, fit_indices]
            fit_target = normalized_tracks[transition + 1, fit_indices]
            matrix, fit_inliers, _ = _robust_affine(
                fit_source,
                fit_target,
                config,
            )
            if float(np.mean(fit_inliers)) < config.minimum_camera_inlier_fraction:
                _fail(
                    "low_crossfit_camera_inlier_fraction",
                    f"transition {transition} fold {eval_fold}",
                )
            eval_source = normalized_tracks[transition, eval_indices]
            eval_target = normalized_tracks[transition + 1, eval_indices]
            raw_values.append(
                np.linalg.norm(eval_target - eval_source, axis=1)
            )
            residual_values.append(
                np.linalg.norm(
                    eval_target - _apply_affine(matrix, eval_source),
                    axis=1,
                )
            )
    raw_median = float(np.median(np.concatenate(raw_values)))
    residual_median = float(np.median(np.concatenate(residual_values)))
    # A static camera has no removable camera signal.  Reporting a perfect
    # reduction for 0/0 would be a false positive, so its reduction is zero.
    reduction = (
        0.0
        if raw_median <= config.eps
        else float(
            np.clip(
                1.0 - residual_median / raw_median,
                -1.0,
                1.0,
            )
        )
    )
    return raw_median, residual_median, reduction


def robust_camera_compensation(
    tracks: Any,
    visibility: Any,
    frame_size: tuple[int, int],
    *,
    config: TemporalTeacherConfig | None = None,
) -> CameraCompensation:
    """Estimate robust affine camera motion and stabilize point tracks."""

    cfg = config or TemporalTeacherConfig()
    cfg.validate()
    track_array = _finite_array(tracks, name="tracks", ndim=3).astype(
        np.float64,
        copy=False,
    )
    if track_array.shape[-1] != 2:
        _fail("invalid_shape", "tracks must have shape [T,N,2]")
    visibility_array = _finite_array(
        visibility,
        name="visibility",
        ndim=2,
    ).astype(np.float64, copy=False)
    if visibility_array.shape != track_array.shape[:2]:
        _fail("invalid_shape", "visibility must have shape [T,N]")
    if bool(((visibility_array < 0.0) | (visibility_array > 1.0)).any()):
        _fail("invalid_visibility", "visibility must be in [0,1]")
    frame_count, track_count, _ = track_array.shape
    if frame_count < cfg.minimum_frames:
        _fail(
            "insufficient_frames",
            f"need {cfg.minimum_frames}, received {frame_count}",
        )
    if track_count < cfg.minimum_tracks:
        _fail(
            "insufficient_tracks",
            f"need {cfg.minimum_tracks}, received {track_count}",
        )
    normalized = _normalize_tracks(track_array, frame_size)
    transition_affines: list[np.ndarray] = []
    cumulative = [np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])]
    inlier_fractions: list[float] = []
    valid_counts: list[int] = []
    raw_background: list[np.ndarray] = []
    residual_background: list[np.ndarray] = []
    raw_square_sum = 0.0
    residual_square_sum = 0.0
    for index in range(frame_count - 1):
        valid = (
            (visibility_array[index] >= cfg.visibility_threshold)
            & (visibility_array[index + 1] >= cfg.visibility_threshold)
        )
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) < cfg.minimum_camera_tracks:
            _fail(
                "insufficient_camera_tracks",
                f"transition {index} has only {len(valid_indices)} visible tracks",
            )
        source = normalized[index, valid_indices]
        target = normalized[index + 1, valid_indices]
        matrix, inliers, residual = _robust_affine(source, target, cfg)
        inlier_fraction = float(np.mean(inliers))
        if inlier_fraction < cfg.minimum_camera_inlier_fraction:
            _fail(
                "low_camera_inlier_fraction",
                f"transition {index} inlier fraction={inlier_fraction:.4f}",
            )
        transition_affines.append(matrix)
        valid_counts.append(len(valid_indices))
        inlier_fractions.append(inlier_fraction)
        background_indices = np.flatnonzero(inliers)
        raw = np.linalg.norm(
            target[background_indices] - source[background_indices],
            axis=1,
        )
        camera_residual = residual[background_indices]
        raw_background.append(raw)
        residual_background.append(camera_residual)
        raw_square_sum += float(np.sum(raw * raw))
        residual_square_sum += float(np.sum(camera_residual * camera_residual))

        previous_h = np.vstack(
            (cumulative[-1], np.asarray([0.0, 0.0, 1.0]))
        )
        step_h = np.vstack((matrix, np.asarray([0.0, 0.0, 1.0])))
        composed = step_h @ previous_h
        if (
            not np.isfinite(composed).all()
            or abs(float(np.linalg.det(composed[:2, :2]))) <= cfg.eps
        ):
            _fail(
                "invalid_camera_model",
                f"cumulative camera transform failed at transition {index}",
            )
        cumulative.append(composed[:2])

    cumulative_array = np.stack(cumulative)
    stabilized = np.empty_like(normalized)
    for index, matrix in enumerate(cumulative_array):
        homogeneous = np.vstack((matrix, np.asarray([0.0, 0.0, 1.0])))
        inverse = np.linalg.inv(homogeneous)
        points_h = np.concatenate(
            (
                normalized[index],
                np.ones((track_count, 1), dtype=np.float64),
            ),
            axis=1,
        )
        stabilized[index] = (points_h @ inverse.T)[:, :2]
    if not np.isfinite(stabilized).all():
        _fail("invalid_camera_model", "stabilized tracks are non-finite")

    try:
        (
            crossfit_raw_median,
            crossfit_residual_median,
            crossfit_reduction,
        ) = _crossfit_camera_residuals(
            normalized,
            visibility_array,
            cfg,
        )
        crossfit_valid = True
    except TemporalTeacherError:
        # Cross-fit is an independent audit, not an excuse to discard an
        # otherwise valid base teacher.  Invalid audit metrics are explicit
        # and are excluded from the camera criterion, whose minimum sample
        # count then fails closed.
        crossfit_valid = False
        crossfit_raw_median = 0.0
        crossfit_residual_median = 0.0
        crossfit_reduction = 0.0

    raw_values = np.concatenate(raw_background)
    residual_values = np.concatenate(residual_background)
    raw_median = float(np.median(raw_values))
    residual_median = float(np.median(residual_values))
    if raw_median <= cfg.eps:
        reduction = 1.0 if residual_median <= cfg.eps else 0.0
    else:
        reduction = float(
            np.clip(1.0 - residual_median / raw_median, -1.0, 1.0)
        )
    explained = (
        1.0
        if raw_square_sum <= cfg.eps and residual_square_sum <= cfg.eps
        else float(
            np.clip(
                1.0 - residual_square_sum / (raw_square_sum + cfg.eps),
                -1.0,
                1.0,
            )
        )
    )
    return CameraCompensation(
        normalized_tracks=normalized.astype(np.float32),
        stabilized_tracks=stabilized.astype(np.float32),
        transition_affines=np.stack(transition_affines).astype(np.float32),
        cumulative_affines=cumulative_array.astype(np.float32),
        transition_inlier_fraction=np.asarray(
            inlier_fractions,
            dtype=np.float32,
        ),
        transition_valid_counts=np.asarray(valid_counts, dtype=np.int32),
        raw_background_median=raw_median,
        residual_background_median=residual_median,
        background_residual_reduction=reduction,
        camera_explained_ratio=explained,
        crossfit_valid=crossfit_valid,
        crossfit_raw_median=crossfit_raw_median,
        crossfit_residual_median=crossfit_residual_median,
        crossfit_residual_reduction=crossfit_reduction,
    )


def select_event_window(
    transition_energy: Any,
    frame_times: Any,
    *,
    config: TemporalTeacherConfig | None = None,
) -> EventWindow:
    """Select the shortest continuous window capturing configured energy."""

    cfg = config or TemporalTeacherConfig()
    cfg.validate()
    energy = _finite_array(
        transition_energy,
        name="transition_energy",
        ndim=1,
    ).astype(np.float64, copy=False)
    times = _finite_array(
        frame_times,
        name="frame_times",
        ndim=1,
    ).astype(np.float64, copy=False)
    if len(times) != len(energy) + 1:
        _fail(
            "invalid_shape",
            "frame_times must have one more entry than transition_energy",
        )
    if bool((np.diff(times) <= 0.0).any()):
        _fail("invalid_frame_times", "frame_times must strictly increase")
    if bool((energy < 0.0).any()):
        _fail("invalid_energy", "transition energy must be nonnegative")
    if len(energy) < cfg.minimum_event_transitions:
        _fail("insufficient_frames", "too few transitions for an event")
    if float(np.max(energy)) <= cfg.eps:
        _fail("no_actor_event", "transition energy is zero")

    # Mild smoothing reduces one-transition tracker spikes without moving the
    # event boundary materially.
    padded = np.pad(energy, (1, 1), mode="edge")
    smooth = (
        0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]
    )
    floor = float(np.quantile(smooth, 0.20))
    signal = np.maximum(smooth - floor, 0.0)
    if float(np.sum(signal)) <= cfg.eps:
        signal = smooth.copy()
    total = float(np.sum(signal))
    target = cfg.event_energy_fraction * total
    best: tuple[int, float, int, int] | None = None
    prefix = np.concatenate(([0.0], np.cumsum(signal)))
    transition_count = len(signal)
    for start in range(transition_count):
        for stop in range(
            start + cfg.minimum_event_transitions,
            transition_count + 1,
        ):
            captured = float(prefix[stop] - prefix[start])
            if captured + cfg.eps < target:
                continue
            length = stop - start
            density = captured / length
            candidate = (length, -density, start, stop)
            if best is None or candidate < best:
                best = candidate
            break
    if best is None:
        _fail(
            "event_localization_failed",
            "no continuous event window captures the required energy",
        )
    _, _, start, stop = best
    start = max(0, start - cfg.event_padding_transitions)
    stop = min(
        transition_count,
        stop + cfg.event_padding_transitions,
    )
    clip_duration = float(times[-1] - times[0])
    duration = float(times[stop] - times[start])
    if duration <= cfg.eps or clip_duration <= cfg.eps:
        _fail("invalid_event_duration", "event duration is non-positive")
    return EventWindow(
        transition_start=start,
        transition_stop=stop,
        frame_start=start,
        frame_stop=stop + 1,
        start_time=float(times[start]),
        end_time=float(times[stop]),
        duration=duration,
        normalized_start=float((times[start] - times[0]) / clip_duration),
        normalized_end=float((times[stop] - times[0]) / clip_duration),
        captured_energy_fraction=float(
            np.sum(signal[start:stop]) / max(total, cfg.eps)
        ),
    )


def _interp_vector(
    source_times: np.ndarray,
    values: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    output = np.empty((len(target_times), values.shape[1]), dtype=np.float64)
    for dimension in range(values.shape[1]):
        output[:, dimension] = np.interp(
            target_times,
            source_times,
            values[:, dimension],
        )
    return output


def _camera_parameters(
    cumulative_affines: np.ndarray,
) -> np.ndarray:
    center = np.asarray([0.5, 0.5], dtype=np.float64)
    parameters = []
    for matrix in cumulative_affines:
        projected = _apply_affine(matrix, center[None])[0]
        linear = np.asarray(matrix[:, :2], dtype=np.float64)
        determinant = float(np.linalg.det(linear))
        if determinant <= 0.0:
            _fail(
                "invalid_camera_model",
                "camera affine contains a reflection or zero scale",
            )
        angle = math.atan2(float(linear[1, 0]), float(linear[0, 0]))
        log_scale = 0.5 * math.log(max(determinant, 1e-12))
        parameters.append(
            [
                projected[0] - center[0],
                projected[1] - center[1],
                angle,
                log_scale,
            ]
        )
    return np.asarray(parameters, dtype=np.float64)


def build_temporal_teacher(
    tracks: Any,
    visibility: Any,
    frame_times: Any,
    frame_size: tuple[int, int],
    *,
    config: TemporalTeacherConfig | None = None,
) -> TemporalTeacher:
    """Build one fail-closed, camera-compensated event teacher."""

    cfg = config or TemporalTeacherConfig()
    cfg.validate()
    observations = TrackObservations.create(
        tracks=tracks,
        visibility=visibility,
        frame_times=frame_times,
        frame_size=frame_size,
        backend="array-core",
    )
    if len(observations.tracks) < cfg.minimum_frames:
        _fail(
            "insufficient_frames",
            f"need {cfg.minimum_frames} frames",
        )
    if observations.tracks.shape[1] < cfg.minimum_tracks:
        _fail(
            "insufficient_tracks",
            f"need {cfg.minimum_tracks} tracks",
        )
    compensation = robust_camera_compensation(
        observations.tracks,
        observations.visibility,
        observations.frame_size,
        config=cfg,
    )
    visible = observations.visibility >= cfg.visibility_threshold
    transition_visible = visible[:-1] & visible[1:]
    dt = np.diff(observations.frame_times)
    residual_step = np.diff(
        compensation.stabilized_tracks.astype(np.float64),
        axis=0,
    )
    residual_speed = np.linalg.norm(residual_step, axis=-1) / dt[:, None]
    residual_speed[~transition_visible] = np.nan
    track_visibility = np.mean(visible, axis=0)
    eligible = track_visibility >= cfg.minimum_track_visibility
    track_scores = np.zeros(observations.tracks.shape[1], dtype=np.float64)
    for track_index in np.flatnonzero(eligible):
        values = residual_speed[:, track_index]
        values = values[np.isfinite(values)]
        if len(values):
            # A whole-clip upper quantile finds short actions without using a
            # single tracker spike as the score.  q75 would erase actions
            # occupying less than one quarter of a clip, the failure mode R7
            # is specifically intended to fix.
            track_scores[track_index] = float(np.quantile(values, 0.90))
    eligible_scores = track_scores[eligible]
    if len(eligible_scores) < cfg.minimum_camera_tracks:
        _fail(
            "insufficient_visible_tracks",
            "too few tracks meet the clip visibility threshold",
        )
    median = float(np.median(eligible_scores))
    mad = float(np.median(np.abs(eligible_scores - median)))
    actor_threshold = max(
        cfg.minimum_actor_speed,
        median + cfg.actor_mad_multiplier * max(1.4826 * mad, cfg.eps),
    )
    active = np.flatnonzero(eligible & (track_scores > actor_threshold))
    if len(active) < cfg.minimum_actor_tracks:
        _fail(
            "no_actor_tracks",
            (
                f"only {len(active)} tracks exceed actor threshold "
                f"{actor_threshold:.6g}"
            ),
        )
    if len(active) / max(int(np.sum(eligible)), 1) > cfg.maximum_actor_fraction:
        _fail(
            "actor_background_not_separable",
            "active tracks are not a minority; camera estimate is ambiguous",
        )

    transition_energy = np.zeros(len(dt), dtype=np.float64)
    for index in range(len(dt)):
        values = residual_speed[index, active]
        values = values[np.isfinite(values)]
        if len(values) >= cfg.minimum_actor_tracks:
            transition_energy[index] = float(np.median(values))
    event = select_event_window(
        transition_energy,
        observations.frame_times,
        config=cfg,
    )
    event_frames = slice(event.frame_start, event.frame_stop)
    event_visibility = np.mean(visible[event_frames, active], axis=0)
    event_active = active[
        event_visibility >= cfg.minimum_event_track_visibility
    ]
    if len(event_active) < cfg.minimum_actor_tracks:
        _fail(
            "insufficient_event_visibility",
            "too few active tracks remain visible through the event",
        )
    event_active = np.asarray(
        sorted(
            event_active.tolist(),
            key=lambda index: (-track_scores[index], index),
        ),
        dtype=np.int64,
    )
    phase_times = np.linspace(
        event.start_time,
        event.end_time,
        cfg.phase_steps,
        dtype=np.float64,
    )
    per_track: list[np.ndarray] = []
    per_track_visibility: list[np.ndarray] = []
    for track_index in event_active:
        keep = (
            visible[event_frames, track_index]
            & np.isfinite(
                compensation.stabilized_tracks[
                    event_frames,
                    track_index,
                ]
            ).all(axis=1)
        )
        source_times = observations.frame_times[event_frames][keep]
        source_values = compensation.stabilized_tracks[
            event_frames,
            track_index,
        ][keep].astype(np.float64)
        if len(source_times) < 2:
            continue
        trajectory = _interp_vector(source_times, source_values, phase_times)
        trajectory -= trajectory[0]
        per_track.append(trajectory)
        visible_values = observations.visibility[
            event_frames,
            track_index,
        ].astype(np.float64)
        per_track_visibility.append(
            np.interp(
                phase_times,
                observations.frame_times[event_frames],
                visible_values,
            )
        )
    if len(per_track) < cfg.minimum_actor_tracks:
        _fail(
            "insufficient_event_visibility",
            "active trajectories cannot be interpolated through the event",
        )
    track_trajectories = np.stack(per_track)
    aggregate = np.median(track_trajectories, axis=0)
    phase_visibility = np.mean(np.stack(per_track_visibility), axis=0)
    phase_uncertainty = np.median(
        np.linalg.norm(track_trajectories - aggregate[None], axis=-1),
        axis=0,
    )
    edge_order = 2 if cfg.phase_steps >= 3 else 1
    velocity = np.gradient(
        aggregate,
        phase_times,
        axis=0,
        edge_order=edge_order,
    )
    acceleration = np.gradient(
        velocity,
        phase_times,
        axis=0,
        edge_order=edge_order,
    )
    transition_midtimes = (
        observations.frame_times[:-1] + observations.frame_times[1:]
    ) / 2.0
    phase_energy = np.interp(
        phase_times,
        transition_midtimes,
        transition_energy,
    )
    camera_frames = _camera_parameters(
        compensation.cumulative_affines.astype(np.float64)
    )
    camera_trajectory = _interp_vector(
        observations.frame_times,
        camera_frames,
        phase_times,
    )
    camera_trajectory -= camera_trajectory[0]
    padded_tracks = np.zeros(
        (cfg.output_actor_tracks, cfg.phase_steps, 2),
        dtype=np.float32,
    )
    track_mask = np.zeros(cfg.output_actor_tracks, dtype=bool)
    stored = min(cfg.output_actor_tracks, len(track_trajectories))
    padded_tracks[:stored] = track_trajectories[:stored].astype(np.float32)
    track_mask[:stored] = True
    output_arrays = (
        aggregate,
        velocity,
        acceleration,
        camera_trajectory,
        phase_visibility,
        phase_uncertainty,
        phase_energy,
    )
    if not all(np.isfinite(value).all() for value in output_arrays):
        _fail("non_finite_output", "temporal teacher output is non-finite")
    teacher = TemporalTeacher(
        event_window=event,
        actor_trajectory=aggregate.astype(np.float32),
        actor_velocity=velocity.astype(np.float32),
        actor_acceleration=acceleration.astype(np.float32),
        actor_track_trajectories=padded_tracks,
        actor_track_mask=track_mask,
        camera_trajectory=camera_trajectory.astype(np.float32),
        phase_visibility=phase_visibility.astype(np.float32),
        phase_uncertainty=phase_uncertainty.astype(np.float32),
        phase_energy=phase_energy.astype(np.float32),
        active_track_indices=event_active,
        active_track_scores=track_scores[event_active].astype(np.float32),
        event_duration=event.duration,
        mean_visibility=float(np.mean(phase_visibility)),
        background_residual_reduction=(
            compensation.background_residual_reduction
        ),
        camera_explained_ratio=compensation.camera_explained_ratio,
        camera_inlier_fraction=float(
            np.median(compensation.transition_inlier_fraction)
        ),
        camera_crossfit_valid=compensation.crossfit_valid,
        camera_crossfit_raw_median=compensation.crossfit_raw_median,
        camera_crossfit_residual_median=(
            compensation.crossfit_residual_median
        ),
        camera_crossfit_residual_reduction=(
            compensation.crossfit_residual_reduction
        ),
    )
    # This final check also rejects a non-moving numerical artifact.
    teacher.embedding()
    return teacher


def deterministic_track_time_perturbation(
    tracks: Any,
    visibility: Any,
    frame_times: Any,
    frame_size: tuple[int, int],
    *,
    seed: int,
    config: TemporalTeacherConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply deterministic track dropout, visibility dropout, and time jitter."""

    cfg = config or TemporalTeacherConfig()
    cfg.validate()
    observations = TrackObservations.create(
        tracks=tracks,
        visibility=visibility,
        frame_times=frame_times,
        frame_size=frame_size,
        backend="array-perturbation",
    )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        _fail("invalid_seed", "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    perturbed_tracks = observations.tracks.astype(np.float64).copy()
    perturbed_visibility = observations.visibility.copy()
    track_count = perturbed_tracks.shape[1]
    drop_count = int(math.floor(cfg.perturb_track_drop_fraction * track_count))
    if drop_count:
        dropped = rng.choice(track_count, size=drop_count, replace=False)
        perturbed_visibility[:, dropped] = 0.0
    remaining_visible = np.argwhere(
        perturbed_visibility >= cfg.visibility_threshold
    )
    visibility_drop_count = int(
        math.floor(
            cfg.perturb_visibility_drop_fraction * len(remaining_visible)
        )
    )
    if visibility_drop_count:
        selected = rng.choice(
            len(remaining_visible),
            size=visibility_drop_count,
            replace=False,
        )
        coordinates = remaining_visible[selected]
        perturbed_visibility[coordinates[:, 0], coordinates[:, 1]] = 0.0
    if cfg.perturb_coordinate_jitter > 0.0:
        height, width = frame_size
        normalized_noise = rng.normal(
            0.0,
            cfg.perturb_coordinate_jitter,
            size=perturbed_tracks.shape,
        )
        perturbed_tracks += normalized_noise * np.asarray([width, height])
    perturbed_times = observations.frame_times.astype(np.float64).copy()
    intervals = np.diff(perturbed_times)
    if len(perturbed_times) > 2 and cfg.perturb_time_jitter_fraction > 0.0:
        local_scale = np.minimum(intervals[:-1], intervals[1:])
        jitter = rng.uniform(
            -cfg.perturb_time_jitter_fraction,
            cfg.perturb_time_jitter_fraction,
            size=len(perturbed_times) - 2,
        )
        perturbed_times[1:-1] += jitter * local_scale
    if bool((np.diff(perturbed_times) <= 0.0).any()):
        _fail(
            "invalid_perturbation",
            "time perturbation broke strict ordering",
        )
    return (
        perturbed_tracks.astype(np.float32),
        perturbed_visibility.astype(np.float32),
        perturbed_times,
    )


def event_window_iou(first: EventWindow, second: EventWindow) -> float:
    """Temporal IoU in normalized clip time."""

    intersection = max(
        0.0,
        min(first.normalized_end, second.normalized_end)
        - max(first.normalized_start, second.normalized_start),
    )
    union = max(
        first.normalized_end,
        second.normalized_end,
    ) - min(first.normalized_start, second.normalized_start)
    return float(intersection / union) if union > 0.0 else 0.0


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    first64 = np.asarray(first, dtype=np.float64).reshape(-1)
    second64 = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = float(
        np.linalg.norm(first64) * np.linalg.norm(second64)
    )
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(first64, second64) / denominator, -1.0, 1.0))


def _evaluate_teacher_perturbation(
    base: TemporalTeacher,
    tracks: Any,
    visibility: Any,
    frame_times: Any,
    frame_size: tuple[int, int],
    *,
    seed: int,
    config: TemporalTeacherConfig,
) -> tuple[
    TemporalTeacher | None,
    PerturbationStability | None,
    str | None,
]:
    """Evaluate one downstream perturbation independently of other seeds."""

    try:
        perturbed_tracks, perturbed_visibility, perturbed_times = (
            deterministic_track_time_perturbation(
                tracks,
                visibility,
                frame_times,
                frame_size,
                seed=seed,
                config=config,
            )
        )
        perturbed = build_temporal_teacher(
            perturbed_tracks,
            perturbed_visibility,
            perturbed_times,
            frame_size,
            config=config,
        )
    except TemporalTeacherError as error:
        return None, None, error.reason
    overlap = event_window_iou(base.event_window, perturbed.event_window)
    base_embedding = base.embedding().astype(np.float64)
    perturbed_embedding = perturbed.embedding().astype(np.float64)
    cosine = _cosine(base_embedding, perturbed_embedding)
    duration_error = abs(
        perturbed.event_duration - base.event_duration
    ) / max(base.event_duration, config.eps)
    base_norm = float(np.linalg.norm(base_embedding))
    perturbed_norm = float(np.linalg.norm(perturbed_embedding))
    norm_error = abs(perturbed_norm - base_norm) / max(
        base_norm,
        config.eps,
    )
    trajectory_rmse = float(
        np.sqrt(
            np.mean(
                (
                    perturbed.actor_trajectory.astype(np.float64)
                    - base.actor_trajectory.astype(np.float64)
                )
                ** 2
            )
        )
    )
    passed = (
        overlap >= config.stability_event_iou_threshold
        and cosine >= config.stability_embedding_cosine_threshold
        and duration_error
        <= config.stability_duration_relative_error_threshold
        and norm_error
        <= config.stability_embedding_norm_relative_error_threshold
        and trajectory_rmse <= config.stability_trajectory_rmse_threshold
    )
    stability = PerturbationStability(
        event_window_iou=overlap,
        embedding_cosine=cosine,
        event_duration_relative_error=float(duration_error),
        embedding_norm_relative_error=float(norm_error),
        trajectory_rmse=trajectory_rmse,
        base_active_tracks=len(base.active_track_indices),
        perturbed_active_tracks=len(perturbed.active_track_indices),
        passed=passed,
    )
    return perturbed, stability, None


def build_temporal_teacher_with_stability(
    tracks: Any,
    visibility: Any,
    frame_times: Any,
    frame_size: tuple[int, int],
    *,
    seed: int,
    audit_seed: int | None = None,
    config: TemporalTeacherConfig | None = None,
) -> StableTemporalTeacher:
    """Build a base teacher plus independent screening/audit perturbations.

    ``seed`` controls the screening perturbation and therefore
    ``diagnostic_ready``.  ``audit_seed``, when supplied, is evaluated
    independently even when screening fails; it never changes usability.
    Both perturbations operate downstream on the same tracker output.  They
    measure numerical teacher robustness, not visual re-tracking stability.
    """

    cfg = config or TemporalTeacherConfig()
    cfg.validate()
    for name, value in (("seed", seed), ("audit_seed", audit_seed)):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            _fail("invalid_seed", f"{name} must be a nonnegative integer")
    if audit_seed is not None and audit_seed == seed:
        _fail("invalid_seed", "audit_seed must differ from screening seed")
    base = build_temporal_teacher(
        tracks,
        visibility,
        frame_times,
        frame_size,
        config=cfg,
    )
    perturbed, stability, screening_error = _evaluate_teacher_perturbation(
        base,
        tracks,
        visibility,
        frame_times,
        frame_size,
        seed=seed,
        config=cfg,
    )
    audit_perturbed: TemporalTeacher | None = None
    audit_stability: PerturbationStability | None = None
    audit_error: str | None = None
    if audit_seed is not None:
        (
            audit_perturbed,
            audit_stability,
            audit_error,
        ) = _evaluate_teacher_perturbation(
            base,
            tracks,
            visibility,
            frame_times,
            frame_size,
            seed=audit_seed,
            config=cfg,
        )
    screening_passed = bool(stability is not None and stability.passed)
    audit_available = audit_stability is not None
    audit_passed = bool(audit_stability is not None and audit_stability.passed)
    return StableTemporalTeacher(
        base=base,
        perturbed=perturbed,
        stability=stability,
        diagnostic_ready=screening_passed,
        failure_reason=(
            None
            if screening_passed
            else screening_error or "unstable_teacher"
        ),
        audit_perturbed=audit_perturbed,
        audit_stability=audit_stability,
        audit_available=audit_available,
        audit_passed=audit_passed,
        audit_failure_reason=(
            None
            if audit_seed is None or audit_passed
            else audit_error or "unstable_teacher"
        ),
    )


class LazyCoTrackerAdapter:
    """Optional CoTracker adapter with no import-time torch dependency."""

    def __init__(
        self,
        *,
        checkpoint: str | Path | None = None,
        device: str = "cuda",
        grid_size: int = 10,
        backward_tracking: bool = False,
        predictor_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if (
            isinstance(grid_size, bool)
            or not isinstance(grid_size, int)
            or grid_size < 2
        ):
            _fail("invalid_tracker_config", "grid_size must be an integer >= 2")
        device_value = str(device).strip()
        if not device_value:
            _fail("invalid_tracker_config", "device is empty")
        self.checkpoint = None if checkpoint is None else Path(checkpoint)
        self.device = device_value
        self.grid_size = grid_size
        self.backward_tracking = bool(backward_tracking)
        self.predictor_kwargs = dict(predictor_kwargs or {})
        self._torch: Any | None = None
        self._predictor: Any | None = None

    @property
    def loaded(self) -> bool:
        return self._predictor is not None

    def _load(self) -> tuple[Any, Any]:
        if self._predictor is not None:
            return self._torch, self._predictor
        if self.checkpoint is not None and not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        try:
            torch_module = importlib.import_module("torch")
            predictor_module = importlib.import_module("cotracker.predictor")
            predictor_class = getattr(predictor_module, "CoTrackerPredictor")
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "CoTracker is optional; install torch and the CoTracker source "
                "package before calling LazyCoTrackerAdapter.track"
            ) from error
        kwargs = dict(self.predictor_kwargs)
        if self.checkpoint is not None:
            kwargs.setdefault("checkpoint", str(self.checkpoint))
        predictor = predictor_class(**kwargs)
        if hasattr(predictor, "to"):
            predictor = predictor.to(self.device)
        if hasattr(predictor, "eval"):
            predictor.eval()
        self._torch = torch_module
        self._predictor = predictor
        return torch_module, predictor

    def track(
        self,
        frames_rgb: Any,
        *,
        frame_times: Any | None = None,
    ) -> TrackObservations:
        """Track an RGB uint8/float array with shape ``[T,H,W,3]``."""

        frames = _finite_array(
            frames_rgb,
            name="frames_rgb",
            ndim=4,
        )
        if frames.shape[-1] != 3:
            _fail("invalid_shape", "frames_rgb must have shape [T,H,W,3]")
        if len(frames) < 2:
            _fail("insufficient_frames", "tracker input needs at least 2 frames")
        height, width = int(frames.shape[1]), int(frames.shape[2])
        if height < 2 or width < 2:
            _fail("invalid_frame_size", "tracker frames are too small")
        if np.issubdtype(frames.dtype, np.floating):
            minimum = float(np.min(frames))
            maximum = float(np.max(frames))
            if minimum < 0.0 or maximum > 255.0:
                _fail(
                    "invalid_pixel_range",
                    "floating frames must be in [0,255]",
                )
        torch_module, predictor = self._load()
        video = torch_module.from_numpy(
            np.ascontiguousarray(frames)
        ).permute(0, 3, 1, 2)[None].float().to(self.device)
        context = (
            torch_module.inference_mode()
            if hasattr(torch_module, "inference_mode")
            else torch_module.no_grad()
        )
        with context:
            result = predictor(
                video,
                grid_size=self.grid_size,
                grid_query_frame=0,
                backward_tracking=self.backward_tracking,
            )
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            _fail(
                "invalid_tracker_output",
                "CoTracker must return (tracks, visibility)",
            )
        track_tensor, visibility_tensor = result[:2]
        track_array = np.asarray(
            track_tensor.detach().cpu().numpy(),
            dtype=np.float32,
        )
        visibility_array = np.asarray(
            visibility_tensor.detach().cpu().numpy(),
            dtype=np.float32,
        )
        if track_array.ndim != 4 or track_array.shape[0] != 1:
            _fail(
                "invalid_tracker_output",
                "CoTracker tracks must have shape [1,T,N,2]",
            )
        if visibility_array.ndim == 4 and visibility_array.shape[-1] == 1:
            visibility_array = visibility_array[..., 0]
        if visibility_array.ndim != 3 or visibility_array.shape[0] != 1:
            _fail(
                "invalid_tracker_output",
                "CoTracker visibility must have shape [1,T,N]",
            )
        if frame_times is None:
            times = np.arange(len(frames), dtype=np.float64)
        else:
            times = frame_times
        checkpoint_value = (
            None
            if self.checkpoint is None
            else str(self.checkpoint.resolve())
        )
        return TrackObservations.create(
            tracks=track_array[0],
            visibility=np.clip(visibility_array[0], 0.0, 1.0),
            frame_times=times,
            frame_size=(height, width),
            backend="cotracker",
            provenance={
                "checkpoint": checkpoint_value,
                "grid_size": self.grid_size,
                "query_frame": 0,
                "backward_tracking": self.backward_tracking,
                "device": self.device,
            },
        )


__all__ = [
    "CameraCompensation",
    "EventWindow",
    "LazyCoTrackerAdapter",
    "PerturbationStability",
    "R7_PHASE_STEPS",
    "R7_TEMPORAL_TEACHER_SCHEMA",
    "R7_TRACK_SCHEMA",
    "StableTemporalTeacher",
    "TemporalTeacher",
    "TemporalTeacherConfig",
    "TemporalTeacherError",
    "TrackObservations",
    "build_temporal_teacher",
    "build_temporal_teacher_with_stability",
    "deterministic_track_time_perturbation",
    "event_window_iou",
    "robust_camera_compensation",
    "select_event_window",
]
