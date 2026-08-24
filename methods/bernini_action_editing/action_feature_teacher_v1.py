#!/usr/bin/env python3
"""Deterministic R7 geometry candidate bridge for Bernini action editing.

This module converts the *camera-compensated temporal geometry* emitted by
``motive.r7_temporal_teacher`` into the exact ``21 x 256`` phase-token plus
``256`` global-token geometry consumed by :class:`ActionPlanPredictorV1`.

The bridge is intentionally narrow:

* RGB pixels, VAE latents, appearance features, and renderer hidden states are
  not accepted;
* target teachers and self-generated-anchor teachers use the same frozen
  numerical transform;
* a target action is represented as ``target - source/no-op`` and an anchor
  action as ``anchor-action - anchor-no-op``;
* camera trajectories are validated and recorded but excluded from the action
  token projection;
* the projection is a code-defined, deterministic signed random lift.  It is not a
  learned teacher and cannot silently co-adapt with the student.

The output remains an unqualified candidate descriptor.  It is never an
inference input and must not be described as a qualified action encoder or as
evidence that a trained renderer follows it; that requires independent
teacher qualification plus the intervention and decoded-video gates in the
0817 contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "bernini-action-feature-teacher-v1"
SOURCE_TEACHER_SCHEMA = "motive-r7-event-temporal-teacher-v2"
PROJECTION_SCHEMA = "sha256-signed-random-lift-v1"
UPSTREAM_AUTHORITY_SCHEMA = "bernini-r7-geometry-input-authority-v1"
INPUT_PHASES = 32
OUTPUT_PHASES = 21
ACTION_WIDTH = 256
PHASE_FEATURES = 12
GLOBAL_FEATURES = PHASE_FEATURES * 3 + 1
PHASE_WEIGHTS = np.asarray(
    (
        1.0,
        1.0,
        0.25,
        0.25,
        0.02,
        0.02,
        0.10,
        0.50,
        0.50,
        0.50,
        0.50,
        0.25,
    ),
    dtype=np.float32,
)
PHASE_WEIGHTS.setflags(write=False)
_ROLES = frozenset(("target", "anchor"))
_AUTHORITY_ROLES = frozenset(
    ("target_action", "target_noop", "anchor_action", "anchor_noop")
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RECEIPT_KEYS = frozenset(
    (
        "schema_version",
        "role",
        "source_teacher_schema",
        "input_phases",
        "output_phases",
        "action_width",
        "phase_features",
        "global_features",
        "phase_weights",
        "projection",
        "action_embedding_sha256",
        "action_camera_sha256_audit_only",
        "action_upstream_authority_sha256",
        "baseline_mode",
        "baseline_embedding_sha256",
        "baseline_camera_sha256_audit_only",
        "baseline_upstream_authority_sha256",
        "action_event_duration",
        "action_event_normalized_start",
        "action_event_normalized_end",
        "baseline_event_duration",
        "baseline_event_normalized_start",
        "baseline_event_normalized_end",
        "delta_feature_sha256",
        "delta_feature_l2",
        "phase_tokens_sha256",
        "global_token_sha256",
        "camera_trajectory_excluded_from_tokens",
        "camera_invariance_claimed",
        "direct_rgb_or_latent_feature_input",
        "appearance_invariance_claimed",
        "actor_object_contact_geometry_in_tokens",
        "training_only_not_inference_input",
        "teacher_qualification_status",
        "point_distillation_authorized",
        "action_following_claimed",
        "receipt_sha256",
    )
)
_UPSTREAM_KEYS = frozenset(
    (
        "schema_version",
        "media_role",
        "content_group_id",
        "counterfactual_pair_sha256",
        "media_sha256",
        "media_size",
        "track_cache_sha256",
        "track_cache_size",
        "tracker_authority_sha256",
        "temporal_teacher_source_sha256",
        "temporal_teacher_config_sha256",
        "temporal_teacher_arrays_sha256",
        "stability_receipt_sha256",
        "actor_binding_sha256",
        "object_binding_sha256",
        "instruction_semantics_sha256",
        "temporal_teacher_present",
        "static_noop_verified",
        "camera_crossfit_valid",
        "perturbation_stability_passed",
        "full_video_quality_passed",
        "authority_sha256",
    )
)


class ActionFeatureTeacherError(ValueError):
    """Raised when a teacher cannot be converted without ambiguity."""


def _fail(message: str) -> None:
    raise ActionFeatureTeacherError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ActionFeatureTeacherError(
            "value is not finite canonical ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            _fail(f"temporal teacher is missing {name}")
        return value[name]
    if not hasattr(value, name):
        _fail(f"temporal teacher is missing {name}")
    return getattr(value, name)


def _array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != shape
        or array.dtype != np.dtype(np.float32)
        or not array.flags.c_contiguous
    ):
        _fail(f"{name} must have exact C-contiguous float32 shape {shape}")
    if not np.isfinite(array).all():
        _fail(f"{name} contains NaN or infinity")
    return array


def _float(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (float, int, np.floating, np.integer)
    ):
        _fail(f"{name} must be one finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        _fail(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def float32_sha256(value: np.ndarray) -> str:
    array = np.asarray(value, dtype="<f4")
    if not np.isfinite(array).all():
        _fail("cannot hash non-finite float32 tensor")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _signed_projection(rows: int, columns: int, *, namespace: str) -> np.ndarray:
    if type(rows) is not int or type(columns) is not int or rows <= 0 or columns <= 0:
        _fail("projection geometry must use positive exact integers")
    matrix = np.empty((rows, columns), dtype=np.float32)
    scale = np.float32(1.0 / math.sqrt(float(rows)))
    prefix = namespace.encode("ascii") + b"\0"
    for row in range(rows):
        row_bytes = row.to_bytes(4, "big", signed=False)
        for column in range(columns):
            digest = hashlib.sha256(
                prefix + row_bytes + column.to_bytes(4, "big", signed=False)
            ).digest()
            matrix[row, column] = scale if digest[0] & 1 else -scale
    return matrix


_PHASE_PROJECTION = _signed_projection(
    PHASE_FEATURES,
    ACTION_WIDTH,
    namespace="bernini-action-feature-teacher-v1/phase",
)
_GLOBAL_PROJECTION = _signed_projection(
    GLOBAL_FEATURES,
    ACTION_WIDTH,
    namespace="bernini-action-feature-teacher-v1/global",
)
_PHASE_PROJECTION.setflags(write=False)
_GLOBAL_PROJECTION.setflags(write=False)
PHASE_WEIGHTS_SHA256 = (
    "98a33be2a712d72441cb7a5740054deb9a4edf601870a14ac2791a07af9a52a0"
)
PHASE_PROJECTION_SHA256 = (
    "30ef308aefe27c77520e53c0a7e164a122f684c437c9214286508ebffdd2883a"
)
GLOBAL_PROJECTION_SHA256 = (
    "040d69b405bafaa637e845ee03c60d54c0c14e0017fb9f91b3816ab25a6b301d"
)


def _assert_transform_integrity() -> None:
    if (
        PHASE_WEIGHTS.flags.writeable
        or _PHASE_PROJECTION.flags.writeable
        or _GLOBAL_PROJECTION.flags.writeable
        or float32_sha256(PHASE_WEIGHTS) != PHASE_WEIGHTS_SHA256
        or float32_sha256(_PHASE_PROJECTION) != PHASE_PROJECTION_SHA256
        or float32_sha256(_GLOBAL_PROJECTION) != GLOBAL_PROJECTION_SHA256
    ):
        _fail("action feature transform bytes differ from pinned authority")


def _fixed_project(array: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """Project with a fixed reduction order, independent of BLAS/NumPy."""

    if array.ndim != 2 or projection.ndim != 2 or array.shape[1] != projection.shape[0]:
        _fail("action feature projection geometry differs")
    result = np.empty((array.shape[0], projection.shape[1]), dtype=np.float32)
    for row in range(array.shape[0]):
        for column in range(projection.shape[1]):
            result[row, column] = np.float32(
                math.fsum(
                    float(array[row, feature]) * float(projection[feature, column])
                    for feature in range(array.shape[1])
                )
            )
    return result


def _fixed_column_statistics(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Population mean/std with an explicit platform-stable reduction order."""

    if array.ndim != 2 or array.shape[0] <= 0:
        _fail("action feature statistics geometry differs")
    mean = np.empty(array.shape[1], dtype=np.float32)
    std = np.empty(array.shape[1], dtype=np.float32)
    count = float(array.shape[0])
    for column in range(array.shape[1]):
        values = [float(array[row, column]) for row in range(array.shape[0])]
        column_mean = math.fsum(values) / count
        column_variance = math.fsum(
            (value - column_mean) * (value - column_mean) for value in values
        ) / count
        mean[column] = np.float32(column_mean)
        std[column] = np.float32(math.sqrt(max(column_variance, 0.0)))
    return mean, std


def _fixed_track_moments(relative_tracks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median/std/p90 across at most eight tracks with fixed interpolation."""

    if relative_tracks.ndim != 3 or not 2 <= relative_tracks.shape[0] <= 8:
        _fail("relative actor-track geometry differs")
    count, phases, coordinates = relative_tracks.shape
    median = np.empty((phases, coordinates), dtype=np.float32)
    std = np.empty((phases, coordinates), dtype=np.float32)
    p90 = np.empty(phases, dtype=np.float32)
    for phase in range(phases):
        for coordinate in range(coordinates):
            values = sorted(
                float(relative_tracks[index, phase, coordinate])
                for index in range(count)
            )
            midpoint = count // 2
            median_value = (
                values[midpoint]
                if count % 2
                else math.fsum((values[midpoint - 1], values[midpoint])) / 2.0
            )
            mean_value = math.fsum(values) / float(count)
            variance = math.fsum(
                (value - mean_value) * (value - mean_value) for value in values
            ) / float(count)
            median[phase, coordinate] = np.float32(median_value)
            std[phase, coordinate] = np.float32(math.sqrt(max(variance, 0.0)))
        norms = sorted(
            math.hypot(
                float(relative_tracks[index, phase, 0]),
                float(relative_tracks[index, phase, 1]),
            )
            for index in range(count)
        )
        position = 0.90 * float(count - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        fraction = position - float(lower)
        p90[phase] = np.float32(
            math.fsum(
                (
                    norms[lower] * (1.0 - fraction),
                    norms[upper] * fraction,
                )
            )
        )
    return median, std, p90


_assert_transform_integrity()


def _absolute_phase_resample(
    array: np.ndarray,
    *,
    normalized_start: float,
    normalized_end: float,
) -> np.ndarray:
    """Place one event-local trajectory on the absolute 21-phase clip grid.

    State features (trajectory and local actor configuration) hold their final
    value after the event. Dynamic features (velocity, acceleration, energy)
    are zero outside the event. This retains onset and completion/hold timing
    instead of stretching every event over the full clip.
    """

    if array.ndim != 2 or array.shape[0] != INPUT_PHASES:
        _fail("phase feature input must use the exact 32-phase teacher grid")
    if (
        not math.isfinite(normalized_start)
        or not math.isfinite(normalized_end)
        or not 0.0 <= normalized_start < normalized_end <= 1.0
    ):
        _fail("event window must be one nonempty interval inside [0,1]")
    result = np.empty((OUTPUT_PHASES, array.shape[1]), dtype=np.float32)
    state_columns = frozenset((0, 1, 7, 8, 9, 10, 11))
    for output_phase in range(OUTPUT_PHASES):
        time_value = float(output_phase) / float(OUTPUT_PHASES - 1)
        for column in range(array.shape[1]):
            if time_value < normalized_start:
                value = 0.0
            elif time_value > normalized_end:
                value = float(array[-1, column]) if column in state_columns else 0.0
            else:
                position = (
                    (time_value - normalized_start)
                    / (normalized_end - normalized_start)
                    * float(INPUT_PHASES - 1)
                )
                lower = min(int(math.floor(position)), INPUT_PHASES - 1)
                upper = min(lower + 1, INPUT_PHASES - 1)
                fraction = position - float(lower)
                value = math.fsum(
                    (
                        float(array[lower, column]) * (1.0 - fraction),
                        float(array[upper, column]) * fraction,
                    )
                )
            result[output_phase, column] = np.float32(value)
    return result


@dataclass(frozen=True)
class TemporalActionCore:
    """Direct-RGB-free numerical core extracted from one R7 teacher."""

    phase_features: np.ndarray
    event_duration: float
    event_normalized_start: float
    event_normalized_end: float
    camera_sha256: str
    temporal_teacher_arrays_sha256: str
    source_embedding_sha256: str


def _event_value(event: Any, name: str) -> Any:
    return _field(event, name)


def _teacher_arrays_identity(
    *,
    trajectory: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    energy: np.ndarray,
    visibility: np.ndarray,
    uncertainty: np.ndarray,
    track_trajectories: np.ndarray,
    track_mask: np.ndarray,
    camera: np.ndarray,
    event_duration: float,
    event_start: float,
    event_end: float,
    camera_diagnostics: Mapping[str, Any],
) -> str:
    arrays = {
        "actor_trajectory": trajectory,
        "actor_velocity": velocity,
        "actor_acceleration": acceleration,
        "phase_energy": energy,
        "phase_visibility": visibility,
        "phase_uncertainty": uncertainty,
        "actor_track_trajectories": track_trajectories,
        "camera_trajectory": camera,
    }
    identity = {
        name: {
            "dtype": "float32",
            "shape": list(array.shape),
            "sha256": float32_sha256(array),
        }
        for name, array in sorted(arrays.items())
    }
    identity["actor_track_mask"] = {
        "dtype": "bool",
        "shape": [8],
        "sha256": hashlib.sha256(
            np.ascontiguousarray(track_mask, dtype=np.bool_).tobytes(order="C")
        ).hexdigest(),
    }
    identity["event"] = {
        "duration": event_duration,
        "normalized_start": event_start,
        "normalized_end": event_end,
    }
    identity["camera_diagnostics"] = dict(camera_diagnostics)
    return object_sha256(identity)


def validate_upstream_authority(
    authority: Mapping[str, Any],
    *,
    expected_role: str,
    expected_arrays_sha256: str | None,
) -> Mapping[str, Any]:
    """Validate one externally pinned media/tracker/R7 authority envelope."""

    if not isinstance(authority, Mapping) or set(authority) != _UPSTREAM_KEYS:
        _fail("upstream teacher authority schema differs")
    if expected_role not in _AUTHORITY_ROLES:
        _fail("internal expected upstream role differs")
    if (
        authority.get("schema_version") != UPSTREAM_AUTHORITY_SCHEMA
        or authority.get("media_role") != expected_role
    ):
        _fail("upstream teacher authority role differs")
    for name in (
        "content_group_id",
    ):
        if type(authority.get(name)) is not str or not authority[name]:
            _fail(f"upstream authority {name} differs")
    for name in ("media_size", "track_cache_size"):
        if type(authority.get(name)) is not int or authority[name] <= 0:
            _fail(f"upstream authority {name} differs")
    hashes = (
        "media_sha256",
        "counterfactual_pair_sha256",
        "track_cache_sha256",
        "tracker_authority_sha256",
        "temporal_teacher_source_sha256",
        "temporal_teacher_config_sha256",
        "stability_receipt_sha256",
        "actor_binding_sha256",
        "object_binding_sha256",
        "instruction_semantics_sha256",
        "authority_sha256",
    )
    if any(
        type(authority.get(name)) is not str
        or _SHA256.fullmatch(authority[name]) is None
        for name in hashes
    ):
        _fail("upstream authority SHA field differs")
    present = authority.get("temporal_teacher_present")
    static = authority.get("static_noop_verified")
    if type(present) is not bool or type(static) is not bool:
        _fail("upstream authority teacher/static flags differ")
    arrays_sha = authority.get("temporal_teacher_arrays_sha256")
    if present:
        if (
            static is not False
            or type(arrays_sha) is not str
            or _SHA256.fullmatch(arrays_sha) is None
            or arrays_sha != expected_arrays_sha256
        ):
            _fail("upstream authority temporal teacher binding differs")
        if authority.get("camera_crossfit_valid") is not True or authority.get(
            "perturbation_stability_passed"
        ) is not True:
            _fail("upstream temporal teacher stability/camera gate differs")
    else:
        if (
            expected_arrays_sha256 is not None
            or static is not True
            or arrays_sha is not None
            or authority.get("camera_crossfit_valid") is not False
            or authority.get("perturbation_stability_passed") is not False
        ):
            _fail("static-noop authority binding differs")
    if authority.get("full_video_quality_passed") is not True:
        _fail("upstream media has not passed full-video quality")
    raw = dict(authority)
    claimed = raw.pop("authority_sha256")
    if claimed != object_sha256(raw):
        _fail("upstream authority digest differs")
    return dict(authority)


def temporal_action_core(teacher: Any) -> TemporalActionCore:
    """Validate and convert one R7 temporal teacher to 12 phase features."""

    schema = _field(teacher, "schema_version")
    if schema != SOURCE_TEACHER_SCHEMA:
        _fail("temporal teacher schema differs")
    duration = _float(
        _field(teacher, "event_duration"),
        name="event_duration",
        positive=True,
    )
    event = _field(teacher, "event_window")
    event_start = _float(
        _event_value(event, "normalized_start"),
        name="event_window.normalized_start",
    )
    event_end = _float(
        _event_value(event, "normalized_end"),
        name="event_window.normalized_end",
    )
    if not 0.0 <= event_start < event_end <= 1.0:
        _fail("event window normalized interval differs")
    raw_trajectory = _array(
        _field(teacher, "actor_trajectory"),
        name="actor_trajectory",
        shape=(INPUT_PHASES, 2),
    )
    velocity = _array(
        _field(teacher, "actor_velocity"),
        name="actor_velocity",
        shape=(INPUT_PHASES, 2),
    )
    acceleration = _array(
        _field(teacher, "actor_acceleration"),
        name="actor_acceleration",
        shape=(INPUT_PHASES, 2),
    )
    energy = _array(
        _field(teacher, "phase_energy"),
        name="phase_energy",
        shape=(INPUT_PHASES,),
    )
    visibility = _array(
        _field(teacher, "phase_visibility"),
        name="phase_visibility",
        shape=(INPUT_PHASES,),
    )
    uncertainty = _array(
        _field(teacher, "phase_uncertainty"),
        name="phase_uncertainty",
        shape=(INPUT_PHASES,),
    )
    track_trajectories = _array(
        _field(teacher, "actor_track_trajectories"),
        name="actor_track_trajectories",
        shape=(8, INPUT_PHASES, 2),
    )
    track_mask_raw = np.asarray(_field(teacher, "actor_track_mask"))
    if (
        track_mask_raw.shape != (8,)
        or track_mask_raw.dtype != np.bool_
        or not track_mask_raw.flags.c_contiguous
    ):
        _fail("actor_track_mask must have exact C-contiguous bool shape (8,)")
    if int(track_mask_raw.sum()) < 2:
        _fail("action teacher requires at least two actor tracks")
    camera = _array(
        _field(teacher, "camera_trajectory"),
        name="camera_trajectory",
        shape=(INPUT_PHASES, 4),
    )
    camera_crossfit_valid = _field(teacher, "camera_crossfit_valid")
    if type(camera_crossfit_valid) is not bool or camera_crossfit_valid is not True:
        _fail("temporal teacher camera cross-fit gate differs")
    camera_diagnostics = {
        "background_residual_reduction": _float(
            _field(teacher, "background_residual_reduction"),
            name="background_residual_reduction",
        ),
        "camera_explained_ratio": _float(
            _field(teacher, "camera_explained_ratio"),
            name="camera_explained_ratio",
        ),
        "camera_inlier_fraction": _float(
            _field(teacher, "camera_inlier_fraction"),
            name="camera_inlier_fraction",
        ),
        "camera_crossfit_valid": camera_crossfit_valid,
        "camera_crossfit_raw_median": _float(
            _field(teacher, "camera_crossfit_raw_median"),
            name="camera_crossfit_raw_median",
        ),
        "camera_crossfit_residual_median": _float(
            _field(teacher, "camera_crossfit_residual_median"),
            name="camera_crossfit_residual_median",
        ),
        "camera_crossfit_residual_reduction": _float(
            _field(teacher, "camera_crossfit_residual_reduction"),
            name="camera_crossfit_residual_reduction",
        ),
    }
    if not 0.0 <= camera_diagnostics["camera_inlier_fraction"] <= 1.0:
        _fail("temporal teacher camera inlier fraction differs")
    if bool((energy < 0.0).any()):
        _fail("phase_energy must be nonnegative")
    if bool(((visibility < 0.0) | (visibility > 1.0)).any()):
        _fail("phase_visibility must lie in [0,1]")
    if bool((uncertainty < 0.0).any()):
        _fail("phase_uncertainty must be nonnegative")
    energy_norm = math.sqrt(
        math.fsum(float(value) * float(value) for value in energy.flat)
    )
    if energy_norm <= 1.0e-12:
        _fail("temporal teacher has zero action energy")

    # R7 defines velocity and acceleration as derivatives on the evenly-spaced
    # event-time grid.  Recompute them from the supplied trajectory so a
    # self-consistent authority envelope cannot bind unrelated derivative
    # arrays.  Tolerances cover only the upstream float64 -> float32 cast.
    phase_times = np.linspace(0.0, duration, INPUT_PHASES, dtype=np.float64)
    expected_velocity = np.gradient(
        raw_trajectory.astype(np.float64),
        phase_times,
        axis=0,
        edge_order=2,
    )
    expected_acceleration = np.gradient(
        expected_velocity,
        phase_times,
        axis=0,
        edge_order=2,
    )
    if not np.allclose(
        velocity.astype(np.float64),
        expected_velocity,
        rtol=5.0e-4,
        atol=5.0e-5,
    ):
        _fail("actor_velocity is inconsistent with actor_trajectory")
    if not np.allclose(
        acceleration.astype(np.float64),
        expected_acceleration,
        rtol=2.0e-3,
        atol=5.0e-4,
    ):
        _fail("actor_acceleration is inconsistent with actor_velocity")

    # Translation is irrelevant. Physical velocity/acceleration and raw
    # normalized-frame energy retain speed/amplitude; confidence is audited
    # upstream but is deliberately not encoded as a shortcut feature.
    trajectory = raw_trajectory - raw_trajectory[:1]
    active_tracks = track_trajectories[track_mask_raw]
    relative_tracks = active_tracks - trajectory[None, :, :]
    relative_median, relative_std, relative_p90 = _fixed_track_moments(
        relative_tracks
    )
    feature = np.concatenate(
        (
            trajectory,
            velocity,
            acceleration,
            energy[:, None],
            relative_median,
            relative_std,
            relative_p90[:, None],
        ),
        axis=1,
    )
    feature = _absolute_phase_resample(
        feature,
        normalized_start=event_start,
        normalized_end=event_end,
    ) * PHASE_WEIGHTS[None, :]
    feature_norm = math.sqrt(
        math.fsum(float(value) * float(value) for value in feature.flat)
    )
    if not np.isfinite(feature).all() or feature_norm <= 1.0e-12:
        _fail("temporal teacher produces a zero/non-finite action core")
    source_embedding = np.concatenate(
        (
            trajectory.reshape(-1),
            velocity.reshape(-1),
            acceleration.reshape(-1),
            energy.reshape(-1),
            relative_median.reshape(-1),
            relative_std.reshape(-1),
            relative_p90.reshape(-1),
        )
    )
    arrays_sha256 = _teacher_arrays_identity(
        trajectory=raw_trajectory,
        velocity=velocity,
        acceleration=acceleration,
        energy=energy,
        visibility=visibility,
        uncertainty=uncertainty,
        track_trajectories=track_trajectories,
        track_mask=track_mask_raw,
        camera=camera,
        event_duration=duration,
        event_start=event_start,
        event_end=event_end,
        camera_diagnostics=camera_diagnostics,
    )
    return TemporalActionCore(
        phase_features=np.ascontiguousarray(feature, dtype=np.float32),
        event_duration=duration,
        event_normalized_start=event_start,
        event_normalized_end=event_end,
        camera_sha256=float32_sha256(camera),
        temporal_teacher_arrays_sha256=arrays_sha256,
        source_embedding_sha256=float32_sha256(source_embedding),
    )


@dataclass(frozen=True)
class ActionFeatureTokens:
    """Frozen teacher tokens plus their canonical receipt."""

    phase_tokens: np.ndarray
    global_token: np.ndarray
    receipt: Mapping[str, Any]

    def validate(self) -> None:
        _assert_transform_integrity()
        phase = _array(
            self.phase_tokens,
            name="phase_tokens",
            shape=(OUTPUT_PHASES, ACTION_WIDTH),
        )
        global_token = _array(
            self.global_token,
            name="global_token",
            shape=(ACTION_WIDTH,),
        )
        if not isinstance(self.receipt, Mapping):
            _fail("action feature receipt must be one mapping")
        if set(self.receipt) != _RECEIPT_KEYS:
            _fail("action feature receipt schema differs")
        if (
            self.receipt.get("schema_version") != SCHEMA_VERSION
            or self.receipt.get("role") not in _ROLES
            or self.receipt.get("source_teacher_schema") != SOURCE_TEACHER_SCHEMA
            or type(self.receipt.get("input_phases")) is not int
            or self.receipt.get("input_phases") != INPUT_PHASES
            or type(self.receipt.get("output_phases")) is not int
            or self.receipt.get("output_phases") != OUTPUT_PHASES
            or type(self.receipt.get("action_width")) is not int
            or self.receipt.get("action_width") != ACTION_WIDTH
            or type(self.receipt.get("phase_features")) is not int
            or self.receipt.get("phase_features") != PHASE_FEATURES
            or type(self.receipt.get("global_features")) is not int
            or self.receipt.get("global_features") != GLOBAL_FEATURES
        ):
            _fail("action feature receipt fixed geometry differs")
        weights = self.receipt.get("phase_weights")
        if (
            not isinstance(weights, list)
            or len(weights) != PHASE_FEATURES
            or any(type(value) is not float for value in weights)
            or weights != [float(value) for value in PHASE_WEIGHTS]
        ):
            _fail("action feature receipt phase weights differ")
        projection = self.receipt.get("projection")
        if (
            not isinstance(projection, Mapping)
            or set(projection) != {"schema", "phase_sha256", "global_sha256"}
            or projection.get("schema") != PROJECTION_SCHEMA
            or projection.get("phase_sha256") != PHASE_PROJECTION_SHA256
            or projection.get("global_sha256") != GLOBAL_PROJECTION_SHA256
        ):
            _fail("action feature projection authority differs")
        required_hashes = (
            "action_embedding_sha256",
            "action_camera_sha256_audit_only",
            "action_upstream_authority_sha256",
            "delta_feature_sha256",
            "phase_tokens_sha256",
            "global_token_sha256",
            "receipt_sha256",
        )
        if any(
            type(self.receipt.get(name)) is not str
            or _SHA256.fullmatch(self.receipt[name]) is None
            for name in required_hashes
        ):
            _fail("action feature receipt SHA field differs")
        mode = self.receipt.get("baseline_mode")
        baseline_hashes = (
            self.receipt.get("baseline_embedding_sha256"),
            self.receipt.get("baseline_camera_sha256_audit_only"),
        )
        baseline_authority_sha256 = self.receipt.get(
            "baseline_upstream_authority_sha256"
        )
        if (
            type(baseline_authority_sha256) is not str
            or _SHA256.fullmatch(baseline_authority_sha256) is None
        ):
            _fail("baseline upstream authority SHA differs")
        if mode == "externally_verified_static_noop":
            if baseline_hashes != (None, None):
                _fail("static-noop receipt must not invent baseline teacher hashes")
        elif mode == "explicit_temporal_teacher":
            if any(
                type(value) is not str or _SHA256.fullmatch(value) is None
                for value in baseline_hashes
            ):
                _fail("explicit baseline teacher hashes differ")
        else:
            _fail("action feature baseline mode differs")
        for name in ("action_event_duration", "baseline_event_duration", "delta_feature_l2"):
            value = self.receipt.get(name)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                _fail(f"action feature receipt {name} differs")
        action_start = self.receipt.get("action_event_normalized_start")
        action_end = self.receipt.get("action_event_normalized_end")
        if (
            type(action_start) is not float
            or type(action_end) is not float
            or not 0.0 <= action_start < action_end <= 1.0
        ):
            _fail("action event-window receipt differs")
        baseline_start = self.receipt.get("baseline_event_normalized_start")
        baseline_end = self.receipt.get("baseline_event_normalized_end")
        if mode == "externally_verified_static_noop":
            if baseline_start is not None or baseline_end is not None:
                _fail("static-noop receipt must not invent an event window")
        elif (
            type(baseline_start) is not float
            or type(baseline_end) is not float
            or not 0.0 <= baseline_start < baseline_end <= 1.0
        ):
            _fail("baseline event-window receipt differs")
        exact_flags = {
            "camera_trajectory_excluded_from_tokens": True,
            "camera_invariance_claimed": False,
            "direct_rgb_or_latent_feature_input": False,
            "appearance_invariance_claimed": False,
            "actor_object_contact_geometry_in_tokens": False,
            "training_only_not_inference_input": True,
            "point_distillation_authorized": False,
            "action_following_claimed": False,
        }
        if any(self.receipt.get(name) is not expected for name, expected in exact_flags.items()):
            _fail("action feature receipt semantic flags differ")
        if self.receipt.get("teacher_qualification_status") != "candidate_unqualified":
            _fail("candidate teacher may not self-authorize qualification")
        if self.receipt.get("phase_tokens_sha256") != float32_sha256(phase):
            _fail("phase token digest differs")
        if self.receipt.get("global_token_sha256") != float32_sha256(global_token):
            _fail("global token digest differs")
        receipt = dict(self.receipt)
        claimed = receipt.pop("receipt_sha256", None)
        if claimed != object_sha256(receipt):
            _fail("action feature receipt digest differs")


def build_action_feature_tokens(
    action_teacher: Any,
    *,
    role: str,
    action_authority: Mapping[str, Any],
    baseline_authority: Mapping[str, Any],
    baseline_teacher: Any | None = None,
) -> ActionFeatureTokens:
    """Build target ``q_y`` or anchor ``q_anchor`` with one frozen transform.

    A missing baseline is legal only when an external, content-bound no-op
    authority has established that the baseline contains no event. Callers
    must additionally pin both authority digests in the row-level manifest;
    self-hashes do not constitute launch or training authority.
    """

    _assert_transform_integrity()
    if type(role) is not str or role not in _ROLES:
        _fail("role must be exactly target or anchor")
    action = temporal_action_core(action_teacher)
    action_role = f"{role}_action"
    noop_role = f"{role}_noop"
    checked_action_authority = validate_upstream_authority(
        action_authority,
        expected_role=action_role,
        expected_arrays_sha256=action.temporal_teacher_arrays_sha256,
    )
    if baseline_teacher is None:
        checked_baseline_authority = validate_upstream_authority(
            baseline_authority,
            expected_role=noop_role,
            expected_arrays_sha256=None,
        )
        baseline_features = np.zeros_like(action.phase_features)
        baseline_duration = action.event_duration
        baseline_start = None
        baseline_end = None
        baseline_mode = "externally_verified_static_noop"
        baseline_embedding_sha256 = None
        baseline_camera_sha256 = None
    else:
        baseline = temporal_action_core(baseline_teacher)
        checked_baseline_authority = validate_upstream_authority(
            baseline_authority,
            expected_role=noop_role,
            expected_arrays_sha256=baseline.temporal_teacher_arrays_sha256,
        )
        baseline_features = baseline.phase_features
        baseline_duration = baseline.event_duration
        baseline_start = baseline.event_normalized_start
        baseline_end = baseline.event_normalized_end
        baseline_mode = "explicit_temporal_teacher"
        baseline_embedding_sha256 = baseline.source_embedding_sha256
        baseline_camera_sha256 = baseline.camera_sha256

    for name in (
        "content_group_id",
        "counterfactual_pair_sha256",
        "actor_binding_sha256",
        "object_binding_sha256",
        "instruction_semantics_sha256",
        "tracker_authority_sha256",
        "temporal_teacher_source_sha256",
        "temporal_teacher_config_sha256",
    ):
        if checked_action_authority[name] != checked_baseline_authority[name]:
            _fail(f"action/no-op counterfactual {name} differs")
    for name in ("media_sha256", "track_cache_sha256"):
        if checked_action_authority[name] == checked_baseline_authority[name]:
            _fail(f"action/no-op counterfactual {name} aliases")

    delta = np.ascontiguousarray(
        action.phase_features - baseline_features,
        dtype=np.float32,
    )
    delta_norm = math.sqrt(
        math.fsum(float(value) * float(value) for value in delta.flat)
    )
    if not math.isfinite(delta_norm) or delta_norm <= 1.0e-8:
        _fail("action-minus-baseline teacher is degenerate")
    delta_mean, delta_std = _fixed_column_statistics(delta)
    global_features = np.concatenate(
        (
            delta_mean,
            delta_std,
            delta[-1],
            np.asarray([action.event_duration], dtype=np.float32),
        )
    ).astype(np.float32)
    if global_features.shape != (GLOBAL_FEATURES,):
        _fail("global teacher feature geometry differs")
    phase_tokens = _fixed_project(delta, _PHASE_PROJECTION)
    global_token = _fixed_project(
        global_features.reshape(1, -1), _GLOBAL_PROJECTION
    )[0].copy()
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "role": role,
        "source_teacher_schema": SOURCE_TEACHER_SCHEMA,
        "input_phases": INPUT_PHASES,
        "output_phases": OUTPUT_PHASES,
        "action_width": ACTION_WIDTH,
        "phase_features": PHASE_FEATURES,
        "global_features": GLOBAL_FEATURES,
        "phase_weights": [float(value) for value in PHASE_WEIGHTS],
        "projection": {
            "schema": PROJECTION_SCHEMA,
            "phase_sha256": PHASE_PROJECTION_SHA256,
            "global_sha256": GLOBAL_PROJECTION_SHA256,
        },
        "action_embedding_sha256": action.source_embedding_sha256,
        "action_camera_sha256_audit_only": action.camera_sha256,
        "action_upstream_authority_sha256": checked_action_authority[
            "authority_sha256"
        ],
        "baseline_mode": baseline_mode,
        "baseline_embedding_sha256": baseline_embedding_sha256,
        "baseline_camera_sha256_audit_only": baseline_camera_sha256,
        "baseline_upstream_authority_sha256": checked_baseline_authority[
            "authority_sha256"
        ],
        "action_event_duration": action.event_duration,
        "action_event_normalized_start": action.event_normalized_start,
        "action_event_normalized_end": action.event_normalized_end,
        "baseline_event_duration": baseline_duration,
        "baseline_event_normalized_start": baseline_start,
        "baseline_event_normalized_end": baseline_end,
        "delta_feature_sha256": float32_sha256(delta),
        "delta_feature_l2": delta_norm,
        "phase_tokens_sha256": float32_sha256(phase_tokens),
        "global_token_sha256": float32_sha256(global_token),
        "camera_trajectory_excluded_from_tokens": True,
        "camera_invariance_claimed": False,
        "direct_rgb_or_latent_feature_input": False,
        "appearance_invariance_claimed": False,
        "actor_object_contact_geometry_in_tokens": False,
        "training_only_not_inference_input": True,
        "teacher_qualification_status": "candidate_unqualified",
        "point_distillation_authorized": False,
        "action_following_claimed": False,
    }
    receipt["receipt_sha256"] = object_sha256(receipt)
    result = ActionFeatureTokens(
        phase_tokens=phase_tokens,
        global_token=global_token,
        receipt=receipt,
    )
    result.validate()
    return result


def token_cosine(left: ActionFeatureTokens, right: ActionFeatureTokens) -> float:
    """Cosine over phase and global tokens for diagnostics and compatibility."""

    left.validate()
    right.validate()
    a = np.concatenate((left.phase_tokens.reshape(-1), left.global_token)).astype(
        np.float64
    )
    b = np.concatenate((right.phase_tokens.reshape(-1), right.global_token)).astype(
        np.float64
    )
    a_norm = math.sqrt(math.fsum(float(value) * float(value) for value in a))
    b_norm = math.sqrt(math.fsum(float(value) * float(value) for value in b))
    denominator = a_norm * b_norm
    if denominator <= 1.0e-12:
        _fail("cannot compare zero action-feature tokens")
    numerator = math.fsum(float(left) * float(right) for left, right in zip(a, b))
    return numerator / denominator


__all__ = [
    "ACTION_WIDTH",
    "ActionFeatureTeacherError",
    "ActionFeatureTokens",
    "GLOBAL_PROJECTION_SHA256",
    "INPUT_PHASES",
    "OUTPUT_PHASES",
    "PHASE_PROJECTION_SHA256",
    "SCHEMA_VERSION",
    "SOURCE_TEACHER_SCHEMA",
    "UPSTREAM_AUTHORITY_SCHEMA",
    "build_action_feature_tokens",
    "canonical_json_bytes",
    "float32_sha256",
    "object_sha256",
    "temporal_action_core",
    "token_cosine",
    "validate_upstream_authority",
]
