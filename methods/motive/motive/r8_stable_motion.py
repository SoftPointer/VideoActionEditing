"""Development-only stable set motion representation for R8.

R7 exposes a coherent component as per-track phase trajectories, a phase mask,
and phase energy.  Taking a median of *absolute* positions before subtracting
the initial median can turn changing track membership into apparent motion:
different static tracks carry different shape offsets.

This independent prototype removes that failure mode in two steps:

1. estimate every adjacent-phase displacement only from tracks visible at
   both endpoints, using a deterministic robust set center;
2. integrate those robust displacements from zero;
3. retain first-valid re-anchored per-track trajectories only as evidence.

The output retains the exact component membership, phase mask, and upstream
phase energy.  It is a geometric development representation only.  It cannot
decide whether the component is a semantic actor and must not be used as a
production teacher or generation authorization.

The set center is the unique minimizer of a fixed-delta pseudo-Huber spatial
objective in caller coordinates.  A proven global or local position
certificate is the only success condition; a small optimizer step alone never
authorizes output.  This is an explicitly smoothed robust center, not an exact
geometric median.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np


R8_STABLE_MOTION_SCHEMA = "motive-r8-stable-motion-dev-v6"
R8_STABLE_MOTION_PHASE_STEPS = 32
R8_STABLE_MOTION_SHAPE_DIM = 8
R8_SMOOTHED_CENTER_DEFAULT_DELTA = 1e-4
R8_SMOOTHED_CENTER_DEFAULT_ABSOLUTE_POSITION_TOLERANCE = 1e-7
R8_SMOOTHED_CENTER_MIN_NORMALIZED_VALUE = 1e-12
R8_SMOOTHED_CENTER_MAX_NORMALIZED_VALUE = 1.0
R8_SMOOTHED_CENTER_MAX_ITERATIONS = 4096
R8_SMOOTHED_CENTER_ARMIJO_FACTOR = 1e-4
R8_SMOOTHED_CENTER_MAX_BACKTRACKS = 64
R8_STABLE_MOTION_SCOPE = (
    "development-only stable geometric motion representation; preserves an "
    "upstream component but does not identify a semantic actor"
)
R8_SHAPE_TOKEN_FIELDS = (
    "transition_residual_median_radius",
    "transition_residual_radius_mad",
    "transition_residual_radius_q75",
    "transition_residual_radius_q90",
    "transition_residual_robust_covariance_xx",
    "transition_residual_robust_covariance_xy",
    "transition_residual_robust_covariance_yy",
    "transition_residual_robust_covariance_anisotropy",
)


class _StableMotionFailure(ValueError):
    """Internal fail-closed control flow for expected invalid inputs."""

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = str(reason)
        self.detail = str(detail)
        super().__init__(f"{self.reason}: {self.detail}")


def _fail(reason: str, detail: str) -> None:
    raise _StableMotionFailure(reason, detail)


@dataclass(frozen=True)
class StableMotionConfig:
    """Data-independent validity and robust aggregation settings."""

    minimum_tracks: int = 3
    minimum_track_valid_phases: int = 2
    minimum_phase_tracks: int = 2
    minimum_phase_track_fraction: float = 0.50
    minimum_transition_tracks: int = 2
    minimum_transition_track_fraction: float = 0.50
    smoothed_center_max_iterations: int = 64
    smoothed_center_delta: float = R8_SMOOTHED_CENTER_DEFAULT_DELTA
    smoothed_center_absolute_position_tolerance: float = (
        R8_SMOOTHED_CENTER_DEFAULT_ABSOLUTE_POSITION_TOLERANCE
    )
    covariance_clip_mad_scale: float = 3.0
    eps: float = 1e-8

    def validate(self) -> None:
        integer_values = (
            ("minimum_tracks", self.minimum_tracks, 2),
            (
                "minimum_track_valid_phases",
                self.minimum_track_valid_phases,
                2,
            ),
            ("minimum_phase_tracks", self.minimum_phase_tracks, 1),
            (
                "minimum_transition_tracks",
                self.minimum_transition_tracks,
                1,
            ),
            (
                "smoothed_center_max_iterations",
                self.smoothed_center_max_iterations,
                8,
            ),
        )
        for name, value, minimum in integer_values:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                _fail("invalid_config", f"{name} must be >= {minimum}")
        if (
            self.smoothed_center_max_iterations
            > R8_SMOOTHED_CENTER_MAX_ITERATIONS
        ):
            _fail(
                "invalid_config",
                "smoothed_center_max_iterations must be <= "
                f"{R8_SMOOTHED_CENTER_MAX_ITERATIONS}",
            )
        if self.minimum_phase_tracks > self.minimum_tracks:
            _fail(
                "invalid_config",
                "minimum_phase_tracks cannot exceed minimum_tracks",
            )
        if self.minimum_transition_tracks > self.minimum_tracks:
            _fail(
                "invalid_config",
                "minimum_transition_tracks cannot exceed minimum_tracks",
            )
        if self.minimum_track_valid_phases > R8_STABLE_MOTION_PHASE_STEPS:
            _fail(
                "invalid_config",
                "minimum_track_valid_phases exceeds the fixed phase count",
            )
        for name in (
            "minimum_phase_track_fraction",
            "minimum_transition_track_fraction",
        ):
            raw_fraction = getattr(self, name)
            if isinstance(raw_fraction, bool) or not isinstance(
                raw_fraction,
                Real,
            ):
                _fail(
                    "invalid_config",
                    f"{name} must be in (0,1]",
                )
            try:
                fraction = float(raw_fraction)
            except (TypeError, ValueError, OverflowError):
                _fail(
                    "invalid_config",
                    f"{name} must be in (0,1]",
                )
            if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
                _fail(
                    "invalid_config",
                    f"{name} must be in (0,1]",
                )
        for name in (
            "smoothed_center_delta",
            "smoothed_center_absolute_position_tolerance",
            "covariance_clip_mad_scale",
            "eps",
        ):
            raw_value = getattr(self, name)
            if isinstance(raw_value, bool) or not isinstance(
                raw_value,
                Real,
            ):
                _fail(
                    "invalid_config",
                    f"{name} must be finite and positive",
                )
            try:
                value = float(raw_value)
            except (TypeError, ValueError, OverflowError):
                _fail(
                    "invalid_config",
                    f"{name} must be finite and positive",
                )
            if not math.isfinite(value) or value <= 0.0:
                _fail(
                    "invalid_config",
                    f"{name} must be finite and positive",
                )
        for name in (
            "smoothed_center_delta",
            "smoothed_center_absolute_position_tolerance",
        ):
            value = float(getattr(self, name))
            if not (
                R8_SMOOTHED_CENTER_MIN_NORMALIZED_VALUE
                <= value
                <= R8_SMOOTHED_CENTER_MAX_NORMALIZED_VALUE
            ):
                _fail(
                    "invalid_config",
                    (
                        f"{name} must be in "
                        f"[{R8_SMOOTHED_CENTER_MIN_NORMALIZED_VALUE},"
                        f"{R8_SMOOTHED_CENTER_MAX_NORMALIZED_VALUE}] for "
                        "normalized-unit-square coordinates"
                    ),
                )


@dataclass(frozen=True)
class StableMotionRepresentation:
    """Fixed-phase output plus auditable variable-length component evidence."""

    diagnostic_ready: bool
    failure_reason: str | None
    failure_detail: str | None
    trajectory: np.ndarray
    transition_displacement: np.ndarray
    center_certificate_kind: tuple[str, ...]
    center_position_error_upper_bound: np.ndarray
    center_global_curvature_lower_bound: np.ndarray
    center_gradient_upper_bound: np.ndarray
    phase_energy: np.ndarray
    shape_tokens: np.ndarray
    phase_support: np.ndarray
    transition_support: np.ndarray
    transition_support_count: np.ndarray
    phase_times: np.ndarray
    component_track_indices: np.ndarray
    anchored_track_trajectories: np.ndarray
    track_phase_mask: np.ndarray
    track_anchor_phase: np.ndarray
    input_track_count: int
    required_phase_support: int
    required_transition_support: int
    smoothed_center_delta: float | None
    smoothed_center_absolute_position_tolerance: float | None
    coordinate_space: str
    semantic_actor_identified: bool = False
    formal_status: str = "INSUFFICIENT"
    production_decision: bool = False
    generation_authorized: bool = False
    scope: str = R8_STABLE_MOTION_SCOPE
    schema_version: str = R8_STABLE_MOTION_SCHEMA

    def to_summary(self) -> dict[str, Any]:
        """Return JSON-safe contract metadata without dense token arrays."""

        return {
            "schema_version": self.schema_version,
            "diagnostic_ready": self.diagnostic_ready,
            "failure_reason": self.failure_reason,
            "failure_detail": self.failure_detail,
            "phase_steps": R8_STABLE_MOTION_PHASE_STEPS,
            "shape_token_dim": R8_STABLE_MOTION_SHAPE_DIM,
            "shape_token_fields": list(R8_SHAPE_TOKEN_FIELDS),
            "component_track_indices": (
                self.component_track_indices.tolist()
            ),
            "input_track_count": self.input_track_count,
            "required_phase_support": self.required_phase_support,
            "required_transition_support": (
                self.required_transition_support
            ),
            "smoothed_center_delta": self.smoothed_center_delta,
            "smoothed_center_absolute_position_tolerance": (
                self.smoothed_center_absolute_position_tolerance
            ),
            "smoothed_center_normalized_parameter_bounds": {
                "minimum": R8_SMOOTHED_CENTER_MIN_NORMALIZED_VALUE,
                "maximum": R8_SMOOTHED_CENTER_MAX_NORMALIZED_VALUE,
            },
            "smoothed_center_max_iterations_limit": (
                R8_SMOOTHED_CENTER_MAX_ITERATIONS
            ),
            "coordinate_space": self.coordinate_space,
            "semantic_actor_identified": self.semantic_actor_identified,
            "formal_status": self.formal_status,
            "production_decision": self.production_decision,
            "generation_authorized": self.generation_authorized,
            "scope": self.scope,
            "energy_is_upstream_preserved": True,
            "aggregation": (
                "adjacent-phase common-visible displacement; certified "
                "fixed-delta pseudo-Huber spatial center; integrate from zero"
            ),
            "center_objective": (
                "mean sqrt(squared caller-coordinate residual plus delta "
                "squared)"
            ),
            "center_objective_is_exact_geometric_median": False,
            "center_storage_dtype": "float64",
            "center_certificate_kind_counts": {
                kind: self.center_certificate_kind.count(kind)
                for kind in sorted(set(self.center_certificate_kind))
            }
            if self.diagnostic_ready
            else {},
            "max_center_position_error_upper_bound": (
                float(np.max(self.center_position_error_upper_bound))
                if self.diagnostic_ready
                else None
            ),
            "gradient_roundoff_guard": (
                "64*float64_eps*(1+(norm(center)+max_norm(values))/delta)"
            ),
            "center_success_condition": (
                "minimum of proven global-strong-convexity and local "
                "Hessian-Lipschitz position certificates"
            ),
            "transition_quantity": (
                "coordinate displacement, not time-normalized velocity"
            ),
            "track_reanchor_role": "evidence-only",
            "phase_zero_has_no_transition": True,
        }


def _failure_result(
    reason: str,
    detail: str,
    *,
    input_track_count: int = 0,
    coordinate_space: str = "caller-defined",
    config: StableMotionConfig | None = None,
) -> StableMotionRepresentation:
    phase_steps = R8_STABLE_MOTION_PHASE_STEPS
    delta: float | None = None
    position_tolerance: float | None = None
    if isinstance(config, StableMotionConfig):
        try:
            raw_delta = float(config.smoothed_center_delta)
            raw_tolerance = float(
                config.smoothed_center_absolute_position_tolerance
            )
        except (TypeError, ValueError, OverflowError):
            pass
        else:
            if (
                math.isfinite(raw_delta)
                and R8_SMOOTHED_CENTER_MIN_NORMALIZED_VALUE
                <= raw_delta
                <= R8_SMOOTHED_CENTER_MAX_NORMALIZED_VALUE
            ):
                delta = raw_delta
            if (
                math.isfinite(raw_tolerance)
                and R8_SMOOTHED_CENTER_MIN_NORMALIZED_VALUE
                <= raw_tolerance
                <= R8_SMOOTHED_CENTER_MAX_NORMALIZED_VALUE
            ):
                position_tolerance = raw_tolerance
    return StableMotionRepresentation(
        diagnostic_ready=False,
        failure_reason=str(reason),
        failure_detail=str(detail),
        trajectory=np.zeros((phase_steps, 2), dtype=np.float64),
        transition_displacement=np.zeros(
            (phase_steps, 2),
            dtype=np.float64,
        ),
        center_certificate_kind=tuple(
            "unavailable" for _ in range(phase_steps)
        ),
        center_position_error_upper_bound=np.zeros(
            phase_steps,
            dtype=np.float64,
        ),
        center_global_curvature_lower_bound=np.zeros(
            phase_steps,
            dtype=np.float64,
        ),
        center_gradient_upper_bound=np.zeros(
            phase_steps,
            dtype=np.float64,
        ),
        phase_energy=np.zeros(phase_steps, dtype=np.float32),
        shape_tokens=np.zeros(
            (phase_steps, R8_STABLE_MOTION_SHAPE_DIM),
            dtype=np.float32,
        ),
        phase_support=np.zeros(phase_steps, dtype=np.float32),
        transition_support=np.zeros(phase_steps, dtype=np.float32),
        transition_support_count=np.zeros(phase_steps, dtype=np.int64),
        phase_times=np.zeros(phase_steps, dtype=np.float64),
        component_track_indices=np.zeros(0, dtype=np.int64),
        anchored_track_trajectories=np.zeros(
            (0, phase_steps, 2),
            dtype=np.float32,
        ),
        track_phase_mask=np.zeros((0, phase_steps), dtype=bool),
        track_anchor_phase=np.zeros(0, dtype=np.int64),
        input_track_count=max(int(input_track_count), 0),
        required_phase_support=0,
        required_transition_support=0,
        smoothed_center_delta=delta,
        smoothed_center_absolute_position_tolerance=position_tolerance,
        coordinate_space=str(coordinate_space),
    )


def _infer_track_count(value: Any) -> int:
    try:
        array = np.asarray(value)
    except Exception:
        return 0
    if (
        array.ndim == 3
        and array.shape[1:] == (R8_STABLE_MOTION_PHASE_STEPS, 2)
    ):
        return int(array.shape[0])
    return 0


def _real_numeric_array(
    value: Any,
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception:
        _fail("invalid_array", f"{name} cannot be converted to an array")
    if array.ndim != ndim:
        _fail("invalid_shape", f"{name} must have rank {ndim}")
    if (
        not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        _fail("invalid_dtype", f"{name} must be real numeric")
    output = array.astype(np.float64, copy=False)
    if not np.isfinite(output).all():
        _fail("non_finite_input", f"{name} contains NaN or infinity")
    return np.ascontiguousarray(output)


def _finite_float32_output(value: np.ndarray, *, name: str) -> np.ndarray:
    """Cast an output tensor without silently turning finite values into inf."""

    with np.errstate(over="ignore", invalid="ignore"):
        output = np.asarray(value, dtype=np.float32)
    if not np.isfinite(output).all():
        _fail(
            "non_finite_output",
            f"{name} is not representable as finite float32",
        )
    return output


def _validated_inputs(
    track_trajectories: Any,
    track_phase_mask: Any,
    phase_energy: Any,
    *,
    component_track_indices: Any,
    phase_times: Any,
    coordinate_space: Any,
    config: StableMotionConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str,
]:
    config.validate()
    trajectories = _real_numeric_array(
        track_trajectories,
        name="track_trajectories",
        ndim=3,
    )
    expected_tail = (R8_STABLE_MOTION_PHASE_STEPS, 2)
    if trajectories.shape[1:] != expected_tail:
        _fail(
            "invalid_shape",
            f"track_trajectories must have shape [K,{expected_tail[0]},2]",
        )
    track_count = int(trajectories.shape[0])
    if track_count < config.minimum_tracks:
        _fail(
            "insufficient_tracks",
            f"need at least {config.minimum_tracks} component tracks",
        )

    try:
        mask = np.asarray(track_phase_mask)
    except Exception:
        _fail(
            "invalid_array",
            "track_phase_mask cannot be converted to an array",
        )
    if mask.shape != trajectories.shape[:2]:
        _fail(
            "invalid_shape",
            "track_phase_mask must match track_trajectories[:2]",
        )
    if not np.issubdtype(mask.dtype, np.bool_):
        _fail(
            "invalid_mask_dtype",
            "track_phase_mask must have boolean dtype",
        )
    mask = np.ascontiguousarray(mask.astype(bool, copy=False))
    valid_counts = np.sum(mask, axis=1)
    if bool((valid_counts < config.minimum_track_valid_phases).any()):
        _fail(
            "insufficient_track_support",
            "every preserved component track needs multiple valid phases",
        )

    energy = _real_numeric_array(
        phase_energy,
        name="phase_energy",
        ndim=1,
    )
    if energy.shape != (R8_STABLE_MOTION_PHASE_STEPS,):
        _fail(
            "invalid_shape",
            "phase_energy must have shape [32]",
        )
    if bool((energy < 0.0).any()):
        _fail(
            "invalid_energy",
            "phase_energy must be nonnegative",
        )

    if phase_times is None:
        times = np.linspace(
            0.0,
            1.0,
            R8_STABLE_MOTION_PHASE_STEPS,
            dtype=np.float64,
        )
    else:
        times = _real_numeric_array(
            phase_times,
            name="phase_times",
            ndim=1,
        )
        if times.shape != (R8_STABLE_MOTION_PHASE_STEPS,):
            _fail(
                "invalid_shape",
                "phase_times must have shape [32]",
            )
        if bool((np.diff(times) <= 0.0).any()):
            _fail(
                "invalid_phase_times",
                "phase_times must be strictly increasing",
            )

    if component_track_indices is None:
        indices = np.arange(track_count, dtype=np.int64)
    else:
        try:
            raw_indices = np.asarray(component_track_indices)
        except Exception:
            _fail(
                "invalid_array",
                "component_track_indices cannot be converted to an array",
            )
        if raw_indices.shape != (track_count,):
            _fail(
                "invalid_shape",
                "component_track_indices must have shape [K]",
            )
        if (
            not np.issubdtype(raw_indices.dtype, np.integer)
            or np.issubdtype(raw_indices.dtype, np.bool_)
        ):
            _fail(
                "invalid_component_membership",
                "component track indices must be integers",
            )
        indices = raw_indices.astype(np.int64, copy=False)
        if bool((indices < 0).any()) or len(np.unique(indices)) != track_count:
            _fail(
                "invalid_component_membership",
                "component track indices must be unique and nonnegative",
            )
        indices = np.ascontiguousarray(indices)

    if not isinstance(coordinate_space, str) or not coordinate_space.strip():
        _fail(
            "invalid_coordinate_space",
            "coordinate_space must be a nonempty string",
        )
    return trajectories, mask, energy, times, indices, coordinate_space.strip()


def _smoothed_center_state(
    values: np.ndarray,
    current: np.ndarray,
    *,
    delta: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return objective, gradient, Hessian, smooth radii, and max distance."""

    difference = current[None, :] - values
    euclidean = np.hypot(difference[:, 0], difference[:, 1])
    smooth = np.hypot(euclidean, delta)
    unit = difference / smooth[:, None]
    count = len(values)
    try:
        objective = math.fsum(float(value) for value in smooth) / count
        gradient = np.asarray(
            (
                math.fsum(float(value) for value in unit[:, 0]) / count,
                math.fsum(float(value) for value in unit[:, 1]) / count,
            ),
            dtype=np.float64,
        )
        hessian_xy = (
            math.fsum(
                float(value)
                for value in (
                    -unit[:, 0] * unit[:, 1] / smooth
                )
            )
            / count
        )
        hessian = np.asarray(
            (
                (
                    math.fsum(
                        float(value)
                        for value in (1.0 - unit[:, 0] ** 2) / smooth
                    )
                    / count,
                    hessian_xy,
                ),
                (
                    hessian_xy,
                    math.fsum(
                        float(value)
                        for value in (1.0 - unit[:, 1] ** 2) / smooth
                    )
                    / count,
                ),
            ),
            dtype=np.float64,
        )
    except (OverflowError, ValueError):
        _fail(
            "smoothed_median_numeric_failure",
            "pseudo-Huber state cannot be represented in float64",
        )
    arrays = (gradient, hessian, smooth)
    if (
        not math.isfinite(objective)
        or not all(np.isfinite(value).all() for value in arrays)
    ):
        _fail(
            "smoothed_median_numeric_failure",
            "pseudo-Huber state is non-finite",
        )
    return (
        objective,
        gradient,
        hessian,
        smooth,
        float(np.max(euclidean)),
    )


def _smoothed_center_objective(
    values: np.ndarray,
    candidate: np.ndarray,
    *,
    delta: float,
) -> float:
    difference = candidate[None, :] - values
    euclidean = np.hypot(difference[:, 0], difference[:, 1])
    smooth = np.hypot(euclidean, delta)
    try:
        return math.fsum(float(value) for value in smooth) / len(values)
    except (OverflowError, ValueError):
        return math.inf


def _smoothed_center_position_bound(
    *,
    values: np.ndarray,
    current: np.ndarray,
    center: np.ndarray,
    radius: float,
    gradient: np.ndarray,
    hessian: np.ndarray,
    maximum_distance: float,
    delta: float,
) -> tuple[float, str, float, float]:
    """Return a conservative global-or-local distance-to-minimizer bound."""

    unit_roundoff = np.finfo(np.float64).eps
    value_norm = np.hypot(values[:, 0], values[:, 1])
    current_norm = float(math.hypot(current[0], current[1]))
    maximum_value_norm = float(np.max(value_norm))
    magnitude = current_norm + maximum_value_norm
    gradient_error = (
        64.0
        * unit_roundoff
        * (1.0 + magnitude / delta)
    )
    hessian_error = (
        256.0
        * unit_roundoff
        * (
            1.0 / delta
            + magnitude / delta**2
        )
    )
    gradient_upper = float(np.linalg.norm(gradient)) + gradient_error

    center_norm = float(math.hypot(center[0], center[1]))
    bound_inflation = (
        64.0
        * unit_roundoff
        * (1.0 + current_norm + maximum_value_norm + center_norm)
    )
    distance_bound = max(2.0 * radius, maximum_distance)
    distance_bound = (
        distance_bound * (1.0 + 64.0 * unit_roundoff)
        + bound_inflation
    )
    smooth_bound = math.hypot(
        distance_bound,
        delta,
    )
    global_curvature = (
        delta**2 / smooth_bound**3
    ) * (1.0 - 32.0 * unit_roundoff)
    global_position_bound = (
        gradient_upper / global_curvature
        if global_curvature > 0.0
        else math.inf
    )

    eigenvalues = np.linalg.eigvalsh(hessian)
    local_curvature = max(float(eigenvalues[0]) - hessian_error, 0.0)
    hessian_lipschitz = 6.0 / delta**2
    discriminant = (
        local_curvature**2
        - 2.0 * hessian_lipschitz * gradient_upper
    )
    local_position_bound = math.inf
    if discriminant >= 0.0 and local_curvature > 0.0:
        denominator = local_curvature + math.sqrt(discriminant)
        if denominator > 0.0:
            local_position_bound = 2.0 * gradient_upper / denominator
    if global_position_bound <= local_position_bound:
        return (
            global_position_bound,
            "global",
            global_curvature,
            gradient_upper,
        )
    return (
        local_position_bound,
        "local",
        global_curvature,
        gradient_upper,
    )


@dataclass(frozen=True)
class _SmoothedCenterResult:
    center: np.ndarray
    certificate_kind: str
    position_error_upper_bound: float
    global_curvature_lower_bound: float
    gradient_upper_bound: float


def _deterministic_smoothed_spatial_center(
    vectors: np.ndarray,
    *,
    config: StableMotionConfig,
) -> _SmoothedCenterResult:
    """Solve and certify the fixed-delta pseudo-Huber spatial center."""

    values = np.asarray(vectors, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or not len(values)
        or not np.isfinite(values).all()
    ):
        _fail(
            "invalid_transition_vectors",
            "smoothed center requires a finite nonempty [N,2] array",
        )
    norms = np.hypot(values[:, 0], values[:, 1])
    if not np.isfinite(norms).all():
        _fail(
            "smoothed_median_numeric_failure",
            "transition-vector norms are non-finite",
        )
    order = sorted(
        range(len(values)),
        key=lambda index: (
            float(norms[index]),
            float(values[index, 0]),
            float(values[index, 1]),
        ),
    )
    values = values[np.asarray(order, dtype=np.int64)]
    try:
        center = np.asarray(
            (
                math.fsum(float(value) for value in values[:, 0])
                / len(values),
                math.fsum(float(value) for value in values[:, 1])
                / len(values),
            ),
            dtype=np.float64,
        )
    except (OverflowError, ValueError):
        _fail(
            "smoothed_median_numeric_failure",
            "transition-vector mean overflows float64",
        )
    if not np.isfinite(center).all():
        _fail(
            "smoothed_median_numeric_failure",
            "transition-vector mean is non-finite",
        )
    centered = values - center[None, :]
    radius_values = np.hypot(centered[:, 0], centered[:, 1])
    if not np.isfinite(radius_values).all():
        _fail(
            "smoothed_median_numeric_failure",
            "transition-vector radius is non-finite",
        )
    radius = float(np.max(radius_values))

    current = center.copy()
    for iteration in range(config.smoothed_center_max_iterations + 1):
        (
            objective,
            gradient,
            hessian,
            smooth,
            maximum_distance,
        ) = _smoothed_center_state(
            values,
            current,
            delta=config.smoothed_center_delta,
        )
        (
            position_bound,
            certificate_kind,
            global_curvature,
            gradient_upper,
        ) = _smoothed_center_position_bound(
            values=values,
            current=current,
            center=center,
            radius=radius,
            gradient=gradient,
            hessian=hessian,
            maximum_distance=maximum_distance,
            delta=config.smoothed_center_delta,
        )
        if (
            math.isfinite(position_bound)
            and position_bound
            <= config.smoothed_center_absolute_position_tolerance
        ):
            return _SmoothedCenterResult(
                center=current.copy(),
                certificate_kind=certificate_kind,
                position_error_upper_bound=position_bound,
                global_curvature_lower_bound=global_curvature,
                gradient_upper_bound=gradient_upper,
            )
        if iteration == config.smoothed_center_max_iterations:
            break

        directions: list[np.ndarray] = []
        try:
            newton_direction = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            newton_direction = np.full(2, np.nan, dtype=np.float64)
        if (
            np.isfinite(newton_direction).all()
            and float(gradient @ newton_direction) < 0.0
        ):
            directions.append(newton_direction)

        inverse = 1.0 / smooth
        try:
            denominator = math.fsum(float(value) for value in inverse)
            irls_target = np.asarray(
                (
                    math.fsum(
                        float(weight * value)
                        for weight, value in zip(inverse, values[:, 0])
                    )
                    / denominator,
                    math.fsum(
                        float(weight * value)
                        for weight, value in zip(inverse, values[:, 1])
                    )
                    / denominator,
                ),
                dtype=np.float64,
            )
        except (OverflowError, ValueError):
            irls_target = np.full(2, np.nan, dtype=np.float64)
        irls_direction = irls_target - current
        if (
            np.isfinite(irls_direction).all()
            and float(gradient @ irls_direction) < 0.0
        ):
            directions.append(irls_direction)

        accepted: np.ndarray | None = None
        for direction in directions:
            directional_derivative = float(gradient @ direction)
            for backtrack in range(R8_SMOOTHED_CENTER_MAX_BACKTRACKS):
                step = math.ldexp(1.0, -backtrack)
                candidate = current + step * direction
                if not np.isfinite(candidate).all():
                    continue
                candidate_objective = _smoothed_center_objective(
                    values,
                    candidate,
                    delta=config.smoothed_center_delta,
                )
                armijo_limit = (
                    objective
                    + R8_SMOOTHED_CENTER_ARMIJO_FACTOR
                    * step
                    * directional_derivative
                )
                if candidate_objective <= armijo_limit:
                    accepted = candidate
                    break
            if accepted is not None:
                break
        if accepted is None:
            _fail(
                "smoothed_median_uncertified",
                "Newton and IRLS directions failed guarded Armijo descent",
            )
        current = accepted
    _fail(
        "smoothed_median_uncertified",
        "pseudo-Huber center exceeded the certified iteration budget",
    )


def _robust_shape_token(
    residual: np.ndarray,
    *,
    config: StableMotionConfig,
) -> np.ndarray:
    """Return robust permutation-invariant displacement-shape statistics."""

    radius = np.linalg.norm(residual, axis=1)
    if float(np.max(radius)) <= config.eps:
        return np.zeros(R8_STABLE_MOTION_SHAPE_DIM, dtype=np.float64)
    median_radius = float(np.median(radius))
    radius_mad = float(np.median(np.abs(radius - median_radius)))
    radius_q75 = float(np.quantile(radius, 0.75))
    radius_q90 = float(np.quantile(radius, 0.90))
    robust_sigma = 1.4826 * radius_mad
    clip_radius = median_radius + (
        config.covariance_clip_mad_scale * robust_sigma
    )
    if clip_radius <= config.eps:
        clipped = np.zeros_like(residual)
    else:
        scale = np.minimum(
            1.0,
            clip_radius / np.maximum(radius, config.eps),
        )
        clipped = residual * scale[:, None]
    covariance = (clipped.T @ clipped) / max(len(clipped), 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    trace = float(np.sum(eigenvalues))
    anisotropy = (
        float((eigenvalues[-1] - eigenvalues[0]) / trace)
        if trace > config.eps
        else 0.0
    )
    token = np.asarray(
        (
            median_radius,
            radius_mad,
            radius_q75,
            radius_q90,
            float(covariance[0, 0]),
            float(covariance[0, 1]),
            float(covariance[1, 1]),
            anisotropy,
        ),
        dtype=np.float64,
    )
    if not np.isfinite(token).all():
        _fail("non_finite_output", "shape token is non-finite")
    return token


def build_stable_motion_representation(
    track_trajectories: Any,
    track_phase_mask: Any,
    phase_energy: Any,
    *,
    component_track_indices: Any = None,
    phase_times: Any = None,
    coordinate_space: str = "caller-defined",
    config: StableMotionConfig | None = None,
) -> StableMotionRepresentation:
    """Build a fixed 32-phase transition-set representation, or fail closed.

    Tracks may contain absolute positions or an already-relative R7 tensor.
    The global trajectory depends only on common-visible adjacent-phase
    displacements.  Re-anchoring an already-relative tensor is idempotent and
    is retained only as evidence.  No track is silently removed: component
    membership and its corresponding mask are preserved in canonical
    component-index order.
    """

    inferred_track_count = _infer_track_count(track_trajectories)
    if config is not None and not isinstance(config, StableMotionConfig):
        return _failure_result(
            "invalid_config",
            "config must be a StableMotionConfig instance",
            input_track_count=inferred_track_count,
            coordinate_space=(
                coordinate_space
                if isinstance(coordinate_space, str)
                else "invalid"
            ),
        )
    cfg = config or StableMotionConfig()
    try:
        (
            trajectories,
            mask,
            energy,
            times,
            indices,
            coordinate_name,
        ) = _validated_inputs(
            track_trajectories,
            track_phase_mask,
            phase_energy,
            component_track_indices=component_track_indices,
            phase_times=phase_times,
            coordinate_space=coordinate_space,
            config=cfg,
        )
        order = np.argsort(indices, kind="stable")
        trajectories = trajectories[order]
        mask = mask[order]
        indices = indices[order]
        track_count = len(indices)

        required_support = max(
            cfg.minimum_phase_tracks,
            int(math.ceil(cfg.minimum_phase_track_fraction * track_count)),
        )
        support_count = np.sum(mask, axis=0)
        if bool((support_count < required_support).any()):
            _fail(
                "insufficient_phase_support",
                (
                    "every phase must retain the pre-registered minimum "
                    "component support"
                ),
            )
        required_transition_support = max(
            cfg.minimum_transition_tracks,
            int(
                math.ceil(
                    cfg.minimum_transition_track_fraction * track_count
                )
            ),
        )
        common_transition = mask[:, :-1] & mask[:, 1:]
        common_transition_count = np.sum(common_transition, axis=0)
        if bool(
            (
                common_transition_count
                < required_transition_support
            ).any()
        ):
            _fail(
                "insufficient_transition_support",
                (
                    "every adjacent phase must retain enough common-visible "
                    "component tracks"
                ),
            )

        anchor_phase = np.argmax(mask, axis=1).astype(np.int64)
        anchored = np.zeros_like(trajectories)
        for track in range(track_count):
            anchor = trajectories[track, anchor_phase[track]].copy()
            anchored[track, mask[track]] = (
                trajectories[track, mask[track]] - anchor
            )

        trajectory = np.zeros(
            (R8_STABLE_MOTION_PHASE_STEPS, 2),
            dtype=np.float64,
        )
        transition_displacement = np.zeros_like(trajectory)
        shape_tokens = np.zeros(
            (
                R8_STABLE_MOTION_PHASE_STEPS,
                R8_STABLE_MOTION_SHAPE_DIM,
            ),
            dtype=np.float64,
        )
        certificate_kind = ["no-transition"] * (
            R8_STABLE_MOTION_PHASE_STEPS
        )
        position_error_bound = np.zeros(
            R8_STABLE_MOTION_PHASE_STEPS,
            dtype=np.float64,
        )
        global_curvature_bound = np.zeros_like(position_error_bound)
        gradient_upper_bound = np.zeros_like(position_error_bound)
        for phase in range(1, R8_STABLE_MOTION_PHASE_STEPS):
            common = common_transition[:, phase - 1]
            values = (
                trajectories[common, phase]
                - trajectories[common, phase - 1]
            )
            center_result = _deterministic_smoothed_spatial_center(
                values,
                config=cfg,
            )
            center = center_result.center
            transition_displacement[phase] = center
            trajectory[phase] = trajectory[phase - 1] + center
            certificate_kind[phase] = center_result.certificate_kind
            position_error_bound[phase] = (
                center_result.position_error_upper_bound
            )
            global_curvature_bound[phase] = (
                center_result.global_curvature_lower_bound
            )
            gradient_upper_bound[phase] = (
                center_result.gradient_upper_bound
            )
            shape_tokens[phase] = _robust_shape_token(
                values - center,
                config=cfg,
            )

        arrays = (
            anchored,
            trajectory,
            transition_displacement,
            shape_tokens,
            position_error_bound,
            global_curvature_bound,
            gradient_upper_bound,
        )
        if not all(np.isfinite(value).all() for value in arrays):
            _fail(
                "non_finite_output",
                "stable representation contains NaN or infinity",
            )
        phase_support = support_count.astype(np.float64) / track_count
        transition_support = np.concatenate(
            (
                np.zeros(1, dtype=np.float64),
                common_transition_count.astype(np.float64)
                / track_count,
            )
        )
        trajectory_output = trajectory.copy()
        transition_output = transition_displacement.copy()
        energy_output = _finite_float32_output(
            energy,
            name="phase_energy",
        )
        shape_output = _finite_float32_output(
            shape_tokens,
            name="shape_tokens",
        )
        phase_support_output = _finite_float32_output(
            phase_support,
            name="phase_support",
        )
        transition_support_output = _finite_float32_output(
            transition_support,
            name="transition_support",
        )
        anchored_output = _finite_float32_output(
            anchored,
            name="anchored_track_trajectories",
        )
        position_error_output = position_error_bound.copy()
        global_curvature_output = global_curvature_bound.copy()
        gradient_upper_output = gradient_upper_bound.copy()
        return StableMotionRepresentation(
            diagnostic_ready=True,
            failure_reason=None,
            failure_detail=None,
            trajectory=trajectory_output,
            transition_displacement=transition_output,
            center_certificate_kind=tuple(certificate_kind),
            center_position_error_upper_bound=position_error_output,
            center_global_curvature_lower_bound=(
                global_curvature_output
            ),
            center_gradient_upper_bound=gradient_upper_output,
            phase_energy=energy_output,
            shape_tokens=shape_output,
            phase_support=phase_support_output,
            transition_support=transition_support_output,
            transition_support_count=np.concatenate(
                (
                    np.zeros(1, dtype=np.int64),
                    common_transition_count.astype(np.int64),
                )
            ),
            phase_times=times.copy(),
            component_track_indices=indices.copy(),
            anchored_track_trajectories=anchored_output,
            track_phase_mask=mask.copy(),
            track_anchor_phase=anchor_phase,
            input_track_count=track_count,
            required_phase_support=required_support,
            required_transition_support=required_transition_support,
            smoothed_center_delta=cfg.smoothed_center_delta,
            smoothed_center_absolute_position_tolerance=(
                cfg.smoothed_center_absolute_position_tolerance
            ),
            coordinate_space=coordinate_name,
        )
    except _StableMotionFailure as error:
        return _failure_result(
            error.reason,
            error.detail,
            input_track_count=inferred_track_count,
            coordinate_space=(
                coordinate_space
                if isinstance(coordinate_space, str)
                else "invalid"
            ),
            config=cfg if isinstance(cfg, StableMotionConfig) else None,
        )


__all__ = [
    "R8_SHAPE_TOKEN_FIELDS",
    "R8_SMOOTHED_CENTER_DEFAULT_ABSOLUTE_POSITION_TOLERANCE",
    "R8_SMOOTHED_CENTER_DEFAULT_DELTA",
    "R8_SMOOTHED_CENTER_MAX_ITERATIONS",
    "R8_SMOOTHED_CENTER_MAX_NORMALIZED_VALUE",
    "R8_SMOOTHED_CENTER_MIN_NORMALIZED_VALUE",
    "R8_STABLE_MOTION_PHASE_STEPS",
    "R8_STABLE_MOTION_SCHEMA",
    "R8_STABLE_MOTION_SCOPE",
    "R8_STABLE_MOTION_SHAPE_DIM",
    "StableMotionConfig",
    "StableMotionRepresentation",
    "build_stable_motion_representation",
]
