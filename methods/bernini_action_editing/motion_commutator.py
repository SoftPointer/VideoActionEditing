#!/usr/bin/env python3
"""Pure-tensor motion commutator for Bernini action editing (v7).

The adapter is evaluated under both the requested action and a semantic no-op.
This makes its deployable contribution a difference of differences rather than
the raw adapted action branch::

    B0       = Q0(A0 - N0)
    C_theta  = Q0((A_theta - N_theta) - (A0 - N0))
    B_final  = B0 + bounded(C_theta)

Here ``A`` and ``N`` are action/no-op clean fields, ``0`` denotes the frozen
editor, and ``theta`` denotes the adapted editor.  Any adapter change shared
by the action and no-op branches therefore cancels before it can reach the
motion correction.  No mask, flow, pose, target video, generator, or first
frame anchor is part of the inference operator.

The trust region acts independently on each *temporal increment*.  It never
spatially filters the value-bearing correction.  An optional fixed three-tap
temporal filter can suppress phase-to-phase jitter before the hard projection.
The immutable forty-step release schedule is imported from v6.  At rho zero,
execution returns the exact frozen input tensor object without arithmetic.

PyTorch is imported lazily so the schedule/configuration contract remains
auditable in lightweight environments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

try:  # Package import.
    from . import cross_mode_motion_spectrum as _v6_spectrum
except ImportError:  # Direct import when METHOD_ROOT is placed on sys.path.
    import cross_mode_motion_spectrum as _v6_spectrum


METHOD_NAME = "counterfactual-motion-commutator-v7"
EXPECTED_PHASES = 21
NUM_DENOISING_STEPS = _v6_spectrum.NUM_DENOISING_STEPS
DEFAULT_MAX_CORRECTION_INCREMENT_RATIO = 0.25
DEFAULT_CORRECTION_INCREMENT_RMS_FLOOR = 1.0e-3
DEFAULT_EPSILON = 1.0e-8
FIXED_TEMPORAL_SMOOTHING_KERNEL = (0.25, 0.5, 0.25)


class MotionCommutatorError(RuntimeError):
    """Raised when a v7 tensor or immutable operator contract is violated."""


@dataclass(frozen=True)
class MotionCommutatorConfig:
    """Auditable, deliberately conservative commutator hyperparameters.

    The per-phase correction-increment RMS cap is

    ``max(max_correction_increment_ratio * frozen_increment_rms,
          correction_increment_rms_floor)``.

    The small positive default floor permits a genuinely new action increment
    when the frozen editor is locally static, while the relative term prevents
    a large adapter residual from overwhelming an already active frozen prior.
    Setting the floor explicitly to zero gives a purely relative trust region.
    """

    max_correction_increment_ratio: float = (
        DEFAULT_MAX_CORRECTION_INCREMENT_RATIO
    )
    correction_increment_rms_floor: float = (
        DEFAULT_CORRECTION_INCREMENT_RMS_FLOOR
    )
    temporal_smoothing: bool = False
    epsilon: float = DEFAULT_EPSILON

    def validate(self) -> None:
        _validate_nonnegative_finite(
            "max_correction_increment_ratio",
            self.max_correction_increment_ratio,
        )
        _validate_nonnegative_finite(
            "correction_increment_rms_floor",
            self.correction_increment_rms_floor,
        )
        if type(self.temporal_smoothing) is not bool:
            raise MotionCommutatorError("temporal_smoothing must be a bool")
        if isinstance(self.epsilon, bool):
            raise MotionCommutatorError("epsilon must be finite and positive")
        try:
            epsilon = float(self.epsilon)
        except (TypeError, ValueError) as error:
            raise MotionCommutatorError(
                "epsilon must be finite and positive"
            ) from error
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise MotionCommutatorError("epsilon must be finite and positive")


@dataclass(frozen=True)
class MotionCommutatorDiagnostics:
    """Per-phase evidence for the hard increment-domain trust projection."""

    frozen_increment_rms: Any
    raw_correction_increments: Any
    candidate_correction_increments: Any
    candidate_correction_increment_rms: Any
    correction_increment_rms_cap: Any
    bound_scale: Any
    bounded_correction_increments: Any
    bounded_correction_increment_rms: Any
    temporal_smoothing_applied: bool


@dataclass(frozen=True)
class MotionCommutatorResult:
    """The frozen direction, adapter commutator, and bounded final direction."""

    frozen_official_direction: Any
    raw_commutator_correction: Any
    candidate_commutator_correction: Any
    bounded_commutator_correction: Any
    final_direction: Any
    diagnostics: MotionCommutatorDiagnostics


@dataclass(frozen=True)
class RawMotionCommutatorResult:
    """Unbounded training representation before any deployment projection."""

    frozen_official_direction: Any
    raw_commutator_correction: Any
    unbounded_final_direction: Any


@dataclass(frozen=True)
class BoundedMotionCorrectionResult:
    """Deployment-only projection of a raw commutator correction."""

    candidate_commutator_correction: Any
    bounded_commutator_correction: Any
    final_direction: Any
    diagnostics: MotionCommutatorDiagnostics


@dataclass(frozen=True)
class MotionCommutatorExecution:
    """One scheduled release of a previously bounded commutator correction."""

    frozen_official_direction: Any
    bounded_commutator_correction: Any
    rho: float
    executed_direction: Any


@dataclass(frozen=True)
class OfficialTensorExecution:
    """Scheduler-boundary reconstruction with exact rho-zero object identity."""

    frozen_official_tensor: Any
    bounded_commutator_correction: Any
    rho: float
    executed_official_tensor: Any


def _validate_nonnegative_finite(name: str, value: Any) -> None:
    if isinstance(value, bool):
        raise MotionCommutatorError(f"{name} must be finite and nonnegative")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise MotionCommutatorError(
            f"{name} must be finite and nonnegative"
        ) from error
    if not math.isfinite(numeric) or numeric < 0.0:
        raise MotionCommutatorError(f"{name} must be finite and nonnegative")


def _validate_positive_finite(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise MotionCommutatorError(f"{name} must be finite and positive")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise MotionCommutatorError(
            f"{name} must be finite and positive"
        ) from error
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise MotionCommutatorError(f"{name} must be finite and positive")
    return numeric


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        raise MotionCommutatorError(
            "motion commutator tensor operations require PyTorch"
        ) from error
    return torch


def _validate_fields(*fields: Any) -> None:
    """Require identical finite float32 ``[B,21,S,D]`` tensors."""

    torch = _require_torch()
    if not fields:
        raise MotionCommutatorError("at least one field is required")
    reference = fields[0]
    if not isinstance(reference, torch.Tensor):
        raise MotionCommutatorError("fields must be torch tensors")
    if reference.ndim != 4 or int(reference.shape[1]) != EXPECTED_PHASES:
        raise MotionCommutatorError("fields must have exact shape [B,21,S,D]")
    if any(int(reference.shape[index]) <= 0 for index in (0, 2, 3)):
        raise MotionCommutatorError("B, S, and D must be non-empty")
    if reference.dtype != torch.float32:
        raise MotionCommutatorError("fields must be torch.float32")
    for field in fields:
        if not isinstance(field, torch.Tensor):
            raise MotionCommutatorError("fields must be torch tensors")
        if tuple(field.shape) != tuple(reference.shape):
            raise MotionCommutatorError("field shapes differ")
        if field.dtype != reference.dtype or field.device != reference.device:
            raise MotionCommutatorError("field dtype or device differs")
        if not bool(torch.isfinite(field).all()):
            raise MotionCommutatorError("field contains non-finite values")


def _validate_directions(*directions: Any) -> None:
    """Validate fields and require an exact causal phase-zero gauge."""

    torch = _require_torch()
    _validate_fields(*directions)
    for direction in directions:
        if not bool(
            torch.equal(direction[:, 0], torch.zeros_like(direction[:, 0]))
        ):
            raise MotionCommutatorError(
                "directions must have an exact zero phase zero"
            )


def causal_gauge(field: Any) -> Any:
    """Apply ``Q0(x)_t = x_t - x_0`` with exact phase-zero verification."""

    torch = _require_torch()
    _validate_fields(field)
    projected = field - field[:, :1]
    if not bool(
        torch.equal(projected[:, 0], torch.zeros_like(projected[:, 0]))
    ):
        raise MotionCommutatorError("Q0 failed to make phase zero exact")
    return projected


def phase_increments(direction: Any) -> Any:
    """Return 21 causal increments, prepending an exact zero increment."""

    torch = _require_torch()
    _validate_directions(direction)
    return torch.cat(
        (
            torch.zeros_like(direction[:, :1]),
            direction[:, 1:] - direction[:, :-1],
        ),
        dim=1,
    )


def _integrate_increments(increments: Any) -> Any:
    torch = _require_torch()
    _validate_directions(increments)
    integrated = torch.cumsum(increments, dim=1)
    if not bool(
        torch.equal(integrated[:, 0], torch.zeros_like(integrated[:, 0]))
    ) or not bool(torch.isfinite(integrated).all()):
        raise MotionCommutatorError(
            "increment integration violated the causal finite contract"
        )
    return integrated


def _phase_rms(field: Any) -> Any:
    """Return RMS over only spatial-token and channel axes."""

    return field.square().mean(dim=(2, 3)).sqrt()


def _fixed_temporal_smooth(increments: Any) -> Any:
    """Smooth only the temporal axis with the immutable [1/4,1/2,1/4] FIR."""

    torch = _require_torch()
    _validate_directions(increments)
    active = increments[:, 1:]
    previous = torch.cat((active[:, :1], active[:, :-1]), dim=1)
    following = torch.cat((active[:, 1:], active[:, -1:]), dim=1)
    smoothed_active = (
        FIXED_TEMPORAL_SMOOTHING_KERNEL[0] * previous
        + FIXED_TEMPORAL_SMOOTHING_KERNEL[1] * active
        + FIXED_TEMPORAL_SMOOTHING_KERNEL[2] * following
    )
    return torch.cat((torch.zeros_like(increments[:, :1]), smoothed_active), dim=1)


def build_raw_motion_commutator(
    adapted_action_field: Any,
    adapted_noop_field: Any,
    frozen_action_field: Any,
    frozen_noop_field: Any,
) -> RawMotionCommutatorResult:
    """Build the unbounded action-specific residual used by training.

    All four inputs are editor clean fields at one shared diffusion state.  The
    function has no target-derived or spatial-control input.  It deliberately
    performs no trust-region projection: training supervises the raw residual,
    while deployment calls :func:`bound_motion_commutator_correction`.
    """

    _validate_fields(
        adapted_action_field,
        adapted_noop_field,
        frozen_action_field,
        frozen_noop_field,
    )

    frozen_semantic = frozen_action_field - frozen_noop_field
    adapted_semantic = adapted_action_field - adapted_noop_field
    frozen_direction = causal_gauge(frozen_semantic)
    raw_correction = causal_gauge(adapted_semantic - frozen_semantic)
    unbounded_final = frozen_direction + raw_correction
    _validate_directions(frozen_direction, raw_correction, unbounded_final)
    return RawMotionCommutatorResult(
        frozen_official_direction=frozen_direction,
        raw_commutator_correction=raw_correction,
        unbounded_final_direction=unbounded_final,
    )


def bound_motion_commutator_correction(
    frozen_official_direction: Any,
    raw_commutator_correction: Any,
    *,
    config: MotionCommutatorConfig = MotionCommutatorConfig(),
) -> BoundedMotionCorrectionResult:
    """Apply the deployment-only temporal filter and hard increment bound."""

    torch = _require_torch()
    config.validate()
    _validate_directions(frozen_official_direction, raw_commutator_correction)

    frozen_increments = phase_increments(frozen_official_direction)
    raw_correction_increments = phase_increments(raw_commutator_correction)
    candidate_increments = (
        _fixed_temporal_smooth(raw_correction_increments)
        if config.temporal_smoothing
        else raw_correction_increments
    )

    frozen_rms = _phase_rms(frozen_increments)
    candidate_rms = _phase_rms(candidate_increments)
    relative_cap = (
        float(config.max_correction_increment_ratio) * frozen_rms
    )
    absolute_floor = torch.full_like(
        relative_cap,
        float(config.correction_increment_rms_floor),
    )
    cap = torch.maximum(relative_cap, absolute_floor)
    scale = torch.where(
        candidate_rms > cap,
        cap / candidate_rms.clamp_min(float(config.epsilon)),
        torch.ones_like(candidate_rms),
    )
    bounded_increments = candidate_increments * scale[..., None, None]
    bounded_rms = _phase_rms(bounded_increments)
    tolerance = max(float(config.epsilon), 8.0 * torch.finfo(torch.float32).eps)
    if bool((bounded_rms > cap + tolerance).any()):
        raise MotionCommutatorError("hard increment trust bound was violated")

    candidate_correction = _integrate_increments(candidate_increments)
    bounded_correction = _integrate_increments(bounded_increments)
    final_direction = frozen_official_direction + bounded_correction
    _validate_directions(
        frozen_official_direction,
        raw_commutator_correction,
        candidate_correction,
        bounded_correction,
        final_direction,
    )

    diagnostics = MotionCommutatorDiagnostics(
        frozen_increment_rms=frozen_rms,
        raw_correction_increments=raw_correction_increments,
        candidate_correction_increments=candidate_increments,
        candidate_correction_increment_rms=candidate_rms,
        correction_increment_rms_cap=cap,
        bound_scale=scale,
        bounded_correction_increments=bounded_increments,
        bounded_correction_increment_rms=bounded_rms,
        temporal_smoothing_applied=config.temporal_smoothing,
    )
    return BoundedMotionCorrectionResult(
        candidate_commutator_correction=candidate_correction,
        bounded_commutator_correction=bounded_correction,
        final_direction=final_direction,
        diagnostics=diagnostics,
    )


def build_motion_commutator(
    adapted_action_field: Any,
    adapted_noop_field: Any,
    frozen_action_field: Any,
    frozen_noop_field: Any,
    *,
    config: MotionCommutatorConfig = MotionCommutatorConfig(),
) -> MotionCommutatorResult:
    """Build the raw training residual and its bounded deployment form."""

    raw = build_raw_motion_commutator(
        adapted_action_field,
        adapted_noop_field,
        frozen_action_field,
        frozen_noop_field,
    )
    bounded = bound_motion_commutator_correction(
        raw.frozen_official_direction,
        raw.raw_commutator_correction,
        config=config,
    )
    return MotionCommutatorResult(
        frozen_official_direction=raw.frozen_official_direction,
        raw_commutator_correction=raw.raw_commutator_correction,
        candidate_commutator_correction=bounded.candidate_commutator_correction,
        bounded_commutator_correction=bounded.bounded_commutator_correction,
        final_direction=bounded.final_direction,
        diagnostics=bounded.diagnostics,
    )


def release_rho(step_index: int) -> float:
    """Reuse the immutable v6 40-step release schedule verbatim."""

    try:
        return float(_v6_spectrum.release_rho(step_index))
    except _v6_spectrum.CrossModeMotionSpectrumError as error:
        raise MotionCommutatorError(str(error)) from error


def release_rho_schedule() -> tuple[float, ...]:
    """Return all forty v6 release coefficients."""

    return tuple(release_rho(step) for step in range(NUM_DENOISING_STEPS))


def execute_motion_commutator(
    frozen_official_direction: Any,
    bounded_commutator_correction: Any,
    *,
    step_index: int,
) -> MotionCommutatorExecution:
    """Release ``B0 + rho*C`` and alias ``B0`` itself whenever rho is zero."""

    torch = _require_torch()
    _validate_directions(
        frozen_official_direction,
        bounded_commutator_correction,
    )
    rho = release_rho(step_index)
    executed = (
        frozen_official_direction
        if rho == 0.0
        else frozen_official_direction
        + rho * bounded_commutator_correction
    )
    _validate_directions(executed)
    if rho == 0.0 and executed is not frozen_official_direction:
        raise MotionCommutatorError("rho-zero direction did not preserve identity")
    if not bool(torch.isfinite(executed).all()):  # Defensive after arithmetic.
        raise MotionCommutatorError("executed direction is non-finite")
    return MotionCommutatorExecution(
        frozen_official_direction=frozen_official_direction,
        bounded_commutator_correction=bounded_commutator_correction,
        rho=rho,
        executed_direction=executed,
    )


def apply_motion_commutator_to_official_tensor(
    frozen_official_tensor: Any,
    bounded_commutator_correction: Any,
    *,
    step_index: int,
) -> OfficialTensorExecution:
    """Apply a released correction at the scheduler boundary.

    ``frozen_official_tensor`` may have a nonzero phase zero.  The correction
    must be in the exact Q0 gauge, so its release preserves that phase.  At all
    rho-zero steps the returned tensor is the exact same Python object as the
    frozen official tensor, which permits untouched official scheduler replay.
    """

    torch = _require_torch()
    _validate_fields(frozen_official_tensor, bounded_commutator_correction)
    _validate_directions(bounded_commutator_correction)
    rho = release_rho(step_index)
    executed = (
        frozen_official_tensor
        if rho == 0.0
        else frozen_official_tensor
        + rho * bounded_commutator_correction
    )
    _validate_fields(executed)
    if not bool(torch.equal(executed[:, 0], frozen_official_tensor[:, 0])):
        raise MotionCommutatorError("official phase-zero boundary changed")
    if rho == 0.0 and executed is not frozen_official_tensor:
        raise MotionCommutatorError("rho-zero official tensor did not alias input")
    return OfficialTensorExecution(
        frozen_official_tensor=frozen_official_tensor,
        bounded_commutator_correction=bounded_commutator_correction,
        rho=rho,
        executed_official_tensor=executed,
    )


def build_target_correction(
    target_motion_direction: Any,
    frozen_official_direction: Any,
) -> Any:
    """Build the training-only residual target ``Q0(T - B0)``.

    This helper is deliberately separate from every inference API.  Both
    arguments must already be causal motion directions with exact zero phase.
    """

    _validate_directions(target_motion_direction, frozen_official_direction)
    correction = causal_gauge(
        target_motion_direction - frozen_official_direction
    )
    _validate_directions(correction)
    return correction


def adapted_noop_preservation_loss(
    adapted_noop_field: Any,
    frozen_noop_field: Any,
    *,
    epsilon: float = 1.0e-3,
) -> Any:
    """Robustly anchor the adapted no-op clean field to the frozen editor."""

    epsilon = _validate_positive_finite("epsilon", epsilon)
    _validate_fields(adapted_noop_field, frozen_noop_field)
    difference = adapted_noop_field - frozen_noop_field
    return ((difference.square() + epsilon**2).sqrt() - epsilon).mean()


def target_correction_loss(
    predicted_correction: Any,
    target_correction: Any,
    *,
    epsilon: float = 1.0e-3,
) -> Any:
    """Robustly supervise the signed temporal increments of ``C_theta``."""

    epsilon = _validate_positive_finite("epsilon", epsilon)
    _validate_directions(predicted_correction, target_correction)
    error = phase_increments(predicted_correction) - phase_increments(
        target_correction
    )
    return ((error.square() + epsilon**2).sqrt() - epsilon).mean()


__all__ = [
    "DEFAULT_CORRECTION_INCREMENT_RMS_FLOOR",
    "DEFAULT_EPSILON",
    "DEFAULT_MAX_CORRECTION_INCREMENT_RATIO",
    "EXPECTED_PHASES",
    "FIXED_TEMPORAL_SMOOTHING_KERNEL",
    "METHOD_NAME",
    "MotionCommutatorConfig",
    "BoundedMotionCorrectionResult",
    "MotionCommutatorDiagnostics",
    "MotionCommutatorError",
    "MotionCommutatorExecution",
    "MotionCommutatorResult",
    "RawMotionCommutatorResult",
    "NUM_DENOISING_STEPS",
    "OfficialTensorExecution",
    "adapted_noop_preservation_loss",
    "apply_motion_commutator_to_official_tensor",
    "bound_motion_commutator_correction",
    "build_motion_commutator",
    "build_raw_motion_commutator",
    "build_target_correction",
    "causal_gauge",
    "execute_motion_commutator",
    "phase_increments",
    "release_rho",
    "release_rho_schedule",
    "target_correction_loss",
]
