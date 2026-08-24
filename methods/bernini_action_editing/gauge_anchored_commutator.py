#!/usr/bin/env python3
"""Reconstruction-section feasible quotient transport for Bernini v8.

V7 cannot undo identity drift already present in the coefficient-one frozen
action field.  V8 instead uses the frozen semantic no-op field as the
appearance/reconstruction section and transports the *complete* adapted
action/no-op quotient through a zero-centred, source-only trust region::

    q_theta = Q0(A_theta - N_theta)
    z_theta = FIR(D q_theta)
    q_bar   = Integrate(Project_radius(z_theta))
    E_theta = N0 + rho * q_bar

The frozen quotient and frozen no-op dynamics determine ``radius``; neither a
target nor an inference-only localization oracle is consulted.  The auxiliary
``build_noop_gauge_anchor`` routine exposes the frozen special case
``N0 + Q0(A0-N0)`` for diagnostics.

PyTorch remains a lazy dependency so receipt and configuration code can be
inspected in lightweight environments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

try:  # Package import.
    from . import motion_commutator as commutator
except ImportError:  # Direct import with METHOD_ROOT on sys.path.
    import motion_commutator as commutator


METHOD_NAME = "bernini-noop-gauge-appearance-carrier-v8"
APPEARANCE_CARRIER = "frozen_noop_reconstruction_section"
FULL_QUOTIENT_OPERATOR = "reconstruction_section_feasible_quotient_transport"
DEFAULT_FROZEN_QUOTIENT_RADIUS_RATIO = 1.0
DEFAULT_NOOP_DYNAMICS_RADIUS_RATIO = 0.25
DEFAULT_RADIUS_FLOOR = 1.0e-3
DEFAULT_EPSILON = 1.0e-8


class GaugeAnchoredCommutatorError(RuntimeError):
    """Raised before an invalid gauge-anchored field reaches the scheduler."""


@dataclass(frozen=True)
class NoopGaugeAnchor:
    """Frozen action field expressed in the semantic no-op gauge."""

    frozen_action_field: Any
    frozen_noop_field: Any
    frozen_semantic_motion: Any
    anchored_action_field: Any
    removed_common_mode: Any
    phase_increment_rms_error: Any
    phase_increment_tolerance: float


@dataclass(frozen=True)
class GaugeAnchoredExecution:
    """One scheduler-boundary clean field and its auditable gauge evidence."""

    anchor: NoopGaugeAnchor
    bounded_commutator_correction: Any
    rho: float
    executed_clean_field: Any


@dataclass(frozen=True)
class FeasibleQuotientConfig:
    """Source-only trust radius for the complete action/no-op quotient."""

    frozen_quotient_radius_ratio: float = (
        DEFAULT_FROZEN_QUOTIENT_RADIUS_RATIO
    )
    noop_dynamics_radius_ratio: float = DEFAULT_NOOP_DYNAMICS_RADIUS_RATIO
    radius_floor: float = DEFAULT_RADIUS_FLOOR
    epsilon: float = DEFAULT_EPSILON

    def validate(self) -> None:
        for name in (
            "frozen_quotient_radius_ratio",
            "noop_dynamics_radius_ratio",
            "radius_floor",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise GaugeAnchoredCommutatorError(
                    f"{name} must be finite and nonnegative"
                )
        if (
            isinstance(self.epsilon, bool)
            or not math.isfinite(float(self.epsilon))
            or float(self.epsilon) <= 0.0
        ):
            raise GaugeAnchoredCommutatorError(
                "epsilon must be finite and strictly positive"
            )


@dataclass(frozen=True)
class FeasibleQuotientDiagnostics:
    """Per-phase evidence for the centered, full-quotient trust region."""

    frozen_quotient: Any
    adapted_quotient: Any
    noop_motion_direction: Any
    adapted_raw_increments: Any
    adapted_smoothed_increments: Any
    frozen_smoothed_increment_rms: Any
    noop_smoothed_increment_rms: Any
    adapted_smoothed_increment_rms: Any
    radius: Any
    scale: Any
    bounded_increments: Any
    bounded_increment_rms: Any


@dataclass(frozen=True)
class FeasibleQuotientProjection:
    """Complete adapted motion quotient projected around zero, not q0."""

    bounded_quotient: Any
    diagnostics: FeasibleQuotientDiagnostics


@dataclass(frozen=True)
class FeasibleQuotientExecution:
    """Lift a bounded motion quotient from the frozen no-op section."""

    frozen_noop_field: Any
    bounded_quotient: Any
    rho: float
    executed_clean_field: Any


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependent
        raise GaugeAnchoredCommutatorError(
            "v8 gauge anchoring requires PyTorch"
        ) from error
    return torch


def _validate_pair(action: Any, noop: Any) -> None:
    torch = _require_torch()
    if not isinstance(action, torch.Tensor) or not isinstance(noop, torch.Tensor):
        raise GaugeAnchoredCommutatorError("action and no-op fields must be tensors")
    if (
        action.ndim != 4
        or tuple(action.shape) != tuple(noop.shape)
        or int(action.shape[1]) != commutator.EXPECTED_PHASES
        or action.dtype != noop.dtype
        or action.device != noop.device
        or not bool(torch.isfinite(action).all())
        or not bool(torch.isfinite(noop).all())
    ):
        raise GaugeAnchoredCommutatorError(
            "action/no-op fields must be matching finite [B,21,S,D] tensors"
        )


def _rms(value: Any) -> Any:
    return value.square().mean(dim=tuple(range(1, value.ndim))).sqrt()


def phase_rms(value: Any) -> Any:
    """Return RMS over spatial/token and channel dimensions for each phase."""

    return value.square().mean(dim=(2, 3)).sqrt()


def smooth_phase_increments(increments: Any) -> Any:
    """Apply the same replicated-edge [1/4,1/2,1/4] FIR as v7."""

    torch = _require_torch()
    active = increments[:, 1:]
    previous = torch.cat((active[:, :1], active[:, :-1]), dim=1)
    following = torch.cat((active[:, 1:], active[:, -1:]), dim=1)
    smoothed = 0.25 * previous + 0.5 * active + 0.25 * following
    return torch.cat((torch.zeros_like(increments[:, :1]), smoothed), dim=1)


def integrate_phase_increments(increments: Any) -> Any:
    torch = _require_torch()
    integrated = torch.cumsum(increments, dim=1)
    if not bool(torch.equal(
        integrated[:, 0], torch.zeros_like(integrated[:, 0])
    )) or not bool(torch.isfinite(integrated).all()):
        raise GaugeAnchoredCommutatorError(
            "full-quotient increment integration is not finite and causal"
        )
    return integrated


def radial_project_phase_increments(
    increments: Any,
    radius: Any,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[Any, Any]:
    """Project each phase RMS into a zero-centred ball.

    This differentiable primitive is shared verbatim by v8 training and
    deployment.  It returns ``(bounded_increments, scale)`` with scale shaped
    ``[B,21]``.
    """

    torch = _require_torch()
    if (
        not isinstance(increments, torch.Tensor)
        or increments.ndim != 4
        or int(increments.shape[1]) != commutator.EXPECTED_PHASES
        or not isinstance(radius, torch.Tensor)
        or tuple(radius.shape) != tuple(increments.shape[:2])
        or radius.dtype != increments.dtype
        or radius.device != increments.device
        or not bool(torch.isfinite(increments).all())
        or not bool(torch.isfinite(radius).all())
        or bool((radius < 0.0).any())
        or not bool(torch.equal(
            increments[:, 0], torch.zeros_like(increments[:, 0])
        ))
        or isinstance(epsilon, bool)
        or not math.isfinite(float(epsilon))
        or float(epsilon) <= 0.0
    ):
        raise GaugeAnchoredCommutatorError(
            "radial projection requires finite causal increments and [B,21] radius"
        )
    increment_rms = phase_rms(increments)
    scale = torch.where(
        increment_rms > radius,
        radius / increment_rms.clamp_min(float(epsilon)),
        torch.ones_like(increment_rms),
    )
    return increments * scale[..., None, None], scale


def build_noop_gauge_anchor(
    frozen_action_field: Any,
    frozen_noop_field: Any,
) -> NoopGaugeAnchor:
    """Replace only the action field's temporal common mode with no-op.

    Algebraically ``G0 = A0 - (A0-N0)[:,0]``.  Consequently ``G0[:,0]`` is
    exactly ``N0[:,0]`` and every phase increment of ``G0`` matches ``A0`` up
    to floating-point reassociation.  No source or target endpoint is read.
    """

    torch = _require_torch()
    _validate_pair(frozen_action_field, frozen_noop_field)
    try:
        semantic_motion = commutator.causal_gauge(
            frozen_action_field - frozen_noop_field
        )
        # ``phase_increments`` accepts causal directions, not absolute clean
        # fields.  Gauging first changes no temporal increment but gives the
        # exact zero phase required by its contract.
        action_increments = commutator.phase_increments(
            commutator.causal_gauge(frozen_action_field)
        )
    except commutator.MotionCommutatorError as error:
        raise GaugeAnchoredCommutatorError(str(error)) from error
    anchored = frozen_noop_field + semantic_motion
    if not bool(torch.equal(anchored[:, 0], frozen_noop_field[:, 0])):
        raise GaugeAnchoredCommutatorError(
            "no-op gauge did not preserve the exact no-op phase zero"
        )
    try:
        anchored_increments = commutator.phase_increments(
            commutator.causal_gauge(anchored)
        )
    except commutator.MotionCommutatorError as error:
        raise GaugeAnchoredCommutatorError(str(error)) from error
    increment_error = anchored_increments - action_increments
    scale = max(
        1.0,
        float(action_increments.detach().abs().max().cpu().item()),
        float(anchored_increments.detach().abs().max().cpu().item()),
    )
    tolerance = 16.0 * float(torch.finfo(frozen_action_field.dtype).eps) * scale
    if not bool(torch.allclose(
        anchored_increments,
        action_increments,
        rtol=0.0,
        atol=tolerance,
    )):
        raise GaugeAnchoredCommutatorError(
            "no-op gauge changed frozen action temporal increments"
        )
    removed = frozen_action_field - anchored
    expected_removed = (
        frozen_action_field[:, :1] - frozen_noop_field[:, :1]
    ).expand_as(removed)
    if not bool(torch.allclose(
        removed,
        expected_removed,
        rtol=0.0,
        atol=tolerance,
    )):
        raise GaugeAnchoredCommutatorError(
            "removed field is not one temporally constant semantic offset"
        )
    return NoopGaugeAnchor(
        frozen_action_field=frozen_action_field,
        frozen_noop_field=frozen_noop_field,
        frozen_semantic_motion=semantic_motion,
        anchored_action_field=anchored,
        removed_common_mode=removed,
        phase_increment_rms_error=_rms(increment_error),
        phase_increment_tolerance=tolerance,
    )


def execute_gauge_anchored_commutator(
    frozen_action_field: Any,
    frozen_noop_field: Any,
    bounded_commutator_correction: Any,
    *,
    step_index: int,
) -> GaugeAnchoredExecution:
    """Release a bounded correction around the no-op-gauge action carrier."""

    torch = _require_torch()
    anchor = build_noop_gauge_anchor(frozen_action_field, frozen_noop_field)
    if (
        not isinstance(bounded_commutator_correction, torch.Tensor)
        or tuple(bounded_commutator_correction.shape)
        != tuple(anchor.anchored_action_field.shape)
        or bounded_commutator_correction.dtype
        != anchor.anchored_action_field.dtype
        or bounded_commutator_correction.device
        != anchor.anchored_action_field.device
        or not bool(torch.isfinite(bounded_commutator_correction).all())
    ):
        raise GaugeAnchoredCommutatorError(
            "bounded correction differs from the anchored clean-field geometry"
        )
    if not bool(torch.equal(
        bounded_commutator_correction[:, 0],
        torch.zeros_like(bounded_commutator_correction[:, 0]),
    )):
        raise GaugeAnchoredCommutatorError(
            "bounded commutator correction must have exact zero phase"
        )
    try:
        rho = float(commutator.release_rho(step_index))
    except commutator.MotionCommutatorError as error:
        raise GaugeAnchoredCommutatorError(str(error)) from error
    if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise GaugeAnchoredCommutatorError("release coefficient is invalid")
    executed = (
        anchor.anchored_action_field
        if rho == 0.0
        else anchor.anchored_action_field
        + rho * bounded_commutator_correction
    )
    if not bool(torch.equal(executed[:, 0], frozen_noop_field[:, 0])):
        raise GaugeAnchoredCommutatorError(
            "gauge-anchored execution changed the no-op phase zero"
        )
    if rho == 0.0 and executed is not anchor.anchored_action_field:
        raise GaugeAnchoredCommutatorError(
            "rho-zero execution did not alias the gauge anchor"
        )
    return GaugeAnchoredExecution(
        anchor=anchor,
        bounded_commutator_correction=bounded_commutator_correction,
        rho=rho,
        executed_clean_field=executed,
    )


def project_complete_action_quotient(
    *,
    frozen_action_field: Any,
    frozen_noop_field: Any,
    adapted_action_field: Any,
    adapted_noop_field: Any,
    config: FeasibleQuotientConfig = FeasibleQuotientConfig(),
) -> FeasibleQuotientProjection:
    """Project the *complete* adapted quotient into a zero-centered ball.

    Unlike v7, the frozen quotient is not a coefficient-one bypass plus a
    small residual ball.  It only defines one source-side radius term.  The
    second term derives from frozen no-op dynamics, so a weak or wrong frozen
    action can be rotated, suppressed, or replaced without an inference-only
    oracle.
    """

    torch = _require_torch()
    config.validate()
    _validate_pair(frozen_action_field, frozen_noop_field)
    _validate_pair(adapted_action_field, adapted_noop_field)
    if tuple(adapted_action_field.shape) != tuple(frozen_action_field.shape):
        raise GaugeAnchoredCommutatorError(
            "frozen and adapted quotient fields have different geometry"
        )
    try:
        frozen_quotient = commutator.causal_gauge(
            frozen_action_field - frozen_noop_field
        )
        adapted_quotient = commutator.causal_gauge(
            adapted_action_field - adapted_noop_field
        )
        noop_motion = commutator.causal_gauge(frozen_noop_field)
        frozen_increments = commutator.phase_increments(frozen_quotient)
        adapted_increments = commutator.phase_increments(adapted_quotient)
        noop_increments = commutator.phase_increments(noop_motion)
    except commutator.MotionCommutatorError as error:
        raise GaugeAnchoredCommutatorError(str(error)) from error
    frozen_smoothed = smooth_phase_increments(frozen_increments)
    adapted_smoothed = smooth_phase_increments(adapted_increments)
    noop_smoothed = smooth_phase_increments(noop_increments)
    frozen_rms = phase_rms(frozen_smoothed)
    adapted_rms = phase_rms(adapted_smoothed)
    noop_rms = phase_rms(noop_smoothed)
    semantic_radius = float(config.frozen_quotient_radius_ratio) * frozen_rms
    noop_radius = float(config.noop_dynamics_radius_ratio) * noop_rms
    floor = torch.full_like(semantic_radius, float(config.radius_floor))
    radius = torch.maximum(torch.maximum(semantic_radius, noop_radius), floor)
    bounded_increments, scale = radial_project_phase_increments(
        adapted_smoothed,
        radius,
        epsilon=float(config.epsilon),
    )
    bounded_rms = phase_rms(bounded_increments)
    tolerance = max(
        float(config.epsilon), 8.0 * float(torch.finfo(torch.float32).eps)
    )
    if bool((bounded_rms > radius + tolerance).any()):
        raise GaugeAnchoredCommutatorError(
            "complete action quotient violated its source-only trust radius"
        )
    bounded_quotient = integrate_phase_increments(bounded_increments)
    return FeasibleQuotientProjection(
        bounded_quotient=bounded_quotient,
        diagnostics=FeasibleQuotientDiagnostics(
            frozen_quotient=frozen_quotient,
            adapted_quotient=adapted_quotient,
            noop_motion_direction=noop_motion,
            adapted_raw_increments=adapted_increments,
            adapted_smoothed_increments=adapted_smoothed,
            frozen_smoothed_increment_rms=frozen_rms,
            noop_smoothed_increment_rms=noop_rms,
            adapted_smoothed_increment_rms=adapted_rms,
            radius=radius,
            scale=scale,
            bounded_increments=bounded_increments,
            bounded_increment_rms=bounded_rms,
        ),
    )


def execute_feasible_quotient_transport(
    frozen_noop_field: Any,
    bounded_quotient: Any,
    *,
    step_index: int,
) -> FeasibleQuotientExecution:
    """Lift a bounded quotient through the frozen no-op reconstruction section."""

    torch = _require_torch()
    _validate_pair(frozen_noop_field, frozen_noop_field)
    if (
        not isinstance(bounded_quotient, torch.Tensor)
        or tuple(bounded_quotient.shape) != tuple(frozen_noop_field.shape)
        or bounded_quotient.dtype != frozen_noop_field.dtype
        or bounded_quotient.device != frozen_noop_field.device
        or not bool(torch.isfinite(bounded_quotient).all())
        or not bool(torch.equal(
            bounded_quotient[:, 0], torch.zeros_like(bounded_quotient[:, 0])
        ))
    ):
        raise GaugeAnchoredCommutatorError(
            "bounded full quotient differs from the no-op field contract"
        )
    try:
        rho = float(commutator.release_rho(step_index))
    except commutator.MotionCommutatorError as error:
        raise GaugeAnchoredCommutatorError(str(error)) from error
    executed = (
        frozen_noop_field
        if rho == 0.0
        else frozen_noop_field + rho * bounded_quotient
    )
    if not bool(torch.equal(executed[:, 0], frozen_noop_field[:, 0])):
        raise GaugeAnchoredCommutatorError(
            "feasible quotient transport changed the no-op phase zero"
        )
    if rho == 0.0 and executed is not frozen_noop_field:
        raise GaugeAnchoredCommutatorError(
            "rho-zero feasible transport did not alias frozen no-op"
        )
    return FeasibleQuotientExecution(
        frozen_noop_field=frozen_noop_field,
        bounded_quotient=bounded_quotient,
        rho=rho,
        executed_clean_field=executed,
    )


__all__ = [
    "APPEARANCE_CARRIER",
    "DEFAULT_EPSILON",
    "DEFAULT_FROZEN_QUOTIENT_RADIUS_RATIO",
    "DEFAULT_NOOP_DYNAMICS_RADIUS_RATIO",
    "DEFAULT_RADIUS_FLOOR",
    "FULL_QUOTIENT_OPERATOR",
    "FeasibleQuotientConfig",
    "FeasibleQuotientDiagnostics",
    "FeasibleQuotientExecution",
    "FeasibleQuotientProjection",
    "GaugeAnchoredCommutatorError",
    "GaugeAnchoredExecution",
    "METHOD_NAME",
    "NoopGaugeAnchor",
    "build_noop_gauge_anchor",
    "execute_gauge_anchored_commutator",
    "execute_feasible_quotient_transport",
    "integrate_phase_increments",
    "phase_rms",
    "project_complete_action_quotient",
    "radial_project_phase_increments",
    "smooth_phase_increments",
]
