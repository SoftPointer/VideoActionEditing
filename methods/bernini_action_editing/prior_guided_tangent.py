#!/usr/bin/env python3
"""Pure-tensor core for Prior-Guided Tangent Trust-Region LoRA v5.

The operator consumes already reconstructed FP32 clean fields.  APG/model
forward logic intentionally lives outside this module so training and
inference can call the exact same numerical program.

For frozen base action/no-op fields ``A0`` and ``N0``, first form the raw
difference ``B_raw = A0 - N0`` and then the executable causal prior
``B = Q0(B_raw)``.  The adapter is allowed to contribute only a causal-gauge
correction ``R = Q0(A_theta - A0)``.  A phasewise smooth trust region keeps
correction parallel to ``B`` within ``kappa_parallel`` and bounds orthogonal
innovation relative to the RMS of ``B``.  The executed field is
``B + gamma(step) * C_B(R)`` and therefore has an exact zero first phase.

PyTorch is imported lazily.  Consequently schedule/configuration contract
tests can import this file in environments where torch is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


METHOD_NAME = "prior-guided-tangent-trust-region-lora-v5"
NUM_DENOISING_STEPS = 40
FULL_CORRECTION_LAST_STEP = 23
TAPER_FIRST_STEP = 24
TAPER_LAST_STEP = 34
FROZEN_REPLAY_FIRST_STEP = 35
DEFAULT_KAPPA_PARALLEL = 0.5
DEFAULT_KAPPA_PERP = 0.15
DEFAULT_EPSILON = 1.0e-6
DEFAULT_PHASE_DIM = 1


class PriorGuidedTangentError(RuntimeError):
    """Raised when a v5 tensor or schedule invariant differs."""


@dataclass(frozen=True)
class TangentTrustRegionConfig:
    """Fixed main-arm trust-region hyperparameters."""

    kappa_parallel: float = DEFAULT_KAPPA_PARALLEL
    kappa_perp: float = DEFAULT_KAPPA_PERP
    epsilon: float = DEFAULT_EPSILON
    phase_dim: int = DEFAULT_PHASE_DIM

    def validate(self) -> None:
        for name in ("kappa_parallel", "kappa_perp"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise PriorGuidedTangentError(f"{name} must be finite and non-negative")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as error:
                raise PriorGuidedTangentError(
                    f"{name} must be finite and non-negative"
                ) from error
            if not math.isfinite(numeric) or numeric < 0.0:
                raise PriorGuidedTangentError(
                    f"{name} must be finite and non-negative"
                )
        if isinstance(self.epsilon, bool):
            raise PriorGuidedTangentError("epsilon must be finite and positive")
        try:
            epsilon = float(self.epsilon)
        except (TypeError, ValueError) as error:
            raise PriorGuidedTangentError(
                "epsilon must be finite and positive"
            ) from error
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise PriorGuidedTangentError("epsilon must be finite and positive")
        if type(self.phase_dim) is not int or self.phase_dim == 0:
            raise PriorGuidedTangentError(
                "phase_dim must be an integer distinct from batch dimension zero"
            )


@dataclass(frozen=True)
class TrustRegionResult:
    """Auditable components of one phasewise trust-region projection."""

    motion_reference: Any
    raw_correction: Any
    raw_parallel_coefficient: Any
    parallel_correction: Any
    raw_perpendicular_correction: Any
    bounded_perpendicular_correction: Any
    trusted_correction: Any


@dataclass(frozen=True)
class ExecutedFieldResult:
    """Causal frozen prior plus a scheduled, trust-region-bounded correction."""

    prior: Any
    gamma: float
    trust_region: TrustRegionResult
    executed_field: Any


def correction_gamma(step_index: int) -> float:
    """Return the exact fixed 40-step correction schedule.

    Steps 0--23 are one; steps 24--34 follow an inclusive cosine taper from
    one to zero; and steps 35--39 are exactly zero.
    """

    if type(step_index) is not int or not 0 <= step_index < NUM_DENOISING_STEPS:
        raise PriorGuidedTangentError("step_index must be an integer in [0,40)")
    if step_index <= FULL_CORRECTION_LAST_STEP:
        return 1.0
    if step_index >= FROZEN_REPLAY_FIRST_STEP:
        return 0.0
    progress = (step_index - TAPER_FIRST_STEP) / (
        TAPER_LAST_STEP - TAPER_FIRST_STEP
    )
    value = 0.5 * (1.0 + math.cos(math.pi * progress))
    if step_index == TAPER_FIRST_STEP:
        return 1.0
    if step_index == TAPER_LAST_STEP:
        return 0.0
    return float(value)


def correction_gamma_schedule() -> tuple[float, ...]:
    """Materialize all 40 immutable gamma values."""

    return tuple(correction_gamma(step) for step in range(NUM_DENOISING_STEPS))


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised without torch
        raise PriorGuidedTangentError(
            "prior-guided tangent tensor operations require PyTorch"
        ) from error
    return torch


def _canonical_phase_dim(tensor: Any, phase_dim: int) -> int:
    ndim = getattr(tensor, "ndim", None)
    if type(ndim) is not int or ndim < 3:
        raise PriorGuidedTangentError(
            "fields must have shape [batch,phase,...] with at least three axes"
        )
    resolved = phase_dim if phase_dim >= 0 else ndim + phase_dim
    if not 0 <= resolved < ndim or resolved == 0:
        raise PriorGuidedTangentError("phase_dim is invalid for the field rank")
    if int(tensor.shape[0]) <= 0 or int(tensor.shape[resolved]) <= 0:
        raise PriorGuidedTangentError("batch and phase axes must be non-empty")
    return resolved


def _validate_fields(*fields: Any, phase_dim: int) -> int:
    torch = _require_torch()
    if not fields:
        raise PriorGuidedTangentError("at least one field is required")
    reference = fields[0]
    if not isinstance(reference, torch.Tensor):
        raise PriorGuidedTangentError("fields must be torch tensors")
    resolved = _canonical_phase_dim(reference, phase_dim)
    if reference.dtype != torch.float32:
        raise PriorGuidedTangentError("v5 clean fields must be exact torch.float32")
    for field in fields:
        if not isinstance(field, torch.Tensor):
            raise PriorGuidedTangentError("fields must be torch tensors")
        if tuple(field.shape) != tuple(reference.shape):
            raise PriorGuidedTangentError("field shapes differ")
        if field.device != reference.device or field.dtype != reference.dtype:
            raise PriorGuidedTangentError("field device/dtype differs")
        if not bool(torch.isfinite(field).all()):
            raise PriorGuidedTangentError("field contains non-finite values")
    return resolved


def q0(field: Any, *, phase_dim: int = DEFAULT_PHASE_DIM) -> Any:
    """Apply the idempotent causal gauge ``Q0(d)_t = d_t - d_0``."""

    resolved = _validate_fields(field, phase_dim=phase_dim)
    first_phase = field.select(resolved, 0).unsqueeze(resolved)
    projected = field - first_phase
    torch = _require_torch()
    if not bool(
        torch.equal(
            projected.select(resolved, 0),
            torch.zeros_like(projected.select(resolved, 0)),
        )
    ):
        raise PriorGuidedTangentError("Q0 did not produce an exact zero first phase")
    return projected


def raw_frozen_prior(
    base_action_field: Any,
    base_noop_field: Any,
    *,
    phase_dim: int = DEFAULT_PHASE_DIM,
) -> Any:
    """Return the raw frozen guided difference ``B_raw = A0 - N0``."""

    _validate_fields(base_action_field, base_noop_field, phase_dim=phase_dim)
    return base_action_field - base_noop_field


def frozen_prior(
    base_action_field: Any,
    base_noop_field: Any,
    *,
    phase_dim: int = DEFAULT_PHASE_DIM,
) -> Any:
    """Return the executable causal prior ``B = Q0(A0 - N0)``."""

    return q0(
        raw_frozen_prior(
            base_action_field,
            base_noop_field,
            phase_dim=phase_dim,
        ),
        phase_dim=phase_dim,
    )


def adapter_correction(
    adapted_action_field: Any,
    base_action_field: Any,
    *,
    phase_dim: int = DEFAULT_PHASE_DIM,
) -> Any:
    """Return ``R_theta = Q0(A_theta - A0)`` for one-sided action LoRA."""

    _validate_fields(adapted_action_field, base_action_field, phase_dim=phase_dim)
    return q0(adapted_action_field - base_action_field, phase_dim=phase_dim)


def teacher_correction(
    source: Any,
    target: Any,
    prior: Any,
    *,
    phase_dim: int = DEFAULT_PHASE_DIM,
) -> Any:
    """Return teacher input ``R* = Q0((T-S)-B)`` for causal prior ``B``."""

    _validate_fields(source, target, prior, phase_dim=phase_dim)
    return q0((target - source) - prior, phase_dim=phase_dim)


def phasewise_trust_region(
    prior: Any,
    raw_correction: Any,
    config: TangentTrustRegionConfig = TangentTrustRegionConfig(),
) -> TrustRegionResult:
    """Apply the smooth phasewise parallel/perpendicular trust region.

    All non-batch, non-phase axes are one vector.  The parallel coefficient is
    smoothly limited by ``kappa_parallel * tanh(a/kappa_parallel)``.  The
    perpendicular RMS is smoothly bounded by
    ``kappa_perp * RMS(B) + epsilon`` for causal prior ``B``.
    """

    config.validate()
    phase_dim = _validate_fields(prior, raw_correction, phase_dim=config.phase_dim)
    torch = _require_torch()
    motion_reference = q0(prior, phase_dim=phase_dim)
    correction = q0(raw_correction, phase_dim=phase_dim)
    reference_canonical = motion_reference.movedim(phase_dim, 1)
    correction_canonical = correction.movedim(phase_dim, 1)
    reduction_axes = tuple(range(2, reference_canonical.ndim))
    broadcast_shape = (*reference_canonical.shape[:2],) + (1,) * len(reduction_axes)

    dot = (correction_canonical * reference_canonical).sum(
        dim=reduction_axes, keepdim=False
    )
    reference_energy = reference_canonical.square().sum(
        dim=reduction_axes, keepdim=False
    )
    coefficient = dot / (reference_energy + float(config.epsilon))
    if float(config.kappa_parallel) == 0.0:
        bounded_coefficient = torch.zeros_like(coefficient)
    else:
        bound = float(config.kappa_parallel)
        bounded_coefficient = bound * torch.tanh(coefficient / bound)
    parallel = bounded_coefficient.reshape(broadcast_shape) * reference_canonical

    raw_parallel = coefficient.reshape(broadcast_shape) * reference_canonical
    raw_perpendicular = correction_canonical - raw_parallel
    if float(config.kappa_perp) == 0.0:
        bounded_perpendicular = torch.zeros_like(raw_perpendicular)
    else:
        vector_width = math.prod(
            int(reference_canonical.shape[axis]) for axis in reduction_axes
        )
        rms_denominator = math.sqrt(vector_width)
        # vector_norm defines the zero-vector gradient as zero.  The equivalent
        # sqrt(mean(square(.))) expression yields NaN gradients at exact zero,
        # which is precisely the adapter initialization state.
        perpendicular_rms = torch.linalg.vector_norm(
            raw_perpendicular,
            ord=2,
            dim=reduction_axes,
            keepdim=True,
        ) / rms_denominator
        reference_rms = torch.linalg.vector_norm(
            reference_canonical,
            ord=2,
            dim=reduction_axes,
            keepdim=True,
        ) / rms_denominator
        cap = float(config.kappa_perp) * reference_rms + float(config.epsilon)
        bounded_perpendicular = raw_perpendicular / torch.sqrt(
            1.0 + (perpendicular_rms / cap).square()
        )
    trusted = parallel + bounded_perpendicular

    def restore(value: Any) -> Any:
        return value.movedim(1, phase_dim)

    trusted_restored = restore(trusted)
    first = trusted_restored.select(phase_dim, 0)
    if not bool(torch.equal(first, torch.zeros_like(first))):
        raise PriorGuidedTangentError(
            "trust-region correction changed the exact phase-zero boundary"
        )
    if not bool(torch.isfinite(trusted_restored).all()):
        raise PriorGuidedTangentError("trust-region correction is non-finite")
    return TrustRegionResult(
        motion_reference=motion_reference,
        raw_correction=correction,
        raw_parallel_coefficient=coefficient,
        parallel_correction=restore(parallel),
        raw_perpendicular_correction=restore(raw_perpendicular),
        bounded_perpendicular_correction=restore(bounded_perpendicular),
        trusted_correction=trusted_restored,
    )


def execute_prior_guided_field(
    prior: Any,
    raw_correction: Any,
    *,
    step_index: int,
    config: TangentTrustRegionConfig = TangentTrustRegionConfig(),
) -> ExecutedFieldResult:
    """Execute ``B + gamma(step) * C_B(R)`` for an exact causal prior ``B``."""

    phase_dim = _validate_fields(
        prior,
        raw_correction,
        phase_dim=config.phase_dim,
    )
    torch = _require_torch()
    prior_phase_zero = prior.select(phase_dim, 0)
    if not bool(torch.equal(prior_phase_zero, torch.zeros_like(prior_phase_zero))):
        raise PriorGuidedTangentError(
            "executed prior must have an exact zero first phase"
        )
    trust = phasewise_trust_region(prior, raw_correction, config)
    gamma = correction_gamma(step_index)
    # Avoid even a multiply/add at frozen replay steps.  This makes steps
    # 35--39 exact aliases of the frozen prior and intentionally removes all
    # adapter gradient from late detail refinement.
    executed = prior if gamma == 0.0 else prior + gamma * trust.trusted_correction
    executed_phase_zero = executed.select(phase_dim, 0)
    if not bool(
        torch.equal(executed_phase_zero, torch.zeros_like(executed_phase_zero))
    ):
        raise PriorGuidedTangentError(
            "executed motion changed the exact phase-zero source boundary"
        )
    return ExecutedFieldResult(
        prior=prior,
        gamma=gamma,
        trust_region=trust,
        executed_field=executed,
    )


def student_executed_field(
    base_action_field: Any,
    base_noop_field: Any,
    adapted_action_field: Any,
    *,
    step_index: int,
    config: TangentTrustRegionConfig = TangentTrustRegionConfig(),
) -> Any:
    """Return student ``E_theta`` from frozen and adapted action fields."""

    prior = frozen_prior(
        base_action_field,
        base_noop_field,
        phase_dim=config.phase_dim,
    )
    correction = adapter_correction(
        adapted_action_field,
        base_action_field,
        phase_dim=config.phase_dim,
    )
    return execute_prior_guided_field(
        prior,
        correction,
        step_index=step_index,
        config=config,
    ).executed_field


def teacher_executed_field(
    source: Any,
    target: Any,
    base_action_field: Any,
    base_noop_field: Any,
    *,
    step_index: int,
    config: TangentTrustRegionConfig = TangentTrustRegionConfig(),
) -> Any:
    """Return teacher ``E*`` through the same shared execution operator."""

    prior = frozen_prior(
        base_action_field,
        base_noop_field,
        phase_dim=config.phase_dim,
    )
    correction = teacher_correction(
        source,
        target,
        prior,
        phase_dim=config.phase_dim,
    )
    return execute_prior_guided_field(
        prior,
        correction,
        step_index=step_index,
        config=config,
    ).executed_field
