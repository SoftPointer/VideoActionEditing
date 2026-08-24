#!/usr/bin/env python3
"""Parameter-free signed-relational motion score and nuisance-null interventions.

The scorer combines two views of one same-state ``action - no-op`` hidden
residual.  A normalized ``21x21`` temporal Gram is basis-free and records
phase relations, but is unsigned: by itself it cannot distinguish ``R`` from
``-R``.  A second, spatial-orderless temporal quotient keeps the shared
Bernini channel sign through signed spatial means, then adds lag-1/2/4,
start-to-terminal and hold features.  The signed term removes the Gram sign
ambiguity without requiring teacher/current pixel correspondence.

The teacher is converted to a detached registered buffer at construction.
The current residual remains differentiable, and the module owns no trainable
parameters.  Generated teacher media, source masks, tracks, poses and flows
are outside this tensor-only boundary.

The second half of the module projects one exact81 clean-latent cotangent onto
a fixed nuisance-null linear subspace.  Phase zero is exactly zero; the remaining
20 phases have zero temporal mean; and every channel/phase spatial map is
orthogonal to the fixed affine basis ``{1, x, y}``.  A fixed-RMS symmetric
``+/-`` intervention can then be constructed without choosing between arms.
These algebraic constraints do not prove source identity or camera retention;
decoded exact81 gates remain authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn


LATENT_PHASES = 21
LATENT_CHANNELS = 16
_MIN_ENERGY = 1.0e-12
MIN_TEMPORAL_RESIDUAL_RMS = 1.0e-6
MIN_DYNAMIC_FRACTION = 1.0e-3
MIN_SIGNED_FRACTION = 0.05
MIN_MEANINGFUL_MISMATCH = 1.0e-2
MIN_PROJECTION_SURVIVAL_RATIO = 0.10
TEMPORAL_LAGS = (1, 2, 4)
HOLD_PHASES = 4
SIGNED_LOSS_WEIGHT = 0.5
MAGNITUDE_LOSS_WEIGHT = 0.25
RELATIONAL_LOSS_WEIGHT = 0.25
GRAM_STABILIZER_FRACTION = 1.0e-3
_CONSTRAINT_REFINEMENT_STEPS = 3


class RelationalMotionError(RuntimeError):
    """A tensor cannot satisfy the closed relational-motion contract."""


def _finite_positive_scalar(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RelationalMotionError(f"{label} must be a finite positive scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RelationalMotionError(f"{label} must be a finite positive scalar")
    return result


def _validate_residual(
    value: Any,
    *,
    label: str,
    expected_shape: tuple[int, int, int, int] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise RelationalMotionError(f"{label} must be a torch.Tensor")
    if value.layout != torch.strided or value.device.type == "meta":
        raise RelationalMotionError(f"{label} must be dense and materialized")
    if value.dtype != torch.float32 or value.ndim != 4:
        raise RelationalMotionError(f"{label} must be FP32 [B,21,K,D]")
    shape = tuple(int(item) for item in value.shape)
    if (
        shape[0] <= 0
        or shape[1] != LATENT_PHASES
        or shape[2] <= 0
        or shape[3] <= 0
    ):
        raise RelationalMotionError(f"{label} must be nonempty FP32 [B,21,K,D]")
    if expected_shape is not None and shape != expected_shape:
        raise RelationalMotionError(
            f"{label} shape {shape} differs from required {expected_shape}"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise RelationalMotionError(f"{label} contains NaN or infinity")
    return value


def temporal_dc_residual(value: torch.Tensor) -> torch.Tensor:
    """Remove the exact phase-axis DC component from ``[B,21,K,D]``."""

    tensor = _validate_residual(value, label="sketched residual")
    return tensor - tensor.mean(dim=1, keepdim=True)


def _temporal_residual_rms(value: torch.Tensor) -> torch.Tensor:
    centered = temporal_dc_residual(value)
    return centered.to(dtype=torch.float64).square().mean(dim=(1, 2, 3)).sqrt()


def _packetize_phase_feature(phase_feature: torch.Tensor) -> torch.Tensor:
    phase_feature = phase_feature - phase_feature.mean(dim=1, keepdim=True)
    components = [
        phase_feature.reshape(phase_feature.shape[0], -1)
        / math.sqrt(float(LATENT_PHASES))
    ]
    for lag in TEMPORAL_LAGS:
        prefix = torch.zeros_like(phase_feature[:, :lag])
        difference = torch.cat(
            (prefix, phase_feature[:, lag:] - phase_feature[:, :-lag]), dim=1
        )
        components.append(
            difference.reshape(difference.shape[0], -1)
            / math.sqrt(float(LATENT_PHASES - lag))
        )

    initial = phase_feature[:, :HOLD_PHASES]
    terminal = phase_feature[:, -HOLD_PHASES:]
    initial_mean = initial.mean(dim=1, keepdim=True)
    terminal_mean = terminal.mean(dim=1, keepdim=True)
    boundary = terminal_mean - initial_mean
    components.extend(
        (
            boundary.reshape(boundary.shape[0], -1),
            0.5
            * (initial - initial_mean).reshape(initial.shape[0], -1)
            / math.sqrt(float(HOLD_PHASES)),
            0.5
            * (terminal - terminal_mean).reshape(terminal.shape[0], -1)
            / math.sqrt(float(HOLD_PHASES)),
        )
    )
    feature = torch.cat(components, dim=1).float().contiguous()
    if not bool(torch.isfinite(feature).all().item()):
        raise RelationalMotionError("temporal motion packet is non-finite")
    return feature


def signed_temporal_motion_packets(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return separate signed and magnitude packets after spatial pooling."""

    tensor = _validate_residual(value, label="sketched residual")
    centered_hidden = temporal_dc_residual(tensor)
    signed_mean = centered_hidden.mean(dim=2)
    rms_epsilon = 1.0e-12
    centered_rms = torch.sqrt(
        centered_hidden.square().mean(dim=2) + rms_epsilon
    ) - math.sqrt(rms_epsilon)
    return (
        _packetize_phase_feature(signed_mean),
        _packetize_phase_feature(centered_rms),
    )


def signed_temporal_motion_feature(value: torch.Tensor) -> torch.Tensor:
    """Return the concatenated signed/magnitude packets for diagnostics."""

    signed, magnitude = signed_temporal_motion_packets(value)
    return torch.cat((signed, magnitude), dim=1).contiguous()


def _raw_temporal_gram(value: torch.Tensor) -> torch.Tensor:
    centered = temporal_dc_residual(value)
    flattened = centered.reshape(centered.shape[0], centered.shape[1], -1)
    return torch.bmm(flattened, flattened.transpose(1, 2))


def normalized_temporal_gram(value: torch.Tensor) -> torch.Tensor:
    """Return a unit-Frobenius, identity-basis-invariant temporal Gram.

    The flattened feature axis may be transformed by any single orthogonal
    matrix shared by all phases and batches without changing the result.
    Degenerate residuals with no temporal energy fail closed because a zero
    Gram cannot define a normalized motion relation.
    """

    centered = temporal_dc_residual(value)
    batch, phases, sketches, channels = centered.shape
    flattened = centered.reshape(batch, phases, sketches * channels)
    gram = torch.bmm(flattened, flattened.transpose(1, 2))
    norm = torch.linalg.vector_norm(gram, ord=2, dim=(-2, -1), keepdim=True)
    if bool((norm.detach() <= _MIN_ENERGY).any().item()):
        raise RelationalMotionError(
            "sketched residual has no nondegenerate temporal Gram energy"
        )
    normalized = gram / norm
    if not bool(torch.isfinite(normalized).all().item()):
        raise RelationalMotionError("normalized temporal Gram is non-finite")
    return normalized


@dataclass(frozen=True)
class RelationalMotionScore:
    """Differentiable score plus compact audit diagnostics."""

    score: torch.Tensor
    per_sample_score: torch.Tensor
    normalized_frobenius_mismatch: torch.Tensor
    current_temporal_gram: torch.Tensor
    signed_feature_loss: torch.Tensor
    magnitude_feature_loss: torch.Tensor
    relational_gram_loss: torch.Tensor
    objective_mismatch: torch.Tensor
    meaningful_mismatch: torch.Tensor


class FrozenRelationalMotionScorer(nn.Module):
    """Compare current motion with one frozen signed-relational teacher.

    ``teacher_residual`` is detached immediately.  The module retains a
    stabilized temporal Gram plus signed and magnitude temporal packets, never
    the teacher residual itself.  The signed packet uses Bernini's shared
    channel basis and may still contain temporal identity correlations; it is
    an action hypothesis, not an identity-free guarantee. ``parameters()`` is
    empty.
    """

    def __init__(self, teacher_residual: torch.Tensor) -> None:
        super().__init__()
        self.last_score_components: dict[str, float | bool] | None = None
        teacher = _validate_residual(
            teacher_residual, label="teacher sketched residual"
        )
        self.expected_shape = tuple(int(item) for item in teacher.shape)
        with torch.no_grad():
            detached = teacher.detach().clone().contiguous()
            teacher_signed, teacher_magnitude = signed_temporal_motion_packets(
                detached
            )
            teacher_signed_norm = torch.linalg.vector_norm(
                teacher_signed, ord=2, dim=1, keepdim=True
            )
            teacher_magnitude_norm = torch.linalg.vector_norm(
                teacher_magnitude, ord=2, dim=1, keepdim=True
            )
            teacher_temporal_rms = _temporal_residual_rms(teacher)
            teacher_raw_rms = teacher.to(torch.float64).square().mean(
                dim=(1, 2, 3)
            ).sqrt()
            centered = temporal_dc_residual(teacher)
            signed_mean_rms = centered.mean(dim=2).to(torch.float64).square().mean(
                dim=(1, 2)
            ).sqrt()
            dynamic_fraction = teacher_temporal_rms / torch.clamp(
                teacher_raw_rms, min=_MIN_ENERGY
            )
            signed_fraction = signed_mean_rms / torch.clamp(
                teacher_temporal_rms, min=_MIN_ENERGY
            )
            if bool(
                (teacher_signed_norm <= _MIN_ENERGY).any().item()
                or (teacher_temporal_rms < MIN_TEMPORAL_RESIDUAL_RMS).any().item()
                or (dynamic_fraction < MIN_DYNAMIC_FRACTION).any().item()
                or (signed_fraction < MIN_SIGNED_FRACTION).any().item()
            ):
                raise RelationalMotionError(
                    "teacher residual has insufficient signed temporal energy"
                )
            teacher_raw_gram = _raw_temporal_gram(detached)
            teacher_gram_norm = torch.linalg.vector_norm(
                teacher_raw_gram, ord=2, dim=(-2, -1), keepdim=True
            )
            gram_gamma = GRAM_STABILIZER_FRACTION * teacher_gram_norm
            teacher_stabilized_gram = teacher_raw_gram / torch.sqrt(
                teacher_gram_norm.square() + gram_gamma.square()
            )
            magnitude_scale = torch.maximum(
                teacher_magnitude_norm, 0.1 * teacher_signed_norm
            )
        self.register_buffer(
            "teacher_temporal_gram",
            teacher_stabilized_gram.detach().clone().contiguous(),
            persistent=True,
        )
        self.register_buffer(
            "teacher_signed_feature",
            teacher_signed.detach().clone().contiguous(),
            persistent=True,
        )
        self.register_buffer(
            "teacher_magnitude_feature",
            teacher_magnitude.detach().clone().contiguous(),
            persistent=True,
        )
        self.register_buffer(
            "teacher_signed_norm",
            teacher_signed_norm.detach().clone().contiguous(),
            persistent=True,
        )
        self.register_buffer(
            "teacher_magnitude_scale",
            magnitude_scale.detach().clone().contiguous(),
            persistent=True,
        )
        self.register_buffer(
            "gram_gamma",
            gram_gamma.detach().clone().contiguous(),
            persistent=True,
        )

    def forward_sketched_residual(
        self,
        current: torch.Tensor,
        *,
        require_input_grad: bool = True,
    ) -> RelationalMotionScore:
        current_value = _validate_residual(
            current,
            label="current sketched residual",
            expected_shape=self.expected_shape,
        )
        if current_value.device != self.teacher_temporal_gram.device:
            raise RelationalMotionError(
                "current residual and frozen teacher Gram must share one device"
            )
        if require_input_grad and not current_value.requires_grad:
            raise RelationalMotionError(
                "current residual must require gradients for a live score VJP"
            )
        if require_input_grad and current_value.grad_fn is None:
            raise RelationalMotionError(
                "current residual must remain connected to the live Bernini graph"
            )
        current_temporal_rms = _temporal_residual_rms(current_value)
        if bool((current_temporal_rms < MIN_TEMPORAL_RESIDUAL_RMS).any().item()):
            raise RelationalMotionError(
                "current residual has insufficient temporal energy"
            )
        current_raw_gram = _raw_temporal_gram(current_value)
        current_gram_norm = torch.linalg.vector_norm(
            current_raw_gram, ord=2, dim=(-2, -1), keepdim=True
        )
        current_gram = current_raw_gram / torch.sqrt(
            current_gram_norm.square() + self.gram_gamma.detach().square()
        )
        teacher_gram = self.teacher_temporal_gram.detach()
        gram_delta = current_gram - teacher_gram
        gram_loss = 0.5 * gram_delta.square().sum(dim=(-2, -1))
        mismatch = torch.sqrt(torch.clamp(2.0 * gram_loss, min=0.0)) / math.sqrt(2.0)

        current_signed, current_magnitude = signed_temporal_motion_packets(
            current_value
        )
        signed_loss = (
            (current_signed - self.teacher_signed_feature.detach())
            .square()
            .sum(dim=1)
            / (4.0 * self.teacher_signed_norm.detach().square().reshape(-1))
        )
        magnitude_loss = (
            (current_magnitude - self.teacher_magnitude_feature.detach())
            .square()
            .sum(dim=1)
            / (4.0 * self.teacher_magnitude_scale.detach().square().reshape(-1))
        )
        objective_mismatch = (
            RELATIONAL_LOSS_WEIGHT * gram_loss
            + MAGNITUDE_LOSS_WEIGHT * magnitude_loss
            + SIGNED_LOSS_WEIGHT * signed_loss
        )
        meaningful_mismatch = torch.sqrt(torch.clamp(objective_mismatch, min=0.0))
        if require_input_grad and bool(
            (meaningful_mismatch.detach() < MIN_MEANINGFUL_MISMATCH).any().item()
        ):
            raise RelationalMotionError(
                "current residual already matches teacher below the meaningful-mismatch floor"
            )
        per_sample_score = -objective_mismatch
        score = per_sample_score.mean()
        if score.ndim != 0 or not bool(torch.isfinite(score).item()):
            raise RelationalMotionError("relational motion score is non-finite")
        self.last_score_components = {
            "score": float(score.detach().item()),
            "objective_mismatch": float(objective_mismatch.detach().mean().item()),
            "meaningful_mismatch": float(meaningful_mismatch.detach().mean().item()),
            "signed_feature_loss": float(signed_loss.detach().mean().item()),
            "magnitude_feature_loss": float(magnitude_loss.detach().mean().item()),
            "relational_gram_loss": float(gram_loss.detach().mean().item()),
            "intervention_allowed": bool(
                (meaningful_mismatch.detach() >= MIN_MEANINGFUL_MISMATCH)
                .all()
                .item()
            ),
        }
        return RelationalMotionScore(
            score=score,
            per_sample_score=per_sample_score,
            normalized_frobenius_mismatch=mismatch,
            current_temporal_gram=current_gram,
            signed_feature_loss=signed_loss,
            magnitude_feature_loss=magnitude_loss,
            relational_gram_loss=gram_loss,
            objective_mismatch=objective_mismatch,
            meaningful_mismatch=meaningful_mismatch,
        )

    def forward(
        self,
        current: torch.Tensor,
        *,
        require_input_grad: bool = True,
    ) -> RelationalMotionScore:
        return self.forward_sketched_residual(
            current, require_input_grad=require_input_grad
        )


def _validate_clean_latent(value: Any, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise RelationalMotionError(f"{label} must be a torch.Tensor")
    if value.layout != torch.strided or value.device.type == "meta":
        raise RelationalMotionError(f"{label} must be dense and materialized")
    if value.dtype != torch.float32 or value.ndim != 5:
        raise RelationalMotionError(
            f"{label} must be FP32 [1,16,21,H,W]"
        )
    shape = tuple(int(item) for item in value.shape)
    if (
        shape[0] != 1
        or shape[1] != LATENT_CHANNELS
        or shape[2] != LATENT_PHASES
        or shape[3] < 2
        or shape[4] < 2
    ):
        raise RelationalMotionError(
            f"{label} must be FP32 [1,16,21,H>=2,W>=2]"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise RelationalMotionError(f"{label} contains NaN or infinity")
    return value


def _orthonormal_affine_basis(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Construct the fixed orthonormal spatial basis ``{1,x,y}``."""

    x = torch.linspace(-1.0, 1.0, width, dtype=dtype, device=device)
    y = torch.linspace(-1.0, 1.0, height, dtype=dtype, device=device)
    x_grid = x.unsqueeze(0).expand(height, width)
    y_grid = y.unsqueeze(1).expand(height, width)
    basis = torch.stack(
        [torch.ones_like(x_grid), x_grid, y_grid], dim=0
    ).reshape(3, height * width)
    norms = torch.linalg.vector_norm(basis, ord=2, dim=1, keepdim=True)
    if bool((norms <= 0.0).any().item()):
        raise RelationalMotionError("fixed affine spatial basis is degenerate")
    normalized = basis / norms
    gram = normalized @ normalized.transpose(0, 1)
    tolerance = 1.0e-8 if dtype == torch.float64 else 1.0e-5
    if not bool(
        torch.allclose(
            gram,
            torch.eye(3, dtype=dtype, device=device),
            rtol=tolerance,
            atol=tolerance,
        )
    ):
        raise RelationalMotionError("fixed affine spatial basis is not orthonormal")
    return normalized


def _remove_spatial_affine(value: torch.Tensor) -> torch.Tensor:
    batch, channels, phases, height, width = value.shape
    flat = value.reshape(batch * channels * phases, height * width)
    basis = _orthonormal_affine_basis(
        height, width, device=value.device, dtype=value.dtype
    )
    coefficients = torch.matmul(flat, basis.transpose(0, 1))
    projected = flat - torch.matmul(coefficients, basis)
    return projected.reshape(batch, channels, phases, height, width)


def _zero_phase0_and_center_active_phases(value: torch.Tensor) -> torch.Tensor:
    """Closed-form projection onto ``q[0]=0`` and ``sum_t q[t]=0``.

    Once phase zero is fixed to zero, subtracting the mean of phases 1..20
    is the orthogonal projection onto their zero-sum subspace.  Concatenating
    an explicit zero tensor makes phase zero byte-exact rather than relying on
    multiplication by a mask.
    """

    active = value[:, :, 1:]
    active = active - active.mean(dim=2, keepdim=True)
    phase0 = torch.zeros_like(value[:, :, :1])
    return torch.cat([phase0, active], dim=2)


def _constraint_projection(value: torch.Tensor) -> torch.Tensor:
    result = _remove_spatial_affine(value)
    result = _zero_phase0_and_center_active_phases(result)
    return result


@dataclass(frozen=True)
class NuisanceProjectionDiagnostics:
    raw_rms: float
    projected_rms: float
    survival_ratio: float
    ascent_cosine: float
    phase0_max_abs: float
    temporal_sum_max_abs: float
    spatial_affine_max_abs_dot: float


def nuisance_projection_diagnostics(
    raw: torch.Tensor, projected: torch.Tensor
) -> NuisanceProjectionDiagnostics:
    raw_value = _validate_clean_latent(raw, label="raw clean-latent cotangent")
    projected_value = _validate_clean_latent(
        projected, label="projected clean-latent cotangent"
    )
    if tuple(raw_value.shape) != tuple(projected_value.shape):
        raise RelationalMotionError("raw/projected cotangent shapes differ")
    raw64 = raw_value.to(torch.float64)
    projected64 = projected_value.to(torch.float64)
    raw_norm = torch.linalg.vector_norm(raw64)
    projected_norm = torch.linalg.vector_norm(projected64)
    if (
        not bool(torch.isfinite(raw_norm).item())
        or not bool(torch.isfinite(projected_norm).item())
        or float(raw_norm.item()) <= _MIN_ENERGY
        or float(projected_norm.item()) <= _MIN_ENERGY
    ):
        raise RelationalMotionError("raw/projected cotangent is degenerate")
    survival_ratio = projected_norm / raw_norm
    ascent_cosine = torch.dot(raw64.reshape(-1), projected64.reshape(-1)) / (
        raw_norm * projected_norm
    )
    height, width = int(projected_value.shape[-2]), int(projected_value.shape[-1])
    basis = _orthonormal_affine_basis(
        height, width, device=projected_value.device, dtype=torch.float64
    )
    affine_dot = projected64.reshape(-1, height * width) @ basis.T
    return NuisanceProjectionDiagnostics(
        raw_rms=float(raw64.square().mean().sqrt().detach().item()),
        projected_rms=float(projected64.square().mean().sqrt().detach().item()),
        survival_ratio=float(survival_ratio.detach().item()),
        ascent_cosine=float(ascent_cosine.detach().item()),
        phase0_max_abs=float(projected_value[:, :, 0].abs().max().detach().item()),
        temporal_sum_max_abs=float(
            projected64.sum(dim=2).abs().max().detach().item()
        ),
        spatial_affine_max_abs_dot=float(affine_dot.abs().max().detach().item()),
    )


def project_source_safe_cotangent(
    q: torch.Tensor,
    *,
    minimum_survival_ratio: float = MIN_PROJECTION_SURVIVAL_RATIO,
) -> torch.Tensor:
    """Project one exact81 cotangent into a fixed nuisance-null subspace.

    The projection is evaluated once in FP64, then refined a fixed three times
    in FP32 to reduce cast-rounding residuals.  Spatial and temporal projectors
    act on different tensor axes and commute.  A fixed raw-to-projected
    survival floor prevents an almost fully removed numerical residue from
    being normalized back to a visible intervention.
    """

    value = _validate_clean_latent(q, label="clean-latent cotangent")
    survival_floor = _finite_positive_scalar(
        minimum_survival_ratio, label="minimum projection survival ratio"
    )
    if survival_floor > 1.0:
        raise RelationalMotionError(
            "minimum projection survival ratio must not exceed one"
        )
    projected = _constraint_projection(value.to(dtype=torch.float64))
    projected = projected.to(dtype=torch.float32)
    for _ in range(_CONSTRAINT_REFINEMENT_STEPS):
        projected = _constraint_projection(projected)

    projected = projected.contiguous()
    if not bool(torch.isfinite(projected).all().item()):
        raise RelationalMotionError("projected cotangent is non-finite")
    projected_rms = projected.to(torch.float64).square().mean().sqrt()
    if float(projected_rms.detach().item()) <= _MIN_ENERGY:
        raise RelationalMotionError(
            "nuisance projection removed the entire cotangent"
        )
    diagnostics = nuisance_projection_diagnostics(value, projected)
    if (
        diagnostics.survival_ratio < survival_floor
        or diagnostics.ascent_cosine < survival_floor
    ):
        raise RelationalMotionError(
            "clean-latent cotangent did not survive the nuisance projection"
        )
    if diagnostics.phase0_max_abs != 0.0:
        raise RelationalMotionError("phase-zero projection is not exact")
    if diagnostics.temporal_sum_max_abs > 3.0e-6:
        raise RelationalMotionError("temporal nuisance projection did not close")
    if diagnostics.spatial_affine_max_abs_dot > 3.0e-6:
        raise RelationalMotionError("spatial-affine nuisance projection did not close")
    return projected


@dataclass(frozen=True)
class SymmetricLatentInterventions:
    """One fixed-dose, non-selected pair around the same clean latent."""

    plus: torch.Tensor
    minus: torch.Tensor
    delta: torch.Tensor
    projected_cotangent: torch.Tensor
    projection_diagnostics: NuisanceProjectionDiagnostics
    dose_rms: float


def symmetric_latent_interventions(
    clean: torch.Tensor,
    q: torch.Tensor,
    *,
    dose_rms: float,
    minimum_survival_ratio: float = MIN_PROJECTION_SURVIVAL_RATIO,
) -> SymmetricLatentInterventions:
    """Construct exact paired ``clean +/- delta`` at one fixed RMS dose."""

    clean_value = _validate_clean_latent(clean, label="clean latent")
    q_value = _validate_clean_latent(q, label="clean-latent cotangent")
    if tuple(q_value.shape) != tuple(clean_value.shape) or q_value.device != clean_value.device:
        raise RelationalMotionError(
            "clean latent and cotangent must have identical shape and device"
        )
    dose = _finite_positive_scalar(dose_rms, label="intervention RMS dose")
    direction = project_source_safe_cotangent(
        q_value, minimum_survival_ratio=minimum_survival_ratio
    )
    projection_diagnostics = nuisance_projection_diagnostics(q_value, direction)
    direction_rms = direction.to(dtype=torch.float64).square().mean().sqrt()
    scale = (
        torch.as_tensor(dose, dtype=torch.float64, device=clean_value.device)
        / direction_rms
    ).to(dtype=torch.float32)
    delta = (direction * scale).contiguous()

    # One deterministic correction compensates the FP32 scale cast.  A scalar
    # rescale preserves phase-zero, temporal-zero and affine-null constraints.
    observed = delta.to(dtype=torch.float64).square().mean().sqrt()
    delta = (
        delta
        * (
            torch.as_tensor(dose, dtype=torch.float64, device=clean_value.device)
            / observed
        ).to(dtype=torch.float32)
    ).contiguous()
    plus = (clean_value + delta).contiguous()
    minus = (clean_value - delta).contiguous()
    if not (
        bool(torch.isfinite(delta).all().item())
        and bool(torch.isfinite(plus).all().item())
        and bool(torch.isfinite(minus).all().item())
    ):
        raise RelationalMotionError("fixed-RMS interventions are non-finite")
    actual_dose = float(
        delta.to(dtype=torch.float64).square().mean().sqrt().detach().item()
    )
    if not math.isclose(actual_dose, dose, rel_tol=1.0e-6, abs_tol=1.0e-8):
        raise RelationalMotionError("fixed-RMS intervention dose differs")
    return SymmetricLatentInterventions(
        plus=plus,
        minus=minus,
        delta=delta,
        projected_cotangent=direction,
        projection_diagnostics=projection_diagnostics,
        dose_rms=dose,
    )


__all__ = [
    "FrozenRelationalMotionScorer",
    "NuisanceProjectionDiagnostics",
    "RelationalMotionError",
    "RelationalMotionScore",
    "SymmetricLatentInterventions",
    "normalized_temporal_gram",
    "nuisance_projection_diagnostics",
    "project_source_safe_cotangent",
    "signed_temporal_motion_feature",
    "signed_temporal_motion_packets",
    "symmetric_latent_interventions",
    "temporal_dc_residual",
]
