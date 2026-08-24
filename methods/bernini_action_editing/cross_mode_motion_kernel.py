#!/usr/bin/env python3
"""Pure-tensor Cross-Mode Motion Kernel Distillation (CMKD) v7 core.

CMKD compares a T2V teacher and a source-conditioned target without assuming
that their spatial tokens or channels share coordinates.  Given an already
gauge-fixed motion field ``x`` with shape ``[B, 21, S, D]`` and exact
``x[:, 0] == 0``, it forms causal phase increments and discards the synthetic
zero increment at phase zero.  If ``U`` contains the row-normalized flattened
increments, the temporal self-kernel is

``K = U @ U.transpose(-1, -2)``.

Consequently, ``K`` is invariant to any time-constant orthogonal change of
the flattened coordinate system.  This includes token permutations, sign
changes, and independent orthogonal transforms of spatial and channel axes.
The construction says nothing about where motion occurs and does not, by
itself, establish semantic correspondence between two videos.

The unit diagonal of ``K`` is discarded before any cross-video comparison:
row normalization makes that diagonal nearly constant even for unrelated
high-dimensional motion.  Teacher eligibility therefore uses only a centered
*cross-phase relational kernel*, its off-diagonal relational energy, and
invariant increment-energy envelopes.  It never computes a teacher/target
pointwise cosine.  Pointwise comparison is reserved for the source-aligned
target and student fields.  Every public tensor operation validates shape,
dtype, phase-zero, and finiteness and either returns auditable diagnostics or
raises.

PyTorch is imported lazily so configuration contracts can be inspected in a
lightweight environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


METHOD_NAME = "cross-mode-motion-kernel-distillation-v7"
EXPECTED_PHASES = 21
ACTIVE_INCREMENTS = EXPECTED_PHASES - 1
DEFAULT_EPSILON = 1.0e-6


class CrossModeMotionKernelError(RuntimeError):
    """Raised when a CMKD tensor or configuration invariant differs."""


@dataclass(frozen=True)
class CrossModeMotionKernelConfig:
    """Auditable eligibility thresholds and student-loss weights.

    The energy-ratio bounds are deliberately broad: they reject collapsed or
    explosive teachers while the envelope alignment checks temporal energy
    shape.  They do not calibrate semantic correctness.
    """

    min_centered_kernel_alignment: float = 0.55
    min_off_diagonal_relational_rms: float = 0.05
    min_envelope_cosine: float = 0.80
    max_envelope_relative_error: float = 0.65
    min_teacher_target_energy_ratio: float = 0.10
    max_teacher_target_energy_ratio: float = 10.0
    min_motion_rms: float = 1.0e-5
    target_direction_weight: float = 1.0
    teacher_kernel_weight: float = 0.5
    amplitude_envelope_weight: float = 0.25
    temporal_jitter_weight: float = 0.05
    epsilon: float = DEFAULT_EPSILON

    def validate(self) -> None:
        _finite_interval(
            "min_centered_kernel_alignment",
            self.min_centered_kernel_alignment,
            lower=-1.0,
            upper=1.0,
        )
        _finite_interval(
            "min_off_diagonal_relational_rms",
            self.min_off_diagonal_relational_rms,
            lower=0.0,
            upper=1.0,
            strictly_lower=True,
        )
        _finite_interval(
            "min_envelope_cosine",
            self.min_envelope_cosine,
            lower=0.0,
            upper=1.0,
        )
        _finite_interval(
            "max_envelope_relative_error",
            self.max_envelope_relative_error,
            lower=0.0,
            upper=None,
        )
        _finite_interval(
            "min_teacher_target_energy_ratio",
            self.min_teacher_target_energy_ratio,
            lower=0.0,
            upper=None,
            strictly_lower=True,
        )
        _finite_interval(
            "max_teacher_target_energy_ratio",
            self.max_teacher_target_energy_ratio,
            lower=0.0,
            upper=None,
            strictly_lower=True,
        )
        if (
            float(self.max_teacher_target_energy_ratio)
            < float(self.min_teacher_target_energy_ratio)
        ):
            raise CrossModeMotionKernelError(
                "energy-ratio maximum must not be below its minimum"
            )
        _finite_interval(
            "min_motion_rms",
            self.min_motion_rms,
            lower=0.0,
            upper=None,
            strictly_lower=True,
        )
        for name in (
            "target_direction_weight",
            "teacher_kernel_weight",
            "amplitude_envelope_weight",
            "temporal_jitter_weight",
        ):
            _finite_interval(
                name,
                getattr(self, name),
                lower=0.0,
                upper=None,
            )
        if not any(
            float(getattr(self, name)) > 0.0
            for name in (
                "target_direction_weight",
                "teacher_kernel_weight",
                "amplitude_envelope_weight",
                "temporal_jitter_weight",
            )
        ):
            raise CrossModeMotionKernelError(
                "at least one student-loss weight must be strictly positive"
            )
        _finite_interval(
            "epsilon",
            self.epsilon,
            lower=0.0,
            upper=None,
            strictly_lower=True,
        )


@dataclass(frozen=True)
class MotionKernelStatistics:
    """Coordinate-invariant temporal statistics for one motion field.

    ``phase_increments`` and ``phase_rms_envelope`` retain 21 phases so their
    first entries audit the exact-zero boundary.  Kernels use only phases
    1--20 and therefore have shape ``[B, 20, 20]``.  The raw self-kernel is
    diagnostic; only its diagonal-free relational form is centered and used
    across videos.  Frequency power is an rFFT over the 20 active increments
    and has shape ``[B, 11]``.
    """

    phase_increments: Any
    active_increment_mask: Any
    temporal_self_kernel: Any
    cross_phase_relational_kernel: Any
    centered_cross_phase_relational_kernel: Any
    centered_relational_frobenius: Any
    off_diagonal_relational_rms: Any
    phase_rms_envelope: Any
    temporal_frequency_power: Any
    normalized_temporal_frequency_power: Any
    total_increment_rms: Any


@dataclass(frozen=True)
class TeacherTargetEligibilityDiagnostics:
    """Every scalar and decision used by the teacher gate."""

    teacher: MotionKernelStatistics
    target: MotionKernelStatistics
    centered_kernel_alignment: Any
    envelope_cosine: Any
    teacher_to_target_envelope_scale: Any
    envelope_relative_error: Any
    teacher_target_energy_ratio: Any
    frequency_power_cosine: Any
    teacher_motion_non_degenerate: Any
    target_motion_non_degenerate: Any
    teacher_relational_non_degenerate: Any
    target_relational_non_degenerate: Any
    centered_relational_non_degenerate: Any
    kernel_alignment_pass: Any
    envelope_cosine_pass: Any
    envelope_error_pass: Any
    energy_ratio_pass: Any
    off_diagonal_relational_pass: Any
    finite_pass: Any
    eligible: Any


@dataclass(frozen=True)
class StudentLossDiagnostics:
    """Per-sample CMKD loss terms plus the immutable eligibility audit."""

    eligibility: TeacherTargetEligibilityDiagnostics
    student: MotionKernelStatistics
    target_direction_loss: Any
    teacher_kernel_loss: Any
    amplitude_envelope_loss: Any
    temporal_jitter_loss: Any
    weighted_total_loss: Any
    eligible_sample_count: Any


@dataclass(frozen=True)
class StudentLossResult:
    """A scalar differentiable loss and complete per-sample diagnostics."""

    loss: Any
    diagnostics: StudentLossDiagnostics


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        raise CrossModeMotionKernelError(
            "CMKD tensor operations require PyTorch"
        ) from error
    return torch


def _finite_interval(
    name: str,
    value: Any,
    *,
    lower: float,
    upper: float | None,
    strictly_lower: bool = False,
) -> None:
    if isinstance(value, bool):
        raise CrossModeMotionKernelError(f"{name} is outside its valid interval")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise CrossModeMotionKernelError(
            f"{name} is outside its valid interval"
        ) from error
    below = numeric <= lower if strictly_lower else numeric < lower
    if (
        not math.isfinite(numeric)
        or below
        or (upper is not None and numeric > upper)
    ):
        raise CrossModeMotionKernelError(f"{name} is outside its valid interval")


def _validate_motion_fields(*fields: Any) -> None:
    torch = _require_torch()
    if not fields:
        raise CrossModeMotionKernelError("at least one motion field is required")
    reference = fields[0]
    if not isinstance(reference, torch.Tensor):
        raise CrossModeMotionKernelError("motion fields must be torch tensors")
    if reference.ndim != 4:
        raise CrossModeMotionKernelError(
            "motion fields must have exact shape [B,21,S,D]"
        )
    if (
        int(reference.shape[0]) <= 0
        or int(reference.shape[1]) != EXPECTED_PHASES
        or int(reference.shape[2]) <= 0
        or int(reference.shape[3]) <= 0
    ):
        raise CrossModeMotionKernelError(
            "motion fields must have exact non-empty shape [B,21,S,D]"
        )
    if reference.dtype != torch.float32:
        raise CrossModeMotionKernelError("CMKD motion fields must be torch.float32")
    for field in fields:
        if not isinstance(field, torch.Tensor):
            raise CrossModeMotionKernelError("motion fields must be torch tensors")
        if tuple(field.shape) != tuple(reference.shape):
            raise CrossModeMotionKernelError("motion field shapes differ")
        if field.dtype != reference.dtype or field.device != reference.device:
            raise CrossModeMotionKernelError(
                "motion field dtype or device differs"
            )
        if not bool(torch.isfinite(field).all()):
            raise CrossModeMotionKernelError(
                "motion field contains non-finite values"
            )
        if not bool(
            torch.equal(field[:, 0], torch.zeros_like(field[:, 0]))
        ):
            raise CrossModeMotionKernelError(
                "CMKD inputs must be Q0 fields with exact-zero phase zero"
            )


def _assert_finite(name: str, value: Any) -> None:
    torch = _require_torch()
    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
        raise CrossModeMotionKernelError(f"{name} is non-finite")


def _phase_increments(direction: Any) -> Any:
    """Return 21 increments with a synthetic exact-zero phase-zero entry."""

    torch = _require_torch()
    zero = torch.zeros_like(direction[:, :1])
    increments = torch.cat(
        (zero, direction[:, 1:] - direction[:, :-1]), dim=1
    )
    if not bool(torch.equal(increments[:, 0], zero[:, 0])):
        raise CrossModeMotionKernelError(
            "phase-increment construction changed exact-zero phase zero"
        )
    return increments


def _center_kernel(kernel: Any) -> Any:
    row_mean = kernel.mean(dim=-1, keepdim=True)
    column_mean = kernel.mean(dim=-2, keepdim=True)
    grand_mean = kernel.mean(dim=(-2, -1), keepdim=True)
    return kernel - row_mean - column_mean + grand_mean


def _remove_kernel_diagonal(kernel: Any) -> tuple[Any, Any]:
    """Return the cross-phase kernel and its RMS over off-diagonal entries."""

    torch = _require_torch()
    phases = int(kernel.shape[-1])
    diagonal = torch.eye(phases, dtype=torch.bool, device=kernel.device)
    relational = kernel.masked_fill(diagonal[None], 0.0)
    denominator = phases * (phases - 1)
    relational_rms = torch.sqrt(
        relational.square().sum(dim=(-2, -1)) / float(denominator)
    )
    # Returning the mask keeps the exact definition local and prevents a
    # caller from accidentally including the uninformative diagonal.
    return relational, relational_rms


def _safe_cosine(left: Any, right: Any, *, epsilon: float) -> tuple[Any, Any]:
    """Return batched flattened cosine and a non-degeneracy mask."""

    torch = _require_torch()
    left_flat = left.flatten(start_dim=1)
    right_flat = right.flatten(start_dim=1)
    numerator = (left_flat * right_flat).sum(dim=1)
    left_norm = torch.linalg.vector_norm(left_flat, dim=1)
    right_norm = torch.linalg.vector_norm(right_flat, dim=1)
    non_degenerate = (left_norm > float(epsilon)) & (
        right_norm > float(epsilon)
    )
    denominator = (left_norm * right_norm).clamp_min(float(epsilon))
    cosine = torch.where(
        non_degenerate,
        torch.clamp(numerator / denominator, min=-1.0, max=1.0),
        torch.zeros_like(numerator),
    )
    return cosine, non_degenerate


def motion_kernel_statistics(
    direction: Any,
    *,
    config: CrossModeMotionKernelConfig = CrossModeMotionKernelConfig(),
) -> MotionKernelStatistics:
    """Compute the invariant CMKD statistics of one exact-Q0 motion field."""

    config.validate()
    _validate_motion_fields(direction)
    torch = _require_torch()
    epsilon = float(config.epsilon)

    increments = _phase_increments(direction)
    active = increments[:, 1:]
    batch = int(active.shape[0])
    flat = active.reshape(batch, ACTIVE_INCREMENTS, -1)
    row_norm = torch.linalg.vector_norm(flat, dim=2, keepdim=True)
    normalized = torch.where(
        row_norm > epsilon,
        flat / row_norm.clamp_min(epsilon),
        torch.zeros_like(flat),
    )
    kernel = normalized @ normalized.transpose(-1, -2)
    relational_kernel, relational_rms = _remove_kernel_diagonal(kernel)
    centered_relational_kernel = _center_kernel(relational_kernel)
    centered_frobenius = torch.linalg.vector_norm(
        centered_relational_kernel, dim=(-2, -1)
    )

    coordinate_count = int(direction.shape[2]) * int(direction.shape[3])
    envelope = torch.linalg.vector_norm(
        increments.reshape(batch, EXPECTED_PHASES, coordinate_count), dim=2
    ) / math.sqrt(coordinate_count)
    if not bool(torch.equal(envelope[:, 0], torch.zeros_like(envelope[:, 0]))):
        raise CrossModeMotionKernelError(
            "phase RMS envelope changed exact-zero phase zero"
        )

    frequency = torch.fft.rfft(flat, dim=1, norm="ortho")
    frequency_power = frequency.abs().square().mean(dim=2)
    power_sum = frequency_power.sum(dim=1, keepdim=True)
    normalized_power = torch.where(
        power_sum > epsilon,
        frequency_power / power_sum.clamp_min(epsilon),
        torch.zeros_like(frequency_power),
    )
    total_rms = torch.sqrt(active.square().mean(dim=(1, 2, 3)))
    active_mask = row_norm.squeeze(2) > epsilon

    for name, value in (
        ("phase increments", increments),
        ("temporal self-kernel", kernel),
        ("cross-phase relational kernel", relational_kernel),
        ("centered cross-phase relational kernel", centered_relational_kernel),
        ("centered relational kernel norm", centered_frobenius),
        ("off-diagonal relational RMS", relational_rms),
        ("phase RMS envelope", envelope),
        ("temporal-frequency power", frequency_power),
        ("normalized temporal-frequency power", normalized_power),
        ("total increment RMS", total_rms),
    ):
        _assert_finite(name, value)

    return MotionKernelStatistics(
        phase_increments=increments,
        active_increment_mask=active_mask,
        temporal_self_kernel=kernel,
        cross_phase_relational_kernel=relational_kernel,
        centered_cross_phase_relational_kernel=centered_relational_kernel,
        centered_relational_frobenius=centered_frobenius,
        off_diagonal_relational_rms=relational_rms,
        phase_rms_envelope=envelope,
        temporal_frequency_power=frequency_power,
        normalized_temporal_frequency_power=normalized_power,
        total_increment_rms=total_rms,
    )


def evaluate_teacher_target_eligibility(
    teacher_direction: Any,
    target_direction: Any,
    *,
    config: CrossModeMotionKernelConfig = CrossModeMotionKernelConfig(),
) -> TeacherTargetEligibilityDiagnostics:
    """Gate a teacher using no teacher/target pointwise-coordinate metric."""

    config.validate()
    _validate_motion_fields(teacher_direction, target_direction)
    torch = _require_torch()
    epsilon = float(config.epsilon)
    teacher = motion_kernel_statistics(teacher_direction, config=config)
    target = motion_kernel_statistics(target_direction, config=config)

    kernel_alignment, kernel_non_degenerate = _safe_cosine(
        teacher.centered_cross_phase_relational_kernel,
        target.centered_cross_phase_relational_kernel,
        epsilon=epsilon,
    )
    teacher_envelope = teacher.phase_rms_envelope[:, 1:]
    target_envelope = target.phase_rms_envelope[:, 1:]
    envelope_cosine, envelope_non_degenerate = _safe_cosine(
        teacher_envelope,
        target_envelope,
        epsilon=epsilon,
    )

    teacher_envelope_square = teacher_envelope.square().sum(dim=1)
    envelope_scale = (
        (teacher_envelope * target_envelope).sum(dim=1)
        / teacher_envelope_square.clamp_min(epsilon)
    )
    aligned_teacher_envelope = teacher_envelope * envelope_scale[:, None]
    envelope_relative_error = torch.linalg.vector_norm(
        aligned_teacher_envelope - target_envelope, dim=1
    ) / torch.linalg.vector_norm(target_envelope, dim=1).clamp_min(epsilon)

    energy_ratio = teacher.total_increment_rms / target.total_increment_rms.clamp_min(
        epsilon
    )
    frequency_cosine, _ = _safe_cosine(
        teacher.temporal_frequency_power,
        target.temporal_frequency_power,
        epsilon=epsilon,
    )
    teacher_motion_non_degenerate = (
        teacher.total_increment_rms >= float(config.min_motion_rms)
    )
    target_motion_non_degenerate = (
        target.total_increment_rms >= float(config.min_motion_rms)
    )
    finite_pass = (
        torch.isfinite(kernel_alignment)
        & torch.isfinite(envelope_cosine)
        & torch.isfinite(envelope_scale)
        & torch.isfinite(envelope_relative_error)
        & torch.isfinite(energy_ratio)
        & torch.isfinite(frequency_cosine)
    )
    kernel_pass = kernel_alignment >= float(
        config.min_centered_kernel_alignment
    )
    envelope_cosine_pass = (
        envelope_cosine >= float(config.min_envelope_cosine)
    ) & envelope_non_degenerate
    envelope_error_pass = (
        envelope_relative_error <= float(config.max_envelope_relative_error)
    )
    energy_ratio_pass = (
        energy_ratio >= float(config.min_teacher_target_energy_ratio)
    ) & (energy_ratio <= float(config.max_teacher_target_energy_ratio))
    teacher_relational_non_degenerate = (
        teacher.off_diagonal_relational_rms
        >= float(config.min_off_diagonal_relational_rms)
    )
    target_relational_non_degenerate = (
        target.off_diagonal_relational_rms
        >= float(config.min_off_diagonal_relational_rms)
    )
    centered_relational_non_degenerate = kernel_non_degenerate
    off_diagonal_relational_pass = (
        teacher_relational_non_degenerate & target_relational_non_degenerate
    )
    eligible = (
        finite_pass
        & teacher_motion_non_degenerate
        & target_motion_non_degenerate
        & centered_relational_non_degenerate
        & off_diagonal_relational_pass
        & kernel_pass
        & envelope_cosine_pass
        & envelope_error_pass
        & energy_ratio_pass
    )

    for name, value in (
        ("centered kernel alignment", kernel_alignment),
        ("envelope cosine", envelope_cosine),
        ("teacher-to-target envelope scale", envelope_scale),
        ("envelope relative error", envelope_relative_error),
        ("teacher-target energy ratio", energy_ratio),
        ("frequency-power cosine", frequency_cosine),
    ):
        _assert_finite(name, value)

    return TeacherTargetEligibilityDiagnostics(
        teacher=teacher,
        target=target,
        centered_kernel_alignment=kernel_alignment,
        envelope_cosine=envelope_cosine,
        teacher_to_target_envelope_scale=envelope_scale,
        envelope_relative_error=envelope_relative_error,
        teacher_target_energy_ratio=energy_ratio,
        frequency_power_cosine=frequency_cosine,
        teacher_motion_non_degenerate=teacher_motion_non_degenerate,
        target_motion_non_degenerate=target_motion_non_degenerate,
        teacher_relational_non_degenerate=teacher_relational_non_degenerate,
        target_relational_non_degenerate=target_relational_non_degenerate,
        centered_relational_non_degenerate=centered_relational_non_degenerate,
        kernel_alignment_pass=kernel_pass,
        envelope_cosine_pass=envelope_cosine_pass,
        envelope_error_pass=envelope_error_pass,
        energy_ratio_pass=energy_ratio_pass,
        off_diagonal_relational_pass=off_diagonal_relational_pass,
        finite_pass=finite_pass,
        eligible=eligible,
    )


def cmkd_student_loss(
    student_direction: Any,
    target_direction: Any,
    teacher_direction: Any,
    *,
    config: CrossModeMotionKernelConfig = CrossModeMotionKernelConfig(),
) -> StudentLossResult:
    """Return the four-term CMKD v7 loss for an eligible teacher/target pair.

    The target-direction and residual-jitter terms compare coordinates only
    between the source-aligned student and target.  The independently
    coordinated teacher enters solely through its diagonal-free temporal
    relational kernel.
    Amplitude follows the target envelope, avoiding uncalibrated T2V scale.
    Ineligible teacher/target pairs raise instead of silently contributing a
    gradient.
    """

    config.validate()
    _validate_motion_fields(student_direction, target_direction, teacher_direction)
    torch = _require_torch()
    epsilon = float(config.epsilon)
    eligibility = evaluate_teacher_target_eligibility(
        teacher_direction, target_direction, config=config
    )
    if not bool(eligibility.eligible.all()):
        rejected = (~eligibility.eligible).nonzero(as_tuple=False).flatten().tolist()
        raise CrossModeMotionKernelError(
            f"CMKD student loss requires eligible teacher/target pairs; rejected={rejected}"
        )

    student = motion_kernel_statistics(student_direction, config=config)
    target = eligibility.target
    teacher = eligibility.teacher

    # Only this source-aligned pair is compared point by point.  Phase zero is
    # excluded because it is a fixed gauge boundary, not a training example.
    target_active = target_direction[:, 1:]
    student_active = student_direction[:, 1:]
    target_scale_square = target_active.square().mean(dim=(1, 2, 3)).clamp_min(
        epsilon
    )
    target_direction_loss = (
        (student_active - target_active).square().mean(dim=(1, 2, 3))
        / target_scale_square
    )

    student_teacher_alignment, student_kernel_non_degenerate = _safe_cosine(
        student.centered_cross_phase_relational_kernel,
        teacher.centered_cross_phase_relational_kernel,
        epsilon=epsilon,
    )
    teacher_kernel_loss = torch.where(
        student_kernel_non_degenerate,
        1.0 - student_teacher_alignment,
        torch.ones_like(student_teacher_alignment),
    )

    target_envelope = target.phase_rms_envelope[:, 1:]
    student_envelope = student.phase_rms_envelope[:, 1:]
    amplitude_envelope_loss = (
        (student_envelope - target_envelope).square().mean(dim=1)
        / target_envelope.square().mean(dim=1).clamp_min(epsilon)
    )

    # Penalize acceleration of the student-target residual, not acceleration
    # intrinsic to the desired target motion.  This is a source-aligned
    # pointwise operation and never compares coordinates with the teacher.
    residual_increments = (
        student.phase_increments[:, 1:] - target.phase_increments[:, 1:]
    )
    residual_acceleration = (
        residual_increments[:, 1:] - residual_increments[:, :-1]
    )
    target_increment_scale_square = target.phase_increments[:, 1:].square().mean(
        dim=(1, 2, 3)
    ).clamp_min(epsilon)
    temporal_jitter_loss = residual_acceleration.square().mean(
        dim=(1, 2, 3)
    ) / target_increment_scale_square

    weighted_total = (
        float(config.target_direction_weight) * target_direction_loss
        + float(config.teacher_kernel_weight) * teacher_kernel_loss
        + float(config.amplitude_envelope_weight) * amplitude_envelope_loss
        + float(config.temporal_jitter_weight) * temporal_jitter_loss
    )
    loss = weighted_total.mean()
    for name, value in (
        ("target-direction loss", target_direction_loss),
        ("teacher-kernel loss", teacher_kernel_loss),
        ("amplitude-envelope loss", amplitude_envelope_loss),
        ("temporal-jitter loss", temporal_jitter_loss),
        ("weighted total loss", weighted_total),
        ("scalar CMKD loss", loss),
    ):
        _assert_finite(name, value)

    eligible_count = eligibility.eligible.to(dtype=torch.int64).sum()
    return StudentLossResult(
        loss=loss,
        diagnostics=StudentLossDiagnostics(
            eligibility=eligibility,
            student=student,
            target_direction_loss=target_direction_loss,
            teacher_kernel_loss=teacher_kernel_loss,
            amplitude_envelope_loss=amplitude_envelope_loss,
            temporal_jitter_loss=temporal_jitter_loss,
            weighted_total_loss=weighted_total,
            eligible_sample_count=eligible_count,
        ),
    )


# Descriptive aliases for integration sites.
compute_motion_kernel_statistics = motion_kernel_statistics
compute_cmkd_student_loss = cmkd_student_loss


__all__ = [
    "ACTIVE_INCREMENTS",
    "CrossModeMotionKernelConfig",
    "CrossModeMotionKernelError",
    "EXPECTED_PHASES",
    "METHOD_NAME",
    "MotionKernelStatistics",
    "StudentLossDiagnostics",
    "StudentLossResult",
    "TeacherTargetEligibilityDiagnostics",
    "cmkd_student_loss",
    "compute_cmkd_student_loss",
    "compute_motion_kernel_statistics",
    "evaluate_teacher_target_eligibility",
    "motion_kernel_statistics",
]
