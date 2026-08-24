#!/usr/bin/env python3
"""Pure-tensor Cross-Mode Motion Spectrum Guidance (CMSG) v6.

CMSG uses a generator only as a *motion-statistics oracle*.  It never adds a
generator clean field, latent, or phase increment to the editor.  Instead it
measures the agreement and relative energy of temporal increments from two
independently classifier-free-guided directions:

``official_editor = Q0(editor_action - editor_noop)`` and
``generator_motion = Q0(generator_action - generator_uncond)``.

A fixed 3x3 spatial low pass is used only to measure spatial and channel
increment spectra.  If both spectra align, clipped generator energy produces
one scalar gain per batch item and phase.  That scalar multiplies the original
(not low-passed) editor increment, so the source-aware editor direction is the
only value-bearing carrier.  The plan is reconstructed causally from phase
zero.  A separate release operator bounds ``plan - official_editor`` before
applying the immutable 40-step denoising schedule.

PyTorch is imported lazily so schedule and configuration contracts remain
importable in lightweight environments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


METHOD_NAME = "cross-mode-motion-spectrum-guidance-v6"
EXPECTED_PHASES = 21
NUM_DENOISING_STEPS = 40
FULL_RELEASE_LAST_STEP = 19
TAPER_FIRST_STEP = 20
TAPER_LAST_STEP = 31
ZERO_RELEASE_FIRST_STEP = 32
DEFAULT_ALIGNMENT_THRESHOLD = 0.1
DEFAULT_GENERATOR_LAMBDA = 0.5
DEFAULT_MAX_PLAN_DELTA_RATIO = 0.5
DEFAULT_EPSILON = 1.0e-6


class CrossModeMotionSpectrumError(RuntimeError):
    """Raised when a CMSG tensor, geometry, or schedule invariant differs."""


@dataclass(frozen=True)
class CrossModeMotionSpectrumConfig:
    """Auditable CMSG hyperparameters.

    ``generator_lambda`` interpolates between the official editor increments
    and their generator-energy modulation.  ``max_plan_delta_ratio`` bounds
    the phasewise RMS of the released plan delta relative to the official
    editor field.
    """

    generator_lambda: float = DEFAULT_GENERATOR_LAMBDA
    alignment_threshold: float = DEFAULT_ALIGNMENT_THRESHOLD
    max_plan_delta_ratio: float = DEFAULT_MAX_PLAN_DELTA_RATIO
    epsilon: float = DEFAULT_EPSILON

    def validate(self) -> None:
        _validate_finite_interval(
            "generator_lambda", self.generator_lambda, lower=0.0, upper=1.0
        )
        _validate_finite_interval(
            "alignment_threshold",
            self.alignment_threshold,
            lower=-1.0,
            upper=1.0,
        )
        _validate_finite_interval(
            "max_plan_delta_ratio",
            self.max_plan_delta_ratio,
            lower=0.0,
            upper=None,
        )
        if isinstance(self.epsilon, bool):
            raise CrossModeMotionSpectrumError(
                "epsilon must be finite and strictly positive"
            )
        try:
            epsilon = float(self.epsilon)
        except (TypeError, ValueError) as error:
            raise CrossModeMotionSpectrumError(
                "epsilon must be finite and strictly positive"
            ) from error
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise CrossModeMotionSpectrumError(
                "epsilon must be finite and strictly positive"
            )


@dataclass(frozen=True)
class MotionSpectrumDiagnostics:
    """Generator-derived scalars that audit a CMSG plan.

    Scalar energy, cosine, gate, ratio, and gain tensors have shape ``[B,21]``.
    The nonnegative spatial energy profiles have shape ``[B,21,S]`` and the
    channel energy profiles have shape ``[B,21,D]``.  Generator clean values
    are deliberately absent from this result.
    """

    editor_spatial_increment_energy: Any
    generator_spatial_increment_energy: Any
    clipped_generator_spatial_increment_energy: Any
    editor_spatial_energy_profile: Any
    generator_spatial_energy_profile: Any
    spatial_increment_cosine: Any
    editor_channel_increment_energy: Any
    generator_channel_increment_energy: Any
    clipped_generator_channel_increment_energy: Any
    editor_channel_energy_profile: Any
    generator_channel_energy_profile: Any
    channel_increment_cosine: Any
    mean_alignment: Any
    alignment_gate: Any
    spatial_energy_ratio: Any
    channel_energy_ratio: Any
    increment_gain: Any


@dataclass(frozen=True)
class MotionSpectrumPlanResult:
    """The official source-aware editor direction and its CMSG plan."""

    official_editor: Any
    plan: Any
    diagnostics: MotionSpectrumDiagnostics


@dataclass(frozen=True)
class BoundedPlanDiagnostics:
    """Auditable phasewise trust bound for ``plan - official_editor``."""

    raw_plan_delta: Any
    raw_plan_delta_rms: Any
    official_editor_rms: Any
    delta_rms_cap: Any
    bound_scale: Any
    bounded_plan_delta: Any


@dataclass(frozen=True)
class ExecutedMotionSpectrumResult:
    """One denoising step of bounded CMSG release."""

    official_editor: Any
    plan: Any
    rho: float
    bound: BoundedPlanDiagnostics
    executed_field: Any


def _validate_finite_interval(
    name: str,
    value: Any,
    *,
    lower: float,
    upper: float | None,
) -> None:
    if isinstance(value, bool):
        raise CrossModeMotionSpectrumError(f"{name} is outside its valid interval")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise CrossModeMotionSpectrumError(
            f"{name} is outside its valid interval"
        ) from error
    if (
        not math.isfinite(numeric)
        or numeric < lower
        or (upper is not None and numeric > upper)
    ):
        raise CrossModeMotionSpectrumError(f"{name} is outside its valid interval")


def release_rho(step_index: int) -> float:
    """Return the exact immutable CMSG release schedule.

    Steps 0--19 are one.  Steps 20--31 are an inclusive cosine from one
    through zero.  Steps 32--39 are exactly zero.  Consequently step 31 is
    also zero as the inclusive taper endpoint.
    """

    if type(step_index) is not int or not 0 <= step_index < NUM_DENOISING_STEPS:
        raise CrossModeMotionSpectrumError(
            "step_index must be an integer in [0,40)"
        )
    if step_index <= FULL_RELEASE_LAST_STEP:
        return 1.0
    if step_index >= ZERO_RELEASE_FIRST_STEP:
        return 0.0
    if step_index == TAPER_FIRST_STEP:
        return 1.0
    if step_index == TAPER_LAST_STEP:
        return 0.0
    progress = (step_index - TAPER_FIRST_STEP) / (
        TAPER_LAST_STEP - TAPER_FIRST_STEP
    )
    return float(0.5 * (1.0 + math.cos(math.pi * progress)))


def release_rho_schedule() -> tuple[float, ...]:
    """Materialize all 40 exact release coefficients."""

    return tuple(release_rho(step) for step in range(NUM_DENOISING_STEPS))


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised without torch
        raise CrossModeMotionSpectrumError(
            "CMSG tensor operations require PyTorch"
        ) from error
    return torch


def _validate_fields(*fields: Any) -> None:
    torch = _require_torch()
    if not fields:
        raise CrossModeMotionSpectrumError("at least one field is required")
    reference = fields[0]
    if not isinstance(reference, torch.Tensor):
        raise CrossModeMotionSpectrumError("clean fields must be torch tensors")
    if reference.ndim != 4:
        raise CrossModeMotionSpectrumError(
            "clean fields must have exact shape [B,21,S,D]"
        )
    if int(reference.shape[0]) <= 0 or int(reference.shape[2]) <= 0 or int(
        reference.shape[3]
    ) <= 0:
        raise CrossModeMotionSpectrumError("B, S, and D must be non-empty")
    if int(reference.shape[1]) != EXPECTED_PHASES:
        raise CrossModeMotionSpectrumError(
            "clean fields must contain exactly 21 causal phases"
        )
    if reference.dtype != torch.float32:
        raise CrossModeMotionSpectrumError("CMSG clean fields must be torch.float32")
    for field in fields:
        if not isinstance(field, torch.Tensor):
            raise CrossModeMotionSpectrumError("clean fields must be torch tensors")
        if tuple(field.shape) != tuple(reference.shape):
            raise CrossModeMotionSpectrumError("clean field shapes differ")
        if field.device != reference.device or field.dtype != reference.dtype:
            raise CrossModeMotionSpectrumError(
                "clean field device or dtype differs"
            )
        if not bool(torch.isfinite(field).all()):
            raise CrossModeMotionSpectrumError("clean field contains non-finite values")


def _resolve_spatial_hw(
    spatial_tokens: int,
    spatial_hw: tuple[int, int] | None,
) -> tuple[int, int]:
    if spatial_hw is None:
        side = math.isqrt(spatial_tokens)
        if side * side != spatial_tokens:
            raise CrossModeMotionSpectrumError(
                "spatial_hw is required when S is not a perfect square"
            )
        return side, side
    if (
        type(spatial_hw) is not tuple
        or len(spatial_hw) != 2
        or any(type(value) is not int or value <= 0 for value in spatial_hw)
    ):
        raise CrossModeMotionSpectrumError(
            "spatial_hw must be a pair of strictly positive integers"
        )
    height, width = spatial_hw
    if height * width != spatial_tokens:
        raise CrossModeMotionSpectrumError(
            "spatial_hw product must equal the clean field S dimension"
        )
    return height, width


def q0(field: Any) -> Any:
    """Apply the exact causal gauge ``Q0(x)_t = x_t - x_0``."""

    _validate_fields(field)
    torch = _require_torch()
    projected = field - field[:, :1]
    if not bool(torch.equal(projected[:, 0], torch.zeros_like(projected[:, 0]))):
        raise CrossModeMotionSpectrumError("Q0 failed to make phase zero exact")
    return projected


def _phase_increments(direction: Any) -> Any:
    """Return 21 increments, with an exact zero phase-zero increment."""

    torch = _require_torch()
    zero = torch.zeros_like(direction[:, :1])
    return torch.cat((zero, direction[:, 1:] - direction[:, :-1]), dim=1)


def _spatial_low_pass(increments: Any, height: int, width: int) -> Any:
    """Apply a fixed replicate-padded 3x3 average independently per channel."""

    torch = _require_torch()
    batch, phases, spatial_tokens, channels = increments.shape
    images = increments.reshape(batch, phases, height, width, channels)
    images = images.permute(0, 1, 4, 2, 3).reshape(
        batch * phases, channels, height, width
    )
    padded = torch.nn.functional.pad(images, (1, 1, 1, 1), mode="replicate")
    filtered = torch.nn.functional.avg_pool2d(
        padded, kernel_size=3, stride=1, padding=0
    )
    return (
        filtered.reshape(batch, phases, channels, height, width)
        .permute(0, 1, 3, 4, 2)
        .reshape(batch, phases, spatial_tokens, channels)
    )


def _active_group_cosine(
    editor_vectors: Any,
    generator_vectors: Any,
    *,
    vector_dim: int,
    group_dim: int,
    epsilon: float,
) -> Any:
    """Average stable signed cosines without marginalizing vector values.

    For the spatial spectrum, every channel is one vector over ``S``.  For
    the channel spectrum, every spatial token is one vector over ``D``.  This
    keeps opposite signed components separate until after their energies are
    measured, so a checkerboard/zero-marginal motion cannot disappear.
    """

    torch = _require_torch()
    editor_energy = editor_vectors.square().sum(dim=vector_dim)
    generator_energy = generator_vectors.square().sum(dim=vector_dim)
    dot = (editor_vectors * generator_vectors).sum(dim=vector_dim)
    denominator = torch.sqrt(
        editor_energy * generator_energy + float(epsilon) ** 2
    )
    group_cosine = torch.clamp(dot / denominator, min=-1.0, max=1.0)
    active = (editor_energy > float(epsilon)) & (
        generator_energy > float(epsilon)
    )
    active_value = active.to(dtype=editor_vectors.dtype)
    active_count = active_value.sum(dim=group_dim).clamp_min(1.0)
    return (group_cosine * active_value).sum(dim=group_dim) / active_count


def _motion_spectrum_statistics(
    editor_increments: Any,
    generator_increments: Any,
    *,
    epsilon: float,
) -> tuple[Any, ...]:
    """Return non-cancelling spatial/channel profiles, energies, and cosines."""

    # Squaring precedes every marginalization.  The profiles therefore expose
    # motion support even when signed row and column sums are both exactly zero.
    editor_square = editor_increments.square()
    generator_square = generator_increments.square()
    editor_spatial_profile = editor_square.mean(dim=3)
    generator_spatial_profile = generator_square.mean(dim=3)
    editor_channel_profile = editor_square.mean(dim=2)
    generator_channel_profile = generator_square.mean(dim=2)
    editor_spatial_energy = editor_spatial_profile.mean(dim=2)
    generator_spatial_energy = generator_spatial_profile.mean(dim=2)
    editor_channel_energy = editor_channel_profile.mean(dim=2)
    generator_channel_energy = generator_channel_profile.mean(dim=2)

    spatial_cosine = _active_group_cosine(
        editor_increments,
        generator_increments,
        vector_dim=2,
        group_dim=2,
        epsilon=epsilon,
    )
    channel_cosine = _active_group_cosine(
        editor_increments,
        generator_increments,
        vector_dim=3,
        group_dim=2,
        epsilon=epsilon,
    )
    return (
        editor_spatial_profile,
        generator_spatial_profile,
        editor_channel_profile,
        generator_channel_profile,
        editor_spatial_energy,
        generator_spatial_energy,
        editor_channel_energy,
        generator_channel_energy,
        spatial_cosine,
        channel_cosine,
    )


def build_cmsg_plan(
    editor_action_field: Any,
    editor_noop_field: Any,
    generator_action_field: Any,
    generator_uncond_field: Any,
    *,
    spatial_hw: tuple[int, int] | None = None,
    config: CrossModeMotionSpectrumConfig = CrossModeMotionSpectrumConfig(),
) -> MotionSpectrumPlanResult:
    """Build a causal plan without injecting any generator value.

    Each channel is treated as a signed spatial vector and each spatial token
    as a signed channel vector.  Energy profiles square before reducing, while
    cosines average only over active vectors.  Their statistics determine a
    single gain ``[B,21,1,1]``.  The gain scales only the original official
    editor increment.  Generator energies are independently clipped to
    ``[aE, 3*aE]`` before their ratios are formed.
    """

    config.validate()
    _validate_fields(
        editor_action_field,
        editor_noop_field,
        generator_action_field,
        generator_uncond_field,
    )
    torch = _require_torch()
    height, width = _resolve_spatial_hw(
        int(editor_action_field.shape[2]), spatial_hw
    )

    official_editor = q0(editor_action_field - editor_noop_field)
    generator_motion = q0(generator_action_field - generator_uncond_field)
    editor_increments = _phase_increments(official_editor)
    generator_increments = _phase_increments(generator_motion)

    editor_lp = _spatial_low_pass(editor_increments, height, width)
    generator_lp = _spatial_low_pass(generator_increments, height, width)
    epsilon = float(config.epsilon)
    (
        editor_spatial_profile,
        generator_spatial_profile,
        editor_channel_profile,
        generator_channel_profile,
        editor_spatial_energy,
        generator_spatial_energy,
        editor_channel_energy,
        generator_channel_energy,
        spatial_cosine,
        channel_cosine,
    ) = _motion_spectrum_statistics(
        editor_lp,
        generator_lp,
        epsilon=epsilon,
    )

    clipped_generator_spatial_energy = torch.maximum(
        editor_spatial_energy,
        torch.minimum(generator_spatial_energy, 3.0 * editor_spatial_energy),
    )
    clipped_generator_channel_energy = torch.maximum(
        editor_channel_energy,
        torch.minimum(generator_channel_energy, 3.0 * editor_channel_energy),
    )
    spatial_energy_ratio = torch.sqrt(
        (clipped_generator_spatial_energy + epsilon)
        / (editor_spatial_energy + epsilon)
    )
    channel_energy_ratio = torch.sqrt(
        (clipped_generator_channel_energy + epsilon)
        / (editor_channel_energy + epsilon)
    )

    mean_alignment = 0.5 * (spatial_cosine + channel_cosine)
    # A high score on one axis may not compensate an anti-aligned score on the
    # other.  Both motion spectra must independently pass the fixed threshold.
    alignment_gate = (
        spatial_cosine >= float(config.alignment_threshold)
    ) & (channel_cosine >= float(config.alignment_threshold))
    oracle_ratio = 0.5 * (spatial_energy_ratio + channel_energy_ratio)
    gate_value = alignment_gate.to(dtype=official_editor.dtype)
    increment_gain = 1.0 + float(config.generator_lambda) * gate_value * (
        oracle_ratio - 1.0
    )

    # This is the key non-leakage construction: the generator contributes
    # only scalar statistics.  Every value-bearing plan increment remains an
    # exact scalar multiple of the source-aware editor increment.
    increment_delta = editor_increments * (increment_gain[..., None, None] - 1.0)
    accumulated_delta = torch.cumsum(increment_delta, dim=1)
    plan = (
        official_editor
        if float(config.generator_lambda) == 0.0
        else official_editor + accumulated_delta
    )
    if not bool(torch.equal(plan[:, 0], torch.zeros_like(plan[:, 0]))):
        raise CrossModeMotionSpectrumError(
            "CMSG accumulation changed the exact phase-zero boundary"
        )
    if not bool(torch.isfinite(plan).all()):
        raise CrossModeMotionSpectrumError("CMSG plan is non-finite")

    diagnostics = MotionSpectrumDiagnostics(
        editor_spatial_increment_energy=editor_spatial_energy,
        generator_spatial_increment_energy=generator_spatial_energy,
        clipped_generator_spatial_increment_energy=(
            clipped_generator_spatial_energy
        ),
        editor_spatial_energy_profile=editor_spatial_profile,
        generator_spatial_energy_profile=generator_spatial_profile,
        spatial_increment_cosine=spatial_cosine,
        editor_channel_increment_energy=editor_channel_energy,
        generator_channel_increment_energy=generator_channel_energy,
        clipped_generator_channel_increment_energy=(
            clipped_generator_channel_energy
        ),
        editor_channel_energy_profile=editor_channel_profile,
        generator_channel_energy_profile=generator_channel_profile,
        channel_increment_cosine=channel_cosine,
        mean_alignment=mean_alignment,
        alignment_gate=alignment_gate,
        spatial_energy_ratio=spatial_energy_ratio,
        channel_energy_ratio=channel_energy_ratio,
        increment_gain=increment_gain,
    )
    return MotionSpectrumPlanResult(
        official_editor=official_editor,
        plan=plan,
        diagnostics=diagnostics,
    )


def bound_plan_delta(
    official_editor: Any,
    plan: Any,
    *,
    config: CrossModeMotionSpectrumConfig = CrossModeMotionSpectrumConfig(),
) -> BoundedPlanDiagnostics:
    """Hard-project each phase's plan-delta RMS into its editor-relative cap."""

    config.validate()
    _validate_fields(official_editor, plan)
    torch = _require_torch()
    zero = torch.zeros_like(official_editor[:, 0])
    if not bool(torch.equal(official_editor[:, 0], zero)):
        raise CrossModeMotionSpectrumError(
            "official_editor must have an exact zero first phase"
        )
    if not bool(torch.equal(plan[:, 0], zero)):
        raise CrossModeMotionSpectrumError("plan must have an exact zero first phase")

    raw_delta = q0(plan - official_editor)
    vector_width = int(raw_delta.shape[2]) * int(raw_delta.shape[3])
    rms_denominator = math.sqrt(vector_width)
    raw_rms = torch.linalg.vector_norm(raw_delta, dim=(2, 3)) / rms_denominator
    official_rms = (
        torch.linalg.vector_norm(official_editor, dim=(2, 3)) / rms_denominator
    )
    cap = float(config.max_plan_delta_ratio) * official_rms
    scale = torch.where(
        raw_rms > cap,
        cap / (raw_rms + float(config.epsilon)),
        torch.ones_like(raw_rms),
    )
    bounded = raw_delta * scale[..., None, None]
    if not bool(torch.equal(bounded[:, 0], zero)):
        raise CrossModeMotionSpectrumError(
            "plan bound changed the exact phase-zero boundary"
        )
    if not bool(torch.isfinite(bounded).all()):
        raise CrossModeMotionSpectrumError("bounded plan delta is non-finite")
    return BoundedPlanDiagnostics(
        raw_plan_delta=raw_delta,
        raw_plan_delta_rms=raw_rms,
        official_editor_rms=official_rms,
        delta_rms_cap=cap,
        bound_scale=scale,
        bounded_plan_delta=bounded,
    )


def execute_cmsg_plan(
    official_editor: Any,
    plan: Any,
    *,
    step_index: int,
    config: CrossModeMotionSpectrumConfig = CrossModeMotionSpectrumConfig(),
) -> ExecutedMotionSpectrumResult:
    """Execute ``official + rho * bounded(plan - official)``.

    At every zero-release step the returned ``executed_field`` is the exact
    same tensor object as ``official_editor``; no multiply or add is performed.
    """

    bound = bound_plan_delta(official_editor, plan, config=config)
    rho = release_rho(step_index)
    executed = (
        official_editor
        if rho == 0.0
        else official_editor + rho * bound.bounded_plan_delta
    )
    torch = _require_torch()
    if not bool(torch.equal(executed[:, 0], torch.zeros_like(executed[:, 0]))):
        raise CrossModeMotionSpectrumError(
            "CMSG execution changed the exact phase-zero boundary"
        )
    return ExecutedMotionSpectrumResult(
        official_editor=official_editor,
        plan=plan,
        rho=rho,
        bound=bound,
        executed_field=executed,
    )


# Descriptive aliases for integration sites that prefer the unabbreviated API.
build_cross_mode_motion_spectrum_plan = build_cmsg_plan
execute_cross_mode_motion_spectrum_plan = execute_cmsg_plan
