#!/usr/bin/env python3
"""Auditable training core for Bernini Cross-Mode CMSG LoRA v6.

This module deliberately stops at the boundary that can be verified without a
real Bernini checkpoint and a pinned multi-rank runtime.  It provides:

* the exact 46-module LoRA scope (all 30 cross-attention ``to_q`` projections
  plus self-attention ``to_q`` in blocks 7 through 22), rank/alpha 8/8;
* the only legal editor-to-generator batch conversion, delegated to
  :mod:`cross_mode_branches`, whose generator tensors are target-tail views of
  the editor tensors and therefore cannot be rebuilt or resampled;
* a differentiable, pure-torch generator-to-editor distillation cell for
  editor direction, log amplitude, cross-mode spectral consistency,
  high-frequency detail, and late frozen replay;
* a non-differentiable frozen-prior eligibility gate; and
* exact late routing through :func:`cross_mode_motion_spectrum.execute_cmsg_plan`.

The target-only generator and paired target are training-only teachers.  The
generator's signed Q0 temporal increments may teach a direction only after
their agreement with ground-truth target-minus-source motion passes the frozen
prior gate.  Deployment remains the official source-video + instruction
editor path: no generator branch, paired target, mask, optical flow, pose,
track, or other inference-time oracle is accepted.

The generator action is tokenized with Bernini's official T2V system prompt,
not by reusing the editor/MV2V token ids.  Only latent/RoPE target-tail storage
and diffusion timestep are shared across modes; text geometry remains
mode-native.

This distinction is essential: the scalar spectrum plan in
``cross_mode_motion_spectrum`` cannot invent a direction missing from the
editor.  Here the frozen generator direction is distilled into the editor
LoRA during training; generator values are never routed at inference.

The full Bernini forward/optimizer/receipt path is intentionally *not* claimed
to be integrated here.  Executing this file without ``--preflight-only`` fails
closed.  A later integration must bind adapter-on/off branch order, official
APG reconstruction, Ulysses gradient reduction, and inference receipt parity
before it may perform an optimizer update.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import re
from typing import Any, Mapping, Optional, Sequence

import cross_mode_branches as cross_mode
import cross_mode_motion_spectrum as spectrum


METHOD_NAME = "bernini-cross-mode-cmsg-lora-v6"
RECEIPT_SCHEMA = "bernini-r-1p3b-cross-mode-cmsg-lora-receipt-v6"
NUM_FRAMES = 81
LATENT_PHASES = 21
LORA_RANK = 8
LORA_ALPHA = 8
LORA_SCOPE = "cross_q_plus_mid_self_q"
MIDDLE_SELF_BLOCKS = (7, 22)
EXPECTED_CROSS_Q_MODULES = 30
EXPECTED_MIDDLE_SELF_Q_MODULES = 16
EXPECTED_LORA_MODULES = (
    EXPECTED_CROSS_Q_MODULES + EXPECTED_MIDDLE_SELF_Q_MODULES
)
INFERENCE_CONDITIONS = ("source_video", "action_instruction")
TRAINING_ONLY_CONDITIONS = (
    "paired_target_video",
    "frozen_target_only_generator_teacher",
)
GENERATOR_ACTION_TEXT_CONTRACT = (
    "official_t2v_system_prompt_plus_action_instruction"
)
GENERATOR_NEGATIVE_TEXT_CONTRACT = "official_t2v_negative_prompt_verbatim"
FORBIDDEN_INFERENCE_CONDITIONS = (
    "target_video",
    "mask",
    "track",
    "swept_tube",
    "pose",
    "trajectory",
    "optical_flow",
    "first_frame_anchor",
)

_TARGET_RE = re.compile(
    r"^diff_dec\.transformer\.blocks\.(?P<block>\d+)\."
    r"attn(?P<attention>[12])\.to_q$"
)


class CrossModeCMSGTrainingError(RuntimeError):
    """Raised before an unsupported or scientifically invalid update."""


class FrozenPriorGateRejected(CrossModeCMSGTrainingError):
    """Raised when the frozen cross-mode prior does not support the target."""


@dataclass(frozen=True)
class CMSGTrainingLossConfig:
    """Weights and frozen-prior eligibility thresholds for the v6 core."""

    editor_direction_weight: float = 1.0
    log_amplitude_weight: float = 0.25
    generator_spectrum_weight: float = 0.10
    high_frequency_detail_weight: float = 0.25
    late_frozen_replay_weight: float = 0.10
    charbonnier_scale: float = 0.10
    active_relative_floor: float = 0.05
    gate_min_active_phases: int = 2
    gate_min_mean_cosine: float = 0.25
    gate_phase_min_cosine: float = 0.10
    gate_max_log_amplitude_error: float = math.log(3.0)
    gate_min_coverage: float = 0.50
    gate_max_normalized_rmse: float = 1.50
    epsilon: float = 1.0e-6
    enforce_frozen_prior_gate: bool = True

    def validate(self) -> None:
        positive_weights = (
            "editor_direction_weight",
            "log_amplitude_weight",
            "generator_spectrum_weight",
            "high_frequency_detail_weight",
            "late_frozen_replay_weight",
        )
        for name in positive_weights:
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
                raise CrossModeCMSGTrainingError(
                    f"{name} must be finite and non-negative"
                )
        for name in ("charbonnier_scale", "epsilon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                raise CrossModeCMSGTrainingError(
                    f"{name} must be finite and strictly positive"
                )
        if (
            isinstance(self.active_relative_floor, bool)
            or not math.isfinite(float(self.active_relative_floor))
            or not 0.0 < float(self.active_relative_floor) <= 1.0
        ):
            raise CrossModeCMSGTrainingError(
                "active_relative_floor must lie in (0,1]"
            )
        if (
            type(self.gate_min_active_phases) is not int
            or not 1 <= self.gate_min_active_phases < LATENT_PHASES
        ):
            raise CrossModeCMSGTrainingError(
                "gate_min_active_phases must be an integer in [1,20]"
            )
        for name in ("gate_min_mean_cosine", "gate_phase_min_cosine"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or not -1.0 <= float(value) <= 1.0
            ):
                raise CrossModeCMSGTrainingError(f"{name} must lie in [-1,1]")
        if (
            isinstance(self.gate_min_coverage, bool)
            or not math.isfinite(float(self.gate_min_coverage))
            or not 0.0 <= float(self.gate_min_coverage) <= 1.0
        ):
            raise CrossModeCMSGTrainingError(
                "gate_min_coverage must lie in [0,1]"
            )
        for name in (
            "gate_max_log_amplitude_error",
            "gate_max_normalized_rmse",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
                raise CrossModeCMSGTrainingError(
                    f"{name} must be finite and non-negative"
                )
        if type(self.enforce_frozen_prior_gate) is not bool:
            raise CrossModeCMSGTrainingError(
                "enforce_frozen_prior_gate must be boolean"
            )


@dataclass(frozen=True)
class FrozenPriorGateResult:
    """Detached per-sample evidence for accepting a frozen CMSG prior."""

    passed: Any
    active_phase_count: Any
    mean_direction_cosine: Any
    log_amplitude_mae: Any
    covered_phase_fraction: Any
    normalized_rmse: Any
    frozen_prior_rms: Any
    target_motion_rms: Any


@dataclass(frozen=True)
class DistilledEditorExecution:
    """Editor-only deployment field with exact late adapter-off replay."""

    frozen_editor: Any
    adapted_editor: Any
    rho: float
    executed_field: Any


@dataclass(frozen=True)
class CMSGTrainingLossResult:
    """All loss components and the exact field presented to integration."""

    total: Any
    editor_direction: Any
    log_amplitude: Any
    generator_spectral_consistency: Any
    high_frequency_detail: Any
    late_frozen_replay: Any
    rho: float
    frozen_prior_gate: FrozenPriorGateResult
    frozen_editor_direction: Any
    generator_teacher_direction: Any
    student_editor_direction: Any
    cross_mode_diagnostics: Any
    student_execution: Any


def select_cmsg_lora_targets(available_modules: Sequence[str]) -> list[str]:
    """Select 30 cross ``to_q`` plus blocks 7..22 self ``to_q`` modules."""

    if isinstance(available_modules, (str, bytes)):
        raise CrossModeCMSGTrainingError(
            "available_modules must be a sequence of fully-qualified names"
        )
    names = list(available_modules)
    if not names or not all(isinstance(name, str) and name for name in names):
        raise CrossModeCMSGTrainingError(
            "available_modules must contain non-empty strings"
        )
    if len(set(names)) != len(names):
        raise CrossModeCMSGTrainingError("available_modules contains duplicates")

    start, end = MIDDLE_SELF_BLOCKS
    selected: list[str] = []
    cross_blocks: set[int] = set()
    middle_self_blocks: set[int] = set()
    for name in names:
        match = _TARGET_RE.fullmatch(name)
        if match is None:
            continue
        block = int(match.group("block"))
        attention = int(match.group("attention"))
        if attention == 2 and 0 <= block < EXPECTED_CROSS_Q_MODULES:
            selected.append(name)
            cross_blocks.add(block)
        elif attention == 1 and start <= block <= end:
            selected.append(name)
            middle_self_blocks.add(block)

    expected_cross = set(range(EXPECTED_CROSS_Q_MODULES))
    expected_middle = set(range(start, end + 1))
    if cross_blocks != expected_cross or middle_self_blocks != expected_middle:
        raise CrossModeCMSGTrainingError(
            "CMSG LoRA scope must contain all 30 cross to_q and self to_q "
            "for blocks 7..22"
        )
    selected = sorted(selected)
    if len(selected) != EXPECTED_LORA_MODULES:
        raise CrossModeCMSGTrainingError(
            f"CMSG LoRA scope resolved {len(selected)} modules, expected "
            f"{EXPECTED_LORA_MODULES}"
        )
    return selected


def lora_contract(available_modules: Sequence[str]) -> dict[str, Any]:
    """Return a serialization-ready immutable LoRA scope contract."""

    targets = select_cmsg_lora_targets(available_modules)
    return {
        "scope": LORA_SCOPE,
        "rank": LORA_RANK,
        "alpha": LORA_ALPHA,
        "dropout": 0.0,
        "bias": "none",
        "target_module_count": len(targets),
        "target_modules": targets,
        "middle_self_blocks_inclusive": list(MIDDLE_SELF_BLOCKS),
    }


def build_training_branches(
    editor_action_batch: Mapping[str, Any],
    generator_action_text_fields: Mapping[str, Any],
    generator_negative_text_fields: Mapping[str, Any],
) -> cross_mode.CrossModeBranches:
    """Build official-T2V teacher branches from editor target-tail views.

    The generator action text is intentionally distinct from the editor text:
    Bernini's MV2V and T2V system prefixes are different.  Reusing editor token
    ids would query neither the proven frozen T2V prior nor the deployed editor
    distribution and is therefore rejected by the delegated branch contract.
    """

    try:
        result = cross_mode.build_generator_branches(
            editor_action_batch,
            generator_action_text_fields,
            generator_negative_text_fields,
        )
    except TypeError as error:
        raise CrossModeCMSGTrainingError(
            "cross_mode_branches lacks the required separate official-T2V "
            "action/negative text API"
        ) from error
    except cross_mode.CrossModeBranchError as error:
        raise CrossModeCMSGTrainingError(str(error)) from error

    # The dependency validates exact values.  Training additionally requires
    # storage aliasing, because an equal clone would weaken the no-resampling
    # provenance claim.
    try:
        editor_latents = result.editor_action["input_vae_latents"]
        editor_rope = result.editor_action["input_vae_rope"]
        generator_latents = result.generator_action["input_vae_latents"]
        generator_rope = result.generator_action["input_vae_rope"]
        latent_alias = (
            editor_latents.untyped_storage().data_ptr()
            == generator_latents.untyped_storage().data_ptr()
        )
        rope_alias = (
            editor_rope.untyped_storage().data_ptr()
            == generator_rope.untyped_storage().data_ptr()
        )
    except (AttributeError, KeyError, RuntimeError) as error:
        raise CrossModeCMSGTrainingError(
            "cannot certify generator target-tail storage provenance"
        ) from error
    if not latent_alias or not rope_alias:
        raise CrossModeCMSGTrainingError(
            "generator latent and RoPE must be target-tail views, never copies"
        )
    return result


def _validate_clean_fields(*fields: Any) -> None:
    """Translate the spectrum module's strict [B,21,S,D] contract."""

    try:
        spectrum._validate_fields(*fields)
    except spectrum.CrossModeMotionSpectrumError as error:
        raise CrossModeCMSGTrainingError(str(error)) from error


def _require_frozen(label: str, value: Any) -> None:
    if bool(getattr(value, "requires_grad", False)):
        raise CrossModeCMSGTrainingError(
            f"{label} must be adapter-off and graph-free"
        )


def _phase_increments(value: Any) -> Any:
    import torch

    zero = torch.zeros_like(value[:, :1])
    return torch.cat((zero, value[:, 1:] - value[:, :-1]), dim=1)


def _phase_rms(value: Any, *, epsilon: float = 0.0) -> Any:
    import torch

    mean_square = value.square().mean(dim=(2, 3))
    if epsilon > 0.0:
        mean_square = mean_square + float(epsilon) ** 2
    return torch.sqrt(mean_square)


def _phase_cosine(left: Any, right: Any, *, epsilon: float) -> Any:
    import torch

    dot = (left * right).mean(dim=(2, 3))
    left_rms = _phase_rms(left, epsilon=epsilon)
    right_rms = _phase_rms(right, epsilon=epsilon)
    return torch.clamp(dot / (left_rms * right_rms), min=-1.0, max=1.0)


def _active_target_phases(target_increments: Any, *, relative_floor: float) -> Any:
    import torch

    target_rms = _phase_rms(target_increments)
    maximum = target_rms.amax(dim=1, keepdim=True)
    active = target_rms >= float(relative_floor) * maximum
    active = active & (maximum > 0.0) & (target_rms > 0.0)
    active[:, 0] = False
    return active


def _masked_sample_mean(value: Any, mask: Any) -> Any:
    import torch

    count = mask.sum(dim=1)
    numerator = torch.where(mask, value, torch.zeros_like(value)).sum(dim=1)
    return numerator / count.clamp_min(1).to(dtype=value.dtype)


def _masked_scalar_mean(value: Any, mask: Any) -> Any:
    import torch

    numerator = torch.where(mask, value, torch.zeros_like(value)).sum()
    return numerator / mask.sum().clamp_min(1).to(dtype=value.dtype)


def compute_frozen_prior_gate(
    frozen_prior_field: Any,
    target_motion_field: Any,
    *,
    config: CMSGTrainingLossConfig = CMSGTrainingLossConfig(),
) -> FrozenPriorGateResult:
    """Measure whether a frozen generator prior supports target motion.

    The gate is deliberately detached and cannot become a reward that the
    adapter learns to game.  Coverage requires both signed direction agreement
    and an amplitude ratio within a factor of three on active target phases.
    """

    import torch

    config.validate()
    _validate_clean_fields(frozen_prior_field, target_motion_field)
    _require_frozen("frozen_prior_field", frozen_prior_field)
    _require_frozen("target_motion_field", target_motion_field)
    with torch.no_grad():
        frozen = spectrum.q0(frozen_prior_field.float())
        target = spectrum.q0(target_motion_field.float())
        frozen_increments = _phase_increments(frozen)
        target_increments = _phase_increments(target)
        active = _active_target_phases(
            target_increments,
            relative_floor=float(config.active_relative_floor),
        )
        count = active.sum(dim=1)
        cosine = _phase_cosine(
            frozen_increments,
            target_increments,
            epsilon=float(config.epsilon),
        )
        frozen_amp = _phase_rms(frozen_increments)
        target_amp = _phase_rms(target_increments)
        log_error = (
            torch.log(frozen_amp + float(config.epsilon))
            - torch.log(target_amp + float(config.epsilon))
        ).abs()
        mean_cosine = _masked_sample_mean(cosine, active)
        log_amplitude_mae = _masked_sample_mean(log_error, active)
        covered = (
            (cosine >= float(config.gate_phase_min_cosine))
            & (log_error <= float(config.gate_max_log_amplitude_error))
            & active
        )
        coverage = covered.sum(dim=1).to(torch.float32) / count.clamp_min(1).to(
            torch.float32
        )
        residual_rms = _phase_rms(frozen_increments - target_increments)
        target_increment_rms = _phase_rms(target_increments)
        normalized_rmse = _masked_sample_mean(
            residual_rms / (target_increment_rms + float(config.epsilon)),
            active,
        )
        passed = (
            (count >= int(config.gate_min_active_phases))
            & (mean_cosine >= float(config.gate_min_mean_cosine))
            & (
                log_amplitude_mae
                <= float(config.gate_max_log_amplitude_error)
            )
            & (coverage >= float(config.gate_min_coverage))
            & (normalized_rmse <= float(config.gate_max_normalized_rmse))
        )
        return FrozenPriorGateResult(
            passed=passed.detach(),
            active_phase_count=count.detach(),
            mean_direction_cosine=mean_cosine.detach(),
            log_amplitude_mae=log_amplitude_mae.detach(),
            covered_phase_fraction=coverage.detach(),
            normalized_rmse=normalized_rmse.detach(),
            frozen_prior_rms=frozen.square().mean(dim=(1, 2, 3)).sqrt().detach(),
            target_motion_rms=target.square().mean(dim=(1, 2, 3)).sqrt().detach(),
        )


def _charbonnier(left: Any, right: Any, *, scale: float) -> Any:
    import torch

    residual = left - right
    return (torch.sqrt(residual.square() + float(scale) ** 2) - float(scale)).mean()


def _resolve_spatial_hw(value: Any, spatial_hw: tuple[int, int]) -> tuple[int, int]:
    if (
        type(spatial_hw) is not tuple
        or len(spatial_hw) != 2
        or any(type(item) is not int or item <= 0 for item in spatial_hw)
        or spatial_hw[0] * spatial_hw[1] != int(value.shape[2])
    ):
        raise CrossModeCMSGTrainingError(
            "spatial_hw must be positive integers whose product equals S"
        )
    return spatial_hw


def _high_pass(value: Any, *, spatial_hw: tuple[int, int]) -> Any:
    import torch

    height, width = _resolve_spatial_hw(value, spatial_hw)
    batch, phases, _, channels = value.shape
    images = (
        value.reshape(batch, phases, height, width, channels)
        .permute(0, 1, 4, 2, 3)
        .reshape(batch * phases, channels, height, width)
    )
    padded = torch.nn.functional.pad(images, (1, 1, 1, 1), mode="replicate")
    low = torch.nn.functional.avg_pool2d(padded, kernel_size=3, stride=1)
    high = images - low
    return (
        high.reshape(batch, phases, channels, height, width)
        .permute(0, 1, 3, 4, 2)
        .reshape_as(value)
    )


def _cross_mode_spectral_consistency(
    cross_mode_plan: Any,
    active: Any,
    *,
    epsilon: float,
) -> Any:
    """Match editor spectra to the frozen generator teacher on active phases."""

    diagnostics = cross_mode_plan.diagnostics
    terms = []
    for editor_name, generator_name in (
        (
            "editor_spatial_energy_profile",
            "generator_spatial_energy_profile",
        ),
        (
            "editor_channel_energy_profile",
            "generator_channel_energy_profile",
        ),
    ):
        editor_energy = getattr(diagnostics, editor_name)
        generator_energy = getattr(diagnostics, generator_name)
        log_error = (
            (
                (editor_energy + float(epsilon)).log()
                - (generator_energy + float(epsilon)).log()
            )
            .abs()
            .mean(dim=2)
        )
        terms.append(_masked_scalar_mean(log_error, active))
    spatial_alignment = _masked_scalar_mean(
        1.0 - diagnostics.spatial_increment_cosine, active
    )
    channel_alignment = _masked_scalar_mean(
        1.0 - diagnostics.channel_increment_cosine, active
    )
    return (
        terms[0] + terms[1] + spatial_alignment + channel_alignment
    ) / 4.0


def execute_distilled_editor(
    frozen_editor_direction: Any,
    adapted_editor_direction: Any,
    *,
    step_index: int,
) -> DistilledEditorExecution:
    """Release the distilled editor early and return frozen prior exactly late.

    The target-only generator is intentionally absent from this signature.
    It is a training teacher, not a deployment branch.  At rho one the adapted
    editor object is returned directly; at rho zero the frozen editor object is
    returned directly, with no multiply/add that could perturb late detail.
    """

    import torch

    _validate_clean_fields(frozen_editor_direction, adapted_editor_direction)
    zero = torch.zeros_like(frozen_editor_direction[:, 0])
    if not bool(torch.equal(frozen_editor_direction[:, 0], zero)):
        raise CrossModeCMSGTrainingError(
            "frozen editor direction must have exact zero phase zero"
        )
    if not bool(torch.equal(adapted_editor_direction[:, 0], zero)):
        raise CrossModeCMSGTrainingError(
            "adapted editor direction must have exact zero phase zero"
        )
    try:
        rho = spectrum.release_rho(step_index)
    except spectrum.CrossModeMotionSpectrumError as error:
        raise CrossModeCMSGTrainingError(str(error)) from error
    if rho == 0.0:
        executed = frozen_editor_direction
    elif rho == 1.0:
        executed = adapted_editor_direction
    else:
        executed = frozen_editor_direction + rho * (
            adapted_editor_direction - frozen_editor_direction
        )
    if not bool(torch.equal(executed[:, 0], zero)) or not bool(
        torch.isfinite(executed).all()
    ):
        raise CrossModeCMSGTrainingError(
            "distilled editor execution violated the causal finite contract"
        )
    return DistilledEditorExecution(
        frozen_editor=frozen_editor_direction,
        adapted_editor=adapted_editor_direction,
        rho=float(rho),
        executed_field=executed,
    )


def compute_cmsg_lora_loss(
    *,
    adapted_editor_action_field: Any,
    frozen_editor_action_field: Any,
    editor_noop_field: Any,
    frozen_generator_action_field: Any,
    generator_uncond_field: Any,
    target_motion_field: Any,
    step_index: int,
    spatial_hw: tuple[int, int],
    loss_config: CMSGTrainingLossConfig = CMSGTrainingLossConfig(),
    spectrum_config: spectrum.CrossModeMotionSpectrumConfig = (
        spectrum.CrossModeMotionSpectrumConfig()
    ),
) -> CMSGTrainingLossResult:
    """Distill a frozen target-only generator direction into the editor LoRA.

    Frozen editor/no-op/generator/unconditional fields must be graph-free; the
    adapted editor action is the sole differentiable branch.  The generator's
    Q0 temporal increments supervise signed direction and cross-mode spectra,
    but only after agreeing with target-minus-source motion at the frozen-prior
    gate.  At deployment only the adapted editor direction is released.  Every
    zero-rho step returns the exact adapter-off editor tensor object.
    """

    import torch

    loss_config.validate()
    spectrum_config.validate()
    fields = (
        adapted_editor_action_field,
        frozen_editor_action_field,
        editor_noop_field,
        frozen_generator_action_field,
        generator_uncond_field,
        target_motion_field,
    )
    _validate_clean_fields(*fields)
    _resolve_spatial_hw(adapted_editor_action_field, spatial_hw)
    for label, value in (
        ("frozen_editor_action_field", frozen_editor_action_field),
        ("editor_noop_field", editor_noop_field),
        ("frozen_generator_action_field", frozen_generator_action_field),
        ("generator_uncond_field", generator_uncond_field),
        ("target_motion_field", target_motion_field),
    ):
        _require_frozen(label, value)
    if not bool(adapted_editor_action_field.requires_grad):
        raise CrossModeCMSGTrainingError(
            "adapted editor action must carry the sole LoRA gradient graph"
        )

    try:
        frozen_editor_direction = spectrum.q0(
            frozen_editor_action_field - editor_noop_field
        )
        student_editor_direction = spectrum.q0(
            adapted_editor_action_field - editor_noop_field
        )
        generator_teacher_direction = spectrum.q0(
            frozen_generator_action_field - generator_uncond_field
        )
        # The public CMSG plan is used here only for cross-mode spectrum
        # diagnostics/loss.  Its scalar-modulated plan is never executed and
        # is absent from the inference signature.
        cross_mode_diagnostics = spectrum.build_cmsg_plan(
            adapted_editor_action_field,
            editor_noop_field,
            frozen_generator_action_field,
            generator_uncond_field,
            spatial_hw=spatial_hw,
            config=spectrum_config,
        )
        student_execution = execute_distilled_editor(
            frozen_editor_direction,
            student_editor_direction,
            step_index=step_index,
        )
    except spectrum.CrossModeMotionSpectrumError as error:
        raise CrossModeCMSGTrainingError(str(error)) from error

    gate = compute_frozen_prior_gate(
        generator_teacher_direction.detach(),
        target_motion_field,
        config=loss_config,
    )
    # At rho==0 the generator teacher contributes exactly no value and the
    # sole objective is adapter-off replay.  Rejecting that update because an
    # unused teacher disagrees with the pair would bias the preservation
    # rehearsal stream and can deadlock late schedule strata.  Keep computing
    # the detached diagnostic, but enforce eligibility only when a teacher
    # direction can actually reach the student loss.
    if (
        student_execution.rho > 0.0
        and loss_config.enforce_frozen_prior_gate
        and not bool(gate.passed.all().item())
    ):
        raise FrozenPriorGateRejected(
            "frozen generator prior failed direction/amplitude/coverage gate"
        )

    target = spectrum.q0(target_motion_field)
    student = student_execution.executed_field
    student_increments = _phase_increments(student)
    teacher_increments = _phase_increments(generator_teacher_direction)
    target_increments = _phase_increments(target)
    active = _active_target_phases(
        target_increments,
        relative_floor=float(loss_config.active_relative_floor),
    )
    cosine = _phase_cosine(
        student_increments,
        teacher_increments,
        epsilon=float(loss_config.epsilon),
    )
    editor_direction = _masked_scalar_mean(1.0 - cosine, active)
    student_amp = _phase_rms(student_increments)
    target_amp = _phase_rms(target_increments)
    amplitude_error = (
        torch.log(student_amp + float(loss_config.epsilon))
        - torch.log(target_amp + float(loss_config.epsilon))
    ).abs()
    log_amplitude = _masked_scalar_mean(amplitude_error, active)
    generator_spectral = _cross_mode_spectral_consistency(
        cross_mode_diagnostics,
        active,
        epsilon=float(loss_config.epsilon),
    )
    student_high = _high_pass(student, spatial_hw=spatial_hw)
    frozen_high = _high_pass(
        frozen_editor_direction, spatial_hw=spatial_hw
    )
    high_frequency_detail = _charbonnier(
        student_high,
        frozen_high,
        scale=float(loss_config.charbonnier_scale),
    )
    replay_editor = _charbonnier(
        adapted_editor_action_field,
        frozen_editor_action_field,
        scale=float(loss_config.charbonnier_scale),
    )
    late_frozen_replay = replay_editor
    rho = float(student_execution.rho)
    motion_weight = rho
    replay_weight = 1.0 - rho
    total = (
        motion_weight
        * (
            float(loss_config.editor_direction_weight) * editor_direction
            + float(loss_config.log_amplitude_weight) * log_amplitude
            + float(loss_config.generator_spectrum_weight)
            * generator_spectral
            + float(loss_config.high_frequency_detail_weight)
            * high_frequency_detail
        )
        + float(loss_config.late_frozen_replay_weight)
        * replay_weight
        * late_frozen_replay
    )
    if not bool(torch.isfinite(total)):
        raise CrossModeCMSGTrainingError("CMSG total loss is non-finite")
    return CMSGTrainingLossResult(
        total=total,
        editor_direction=editor_direction,
        log_amplitude=log_amplitude,
        generator_spectral_consistency=generator_spectral,
        high_frequency_detail=high_frequency_detail,
        late_frozen_replay=late_frozen_replay,
        rho=rho,
        frozen_prior_gate=gate,
        frozen_editor_direction=frozen_editor_direction,
        generator_teacher_direction=generator_teacher_direction,
        student_editor_direction=student_editor_direction,
        cross_mode_diagnostics=cross_mode_diagnostics,
        student_execution=student_execution,
    )


def canonical_attention_modules() -> list[str]:
    """Return a synthetic name universe for lightweight contract preflight."""

    projections = ("to_q", "to_k", "to_v", "to_out.0")
    return [
        f"diff_dec.transformer.blocks.{block}.attn{attention}.{projection}"
        for block in range(30)
        for attention in (1, 2)
        for projection in projections
    ]


def preflight_contract(
    available_modules: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Validate the executable core while explicitly denying full integration."""

    modules = (
        canonical_attention_modules()
        if available_modules is None
        else list(available_modules)
    )
    contract = lora_contract(modules)
    spectrum.CrossModeMotionSpectrumConfig().validate()
    schedule = spectrum.release_rho_schedule()
    if (
        len(schedule) != 40
        or schedule[0] != 1.0
        or schedule[31] != 0.0
        or schedule[32:] != (0.0,) * 8
    ):
        raise CrossModeCMSGTrainingError("CMSG release schedule differs")
    return {
        "method": METHOD_NAME,
        "schema_version": RECEIPT_SCHEMA,
        "core_api_ready": True,
        "full_bernini_training_integrated": False,
        "optimizer_updates_authorized": False,
        "frames": NUM_FRAMES,
        "latent_phases": LATENT_PHASES,
        "lora": contract,
        "release_schedule": list(schedule),
        "inference_conditions": list(INFERENCE_CONDITIONS),
        "training_only_conditions": list(TRAINING_ONLY_CONDITIONS),
        "forbidden_inference_conditions": list(
            FORBIDDEN_INFERENCE_CONDITIONS
        ),
        "generator_state_provenance": (
            "direct editor target-tail latent/RoPE views; shared timestep; "
            "never rebuilt or resampled"
        ),
        "distillation_direction": (
            "training-only frozen generator Q0 temporal increments -> editor LoRA; "
            "ground-truth target-source motion gates alignment and amplitude"
        ),
        "generator_action_text_contract": GENERATOR_ACTION_TEXT_CONTRACT,
        "generator_negative_text_contract": GENERATOR_NEGATIVE_TEXT_CONTRACT,
        "inference_execution": (
            "official source-video + instruction editor path only; generator "
            "teacher absent; exact adapter-off late replay"
        ),
        "scalar_spectrum_limitation": (
            "scalar spectrum modulation cannot invent a missing editor direction"
        ),
        "blocking_integrations": [
            "pinned Bernini adapter-on/off forward ordering",
            "official editor APG and target-only generator CFG clean-field reconstruction",
            "four-rank Ulysses LoRA gradient reduction",
            "checkpoint/optimizer/inference receipt parity",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight the Bernini Cross-Mode CMSG LoRA v6 core"
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate pure contracts without claiming an optimizer update",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.preflight_only:
        raise CrossModeCMSGTrainingError(
            "full Bernini CMSG training is not yet integrated or authorized; "
            "only --preflight-only is executable"
        )
    print(json.dumps(preflight_contract(), sort_keys=True))
    return 0


__all__ = [
    "CMSGTrainingLossConfig",
    "CMSGTrainingLossResult",
    "CrossModeCMSGTrainingError",
    "DistilledEditorExecution",
    "FrozenPriorGateRejected",
    "FrozenPriorGateResult",
    "build_parser",
    "build_training_branches",
    "canonical_attention_modules",
    "compute_cmsg_lora_loss",
    "compute_frozen_prior_gate",
    "execute_distilled_editor",
    "lora_contract",
    "main",
    "preflight_contract",
    "select_cmsg_lora_targets",
]


if __name__ == "__main__":
    raise SystemExit(main())
