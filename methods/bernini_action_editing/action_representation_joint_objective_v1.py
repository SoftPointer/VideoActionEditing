#!/usr/bin/env python3
"""Separated action/preservation objective for 2026-08-24 action routing.

This module intentionally does not accept target pixels, VAE latents, clean
latents, or absolute target hidden states.  Its teacher input is an already
detached, source-aligned action representation with shape ``[B,P,N,D]``.

The action and preservation arms remain separate through autograd.  They are
combined only by an audited preservation-priority conflict projection; a
single weighted reward is not exposed by this API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bernini-action-representation-joint-objective-v1"
REQUIRED_CONTROLS = (
    "zero_or_noop",
    "temporal_shuffle",
    "reverse",
    "incomplete",
    "wrong_action_energy_matched",
)


class ActionRepresentationObjectiveError(RuntimeError):
    """Raised before an invalid objective can update trainable parameters."""


def _fail(message: str) -> None:
    raise ActionRepresentationObjectiveError(message)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        _fail(f"objective contract is not canonical JSON: {error}")


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependent
        raise ActionRepresentationObjectiveError(
            "action representation objective requires PyTorch"
        ) from error
    return torch


@dataclass(frozen=True)
class JointObjectiveConfig:
    huber_delta: float = 0.10
    cosine_weight: float = 1.0
    local_huber_weight: float = 1.0
    energy_band_weight: float = 0.25
    minimum_energy_ratio: float = 0.50
    maximum_energy_ratio: float = 1.75
    counterfactual_margin: float = 0.10
    counterfactual_weight: float = 1.0
    onset_weight: float = 0.50
    transition_weight: float = 0.50
    terminal_weight: float = 0.50
    replay_weight: float = 1.0
    phase0_weight: float = 1.0
    outside_weight: float = 1.0
    normalization_floor: float = 1.0e-6
    pcgrad_epsilon: float = 1.0e-20
    maximum_gradient_norm: float = 1.0
    noop_trust_radius: float = 1.0e-8

    def validate(self) -> None:
        positive = (
            "huber_delta",
            "cosine_weight",
            "local_huber_weight",
            "energy_band_weight",
            "counterfactual_margin",
            "counterfactual_weight",
            "onset_weight",
            "transition_weight",
            "terminal_weight",
            "replay_weight",
            "phase0_weight",
            "outside_weight",
            "normalization_floor",
            "pcgrad_epsilon",
            "maximum_gradient_norm",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0.0:
                _fail(f"{name} must be finite and strictly positive")
        if (
            isinstance(self.minimum_energy_ratio, bool)
            or isinstance(self.maximum_energy_ratio, bool)
            or not math.isfinite(float(self.minimum_energy_ratio))
            or not math.isfinite(float(self.maximum_energy_ratio))
            or not 0.0 < float(self.minimum_energy_ratio) < 1.0
            or not 1.0 < float(self.maximum_energy_ratio)
        ):
            _fail("energy ratios must straddle one")
        if (
            isinstance(self.noop_trust_radius, bool)
            or not math.isfinite(float(self.noop_trust_radius))
            or float(self.noop_trust_radius) < 0.0
        ):
            _fail("noop_trust_radius must be finite and nonnegative")


@dataclass(frozen=True)
class JointObjectiveInputs:
    """Source-coordinate route residuals; only student tensors retain grad."""

    student_correct: Any
    student_controls: Mapping[str, Any]
    detached_teacher_correct: Any
    student_route_off: Any
    detached_frozen_route_off: Any
    action_activity: Any


@dataclass(frozen=True)
class JointObjectiveResult:
    action: Any
    preservation: Any
    action_components: Mapping[str, Any]
    preservation_components: Mapping[str, Any]
    diagnostics: Mapping[str, Any]


def objective_contract(
    config: JointObjectiveConfig = JointObjectiveConfig(),
) -> dict[str, Any]:
    config.validate()
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "teacher_abi": "detached_source_aligned_action_representation_BPND",
        "target_rgb_accepted": False,
        "target_vae_or_clean_latent_accepted": False,
        "target_absolute_hidden_or_value_accepted": False,
        "required_controls": list(REQUIRED_CONTROLS),
        "counterfactual_gate": "each_control_independent_no_weighted_compensation",
        "action_and_preservation_losses_separate": True,
        "single_weighted_reward_exposed": False,
        "conflict_policy": "project_action_gradient_against_preservation_then_add",
        "config": asdict(config),
    }
    value["contract_digest"] = _object_sha256(value)
    return value


def _validate_field(value: Any, *, label: str, requires_grad: bool | None) -> None:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 4
        or not bool(value.is_floating_point())
        or any(int(size) <= 0 for size in value.shape)
        or int(value.shape[1]) < 3
        or not bool(torch.isfinite(value).all().item())
    ):
        _fail(f"{label} must be finite [B,P>=3,N,D]")
    if requires_grad is not None and bool(value.requires_grad) is not requires_grad:
        _fail(f"{label} requires_grad differs")


def _validate_inputs(inputs: JointObjectiveInputs) -> Any:
    torch = _torch()
    _validate_field(inputs.student_correct, label="student_correct", requires_grad=True)
    _validate_field(
        inputs.detached_teacher_correct,
        label="detached_teacher_correct",
        requires_grad=False,
    )
    _validate_field(inputs.student_route_off, label="student_route_off", requires_grad=None)
    _validate_field(
        inputs.detached_frozen_route_off,
        label="detached_frozen_route_off",
        requires_grad=False,
    )
    shape = tuple(inputs.student_correct.shape)
    for label, value in (
        ("detached_teacher_correct", inputs.detached_teacher_correct),
        ("student_route_off", inputs.student_route_off),
        ("detached_frozen_route_off", inputs.detached_frozen_route_off),
    ):
        if tuple(value.shape) != shape:
            _fail(f"{label} geometry differs from student_correct")
        if value.device != inputs.student_correct.device:
            _fail(f"{label} device differs from student_correct")
    if not isinstance(inputs.student_controls, Mapping) or set(inputs.student_controls) != set(REQUIRED_CONTROLS):
        _fail("student_controls must contain the exact five G1 controls")
    for name in REQUIRED_CONTROLS:
        value = inputs.student_controls[name]
        _validate_field(value, label=f"student_controls.{name}", requires_grad=None)
        if tuple(value.shape) != shape or value.device != inputs.student_correct.device:
            _fail(f"student_controls.{name} geometry/device differs")
    mask = inputs.action_activity
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        _fail("action_activity must be a bool tensor")
    if mask.ndim == 3:
        mask = mask.unsqueeze(-1)
    if mask.ndim != 4 or tuple(mask.shape[:3]) != tuple(shape[:3]) or int(mask.shape[3]) != 1:
        _fail("action_activity must have shape [B,P,N] or [B,P,N,1]")
    if mask.device != inputs.student_correct.device:
        _fail("action_activity device differs")
    if not bool(mask.any().item()) or not bool((~mask).any().item()):
        _fail("action_activity must contain both active and outside tokens")
    if bool(mask[:, 0].any().item()):
        _fail("action_activity phase0 must be entirely inactive")
    if not bool(mask[:, 1].any().item()):
        _fail("action_activity phase1 must cover the onset transition")
    if not bool(mask[:, -1].any().item()):
        _fail("action_activity must cover the terminal phase")
    if bool(inputs.detached_teacher_correct.requires_grad) or bool(inputs.detached_frozen_route_off.requires_grad):
        _fail("teacher and frozen route-off must be detached")
    return mask


def _masked_mean(value: Any, mask: Any) -> Any:
    expanded = mask.expand_as(value)
    denominator = expanded.sum().clamp_min(1)
    return value.masked_select(expanded).sum() / denominator


def _masked_alignment_error(
    student: Any,
    teacher: Any,
    mask: Any,
    *,
    config: JointObjectiveConfig,
) -> tuple[Any, Any, Any]:
    torch = _torch()
    floor = float(config.normalization_floor)
    # Clamp the squared quantity before sqrt.  ``sqrt(0).clamp_min(floor)``
    # has an infinite local sqrt derivative and can yield NaN at the exact
    # zero-init residual even though its forward value is subsequently
    # clamped.  The pre-sqrt floor keeps both forward and backward finite.
    squared_floor = floor * floor
    student_norm = (
        student.float()
        .square()
        .sum(dim=-1, keepdim=True)
        .clamp_min(squared_floor)
        .sqrt()
    )
    teacher_norm = (
        teacher.float()
        .square()
        .sum(dim=-1, keepdim=True)
        .clamp_min(squared_floor)
        .sqrt()
    )
    student_unit = student.float() / student_norm
    teacher_unit = teacher.float() / teacher_norm
    cosine = 1.0 - (student_unit * teacher_unit).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    normalized_residual = (student.float() - teacher.float()) / teacher_norm.detach()
    huber = torch.nn.functional.huber_loss(
        normalized_residual,
        torch.zeros_like(normalized_residual),
        delta=float(config.huber_delta),
        reduction="none",
    )
    directional = mask & (teacher_norm > floor)
    cosine_loss = (
        _masked_mean(cosine, directional)
        if bool(directional.any().item())
        else cosine.sum().mul(0.0)
    )
    huber_loss = _masked_mean(huber, mask)
    total = float(config.cosine_weight) * cosine_loss + float(config.local_huber_weight) * huber_loss
    return total, cosine_loss, huber_loss


def _energy_band(
    student: Any,
    teacher: Any,
    mask: Any,
    config: JointObjectiveConfig,
) -> tuple[Any, Any, Any, Any]:
    torch = _torch()
    expanded = mask.expand_as(student)
    count = expanded.sum().clamp_min(1)
    student_mean_square = (
        student.float().square().masked_select(expanded).sum().div(count)
    )
    teacher_mean_square = (
        teacher.float().square().masked_select(expanded).sum().div(count)
    )
    squared_floor = float(config.normalization_floor) ** 2
    student_rms = student_mean_square.clamp_min(squared_floor).sqrt()
    teacher_rms = teacher_mean_square.clamp_min(squared_floor).sqrt().detach()
    if float(teacher_rms.item()) <= float(config.normalization_floor):
        _fail("teacher action energy is too small for admission/training")
    low = float(config.minimum_energy_ratio) * teacher_rms
    high = float(config.maximum_energy_ratio) * teacher_rms
    penalty = torch.relu(low - student_rms).square() + torch.relu(student_rms - high).square()
    # Raw RMS is diagnostics-only and detached before sqrt, so reporting the
    # true numeric zero cannot reintroduce the unstable autograd edge.
    student_raw_rms = student_mean_square.detach().sqrt()
    return penalty, student_rms, teacher_rms, student_raw_rms


def compute_joint_objectives(
    inputs: JointObjectiveInputs,
    config: JointObjectiveConfig = JointObjectiveConfig(),
) -> JointObjectiveResult:
    """Build separated losses without performing backward or an optimizer step."""

    torch = _torch()
    config.validate()
    activity = _validate_inputs(inputs)
    correct_alignment, correct_cosine, correct_huber = _masked_alignment_error(
        inputs.student_correct,
        inputs.detached_teacher_correct,
        activity,
        config=config,
    )
    energy_penalty, student_rms, teacher_rms, student_raw_rms = _energy_band(
        inputs.student_correct,
        inputs.detached_teacher_correct,
        activity,
        config,
    )
    control_errors: dict[str, Any] = {}
    margins: dict[str, Any] = {}
    for name in REQUIRED_CONTROLS:
        error, _, _ = _masked_alignment_error(
            inputs.student_controls[name],
            inputs.detached_teacher_correct,
            activity,
            config=config,
        )
        control_errors[name] = error
        margins[name] = torch.relu(
            correct_alignment - error + float(config.counterfactual_margin)
        )
    counterfactual = torch.stack(tuple(margins.values())).mean()

    student_delta = inputs.student_correct[:, 1:] - inputs.student_correct[:, :-1]
    teacher_delta = (
        inputs.detached_teacher_correct[:, 1:]
        - inputs.detached_teacher_correct[:, :-1]
    )
    transition_mask = activity[:, 1:] | activity[:, :-1]
    transition, _, _ = _masked_alignment_error(
        student_delta,
        teacher_delta,
        transition_mask,
        config=config,
    )
    onset_mask = transition_mask[:, :1]
    onset, _, _ = _masked_alignment_error(
        student_delta[:, :1],
        teacher_delta[:, :1],
        onset_mask,
        config=config,
    )
    terminal_state = torch.nn.functional.smooth_l1_loss(
        inputs.student_correct[:, -1].float(),
        inputs.detached_teacher_correct[:, -1].float(),
        reduction="none",
        beta=float(config.huber_delta),
    )
    terminal_velocity = torch.nn.functional.smooth_l1_loss(
        student_delta[:, -1].float(),
        teacher_delta[:, -1].float(),
        reduction="none",
        beta=float(config.huber_delta),
    )
    terminal_mask = activity[:, -1:].expand_as(inputs.student_correct[:, -1:])[:, 0]
    terminal = (
        terminal_state.masked_select(terminal_mask).mean()
        + terminal_velocity.masked_select(terminal_mask).mean()
    )

    action_components = {
        "alignment": correct_alignment,
        "energy_band": energy_penalty,
        "counterfactual_margin": counterfactual,
        "onset": onset,
        "ordered_transition": transition,
        "terminal_hold": terminal,
    }
    action = (
        correct_alignment
        + float(config.energy_band_weight) * energy_penalty
        + float(config.counterfactual_weight) * counterfactual
        + float(config.onset_weight) * onset
        + float(config.transition_weight) * transition
        + float(config.terminal_weight) * terminal
    )

    replay = torch.nn.functional.mse_loss(
        inputs.student_route_off.float(),
        inputs.detached_frozen_route_off.float(),
    )
    phase0 = inputs.student_correct[:, :1].float().square().mean()
    outside = inputs.student_correct.float().square().masked_select(
        (~activity).expand_as(inputs.student_correct)
    ).mean()
    preservation_components = {
        "exact_zero_route_replay": replay,
        "phase0_source_tether": phase0,
        "outside_action_tube_tether": outside,
    }
    preservation = (
        float(config.replay_weight) * replay
        + float(config.phase0_weight) * phase0
        + float(config.outside_weight) * outside
    )
    if not bool(torch.isfinite(action).item()) or not bool(torch.isfinite(preservation).item()):
        _fail("joint objective produced a non-finite scalar")
    return JointObjectiveResult(
        action=action,
        preservation=preservation,
        action_components=action_components,
        preservation_components=preservation_components,
        diagnostics={
            "correct_cosine": correct_cosine.detach(),
            "correct_huber": correct_huber.detach(),
            "student_action_rms": student_rms.detach(),
            "student_action_rms_raw": student_raw_rms,
            "teacher_action_rms": teacher_rms.detach(),
            "rms_stabilization": "mean_square_clamp_floor_squared_before_sqrt",
            "control_errors": {name: value.detach() for name, value in control_errors.items()},
            "independent_control_margins": {name: value.detach() for name, value in margins.items()},
        },
    )


@dataclass(frozen=True)
class PCGradResult:
    gradients: tuple[Any, ...]
    diagnostics: Mapping[str, Any]


def project_action_against_preservation_gradients(
    action_gradients: Sequence[Any],
    preservation_gradients: Sequence[Any],
    *,
    epsilon: float = 1.0e-20,
    maximum_gradient_norm: float = 1.0,
) -> PCGradResult:
    """Project conflicting action gradient, then add preservation gradient."""

    torch = _torch()
    action = tuple(action_gradients)
    preservation = tuple(preservation_gradients)
    if not action or len(action) != len(preservation):
        _fail("gradient lists must be equally non-empty")
    if (
        isinstance(epsilon, bool)
        or isinstance(maximum_gradient_norm, bool)
        or not math.isfinite(float(epsilon))
        or not math.isfinite(float(maximum_gradient_norm))
        or float(epsilon) <= 0.0
        or float(maximum_gradient_norm) <= 0.0
    ):
        _fail("gradient epsilon/norm must be finite and positive")
    if any(
        not isinstance(a, torch.Tensor)
        or not isinstance(p, torch.Tensor)
        or tuple(a.shape) != tuple(p.shape)
        or not bool(torch.isfinite(a).all().item())
        or not bool(torch.isfinite(p).all().item())
        for a, p in zip(action, preservation)
    ):
        _fail("action/preservation gradients must be finite aligned tensors")
    dot = sum((a.float() * p.float()).sum() for a, p in zip(action, preservation))
    action_sq = sum(a.float().square().sum() for a in action)
    preservation_sq = sum(p.float().square().sum() for p in preservation)
    conflict = bool(float(dot.detach().item()) < 0.0)
    coefficient = (
        dot / preservation_sq.clamp_min(float(epsilon))
        if conflict
        else dot.new_zeros(())
    )
    projected_action = tuple(
        a - coefficient.to(device=a.device, dtype=a.dtype) * p
        for a, p in zip(action, preservation)
    )
    projection_delta_sq = sum(
        (projected.float() - original.float()).square().sum()
        for projected, original in zip(projected_action, action)
    )
    combined = tuple(a + p for a, p in zip(projected_action, preservation))
    combined_sq = sum(value.float().square().sum() for value in combined)
    combined_norm = combined_sq.sqrt()
    clip_scale = torch.minimum(
        torch.ones_like(combined_norm),
        combined_norm.new_tensor(float(maximum_gradient_norm))
        / combined_norm.clamp_min(float(epsilon)),
    )
    combined = tuple(value * clip_scale.to(dtype=value.dtype) for value in combined)
    cosine = dot / (
        action_sq.sqrt().mul(preservation_sq.sqrt()).clamp_min(float(epsilon))
    )
    return PCGradResult(
        gradients=combined,
        diagnostics={
            "action_preservation_dot": dot.detach(),
            "action_preservation_cosine": cosine.detach(),
            "action_norm": action_sq.sqrt().detach(),
            "preservation_norm": preservation_sq.sqrt().detach(),
            "conflict_projected": conflict,
            "action_projection_norm": projection_delta_sq.sqrt().detach(),
            "combined_preclip_norm": combined_norm.detach(),
            "gradient_clip_scale": clip_scale.detach(),
        },
    )


def backward_with_preservation_pcgrad(
    result: JointObjectiveResult,
    parameters: Sequence[Any],
    config: JointObjectiveConfig = JointObjectiveConfig(),
) -> Mapping[str, Any]:
    """Compute two gradient arms, enforce trust region, and assign ``.grad``."""

    torch = _torch()
    config.validate()
    params = tuple(parameters)
    if not params or len({id(parameter) for parameter in params}) != len(params):
        _fail("trainable parameter closure must be non-empty and unique")
    if any(not isinstance(parameter, torch.nn.Parameter) or not parameter.requires_grad for parameter in params):
        _fail("PCGrad parameters must be trainable torch Parameters")
    replay = result.preservation_components.get("exact_zero_route_replay")
    if replay is None or not isinstance(replay, torch.Tensor) or replay.numel() != 1:
        _fail("exact zero-route replay diagnostic is missing")
    replay_value = float(replay.detach().item())
    if (
        not math.isfinite(replay_value)
        or replay_value < 0.0
        or replay_value > float(config.noop_trust_radius)
    ):
        _fail(
            "no-op trust region rejected update: "
            f"replay={replay_value:.9g} radius={float(config.noop_trust_radius):.9g}"
        )
    if (
        not isinstance(result.preservation, torch.Tensor)
        or result.preservation.numel() != 1
    ):
        _fail("preservation objective must be a scalar tensor")
    preservation_value = float(result.preservation.detach().item())
    if not math.isfinite(preservation_value):
        _fail("preservation objective must be finite")
    action_raw = torch.autograd.grad(
        result.action,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    preservation_raw = torch.autograd.grad(
        result.preservation,
        params,
        retain_graph=False,
        allow_unused=True,
    )
    action = tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(params, action_raw)
    )
    preservation = tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(params, preservation_raw)
    )
    if not any(bool(torch.count_nonzero(value).item()) for value in action):
        _fail("action objective produced no trainable gradient")
    preservation_gradient_nonzero = any(
        bool(torch.count_nonzero(value).item()) for value in preservation
    )
    zero_preservation_fallback = not preservation_gradient_nonzero
    preservation_scalar_within_trust = (
        0.0 <= preservation_value <= float(config.noop_trust_radius)
    )
    if zero_preservation_fallback and not preservation_scalar_within_trust:
        _fail(
            "zero preservation gradient cannot use the initialization "
            "fail-safe outside the preservation trust boundary: "
            f"preservation={preservation_value:.9g} "
            f"radius={float(config.noop_trust_radius):.9g}"
        )
    projected = project_action_against_preservation_gradients(
        action,
        preservation,
        epsilon=float(config.pcgrad_epsilon),
        maximum_gradient_norm=float(config.maximum_gradient_norm),
    )
    for parameter, gradient in zip(params, projected.gradients):
        parameter.grad = gradient.detach().clone()
    return {
        **dict(projected.diagnostics),
        "gradient_combination_mode": (
            "initial_zero_preservation_action_only_failsafe"
            if zero_preservation_fallback
            else "preservation_priority_pcgrad"
        ),
        "zero_preservation_gradient_fallback": zero_preservation_fallback,
        "preservation_gradient_nonzero": preservation_gradient_nonzero,
        "preservation_scalar": result.preservation.detach(),
        "preservation_scalar_within_noop_trust": (
            preservation_scalar_within_trust
        ),
        "fallback_establishes_tp_preservation": False,
        "noop_replay": replay.detach(),
        "noop_replay_within_trust": True,
        "noop_trust_radius": float(config.noop_trust_radius),
        "action_unused_parameter_count": sum(item is None for item in action_raw),
        "preservation_unused_parameter_count": sum(
            item is None for item in preservation_raw
        ),
    }


__all__ = [
    "ActionRepresentationObjectiveError",
    "JointObjectiveConfig",
    "JointObjectiveInputs",
    "JointObjectiveResult",
    "PCGradResult",
    "REQUIRED_CONTROLS",
    "SCHEMA_VERSION",
    "backward_with_preservation_pcgrad",
    "compute_joint_objectives",
    "objective_contract",
    "project_action_against_preservation_gradients",
]
