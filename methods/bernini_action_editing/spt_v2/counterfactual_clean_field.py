#!/usr/bin/env python3
"""Same-state counterfactual clean-field execution for Bernini SPT.

This module owns only the tensor algebra at one denoising state.  The caller
is responsible for obtaining *post official CFG/APG* action and no-op clean
predictions from the exact same noisy sample ``y`` and scalar ``sigma``.  The
core then forms

``delta = x_action - x_noop``
``x_cf = source + alpha * delta``

and executes ``P*source + T*transport(source) + G*x_cf``.  ``x_cf`` is always
ungated: the generate gate is applied exactly once by ``execute_clean_plan``.

There is deliberately no target-video, mask, track, flow, pose, first-frame
anchor, PEFT, model-forward, scheduler callback, or custom integrator API here.
This algebra layer cannot prove same-state provenance by itself; the audited
tri-branch sampler hook owns that proof and the sole official UniPC call.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from .phase_transport import (
    LATENT_PHASES,
    PhasePlan,
    PhaseTransportError,
    execute_clean_plan,
    packed_to_video,
    velocity_from_clean,
    video_to_packed,
)


MIN_SCHEDULER_SIGMA = 1e-4


class CounterfactualCleanFieldError(PhaseTransportError):
    """Raised before scheduler integration when the clean-field contract fails."""


@dataclass(frozen=True)
class CounterfactualStepRecord:
    """Tensor-free diagnostics for one successfully integrated shared state."""

    sigma: float
    alpha: float
    delta_mean_square: float
    delta_rms: float
    source_anchor_displacement_rms: float
    noop_action_parity_rms_error: float
    noop_action_parity_max_abs_error: float
    preserve_gate_mass: float
    transport_gate_mass: float
    generate_gate_mass: float


@dataclass
class CounterfactualTrace:
    """Serializable records for frozen-base and later joint-routing diagnosis."""

    records: list[CounterfactualStepRecord] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract": runtime_contract(),
            "step_count": len(self.records),
            "steps": [asdict(record) for record in self.records],
        }


def runtime_contract() -> dict[str, Any]:
    """Return a tensor-free description of the deployable boundary."""

    return {
        "method": "bernini-spt-same-state-counterfactual-clean-field-v1",
        "status": "algebra-only-requires-authoritative-same-state-sampler-hook",
        "inference_conditions": ["source_video", "edit_instruction"],
        "internal_noop_condition": True,
        "forbidden_conditions": [
            "target_video",
            "paired_oracle_plan",
            "mask",
            "track",
            "pose",
            "optical_flow",
            "trajectory",
            "first_frame_anchor",
        ],
        "prediction_boundary": "post_official_cfg_apg_clean_prediction",
        "same_state_obligation": "one_identical_noisy_y_and_sigma_for_action_and_noop",
        "same_state_enforced_here": False,
        "same_state_authority": "tri_branch_unipc_hook",
        "delta_formula": "x_action_clean-x_noop_clean",
        "counterfactual_formula": "source_clean+alpha*delta_clean",
        "parity_control_formula": "x_noop_clean+delta_clean=x_action_clean",
        "parity_control_scope": "algebraic_identity_not_same_state_evidence",
        "execution_formula": (
            "P*source_clean+T*transport(source_clean)+G*x_counterfactual_clean"
        ),
        "generate_gate_application_count": 1,
        "diagnostics": [
            "delta_mean_square",
            "delta_rms",
            "source_anchor_displacement_rms",
            "noop_action_parity_rms_error",
            "noop_action_parity_max_abs_error",
            "preserve_gate_mass",
            "transport_gate_mass",
            "generate_gate_mass",
        ],
        "required_plan_provenance": "student",
        "integrator": "owned_externally_by_tri_branch_unipc_hook",
        "custom_integrator": False,
        "zero_sigma_policy": "fail_before_velocity_projection",
        "minimum_scheduler_sigma": MIN_SCHEDULER_SIGMA,
        "latent_phases": LATENT_PHASES,
        "peft_dependency": False,
    }


def _tensor_device(value: Any) -> Any:
    return getattr(value, "device", None)


def _validate_clean_video(value: Any, *, label: str) -> None:
    import torch

    if not isinstance(value, torch.Tensor):
        raise CounterfactualCleanFieldError(f"{label} must be a torch.Tensor")
    if value.ndim != 5:
        raise CounterfactualCleanFieldError(f"{label} must be [B,T,H,W,D]")
    if int(value.shape[1]) != LATENT_PHASES:
        raise CounterfactualCleanFieldError(
            f"{label} must contain exactly {LATENT_PHASES} latent phases"
        )
    if any(int(size) <= 0 for size in value.shape):
        raise CounterfactualCleanFieldError(f"{label} has an empty dimension")
    if not torch.is_floating_point(value):
        raise CounterfactualCleanFieldError(f"{label} must be floating point")
    if not bool(torch.isfinite(value).all()):
        raise CounterfactualCleanFieldError(f"{label} contains non-finite values")


def _validate_same_layout(reference: Any, value: Any, *, label: str) -> None:
    _validate_clean_video(value, label=label)
    if tuple(value.shape) != tuple(reference.shape):
        raise CounterfactualCleanFieldError(
            f"{label} shape differs from the shared denoising state"
        )
    if _tensor_device(value) != _tensor_device(reference):
        raise CounterfactualCleanFieldError(
            f"{label} device differs from the shared denoising state"
        )


def _validate_plan(plan: PhasePlan, source: Any) -> None:
    if not isinstance(plan, PhasePlan):
        raise CounterfactualCleanFieldError("plan must be a PhasePlan")
    if plan.provenance != "student":
        raise CounterfactualCleanFieldError(
            "counterfactual inference requires a source+instruction student plan"
        )
    plan.validate(source)
    for label, value in (("plan offsets", plan.offsets), ("plan gates", plan.gate_probs)):
        if _tensor_device(value) != _tensor_device(source):
            raise CounterfactualCleanFieldError(f"{label} device differs from source")


def _coerce_alpha(alpha: Any) -> float:
    if isinstance(alpha, bool):
        raise CounterfactualCleanFieldError("alpha must be a finite non-negative scalar")
    try:
        if hasattr(alpha, "numel") and int(alpha.numel()) != 1:
            raise CounterfactualCleanFieldError("alpha must be scalar")
        value = float(alpha.detach().cpu().item() if hasattr(alpha, "detach") else alpha)
    except CounterfactualCleanFieldError:
        raise
    except Exception as error:
        raise CounterfactualCleanFieldError(
            "alpha must be a finite non-negative scalar"
        ) from error
    if not math.isfinite(value) or value < 0.0:
        raise CounterfactualCleanFieldError(
            "alpha must be a finite non-negative scalar"
        )
    return value


def _validate_sigma(sigma: Any) -> None:
    if isinstance(sigma, bool):
        raise CounterfactualCleanFieldError(
            f"sigma must be finite and >= {MIN_SCHEDULER_SIGMA}"
        )
    try:
        if hasattr(sigma, "numel") and int(sigma.numel()) != 1:
            raise CounterfactualCleanFieldError("scheduler sigma must be scalar")
        value = float(sigma.detach().cpu().item() if hasattr(sigma, "detach") else sigma)
    except CounterfactualCleanFieldError:
        raise
    except Exception as error:
        raise CounterfactualCleanFieldError("scheduler sigma must be scalar") from error
    if not math.isfinite(value) or value < MIN_SCHEDULER_SIGMA:
        raise CounterfactualCleanFieldError(
            f"sigma must be finite and >= {MIN_SCHEDULER_SIGMA}"
        )


def _scalar(value: Any) -> float:
    return float(value.detach().float().cpu().item())


def _step_record(
    *,
    sigma: Any,
    alpha: Any,
    source: Any,
    action_clean: Any,
    noop_clean: Any,
    plan: PhasePlan,
) -> CounterfactualStepRecord:
    import torch

    sigma_value = float(
        sigma.detach().cpu().item() if hasattr(sigma, "detach") else sigma
    )
    alpha_value = _coerce_alpha(alpha)
    delta = same_state_clean_delta(action_clean, noop_clean)
    mean_square = delta.square().mean()
    parity_error = noop_clean.float() + delta - action_clean.float()
    masses = plan.gate_probs.float().mean(dim=(0, 2, 3, 4))
    if not bool(torch.isclose(masses.sum(), torch.ones_like(masses.sum()), atol=2e-5, rtol=0.0)):
        raise CounterfactualCleanFieldError("mean P/T/G gate masses must sum to one")
    displacement = alpha_value * delta
    # The subtraction explicitly re-evaluates the primary field identity; it
    # is kept as a diagnostic instead of assuming scale*delta by definition.
    field = source.float() + displacement
    return CounterfactualStepRecord(
        sigma=sigma_value,
        alpha=alpha_value,
        delta_mean_square=_scalar(mean_square),
        delta_rms=_scalar(torch.sqrt(mean_square)),
        source_anchor_displacement_rms=_scalar(
            torch.sqrt((field - source.float()).square().mean())
        ),
        noop_action_parity_rms_error=_scalar(
            torch.sqrt(parity_error.square().mean())
        ),
        noop_action_parity_max_abs_error=_scalar(parity_error.abs().max()),
        preserve_gate_mass=_scalar(masses[0]),
        transport_gate_mass=_scalar(masses[1]),
        generate_gate_mass=_scalar(masses[2]),
    )


def same_state_clean_delta(action_clean: Any, noop_clean: Any) -> Any:
    """Return ``x_action - x_noop`` in float32 from one shared denoising state."""

    _validate_clean_video(action_clean, label="action_clean")
    _validate_same_layout(action_clean, noop_clean, label="noop_clean")
    return action_clean.float() - noop_clean.float()


def counterfactual_clean_field(
    source: Any,
    action_clean: Any,
    noop_clean: Any,
    *,
    alpha: Any = 1.0,
) -> Any:
    """Build the ungated source-anchored field ``source + alpha*(action-noop)``."""

    _validate_clean_video(source, label="source")
    _validate_same_layout(source, action_clean, label="action_clean")
    _validate_same_layout(source, noop_clean, label="noop_clean")
    scale = _coerce_alpha(alpha)
    delta = same_state_clean_delta(action_clean, noop_clean)
    # Do not multiply by plan.G here.  The sole G multiplication happens in
    # execute_clean_plan, after P/T/G have been validated as a partition.
    return source.float() + scale * delta


def execute_counterfactual_clean_plan(
    *,
    source: Any,
    action_clean: Any,
    noop_clean: Any,
    plan: PhasePlan,
    alpha: Any = 1.0,
    detach_source_bank: bool = True,
) -> Any:
    """Execute one student P/T/G plan using the ungated counterfactual field."""

    if type(detach_source_bank) is not bool:
        raise CounterfactualCleanFieldError("detach_source_bank must be boolean")
    _validate_clean_video(source, label="source")
    _validate_same_layout(source, action_clean, label="action_clean")
    _validate_same_layout(source, noop_clean, label="noop_clean")
    _validate_plan(plan, source)
    field = counterfactual_clean_field(
        source, action_clean, noop_clean, alpha=alpha
    )
    return execute_clean_plan(
        source,
        field,
        plan,
        detach_source_bank=detach_source_bank,
    )


def counterfactual_plan_velocity_with_diagnostics(
    *,
    noisy: Any,
    sigma: Any,
    source: Any,
    action_clean: Any,
    noop_clean: Any,
    plan: PhasePlan,
    alpha: Any = 1.0,
    detach_source_bank: bool = True,
) -> tuple[Any, CounterfactualStepRecord]:
    """Project one plan and return tensor-free same-state diagnostics."""

    _validate_sigma(sigma)
    _validate_clean_video(noisy, label="noisy")
    _validate_same_layout(noisy, source, label="source")
    _validate_same_layout(noisy, action_clean, label="action_clean")
    _validate_same_layout(noisy, noop_clean, label="noop_clean")
    executed = execute_counterfactual_clean_plan(
        source=source,
        action_clean=action_clean,
        noop_clean=noop_clean,
        plan=plan,
        alpha=alpha,
        detach_source_bank=detach_source_bank,
    )
    velocity = velocity_from_clean(
        noisy, executed, sigma, eps=MIN_SCHEDULER_SIGMA
    )
    record = _step_record(
        sigma=sigma,
        alpha=alpha,
        source=source,
        action_clean=action_clean,
        noop_clean=noop_clean,
        plan=plan,
    )
    return velocity, record


def counterfactual_plan_velocity(
    *,
    noisy: Any,
    sigma: Any,
    source: Any,
    action_clean: Any,
    noop_clean: Any,
    plan: PhasePlan,
    alpha: Any = 1.0,
    detach_source_bank: bool = True,
) -> Any:
    """Project the executed clean plan to the official scheduler velocity."""

    _validate_sigma(sigma)
    _validate_clean_video(noisy, label="noisy")
    _validate_same_layout(noisy, source, label="source")
    _validate_same_layout(noisy, action_clean, label="action_clean")
    _validate_same_layout(noisy, noop_clean, label="noop_clean")
    executed = execute_counterfactual_clean_plan(
        source=source,
        action_clean=action_clean,
        noop_clean=noop_clean,
        plan=plan,
        alpha=alpha,
        detach_source_bank=detach_source_bank,
    )
    return velocity_from_clean(
        noisy, executed, sigma, eps=MIN_SCHEDULER_SIGMA
    )


def counterfactual_packed_velocity_with_diagnostics(
    *,
    noisy_packed: Any,
    sigma: Any,
    source_packed: Any,
    action_clean_packed: Any,
    noop_clean_packed: Any,
    height: int,
    width: int,
    plan: PhasePlan,
    alpha: Any = 1.0,
    detach_source_bank: bool = True,
) -> tuple[Any, CounterfactualStepRecord]:
    """Packed adapter returning velocity plus frozen-base diagnostics."""

    packed = {
        "noisy_packed": noisy_packed,
        "source_packed": source_packed,
        "action_clean_packed": action_clean_packed,
        "noop_clean_packed": noop_clean_packed,
    }
    shapes = {label: tuple(getattr(value, "shape", ())) for label, value in packed.items()}
    if len(set(shapes.values())) != 1:
        raise CounterfactualCleanFieldError(
            "all packed clean/noisy tensors must share one exact shape"
        )
    try:
        videos = {
            label: packed_to_video(value, height=height, width=width)
            for label, value in packed.items()
        }
    except PhaseTransportError as error:
        raise CounterfactualCleanFieldError(str(error)) from error
    velocity, record = counterfactual_plan_velocity_with_diagnostics(
        noisy=videos["noisy_packed"],
        sigma=sigma,
        source=videos["source_packed"],
        action_clean=videos["action_clean_packed"],
        noop_clean=videos["noop_clean_packed"],
        plan=plan,
        alpha=alpha,
        detach_source_bank=detach_source_bank,
    )
    return video_to_packed(velocity), record


def counterfactual_packed_velocity(
    *,
    noisy_packed: Any,
    sigma: Any,
    source_packed: Any,
    action_clean_packed: Any,
    noop_clean_packed: Any,
    height: int,
    width: int,
    plan: PhasePlan,
    alpha: Any = 1.0,
    detach_source_bank: bool = True,
) -> Any:
    """Packed ``[B,T*H*W,D]`` adapter for the Bernini sampling boundary."""

    packed = {
        "noisy_packed": noisy_packed,
        "source_packed": source_packed,
        "action_clean_packed": action_clean_packed,
        "noop_clean_packed": noop_clean_packed,
    }
    shapes = {label: tuple(getattr(value, "shape", ())) for label, value in packed.items()}
    if len(set(shapes.values())) != 1:
        raise CounterfactualCleanFieldError(
            "all packed clean/noisy tensors must share one exact shape"
        )
    try:
        videos = {
            label: packed_to_video(value, height=height, width=width)
            for label, value in packed.items()
        }
    except PhaseTransportError as error:
        raise CounterfactualCleanFieldError(str(error)) from error
    velocity = counterfactual_plan_velocity(
        noisy=videos["noisy_packed"],
        sigma=sigma,
        source=videos["source_packed"],
        action_clean=videos["action_clean_packed"],
        noop_clean=videos["noop_clean_packed"],
        plan=plan,
        alpha=alpha,
        detach_source_bank=detach_source_bank,
    )
    return video_to_packed(velocity)


__all__ = [
    "CounterfactualCleanFieldError",
    "CounterfactualStepRecord",
    "CounterfactualTrace",
    "MIN_SCHEDULER_SIGMA",
    "counterfactual_clean_field",
    "counterfactual_packed_velocity",
    "counterfactual_packed_velocity_with_diagnostics",
    "counterfactual_plan_velocity",
    "counterfactual_plan_velocity_with_diagnostics",
    "execute_counterfactual_clean_plan",
    "runtime_contract",
    "same_state_clean_delta",
]
