#!/usr/bin/env python3
"""Projected Differential Flow (PDF), an intentionally diagnostic bridge arm.

PDF fixes one concrete train/inference mismatch in CDF: both training and
sampling consume the same hard temporal-DC projection of the action-minus-noop
velocity field.  It is not a claim that temporal projection is a sufficient
motion representation.  Inference remains source-video + instruction only;
there is no target, mask, track, pose, trajectory, or frame anchor.

The solver uses at least two Euler substeps per scheduler interval.  This
reduces the last-interval discretisation error and records the magnitude of
every actual latent update, rather than only the unscaled field magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Optional, Sequence

import differential_sampler as cdf


class ProjectedFlowContractError(RuntimeError):
    pass


_DEFAULT_SUBSTEPS = 2
_LAST_TRACE: Optional["ProjectedFlowTrace"] = None


def set_default_substeps(value: int) -> None:
    global _DEFAULT_SUBSTEPS
    if type(value) is not int or value < 2 or value > 16:
        raise ProjectedFlowContractError("solver substeps must be an integer in [2, 16]")
    _DEFAULT_SUBSTEPS = value


def get_last_trace() -> Optional["ProjectedFlowTrace"]:
    return _LAST_TRACE


def _require_torch() -> Any:
    import torch
    return torch


def project_temporal_dc(field: Any, *, latent_frames: int = 21) -> Any:
    """Hard-project packed ``[B,N,D]`` fields onto the temporal zero-mean space."""

    if getattr(field, "ndim", None) != 3:
        raise ProjectedFlowContractError("field must have packed shape [B,N,D]")
    if type(latent_frames) is not int or latent_frames <= 1:
        raise ProjectedFlowContractError("latent_frames must be greater than one")
    tokens = int(field.shape[1])
    if tokens <= 0 or tokens % latent_frames:
        raise ProjectedFlowContractError("packed tokens are not divisible by latent_frames")
    grid = field.reshape(
        int(field.shape[0]), latent_frames, tokens // latent_frames, int(field.shape[2])
    )
    projected = grid - grid.mean(dim=1, keepdim=True)
    return projected.reshape_as(field)


def temporal_dc(field: Any, *, latent_frames: int = 21) -> Any:
    if getattr(field, "ndim", None) != 3:
        raise ProjectedFlowContractError("field must have packed shape [B,N,D]")
    tokens = int(field.shape[1])
    if tokens <= 0 or tokens % latent_frames:
        raise ProjectedFlowContractError("packed tokens are not divisible by latent_frames")
    return field.reshape(
        int(field.shape[0]), latent_frames, tokens // latent_frames, int(field.shape[2])
    ).mean(dim=1)


def shifted_inference_sigmas(
    *, num_steps: int = 40, flow_shift: float = 5.0
) -> tuple[float, ...]:
    """Return the analytic shifted-flow grid, including terminal zero."""

    if type(num_steps) is not int or num_steps <= 0:
        raise ProjectedFlowContractError("num_steps must be positive")
    if not math.isfinite(float(flow_shift)) or float(flow_shift) <= 0.0:
        raise ProjectedFlowContractError("flow_shift must be finite and positive")
    values = []
    for index in range(num_steps):
        base = 1.0 - index / num_steps
        values.append(float(flow_shift) * base / (1.0 + (float(flow_shift) - 1.0) * base))
    values.append(0.0)
    return tuple(values)


def integration_interval_weight(
    sigma: Any,
    *,
    num_steps: int = 40,
    flow_shift: float = 5.0,
    power: float = 1.0,
) -> Any:
    """Weight a training sigma by its nearest inference integration interval.

    Weights are normalised to mean one over solver intervals.  Unlike CDF's
    high-noise heuristic, this explicitly exposes the large terminal interval
    that materially dominated the failed sample's update trace.
    """

    torch = _require_torch()
    if not math.isfinite(float(power)) or float(power) <= 0.0:
        raise ProjectedFlowContractError("interval-weight power must be positive")
    schedule = shifted_inference_sigmas(num_steps=num_steps, flow_shift=flow_shift)
    centers = torch.as_tensor(
        [(a + b) * 0.5 for a, b in zip(schedule, schedule[1:])],
        device=sigma.device,
        dtype=torch.float32,
    )
    widths = torch.as_tensor(
        [abs(a - b) ** float(power) for a, b in zip(schedule, schedule[1:])],
        device=sigma.device,
        dtype=torch.float32,
    )
    widths = widths / widths.mean()
    values = sigma.float().reshape(-1).clamp(0.0, 1.0)
    nearest = (values[:, None] - centers[None, :]).abs().argmin(dim=1)
    return widths[nearest].reshape(sigma.shape)


@dataclass(frozen=True)
class DifferentialFlowConfig:
    num_inference_steps: int = 40
    flow_shift: float = 5.0
    seed: int = 20260806
    motion_scale: float = 1.0
    substeps: int = field(default_factory=lambda: _DEFAULT_SUBSTEPS)

    def validate(self) -> "DifferentialFlowConfig":
        cdf.DifferentialFlowConfig(
            self.num_inference_steps, self.flow_shift, self.seed, self.motion_scale
        ).validate()
        if type(self.substeps) is not int or not 2 <= self.substeps <= 16:
            raise ProjectedFlowContractError("substeps must be an integer in [2, 16]")
        return self


@dataclass(frozen=True)
class ProjectedFlowTrace:
    identity_bypassed: bool
    sigmas: tuple[float, ...]
    delta_rms: tuple[float, ...]
    contribution_rms: tuple[float, ...] = ()
    cumulative_update_rms: tuple[float, ...] = ()
    temporal_dc_rms_before_projection: tuple[float, ...] = ()
    interval_index: tuple[int, ...] = ()
    substep_index: tuple[int, ...] = ()


# Compatibility name used by infer_delta_lora's identity branch.
DifferentialFlowTrace = ProjectedFlowTrace


def sampler_contract() -> dict[str, Any]:
    return {
        "method": "bernini_projected_differential_flow_v2_diagnostic",
        "status": "diagnostic_bridge_not_final_method",
        "inference_conditions": ["source_video", "edit_instruction"],
        "forbidden_conditions": [
            "target_video", "mask", "track", "pose", "trajectory",
            "optical_flow", "first_frame_anchor",
        ],
        "field": "P_temporal_dc_zero(V_action-V_noop)",
        "train_inference_projection_identical": True,
        "temporal_dc_constraint": "hard_zero_mean_over_latent_time_per_spatial_token",
        "integrator": "descending_sigma_substep_euler",
        "minimum_substeps_per_scheduler_interval": 2,
        "trace": [
            "delta_rms", "contribution_rms", "cumulative_update_rms",
            "temporal_dc_rms_before_projection", "interval_index", "substep_index",
        ],
        "noise_policy": "one_fixed_shared_gaussian",
        "anc": False,
    }


def _identity_trace() -> ProjectedFlowTrace:
    return ProjectedFlowTrace(True, (), (), (), (), (), (), ())


def sample_differential_flow(
    renderer_or_diffusion: Any,
    *,
    source_latent: Any,
    action_prompt_embeds: Any,
    noop_prompt_embeds: Any,
    config: Optional[DifferentialFlowConfig] = None,
    return_trace: bool = False,
) -> Any:
    """Integrate the hard-projected action/no-op field with Euler substeps."""

    global _LAST_TRACE
    runtime = (config or DifferentialFlowConfig()).validate()
    identity = float(runtime.motion_scale) == 0.0 or cdf.prompts_are_exactly_identical(
        action_prompt_embeds, noop_prompt_embeds
    )
    if identity:
        trace = _identity_trace()
        _LAST_TRACE = trace
        return (source_latent, trace) if return_trace else source_latent

    torch = _require_torch()
    diffusion = cdf.resolve_diffusion_core(renderer_or_diffusion)
    layout, transformer = cdf._validate_runtime_inputs(
        diffusion, source_latent, action_prompt_embeds, noop_prompt_embeds
    )
    timesteps, intervals = cdf._set_scheduler_timesteps(
        diffusion,
        cdf.DifferentialFlowConfig(
            runtime.num_inference_steps, runtime.flow_shift, runtime.seed, runtime.motion_scale
        ),
        source_latent.device,
    )
    fixed_noise = cdf._make_fixed_noise(source_latent, seed=runtime.seed)
    source_clean = source_latent.detach().to(dtype=torch.float32)
    source_packed = cdf._pack_spatial_latent(source_clean, layout)
    noise_packed = cdf._pack_spatial_latent(fixed_noise, layout)
    edit_packed = source_packed.clone()
    initial_edit = edit_packed.clone()

    trace_sigma: list[float] = []
    delta_rms: list[float] = []
    contribution_rms: list[float] = []
    cumulative_rms: list[float] = []
    dc_rms: list[float] = []
    interval_ids: list[int] = []
    substep_ids: list[int] = []
    with torch.no_grad():
        source_condition = cdf._patch_source_condition(transformer, source_clean)
        for interval_index, (sigma_start, sigma_end) in enumerate(intervals):
            timestep_start = timesteps[interval_index].float()
            timestep_end = (
                timesteps[interval_index + 1].float()
                if interval_index + 1 < len(timesteps)
                else torch.zeros_like(timestep_start)
            )
            for substep_index in range(runtime.substeps):
                fraction = substep_index / runtime.substeps
                next_fraction = (substep_index + 1) / runtime.substeps
                sigma_value = sigma_start + (sigma_end - sigma_start) * fraction
                next_sigma = sigma_start + (sigma_end - sigma_start) * next_fraction
                # The scheduler's public timesteps do not necessarily scale
                # linearly with sigma after flow shifting.  Interpolate in the
                # same local interval only; this is still an explicit Euler
                # diagnostic rather than a claim of a higher-order solver.
                timestep = timestep_start + (timestep_end - timestep_start) * fraction
                sigma = torch.as_tensor(
                    sigma_value, device=source_clean.device, dtype=torch.float32
                )
                source_state_packed = (1.0 - sigma) * source_packed + sigma * noise_packed
                target_state_packed = edit_packed + source_state_packed - source_packed
                source_state = cdf._unpack_spatial_latent(source_state_packed, layout)
                target_state = cdf._unpack_spatial_latent(target_state_packed, layout)
                action_velocity = cdf._predict_source_conditioned_velocity(
                    diffusion=diffusion,
                    transformer=transformer,
                    source_condition=source_condition,
                    query_latent=target_state,
                    prompt_embeds=action_prompt_embeds,
                    timestep=timestep,
                )
                noop_velocity = cdf._predict_source_conditioned_velocity(
                    diffusion=diffusion,
                    transformer=transformer,
                    source_condition=source_condition,
                    query_latent=source_state,
                    prompt_embeds=noop_prompt_embeds,
                    timestep=timestep,
                )
                raw_delta = action_velocity.float() - noop_velocity.float()
                expected = (layout.batch, layout.tokens, layout.packed_channels)
                if tuple(int(x) for x in raw_delta.shape) != expected:
                    raise ProjectedFlowContractError("projected field shape differs from source")
                projected = project_temporal_dc(raw_delta, latent_frames=layout.frames)
                contribution = (
                    float(runtime.motion_scale) * float(next_sigma - sigma_value) * projected
                )
                edit_packed = edit_packed + contribution
                trace_sigma.append(float(sigma_value))
                delta_rms.append(float(projected.square().mean().sqrt().cpu().item()))
                contribution_rms.append(float(contribution.square().mean().sqrt().cpu().item()))
                cumulative_rms.append(
                    float((edit_packed - initial_edit).square().mean().sqrt().cpu().item())
                )
                dc_rms.append(
                    float(temporal_dc(raw_delta, latent_frames=layout.frames).square().mean().sqrt().cpu().item())
                )
                interval_ids.append(interval_index)
                substep_ids.append(substep_index)

    trace_sigma.append(0.0)
    result = cdf._unpack_spatial_latent(edit_packed, layout)
    trace = ProjectedFlowTrace(
        False, tuple(trace_sigma), tuple(delta_rms), tuple(contribution_rms),
        tuple(cumulative_rms), tuple(dc_rms), tuple(interval_ids), tuple(substep_ids)
    )
    _LAST_TRACE = trace
    return (result, trace) if return_trace else result


__all__ = [
    "DifferentialFlowConfig", "DifferentialFlowTrace", "ProjectedFlowTrace",
    "ProjectedFlowContractError", "get_last_trace", "integration_interval_weight",
    "project_temporal_dc", "sample_differential_flow", "sampler_contract",
    "set_default_substeps", "shifted_inference_sigmas", "temporal_dc",
]
