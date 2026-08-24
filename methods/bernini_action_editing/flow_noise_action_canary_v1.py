#!/usr/bin/env python3
"""Dense optical-flow transport of Bernini's native initial Gaussian.

The operator never derives RGB or latent appearance from the action anchor.
It receives only backward flow in latent-cell coordinates and transports the
Gaussian already drawn by Bernini.  A fresh native IID slice fills invalid or
occluded cells, and every phase/channel is renormalized to preserve the first
two Gaussian moments after bilinear interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


SCHEMA_VERSION = "bernini-flow-noise-action-canary-v1"
LATENT_PHASES = 21


class FlowNoiseActionError(RuntimeError):
    pass


@dataclass(frozen=True)
class FlowNoiseResult:
    initial_noise: Any
    receipt: dict[str, Any]


def _validate_inputs(baseline: Any, backward_flow: Any, validity: Any) -> None:
    import torch

    if (
        not isinstance(baseline, torch.Tensor)
        or baseline.ndim != 5
        or int(baseline.shape[0]) != 1
        or int(baseline.shape[2]) != LATENT_PHASES
        or not bool(torch.isfinite(baseline).all().item())
    ):
        raise FlowNoiseActionError("baseline must be finite [1,C,21,H,W]")
    expected_flow = (
        LATENT_PHASES - 1,
        2,
        int(baseline.shape[3]),
        int(baseline.shape[4]),
    )
    expected_validity = (
        LATENT_PHASES - 1,
        1,
        int(baseline.shape[3]),
        int(baseline.shape[4]),
    )
    if (
        not isinstance(backward_flow, torch.Tensor)
        or tuple(backward_flow.shape) != expected_flow
        or not bool(torch.isfinite(backward_flow).all().item())
    ):
        raise FlowNoiseActionError(
            f"backward flow must be finite {expected_flow}"
        )
    if (
        not isinstance(validity, torch.Tensor)
        or tuple(validity.shape) != expected_validity
        or not bool(torch.isfinite(validity).all().item())
        or bool((validity < 0).any().item())
        or bool((validity > 1).any().item())
    ):
        raise FlowNoiseActionError(
            f"validity must lie in [0,1] with shape {expected_validity}"
        )


def _backward_warp(value: Any, flow: Any) -> tuple[Any, Any]:
    import torch
    import torch.nn.functional as F

    _, _, height, width = value.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=value.device, dtype=torch.float32),
        torch.arange(width, device=value.device, dtype=torch.float32),
        indexing="ij",
    )
    sample_x = xx + flow[:, 0]
    sample_y = yy + flow[:, 1]
    inside = (
        (sample_x >= 0)
        & (sample_x <= max(width - 1, 0))
        & (sample_y >= 0)
        & (sample_y <= max(height - 1, 0))
    ).unsqueeze(1)
    grid_x = 2.0 * sample_x / max(width - 1, 1) - 1.0
    grid_y = 2.0 * sample_y / max(height - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    warped = F.grid_sample(
        value.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped, inside.to(dtype=torch.float32)


def _standardize(value: Any, reference: Any) -> Any:
    mean = value.mean(dim=(-2, -1), keepdim=True)
    std = value.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1.0e-5)
    normalized = (value - mean) / std
    reference_mean = reference.mean(dim=(-2, -1), keepdim=True)
    reference_std = reference.std(
        dim=(-2, -1), keepdim=True, unbiased=False
    ).clamp_min(1.0e-5)
    return normalized * reference_std + reference_mean


def build_flow_transported_noise(
    baseline: Any,
    backward_flow: Any,
    validity: Any,
    *,
    degradation: float,
    use_validity: bool = True,
) -> FlowNoiseResult:
    """Transport a matched IID Gaussian along one 21-phase motion field."""

    import torch

    _validate_inputs(baseline, backward_flow, validity)
    if (
        isinstance(degradation, bool)
        or not math.isfinite(float(degradation))
        or not 0.0 <= float(degradation) <= 1.0
    ):
        raise FlowNoiseActionError("degradation must lie in [0,1]")
    if type(use_validity) is not bool:
        raise FlowNoiseActionError("use_validity must be boolean")

    device = baseline.device
    original_dtype = baseline.dtype
    flow = backward_flow.to(device=device, dtype=torch.float32)
    confidence = validity.to(device=device, dtype=torch.float32)
    phases = [baseline[:, :, 0].float()]
    valid_fractions = []
    pre_normalization_std = []
    for phase_index in range(1, LATENT_PHASES):
        warped, inside = _backward_warp(
            phases[-1], flow[phase_index - 1 : phase_index]
        )
        mask = inside
        if use_validity:
            mask = mask * confidence[phase_index - 1 : phase_index]
        fresh = baseline[:, :, phase_index].float()
        propagated = mask * warped + (1.0 - mask) * fresh
        if float(degradation) > 0.0:
            propagated = (
                math.sqrt(1.0 - float(degradation)) * propagated
                + math.sqrt(float(degradation)) * fresh
            )
        pre_normalization_std.append(
            float(
                propagated.std(dim=(-2, -1), unbiased=False).mean().item()
            )
        )
        propagated = _standardize(propagated, fresh)
        phases.append(propagated)
        valid_fractions.append(float(mask.mean().item()))
    result = torch.stack(phases, dim=2).to(dtype=original_dtype).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise FlowNoiseActionError("flow-transported noise is non-finite")
    if torch.equal(result, baseline):
        raise FlowNoiseActionError("active flow operator returned baseline exactly")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "operator": "sequential_backward_flow_transport",
        "latent_shape": list(map(int, baseline.shape)),
        "degradation": float(degradation),
        "use_validity": use_validity,
        "valid_fraction_mean": sum(valid_fractions) / len(valid_fractions),
        "valid_fraction_min": min(valid_fractions),
        "pre_normalization_std_mean": sum(pre_normalization_std)
        / len(pre_normalization_std),
        "per_phase_channel_moment_match": True,
        "anchor_rgb_or_latent_copied": False,
        "native_baseline_draw_performed_first": True,
    }
    return FlowNoiseResult(initial_noise=result, receipt=receipt)


__all__ = [
    "FlowNoiseActionError",
    "FlowNoiseResult",
    "LATENT_PHASES",
    "SCHEMA_VERSION",
    "build_flow_transported_noise",
]
