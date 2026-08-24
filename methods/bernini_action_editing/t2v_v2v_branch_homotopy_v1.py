#!/usr/bin/env python3
"""Pointwise homotopy from a pure T2V action prior to source-only V2V.

Both inputs are already-guided, scheduler-bound Bernini velocities evaluated
at the same packed target state.  The high-noise branch is official target-only
T2V APG with a T2V-native action prompt.  The low-noise branch is the stock
source-video-only ``v2v_apg`` field with an MV2V-native action prompt.  This
module only performs the pinned FP32 velocity interpolation; it does not call a
model, reconstruct APG, or advance a scheduler.

For active flow sigma ``s``, the pure-T2V weight is::

    h(s) = 0                                  if s <= 0.75
           u^2 (3 - 2u), u=(s-.75)/.20        if 0.75 < s < 0.95
           1                                  if s >= 0.95

Endpoints return the selected input tensor object directly.  Transition
arithmetic is FP32 and is cast back to the common branch dtype.  APG momentum
must be exactly zero so both inputs remain pointwise fields at one state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


SIGMA_LOW = 0.75
SIGMA_HIGH = 0.95
STEP_TRACE_SCHEMA = "bernini-t2v-v2v-branch-homotopy-step-v1"


class T2VV2VBranchHomotopyError(RuntimeError):
    """Raised before integration when the homotopy contract fails."""


@dataclass(frozen=True)
class T2VV2VBranchHomotopyStep:
    """One combined velocity and a JSON-safe audit trace."""

    velocity: Any
    sigma: float
    high_pure_t2v_weight: float
    low_source_v2v_weight: float
    endpoint: str

    def trace_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STEP_TRACE_SCHEMA,
            "sigma": self.sigma,
            "high_pure_t2v_weight": self.high_pure_t2v_weight,
            "low_source_v2v_weight": self.low_source_v2v_weight,
            "endpoint": self.endpoint,
            "endpoint_exact": self.endpoint != "transition",
            "high_branch": "official_pure_t2v_apg_target_only",
            "low_branch": "stock_source_video_only_v2v_apg",
            "interpolation_space": "scheduler_bound_velocity",
            "interpolation_dtype": "float32",
            "apg_momentum": 0.0,
        }


def _require_torch() -> Any:
    try:
        import torch
    except Exception as error:  # pragma: no cover - exercised on AUH
        raise T2VV2VBranchHomotopyError(
            "PyTorch is required for T2V/V2V branch homotopy"
        ) from error
    return torch


def _validate_sigma(sigma: Any) -> Any:
    torch = _require_torch()
    if (
        not isinstance(sigma, torch.Tensor)
        or sigma.ndim != 0
        or sigma.dtype != torch.float32
        or not bool(torch.isfinite(sigma).item())
        or not bool((sigma > 0).item())
    ):
        raise T2VV2VBranchHomotopyError(
            "sigma must be one finite positive FP32 scalar tensor"
        )
    return sigma


def _validate_zero_momentum(value: Any, *, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) != 0.0
    ):
        raise T2VV2VBranchHomotopyError(f"{label} must be exactly zero")


def _validate_geometry(
    target_packed_state: Any,
    high_pure_t2v_apg_velocity: Any,
    low_source_v2v_apg_velocity: Any,
) -> None:
    torch = _require_torch()
    tensors = (
        ("target packed state", target_packed_state),
        ("high pure-T2V APG velocity", high_pure_t2v_apg_velocity),
        ("low source-only V2V APG velocity", low_source_v2v_apg_velocity),
    )
    for label, value in tensors:
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 3
            or value.numel() <= 0
            or not value.dtype.is_floating_point
            or not bool(torch.isfinite(value).all().item())
        ):
            raise T2VV2VBranchHomotopyError(
                f"{label} must be one finite non-empty floating [B,N,C] tensor"
            )

    expected_shape = tuple(int(item) for item in target_packed_state.shape)
    expected_device = target_packed_state.device
    for label, value in tensors[1:]:
        if tuple(int(item) for item in value.shape) != expected_shape:
            raise T2VV2VBranchHomotopyError(
                f"{label} does not share the target packed-state shape"
            )
        if value.device != expected_device:
            raise T2VV2VBranchHomotopyError(
                f"{label} does not share the target packed-state device"
            )
    if high_pure_t2v_apg_velocity.dtype != low_source_v2v_apg_velocity.dtype:
        raise T2VV2VBranchHomotopyError(
            "high/low APG velocities must share one scheduler-bound dtype"
        )


def smoothstep_pure_t2v_weight(sigma: Any) -> Any:
    """Return the pinned high-noise pure-T2V weight as scalar FP32."""

    torch = _require_torch()
    sigma = _validate_sigma(sigma)
    if bool((sigma <= SIGMA_LOW).item()):
        return torch.zeros_like(sigma, dtype=torch.float32)
    if bool((sigma >= SIGMA_HIGH).item()):
        return torch.ones_like(sigma, dtype=torch.float32)

    low = torch.tensor(SIGMA_LOW, dtype=torch.float32, device=sigma.device)
    span = torch.tensor(
        SIGMA_HIGH - SIGMA_LOW,
        dtype=torch.float32,
        device=sigma.device,
    )
    three = torch.tensor(3.0, dtype=torch.float32, device=sigma.device)
    two = torch.tensor(2.0, dtype=torch.float32, device=sigma.device)
    unit = (sigma - low) / span
    weight = unit.square() * (three - two * unit)
    if weight.dtype != torch.float32 or not bool(torch.isfinite(weight).item()):
        raise T2VV2VBranchHomotopyError(
            "FP32 T2V/V2V smoothstep produced an invalid weight"
        )
    return weight


def t2v_v2v_branch_homotopy_step(
    target_packed_state: Any,
    high_pure_t2v_apg_velocity: Any,
    low_source_v2v_apg_velocity: Any,
    sigma: Any,
    *,
    high_pure_t2v_momentum: float = 0.0,
    low_source_v2v_momentum: float = 0.0,
) -> T2VV2VBranchHomotopyStep:
    """Combine two independently guided velocities at one target state."""

    torch = _require_torch()
    _validate_zero_momentum(
        high_pure_t2v_momentum,
        label="high pure-T2V APG momentum",
    )
    _validate_zero_momentum(
        low_source_v2v_momentum,
        label="low source-only V2V APG momentum",
    )
    _validate_geometry(
        target_packed_state,
        high_pure_t2v_apg_velocity,
        low_source_v2v_apg_velocity,
    )
    weight = smoothstep_pure_t2v_weight(sigma)
    high_weight = float(weight.item())
    sigma_value = float(sigma.item())

    if high_weight == 0.0:
        return T2VV2VBranchHomotopyStep(
            velocity=low_source_v2v_apg_velocity,
            sigma=sigma_value,
            high_pure_t2v_weight=0.0,
            low_source_v2v_weight=1.0,
            endpoint="low_source_v2v_apg",
        )
    if high_weight == 1.0:
        return T2VV2VBranchHomotopyStep(
            velocity=high_pure_t2v_apg_velocity,
            sigma=sigma_value,
            high_pure_t2v_weight=1.0,
            low_source_v2v_weight=0.0,
            endpoint="high_pure_t2v_apg",
        )

    target_weight = weight.to(
        device=target_packed_state.device,
        dtype=torch.float32,
    )
    transition_fp32 = torch.lerp(
        low_source_v2v_apg_velocity.float(),
        high_pure_t2v_apg_velocity.float(),
        target_weight,
    )
    if not bool(torch.isfinite(transition_fp32).all().item()):
        raise T2VV2VBranchHomotopyError(
            "FP32 T2V/V2V branch homotopy is non-finite"
        )
    velocity = transition_fp32.to(dtype=low_source_v2v_apg_velocity.dtype)
    low_weight = float(
        (torch.ones_like(weight, dtype=torch.float32) - weight).item()
    )
    return T2VV2VBranchHomotopyStep(
        velocity=velocity,
        sigma=sigma_value,
        high_pure_t2v_weight=high_weight,
        low_source_v2v_weight=low_weight,
        endpoint="transition",
    )


def combine_t2v_v2v_apg_velocities(
    target_packed_state: Any,
    high_pure_t2v_apg_velocity: Any,
    low_source_v2v_apg_velocity: Any,
    sigma: Any,
    *,
    high_pure_t2v_momentum: float = 0.0,
    low_source_v2v_momentum: float = 0.0,
) -> Any:
    """Tensor-only convenience wrapper for the pinned homotopy step."""

    return t2v_v2v_branch_homotopy_step(
        target_packed_state,
        high_pure_t2v_apg_velocity,
        low_source_v2v_apg_velocity,
        sigma,
        high_pure_t2v_momentum=high_pure_t2v_momentum,
        low_source_v2v_momentum=low_source_v2v_momentum,
    ).velocity


__all__ = [
    "SIGMA_HIGH",
    "SIGMA_LOW",
    "STEP_TRACE_SCHEMA",
    "T2VV2VBranchHomotopyError",
    "T2VV2VBranchHomotopyStep",
    "combine_t2v_v2v_apg_velocities",
    "smoothstep_pure_t2v_weight",
    "t2v_v2v_branch_homotopy_step",
]
