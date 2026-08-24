#!/usr/bin/env python3
"""Pointwise homotopy between two native Bernini APG velocities.

The high-noise branch is the references-only ``R2V-4`` APG proposal.  The
low-noise branch is Bernini's official ``v2v_apg`` proposal conditioned on the
full source video and the same four references.  Both proposals must have
already been evaluated on one target packed state.  This module only combines
their scheduler-bound velocities; it neither reconstructs APG nor advances an
integrator.

For active flow sigma ``s``, the high-branch weight is the pinned FP32
smoothstep

    h(s) = 0                         if s <= 0.60
           u^2 (3 - 2u), u=(s-.60)/.30, if 0.60 < s < 0.90
           1                         if s >= 0.90

and the velocity is ``(1-h) * v_v2v + h * v_r2v4``.  Endpoint branches are
returned directly, before any cast or arithmetic, so ``h=0`` and ``h=1`` are
bit exact.  Transition arithmetic is FP32 and is cast back to the common
branch dtype.  APG momentum is required to be exactly zero: with historyful
momentum the two fields would not define a pointwise homotopy at one state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


SIGMA_LOW = 0.60
SIGMA_HIGH = 0.90
STEP_TRACE_SCHEMA = "bernini-native-branch-homotopy-step-v1"


class NativeBranchHomotopyError(RuntimeError):
    """Raised before integration when the pointwise homotopy contract fails."""


@dataclass(frozen=True)
class NativeBranchHomotopyStep:
    """One combined velocity and its small, JSON-safe audit trace."""

    velocity: Any
    sigma: float
    high_r2v4_weight: float
    low_official_v2v_apg_weight: float
    endpoint: str

    def trace_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STEP_TRACE_SCHEMA,
            "sigma": self.sigma,
            "high_r2v4_weight": self.high_r2v4_weight,
            "low_official_v2v_apg_weight": self.low_official_v2v_apg_weight,
            "endpoint": self.endpoint,
            "endpoint_exact": self.endpoint != "transition",
            "high_branch": "references_only_r2v4_apg",
            "low_branch": "official_full_source_plus_four_refs_v2v_apg",
            "interpolation_dtype": "float32",
            "apg_momentum": 0.0,
        }


def _require_torch() -> Any:
    try:
        import torch
    except Exception as error:  # pragma: no cover - exercised on AUH
        raise NativeBranchHomotopyError(
            "PyTorch is required for native branch homotopy"
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
        raise NativeBranchHomotopyError(
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
        raise NativeBranchHomotopyError(f"{label} must be exactly zero")


def _validate_geometry(
    target_packed_state: Any,
    high_r2v4_apg_velocity: Any,
    low_official_v2v_apg_velocity: Any,
) -> None:
    torch = _require_torch()
    tensors = (
        ("target packed state", target_packed_state),
        ("high R2V-4 APG velocity", high_r2v4_apg_velocity),
        ("low official v2v_apg velocity", low_official_v2v_apg_velocity),
    )
    for label, value in tensors:
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 3
            or value.numel() <= 0
            or not value.dtype.is_floating_point
            or not bool(torch.isfinite(value).all().item())
        ):
            raise NativeBranchHomotopyError(
                f"{label} must be one finite non-empty floating [B,N,C] tensor"
            )

    expected_shape = tuple(int(item) for item in target_packed_state.shape)
    expected_device = target_packed_state.device
    for label, value in tensors[1:]:
        if tuple(int(item) for item in value.shape) != expected_shape:
            raise NativeBranchHomotopyError(
                f"{label} does not share the target packed-state shape"
            )
        if value.device != expected_device:
            raise NativeBranchHomotopyError(
                f"{label} does not share the target packed-state device"
            )
    if high_r2v4_apg_velocity.dtype != low_official_v2v_apg_velocity.dtype:
        raise NativeBranchHomotopyError(
            "high/low APG velocities must share one scheduler-bound dtype"
        )


def smoothstep_high_branch_weight(sigma: Any) -> Any:
    """Return the pinned high-noise R2V-4 weight as a scalar FP32 tensor."""

    torch = _require_torch()
    sigma = _validate_sigma(sigma)
    if bool((sigma <= SIGMA_LOW).item()):
        return torch.zeros_like(sigma, dtype=torch.float32)
    if bool((sigma >= SIGMA_HIGH).item()):
        return torch.ones_like(sigma, dtype=torch.float32)

    # Materialize every constant in FP32 on sigma's device.  This prevents a
    # Python-double smoothstep from silently defining a different schedule.
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
        raise NativeBranchHomotopyError("FP32 smoothstep produced an invalid weight")
    return weight


def native_branch_homotopy_step(
    target_packed_state: Any,
    high_r2v4_apg_velocity: Any,
    low_official_v2v_apg_velocity: Any,
    sigma: Any,
    *,
    high_r2v4_momentum: float = 0.0,
    low_official_v2v_apg_momentum: float = 0.0,
) -> NativeBranchHomotopyStep:
    """Combine two already-guided native velocities at one target state."""

    torch = _require_torch()
    _validate_zero_momentum(
        high_r2v4_momentum,
        label="high R2V-4 APG momentum",
    )
    _validate_zero_momentum(
        low_official_v2v_apg_momentum,
        label="low official v2v_apg momentum",
    )
    _validate_geometry(
        target_packed_state,
        high_r2v4_apg_velocity,
        low_official_v2v_apg_velocity,
    )
    weight = smoothstep_high_branch_weight(sigma)
    high_weight = float(weight.item())
    sigma_value = float(sigma.item())

    # These direct returns are intentional endpoint certificates: NaNs or
    # roundoff in an inactive branch cannot contaminate an active endpoint.
    # (Both branches were nevertheless validated as finite above.)
    if high_weight == 0.0:
        return NativeBranchHomotopyStep(
            velocity=low_official_v2v_apg_velocity,
            sigma=sigma_value,
            high_r2v4_weight=0.0,
            low_official_v2v_apg_weight=1.0,
            endpoint="low_official_v2v_apg",
        )
    if high_weight == 1.0:
        return NativeBranchHomotopyStep(
            velocity=high_r2v4_apg_velocity,
            sigma=sigma_value,
            high_r2v4_weight=1.0,
            low_official_v2v_apg_weight=0.0,
            endpoint="high_r2v4_apg",
        )

    weight_on_target = weight.to(
        device=target_packed_state.device,
        dtype=torch.float32,
    )
    transition_fp32 = torch.lerp(
        low_official_v2v_apg_velocity.float(),
        high_r2v4_apg_velocity.float(),
        weight_on_target,
    )
    if not bool(torch.isfinite(transition_fp32).all().item()):
        raise NativeBranchHomotopyError("FP32 branch homotopy is non-finite")
    velocity = transition_fp32.to(dtype=low_official_v2v_apg_velocity.dtype)
    low_weight = float(
        (torch.ones_like(weight, dtype=torch.float32) - weight).item()
    )
    return NativeBranchHomotopyStep(
        velocity=velocity,
        sigma=sigma_value,
        high_r2v4_weight=high_weight,
        low_official_v2v_apg_weight=low_weight,
        endpoint="transition",
    )


def combine_native_apg_velocities(
    target_packed_state: Any,
    high_r2v4_apg_velocity: Any,
    low_official_v2v_apg_velocity: Any,
    sigma: Any,
    *,
    high_r2v4_momentum: float = 0.0,
    low_official_v2v_apg_momentum: float = 0.0,
) -> Any:
    """Tensor-only convenience wrapper around :func:`native_branch_homotopy_step`."""

    return native_branch_homotopy_step(
        target_packed_state,
        high_r2v4_apg_velocity,
        low_official_v2v_apg_velocity,
        sigma,
        high_r2v4_momentum=high_r2v4_momentum,
        low_official_v2v_apg_momentum=low_official_v2v_apg_momentum,
    ).velocity


__all__ = [
    "NativeBranchHomotopyError",
    "NativeBranchHomotopyStep",
    "SIGMA_HIGH",
    "SIGMA_LOW",
    "STEP_TRACE_SCHEMA",
    "combine_native_apg_velocities",
    "native_branch_homotopy_step",
    "smoothstep_high_branch_weight",
]
