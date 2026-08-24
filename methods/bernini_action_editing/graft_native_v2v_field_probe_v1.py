#!/usr/bin/env python3
"""Training-mode bridge for Bernini's authenticated native V-only field.

Two existing read-only runtimes already prove the native sampler geometry:

* :mod:`pair_v7_vonly_exact81_route_runtime` proves that the negative and
  action forwards receive the *same object* for the source-prefix +
  target-suffix visual pack, rotary tensor, and timestep.
* :mod:`t2v_v2v_branch_homotopy_runtime_v1` additionally proves the native
  ``patch_vae_latent`` order ``source_id=1 -> source_id=0`` and that APG uses
  the target suffix of both raw transformer predictions.

Those observers intentionally reject tensors connected to autograd and wrap
``GEN_Wanx22.sample``, which is itself decorated with ``torch.no_grad``.  They
therefore authenticate inference wiring but cannot be reused as a training
field.  This module adds only the missing narrow bridge:

1. validate the already-authenticated two-forward V pack while preserving
   connected raw predictions and return their target-suffix views; and
2. prove that the pinned vendor ``normalized_guidance`` accepts FP32 leaves
   and has the same finite forward/VJP as an independent algebraic spelling.

It does not sample, train, create an optimizer, or claim output quality.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from typing import Any, Callable, Mapping, Optional

import pair_v7_vonly_exact81_route_runtime as vonly
import t2v_v2v_branch_homotopy_runtime_v1 as homotopy


SCHEMA_VERSION = "bernini-graft-native-v2v-field-probe-v1"
PINNED_BERNINI_COMMIT = homotopy.PINNED_BERNINI_COMMIT
PINNED_WAN_DIFFUSION_SHA256 = homotopy.PINNED_WAN_DIFFUSION_SHA256
EXPECTED_PATCH_SOURCE_IDS = homotopy.EXPECTED_PATCH_SOURCE_IDS
EXPECTED_FORWARD_ORDER = vonly.FORWARD_ORDER
EXPECTED_GUIDANCE_MODE = vonly.EXPECTED_GUIDANCE_MODE
EXPECTED_HIDDEN_DIM = vonly.EXPECTED_HIDDEN_DIM
EXPECTED_OUTPUT_CHANNELS = 64
APG_REDUCTION_DIMS = (-1, -2, -4)


class GraftNativeV2VFieldProbeError(RuntimeError):
    """Raised when inference wiring cannot be promoted to a live field."""


def native_field_wiring_receipt() -> dict[str, Any]:
    """Describe exactly which existing observers authenticate this bridge."""

    if EXPECTED_PATCH_SOURCE_IDS != (1.0, 0.0):
        raise GraftNativeV2VFieldProbeError("native patch source-id order differs")
    if EXPECTED_FORWARD_ORDER != ("negative", "action"):
        raise GraftNativeV2VFieldProbeError("native two-forward order differs")
    if EXPECTED_GUIDANCE_MODE != "v2v_apg":
        raise GraftNativeV2VFieldProbeError("native guidance mode differs")
    return {
        "schema_version": SCHEMA_VERSION,
        "pinned_bernini_commit": PINNED_BERNINI_COMMIT,
        "pinned_wan_diffusion_sha256": PINNED_WAN_DIFFUSION_SHA256,
        "guidance_mode": EXPECTED_GUIDANCE_MODE,
        "patch_source_ids": list(EXPECTED_PATCH_SOURCE_IDS),
        "forward_order": list(EXPECTED_FORWARD_ORDER),
        "negative_positive_same_visual_pack": True,
        "visual_pack_layout": "source_id_1_prefix_then_source_id_0_target_suffix",
        "raw_output_target_selection": "last_target_tokens",
        "authenticated_by": {
            "same_pack": (
                "pair_v7_vonly_exact81_route_runtime."
                "PairV7VOnlyExact81RoutePatch._validate_shared_call"
            ),
            "patch_order_and_target_tail": (
                "t2v_v2v_branch_homotopy_runtime_v1."
                "T2VV2VBranchHomotopyRuntimePatch"
            ),
            "vendor_apg": "bernini.models.wan_diffusion.normalized_guidance",
        },
        "existing_observers_training_usable": False,
        "existing_observer_blocker": (
            "both observers require detached tensors; GEN_Wanx22.sample is "
            "torch.no_grad"
        ),
        "optimizer_created": False,
        "parameters_updated": False,
        "quality_claim_authorized": False,
        "training_claim_authorized": False,
    }


def _torch() -> Any:
    try:
        import torch
    except Exception as error:  # pragma: no cover - import failure is host-specific
        raise GraftNativeV2VFieldProbeError("PyTorch is required") from error
    return torch


def _finite_tensor(value: Any, *, label: str) -> None:
    torch = _torch()
    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all().item()):
        raise GraftNativeV2VFieldProbeError(f"{label} must be a finite torch tensor")


@dataclass(frozen=True)
class ConnectedTargetTailPair:
    """Graph-preserving target suffixes from one native negative/action pair."""

    negative: Any
    action: Any
    source_tokens: int
    target_tokens: int
    total_tokens: int


def connected_target_tail_pair(
    *,
    negative_visual_pack: Any,
    action_visual_pack: Any,
    negative_rotary: Any,
    action_rotary: Any,
    negative_timestep: Any,
    action_timestep: Any,
    negative_raw_output: Any,
    action_raw_output: Any,
    source_tokens: int,
    target_tokens: int,
) -> ConnectedTargetTailPair:
    """Validate native two-forward identity and preserve target-tail graphs.

    The upstream inference observer must already have authenticated
    ``source_id=(1, 0)``.  Object identity is required here because equality
    alone would not prove that negative/action forwards saw one shared native
    state.  In contrast to the old observers, raw outputs must remain
    autograd-connected.
    """

    torch = _torch()
    if (
        type(source_tokens) is not int
        or type(target_tokens) is not int
        or source_tokens <= 0
        or target_tokens <= 0
        or source_tokens != target_tokens
    ):
        raise GraftNativeV2VFieldProbeError(
            "native V-only pack requires equal positive source/target token counts"
        )
    total_tokens = source_tokens + target_tokens
    for label, left, right in (
        ("visual pack", negative_visual_pack, action_visual_pack),
        ("rotary", negative_rotary, action_rotary),
        ("timestep", negative_timestep, action_timestep),
    ):
        if left is not right:
            raise GraftNativeV2VFieldProbeError(
                f"negative/action {label} must be the same object"
            )
        _finite_tensor(left, label=f"shared {label}")
    if tuple(negative_visual_pack.shape) != (
        1,
        total_tokens,
        EXPECTED_HIDDEN_DIM,
    ):
        raise GraftNativeV2VFieldProbeError(
            "visual pack is not source-prefix + target-suffix Bernini 1.3B geometry"
        )
    if tuple(negative_timestep.shape) != (1,):
        raise GraftNativeV2VFieldProbeError("native timestep must be one expanded scalar")
    rotary_shape = tuple(negative_rotary.shape)
    if len(rotary_shape) != 4 or rotary_shape[0] != 1 or rotary_shape[2] != total_tokens:
        raise GraftNativeV2VFieldProbeError("native rotary pack geometry differs")

    tails = []
    for label, value in (
        ("negative", negative_raw_output),
        ("action", action_raw_output),
    ):
        _finite_tensor(value, label=f"{label} raw output")
        if tuple(value.shape) != (1, total_tokens, EXPECTED_OUTPUT_CHANNELS):
            raise GraftNativeV2VFieldProbeError(
                f"{label} raw output does not preserve the full native V pack"
            )
        if value.requires_grad is not True or value.grad_fn is None:
            raise GraftNativeV2VFieldProbeError(
                f"{label} raw output is detached; old inference observer is not a trainer"
            )
        tail = value[:, -target_tokens:, :]
        if (
            tuple(tail.shape) != (1, target_tokens, EXPECTED_OUTPUT_CHANNELS)
            or tail.requires_grad is not True
            or tail.grad_fn is None
        ):
            raise GraftNativeV2VFieldProbeError(
                f"{label} target suffix lost its live graph"
            )
        tails.append(tail)
    return ConnectedTargetTailPair(
        negative=tails[0],
        action=tails[1],
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        total_tokens=total_tokens,
    )


def differentiable_normalized_guidance(
    pred_cond: Any,
    pred_uncond: Any,
    guidance_scale: float,
    *,
    eta: float,
    norm_threshold: float,
) -> Any:
    """Independent differentiable spelling of pinned Bernini single APG."""

    torch = _torch()
    import torch.nn.functional as functional

    diff = pred_cond - pred_uncond
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=list(APG_REDUCTION_DIMS), keepdim=True)
        diff = diff * torch.minimum(ones, norm_threshold / diff_norm)
    projected, base = diff.double(), pred_cond.double()
    base = functional.normalize(base, dim=list(APG_REDUCTION_DIMS))
    parallel = (projected * base).sum(
        dim=list(APG_REDUCTION_DIMS), keepdim=True
    ) * base
    orthogonal = projected - parallel
    normalized = orthogonal.to(diff.dtype) + eta * parallel.to(diff.dtype)
    return pred_uncond + guidance_scale * normalized


def _finite_scalar(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GraftNativeV2VFieldProbeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise GraftNativeV2VFieldProbeError(f"{label} must be finite")
    return result


def _validate_fp32_leaf(value: Any, *, label: str) -> None:
    torch = _torch()
    _finite_tensor(value, label=label)
    if (
        value.dtype != torch.float32
        or value.ndim != 5
        or value.requires_grad is not True
        or value.is_leaf is not True
        or value.grad_fn is not None
    ):
        raise GraftNativeV2VFieldProbeError(
            f"{label} must be a connected-ready FP32 five-dimensional leaf"
        )


def _maximum_absolute(left: Any, right: Any) -> float:
    return float((left.detach().double() - right.detach().double()).abs().max().item())


def normalized_guidance_vjp_parity(
    *,
    vendor_normalized_guidance: Callable[..., Any],
    momentum_buffer_factory: Callable[[float], Any],
    pred_cond: Any,
    pred_uncond: Any,
    cotangent: Any,
    guidance_scale: float = 4.0,
    eta: float = 0.5,
    norm_threshold: float = 50.0,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> Mapping[str, Any]:
    """Check pinned vendor APG forward and VJP against independent algebra."""

    torch = _torch()
    if not callable(vendor_normalized_guidance) or not callable(momentum_buffer_factory):
        raise GraftNativeV2VFieldProbeError("vendor APG symbols must be callable")
    try:
        parameters = tuple(inspect.signature(vendor_normalized_guidance).parameters)
    except (TypeError, ValueError) as error:
        raise GraftNativeV2VFieldProbeError("vendor APG signature is unavailable") from error
    if parameters != (
        "pred_cond",
        "pred_uncond",
        "guidance_scale",
        "momentum_buffer",
        "eta",
        "norm_threshold",
    ):
        raise GraftNativeV2VFieldProbeError("vendor normalized_guidance signature differs")
    _validate_fp32_leaf(pred_cond, label="conditional clean prediction")
    _validate_fp32_leaf(pred_uncond, label="negative clean prediction")
    _finite_tensor(cotangent, label="VJP cotangent")
    if (
        tuple(pred_cond.shape) != tuple(pred_uncond.shape)
        or tuple(cotangent.shape) != tuple(pred_cond.shape)
        or cotangent.dtype != torch.float32
        or cotangent.requires_grad
    ):
        raise GraftNativeV2VFieldProbeError("APG leaves/cotangent geometry differs")
    scale = _finite_scalar(guidance_scale, label="guidance_scale")
    eta_value = _finite_scalar(eta, label="eta")
    threshold = _finite_scalar(norm_threshold, label="norm_threshold")
    if threshold < 0 or atol < 0 or rtol < 0:
        raise GraftNativeV2VFieldProbeError("APG thresholds/tolerances must be non-negative")
    momentum = momentum_buffer_factory(0.0)
    if float(getattr(momentum, "momentum", float("nan"))) != 0.0:
        raise GraftNativeV2VFieldProbeError("vendor APG requires a fresh momentum-zero buffer")

    try:
        with torch.enable_grad():
            vendor = vendor_normalized_guidance(
                pred_cond=pred_cond,
                pred_uncond=pred_uncond,
                guidance_scale=scale,
                momentum_buffer=momentum,
                eta=eta_value,
                norm_threshold=threshold,
            )
            independent = differentiable_normalized_guidance(
                pred_cond,
                pred_uncond,
                scale,
                eta=eta_value,
                norm_threshold=threshold,
            )
            if vendor.requires_grad is not True or vendor.grad_fn is None:
                raise GraftNativeV2VFieldProbeError(
                    "vendor normalized_guidance detached FP32 leaves"
                )
            torch.testing.assert_close(vendor, independent, atol=atol, rtol=rtol)
            vendor_vjp = torch.autograd.grad(
                vendor,
                (pred_cond, pred_uncond),
                grad_outputs=cotangent,
                retain_graph=True,
                create_graph=False,
            )
            independent_vjp = torch.autograd.grad(
                independent,
                (pred_cond, pred_uncond),
                grad_outputs=cotangent,
                retain_graph=False,
                create_graph=False,
            )
            for observed, expected in zip(vendor_vjp, independent_vjp):
                torch.testing.assert_close(observed, expected, atol=atol, rtol=rtol)
    except GraftNativeV2VFieldProbeError:
        raise
    except Exception as error:
        raise GraftNativeV2VFieldProbeError(
            "vendor normalized_guidance forward/VJP parity failed"
        ) from error
    for label, gradient in zip(("conditional", "negative"), vendor_vjp):
        if not bool(torch.isfinite(gradient).all().item()) or not bool(torch.count_nonzero(gradient).item()):
            raise GraftNativeV2VFieldProbeError(f"vendor {label} VJP is not finite/nonzero")

    return {
        "schema_version": SCHEMA_VERSION,
        "vendor_function": "bernini.models.wan_diffusion.normalized_guidance",
        "input_dtype": "torch.float32",
        "input_shape": [int(item) for item in pred_cond.shape],
        "leaves": True,
        "vendor_output_connected": True,
        "vendor_forward_finite": bool(torch.isfinite(vendor).all().item()),
        "vendor_vjp_finite": True,
        "vendor_forward_independent_parity": True,
        "vendor_vjp_independent_parity": True,
        "forward_max_abs_error": _maximum_absolute(vendor, independent),
        "conditional_vjp_max_abs_error": _maximum_absolute(
            vendor_vjp[0], independent_vjp[0]
        ),
        "negative_vjp_max_abs_error": _maximum_absolute(
            vendor_vjp[1], independent_vjp[1]
        ),
        "guidance_scale": scale,
        "eta": eta_value,
        "norm_threshold": threshold,
        "momentum": 0.0,
        "optimizer_created": False,
        "parameters_updated": False,
        "training_claim_authorized": False,
        "quality_claim_authorized": False,
    }


def pinned_vendor_normalized_guidance_vjp_parity(
    *,
    pred_cond: Any,
    pred_uncond: Any,
    cotangent: Any,
    guidance_scale: float = 4.0,
    eta: float = 0.5,
    norm_threshold: float = 50.0,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> Mapping[str, Any]:
    """Resolve hash-bound Bernini APG symbols through the existing observer."""

    vendor, momentum_factory = homotopy._resolve_vendor_apg_symbols()
    return normalized_guidance_vjp_parity(
        vendor_normalized_guidance=vendor,
        momentum_buffer_factory=momentum_factory,
        pred_cond=pred_cond,
        pred_uncond=pred_uncond,
        cotangent=cotangent,
        guidance_scale=guidance_scale,
        eta=eta,
        norm_threshold=norm_threshold,
        atol=atol,
        rtol=rtol,
    )


__all__ = [
    "ConnectedTargetTailPair",
    "EXPECTED_FORWARD_ORDER",
    "EXPECTED_PATCH_SOURCE_IDS",
    "GraftNativeV2VFieldProbeError",
    "PINNED_BERNINI_COMMIT",
    "PINNED_WAN_DIFFUSION_SHA256",
    "SCHEMA_VERSION",
    "connected_target_tail_pair",
    "differentiable_normalized_guidance",
    "native_field_wiring_receipt",
    "normalized_guidance_vjp_parity",
    "pinned_vendor_normalized_guidance_vjp_parity",
]
