"""Geometry-safe proposal carrier for few-shot Bernini motion codes.

The original CPMR oracle was frozen to the canonical dog's ``31 x 30`` patch
grid.  The K=2 micro-program uses the equally sized but differently oriented
``30 x 31`` grid.  This module accepts only those two audited geometries and
never transposes one into the other: adaptive pooling observes the real y/x
layout and the fixed 3-D coordinate code is added afterwards.

Action/no-op proposals are produced internally by the frozen Bernini model
from the source video and instruction.  Paired targets and support videos are
not inputs to this module and are not available at inference.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

import counterfactual_proposal_motion_rebinding as cpmr


METHOD_NAME = "fewshot-counterfactual-proposal-motion-carrier"
SCHEMA_VERSION = "bernini-epmc-proposal-carrier-v1"

LATENT_PHASES = 21
LATENT_CHANNELS = 16
HIDDEN_SIZE = 1536
POOL_HEIGHT = 8
POOL_WIDTH = 8
CARRIER_TOKENS = LATENT_PHASES * POOL_HEIGHT * POOL_WIDTH
PATCH_GRIDS = ((30, 31), (31, 30))
PATCH_TOKENS_PER_PHASE = 930
EPSILON = 1.0e-6
TOKEN_RMS_CAP = 4.0
COORDINATE_SCALE = 0.02


class FewShotCarrierContractError(ValueError):
    """Raised when proposal/carrier geometry or values violate the contract."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _positive_zero(name: str, value: torch.Tensor) -> None:
    if int(torch.count_nonzero(value).item()) != 0:
        raise FewShotCarrierContractError(f"{name} must be exact zero")
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8)
    if int(torch.count_nonzero(raw).item()) != 0:
        raise FewShotCarrierContractError(
            f"{name} must be byte-exact positive zero"
        )


def _finite_fp32(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise FewShotCarrierContractError(f"{name} must be a torch.Tensor")
    if value.device.type == "meta":
        raise FewShotCarrierContractError(f"{name} cannot be a meta tensor")
    if value.dtype != torch.float32:
        raise FewShotCarrierContractError(f"{name} must be float32")
    if not bool(torch.isfinite(value).all().item()):
        raise FewShotCarrierContractError(f"{name} contains NaN or infinity")
    return value


def validate_patch_grid(value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise FewShotCarrierContractError("patch grid must be two exact integers")
    result = (int(value[0]), int(value[1]))
    if result not in PATCH_GRIDS:
        raise FewShotCarrierContractError(
            f"patch grid must be one of {PATCH_GRIDS}, got {result}"
        )
    if result[0] * result[1] != PATCH_TOKENS_PER_PHASE:
        raise FewShotCarrierContractError("patch grid token count differs from 930")
    return result


@dataclass(frozen=True)
class PooledMotionContent:
    clipped_content_fp32: torch.Tensor
    activity: torch.Tensor
    pooled_increment_phase_rms: torch.Tensor
    token_rms_before_clip: torch.Tensor
    clip_fraction: torch.Tensor


def normalize_pooled_motion_content(
    pooled_increments: torch.Tensor,
) -> PooledMotionContent:
    """Normalize detached ``[1,21,8,8,1536]`` temporal increments."""

    value = _finite_fp32("pooled_increments", pooled_increments)
    expected = (1, LATENT_PHASES, POOL_HEIGHT, POOL_WIDTH, HIDDEN_SIZE)
    if tuple(int(item) for item in value.shape) != expected:
        raise FewShotCarrierContractError(
            f"pooled increments must have exact shape {expected}"
        )
    _positive_zero("pooled phase 0", value[:, 0])
    with torch.no_grad():
        phase_rms = value.square().mean(dim=(2, 3, 4)).sqrt()
        activity = value.abs().amax(dim=(2, 3, 4)) > 0
        activity[:, 0] = False
        normalized = value / phase_rms.clamp_min(EPSILON)[:, :, None, None, None]
        normalized = torch.where(
            activity[:, :, None, None, None],
            normalized,
            torch.zeros_like(normalized),
        )
        token_rms = normalized.square().mean(dim=-1).sqrt()
        scale = (token_rms / TOKEN_RMS_CAP).clamp_min(1.0)
        clipped = normalized / scale[..., None]
        clipped = torch.where(
            activity[:, :, None, None, None],
            clipped,
            torch.zeros_like(clipped),
        ).contiguous()
        clip_fraction = (scale > 1.0).float().mean(dim=(2, 3))
        clip_fraction = torch.where(
            activity, clip_fraction, torch.zeros_like(clip_fraction)
        )
    _finite_fp32("clipped_content", clipped)
    _positive_zero("clipped phase 0", clipped[:, 0])
    if bool(
        (clipped.square().mean(dim=-1).sqrt() > TOKEN_RMS_CAP + 1.0e-5)
        .any()
        .item()
    ):
        raise FewShotCarrierContractError("content token RMS cap was violated")
    return PooledMotionContent(
        clipped_content_fp32=clipped,
        activity=activity,
        pooled_increment_phase_rms=phase_rms,
        token_rms_before_clip=token_rms,
        clip_fraction=clip_fraction,
    )


@dataclass(frozen=True)
class FewShotProposalCarrier:
    carrier_fp32: torch.Tensor
    content: PooledMotionContent
    patch_grid: tuple[int, int]
    action_patch_sha256: str
    noop_patch_sha256: str

    @property
    def activity(self) -> torch.Tensor:
        return self.content.activity

    def flattened(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        value = self.carrier_fp32.reshape(1, CARRIER_TOKENS, HIDDEN_SIZE)
        if dtype is not None:
            value = value.to(dtype=dtype)
        _positive_zero("flattened carrier phase 0", value[:, :64])
        return value

    def audit_receipt(self) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD_NAME,
            "patch_grid_yx": list(self.patch_grid),
            "patch_grid_transposed": False,
            "latent_phases": LATENT_PHASES,
            "pool_grid_yx": [POOL_HEIGHT, POOL_WIDTH],
            "carrier_shape": list(self.carrier_fp32.shape),
            "carrier_tokens": CARRIER_TOKENS,
            "temporal_operator": "D[0]=0;D[t]=(A-N)[t]-(A-N)[t-1]",
            "pooling": "adaptive_avg_pool2d_on_true_yx_layout",
            "normalization": "per_phase_rms",
            "token_rms_cap": TOKEN_RMS_CAP,
            "coordinate_scale": COORDINATE_SCALE,
            "coordinate_sha256": cpmr.coordinate_tensor_sha256(),
            "phase0_exact_positive_zero": True,
            "activity_bitset": "".join(
                "1" if bool(item) else "0"
                for item in self.activity.detach().cpu()[0].tolist()
            ),
            "action_patch_sha256": self.action_patch_sha256,
            "noop_patch_sha256": self.noop_patch_sha256,
            "carrier_fp32_sha256": cpmr.tensor_sha256(self.carrier_fp32),
            "carrier_bfloat16_sha256": cpmr.tensor_sha256(
                self.flattened(dtype=torch.bfloat16)
            ),
            "target_or_support_input": False,
        }
        return {**payload, "receipt_sha256": _sha256_json(payload)}


def build_fewshot_proposal_carrier(
    action_patch_field: torch.Tensor,
    noop_patch_field: torch.Tensor,
    *,
    expected_patch_grid: tuple[int, int],
) -> FewShotProposalCarrier:
    """Build a frozen motion carrier without changing the proposal orientation."""

    patch_h, patch_w = validate_patch_grid(expected_patch_grid)
    expected = (1, LATENT_PHASES, patch_h, patch_w, HIDDEN_SIZE)
    for name, value in (
        ("action_patch_field", action_patch_field),
        ("noop_patch_field", noop_patch_field),
    ):
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            raise FewShotCarrierContractError(f"{name} must be floating tensor")
        if value.device.type == "meta":
            raise FewShotCarrierContractError(f"{name} cannot be meta")
        if tuple(int(item) for item in value.shape) != expected:
            raise FewShotCarrierContractError(
                f"{name} must have exact shape {expected}"
            )
        if not bool(torch.isfinite(value).all().item()):
            raise FewShotCarrierContractError(f"{name} contains non-finite values")
    if action_patch_field.device != noop_patch_field.device:
        raise FewShotCarrierContractError("proposal fields must share one device")
    if action_patch_field.dtype != noop_patch_field.dtype:
        raise FewShotCarrierContractError("proposal fields must share one dtype")

    action_sha = cpmr.tensor_sha256(action_patch_field)
    noop_sha = cpmr.tensor_sha256(noop_patch_field)
    with torch.no_grad():
        increments = action_patch_field.detach().float().clone()
        increments.sub_(noop_patch_field.detach().float())
        for phase in range(LATENT_PHASES - 1, 0, -1):
            increments[:, phase].sub_(increments[:, phase - 1])
        increments[:, 0].zero_()
        pool_input = increments.permute(0, 1, 4, 2, 3).reshape(
            LATENT_PHASES, HIDDEN_SIZE, patch_h, patch_w
        )
        pooled_cf = F.adaptive_avg_pool2d(pool_input, (POOL_HEIGHT, POOL_WIDTH))
        pooled = (
            pooled_cf.reshape(
                1, LATENT_PHASES, HIDDEN_SIZE, POOL_HEIGHT, POOL_WIDTH
            )
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
    normalized = normalize_pooled_motion_content(pooled)
    coordinate = cpmr.fixed_3d_coordinate_encoding(
        device=normalized.clipped_content_fp32.device
    )
    with torch.no_grad():
        carrier = torch.where(
            normalized.activity[:, :, None, None, None],
            normalized.clipped_content_fp32 + COORDINATE_SCALE * coordinate,
            torch.zeros_like(normalized.clipped_content_fp32),
        ).contiguous()
    _finite_fp32("carrier", carrier)
    _positive_zero("carrier phase 0", carrier[:, 0])
    return FewShotProposalCarrier(
        carrier_fp32=carrier,
        content=normalized,
        patch_grid=(patch_h, patch_w),
        action_patch_sha256=action_sha,
        noop_patch_sha256=noop_sha,
    )


def build_carrier_from_proposal_latents(
    transformer: Any,
    action_latent: torch.Tensor,
    noop_latent: torch.Tensor,
    *,
    expected_patch_grid: tuple[int, int],
) -> FewShotProposalCarrier:
    """Patch-embed two frozen proposals and retain their true y/x geometry."""

    patch_h, patch_w = validate_patch_grid(expected_patch_grid)
    expected_latent = (
        1,
        LATENT_CHANNELS,
        LATENT_PHASES,
        patch_h * 2,
        patch_w * 2,
    )
    if not isinstance(action_latent, torch.Tensor) or not isinstance(
        noop_latent, torch.Tensor
    ):
        raise FewShotCarrierContractError("proposal latents must be tensors")
    if tuple(action_latent.shape) != expected_latent or tuple(noop_latent.shape) != expected_latent:
        raise FewShotCarrierContractError(
            f"proposal latents must both have exact shape {expected_latent}"
        )
    if action_latent.device != noop_latent.device:
        raise FewShotCarrierContractError("proposal latents must share one device")
    patch_embedding = getattr(transformer, "patch_embedding", None)
    weight = getattr(patch_embedding, "weight", None)
    bias = getattr(patch_embedding, "bias", None)
    if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
        raise FewShotCarrierContractError(
            "patch-embedding weight dtype/device is unavailable"
        )
    if bias is not None and (
        not isinstance(bias, torch.Tensor)
        or bias.dtype != weight.dtype
        or bias.device != weight.device
    ):
        raise FewShotCarrierContractError(
            "patch-embedding weight and bias dtype/device differ"
        )
    if action_latent.device != weight.device:
        raise FewShotCarrierContractError(
            "proposal latents and patch embedding must share one device"
        )
    # Bernini contains mixed-precision submodules: the first transformer
    # parameter is not a reliable proxy for the Conv3d input dtype.  Binding
    # directly to patch_embedding.weight avoids FP32-input/BF16-bias failures.
    dtype = weight.dtype
    with torch.no_grad():
        action_emb = patch_embedding(action_latent.to(dtype=dtype))
        noop_emb = patch_embedding(noop_latent.to(dtype=dtype))
    expected_embedding = (1, HIDDEN_SIZE, LATENT_PHASES, patch_h, patch_w)
    if tuple(action_emb.shape) != expected_embedding or tuple(noop_emb.shape) != expected_embedding:
        raise FewShotCarrierContractError(
            f"patch embedding must preserve true grid {expected_embedding}"
        )
    return build_fewshot_proposal_carrier(
        action_emb.permute(0, 2, 3, 4, 1).contiguous(),
        noop_emb.permute(0, 2, 3, 4, 1).contiguous(),
        expected_patch_grid=(patch_h, patch_w),
    )


__all__ = [
    "CARRIER_TOKENS",
    "FewShotCarrierContractError",
    "FewShotProposalCarrier",
    "HIDDEN_SIZE",
    "LATENT_PHASES",
    "METHOD_NAME",
    "PATCH_GRIDS",
    "PooledMotionContent",
    "SCHEMA_VERSION",
    "build_carrier_from_proposal_latents",
    "build_fewshot_proposal_carrier",
    "normalize_pooled_motion_content",
    "validate_patch_grid",
]
