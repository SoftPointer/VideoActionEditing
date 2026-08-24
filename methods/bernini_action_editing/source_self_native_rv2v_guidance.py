#!/usr/bin/env python3
"""Train-time reproduction of Bernini's pinned four-forward RV2V guidance."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import torch

import source_self_native_ref_contrastive_v3 as native
import source_self_native_target_adapter as target_adapter


SCHEMA_VERSION = "bernini-native-rv2v-guidance-training-v2"
OMEGA_VIDEO = 1.25
OMEGA_IMAGE = 4.5
OMEGA_TEXT = 4.0


class NativeRV2VGuidanceError(RuntimeError):
    """Raised when train-time guidance departs from native Bernini RV2V."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise NativeRV2VGuidanceError(f"guidance receipt is invalid: {error}") from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def guidance_receipt() -> Mapping[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "vendor_formula": (
            "eps_none_u + 1.25*(eps_V_u-eps_none_u) + "
            "4.5*(eps_VI_u-eps_V_u) + 4.0*(eps_VI_c-eps_VI_u)"
        ),
        "forward_order": ["none_uncond", "V_uncond", "VI_uncond", "VI_cond"],
        "omega_video_hex": float(OMEGA_VIDEO).hex(),
        "omega_image_hex": float(OMEGA_IMAGE).hex(),
        "omega_text_hex": float(OMEGA_TEXT).hex(),
        "image_only_axis_built_but_not_forwarded_by_rv2v": True,
        "native_pack_schema": native.SCHEMA_VERSION,
        "native_rv2v4_reference_contract_digest": (
            native.native_rv2v4_reference_contract()["digest"]
        ),
        "target_adapter_schema": target_adapter.SCHEMA_VERSION,
    }
    return {**value, "digest": _object_sha256(value)}


@dataclass(frozen=True)
class NativeRV2VPrediction:
    guided: torch.Tensor
    components: Mapping[str, torch.Tensor]
    receipt: Mapping[str, Any]


def _forward(
    diffusion: Any,
    branch: native.NativeRV2VBranch,
    *,
    timestep: torch.Tensor,
    text: torch.Tensor,
    adapter: target_adapter.NativeTargetAdapterHandle,
    sequence_parallel_rank: int,
    sequence_parallel_size: int,
) -> torch.Tensor:
    route = target_adapter.NativeTargetRoute(
        total_tokens=branch.total_tokens,
        condition_tokens=branch.condition_tokens,
        sequence_parallel_rank=sequence_parallel_rank,
        sequence_parallel_size=sequence_parallel_size,
        branch_name=branch.name,
    )
    # The context encloses the real forward.  Gradient checkpointing is
    # forbidden below because recomputation would happen after this
    # branch-specific context has exited.
    with adapter.route(route):
        return native.forward_native_target_branch(
            diffusion, branch, timestep=timestep, cond_embeds=text
        )


def forward_native_rv2v_guidance(
    diffusion: Any,
    pack: native.NativeRV2VPack,
    *,
    timestep: torch.Tensor,
    cond_embeds: torch.Tensor,
    uncond_embeds: torch.Tensor,
    adapter: target_adapter.NativeTargetAdapterHandle,
    sequence_parallel_rank: int,
    sequence_parallel_size: int = 4,
) -> NativeRV2VPrediction:
    """Run the exact four native RV2V fields and combine them differentiably."""

    if not isinstance(pack, native.NativeRV2VPack):
        raise NativeRV2VGuidanceError("pack must be native NativeRV2VPack")
    if not isinstance(adapter, target_adapter.NativeTargetAdapterHandle):
        raise NativeRV2VGuidanceError("native target adapter handle is required")
    transformer = adapter.transformer
    if bool(getattr(transformer, "gradient_checkpointing", False)) or bool(
        getattr(transformer, "is_gradient_checkpointing", False)
    ):
        raise NativeRV2VGuidanceError(
            "gradient checkpointing must be disabled for branch-local target routes"
        )
    if (
        not isinstance(cond_embeds, torch.Tensor)
        or not isinstance(uncond_embeds, torch.Tensor)
        or cond_embeds.shape != uncond_embeds.shape
        or cond_embeds.device != uncond_embeds.device
        or cond_embeds.requires_grad
        or uncond_embeds.requires_grad
    ):
        raise NativeRV2VGuidanceError(
            "conditional/unconditional text must be frozen and shape matched"
        )
    components = {
        "none_uncond": _forward(
            diffusion,
            pack.none,
            timestep=timestep,
            text=uncond_embeds,
            adapter=adapter,
            sequence_parallel_rank=sequence_parallel_rank,
            sequence_parallel_size=sequence_parallel_size,
        ),
        "V_uncond": _forward(
            diffusion,
            pack.video,
            timestep=timestep,
            text=uncond_embeds,
            adapter=adapter,
            sequence_parallel_rank=sequence_parallel_rank,
            sequence_parallel_size=sequence_parallel_size,
        ),
        "VI_uncond": _forward(
            diffusion,
            pack.video_image,
            timestep=timestep,
            text=uncond_embeds,
            adapter=adapter,
            sequence_parallel_rank=sequence_parallel_rank,
            sequence_parallel_size=sequence_parallel_size,
        ),
        "VI_cond": _forward(
            diffusion,
            pack.video_image,
            timestep=timestep,
            text=cond_embeds,
            adapter=adapter,
            sequence_parallel_rank=sequence_parallel_rank,
            sequence_parallel_size=sequence_parallel_size,
        ),
    }
    shapes = {tuple(value.shape) for value in components.values()}
    if len(shapes) != 1 or any(
        not value.is_floating_point() or not bool(torch.isfinite(value).all().item())
        for value in components.values()
    ):
        raise NativeRV2VGuidanceError("native RV2V component geometry/finite gate failed")
    none = components["none_uncond"]
    video = components["V_uncond"]
    video_image_u = components["VI_uncond"]
    video_image_c = components["VI_cond"]
    guided = (
        none
        + OMEGA_VIDEO * (video - none)
        + OMEGA_IMAGE * (video_image_u - video)
        + OMEGA_TEXT * (video_image_c - video_image_u)
    )
    if not bool(torch.isfinite(guided).all().item()):
        raise NativeRV2VGuidanceError("guided prediction is non-finite")
    if not any(value.requires_grad for value in components.values()):
        raise NativeRV2VGuidanceError("all native RV2V fields are graph-detached")
    return NativeRV2VPrediction(guided, components, guidance_receipt())


__all__ = [
    "NativeRV2VGuidanceError",
    "NativeRV2VPrediction",
    "forward_native_rv2v_guidance",
    "guidance_receipt",
]
