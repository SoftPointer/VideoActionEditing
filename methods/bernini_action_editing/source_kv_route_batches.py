#!/usr/bin/env python3
"""Pinned batch and full-sequence forward primitives for Bernini V9.

Official Bernini supervision packs a clean source prefix followed by an equal
noisy target span.  The legacy training helper returns only target positions.
Source-K/V replay additionally needs a source-only carrier forward, so this
module exposes two narrow operations:

* build a carrier batch by copying the exact source prefix and no-op text;
* execute the pinned Bernini ``shared_step`` and return every sequence token.

The carrier contains no target tail, selector, target velocity, target length,
mask, flow, pose, track, trajectory, or first-frame anchor.  It is attention
memory only and is never decoded as the edited output.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping


EXACT_NOOP_INSTRUCTION = (
    "Keep the source video exactly unchanged, including every subject, "
    "appearance, action, camera motion, background, timing, and composition."
)
EXACT_NOOP_INSTRUCTION_SHA256 = (
    "fb5f23b5b9de175696cff019f035e81eb1ee6a1123db7e3b63afb604b88daf3a"
)
TEXT_FIELDS = ("input_ids", "attention_mask", "t5_input_lens")
PAIR_STATE_FIELDS = (
    "input_vae_latents",
    "input_vae_rope",
    "vae_latents_mask",
    "vae_seqlen",
    "timesteps",
)
CARRIER_MODEL_FIELDS = (
    *TEXT_FIELDS,
    "input_vae_latents",
    "input_vae_rope",
    "vae_seqlen",
    "timesteps",
)
FORBIDDEN_EXTERNAL_FIELDS = frozenset(
    {
        "mask",
        "edit_mask",
        "spatial_mask",
        "segmentation_mask",
        "motion_mask",
        "tracking_mask",
        "track_mask",
        "swept_tube",
        "pose",
        "trajectory",
        "optical_flow",
        "flow",
        "first_frame_anchor",
    }
)


class SourceKVRouteBatchError(RuntimeError):
    """Raised instead of admitting ambiguous carrier or pair geometry."""


def validate_noop_instruction(value: str) -> str:
    if not isinstance(value, str):
        raise SourceKVRouteBatchError("no-op instruction must be a string")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    if value != EXACT_NOOP_INSTRUCTION or digest != EXACT_NOOP_INSTRUCTION_SHA256:
        raise SourceKVRouteBatchError("no-op instruction text or SHA256 differs")
    return digest


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependent
        raise SourceKVRouteBatchError(
            "V9 batch tensor operations require PyTorch"
        ) from error
    return torch


def _reject_forbidden_fields(batch: Mapping[str, Any], *, label: str) -> None:
    forbidden = sorted(FORBIDDEN_EXTERNAL_FIELDS.intersection(batch))
    if forbidden:
        raise SourceKVRouteBatchError(
            f"{label} contains forbidden external conditioning fields: {forbidden}"
        )


def _require_tensor(batch: Mapping[str, Any], name: str, *, label: str) -> Any:
    torch = _require_torch()
    value = batch.get(name)
    if not isinstance(value, torch.Tensor):
        raise SourceKVRouteBatchError(f"{label} lacks tensor field {name}")
    return value


def _pair_boundary(batch: Mapping[str, Any], *, label: str) -> int:
    torch = _require_torch()
    _reject_forbidden_fields(batch, label=label)
    mask = _require_tensor(batch, "vae_latents_mask", label=label)
    if mask.ndim != 2 or int(mask.shape[0]) != 1:
        raise SourceKVRouteBatchError(f"{label} selector must be [1,2N]")
    selector = mask.squeeze(0).bool()
    total = int(selector.numel())
    target = int(selector.sum().item())
    source = total - target
    if source <= 0 or source != target or total != 2 * source:
        raise SourceKVRouteBatchError(
            f"{label} must contain equal non-empty source and target spans"
        )
    if bool(selector[:source].any()) or not bool(selector[source:].all()):
        raise SourceKVRouteBatchError(
            f"{label} selector must be contiguous source-then-target"
        )
    latents = _require_tensor(batch, "input_vae_latents", label=label)
    rope = _require_tensor(batch, "input_vae_rope", label=label)
    if int(latents.shape[0]) != total or int(rope.shape[0]) != total:
        raise SourceKVRouteBatchError(
            f"{label} latent/RoPE length differs from selector"
        )
    vae_seqlen = _require_tensor(batch, "vae_seqlen", label=label)
    positive = vae_seqlen.reshape(-1)
    positive = positive[positive > 0]
    if positive.numel() != 1 or int(positive.item()) != total:
        raise SourceKVRouteBatchError(
            f"{label} must have one positive vae_seqlen equal to 2N"
        )
    timesteps = _require_tensor(batch, "timesteps", label=label)
    if timesteps.numel() != 1 or not bool(torch.isfinite(timesteps.float()).all()):
        raise SourceKVRouteBatchError(
            f"{label} must have one finite timestep"
        )
    for name in TEXT_FIELDS:
        _require_tensor(batch, name, label=label)
    return source


def validate_equal_pair_batches(
    action_batch: Mapping[str, Any],
    noop_batch: Mapping[str, Any],
) -> int:
    """Return ``N`` after proving both prompts share one exact pair state."""

    torch = _require_torch()
    if not isinstance(action_batch, Mapping) or not isinstance(noop_batch, Mapping):
        raise SourceKVRouteBatchError("action and no-op batches must be mappings")
    action_boundary = _pair_boundary(action_batch, label="action pair")
    noop_boundary = _pair_boundary(noop_batch, label="no-op pair")
    if action_boundary != noop_boundary:
        raise SourceKVRouteBatchError("action/no-op pair boundaries differ")
    for name in PAIR_STATE_FIELDS:
        left = _require_tensor(action_batch, name, label="action pair")
        right = _require_tensor(noop_batch, name, label="no-op pair")
        if (
            tuple(left.shape) != tuple(right.shape)
            or left.dtype != right.dtype
            or left.device != right.device
            or not torch.equal(left, right)
        ):
            raise SourceKVRouteBatchError(
                f"action/no-op pair state differs at {name}"
            )
    if all(
        torch.equal(
            _require_tensor(action_batch, name, label="action pair"),
            _require_tensor(noop_batch, name, label="no-op pair"),
        )
        for name in TEXT_FIELDS
    ):
        raise SourceKVRouteBatchError("action and no-op text must be distinct")
    return action_boundary


@dataclass(frozen=True)
class SourceOnlyCarrierBatch:
    batch: Mapping[str, Any]
    source_tokens: int
    noop_instruction_sha256: str


def build_source_only_carrier_batch(
    *,
    action_pair_batch: Mapping[str, Any],
    noop_pair_batch: Mapping[str, Any],
    noop_instruction: str,
) -> SourceOnlyCarrierBatch:
    """Copy one clean source prefix and bind the exact audited no-op text."""

    torch = _require_torch()
    digest = validate_noop_instruction(noop_instruction)
    boundary = validate_equal_pair_batches(action_pair_batch, noop_pair_batch)
    carrier: dict[str, Any] = {}
    for name in TEXT_FIELDS:
        carrier[name] = _require_tensor(
            noop_pair_batch, name, label="no-op pair"
        ).clone()
    carrier["input_vae_latents"] = _require_tensor(
        noop_pair_batch, "input_vae_latents", label="no-op pair"
    )[:boundary].clone()
    carrier["input_vae_rope"] = _require_tensor(
        noop_pair_batch, "input_vae_rope", label="no-op pair"
    )[:boundary].clone()
    carrier["timesteps"] = _require_tensor(
        noop_pair_batch, "timesteps", label="no-op pair"
    ).clone()
    pair_seqlen = _require_tensor(
        noop_pair_batch, "vae_seqlen", label="no-op pair"
    )
    carrier_seqlen = pair_seqlen.clone()
    positive = carrier_seqlen > 0
    if int(positive.sum().item()) != 1:
        raise SourceKVRouteBatchError("pair vae_seqlen lost its single sequence")
    carrier_seqlen[positive] = int(boundary)
    carrier["vae_seqlen"] = carrier_seqlen

    if set(carrier) != set(CARRIER_MODEL_FIELDS):
        raise SourceKVRouteBatchError("carrier fields differ from the minimal contract")
    if any(
        name in carrier
        for name in (
            "vae_latents_mask",
            "target_velocity",
            "target_lens",
            "paired_target_video",
        )
    ):
        raise SourceKVRouteBatchError("carrier unexpectedly contains a target label")
    _reject_forbidden_fields(carrier, label="source-only carrier")
    if (
        int(carrier["input_vae_latents"].shape[0]) != boundary
        or int(carrier["input_vae_rope"].shape[0]) != boundary
        or int(carrier["vae_seqlen"][carrier["vae_seqlen"] > 0].item()) != boundary
        or not torch.equal(
            carrier["input_vae_latents"],
            noop_pair_batch["input_vae_latents"][:boundary],
        )
        or not torch.equal(
            carrier["input_vae_rope"],
            noop_pair_batch["input_vae_rope"][:boundary],
        )
    ):
        raise SourceKVRouteBatchError("carrier is not an exact source prefix")
    return SourceOnlyCarrierBatch(
        batch=carrier,
        source_tokens=boundary,
        noop_instruction_sha256=digest,
    )


def _positive_vae_lengths(batch: Mapping[str, Any]) -> list[int]:
    vae_seqlen = _require_tensor(batch, "vae_seqlen", label="renderer batch")
    if vae_seqlen.ndim < 1:
        raise SourceKVRouteBatchError("vae_seqlen must have a packed dimension")
    values = vae_seqlen.squeeze(0)
    values = values[values > 0]
    lengths = [int(value) for value in values.tolist()]
    if len(lengths) != 1 or lengths[0] <= 0:
        raise SourceKVRouteBatchError("V9 requires one positive VAE sequence")
    return lengths


def renderer_full_velocity_prediction(
    renderer: Any,
    batch: Mapping[str, Any],
) -> Any:
    """Run the pinned Bernini renderer and return the full sequence field."""

    torch = _require_torch()
    if not isinstance(batch, Mapping):
        raise SourceKVRouteBatchError("renderer batch must be a mapping")
    _reject_forbidden_fields(batch, label="renderer batch")
    for name in CARRIER_MODEL_FIELDS:
        _require_tensor(batch, name, label="renderer batch")
    text_lens, text_embs = renderer.get_t5_text_embeddings(
        batch["input_ids"], batch["attention_mask"], batch["t5_input_lens"]
    )
    if len(text_lens) != 1:
        raise SourceKVRouteBatchError("V9 full prediction requires batch size one")
    vae_lengths = _positive_vae_lengths(batch)
    timesteps = batch["timesteps"].squeeze(0)[:1].unsqueeze(0)
    if timesteps.numel() != 1 or not bool(torch.isfinite(timesteps.float()).all()):
        raise SourceKVRouteBatchError("renderer timestep must be one finite scalar")
    decoder = renderer.diff_dec
    if decoder.transformer is not None and decoder.transformer_2 is not None:
        raise SourceKVRouteBatchError("V9 requires exactly one Wan expert")
    if decoder.transformer_2 is None:
        model_id, transformer = "transformer_1", decoder.transformer
    else:
        model_id, transformer = "transformer_2", decoder.transformer_2
    if transformer is None:
        raise SourceKVRouteBatchError("active Wan transformer is unavailable")

    latent_tokens = batch["input_vae_latents"]
    rope_tokens = batch["input_vae_rope"]
    expected_tokens = vae_lengths[0]
    if (
        latent_tokens.ndim < 2
        or rope_tokens.ndim != 3
        or int(latent_tokens.shape[0]) != expected_tokens
        or int(rope_tokens.shape[0]) != expected_tokens
    ):
        raise SourceKVRouteBatchError(
            "latent/RoPE token length differs from the VAE sequence"
        )
    inputs = latent_tokens.unsqueeze(0)
    inputs = transformer.patch_embedding(inputs.squeeze(0)).flatten(1).unsqueeze(0)
    if int(inputs.shape[1]) != expected_tokens:
        raise SourceKVRouteBatchError(
            "patch embedding changed the packed token axis"
        )
    rope = rope_tokens.permute(1, 0, 2).unsqueeze(0)
    prediction = decoder.shared_step(
        model_id=model_id,
        noisy_latents=inputs,
        timesteps=timesteps.squeeze(0),
        cond_embeds=text_embs,
        rotary_embs=rope,
        batch_vae_seqlen=vae_lengths,
        batch_text_seqlen=text_lens,
    )
    if (
        not isinstance(prediction, torch.Tensor)
        or prediction.ndim != 3
        or int(prediction.shape[0]) != 1
        or int(prediction.shape[1]) != expected_tokens
        or int(prediction.shape[2]) <= 0
        or not bool(torch.isfinite(prediction).all())
    ):
        raise SourceKVRouteBatchError(
            "shared_step did not return finite full [1,N,D] prediction"
        )
    return prediction


def select_target_velocity(full_prediction: Any, pair_batch: Mapping[str, Any]) -> Any:
    """Select the exact target suffix from a verified full pair prediction."""

    torch = _require_torch()
    boundary = _pair_boundary(pair_batch, label="selection pair")
    if (
        not isinstance(full_prediction, torch.Tensor)
        or full_prediction.ndim != 3
        or int(full_prediction.shape[0]) != 1
        or int(full_prediction.shape[1]) != 2 * boundary
        or int(full_prediction.shape[2]) <= 0
        or not bool(torch.isfinite(full_prediction).all())
    ):
        raise SourceKVRouteBatchError(
            "full pair prediction must be finite [1,2N,D]"
        )
    selector = pair_batch["vae_latents_mask"].squeeze(0).bool()
    selected = full_prediction[:, selector, :]
    if tuple(selected.shape[:2]) != (1, boundary):
        raise SourceKVRouteBatchError("target selection did not return [1,N,D]")
    return selected


__all__ = [
    "CARRIER_MODEL_FIELDS",
    "EXACT_NOOP_INSTRUCTION",
    "EXACT_NOOP_INSTRUCTION_SHA256",
    "FORBIDDEN_EXTERNAL_FIELDS",
    "SourceKVRouteBatchError",
    "SourceOnlyCarrierBatch",
    "build_source_only_carrier_batch",
    "renderer_full_velocity_prediction",
    "select_target_velocity",
    "validate_equal_pair_batches",
    "validate_noop_instruction",
]
