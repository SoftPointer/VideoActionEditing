#!/usr/bin/env python3
"""Pinned four-rank AUH trainer for Bernini Cross-Mode CMSG LoRA v6.

This is the checkpoint-producing integration layer around the separately
tested CMSG tensor core.  Each accepted optimizer step runs exactly six Wan
forwards on one beta=0 source query and one selected 40-step UniPC sigma:

1. frozen full-source MV2V negative, no-op, and action;
2. adapted full-source MV2V action (the only forward with a graph);
3. frozen target-only T2V negative and action.

The target-only tensors are direct views of the editor target tail and use the
official T2V system prompt.  The editor uses Bernini's momentum-zero APG.  The
generator uses official native-BF16 plain CFG, ``v_n + 4*(v_a-v_n)``, before
clean reconstruction.  Paired target data is an offline Q0 teacher only;
inference conditions remain source video plus action instruction.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import cross_mode_branches as branch_geometry  # noqa: E402
import cross_mode_motion_spectrum as spectrum  # noqa: E402
import inference_sigma_strata as sigma_strata  # noqa: E402
import motion_residual as motion  # noqa: E402
import train_cross_mode_cmsg_lora as core  # noqa: E402
import train_delta_lora as v4  # noqa: E402
import train_lora as legacy  # noqa: E402
import train_prior_tangent_lora as v5  # noqa: E402


METHOD_NAME = "bernini-cross-mode-cmsg-lora-v6-auh"
RECEIPT_SCHEMA = "bernini-r-1p3b-cross-mode-cmsg-auh-receipt-v6"
OPTIMIZER_SCHEMA = "bernini-r-1p3b-cross-mode-cmsg-auh-optimizer-v6"
NUM_FRAMES = 81
LATENT_PHASES = 21
LEARNING_RATE = 2.0e-5
MINIMUM_TRAINING_SIGMA = 0.1
MAX_GATE_ATTEMPTS_DEFAULT = 8
TRAINING_BRIDGE_ENDPOINT = "source(beta=0)"
T2V_GUIDANCE_SCALE = 4.0
T2V_SYSTEM_PROMPT = (
    "You are a helpful assistant specialized in text-to-video generation."
)
T2V_SYSTEM_PROMPT_SHA256 = hashlib.sha256(
    T2V_SYSTEM_PROMPT.encode("utf-8")
).hexdigest()
FORWARD_CELL_ORDER = (
    "frozen_editor_negative_full_source",
    "frozen_editor_noop_full_source",
    "frozen_editor_action_full_source",
    "adapted_editor_action_full_source",
    "frozen_generator_negative_target_only",
    "frozen_generator_action_target_only",
)


class CMSGauhTrainingError(RuntimeError):
    """Raised before a misleading or non-pinned AUH optimizer update."""


@dataclass(frozen=True)
class PreparedCandidate:
    editor_negative: Mapping[str, Any]
    editor_noop: Mapping[str, Any]
    editor_action: Mapping[str, Any]
    generator_action_text_fields: Mapping[str, Any]
    generator_negative_text_fields: Mapping[str, Any]
    auxiliary: Mapping[str, Any]
    spatial_hw: tuple[int, int]
    instruction_sha256: str
    t2v_rope_parity: Mapping[str, Any]


@dataclass(frozen=True)
class ForwardCellResult:
    weighted_loss: Any
    loss_result: Any
    inverse_sigma_weight: Any
    gate_preview: Any


@dataclass(frozen=True)
class MovedCandidate:
    editor_negative: Mapping[str, Any]
    editor_noop: Mapping[str, Any]
    editor_action: Mapping[str, Any]
    generator_action: Mapping[str, Any]
    generator_negative: Mapping[str, Any]
    generator_action_text_fields: Mapping[str, Any]
    generator_negative_text_fields: Mapping[str, Any]
    auxiliary: Mapping[str, Any]
    spatial_hw: tuple[int, int]
    instruction_sha256: str
    t2v_rope_parity: Mapping[str, Any]


def _translate(error: Exception) -> CMSGauhTrainingError:
    return CMSGauhTrainingError(str(error))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train pinned 81f Bernini Cross-Mode CMSG LoRA on four AUH GPUs"
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preprocessed-parquet-dir", required=True)
    parser.add_argument("--dataset-summary", required=True)
    parser.add_argument("--routing-jsonl", required=True)
    parser.add_argument(
        "--expected-routing-jsonl-sha256",
        default=v5.STRICT_ROUTING_SHA256,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-frames", type=int, choices=(81,), default=81)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--save-every", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--max-gate-attempts",
        type=int,
        default=MAX_GATE_ATTEMPTS_DEFAULT,
    )
    parser.add_argument(
        "--disable-frozen-prior-gate",
        dest="enforce_frozen_prior_gate",
        action="store_false",
        help="one-step diagnostic canary only; production default is fail-closed",
    )
    parser.set_defaults(enforce_frozen_prior_gate=True)
    parser.add_argument("--noop-instruction", default=motion.DEFAULT_NOOP_INSTRUCTION)
    parser.add_argument("--negative-prompt", default=v5.DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "--expected-bernini-commit", default=legacy.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=legacy.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=legacy.CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    if args.num_frames != NUM_FRAMES or legacy.LATENT_FRAMES != LATENT_PHASES:
        raise CMSGauhTrainingError("v6 requires exact 81-frame / 21-phase data")
    if type(args.max_steps) is not int or args.max_steps <= 0:
        raise CMSGauhTrainingError("max_steps must be a positive integer")
    if type(args.save_every) is not int or args.save_every < 0:
        raise CMSGauhTrainingError("save_every must be a non-negative integer")
    if (
        type(args.max_gate_attempts) is not int
        or not 1 <= args.max_gate_attempts <= legacy.EXPECTED_STRICT_ROWS
    ):
        raise CMSGauhTrainingError("max_gate_attempts must lie in [1,359]")
    if type(args.enforce_frozen_prior_gate) is not bool:
        raise CMSGauhTrainingError("frozen-prior gate setting must be boolean")
    if not args.enforce_frozen_prior_gate and args.max_steps != 1:
        raise CMSGauhTrainingError(
            "gate-disabled mode is authorized only for an explicit one-step canary"
        )
    if float(args.learning_rate) != LEARNING_RATE:
        raise CMSGauhTrainingError(f"v6 fixes learning_rate to {LEARNING_RATE}")
    if not math.isfinite(float(args.weight_decay)) or args.weight_decay < 0.0:
        raise CMSGauhTrainingError("weight_decay must be finite and non-negative")
    if not math.isfinite(float(args.max_grad_norm)) or args.max_grad_norm <= 0.0:
        raise CMSGauhTrainingError("max_grad_norm must be finite and positive")
    if args.noop_instruction != motion.DEFAULT_NOOP_INSTRUCTION:
        raise CMSGauhTrainingError("v6 pins the semantic no-op instruction")
    if args.negative_prompt != v5.DEFAULT_NEGATIVE_PROMPT:
        raise CMSGauhTrainingError("v6 pins Bernini's verbatim negative prompt")
    if args.expected_routing_jsonl_sha256 != v5.STRICT_ROUTING_SHA256:
        raise CMSGauhTrainingError("v6 requires the hash-bound strict-359 route")
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        if re.fullmatch(r"[0-9a-fA-F]{40}", str(getattr(args, name))) is None:
            raise CMSGauhTrainingError(f"{name} must be a full SHA-1")
    for name in (
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
        "expected_routing_jsonl_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(getattr(args, name))) is None:
            raise CMSGauhTrainingError(f"{name} must be a lowercase SHA-256")
    if args.expected_bernini_commit.lower() != legacy.BERNINI_OFFICIAL_COMMIT:
        raise CMSGauhTrainingError("Bernini revision differs from the pinned release")
    if args.expected_veomni_commit.lower() != legacy.VEOMNI_TESTED_COMMIT:
        raise CMSGauhTrainingError("VeOmni revision differs from the pinned release")
    if args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256:
        raise CMSGauhTrainingError("Bernini 1.3B checkpoint tree differs")
    if (
        core.LORA_RANK != 8
        or core.LORA_ALPHA != 8
        or core.EXPECTED_LORA_MODULES != 46
    ):
        raise CMSGauhTrainingError("v6 requires exact 46-module rank8/alpha8 LoRA")
    core.CMSGTrainingLossConfig(
        enforce_frozen_prior_gate=args.enforce_frozen_prior_gate
    ).validate()
    spectrum.CrossModeMotionSpectrumConfig().validate()


def _text_length(fields: Mapping[str, Any], *, label: str) -> int:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - AUH has torch
        raise CMSGauhTrainingError("PyTorch is required") from error
    ids = fields.get("input_ids")
    mask = fields.get("attention_mask")
    lens = fields.get("t5_input_lens")
    if (
        not isinstance(ids, torch.Tensor)
        or not isinstance(mask, torch.Tensor)
        or not isinstance(lens, torch.Tensor)
        or ids.ndim != 2
        or tuple(ids.shape) != tuple(mask.shape)
        or int(ids.shape[0]) != 1
        or lens.numel() != 1
    ):
        raise CMSGauhTrainingError(f"{label} must contain batched [1,L] T5 fields")
    length = int(lens.item())
    if length <= 0 or length != int(ids.shape[1]):
        raise CMSGauhTrainingError(f"{label} must expose exactly L valid tokens")
    return length


def _bind_text_geometry(
    batch: Mapping[str, Any], text_fields: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """Replace text plus the two Bernini packing lengths as one transaction."""

    import torch

    length = _text_length(text_fields, label=label)
    result = dict(batch)
    for field in branch_geometry.TEXT_FIELDS:
        result[field] = text_fields[field]
    vae_seqlen = result.get("vae_seqlen")
    vlm_seqlen = result.get("vlm_seqlen")
    num_tokens = result.get("num_tokens")
    if any(
        not isinstance(value, torch.Tensor)
        for value in (vae_seqlen, vlm_seqlen, num_tokens)
    ) or vae_seqlen.numel() != 1:
        raise CMSGauhTrainingError(f"{label} lacks Bernini token geometry")
    vae_length = int(vae_seqlen.item())
    result["vlm_seqlen"] = torch.full_like(vlm_seqlen, length)
    result["num_tokens"] = torch.full_like(num_tokens, vae_length + length)
    return result


def _official_t2v_text_fields(
    sample: Mapping[str, Any],
    *,
    tokenizer: Any,
    prompt_cleaner: Any,
    system_prompts: Mapping[str, str],
) -> tuple[dict[str, Any], str, str]:
    """Tokenize the raw instruction using pinned Bernini's exact T2V prefix."""

    import torch

    if system_prompts.get("t2v") != T2V_SYSTEM_PROMPT:
        raise CMSGauhTrainingError("runtime Bernini T2V system prompt differs")
    try:
        messages = json.loads(str(sample["inputs"]))
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise CMSGauhTrainingError("cannot decode raw renderer instruction") from error
    if (
        not isinstance(messages, list)
        or len(messages) != 3
        or not isinstance(messages[1], Mapping)
        or messages[1].get("type") != "text"
    ):
        raise CMSGauhTrainingError("renderer row lacks its single raw instruction")
    instruction = messages[1].get("text")
    if not isinstance(instruction, str) or not instruction.strip() or "\x00" in instruction:
        raise CMSGauhTrainingError("raw action instruction is invalid")
    cleaned = prompt_cleaner(instruction)
    if not isinstance(cleaned, str) or not cleaned.strip():
        raise CMSGauhTrainingError("Wan prompt_clean produced an empty instruction")
    full_prompt = T2V_SYSTEM_PROMPT + cleaned
    encoded = tokenizer(
        full_prompt,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    ids = getattr(encoded, "input_ids", None)
    mask = getattr(encoded, "attention_mask", None)
    if isinstance(encoded, Mapping):
        ids = encoded.get("input_ids", ids)
        mask = encoded.get("attention_mask", mask)
    if (
        not isinstance(ids, torch.Tensor)
        or not isinstance(mask, torch.Tensor)
        or ids.ndim != 2
        or tuple(ids.shape) != tuple(mask.shape)
        or int(ids.shape[0]) != 1
        or not 0 < int(ids.shape[1]) <= 512
    ):
        raise CMSGauhTrainingError("official T2V tokenization must return [1,L<=512]")
    length = int(ids.shape[1])
    fields = {
        "input_ids": ids,
        "attention_mask": mask,
        "t5_input_lens": torch.tensor(
            [[length]], dtype=torch.long, device=ids.device
        ),
    }
    _text_length(fields, label="official T2V action")
    return fields, instruction, hashlib.sha256(full_prompt.encode("utf-8")).hexdigest()


def _spatial_hw_from_sample(sample: Mapping[str, Any], *, z_dim: int) -> tuple[int, int]:
    try:
        shapes = legacy.validate_81_frame_latents(
            sample, expected_parameter_channels=2 * z_dim
        )
    except legacy.TrainingContractError as error:
        raise _translate(error) from error
    source_shape, target_shape = shapes
    if source_shape != target_shape or source_shape[3] % 2 or source_shape[4] % 2:
        raise CMSGauhTrainingError("source/target VAE spatial geometry differs")
    return source_shape[3] // 2, source_shape[4] // 2


def _validate_native_t2v_rope_parity(
    editor_action: Mapping[str, Any],
    *,
    rope: Any,
    z_dim: int,
    spatial_hw: tuple[int, int],
) -> dict[str, Any]:
    """Prove the MV2V target tail is native T2V RoPE, not editor metadata.

    Pinned Bernini ``pack_vae_latents`` assigns ``source_id=0`` to every
    target segment (``vae_mask=True``).  A target-only T2V branch has the same
    rule.  Recomputing the native one-target RoPE at the exact VAE shape and
    requiring bit-exact equality makes that source-level argument executable.
    """

    import torch

    if (
        type(z_dim) is not int
        or z_dim <= 0
        or type(spatial_hw) is not tuple
        or len(spatial_hw) != 2
        or any(type(value) is not int or value <= 0 for value in spatial_hw)
    ):
        raise CMSGauhTrainingError("native T2V RoPE geometry is invalid")
    if (
        getattr(rope, "use_src_id_rotary_emb", None) is not True
        or int(getattr(rope, "attention_head_dim", -1)) != 128
        or tuple(getattr(rope, "patch_size", ())) != (1, 2, 2)
        or int(getattr(rope, "max_seq_len", -1)) != 1024
    ):
        raise CMSGauhTrainingError("runtime RoPE is not pinned Bernini source-id 3D RoPE")
    mask = editor_action.get("vae_latents_mask")
    packed_rope = editor_action.get("input_vae_rope")
    if (
        not isinstance(mask, torch.Tensor)
        or not isinstance(packed_rope, torch.Tensor)
        or mask.ndim != 2
        or int(mask.shape[0]) != 1
        or mask.dtype != torch.bool
    ):
        raise CMSGauhTrainingError("editor target RoPE provenance lacks packed tensors")
    selector = mask.squeeze(0)
    target_tokens = int(selector.sum().item())
    height, width = spatial_hw
    if (
        target_tokens != LATENT_PHASES * height * width
        or int(selector.numel()) != 2 * target_tokens
        or bool(selector[:target_tokens].any())
        or not bool(selector[target_tokens:].all())
    ):
        raise CMSGauhTrainingError("editor target tail is not one 21-phase target")

    # The official rotary module uses only VAE shape and source_id.  Keeping
    # this proof on CPU avoids an extra GPU allocation and happens before the
    # target-tail storage view is formed on the training device.
    probe = torch.empty(
        (1, z_dim, LATENT_PHASES, 2 * height, 2 * width),
        dtype=torch.bfloat16,
        device=packed_rope.device,
    )
    try:
        with torch.no_grad():
            native = rope(probe, source_id=0).squeeze(0).permute(1, 0, 2)
    except (AssertionError, RuntimeError, TypeError, ValueError) as error:
        raise CMSGauhTrainingError("cannot recompute native target-only T2V RoPE") from error
    target_tail = packed_rope[target_tokens:]
    if (
        tuple(native.shape) != tuple(target_tail.shape)
        or native.dtype != target_tail.dtype
        or native.device != target_tail.device
        or not torch.equal(native, target_tail)
    ):
        raise CMSGauhTrainingError(
            "MV2V target-tail RoPE differs from native T2V source_id=0 RoPE"
        )
    return {
        "verified": True,
        "official_pack_rule": "vae_mask=True -> source_id=0",
        "mv2v_target_source_id": 0,
        "native_t2v_target_source_id": 0,
        "same_target_shape": True,
        "exact_tensor_equality": True,
        "target_tokens": target_tokens,
        "rope_shape": list(native.shape),
    }


def _prepare_candidate_cpu(
    *,
    raw_row: Mapping[str, Any],
    tokenizer: Any,
    prompt_cleaner: Any,
    system_prompts: Mapping[str, str],
    rope: Any,
    vae_mean: Any,
    vae_std: Any,
    z_dim: int,
    scheduler: Any,
    noop_instruction: str,
    negative_prompt: str,
    process_renderer_sample: Any,
    selected_stratum: Any,
) -> PreparedCandidate:
    try:
        sample = legacy.sanitize_preprocessed_row(raw_row)
        spatial_hw = _spatial_hw_from_sample(sample, z_dim=z_dim)
        t2v_text, instruction, _instruction_prompt_sha = _official_t2v_text_fields(
            sample,
            tokenizer=tokenizer,
            prompt_cleaner=prompt_cleaner,
            system_prompts=system_prompts,
        )
        endpoints = v5._prepare_prior_bridge_batches(
            raw_row=raw_row,
            tokenizer=tokenizer,
            rope=rope,
            vae_mean=vae_mean,
            vae_std=vae_std,
            z_dim=z_dim,
            scheduler=scheduler,
            noop_instruction=noop_instruction,
            negative_prompt=negative_prompt,
            minimum_training_sigma=MINIMUM_TRAINING_SIGMA,
            process_renderer_sample=process_renderer_sample,
            selected_stratum=selected_stratum,
        )
    except (
        legacy.TrainingContractError,
        motion.MotionContractError,
        sigma_strata.InferenceSigmaStrataError,
        v4.DeltaTrainingError,
        v5.PriorTangentTrainingError,
    ) as error:
        raise _translate(error) from error
    editor_negative, editor_noop, editor_action, auxiliary = endpoints["source"]
    if float(auxiliary.get("bridge_fraction", -1.0)) != 0.0:
        raise CMSGauhTrainingError("main v6 training must use beta=0 source endpoint")
    negative_text = {
        field: editor_negative[field] for field in branch_geometry.TEXT_FIELDS
    }
    editor_negative = _bind_text_geometry(
        editor_negative, negative_text, label="full-source negative"
    )
    t2v_rope_parity = _validate_native_t2v_rope_parity(
        editor_action,
        rope=rope,
        z_dim=z_dim,
        spatial_hw=spatial_hw,
    )
    return PreparedCandidate(
        editor_negative=editor_negative,
        editor_noop=editor_noop,
        editor_action=editor_action,
        generator_action_text_fields=t2v_text,
        generator_negative_text_fields=negative_text,
        auxiliary=auxiliary,
        spatial_hw=spatial_hw,
        instruction_sha256=hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        t2v_rope_parity=t2v_rope_parity,
    )


def _move_candidate_to_device(
    candidate: PreparedCandidate, *, device: Any
) -> MovedCandidate:
    """Move the editor first, then slice GPU target views for T2V branches."""

    try:
        editor_negative = legacy._move_batch(candidate.editor_negative, device)
        editor_noop = legacy._move_batch(candidate.editor_noop, device)
        editor_action = legacy._move_batch(candidate.editor_action, device)
        action_text = legacy._move_batch(
            candidate.generator_action_text_fields, device
        )
        negative_text = legacy._move_batch(
            candidate.generator_negative_text_fields, device
        )
        auxiliary = v4._move_auxiliary_to_device(
            candidate.auxiliary,
            device=device,
            branch_state_mode="source_target_bridge_clean_field",
        )
        branches = core.build_training_branches(
            editor_action,
            action_text,
            negative_text,
        )
    except (
        legacy.TrainingContractError,
        v4.DeltaTrainingError,
        core.CrossModeCMSGTrainingError,
    ) as error:
        raise _translate(error) from error
    moved = MovedCandidate(
        editor_negative=editor_negative,
        editor_noop=editor_noop,
        editor_action=editor_action,
        generator_action=branches.generator_action,
        generator_negative=branches.generator_negative,
        generator_action_text_fields=action_text,
        generator_negative_text_fields=negative_text,
        auxiliary=auxiliary,
        spatial_hw=candidate.spatial_hw,
        instruction_sha256=candidate.instruction_sha256,
        t2v_rope_parity=candidate.t2v_rope_parity,
    )
    _validate_six_branch_state(moved)
    return moved


def _validate_six_branch_state(candidate: MovedCandidate) -> None:
    import torch

    try:
        v5._assert_same_endpoint_state(
            candidate.editor_negative,
            candidate.editor_noop,
            candidate.editor_action,
        )
        branch_geometry.validate_cross_mode_branches(
            candidate.editor_action,
            candidate.generator_action,
            candidate.generator_negative,
            generator_action_text_fields=candidate.generator_action_text_fields,
            generator_negative_text_fields=candidate.generator_negative_text_fields,
        )
    except (
        v5.PriorTangentTrainingError,
        branch_geometry.CrossModeBranchError,
    ) as error:
        raise _translate(error) from error
    selector = candidate.editor_action["vae_latents_mask"].squeeze(0).bool()
    target_tokens = int(selector.sum().item())
    if target_tokens <= 0 or int((~selector).sum().item()) != target_tokens:
        raise CMSGauhTrainingError("editor branch is not source N + target N")
    shared_noisy = candidate.auxiliary.get("shared_noisy")
    if (
        not isinstance(shared_noisy, torch.Tensor)
        or shared_noisy.dtype != torch.float32
        or tuple(shared_noisy.shape[:2]) != (1, target_tokens)
    ):
        raise CMSGauhTrainingError("beta=0 shared noisy target is not fp32 [1,N,D]")
    if target_tokens != LATENT_PHASES * math.prod(candidate.spatial_hw):
        raise CMSGauhTrainingError(
            "VAE-derived spatial_hw does not match the packed 21-phase target"
        )
    editor_tail = candidate.editor_action["input_vae_latents"][selector]
    packed_editor_tail = motion.flatten_velocity_patches(
        editor_tail.unsqueeze(0)
    ).float()
    if not torch.equal(packed_editor_tail, shared_noisy):
        raise CMSGauhTrainingError(
            "generator/editor target tail differs from the audited shared noisy query"
        )
    for generator in (candidate.generator_negative, candidate.generator_action):
        if not torch.equal(
            generator["timesteps"], candidate.editor_action["timesteps"]
        ):
            raise CMSGauhTrainingError("generator/editor timesteps differ")
    if float(candidate.auxiliary.get("bridge_fraction", -1.0)) != 0.0:
        raise CMSGauhTrainingError("target endpoint teacher leakage is forbidden")
    parity = candidate.t2v_rope_parity
    if (
        not isinstance(parity, Mapping)
        or parity.get("verified") is not True
        or parity.get("official_pack_rule") != "vae_mask=True -> source_id=0"
        or parity.get("mv2v_target_source_id") != 0
        or parity.get("native_t2v_target_source_id") != 0
        or parity.get("exact_tensor_equality") is not True
        or parity.get("target_tokens") != target_tokens
    ):
        raise CMSGauhTrainingError("native T2V target-tail RoPE parity is unaudited")


def _generator_plain_cfg_clean(
    *,
    shared_noisy: Any,
    sigma: Any,
    negative_velocity: Any,
    action_velocity: Any,
) -> tuple[Any, Any]:
    """Reproduce pinned T2V native-BF16 CFG before clean reconstruction."""

    import torch

    if (
        negative_velocity.dtype != torch.bfloat16
        or action_velocity.dtype != torch.bfloat16
        or tuple(negative_velocity.shape) != tuple(action_velocity.shape)
    ):
        raise CMSGauhTrainingError("T2V CFG requires matching native-BF16 fields")
    guided_velocity = negative_velocity + T2V_GUIDANCE_SCALE * (
        action_velocity - negative_velocity
    )
    if guided_velocity.dtype != torch.bfloat16:
        raise CMSGauhTrainingError("T2V CFG left the native BF16 numerical path")
    try:
        negative_clean = v5.tri.pinned_raw_condition_clean(
            shared_noisy, negative_velocity, sigma
        )
        guided_clean = v5.tri.pinned_raw_condition_clean(
            shared_noisy, guided_velocity, sigma
        )
        return v5._as_phase_grid(negative_clean), v5._as_phase_grid(guided_clean)
    except (v5.tri.TriBranchHookError, v5.PriorTangentTrainingError) as error:
        raise _translate(error) from error


def _run_six_forward_cell(
    *,
    renderer: Any,
    adapter_controller: Any,
    candidate: MovedCandidate,
    step_index: int,
    enforce_frozen_prior_gate: bool,
) -> ForwardCellResult:
    """Run the literal six-forward cell and delegate its loss to the v6 core."""

    import torch

    shared_noisy = candidate.auxiliary["shared_noisy"]
    sigma = candidate.auxiliary["sigma"]
    with torch.no_grad():
        with adapter_controller.disable_adapter():
            editor_negative_v = motion.renderer_velocity_prediction(
                renderer, candidate.editor_negative
            )
            editor_noop_v = motion.renderer_velocity_prediction(
                renderer, candidate.editor_noop
            )
            editor_action_v = motion.renderer_velocity_prediction(
                renderer, candidate.editor_action
            )
    adapted_editor_action_v = motion.renderer_velocity_prediction(
        renderer, candidate.editor_action
    )
    with torch.no_grad():
        with adapter_controller.disable_adapter():
            generator_negative_v = motion.renderer_velocity_prediction(
                renderer, candidate.generator_negative
            )
            generator_action_v = motion.renderer_velocity_prediction(
                renderer, candidate.generator_action
            )
    velocities = (
        editor_negative_v,
        editor_noop_v,
        editor_action_v,
        adapted_editor_action_v,
        generator_negative_v,
        generator_action_v,
    )
    if any(
        value.dtype != torch.bfloat16
        or tuple(value.shape) != tuple(shared_noisy.shape)
        or not bool(torch.isfinite(value).all())
        for value in velocities
    ):
        raise CMSGauhTrainingError(
            "all six forwards must be finite native-BF16 fields on one noisy query"
        )
    if any(value.requires_grad for value in velocities[:3]) or any(
        value.requires_grad for value in velocities[4:]
    ):
        raise CMSGauhTrainingError("a frozen editor/generator branch retained a graph")
    if not adapted_editor_action_v.requires_grad:
        raise CMSGauhTrainingError("adapted editor action is not the sole graph branch")

    try:
        _, editor_noop = v5._guided_clean(
            shared_noisy=shared_noisy,
            sigma=sigma,
            negative_velocity=editor_negative_v,
            conditional_velocity=editor_noop_v,
        )
        _, frozen_editor_action = v5._guided_clean(
            shared_noisy=shared_noisy,
            sigma=sigma,
            negative_velocity=editor_negative_v,
            conditional_velocity=editor_action_v,
        )
        _, adapted_editor_action = v5._guided_clean(
            shared_noisy=shared_noisy,
            sigma=sigma,
            negative_velocity=editor_negative_v,
            conditional_velocity=adapted_editor_action_v,
        )
        generator_uncond, frozen_generator_action = _generator_plain_cfg_clean(
            shared_noisy=shared_noisy,
            sigma=sigma,
            negative_velocity=generator_negative_v,
            action_velocity=generator_action_v,
        )
        source = v5._as_phase_grid(candidate.auxiliary["source_clean"].float())
        target = v5._as_phase_grid(candidate.auxiliary["target_clean"].float())
        target_motion = spectrum.q0(target - source).detach()
        generator_teacher = spectrum.q0(
            frozen_generator_action - generator_uncond
        ).detach()
        loss_config = core.CMSGTrainingLossConfig(
            enforce_frozen_prior_gate=enforce_frozen_prior_gate
        )
        gate_preview = core.compute_frozen_prior_gate(
            generator_teacher,
            target_motion,
            config=loss_config,
        )
        try:
            result = core.compute_cmsg_lora_loss(
                adapted_editor_action_field=adapted_editor_action,
                frozen_editor_action_field=frozen_editor_action,
                editor_noop_field=editor_noop,
                frozen_generator_action_field=frozen_generator_action,
                generator_uncond_field=generator_uncond,
                target_motion_field=target_motion,
                step_index=step_index,
                spatial_hw=candidate.spatial_hw,
                loss_config=loss_config,
            )
        except core.FrozenPriorGateRejected as error:
            setattr(error, "gate_result", gate_preview)
            raise
        inverse_sigma = motion.clean_field_inverse_sigma_weight(
            sigma,
            weight_floor=sigma_strata.PINNED_POSITIVE_SIGMAS[-1],
        ).mean()
    except (
        core.CrossModeCMSGTrainingError,
        spectrum.CrossModeMotionSpectrumError,
        motion.MotionContractError,
        v5.PriorTangentTrainingError,
    ):
        raise
    weighted = inverse_sigma * result.total
    if not bool(torch.isfinite(weighted)):
        raise CMSGauhTrainingError("inverse-sigma weighted CMSG loss is non-finite")
    return ForwardCellResult(
        weighted_loss=weighted,
        loss_result=result,
        inverse_sigma_weight=inverse_sigma,
        gate_preview=gate_preview,
    )


def _gate_audit_record(
    gate: Any,
    *,
    global_step: int,
    attempt_ordinal: int,
    attempt_in_step: int,
    row_index: int,
    iid: str,
    schedule_index: int,
    timestep: int,
    accepted: bool,
    teacher_active: bool,
) -> dict[str, Any]:
    def scalar(name: str, cast: Any = float) -> Any:
        value = getattr(gate, name)
        return cast(value.reshape(-1)[0].detach().cpu().item())

    return {
        "global_step_before_update": int(global_step),
        "attempt_ordinal": int(attempt_ordinal),
        "attempt_in_step": int(attempt_in_step),
        "row_index": int(row_index),
        "iid": str(iid),
        "sigma_schedule_index": int(schedule_index),
        "sigma_timestep": int(timestep),
        "teacher_active": bool(teacher_active),
        "gate_passed": bool(scalar("passed", bool)),
        "accepted": bool(accepted),
        "active_phase_count": int(scalar("active_phase_count", int)),
        "mean_direction_cosine": scalar("mean_direction_cosine"),
        "log_amplitude_mae": scalar("log_amplitude_mae"),
        "covered_phase_fraction": scalar("covered_phase_fraction"),
        "normalized_rmse": scalar("normalized_rmse"),
        "frozen_prior_rms": scalar("frozen_prior_rms"),
        "target_motion_rms": scalar("target_motion_rms"),
    }


def _assert_gate_record_equal_across_ranks(record: Mapping[str, Any]) -> None:
    """Require the complete detached gate decision, not just its bool, to agree."""

    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    gathered: list[Optional[dict[str, Any]]] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, dict(record))
    if any(candidate != gathered[0] for candidate in gathered[1:]):
        raise CMSGauhTrainingError("frozen-prior gate metrics differ across ranks")


def _loss_metrics(cell: ForwardCellResult) -> dict[str, float]:
    result = cell.loss_result
    return {
        "total_weighted": float(cell.weighted_loss.detach().item()),
        "total_core": float(result.total.detach().item()),
        "editor_direction": float(result.editor_direction.detach().item()),
        "log_amplitude": float(result.log_amplitude.detach().item()),
        "generator_spectral_consistency": float(
            result.generator_spectral_consistency.detach().item()
        ),
        "high_frequency_detail": float(result.high_frequency_detail.detach().item()),
        "late_frozen_replay": float(result.late_frozen_replay.detach().item()),
        "rho": float(result.rho),
        "inverse_sigma_weight": float(cell.inverse_sigma_weight.detach().item()),
    }


def _strict_router(
    args: argparse.Namespace,
    router: Any,
    eligible_routes: Sequence[tuple[int, Any]],
    dataset: Any,
) -> None:
    receipt = router.receipt()
    if (
        receipt.get("path") is None
        or receipt.get("default_tier") != "reject"
        or receipt.get("file_sha256") != args.expected_routing_jsonl_sha256
        or receipt.get("explicit_route_counts")
        != {"full_pair": 0, "motion_only": 359, "reject": 285}
        or len(dataset) != 644
        or len(eligible_routes) != 359
        or any(
            route.tier != "motion_only" or route.full_target_weight != 0.0
            for _, route in eligible_routes
        )
    ):
        raise CMSGauhTrainingError(
            "v6 requires hash-bound strict359 motion-only / reject285 routing"
        )


def _immutable_contract(
    *,
    args: argparse.Namespace,
    dataset: Any,
    dataset_summary: Mapping[str, Any],
    router: Any,
    eligible_routes: Sequence[tuple[int, Any]],
    target_modules: Sequence[str],
    checkpoint: Path,
) -> dict[str, Any]:
    loss_config = asdict(
        core.CMSGTrainingLossConfig(
            enforce_frozen_prior_gate=args.enforce_frozen_prior_gate
        )
    )
    spectrum_config = asdict(spectrum.CrossModeMotionSpectrumConfig())
    release = list(spectrum.release_rho_schedule())
    value = {
        "method": METHOD_NAME,
        "schema_version": RECEIPT_SCHEMA,
        "method_source_revision": args.method_source_revision.lower(),
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "bernini_commit": args.expected_bernini_commit.lower(),
        "veomni_commit": args.expected_veomni_commit.lower(),
        "checkpoint_path": str(checkpoint),
        "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
        "dataset_signature": dataset.signature,
        "dataset_summary_sha256": dataset_summary["sha256"],
        "dataset_index_sha256": dataset_summary["index_sha256"],
        "routing_digest": router.digest,
        "routing_file_sha256": router.file_sha256,
        "eligible_route_count": len(eligible_routes),
        "eligible_route_stream_sha256": legacy.object_sha256(
            [
                {
                    "row_index": row_index,
                    "iid": route.iid,
                    "tier": route.tier,
                    "full_target_weight": route.full_target_weight,
                }
                for row_index, route in eligible_routes
            ]
        ),
        "frames": NUM_FRAMES,
        "latent_phases": LATENT_PHASES,
        "seed": int(args.seed),
        "learning_rate": LEARNING_RATE,
        "weight_decay": float(args.weight_decay),
        "max_grad_norm": float(args.max_grad_norm),
        "lora": {
            "scope": core.LORA_SCOPE,
            "rank": 8,
            "alpha": 8,
            "dropout": 0.0,
            "bias": "none",
            "target_module_count": len(target_modules),
            "target_modules": list(target_modules),
            "target_modules_sha256": legacy.object_sha256(list(target_modules)),
        },
        "training_bridge_endpoint": TRAINING_BRIDGE_ENDPOINT,
        "target_endpoint_teacher_leakage_forbidden": True,
        "forward_cell_order": list(FORWARD_CELL_ORDER),
        "forwards_per_candidate": 6,
        "graph_forwards_per_candidate": 1,
        "training_editor_branches": [
            "frozen_negative_adapter_off_no_grad",
            "frozen_noop_adapter_off_no_grad",
            "frozen_action_adapter_off_no_grad",
            "adapted_action_adapter_on_grad",
        ],
        "inference_editor_branches": [
            "frozen_negative_adapter_off_no_grad",
            "frozen_noop_adapter_off_no_grad",
            "frozen_action_adapter_off_no_grad",
            "adapted_action_adapter_on_no_grad",
        ],
        "editor_guidance": {
            "mode": "official_momentum_zero_apg",
            "guidance_scale": v5.APG_GUIDANCE_SCALE,
            "eta": v5.APG_ETA,
            "norm_threshold": v5.APG_NORM_THRESHOLD,
            "momentum": v5.APG_MOMENTUM,
        },
        "generator_guidance": {
            "mode": "official_t2v_plain_cfg",
            "native_velocity_formula": "v_negative+4*(v_action-v_negative)",
            "scale": T2V_GUIDANCE_SCALE,
            "combine_before_fp32_clean_reconstruction": True,
        },
        "text_contract": {
            "editor_action": "official_mv2v_system_prompt_plus_prompt_clean",
            "generator_action": "official_t2v_system_prompt_plus_prompt_clean",
            "generator_t2v_system_prompt_sha256": T2V_SYSTEM_PROMPT_SHA256,
            "generator_negative": "official_negative_verbatim",
            "generator_negative_sha256": v5.NEGATIVE_PROMPT_SHA256,
        },
        "target_motion_teacher": "Q0(target_clean-source_clean)",
        "target_used_as_model_condition": False,
        "t2v_rope_parity": {
            "official_pack_rule": "vae_mask=True -> source_id=0",
            "mv2v_target_source_id": 0,
            "native_t2v_target_source_id": 0,
            "same_target_shape_required": True,
            "per_candidate_exact_tensor_equality_required": True,
            "generator_uses_direct_editor_target_tail_view": True,
        },
        "generator_target_tail": (
            "direct GPU storage view of editor noisy target; no transform/noise resample"
        ),
        "loss_config": loss_config,
        "spectrum_config": spectrum_config,
        "release_schedule": release,
        "release_schedule_sha256": legacy.object_sha256(release),
        "sigma_schedule": "exact_40_step_flow_shift_5_cycle",
        "sigma_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
        "sigma_selector": "absolute_global_step_mod_40",
        "inverse_sigma_weight_floor": float(
            sigma_strata.PINNED_POSITIVE_SIGMAS[-1]
        ),
        "gate": {
            "enforced": bool(args.enforce_frozen_prior_gate),
            "max_attempts_per_accepted_step": int(args.max_gate_attempts),
            "attempt_stream": "global_attempt_ordinal",
            "complete_record_exact_across_all_four_ranks": True,
            "rho_zero_teacher_inactive_no_rejection": True,
        },
        "candidate_seed_formula": "step_seed(base_seed,attempt_ordinal,row_index)",
        "inference_conditions": list(core.INFERENCE_CONDITIONS),
        "training_only_conditions": list(core.TRAINING_ONLY_CONDITIONS),
        "forbidden_inference_conditions": list(
            core.FORBIDDEN_INFERENCE_CONDITIONS
        ),
        "inference_generator_forwards": 0,
        "frozen_editor_direction": "Q0(frozen_action-frozen_noop)",
        "adapted_editor_direction": "Q0(adapted_action-frozen_noop)",
        "inference_execution": (
            "frozen_action_clean+(execute_distilled_editor(B0,Btheta,k)-B0)"
        ),
        "phase_zero_contract": "official_frozen_action_phase_zero_exactly_preserved",
        "release_contract": (
            "0-19 adapted; 20-31 inclusive cosine with rho31=0; "
            "32-39 exact official adapter-off replay"
        ),
        "zero_release_steps": [
            step for step, rho in enumerate(release) if rho == 0.0
        ],
        "formal_adapter_off_steps": list(range(32, 40)),
        "late_scheduler_boundary": (
            "exact_same_official_frozen_action_model_output_object"
        ),
        "resume_integrated": False,
    }
    if len(target_modules) != 46:
        raise CMSGauhTrainingError(
            f"v6 LoRA scope resolved {len(target_modules)} modules, expected 46"
        )
    return {"value": value, "digest": legacy.object_sha256(value)}


def _supervision_receipt(*, global_step: int) -> dict[str, Any]:
    """Bind the offline teachers and editor-only deployment boundary."""

    if type(global_step) is not int or global_step < 0:
        raise CMSGauhTrainingError("supervision receipt global_step is invalid")
    return {
        "method": METHOD_NAME,
        "full_bernini_training_integrated": True,
        "optimizer_updates_completed": global_step >= spectrum.NUM_DENOISING_STEPS,
        "checkpoint_optimizer_inference_receipt_parity": True,
        "frozen_target_only_generator_teacher": True,
        "generator_teacher_training_only": True,
        "generator_loaded_at_inference": False,
        "generator_forwards_at_inference": 0,
        "paired_target_training_only": True,
        "paired_target_used_at_inference": False,
        "base_editor_branches_adapter_disabled": True,
        "base_editor_branches_no_grad": True,
        "adapted_action_only_trainable_editor_forward": True,
        "official_editor_apg_momentum": 0.0,
        "inference_conditions": list(core.INFERENCE_CONDITIONS),
        "external_mask_track_flow_pose_trajectory": False,
        "first_frame_anchor": False,
        "training_bridge_endpoint": TRAINING_BRIDGE_ENDPOINT,
        "target_endpoint_teacher_leakage_forbidden": True,
        "native_t2v_rope_source_id_zero_parity_enforced": True,
        "resume_integrated": False,
        "production_claim_forbidden": True,
    }


def _optimizer_payload(
    *,
    optimizer: Any,
    global_step: int,
    attempt_ordinal: int,
    rejected_count: int,
    immutable: Mapping[str, Any],
    parameter_names: Sequence[str],
    gate_audit: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": OPTIMIZER_SCHEMA,
        "global_step": int(global_step),
        "attempt_ordinal": int(attempt_ordinal),
        "accepted_count": int(global_step),
        "rejected_count": int(rejected_count),
        "optimizer": optimizer.state_dict(),
        "immutable_contract": dict(immutable),
        "parameter_names": list(parameter_names),
        "gate_audit": list(gate_audit),
        "gate_audit_sha256": legacy.object_sha256(list(gate_audit)),
        "resume_integrated": False,
    }


def _build_receipt(
    *,
    args: argparse.Namespace,
    global_step: int,
    attempt_ordinal: int,
    rejected_count: int,
    metrics: Optional[Mapping[str, float]],
    gate_audit: Sequence[Mapping[str, Any]],
    dataset: Any,
    dataset_summary: Mapping[str, Any],
    router: Any,
    checkpoint: Path,
    bernini_revision: str,
    veomni_revision: str,
    distributed: Any,
    backend: str,
    target_modules: Sequence[str],
    named_trainable: Sequence[tuple[str, Any]],
    initialization_digest: str,
    transformers_version: str,
    immutable: Mapping[str, Any],
    optimizer_payload: Mapping[str, Any],
) -> dict[str, Any]:
    names = v4._optimizer_parameter_names(named_trainable)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "method": METHOD_NAME,
        "global_step": int(global_step),
        "max_steps": int(args.max_steps),
        "attempt_ordinal": int(attempt_ordinal),
        "accepted_count": int(global_step),
        "rejected_count": int(rejected_count),
        "gate_audit": list(gate_audit),
        "gate_audit_sha256": legacy.object_sha256(list(gate_audit)),
        "last_metrics": dict(metrics) if metrics is not None else None,
        "immutable_contract": dict(immutable),
        "bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "checkpoint": {
            "path": str(checkpoint),
            "tree_sha256": args.expected_checkpoint_tree_sha256,
        },
        "dataset": {
            "path": str(dataset.root),
            "rows": len(dataset),
            "signature": dataset.signature,
            "summary": dict(dataset_summary),
            "routing": router.receipt(),
        },
        "supervision": _supervision_receipt(global_step=global_step),
        "inference_sigma_strata": sigma_strata.build_sigma_strata_receipt(
            completed_optimizer_steps=global_step
        ),
        "adapter": {
            "rank": 8,
            "alpha": 8,
            "scope": core.LORA_SCOPE,
            "target_module_count": len(target_modules),
            "target_modules": list(target_modules),
            "target_modules_sha256": legacy.object_sha256(list(target_modules)),
            "trainable_parameter_count": sum(
                int(parameter.numel()) for _, parameter in named_trainable
            ),
            "parameter_names_sha256": legacy.object_sha256(names),
            "initialization_digest": initialization_digest,
            "checkpoint_parameter_digest": v4._checkpoint_parameter_digest(
                named_trainable
            ),
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": float(args.weight_decay),
            "max_gradient_norm": float(args.max_grad_norm),
            "parameter_names": names,
            "checkpoint_state_digest": v4._stable_recursive_digest(
                optimizer_payload
            ),
        },
        "distributed": {
            "world_size": distributed.world_size,
            "ulysses_size": distributed.ulysses_size,
            "backend": backend,
            "same_pair_all_ranks": True,
            "explicit_lora_gradient_all_reduce": True,
        },
        "transformers_version": transformers_version,
        "inference_conditions": list(core.INFERENCE_CONDITIONS),
        "training_only_generator_and_target": True,
        "experimental_training": True,
        "canary_gate_disabled": not bool(args.enforce_frozen_prior_gate),
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
        "resume_integrated": False,
        "inference_loader_parity_pending": False,
    }
    receipt["receipt_digest"] = legacy.object_sha256(receipt)
    return receipt


def _save_checkpoint(
    *,
    model: Any,
    optimizer_payload: Mapping[str, Any],
    output: Path,
    global_step: int,
    receipt: Mapping[str, Any],
    rank: int,
) -> Path:
    try:
        return v5._save_checkpoint(
            model=model,
            optimizer_payload=optimizer_payload,
            output=output,
            global_step=global_step,
            receipt=receipt,
            rank=rank,
        )
    except v5.PriorTangentTrainingError as error:
        raise _translate(error) from error


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    except legacy.TrainingContractError as error:
        raise _translate(error) from error
    if transformer_config["num_attention_heads"] % 4:
        raise CMSGauhTrainingError("1.3B attention heads must divide Ulysses=4")
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import UniPCMultistepScheduler
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, __version__ as transformers_version

    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.training.data import (
        NoiseScheduler,
        SYSTEM_PROMPTS,
        process_renderer_sample,
    )

    if DEFAULT_NEG_PROMPT != v5.DEFAULT_NEGATIVE_PROMPT:
        raise CMSGauhTrainingError("runtime Bernini negative prompt differs")
    if SYSTEM_PROMPTS.get("t2v") != T2V_SYSTEM_PROMPT:
        raise CMSGauhTrainingError("runtime Bernini T2V system prompt differs")

    distributed = legacy.distributed_contract()
    if distributed.world_size != 4 or distributed.ulysses_size != 4:
        raise CMSGauhTrainingError("AUH v6 training requires exactly four ranks")
    device, backend = legacy.initialise_distributed(distributed)
    from bernini.parallel import init_parallel_state

    init_parallel_state(ulysses_size=4)
    legacy.seed_same_sample(args.seed)
    output = Path(args.output).expanduser().resolve()
    dataset = legacy.ParquetRowStore(args.preprocessed_parquet_dir)
    dataset_summary = legacy.validate_preprocessed_dataset_summary(
        args.dataset_summary,
        dataset,
        allow_incomplete=False,
    )
    try:
        router = motion.ReviewRouter.load(args.routing_jsonl, default_tier="reject")
    except motion.MotionContractError as error:
        raise _translate(error) from error
    eligible_routes = v4._build_eligible_routes(dataset, router)
    _strict_router(args, router, eligible_routes, dataset)

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    try:
        legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    except legacy.TrainingContractError as error:
        raise _translate(error) from error
    base_model = BerniniRendererModel(config)
    base_model.requires_grad_(False)
    base_model.t5_text_encoder.eval()
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    try:
        available_modules = legacy.select_attention_projection_names(base_model)
        target_modules = core.select_cmsg_lora_targets(available_modules)
    except (
        legacy.TrainingContractError,
        core.CrossModeCMSGTrainingError,
    ) as error:
        raise _translate(error) from error
    immutable = _immutable_contract(
        args=args,
        dataset=dataset,
        dataset_summary=dataset_summary,
        router=router,
        eligible_routes=eligible_routes,
        target_modules=target_modules,
        checkpoint=checkpoint,
    )
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
            target_modules=target_modules,
        ),
    )
    model.to(device)
    model.eval()
    renderer = model.get_base_model()
    renderer.t5_text_encoder.eval()
    try:
        named_trainable = legacy.trainable_lora_parameters(model)
        initialization_digest = legacy.synchronize_trainable_parameters(
            named_trainable, source_rank=0
        )
    except legacy.TrainingContractError as error:
        raise _translate(error) from error
    parameter_names = v4._optimizer_parameter_names(named_trainable)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named_trainable],
        lr=LEARNING_RATE,
        weight_decay=args.weight_decay,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)
    vae_mean, vae_std, z_dim = legacy._vae_statistics(checkpoint)
    scheduler_kwargs = legacy.noise_scheduler_kwargs()
    scheduler_kwargs["noise_tmin"] = MINIMUM_TRAINING_SIGMA
    scheduler = NoiseScheduler(**scheduler_kwargs)
    inference_scheduler = UniPCMultistepScheduler.from_pretrained(
        str(checkpoint),
        subfolder="scheduler",
        local_files_only=True,
        flow_shift=sigma_strata.FLOW_SHIFT,
    )
    sigma_strata.audit_runtime_unipc_schedule(inference_scheduler)

    global_step = 0
    attempt_ordinal = 0
    rejected_count = 0
    gate_audit: list[dict[str, Any]] = []
    last_metrics: Optional[dict[str, float]] = None
    last_saved = -1

    def save_current() -> None:
        optimizer_payload = _optimizer_payload(
            optimizer=optimizer,
            global_step=global_step,
            attempt_ordinal=attempt_ordinal,
            rejected_count=rejected_count,
            immutable=immutable,
            parameter_names=parameter_names,
            gate_audit=gate_audit,
        )
        receipt = _build_receipt(
            args=args,
            global_step=global_step,
            attempt_ordinal=attempt_ordinal,
            rejected_count=rejected_count,
            metrics=last_metrics,
            gate_audit=gate_audit,
            dataset=dataset,
            dataset_summary=dataset_summary,
            router=router,
            checkpoint=checkpoint,
            bernini_revision=bernini_revision,
            veomni_revision=veomni_revision,
            distributed=distributed,
            backend=backend,
            target_modules=target_modules,
            named_trainable=named_trainable,
            initialization_digest=initialization_digest,
            transformers_version=transformers_version,
            immutable=immutable,
            optimizer_payload=optimizer_payload,
        )
        _save_checkpoint(
            model=model,
            optimizer_payload=optimizer_payload,
            output=output,
            global_step=global_step,
            receipt=receipt,
            rank=distributed.rank,
        )

    while global_step < args.max_steps:
        selected_stratum = sigma_strata.select_sigma_stratum(global_step)
        teacher_active = spectrum.release_rho(selected_stratum.schedule_index) > 0.0
        accepted: Optional[tuple[ForwardCellResult, int, Any, int, int]] = None
        for attempt_in_step in range(args.max_gate_attempts):
            current_ordinal = attempt_ordinal
            row_index, raw_row, route = v4._next_routed_row(
                dataset, eligible_routes, ordinal=current_ordinal
            )
            attempt_ordinal += 1
            identity = legacy.dataset_identity(raw_row, row_index)
            legacy.assert_identical_row(identity)
            current_seed = legacy.step_seed(args.seed, current_ordinal, row_index)
            legacy.seed_same_sample(current_seed)
            prepared = _prepare_candidate_cpu(
                raw_row=raw_row,
                tokenizer=tokenizer,
                prompt_cleaner=prompt_clean,
                system_prompts=SYSTEM_PROMPTS,
                rope=rope,
                vae_mean=vae_mean,
                vae_std=vae_std,
                z_dim=z_dim,
                scheduler=scheduler,
                noop_instruction=args.noop_instruction,
                negative_prompt=args.negative_prompt,
                process_renderer_sample=process_renderer_sample,
                selected_stratum=selected_stratum,
            )
            moved = _move_candidate_to_device(prepared, device=device)
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            cell: Optional[ForwardCellResult] = None
            gate = None
            rejected = False
            with autocast:
                try:
                    cell = _run_six_forward_cell(
                        renderer=renderer,
                        adapter_controller=model,
                        candidate=moved,
                        step_index=selected_stratum.schedule_index,
                        enforce_frozen_prior_gate=args.enforce_frozen_prior_gate,
                    )
                    gate = cell.gate_preview
                except core.FrozenPriorGateRejected as error:
                    rejected = True
                    gate = getattr(error, "gate_result", None)
            any_rejected = legacy._distributed_boolean(rejected, op="any")
            all_rejected = legacy._distributed_boolean(rejected, op="all")
            if any_rejected != all_rejected:
                raise CMSGauhTrainingError("frozen-prior gate differs across ranks")
            if gate is None:
                raise CMSGauhTrainingError("gate decision lacks auditable metrics")
            if rejected and not teacher_active:
                raise CMSGauhTrainingError(
                    "rho=0 late replay was incorrectly rejected by the teacher gate"
                )
            record = _gate_audit_record(
                gate,
                global_step=global_step,
                attempt_ordinal=current_ordinal,
                attempt_in_step=attempt_in_step,
                row_index=row_index,
                iid=route.iid,
                schedule_index=selected_stratum.schedule_index,
                timestep=selected_stratum.timestep,
                accepted=not rejected,
                teacher_active=teacher_active,
            )
            _assert_gate_record_equal_across_ranks(record)
            gate_audit.append(record)
            if rejected:
                rejected_count += 1
                if distributed.rank == 0:
                    print(json.dumps({"event": "gate_rejected", **record}, sort_keys=True), flush=True)
                continue
            if cell is None:
                raise CMSGauhTrainingError("accepted gate has no differentiable loss")
            accepted = (cell, row_index, route, current_seed, current_ordinal)
            break
        if accepted is None:
            raise CMSGauhTrainingError(
                "frozen-prior gate exhausted bounded attempts without an update: "
                f"step={global_step}, attempts={args.max_gate_attempts}, "
                f"audit_sha256={legacy.object_sha256(gate_audit)}"
            )

        cell, row_index, route, current_seed, current_ordinal = accepted
        finite = bool(torch.isfinite(cell.weighted_loss.detach()).item())
        if not legacy._distributed_boolean(finite, op="all"):
            raise CMSGauhTrainingError("non-finite CMSG loss blocked optimizer update")
        cell.weighted_loss.backward()
        try:
            gradient_norm = legacy.all_reduce_lora_gradients(named_trainable)
        except legacy.TrainingContractError as error:
            raise _translate(error) from error
        torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in named_trainable], args.max_grad_norm
        )
        optimizer.step()
        global_step += 1
        last_metrics = {
            **_loss_metrics(cell),
            "preclip_gradient_norm": float(gradient_norm),
            "sigma_schedule_index": float(selected_stratum.schedule_index),
            "sigma_timestep": float(selected_stratum.timestep),
            "attempt_ordinal": float(current_ordinal),
            "gate_rejected_total": float(rejected_count),
        }
        if distributed.rank == 0:
            print(
                json.dumps(
                    {
                        "event": "optimizer_step",
                        "step": global_step,
                        "row": row_index,
                        "iid": route.iid,
                        "seed": current_seed,
                        **last_metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if args.save_every > 0 and global_step % args.save_every == 0:
            save_current()
            last_saved = global_step

    if last_saved != global_step:
        save_current()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    return 0


__all__ = [
    "CMSGauhTrainingError",
    "FORWARD_CELL_ORDER",
    "METHOD_NAME",
    "PreparedCandidate",
    "T2V_GUIDANCE_SCALE",
    "T2V_SYSTEM_PROMPT",
    "TRAINING_BRIDGE_ENDPOINT",
    "_generator_plain_cfg_clean",
    "_official_t2v_text_fields",
    "_run_six_forward_cell",
    "build_parser",
    "main",
    "validate_cli",
]


if __name__ == "__main__":
    raise SystemExit(main())
