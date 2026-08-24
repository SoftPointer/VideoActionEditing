"""Pure runtime contracts for Bernini DCLR teacher queries.

This module deliberately owns no model, tokenizer, scheduler, or distributed
process group.  It validates and assembles the tensors consumed by the pinned
Bernini-R 1.3B ``GEN_Wanx22.shared_step`` boundary:

* one FP32 physical sigma maps directly to ``t = 1000 * sigma``;
* T2V is one target-only visual sequence;
* MV2V is one combined ``[clean source, noisy target]`` visual sequence;
* correct/decoy MV2V calls differ only in source-prefix content; and
* the denoising energy is an FP32 MSE over the target tail only.

The sequence-parallel helper only verifies already-gathered scalar replicas.
It performs no sum, mean, all-reduce, or other cross-rank reduction: pinned
Bernini gathers the full visual prediction onto every SP rank before return.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import struct
from typing import Any

import torch


SCHEMA_VERSION = "bernini-dclr-runtime-contract-v1"
NUM_TRAIN_TIMESTEPS = 1000
PINNED_INNER_DIM = 1536
PINNED_PATCH_DIM = 64
PINNED_ROPE_DIM = 64
PINNED_TEXT_TOKENS = 512
PINNED_TEXT_DIM = 4096


class DCLRRuntimeContractError(RuntimeError):
    """A DCLR teacher query violates the pinned Bernini runtime contract."""


@dataclass(frozen=True)
class PackedVisualBranch:
    """Visual arguments and target selector for one ``shared_step`` call.

    ``batch_vae_seqlen`` always has one entry because source and target are
    parts of one self-attending Bernini sample.  In particular, MV2V uses
    ``(2N,)`` and never ``(N, N)``.
    """

    mode: str
    noisy_latents: torch.Tensor
    rotary_embs: torch.Tensor
    target_selector: torch.Tensor
    batch_vae_seqlen: tuple[int, ...]
    target_token_count: int
    source_id: int | None
    target_source_id: int

    @property
    def total_token_count(self) -> int:
        return int(self.noisy_latents.shape[1])

    @property
    def source_token_count(self) -> int:
        return self.total_token_count - self.target_token_count


def _require_tensor(value: Any, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise DCLRRuntimeContractError(f"{label} must be a torch.Tensor")
    if value.device.type == "meta":
        raise DCLRRuntimeContractError(f"{label} cannot be a meta tensor")
    return value


def _require_finite_detached_numeric_tensor(
    value: Any,
    *,
    label: str,
    allow_complex: bool = False,
) -> torch.Tensor:
    tensor = _require_tensor(value, label=label)
    is_valid_dtype = tensor.is_floating_point() or (
        allow_complex and tensor.is_complex()
    )
    if not is_valid_dtype:
        raise DCLRRuntimeContractError(
            f"{label} must be a floating-point"
            + (" or complex" if allow_complex else "")
            + " tensor"
        )
    if tensor.requires_grad or tensor.grad_fn is not None:
        raise DCLRRuntimeContractError(
            f"{label} must be detached from the student/teacher graph"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise DCLRRuntimeContractError(f"{label} contains NaN or infinity")
    return tensor


def _tensor_exact(left: Any, right: Any) -> bool:
    return bool(
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and tuple(left.shape) == tuple(right.shape)
        and left.dtype == right.dtype
        and left.device == right.device
        and left.layout == right.layout
        and torch.equal(left, right)
    )


def _require_visual_tokens(value: Any, *, label: str) -> torch.Tensor:
    tokens = _require_finite_detached_numeric_tensor(value, label=label)
    if (
        tokens.ndim != 3
        or int(tokens.shape[0]) != 1
        or int(tokens.shape[1]) <= 0
        or int(tokens.shape[2]) != PINNED_INNER_DIM
    ):
        raise DCLRRuntimeContractError(
            f"{label} must be patch-embedded [1,N,{PINNED_INNER_DIM}]"
        )
    return tokens


def _require_rotary(
    value: Any,
    *,
    label: str,
    token_count: int,
) -> torch.Tensor:
    rotary = _require_finite_detached_numeric_tensor(
        value, label=label, allow_complex=True
    )
    if tuple(rotary.shape[:2]) != (1, 1) or rotary.ndim != 4:
        raise DCLRRuntimeContractError(
            f"{label} must have pinned [1,1,N,R] layout"
        )
    if (
        int(rotary.shape[2]) != token_count
        or int(rotary.shape[3]) != PINNED_ROPE_DIM
        or rotary.dtype != torch.complex128
    ):
        raise DCLRRuntimeContractError(
            f"{label} must be pinned complex128 [1,1,{token_count},{PINNED_ROPE_DIM}]"
        )
    return rotary


def _require_same_representation(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    left_label: str,
    right_label: str,
) -> None:
    if (
        left.dtype != right.dtype
        or left.device != right.device
        or left.layout != right.layout
    ):
        raise DCLRRuntimeContractError(
            f"{left_label} and {right_label} representation differs"
        )


def _require_fp32_timestep(value: Any, *, label: str) -> torch.Tensor:
    timestep = _require_finite_detached_numeric_tensor(value, label=label)
    if timestep.dtype != torch.float32 or tuple(timestep.shape) != (1,):
        raise DCLRRuntimeContractError(
            f"{label} must be an exact FP32 [1] model timestep"
        )
    if bool(((timestep < 0.0) | (timestep > NUM_TRAIN_TIMESTEPS)).any().item()):
        raise DCLRRuntimeContractError(
            f"{label} must remain in [0, {NUM_TRAIN_TIMESTEPS}]"
        )
    return timestep


def fp32_sigma_to_timestep(sigma: torch.Tensor) -> torch.Tensor:
    """Map a physical FP32 sigma directly to Bernini model time.

    The pinned scheduler uses ``timesteps = sigmas * 1000``.  Flow shift
    changes which sigma values training samples; it does not define a
    mode-specific inverse map.  The helper consequently accepts no shift or
    task argument and never snaps to a scheduler grid.
    """

    sigma_tensor = _require_finite_detached_numeric_tensor(sigma, label="sigma")
    if sigma_tensor.dtype != torch.float32 or sigma_tensor.numel() == 0:
        raise DCLRRuntimeContractError("sigma must be a nonempty FP32 tensor")
    if bool(((sigma_tensor < 0.0) | (sigma_tensor > 1.0)).any().item()):
        raise DCLRRuntimeContractError("sigma must remain in [0, 1]")
    timestep = sigma_tensor * sigma_tensor.new_tensor(
        float(NUM_TRAIN_TIMESTEPS)
    )
    if timestep.dtype != torch.float32 or tuple(timestep.shape) != tuple(
        sigma_tensor.shape
    ):
        raise DCLRRuntimeContractError("sigma-to-timestep mapping changed contract")
    return timestep


def build_t2v_target_branch(
    target_tokens: torch.Tensor,
    target_rotary: torch.Tensor,
    *,
    target_source_id: int = 0,
) -> PackedVisualBranch:
    """Build target-only T2V geometry without copying the target tensors."""

    if target_source_id != 0:
        raise DCLRRuntimeContractError("Bernini target source_id must equal 0")
    tokens = _require_visual_tokens(target_tokens, label="target_tokens")
    token_count = int(tokens.shape[1])
    rotary = _require_rotary(
        target_rotary, label="target_rotary", token_count=token_count
    )
    if rotary.device != tokens.device:
        raise DCLRRuntimeContractError(
            "target rotary device differs from target tokens"
        )
    selector = torch.ones(token_count, dtype=torch.bool, device=tokens.device)
    result = PackedVisualBranch(
        mode="t2v",
        noisy_latents=tokens,
        rotary_embs=rotary,
        target_selector=selector,
        batch_vae_seqlen=(token_count,),
        target_token_count=token_count,
        source_id=None,
        target_source_id=target_source_id,
    )
    _validate_branch(result, label="T2V branch")
    return result


def build_mv2v_target_tail_branch(
    source_tokens: torch.Tensor,
    source_rotary: torch.Tensor,
    target_tokens: torch.Tensor,
    target_rotary: torch.Tensor,
    *,
    source_id: int = 1,
    target_source_id: int = 0,
) -> PackedVisualBranch:
    """Build one MV2V sample as the packed sequence ``[source, target]``."""

    if source_id != 1 or target_source_id != 0:
        raise DCLRRuntimeContractError(
            "one-source MV2V requires source_id=1 and target source_id=0"
        )
    source = _require_visual_tokens(source_tokens, label="source_tokens")
    target = _require_visual_tokens(target_tokens, label="target_tokens")
    if int(source.shape[1]) != int(target.shape[1]):
        raise DCLRRuntimeContractError(
            "DCLR MV2V requires equal source and target token counts"
        )
    _require_same_representation(
        source,
        target,
        left_label="source_tokens",
        right_label="target_tokens",
    )
    token_count = int(target.shape[1])
    source_rope = _require_rotary(
        source_rotary, label="source_rotary", token_count=token_count
    )
    target_rope = _require_rotary(
        target_rotary, label="target_rotary", token_count=token_count
    )
    _require_same_representation(
        source_rope,
        target_rope,
        left_label="source_rotary",
        right_label="target_rotary",
    )
    if source_rope.device != source.device:
        raise DCLRRuntimeContractError(
            "source/target rotary device differs from visual tokens"
        )
    visual_tokens = torch.cat((source, target), dim=1)
    visual_rotary = torch.cat((source_rope, target_rope), dim=2)
    selector = torch.cat(
        (
            torch.zeros(token_count, dtype=torch.bool, device=target.device),
            torch.ones(token_count, dtype=torch.bool, device=target.device),
        ),
        dim=0,
    )
    result = PackedVisualBranch(
        mode="mv2v",
        noisy_latents=visual_tokens,
        rotary_embs=visual_rotary,
        target_selector=selector,
        batch_vae_seqlen=(2 * token_count,),
        target_token_count=token_count,
        source_id=source_id,
        target_source_id=target_source_id,
    )
    _validate_branch(result, label="MV2V branch")
    if not _tensor_exact(result.noisy_latents[:, token_count:, :], target):
        raise DCLRRuntimeContractError("MV2V target tail differs after packing")
    if not _tensor_exact(result.rotary_embs[:, :, token_count:, :], target_rope):
        raise DCLRRuntimeContractError("MV2V target rotary differs after packing")
    return result


def _validate_branch(branch: Any, *, label: str) -> PackedVisualBranch:
    if not isinstance(branch, PackedVisualBranch):
        raise DCLRRuntimeContractError(
            f"{label} must be a PackedVisualBranch"
        )
    visual = _require_visual_tokens(
        branch.noisy_latents, label=f"{label}.noisy_latents"
    )
    total = int(visual.shape[1])
    rotary = _require_rotary(
        branch.rotary_embs,
        label=f"{label}.rotary_embs",
        token_count=total,
    )
    if rotary.device != visual.device:
        raise DCLRRuntimeContractError(f"{label} rotary/token device differs")
    selector = _require_tensor(
        branch.target_selector, label=f"{label}.target_selector"
    )
    if (
        selector.dtype != torch.bool
        or selector.ndim != 1
        or int(selector.numel()) != total
        or selector.device != visual.device
    ):
        raise DCLRRuntimeContractError(
            f"{label} target selector must be bool [{total}] on token device"
        )
    target_count = int(selector.sum().item())
    if target_count <= 0 or target_count != branch.target_token_count:
        raise DCLRRuntimeContractError(f"{label} target count differs")
    boundary = total - target_count
    if bool(selector[:boundary].any().item()) or not bool(
        selector[boundary:].all().item()
    ):
        raise DCLRRuntimeContractError(
            f"{label} selector must choose one contiguous target tail"
        )
    if tuple(branch.batch_vae_seqlen) != (total,):
        raise DCLRRuntimeContractError(
            f"{label} batch_vae_seqlen must contain one combined length"
        )
    if branch.mode == "t2v":
        if (
            boundary != 0
            or branch.source_id is not None
            or branch.target_source_id != 0
        ):
            raise DCLRRuntimeContractError(
                f"{label} is not target-only source_id=0 T2V"
            )
    elif branch.mode == "mv2v":
        if (
            boundary != target_count
            or branch.source_id != 1
            or branch.target_source_id != 0
        ):
            raise DCLRRuntimeContractError(
                f"{label} must be equal source-N plus target-N MV2V"
            )
    else:
        raise DCLRRuntimeContractError(f"{label} has unsupported mode")
    return branch


def shared_step_visual_kwargs(branch: PackedVisualBranch) -> dict[str, Any]:
    """Return the visual subset of the pinned ``shared_step`` keyword args."""

    checked = _validate_branch(branch, label="shared_step branch")
    return {
        "noisy_latents": checked.noisy_latents,
        "rotary_embs": checked.rotary_embs,
        "batch_vae_seqlen": list(checked.batch_vae_seqlen),
    }


def validate_cross_mode_target_tail(
    t2v: PackedVisualBranch,
    mv2v: PackedVisualBranch,
    *,
    t2v_timestep: torch.Tensor,
    mv2v_timestep: torch.Tensor,
) -> int:
    """Prove that T2V and MV2V query the same target state and time."""

    t2v_branch = _validate_branch(t2v, label="T2V branch")
    mv2v_branch = _validate_branch(mv2v, label="MV2V branch")
    if t2v_branch.mode != "t2v" or mv2v_branch.mode != "mv2v":
        raise DCLRRuntimeContractError(
            "cross-mode parity requires T2V then MV2V branches"
        )
    if t2v_branch.target_token_count != mv2v_branch.target_token_count:
        raise DCLRRuntimeContractError("cross-mode target token counts differ")
    target_count = t2v_branch.target_token_count
    if not _tensor_exact(
        t2v_branch.noisy_latents,
        mv2v_branch.noisy_latents[:, -target_count:, :],
    ):
        raise DCLRRuntimeContractError(
            "T2V state differs from the MV2V target tail"
        )
    if not _tensor_exact(
        t2v_branch.rotary_embs,
        mv2v_branch.rotary_embs[:, :, -target_count:, :],
    ):
        raise DCLRRuntimeContractError(
            "T2V rotary differs from the MV2V target tail"
        )
    t2v_time = _require_fp32_timestep(t2v_timestep, label="T2V timestep")
    mv2v_time = _require_fp32_timestep(mv2v_timestep, label="MV2V timestep")
    if not _tensor_exact(t2v_time, mv2v_time):
        raise DCLRRuntimeContractError(
            "T2V and MV2V must use one exact physical-sigma timestep"
        )
    return target_count


def _require_condition(value: Any, *, label: str) -> torch.Tensor:
    condition = _require_finite_detached_numeric_tensor(value, label=label)
    if tuple(condition.shape) != (
        1,
        PINNED_TEXT_TOKENS,
        PINNED_TEXT_DIM,
    ):
        raise DCLRRuntimeContractError(
            f"{label} must be [1,{PINNED_TEXT_TOKENS},{PINNED_TEXT_DIM}]"
        )
    return condition


def _normalize_text_lengths(value: Any, *, label: str) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        if value.device.type == "meta" or value.numel() == 0:
            raise DCLRRuntimeContractError(f"{label} is empty or meta")
        if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
            raise DCLRRuntimeContractError(f"{label} must contain integers")
        values = tuple(int(item) for item in value.reshape(-1).tolist())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if any(type(item) is not int for item in value):
            raise DCLRRuntimeContractError(f"{label} must contain plain integers")
        values = tuple(value)
    else:
        raise DCLRRuntimeContractError(
            f"{label} must be an integer sequence or tensor"
        )
    if values != (PINNED_TEXT_TOKENS,):
        raise DCLRRuntimeContractError(
            f"{label} must equal ({PINNED_TEXT_TOKENS},)"
        )
    return values


def validate_correct_decoy_same_state(
    correct: PackedVisualBranch,
    decoy: PackedVisualBranch,
    *,
    correct_timestep: torch.Tensor,
    decoy_timestep: torch.Tensor,
    correct_cond_embeds: torch.Tensor,
    decoy_cond_embeds: torch.Tensor,
    correct_text_seqlen: Sequence[int] | torch.Tensor,
    decoy_text_seqlen: Sequence[int] | torch.Tensor,
) -> int:
    """Validate a single-variable correct-source/decoy-source comparison.

    Both calls must have the same action text, time, target tail, mask, token
    geometry, and complete rotary tensor.  The source-prefix token values must
    differ; that prefix is the only admitted model-input intervention.
    """

    correct_branch = _validate_branch(correct, label="correct-source branch")
    decoy_branch = _validate_branch(decoy, label="decoy-source branch")
    if correct_branch.mode != "mv2v" or decoy_branch.mode != "mv2v":
        raise DCLRRuntimeContractError(
            "correct/decoy comparison requires two MV2V branches"
        )
    if (
        correct_branch.target_token_count != decoy_branch.target_token_count
        or correct_branch.batch_vae_seqlen != decoy_branch.batch_vae_seqlen
        or not _tensor_exact(
            correct_branch.target_selector, decoy_branch.target_selector
        )
    ):
        raise DCLRRuntimeContractError(
            "correct/decoy MV2V target geometry differs"
        )
    target_count = correct_branch.target_token_count
    if not _tensor_exact(
        correct_branch.noisy_latents[:, -target_count:, :],
        decoy_branch.noisy_latents[:, -target_count:, :],
    ):
        raise DCLRRuntimeContractError(
            "correct/decoy noisy target tails differ"
        )
    if not _tensor_exact(correct_branch.rotary_embs, decoy_branch.rotary_embs):
        raise DCLRRuntimeContractError(
            "correct/decoy rotary differs; geometry or source_id changed"
        )
    correct_source = correct_branch.noisy_latents[:, :-target_count, :]
    decoy_source = decoy_branch.noisy_latents[:, :-target_count, :]
    if _tensor_exact(correct_source, decoy_source):
        raise DCLRRuntimeContractError(
            "decoy source prefix is identical to the correct source"
        )

    correct_time = _require_fp32_timestep(
        correct_timestep, label="correct-source timestep"
    )
    decoy_time = _require_fp32_timestep(
        decoy_timestep, label="decoy-source timestep"
    )
    if not _tensor_exact(correct_time, decoy_time):
        raise DCLRRuntimeContractError("correct/decoy timesteps differ")

    correct_condition = _require_condition(
        correct_cond_embeds, label="correct-source condition"
    )
    decoy_condition = _require_condition(
        decoy_cond_embeds, label="decoy-source condition"
    )
    if not _tensor_exact(correct_condition, decoy_condition):
        raise DCLRRuntimeContractError(
            "correct/decoy action condition differs"
        )
    correct_lengths = _normalize_text_lengths(
        correct_text_seqlen, label="correct-source text lengths"
    )
    decoy_lengths = _normalize_text_lengths(
        decoy_text_seqlen, label="decoy-source text lengths"
    )
    if correct_lengths != decoy_lengths:
        raise DCLRRuntimeContractError(
            "correct/decoy text sequence lengths differ"
        )
    return target_count


def target_only_fp32_mse(
    full_prediction: torch.Tensor,
    true_target_velocity: torch.Tensor,
    target_selector: torch.Tensor,
) -> torch.Tensor:
    """Compute one detached FP32 velocity MSE over a contiguous target tail."""

    prediction = _require_finite_detached_numeric_tensor(
        full_prediction, label="full_prediction"
    )
    if (
        prediction.ndim != 3
        or int(prediction.shape[0]) != 1
        or int(prediction.shape[1]) <= 0
        or int(prediction.shape[2]) != PINNED_PATCH_DIM
    ):
        raise DCLRRuntimeContractError(
            f"full_prediction must be [1,total,{PINNED_PATCH_DIM}]"
        )
    target = _require_finite_detached_numeric_tensor(
        true_target_velocity, label="true_target_velocity"
    )
    if target.dtype != torch.float32 or (
        target.ndim != 3
        or int(target.shape[0]) != 1
        or int(target.shape[1]) <= 0
        or int(target.shape[2]) != PINNED_PATCH_DIM
    ):
        raise DCLRRuntimeContractError(
            f"true_target_velocity must be FP32 [1,N,{PINNED_PATCH_DIM}]"
        )
    if target.device != prediction.device:
        raise DCLRRuntimeContractError(
            "prediction and target velocity devices differ"
        )
    selector = _require_tensor(target_selector, label="target_selector")
    total = int(prediction.shape[1])
    if (
        selector.dtype != torch.bool
        or selector.ndim != 1
        or int(selector.numel()) != total
        or selector.device != prediction.device
    ):
        raise DCLRRuntimeContractError(
            f"target_selector must be bool [{total}] on prediction device"
        )
    target_count = int(selector.sum().item())
    boundary = total - target_count
    if (
        target_count != int(target.shape[1])
        or target_count <= 0
        or bool(selector[:boundary].any().item())
        or not bool(selector[boundary:].all().item())
    ):
        raise DCLRRuntimeContractError(
            "target_selector must select exactly one contiguous target tail"
        )
    selected_prediction = prediction[:, selector, :].to(dtype=torch.float32)
    squared_error = (selected_prediction - target).square()
    energy = squared_error.mean()
    if energy.dtype != torch.float32 or energy.ndim != 0:
        raise DCLRRuntimeContractError("target energy did not remain FP32 scalar")
    if not bool(torch.isfinite(energy).item()):
        raise DCLRRuntimeContractError("target energy is not finite")
    return energy


def _fp32_scalar_bits(value: torch.Tensor, *, label: str) -> bytes:
    scalar = _require_finite_detached_numeric_tensor(value, label=label)
    if scalar.dtype != torch.float32 or scalar.numel() != 1:
        raise DCLRRuntimeContractError(f"{label} must be one FP32 scalar")
    return struct.pack("!f", float(scalar.detach().cpu().item()))


def assert_sp_replicated_scalar(
    local_scalar: torch.Tensor,
    gathered_scalars: Sequence[torch.Tensor] | torch.Tensor,
    *,
    expected_world_size: int | None = None,
) -> torch.Tensor:
    """Assert exact SP scalar replication and return ``local_scalar`` itself.

    The caller supplies values obtained with an all-gather-like operation.
    This function intentionally accepts no reducer and performs no arithmetic
    across ranks.  Returning the original object, rather than a gathered sum
    or average, makes the no-reduction contract directly testable.
    """

    local_bits = _fp32_scalar_bits(local_scalar, label="local SP scalar")
    if isinstance(gathered_scalars, torch.Tensor):
        if gathered_scalars.ndim == 1:
            replicas = tuple(gathered_scalars.unbind(dim=0))
        elif gathered_scalars.ndim == 2 and int(gathered_scalars.shape[1]) == 1:
            replicas = tuple(item.reshape(()) for item in gathered_scalars.unbind(0))
        else:
            raise DCLRRuntimeContractError(
                "gathered SP scalars must have [R] or [R,1] tensor layout"
            )
    elif isinstance(gathered_scalars, Sequence) and not isinstance(
        gathered_scalars, (str, bytes)
    ):
        replicas = tuple(gathered_scalars)
    else:
        raise DCLRRuntimeContractError(
            "gathered SP scalars must be a tensor or tensor sequence"
        )
    if not replicas:
        raise DCLRRuntimeContractError("gathered SP scalars cannot be empty")
    if expected_world_size is not None:
        if type(expected_world_size) is not int or expected_world_size <= 0:
            raise DCLRRuntimeContractError(
                "expected_world_size must be a positive plain integer"
            )
        if len(replicas) != expected_world_size:
            raise DCLRRuntimeContractError(
                "gathered SP scalar count differs from expected world size"
            )
    for rank, replica in enumerate(replicas):
        if _fp32_scalar_bits(replica, label=f"SP rank {rank} scalar") != local_bits:
            raise DCLRRuntimeContractError(
                f"SP rank {rank} scalar is not an exact replica; reduction is forbidden"
            )
    return local_scalar


__all__ = [
    "DCLRRuntimeContractError",
    "NUM_TRAIN_TIMESTEPS",
    "PINNED_INNER_DIM",
    "PINNED_PATCH_DIM",
    "PINNED_ROPE_DIM",
    "PINNED_TEXT_DIM",
    "PINNED_TEXT_TOKENS",
    "PackedVisualBranch",
    "SCHEMA_VERSION",
    "assert_sp_replicated_scalar",
    "build_mv2v_target_tail_branch",
    "build_t2v_target_branch",
    "fp32_sigma_to_timestep",
    "shared_step_visual_kwargs",
    "target_only_fp32_mse",
    "validate_correct_decoy_same_state",
    "validate_cross_mode_target_tail",
]
