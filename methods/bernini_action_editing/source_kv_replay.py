#!/usr/bin/env python3
"""Fail-closed source K/V capture and replay for Bernini-R self-attention.

The V9 carrier is deliberately small: a frozen no-op *source-only* forward
captures the source keys and values produced by the official Bernini
``WanAttnProcessor2_0._project_qkv``.  Later ``[source, target]`` forwards
replace only their source K/V prefix with that detached carrier.  Queries and
the target K/V suffix stay on the current (and potentially adapted) route.

``_project_qkv`` is the important integration boundary.  In official Bernini
it performs projection, q/k normalisation, Ulysses gather-sequence / scatter-
heads, and finally RoPE on q/k.  Consequently this module stores rank-local
head shards *after RoPE* and executes the rest of the official varlen path,
including ``gather_heads_scatter_seq`` on the output.

This module does not infer whether LoRA is enabled or whether a prompt is
negative, no-op, or action-bearing.  The outer runner must supply one of the
audited branch tags through :func:`source_kv_replay_invocation`.  It also does
not implement phase shuffling: any such transform belongs to source content
before q/k projection and RoPE.  Shuffling a post-RoPE K cache would attach the
wrong positional phase and is forbidden by the public contract.

Non-reentrant gradient checkpointing must be configured with
``context_fn=source_kv_replay_checkpoint_context_fn``.  Each checkpoint then
captures its own immutable branch/step/rank invocation and restores that exact
snapshot during backward recomputation.  Source capture itself remains a
frozen no-grad forward and is deliberately not checkpointable.

PyTorch and Bernini imports are lazy so the tensor core and install/restore
handle can be tested without a Bernini checkout.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Iterator, Optional, Sequence


EXPECTED_BLOCK_COUNT = 30
BLOCK_SELECTIONS = ("all", "mid", "late")
MAIN_BLOCK_SELECTION = "all"
MID_ABLATION_START_1P3B = 7
MID_ABLATION_STOP_1P3B = 23
CAPTURE_MODE = "capture"
REPLAY_MODE = "replay"
INVOCATION_MODES = (CAPTURE_MODE, REPLAY_MODE)
CAPTURE_BRANCH_TAG = "frozen_noop_carrier"
REPLAY_BRANCH_TAGS = (
    "frozen_negative",
    "frozen_noop",
    "frozen_action",
    "adapted_noop",
    "adapted_action",
)
CORE_SCHEMA = "bernini-source-kv-replay-v9-core-v1"
EAGER_EXECUTION = "eager"
CHECKPOINT_FORWARD = "checkpoint_forward"
CHECKPOINT_RECOMPUTE = "checkpoint_recompute"
EXECUTION_PHASES = (
    EAGER_EXECUTION,
    CHECKPOINT_FORWARD,
    CHECKPOINT_RECOMPUTE,
)


class SourceKVReplayContractError(RuntimeError):
    """Raised instead of silently using an ambiguous or stale carrier."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SourceKVReplayContractError(
            f"source K/V contract is not canonical JSON: {error}"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _exact_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise SourceKVReplayContractError(f"{label} must be an integer")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "numel"):
        if int(value.numel()) != 1:
            raise SourceKVReplayContractError(f"{label} must be scalar")
        value = value.item()
    try:
        integer = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise SourceKVReplayContractError(f"{label} must be an integer") from error
    if not math.isfinite(numeric) or numeric != float(integer) or integer < 0:
        raise SourceKVReplayContractError(
            f"{label} must be an exact non-negative finite integer"
        )
    return integer


def _exact_positive_int(value: Any, *, label: str) -> int:
    integer = _exact_nonnegative_int(value, label=label)
    if integer <= 0:
        raise SourceKVReplayContractError(f"{label} must be positive")
    return integer


def _as_int_tuple(value: Any, *, label: str) -> tuple[int, ...]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SourceKVReplayContractError(f"{label} must be a sequence")
    return tuple(
        _exact_nonnegative_int(item, label=f"{label} item") for item in value
    )


def resolve_block_indices(num_blocks: int, selection: str) -> tuple[int, ...]:
    """Resolve the main all-block scope and explicitly named ablations.

    Bernini-R 1.3B's ``mid`` ablation is the broad central window 7..22,
    inclusive.  For a non-pinned block count the same definition is the
    central half (floor one-quarter margins); installed Bernini models are
    independently required to contain exactly 30 blocks.
    """

    count = _exact_positive_int(num_blocks, label="transformer block count")
    if count < 3:
        raise SourceKVReplayContractError("transformer must have at least 3 blocks")
    if selection not in BLOCK_SELECTIONS:
        raise SourceKVReplayContractError(
            f"block selection must be one of {BLOCK_SELECTIONS}, got {selection!r}"
        )
    second_cut = (2 * count) // 3
    if selection == "all":
        indices = tuple(range(count))
    elif selection == "mid":
        margin = count // 4
        indices = tuple(range(margin, count - margin))
    else:
        indices = tuple(range(second_cut, count))
    if not indices:
        raise SourceKVReplayContractError("block selection resolved to no blocks")
    return indices


def phase_shuffle_contract() -> dict[str, Any]:
    """Return the only positional-phase policy compatible with this cache."""

    return {
        "implemented_by_this_module": False,
        "post_rope_cache_shuffle_allowed": False,
        "required_stage_if_enabled": "pre_project_qkv_pre_rope_source_content",
        "required_operation": (
            "shuffle source content tokens, then run official _project_qkv so "
            "canonical source positions receive fresh RoPE"
        ),
        "forbidden_operation": "permute or roll captured post-RoPE source keys",
        "reason": "post-RoPE keys already bind content to positional phase",
    }


def source_kv_replay_contract(
    *, selection: str, num_blocks: int = EXPECTED_BLOCK_COUNT
) -> dict[str, Any]:
    indices = resolve_block_indices(num_blocks, selection)
    value: dict[str, Any] = {
        "schema_version": CORE_SCHEMA,
        "status": "isolated_tensor_core_not_integrated_gpu_runner",
        "attention": "self_attention_attn1_only",
        "capture_branch_tag": CAPTURE_BRANCH_TAG,
        "replay_branch_tags": list(REPLAY_BRANCH_TAGS),
        "branch_tag_owner": "outer_runner_explicit_no_processor_inference",
        "capture_layout": "one_source_only_sequence",
        "replay_layout": "one_equal_length_[source,target]_sequence",
        "capture_point": (
            "official__project_qkv_output_after_projection_qk_norm_"
            "ulysses_gather_seq_scatter_heads_and_qk_rope"
        ),
        "captured_tensors": ["source_key_post_rope", "source_value"],
        "cache_autograd": "detached_cloned",
        "rotary_embedding": {
            "required": True,
            "required_layout": "[1,gathered_sequence,1,head_dim/2]_complex",
            "cache_key_position_state": "post_rope",
            "none_allowed": False,
        },
        "replay_operation": (
            "replace_source_kv_prefix_keep_current_queries_and_target_kv_suffix"
        ),
        "varlen_attention_path": "official_full_bidirectional_self_attention",
        "ulysses_output_inverse": "official_gather_heads_scatter_seq",
        "cache_identity": [
            "generation",
            "step_index",
            "timestep_token",
            "block_index",
            "ulysses_rank",
            "ulysses_size",
            "shape",
            "dtype",
            "device",
        ],
        "ordinary_attention_fallback": False,
        "gradient_checkpointing": {
            "supported_mode": "torch_non_reentrant_context_fn",
            "context_fn": "source_kv_replay_checkpoint_context_fn",
            "snapshot_fields": [
                "cache_bank",
                "mode",
                "branch_tag",
                "generation",
                "step_index",
                "timestep_token",
                "rank",
                "ulysses_size",
            ],
            "forward_policy": "assert_exact_active_invocation",
            "recompute_policy": "rebind_snapshot_after_cache_freshness_check",
            "capture_checkpointing_allowed": False,
        },
        "phase_shuffle": phase_shuffle_contract(),
        "block_selection": selection,
        "block_indices": list(indices),
        "scope_role": (
            "main_all_30"
            if selection == MAIN_BLOCK_SELECTION and num_blocks == EXPECTED_BLOCK_COUNT
            else "main_all_blocks"
            if selection == MAIN_BLOCK_SELECTION
            else "ablation"
        ),
        "pinned_mid_ablation_indices": list(
            range(MID_ABLATION_START_1P3B, MID_ABLATION_STOP_1P3B)
        ),
        "num_transformer_blocks": num_blocks,
    }
    value["contract_digest"] = _object_sha256(value)
    return value


def validate_source_only_layout(
    *,
    gathered_sequence_length: int,
    batch_image_vae_seqlen: Any,
    cu_seqlens_q_cache: Any,
    max_seqlen_q_cache: Any,
    origin_hidden_states_seq_len: Any,
) -> int:
    """Prove the carrier forward contains one source sequence and no target."""

    length = _exact_positive_int(
        gathered_sequence_length, label="gathered sequence length"
    )
    lengths = _as_int_tuple(batch_image_vae_seqlen, label="batch_image_vae_seqlen")
    if lengths != (length,):
        raise SourceKVReplayContractError(
            "capture requires batch=1 with one source-only sequence: "
            f"lengths={lengths!r}, gathered={length}"
        )
    cu = _as_int_tuple(cu_seqlens_q_cache, label="cu_seqlens_q_cache")
    if cu != (0, length):
        raise SourceKVReplayContractError(
            f"capture cu_seqlens must be exactly (0,{length}), got {cu!r}"
        )
    maximum = _exact_positive_int(max_seqlen_q_cache, label="max_seqlen_q_cache")
    if maximum != length:
        raise SourceKVReplayContractError(
            f"capture max sequence length {maximum} differs from {length}"
        )
    if origin_hidden_states_seq_len is not None:
        origin = _exact_positive_int(
            origin_hidden_states_seq_len, label="origin_hidden_states_seq_len"
        )
        if origin != length:
            raise SourceKVReplayContractError(
                f"capture Ulysses origin length {origin} differs from {length}"
            )
    return length


def validate_equal_pair_layout(
    *,
    gathered_sequence_length: int,
    batch_image_vae_seqlen: Any,
    cu_seqlens_q_cache: Any,
    max_seqlen_q_cache: Any,
    origin_hidden_states_seq_len: Any,
) -> int:
    """Prove one exact equal-length ``[source, target]`` sequence."""

    total = _exact_positive_int(
        gathered_sequence_length, label="gathered sequence length"
    )
    if total % 2:
        raise SourceKVReplayContractError(
            "replay requires a positive even full sequence"
        )
    lengths = _as_int_tuple(batch_image_vae_seqlen, label="batch_image_vae_seqlen")
    if lengths != (total,):
        raise SourceKVReplayContractError(
            "replay requires batch=1 with one full source+target sequence: "
            f"lengths={lengths!r}, gathered={total}"
        )
    cu = _as_int_tuple(cu_seqlens_q_cache, label="cu_seqlens_q_cache")
    if cu != (0, total):
        raise SourceKVReplayContractError(
            f"replay cu_seqlens must be exactly (0,{total}), got {cu!r}"
        )
    maximum = _exact_positive_int(max_seqlen_q_cache, label="max_seqlen_q_cache")
    if maximum != total:
        raise SourceKVReplayContractError(
            f"replay max sequence length {maximum} differs from {total}"
        )
    if origin_hidden_states_seq_len is not None:
        origin = _exact_positive_int(
            origin_hidden_states_seq_len, label="origin_hidden_states_seq_len"
        )
        if origin != total:
            raise SourceKVReplayContractError(
                f"replay Ulysses origin length {origin} differs from {total}"
            )
    boundary = total // 2
    if boundary <= 0:
        raise SourceKVReplayContractError("source/target sequence cannot be empty")
    return boundary


@dataclass(frozen=True)
class ReplayInvocation:
    cache_bank: Any
    mode: str
    branch_tag: str
    generation: int
    step_index: int
    timestep_token: str
    rank: int
    ulysses_size: int

    @property
    def identity(self) -> tuple[int, int, str, int, int]:
        return (
            self.generation,
            self.step_index,
            self.timestep_token,
            self.rank,
            self.ulysses_size,
        )


@dataclass(frozen=True)
class CheckpointInvocationBinding:
    """One immutable invocation snapshot bound to a checkpoint execution."""

    invocation: ReplayInvocation
    phase: str


_CURRENT_INVOCATION: ContextVar[Optional[ReplayInvocation]] = ContextVar(
    "bernini_source_kv_replay_invocation", default=None
)
_CURRENT_CHECKPOINT_BINDING: ContextVar[Optional[CheckpointInvocationBinding]] = (
    ContextVar("bernini_source_kv_checkpoint_binding", default=None)
)


def _same_invocation(left: ReplayInvocation, right: ReplayInvocation) -> bool:
    return (
        left.cache_bank is right.cache_bank
        and left.mode == right.mode
        and left.branch_tag == right.branch_tag
        and left.generation == right.generation
        and left.step_index == right.step_index
        and left.timestep_token == right.timestep_token
        and left.rank == right.rank
        and left.ulysses_size == right.ulysses_size
    )


def _snapshot_invocation(invocation: ReplayInvocation) -> ReplayInvocation:
    """Copy every invocation field while retaining the exact rank-local bank."""

    snapshot = ReplayInvocation(
        cache_bank=invocation.cache_bank,
        mode=invocation.mode,
        branch_tag=invocation.branch_tag,
        generation=invocation.generation,
        step_index=invocation.step_index,
        timestep_token=invocation.timestep_token,
        rank=invocation.rank,
        ulysses_size=invocation.ulysses_size,
    )
    _validate_invocation(snapshot)
    return snapshot


def _current_execution_phase(invocation: ReplayInvocation) -> str:
    binding = _CURRENT_CHECKPOINT_BINDING.get()
    if binding is None:
        return EAGER_EXECUTION
    if not _same_invocation(binding.invocation, invocation):
        raise SourceKVReplayContractError(
            "checkpoint binding differs from active source K/V invocation"
        )
    if binding.phase not in (CHECKPOINT_FORWARD, CHECKPOINT_RECOMPUTE):
        raise SourceKVReplayContractError("checkpoint execution phase is invalid")
    return binding.phase


def current_execution_phase(invocation: ReplayInvocation) -> str:
    """Public read-only phase query shared by source-carrier extensions."""

    return _current_execution_phase(invocation)


def _validate_invocation(invocation: ReplayInvocation) -> None:
    if invocation.mode not in INVOCATION_MODES:
        raise SourceKVReplayContractError(
            f"mode must be one of {INVOCATION_MODES}, got {invocation.mode!r}"
        )
    if invocation.mode == CAPTURE_MODE:
        if invocation.branch_tag != CAPTURE_BRANCH_TAG:
            raise SourceKVReplayContractError(
                f"capture branch must be {CAPTURE_BRANCH_TAG!r}"
            )
    elif invocation.branch_tag not in REPLAY_BRANCH_TAGS:
        raise SourceKVReplayContractError(
            f"replay branch must be one of {REPLAY_BRANCH_TAGS}, "
            f"got {invocation.branch_tag!r}"
        )
    _exact_nonnegative_int(invocation.generation, label="generation")
    _exact_nonnegative_int(invocation.step_index, label="step index")
    if (
        type(invocation.timestep_token) is not str
        or not invocation.timestep_token
        or invocation.timestep_token.strip() != invocation.timestep_token
    ):
        raise SourceKVReplayContractError(
            "timestep_token must be a non-empty canonical string without edge whitespace"
        )
    rank = _exact_nonnegative_int(invocation.rank, label="Ulysses rank")
    size = _exact_positive_int(invocation.ulysses_size, label="Ulysses size")
    if rank >= size:
        raise SourceKVReplayContractError(
            f"Ulysses rank {rank} is outside size {size}"
        )


def current_source_kv_invocation() -> ReplayInvocation:
    invocation = _CURRENT_INVOCATION.get()
    if invocation is None:
        raise SourceKVReplayContractError(
            "source K/V processor called outside source_kv_replay_invocation"
        )
    return invocation


@dataclass(frozen=True)
class CapturedSourceKV:
    generation: int
    step_index: int
    timestep_token: str
    block_index: int
    rank: int
    ulysses_size: int
    source_tokens: int
    local_heads: int
    head_dim: int
    key: Any
    value: Any


class SourceKVCacheBank:
    """One-step, rank-local, detached K/V bank for selected attn1 blocks."""

    def __init__(self, selected_block_indices: Sequence[int]) -> None:
        if isinstance(selected_block_indices, (str, bytes)):
            raise SourceKVReplayContractError("selected block indices must be a sequence")
        indices = tuple(
            _exact_nonnegative_int(value, label="selected block index")
            for value in selected_block_indices
        )
        if not indices or len(set(indices)) != len(indices):
            raise SourceKVReplayContractError(
                "selected block indices must be non-empty and unique"
            )
        if tuple(sorted(indices)) != indices:
            raise SourceKVReplayContractError("selected block indices must be sorted")
        self.selected_block_indices = indices
        self._identity: Optional[tuple[int, int, str, int, int]] = None
        self._entries: dict[int, CapturedSourceKV] = {}
        self._retired_identities: set[tuple[int, int, str, int, int]] = set()
        self.capture_calls = 0
        self.replay_lookups = 0
        self.replay_branch_counts: dict[str, int] = {}
        self.replay_phase_counts: dict[str, int] = {
            phase: 0 for phase in EXECUTION_PHASES
        }
        self.replay_branch_phase_counts: dict[str, dict[str, int]] = {}
        self.checkpoint_context_counts: dict[str, int] = {
            CHECKPOINT_FORWARD: 0,
            CHECKPOINT_RECOMPUTE: 0,
        }
        self.checkpoint_branch_counts: dict[str, dict[str, int]] = {}

    @property
    def identity(self) -> Optional[tuple[int, int, str, int, int]]:
        return self._identity

    @property
    def complete(self) -> bool:
        return tuple(sorted(self._entries)) == self.selected_block_indices

    def _enter(self, invocation: ReplayInvocation) -> None:
        if invocation.cache_bank is not self:
            raise SourceKVReplayContractError("invocation refers to a different cache bank")
        if invocation.mode == CAPTURE_MODE:
            if self._identity is None:
                if invocation.identity in self._retired_identities:
                    raise SourceKVReplayContractError(
                        "capture identity was retired; generation/step token must be unique"
                    )
                self._identity = invocation.identity
            elif self._identity != invocation.identity:
                raise SourceKVReplayContractError(
                    "cache still belongs to another generation/step; clear it explicitly"
                )
            if self._entries:
                raise SourceKVReplayContractError(
                    "capture cache is already populated; refusing silent overwrite"
                )
        else:
            if self._identity != invocation.identity:
                raise SourceKVReplayContractError(
                    "replay generation/step/rank identity differs from captured carrier"
                )
            self.assert_complete()

    def assert_complete(self) -> None:
        missing = tuple(
            index for index in self.selected_block_indices if index not in self._entries
        )
        unexpected = tuple(
            index for index in self._entries if index not in self.selected_block_indices
        )
        if missing or unexpected:
            raise SourceKVReplayContractError(
                f"source K/V cache incomplete: missing={missing}, unexpected={unexpected}"
            )

    def _record_checkpoint_context(
        self, invocation: ReplayInvocation, phase: str
    ) -> None:
        if invocation.mode != REPLAY_MODE:
            raise SourceKVReplayContractError(
                "gradient checkpoint replay is forbidden for source capture"
            )
        if phase not in (CHECKPOINT_FORWARD, CHECKPOINT_RECOMPUTE):
            raise SourceKVReplayContractError("invalid checkpoint context phase")
        if invocation.cache_bank is not self or invocation.identity != self._identity:
            raise SourceKVReplayContractError(
                "checkpoint invocation/cache identity is stale"
            )
        self.assert_complete()
        self.checkpoint_context_counts[phase] += 1
        branch_counts = self.checkpoint_branch_counts.setdefault(
            invocation.branch_tag,
            {CHECKPOINT_FORWARD: 0, CHECKPOINT_RECOMPUTE: 0},
        )
        branch_counts[phase] += 1

    def clear(self) -> None:
        active = _CURRENT_INVOCATION.get()
        if active is not None and active.cache_bank is self:
            raise SourceKVReplayContractError("cannot clear an active source K/V invocation")
        if self._identity is not None:
            self._retired_identities.add(self._identity)
        self._identity = None
        self._entries.clear()

    @staticmethod
    def _validate_projected_pair(key: Any, value: Any) -> tuple[int, int, int]:
        for label, tensor in (("key", key), ("value", value)):
            if getattr(tensor, "ndim", None) != 4 or int(tensor.shape[0]) != 1:
                raise SourceKVReplayContractError(
                    f"projected {label} must have shape [1,S,H,D]"
                )
        if tuple(key.shape) != tuple(value.shape):
            raise SourceKVReplayContractError("source key/value shapes differ")
        if key.dtype != value.dtype or key.device != value.device:
            raise SourceKVReplayContractError(
                "source key/value dtype or device differs"
            )
        source_tokens = _exact_positive_int(key.shape[1], label="source token count")
        local_heads = _exact_positive_int(key.shape[2], label="local head count")
        head_dim = _exact_positive_int(key.shape[3], label="head dimension")
        return source_tokens, local_heads, head_dim

    def capture(
        self,
        *,
        invocation: ReplayInvocation,
        block_index: int,
        key: Any,
        value: Any,
    ) -> None:
        if invocation.mode != CAPTURE_MODE or invocation.identity != self._identity:
            raise SourceKVReplayContractError("capture invocation/cache identity mismatch")
        index = _exact_nonnegative_int(block_index, label="block index")
        if index not in self.selected_block_indices:
            raise SourceKVReplayContractError(f"block {index} is outside cache scope")
        if index in self._entries:
            raise SourceKVReplayContractError(
                f"block {index} source K/V was captured more than once"
            )
        source_tokens, local_heads, head_dim = self._validate_projected_pair(key, value)
        # Clone as well as detach.  The cache must not share mutable storage with
        # a fused projection workspace or retain any path into the carrier graph.
        cached_key = key.detach().clone().contiguous()
        cached_value = value.detach().clone().contiguous()
        self._entries[index] = CapturedSourceKV(
            generation=invocation.generation,
            step_index=invocation.step_index,
            timestep_token=invocation.timestep_token,
            block_index=index,
            rank=invocation.rank,
            ulysses_size=invocation.ulysses_size,
            source_tokens=source_tokens,
            local_heads=local_heads,
            head_dim=head_dim,
            key=cached_key,
            value=cached_value,
        )
        self.capture_calls += 1

    def _lookup(
        self,
        *,
        invocation: ReplayInvocation,
        block_index: int,
        current_key: Any,
        current_value: Any,
        source_tokens: int,
    ) -> CapturedSourceKV:
        if invocation.mode != REPLAY_MODE or invocation.identity != self._identity:
            raise SourceKVReplayContractError("replay invocation/cache identity mismatch")
        index = _exact_nonnegative_int(block_index, label="block index")
        entry = self._entries.get(index)
        if entry is None:
            raise SourceKVReplayContractError(
                f"block {index} has no captured source K/V"
            )
        if (
            entry.generation != invocation.generation
            or entry.step_index != invocation.step_index
            or entry.timestep_token != invocation.timestep_token
            or entry.rank != invocation.rank
            or entry.ulysses_size != invocation.ulysses_size
        ):
            raise SourceKVReplayContractError(
                f"block {index} cache metadata is stale or cross-rank"
            )
        expected_source = _exact_positive_int(source_tokens, label="replay source tokens")
        if entry.source_tokens != expected_source:
            raise SourceKVReplayContractError(
                f"block {index} cached {entry.source_tokens} source tokens, "
                f"replay requires {expected_source}"
            )
        _, current_heads, current_dim = self._validate_projected_pair(
            current_key, current_value
        )
        if entry.local_heads != current_heads or entry.head_dim != current_dim:
            raise SourceKVReplayContractError(
                f"block {index} cached head shape "
                f"({entry.local_heads},{entry.head_dim}) differs from current "
                f"({current_heads},{current_dim})"
            )
        for label, cached, current in (
            ("key", entry.key, current_key),
            ("value", entry.value, current_value),
        ):
            if cached.dtype != current.dtype:
                raise SourceKVReplayContractError(
                    f"block {index} cached {label} dtype {cached.dtype} differs "
                    f"from current {current.dtype}"
                )
            if cached.device != current.device:
                raise SourceKVReplayContractError(
                    f"block {index} cached {label} device {cached.device} differs "
                    f"from current {current.device}"
                )
            if cached.requires_grad or cached.grad_fn is not None:
                raise SourceKVReplayContractError(
                    f"block {index} cached {label} unexpectedly retains autograd"
                )
        phase = _current_execution_phase(invocation)
        self.replay_lookups += 1
        self.replay_branch_counts[invocation.branch_tag] = (
            self.replay_branch_counts.get(invocation.branch_tag, 0) + 1
        )
        self.replay_phase_counts[phase] += 1
        branch_phases = self.replay_branch_phase_counts.setdefault(
            invocation.branch_tag, {item: 0 for item in EXECUTION_PHASES}
        )
        branch_phases[phase] += 1
        return entry

    def lookup(
        self,
        *,
        invocation: ReplayInvocation,
        block_index: int,
        current_key: Any,
        current_value: Any,
        source_tokens: int,
    ) -> CapturedSourceKV:
        """Validated public replay lookup for compatible carrier operators."""

        return self._lookup(
            invocation=invocation,
            block_index=block_index,
            current_key=current_key,
            current_value=current_value,
            source_tokens=source_tokens,
        )

    def inspect_entry(self, block_index: int) -> CapturedSourceKV:
        """Return a detached copy for tests/receipts without exposing cache storage."""

        index = _exact_nonnegative_int(block_index, label="block index")
        entry = self._entries.get(index)
        if entry is None:
            raise SourceKVReplayContractError(f"block {index} is not cached")
        values = dict(entry.__dict__)
        values["key"] = entry.key.detach().clone()
        values["value"] = entry.value.detach().clone()
        return CapturedSourceKV(**values)

    def receipt(self) -> dict[str, Any]:
        identity = self._identity
        entries = []
        for index in sorted(self._entries):
            entry = self._entries[index]
            entries.append(
                {
                    "block_index": index,
                    "shape": list(entry.key.shape),
                    "dtype": str(entry.key.dtype),
                    "device": str(entry.key.device),
                    "key_position_state": "post_rope_verified_non_none_rotary",
                    "detached": (
                        not entry.key.requires_grad
                        and entry.key.grad_fn is None
                        and not entry.value.requires_grad
                        and entry.value.grad_fn is None
                    ),
                }
            )
        value: dict[str, Any] = {
            "identity": None
            if identity is None
            else {
                "generation": identity[0],
                "step_index": identity[1],
                "timestep_token": identity[2],
                "rank": identity[3],
                "ulysses_size": identity[4],
            },
            "selected_blocks": list(self.selected_block_indices),
            "captured_blocks": sorted(self._entries),
            "complete": self.complete,
            "capture_calls": self.capture_calls,
            "replay_lookups": self.replay_lookups,
            "replay_branch_counts": dict(sorted(self.replay_branch_counts.items())),
            "replay_phase_counts": dict(self.replay_phase_counts),
            "replay_branch_phase_counts": {
                branch: dict(counts)
                for branch, counts in sorted(self.replay_branch_phase_counts.items())
            },
            "checkpoint_context_counts": dict(self.checkpoint_context_counts),
            "checkpoint_branch_counts": {
                branch: dict(counts)
                for branch, counts in sorted(self.checkpoint_branch_counts.items())
            },
            "retired_identity_count": len(self._retired_identities),
            "entries": entries,
        }
        value["cache_digest"] = _object_sha256(value)
        return value


@contextmanager
def source_kv_replay_invocation(
    cache_bank: SourceKVCacheBank,
    *,
    mode: str,
    branch_tag: str,
    generation: int,
    step_index: int,
    timestep_token: str,
    rank: int,
    ulysses_size: int,
) -> Iterator[ReplayInvocation]:
    """Bind one explicit branch/step contract for patched attention calls."""

    if not isinstance(cache_bank, SourceKVCacheBank):
        raise SourceKVReplayContractError("cache_bank has the wrong type")
    invocation = ReplayInvocation(
        cache_bank=cache_bank,
        mode=mode,
        branch_tag=branch_tag,
        generation=_exact_nonnegative_int(generation, label="generation"),
        step_index=_exact_nonnegative_int(step_index, label="step index"),
        timestep_token=timestep_token,
        rank=_exact_nonnegative_int(rank, label="Ulysses rank"),
        ulysses_size=_exact_positive_int(ulysses_size, label="Ulysses size"),
    )
    _validate_invocation(invocation)
    if (
        _CURRENT_INVOCATION.get() is not None
        or _CURRENT_CHECKPOINT_BINDING.get() is not None
    ):
        raise SourceKVReplayContractError("nested source K/V invocations are forbidden")
    cache_bank._enter(invocation)
    token = _CURRENT_INVOCATION.set(invocation)
    succeeded = False
    try:
        yield invocation
        succeeded = True
    finally:
        _CURRENT_INVOCATION.reset(token)
        if succeeded and invocation.mode == CAPTURE_MODE:
            cache_bank.assert_complete()


@contextmanager
def _checkpoint_forward_context(
    snapshot: ReplayInvocation,
) -> Iterator[None]:
    active = _CURRENT_INVOCATION.get()
    if active is None or not _same_invocation(active, snapshot):
        raise SourceKVReplayContractError(
            "checkpoint forward context differs from its captured invocation"
        )
    if _CURRENT_CHECKPOINT_BINDING.get() is not None:
        raise SourceKVReplayContractError("nested checkpoint bindings are forbidden")
    binding = CheckpointInvocationBinding(
        invocation=snapshot, phase=CHECKPOINT_FORWARD
    )
    token = _CURRENT_CHECKPOINT_BINDING.set(binding)
    try:
        snapshot.cache_bank._record_checkpoint_context(
            snapshot, CHECKPOINT_FORWARD
        )
        yield
    finally:
        _CURRENT_CHECKPOINT_BINDING.reset(token)


@contextmanager
def _checkpoint_recompute_context(
    snapshot: ReplayInvocation,
) -> Iterator[None]:
    if (
        _CURRENT_INVOCATION.get() is not None
        or _CURRENT_CHECKPOINT_BINDING.get() is not None
    ):
        raise SourceKVReplayContractError(
            "checkpoint recompute requires an otherwise empty invocation context"
        )
    # Revalidate immediately before recomputation.  Clearing or advancing the
    # rank-local bank between forward and backward must fail instead of replaying
    # a carrier from another step.
    snapshot.cache_bank._enter(snapshot)
    invocation_token = _CURRENT_INVOCATION.set(snapshot)
    binding = CheckpointInvocationBinding(
        invocation=snapshot, phase=CHECKPOINT_RECOMPUTE
    )
    binding_token = _CURRENT_CHECKPOINT_BINDING.set(binding)
    try:
        snapshot.cache_bank._record_checkpoint_context(
            snapshot, CHECKPOINT_RECOMPUTE
        )
        yield
    finally:
        _CURRENT_CHECKPOINT_BINDING.reset(binding_token)
        _CURRENT_INVOCATION.reset(invocation_token)


def source_kv_replay_checkpoint_context_fn() -> tuple[Any, Any]:
    """Return contexts for one non-reentrant ``torch.checkpoint`` call.

    PyTorch calls this function when each checkpoint is created, while the
    branch's outer :func:`source_kv_replay_invocation` is active.  The immutable
    snapshot is therefore private to that checkpoint.  During backward the
    recompute context restores the exact branch even when losses from multiple
    adapted branches are combined before one backward call.

    Capture is intentionally unsupported: the carrier must be produced by a
    frozen, no-grad source-only forward and must never be recaptured during
    checkpoint recomputation.
    """

    active = current_source_kv_invocation()
    snapshot = _snapshot_invocation(active)
    if snapshot.mode != REPLAY_MODE:
        raise SourceKVReplayContractError(
            "source capture must be frozen/no-grad and cannot use checkpoint recompute"
        )
    return (
        _checkpoint_forward_context(snapshot),
        _checkpoint_recompute_context(snapshot),
    )


def _projected_qkv_shape(query: Any, key: Any, value: Any) -> tuple[int, int, int, int]:
    shapes = []
    for label, tensor in (("query", query), ("key", key), ("value", value)):
        if getattr(tensor, "ndim", None) != 4 or int(tensor.shape[0]) != 1:
            raise SourceKVReplayContractError(
                f"official projected {label} must have shape [1,S,H,D]"
            )
        shapes.append(tuple(int(item) for item in tensor.shape))
    if not (shapes[0] == shapes[1] == shapes[2]):
        raise SourceKVReplayContractError(
            f"self-attention q/k/v shapes differ after official projection: {shapes!r}"
        )
    if not (query.dtype == key.dtype == value.dtype):
        raise SourceKVReplayContractError("projected q/k/v dtypes differ")
    if not (query.device == key.device == value.device):
        raise SourceKVReplayContractError("projected q/k/v devices differ")
    return shapes[0]


def _require_rotary_embedding(rotary_emb: Any) -> None:
    if rotary_emb is None:
        raise SourceKVReplayContractError(
            "Bernini self-attention rotary_emb is required; a missing value "
            "would make the advertised post-RoPE cache false"
        )
    if getattr(rotary_emb, "ndim", None) != 4:
        raise SourceKVReplayContractError(
            "rotary_emb must have official shape [1,S,1,head_dim/2]"
        )
    shape = tuple(int(item) for item in rotary_emb.shape)
    if shape[0] != 1 or shape[1] <= 0 or shape[2] != 1 or shape[3] <= 0:
        raise SourceKVReplayContractError(
            "rotary_emb must have official shape [1,S,1,head_dim/2]"
        )
    if not bool(getattr(getattr(rotary_emb, "dtype", None), "is_complex", False)):
        raise SourceKVReplayContractError("rotary_emb must use a complex dtype")


def _validate_projected_rotary_embedding(
    rotary_emb: Any,
    *,
    projected_shape: tuple[int, int, int, int],
    projected_device: Any,
) -> None:
    _require_rotary_embedding(rotary_emb)
    _, sequence_length, _, head_dim = projected_shape
    if int(rotary_emb.shape[1]) != sequence_length:
        raise SourceKVReplayContractError(
            "rotary_emb sequence length differs from gathered q/k sequence"
        )
    if int(rotary_emb.shape[3]) * 2 != head_dim:
        raise SourceKVReplayContractError(
            "rotary_emb width differs from the projected attention head dimension"
        )
    if rotary_emb.device != projected_device:
        raise SourceKVReplayContractError(
            "rotary_emb device differs from projected q/k device"
        )


def _maybe_call(value: Any) -> Any:
    return value() if callable(value) else value


def _parallel_identity(state: Any) -> tuple[bool, int, int]:
    enabled = bool(_maybe_call(getattr(state, "ulysses_enabled", False)))
    if not enabled:
        return False, 0, 1

    rank = None
    size = None
    for name in ("ulysses_rank", "rank"):
        candidate = getattr(state, name, None)
        if candidate is not None:
            rank = _maybe_call(candidate)
            break
    for name in ("ulysses_size", "world_size"):
        candidate = getattr(state, name, None)
        if candidate is not None:
            size = _maybe_call(candidate)
            break

    if rank is None or size is None:
        try:
            torch = __import__("torch")
            distributed = torch.distributed
            if distributed.is_available() and distributed.is_initialized():
                if rank is None:
                    rank = distributed.get_rank()
                if size is None:
                    size = distributed.get_world_size()
        except Exception:
            pass
    if rank is None or size is None:
        raise SourceKVReplayContractError(
            "Ulysses is enabled but runtime rank/size cannot be proven"
        )
    rank_int = _exact_nonnegative_int(rank, label="runtime Ulysses rank")
    size_int = _exact_positive_int(size, label="runtime Ulysses size")
    if rank_int >= size_int:
        raise SourceKVReplayContractError("runtime Ulysses rank is outside group")
    return True, rank_int, size_int


def projected_qkv_shape(query: Any, key: Any, value: Any) -> tuple[int, int, int, int]:
    """Public validation of the official gathered Q/K/V tensor contract."""

    return _projected_qkv_shape(query, key, value)


def require_rotary_embedding(rotary_emb: Any) -> None:
    """Public fail-closed Bernini self-attention RoPE validator."""

    _require_rotary_embedding(rotary_emb)


def validate_projected_rotary_embedding(
    rotary_emb: Any,
    *,
    projected_shape: tuple[int, int, int, int],
    projected_device: Any,
) -> None:
    """Public shape/device validation after official Q/K RoPE projection."""

    _validate_projected_rotary_embedding(
        rotary_emb,
        projected_shape=projected_shape,
        projected_device=projected_device,
    )


def parallel_identity(state: Any) -> tuple[bool, int, int]:
    """Public Ulysses runtime identity query shared by carrier operators."""

    return _parallel_identity(state)


class SourceKVReplaySelfAttnProcessor:
    """Wrap the official processor at its post-RoPE ``_project_qkv`` boundary."""

    def __init__(
        self,
        base_processor: Any,
        *,
        block_index: int,
        cache_bank: SourceKVCacheBank,
        varlen_attention_fn: Optional[Callable[..., Any]] = None,
        get_parallel_state_fn: Optional[Callable[[], Any]] = None,
        gather_heads_scatter_seq_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not callable(getattr(base_processor, "_project_qkv", None)):
            raise SourceKVReplayContractError(
                "base attn1 processor lacks official _project_qkv"
            )
        index = _exact_nonnegative_int(block_index, label="block index")
        if not isinstance(cache_bank, SourceKVCacheBank):
            raise SourceKVReplayContractError("cache bank has the wrong type")
        if index not in cache_bank.selected_block_indices:
            raise SourceKVReplayContractError(
                f"block {index} is outside the cache bank scope"
            )
        self.base_processor = base_processor
        self.block_index = index
        self.cache_bank = cache_bank
        self._varlen_attention_fn = varlen_attention_fn
        self._get_parallel_state_fn = get_parallel_state_fn
        self._gather_heads_scatter_seq_fn = gather_heads_scatter_seq_fn
        self.capture_calls = 0
        self.replay_calls = 0
        self.branch_counts: dict[str, int] = {}
        self.execution_phase_counts: dict[str, int] = {
            phase: 0 for phase in EXECUTION_PHASES
        }
        self.verified_post_rope_project_qkv_calls = 0
        self.post_rope_phase_counts: dict[str, int] = {
            phase: 0 for phase in EXECUTION_PHASES
        }
        self.saw_ulysses = False
        self.last_source_tokens: Optional[int] = None

    def _runtime_ops(
        self,
    ) -> tuple[Callable[..., Any], Callable[[], Any], Callable[..., Any]]:
        varlen_fn = self._varlen_attention_fn
        state_fn = self._get_parallel_state_fn
        inverse_fn = self._gather_heads_scatter_seq_fn
        if varlen_fn is None:
            from bernini.attention import varlen_attention as varlen_fn
        if state_fn is None or inverse_fn is None:
            from bernini.parallel import gather_heads_scatter_seq, get_parallel_state

            if state_fn is None:
                state_fn = get_parallel_state
            if inverse_fn is None:
                inverse_fn = gather_heads_scatter_seq
        return varlen_fn, state_fn, inverse_fn

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Optional[Any] = None,
        attention_mask: Optional[Any] = None,
        rotary_emb: Optional[Any] = None,
        batch_image_vae_seqlen=None,
        text_features_length=None,
        origin_hidden_states_seq_len: Optional[int] = None,
        split_hidden_states_seq_len: Optional[int] = None,
        cu_seqlens_q_cache=None,
        max_seqlen_q_cache=None,
        cu_seqlens_k_cross_cache=None,
        cu_seqlens_q_cross_cache=None,
        max_seqlen_k_cross_cache=None,
        max_seqlen_q_cross_cache=None,
    ) -> Any:
        del (
            text_features_length,
            split_hidden_states_seq_len,
            cu_seqlens_k_cross_cache,
            cu_seqlens_q_cross_cache,
            max_seqlen_k_cross_cache,
            max_seqlen_q_cross_cache,
        )
        invocation = current_source_kv_invocation()
        if invocation.cache_bank is not self.cache_bank:
            raise SourceKVReplayContractError(
                "processor and invocation use different cache banks"
            )
        execution_phase = _current_execution_phase(invocation)
        if encoder_hidden_states is not None:
            raise SourceKVReplayContractError(
                "source K/V replay may only replace attn1 self-attention"
            )
        if attention_mask is not None:
            raise SourceKVReplayContractError(
                "an extra attention mask makes source K/V replay ambiguous"
            )
        if getattr(hidden_states, "ndim", None) != 3 or int(hidden_states.shape[0]) != 1:
            raise SourceKVReplayContractError(
                "source K/V replay requires hidden_states shaped [1,L,D]"
            )
        _require_rotary_embedding(rotary_emb)

        # Official Bernini establishes the cache coordinate system here:
        # projection -> q/k norm -> Ulysses gather/scatter -> q/k RoPE.
        query, key, value = self.base_processor._project_qkv(
            attn,
            hidden_states,
            None,
            rotary_emb,
            origin_hidden_states_seq_len,
            False,
        )
        projected_shape = _projected_qkv_shape(query, key, value)
        _, gathered_length, _, _ = projected_shape
        _validate_projected_rotary_embedding(
            rotary_emb,
            projected_shape=projected_shape,
            projected_device=query.device,
        )
        self.verified_post_rope_project_qkv_calls += 1
        self.post_rope_phase_counts[execution_phase] += 1

        varlen_fn, state_fn, inverse_fn = self._runtime_ops()
        parallel_state = state_fn()
        ulysses_enabled, runtime_rank, runtime_size = _parallel_identity(parallel_state)
        if (runtime_rank, runtime_size) != (invocation.rank, invocation.ulysses_size):
            raise SourceKVReplayContractError(
                "outer invocation rank/Ulysses size differs from runtime state: "
                f"outer=({invocation.rank},{invocation.ulysses_size}), "
                f"runtime=({runtime_rank},{runtime_size})"
            )

        if invocation.mode == CAPTURE_MODE:
            source_tokens = validate_source_only_layout(
                gathered_sequence_length=gathered_length,
                batch_image_vae_seqlen=batch_image_vae_seqlen,
                cu_seqlens_q_cache=cu_seqlens_q_cache,
                max_seqlen_q_cache=max_seqlen_q_cache,
                origin_hidden_states_seq_len=origin_hidden_states_seq_len,
            )
            self.cache_bank.capture(
                invocation=invocation,
                block_index=self.block_index,
                key=key,
                value=value,
            )
        else:
            source_tokens = validate_equal_pair_layout(
                gathered_sequence_length=gathered_length,
                batch_image_vae_seqlen=batch_image_vae_seqlen,
                cu_seqlens_q_cache=cu_seqlens_q_cache,
                max_seqlen_q_cache=max_seqlen_q_cache,
                origin_hidden_states_seq_len=origin_hidden_states_seq_len,
            )
            entry = self.cache_bank._lookup(
                invocation=invocation,
                block_index=self.block_index,
                current_key=key,
                current_value=value,
                source_tokens=source_tokens,
            )
            torch = __import__("torch")
            key = torch.cat((entry.key, key[:, source_tokens:]), dim=1)
            value = torch.cat((entry.value, value[:, source_tokens:]), dim=1)

        query_for_dtype = query
        query = query.squeeze(0).contiguous()
        key = key.squeeze(0).contiguous()
        value = value.squeeze(0).contiguous()
        output = varlen_fn(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q_cache,
            cu_seqlens_k=cu_seqlens_q_cache,
            max_seqlen_q=max_seqlen_q_cache,
            max_seqlen_k=max_seqlen_q_cache,
            causal=False,
        )
        if tuple(int(item) for item in output.shape) != tuple(int(item) for item in query.shape):
            raise SourceKVReplayContractError(
                "official varlen self-attention output shape differs from query"
            )
        output = output.unsqueeze(0)
        if ulysses_enabled:
            self.saw_ulysses = True
            output = inverse_fn(output, head_dim=2, seq_dim=1)
        if getattr(output, "ndim", None) != 4:
            raise SourceKVReplayContractError(
                "attention output must be [1,S,H,D] before head flattening"
            )
        output = output.flatten(2, 3).contiguous().type_as(query_for_dtype)
        output = attn.to_out[0](output)
        output = attn.to_out[1](output)

        self.last_source_tokens = source_tokens
        self.branch_counts[invocation.branch_tag] = (
            self.branch_counts.get(invocation.branch_tag, 0) + 1
        )
        self.execution_phase_counts[execution_phase] += 1
        if invocation.mode == CAPTURE_MODE:
            self.capture_calls += 1
        else:
            self.replay_calls += 1
        return output

    def statistics(self) -> dict[str, Any]:
        return {
            "block_index": self.block_index,
            "capture_calls": self.capture_calls,
            "replay_calls": self.replay_calls,
            "branch_counts": dict(sorted(self.branch_counts.items())),
            "execution_phase_counts": dict(self.execution_phase_counts),
            "verified_post_rope_project_qkv_calls": (
                self.verified_post_rope_project_qkv_calls
            ),
            "post_rope_phase_counts": dict(self.post_rope_phase_counts),
            "rotary_emb_required_non_none": True,
            "last_source_tokens": self.last_source_tokens,
            "ulysses_observed": self.saw_ulysses,
        }


def resolve_wan_transformer(model: Any) -> Any:
    """Resolve the single official 30-block Wan transformer through wrappers."""

    queue = [model]
    seen: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        blocks = getattr(candidate, "blocks", None)
        if blocks is not None and callable(getattr(candidate, "patch_vae_latent", None)):
            if len(blocks) != EXPECTED_BLOCK_COUNT:
                raise SourceKVReplayContractError(
                    f"Bernini-R 1.3B must have {EXPECTED_BLOCK_COUNT} blocks, "
                    f"got {len(blocks)}"
                )
            return candidate
        get_base_model = getattr(candidate, "get_base_model", None)
        if callable(get_base_model):
            try:
                queue.append(get_base_model())
            except Exception:
                pass
        diff_dec = getattr(candidate, "diff_dec", None)
        if diff_dec is not None:
            queue.append(diff_dec)
        for name in ("transformer", "base_model", "model", "module"):
            nested = getattr(candidate, name, None)
            if nested is not None and nested is not candidate:
                queue.append(nested)
    raise SourceKVReplayContractError(
        "could not resolve the official 30-block Bernini-R Wan transformer"
    )


@dataclass
class SourceKVReplayPatchHandle:
    transformer: Any
    selection: str
    indices: tuple[int, ...]
    cache_bank: SourceKVCacheBank
    processors: tuple[SourceKVReplaySelfAttnProcessor, ...]
    original_processors: tuple[Any, ...]
    restored: bool = False

    def restore(self) -> None:
        if self.restored:
            return
        blocks = self.transformer.blocks
        # Validate the whole handle first so a conflict cannot cause a partial restore.
        for index, installed in zip(self.indices, self.processors):
            if getattr(blocks[index].attn1, "processor", None) is not installed:
                raise SourceKVReplayContractError(
                    f"block {index} attn1 processor changed behind patch handle"
                )
        for index, original in zip(self.indices, self.original_processors):
            attn = blocks[index].attn1
            setter = getattr(attn, "set_processor", None)
            if callable(setter):
                setter(original)
            else:
                attn.processor = original
        self.restored = True

    def receipt(self) -> dict[str, Any]:
        value = source_kv_replay_contract(
            selection=self.selection, num_blocks=len(self.transformer.blocks)
        )
        value["runtime"] = {
            "installed_block_count": len(self.indices),
            "restored": self.restored,
            "cache": self.cache_bank.receipt(),
            "per_block": [processor.statistics() for processor in self.processors],
        }
        value["runtime_digest"] = _object_sha256(value["runtime"])
        return value


def install_source_kv_replay(
    model: Any,
    *,
    selection: str = "all",
    cache_bank: Optional[SourceKVCacheBank] = None,
    processor_factory: Optional[
        Callable[[Any, int, SourceKVCacheBank], SourceKVReplaySelfAttnProcessor]
    ] = None,
) -> SourceKVReplayPatchHandle:
    """Patch selected ``attn1`` processors and return an exact restore handle."""

    transformer = resolve_wan_transformer(model)
    indices = resolve_block_indices(len(transformer.blocks), selection)
    bank = cache_bank if cache_bank is not None else SourceKVCacheBank(indices)
    if not isinstance(bank, SourceKVCacheBank) or bank.selected_block_indices != indices:
        raise SourceKVReplayContractError(
            "provided cache bank scope differs from selected transformer blocks"
        )
    originals: list[Any] = []
    installed: list[SourceKVReplaySelfAttnProcessor] = []
    installed_indices: list[int] = []
    try:
        for index in indices:
            attn = transformer.blocks[index].attn1
            original = getattr(attn, "processor", None)
            if original is None:
                raise SourceKVReplayContractError(
                    f"block {index} attn1 lacks a processor"
                )
            if isinstance(original, SourceKVReplaySelfAttnProcessor):
                raise SourceKVReplayContractError(
                    f"block {index} already has source K/V replay"
                )
            processor = (
                processor_factory(original, index, bank)
                if processor_factory is not None
                else SourceKVReplaySelfAttnProcessor(
                    original, block_index=index, cache_bank=bank
                )
            )
            if not isinstance(processor, SourceKVReplaySelfAttnProcessor):
                raise SourceKVReplayContractError(
                    "processor_factory returned the wrong type"
                )
            setter = getattr(attn, "set_processor", None)
            if callable(setter):
                setter(processor)
            else:
                attn.processor = processor
            originals.append(original)
            installed.append(processor)
            installed_indices.append(index)
    except Exception:
        for index, original, processor in zip(
            reversed(installed_indices), reversed(originals), reversed(installed)
        ):
            attn = transformer.blocks[index].attn1
            if getattr(attn, "processor", None) is processor:
                setter = getattr(attn, "set_processor", None)
                if callable(setter):
                    setter(original)
                else:
                    attn.processor = original
        raise
    return SourceKVReplayPatchHandle(
        transformer=transformer,
        selection=selection,
        indices=indices,
        cache_bank=bank,
        processors=tuple(installed),
        original_processors=tuple(originals),
    )


@contextmanager
def source_kv_replay(
    model: Any, *, selection: str = "all"
) -> Iterator[SourceKVReplayPatchHandle]:
    """Temporarily install the V9 tensor core and restore prior processors."""

    handle = install_source_kv_replay(model, selection=selection)
    try:
        yield handle
    finally:
        handle.restore()


__all__ = [
    "BLOCK_SELECTIONS",
    "CAPTURE_BRANCH_TAG",
    "CAPTURE_MODE",
    "CHECKPOINT_FORWARD",
    "CHECKPOINT_RECOMPUTE",
    "CheckpointInvocationBinding",
    "CORE_SCHEMA",
    "CapturedSourceKV",
    "EAGER_EXECUTION",
    "EXPECTED_BLOCK_COUNT",
    "EXECUTION_PHASES",
    "INVOCATION_MODES",
    "MAIN_BLOCK_SELECTION",
    "MID_ABLATION_START_1P3B",
    "MID_ABLATION_STOP_1P3B",
    "REPLAY_BRANCH_TAGS",
    "REPLAY_MODE",
    "ReplayInvocation",
    "SourceKVCacheBank",
    "SourceKVReplayContractError",
    "SourceKVReplayPatchHandle",
    "SourceKVReplaySelfAttnProcessor",
    "current_execution_phase",
    "current_source_kv_invocation",
    "install_source_kv_replay",
    "phase_shuffle_contract",
    "parallel_identity",
    "projected_qkv_shape",
    "require_rotary_embedding",
    "resolve_block_indices",
    "resolve_wan_transformer",
    "source_kv_replay",
    "source_kv_replay_checkpoint_context_fn",
    "source_kv_replay_contract",
    "source_kv_replay_invocation",
    "validate_equal_pair_layout",
    "validate_projected_rotary_embedding",
    "validate_source_only_layout",
]
