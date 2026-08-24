"""Online pure-T2V action-minus-noop transport at Bernini cross-attention.

The self-attention route in :mod:`anchor_qk_transport` mixes motion, spatial
layout and the current denoising state.  This module instead captures the
post-projection ``attn2`` output produced by two forwards of the *same* noised
pure-T2V anchor video: one with the action caption and one with a matched no-op
caption.  Only their phase-0-relative difference is added to the target suffix
of the target-conditional field call.  Source rows and the target frame-zero
phase remain byte-identical to the current branch.

The cache is step/candidate/rank/block/slot bound and fail-closed.  It is an
inference-time attention intervention; it neither trains parameters nor uses
the anchor as a reward or preselected endpoint.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, replace
import math
from typing import Any, Iterator, Optional, Sequence

import torch

import anchor_qk_transport as self_transport
import source_kv_replay as replay_runtime


METHOD = "bernini-online-pure-t2v-anchor-cross-attention-contrast-v1"
TEMPORAL_CONTRAST_CROSS_ATTN_OUTPUT = "temporal_contrast_cross_attn_output"
BLOCK_COUNT = 30
CAPTURE = "anchor_capture"
REPLAY = "target_replay"
MODES = (CAPTURE, REPLAY)
PAIRED_SUFFIX = self_transport.PAIRED_SUFFIX
FULL_SEQUENCE = self_transport.FULL_SEQUENCE
REPLAY_SCOPES = (PAIRED_SUFFIX, FULL_SEQUENCE)
ACTION_SLOT = self_transport.ACTION_SLOT
NOOP_SLOT = self_transport.NOOP_SLOT
SLOTS = (ACTION_SLOT, NOOP_SLOT)
LATENT_PHASES = self_transport.LATENT_PHASES


class AnchorCrossAttentionError(RuntimeError):
    pass


def _exact_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise AnchorCrossAttentionError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise AnchorCrossAttentionError(f"{label} must be an integer") from error
    if result != value:
        raise AnchorCrossAttentionError(f"{label} must be exact")
    return result


def _shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, torch.Tensor):
        raise AnchorCrossAttentionError("attention output must be a tensor")
    return tuple(int(item) for item in value.shape)


def _global_token_count(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AnchorCrossAttentionError("batch visual lengths must be a sequence")
    lengths = tuple(_exact_int(item, label="batch visual length") for item in value)
    if not lengths or any(item <= 0 for item in lengths):
        raise AnchorCrossAttentionError("batch visual lengths must be positive")
    return sum(lengths)


def _gather_global_sequence(
    local: torch.Tensor,
    *,
    total_tokens: int,
) -> tuple[torch.Tensor, int, int]:
    """Undo Bernini's append-pad/contiguous Ulysses sequence shard."""

    try:
        from bernini.parallel import get_parallel_state

        state = get_parallel_state()
        enabled, rank, size = replay_runtime.parallel_identity(state)
    except (ImportError, AttributeError):
        enabled, rank, size, state = False, 0, 1, None
    expected_local = math.ceil(total_tokens / size)
    if int(local.shape[1]) != expected_local:
        raise AnchorCrossAttentionError(
            "local cross-attention length differs from contiguous Ulysses layout"
        )
    if not enabled:
        if size != 1 or int(local.shape[1]) != total_tokens:
            raise AnchorCrossAttentionError("disabled Ulysses layout differs")
        return local, rank, size

    import torch.distributed as dist

    group = getattr(state, "ulysses_group", None)
    if (
        group is None
        or not dist.is_initialized()
        or dist.get_world_size(group) != size
        or dist.get_rank(group) != rank
    ):
        raise AnchorCrossAttentionError("live Ulysses group identity differs")
    shards = [torch.empty_like(local) for _ in range(size)]
    dist.all_gather(shards, local.contiguous(), group=group)
    full = torch.cat(shards, dim=1)[:, :total_tokens].contiguous()
    if int(full.shape[1]) != total_tokens:
        raise AnchorCrossAttentionError("gathered cross-attention length differs")
    return full, rank, size


def _scatter_global_sequence(
    full: torch.Tensor,
    *,
    original_local: torch.Tensor,
    total_tokens: int,
    rank: int,
    size: int,
) -> torch.Tensor:
    local_length = math.ceil(total_tokens / size)
    if int(original_local.shape[1]) != local_length or int(full.shape[1]) != total_tokens:
        raise AnchorCrossAttentionError("cross-attention scatter geometry differs")
    start = rank * local_length
    valid = max(0, min(total_tokens, start + local_length) - start)
    result = original_local.clone()
    if valid:
        result[:, :valid].copy_(full[:, start : start + valid])
    return result


class AnchorCrossAttentionCache:
    def __init__(self, selected_block_indices: Sequence[int]) -> None:
        indices = tuple(
            _exact_int(item, label="selected block index")
            for item in selected_block_indices
        )
        if (
            not indices
            or indices != tuple(sorted(set(indices)))
            or any(item < 0 or item >= BLOCK_COUNT for item in indices)
        ):
            raise AnchorCrossAttentionError(
                "selected blocks must be a non-empty increasing subset of 0..29"
            )
        self.selected_block_indices = indices
        self._entries: dict[
            tuple[int, int, int, int, str], tuple[torch.Tensor, int, int]
        ] = {}
        self.capture_count = 0
        self.replay_count = 0

    @staticmethod
    def _key(
        invocation: "AnchorCrossAttentionInvocation", block_index: int
    ) -> tuple[int, int, int, int, str]:
        return (
            invocation.step_index,
            invocation.candidate_index,
            invocation.rank,
            block_index,
            invocation.slot,
        )

    def capture(
        self,
        *,
        invocation: "AnchorCrossAttentionInvocation",
        block_index: int,
        output: torch.Tensor,
    ) -> None:
        if invocation.mode != CAPTURE:
            raise AnchorCrossAttentionError("only a capture invocation may write")
        if output.ndim != 3 or int(output.shape[0]) != 1:
            raise AnchorCrossAttentionError("captured cross-attention shape differs")
        if int(output.shape[1]) % LATENT_PHASES:
            raise AnchorCrossAttentionError("captured output lacks 21 latent phases")
        if not bool(torch.isfinite(output).all()):
            raise AnchorCrossAttentionError("captured output is non-finite")
        key = self._key(invocation, block_index)
        if key in self._entries:
            raise AnchorCrossAttentionError("cross-attention cache entry already exists")
        self._entries[key] = (
            output.detach().clone(),
            invocation.replay_uses,
            invocation.replay_uses,
        )
        self.capture_count += 1

    def consume(
        self,
        *,
        invocation: "AnchorCrossAttentionInvocation",
        block_index: int,
        current_output: torch.Tensor,
    ) -> torch.Tensor:
        if invocation.mode != REPLAY:
            raise AnchorCrossAttentionError("only a replay invocation may consume")
        key = self._key(invocation, block_index)
        stored = self._entries.get(key)
        if stored is None:
            raise AnchorCrossAttentionError("matching cross-attention cache is absent")
        output, remaining, total = stored
        if invocation.replay_uses != total:
            raise AnchorCrossAttentionError("cross-attention replay-use contract differs")
        current_tokens = int(current_output.shape[1])
        target_tokens = (
            current_tokens // 2
            if invocation.replay_scope == PAIRED_SUFFIX
            else current_tokens
        )
        if (
            current_output.ndim != 3
            or int(current_output.shape[0]) != 1
            or (
                invocation.replay_scope == PAIRED_SUFFIX
                and current_tokens % 2
            )
            or _shape(output)
            != (int(current_output.shape[0]), target_tokens, int(current_output.shape[2]))
            or output.dtype != current_output.dtype
            or output.device != current_output.device
        ):
            raise AnchorCrossAttentionError("anchor/current cross-attention geometry differs")
        if remaining == 1:
            self._entries.pop(key)
        else:
            self._entries[key] = (output, remaining - 1, total)
        self.replay_count += 1
        return output

    def assert_empty(self) -> None:
        if self._entries:
            raise AnchorCrossAttentionError("unconsumed cross-attention entries remain")

    def receipt(self) -> dict[str, Any]:
        return {
            "method": METHOD,
            "selected_block_indices": list(self.selected_block_indices),
            "capture_count": self.capture_count,
            "replay_count": self.replay_count,
            "pending_entries": len(self._entries),
        }


@dataclass(frozen=True)
class AnchorCrossAttentionInvocation:
    mode: str
    cache_bank: AnchorCrossAttentionCache
    step_index: int
    candidate_index: int
    rank: int
    ulysses_size: int
    transport_strength: float
    replay_uses: int = 1
    replay_scope: str = PAIRED_SUFFIX
    slot: str = ACTION_SLOT

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise AnchorCrossAttentionError(f"mode must be one of {MODES}")
        if not isinstance(self.cache_bank, AnchorCrossAttentionCache):
            raise AnchorCrossAttentionError("cache bank has the wrong type")
        for label in ("step_index", "candidate_index", "rank", "ulysses_size"):
            value = _exact_int(getattr(self, label), label=label)
            if value < 0 or (label == "ulysses_size" and value < 1):
                raise AnchorCrossAttentionError(f"{label} is outside its domain")
        if self.rank >= self.ulysses_size:
            raise AnchorCrossAttentionError("rank must be smaller than ulysses_size")
        if isinstance(self.replay_uses, bool) or self.replay_uses not in (1, 2):
            raise AnchorCrossAttentionError("replay_uses must be one or two")
        if self.replay_scope not in REPLAY_SCOPES:
            raise AnchorCrossAttentionError("unknown replay scope")
        if self.slot not in SLOTS:
            raise AnchorCrossAttentionError("unknown anchor slot")
        if (
            isinstance(self.transport_strength, bool)
            or not isinstance(self.transport_strength, (int, float))
            or not math.isfinite(float(self.transport_strength))
            or not 0.0 < float(self.transport_strength) <= 1.0
        ):
            raise AnchorCrossAttentionError("transport strength must be in (0,1]")


_CURRENT: contextvars.ContextVar[Optional[AnchorCrossAttentionInvocation]] = (
    contextvars.ContextVar("anchor_cross_attention_invocation", default=None)
)


@contextlib.contextmanager
def anchor_cross_attention_invocation(
    invocation: AnchorCrossAttentionInvocation,
) -> Iterator[None]:
    if _CURRENT.get() is not None:
        raise AnchorCrossAttentionError("cross-attention invocations may not nest")
    token = _CURRENT.set(invocation)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def current_anchor_cross_attention_invocation(
) -> Optional[AnchorCrossAttentionInvocation]:
    return _CURRENT.get()


class AnchorCrossAttentionProcessor:
    """Wrap one official Bernini ``attn2`` processor at its output seam."""

    def __init__(
        self,
        base_processor: Any,
        *,
        block_index: int,
        cache_bank: AnchorCrossAttentionCache,
    ) -> None:
        if not callable(base_processor):
            raise AnchorCrossAttentionError("base attn2 processor is not callable")
        self.base_processor = base_processor
        self.block_index = _exact_int(block_index, label="block index")
        self.cache_bank = cache_bank
        self.base_delegations = 0
        self.capture_calls = 0
        self.replay_calls = 0

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
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
    ) -> torch.Tensor:
        output = self.base_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            rotary_emb=rotary_emb,
            batch_image_vae_seqlen=batch_image_vae_seqlen,
            text_features_length=text_features_length,
            origin_hidden_states_seq_len=origin_hidden_states_seq_len,
            split_hidden_states_seq_len=split_hidden_states_seq_len,
            cu_seqlens_q_cache=cu_seqlens_q_cache,
            max_seqlen_q_cache=max_seqlen_q_cache,
            cu_seqlens_k_cross_cache=cu_seqlens_k_cross_cache,
            cu_seqlens_q_cross_cache=cu_seqlens_q_cross_cache,
            max_seqlen_k_cross_cache=max_seqlen_k_cross_cache,
            max_seqlen_q_cross_cache=max_seqlen_q_cross_cache,
        )
        if (
            not isinstance(output, torch.Tensor)
            or output.ndim != 3
            or _shape(output) != _shape(hidden_states)
            or output.dtype != hidden_states.dtype
            or output.device != hidden_states.device
            or not bool(torch.isfinite(output).all())
        ):
            raise AnchorCrossAttentionError("official attn2 output contract differs")
        invocation = current_anchor_cross_attention_invocation()
        if invocation is None:
            self.base_delegations += 1
            return output
        if invocation.cache_bank is not self.cache_bank:
            raise AnchorCrossAttentionError("processor and invocation cache banks differ")
        total_tokens = _global_token_count(batch_image_vae_seqlen)
        full_output, rank, size = _gather_global_sequence(
            output, total_tokens=total_tokens
        )
        if (rank, size) != (invocation.rank, invocation.ulysses_size):
            raise AnchorCrossAttentionError("runtime and invocation Ulysses identity differs")
        if invocation.mode == CAPTURE:
            self.cache_bank.capture(
                invocation=invocation,
                block_index=self.block_index,
                output=full_output,
            )
            self.capture_calls += 1
            return output

        action = self.cache_bank.consume(
            invocation=invocation,
            block_index=self.block_index,
            current_output=full_output,
        )
        noop = self.cache_bank.consume(
            invocation=replace(invocation, slot=NOOP_SLOT),
            block_index=self.block_index,
            current_output=full_output,
        )
        source_tokens = (
            int(full_output.shape[1]) // 2
            if invocation.replay_scope == PAIRED_SUFFIX
            else 0
        )
        routed = self_transport._sparse_frame0_additive_contrast(
            full_output[:, source_tokens:].unsqueeze(2),
            action.unsqueeze(2),
            noop.unsqueeze(2),
            strength=invocation.transport_strength,
        ).squeeze(2)
        full_result = torch.cat((full_output[:, :source_tokens], routed), dim=1)
        result = _scatter_global_sequence(
            full_result,
            original_local=output,
            total_tokens=total_tokens,
            rank=rank,
            size=size,
        )
        if (
            _shape(result) != _shape(output)
            or result.dtype != output.dtype
            or result.device != output.device
            or not bool(torch.isfinite(result).all())
        ):
            raise AnchorCrossAttentionError("routed cross-attention output differs")
        self.replay_calls += 1
        return result


class AnchorCrossAttentionPatchHandle:
    def __init__(self, transformer: Any, cache_bank: AnchorCrossAttentionCache) -> None:
        blocks = getattr(transformer, "blocks", None)
        if not isinstance(blocks, (list, torch.nn.ModuleList)) or len(blocks) != BLOCK_COUNT:
            raise AnchorCrossAttentionError("transformer must expose exactly 30 blocks")
        self.transformer = transformer
        self.cache_bank = cache_bank
        self.originals: dict[int, Any] = {}

    def install(self) -> None:
        if self.originals:
            raise AnchorCrossAttentionError("cross-attention patch is already installed")
        for index in self.cache_bank.selected_block_indices:
            attn = self.transformer.blocks[index].attn2
            original = getattr(attn, "processor", None)
            wrapper = AnchorCrossAttentionProcessor(
                original, block_index=index, cache_bank=self.cache_bank
            )
            self.originals[index] = original
            setter = getattr(attn, "set_processor", None)
            if callable(setter):
                setter(wrapper)
            else:
                attn.processor = wrapper

    def restore(self) -> None:
        for index, original in self.originals.items():
            attn = self.transformer.blocks[index].attn2
            setter = getattr(attn, "set_processor", None)
            if callable(setter):
                setter(original)
            else:
                attn.processor = original
        self.originals.clear()


@contextlib.contextmanager
def install_anchor_cross_attention_transport(
    transformer: Any,
    *,
    selected_block_indices: Sequence[int],
) -> Iterator[AnchorCrossAttentionPatchHandle]:
    bank = AnchorCrossAttentionCache(selected_block_indices)
    handle = AnchorCrossAttentionPatchHandle(transformer, bank)
    handle.install()
    try:
        yield handle
        bank.assert_empty()
    finally:
        handle.restore()


__all__ = [
    "ACTION_SLOT",
    "NOOP_SLOT",
    "CAPTURE",
    "REPLAY",
    "TEMPORAL_CONTRAST_CROSS_ATTN_OUTPUT",
    "PAIRED_SUFFIX",
    "FULL_SEQUENCE",
    "AnchorCrossAttentionCache",
    "AnchorCrossAttentionError",
    "AnchorCrossAttentionInvocation",
    "AnchorCrossAttentionProcessor",
    "anchor_cross_attention_invocation",
    "install_anchor_cross_attention_transport",
]
