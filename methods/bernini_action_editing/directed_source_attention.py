#!/usr/bin/env python3
"""Fail-closed directed self-attention for Bernini-R video editing.

Bernini concatenates clean source-video tokens followed by noisy target tokens
inside one self-attention sequence.  Its stock attention is bidirectional, so
the representation used as the source reference can absorb target noise at
every block.  This diagnostic processor changes only that visibility graph::

    source queries -> source keys/values
    target queries -> source + target keys/values

It is deliberately an *oracle diagnostic*, not a trained method.  The split is
accepted only for one packed sample containing two equal-length video token
segments in ``[source, target]`` order.  There is no fallback to ordinary
attention when the contract is not provable.

The processor calls the official processor's ``_project_qkv`` first.  Thus, on
four-rank Ulysses, the sequence has already been gathered and the heads have
already been scattered before it is split.  The official inverse
``gather_heads_scatter_seq`` is applied after the two attention calls.

PyTorch and Bernini are imported lazily so contract tests can run in the small
local environment.  ``install_directed_source_attention`` monkey-patches only
``attn1`` processors and returns a handle that restores the exact prior
processor objects.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence


EXPECTED_BLOCK_COUNT = 30
BLOCK_SELECTIONS = ("all", "mid", "late")
ORACLE_SCHEMA = "bernini-directed-source-attention-oracle-v1"


class DirectedAttentionContractError(RuntimeError):
    """Raised instead of silently applying an ambiguous attention split."""


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
        raise DirectedAttentionContractError(
            f"oracle contract is not canonical JSON: {error}"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def resolve_block_indices(
    num_blocks: int, selection: str
) -> tuple[int, ...]:
    """Resolve the stable all/middle-third/late-third ablation scopes."""

    if type(num_blocks) is not int or num_blocks < 3:
        raise DirectedAttentionContractError("transformer must have at least 3 blocks")
    if selection not in BLOCK_SELECTIONS:
        raise DirectedAttentionContractError(
            f"block selection must be one of {BLOCK_SELECTIONS}, got {selection!r}"
        )
    first_cut = num_blocks // 3
    second_cut = (2 * num_blocks) // 3
    if selection == "all":
        indices = tuple(range(num_blocks))
    elif selection == "mid":
        indices = tuple(range(first_cut, second_cut))
    else:
        indices = tuple(range(second_cut, num_blocks))
    if not indices:
        raise DirectedAttentionContractError("block selection resolved to no blocks")
    return indices


def oracle_contract(*, selection: str, num_blocks: int = EXPECTED_BLOCK_COUNT) -> dict[str, Any]:
    indices = resolve_block_indices(num_blocks, selection)
    value: dict[str, Any] = {
        "schema_version": ORACLE_SCHEMA,
        "status": "untrained_architecture_oracle_not_production_method",
        "attention": "self_attention_attn1_only",
        "sequence_order": ["clean_source_video", "noisy_target_video"],
        "batch_size": 1,
        "equal_source_target_token_lengths": True,
        "source_query_keys_values": ["source"],
        "target_query_keys_values": ["source", "target"],
        "split_point": "half_of_verified_full_sequence_after_qkv_ulysses_gather",
        "ordinary_attention_fallback": False,
        "external_conditions": ["source_video", "edit_instruction"],
        "forbidden_external_conditions": [
            "target_video",
            "mask",
            "track",
            "swept_tube",
            "pose",
            "trajectory",
            "optical_flow",
            "first_frame_anchor",
        ],
        "block_selection": selection,
        "block_indices": list(indices),
        "num_transformer_blocks": num_blocks,
        "trained_parameters": 0,
    }
    value["contract_digest"] = _object_sha256(value)
    return value


def _as_single_int(value: Any, *, label: str) -> int:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "numel"):
        if int(value.numel()) != 1:
            raise DirectedAttentionContractError(f"{label} must be scalar")
        value = value.item()
    if isinstance(value, bool):
        raise DirectedAttentionContractError(f"{label} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise DirectedAttentionContractError(f"{label} must be an integer") from error
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise DirectedAttentionContractError(f"{label} must be numeric") from error
    if not math.isfinite(numeric) or numeric != float(integer):
        raise DirectedAttentionContractError(f"{label} must be an exact finite integer")
    return integer


def _as_int_tuple(value: Any, *, label: str) -> tuple[int, ...]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DirectedAttentionContractError(f"{label} must be a sequence")
    return tuple(_as_single_int(item, label=f"{label} item") for item in value)


def validate_equal_pair_layout(
    *,
    gathered_sequence_length: int,
    batch_image_vae_seqlen: Any,
    cu_seqlens_q_cache: Any,
    max_seqlen_q_cache: Any,
    origin_hidden_states_seq_len: Any,
) -> int:
    """Validate one exact ``[source, target]`` pair and return its boundary."""

    total = _as_single_int(gathered_sequence_length, label="gathered sequence length")
    if total <= 0 or total % 2:
        raise DirectedAttentionContractError(
            "directed attention requires a positive even full sequence"
        )
    lengths = _as_int_tuple(batch_image_vae_seqlen, label="batch_image_vae_seqlen")
    if lengths != (total,):
        raise DirectedAttentionContractError(
            "directed attention requires batch=1 and one full source+target sequence: "
            f"lengths={lengths!r}, gathered={total}"
        )
    cu = _as_int_tuple(cu_seqlens_q_cache, label="cu_seqlens_q_cache")
    if cu != (0, total):
        raise DirectedAttentionContractError(
            f"self-attention cu_seqlens must be exactly (0,{total}), got {cu!r}"
        )
    maximum = _as_single_int(max_seqlen_q_cache, label="max_seqlen_q_cache")
    if maximum != total:
        raise DirectedAttentionContractError(
            f"max self-attention length {maximum} differs from full sequence {total}"
        )
    if origin_hidden_states_seq_len is not None:
        origin = _as_single_int(
            origin_hidden_states_seq_len, label="origin_hidden_states_seq_len"
        )
        if origin != total:
            raise DirectedAttentionContractError(
                f"Ulysses origin length {origin} differs from gathered length {total}"
            )
    boundary = total // 2
    if boundary <= 0 or total - boundary != boundary:
        raise DirectedAttentionContractError("source and target token lengths differ")
    return boundary


class DirectedSourceSelfAttnProcessor:
    """Drop-in wrapper around Bernini's official ``WanAttnProcessor2_0``."""

    def __init__(
        self,
        base_processor: Any,
        *,
        block_index: int,
        varlen_attention_fn: Optional[Callable[..., Any]] = None,
        get_parallel_state_fn: Optional[Callable[[], Any]] = None,
        gather_heads_scatter_seq_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not callable(getattr(base_processor, "_project_qkv", None)):
            raise DirectedAttentionContractError(
                "base attn1 processor lacks official _project_qkv"
            )
        if type(block_index) is not int or block_index < 0:
            raise DirectedAttentionContractError("block index must be non-negative")
        self.base_processor = base_processor
        self.block_index = block_index
        self._varlen_attention_fn = varlen_attention_fn
        self._get_parallel_state_fn = get_parallel_state_fn
        self._gather_heads_scatter_seq_fn = gather_heads_scatter_seq_fn
        self.call_count = 0
        self.full_sequence_length: Optional[int] = None
        self.source_sequence_length: Optional[int] = None
        self.saw_ulysses = False

    def _runtime_ops(self) -> tuple[Callable[..., Any], Callable[[], Any], Callable[..., Any]]:
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
        if encoder_hidden_states is not None:
            raise DirectedAttentionContractError(
                "directed source processor may only replace attn1 self-attention"
            )
        if attention_mask is not None:
            raise DirectedAttentionContractError(
                "an additional attention mask makes the directed oracle ambiguous"
            )
        if getattr(hidden_states, "ndim", None) != 3 or int(hidden_states.shape[0]) != 1:
            raise DirectedAttentionContractError(
                "directed attention requires rank-local hidden_states shaped [1,L,D]"
            )

        # This is intentionally delegated before the split.  The official
        # processor applies q/k normalisation, Ulysses gather-seq/scatter-heads,
        # and RoPE here.
        query, key, value = self.base_processor._project_qkv(
            attn,
            hidden_states,
            None,
            rotary_emb,
            origin_hidden_states_seq_len,
            False,
        )
        for label, tensor in (("query", query), ("key", key), ("value", value)):
            if getattr(tensor, "ndim", None) != 4 or int(tensor.shape[0]) != 1:
                raise DirectedAttentionContractError(
                    f"official projected {label} must have shape [1,S,H,D]"
                )
        shapes = tuple(tuple(int(item) for item in tensor.shape) for tensor in (query, key, value))
        if not (shapes[0] == shapes[1] == shapes[2]):
            raise DirectedAttentionContractError(
                f"self-attention q/k/v shapes differ after Ulysses gather: {shapes!r}"
            )
        full_length = int(query.shape[1])
        boundary = validate_equal_pair_layout(
            gathered_sequence_length=full_length,
            batch_image_vae_seqlen=batch_image_vae_seqlen,
            cu_seqlens_q_cache=cu_seqlens_q_cache,
            max_seqlen_q_cache=max_seqlen_q_cache,
            origin_hidden_states_seq_len=origin_hidden_states_seq_len,
        )
        if self.full_sequence_length not in (None, full_length):
            raise DirectedAttentionContractError(
                "packed sequence length changed across denoising calls"
            )
        self.full_sequence_length = full_length
        self.source_sequence_length = boundary

        varlen_fn, state_fn, inverse_fn = self._runtime_ops()
        torch = __import__("torch")
        device = query.device
        cu_half = torch.tensor([0, boundary], dtype=torch.int32, device=device)
        cu_full = torch.tensor([0, full_length], dtype=torch.int32, device=device)
        query = query.squeeze(0).contiguous()
        key = key.squeeze(0).contiguous()
        value = value.squeeze(0).contiguous()

        source_output = varlen_fn(
            query[:boundary],
            key[:boundary],
            value[:boundary],
            cu_seqlens_q=cu_half,
            cu_seqlens_k=cu_half,
            max_seqlen_q=boundary,
            max_seqlen_k=boundary,
            causal=False,
        )
        target_output = varlen_fn(
            query[boundary:],
            key,
            value,
            cu_seqlens_q=cu_half,
            cu_seqlens_k=cu_full,
            max_seqlen_q=boundary,
            max_seqlen_k=full_length,
            causal=False,
        )
        expected_half_shape = tuple(int(item) for item in query[:boundary].shape)
        if tuple(int(item) for item in source_output.shape) != expected_half_shape:
            raise DirectedAttentionContractError("source attention output shape differs")
        if tuple(int(item) for item in target_output.shape) != expected_half_shape:
            raise DirectedAttentionContractError("target attention output shape differs")
        output = torch.cat((source_output, target_output), dim=0).unsqueeze(0)

        parallel_state = state_fn()
        ulysses_enabled = bool(getattr(parallel_state, "ulysses_enabled", False))
        if ulysses_enabled:
            self.saw_ulysses = True
            output = inverse_fn(output, head_dim=2, seq_dim=1)
        output = output.flatten(2, 3).contiguous().type_as(query)
        output = attn.to_out[0](output)
        output = attn.to_out[1](output)
        self.call_count += 1
        return output

    def statistics(self) -> dict[str, Any]:
        return {
            "block_index": self.block_index,
            "call_count": self.call_count,
            "full_sequence_length": self.full_sequence_length,
            "source_sequence_length": self.source_sequence_length,
            "target_sequence_length": self.source_sequence_length,
            "ulysses_observed": self.saw_ulysses,
        }


def resolve_wan_transformer(model: Any) -> Any:
    """Resolve the single official Wan transformer through renderer/PEFT wrappers."""

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
                raise DirectedAttentionContractError(
                    f"Bernini-R 1.3B must have {EXPECTED_BLOCK_COUNT} blocks, got {len(blocks)}"
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
    raise DirectedAttentionContractError(
        "could not resolve the official 30-block Bernini-R Wan transformer"
    )


@dataclass
class DirectedAttentionPatchHandle:
    transformer: Any
    selection: str
    indices: tuple[int, ...]
    processors: tuple[DirectedSourceSelfAttnProcessor, ...]
    original_processors: tuple[Any, ...]
    restored: bool = False

    def restore(self) -> None:
        if self.restored:
            return
        blocks = self.transformer.blocks
        for index, original, installed in zip(
            self.indices, self.original_processors, self.processors
        ):
            attn = blocks[index].attn1
            if getattr(attn, "processor", None) is not installed:
                raise DirectedAttentionContractError(
                    f"block {index} attn1 processor changed behind patch handle"
                )
            setter = getattr(attn, "set_processor", None)
            if callable(setter):
                setter(original)
            else:
                attn.processor = original
        self.restored = True

    def receipt(self) -> dict[str, Any]:
        value = oracle_contract(
            selection=self.selection, num_blocks=len(self.transformer.blocks)
        )
        value["runtime"] = {
            "installed_block_count": len(self.indices),
            "restored": self.restored,
            "per_block": [processor.statistics() for processor in self.processors],
        }
        value["runtime_digest"] = _object_sha256(value["runtime"])
        return value


def install_directed_source_attention(
    model: Any,
    *,
    selection: str = "all",
    processor_factory: Optional[Callable[[Any, int], DirectedSourceSelfAttnProcessor]] = None,
) -> DirectedAttentionPatchHandle:
    """Install the oracle on selected ``attn1`` blocks and return a restore handle."""

    transformer = resolve_wan_transformer(model)
    indices = resolve_block_indices(len(transformer.blocks), selection)
    originals: list[Any] = []
    installed: list[DirectedSourceSelfAttnProcessor] = []
    for index in indices:
        attn = transformer.blocks[index].attn1
        original = getattr(attn, "processor", None)
        if original is None:
            raise DirectedAttentionContractError(f"block {index} attn1 lacks processor")
        if isinstance(original, DirectedSourceSelfAttnProcessor):
            raise DirectedAttentionContractError(
                f"block {index} already has directed source attention"
            )
        processor = (
            processor_factory(original, index)
            if processor_factory is not None
            else DirectedSourceSelfAttnProcessor(original, block_index=index)
        )
        if not isinstance(processor, DirectedSourceSelfAttnProcessor):
            raise DirectedAttentionContractError("processor_factory returned the wrong type")
        setter = getattr(attn, "set_processor", None)
        if callable(setter):
            setter(processor)
        else:
            attn.processor = processor
        originals.append(original)
        installed.append(processor)
    return DirectedAttentionPatchHandle(
        transformer=transformer,
        selection=selection,
        indices=indices,
        processors=tuple(installed),
        original_processors=tuple(originals),
    )


@contextmanager
def directed_source_attention(
    model: Any, *, selection: str = "all"
) -> Iterator[DirectedAttentionPatchHandle]:
    """Temporarily install directed source attention and restore it exactly."""

    handle = install_directed_source_attention(model, selection=selection)
    try:
        yield handle
    finally:
        handle.restore()


__all__ = [
    "BLOCK_SELECTIONS",
    "DirectedAttentionContractError",
    "DirectedAttentionPatchHandle",
    "DirectedSourceSelfAttnProcessor",
    "EXPECTED_BLOCK_COUNT",
    "ORACLE_SCHEMA",
    "directed_source_attention",
    "install_directed_source_attention",
    "oracle_contract",
    "resolve_block_indices",
    "resolve_wan_transformer",
    "validate_equal_pair_layout",
]
