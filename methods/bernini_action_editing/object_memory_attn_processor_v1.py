#!/usr/bin/env python3
"""Local, fail-closed object-memory attention core for Bernini-R.

This module is an isolated architecture scaffold, not a production renderer
integration and not evidence that identity preservation works.  It wraps an
official Bernini ``attn1`` processor and, for a verified one-sample packed
``[source, target]`` sequence, adds target-row-only reads from four typed
source-object slots:

``dog_head``, ``dog_body``, ``dog_collar``, and ``bone``.

The base self-attention follows the same official ``_project_qkv`` -> varlen
attention -> Ulysses inverse -> ``to_out`` path used by
``source_kv_replay.py`` and ``source_value_residual.py``.  Official projected
Q/K are post-RoPE, so the memory branch explicitly removes the unit-modulus
rotary phase before masked Q/K matching.  Values are read from source rows;
only rows selected by the corresponding target responsibility mask receive a
residual.  The source half is never directly written by this branch.

``drop`` and ``swap`` are structural controls.  Drop suppresses named target
slots.  Swap redirects a target slot to another typed source mask (for
example, ``bone -> dog_collar``); it does not yet accept a second video's
external donor K/V.  A fixed zero gate immediately calls the original
processor exactly once, without importing Bernini, projecting Q/K/V, or
requiring an object-memory invocation.

Important integration boundary: the tensor math mirrors the audited official
SP1/Ulysses path, but it has not been exercised on native four-rank Bernini.
There is no cross-rank mask-content consensus, no checkpoint context rebinding,
no runner ABI, and no trained slot encoder/gate.  Active use therefore remains
an unintegrated local oracle scaffold and fails closed when its local shape,
dtype, device, packed-layout, RoPE, or declared world identity is ambiguous.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import source_kv_replay as replay


CORE_SCHEMA = "bernini-object-memory-attn-processor-v1-local-core-v1"
SLOT_NAMES = ("dog_head", "dog_body", "dog_collar", "bone")
MEMORY_MODES = ("read", "drop", "swap")
DEFAULT_BLOCK_INDICES = (19, 24, 29)


class ObjectMemoryContractError(RuntimeError):
    """Raised instead of silently applying an ambiguous object-memory read."""


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
        raise ObjectMemoryContractError(
            f"object-memory contract is not canonical JSON: {error}"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _exact_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ObjectMemoryContractError(f"{label} must be an integer")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "numel"):
        if int(value.numel()) != 1:
            raise ObjectMemoryContractError(f"{label} must be scalar")
        value = value.item()
    try:
        integer = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ObjectMemoryContractError(f"{label} must be an integer") from error
    if not math.isfinite(numeric) or numeric != float(integer) or integer < 0:
        raise ObjectMemoryContractError(
            f"{label} must be an exact non-negative finite integer"
        )
    return integer


def _exact_positive_int(value: Any, *, label: str) -> int:
    integer = _exact_nonnegative_int(value, label=label)
    if integer <= 0:
        raise ObjectMemoryContractError(f"{label} must be positive")
    return integer


def validate_gate(value: Any) -> float:
    """Return a finite fixed gate in [0, 1], rejecting bool coercion."""

    if isinstance(value, bool):
        raise ObjectMemoryContractError("object-memory gate must be numeric")
    try:
        gate = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ObjectMemoryContractError("object-memory gate must be numeric") from error
    if not math.isfinite(gate) or gate < 0.0 or gate > 1.0:
        raise ObjectMemoryContractError(
            "object-memory gate must be finite and in [0,1]"
        )
    return gate


def _validate_slot_keys(mapping: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(mapping, Mapping):
        raise ObjectMemoryContractError(f"{label} must be a mapping")
    copied = dict(mapping)
    if set(copied) != set(SLOT_NAMES):
        raise ObjectMemoryContractError(
            f"{label} keys must be exactly {SLOT_NAMES}, got {tuple(copied)!r}"
        )
    return {name: copied[name] for name in SLOT_NAMES}


def _validate_slot_sequence(value: Sequence[str], *, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ObjectMemoryContractError(f"{label} must be a sequence of slot names")
    slots = tuple(value)
    if len(set(slots)) != len(slots):
        raise ObjectMemoryContractError(f"{label} contains duplicate slots")
    invalid = tuple(slot for slot in slots if slot not in SLOT_NAMES)
    if invalid:
        raise ObjectMemoryContractError(f"{label} contains unknown slots: {invalid!r}")
    return slots


def _validate_swap_sources(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ObjectMemoryContractError("swap_sources must be a mapping")
    copied = dict(value)
    if not copied:
        raise ObjectMemoryContractError("swap mode requires at least one redirected slot")
    for target, source in copied.items():
        if target not in SLOT_NAMES or source not in SLOT_NAMES:
            raise ObjectMemoryContractError(
                "swap_sources keys and values must be declared object-memory slots"
            )
        if target == source:
            raise ObjectMemoryContractError(
                f"swap_sources contains an identity redirect for {target!r}"
            )
    return {name: copied[name] for name in SLOT_NAMES if name in copied}


@dataclass(frozen=True)
class ObjectMemoryInvocation:
    """One immutable structural binding for an active processor call."""

    source_masks: Mapping[str, Any]
    target_responsibility_masks: Mapping[str, Any]
    mode: str
    drop_slots: tuple[str, ...]
    swap_sources: Mapping[str, str]
    rank: int
    ulysses_size: int
    invocation_token: str

    def source_slot_for(self, target_slot: str) -> Optional[str]:
        if self.mode == "drop" and target_slot in self.drop_slots:
            return None
        if self.mode == "swap":
            return self.swap_sources.get(target_slot, target_slot)
        return target_slot


_CURRENT_INVOCATION: ContextVar[Optional[ObjectMemoryInvocation]] = ContextVar(
    "bernini_object_memory_invocation_v1", default=None
)


def _build_invocation(
    *,
    source_masks: Mapping[str, Any],
    target_responsibility_masks: Mapping[str, Any],
    mode: str,
    drop_slots: Sequence[str],
    swap_sources: Mapping[str, str],
    rank: Any,
    ulysses_size: Any,
    invocation_token: str,
) -> ObjectMemoryInvocation:
    source = _validate_slot_keys(source_masks, label="source_masks")
    target = _validate_slot_keys(
        target_responsibility_masks, label="target_responsibility_masks"
    )
    if mode not in MEMORY_MODES:
        raise ObjectMemoryContractError(
            f"memory mode must be one of {MEMORY_MODES}, got {mode!r}"
        )
    dropped = _validate_slot_sequence(drop_slots, label="drop_slots")
    swaps = dict(swap_sources)
    if mode == "read":
        if dropped or swaps:
            raise ObjectMemoryContractError(
                "read mode forbids drop_slots and swap_sources controls"
            )
    elif mode == "drop":
        if not dropped or swaps:
            raise ObjectMemoryContractError(
                "drop mode requires drop_slots and forbids swap_sources"
            )
    else:
        if dropped:
            raise ObjectMemoryContractError("swap mode forbids drop_slots")
        swaps = _validate_swap_sources(swaps)
    runtime_rank = _exact_nonnegative_int(rank, label="declared Ulysses rank")
    runtime_size = _exact_positive_int(
        ulysses_size, label="declared Ulysses size"
    )
    if runtime_rank >= runtime_size:
        raise ObjectMemoryContractError("declared Ulysses rank is outside its group")
    if not isinstance(invocation_token, str) or not invocation_token:
        raise ObjectMemoryContractError("invocation_token must be a non-empty string")
    return ObjectMemoryInvocation(
        source_masks=MappingProxyType(source),
        target_responsibility_masks=MappingProxyType(target),
        mode=mode,
        drop_slots=dropped,
        swap_sources=MappingProxyType(swaps),
        rank=runtime_rank,
        ulysses_size=runtime_size,
        invocation_token=invocation_token,
    )


@contextmanager
def object_memory_invocation(
    *,
    source_masks: Mapping[str, Any],
    target_responsibility_masks: Mapping[str, Any],
    mode: str = "read",
    drop_slots: Sequence[str] = (),
    swap_sources: Mapping[str, str] = MappingProxyType({}),
    rank: int = 0,
    ulysses_size: int = 1,
    invocation_token: str,
) -> Iterator[ObjectMemoryInvocation]:
    """Bind masks and a typed read/drop/swap intervention for one call scope."""

    if _CURRENT_INVOCATION.get() is not None:
        raise ObjectMemoryContractError("nested object-memory invocations are forbidden")
    invocation = _build_invocation(
        source_masks=source_masks,
        target_responsibility_masks=target_responsibility_masks,
        mode=mode,
        drop_slots=drop_slots,
        swap_sources=swap_sources,
        rank=rank,
        ulysses_size=ulysses_size,
        invocation_token=invocation_token,
    )
    token = _CURRENT_INVOCATION.set(invocation)
    try:
        yield invocation
    finally:
        _CURRENT_INVOCATION.reset(token)


def current_object_memory_invocation() -> ObjectMemoryInvocation:
    invocation = _CURRENT_INVOCATION.get()
    if invocation is None:
        raise ObjectMemoryContractError(
            "positive-gate object-memory call lacks an explicit invocation"
        )
    return invocation


def object_memory_contract(
    *, gate: Any, block_indices: Sequence[int] = DEFAULT_BLOCK_INDICES
) -> dict[str, Any]:
    """Describe the implemented local core and its unresolved integration P0s."""

    fixed_gate = validate_gate(gate)
    if isinstance(block_indices, (str, bytes)) or not isinstance(
        block_indices, Sequence
    ):
        raise ObjectMemoryContractError("block_indices must be a sequence")
    indices = tuple(
        _exact_nonnegative_int(item, label="block index") for item in block_indices
    )
    if not indices or len(set(indices)) != len(indices):
        raise ObjectMemoryContractError(
            "block_indices must be non-empty and contain no duplicates"
        )
    value: dict[str, Any] = {
        "schema_version": CORE_SCHEMA,
        "status": "local_tensor_core_not_integrated_not_production",
        "production_ready": False,
        "trained_parameters": 0,
        "attention": "self_attention_attn1_only",
        "packed_sequence": "one_equal_length_[source,target]_sample",
        "slots": list(SLOT_NAMES),
        "memory_modes": list(MEMORY_MODES),
        "fixed_gate": fixed_gate,
        "zero_gate": "immediate_exact_single_call_delegate_to_original_processor",
        "memory_read": (
            "position_free_target_Q_by_source_masked_position_free_K_and_source_V"
        ),
        "write_scope": "target_responsibility_rows_only",
        "source_rows_directly_written": False,
        "official_base_path": [
            "base_processor._project_qkv",
            "official_varlen_attention",
            "official_gather_heads_scatter_seq_if_ulysses",
            "attn.to_out",
        ],
        "position_policy": {
            "input_qk_state": "official_post_rope",
            "memory_qk_state": "unit_rotary_phase_removed_in_float64_complex_math",
            "value_state": "official_projected_unrotated_value",
        },
        "block_indices": list(indices),
        "fail_closed": [
            "cross_attention_or_extra_attention_mask",
            "non_batch1_or_non_equal_source_target_layout",
            "qkv_shape_dtype_device_mismatch",
            "non_complex_non_unit_or_shape_mismatched_rotary",
            "mask_key_shape_dtype_device_or_empty_mismatch",
            "declared_vs_runtime_rank_or_world_size_mismatch",
            "missing_or_nested_positive_gate_invocation",
        ],
        "unresolved_p0": [
            "native_world4_ulysses_forward_and_inverse_parity",
            "cross_rank_mask_content_consensus",
            "gradient_checkpoint_context_rebinding",
            "infer_lora_runner_condition_abi_and_receipt_wiring",
        ],
        "unresolved_p1": [
            "trained_slot_encoder_and_gate",
            "external_wrong_object_donor_kv_for_swap",
            "target_responsibility_prediction_without_oracle_tracks",
        ],
    }
    value["contract_digest"] = _object_sha256(value)
    return value


def _validate_mask(
    mask: Any, *, label: str, expected_tokens: int, device: Any
) -> Any:
    torch = __import__("torch")
    if not isinstance(mask, torch.Tensor):
        raise ObjectMemoryContractError(f"{label} must be a torch.Tensor")
    if mask.ndim != 1 or int(mask.numel()) != expected_tokens:
        raise ObjectMemoryContractError(
            f"{label} must have shape [{expected_tokens}], got {tuple(mask.shape)!r}"
        )
    if mask.dtype != torch.bool:
        raise ObjectMemoryContractError(f"{label} must use torch.bool")
    if mask.device != device:
        raise ObjectMemoryContractError(
            f"{label} device {mask.device} differs from projected Q/K/V device {device}"
        )
    if mask.requires_grad:
        raise ObjectMemoryContractError(f"{label} must not require gradients")
    if not bool(mask.any().detach().cpu().item()):
        raise ObjectMemoryContractError(f"{label} must select at least one token")
    return mask.contiguous()


def _validate_and_remove_rotary(tensor: Any, rotary_emb: Any, *, label: str) -> Any:
    """Invert Bernini's unit RoPE in the same float64 complex coordinate system."""

    torch = __import__("torch")
    if tensor.ndim != 4 or int(tensor.shape[0]) != 1:
        raise ObjectMemoryContractError(f"{label} must have shape [1,S,H,D]")
    if int(tensor.shape[3]) % 2:
        raise ObjectMemoryContractError(f"{label} head dimension must be even")
    try:
        replay.validate_projected_rotary_embedding(
            rotary_emb,
            projected_shape=tuple(int(item) for item in tensor.shape),
            projected_device=tensor.device,
        )
    except replay.SourceKVReplayContractError as error:
        raise ObjectMemoryContractError(str(error)) from error
    finite_rotary = torch.isfinite(rotary_emb.real).all().logical_and(
        torch.isfinite(rotary_emb.imag).all()
    )
    if not bool(finite_rotary.detach().cpu().item()):
        raise ObjectMemoryContractError("rotary_emb contains non-finite values")
    magnitude = rotary_emb.abs().to(torch.float64)
    if not bool(
        torch.allclose(
            magnitude,
            torch.ones_like(magnitude),
            rtol=1e-10,
            atol=1e-10,
        )
    ):
        raise ObjectMemoryContractError(
            "position-free memory requires unit-modulus rotary phases"
        )
    as_complex = torch.view_as_complex(
        tensor.to(torch.float64).unflatten(3, (-1, 2)).contiguous()
    )
    phase = rotary_emb.to(torch.complex128)
    unrotated = torch.view_as_real(as_complex * phase.conj()).flatten(3, 4)
    return unrotated.type_as(tensor)


class ObjectMemorySelfAttnProcessorV1:
    """Official-path Bernini ``attn1`` wrapper with typed masked memory reads."""

    def __init__(
        self,
        base_processor: Any,
        *,
        block_index: int,
        gate: Any,
        varlen_attention_fn: Optional[Callable[..., Any]] = None,
        get_parallel_state_fn: Optional[Callable[[], Any]] = None,
        gather_heads_scatter_seq_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not callable(base_processor):
            raise ObjectMemoryContractError("base attn1 processor must be callable")
        self.gate = validate_gate(gate)
        if self.gate > 0.0 and not callable(
            getattr(base_processor, "_project_qkv", None)
        ):
            raise ObjectMemoryContractError(
                "positive-gate base attn1 processor lacks official _project_qkv"
            )
        self.block_index = _exact_nonnegative_int(block_index, label="block index")
        self.base_processor = base_processor
        self._varlen_attention_fn = varlen_attention_fn
        self._get_parallel_state_fn = get_parallel_state_fn
        self._gather_heads_scatter_seq_fn = gather_heads_scatter_seq_fn
        self.call_count = 0
        self.zero_gate_delegations = 0
        self.active_calls = 0
        self.memory_varlen_calls = 0
        self.mode_counts = {mode: 0 for mode in MEMORY_MODES}
        self.slot_read_counts = {slot: 0 for slot in SLOT_NAMES}
        self.source_slot_use_counts = {slot: 0 for slot in SLOT_NAMES}
        self.drop_counts = {slot: 0 for slot in SLOT_NAMES}
        self.swap_count = 0
        self.saw_ulysses = False
        self.last_invocation_token: Optional[str] = None
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
        # This branch is deliberately first.  It does not validate, project, or
        # import runtime operators; it calls the original processor once with
        # the exact current ABI arguments and returns its result unchanged.
        if self.gate == 0.0:
            self.call_count += 1
            self.zero_gate_delegations += 1
            return self.base_processor(
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

        del (
            text_features_length,
            split_hidden_states_seq_len,
            cu_seqlens_k_cross_cache,
            cu_seqlens_q_cross_cache,
            max_seqlen_k_cross_cache,
            max_seqlen_q_cross_cache,
        )
        invocation = current_object_memory_invocation()
        if encoder_hidden_states is not None:
            raise ObjectMemoryContractError(
                "object memory may only wrap attn1 self-attention"
            )
        if attention_mask is not None:
            raise ObjectMemoryContractError(
                "an extra attention mask makes object responsibility ambiguous"
            )
        if getattr(hidden_states, "ndim", None) != 3 or int(
            hidden_states.shape[0]
        ) != 1:
            raise ObjectMemoryContractError(
                "object memory requires hidden_states shaped [1,L,D]"
            )
        try:
            replay.require_rotary_embedding(rotary_emb)
        except replay.SourceKVReplayContractError as error:
            raise ObjectMemoryContractError(str(error)) from error

        query, key, value = self.base_processor._project_qkv(
            attn,
            hidden_states,
            None,
            rotary_emb,
            origin_hidden_states_seq_len,
            False,
        )
        try:
            projected_shape = replay.projected_qkv_shape(query, key, value)
            replay.validate_projected_rotary_embedding(
                rotary_emb,
                projected_shape=projected_shape,
                projected_device=query.device,
            )
        except replay.SourceKVReplayContractError as error:
            raise ObjectMemoryContractError(str(error)) from error
        _, gathered_length, _, _ = projected_shape
        try:
            source_tokens = replay.validate_equal_pair_layout(
                gathered_sequence_length=gathered_length,
                batch_image_vae_seqlen=batch_image_vae_seqlen,
                cu_seqlens_q_cache=cu_seqlens_q_cache,
                max_seqlen_q_cache=max_seqlen_q_cache,
                origin_hidden_states_seq_len=origin_hidden_states_seq_len,
            )
        except replay.SourceKVReplayContractError as error:
            raise ObjectMemoryContractError(str(error)) from error

        varlen_fn, state_fn, inverse_fn = self._runtime_ops()
        try:
            ulysses_enabled, runtime_rank, runtime_size = replay.parallel_identity(
                state_fn()
            )
        except replay.SourceKVReplayContractError as error:
            raise ObjectMemoryContractError(str(error)) from error
        if (runtime_rank, runtime_size) != (
            invocation.rank,
            invocation.ulysses_size,
        ):
            raise ObjectMemoryContractError(
                "declared invocation rank/Ulysses size differs from runtime: "
                f"declared=({invocation.rank},{invocation.ulysses_size}), "
                f"runtime=({runtime_rank},{runtime_size})"
            )

        target_tokens = gathered_length - source_tokens
        source_masks = {
            slot: _validate_mask(
                invocation.source_masks[slot],
                label=f"source_masks[{slot!r}]",
                expected_tokens=source_tokens,
                device=query.device,
            )
            for slot in SLOT_NAMES
        }
        target_masks = {
            slot: _validate_mask(
                invocation.target_responsibility_masks[slot],
                label=f"target_responsibility_masks[{slot!r}]",
                expected_tokens=target_tokens,
                device=query.device,
            )
            for slot in SLOT_NAMES
        }

        torch = __import__("torch")
        if not bool(
            torch.isfinite(query).all()
            .logical_and(torch.isfinite(key).all())
            .logical_and(torch.isfinite(value).all())
            .detach()
            .cpu()
            .item()
        ):
            raise ObjectMemoryContractError("projected Q/K/V contain non-finite values")
        query_position_free = _validate_and_remove_rotary(
            query, rotary_emb, label="query"
        )
        key_position_free = _validate_and_remove_rotary(key, rotary_emb, label="key")

        query_for_dtype = query
        query_packed = query.squeeze(0).contiguous()
        key_packed = key.squeeze(0).contiguous()
        value_packed = value.squeeze(0).contiguous()
        base_output = varlen_fn(
            query_packed,
            key_packed,
            value_packed,
            cu_seqlens_q=cu_seqlens_q_cache,
            cu_seqlens_k=cu_seqlens_q_cache,
            max_seqlen_q=max_seqlen_q_cache,
            max_seqlen_k=max_seqlen_q_cache,
            causal=False,
        )
        if tuple(int(item) for item in base_output.shape) != tuple(
            int(item) for item in query_packed.shape
        ):
            raise ObjectMemoryContractError(
                "official base varlen attention output shape differs from query"
            )

        query_pf = query_position_free.squeeze(0).contiguous()
        key_pf = key_position_free.squeeze(0).contiguous()
        residual = torch.zeros_like(base_output)
        for target_slot in SLOT_NAMES:
            source_slot = invocation.source_slot_for(target_slot)
            if source_slot is None:
                self.drop_counts[target_slot] += 1
                continue
            source_mask = source_masks[source_slot]
            target_mask = target_masks[target_slot]
            source_indices = torch.nonzero(source_mask, as_tuple=False).flatten()
            target_local_indices = torch.nonzero(
                target_mask, as_tuple=False
            ).flatten()
            target_global_indices = target_local_indices + source_tokens
            source_count = int(source_indices.numel())
            target_count = int(target_local_indices.numel())
            cu_source = torch.tensor(
                [0, source_count], dtype=torch.int32, device=query.device
            )
            cu_target = torch.tensor(
                [0, target_count], dtype=torch.int32, device=query.device
            )
            memory = varlen_fn(
                query_pf.index_select(0, target_global_indices).contiguous(),
                key_pf.index_select(0, source_indices).contiguous(),
                value_packed.index_select(0, source_indices).contiguous(),
                cu_seqlens_q=cu_target,
                cu_seqlens_k=cu_source,
                max_seqlen_q=target_count,
                max_seqlen_k=source_count,
                causal=False,
            )
            expected_memory_shape = (
                target_count,
                int(query_packed.shape[1]),
                int(query_packed.shape[2]),
            )
            if tuple(int(item) for item in memory.shape) != expected_memory_shape:
                raise ObjectMemoryContractError(
                    f"memory read for {target_slot!r} returned shape "
                    f"{tuple(memory.shape)!r}, expected {expected_memory_shape!r}"
                )
            if not bool(torch.isfinite(memory).all().detach().cpu().item()):
                raise ObjectMemoryContractError(
                    f"memory read for {target_slot!r} contains non-finite values"
                )
            residual.index_add_(0, target_global_indices, memory)
            self.memory_varlen_calls += 1
            self.slot_read_counts[target_slot] += 1
            self.source_slot_use_counts[source_slot] += 1
            if source_slot != target_slot:
                self.swap_count += 1

        if bool(residual[:source_tokens].count_nonzero().detach().cpu().item()):
            raise ObjectMemoryContractError(
                "internal error: object-memory residual wrote source rows"
            )
        combined = base_output + self.gate * residual
        if not bool(torch.isfinite(combined).all().detach().cpu().item()):
            raise ObjectMemoryContractError(
                "combined object-memory attention output contains non-finite values"
            )
        output = combined.unsqueeze(0)
        if ulysses_enabled:
            self.saw_ulysses = True
            output = inverse_fn(output, head_dim=2, seq_dim=1)
        if getattr(output, "ndim", None) != 4:
            raise ObjectMemoryContractError(
                "attention output must be [1,S,H,D] before head flattening"
            )
        output = output.flatten(2, 3).contiguous().type_as(query_for_dtype)
        output = attn.to_out[0](output)
        output = attn.to_out[1](output)
        if not bool(torch.isfinite(output).all().detach().cpu().item()):
            raise ObjectMemoryContractError(
                "projected object-memory processor output contains non-finite values"
            )

        self.call_count += 1
        self.active_calls += 1
        self.mode_counts[invocation.mode] += 1
        self.last_invocation_token = invocation.invocation_token
        self.last_source_tokens = source_tokens
        return output

    def statistics(self) -> dict[str, Any]:
        return {
            "schema_version": CORE_SCHEMA,
            "block_index": self.block_index,
            "fixed_gate": self.gate,
            "call_count": self.call_count,
            "zero_gate_delegations": self.zero_gate_delegations,
            "active_calls": self.active_calls,
            "memory_varlen_calls": self.memory_varlen_calls,
            "mode_counts": dict(self.mode_counts),
            "slot_read_counts": dict(self.slot_read_counts),
            "source_slot_use_counts": dict(self.source_slot_use_counts),
            "drop_counts": dict(self.drop_counts),
            "swap_count": self.swap_count,
            "ulysses_observed": self.saw_ulysses,
            "last_invocation_token": self.last_invocation_token,
            "last_source_tokens": self.last_source_tokens,
            "production_ready": False,
        }


__all__ = [
    "CORE_SCHEMA",
    "DEFAULT_BLOCK_INDICES",
    "MEMORY_MODES",
    "ObjectMemoryContractError",
    "ObjectMemoryInvocation",
    "ObjectMemorySelfAttnProcessorV1",
    "SLOT_NAMES",
    "current_object_memory_invocation",
    "object_memory_contract",
    "object_memory_invocation",
    "validate_gate",
]
