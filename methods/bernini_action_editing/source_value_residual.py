#!/usr/bin/env python3
"""Counterfactual source-value residual for Bernini-R self-attention.

V9 hard replay replaced the source prefix of both K and V with tensors from a
source-only/no-op trajectory.  The frozen dog oracle falsified that carrier:
replacing all layers changed the entire scene and subject.  This V10 tensor
core leaves the official full-pair Q/K route and base attention output intact.
For target queries only it adds the exact value-interpolation residual

    dV = [V_cache - V_current_source ; 0_target]
    dM = Attn(Q_target, K_current_full, dV)
    M_target' = M_target + gate * dM.

Because attention is linear in V and Bernini's pinned varlen attention has no
attention dropout, this is the counterfactual obtained by interpolating only
the source values while keeping every current Q/K logit and the full-pair
softmax normalization fixed.  Within that block, source-query output is not
directly modified.  This is not an end-to-end source-stream invariant: a
target residual from an earlier block can feed back through a later block's
official bidirectional self-attention.

The main operator is ``full_k_value``.  Two deliberately named diagnostics
are also implemented: source-only renormalisation and centered cached-K/V.
They may be useful falsifiers but are not the proposed training operator.

The source cache identity, capture lifetime, post-RoPE validation, checkpoint
context, and Ulysses rank contract are reused from :mod:`source_kv_replay`.
Only detached ``entry.value`` participates in the main residual; cached K is
ignored there.  A fixed zero gate delegates the replay call to the untouched
official processor, providing a byte-exact frozen-oracle control.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Iterator, Mapping, Optional

import source_kv_replay as replay


CORE_SCHEMA = "bernini-counterfactual-source-value-residual-v10-core-v1"
MAIN_OPERATOR = "full_k_value"
SOURCE_NORMALIZED_DIAGNOSTIC = "source_normalized_value"
CACHED_KV_DIAGNOSTIC = "centered_cached_kv"
OPERATORS = (
    MAIN_OPERATOR,
    SOURCE_NORMALIZED_DIAGNOSTIC,
    CACHED_KV_DIAGNOSTIC,
)


class SourceValueResidualContractError(replay.SourceKVReplayContractError):
    """Raised instead of silently changing the advertised V10 operator."""


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
        raise SourceValueResidualContractError(
            f"source-value contract is not canonical JSON: {error}"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def validate_fixed_gate(value: Any) -> float:
    if isinstance(value, bool):
        raise SourceValueResidualContractError("fixed gate must be numeric")
    try:
        gate = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise SourceValueResidualContractError("fixed gate must be numeric") from error
    if not math.isfinite(gate) or gate < 0.0 or gate > 1.0:
        raise SourceValueResidualContractError(
            "frozen-oracle fixed gate must be finite and in [0,1]"
        )
    return gate


def source_value_residual_contract(
    *,
    selection: str,
    operator: str = MAIN_OPERATOR,
    gate: float,
    num_blocks: int = replay.EXPECTED_BLOCK_COUNT,
) -> dict[str, Any]:
    indices = replay.resolve_block_indices(num_blocks, selection)
    if operator not in OPERATORS:
        raise SourceValueResidualContractError(
            f"operator must be one of {OPERATORS}, got {operator!r}"
        )
    fixed_gate = validate_fixed_gate(gate)
    value: dict[str, Any] = {
        "schema_version": CORE_SCHEMA,
        "status": "integrated_frozen_oracle_tensor_core_not_trained_method",
        "attention": "self_attention_attn1_only",
        "capture": "detached_source_only_noop_post_project_qkv_value",
        "operator": operator,
        "main_operator": MAIN_OPERATOR,
        "operator_is_main": operator == MAIN_OPERATOR,
        "active_operator_uses_cached_key": operator == CACHED_KV_DIAGNOSTIC,
        "fixed_gate": fixed_gate,
        "zero_gate_replay": "delegate_exact_official_processor",
        "sequence_order": ["source", "target"],
        "source_query_output": "locally_unchanged_within_each_patched_block",
        "cross_block_source_stream_invariant_claimed": False,
        "target_query_base_output": "official_full_pair_output_unchanged",
        "target_query_residual": (
            "Attn(Q_target,K_current_full,"
            "[V_cached_source-V_current_source;zeros_target])"
            if operator == MAIN_OPERATOR
            else "diagnostic_operator_explicitly_not_main"
        ),
        "main_route_invariants": {
            "current_full_pair_queries": True,
            "current_full_pair_keys": True,
            "current_full_pair_attention_logits": True,
            "current_full_pair_softmax_normalization": True,
            "current_source_vs_target_attention_mass": True,
            "cached_key_used": False,
        },
        "attention_linearity_requirement": {
            "attention_dropout": 0,
            "varlen_value_linearity": True,
        },
        "carrier_runtime_dependency_schema": replay.CORE_SCHEMA,
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


def _rms_from_sum(sum_square: Any, count: int) -> Optional[float]:
    if sum_square is None or count <= 0:
        return None
    return math.sqrt(float(sum_square.detach().cpu().item()) / float(count))


class SourceValueResidualSelfAttnProcessor:
    """Official-path wrapper implementing a fixed-gate V10 oracle arm."""

    def __init__(
        self,
        base_processor: Any,
        *,
        block_index: int,
        cache_bank: replay.SourceKVCacheBank,
        operator: str = MAIN_OPERATOR,
        gate: float,
        varlen_attention_fn: Optional[Callable[..., Any]] = None,
        get_parallel_state_fn: Optional[Callable[[], Any]] = None,
        gather_heads_scatter_seq_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not callable(getattr(base_processor, "_project_qkv", None)):
            raise SourceValueResidualContractError(
                "base attn1 processor lacks official _project_qkv"
            )
        if not callable(base_processor):
            raise SourceValueResidualContractError(
                "base attn1 processor must be callable for zero-gate delegation"
            )
        try:
            index = int(block_index)
        except (TypeError, ValueError, OverflowError) as error:
            raise SourceValueResidualContractError(
                "block index must be an integer"
            ) from error
        if isinstance(block_index, bool) or index < 0 or float(block_index) != index:
            raise SourceValueResidualContractError(
                "block index must be a non-negative integer"
            )
        if not isinstance(cache_bank, replay.SourceKVCacheBank):
            raise SourceValueResidualContractError("cache bank has the wrong type")
        if index not in cache_bank.selected_block_indices:
            raise SourceValueResidualContractError(
                f"block {index} is outside the cache bank scope"
            )
        if operator not in OPERATORS:
            raise SourceValueResidualContractError(
                f"operator must be one of {OPERATORS}, got {operator!r}"
            )
        self.base_processor = base_processor
        self.block_index = index
        self.cache_bank = cache_bank
        self.operator = operator
        self.gate = validate_fixed_gate(gate)
        self._varlen_attention_fn = varlen_attention_fn
        self._get_parallel_state_fn = get_parallel_state_fn
        self._gather_heads_scatter_seq_fn = gather_heads_scatter_seq_fn
        self.capture_calls = 0
        self.replay_calls = 0
        self.zero_gate_delegations = 0
        self.residual_varlen_calls = 0
        self.branch_counts: dict[str, int] = {}
        self.execution_phase_counts = {
            phase: 0 for phase in replay.EXECUTION_PHASES
        }
        self.saw_ulysses = False
        self.last_source_tokens: Optional[int] = None
        self._metric_calls = 0
        self._metric_elements = 0
        self._base_target_sq_sum = None
        self._value_delta_sq_sum = None
        self._delta_memory_sq_sum = None
        self._gated_delta_sq_sum = None
        self._finite = None
        self._combined_finite = None
        self._projected_output_finite = None

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

    def _record_branch(self, invocation: replay.ReplayInvocation, phase: str) -> None:
        self.branch_counts[invocation.branch_tag] = (
            self.branch_counts.get(invocation.branch_tag, 0) + 1
        )
        self.execution_phase_counts[phase] += 1
        if invocation.mode == replay.CAPTURE_MODE:
            self.capture_calls += 1
        else:
            self.replay_calls += 1

    def _accumulate_metrics(
        self, *, base_target: Any, value_delta: Any, delta_memory: Any
    ) -> None:
        torch = __import__("torch")
        tensors = (base_target, value_delta, delta_memory)
        finite = torch.stack(
            [torch.isfinite(tensor.detach()).all() for tensor in tensors]
        ).all()
        self._finite = finite if self._finite is None else self._finite.logical_and(finite)
        base_sum = base_target.detach().float().square().sum()
        value_sum = value_delta.detach().float().square().sum()
        delta_sum = delta_memory.detach().float().square().sum()
        gated_sum = (delta_memory.detach().float() * self.gate).square().sum()
        self._base_target_sq_sum = (
            base_sum
            if self._base_target_sq_sum is None
            else self._base_target_sq_sum + base_sum
        )
        self._value_delta_sq_sum = (
            value_sum
            if self._value_delta_sq_sum is None
            else self._value_delta_sq_sum + value_sum
        )
        self._delta_memory_sq_sum = (
            delta_sum
            if self._delta_memory_sq_sum is None
            else self._delta_memory_sq_sum + delta_sum
        )
        self._gated_delta_sq_sum = (
            gated_sum
            if self._gated_delta_sq_sum is None
            else self._gated_delta_sq_sum + gated_sum
        )
        self._metric_calls += 1
        self._metric_elements += int(delta_memory.numel())

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
        invocation = replay.current_source_kv_invocation()
        if invocation.cache_bank is not self.cache_bank:
            raise SourceValueResidualContractError(
                "processor and invocation use different cache banks"
            )
        phase = replay.current_execution_phase(invocation)

        # The fixed frozen-oracle zero arm must execute the exact official pair
        # processor.  Cache completeness and step identity were already checked
        # when the replay invocation was entered; capture still runs below.
        if invocation.mode == replay.REPLAY_MODE and self.gate == 0.0:
            _, state_fn, _ = self._runtime_ops()
            ulysses_enabled, runtime_rank, runtime_size = replay.parallel_identity(
                state_fn()
            )
            if (runtime_rank, runtime_size) != (
                invocation.rank,
                invocation.ulysses_size,
            ):
                raise SourceValueResidualContractError(
                    "zero-gate invocation rank/Ulysses size differs from runtime"
                )
            if ulysses_enabled:
                self.saw_ulysses = True
            self.zero_gate_delegations += 1
            self._record_branch(invocation, phase)
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
        if encoder_hidden_states is not None:
            raise SourceValueResidualContractError(
                "source-value residual may only wrap attn1 self-attention"
            )
        if attention_mask is not None:
            raise SourceValueResidualContractError(
                "an extra attention mask makes the residual route ambiguous"
            )
        if getattr(hidden_states, "ndim", None) != 3 or int(hidden_states.shape[0]) != 1:
            raise SourceValueResidualContractError(
                "source-value residual requires hidden_states shaped [1,L,D]"
            )
        replay.require_rotary_embedding(rotary_emb)
        query, key, value = self.base_processor._project_qkv(
            attn,
            hidden_states,
            None,
            rotary_emb,
            origin_hidden_states_seq_len,
            False,
        )
        projected_shape = replay.projected_qkv_shape(query, key, value)
        _, gathered_length, _, _ = projected_shape
        replay.validate_projected_rotary_embedding(
            rotary_emb,
            projected_shape=projected_shape,
            projected_device=query.device,
        )
        varlen_fn, state_fn, inverse_fn = self._runtime_ops()
        parallel_state = state_fn()
        ulysses_enabled, runtime_rank, runtime_size = replay.parallel_identity(
            parallel_state
        )
        if (runtime_rank, runtime_size) != (invocation.rank, invocation.ulysses_size):
            raise SourceValueResidualContractError(
                "outer invocation rank/Ulysses size differs from runtime state"
            )

        if invocation.mode == replay.CAPTURE_MODE:
            source_tokens = replay.validate_source_only_layout(
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
            entry = None
        else:
            source_tokens = replay.validate_equal_pair_layout(
                gathered_sequence_length=gathered_length,
                batch_image_vae_seqlen=batch_image_vae_seqlen,
                cu_seqlens_q_cache=cu_seqlens_q_cache,
                max_seqlen_q_cache=max_seqlen_q_cache,
                origin_hidden_states_seq_len=origin_hidden_states_seq_len,
            )
            entry = self.cache_bank.lookup(
                invocation=invocation,
                block_index=self.block_index,
                current_key=key,
                current_value=value,
                source_tokens=source_tokens,
            )

        query_for_dtype = query
        query = query.squeeze(0).contiguous()
        key = key.squeeze(0).contiguous()
        value = value.squeeze(0).contiguous()
        base_output = varlen_fn(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q_cache,
            cu_seqlens_k=cu_seqlens_q_cache,
            max_seqlen_q=max_seqlen_q_cache,
            max_seqlen_k=max_seqlen_q_cache,
            causal=False,
        )
        if tuple(int(item) for item in base_output.shape) != tuple(
            int(item) for item in query.shape
        ):
            raise SourceValueResidualContractError(
                "official base attention output shape differs from query"
            )

        if entry is not None:
            torch = __import__("torch")
            cu_half = torch.tensor(
                [0, source_tokens], dtype=torch.int32, device=query.device
            )
            cached_key = entry.key.squeeze(0).contiguous()
            cached_value = entry.value.squeeze(0).contiguous()
            value_delta = cached_value - value[:source_tokens]
            if self.operator == MAIN_OPERATOR:
                delta_value_full = torch.cat(
                    (value_delta, torch.zeros_like(value[source_tokens:])), dim=0
                ).contiguous()
                delta_target = varlen_fn(
                    query[source_tokens:],
                    key,
                    delta_value_full,
                    cu_seqlens_q=cu_half,
                    cu_seqlens_k=cu_seqlens_q_cache,
                    max_seqlen_q=source_tokens,
                    max_seqlen_k=max_seqlen_q_cache,
                    causal=False,
                )
                self.residual_varlen_calls += 1
            elif self.operator == SOURCE_NORMALIZED_DIAGNOSTIC:
                delta_target = varlen_fn(
                    query[source_tokens:],
                    key[:source_tokens],
                    value_delta.contiguous(),
                    cu_seqlens_q=cu_half,
                    cu_seqlens_k=cu_half,
                    max_seqlen_q=source_tokens,
                    max_seqlen_k=source_tokens,
                    causal=False,
                )
                self.residual_varlen_calls += 1
            else:
                cached_output = varlen_fn(
                    query[source_tokens:],
                    cached_key,
                    cached_value,
                    cu_seqlens_q=cu_half,
                    cu_seqlens_k=cu_half,
                    max_seqlen_q=source_tokens,
                    max_seqlen_k=source_tokens,
                    causal=False,
                )
                current_output = varlen_fn(
                    query[source_tokens:],
                    key[:source_tokens],
                    value[:source_tokens],
                    cu_seqlens_q=cu_half,
                    cu_seqlens_k=cu_half,
                    max_seqlen_q=source_tokens,
                    max_seqlen_k=source_tokens,
                    causal=False,
                )
                delta_target = cached_output - current_output
                self.residual_varlen_calls += 2
            expected_half_shape = tuple(
                int(item) for item in query[source_tokens:].shape
            )
            if tuple(int(item) for item in delta_target.shape) != expected_half_shape:
                raise SourceValueResidualContractError(
                    "target residual attention output shape differs from target query"
                )
            residual_pair = torch.cat(
                (torch.zeros_like(base_output[:source_tokens]), delta_target), dim=0
            )
            self._accumulate_metrics(
                base_target=base_output[source_tokens:],
                value_delta=value_delta,
                delta_memory=delta_target,
            )
            base_output = base_output + self.gate * residual_pair
            combined_finite = torch.isfinite(base_output.detach()).all()
            self._combined_finite = (
                combined_finite
                if self._combined_finite is None
                else self._combined_finite.logical_and(combined_finite)
            )

        output = base_output.unsqueeze(0)
        if ulysses_enabled:
            self.saw_ulysses = True
            output = inverse_fn(output, head_dim=2, seq_dim=1)
        if getattr(output, "ndim", None) != 4:
            raise SourceValueResidualContractError(
                "attention output must be [1,S,H,D] before head flattening"
            )
        output = output.flatten(2, 3).contiguous().type_as(query_for_dtype)
        output = attn.to_out[0](output)
        output = attn.to_out[1](output)
        torch = __import__("torch")
        projected_finite = torch.isfinite(output.detach()).all()
        self._projected_output_finite = (
            projected_finite
            if self._projected_output_finite is None
            else self._projected_output_finite.logical_and(projected_finite)
        )
        self.last_source_tokens = source_tokens
        self._record_branch(invocation, phase)
        return output

    def statistics(self) -> dict[str, Any]:
        base_rms = _rms_from_sum(self._base_target_sq_sum, self._metric_elements)
        gated_rms = _rms_from_sum(self._gated_delta_sq_sum, self._metric_elements)
        return {
            "block_index": self.block_index,
            "operator": self.operator,
            "fixed_gate": self.gate,
            "capture_calls": self.capture_calls,
            "replay_calls": self.replay_calls,
            "zero_gate_delegations": self.zero_gate_delegations,
            "residual_varlen_calls": self.residual_varlen_calls,
            "branch_counts": dict(sorted(self.branch_counts.items())),
            "execution_phase_counts": dict(self.execution_phase_counts),
            "last_source_tokens": self.last_source_tokens,
            "ulysses_observed": self.saw_ulysses,
            "metrics": {
                "calls": self._metric_calls,
                "accumulated_elements_including_recompute": self._metric_elements,
                "all_finite": (
                    None
                    if self._finite is None
                    else bool(self._finite.detach().cpu().item())
                ),
                "combined_attention_output_all_finite": (
                    None
                    if self._combined_finite is None
                    else bool(self._combined_finite.detach().cpu().item())
                ),
                "projected_output_all_finite": (
                    None
                    if self._projected_output_finite is None
                    else bool(
                        self._projected_output_finite.detach().cpu().item()
                    )
                ),
                "base_target_rms": base_rms,
                "source_value_delta_rms": _rms_from_sum(
                    self._value_delta_sq_sum, self._metric_elements
                ),
                "delta_memory_rms": _rms_from_sum(
                    self._delta_memory_sq_sum, self._metric_elements
                ),
                "gated_delta_rms": gated_rms,
                "gated_to_base_rms_ratio": (
                    None
                    if base_rms in (None, 0.0) or gated_rms is None
                    else gated_rms / base_rms
                ),
            },
        }


@dataclass
class SourceValueResidualPatchHandle:
    transformer: Any
    selection: str
    operator: str
    gate: float
    indices: tuple[int, ...]
    cache_bank: replay.SourceKVCacheBank
    processors: tuple[SourceValueResidualSelfAttnProcessor, ...]
    original_processors: tuple[Any, ...]
    restored: bool = False

    def restore(self) -> None:
        if self.restored:
            return
        blocks = self.transformer.blocks
        for index, installed in zip(self.indices, self.processors):
            if getattr(blocks[index].attn1, "processor", None) is not installed:
                raise SourceValueResidualContractError(
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
        value = source_value_residual_contract(
            selection=self.selection,
            operator=self.operator,
            gate=self.gate,
            num_blocks=len(self.transformer.blocks),
        )
        value["runtime"] = {
            "installed_block_count": len(self.indices),
            "restored": self.restored,
            "cache": self.cache_bank.receipt(),
            "per_block": [processor.statistics() for processor in self.processors],
        }
        value["runtime_digest"] = _object_sha256(value["runtime"])
        return value


def install_source_value_residual(
    model: Any,
    *,
    selection: str = "late",
    operator: str = MAIN_OPERATOR,
    gate: float,
    cache_bank: Optional[replay.SourceKVCacheBank] = None,
    processor_factory: Optional[
        Callable[
            [Any, int, replay.SourceKVCacheBank],
            SourceValueResidualSelfAttnProcessor,
        ]
    ] = None,
) -> SourceValueResidualPatchHandle:
    transformer = replay.resolve_wan_transformer(model)
    indices = replay.resolve_block_indices(len(transformer.blocks), selection)
    fixed_gate = validate_fixed_gate(gate)
    if operator not in OPERATORS:
        raise SourceValueResidualContractError(
            f"operator must be one of {OPERATORS}, got {operator!r}"
        )
    bank = cache_bank if cache_bank is not None else replay.SourceKVCacheBank(indices)
    if not isinstance(bank, replay.SourceKVCacheBank):
        raise SourceValueResidualContractError("provided cache bank has wrong type")
    if bank.selected_block_indices != indices:
        raise SourceValueResidualContractError(
            "provided cache bank scope differs from selected transformer blocks"
        )
    originals: list[Any] = []
    installed: list[SourceValueResidualSelfAttnProcessor] = []
    installed_indices: list[int] = []
    try:
        for index in indices:
            attn = transformer.blocks[index].attn1
            original = getattr(attn, "processor", None)
            if original is None:
                raise SourceValueResidualContractError(
                    f"block {index} attn1 lacks a processor"
                )
            if isinstance(
                original,
                (SourceValueResidualSelfAttnProcessor, replay.SourceKVReplaySelfAttnProcessor),
            ):
                raise SourceValueResidualContractError(
                    f"block {index} already has an experimental source carrier"
                )
            processor = (
                processor_factory(original, index, bank)
                if processor_factory is not None
                else SourceValueResidualSelfAttnProcessor(
                    original,
                    block_index=index,
                    cache_bank=bank,
                    operator=operator,
                    gate=fixed_gate,
                )
            )
            if not isinstance(processor, SourceValueResidualSelfAttnProcessor):
                raise SourceValueResidualContractError(
                    "processor_factory returned the wrong type"
                )
            if (
                processor.operator != operator
                or processor.gate != fixed_gate
                or processor.block_index != index
                or processor.cache_bank is not bank
                or processor.base_processor is not original
            ):
                raise SourceValueResidualContractError(
                    "processor_factory changed block/cache/base/operator/gate identity"
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
    return SourceValueResidualPatchHandle(
        transformer=transformer,
        selection=selection,
        operator=operator,
        gate=fixed_gate,
        indices=indices,
        cache_bank=bank,
        processors=tuple(installed),
        original_processors=tuple(originals),
    )


@contextmanager
def source_value_residual(
    model: Any,
    *,
    selection: str = "late",
    operator: str = MAIN_OPERATOR,
    gate: float,
) -> Iterator[SourceValueResidualPatchHandle]:
    handle = install_source_value_residual(
        model, selection=selection, operator=operator, gate=gate
    )
    try:
        yield handle
    finally:
        handle.restore()


__all__ = [
    "CACHED_KV_DIAGNOSTIC",
    "CORE_SCHEMA",
    "MAIN_OPERATOR",
    "OPERATORS",
    "SOURCE_NORMALIZED_DIAGNOSTIC",
    "SourceValueResidualContractError",
    "SourceValueResidualPatchHandle",
    "SourceValueResidualSelfAttnProcessor",
    "install_source_value_residual",
    "source_value_residual",
    "source_value_residual_contract",
    "validate_fixed_gate",
]
