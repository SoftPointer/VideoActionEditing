#!/usr/bin/env python3
"""Observer-only Bernini attn2 capture with an explicit 64-span null bank.

The wrapper returns the exact official processor output object.  It performs a
second detached official QKV projection only while the authenticated v15
source-observer context is active.  It stores role, legacy-control, shuffled,
and every preregistered null-span affinity.  No collective occurs inside
attn2; the SP4 harness gathers the explicit 75-channel tensor afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch

try:
    from . import source_owned_role_locator_v15 as locator
    from . import source_owned_role_null_registry_v15b_r6 as null_registry
except ImportError:  # pragma: no cover - flat AUH deployment
    import source_owned_role_locator_v15 as locator
    import source_owned_role_null_registry_v15b_r6 as null_registry


SCHEMA_VERSION = "bernini-source-owned-role-null-bank-observer-v15b-r6"
NULL_SPAN_COUNT = null_registry.SPAN_COUNT
EXPECTED_BLOCK_COUNT = locator.EXPECTED_BLOCK_COUNT


class NullBankObserverV15BR6Error(RuntimeError):
    """Fail-closed r6 observer/shard/assembly violation."""


def _shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, torch.Tensor):
        raise NullBankObserverV15BR6Error("expected tensor")
    return tuple(int(item) for item in value.shape)


def _exact_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(item not in "0123456789abcdef" for item in value)
    ):
        raise NullBankObserverV15BR6Error(f"{label} is not lowercase SHA-256")
    return value


@dataclass(frozen=True)
class NullBankAffinityShardV15BR6:
    event_id: str
    source_text_provenance_sha256: str
    null_registry_sha256: str
    step_index: int
    block_index: int
    role_names: tuple[str, ...]
    layout: locator.UlyssesVisualShard
    affinity: torch.Tensor
    legacy_null_affinity: torch.Tensor
    shuffled_affinity: torch.Tensor
    null_span_affinity: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id:
            raise NullBankObserverV15BR6Error("shard event is invalid")
        _exact_sha(self.source_text_provenance_sha256, "source text provenance")
        if _exact_sha(self.null_registry_sha256, "null registry") != null_registry.REGISTRY_SHA256:
            raise NullBankObserverV15BR6Error("shard null registry differs")
        if (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or self.step_index < 0
            or isinstance(self.block_index, bool)
            or not isinstance(self.block_index, int)
            or not 0 <= self.block_index < EXPECTED_BLOCK_COUNT
            or not isinstance(self.layout, locator.UlyssesVisualShard)
            or not isinstance(self.role_names, tuple)
            or not self.role_names
            or len(set(self.role_names)) != len(self.role_names)
        ):
            raise NullBankObserverV15BR6Error("shard scalar authority differs")
        valid = self.layout.valid_local_tokens
        contracts = (
            (self.affinity, (len(self.role_names), valid)),
            (self.legacy_null_affinity, (valid,)),
            (self.shuffled_affinity, (len(self.role_names), valid)),
            (self.null_span_affinity, (NULL_SPAN_COUNT, valid)),
        )
        for tensor, expected in contracts:
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float32
                or tensor.requires_grad
                or tensor.grad_fn is not None
                or not tensor.is_contiguous()
                or _shape(tensor) != expected
                or not bool(torch.isfinite(tensor).all().item())
            ):
                raise NullBankObserverV15BR6Error("shard tensor contract differs")
        if any(tensor.device != self.affinity.device for tensor, _expected in contracts):
            raise NullBankObserverV15BR6Error("shard tensor devices differ")

    @property
    def channel_count(self) -> int:
        return 2 * len(self.role_names) + 1 + NULL_SPAN_COUNT

    def padded_collective_tensor(self) -> torch.Tensor:
        roles = len(self.role_names)
        result = self.affinity.new_zeros(
            (self.channel_count, self.layout.padded_local_tokens)
        )
        stop = self.layout.valid_local_tokens
        result[:roles, :stop].copy_(self.affinity)
        result[roles, :stop].copy_(self.legacy_null_affinity)
        result[roles + 1 : 2 * roles + 1, :stop].copy_(self.shuffled_affinity)
        result[2 * roles + 1 :, :stop].copy_(self.null_span_affinity)
        return result

    def collective_metadata(self) -> Mapping[str, Any]:
        roles = len(self.role_names)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "event_id": self.event_id,
            "source_text_provenance_sha256": self.source_text_provenance_sha256,
            "null_registry_sha256": self.null_registry_sha256,
            "step_index": self.step_index,
            "block_index": self.block_index,
            "role_names": list(self.role_names),
            "height": self.layout.geometry.height,
            "width": self.layout.geometry.width,
            "phases": self.layout.geometry.phases,
            "rank": self.layout.rank,
            "size": self.layout.size,
            "global_start": self.layout.global_start,
            "global_stop": self.layout.global_stop,
            "padded_local_tokens": self.layout.padded_local_tokens,
            "valid_local_tokens": self.layout.valid_local_tokens,
            "null_span_count": NULL_SPAN_COUNT,
            "collective_channels": {
                "real": [0, roles],
                "legacy_null": roles,
                "cyclic_shuffled": [roles + 1, 2 * roles + 1],
                "null_spans": [2 * roles + 1, 2 * roles + 1 + NULL_SPAN_COUNT],
            },
        }
        return {**payload, "metadata_sha256": locator.object_sha256(payload)}

    @classmethod
    def from_collective(
        cls,
        tensor: torch.Tensor,
        metadata: Mapping[str, Any],
    ) -> "NullBankAffinityShardV15BR6":
        required = {
            "schema_version",
            "event_id",
            "source_text_provenance_sha256",
            "null_registry_sha256",
            "step_index",
            "block_index",
            "role_names",
            "height",
            "width",
            "phases",
            "rank",
            "size",
            "global_start",
            "global_stop",
            "padded_local_tokens",
            "valid_local_tokens",
            "null_span_count",
            "collective_channels",
            "metadata_sha256",
        }
        if not isinstance(metadata, Mapping) or set(metadata) != required:
            raise NullBankObserverV15BR6Error("collective metadata fields differ")
        payload = dict(metadata)
        digest = payload.pop("metadata_sha256", None)
        if locator.object_sha256(payload) != _exact_sha(digest, "metadata SHA"):
            raise NullBankObserverV15BR6Error("collective metadata SHA differs")
        if payload["schema_version"] != SCHEMA_VERSION:
            raise NullBankObserverV15BR6Error("collective schema differs")
        geometry = locator.SourceVisualGeometry(
            height=metadata["height"],
            width=metadata["width"],
            phases=metadata["phases"],
        )
        layout = locator.UlyssesVisualShard(
            geometry=geometry, rank=metadata["rank"], size=metadata["size"]
        )
        roles = len(metadata["role_names"])
        channels = {
            "real": [0, roles],
            "legacy_null": roles,
            "cyclic_shuffled": [roles + 1, 2 * roles + 1],
            "null_spans": [2 * roles + 1, 2 * roles + 1 + NULL_SPAN_COUNT],
        }
        if (
            metadata["null_span_count"] != NULL_SPAN_COUNT
            or metadata["collective_channels"] != channels
            or metadata["global_start"] != layout.global_start
            or metadata["global_stop"] != layout.global_stop
            or metadata["padded_local_tokens"] != layout.padded_local_tokens
            or metadata["valid_local_tokens"] != layout.valid_local_tokens
            or _shape(tensor)
            != (2 * roles + 1 + NULL_SPAN_COUNT, layout.padded_local_tokens)
        ):
            raise NullBankObserverV15BR6Error("collective geometry/channels differ")
        stop = layout.valid_local_tokens
        return cls(
            event_id=metadata["event_id"],
            source_text_provenance_sha256=metadata["source_text_provenance_sha256"],
            null_registry_sha256=metadata["null_registry_sha256"],
            step_index=metadata["step_index"],
            block_index=metadata["block_index"],
            role_names=tuple(metadata["role_names"]),
            layout=layout,
            affinity=tensor[:roles, :stop].detach().float().contiguous(),
            legacy_null_affinity=tensor[roles, :stop].detach().float().contiguous(),
            shuffled_affinity=tensor[roles + 1 : 2 * roles + 1, :stop]
            .detach()
            .float()
            .contiguous(),
            null_span_affinity=tensor[2 * roles + 1 :, :stop]
            .detach()
            .float()
            .contiguous(),
        )


class NullBankCaptureBankV15BR6(locator.SourceRoleCaptureBank):
    """Invocation-compatible capture bank with an r6-only shard type."""

    def __init__(
        self,
        selected_block_indices: Sequence[int],
        *,
        registry: null_registry.NullTokenRegistryV15BR6,
    ) -> None:
        super().__init__(selected_block_indices)
        if not isinstance(registry, null_registry.NullTokenRegistryV15BR6):
            raise NullBankObserverV15BR6Error("capture bank lacks null registry")
        self.registry = registry
        self._null_shards: dict[
            tuple[str, int, int, int], NullBankAffinityShardV15BR6
        ] = {}

    def capture(self, shard: NullBankAffinityShardV15BR6) -> None:
        if not isinstance(shard, NullBankAffinityShardV15BR6):
            raise NullBankObserverV15BR6Error("capture requires an r6 shard")
        if (
            shard.block_index not in self.selected_block_indices
            or shard.null_registry_sha256 != self.registry.registry_sha256
        ):
            raise NullBankObserverV15BR6Error("capture block/registry differs")
        key = (shard.event_id, shard.step_index, shard.block_index, shard.layout.rank)
        if key in self._null_shards:
            raise NullBankObserverV15BR6Error("duplicate r6 capture")
        self._null_shards[key] = shard
        self.capture_count += 1

    def shards_for(
        self, *, event_id: str, step_index: int, block_index: int
    ) -> tuple[NullBankAffinityShardV15BR6, ...]:
        rows = [
            value
            for (event, step, block, _rank), value in self._null_shards.items()
            if (event, step, block) == (event_id, step_index, block_index)
        ]
        return tuple(sorted(rows, key=lambda item: item.layout.rank))

    def receipt(self) -> Mapping[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "selected_block_indices": list(self.selected_block_indices),
            "null_registry_sha256": self.registry.registry_sha256,
            "null_span_count": NULL_SPAN_COUNT,
            "capture_count": self.capture_count,
            "stored_shards": len(self._null_shards),
            "implicit_collective_calls": 0,
            "route_authorized": False,
        }


def source_role_null_bank_affinity_v15b_r6(
    query: torch.Tensor,
    key: torch.Tensor,
    roles: Sequence[locator.LockedRoleSpan],
    registry: null_registry.NullTokenRegistryV15BR6,
    *,
    valid_local_tokens: int,
    active_source_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    real, legacy_null, shuffled = locator._source_role_affinity(
        query,
        key,
        roles,
        valid_local_tokens=valid_local_tokens,
        active_source_tokens=active_source_tokens,
    )
    if (
        not isinstance(registry, null_registry.NullTokenRegistryV15BR6)
        or active_source_tokens != null_registry.ACTIVE_TOKEN_COUNT
        or int(key.shape[1]) < active_source_tokens
    ):
        raise NullBankObserverV15BR6Error("null-bank text authority differs")
    q = torch.nn.functional.normalize(
        query[:, :valid_local_tokens].detach().float(), dim=-1, eps=1e-12
    )
    span_keys = []
    for span in registry.spans:
        tokens = key[:, span.token_start : span.token_end].detach().float()
        span_key = torch.nn.functional.normalize(tokens, dim=-1, eps=1e-12).mean(dim=1)
        span_keys.append(torch.nn.functional.normalize(span_key, dim=-1, eps=1e-12).squeeze(0))
    keys = torch.stack(span_keys, dim=0)
    null_spans = (
        torch.einsum("blhd,rhd->brlh", q, keys)
        .mean(dim=-1)
        .squeeze(0)
        .detach()
        .float()
        .contiguous()
    )
    if (
        _shape(null_spans) != (NULL_SPAN_COUNT, valid_local_tokens)
        or not bool(torch.isfinite(null_spans).all().item())
    ):
        raise NullBankObserverV15BR6Error("explicit null-span affinity differs")
    return real, legacy_null, shuffled, null_spans


class SourceOwnedRoleNullBankAttn2ObserverV15BR6:
    """Exact-output wrapper around one official Bernini attn2 processor."""

    def __init__(
        self,
        base_processor: Any,
        *,
        block_index: int,
        capture_bank: NullBankCaptureBankV15BR6,
    ) -> None:
        processor_type = type(base_processor)
        if (
            not callable(base_processor)
            or not callable(getattr(base_processor, "_project_qkv", None))
            or processor_type.__module__ != locator.OFFICIAL_ATTN2_PROCESSOR_MODULE
            or processor_type.__name__ != locator.OFFICIAL_ATTN2_PROCESSOR_CLASS
            or block_index not in capture_bank.selected_block_indices
        ):
            raise NullBankObserverV15BR6Error("base processor/block contract differs")
        self.base_processor = base_processor
        self.block_index = block_index
        self.capture_bank = capture_bank
        self.base_calls = 0
        self.observer_calls = 0

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
        self.base_calls += 1
        invocation = locator.current_source_role_observer()
        if invocation is None:
            return output
        if invocation.capture_bank is not self.capture_bank:
            raise NullBankObserverV15BR6Error("observer capture-bank ownership differs")
        locator._validate_frozen_stateless_attention(attn)
        if encoder_hidden_states is None or attention_mask is not None or rotary_emb is not None:
            raise NullBankObserverV15BR6Error("r6 requires unmasked non-RoPE attn2")
        provenance = invocation.source_text_provenance
        provenance.validate_rank_local_view(
            encoder_hidden_states, require_sp_view=invocation.ulysses.size > 1
        )
        if (
            provenance.tokenization.input_ids_sha256
            != self.capture_bank.registry.token_input_ids_sha256
            or provenance.tokenization.attention_mask_sha256
            != self.capture_bank.registry.token_attention_mask_sha256
            or provenance.tokenization.active_token_count != null_registry.ACTIVE_TOKEN_COUNT
            or not isinstance(hidden_states, torch.Tensor)
            or hidden_states.ndim != 3
            or int(hidden_states.shape[0]) != 1
            or int(hidden_states.shape[1]) != invocation.ulysses.padded_local_tokens
            or hidden_states.device != encoder_hidden_states.device
            or hidden_states.dtype != encoder_hidden_states.dtype
            or not bool(torch.isfinite(hidden_states).all().item())
        ):
            raise NullBankObserverV15BR6Error("r6 source text/visual binding differs")
        if (
            locator._one_length(batch_image_vae_seqlen, label="batch_image_vae_seqlen")
            != invocation.geometry.global_tokens
            or origin_hidden_states_seq_len != invocation.geometry.global_tokens
            or locator._one_length(text_features_length, label="text_features_length")
            != provenance.renderer_text_length
        ):
            raise NullBankObserverV15BR6Error("r6 source length binding differs")
        fork_devices: list[int] = []
        if hidden_states.device.type == "cuda":
            device_index = hidden_states.device.index
            fork_devices.append(
                int(torch.cuda.current_device()) if device_index is None else device_index
            )
        with torch.random.fork_rng(devices=fork_devices, enabled=True), torch.no_grad():
            query, key, _value = self.base_processor._project_qkv(
                attn,
                hidden_states.detach(),
                encoder_hidden_states.detach(),
                None,
                invocation.geometry.global_tokens,
                True,
            )
            real, legacy_null, shuffled, null_spans = (
                source_role_null_bank_affinity_v15b_r6(
                    query,
                    key,
                    invocation.event_spec.roles,
                    self.capture_bank.registry,
                    valid_local_tokens=invocation.ulysses.valid_local_tokens,
                    active_source_tokens=provenance.tokenization.active_token_count,
                )
            )
        self.capture_bank.capture(
            NullBankAffinityShardV15BR6(
                event_id=invocation.event_spec.event_id,
                source_text_provenance_sha256=provenance.receipt_sha256,
                null_registry_sha256=self.capture_bank.registry.registry_sha256,
                step_index=invocation.step_index,
                block_index=self.block_index,
                role_names=invocation.event_spec.role_names,
                layout=invocation.ulysses,
                affinity=real,
                legacy_null_affinity=legacy_null,
                shuffled_affinity=shuffled,
                null_span_affinity=null_spans,
            )
        )
        self.observer_calls += 1
        return output

    def statistics(self) -> Mapping[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "block_index": self.block_index,
            "base_calls": self.base_calls,
            "observer_calls": self.observer_calls,
            "null_span_count": NULL_SPAN_COUNT,
            "output_modified": False,
            "implicit_collective_calls": 0,
        }


@dataclass(frozen=True)
class GlobalNullBankAffinityV15BR6:
    event_id: str
    source_text_provenance_sha256: str
    null_registry_sha256: str
    step_index: int
    block_index: int
    role_names: tuple[str, ...]
    geometry: locator.SourceVisualGeometry
    affinity: torch.Tensor
    legacy_null_affinity: torch.Tensor
    shuffled_affinity: torch.Tensor
    null_span_affinity: torch.Tensor

    def __post_init__(self) -> None:
        phases, height, width = (
            self.geometry.phases,
            self.geometry.height,
            self.geometry.width,
        )
        contracts = (
            (self.affinity, (len(self.role_names), phases, height, width)),
            (self.legacy_null_affinity, (phases, height, width)),
            (self.shuffled_affinity, (len(self.role_names), phases, height, width)),
            (self.null_span_affinity, (NULL_SPAN_COUNT, phases, height, width)),
        )
        for tensor, expected in contracts:
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float32
                or tensor.requires_grad
                or tensor.grad_fn is not None
                or not tensor.is_contiguous()
                or _shape(tensor) != expected
                or not bool(torch.isfinite(tensor).all().item())
            ):
                raise NullBankObserverV15BR6Error("global tensor contract differs")
        if any(tensor.device != self.affinity.device for tensor, _expected in contracts):
            raise NullBankObserverV15BR6Error("global tensor devices differ")
        _exact_sha(self.source_text_provenance_sha256, "global provenance")
        if self.null_registry_sha256 != null_registry.REGISTRY_SHA256:
            raise NullBankObserverV15BR6Error("global registry differs")


def assemble_global_null_bank_affinity_v15b_r6(
    shards: Sequence[NullBankAffinityShardV15BR6],
) -> GlobalNullBankAffinityV15BR6:
    rows = tuple(shards)
    if not rows or not all(isinstance(item, NullBankAffinityShardV15BR6) for item in rows):
        raise NullBankObserverV15BR6Error("global assembly lacks r6 shards")
    first = rows[0]
    if len(rows) != first.layout.size or tuple(item.layout.rank for item in rows) != tuple(
        range(first.layout.size)
    ):
        raise NullBankObserverV15BR6Error("global assembly rank closure differs")
    shared = (
        first.event_id,
        first.source_text_provenance_sha256,
        first.null_registry_sha256,
        first.step_index,
        first.block_index,
        first.role_names,
        first.layout.geometry,
        first.layout.size,
        first.affinity.device,
    )
    if any(
        (
            item.event_id,
            item.source_text_provenance_sha256,
            item.null_registry_sha256,
            item.step_index,
            item.block_index,
            item.role_names,
            item.layout.geometry,
            item.layout.size,
            item.affinity.device,
        )
        != shared
        for item in rows
    ):
        raise NullBankObserverV15BR6Error("global shard authority differs")
    expected_start = 0
    for item in rows:
        if item.layout.global_start != expected_start:
            raise NullBankObserverV15BR6Error("global shard intervals are not contiguous")
        expected_start = item.layout.global_stop
    if expected_start != first.layout.geometry.global_tokens:
        raise NullBankObserverV15BR6Error("global shard interval closure differs")
    geometry = first.layout.geometry
    roles = len(first.role_names)
    real = geometry.reshape_global(
        torch.cat([item.affinity for item in rows], dim=1).contiguous(), leading=roles
    ).contiguous()
    shuffled = geometry.reshape_global(
        torch.cat([item.shuffled_affinity for item in rows], dim=1).contiguous(),
        leading=roles,
    ).contiguous()
    null_spans = geometry.reshape_global(
        torch.cat([item.null_span_affinity for item in rows], dim=1).contiguous(),
        leading=NULL_SPAN_COUNT,
    ).contiguous()
    legacy_null = torch.cat(
        [item.legacy_null_affinity for item in rows], dim=0
    ).reshape(geometry.phases, geometry.height, geometry.width).contiguous()
    return GlobalNullBankAffinityV15BR6(
        event_id=first.event_id,
        source_text_provenance_sha256=first.source_text_provenance_sha256,
        null_registry_sha256=first.null_registry_sha256,
        step_index=first.step_index,
        block_index=first.block_index,
        role_names=first.role_names,
        geometry=geometry,
        affinity=real,
        legacy_null_affinity=legacy_null,
        shuffled_affinity=shuffled,
        null_span_affinity=null_spans,
    )


@dataclass
class NullBankObserverPatchHandleV15BR6:
    transformer: Any
    block_indices: tuple[int, ...]
    processors: tuple[SourceOwnedRoleNullBankAttn2ObserverV15BR6, ...]
    original_processors: tuple[Any, ...]
    restored: bool = False

    def restore(self) -> None:
        if self.restored:
            return
        for index, processor in zip(self.block_indices, self.processors):
            if getattr(self.transformer.blocks[index].attn2, "processor", None) is not processor:
                raise NullBankObserverV15BR6Error("observer processor changed behind handle")
        for index, original in zip(self.block_indices, self.original_processors):
            attn2 = self.transformer.blocks[index].attn2
            setter = getattr(attn2, "set_processor", None)
            if callable(setter):
                setter(original)
            else:
                attn2.processor = original
        self.restored = True

    def receipt(self) -> Mapping[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "block_indices": list(self.block_indices),
            "attn2_only": True,
            "parameters_added": 0,
            "output_modified": False,
            "restored": self.restored,
            "processors": [item.statistics() for item in self.processors],
        }

    def __enter__(self) -> "NullBankObserverPatchHandleV15BR6":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.restore()


def install_source_owned_role_null_bank_observer_v15b_r6(
    model: Any,
    *,
    capture_bank: NullBankCaptureBankV15BR6,
) -> NullBankObserverPatchHandleV15BR6:
    if not isinstance(capture_bank, NullBankCaptureBankV15BR6):
        raise NullBankObserverV15BR6Error("installation requires r6 capture bank")
    transformer = locator._resolve_wan_transformer(model)
    originals: list[Any] = []
    processors: list[SourceOwnedRoleNullBankAttn2ObserverV15BR6] = []
    installed: list[int] = []
    try:
        for index in capture_bank.selected_block_indices:
            attn2 = transformer.blocks[index].attn2
            original = getattr(attn2, "processor", None)
            processor = SourceOwnedRoleNullBankAttn2ObserverV15BR6(
                original, block_index=index, capture_bank=capture_bank
            )
            setter = getattr(attn2, "set_processor", None)
            if callable(setter):
                setter(processor)
            else:
                attn2.processor = processor
            if getattr(attn2, "processor", None) is not processor:
                raise NullBankObserverV15BR6Error("observer installation did not stick")
            originals.append(original)
            processors.append(processor)
            installed.append(index)
    except Exception:
        for index, original in zip(installed, originals):
            attn2 = transformer.blocks[index].attn2
            setter = getattr(attn2, "set_processor", None)
            if callable(setter):
                setter(original)
            else:
                attn2.processor = original
        raise
    return NullBankObserverPatchHandleV15BR6(
        transformer=transformer,
        block_indices=tuple(installed),
        processors=tuple(processors),
        original_processors=tuple(originals),
    )


__all__ = [
    "GlobalNullBankAffinityV15BR6",
    "NULL_SPAN_COUNT",
    "NullBankAffinityShardV15BR6",
    "NullBankCaptureBankV15BR6",
    "NullBankObserverV15BR6Error",
    "SCHEMA_VERSION",
    "SourceOwnedRoleNullBankAttn2ObserverV15BR6",
    "assemble_global_null_bank_affinity_v15b_r6",
    "install_source_owned_role_null_bank_observer_v15b_r6",
    "source_role_null_bank_affinity_v15b_r6",
]
