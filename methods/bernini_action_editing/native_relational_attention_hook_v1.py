#!/usr/bin/env python3
"""Reversible observer-only Bernini attention hook for relational anchors.

This module wraps the official ``WanAttnProcessor2_0`` instances at blocks
``6/12/18/24``.  The wrappers delegate the official processor exactly once and
return the exact object it returned.  During that same call they temporarily
intercept ``_project_qkv`` to copy:

* attn1 post-Ulysses, post-RoPE Q/K (global visual sequence, rank-local heads),
* attn2 projected Q/K (rank-local visual sequence, replicated text heads).

The pinned Bernini attention ABI does *not* expose the attention probability
tensor: ``varlen_attention`` returns only the value-weighted output.  Replacing
that kernel in order to observe probabilities would invalidate bit-exact P0.
Consequently this module computes an explicitly named
``derived_qk_role_responsibility_proxy`` after the official output exists.  It
uses float32 scaled-QK softmax and an exhaustive text-token role partition.  It
is not represented as an observed backend/FlashAttention weight tensor.

No collective is executed inside either processor.  After the model forward,
the caller must gather four typed rank shards with its already-authenticated
WORLD4 group.  :func:`commit_world4_shards_to_native_bank` validates and joins
the sequence/head layouts, writes four ``NativeBlockCapture`` objects into the
runner's ``InMemoryNativeCaptureBank``, and zeroizes the rank shards.  The
module has no target, decoder, optimizer, adapter, route, or training API.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Iterator, Mapping, Optional, Sequence

import torch

import infer_native_self_generated_relational_graph_observer_v1 as native


METHOD = "bernini-native-relational-read-only-attention-hook-v1"
SCHEMA_VERSION = "bernini-native-relational-read-only-attention-hook-v1"
OFFICIAL_TRANSFORMER_SOURCE_SHA256 = (
    "9fb579611e79e0f534d5d6ccdcd956c35e57b4513c15267e8533ff3832a1f223"
)
OFFICIAL_PROCESSOR_MODULE = "bernini.models.transformer_wan"
OFFICIAL_PROCESSOR_CLASS = "WanAttnProcessor2_0"
BLOCKS = native.BLOCKS
PHASES = native.PHASES
WORLD_SIZE = 4
TOTAL_HEADS = 12
LOCAL_ATTN1_HEADS = TOTAL_HEADS // WORLD_SIZE
HEAD_DIM = 128
EXPECTED_BLOCK_COUNT = 30
RESPONSIBILITY_KIND = "derived_qk_role_responsibility_proxy"
BACKEND_ATTENTION_WEIGHTS_OBSERVED = False
_ROLE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class NativeRelationalAttentionHookError(RuntimeError):
    """Raised instead of accepting an ambiguous hook or WORLD4 layout."""


def _canonical_digest(value: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise NativeRelationalAttentionHookError("value is not canonical JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _exact_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise NativeRelationalAttentionHookError(f"{label} is outside its integer domain")
    return value


def _one_length(value: Any, *, label: str) -> int:
    if isinstance(value, torch.Tensor):
        rows = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = list(value)
    else:
        rows = [value]
    if len(rows) != 1:
        raise NativeRelationalAttentionHookError(f"{label} must contain one length")
    item = rows[0]
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
        if len(item) != 1:
            raise NativeRelationalAttentionHookError(f"{label} nested length differs")
        item = item[0]
    try:
        integer = int(item)
        numeric = float(item)
    except (TypeError, ValueError, OverflowError) as error:
        raise NativeRelationalAttentionHookError(f"{label} is not integral") from error
    if not math.isfinite(numeric) or numeric != float(integer) or integer <= 0:
        raise NativeRelationalAttentionHookError(f"{label} is not positive integral")
    return integer


def _cache_values(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise NativeRelationalAttentionHookError(f"{label} must be one tensor")
    try:
        rows = tuple(int(item) for item in value.detach().cpu().tolist())
    except Exception as error:
        raise NativeRelationalAttentionHookError(f"{label} cannot be read") from error
    return rows


def _finite_detached_contiguous(value: Any, *, label: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or value.requires_grad
        or value.grad_fn is not None
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
    ):
        raise NativeRelationalAttentionHookError(
            f"{label} must be material detached finite contiguous"
        )
    return value


def _zeroize_tensors(values: Sequence[torch.Tensor]) -> None:
    # Captures cloned while the official frozen forward is under
    # ``torch.inference_mode()`` are inference tensors.  PyTorch forbids
    # mutating those later under only ``no_grad``; zeroization must itself run
    # in inference mode.  This remains valid for ordinary detached tensors.
    with torch.inference_mode():
        for value in values:
            if isinstance(value, torch.Tensor) and value.device.type != "meta":
                value.zero_()


def _owned_contiguous_clone(
    value: torch.Tensor, ownership: list[torch.Tensor]
) -> torch.Tensor:
    """Register a clone immediately after its single allocation."""

    result = value.detach().clone(memory_format=torch.contiguous_format)
    ownership.append(result)
    return result


def _owned_contiguous_cat(
    values: Sequence[torch.Tensor],
    *,
    dim: int,
    ownership: list[torch.Tensor],
) -> torch.Tensor:
    """Register a concatenation immediately after its single allocation."""

    # torch.cat materializes a contiguous result; avoid a second unregistered
    # allocation between materialization and ownership registration.
    result = torch.cat(tuple(values), dim=dim)
    ownership.append(result)
    return result


@dataclass(frozen=True)
class ExhaustiveTextRolePartition:
    """Assign every active text key to exactly one named role, including null."""

    role_names: tuple[str, ...]
    token_to_role: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not 2 <= len(self.role_names) <= native.MAX_ROLES
            or len(set(self.role_names)) != len(self.role_names)
            or any(_ROLE_RE.fullmatch(item) is None for item in self.role_names)
        ):
            raise NativeRelationalAttentionHookError("role registry differs")
        if len(self.token_to_role) < len(self.role_names):
            raise NativeRelationalAttentionHookError("active text partition is too short")
        if any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or not 0 <= item < len(self.role_names)
            for item in self.token_to_role
        ):
            raise NativeRelationalAttentionHookError("text token has no exact role owner")
        if set(self.token_to_role) != set(range(len(self.role_names))):
            raise NativeRelationalAttentionHookError("every registered role needs text support")

    @property
    def active_text_tokens(self) -> int:
        return len(self.token_to_role)

    @property
    def digest(self) -> str:
        return _canonical_digest(
            {
                "role_names": list(self.role_names),
                "token_to_role": list(self.token_to_role),
                "exhaustive": True,
            }
        )


@dataclass(frozen=True)
class World4RankLayout:
    """Pinned Bernini-R-1.3B Ulysses layout for one visual token grid."""

    rank: int
    patch_height: int
    patch_width: int
    world_size: int = WORLD_SIZE
    total_heads: int = TOTAL_HEADS
    head_dim: int = HEAD_DIM

    def __post_init__(self) -> None:
        rank = _exact_int(self.rank, label="rank")
        height = _exact_int(self.patch_height, label="patch height", minimum=1)
        width = _exact_int(self.patch_width, label="patch width", minimum=1)
        if self.world_size != WORLD_SIZE or rank >= WORLD_SIZE:
            raise NativeRelationalAttentionHookError("observer requires exact WORLD4 rank layout")
        if self.total_heads != TOTAL_HEADS or self.head_dim != HEAD_DIM:
            raise NativeRelationalAttentionHookError(
                "Bernini-R-1.3B attention geometry differs from 12x128"
            )
        if height * width < 2 or TOTAL_HEADS % WORLD_SIZE:
            raise NativeRelationalAttentionHookError("visual/head geometry is degenerate")

    @property
    def spatial_tokens(self) -> int:
        return self.patch_height * self.patch_width

    @property
    def global_tokens(self) -> int:
        return PHASES * self.spatial_tokens

    @property
    def padded_local_tokens(self) -> int:
        return math.ceil(self.global_tokens / WORLD_SIZE)

    @property
    def global_start(self) -> int:
        return self.rank * self.padded_local_tokens

    @property
    def global_stop(self) -> int:
        return min(self.global_tokens, self.global_start + self.padded_local_tokens)

    @property
    def valid_local_tokens(self) -> int:
        return max(0, self.global_stop - self.global_start)


@dataclass(frozen=True)
class RankCaptureInvocation:
    capture: native.CaptureInvocation
    layout: World4RankLayout
    role_partition: ExhaustiveTextRolePartition
    responsibility_kind: str = RESPONSIBILITY_KIND

    def __post_init__(self) -> None:
        if (
            not isinstance(self.capture, native.CaptureInvocation)
            or not isinstance(self.layout, World4RankLayout)
            or not isinstance(self.role_partition, ExhaustiveTextRolePartition)
        ):
            raise NativeRelationalAttentionHookError("rank capture authority differs")
        if (
            self.capture.patch_height != self.layout.patch_height
            or self.capture.patch_width != self.layout.patch_width
        ):
            raise NativeRelationalAttentionHookError("capture and WORLD4 grid differ")
        if self.responsibility_kind != RESPONSIBILITY_KIND:
            raise NativeRelationalAttentionHookError(
                "backend weights are unavailable; the derived proxy must be explicit"
            )

    @property
    def key(self) -> tuple[str, str, str, int, int]:
        return (*self.capture.key, self.layout.rank)


@dataclass(frozen=True)
class Attn1PostRopeQKRankShard:
    invocation: RankCaptureInvocation
    block_index: int
    query: torch.Tensor
    key: torch.Tensor

    def __post_init__(self) -> None:
        if self.block_index not in BLOCKS:
            raise NativeRelationalAttentionHookError("attn1 block differs")
        query = _finite_detached_contiguous(self.query, label="attn1 query")
        key = _finite_detached_contiguous(self.key, label="attn1 key")
        layout = self.invocation.layout
        expected = (1, layout.global_tokens, LOCAL_ATTN1_HEADS, HEAD_DIM)
        if (
            tuple(query.shape) != expected
            or tuple(key.shape) != expected
            or query.dtype != key.dtype
            or query.device != key.device
        ):
            raise NativeRelationalAttentionHookError(
                "attn1 must be global sequence x rank-local heads after RoPE"
            )

    def zeroize(self) -> None:
        _zeroize_tensors((self.query, self.key))


@dataclass(frozen=True)
class DerivedRoleProxyRankShard:
    invocation: RankCaptureInvocation
    block_index: int
    proxy: torch.Tensor  # [1, valid_local_tokens, roles], float32

    def __post_init__(self) -> None:
        if self.block_index not in BLOCKS:
            raise NativeRelationalAttentionHookError("attn2 proxy block differs")
        proxy = _finite_detached_contiguous(self.proxy, label=RESPONSIBILITY_KIND)
        expected = (
            1,
            self.invocation.layout.valid_local_tokens,
            len(self.invocation.role_partition.role_names),
        )
        if proxy.dtype != torch.float32 or tuple(proxy.shape) != expected:
            raise NativeRelationalAttentionHookError("derived role proxy geometry differs")
        if bool((proxy < 0).any().item()):
            raise NativeRelationalAttentionHookError("derived role proxy is negative")
        mass = proxy.sum(dim=2)
        if not bool(torch.allclose(mass, torch.ones_like(mass), atol=2e-5, rtol=2e-5)):
            raise NativeRelationalAttentionHookError("derived role proxy does not sum to one")

    def zeroize(self) -> None:
        _zeroize_tensors((self.proxy,))


@dataclass(frozen=True)
class World4BlockRankShard:
    invocation: RankCaptureInvocation
    block_index: int
    query: torch.Tensor
    key: torch.Tensor
    derived_qk_role_responsibility_proxy: torch.Tensor

    @classmethod
    def join(
        cls,
        qk: Attn1PostRopeQKRankShard,
        role: DerivedRoleProxyRankShard,
    ) -> "World4BlockRankShard":
        if qk.invocation != role.invocation or qk.block_index != role.block_index:
            raise NativeRelationalAttentionHookError("attn1/attn2 rank shard authority differs")
        return cls(qk.invocation, qk.block_index, qk.query, qk.key, role.proxy)

    def __post_init__(self) -> None:
        Attn1PostRopeQKRankShard(
            self.invocation, self.block_index, self.query, self.key
        )
        DerivedRoleProxyRankShard(
            self.invocation,
            self.block_index,
            self.derived_qk_role_responsibility_proxy,
        )

    def zeroize(self) -> None:
        _zeroize_tensors(
            (self.query, self.key, self.derived_qk_role_responsibility_proxy)
        )

    def collective_metadata(self) -> Mapping[str, Any]:
        """Small JSON authority for tensor-only collectives outside attention."""

        value = _collective_metadata_payload(
            self.invocation, block_index=self.block_index
        )
        return {**value, "metadata_sha256": _canonical_digest(value)}

    def collective_payload_and_zeroize(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
        """Copy padded collective tensors, then clear this local raw shard.

        Q/K payload: ``[2,1,G,3,128]``.  Proxy payload:
        ``[1,ceil(G/4),K]`` with a strictly zero append-padding suffix.
        Neither tensor may be serialized; the caller may only feed them to an
        authenticated WORLD4 tensor collective after the transformer forward.
        """

        qk: Optional[torch.Tensor] = None
        proxy: Optional[torch.Tensor] = None
        succeeded = False
        try:
            layout = self.invocation.layout
            qk = torch.stack((self.query, self.key), dim=0).detach().contiguous()
            proxy = self.derived_qk_role_responsibility_proxy.new_zeros(
                (
                    1,
                    layout.padded_local_tokens,
                    len(self.invocation.role_partition.role_names),
                )
            )
            proxy[:, : layout.valid_local_tokens].copy_(
                self.derived_qk_role_responsibility_proxy
            )
            metadata = self.collective_metadata()
            result = (qk, proxy.detach().contiguous(), metadata)
            succeeded = True
            return result
        finally:
            self.zeroize()
            if not succeeded:
                _zeroize_tensors(
                    tuple(
                        value
                        for value in (qk, proxy)
                        if isinstance(value, torch.Tensor)
                    )
                )


def _collective_metadata_payload(
    invocation: RankCaptureInvocation, *, block_index: int
) -> Mapping[str, Any]:
    layout = invocation.layout
    capture = invocation.capture
    return {
        "schema_version": SCHEMA_VERSION,
        "appearance_id": capture.appearance_id,
        "arm": capture.arm,
        "sigma_band": capture.sigma_cell.band,
        "step_index": capture.sigma_cell.step_index,
        "state_sha256": capture.state_sha256,
        "block_index": block_index,
        "rank": layout.rank,
        "world_size": WORLD_SIZE,
        "patch_height": layout.patch_height,
        "patch_width": layout.patch_width,
        "global_tokens": layout.global_tokens,
        "padded_local_tokens": layout.padded_local_tokens,
        "global_start": layout.global_start,
        "global_stop": layout.global_stop,
        "valid_local_tokens": layout.valid_local_tokens,
        "total_heads": TOTAL_HEADS,
        "local_attn1_heads": LOCAL_ATTN1_HEADS,
        "head_dim": HEAD_DIM,
        "role_names": list(invocation.role_partition.role_names),
        "role_partition_sha256": invocation.role_partition.digest,
        "responsibility_kind": RESPONSIBILITY_KIND,
        "backend_attention_weights_observed": False,
        "collective_location": "after_transformer_forward",
    }


@dataclass
class _PartialBlock:
    qk: Optional[Attn1PostRopeQKRankShard] = None
    role: Optional[DerivedRoleProxyRankShard] = None


_ACTIVE_RANK_CAPTURE: ContextVar[Optional[tuple["InMemoryWorld4RankShardBank", RankCaptureInvocation]]] = (
    ContextVar("bernini_native_relational_rank_capture_v1", default=None)
)


class InMemoryWorld4RankShardBank:
    """Ephemeral rank-shard sink; it intentionally has no serialization API."""

    def __init__(self) -> None:
        self._rows: dict[
            tuple[str, str, str, int, int], dict[int, _PartialBlock]
        ] = {}
        self.attn1_capture_count = 0
        self.proxy_capture_count = 0
        self.taken_rank_count = 0
        self.zeroized_failure_tensor_count = 0

    @contextmanager
    def observe(self, invocation: RankCaptureInvocation) -> Iterator[None]:
        if not isinstance(invocation, RankCaptureInvocation):
            raise NativeRelationalAttentionHookError("rank observer invocation differs")
        if _ACTIVE_RANK_CAPTURE.get() is not None:
            raise NativeRelationalAttentionHookError("nested rank observers are forbidden")
        if invocation.key in self._rows:
            raise NativeRelationalAttentionHookError("duplicate rank observer invocation")
        self._rows[invocation.key] = {block: _PartialBlock() for block in BLOCKS}
        token: Token[Optional[tuple[InMemoryWorld4RankShardBank, RankCaptureInvocation]]] = (
            _ACTIVE_RANK_CAPTURE.set((self, invocation))
        )
        try:
            yield
            rows = self._rows[invocation.key]
            if any(value.qk is None or value.role is None for value in rows.values()):
                raise NativeRelationalAttentionHookError(
                    "rank forward did not close attn1+attn2 at all four blocks"
                )
        except Exception:
            rows = self._rows.pop(invocation.key, {})
            tensors: list[torch.Tensor] = []
            for value in rows.values():
                if value.qk is not None:
                    tensors.extend((value.qk.query, value.qk.key))
                if value.role is not None:
                    tensors.append(value.role.proxy)
            _zeroize_tensors(tensors)
            self.zeroized_failure_tensor_count += len(tensors)
            raise
        finally:
            _ACTIVE_RANK_CAPTURE.reset(token)

    @staticmethod
    def current() -> Optional[tuple["InMemoryWorld4RankShardBank", RankCaptureInvocation]]:
        return _ACTIVE_RANK_CAPTURE.get()

    def _partial(self, block_index: int) -> tuple[RankCaptureInvocation, _PartialBlock]:
        active = self.current()
        if active is None or active[0] is not self:
            raise NativeRelationalAttentionHookError("capture arrived outside its rank context")
        invocation = active[1]
        if block_index not in BLOCKS:
            raise NativeRelationalAttentionHookError("capture block is outside fixed scope")
        return invocation, self._rows[invocation.key][block_index]

    def capture_attn1(self, value: Attn1PostRopeQKRankShard) -> None:
        if not isinstance(value, Attn1PostRopeQKRankShard):
            raise NativeRelationalAttentionHookError("attn1 shard type differs")
        invocation, row = self._partial(value.block_index)
        if value.invocation != invocation or row.qk is not None:
            raise NativeRelationalAttentionHookError("duplicate/foreign attn1 shard")
        row.qk = value
        self.attn1_capture_count += 1

    def capture_proxy(self, value: DerivedRoleProxyRankShard) -> None:
        if not isinstance(value, DerivedRoleProxyRankShard):
            raise NativeRelationalAttentionHookError("role proxy shard type differs")
        invocation, row = self._partial(value.block_index)
        if value.invocation != invocation or row.role is not None:
            raise NativeRelationalAttentionHookError("duplicate/foreign role proxy shard")
        row.role = value
        self.proxy_capture_count += 1

    def take_rank(self, invocation: RankCaptureInvocation) -> tuple[World4BlockRankShard, ...]:
        rows = self._rows.pop(invocation.key, None)
        result: list[World4BlockRankShard] = []
        succeeded = False
        try:
            if rows is None or tuple(sorted(rows)) != BLOCKS:
                raise NativeRelationalAttentionHookError("rank capture is absent")
            for block in BLOCKS:
                row = rows[block]
                if row.qk is None or row.role is None:
                    raise NativeRelationalAttentionHookError("rank capture is incomplete")
                result.append(World4BlockRankShard.join(row.qk, row.role))
            self.taken_rank_count += 1
            succeeded = True
            return tuple(result)
        finally:
            if not succeeded and rows is not None:
                for row in rows.values():
                    if row.qk is not None:
                        row.qk.zeroize()
                    if row.role is not None:
                        row.role.zeroize()
                for value in result:
                    value.zeroize()

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": SCHEMA_VERSION,
            "attn1_post_rope_qk_rank_shards": self.attn1_capture_count,
            RESPONSIBILITY_KIND + "_rank_shards": self.proxy_capture_count,
            "taken_rank_invocations": self.taken_rank_count,
            "resident_rank_invocations": len(self._rows),
            "implicit_collective_calls": 0,
            "persistent_tensor_artifact_created": False,
            "backend_attention_weights_observed": False,
        }
        return {**value, "digest": _canonical_digest(value)}


def _validate_frozen_attention(attn: Any) -> None:
    if not isinstance(attn, torch.nn.Module):
        return
    if attn.training or any(parameter.requires_grad for parameter in attn.parameters()):
        raise NativeRelationalAttentionHookError("observed attention must be frozen/eval")


def _official_processor(value: Any) -> Any:
    value_type = type(value)
    if (
        not callable(value)
        or not callable(getattr(value, "_project_qkv", None))
        or value_type.__module__ != OFFICIAL_PROCESSOR_MODULE
        or value_type.__name__ != OFFICIAL_PROCESSOR_CLASS
        or "_project_qkv" in getattr(value, "__dict__", {})
    ):
        raise NativeRelationalAttentionHookError("attention processor is not pristine official ABI")
    return value


def _delegate_with_same_call_qk(
    base_processor: Any,
    attn: Any,
    hidden_states: torch.Tensor,
    kwargs: Mapping[str, Any],
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    if "_project_qkv" in getattr(base_processor, "__dict__", {}):
        raise NativeRelationalAttentionHookError("projection interceptor is already installed")
    original_projection = base_processor._project_qkv
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    owned_clones: list[torch.Tensor] = []

    def intercept(*args: Any, **inner_kwargs: Any):
        if captured:
            raise NativeRelationalAttentionHookError("official projection executed more than once")
        query, key, value = original_projection(*args, **inner_kwargs)
        captured_query = _owned_contiguous_clone(query, owned_clones)
        captured_key = _owned_contiguous_clone(key, owned_clones)
        captured.append((captured_query, captured_key))
        return query, key, value

    base_processor._project_qkv = intercept
    succeeded = False
    try:
        output = base_processor(attn, hidden_states, **dict(kwargs))
        try:
            delattr(base_processor, "_project_qkv")
        except AttributeError as error:
            raise NativeRelationalAttentionHookError(
                "temporary official projection interception was lost"
            ) from error
        if len(captured) != 1:
            raise NativeRelationalAttentionHookError(
                "official projection was not captured exactly once"
            )
        succeeded = True
        return output, captured[0][0], captured[0][1]
    finally:
        restore_error: Optional[BaseException] = None
        if "_project_qkv" in getattr(base_processor, "__dict__", {}):
            try:
                delattr(base_processor, "_project_qkv")
            except BaseException as error:
                restore_error = error
        if not succeeded:
            _zeroize_tensors(owned_clones)
        if restore_error is not None:
            raise NativeRelationalAttentionHookError(
                "temporary official projection interception could not restore"
            ) from restore_error


def _processor_kwargs(
    *,
    encoder_hidden_states: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    rotary_emb: Optional[torch.Tensor],
    batch_image_vae_seqlen: Any,
    text_features_length: Any,
    origin_hidden_states_seq_len: Optional[int],
    split_hidden_states_seq_len: Optional[int],
    cu_seqlens_q_cache: Any,
    max_seqlen_q_cache: Any,
    cu_seqlens_k_cross_cache: Any,
    cu_seqlens_q_cross_cache: Any,
    max_seqlen_k_cross_cache: Any,
    max_seqlen_q_cross_cache: Any,
) -> Mapping[str, Any]:
    return {
        "encoder_hidden_states": encoder_hidden_states,
        "attention_mask": attention_mask,
        "rotary_emb": rotary_emb,
        "batch_image_vae_seqlen": batch_image_vae_seqlen,
        "text_features_length": text_features_length,
        "origin_hidden_states_seq_len": origin_hidden_states_seq_len,
        "split_hidden_states_seq_len": split_hidden_states_seq_len,
        "cu_seqlens_q_cache": cu_seqlens_q_cache,
        "max_seqlen_q_cache": max_seqlen_q_cache,
        "cu_seqlens_k_cross_cache": cu_seqlens_k_cross_cache,
        "cu_seqlens_q_cross_cache": cu_seqlens_q_cross_cache,
        "max_seqlen_k_cross_cache": max_seqlen_k_cross_cache,
        "max_seqlen_q_cross_cache": max_seqlen_q_cross_cache,
    }


def _active_for(bank: InMemoryWorld4RankShardBank) -> Optional[RankCaptureInvocation]:
    active = bank.current()
    if active is None:
        return None
    if active[0] is not bank:
        raise NativeRelationalAttentionHookError("hook/rank-bank ownership differs")
    return active[1]


def _validate_common_hidden(
    hidden_states: Any, invocation: RankCaptureInvocation
) -> torch.Tensor:
    if (
        not isinstance(hidden_states, torch.Tensor)
        or hidden_states.ndim != 3
        or tuple(hidden_states.shape[:2])
        != (1, invocation.layout.padded_local_tokens)
        or hidden_states.requires_grad
        or hidden_states.grad_fn is not None
        or not bool(torch.isfinite(hidden_states).all().item())
    ):
        raise NativeRelationalAttentionHookError("rank-local frozen hidden stream differs")
    return hidden_states


def _validate_attn1_call(
    invocation: RankCaptureInvocation,
    *,
    encoder_hidden_states: Any,
    attention_mask: Any,
    rotary_emb: Any,
    batch_image_vae_seqlen: Any,
    origin_hidden_states_seq_len: Any,
    split_hidden_states_seq_len: Any,
    cu_seqlens_q_cache: Any,
    max_seqlen_q_cache: Any,
) -> None:
    layout = invocation.layout
    if encoder_hidden_states is not None or attention_mask is not None:
        raise NativeRelationalAttentionHookError("attn1 observer received cross/masked attention")
    if not isinstance(rotary_emb, torch.Tensor) or rotary_emb.device.type == "meta":
        raise NativeRelationalAttentionHookError("attn1 observer requires real rotary phases")
    if _one_length(batch_image_vae_seqlen, label="batch_image_vae_seqlen") != layout.global_tokens:
        raise NativeRelationalAttentionHookError("attn1 global visual length differs")
    if origin_hidden_states_seq_len != layout.global_tokens:
        raise NativeRelationalAttentionHookError("attn1 origin visual length differs")
    if split_hidden_states_seq_len != layout.padded_local_tokens:
        raise NativeRelationalAttentionHookError("attn1 padded local length differs")
    if _cache_values(cu_seqlens_q_cache, label="self cu_seqlens") != (0, layout.global_tokens):
        raise NativeRelationalAttentionHookError("attn1 self-attention boundaries differ")
    if int(max_seqlen_q_cache) != layout.global_tokens:
        raise NativeRelationalAttentionHookError("attn1 max sequence length differs")


def _validate_attn2_call(
    invocation: RankCaptureInvocation,
    *,
    encoder_hidden_states: Any,
    attention_mask: Any,
    rotary_emb: Any,
    batch_image_vae_seqlen: Any,
    text_features_length: Any,
    origin_hidden_states_seq_len: Any,
    split_hidden_states_seq_len: Any,
    cu_seqlens_k_cross_cache: Any,
    cu_seqlens_q_cross_cache: Any,
    max_seqlen_k_cross_cache: Any,
    max_seqlen_q_cross_cache: Any,
) -> None:
    layout = invocation.layout
    text = invocation.role_partition.active_text_tokens
    if (
        not isinstance(encoder_hidden_states, torch.Tensor)
        or encoder_hidden_states.ndim != 3
        or tuple(encoder_hidden_states.shape[:2]) != (1, text)
        or encoder_hidden_states.requires_grad
        or encoder_hidden_states.grad_fn is not None
    ):
        raise NativeRelationalAttentionHookError("attn2 replicated text stream differs")
    if attention_mask is not None or rotary_emb is not None:
        raise NativeRelationalAttentionHookError("official attn2 must be unmasked/non-RoPE")
    if _one_length(batch_image_vae_seqlen, label="batch_image_vae_seqlen") != layout.global_tokens:
        raise NativeRelationalAttentionHookError("attn2 global visual length differs")
    if _one_length(text_features_length, label="text_features_length") != text:
        raise NativeRelationalAttentionHookError("attn2 active text length differs")
    if origin_hidden_states_seq_len != layout.global_tokens:
        raise NativeRelationalAttentionHookError("attn2 origin visual length differs")
    if split_hidden_states_seq_len != layout.padded_local_tokens:
        raise NativeRelationalAttentionHookError("attn2 padded local length differs")
    if _cache_values(cu_seqlens_q_cross_cache, label="cross query cu_seqlens") != (
        0,
        layout.valid_local_tokens,
    ):
        raise NativeRelationalAttentionHookError("attn2 rank-local query boundaries differ")
    if _cache_values(cu_seqlens_k_cross_cache, label="cross key cu_seqlens") != (0, text):
        raise NativeRelationalAttentionHookError("attn2 text key boundaries differ")
    if int(max_seqlen_q_cross_cache) != layout.valid_local_tokens:
        raise NativeRelationalAttentionHookError("attn2 max query length differs")
    if int(max_seqlen_k_cross_cache) != text:
        raise NativeRelationalAttentionHookError("attn2 max key length differs")


def _derive_qk_role_proxy(
    query: torch.Tensor,
    key: torch.Tensor,
    invocation: RankCaptureInvocation,
    *,
    attn: Any,
) -> torch.Tensor:
    layout = invocation.layout
    text = invocation.role_partition.active_text_tokens
    if (
        tuple(query.shape) != (1, layout.padded_local_tokens, TOTAL_HEADS, HEAD_DIM)
        or tuple(key.shape) != (1, text, TOTAL_HEADS, HEAD_DIM)
        or query.dtype != key.dtype
        or query.device != key.device
    ):
        raise NativeRelationalAttentionHookError("same-call attn2 Q/K geometry differs")
    expected_scale = HEAD_DIM ** -0.5
    declared_scale = getattr(attn, "scale", expected_scale)
    try:
        scale = float(declared_scale)
    except (TypeError, ValueError, OverflowError) as error:
        raise NativeRelationalAttentionHookError("attn2 scale is invalid") from error
    if not math.isclose(scale, expected_scale, rel_tol=0.0, abs_tol=1e-12):
        raise NativeRelationalAttentionHookError("attn2 scale differs from official SDPA/Flash ABI")
    valid_query = query[:, : layout.valid_local_tokens].detach().float()
    text_key = key.detach().float()
    logits = torch.einsum("blhd,bshd->blhs", valid_query, text_key) * expected_scale
    token_probability = torch.softmax(logits, dim=-1).mean(dim=2)
    owner = torch.tensor(
        invocation.role_partition.token_to_role,
        dtype=torch.long,
        device=query.device,
    ).reshape(1, 1, text)
    result = token_probability.new_zeros(
        (1, layout.valid_local_tokens, len(invocation.role_partition.role_names))
    )
    result.scatter_add_(2, owner.expand(1, layout.valid_local_tokens, text), token_probability)
    return result.detach().float().contiguous()


class NativeAttn1PostRopeQKObserver:
    """Same-call attn1 Q/K wrapper; official output is delegated unchanged."""

    def __init__(
        self,
        base_processor: Any,
        *,
        block_index: int,
        rank_bank: InMemoryWorld4RankShardBank,
    ) -> None:
        self.base_processor = _official_processor(base_processor)
        if block_index not in BLOCKS or not isinstance(rank_bank, InMemoryWorld4RankShardBank):
            raise NativeRelationalAttentionHookError("attn1 hook installation scope differs")
        self.block_index = block_index
        self.rank_bank = rank_bank
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
    ) -> Any:
        kwargs = _processor_kwargs(**locals_without_self(locals()))
        invocation = _active_for(self.rank_bank)
        if invocation is None:
            output = self.base_processor(attn, hidden_states, **kwargs)
            self.base_calls += 1
            return output
        _validate_frozen_attention(attn)
        _validate_common_hidden(hidden_states, invocation)
        _validate_attn1_call(
            invocation,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            rotary_emb=rotary_emb,
            batch_image_vae_seqlen=batch_image_vae_seqlen,
            origin_hidden_states_seq_len=origin_hidden_states_seq_len,
            split_hidden_states_seq_len=split_hidden_states_seq_len,
            cu_seqlens_q_cache=cu_seqlens_q_cache,
            max_seqlen_q_cache=max_seqlen_q_cache,
        )
        output, query, key = _delegate_with_same_call_qk(
            self.base_processor, attn, hidden_states, kwargs
        )
        self.base_calls += 1
        try:
            self.rank_bank.capture_attn1(
                Attn1PostRopeQKRankShard(invocation, self.block_index, query, key)
            )
        except Exception:
            _zeroize_tensors((query, key))
            raise
        self.observer_calls += 1
        return output


class NativeAttn2DerivedRoleProxyObserver:
    """Same-call attn2 Q/K wrapper with an honestly labelled post-hoc proxy."""

    def __init__(
        self,
        base_processor: Any,
        *,
        block_index: int,
        rank_bank: InMemoryWorld4RankShardBank,
    ) -> None:
        self.base_processor = _official_processor(base_processor)
        if block_index not in BLOCKS or not isinstance(rank_bank, InMemoryWorld4RankShardBank):
            raise NativeRelationalAttentionHookError("attn2 hook installation scope differs")
        self.block_index = block_index
        self.rank_bank = rank_bank
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
    ) -> Any:
        kwargs = _processor_kwargs(**locals_without_self(locals()))
        invocation = _active_for(self.rank_bank)
        if invocation is None:
            output = self.base_processor(attn, hidden_states, **kwargs)
            self.base_calls += 1
            return output
        _validate_frozen_attention(attn)
        _validate_common_hidden(hidden_states, invocation)
        _validate_attn2_call(
            invocation,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            rotary_emb=rotary_emb,
            batch_image_vae_seqlen=batch_image_vae_seqlen,
            text_features_length=text_features_length,
            origin_hidden_states_seq_len=origin_hidden_states_seq_len,
            split_hidden_states_seq_len=split_hidden_states_seq_len,
            cu_seqlens_k_cross_cache=cu_seqlens_k_cross_cache,
            cu_seqlens_q_cross_cache=cu_seqlens_q_cross_cache,
            max_seqlen_k_cross_cache=max_seqlen_k_cross_cache,
            max_seqlen_q_cross_cache=max_seqlen_q_cross_cache,
        )
        output, query, key = _delegate_with_same_call_qk(
            self.base_processor, attn, hidden_states, kwargs
        )
        self.base_calls += 1
        proxy: Optional[torch.Tensor] = None
        try:
            proxy = _derive_qk_role_proxy(query, key, invocation, attn=attn)
        finally:
            _zeroize_tensors((query, key))
        try:
            self.rank_bank.capture_proxy(
                DerivedRoleProxyRankShard(invocation, self.block_index, proxy)
            )
        except Exception:
            if proxy is not None:
                _zeroize_tensors((proxy,))
            raise
        self.observer_calls += 1
        return output


def locals_without_self(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Select only official processor keyword arguments from a wrapper frame."""

    names = (
        "encoder_hidden_states",
        "attention_mask",
        "rotary_emb",
        "batch_image_vae_seqlen",
        "text_features_length",
        "origin_hidden_states_seq_len",
        "split_hidden_states_seq_len",
        "cu_seqlens_q_cache",
        "max_seqlen_q_cache",
        "cu_seqlens_k_cross_cache",
        "cu_seqlens_q_cross_cache",
        "max_seqlen_k_cross_cache",
        "max_seqlen_q_cross_cache",
    )
    return {name: values[name] for name in names}


@dataclass
class NativeRelationalAttentionHookHandle:
    transformer: Any
    attn1_wrappers: tuple[NativeAttn1PostRopeQKObserver, ...]
    attn2_wrappers: tuple[NativeAttn2DerivedRoleProxyObserver, ...]
    original_attn1: tuple[Any, ...]
    original_attn2: tuple[Any, ...]
    restored: bool = False

    def restore(self) -> None:
        if self.restored:
            return
        for index, attn1, attn2 in zip(BLOCKS, self.attn1_wrappers, self.attn2_wrappers):
            block = self.transformer.blocks[index]
            if block.attn1.processor is not attn1 or block.attn2.processor is not attn2:
                raise NativeRelationalAttentionHookError("installed hook changed behind handle")
        for index, original1, original2 in zip(BLOCKS, self.original_attn1, self.original_attn2):
            block = self.transformer.blocks[index]
            _set_processor(block.attn1, original1)
            _set_processor(block.attn2, original2)
        self.restored = True

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": SCHEMA_VERSION,
            "blocks": list(BLOCKS),
            "world_size": WORLD_SIZE,
            "official_transformer_source_sha256": OFFICIAL_TRANSFORMER_SOURCE_SHA256,
            "official_processor_delegated_once": True,
            "official_output_forwarded_same_object": True,
            "candidate_output_modified": False,
            "attn1_projection_calls_added": 0,
            "attn2_projection_calls_added": 0,
            "collective_calls_inside_attention_added": 0,
            "responsibility_kind": RESPONSIBILITY_KIND,
            "backend_attention_weights_observed": False,
            "base_frozen_required": True,
            "optimizer_available": False,
            "decoder_available": False,
            "route_or_injection_available": False,
            "training_authorized": False,
            "gpu_launch_authorized": False,
            "scientific_claim_authorized": False,
            "restored": self.restored,
        }
        return {**value, "digest": _canonical_digest(value)}

    def __enter__(self) -> "NativeRelationalAttentionHookHandle":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.restore()


def _set_processor(attn: Any, processor: Any) -> None:
    setter = getattr(attn, "set_processor", None)
    if callable(setter):
        setter(processor)
    else:
        attn.processor = processor


def _resolve_transformer(model: Any) -> Any:
    queue = [model]
    seen: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        blocks = getattr(candidate, "blocks", None)
        if blocks is not None:
            if len(blocks) != EXPECTED_BLOCK_COUNT:
                raise NativeRelationalAttentionHookError("Bernini transformer must have 30 blocks")
            if isinstance(candidate, torch.nn.Module):
                if candidate.training or any(
                    parameter.requires_grad for parameter in candidate.parameters()
                ):
                    raise NativeRelationalAttentionHookError(
                        "native observer requires a completely frozen eval transformer"
                    )
            return candidate
        get_base_model = getattr(candidate, "get_base_model", None)
        if callable(get_base_model):
            try:
                queue.append(get_base_model())
            except Exception:
                pass
        for name in ("diff_dec", "transformer", "base_model", "model", "module"):
            nested = getattr(candidate, name, None)
            if nested is not None and nested is not candidate:
                queue.append(nested)
    raise NativeRelationalAttentionHookError("cannot resolve Bernini 30-block transformer")


def install_native_relational_attention_hook(
    model: Any, *, rank_bank: InMemoryWorld4RankShardBank
) -> NativeRelationalAttentionHookHandle:
    """Transactionally install exactly eight read-only processor wrappers."""

    if not isinstance(rank_bank, InMemoryWorld4RankShardBank):
        raise NativeRelationalAttentionHookError("hook installation bank differs")
    transformer = _resolve_transformer(model)
    original1: list[Any] = []
    original2: list[Any] = []
    wrappers1: list[NativeAttn1PostRopeQKObserver] = []
    wrappers2: list[NativeAttn2DerivedRoleProxyObserver] = []
    installed: list[int] = []
    try:
        for index in BLOCKS:
            block = transformer.blocks[index]
            base1 = getattr(block.attn1, "processor", None)
            base2 = getattr(block.attn2, "processor", None)
            wrapper1 = NativeAttn1PostRopeQKObserver(
                base1, block_index=index, rank_bank=rank_bank
            )
            wrapper2 = NativeAttn2DerivedRoleProxyObserver(
                base2, block_index=index, rank_bank=rank_bank
            )
            _set_processor(block.attn1, wrapper1)
            _set_processor(block.attn2, wrapper2)
            if block.attn1.processor is not wrapper1 or block.attn2.processor is not wrapper2:
                raise NativeRelationalAttentionHookError("processor hook installation did not stick")
            original1.append(base1)
            original2.append(base2)
            wrappers1.append(wrapper1)
            wrappers2.append(wrapper2)
            installed.append(index)
    except Exception:
        for index, base1, base2 in zip(installed, original1, original2):
            block = transformer.blocks[index]
            _set_processor(block.attn1, base1)
            _set_processor(block.attn2, base2)
        raise
    return NativeRelationalAttentionHookHandle(
        transformer,
        tuple(wrappers1),
        tuple(wrappers2),
        tuple(original1),
        tuple(original2),
    )


def reconstruct_world4_block_from_collectives(
    *,
    invocation: native.CaptureInvocation,
    role_partition: ExhaustiveTextRolePartition,
    block_index: int,
    qk_rank_major: torch.Tensor,
    proxy_rank_major: torch.Tensor,
    rank_metadata: Sequence[Mapping[str, Any]],
) -> tuple[World4BlockRankShard, ...]:
    """Consume two external WORLD4 all-gathers and rebuild typed rank shards.

    ``qk_rank_major`` is ``[4,2,1,G,3,128]`` and
    ``proxy_rank_major`` is ``[4,1,ceil(G/4),K]``.  This function never calls a
    collective.  It copies only validated logical rows and zeroizes both input
    collective tensors on success or failure.
    """

    result: list[World4BlockRankShard] = []
    rank_temporaries: list[torch.Tensor] = []
    succeeded = False
    try:
        if not isinstance(invocation, native.CaptureInvocation):
            raise NativeRelationalAttentionHookError("collective invocation differs")
        if not isinstance(role_partition, ExhaustiveTextRolePartition):
            raise NativeRelationalAttentionHookError("collective role partition differs")
        if block_index not in BLOCKS:
            raise NativeRelationalAttentionHookError("collective block differs")
        qk = _finite_detached_contiguous(qk_rank_major, label="WORLD4 Q/K collective")
        proxy = _finite_detached_contiguous(
            proxy_rank_major, label="WORLD4 role-proxy collective"
        )
        layout0 = World4RankLayout(0, invocation.patch_height, invocation.patch_width)
        if (
            tuple(qk.shape)
            != (
                WORLD_SIZE,
                2,
                1,
                layout0.global_tokens,
                LOCAL_ATTN1_HEADS,
                HEAD_DIM,
            )
            or tuple(proxy.shape)
            != (
                WORLD_SIZE,
                1,
                layout0.padded_local_tokens,
                len(role_partition.role_names),
            )
            or proxy.dtype != torch.float32
            or qk.device != proxy.device
            or len(rank_metadata) != WORLD_SIZE
        ):
            raise NativeRelationalAttentionHookError("WORLD4 collective tensor geometry differs")
        for rank in range(WORLD_SIZE):
            layout = World4RankLayout(
                rank, invocation.patch_height, invocation.patch_width
            )
            rank_invocation = RankCaptureInvocation(
                invocation, layout, role_partition
            )
            metadata = rank_metadata[rank]
            if not isinstance(metadata, Mapping):
                raise NativeRelationalAttentionHookError("WORLD4 metadata row differs")
            expected = _collective_metadata_payload(
                rank_invocation, block_index=block_index
            )
            supplied = dict(metadata)
            digest = supplied.pop("metadata_sha256", None)
            if supplied != expected or digest != _canonical_digest(expected):
                raise NativeRelationalAttentionHookError(
                    "WORLD4 rank metadata/order differs"
                )
            if layout.valid_local_tokens < layout.padded_local_tokens:
                padding = proxy[
                    rank,
                    :,
                    layout.valid_local_tokens : layout.padded_local_tokens,
                ]
                if int(torch.count_nonzero(padding).item()) != 0:
                    raise NativeRelationalAttentionHookError(
                        "WORLD4 role-proxy append padding is nonzero"
                    )
            query = _owned_contiguous_clone(
                qk[rank, 0], rank_temporaries
            )
            key = _owned_contiguous_clone(
                qk[rank, 1], rank_temporaries
            )
            role_proxy = _owned_contiguous_clone(
                proxy[rank, :, : layout.valid_local_tokens],
                rank_temporaries,
            )
            result.append(
                World4BlockRankShard(
                    rank_invocation,
                    block_index,
                    query,
                    key,
                    role_proxy,
                )
            )
        output = tuple(result)
        succeeded = True
        return output
    finally:
        if not succeeded:
            for row in result:
                row.zeroize()
            _zeroize_tensors(rank_temporaries)
        _zeroize_tensors(
            tuple(
                value
                for value in (qk_rank_major, proxy_rank_major)
                if isinstance(value, torch.Tensor)
            )
        )


def commit_world4_shards_to_native_bank(
    *,
    native_bank: native.InMemoryNativeCaptureBank,
    invocation: native.CaptureInvocation,
    rank_shards: Sequence[World4BlockRankShard],
) -> Mapping[str, Any]:
    """Join externally gathered WORLD4 shards and populate the native bank.

    This function performs no distributed call.  ``rank_shards`` must already
    contain all 4 ranks x 4 blocks in canonical rank order authority.
    Supplied raw shards are zeroized on success or failure.
    """

    rows = tuple(rank_shards)
    captures: list[native.NativeBlockCapture] = []
    assembly_temporaries: list[torch.Tensor] = []
    succeeded = False
    try:
        if not isinstance(native_bank, native.InMemoryNativeCaptureBank):
            raise NativeRelationalAttentionHookError("native capture bank type differs")
        if not isinstance(invocation, native.CaptureInvocation):
            raise NativeRelationalAttentionHookError("native invocation type differs")
        if len(rows) != WORLD_SIZE * len(BLOCKS):
            raise NativeRelationalAttentionHookError("WORLD4 requires exactly sixteen rank shards")
        registry: dict[tuple[int, int], World4BlockRankShard] = {}
        partition_digest: Optional[str] = None
        role_names: Optional[tuple[str, ...]] = None
        for row in rows:
            if not isinstance(row, World4BlockRankShard) or row.invocation.capture != invocation:
                raise NativeRelationalAttentionHookError("foreign WORLD4 rank shard")
            key = (row.block_index, row.invocation.layout.rank)
            if key in registry:
                raise NativeRelationalAttentionHookError("duplicate WORLD4 block/rank shard")
            registry[key] = row
            digest = row.invocation.role_partition.digest
            names = row.invocation.role_partition.role_names
            if partition_digest is None:
                partition_digest, role_names = digest, names
            elif digest != partition_digest or names != role_names:
                raise NativeRelationalAttentionHookError("WORLD4 text-role partition differs")
        expected = {(block, rank) for block in BLOCKS for rank in range(WORLD_SIZE)}
        if set(registry) != expected or role_names is None:
            raise NativeRelationalAttentionHookError("WORLD4 block/rank registry differs")

        for block in BLOCKS:
            shards = [registry[(block, rank)] for rank in range(WORLD_SIZE)]
            first = shards[0]
            dtype, device = first.query.dtype, first.query.device
            if any(
                row.query.dtype != dtype
                or row.key.dtype != dtype
                or row.query.device != device
                or row.key.device != device
                or row.derived_qk_role_responsibility_proxy.device != device
                for row in shards
            ):
                raise NativeRelationalAttentionHookError("WORLD4 tensor dtype/device differs")
            # Official Ulysses assigns a contiguous head chunk to each rank.
            query = _owned_contiguous_cat(
                [row.query for row in shards],
                dim=2,
                ownership=assembly_temporaries,
            )
            key = _owned_contiguous_cat(
                [row.key for row in shards],
                dim=2,
                ownership=assembly_temporaries,
            )
            proxy = _owned_contiguous_cat(
                [
                    row.derived_qk_role_responsibility_proxy
                    for row in shards
                ],
                dim=1,
                ownership=assembly_temporaries,
            )
            global_tokens = first.invocation.layout.global_tokens
            spatial = first.invocation.layout.spatial_tokens
            if (
                tuple(query.shape) != (1, global_tokens, TOTAL_HEADS, HEAD_DIM)
                or tuple(key.shape) != tuple(query.shape)
                or tuple(proxy.shape) != (1, global_tokens, len(role_names))
            ):
                raise NativeRelationalAttentionHookError("assembled WORLD4 tensor geometry differs")
            query = query.reshape(1, PHASES, spatial, TOTAL_HEADS, HEAD_DIM)
            key = key.reshape(1, PHASES, spatial, TOTAL_HEADS, HEAD_DIM)
            proxy = (
                proxy.reshape(1, PHASES, spatial, len(role_names))
                .permute(0, 1, 3, 2)
                .contiguous()
            )
            assembly_temporaries.append(proxy)
            capture_query = query.detach()
            assembly_temporaries.append(capture_query)
            capture_key = key.detach()
            assembly_temporaries.append(capture_key)
            capture_proxy = proxy.detach()
            assembly_temporaries.append(capture_proxy)
            captures.append(
                native.NativeBlockCapture(
                    native.CAPTURE_SCHEMA,
                    invocation,
                    block,
                    capture_query,
                    capture_key,
                    capture_proxy,
                )
            )
        with native_bank.observe(invocation):
            for capture in captures:
                native_bank.capture(capture)
        value = {
            "schema_version": SCHEMA_VERSION,
            "blocks": list(BLOCKS),
            "world_size": WORLD_SIZE,
            "rank_registry": list(range(WORLD_SIZE)),
            "attn1_layout": "global_visual_sequence_x_rank_local_heads",
            "attn2_proxy_layout": "rank_local_visual_sequence_x_all_heads",
            "head_assembly": "concatenate_rank_order",
            "sequence_assembly": "concatenate_valid_rank_intervals",
            "responsibility_kind": RESPONSIBILITY_KIND,
            "backend_attention_weights_observed": False,
            "collective_calls_inside_attention_added": 0,
            "external_world4_collective_required": True,
            "native_block_capture_count": len(captures),
            "rank_shards_zeroized": True,
            "target_inputs_consumed": False,
            "candidate_output_modified": False,
            "optimizer_created": False,
            "decoder_called": False,
            "gpu_launch_authorized": False,
            "scientific_claim_authorized": False,
        }
        receipt = {**value, "digest": _canonical_digest(value)}
        succeeded = True
        return receipt
    finally:
        if not succeeded:
            resident_rows: dict[int, native.NativeBlockCapture] = {}
            resident = getattr(native_bank, "_captures", None)
            if isinstance(resident, dict) and isinstance(
                invocation, native.CaptureInvocation
            ):
                value = resident.pop(invocation.key, None)
                if isinstance(value, dict):
                    resident_rows = value
            resident_ids = {id(value) for value in resident_rows.values()}
            if resident_rows and isinstance(
                native_bank, native.InMemoryNativeCaptureBank
            ):
                native_bank.zeroize(tuple(resident_rows.values()))
            for capture in captures:
                if id(capture) not in resident_ids:
                    capture.zeroize()
            _zeroize_tensors(assembly_temporaries)
        for row in rows:
            if isinstance(row, World4BlockRankShard):
                row.zeroize()


__all__ = [
    "BACKEND_ATTENTION_WEIGHTS_OBSERVED",
    "BLOCKS",
    "DerivedRoleProxyRankShard",
    "ExhaustiveTextRolePartition",
    "InMemoryWorld4RankShardBank",
    "NativeAttn1PostRopeQKObserver",
    "NativeAttn2DerivedRoleProxyObserver",
    "NativeRelationalAttentionHookError",
    "NativeRelationalAttentionHookHandle",
    "RESPONSIBILITY_KIND",
    "RankCaptureInvocation",
    "World4BlockRankShard",
    "World4RankLayout",
    "commit_world4_shards_to_native_bank",
    "install_native_relational_attention_hook",
    "reconstruct_world4_block_from_collectives",
]
