#!/usr/bin/env python3
"""Fail-closed same-outer-state no-op co-state routing for Bernini/DMIQ.

This module is intentionally a small tensor/runtime oracle core.  For one
solver step, an outer runner evaluates two frozen-base branches from the exact
same current packed ``[source, target]`` noisy state:

1. a semantic-no-op branch captures, at every selected block, the full-pair
   post-``_project_qkv`` self-attention K/V pair and the post-block local source
   boundary hidden state; then
2. an action-text branch keeps its own Q, target K/V, attn2/text route, and
   FFN.  Its source K/V is routed as a pair and each selected post-block source
   shard is clamped back toward the cached no-op boundary.

The main arm is

``hybrid_source_noop_kv = ([K_noop_source,K_action_target],``
``[V_noop_source,V_action_target])``.

The stronger ``full_pair_noop_value_diagnostic`` arm replaces both halves and
is explicitly diagnostic because it can suppress the requested action.  A
fixed gate ``g=0`` calls the untouched official attn1 processor and makes the
block-boundary hook return ``None``; it is therefore an exact delegate rather
than a reimplementation of the official path.

"Same state" here means the same *outer* packed x_t only.  Semantic-no-op and
action text create different per-layer hidden trajectories.  Combining action
Q/target K/V with no-op source K/V is thus a controlled cross-branch
factorization oracle, not a claim that Q/K/V came from one per-layer state.

The cache is one-use, one-generation, one-step, one-rank, Ulysses-4 state.  It
binds the outer trajectory/current-state digests, exact token geometry, every
block, and the exact RoPE tensor object/storage/version.  Source-only capture,
cross-trajectory reuse, equal-but-distinct RoPE tensors, silent overwrite, and
reuse after a completed action route all fail closed.  The caller must retain
the cache through backward (when no checkpoint recomputation is used), then
explicitly retire it.  This minimal core does not implement a sampler,
launcher, checkpoint-recompute context, or optimizer authorization.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence


METHOD_NAME = "dmiq-same-state-noop-costate-router"
SCHEMA_VERSION = "dmiq-same-state-noop-costate-router-v1"
CACHE_SCHEMA_VERSION = "dmiq-same-state-noop-costate-cache-v1"

CAPTURE_MODE = "semantic_noop_full_pair_capture"
ACTION_MODE = "action_full_pair_route"
CAPTURE_BRANCH = "semantic_noop"
ACTION_BRANCH = "action"
INVOCATION_MODES = (CAPTURE_MODE, ACTION_MODE)

HYBRID_SOURCE_NOOP_KV = "hybrid_source_noop_kv"
FULL_PAIR_NOOP_VALUE_DIAGNOSTIC = "full_pair_noop_value_diagnostic"
COSTATE_ARMS = (
    HYBRID_SOURCE_NOOP_KV,
    FULL_PAIR_NOOP_VALUE_DIAGNOSTIC,
)

REQUIRED_ULYSSES_SIZE = 4


class DMIQSameStateNoopValueRouterError(RuntimeError):
    """Raised before accepting an ambiguous, stale, or off-contract route."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependent
        raise DMIQSameStateNoopValueRouterError(
            "same-state no-op co-state routing requires PyTorch"
        ) from error
    return torch


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
        raise DMIQSameStateNoopValueRouterError(
            f"router receipt is not canonical JSON: {error}"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DMIQSameStateNoopValueRouterError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _exact_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise DMIQSameStateNoopValueRouterError(f"{label} must be an integer")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "numel"):
        if int(value.numel()) != 1:
            raise DMIQSameStateNoopValueRouterError(
                f"{label} must be scalar"
            )
        value = value.item()
    try:
        integer = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise DMIQSameStateNoopValueRouterError(
            f"{label} must be an integer"
        ) from error
    if not math.isfinite(numeric) or numeric != float(integer) or integer < 0:
        raise DMIQSameStateNoopValueRouterError(
            f"{label} must be an exact non-negative finite integer"
        )
    return integer


def _exact_positive_int(value: Any, *, label: str) -> int:
    integer = _exact_nonnegative_int(value, label=label)
    if integer <= 0:
        raise DMIQSameStateNoopValueRouterError(f"{label} must be positive")
    return integer


def validate_fixed_gate(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DMIQSameStateNoopValueRouterError(
            "fixed gate must be a real scalar"
        )
    gate = float(value)
    if not math.isfinite(gate) or gate < 0.0 or gate > 1.0:
        raise DMIQSameStateNoopValueRouterError(
            "fixed gate must be finite and in [0,1]"
        )
    return gate


def _validate_arm(value: Any) -> str:
    if value not in COSTATE_ARMS:
        raise DMIQSameStateNoopValueRouterError(
            f"co-state arm must be one of {COSTATE_ARMS}, got {value!r}"
        )
    return value


def _validate_indices(value: Any) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DMIQSameStateNoopValueRouterError(
            "selected block indices must be a sequence"
        )
    indices = tuple(
        _exact_nonnegative_int(item, label="selected block index")
        for item in value
    )
    if not indices or tuple(sorted(indices)) != indices:
        raise DMIQSameStateNoopValueRouterError(
            "selected block indices must be nonempty and sorted"
        )
    if len(set(indices)) != len(indices):
        raise DMIQSameStateNoopValueRouterError(
            "selected block indices must be unique"
        )
    return indices


def _as_int_tuple(value: Any, *, label: str) -> tuple[int, ...]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DMIQSameStateNoopValueRouterError(
            f"{label} must be a sequence"
        )
    return tuple(
        _exact_nonnegative_int(item, label=f"{label} item") for item in value
    )


def _validate_floating_tensor(
    value: Any,
    *,
    label: str,
    ndim: int,
    frozen: Optional[bool] = None,
    allow_empty: bool = False,
) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise DMIQSameStateNoopValueRouterError(
            f"{label} must be a torch.Tensor"
        )
    if value.layout != torch.strided or value.is_meta:
        raise DMIQSameStateNoopValueRouterError(
            f"{label} must be a dense non-meta tensor"
        )
    if value.ndim != ndim:
        raise DMIQSameStateNoopValueRouterError(
            f"{label} must have rank {ndim}"
        )
    if any(int(size) < 0 for size in value.shape) or (
        not allow_empty and any(int(size) == 0 for size in value.shape)
    ):
        raise DMIQSameStateNoopValueRouterError(
            f"{label} has an invalid empty dimension"
        )
    if not bool(value.dtype.is_floating_point):
        raise DMIQSameStateNoopValueRouterError(
            f"{label} must use a floating dtype"
        )
    if not value.is_contiguous():
        raise DMIQSameStateNoopValueRouterError(
            f"{label} must be contiguous"
        )
    if frozen is True and (value.requires_grad or value.grad_fn is not None):
        raise DMIQSameStateNoopValueRouterError(
            f"{label} must come from a frozen no-grad branch"
        )
    if frozen is False and not value.requires_grad:
        raise DMIQSameStateNoopValueRouterError(
            f"{label} must retain the action computation graph"
        )
    if value.numel() and not bool(torch.isfinite(value).all().item()):
        raise DMIQSameStateNoopValueRouterError(f"{label} must be finite")
    return value


def _storage_identity(value: Any) -> tuple[str, int, int]:
    return (
        str(value.device),
        int(value.untyped_storage().data_ptr()),
        int(value.storage_offset()),
    )


def _tensor_raw_bytes_equal(left: Any, right: Any) -> bool:
    """Compare exact tensor payload bits, preserving signed zero and NaN bits."""

    torch = _torch()
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return False
    if (
        tuple(left.shape) != tuple(right.shape)
        or left.dtype != right.dtype
        or left.device != right.device
    ):
        return False
    left_bytes = left.detach().contiguous().view(torch.uint8)
    right_bytes = right.detach().contiguous().view(torch.uint8)
    return bool(torch.equal(left_bytes, right_bytes))


@dataclass(frozen=True)
class SameStateStepIdentity:
    """Exact rank-local identity of one outer ``[source,target]`` x_t."""

    generation: int
    step_index: int
    timestep_token: str
    outer_trajectory_sha256: str
    current_pair_state_sha256: str
    rank: int
    ulysses_size: int
    global_pair_tokens: int

    def validate(self) -> None:
        _exact_nonnegative_int(self.generation, label="generation")
        _exact_nonnegative_int(self.step_index, label="step index")
        if (
            type(self.timestep_token) is not str
            or not self.timestep_token
            or self.timestep_token != self.timestep_token.strip()
        ):
            raise DMIQSameStateNoopValueRouterError(
                "timestep_token must be a nonempty canonical string"
            )
        _require_sha256(
            self.outer_trajectory_sha256,
            label="outer_trajectory_sha256",
        )
        _require_sha256(
            self.current_pair_state_sha256,
            label="current_pair_state_sha256",
        )
        rank = _exact_nonnegative_int(self.rank, label="Ulysses rank")
        size = _exact_positive_int(self.ulysses_size, label="Ulysses size")
        if size != REQUIRED_ULYSSES_SIZE:
            raise DMIQSameStateNoopValueRouterError(
                f"this frozen oracle requires Ulysses={REQUIRED_ULYSSES_SIZE}"
            )
        if rank >= size:
            raise DMIQSameStateNoopValueRouterError(
                "Ulysses rank is outside the group"
            )
        total = _exact_positive_int(
            self.global_pair_tokens,
            label="global pair token count",
        )
        if total % (2 * size) != 0:
            raise DMIQSameStateNoopValueRouterError(
                "equal source/target pair must shard evenly across Ulysses-4"
            )

    @property
    def source_tokens(self) -> int:
        self.validate()
        return self.global_pair_tokens // 2

    @property
    def local_sequence_tokens(self) -> int:
        self.validate()
        return self.global_pair_tokens // self.ulysses_size

    @property
    def local_sequence_start(self) -> int:
        return self.rank * self.local_sequence_tokens

    @property
    def local_source_span(self) -> tuple[int, int]:
        global_start = self.local_sequence_start
        global_stop = global_start + self.local_sequence_tokens
        source_stop = self.source_tokens
        intersection_start = min(max(global_start, 0), source_stop)
        intersection_stop = min(max(global_stop, 0), source_stop)
        if intersection_stop <= intersection_start:
            return (0, 0)
        return (
            intersection_start - global_start,
            intersection_stop - global_start,
        )

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "generation": self.generation,
            "step_index": self.step_index,
            "timestep_token": self.timestep_token,
            "outer_trajectory_sha256": self.outer_trajectory_sha256,
            "current_pair_state_sha256": self.current_pair_state_sha256,
            "rank": self.rank,
            "ulysses_size": self.ulysses_size,
            "global_pair_tokens": self.global_pair_tokens,
            "source_tokens": self.source_tokens,
            "local_sequence_tokens": self.local_sequence_tokens,
            "local_sequence_start": self.local_sequence_start,
            "local_source_span": list(self.local_source_span),
        }

    def digest(self) -> str:
        return _object_sha256(self.as_dict())


@dataclass(frozen=True)
class SameStateNoopValueInvocation:
    cache_bank: Any
    mode: str
    branch_tag: str
    arm: str
    gate: float
    identity: SameStateStepIdentity

    def validate(self) -> None:
        if self.mode not in INVOCATION_MODES:
            raise DMIQSameStateNoopValueRouterError(
                f"mode must be one of {INVOCATION_MODES}"
            )
        expected_branch = (
            CAPTURE_BRANCH if self.mode == CAPTURE_MODE else ACTION_BRANCH
        )
        if self.branch_tag != expected_branch:
            raise DMIQSameStateNoopValueRouterError(
                f"{self.mode} requires branch_tag={expected_branch!r}"
            )
        _validate_arm(self.arm)
        validate_fixed_gate(self.gate)
        if not isinstance(self.identity, SameStateStepIdentity):
            raise DMIQSameStateNoopValueRouterError(
                "invocation identity has the wrong type"
            )
        self.identity.validate()


def same_state_noop_value_router_contract(
    *,
    selected_block_indices: Sequence[int],
    arm: str = HYBRID_SOURCE_NOOP_KV,
    gate: float,
) -> dict[str, Any]:
    """Return the dependency-free frozen-oracle contract and limitations."""

    indices = _validate_indices(selected_block_indices)
    selected_arm = _validate_arm(arm)
    fixed_gate = validate_fixed_gate(gate)
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD_NAME,
        "status": "minimal_frozen_oracle_tensor_runtime_core_not_sampler",
        "main_arm": HYBRID_SOURCE_NOOP_KV,
        "selected_arm": selected_arm,
        "selected_arm_is_main": selected_arm == HYBRID_SOURCE_NOOP_KV,
        "arms": {
            HYBRID_SOURCE_NOOP_KV: {
                "key": "[K_noop_source,K_action_target]",
                "value": "[V_noop_source,V_action_target]",
                "role": "main_identity_preserving_motion_arm",
                "action_query_kept": True,
                "noop_source_key_value_from_same_projection": True,
                "action_target_key_kept_byte_exact": True,
                "action_target_value_kept_byte_exact": True,
                "block_boundary_source_clamp": True,
            },
            FULL_PAIR_NOOP_VALUE_DIAGNOSTIC: {
                "key": "K_action_full_pair",
                "value": "V_noop_full_pair",
                "role": "strong_identity_action_suppression_diagnostic_only",
                "eligible_as_main_claim": False,
                "block_boundary_source_clamp": True,
            },
        },
        "fixed_gate": fixed_gate,
        "zero_gate": {
            "attn1": "exact_untouched_official_processor_delegate",
            "block_boundary_hook": "returns_None_without_tensor_write",
            "cache_required": False,
        },
        "outer_state": {
            "capture_and_action_share_exact_current_[source,target]_x_t": True,
            "semantic_noop_capture_is_full_pair": True,
            "source_only_capture_allowed": False,
            "same_outer_x_t_implies_same_per_layer_hidden": False,
            "per_layer_same_state_claimed": False,
            "scientific_description": (
                "controlled_cross_branch_factorization_oracle"
            ),
        },
        "action_branch_invariants": {
            "own_attn1_queries_full_pair": True,
            "own_attn1_target_keys": True,
            "own_attn1_target_values": True,
            "own_attn2_text_route": True,
            "own_ffn": True,
            "main_routes_paired_source_key_and_value": True,
            "diagnostic_routes_only_full_pair_value": True,
            "post_block_target_hidden_is_never_clamped": True,
        },
        "cache_lifecycle": {
            "capture": (
                "one frozen semantic-noop full-pair forward at the same outer x_t"
            ),
            "contents_per_block": [
                "detached_full_pair_attn1_key_value_from_one_official_project_qkv",
                "detached_post_block_rank_local_noop_source_hidden",
            ],
            "identity": [
                "generation",
                "step_index",
                "timestep_token",
                "outer_trajectory_sha256",
                "current_pair_state_sha256",
                "block_index",
                "Ulysses_rank",
                "Ulysses_size_4",
                "full_and_local_shapes",
                "dtype",
                "device",
                "exact_RoPE_object_storage_offset_stride_version",
            ],
            "reuse": "one action forward only; explicit retire after backward",
            "failure": "poison_then_explicit_discard",
            "cross_step_or_cross_trajectory_reuse": False,
            "equal_but_distinct_RoPE_tensor_allowed": False,
        },
        "ulysses": {
            "required_size": REQUIRED_ULYSSES_SIZE,
            "cached_key_value": "full_sequence_rank_local_head_shards",
            "boundary_hidden": "contiguous_rank_local_sequence_shard",
            "source_boundary_alignment": "equal pair splits over ranks 0_and_1",
            "output_inverse": "official_gather_heads_scatter_seq",
        },
        "difference_from_failed_v9_v10": {
            "V9": (
                "uses_no_source_only_cache; paired_source_KV_comes_from_a_"
                "full_pair_noop_costate_at_the_same_outer_xt_and_source_is_clamped"
            ),
            "V10": (
                "does_not_use_source_only_V_residual; captures_full_pair_noop_"
                "costate_and_reinjects_noop_source_at_each_block_boundary"
            ),
            "new_causal_variable": (
                "factorize_action_Q_and_target_KV_from_action_branch_with_"
                "paired_source_KV_and_source_boundaries_from_same_outer_xt_noop_branch"
            ),
        },
        "external_inputs_training_and_inference": [
            "source_video",
            "edit_instruction",
        ],
        "internal_not_external_target": (
            "target_half_is_the_evolving_noisy_sample_inside_current_x_t_not_"
            "a_ground_truth_or_proposal_video"
        ),
        "forbidden_external_inputs": [
            "target_video",
            "proposal_video",
            "proposal_latent",
            "mask",
            "track",
            "swept_tube",
            "pose",
            "trajectory",
            "optical_flow",
            "first_frame_anchor",
        ],
        "fatal_limitations": [
            "same_outer_x_t_does_not_make_noop_source_KV_and_action_Q_same_layer_state",
            "cross_branch_Q_and_KV_factorization_can_be_off_manifold",
            "source_boundary_clamp_can_remove_real_source_target_interaction_feedback",
            "full_pair_noop_value_can_suppress_the_requested_action",
            "no_identity_or_action_success_is_implied_by_tensor_correctness",
            "minimal_core_has_no_gradient_checkpoint_recompute_context",
            "requires_a_second_full_noop_forward_per_solver_step",
            "requires_exact_reuse_of_the_same_RoPE_tensor_object",
            "outer_runner_must_authenticate_the_global_current_pair_state_digest",
            "real_Bernini_FA2_Ulysses4_parity_is_not_proven_by_small_tensor_tests",
        ],
        "selected_block_indices": list(indices),
        "trained_parameters": 0,
        "optimizer_updates_authorized": False,
    }
    value["contract_digest"] = _object_sha256(value)
    return value


@dataclass(frozen=True)
class RoPECacheIdentity:
    """Identity of the exact RoPE cache tensor, not merely equal values."""

    object_id: int
    storage_device: str
    storage_pointer: int
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    version: int

    def receipt(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "storage_device": self.storage_device,
            "storage_pointer": self.storage_pointer,
            "storage_offset": self.storage_offset,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "dtype": self.dtype,
            "version": self.version,
        }


def _describe_rope(
    rotary_emb: Any,
    *,
    global_pair_tokens: int,
    head_dim: int,
    projected_device: Any,
) -> RoPECacheIdentity:
    torch = _torch()
    if not isinstance(rotary_emb, torch.Tensor):
        raise DMIQSameStateNoopValueRouterError(
            "rotary_emb must be a torch.Tensor"
        )
    if rotary_emb.layout != torch.strided or rotary_emb.is_meta:
        raise DMIQSameStateNoopValueRouterError(
            "rotary_emb must be a dense non-meta tensor"
        )
    if rotary_emb.ndim != 4:
        raise DMIQSameStateNoopValueRouterError(
            "rotary_emb must have shape [1,full_pair,1,head_dim/2]"
        )
    expected_shape = (1, int(global_pair_tokens), 1, int(head_dim) // 2)
    if int(head_dim) % 2 or tuple(int(item) for item in rotary_emb.shape) != (
        expected_shape
    ):
        raise DMIQSameStateNoopValueRouterError(
            "rotary_emb shape differs from the full gathered Q/K geometry"
        )
    if not bool(rotary_emb.dtype.is_complex):
        raise DMIQSameStateNoopValueRouterError(
            "rotary_emb must use a complex dtype"
        )
    if rotary_emb.device != projected_device:
        raise DMIQSameStateNoopValueRouterError(
            "rotary_emb device differs from projected Q/K/V"
        )
    if rotary_emb.requires_grad or rotary_emb.grad_fn is not None:
        raise DMIQSameStateNoopValueRouterError(
            "rotary_emb must be frozen"
        )
    return RoPECacheIdentity(
        object_id=id(rotary_emb),
        storage_device=str(rotary_emb.device),
        storage_pointer=int(rotary_emb.untyped_storage().data_ptr()),
        storage_offset=int(rotary_emb.storage_offset()),
        shape=tuple(int(item) for item in rotary_emb.shape),
        stride=tuple(int(item) for item in rotary_emb.stride()),
        dtype=str(rotary_emb.dtype),
        version=int(getattr(rotary_emb, "_version", -1)),
    )


@dataclass(frozen=True)
class CapturedNoopBlockState:
    """Detached paired full K/V and rank-local source boundary for one block."""

    identity_digest: str
    block_index: int
    full_key_value_shape: tuple[int, ...]
    boundary_source_shape: tuple[int, ...]
    key_dtype: str
    key_device: str
    value_dtype: str
    value_device: str
    boundary_dtype: str
    boundary_device: str
    noop_full_key: Any
    noop_full_value: Any
    noop_source_boundary: Any


@dataclass
class _PartialNoopBlockState:
    noop_full_key: Any = None
    noop_full_value: Any = None
    noop_source_boundary: Any = None


_CURRENT_INVOCATION: ContextVar[Optional[SameStateNoopValueInvocation]] = (
    ContextVar("dmiq_same_state_noop_value_invocation", default=None)
)


def current_same_state_noop_value_invocation() -> SameStateNoopValueInvocation:
    invocation = _CURRENT_INVOCATION.get()
    if invocation is None:
        raise DMIQSameStateNoopValueRouterError(
            "router processor called outside an explicit invocation"
        )
    return invocation


class SameStateNoopValueCacheBank:
    """One-use rank-local cache for one full-pair no-op/action factorization."""

    def __init__(self, selected_block_indices: Sequence[int]) -> None:
        self.selected_block_indices = _validate_indices(selected_block_indices)
        self._identity: Optional[SameStateStepIdentity] = None
        self._entries: dict[int, _PartialNoopBlockState] = {}
        self._rope_ref: Any = None
        self._rope_identity: Optional[RoPECacheIdentity] = None
        self._retired_identity_digests: set[str] = set()
        self._last_retired_identity_digest: Optional[str] = None
        self._route_key_value_seen: set[int] = set()
        self._route_boundary_seen: set[int] = set()
        self._route_complete = False
        self._poisoned = False
        self.capture_key_value_calls = 0
        self.capture_boundary_calls = 0
        self.route_key_value_lookups = 0
        self.route_boundary_lookups = 0
        self.zero_gate_invocations = 0

    @property
    def identity(self) -> Optional[SameStateStepIdentity]:
        return self._identity

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def complete(self) -> bool:
        expected = set(self.selected_block_indices)
        if set(self._entries) != expected:
            return False
        return all(
            self._entries[index].noop_full_key is not None
            and self._entries[index].noop_full_value is not None
            and self._entries[index].noop_source_boundary is not None
            for index in self.selected_block_indices
        )

    @property
    def route_complete(self) -> bool:
        return self._route_complete

    def _require_active(
        self,
        invocation: SameStateNoopValueInvocation,
        *,
        mode: str,
    ) -> None:
        if _CURRENT_INVOCATION.get() is not invocation:
            raise DMIQSameStateNoopValueRouterError(
                "cache access is outside its exact active invocation"
            )
        if invocation.cache_bank is not self or invocation.mode != mode:
            raise DMIQSameStateNoopValueRouterError(
                "cache access mode/bank differs from invocation"
            )
        if (
            mode == CAPTURE_MODE or invocation.gate > 0.0
        ) and self._identity != invocation.identity:
            raise DMIQSameStateNoopValueRouterError(
                "cache identity differs from the active generation/step/rank"
            )
        if self._poisoned:
            raise DMIQSameStateNoopValueRouterError(
                "cache is poisoned and must be explicitly discarded"
            )

    def _enter(self, invocation: SameStateNoopValueInvocation) -> None:
        if invocation.cache_bank is not self:
            raise DMIQSameStateNoopValueRouterError(
                "invocation refers to another cache bank"
            )
        if invocation.mode == ACTION_MODE and invocation.gate == 0.0:
            self.zero_gate_invocations += 1
            return
        if self._poisoned:
            raise DMIQSameStateNoopValueRouterError(
                "cache is poisoned and must be explicitly discarded"
            )
        if invocation.mode == CAPTURE_MODE:
            digest = invocation.identity.digest()
            if digest in self._retired_identity_digests:
                raise DMIQSameStateNoopValueRouterError(
                    "retired outer state cannot be captured again"
                )
            if self._identity is not None or self._entries:
                raise DMIQSameStateNoopValueRouterError(
                    "cache is occupied; silent overwrite is forbidden"
                )
            self._identity = invocation.identity
            self._route_complete = False
            self._route_key_value_seen.clear()
            self._route_boundary_seen.clear()
        else:
            if self._identity != invocation.identity:
                raise DMIQSameStateNoopValueRouterError(
                    "action route is cross-step, cross-rank, or cross-trajectory"
                )
            if not self.complete:
                raise DMIQSameStateNoopValueRouterError(
                    "no-op cache is incomplete"
                )
            if self._route_complete:
                raise DMIQSameStateNoopValueRouterError(
                    "no-op cache was already consumed by an action forward"
                )
            if self._route_key_value_seen or self._route_boundary_seen:
                raise DMIQSameStateNoopValueRouterError(
                    "a partial action route must be discarded, not resumed"
                )

    def _finish(self, invocation: SameStateNoopValueInvocation) -> None:
        if invocation.mode == CAPTURE_MODE:
            if not self.complete:
                raise DMIQSameStateNoopValueRouterError(
                    "semantic-noop capture missed a K/V pair or block boundary"
                )
            return
        if invocation.gate == 0.0:
            return
        expected = set(self.selected_block_indices)
        if self._route_key_value_seen != expected:
            raise DMIQSameStateNoopValueRouterError(
                "action route did not consume every selected attn1 K/V pair"
            )
        if self._route_boundary_seen != expected:
            raise DMIQSameStateNoopValueRouterError(
                "action route did not visit every selected block boundary"
            )
        self._route_complete = True

    def _poison(self) -> None:
        if self._identity is not None:
            self._poisoned = True

    def _validate_block(self, block_index: Any) -> int:
        index = _exact_nonnegative_int(block_index, label="block index")
        if index not in self.selected_block_indices:
            raise DMIQSameStateNoopValueRouterError(
                f"block {index} is outside cache scope"
            )
        return index

    def _partial(self, index: int) -> _PartialNoopBlockState:
        return self._entries.setdefault(index, _PartialNoopBlockState())

    def _bind_or_validate_rope(
        self,
        rotary_emb: Any,
        *,
        identity: SameStateStepIdentity,
        head_dim: int,
        projected_device: Any,
    ) -> None:
        description = _describe_rope(
            rotary_emb,
            global_pair_tokens=identity.global_pair_tokens,
            head_dim=head_dim,
            projected_device=projected_device,
        )
        if self._rope_ref is None:
            self._rope_ref = rotary_emb
            self._rope_identity = description
            return
        if rotary_emb is not self._rope_ref or description != self._rope_identity:
            raise DMIQSameStateNoopValueRouterError(
                "RoPE cache identity differs; equal cloned RoPE is forbidden"
            )

    def capture_key_value(
        self,
        *,
        invocation: SameStateNoopValueInvocation,
        block_index: int,
        noop_full_key: Any,
        noop_full_value: Any,
        rotary_emb: Any,
    ) -> None:
        self._require_active(invocation, mode=CAPTURE_MODE)
        index = self._validate_block(block_index)
        key = _validate_floating_tensor(
            noop_full_key,
            label="semantic-noop full-pair key",
            ndim=4,
            frozen=True,
        )
        value = _validate_floating_tensor(
            noop_full_value,
            label="semantic-noop full-pair value",
            ndim=4,
            frozen=True,
        )
        shape = tuple(int(item) for item in key.shape)
        if tuple(value.shape) != tuple(key.shape):
            raise DMIQSameStateNoopValueRouterError(
                "semantic-noop K/V from one projection must have equal shapes"
            )
        if key.dtype != value.dtype or key.device != value.device:
            raise DMIQSameStateNoopValueRouterError(
                "semantic-noop K/V dtype/device differs"
            )
        if shape[0] != 1 or shape[1] != invocation.identity.global_pair_tokens:
            raise DMIQSameStateNoopValueRouterError(
                "capture must be full [source,target] K/V, never source-only"
            )
        self._bind_or_validate_rope(
            rotary_emb,
            identity=invocation.identity,
            head_dim=shape[3],
            projected_device=key.device,
        )
        partial = self._partial(index)
        if (
            partial.noop_full_key is not None
            or partial.noop_full_value is not None
        ):
            raise DMIQSameStateNoopValueRouterError(
                f"block {index} no-op K/V was captured twice"
            )
        cached_key = key.detach().clone().contiguous()
        cached_value = value.detach().clone().contiguous()
        if _storage_identity(cached_key) == _storage_identity(key):
            raise DMIQSameStateNoopValueRouterError(
                "captured key aliases projection workspace"
            )
        if _storage_identity(cached_value) == _storage_identity(value):
            raise DMIQSameStateNoopValueRouterError(
                "captured value aliases projection workspace"
            )
        if _storage_identity(cached_key) == _storage_identity(cached_value):
            raise DMIQSameStateNoopValueRouterError(
                "cached no-op K and V unexpectedly alias"
            )
        if not _tensor_raw_bytes_equal(cached_key, key) or not (
            _tensor_raw_bytes_equal(cached_value, value)
        ):
            raise DMIQSameStateNoopValueRouterError(
                "detached K/V capture changed raw projection bits"
            )
        partial.noop_full_key = cached_key
        partial.noop_full_value = cached_value
        self.capture_key_value_calls += 1

    def capture_boundary(
        self,
        *,
        invocation: SameStateNoopValueInvocation,
        block_index: int,
        noop_block_output: Any,
    ) -> None:
        self._require_active(invocation, mode=CAPTURE_MODE)
        index = self._validate_block(block_index)
        hidden = _validate_floating_tensor(
            noop_block_output,
            label="semantic-noop block-boundary hidden",
            ndim=3,
            frozen=True,
        )
        expected = (
            1,
            invocation.identity.local_sequence_tokens,
            int(hidden.shape[2]),
        )
        if tuple(int(item) for item in hidden.shape) != expected:
            raise DMIQSameStateNoopValueRouterError(
                "block-boundary hidden differs from canonical Ulysses-4 shard"
            )
        start, stop = invocation.identity.local_source_span
        cached = hidden[:, start:stop].detach().clone().contiguous()
        partial = self._partial(index)
        if partial.noop_source_boundary is not None:
            raise DMIQSameStateNoopValueRouterError(
                f"block {index} no-op source boundary was captured twice"
            )
        partial.noop_source_boundary = cached
        self.capture_boundary_calls += 1

    def lookup_key_value(
        self,
        *,
        invocation: SameStateNoopValueInvocation,
        block_index: int,
        current_full_key: Any,
        current_full_value: Any,
        rotary_emb: Any,
    ) -> tuple[Any, Any]:
        self._require_active(invocation, mode=ACTION_MODE)
        if invocation.gate == 0.0:
            raise DMIQSameStateNoopValueRouterError(
                "zero-gate delegate must not read the no-op cache"
            )
        index = self._validate_block(block_index)
        if index in self._route_key_value_seen:
            raise DMIQSameStateNoopValueRouterError(
                f"block {index} K/V was routed twice"
            )
        current_key = _validate_floating_tensor(
            current_full_key,
            label="action full-pair key",
            ndim=4,
        )
        current = _validate_floating_tensor(
            current_full_value,
            label="action full-pair value",
            ndim=4,
        )
        partial = self._entries.get(index)
        if (
            partial is None
            or partial.noop_full_key is None
            or partial.noop_full_value is None
        ):
            raise DMIQSameStateNoopValueRouterError(
                f"block {index} has no paired no-op K/V"
            )
        cached_key = partial.noop_full_key
        cached_value = partial.noop_full_value
        if not (
            tuple(current_key.shape)
            == tuple(current.shape)
            == tuple(cached_key.shape)
            == tuple(cached_value.shape)
        ):
            raise DMIQSameStateNoopValueRouterError(
                f"block {index} action/no-op K/V shapes differ"
            )
        if not (
            current_key.dtype
            == current.dtype
            == cached_key.dtype
            == cached_value.dtype
        ) or not (
            current_key.device
            == current.device
            == cached_key.device
            == cached_value.device
        ):
            raise DMIQSameStateNoopValueRouterError(
                f"block {index} action/no-op K/V dtype/device differs"
            )
        self._bind_or_validate_rope(
            rotary_emb,
            identity=invocation.identity,
            head_dim=int(current.shape[3]),
            projected_device=current_key.device,
        )
        if any(
            tensor.requires_grad or tensor.grad_fn is not None
            for tensor in (cached_key, cached_value)
        ):
            raise DMIQSameStateNoopValueRouterError(
                "cached no-op K/V unexpectedly retains autograd"
            )
        self._route_key_value_seen.add(index)
        self.route_key_value_lookups += 1
        return cached_key, cached_value

    def lookup_boundary(
        self,
        *,
        invocation: SameStateNoopValueInvocation,
        block_index: int,
        current_block_output: Any,
    ) -> Any:
        self._require_active(invocation, mode=ACTION_MODE)
        if invocation.gate == 0.0:
            raise DMIQSameStateNoopValueRouterError(
                "zero-gate boundary delegate must not read the cache"
            )
        index = self._validate_block(block_index)
        if index in self._route_boundary_seen:
            raise DMIQSameStateNoopValueRouterError(
                f"block {index} boundary was routed twice"
            )
        hidden = _validate_floating_tensor(
            current_block_output,
            label="action block-boundary hidden",
            ndim=3,
        )
        if int(hidden.shape[0]) != 1 or int(hidden.shape[1]) != (
            invocation.identity.local_sequence_tokens
        ):
            raise DMIQSameStateNoopValueRouterError(
                "action boundary shape differs from canonical Ulysses-4 shard"
            )
        partial = self._entries.get(index)
        if partial is None or partial.noop_source_boundary is None:
            raise DMIQSameStateNoopValueRouterError(
                f"block {index} has no no-op source boundary"
            )
        start, stop = invocation.identity.local_source_span
        current_source = hidden[:, start:stop]
        cached = partial.noop_source_boundary
        if tuple(current_source.shape) != tuple(cached.shape):
            raise DMIQSameStateNoopValueRouterError(
                f"block {index} action/no-op source boundary shapes differ"
            )
        if current_source.dtype != cached.dtype or current_source.device != cached.device:
            raise DMIQSameStateNoopValueRouterError(
                f"block {index} action/no-op boundary dtype/device differs"
            )
        if cached.requires_grad or cached.grad_fn is not None:
            raise DMIQSameStateNoopValueRouterError(
                "cached no-op source boundary unexpectedly retains autograd"
            )
        self._route_boundary_seen.add(index)
        self.route_boundary_lookups += 1
        return cached

    def inspect_block(self, block_index: int) -> CapturedNoopBlockState:
        index = self._validate_block(block_index)
        if self._identity is None:
            raise DMIQSameStateNoopValueRouterError("cache has no active identity")
        partial = self._entries.get(index)
        if (
            partial is None
            or partial.noop_full_key is None
            or partial.noop_full_value is None
            or partial.noop_source_boundary is None
        ):
            raise DMIQSameStateNoopValueRouterError(
                f"block {index} capture is incomplete"
            )
        key = partial.noop_full_key.detach().clone().contiguous()
        value = partial.noop_full_value.detach().clone().contiguous()
        boundary = partial.noop_source_boundary.detach().clone().contiguous()
        return CapturedNoopBlockState(
            identity_digest=self._identity.digest(),
            block_index=index,
            full_key_value_shape=tuple(int(item) for item in value.shape),
            boundary_source_shape=tuple(int(item) for item in boundary.shape),
            key_dtype=str(key.dtype),
            key_device=str(key.device),
            value_dtype=str(value.dtype),
            value_device=str(value.device),
            boundary_dtype=str(boundary.dtype),
            boundary_device=str(boundary.device),
            noop_full_key=key,
            noop_full_value=value,
            noop_source_boundary=boundary,
        )

    def retire(self, identity: SameStateStepIdentity) -> None:
        if _CURRENT_INVOCATION.get() is not None:
            raise DMIQSameStateNoopValueRouterError(
                "cannot retire a cache during an active invocation"
            )
        if not isinstance(identity, SameStateStepIdentity):
            raise DMIQSameStateNoopValueRouterError(
                "retire identity has the wrong type"
            )
        identity.validate()
        if self._identity != identity:
            raise DMIQSameStateNoopValueRouterError(
                "retire identity differs from cached outer state"
            )
        if self._poisoned:
            raise DMIQSameStateNoopValueRouterError(
                "poisoned cache must be discarded, not retired as successful"
            )
        if not self._route_complete:
            raise DMIQSameStateNoopValueRouterError(
                "cache cannot retire before one complete action route"
            )
        self._clear_current(retire=True)

    def discard(self) -> None:
        if _CURRENT_INVOCATION.get() is not None:
            raise DMIQSameStateNoopValueRouterError(
                "cannot discard a cache during an active invocation"
            )
        self._clear_current(retire=self._identity is not None)

    def _clear_current(self, *, retire: bool) -> None:
        if retire and self._identity is not None:
            digest = self._identity.digest()
            self._retired_identity_digests.add(digest)
            self._last_retired_identity_digest = digest
        self._identity = None
        self._entries.clear()
        self._rope_ref = None
        self._rope_identity = None
        self._route_key_value_seen.clear()
        self._route_boundary_seen.clear()
        self._route_complete = False
        self._poisoned = False

    def receipt(self) -> dict[str, Any]:
        entries = []
        for index in sorted(self._entries):
            partial = self._entries[index]
            key = partial.noop_full_key
            value = partial.noop_full_value
            boundary = partial.noop_source_boundary
            entries.append(
                {
                    "block_index": index,
                    "key_value_shape": None
                    if key is None or value is None
                    else list(key.shape),
                    "paired_key_value": bool(
                        key is not None
                        and value is not None
                        and tuple(key.shape) == tuple(value.shape)
                        and key.dtype == value.dtype
                        and key.device == value.device
                    ),
                    "boundary_source_shape": None
                    if boundary is None
                    else list(boundary.shape),
                    "detached": bool(
                        key is not None
                        and value is not None
                        and boundary is not None
                        and not key.requires_grad
                        and key.grad_fn is None
                        and not value.requires_grad
                        and value.grad_fn is None
                        and not boundary.requires_grad
                        and boundary.grad_fn is None
                    ),
                }
            )
        value: dict[str, Any] = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "identity": None
            if self._identity is None
            else self._identity.as_dict(),
            "selected_blocks": list(self.selected_block_indices),
            "captured_blocks": sorted(self._entries),
            "complete": self.complete,
            "route_complete": self._route_complete,
            "poisoned": self._poisoned,
            "capture_key_value_calls": self.capture_key_value_calls,
            "capture_boundary_calls": self.capture_boundary_calls,
            "route_key_value_lookups": self.route_key_value_lookups,
            "route_boundary_lookups": self.route_boundary_lookups,
            "zero_gate_invocations": self.zero_gate_invocations,
            "retired_identity_count": len(self._retired_identity_digests),
            "last_retired_identity_digest": self._last_retired_identity_digest,
            "rope_identity": None
            if self._rope_identity is None
            else self._rope_identity.receipt(),
            "entries": entries,
            "cross_trajectory_reuse_authorized": False,
            "source_only_capture_authorized": False,
        }
        value["cache_receipt_digest"] = _object_sha256(value)
        return value


@contextmanager
def same_state_noop_value_invocation(
    cache_bank: SameStateNoopValueCacheBank,
    *,
    mode: str,
    branch_tag: str,
    arm: str,
    gate: float,
    identity: SameStateStepIdentity,
) -> Iterator[SameStateNoopValueInvocation]:
    """Bind exactly one full-pair no-op capture or action route forward."""

    if not isinstance(cache_bank, SameStateNoopValueCacheBank):
        raise DMIQSameStateNoopValueRouterError(
            "cache_bank has the wrong type"
        )
    invocation = SameStateNoopValueInvocation(
        cache_bank=cache_bank,
        mode=mode,
        branch_tag=branch_tag,
        arm=arm,
        gate=validate_fixed_gate(gate),
        identity=identity,
    )
    invocation.validate()
    if _CURRENT_INVOCATION.get() is not None:
        raise DMIQSameStateNoopValueRouterError(
            "nested same-state router invocations are forbidden"
        )
    cache_bank._enter(invocation)
    token = _CURRENT_INVOCATION.set(invocation)
    try:
        yield invocation
    except BaseException:
        _CURRENT_INVOCATION.reset(token)
        if mode == CAPTURE_MODE or invocation.gate > 0.0:
            cache_bank._poison()
        raise
    else:
        _CURRENT_INVOCATION.reset(token)
        try:
            cache_bank._finish(invocation)
        except BaseException:
            if mode == CAPTURE_MODE or invocation.gate > 0.0:
                cache_bank._poison()
            raise


def _maybe_call(value: Any) -> Any:
    return value() if callable(value) else value


def _parallel_identity(state: Any) -> tuple[bool, int, int]:
    enabled = bool(_maybe_call(getattr(state, "ulysses_enabled", False)))
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
    if not enabled or rank is None or size is None:
        raise DMIQSameStateNoopValueRouterError(
            "same-state router requires an observable enabled Ulysses state"
        )
    rank_int = _exact_nonnegative_int(rank, label="runtime Ulysses rank")
    size_int = _exact_positive_int(size, label="runtime Ulysses size")
    if size_int != REQUIRED_ULYSSES_SIZE or rank_int >= size_int:
        raise DMIQSameStateNoopValueRouterError(
            "runtime must be the exact configured Ulysses-4 rank"
        )
    return True, rank_int, size_int


def _validate_projected_qkv(
    query: Any,
    key: Any,
    value: Any,
    *,
    identity: SameStateStepIdentity,
) -> tuple[int, int, int, int]:
    torch = _torch()
    tensors = (("query", query), ("key", key), ("value", value))
    shapes: list[tuple[int, ...]] = []
    for label, tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise DMIQSameStateNoopValueRouterError(
                f"projected {label} must be a tensor"
            )
        if tensor.ndim != 4 or int(tensor.shape[0]) != 1:
            raise DMIQSameStateNoopValueRouterError(
                f"projected {label} must be [1,full_pair,local_heads,head_dim]"
            )
        if not bool(tensor.dtype.is_floating_point):
            raise DMIQSameStateNoopValueRouterError(
                f"projected {label} must use a floating dtype"
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise DMIQSameStateNoopValueRouterError(
                f"projected {label} must be finite"
            )
        shapes.append(tuple(int(item) for item in tensor.shape))
    if not (shapes[0] == shapes[1] == shapes[2]):
        raise DMIQSameStateNoopValueRouterError(
            f"official projected Q/K/V shapes differ: {shapes!r}"
        )
    if not (query.dtype == key.dtype == value.dtype):
        raise DMIQSameStateNoopValueRouterError(
            "official projected Q/K/V dtypes differ"
        )
    if not (query.device == key.device == value.device):
        raise DMIQSameStateNoopValueRouterError(
            "official projected Q/K/V devices differ"
        )
    shape = shapes[0]
    if shape[1] != identity.global_pair_tokens:
        raise DMIQSameStateNoopValueRouterError(
            "projected sequence is not the exact full [source,target] pair; "
            "source-only caches are forbidden"
        )
    if shape[2] <= 0 or shape[3] <= 0:
        raise DMIQSameStateNoopValueRouterError(
            "projected local head geometry must be positive"
        )
    return shape  # type: ignore[return-value]


def _validate_pair_metadata(
    *,
    identity: SameStateStepIdentity,
    hidden_states: Any,
    batch_image_vae_seqlen: Any,
    cu_seqlens_q_cache: Any,
    max_seqlen_q_cache: Any,
    origin_hidden_states_seq_len: Any,
    split_hidden_states_seq_len: Any,
) -> None:
    hidden = _validate_floating_tensor(
        hidden_states,
        label="rank-local full-pair hidden shard",
        ndim=3,
    )
    if tuple(int(item) for item in hidden.shape[:2]) != (
        1,
        identity.local_sequence_tokens,
    ):
        raise DMIQSameStateNoopValueRouterError(
            "hidden shard shape differs from contiguous Ulysses-4 layout"
        )
    total = identity.global_pair_tokens
    if _as_int_tuple(
        batch_image_vae_seqlen,
        label="batch_image_vae_seqlen",
    ) != (total,):
        raise DMIQSameStateNoopValueRouterError(
            "router requires batch=1 with one full packed pair"
        )
    if _as_int_tuple(
        cu_seqlens_q_cache,
        label="cu_seqlens_q_cache",
    ) != (0, total):
        raise DMIQSameStateNoopValueRouterError(
            "self-attention cu_seqlens must bind one full packed pair"
        )
    if _exact_positive_int(
        max_seqlen_q_cache,
        label="max_seqlen_q_cache",
    ) != total:
        raise DMIQSameStateNoopValueRouterError(
            "max self-attention length differs from full pair"
        )
    if _exact_positive_int(
        origin_hidden_states_seq_len,
        label="origin_hidden_states_seq_len",
    ) != total:
        raise DMIQSameStateNoopValueRouterError(
            "Ulysses origin length differs from full pair"
        )
    if _exact_positive_int(
        split_hidden_states_seq_len,
        label="split_hidden_states_seq_len",
    ) != identity.local_sequence_tokens:
        raise DMIQSameStateNoopValueRouterError(
            "Ulysses split length differs from rank-local shard"
        )


class SameStateNoopValueSelfAttnProcessor:
    """Attn1 wrapper routing a paired no-op source K/V co-state."""

    def __init__(
        self,
        base_processor: Any,
        *,
        block_index: int,
        cache_bank: SameStateNoopValueCacheBank,
        arm: str,
        gate: float,
        varlen_attention_fn: Optional[Callable[..., Any]] = None,
        get_parallel_state_fn: Optional[Callable[[], Any]] = None,
        gather_heads_scatter_seq_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not callable(base_processor):
            raise DMIQSameStateNoopValueRouterError(
                "base attn1 processor must be callable"
            )
        if not callable(getattr(base_processor, "_project_qkv", None)):
            raise DMIQSameStateNoopValueRouterError(
                "base attn1 processor lacks official _project_qkv"
            )
        if not isinstance(cache_bank, SameStateNoopValueCacheBank):
            raise DMIQSameStateNoopValueRouterError(
                "cache bank has the wrong type"
            )
        index = cache_bank._validate_block(block_index)
        self.base_processor = base_processor
        self.block_index = index
        self.cache_bank = cache_bank
        self.arm = _validate_arm(arm)
        self.gate = validate_fixed_gate(gate)
        self._varlen_attention_fn = varlen_attention_fn
        self._get_parallel_state_fn = get_parallel_state_fn
        self._gather_heads_scatter_seq_fn = gather_heads_scatter_seq_fn
        self.capture_calls = 0
        self.action_route_calls = 0
        self.zero_gate_delegations = 0
        self.saw_ulysses4 = False
        self.outputs_finite = True

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

    def _validate_invocation(
        self,
        invocation: SameStateNoopValueInvocation,
    ) -> None:
        if invocation.cache_bank is not self.cache_bank:
            raise DMIQSameStateNoopValueRouterError(
                "processor and invocation use different cache banks"
            )
        if invocation.arm != self.arm or invocation.gate != self.gate:
            raise DMIQSameStateNoopValueRouterError(
                "processor arm/gate differs from invocation"
            )

    def _delegate(
        self,
        attn: Any,
        hidden_states: Any,
        *,
        encoder_hidden_states: Optional[Any],
        attention_mask: Optional[Any],
        rotary_emb: Optional[Any],
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
    ) -> Any:
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

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Optional[Any] = None,
        attention_mask: Optional[Any] = None,
        rotary_emb: Optional[Any] = None,
        batch_image_vae_seqlen: Any = None,
        text_features_length: Any = None,
        origin_hidden_states_seq_len: Optional[int] = None,
        split_hidden_states_seq_len: Optional[int] = None,
        cu_seqlens_q_cache: Any = None,
        max_seqlen_q_cache: Any = None,
        cu_seqlens_k_cross_cache: Any = None,
        cu_seqlens_q_cross_cache: Any = None,
        max_seqlen_k_cross_cache: Any = None,
        max_seqlen_q_cross_cache: Any = None,
    ) -> Any:
        invocation = current_same_state_noop_value_invocation()
        self._validate_invocation(invocation)
        varlen_fn, state_fn, inverse_fn = self._runtime_ops()
        _, runtime_rank, runtime_size = _parallel_identity(state_fn())
        if (runtime_rank, runtime_size) != (
            invocation.identity.rank,
            invocation.identity.ulysses_size,
        ):
            raise DMIQSameStateNoopValueRouterError(
                "runtime rank/Ulysses identity differs from invocation"
            )
        self.saw_ulysses4 = True

        if invocation.mode == ACTION_MODE and self.gate == 0.0:
            self.zero_gate_delegations += 1
            return self._delegate(
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
            cu_seqlens_k_cross_cache,
            cu_seqlens_q_cross_cache,
            max_seqlen_k_cross_cache,
            max_seqlen_q_cross_cache,
        )
        if encoder_hidden_states is not None:
            raise DMIQSameStateNoopValueRouterError(
                "router may only wrap attn1 self-attention"
            )
        if attention_mask is not None:
            raise DMIQSameStateNoopValueRouterError(
                "extra self-attention masks make the route ambiguous"
            )
        _validate_pair_metadata(
            identity=invocation.identity,
            hidden_states=hidden_states,
            batch_image_vae_seqlen=batch_image_vae_seqlen,
            cu_seqlens_q_cache=cu_seqlens_q_cache,
            max_seqlen_q_cache=max_seqlen_q_cache,
            origin_hidden_states_seq_len=origin_hidden_states_seq_len,
            split_hidden_states_seq_len=split_hidden_states_seq_len,
        )
        if invocation.mode == CAPTURE_MODE:
            _validate_floating_tensor(
                hidden_states,
                label="semantic-noop hidden shard",
                ndim=3,
                frozen=True,
            )
        query, key, value = self.base_processor._project_qkv(
            attn,
            hidden_states,
            None,
            rotary_emb,
            origin_hidden_states_seq_len,
            False,
        )
        projected_shape = _validate_projected_qkv(
            query,
            key,
            value,
            identity=invocation.identity,
        )
        _describe_rope(
            rotary_emb,
            global_pair_tokens=invocation.identity.global_pair_tokens,
            head_dim=projected_shape[3],
            projected_device=query.device,
        )

        if invocation.mode == CAPTURE_MODE:
            self.cache_bank.capture_key_value(
                invocation=invocation,
                block_index=self.block_index,
                noop_full_key=key.contiguous(),
                noop_full_value=value.contiguous(),
                rotary_emb=rotary_emb,
            )
            routed_key = key
            routed_value = value
            self.capture_calls += 1
        else:
            cached_noop_key, cached_noop_value = (
                self.cache_bank.lookup_key_value(
                    invocation=invocation,
                    block_index=self.block_index,
                    current_full_key=key,
                    current_full_value=value,
                    rotary_emb=rotary_emb,
                )
            )
            source_tokens = invocation.identity.source_tokens
            if self.arm == HYBRID_SOURCE_NOOP_KV:
                if self.gate == 1.0:
                    routed_source_key = cached_noop_key[:, :source_tokens]
                    routed_source_value = cached_noop_value[:, :source_tokens]
                else:
                    current_source_key = key[:, :source_tokens]
                    current_source = value[:, :source_tokens]
                    routed_source_key = current_source_key + self.gate * (
                        cached_noop_key[:, :source_tokens] - current_source_key
                    )
                    routed_source_value = current_source + self.gate * (
                        cached_noop_value[:, :source_tokens] - current_source
                    )
                routed_key = _torch().cat(
                    (routed_source_key, key[:, source_tokens:]),
                    dim=1,
                ).contiguous()
                routed_value = _torch().cat(
                    (routed_source_value, value[:, source_tokens:]),
                    dim=1,
                ).contiguous()
                if not _tensor_raw_bytes_equal(
                    routed_key[:, source_tokens:],
                    key[:, source_tokens:],
                ) or not _tensor_raw_bytes_equal(
                    routed_value[:, source_tokens:],
                    value[:, source_tokens:],
                ):
                    raise DMIQSameStateNoopValueRouterError(
                        "main arm changed action target K/V suffix"
                    )
            else:
                routed_key = key
                routed_value = (
                    cached_noop_value
                    if self.gate == 1.0
                    else value + self.gate * (cached_noop_value - value)
                ).contiguous()
            self.action_route_calls += 1

        query_for_dtype = query
        query_work = query.squeeze(0).contiguous()
        key_work = routed_key.squeeze(0).contiguous()
        value_work = routed_value.squeeze(0).contiguous()
        attention_output = varlen_fn(
            query_work,
            key_work,
            value_work,
            cu_seqlens_q=cu_seqlens_q_cache,
            cu_seqlens_k=cu_seqlens_q_cache,
            max_seqlen_q=max_seqlen_q_cache,
            max_seqlen_k=max_seqlen_q_cache,
            causal=False,
        )
        if tuple(int(item) for item in attention_output.shape) != tuple(
            int(item) for item in query_work.shape
        ):
            raise DMIQSameStateNoopValueRouterError(
                "varlen attention output shape differs from gathered query"
            )
        output = inverse_fn(
            attention_output.unsqueeze(0),
            head_dim=2,
            seq_dim=1,
        )
        if (
            getattr(output, "ndim", None) != 4
            or int(output.shape[0]) != 1
            or int(output.shape[1])
            != invocation.identity.local_sequence_tokens
        ):
            raise DMIQSameStateNoopValueRouterError(
                "Ulysses inverse output is not the canonical local sequence shard"
            )
        output = output.flatten(2, 3).contiguous().type_as(query_for_dtype)
        try:
            output = attn.to_out[0](output)
            output = attn.to_out[1](output)
        except (AttributeError, IndexError, TypeError) as error:
            raise DMIQSameStateNoopValueRouterError(
                "attn1 lacks the official two-stage to_out path"
            ) from error
        torch = _torch()
        finite = bool(torch.isfinite(output.detach()).all().item())
        self.outputs_finite = bool(self.outputs_finite and finite)
        if not finite:
            raise DMIQSameStateNoopValueRouterError(
                "routed attention output is non-finite"
            )
        return output

    def statistics(self) -> dict[str, Any]:
        return {
            "block_index": self.block_index,
            "arm": self.arm,
            "gate": self.gate,
            "capture_calls": self.capture_calls,
            "action_route_calls": self.action_route_calls,
            "zero_gate_delegations": self.zero_gate_delegations,
            "saw_ulysses4": self.saw_ulysses4,
            "outputs_finite": self.outputs_finite,
            "action_query_source": "current_action_branch",
            "main_key_value": "paired_noop_source_plus_action_target",
            "diagnostic_key_value": "action_full_key_plus_noop_full_value",
            "capture_projection_pairing": "one_project_qkv_call_per_block",
        }


def _extract_block_hidden(output: Any) -> tuple[Any, Callable[[Any], Any]]:
    torch = _torch()
    if isinstance(output, torch.Tensor):
        return output, lambda replacement: replacement
    if type(output) is tuple and output and isinstance(output[0], torch.Tensor):
        suffix = output[1:]
        return output[0], lambda replacement: (replacement, *suffix)
    raise DMIQSameStateNoopValueRouterError(
        "selected block output must be a tensor or plain tuple starting with one"
    )


class SameStateNoopSourceBoundaryRouter:
    """Post-block hook capturing or clamping the rank-local source shard."""

    def __init__(
        self,
        *,
        block_index: int,
        cache_bank: SameStateNoopValueCacheBank,
        arm: str,
        gate: float,
    ) -> None:
        if not isinstance(cache_bank, SameStateNoopValueCacheBank):
            raise DMIQSameStateNoopValueRouterError(
                "cache bank has the wrong type"
            )
        self.block_index = cache_bank._validate_block(block_index)
        self.cache_bank = cache_bank
        self.arm = _validate_arm(arm)
        self.gate = validate_fixed_gate(gate)
        self.capture_calls = 0
        self.action_clamp_calls = 0
        self.zero_gate_delegations = 0
        self.target_only_rank_delegations = 0
        self._installed_hook_id: Optional[int] = None

    def bind_installed_hook(self, module: Any, handle: Any) -> None:
        if self._installed_hook_id is not None:
            raise DMIQSameStateNoopValueRouterError(
                "boundary router hook identity was already bound"
            )
        identifier = getattr(handle, "id", None)
        registry = getattr(module, "_forward_hooks", None)
        if (
            type(identifier) is not int
            or not isinstance(registry, Mapping)
            or len(registry) != 1
            or registry.get(identifier) is not self
        ):
            raise DMIQSameStateNoopValueRouterError(
                "installed boundary hook identity cannot be proven"
            )
        self._installed_hook_id = identifier

    def _audit_installed_hook(self, module: Any) -> None:
        if self._installed_hook_id is None:
            return
        registry = getattr(module, "_forward_hooks", None)
        if (
            not isinstance(registry, Mapping)
            or len(registry) != 1
            or registry.get(self._installed_hook_id) is not self
        ):
            raise DMIQSameStateNoopValueRouterError(
                "block hook set changed; boundary clamp order is ambiguous"
            )

    def __call__(self, module: Any, inputs: Any, output: Any) -> Optional[Any]:
        self._audit_installed_hook(module)
        del inputs
        invocation = current_same_state_noop_value_invocation()
        if invocation.cache_bank is not self.cache_bank:
            raise DMIQSameStateNoopValueRouterError(
                "boundary router and invocation use different cache banks"
            )
        if invocation.arm != self.arm or invocation.gate != self.gate:
            raise DMIQSameStateNoopValueRouterError(
                "boundary router arm/gate differs from invocation"
            )
        if invocation.mode == ACTION_MODE and self.gate == 0.0:
            self.zero_gate_delegations += 1
            return None

        hidden, rebuild = _extract_block_hidden(output)
        if invocation.mode == CAPTURE_MODE:
            self.cache_bank.capture_boundary(
                invocation=invocation,
                block_index=self.block_index,
                noop_block_output=hidden,
            )
            self.capture_calls += 1
            return None

        cached_source = self.cache_bank.lookup_boundary(
            invocation=invocation,
            block_index=self.block_index,
            current_block_output=hidden,
        )
        start, stop = invocation.identity.local_source_span
        if start == stop:
            self.target_only_rank_delegations += 1
            return None
        current_source = hidden[:, start:stop]
        routed_source = (
            cached_source
            if self.gate == 1.0
            else current_source + self.gate * (cached_source - current_source)
        )
        torch = _torch()
        routed_hidden = torch.cat(
            (hidden[:, :start], routed_source, hidden[:, stop:]),
            dim=1,
        ).contiguous()
        if not _tensor_raw_bytes_equal(
            routed_hidden[:, stop:], hidden[:, stop:]
        ):
            raise DMIQSameStateNoopValueRouterError(
                "source clamp changed a target-side local boundary suffix"
            )
        self.action_clamp_calls += 1
        return rebuild(routed_hidden)

    def statistics(self) -> dict[str, Any]:
        return {
            "block_index": self.block_index,
            "arm": self.arm,
            "gate": self.gate,
            "capture_calls": self.capture_calls,
            "action_clamp_calls": self.action_clamp_calls,
            "zero_gate_delegations": self.zero_gate_delegations,
            "target_only_rank_delegations": self.target_only_rank_delegations,
            "clamped_tensor": "post_block_rank_local_source_hidden_only",
        }


def resolve_wan_transformer(model: Any) -> Any:
    """Resolve a Bernini/Wan transformer without importing either package."""

    candidates = [model]
    diff_dec = getattr(model, "diff_dec", None)
    if diff_dec is not None:
        candidates.append(getattr(diff_dec, "transformer", None))
    candidates.append(getattr(model, "transformer", None))
    matches = []
    seen: set[int] = set()
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        blocks = getattr(candidate, "blocks", None)
        if blocks is not None:
            matches.append(candidate)
    if len(matches) != 1:
        raise DMIQSameStateNoopValueRouterError(
            "could not resolve exactly one Wan transformer with blocks"
        )
    return matches[0]


def _set_processor(attn: Any, processor: Any) -> None:
    setter = getattr(attn, "set_processor", None)
    if callable(setter):
        setter(processor)
    else:
        attn.processor = processor


def _hook_is_installed(block: Any, handle: Any, hook: Any) -> bool:
    registry = getattr(block, "_forward_hooks", None)
    identifier = getattr(handle, "id", None)
    if not isinstance(registry, Mapping) or identifier is None:
        raise DMIQSameStateNoopValueRouterError(
            "block hook registry cannot be audited"
        )
    return len(registry) == 1 and registry.get(identifier) is hook


@dataclass
class SameStateNoopValuePatchHandle:
    transformer: Any
    indices: tuple[int, ...]
    arm: str
    gate: float
    cache_bank: SameStateNoopValueCacheBank
    processors: tuple[SameStateNoopValueSelfAttnProcessor, ...]
    boundary_routers: tuple[SameStateNoopSourceBoundaryRouter, ...]
    original_processors: tuple[Any, ...]
    hook_handles: tuple[Any, ...]
    restored: bool = False

    def restore(self) -> None:
        if self.restored:
            return
        blocks = self.transformer.blocks
        for index, processor, boundary, hook_handle in zip(
            self.indices,
            self.processors,
            self.boundary_routers,
            self.hook_handles,
        ):
            block = blocks[index]
            if getattr(block.attn1, "processor", None) is not processor:
                raise DMIQSameStateNoopValueRouterError(
                    f"block {index} attn1 processor changed behind patch handle"
                )
            if not _hook_is_installed(block, hook_handle, boundary):
                raise DMIQSameStateNoopValueRouterError(
                    f"block {index} boundary hook changed behind patch handle"
                )
        for index, original, hook_handle in zip(
            self.indices,
            self.original_processors,
            self.hook_handles,
        ):
            _set_processor(blocks[index].attn1, original)
            hook_handle.remove()
        self.restored = True

    def receipt(self) -> dict[str, Any]:
        value = same_state_noop_value_router_contract(
            selected_block_indices=self.indices,
            arm=self.arm,
            gate=self.gate,
        )
        value["runtime"] = {
            "installed_block_count": len(self.indices),
            "restored": self.restored,
            "cache": self.cache_bank.receipt(),
            "processors": [item.statistics() for item in self.processors],
            "boundaries": [
                item.statistics() for item in self.boundary_routers
            ],
            "optimizer_updates_authorized": False,
        }
        value["runtime_digest"] = _object_sha256(value["runtime"])
        return value


def install_same_state_noop_value_router(
    model: Any,
    *,
    selected_block_indices: Optional[Sequence[int]] = None,
    arm: str = HYBRID_SOURCE_NOOP_KV,
    gate: float,
    cache_bank: Optional[SameStateNoopValueCacheBank] = None,
    varlen_attention_fn: Optional[Callable[..., Any]] = None,
    get_parallel_state_fn: Optional[Callable[[], Any]] = None,
    gather_heads_scatter_seq_fn: Optional[Callable[..., Any]] = None,
) -> SameStateNoopValuePatchHandle:
    """Patch selected attn1 processors and post-block boundaries atomically."""

    transformer = resolve_wan_transformer(model)
    blocks = transformer.blocks
    num_blocks = len(blocks)
    indices = _validate_indices(
        tuple(range(num_blocks))
        if selected_block_indices is None
        else selected_block_indices
    )
    if indices[-1] >= num_blocks:
        raise DMIQSameStateNoopValueRouterError(
            "selected block index is outside the transformer"
        )
    selected_arm = _validate_arm(arm)
    fixed_gate = validate_fixed_gate(gate)
    bank = (
        SameStateNoopValueCacheBank(indices)
        if cache_bank is None
        else cache_bank
    )
    if not isinstance(bank, SameStateNoopValueCacheBank):
        raise DMIQSameStateNoopValueRouterError(
            "provided cache bank has the wrong type"
        )
    if bank.selected_block_indices != indices:
        raise DMIQSameStateNoopValueRouterError(
            "provided cache scope differs from selected blocks"
        )

    originals: list[Any] = []
    processors: list[SameStateNoopValueSelfAttnProcessor] = []
    boundaries: list[SameStateNoopSourceBoundaryRouter] = []
    hook_handles: list[Any] = []
    installed_indices: list[int] = []
    try:
        for index in indices:
            block = blocks[index]
            attn = getattr(block, "attn1", None)
            original = getattr(attn, "processor", None)
            if original is None:
                raise DMIQSameStateNoopValueRouterError(
                    f"block {index} attn1 lacks a processor"
                )
            if isinstance(original, SameStateNoopValueSelfAttnProcessor) or (
                hasattr(original, "cache_bank")
                and hasattr(original, "block_index")
            ):
                raise DMIQSameStateNoopValueRouterError(
                    f"block {index} already has an experimental cache processor"
                )
            register_hook = getattr(block, "register_forward_hook", None)
            if not callable(register_hook):
                raise DMIQSameStateNoopValueRouterError(
                    f"block {index} cannot install a boundary hook"
                )
            existing_hooks = getattr(block, "_forward_hooks", None)
            if not isinstance(existing_hooks, Mapping) or existing_hooks:
                raise DMIQSameStateNoopValueRouterError(
                    f"block {index} must have an empty auditable hook registry"
                )
            processor = SameStateNoopValueSelfAttnProcessor(
                original,
                block_index=index,
                cache_bank=bank,
                arm=selected_arm,
                gate=fixed_gate,
                varlen_attention_fn=varlen_attention_fn,
                get_parallel_state_fn=get_parallel_state_fn,
                gather_heads_scatter_seq_fn=gather_heads_scatter_seq_fn,
            )
            boundary = SameStateNoopSourceBoundaryRouter(
                block_index=index,
                cache_bank=bank,
                arm=selected_arm,
                gate=fixed_gate,
            )
            _set_processor(attn, processor)
            hook_handle = None
            try:
                hook_handle = register_hook(boundary)
                boundary.bind_installed_hook(block, hook_handle)
            except BaseException:
                if hook_handle is not None:
                    hook_handle.remove()
                _set_processor(attn, original)
                raise
            originals.append(original)
            processors.append(processor)
            boundaries.append(boundary)
            hook_handles.append(hook_handle)
            installed_indices.append(index)
    except BaseException:
        for index, original, processor, boundary, hook_handle in zip(
            reversed(installed_indices),
            reversed(originals),
            reversed(processors),
            reversed(boundaries),
            reversed(hook_handles),
        ):
            block = blocks[index]
            if getattr(block.attn1, "processor", None) is processor:
                _set_processor(block.attn1, original)
            if _hook_is_installed(block, hook_handle, boundary):
                hook_handle.remove()
        raise

    return SameStateNoopValuePatchHandle(
        transformer=transformer,
        indices=indices,
        arm=selected_arm,
        gate=fixed_gate,
        cache_bank=bank,
        processors=tuple(processors),
        boundary_routers=tuple(boundaries),
        original_processors=tuple(originals),
        hook_handles=tuple(hook_handles),
    )


@contextmanager
def same_state_noop_value_router(
    model: Any,
    *,
    selected_block_indices: Optional[Sequence[int]] = None,
    arm: str = HYBRID_SOURCE_NOOP_KV,
    gate: float,
    varlen_attention_fn: Optional[Callable[..., Any]] = None,
    get_parallel_state_fn: Optional[Callable[[], Any]] = None,
    gather_heads_scatter_seq_fn: Optional[Callable[..., Any]] = None,
) -> Iterator[SameStateNoopValuePatchHandle]:
    """Install the minimal co-state route and restore it on context exit."""

    handle = install_same_state_noop_value_router(
        model,
        selected_block_indices=selected_block_indices,
        arm=arm,
        gate=gate,
        varlen_attention_fn=varlen_attention_fn,
        get_parallel_state_fn=get_parallel_state_fn,
        gather_heads_scatter_seq_fn=gather_heads_scatter_seq_fn,
    )
    try:
        yield handle
    finally:
        handle.restore()


__all__ = [
    "ACTION_BRANCH",
    "ACTION_MODE",
    "CACHE_SCHEMA_VERSION",
    "CAPTURE_BRANCH",
    "CAPTURE_MODE",
    "CapturedNoopBlockState",
    "DMIQSameStateNoopValueRouterError",
    "FULL_PAIR_NOOP_VALUE_DIAGNOSTIC",
    "HYBRID_SOURCE_NOOP_KV",
    "METHOD_NAME",
    "REQUIRED_ULYSSES_SIZE",
    "RoPECacheIdentity",
    "SCHEMA_VERSION",
    "SameStateNoopSourceBoundaryRouter",
    "SameStateNoopValueCacheBank",
    "SameStateNoopValueInvocation",
    "SameStateNoopValuePatchHandle",
    "SameStateNoopValueSelfAttnProcessor",
    "SameStateStepIdentity",
    "COSTATE_ARMS",
    "current_same_state_noop_value_invocation",
    "install_same_state_noop_value_router",
    "resolve_wan_transformer",
    "same_state_noop_value_invocation",
    "same_state_noop_value_router",
    "same_state_noop_value_router_contract",
    "validate_fixed_gate",
]
