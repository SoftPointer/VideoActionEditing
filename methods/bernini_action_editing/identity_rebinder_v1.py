#!/usr/bin/env python3
"""Orderless source-identity memory for Bernini-R action editing.

``IdentityRebinderV1`` is deliberately not another motion LoRA.  A
frame-independent 2-D patch encoder turns every source frame into local
appearance fragments, drops frame indices, spatial coordinates and frame
boundaries, and pools the resulting set into a fixed-size identity atlas.
Selected Bernini ``attn1.to_out[0]`` projections then add a low-rank
cross-attention residual from that atlas to the current target rows only.
Later frozen transformer layers can propagate those modified rows; the claim
is about this adapter's direct write, not end-to-end causal isolation.

The residual output projection is exactly zero at installation, so a newly
installed adapter is the frozen Bernini model bit-for-bit.  Source memory is
owned only by the native V/VI branches, is inactive at high sigma, and cannot
write condition or SP-padding rows.  The module contains no mask, track,
flow, pose, source caption, action label, temporal convolution, frame
position, frame difference, or ordered source-token replay.  The unordered
multiset can nevertheless expose static pose, appearance frequency and dwell
time through its patch content and multiplicities; it removes order, not all
motion-correlated evidence.

This file is an executable shape/routing core.  It has not yet been validated
against the full Bernini checkpoint on GPU; ``install_identity_rebinder_v1``
therefore fails closed on the audited 30-block/Conv3d/attn1 structure and
makes no quality claim.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from typing import Any, Iterator, Mapping, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F


SCHEMA_VERSION = "bernini-identity-rebinder-v1"
ATLAS_SCHEMA_VERSION = "bernini-orderless-identity-atlas-v1"
PRETRAIN_OBJECTIVE_SCHEMA = "bernini-identity-rebinder-raw-video-pretrain-v1"
TOTAL_BLOCKS_1P3B = 30
HIDDEN_SIZE_1P3B = 1536

# These coordinates bind the prototype to the exact renderer source/checkpoint
# family audited for this repository.  They are explicit install arguments so
# a copied module cannot silently claim that an arbitrary Wan-like transformer
# is Bernini-R 1.3B.
PINNED_BERNINI_SOURCE_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
PINNED_BERNINI_MODEL_REVISION = "ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce"
PINNED_TRANSFORMER_CLASS_MODULE = "bernini.models.transformer_wan"
PINNED_TRANSFORMER_CLASS_NAME = "WanTransformer3DModel"
PINNED_SCHEDULER_CLASS_MODULE = (
    "diffusers.schedulers.scheduling_unipc_multistep"
)
PINNED_SCHEDULER_CLASS_NAME = "UniPCMultistepScheduler"
PINNED_TRANSFORMER_CONFIG = {
    "num_layers": TOTAL_BLOCKS_1P3B,
    "num_attention_heads": 12,
    "attention_head_dim": 128,
    "in_channels": 16,
    "out_channels": 16,
    "patch_size": (1, 2, 2),
    "ffn_dim": 8960,
    "text_dim": 4096,
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
}

# Early blocks are left to establish native geometry and motion.  Blocks
# 23--29 are left as a frozen synthesis guard.  The all-early/mid scope is an
# explicit ablation, not an implicit CLI free-for-all.
DEFAULT_BLOCK_INDICES = tuple(range(8, 23))
ALL_EARLY_MID_ABLATION = tuple(range(23))
ALLOWED_BLOCK_SCOPES = {DEFAULT_BLOCK_INDICES, ALL_EARLY_MID_ABLATION}
ALLOWED_SP_SIZES = {1, 4}
SOURCE_MEMORY_BRANCHES = {"V", "VI"}
SOURCE_FREE_BRANCHES = {"none", "I"}

DEFAULT_LOW_SIGMA = 0.25
DEFAULT_HIGH_SIGMA = 0.75


class IdentityRebinderContractError(RuntimeError):
    """Raised instead of accepting an ambiguous memory or Bernini route."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise IdentityRebinderContractError(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _lower_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise IdentityRebinderContractError(f"{label} must be lowercase SHA-256")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise IdentityRebinderContractError(f"{label} must be a positive integer")
    return value


def _finite_unit(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise IdentityRebinderContractError(f"{label} must be finite in [0,1]")
    return float(value)


def _tensor_version_or_none(value: torch.Tensor) -> Optional[int]:
    """Return the live mutation counter when PyTorch exposes one.

    Tensors created inside ``torch.inference_mode`` intentionally have no
    version counter.  Those packs remain bound by object/storage/shape/dtype/
    device identity; ordinary training tensors additionally get mutation
    detection through ``_version``.
    """

    try:
        return int(value._version)
    except RuntimeError:
        return None


def mid_low_sigma_gate(
    sigma: Any,
    *,
    low_sigma: float = DEFAULT_LOW_SIGMA,
    high_sigma: float = DEFAULT_HIGH_SIGMA,
) -> float:
    """C1 gate: full at low sigma, smooth ramp in mid, zero at high sigma."""

    value = _finite_unit(sigma, label="sigma")
    low = _finite_unit(low_sigma, label="low_sigma")
    high = _finite_unit(high_sigma, label="high_sigma")
    if not low < high:
        raise IdentityRebinderContractError("low_sigma must be below high_sigma")
    if value <= low:
        return 1.0
    if value >= high:
        return 0.0
    # Smoothstep from one at low to zero at high.
    u = (high - value) / (high - low)
    return u * u * (3.0 - 2.0 * u)


@dataclass(frozen=True)
class IdentityAtlas:
    """Tensor-only forward carrier plus an orderless audit receipt.

    ``tokens`` intentionally has no frame axis.  Frame paths, indices and
    timestamps are provenance-side information and are forbidden here.
    """

    tokens: torch.Tensor
    source_video_sha256: str
    source_frame_count: int
    construction_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.tokens, torch.Tensor)
            or self.tokens.layout != torch.strided
            or self.tokens.device.type == "meta"
            or self.tokens.ndim != 3
            or int(self.tokens.shape[0]) != 1
            or int(self.tokens.shape[1]) <= 0
            or int(self.tokens.shape[2]) <= 0
            or not self.tokens.is_contiguous()
            or not bool(torch.isfinite(self.tokens.detach()).all().item())
        ):
            raise IdentityRebinderContractError(
                "atlas tokens must be contiguous finite [1,M,D]"
            )
        _lower_sha256(self.source_video_sha256, label="source_video_sha256")
        _positive_int(self.source_frame_count, label="source_frame_count")
        _lower_sha256(self.construction_digest, label="construction_digest")

    @property
    def token_count(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def hidden_size(self) -> int:
        return int(self.tokens.shape[2])

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": ATLAS_SCHEMA_VERSION,
            "source_video_sha256": self.source_video_sha256,
            "source_frame_count": self.source_frame_count,
            "atlas_shape": list(self.tokens.shape),
            "construction_digest": self.construction_digest,
            "frame_axis_present": False,
            "frame_order_present": False,
            "timestamp_present": False,
            "explicit_temporal_motion_feature_present": False,
            "static_pose_leakage_information_theoretically_excluded": False,
            "appearance_multiplicity_or_dwell_time_leakage_excluded": False,
        }
        return {**value, "digest": object_sha256(value)}


class OrderlessIdentityAtlasEncoder(nn.Module):
    """Permutation-invariant atlas from frame-independent local RGB patches.

    Each frame is patched independently by the same Conv2d.  Patch tokens are
    then flattened into one set, with no frame/spatial positional embedding
    and no frame-boundary marker.  Learned slots pool the set by attention.
    This structurally removes trajectory, direction and event timing.  It
    cannot prove that static pose evidence is absent from individual RGB
    patches; that limitation is explicit in the research note.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        atlas_width: int = 128,
        atlas_tokens: int = 32,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_size = _positive_int(hidden_size, label="hidden_size")
        self.atlas_width = _positive_int(atlas_width, label="atlas_width")
        self.atlas_tokens = _positive_int(atlas_tokens, label="atlas_tokens")
        self.patch_size = _positive_int(patch_size, label="patch_size")
        if self.atlas_tokens > 256:
            raise IdentityRebinderContractError("atlas_tokens exceeds the closed prototype cap")
        self.patchifier = nn.Conv2d(
            3,
            self.atlas_width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
            dtype=torch.float32,
        )
        self.patch_norm = nn.LayerNorm(self.atlas_width, dtype=torch.float32)
        self.slot_queries = nn.Parameter(
            torch.empty(self.atlas_tokens, self.atlas_width, dtype=torch.float32)
        )
        self.output_norm = nn.LayerNorm(self.atlas_width, dtype=torch.float32)
        self.output_projection = nn.Linear(
            self.atlas_width, self.hidden_size, bias=False, dtype=torch.float32
        )
        nn.init.normal_(self.slot_queries, mean=0.0, std=self.atlas_width**-0.5)

    def architecture_receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": ATLAS_SCHEMA_VERSION,
            "input": "normalized_raw_rgb_[B,F,3,H,W]",
            "frame_encoder": "shared_conv2d_per_frame",
            "frame_axis_after_patchifier": False,
            "frame_position_embedding": False,
            "spatial_position_embedding": False,
            "frame_boundary_marker": False,
            "temporal_convolution": False,
            "frame_difference": False,
            "flow_pose_track_mask": False,
            "pooler": "learned_slot_attention_over_one_unordered_patch_set",
            "permutation_invariant_but_not_multiplicity_invariant": True,
            "static_pose_or_dwell_time_leakage_excluded": False,
            "hidden_size": self.hidden_size,
            "atlas_width": self.atlas_width,
            "atlas_tokens": self.atlas_tokens,
            "patch_size": self.patch_size,
        }
        return {**value, "digest": object_sha256(value)}

    def forward(self, source_frames: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(source_frames, torch.Tensor)
            or source_frames.layout != torch.strided
            or source_frames.device.type == "meta"
            or source_frames.dtype != torch.float32
            or source_frames.ndim != 5
            or int(source_frames.shape[0]) != 1
            or int(source_frames.shape[1]) < 2
            or int(source_frames.shape[2]) != 3
            or int(source_frames.shape[3]) < self.patch_size
            or int(source_frames.shape[4]) < self.patch_size
            or not source_frames.is_contiguous()
            or not bool(torch.isfinite(source_frames.detach()).all().item())
            or float(source_frames.detach().amin().item()) < -1.0
            or float(source_frames.detach().amax().item()) > 1.0
        ):
            raise IdentityRebinderContractError(
                "source frames must be contiguous finite FP32 [1,F>=2,3,H,W] in [-1,1]"
            )
        _, frames, channels, height, width = source_frames.shape
        flat = source_frames.reshape(frames, channels, height, width)
        with torch.autocast(device_type=source_frames.device.type, enabled=False):
            patches = self.patchifier(flat.float()).flatten(2).transpose(1, 2)
            # [F,P,C] -> [1,F*P,C].  No frame boundary survives this reshape.
            patch_set = self.patch_norm(patches.reshape(1, -1, self.atlas_width))
            logits = torch.einsum("mc,bnc->bmn", self.slot_queries, patch_set)
            logits = logits * (self.atlas_width**-0.5)
            weights = torch.softmax(logits, dim=-1)
            pooled = torch.einsum("bmn,bnc->bmc", weights, patch_set)
            output = self.output_projection(self.output_norm(pooled))
        return output.float().contiguous()

    def build_atlas(
        self,
        source_frames: torch.Tensor,
        *,
        source_video_sha256: str,
    ) -> IdentityAtlas:
        receipt = self.architecture_receipt()
        tokens = self(source_frames)
        construction = {
            "architecture_digest": receipt["digest"],
            "source_video_sha256": source_video_sha256,
            "source_frame_count": int(source_frames.shape[1]),
            "input_order_consumed": False,
        }
        return IdentityAtlas(
            tokens=tokens,
            source_video_sha256=source_video_sha256,
            source_frame_count=int(source_frames.shape[1]),
            construction_digest=object_sha256(construction),
        )


@dataclass(frozen=True)
class IdentityRebinderRoute:
    """One native branch route after append-padding and SP slicing."""

    total_tokens: int
    condition_tokens: int
    sequence_parallel_rank: int
    sequence_parallel_size: int
    branch_name: str
    sigma: float
    atlas: Optional[IdentityAtlas]
    enabled: bool = True
    low_sigma: float = DEFAULT_LOW_SIGMA
    high_sigma: float = DEFAULT_HIGH_SIGMA

    def __post_init__(self) -> None:
        total = _positive_int(self.total_tokens, label="total_tokens")
        if (
            isinstance(self.condition_tokens, bool)
            or not isinstance(self.condition_tokens, int)
            or not 0 <= self.condition_tokens < total
        ):
            raise IdentityRebinderContractError(
                "condition_tokens must identify a strict target suffix"
            )
        size = _positive_int(self.sequence_parallel_size, label="sequence_parallel_size")
        rank = self.sequence_parallel_rank
        if size not in ALLOWED_SP_SIZES:
            raise IdentityRebinderContractError("only SP1 tests and native SP4 are supported")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < size:
            raise IdentityRebinderContractError("SP rank lies outside its group")
        if self.branch_name not in SOURCE_MEMORY_BRANCHES | SOURCE_FREE_BRANCHES:
            raise IdentityRebinderContractError("branch_name is not a native RV2V axis")
        if not isinstance(self.enabled, bool):
            raise IdentityRebinderContractError("enabled must be boolean")
        mid_low_sigma_gate(
            self.sigma, low_sigma=self.low_sigma, high_sigma=self.high_sigma
        )
        if self.enabled and self.branch_name in SOURCE_MEMORY_BRANCHES:
            if not isinstance(self.atlas, IdentityAtlas):
                raise IdentityRebinderContractError("V/VI route requires one identity atlas")
        elif self.atlas is not None:
            raise IdentityRebinderContractError(
                "none/I/disabled route must not receive source identity memory"
            )

    @property
    def target_tokens(self) -> int:
        return self.total_tokens - self.condition_tokens

    @property
    def local_length(self) -> int:
        return math.ceil(self.total_tokens / self.sequence_parallel_size)

    @property
    def gate(self) -> float:
        if not self.enabled or self.branch_name not in SOURCE_MEMORY_BRANCHES:
            return 0.0
        return mid_low_sigma_gate(
            self.sigma, low_sigma=self.low_sigma, high_sigma=self.high_sigma
        )

    def local_target_selector(self, *, device: torch.device) -> torch.Tensor:
        selector = torch.cat(
            (
                torch.zeros(self.condition_tokens, dtype=torch.bool, device=device),
                torch.ones(self.target_tokens, dtype=torch.bool, device=device),
            )
        )
        padded = self.local_length * self.sequence_parallel_size
        if padded > self.total_tokens:
            selector = torch.cat(
                (
                    selector,
                    torch.zeros(
                        padded - self.total_tokens, dtype=torch.bool, device=device
                    ),
                )
            )
        start = self.sequence_parallel_rank * self.local_length
        return selector[start : start + self.local_length].contiguous()

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "branch_name": self.branch_name,
            "total_tokens": self.total_tokens,
            "condition_tokens": self.condition_tokens,
            "target_tokens": self.target_tokens,
            "sequence_parallel_rank": self.sequence_parallel_rank,
            "sequence_parallel_size": self.sequence_parallel_size,
            "sigma_hex": float(self.sigma).hex(),
            "gate_hex": float(self.gate).hex(),
            "atlas_receipt_digest": (
                self.atlas.receipt()["digest"] if self.atlas is not None else None
            ),
            "source_memory_owned_by_V_VI_only": True,
            "enabled": self.enabled,
        }
        return {**value, "digest": object_sha256(value)}


_ACTIVE_ROUTE: ContextVar[Optional[IdentityRebinderRoute]] = ContextVar(
    "bernini_identity_rebinder_v1_route", default=None
)


def active_route() -> Optional[IdentityRebinderRoute]:
    return _ACTIVE_ROUTE.get()


@contextmanager
def activate_route(route: IdentityRebinderRoute) -> Iterator[None]:
    if not isinstance(route, IdentityRebinderRoute):
        raise IdentityRebinderContractError("route must be IdentityRebinderRoute")
    if active_route() is not None:
        raise IdentityRebinderContractError("nested identity rebinder routes are forbidden")
    token: Token[Optional[IdentityRebinderRoute]] = _ACTIVE_ROUTE.set(route)
    try:
        yield
    finally:
        _ACTIVE_ROUTE.reset(token)


class TargetQueryIdentityCrossAttention(nn.Module):
    """Frozen base output projection plus target-only low-rank memory residual."""

    def __init__(
        self,
        base: nn.Module,
        *,
        rank: int,
        alpha: float,
        require_explicit_route: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise IdentityRebinderContractError("attn1.to_out[0] must be nn.Linear")
        if base.in_features != base.out_features:
            raise IdentityRebinderContractError("Bernini output projection must be square")
        self.base = base
        self.hidden_size = int(base.in_features)
        self.rank = _positive_int(rank, label="cross-attention rank")
        if self.rank > self.hidden_size:
            raise IdentityRebinderContractError("cross-attention rank exceeds hidden size")
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(float(alpha))
            or float(alpha) <= 0.0
        ):
            raise IdentityRebinderContractError("alpha must be finite and positive")
        self.alpha = float(alpha)
        if not isinstance(require_explicit_route, bool):
            raise IdentityRebinderContractError(
                "require_explicit_route must be boolean"
            )
        self.require_explicit_route = require_explicit_route
        self.query = nn.Linear(self.hidden_size, self.rank, bias=False, dtype=torch.float32)
        self.key = nn.Linear(self.hidden_size, self.rank, bias=False, dtype=torch.float32)
        self.value = nn.Linear(self.hidden_size, self.rank, bias=False, dtype=torch.float32)
        self.output = nn.Linear(self.rank, self.hidden_size, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.output.weight)

    @property
    def weight(self) -> Any:
        return self.base.weight

    @property
    def bias(self) -> Any:
        return self.base.bias

    @property
    def scale(self) -> float:
        return self.alpha / float(self.rank)

    def adapter_delta(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(hidden_states, torch.Tensor)
            or hidden_states.layout != torch.strided
            or hidden_states.device.type == "meta"
            or hidden_states.ndim != 3
            or int(hidden_states.shape[0]) != 1
            or int(hidden_states.shape[2]) != self.hidden_size
        ):
            raise IdentityRebinderContractError(
                "Bernini output projection input must be dense [1,local_N,D]"
            )
        route = active_route()
        result = torch.zeros(
            (int(hidden_states.shape[0]), int(hidden_states.shape[1]), self.hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if route is None:
            if self.require_explicit_route:
                raise IdentityRebinderContractError(
                    "strict identity rebinder forward lacks an authenticated route"
                )
            return result
        if route.gate == 0.0:
            return result
        if route.atlas is None or route.atlas.hidden_size != self.hidden_size:
            raise IdentityRebinderContractError("atlas hidden size differs from Bernini")
        selector = route.local_target_selector(device=hidden_states.device)
        if int(hidden_states.shape[1]) != int(selector.numel()):
            raise IdentityRebinderContractError(
                "local sequence differs from append-pad/SP target selector"
            )
        if not bool(selector.any().item()):
            return result
        memory = route.atlas.tokens
        if memory.device != hidden_states.device:
            raise IdentityRebinderContractError("atlas and hidden states must share a device")
        target_queries = hidden_states[:, selector, :]
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            query = self.query(target_queries.float())
            key = self.key(memory.float())
            value = self.value(memory.float())
            logits = torch.matmul(query, key.transpose(-1, -2)) * (self.rank**-0.5)
            weights = torch.softmax(logits, dim=-1)
            low_rank = torch.matmul(weights, value)
            delta = self.output(low_rank) * self.scale * route.gate
        result[:, selector, :] = delta.to(hidden_states.dtype)
        return result.contiguous()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(hidden_states, torch.Tensor)
            or hidden_states.layout != torch.strided
            or hidden_states.device.type == "meta"
            or hidden_states.ndim != 3
            or int(hidden_states.shape[0]) != 1
            or int(hidden_states.shape[2]) != self.hidden_size
        ):
            raise IdentityRebinderContractError(
                "Bernini output projection input must be dense [1,local_N,D]"
            )
        base = self.base(hidden_states)
        route = active_route()
        if route is None:
            if self.require_explicit_route:
                raise IdentityRebinderContractError(
                    "strict identity rebinder forward lacks an authenticated route"
                )
            return base
        if route.gate == 0.0:
            return base
        # At initialization adapter_delta is numerically zero but remains in
        # the graph, so the zero output projection receives the first update.
        # Returning ``base`` early here would create a silent dead adapter.
        return base + self.adapter_delta(hidden_states).to(base.dtype)


@dataclass
class IdentityRebinderHandle:
    transformer: nn.Module
    atlas_encoder: OrderlessIdentityAtlasEncoder
    wrappers: tuple[tuple[int, TargetQueryIdentityCrossAttention], ...]
    originals: tuple[tuple[int, nn.Module], ...]
    block_indices: tuple[int, ...]
    original_patch_embedding_id: int
    runtime_source_commit: str
    model_revision: str
    checkpoint_manifest_sha256: str
    transformer_config_digest: str
    restored: bool = False

    def trainable_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        if self.restored:
            raise IdentityRebinderContractError("identity rebinder has been restored")
        result: list[tuple[str, nn.Parameter]] = [
            (f"atlas_encoder.{name}", parameter)
            for name, parameter in self.atlas_encoder.named_parameters()
        ]
        for index, wrapper in self.wrappers:
            for name in ("query", "key", "value", "output"):
                projection = getattr(wrapper, name)
                result.append(
                    (f"blocks.{index}.attn1.to_out.0.identity_rebinder.{name}.weight", projection.weight)
                )
        if len({id(parameter) for _, parameter in result}) != len(result):
            raise IdentityRebinderContractError("trainable parameter aliases another")
        if any(not parameter.requires_grad for _, parameter in result):
            raise IdentityRebinderContractError("rebinder parameter is unexpectedly frozen")
        return tuple(result)

    def base_parameters_frozen(self) -> bool:
        trainable = {id(parameter) for _, parameter in self.trainable_named_parameters()}
        return all(
            id(parameter) in trainable or not parameter.requires_grad
            for parameter in self.transformer.parameters()
        )

    def build_atlas(
        self, source_frames: torch.Tensor, *, source_video_sha256: str
    ) -> IdentityAtlas:
        if self.restored:
            raise IdentityRebinderContractError("cannot use a restored rebinder")
        return self.atlas_encoder.build_atlas(
            source_frames, source_video_sha256=source_video_sha256
        )

    @contextmanager
    def route(self, route: IdentityRebinderRoute) -> Iterator[None]:
        if self.restored:
            raise IdentityRebinderContractError("cannot route a restored rebinder")
        with activate_route(route):
            yield

    def state_dict_for_save(self) -> Mapping[str, torch.Tensor]:
        return {
            name: parameter.detach().float().cpu().contiguous()
            for name, parameter in self.trainable_named_parameters()
        }

    def receipt(self) -> Mapping[str, Any]:
        patch = getattr(self.transformer, "patch_embedding", None)
        value = {
            "schema_version": SCHEMA_VERSION,
            "gpu_validated": False,
            "scientific_quality_claim": False,
            "runtime_source_commit": self.runtime_source_commit,
            "model_revision": self.model_revision,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "transformer_config_digest": self.transformer_config_digest,
            "transformer_class": (
                f"{self.transformer.__class__.__module__}."
                f"{self.transformer.__class__.__name__}"
            ),
            "block_indices": list(self.block_indices),
            "default_middle_scope_8_through_22": self.block_indices
            == DEFAULT_BLOCK_INDICES,
            "frozen_early_blocks": list(range(8)),
            "frozen_late_blocks": list(range(23, TOTAL_BLOCKS_1P3B)),
            "insertion": "blocks[i].attn1.to_out[0]",
            "target_queries_only": True,
            "direct_write_scope_only_later_layers_may_propagate": True,
            "memory_keys_values_only": True,
            "condition_rows_written": False,
            "source_memory_branches": sorted(SOURCE_MEMORY_BRANCHES),
            "source_free_branches": sorted(SOURCE_FREE_BRANCHES),
            "high_sigma_exactly_off": True,
            "explicit_route_required": True,
            "low_sigma_hex": DEFAULT_LOW_SIGMA.hex(),
            "high_sigma_hex": DEFAULT_HIGH_SIGMA.hex(),
            "zero_initialized_output_projection": True,
            "patch_embedding_untouched": id(patch) == self.original_patch_embedding_id,
            "patch_vae_latent_untouched": True,
            "base_parameters_frozen": self.base_parameters_frozen(),
            "atlas": self.atlas_encoder.architecture_receipt(),
            "no_mask_track_flow_pose": True,
            "no_source_temporal_order_or_difference": True,
            "static_pose_leakage_information_theoretically_excluded": False,
            "trainable": [
                {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
                for name, parameter in self.trainable_named_parameters()
            ],
        }
        return {**value, "digest": object_sha256(value)}

    def restore(self) -> None:
        if self.restored or active_route() is not None:
            raise IdentityRebinderContractError("rebinder cannot be restored now")
        blocks = tuple(getattr(self.transformer, "blocks", ()))
        if len(blocks) != TOTAL_BLOCKS_1P3B:
            raise IdentityRebinderContractError("transformer block count changed")
        if id(getattr(self.transformer, "patch_embedding", None)) != self.original_patch_embedding_id:
            raise IdentityRebinderContractError("native patch embedding changed")
        for index, original in self.originals:
            blocks[index].attn1.to_out[0] = original
        self.restored = True


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _strict_transformer_config_receipt(transformer: nn.Module) -> Mapping[str, Any]:
    observed_class = (
        transformer.__class__.__module__,
        transformer.__class__.__name__,
    )
    expected_class = (
        PINNED_TRANSFORMER_CLASS_MODULE,
        PINNED_TRANSFORMER_CLASS_NAME,
    )
    if observed_class != expected_class:
        raise IdentityRebinderContractError(
            "transformer class is not the pinned Bernini WanTransformer3DModel"
        )
    config = getattr(transformer, "config", None)
    if config is None:
        raise IdentityRebinderContractError("pinned Bernini transformer config is absent")
    observed: dict[str, Any] = {}
    for name, expected in PINNED_TRANSFORMER_CONFIG.items():
        value = _config_value(config, name)
        if name == "patch_size" and isinstance(value, (list, tuple)):
            value = tuple(value)
        if value != expected:
            raise IdentityRebinderContractError(
                f"pinned Bernini transformer config differs at {name}"
            )
        observed[name] = list(value) if isinstance(value, tuple) else value
    observed["hidden_size"] = (
        int(observed["num_attention_heads"])
        * int(observed["attention_head_dim"])
    )
    if observed["hidden_size"] != HIDDEN_SIZE_1P3B:
        raise IdentityRebinderContractError("pinned Bernini hidden size differs")
    value = {
        "class_module": observed_class[0],
        "class_name": observed_class[1],
        "config": observed,
    }
    return {**value, "digest": object_sha256(value)}


def install_identity_rebinder_v1(
    transformer: nn.Module,
    *,
    runtime_source_commit: str,
    model_revision: str,
    checkpoint_manifest_sha256: str,
    rank: int = 64,
    alpha: float = 64.0,
    atlas_width: int = 128,
    atlas_tokens: int = 32,
    atlas_patch_size: int = 16,
    block_indices: Sequence[int] = DEFAULT_BLOCK_INDICES,
) -> IdentityRebinderHandle:
    """Install the fail-closed Bernini block adapter and orderless encoder."""

    if not isinstance(transformer, nn.Module):
        raise IdentityRebinderContractError("transformer must be nn.Module")
    if runtime_source_commit != PINNED_BERNINI_SOURCE_COMMIT:
        raise IdentityRebinderContractError("Bernini runtime source commit is not pinned")
    if model_revision != PINNED_BERNINI_MODEL_REVISION:
        raise IdentityRebinderContractError("Bernini model revision is not pinned")
    _lower_sha256(
        checkpoint_manifest_sha256, label="checkpoint_manifest_sha256"
    )
    config_receipt = _strict_transformer_config_receipt(transformer)
    if any(parameter.requires_grad for parameter in transformer.parameters()):
        raise IdentityRebinderContractError("freeze the complete transformer first")
    blocks = tuple(getattr(transformer, "blocks", ()))
    patch = getattr(transformer, "patch_embedding", None)
    indices = tuple(block_indices)
    if (
        len(blocks) != TOTAL_BLOCKS_1P3B
        or not isinstance(patch, nn.Conv3d)
        or patch.in_channels != 16
        or patch.out_channels != HIDDEN_SIZE_1P3B
        or tuple(patch.kernel_size) != (1, 2, 2)
        or tuple(patch.stride) != (1, 2, 2)
        or not callable(getattr(transformer, "patch_vae_latent", None))
    ):
        raise IdentityRebinderContractError("audited Bernini 1.3B structure differs")
    if indices not in ALLOWED_BLOCK_SCOPES:
        raise IdentityRebinderContractError("block scope is not preregistered")
    hidden_size = int(patch.out_channels)
    originals: list[tuple[int, nn.Module]] = []
    for index, block in enumerate(blocks):
        attention = getattr(block, "attn1", None)
        output = getattr(attention, "to_out", None)
        if (
            not isinstance(output, nn.ModuleList)
            or len(output) != 2
            or not isinstance(output[0], nn.Linear)
            or output[0].in_features != hidden_size
            or output[0].out_features != hidden_size
        ):
            raise IdentityRebinderContractError(f"block {index} attn1 output differs")
        if index in indices:
            originals.append((index, output[0]))

    device = patch.weight.device
    encoder = OrderlessIdentityAtlasEncoder(
        hidden_size=hidden_size,
        atlas_width=atlas_width,
        atlas_tokens=atlas_tokens,
        patch_size=atlas_patch_size,
    ).to(device=device)
    wrappers: list[tuple[int, TargetQueryIdentityCrossAttention]] = []
    try:
        for index, original in originals:
            wrapper = TargetQueryIdentityCrossAttention(
                original,
                rank=rank,
                alpha=alpha,
                require_explicit_route=True,
            ).to(device=device)
            blocks[index].attn1.to_out[0] = wrapper
            wrappers.append((index, wrapper))
    except Exception:
        for index, original in originals:
            blocks[index].attn1.to_out[0] = original
        raise
    handle = IdentityRebinderHandle(
        transformer=transformer,
        atlas_encoder=encoder,
        wrappers=tuple(wrappers),
        originals=tuple(originals),
        block_indices=indices,
        original_patch_embedding_id=id(patch),
        runtime_source_commit=runtime_source_commit,
        model_revision=model_revision,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        transformer_config_digest=str(config_receipt["digest"]),
    )
    if not handle.base_parameters_frozen() or not handle.receipt()["patch_embedding_untouched"]:
        handle.restore()
        raise IdentityRebinderContractError("identity rebinder scope closure failed")
    return handle


@dataclass(frozen=True)
class _AuthenticatedNativePack:
    branch_name: str
    latent_input: torch.Tensor
    total_tokens: int
    condition_tokens: int
    target_tokens: int
    input_data_ptr: int
    input_version: Optional[int]
    input_dtype: torch.dtype
    input_device: torch.device

    def assert_live(self) -> None:
        if (
            id(self.latent_input) <= 0
            or int(self.latent_input.data_ptr()) != self.input_data_ptr
            or _tensor_version_or_none(self.latent_input) != self.input_version
            or tuple(self.latent_input.shape)
            != (1, self.total_tokens, HIDDEN_SIZE_1P3B)
            or self.latent_input.dtype != self.input_dtype
            or self.latent_input.device != self.input_device
            or not self.latent_input.is_contiguous()
        ):
            raise IdentityRebinderContractError(
                f"authenticated native {self.branch_name} pack changed"
            )


def _native_target_suffix(
    latent_input: torch.Tensor,
    *,
    branch_name: str,
    target_tokens: int,
) -> _AuthenticatedNativePack:
    target_count = _positive_int(target_tokens, label="native target token count")
    if (
        not isinstance(latent_input, torch.Tensor)
        or latent_input.layout != torch.strided
        or latent_input.device.type == "meta"
        or latent_input.ndim != 3
        or int(latent_input.shape[0]) != 1
        or int(latent_input.shape[2]) != HIDDEN_SIZE_1P3B
        or not latent_input.is_contiguous()
        or latent_input.dtype not in (torch.float32, torch.bfloat16)
    ):
        raise IdentityRebinderContractError(
            f"native {branch_name} latent pack differs"
        )
    total = int(latent_input.shape[1])
    if target_count > total:
        raise IdentityRebinderContractError(
            f"native {branch_name} pack is shorter than the none target branch"
        )
    condition = total - target_count
    if branch_name == "none" and condition != 0:
        raise IdentityRebinderContractError("native none pack contains condition rows")
    if branch_name in SOURCE_MEMORY_BRANCHES and condition <= 0:
        raise IdentityRebinderContractError(
            f"native {branch_name} pack lacks video condition rows"
        )
    packet = _AuthenticatedNativePack(
        branch_name=branch_name,
        latent_input=latent_input,
        total_tokens=total,
        condition_tokens=condition,
        target_tokens=target_count,
        input_data_ptr=int(latent_input.data_ptr()),
        input_version=_tensor_version_or_none(latent_input),
        input_dtype=latent_input.dtype,
        input_device=latent_input.device,
    )
    packet.assert_live()
    return packet


def _native_scheduler_sigma(scheduler: Any, timestep: torch.Tensor) -> tuple[int, float]:
    observed_class = (
        scheduler.__class__.__module__,
        scheduler.__class__.__name__,
    )
    if observed_class != (
        PINNED_SCHEDULER_CLASS_MODULE,
        PINNED_SCHEDULER_CLASS_NAME,
    ):
        raise IdentityRebinderContractError("native scheduler class is not pinned UniPC")
    timesteps = getattr(scheduler, "timesteps", None)
    sigmas = getattr(scheduler, "sigmas", None)
    if (
        not isinstance(timestep, torch.Tensor)
        or timestep.numel() != 1
        or timestep.device.type == "meta"
        or not isinstance(timesteps, torch.Tensor)
        or timesteps.ndim != 1
        or timesteps.device.type != "cpu"
        or not isinstance(sigmas, torch.Tensor)
        or sigmas.ndim != 1
        or sigmas.device.type != "cpu"
        or sigmas.dtype != torch.float32
        or int(sigmas.numel()) != int(timesteps.numel()) + 1
    ):
        raise IdentityRebinderContractError("native UniPC timestep/sigma storage differs")
    timestep_value = timestep.detach().to(device="cpu").reshape(()).item()
    matches = (timesteps == timestep_value).nonzero(as_tuple=False).flatten()
    if int(matches.numel()) != 1:
        raise IdentityRebinderContractError(
            "native timestep does not identify exactly one scheduler coordinate"
        )
    index = int(matches[0].item())
    step_index = getattr(scheduler, "step_index", None)
    if step_index is not None and int(step_index) != index:
        raise IdentityRebinderContractError(
            "native scheduler live step index differs from timestep coordinate"
        )
    sigma = float(sigmas[index].item())
    _finite_unit(sigma, label="native physical sigma")
    return index, sigma


def _native_parallel_coordinate() -> tuple[int, int, Mapping[str, Any]]:
    try:
        bernini_parallel = importlib.import_module("bernini.parallel")
        state = bernini_parallel.get_parallel_state()
    except Exception as error:
        raise IdentityRebinderContractError(
            "live Bernini parallel state is unavailable"
        ) from error
    state_type = type(state)
    if not state_type.__module__.startswith("bernini.parallel"):
        raise IdentityRebinderContractError("parallel state is not native Bernini")
    enabled = getattr(state, "ulysses_enabled", None)
    if type(enabled) is not bool:
        raise IdentityRebinderContractError("Bernini Ulysses enabled flag differs")
    if not enabled:
        if (
            getattr(state, "ulysses_size", 1) != 1
            or getattr(state, "ulysses_rank", 0) != 0
        ):
            raise IdentityRebinderContractError("disabled Ulysses state is inconsistent")
        return 0, 1, {
            "parallel_state_type": f"{state_type.__module__}.{state_type.__qualname__}",
            "ulysses_enabled": False,
            "ulysses_rank": 0,
            "ulysses_size": 1,
            "backend": None,
        }

    import torch.distributed as dist

    group = getattr(state, "ulysses_group", None)
    group_type = getattr(dist, "ProcessGroup", None)
    if group_type is None:
        group_type = getattr(
            getattr(dist, "distributed_c10d", None), "ProcessGroup", None
        )
    rank = getattr(state, "ulysses_rank", None)
    size = getattr(state, "ulysses_size", None)
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or group_type is None
        or not isinstance(group, group_type)
        or type(rank) is not int
        or size != 4
        or not 0 <= rank < 4
        or dist.get_world_size(group) != 4
        or dist.get_rank(group) != rank
        or str(dist.get_backend(group)).lower() != "nccl"
    ):
        raise IdentityRebinderContractError(
            "native runtime is not authenticated Bernini Ulysses-SP4/NCCL"
        )
    global_ranks: list[Any] = [None] * 4
    dist.all_gather_object(global_ranks, int(dist.get_rank()), group=group)
    if (
        any(type(value) is not int for value in global_ranks)
        or len(set(global_ranks)) != 4
        or tuple(global_ranks)
        != tuple(range(global_ranks[0], global_ranks[0] + 4))
        or global_ranks[0] % 4 != 0
    ):
        raise IdentityRebinderContractError("native SP4 rank membership differs")
    return rank, 4, {
        "parallel_state_type": f"{state_type.__module__}.{state_type.__qualname__}",
        "ulysses_enabled": True,
        "ulysses_rank": rank,
        "ulysses_size": 4,
        "backend": "nccl",
        "ordered_global_ranks": global_ranks,
    }


class _NativeRV2VStep:
    def __init__(
        self,
        *,
        handle: IdentityRebinderHandle,
        atlas: IdentityAtlas,
        packs: Sequence[_AuthenticatedNativePack],
        sigma: float,
        schedule_index: int,
        sp_rank: int,
        sp_size: int,
        parallel_receipt: Mapping[str, Any],
    ) -> None:
        self.handle = handle
        self.atlas = atlas
        self.packs_by_input_id = {id(packet.latent_input): packet for packet in packs}
        if len(self.packs_by_input_id) != len(tuple(packs)):
            raise IdentityRebinderContractError("native branch input objects alias")
        if len({packet.input_data_ptr for packet in packs}) != len(tuple(packs)):
            raise IdentityRebinderContractError("native branch input storages alias")
        self.sigma = sigma
        self.schedule_index = schedule_index
        self.sp_rank = sp_rank
        self.sp_size = sp_size
        self.parallel_receipt = dict(parallel_receipt)
        self.expected_calls = {"none": 1, "V": 1, "VI": 2}
        self.calls = {name: 0 for name in self.expected_calls}
        self.closed = False
        self.receipt: Optional[Mapping[str, Any]] = None

    @contextmanager
    def route_for_call(
        self,
        *,
        latent_input: torch.Tensor,
    ) -> Iterator[IdentityRebinderRoute]:
        if self.closed:
            raise IdentityRebinderContractError("native RV2V step is closed")
        packet = self.packs_by_input_id.get(id(latent_input))
        if packet is None:
            raise IdentityRebinderContractError(
                "native forward pack was not minted by this RV2V step"
            )
        packet.assert_live()
        if self.calls[packet.branch_name] >= self.expected_calls[packet.branch_name]:
            raise IdentityRebinderContractError(
                f"native {packet.branch_name} branch was called too many times"
            )
        route = IdentityRebinderRoute(
            total_tokens=packet.total_tokens,
            condition_tokens=packet.condition_tokens,
            sequence_parallel_rank=self.sp_rank,
            sequence_parallel_size=self.sp_size,
            branch_name=packet.branch_name,
            sigma=self.sigma,
            atlas=(self.atlas if packet.branch_name in SOURCE_MEMORY_BRANCHES else None),
        )
        self.calls[packet.branch_name] += 1
        with self.handle.route(route):
            yield route

    def close(self) -> Mapping[str, Any]:
        if self.closed or self.calls != self.expected_calls:
            raise IdentityRebinderContractError(
                "native RV2V branch call inventory is incomplete"
            )
        value = {
            "schema_version": "bernini-identity-rebinder-native-rv2v-step-v1",
            "schedule_index": self.schedule_index,
            "sigma_hex": self.sigma.hex(),
            "branch_calls": dict(self.calls),
            "target_suffix_derived_from_none_branch_length": True,
            "source_video_sha256": self.atlas.source_video_sha256,
            "parallel": self.parallel_receipt,
        }
        self.receipt = {**value, "digest": object_sha256(value)}
        self.closed = True
        return self.receipt


class NativeRV2VIdentityRouteBinder:
    """Authenticate official RV2V packs instead of trusting a branch string.

    The official sampler must hand this wrapper the exact ``none_inp``,
    ``v_inp`` and ``vi_inp`` objects before its four
    forwards.  ``route_for_call`` then identifies ownership by object/storage
    identity, derives the shared target suffix from the native none-branch
    length and each conditioned branch length, reads physical sigma from the
    pinned scheduler and reads SP rank/size from the live Bernini process
    group.  No segmentation mask, CLI branch, sigma or rank is accepted.
    """

    def __init__(
        self,
        *,
        handle: IdentityRebinderHandle,
        scheduler: Any,
        atlas: IdentityAtlas,
    ) -> None:
        if not isinstance(handle, IdentityRebinderHandle) or handle.restored:
            raise IdentityRebinderContractError("live rebinder handle is required")
        if not isinstance(atlas, IdentityAtlas):
            raise IdentityRebinderContractError("native binder requires an identity atlas")
        self.handle = handle
        self.scheduler = scheduler
        self.atlas = atlas
        self._step_active = False
        self.last_step_receipt: Optional[Mapping[str, Any]] = None

    @contextmanager
    def native_rv2v_step(
        self,
        *,
        timestep: torch.Tensor,
        none_input: torch.Tensor,
        video_input: torch.Tensor,
        video_image_input: torch.Tensor,
    ) -> Iterator[_NativeRV2VStep]:
        if self._step_active or active_route() is not None:
            raise IdentityRebinderContractError("nested native RV2V steps are forbidden")
        schedule_index, sigma = _native_scheduler_sigma(self.scheduler, timestep)
        sp_rank, sp_size, parallel_receipt = _native_parallel_coordinate()
        if not isinstance(none_input, torch.Tensor) or none_input.ndim != 3:
            raise IdentityRebinderContractError("native none input differs")
        target_tokens = int(none_input.shape[1])
        packs = (
            _native_target_suffix(
                none_input, branch_name="none", target_tokens=target_tokens
            ),
            _native_target_suffix(
                video_input, branch_name="V", target_tokens=target_tokens
            ),
            _native_target_suffix(
                video_image_input, branch_name="VI", target_tokens=target_tokens
            ),
        )
        target_counts = {packet.target_tokens for packet in packs}
        if len(target_counts) != 1:
            raise IdentityRebinderContractError(
                "native RV2V branches do not share one target suffix length"
            )
        step = _NativeRV2VStep(
            handle=self.handle,
            atlas=self.atlas,
            packs=packs,
            sigma=sigma,
            schedule_index=schedule_index,
            sp_rank=sp_rank,
            sp_size=sp_size,
            parallel_receipt=parallel_receipt,
        )
        self.last_step_receipt = None
        self._step_active = True
        try:
            yield step
            self.last_step_receipt = step.close()
        finally:
            self._step_active = False


@dataclass(frozen=True)
class IdentityRebinderPretrainLoss:
    total: torch.Tensor
    recovery: torch.Tensor
    wrong_identity_ranking: torch.Tensor
    view_consistency: torch.Tensor
    identity_contrast: torch.Tensor


def identity_rebinder_pretrain_objective(
    *,
    correct_prediction: torch.Tensor,
    wrong_prediction: torch.Tensor,
    target: torch.Tensor,
    canonical_atlas: torch.Tensor,
    shuffled_atlas: torch.Tensor,
    dropped_atlas: torch.Tensor,
    resampled_atlas: torch.Tensor,
    wrong_atlas: torch.Tensor,
    recovery_rank_margin: float = 0.05,
    identity_margin: float = 0.10,
) -> IdentityRebinderPretrainLoss:
    """Raw-video disjoint recovery + identity contrast + set-view consistency."""

    tensors = {
        "correct_prediction": correct_prediction,
        "wrong_prediction": wrong_prediction,
        "target": target,
        "canonical_atlas": canonical_atlas,
        "shuffled_atlas": shuffled_atlas,
        "dropped_atlas": dropped_atlas,
        "resampled_atlas": resampled_atlas,
        "wrong_atlas": wrong_atlas,
    }
    if any(
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or not value.is_floating_point()
        or not bool(torch.isfinite(value.detach()).all().item())
        for value in tensors.values()
    ):
        raise IdentityRebinderContractError("pretrain tensors must be finite dense floats")
    if correct_prediction.shape != wrong_prediction.shape or correct_prediction.shape != target.shape:
        raise IdentityRebinderContractError("prediction and target shapes differ")
    if correct_prediction.ndim < 2 or int(correct_prediction.shape[0]) <= 0:
        raise IdentityRebinderContractError(
            "predictions require an explicit non-empty batch dimension"
        )
    atlas_shape = canonical_atlas.shape
    if len(atlas_shape) != 3 or any(
        value.shape != atlas_shape
        for value in (shuffled_atlas, dropped_atlas, resampled_atlas, wrong_atlas)
    ):
        raise IdentityRebinderContractError("all fixed-slot atlas shapes must match")
    if int(atlas_shape[0]) != int(correct_prediction.shape[0]):
        raise IdentityRebinderContractError(
            "atlas and prediction batch dimensions differ"
        )
    rank_margin = _finite_unit(recovery_rank_margin, label="recovery_rank_margin")
    contrast_margin = _finite_unit(identity_margin, label="identity_margin")

    # The margin is evaluated independently for every episode.  Reducing the
    # whole batch before the hinge lets one very easy negative hide another
    # sample whose wrong identity reconstructs better than the correct one.
    correct_error_per_example = (
        (correct_prediction.float() - target.float())
        .square()
        .flatten(start_dim=1)
        .mean(dim=1)
    )
    wrong_error_per_example = (
        (wrong_prediction.float() - target.float())
        .square()
        .flatten(start_dim=1)
        .mean(dim=1)
    )
    recovery = correct_error_per_example.mean()
    wrong_ranking = F.relu(
        correct_error_per_example.new_tensor(rank_margin)
        + correct_error_per_example
        - wrong_error_per_example
    ).mean()

    canonical = F.normalize(canonical_atlas.float(), dim=-1)
    views = tuple(
        F.normalize(value.float(), dim=-1)
        for value in (shuffled_atlas, dropped_atlas, resampled_atlas)
    )
    consistency = torch.stack(
        [(canonical - view).square().mean() for view in views]
    ).mean()

    canonical_summary = F.normalize(canonical.mean(dim=1), dim=-1)
    positive_summary = F.normalize(
        torch.stack([view.mean(dim=1) for view in views]).mean(dim=0), dim=-1
    )
    wrong_summary = F.normalize(wrong_atlas.float().mean(dim=1), dim=-1)
    positive_cosine = (canonical_summary * positive_summary).sum(dim=-1)
    wrong_cosine = (canonical_summary * wrong_summary).sum(dim=-1)
    identity_contrast = F.relu(
        wrong_cosine - positive_cosine + canonical.new_tensor(contrast_margin)
    ).mean()

    total = recovery + 0.5 * wrong_ranking + 0.25 * consistency + 0.25 * identity_contrast
    if not bool(torch.isfinite(total.detach()).all().item()):
        raise IdentityRebinderContractError("pretrain objective is non-finite")
    return IdentityRebinderPretrainLoss(
        total=total,
        recovery=recovery,
        wrong_identity_ranking=wrong_ranking,
        view_consistency=consistency,
        identity_contrast=identity_contrast,
    )


def pretrain_objective_receipt() -> Mapping[str, Any]:
    value = {
        "schema_version": PRETRAIN_OBJECTIVE_SCHEMA,
        "supervision": "raw_video_only",
        "memory_target_authority_clip_disjoint": True,
        "memory_target_exact_and_near_duplicate_rejected_by_builder": True,
        "recovery": "heldout_target_frame_denoising_or_feature_recovery",
        "correct_vs_wrong_identity": "per_example_recovery_hinge_then_batch_mean_and_atlas_cosine",
        "consistency_views": ["frame_shuffle", "frame_drop", "frame_resample"],
        "action_labels": False,
        "edited_targets": False,
        "temporal_order_input": False,
        "objective": "Lrec+0.5Lwrong+0.25Lconsistency+0.25Lidentity",
        "train_sigma_support": "sigma<0.75_with_model_gate_applied",
    }
    return {**value, "digest": object_sha256(value)}


__all__ = [
    "ALL_EARLY_MID_ABLATION",
    "ATLAS_SCHEMA_VERSION",
    "DEFAULT_BLOCK_INDICES",
    "IdentityAtlas",
    "IdentityRebinderContractError",
    "IdentityRebinderHandle",
    "IdentityRebinderPretrainLoss",
    "IdentityRebinderRoute",
    "NativeRV2VIdentityRouteBinder",
    "OrderlessIdentityAtlasEncoder",
    "PRETRAIN_OBJECTIVE_SCHEMA",
    "PINNED_BERNINI_MODEL_REVISION",
    "PINNED_BERNINI_SOURCE_COMMIT",
    "PINNED_TRANSFORMER_CONFIG",
    "SCHEMA_VERSION",
    "TargetQueryIdentityCrossAttention",
    "activate_route",
    "active_route",
    "identity_rebinder_pretrain_objective",
    "install_identity_rebinder_v1",
    "mid_low_sigma_gate",
    "pretrain_objective_receipt",
]
