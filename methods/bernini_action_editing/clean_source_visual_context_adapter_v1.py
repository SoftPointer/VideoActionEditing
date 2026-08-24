#!/usr/bin/env python3
"""Persistent clean-source visual context for Bernini action editing.

This module adds a *new* attention path; it never replaces or replays native
self-attention K/V.  A frozen VAE supplies one detached clean source latent.
``CleanSourceVisualEncoder`` patchifies that latent (or the registered
same-noise forward-noised source variant), normalizes/projects the
patches, adds an explicit clean-source role embedding, and returns a bounded
source-only memory.  At selected frozen Bernini blocks, current local target
states query learned K/V made only from that memory.  The residual output
projection is exactly zero at installation, so step 0 is the frozen base.

The implementation is deliberately isolated from the disproven hard source-KV
replacement and no-op velocity-residual routes.  It is a structural/training
slice, not a decoded-quality claim.  The full Bernini GPU runner still has to
bind the route to its authenticated source/target token layout and no-op flow
matching batch; this file provides the fail-closed adapter needed by that
runner.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterator, Mapping, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F


SCHEMA_VERSION = "bernini-clean-source-visual-context-adapter-v1"
MEMORY_SCHEMA_VERSION = "bernini-clean-source-visual-memory-v1"
TOTAL_BLOCKS_1P3B = 30
HIDDEN_SIZE_1P3B = 1536
LATENT_CHANNELS = 16
DEFAULT_PATCH_SIZE = (1, 4, 4)
DEFAULT_ENCODER_WIDTH = 256
DEFAULT_MEMORY_TOKEN_CAP = 1024
DEFAULT_ATTENTION_WIDTH = 64
DEFAULT_ATTENTION_HEADS = 8
MEMORY_INPUT_KINDS = (
    "clean_source",
    "same_noise_forward_noised_source",
)
# Sparse middle-block reads keep the first eight blocks and final synthesis
# blocks completely native.  This is an engineering candidate scope only: it
# cannot authorize optimizer training until decoded Stage-A localization has
# passed both containing middle bands.  The four exact indices remain
# preregistered sparse representatives; Stage-A does not claim per-block
# causal localization inside either band.
DEFAULT_BLOCK_INDICES = (8, 12, 16, 20)
ALLOWED_BLOCK_SCOPES = {
    DEFAULT_BLOCK_INDICES,
    tuple(range(8, 23)),
}
ALLOWED_SP_SIZES = {1, 4}

PINNED_BERNINI_SOURCE_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
PINNED_BERNINI_MODEL_REVISION = "ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce"
PINNED_TRANSFORMER_CLASS_MODULE = "bernini.models.transformer_wan"
PINNED_TRANSFORMER_CLASS_NAME = "WanTransformer3DModel"
PINNED_TRANSFORMER_CONFIG = {
    "num_layers": TOTAL_BLOCKS_1P3B,
    "num_attention_heads": 12,
    "attention_head_dim": 128,
    "in_channels": LATENT_CHANNELS,
    "out_channels": LATENT_CHANNELS,
    "patch_size": (1, 2, 2),
    "ffn_dim": 8960,
    "text_dim": 4096,
    "added_kv_proj_dim": None,
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
}


class CleanSourceVisualContextError(RuntimeError):
    """Raised instead of accepting ambiguous source memory or routing."""


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
        raise CleanSourceVisualContextError(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CleanSourceVisualContextError(f"{label} must be a positive integer")
    return value


def _lower_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CleanSourceVisualContextError(f"{label} must be lowercase SHA-256")
    return value


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


@dataclass(frozen=True)
class CleanSourceVisualMemory:
    """Registered source-derived visual memory with arm-bound provenance.

    ``clean_source`` contains no noise.  The registered comparison arm instead
    consumes the exact same forward-noised source state as the native no-op
    target and therefore truthfully records ``contains_target_noise=true``.
    Neither arm consumes target hidden states, text, or a synthetic target.
    The input latent is detached; memory tokens retain the encoder autograd
    graph required by Stage-B.
    """

    tokens: torch.Tensor
    source_video_sha256: str
    memory_input_latent_sha256: str
    latent_shape: tuple[int, ...]
    patch_grid: tuple[int, int, int]
    pooled_grid: tuple[int, int, int]
    input_kind: str
    construction_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.tokens, torch.Tensor)
            or self.tokens.layout != torch.strided
            or self.tokens.device.type == "meta"
            or self.tokens.ndim != 3
            or int(self.tokens.shape[0]) != 1
            or int(self.tokens.shape[1]) <= 0
            or int(self.tokens.shape[1]) > DEFAULT_MEMORY_TOKEN_CAP
            or int(self.tokens.shape[2]) <= 0
            or not self.tokens.is_contiguous()
            or not bool(torch.isfinite(self.tokens.detach()).all().item())
        ):
            raise CleanSourceVisualContextError(
                "visual memory must be contiguous finite [1,1..cap,D]"
            )
        _lower_sha256(self.source_video_sha256, label="source_video_sha256")
        _lower_sha256(
            self.memory_input_latent_sha256,
            label="memory_input_latent_sha256",
        )
        _lower_sha256(self.construction_digest, label="construction_digest")
        if self.input_kind not in MEMORY_INPUT_KINDS:
            raise CleanSourceVisualContextError("visual memory input kind differs")
        if (
            len(self.latent_shape) != 5
            or self.latent_shape[0] != 1
            or self.latent_shape[1] != LATENT_CHANNELS
            or len(self.patch_grid) != 3
            or len(self.pooled_grid) != 3
            or math.prod(self.pooled_grid) != int(self.tokens.shape[1])
        ):
            raise CleanSourceVisualContextError("visual memory geometry differs")

    @property
    def hidden_size(self) -> int:
        return int(self.tokens.shape[2])

    @property
    def token_count(self) -> int:
        return int(self.tokens.shape[1])

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "source_video_sha256": self.source_video_sha256,
            "memory_input_latent_sha256": self.memory_input_latent_sha256,
            "latent_shape": list(self.latent_shape),
            "patch_grid": list(self.patch_grid),
            "pooled_grid": list(self.pooled_grid),
            "memory_shape": list(self.tokens.shape),
            "construction_digest": self.construction_digest,
            "input_kind": self.input_kind,
            "clean_source_input": self.input_kind == "clean_source",
            "contains_target_noise": (
                self.input_kind == "same_noise_forward_noised_source"
            ),
            "contains_target_hidden": False,
            "contains_text_or_instruction": False,
            "source_derived_input_only_keys_values": True,
            "clean_source_only_keys_values": self.input_kind == "clean_source",
        }
        return {**value, "digest": object_sha256(value)}


def _bounded_spatial_grid(
    *, phases: int, height: int, width: int, token_cap: int
) -> tuple[int, int, int]:
    """Preserve every latent phase and bound only the spatial token grid."""

    phases = _positive_int(phases, label="patch phases")
    height = _positive_int(height, label="patch height")
    width = _positive_int(width, label="patch width")
    token_cap = _positive_int(token_cap, label="memory token cap")
    if phases > token_cap:
        raise CleanSourceVisualContextError(
            "temporal patch count exceeds token cap; temporal pooling is forbidden"
        )
    spatial_budget = token_cap // phases
    if height * width <= spatial_budget:
        return phases, height, width
    aspect = float(height) / float(width)
    pooled_h = max(1, min(height, int(math.floor(math.sqrt(spatial_budget * aspect)))))
    pooled_w = max(1, min(width, spatial_budget // pooled_h))
    while pooled_h * pooled_w > spatial_budget:
        if pooled_w >= pooled_h and pooled_w > 1:
            pooled_w -= 1
        elif pooled_h > 1:
            pooled_h -= 1
        else:  # pragma: no cover - arithmetic guard
            raise CleanSourceVisualContextError("cannot satisfy memory token cap")
    return phases, pooled_h, pooled_w


def fixed_3d_fourier_position_encoding(
    grid: Sequence[int],
    hidden_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return deterministic absolute (phase, y, x) Fourier coordinates.

    Cross-attention is invariant to a bare K/V token permutation.  Binding
    every projected source patch to a fixed 3-D coordinate makes phase order
    observable without adding a learned positional table or pooling time.
    """

    shape = tuple(grid)
    if (
        len(shape) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
    ):
        raise CleanSourceVisualContextError("3-D position grid differs")
    width = _positive_int(hidden_size, label="3-D position hidden_size")
    axes = []
    for size in shape:
        if size == 1:
            coordinate = torch.zeros((1,), dtype=torch.float32, device=device)
        else:
            coordinate = torch.linspace(
                -1.0, 1.0, size, dtype=torch.float32, device=device
            )
        axes.append(coordinate)
    phase, vertical, horizontal = torch.meshgrid(*axes, indexing="ij")
    coordinates = (phase, vertical, horizontal)
    pair_count = math.ceil(width / 2)
    features: list[torch.Tensor] = []
    axis_frequencies = [0, 0, 0]
    for pair_index in range(pair_count):
        axis = pair_index % 3
        axis_frequencies[axis] += 1
        angle = math.pi * float(axis_frequencies[axis]) * coordinates[axis]
        features.extend((torch.sin(angle), torch.cos(angle)))
    encoded = torch.stack(features[:width], dim=-1)
    return encoded.reshape(1, math.prod(shape), width).contiguous()


class CleanSourceVisualEncoder(nn.Module):
    """Patchify/LN/project a detached clean VAE latent into source memory."""

    def __init__(
        self,
        *,
        hidden_size: int,
        encoder_width: int = DEFAULT_ENCODER_WIDTH,
        patch_size: Sequence[int] = DEFAULT_PATCH_SIZE,
        memory_token_cap: int = DEFAULT_MEMORY_TOKEN_CAP,
    ) -> None:
        super().__init__()
        self.hidden_size = _positive_int(hidden_size, label="hidden_size")
        self.encoder_width = _positive_int(encoder_width, label="encoder_width")
        patch = tuple(patch_size)
        if (
            len(patch) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in patch)
            or patch[0] != 1
        ):
            raise CleanSourceVisualContextError(
                "visual patch size must be positive 3-D with temporal stride one"
            )
        self.patch_size = patch
        self.memory_token_cap = _positive_int(
            memory_token_cap, label="memory_token_cap"
        )
        if self.memory_token_cap > DEFAULT_MEMORY_TOKEN_CAP:
            raise CleanSourceVisualContextError("memory token cap exceeds preregistration")
        self.patchifier = nn.Conv3d(
            LATENT_CHANNELS,
            self.encoder_width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
            dtype=torch.float32,
        )
        self.patch_norm = nn.LayerNorm(self.encoder_width, dtype=torch.float32)
        self.projection = nn.Linear(
            self.encoder_width, self.hidden_size, bias=True, dtype=torch.float32
        )
        self.projected_norm = nn.LayerNorm(
            self.hidden_size, elementwise_affine=False, dtype=torch.float32
        )
        # ID 0 is reserved/padding.  ID 1 is exactly the clean-source role.
        self.source_role = nn.Embedding(2, self.hidden_size, dtype=torch.float32)
        nn.init.xavier_uniform_(self.patchifier.weight.flatten(1))
        nn.init.zeros_(self.patchifier.bias)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        nn.init.zeros_(self.source_role.weight)

    def architecture_receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "input": "detached_registered_source_visual_latent_[1,16,F,H,W]",
            "patchifier": "trainable_conv3d",
            "patch_size": list(self.patch_size),
            "temporal_patch_stride": 1,
            "temporal_pooling": False,
            "spatial_pooling_only_when_needed": True,
            "memory_token_cap": self.memory_token_cap,
            "encoder_width": self.encoder_width,
            "hidden_size": self.hidden_size,
            "pipeline": (
                "patchify->spatial_budget->layer_norm->projection->layer_norm"
                "+fixed_3d_fourier_phase_y_x+source_role_id"
            ),
            "position_representation": "fixed_absolute_3d_fourier_phase_y_x_v1",
            "position_parameters_trainable": False,
            "position_added_after_projection": True,
            "explicit_source_role_id": 1,
            "target_noise_argument_present": False,
            "text_argument_present": False,
        }
        return {**value, "digest": object_sha256(value)}

    def _encode(self, memory_input_latent: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int, int]]:
        if (
            not isinstance(memory_input_latent, torch.Tensor)
            or memory_input_latent.layout != torch.strided
            or memory_input_latent.device.type == "meta"
            or memory_input_latent.dtype != torch.float32
            or memory_input_latent.ndim != 5
            or tuple(memory_input_latent.shape[:2]) != (1, LATENT_CHANNELS)
            or memory_input_latent.requires_grad
            or memory_input_latent.grad_fn is not None
            or not memory_input_latent.is_contiguous()
            or not bool(torch.isfinite(memory_input_latent).all().item())
        ):
            raise CleanSourceVisualContextError(
                "memory input must be detached contiguous finite FP32 [1,16,F,H,W]"
            )
        if any(
            int(memory_input_latent.shape[index]) < self.patch_size[index - 2]
            for index in (2, 3, 4)
        ):
            raise CleanSourceVisualContextError("memory input is smaller than one visual patch")
        with torch.autocast(device_type=memory_input_latent.device.type, enabled=False):
            patches = self.patchifier(memory_input_latent.float())
            patch_grid = tuple(int(value) for value in patches.shape[2:])
            pooled_grid = _bounded_spatial_grid(
                phases=patch_grid[0],
                height=patch_grid[1],
                width=patch_grid[2],
                token_cap=self.memory_token_cap,
            )
            if pooled_grid != patch_grid:
                patches = F.adaptive_avg_pool3d(patches, pooled_grid)
            tokens = patches.flatten(2).transpose(1, 2).contiguous()
            tokens = self.patch_norm(tokens)
            tokens = self.projected_norm(self.projection(tokens))
            tokens = tokens + fixed_3d_fourier_position_encoding(
                pooled_grid,
                self.hidden_size,
                device=tokens.device,
            )
            role_ids = torch.ones(
                (1, int(tokens.shape[1])),
                dtype=torch.int64,
                device=tokens.device,
            )
            tokens = tokens + self.source_role(role_ids)
        return tokens.float().contiguous(), patch_grid, pooled_grid

    def forward(self, memory_input_latent: torch.Tensor) -> torch.Tensor:
        tokens, _, _ = self._encode(memory_input_latent)
        return tokens

    def build_memory(
        self,
        memory_input_latent: torch.Tensor,
        *,
        source_video_sha256: str,
        memory_input_latent_sha256: str,
        input_kind: str = "clean_source",
    ) -> CleanSourceVisualMemory:
        _lower_sha256(source_video_sha256, label="source_video_sha256")
        _lower_sha256(
            memory_input_latent_sha256, label="memory_input_latent_sha256"
        )
        if input_kind not in MEMORY_INPUT_KINDS:
            raise CleanSourceVisualContextError("visual memory input kind differs")
        tokens, patch_grid, pooled_grid = self._encode(memory_input_latent)
        construction = {
            "architecture_digest": self.architecture_receipt()["digest"],
            "source_video_sha256": source_video_sha256,
            "memory_input_latent_sha256": memory_input_latent_sha256,
            "input_kind": input_kind,
            "latent_shape": list(memory_input_latent.shape),
            "patch_grid": list(patch_grid),
            "pooled_grid": list(pooled_grid),
        }
        return CleanSourceVisualMemory(
            tokens=tokens,
            source_video_sha256=source_video_sha256,
            memory_input_latent_sha256=memory_input_latent_sha256,
            latent_shape=tuple(int(value) for value in memory_input_latent.shape),
            patch_grid=patch_grid,
            pooled_grid=pooled_grid,
            input_kind=input_kind,
            construction_digest=object_sha256(construction),
        )


@dataclass(frozen=True)
class VisualContextRoute:
    """Authenticated global token layout before append-padding/SP slicing."""

    total_tokens: int
    condition_tokens: int
    sequence_parallel_rank: int
    sequence_parallel_size: int
    memory: Optional[CleanSourceVisualMemory]
    enabled: bool = True

    def __post_init__(self) -> None:
        total = _positive_int(self.total_tokens, label="total_tokens")
        if (
            isinstance(self.condition_tokens, bool)
            or not isinstance(self.condition_tokens, int)
            or not 0 <= self.condition_tokens < total
        ):
            raise CleanSourceVisualContextError(
                "condition_tokens must identify one strict target suffix"
            )
        size = _positive_int(
            self.sequence_parallel_size, label="sequence_parallel_size"
        )
        if size not in ALLOWED_SP_SIZES:
            raise CleanSourceVisualContextError("only SP1 tests and native SP4 are supported")
        if (
            isinstance(self.sequence_parallel_rank, bool)
            or not isinstance(self.sequence_parallel_rank, int)
            or not 0 <= self.sequence_parallel_rank < size
        ):
            raise CleanSourceVisualContextError("sequence-parallel rank is invalid")
        if not isinstance(self.enabled, bool):
            raise CleanSourceVisualContextError("route enabled flag must be boolean")
        if self.enabled and not isinstance(self.memory, CleanSourceVisualMemory):
            raise CleanSourceVisualContextError("enabled route requires clean-source memory")
        if not self.enabled and self.memory is not None:
            raise CleanSourceVisualContextError("disabled route must not carry source memory")

    @property
    def target_tokens(self) -> int:
        return self.total_tokens - self.condition_tokens

    @property
    def local_length(self) -> int:
        return math.ceil(self.total_tokens / self.sequence_parallel_size)

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
                        padded - self.total_tokens,
                        dtype=torch.bool,
                        device=device,
                    ),
                )
            )
        start = self.sequence_parallel_rank * self.local_length
        return selector[start : start + self.local_length].contiguous()

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "total_tokens": self.total_tokens,
            "condition_tokens": self.condition_tokens,
            "target_tokens": self.target_tokens,
            "sequence_parallel_rank": self.sequence_parallel_rank,
            "sequence_parallel_size": self.sequence_parallel_size,
            "enabled": self.enabled,
            "memory_digest": (
                self.memory.receipt()["digest"] if self.memory is not None else None
            ),
            "query_rows": "local_target_suffix_only",
            "key_value_rows": "independent_registered_source_visual_memory_only",
        }
        return {**value, "digest": object_sha256(value)}


_ACTIVE_ROUTE: ContextVar[Optional[VisualContextRoute]] = ContextVar(
    "bernini_clean_source_visual_context_route_v1", default=None
)


def active_route() -> Optional[VisualContextRoute]:
    return _ACTIVE_ROUTE.get()


@contextmanager
def activate_route(route: VisualContextRoute) -> Iterator[None]:
    if not isinstance(route, VisualContextRoute):
        raise CleanSourceVisualContextError("route must be VisualContextRoute")
    if active_route() is not None:
        raise CleanSourceVisualContextError("nested visual-context routes are forbidden")
    token: Token[Optional[VisualContextRoute]] = _ACTIVE_ROUTE.set(route)
    try:
        yield
    finally:
        _ACTIVE_ROUTE.reset(token)


@contextmanager
def _replay_checkpoint_route(route: VisualContextRoute) -> Iterator[None]:
    current = active_route()
    if current is route:
        yield
        return
    if current is not None:
        raise CleanSourceVisualContextError(
            "checkpoint recomputation entered a different visual route"
        )
    with activate_route(route):
        yield


def checkpoint_route_context_fn() -> tuple[Any, Any]:
    """Capture the exact route for non-reentrant checkpoint recomputation."""

    route = active_route()
    if route is None:
        raise CleanSourceVisualContextError(
            "checkpoint was created without an active visual-context route"
        )
    return _replay_checkpoint_route(route), _replay_checkpoint_route(route)


class TargetQuerySourceOnlyAttention(nn.Module):
    """Extra target-Q/source-KV attention; no native projection is wrapped."""

    def __init__(
        self,
        *,
        hidden_size: int,
        attention_width: int = DEFAULT_ATTENTION_WIDTH,
        num_heads: int = DEFAULT_ATTENTION_HEADS,
    ) -> None:
        super().__init__()
        self.hidden_size = _positive_int(hidden_size, label="hidden_size")
        self.attention_width = _positive_int(
            attention_width, label="attention_width"
        )
        self.num_heads = _positive_int(num_heads, label="num_heads")
        if self.attention_width % self.num_heads:
            raise CleanSourceVisualContextError(
                "attention width must be divisible by visual-context heads"
            )
        self.head_dim = self.attention_width // self.num_heads
        self.query_norm = nn.LayerNorm(
            self.hidden_size, elementwise_affine=False, dtype=torch.float32
        )
        self.memory_norm = nn.LayerNorm(
            self.hidden_size, elementwise_affine=False, dtype=torch.float32
        )
        self.query = nn.Linear(
            self.hidden_size, self.attention_width, bias=False, dtype=torch.float32
        )
        self.key = nn.Linear(
            self.hidden_size, self.attention_width, bias=False, dtype=torch.float32
        )
        self.value = nn.Linear(
            self.hidden_size, self.attention_width, bias=False, dtype=torch.float32
        )
        self.output = nn.Linear(
            self.attention_width, self.hidden_size, bias=False, dtype=torch.float32
        )
        # Do not zero both an output and a multiplicative gate: that is a dead
        # bilinear parameterization.  Zero output gives exact-base step 0 and
        # a non-zero first gradient.  Gain starts at one and remains trainable.
        self.residual_gain = nn.Parameter(torch.ones((), dtype=torch.float32))
        for projection in (self.query, self.key, self.value):
            nn.init.xavier_uniform_(projection.weight)
        nn.init.zeros_(self.output.weight)

    def adapter_delta(self, query_states: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(query_states, torch.Tensor)
            or query_states.layout != torch.strided
            or query_states.device.type == "meta"
            or query_states.ndim != 3
            or int(query_states.shape[0]) != 1
            or int(query_states.shape[2]) != self.hidden_size
        ):
            raise CleanSourceVisualContextError(
                "block input must be dense [1,local_N,hidden_size]"
            )
        route = active_route()
        # Every SP rank must expose the same upstream autograd edge even when
        # its contiguous token shard owns no target row.  A plain
        # ``zeros_like`` on those ranks disconnects the frozen block input,
        # while target-owning ranks reach it through Q attention.  Backward
        # then enters Bernini's Ulysses collectives on only part of the SP
        # group.  The trainable scalar makes the first hooked block output
        # require gradients on every rank; multiplication by ``query_states``
        # keeps the zero residual connected to each later frozen block input.
        # The scalar is converted only in dtype (never moved across devices),
        # and multiplying by an exact representable zero preserves step-0
        # numeric parity and writes no condition or padding row.
        graph_zero = self.residual_gain.to(dtype=query_states.dtype) * (
            query_states.new_zeros(())
        )
        result = query_states * graph_zero
        if route is None:
            raise CleanSourceVisualContextError(
                "visual-context block executed without an authenticated route"
            )
        if not route.enabled:
            return result
        memory = route.memory
        if memory is None or memory.hidden_size != self.hidden_size:
            raise CleanSourceVisualContextError("visual memory hidden size differs")
        if memory.tokens.device != query_states.device:
            raise CleanSourceVisualContextError(
                "visual memory and target query must share one device"
            )
        selector = route.local_target_selector(device=query_states.device)
        if int(query_states.shape[1]) != int(selector.numel()):
            raise CleanSourceVisualContextError(
                "local query length differs from append-pad/SP route"
            )
        if not bool(selector.any().item()):
            return result
        target = query_states[:, selector, :]
        with torch.autocast(device_type=query_states.device.type, enabled=False):
            target_fp32 = self.query_norm(target.float())
            memory_fp32 = self.memory_norm(memory.tokens.float())
            batch = int(target_fp32.shape[0])
            q = self.query(target_fp32).view(
                batch, -1, self.num_heads, self.head_dim
            ).transpose(1, 2)
            k = self.key(memory_fp32).view(
                batch, -1, self.num_heads, self.head_dim
            ).transpose(1, 2)
            v = self.value(memory_fp32).view(
                batch, -1, self.num_heads, self.head_dim
            ).transpose(1, 2)
            logits = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim**-0.5)
            weights = torch.softmax(logits, dim=-1)
            attended = torch.matmul(weights, v).transpose(1, 2).reshape(
                batch, -1, self.attention_width
            )
            delta = self.output(attended) * self.residual_gain
        result[:, selector, :] = delta.to(query_states.dtype)
        return result.contiguous()

    def forward(
        self, query_states: torch.Tensor, frozen_block_output: torch.Tensor
    ) -> torch.Tensor:
        if (
            not isinstance(frozen_block_output, torch.Tensor)
            or tuple(frozen_block_output.shape) != tuple(query_states.shape)
            or frozen_block_output.dtype != query_states.dtype
            or frozen_block_output.device != query_states.device
        ):
            raise CleanSourceVisualContextError(
                "frozen block output and current query states differ"
            )
        route = active_route()
        if route is None:
            raise CleanSourceVisualContextError(
                "visual-context block executed without an authenticated route"
            )
        if not route.enabled:
            return frozen_block_output
        # Keep this addition in graph at step 0: zero output weights then get
        # the first optimizer gradient while the numeric result equals base.
        return frozen_block_output + self.adapter_delta(query_states).to(
            frozen_block_output.dtype
        )


class _VisualContextComponents(nn.Module):
    def __init__(
        self,
        *,
        encoder: CleanSourceVisualEncoder,
        adapters: Mapping[str, TargetQuerySourceOnlyAttention],
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.adapters = nn.ModuleDict(dict(adapters))


@dataclass
class CleanSourceVisualContextHandle:
    transformer: nn.Module
    components: _VisualContextComponents
    block_indices: tuple[int, ...]
    hook_handles: tuple[Any, ...]
    native_block_ids: tuple[int, ...]
    native_self_attention_ids: tuple[tuple[int, ...], ...]
    native_text_attention_ids: tuple[tuple[int, ...], ...]
    runtime_source_commit: str
    model_revision: str
    checkpoint_manifest_sha256: str
    transformer_config_digest: str
    restored: bool = False

    @property
    def encoder(self) -> CleanSourceVisualEncoder:
        return self.components.encoder

    def trainable_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        if self.restored:
            raise CleanSourceVisualContextError("visual-context adapter is restored")
        values = tuple(self.components.named_parameters())
        if not values or len({id(parameter) for _, parameter in values}) != len(values):
            raise CleanSourceVisualContextError("visual-context trainable scope aliases")
        if any(not parameter.requires_grad for _, parameter in values):
            raise CleanSourceVisualContextError("visual-context parameter is frozen")
        return values

    def base_parameters_frozen(self) -> bool:
        trainable_ids = {id(parameter) for _, parameter in self.trainable_named_parameters()}
        return all(
            id(parameter) in trainable_ids or not parameter.requires_grad
            for parameter in self.transformer.parameters()
        )

    def native_structure_untouched(self) -> bool:
        blocks = tuple(getattr(self.transformer, "blocks", ()))
        if tuple(id(block) for block in blocks) != self.native_block_ids:
            return False
        try:
            observed_self = _native_attention_identity(
                blocks, attribute="attn1", label="self-attention"
            )
            observed_text = _native_attention_identity(
                blocks, attribute="attn2", label="text cross-attention"
            )
        except CleanSourceVisualContextError:
            return False
        return (
            observed_self == self.native_self_attention_ids
            and observed_text == self.native_text_attention_ids
        )

    def build_memory(
        self,
        memory_input_latent: torch.Tensor,
        *,
        source_video_sha256: str,
        memory_input_latent_sha256: str,
        input_kind: str = "clean_source",
    ) -> CleanSourceVisualMemory:
        if self.restored:
            raise CleanSourceVisualContextError("cannot use a restored adapter")
        return self.encoder.build_memory(
            memory_input_latent,
            source_video_sha256=source_video_sha256,
            memory_input_latent_sha256=memory_input_latent_sha256,
            input_kind=input_kind,
        )

    @contextmanager
    def route(self, route: VisualContextRoute) -> Iterator[None]:
        if self.restored:
            raise CleanSourceVisualContextError("cannot route a restored adapter")
        with activate_route(route):
            yield

    def state_dict_for_save(self) -> Mapping[str, torch.Tensor]:
        return {
            name: parameter.detach().float().cpu().contiguous()
            for name, parameter in self.trainable_named_parameters()
        }

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": SCHEMA_VERSION,
            "gpu_validated": False,
            "scientific_quality_claim": False,
            "runtime_source_commit": self.runtime_source_commit,
            "model_revision": self.model_revision,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "transformer_config_digest": self.transformer_config_digest,
            "block_indices": list(self.block_indices),
            "block_scope_status": "structural_candidate_not_causally_admitted",
            "optimizer_authorized_by_this_receipt": False,
            "insertion": "registered_forward_hook_on_frozen_block_output",
            "query_source": "current_frozen_block_input_target_rows",
            "key_value_source": "independent_source_visual_memory_only",
            "memory_input_kinds_supported": list(MEMORY_INPUT_KINDS),
            "native_self_attention_kv_replaced": False,
            "native_self_attention_kv_replayed": False,
            "native_text_cross_attention_changed": False,
            "native_blocks_replaced": False,
            "native_structure_untouched": self.native_structure_untouched(),
            "condition_rows_directly_written": False,
            "sp_empty_target_rank_graph_anchor": (
                "query_times_trainable_exact_zero_on_every_rank"
            ),
            "sp_collective_backward_graph_isomorphic": True,
            "target_noise_read_by_memory_encoder": "declared_per_memory_receipt",
            "source_reads_target_noise": "declared_per_memory_receipt",
            "zero_initialized_output_projection": True,
            "multiplicative_gain_initial_value": 1.0,
            "double_zero_dead_parameterization": False,
            "checkpoint_context_fn_required": True,
            "base_parameters_frozen": self.base_parameters_frozen(),
            "memory_encoder": self.encoder.architecture_receipt(),
            "trainable": [
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                }
                for name, parameter in self.trainable_named_parameters()
            ],
            "feature_reward": False,
            "vlm_reward": False,
            "synthetic_target_required": False,
        }
        return {**value, "digest": object_sha256(value)}

    def restore(self) -> None:
        if self.restored or active_route() is not None:
            raise CleanSourceVisualContextError("adapter cannot be restored now")
        for hook in self.hook_handles:
            hook.remove()
        if getattr(self.transformer, "clean_source_visual_context_v1", None) is not self.components:
            raise CleanSourceVisualContextError("registered component owner changed")
        delattr(self.transformer, "clean_source_visual_context_v1")
        self.restored = True


def _strict_transformer_config_receipt(transformer: nn.Module) -> Mapping[str, Any]:
    observed_class = (
        transformer.__class__.__module__,
        transformer.__class__.__name__,
    )
    if observed_class != (
        PINNED_TRANSFORMER_CLASS_MODULE,
        PINNED_TRANSFORMER_CLASS_NAME,
    ):
        raise CleanSourceVisualContextError(
            "transformer class is not pinned Bernini WanTransformer3DModel"
        )
    config = getattr(transformer, "config", None)
    if config is None:
        raise CleanSourceVisualContextError("pinned Bernini transformer config is absent")
    observed: dict[str, Any] = {}
    for name, expected in PINNED_TRANSFORMER_CONFIG.items():
        value = _config_value(config, name)
        if name == "patch_size" and isinstance(value, (list, tuple)):
            value = tuple(value)
        if value != expected:
            raise CleanSourceVisualContextError(
                f"pinned Bernini transformer config differs at {name}"
            )
        observed[name] = list(value) if isinstance(value, tuple) else value
    hidden = int(observed["num_attention_heads"]) * int(observed["attention_head_dim"])
    if hidden != HIDDEN_SIZE_1P3B:
        raise CleanSourceVisualContextError("pinned Bernini hidden size differs")
    value = {
        "class_module": observed_class[0],
        "class_name": observed_class[1],
        "config": observed,
        "hidden_size": hidden,
    }
    return {**value, "digest": object_sha256(value)}


def _native_attention_identity(
    blocks: Sequence[nn.Module],
    *,
    attribute: str,
    label: str,
) -> tuple[tuple[int, ...], ...]:
    result = []
    for index, block in enumerate(blocks):
        attention = getattr(block, attribute, None)
        output = getattr(attention, "to_out", None)
        processor = getattr(attention, "processor", None)
        norm_q = getattr(attention, "norm_q", None)
        norm_k = getattr(attention, "norm_k", None)
        if (
            attention is None
            or not isinstance(getattr(attention, "to_q", None), nn.Linear)
            or not isinstance(getattr(attention, "to_k", None), nn.Linear)
            or not isinstance(getattr(attention, "to_v", None), nn.Linear)
            or not isinstance(output, nn.ModuleList)
            or len(output) != 2
            or not isinstance(output[0], nn.Linear)
            or not isinstance(output[1], nn.Module)
            or processor is None
            or not isinstance(norm_q, nn.Module)
            or not isinstance(norm_k, nn.Module)
            or getattr(attention, "added_kv_proj_dim", None) is not None
            or getattr(attention, "add_k_proj", None) is not None
            or getattr(attention, "add_v_proj", None) is not None
        ):
            raise CleanSourceVisualContextError(
                f"block {index} native {label} structure differs"
            )
        for projection in (attention.to_q, attention.to_k, attention.to_v, output[0]):
            if projection.in_features != HIDDEN_SIZE_1P3B or projection.out_features != HIDDEN_SIZE_1P3B:
                raise CleanSourceVisualContextError(
                    f"block {index} native {label} width differs"
                )
        result.append(
            (
                id(attention),
                id(attention.to_q),
                id(attention.to_k),
                id(attention.to_v),
                id(output[0]),
                id(output[1]),
                id(norm_q),
                id(norm_k),
                id(processor),
            )
        )
    return tuple(result)


def _hook_output_tensor(output: Any) -> tuple[torch.Tensor, Any]:
    if isinstance(output, torch.Tensor):
        return output, lambda value: value
    if (
        isinstance(output, tuple)
        and output
        and isinstance(output[0], torch.Tensor)
    ):
        return output[0], lambda value: (value, *output[1:])
    raise CleanSourceVisualContextError(
        "pinned Bernini block output is neither Tensor nor tensor-first tuple"
    )


def install_clean_source_visual_context_adapter_v1(
    transformer: nn.Module,
    *,
    runtime_source_commit: str,
    model_revision: str,
    checkpoint_manifest_sha256: str,
    block_indices: Sequence[int] = DEFAULT_BLOCK_INDICES,
    encoder_width: int = DEFAULT_ENCODER_WIDTH,
    visual_patch_size: Sequence[int] = DEFAULT_PATCH_SIZE,
    memory_token_cap: int = DEFAULT_MEMORY_TOKEN_CAP,
    attention_width: int = DEFAULT_ATTENTION_WIDTH,
    attention_heads: int = DEFAULT_ATTENTION_HEADS,
) -> CleanSourceVisualContextHandle:
    """Register an independent visual-context residual on a frozen base."""

    if not isinstance(transformer, nn.Module):
        raise CleanSourceVisualContextError("transformer must be nn.Module")
    if runtime_source_commit != PINNED_BERNINI_SOURCE_COMMIT:
        raise CleanSourceVisualContextError("Bernini runtime source commit is not pinned")
    if model_revision != PINNED_BERNINI_MODEL_REVISION:
        raise CleanSourceVisualContextError("Bernini model revision is not pinned")
    _lower_sha256(
        checkpoint_manifest_sha256, label="checkpoint_manifest_sha256"
    )
    if hasattr(transformer, "clean_source_visual_context_v1"):
        raise CleanSourceVisualContextError("visual-context adapter is already installed")
    if any(parameter.requires_grad for parameter in transformer.parameters()):
        raise CleanSourceVisualContextError("freeze the complete Bernini transformer first")
    config = _strict_transformer_config_receipt(transformer)
    blocks = tuple(getattr(transformer, "blocks", ()))
    patch = getattr(transformer, "patch_embedding", None)
    indices = tuple(block_indices)
    if (
        len(blocks) != TOTAL_BLOCKS_1P3B
        or not isinstance(patch, nn.Conv3d)
        or patch.in_channels != LATENT_CHANNELS
        or patch.out_channels != HIDDEN_SIZE_1P3B
        or tuple(patch.kernel_size) != (1, 2, 2)
        or tuple(patch.stride) != (1, 2, 2)
        or indices not in ALLOWED_BLOCK_SCOPES
    ):
        raise CleanSourceVisualContextError("audited Bernini structure or block scope differs")
    native_self_attention_ids = _native_attention_identity(
        blocks, attribute="attn1", label="self-attention"
    )
    native_text_attention_ids = _native_attention_identity(
        blocks, attribute="attn2", label="text cross-attention"
    )
    device = patch.weight.device
    encoder = CleanSourceVisualEncoder(
        hidden_size=HIDDEN_SIZE_1P3B,
        encoder_width=encoder_width,
        patch_size=visual_patch_size,
        memory_token_cap=memory_token_cap,
    ).to(device=device)
    adapters = {
        str(index): TargetQuerySourceOnlyAttention(
            hidden_size=HIDDEN_SIZE_1P3B,
            attention_width=attention_width,
            num_heads=attention_heads,
        ).to(device=device)
        for index in indices
    }
    components = _VisualContextComponents(encoder=encoder, adapters=adapters)
    transformer.add_module("clean_source_visual_context_v1", components)
    hooks = []
    try:
        for index in indices:
            adapter = components.adapters[str(index)]

            def callback(
                _module: nn.Module,
                args: tuple[Any, ...],
                output: Any,
                *,
                bound_adapter: TargetQuerySourceOnlyAttention = adapter,
            ) -> Any:
                if not args or not isinstance(args[0], torch.Tensor):
                    raise CleanSourceVisualContextError(
                        "pinned Bernini block input lacks hidden-state tensor"
                    )
                frozen_output, rebuild = _hook_output_tensor(output)
                return rebuild(bound_adapter(args[0], frozen_output))

            hooks.append(blocks[index].register_forward_hook(callback))
    except Exception:
        for hook in hooks:
            hook.remove()
        delattr(transformer, "clean_source_visual_context_v1")
        raise
    handle = CleanSourceVisualContextHandle(
        transformer=transformer,
        components=components,
        block_indices=indices,
        hook_handles=tuple(hooks),
        native_block_ids=tuple(id(block) for block in blocks),
        native_self_attention_ids=native_self_attention_ids,
        native_text_attention_ids=native_text_attention_ids,
        runtime_source_commit=runtime_source_commit,
        model_revision=model_revision,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        transformer_config_digest=str(config["digest"]),
    )
    if not handle.base_parameters_frozen() or not handle.native_structure_untouched():
        handle.restore()
        raise CleanSourceVisualContextError("visual-context scope closure failed")
    return handle


def no_op_flow_matching_loss(
    *, prediction: torch.Tensor, target_velocity: torch.Tensor
) -> torch.Tensor:
    """Single same-real-source no-op FM objective; no synthetic target/reward."""

    if (
        not isinstance(prediction, torch.Tensor)
        or not isinstance(target_velocity, torch.Tensor)
        or prediction.shape != target_velocity.shape
        or prediction.device != target_velocity.device
        or not prediction.is_floating_point()
        or not target_velocity.is_floating_point()
        or prediction.ndim < 2
        or not bool(torch.isfinite(prediction.detach()).all().item())
        or not bool(torch.isfinite(target_velocity.detach()).all().item())
    ):
        raise CleanSourceVisualContextError(
            "prediction/target velocity must be matching finite float tensors"
        )
    loss = F.mse_loss(prediction.float(), target_velocity.float(), reduction="mean")
    if not bool(torch.isfinite(loss.detach()).item()):
        raise CleanSourceVisualContextError("no-op flow-matching loss is non-finite")
    return loss


__all__ = [
    "ALLOWED_BLOCK_SCOPES",
    "CleanSourceVisualContextError",
    "CleanSourceVisualContextHandle",
    "CleanSourceVisualEncoder",
    "CleanSourceVisualMemory",
    "DEFAULT_BLOCK_INDICES",
    "HIDDEN_SIZE_1P3B",
    "MEMORY_INPUT_KINDS",
    "PINNED_BERNINI_MODEL_REVISION",
    "PINNED_BERNINI_SOURCE_COMMIT",
    "PINNED_TRANSFORMER_CONFIG",
    "SCHEMA_VERSION",
    "TargetQuerySourceOnlyAttention",
    "VisualContextRoute",
    "activate_route",
    "active_route",
    "checkpoint_route_context_fn",
    "install_clean_source_visual_context_adapter_v1",
    "fixed_3d_fourier_position_encoding",
    "no_op_flow_matching_loss",
]
