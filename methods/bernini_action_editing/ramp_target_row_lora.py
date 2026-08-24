#!/usr/bin/env python3
"""Role-explicit Bernini compositor for the RAMP C0 temporal-program canary.

This module implements the smallest trainable path that can test whether a
frozen Bernini prior can combine a complete source identity carrier with a
compact temporal program:

* source and noisy-target patches retain Bernini's frozen Conv3d embedding;
* a registered 21x21 temporal transport is serialized into 21 synthetic VAE
  patches and decoded by a trainable program projector (raw donor pixels never
  enter Bernini self-attention);
* zero-initialized learned role embeddings distinguish identity, motion and
  target tokens independently of source-id RoPE;
* only target query rows receive a low-rank update in ``attn1.to_q``.

The role selector is sliced exactly as Bernini's pinned Ulysses path does:
append-only padding to a multiple of the SP world size, followed by contiguous
rank chunks.  The activation context must enclose both forward and backward so
gradient-checkpoint recomputation sees the same immutable role layout.

This is an engineering/training primitive.  It does not accept masks, flows,
poses, tracks, target videos, action proposals, or donor RGB/VAE tensors, and
it does not establish semantic action editing.
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


SCHEMA_VERSION = "bernini-ramp-role-target-query-lora-v1"
LATENT_PHASES = 21
PATCH_CHANNELS = 16
PATCH_SHAPE = (1, 2, 2)
PATCH_VALUES = PATCH_CHANNELS * math.prod(PATCH_SHAPE)

ROLE_PADDING = 0
ROLE_IDENTITY = 1
ROLE_MOTION = 2
ROLE_TARGET = 3
ROLE_NAMES = {
    ROLE_PADDING: "padding",
    ROLE_IDENTITY: "identity",
    ROLE_MOTION: "motion",
    ROLE_TARGET: "target",
}


class RAMPRouteError(RuntimeError):
    """Raised before an ambiguous token route or adapter update."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise RAMPRouteError(f"receipt is not canonical finite ASCII JSON: {error}") from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _exact_positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RAMPRouteError(f"{label} must be an exact positive integer")
    return value


def _role_id(value: Any, *, allow_padding: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RAMPRouteError("role ids must be exact integers")
    allowed = {ROLE_IDENTITY, ROLE_MOTION, ROLE_TARGET}
    if allow_padding:
        allowed.add(ROLE_PADDING)
    if value not in allowed:
        raise RAMPRouteError(f"unsupported role id {value}")
    return value


def _detached_fp32(name: str, value: Any, *, shape: Optional[tuple[int, ...]] = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise RAMPRouteError(f"{name} must be a torch.Tensor")
    if value.layout != torch.strided or value.device.type == "meta":
        raise RAMPRouteError(f"{name} must be dense and materialized")
    if value.dtype != torch.float32 or value.requires_grad or value.grad_fn is not None:
        raise RAMPRouteError(f"{name} must be detached FP32")
    if shape is not None and tuple(int(item) for item in value.shape) != shape:
        raise RAMPRouteError(f"{name} must have shape {shape}")
    if not value.is_contiguous() or not bool(torch.isfinite(value).all().item()):
        raise RAMPRouteError(f"{name} must be contiguous and finite")
    return value


def latent_phase_transport(rgb_output_to_input: Any) -> torch.Tensor:
    """Convert 81-frame coordinates to a compact 21x21 phase transport.

    Bernini's causal VAE must still encode every transformed RGB clip
    independently; this matrix is a *program description*, not a claim that
    VAE latents can be permuted exactly.  Latent phase ``j`` is anchored to RGB
    frame ``4*j`` and linearly expressed in the neighboring phase anchors.
    """

    coordinate = rgb_output_to_input
    if not isinstance(coordinate, torch.Tensor):
        raise RAMPRouteError("RGB output-to-input coordinates must be a tensor")
    if coordinate.layout != torch.strided or coordinate.device.type == "meta":
        raise RAMPRouteError("RGB coordinates must be dense and materialized")
    if coordinate.dtype != torch.float64 or tuple(coordinate.shape) != (81,):
        raise RAMPRouteError("RGB coordinates must be detached FP64 [81]")
    if coordinate.requires_grad or coordinate.grad_fn is not None or not coordinate.is_contiguous():
        raise RAMPRouteError("RGB coordinates must be detached and contiguous")
    if not bool(torch.isfinite(coordinate).all().item()):
        raise RAMPRouteError("RGB coordinates contain NaN or infinity")
    if bool(((coordinate < 0.0) | (coordinate > 80.0)).any().item()):
        raise RAMPRouteError("RGB coordinates must remain inside [0,80]")

    phase_coordinate = coordinate[::4] / 4.0
    if tuple(phase_coordinate.shape) != (LATENT_PHASES,):
        raise RAMPRouteError("81-frame program did not produce 21 phase anchors")
    lower = phase_coordinate.floor().to(torch.int64)
    upper = torch.clamp(lower + 1, max=LATENT_PHASES - 1)
    fraction = phase_coordinate - lower.to(torch.float64)
    rows = torch.arange(LATENT_PHASES, dtype=torch.int64, device=coordinate.device)
    matrix = torch.zeros(
        LATENT_PHASES,
        LATENT_PHASES,
        dtype=torch.float64,
        device=coordinate.device,
    )
    matrix[rows, lower] += 1.0 - fraction
    matrix[rows, upper] += fraction
    if not torch.equal(
        matrix.sum(dim=-1),
        torch.ones(LATENT_PHASES, dtype=torch.float64, device=coordinate.device),
    ):
        raise RAMPRouteError("phase transport rows do not sum exactly to one")
    return matrix.to(torch.float32).detach().contiguous()


def oracle_program_patches(transport: Any) -> torch.Tensor:
    """Serialize one 21x21 row-stochastic transport as 21 VAE-shaped patches."""

    matrix = _detached_fp32(
        "transport", transport, shape=(LATENT_PHASES, LATENT_PHASES)
    )
    if bool((matrix < 0.0).any().item()) or not torch.allclose(
        matrix.sum(dim=-1),
        torch.ones(LATENT_PHASES, dtype=torch.float32, device=matrix.device),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise RAMPRouteError("transport rows must be probability distributions")
    flat = torch.zeros(
        LATENT_PHASES,
        PATCH_VALUES,
        dtype=torch.float32,
        device=matrix.device,
    )
    flat[:, :LATENT_PHASES] = matrix
    return flat.reshape(LATENT_PHASES, PATCH_CHANNELS, *PATCH_SHAPE).contiguous()


def recover_oracle_transport(patches: Any) -> torch.Tensor:
    """Recover and validate the matrix carried by synthetic program patches."""

    value = _detached_fp32(
        "program patches",
        patches,
        shape=(LATENT_PHASES, PATCH_CHANNELS, *PATCH_SHAPE),
    )
    flat = value.reshape(LATENT_PHASES, PATCH_VALUES)
    if not torch.equal(
        flat[:, LATENT_PHASES:],
        torch.zeros_like(flat[:, LATENT_PHASES:]),
    ):
        raise RAMPRouteError("program patch tail must be exactly zero")
    matrix = flat[:, :LATENT_PHASES].contiguous()
    if bool((matrix < 0.0).any().item()) or not torch.allclose(
        matrix.sum(dim=-1),
        torch.ones(LATENT_PHASES, dtype=torch.float32, device=matrix.device),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise RAMPRouteError("serialized transport rows are invalid")
    return matrix


@dataclass(frozen=True)
class TokenRoleLayout:
    """Immutable global visual-token layout before Ulysses slicing."""

    roles: tuple[int, ...]
    source_tokens: int
    motion_tokens: int
    target_tokens: int

    def __post_init__(self) -> None:
        for name in ("source_tokens", "motion_tokens", "target_tokens"):
            _exact_positive_int(getattr(self, name), label=name)
        if len(self.roles) != self.source_tokens + self.motion_tokens + self.target_tokens:
            raise RAMPRouteError("role length differs from declared token spans")
        if self.motion_tokens != LATENT_PHASES:
            raise RAMPRouteError(f"C0 requires exactly {LATENT_PHASES} motion tokens")
        for role in self.roles:
            _role_id(role)
        counts = {
            role: self.roles.count(role)
            for role in (ROLE_IDENTITY, ROLE_MOTION, ROLE_TARGET)
        }
        expected = {
            ROLE_IDENTITY: self.source_tokens,
            ROLE_MOTION: self.motion_tokens,
            ROLE_TARGET: self.target_tokens,
        }
        if counts != expected:
            raise RAMPRouteError("role counts differ from declared spans")

    @classmethod
    def contiguous(
        cls, *, source_tokens: int, target_tokens: int
    ) -> "TokenRoleLayout":
        source = _exact_positive_int(source_tokens, label="source_tokens")
        target = _exact_positive_int(target_tokens, label="target_tokens")
        roles = (
            (ROLE_IDENTITY,) * source
            + (ROLE_MOTION,) * LATENT_PHASES
            + (ROLE_TARGET,) * target
        )
        return cls(roles, source, LATENT_PHASES, target)

    @property
    def total_tokens(self) -> int:
        return len(self.roles)

    def as_dict(self) -> dict[str, Any]:
        value = {
            "source_tokens": self.source_tokens,
            "motion_tokens": self.motion_tokens,
            "target_tokens": self.target_tokens,
            "total_tokens": self.total_tokens,
            "order": ["identity", "motion", "target"],
            "raw_donor_tokens": 0,
        }
        return {**value, "digest": _object_sha256(value)}


@dataclass(frozen=True)
class RouteInvocation:
    layout: TokenRoleLayout
    sequence_parallel_rank: int
    sequence_parallel_size: int
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.layout, TokenRoleLayout):
            raise RAMPRouteError("route invocation requires a TokenRoleLayout")
        size = _exact_positive_int(self.sequence_parallel_size, label="SP size")
        rank = self.sequence_parallel_rank
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < size:
            raise RAMPRouteError("SP rank lies outside the SP group")
        if size not in {1, 4}:
            raise RAMPRouteError("RAMP C0 supports only SP1 engineering or SP4 AUH")
        if not isinstance(self.enabled, bool):
            raise RAMPRouteError("route enabled flag must be boolean")

    @property
    def local_length(self) -> int:
        return math.ceil(self.layout.total_tokens / self.sequence_parallel_size)

    def global_roles(self, *, device: torch.device) -> torch.Tensor:
        return torch.tensor(self.layout.roles, dtype=torch.int64, device=device)

    def local_roles(self, *, device: torch.device) -> torch.Tensor:
        roles = self.global_roles(device=device)
        padded = self.local_length * self.sequence_parallel_size
        if padded > int(roles.numel()):
            roles = torch.cat(
                [
                    roles,
                    torch.full(
                        (padded - int(roles.numel()),),
                        ROLE_PADDING,
                        dtype=torch.int64,
                        device=device,
                    ),
                ]
            )
        start = self.sequence_parallel_rank * self.local_length
        return roles[start : start + self.local_length].contiguous()


_ACTIVE_ROUTE: ContextVar[Optional[RouteInvocation]] = ContextVar(
    "bernini_ramp_active_route", default=None
)


def active_route() -> Optional[RouteInvocation]:
    return _ACTIVE_ROUTE.get()


@contextmanager
def activate_route(invocation: RouteInvocation) -> Iterator[None]:
    """Activate an immutable route; keep this open through ``loss.backward``."""

    if not isinstance(invocation, RouteInvocation):
        raise RAMPRouteError("activate_route requires a RouteInvocation")
    if active_route() is not None:
        raise RAMPRouteError("nested RAMP route contexts are forbidden")
    token: Token[Optional[RouteInvocation]] = _ACTIVE_ROUTE.set(invocation)
    try:
        yield
    finally:
        _ACTIVE_ROUTE.reset(token)


class RoleAwarePatchEmbedding(nn.Module):
    """Preserve Bernini patch embedding except for compact motion rows."""

    def __init__(self, base: nn.Module, *, hidden_size: int):
        super().__init__()
        self.base = base
        self.hidden_size = _exact_positive_int(hidden_size, label="hidden_size")
        self.role_embedding = nn.Embedding(len(ROLE_NAMES), self.hidden_size)
        self.program_projector = nn.Linear(LATENT_PHASES, self.hidden_size, bias=True)
        nn.init.zeros_(self.role_embedding.weight)
        nn.init.xavier_uniform_(self.program_projector.weight, gain=0.1)
        nn.init.zeros_(self.program_projector.bias)
        self.role_embedding.weight.requires_grad_(True)
        self.program_projector.requires_grad_(True)

    @property
    def weight(self) -> Any:
        return getattr(self.base, "weight")

    @property
    def bias(self) -> Any:
        return getattr(self.base, "bias", None)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        base_output = self.base(patches)
        invocation = active_route()
        if invocation is None or not invocation.enabled:
            return base_output
        if patches.ndim != 5 or tuple(int(item) for item in patches.shape[1:]) != (
            PATCH_CHANNELS,
            *PATCH_SHAPE,
        ):
            raise RAMPRouteError("role-aware patch input must be [N,16,1,2,2]")
        if base_output.ndim != 5 or tuple(int(item) for item in base_output.shape[2:]) != (1, 1, 1):
            raise RAMPRouteError("Bernini patch embedding output geometry changed")
        if int(base_output.shape[0]) != invocation.layout.total_tokens:
            raise RAMPRouteError("global patch count differs from active role layout")
        if int(base_output.shape[1]) != self.hidden_size:
            raise RAMPRouteError("patch embedding hidden width differs")

        roles = invocation.global_roles(device=patches.device)
        token_output = base_output.flatten(1)
        motion_rows = roles == ROLE_MOTION
        motion_patches = patches[motion_rows]
        if int(motion_patches.shape[0]) != LATENT_PHASES:
            raise RAMPRouteError("active layout lacks exact C0 motion rows")
        # Program patches are detached data.  Decode their registered 21-value
        # prefixes in FP32; the frozen Conv3d result for those rows is discarded.
        motion_matrix = recover_oracle_transport(
            motion_patches.detach().to(torch.float32).contiguous()
        )
        with torch.autocast(device_type=motion_matrix.device.type, enabled=False):
            projected = self.program_projector(motion_matrix).to(token_output.dtype)
        token_output = token_output.clone()
        token_output[motion_rows] = projected
        role_delta = self.role_embedding(roles).to(token_output.dtype)
        token_output = token_output + role_delta
        return token_output.reshape_as(base_output)


class TargetQueryLoRA(nn.Module):
    """Apply a low-rank update only to local rows with role=target."""

    def __init__(self, base: nn.Module, *, rank: int, alpha: float):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise RAMPRouteError("Bernini attn1.to_q must be nn.Linear")
        self.base = base
        self.rank = _exact_positive_int(rank, label="LoRA rank")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)):
            raise RAMPRouteError("LoRA alpha must be finite")
        if float(alpha) <= 0.0:
            raise RAMPRouteError("LoRA alpha must be positive")
        self.alpha = float(alpha)
        self.lora_a = nn.Linear(base.in_features, self.rank, bias=False, dtype=torch.float32)
        self.lora_b = nn.Linear(self.rank, base.out_features, bias=False, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.lora_b.weight)
        self.lora_a.requires_grad_(True)
        self.lora_b.requires_grad_(True)

    @property
    def scale(self) -> float:
        return self.alpha / float(self.rank)

    @property
    def weight(self) -> Any:
        return self.base.weight

    @property
    def bias(self) -> Any:
        return self.base.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_output = self.base(hidden_states)
        invocation = active_route()
        if invocation is None or not invocation.enabled:
            return base_output
        if hidden_states.ndim != 3 or int(hidden_states.shape[0]) != 1:
            raise RAMPRouteError("Bernini target-query LoRA expects [1,N,D]")
        local_roles = invocation.local_roles(device=hidden_states.device)
        if int(hidden_states.shape[1]) != int(local_roles.numel()):
            raise RAMPRouteError(
                "local hidden length differs from append-pad/contiguous Ulysses selector"
            )
        selector = (local_roles == ROLE_TARGET).view(1, -1, 1)
        # FP32 adapter arithmetic avoids an optimizer state tied to BF16 base
        # weights; only the final residual is cast to Bernini's compute dtype.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            delta = self.lora_b(self.lora_a(hidden_states.float())) * self.scale
        delta = delta.to(base_output.dtype) * selector.to(base_output.dtype)
        return base_output + delta


@dataclass
class RAMPAdapterHandle:
    transformer: nn.Module
    patch_wrapper: RoleAwarePatchEmbedding
    query_wrappers: tuple[TargetQueryLoRA, ...]
    original_patch_embedding: nn.Module
    original_queries: tuple[nn.Module, ...]
    restored: bool = False

    def __post_init__(self) -> None:
        if not self.query_wrappers or len(self.query_wrappers) != len(self.original_queries):
            raise RAMPRouteError("adapter handle has inconsistent query wrappers")

    def trainable_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        if self.restored:
            raise RAMPRouteError("adapter handle has already been restored")
        values: list[tuple[str, nn.Parameter]] = [
            ("role_embedding.weight", self.patch_wrapper.role_embedding.weight),
            ("program_projector.weight", self.patch_wrapper.program_projector.weight),
            ("program_projector.bias", self.patch_wrapper.program_projector.bias),
        ]
        for index, wrapper in enumerate(self.query_wrappers):
            values.extend(
                [
                    (f"blocks.{index}.attn1.to_q.lora_a.weight", wrapper.lora_a.weight),
                    (f"blocks.{index}.attn1.to_q.lora_b.weight", wrapper.lora_b.weight),
                ]
            )
        if len({id(parameter) for _, parameter in values}) != len(values):
            raise RAMPRouteError("trainable adapter parameter is aliased")
        if any(not parameter.requires_grad for _, parameter in values):
            raise RAMPRouteError("adapter parameter was unexpectedly frozen")
        return tuple(values)

    def base_parameters_frozen(self) -> bool:
        trainable_ids = {id(value) for _, value in self.trainable_named_parameters()}
        return all(
            (id(parameter) in trainable_ids) or (not parameter.requires_grad)
            for parameter in self.transformer.parameters()
        )

    @contextmanager
    def route(self, invocation: RouteInvocation) -> Iterator[None]:
        if self.restored:
            raise RAMPRouteError("cannot activate a restored adapter")
        with activate_route(invocation):
            yield

    def receipt(self) -> Mapping[str, Any]:
        trainable = self.trainable_named_parameters()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "query_wrappers": len(self.query_wrappers),
            "target_query_only": True,
            "source_key_value_frozen": True,
            "raw_donor_tokens": 0,
            "motion_tokens": LATENT_PHASES,
            "role_embedding_after_patch": True,
            "source_id_is_not_role": True,
            "sp_selector": "append_pad_then_contiguous_rank_chunk",
            "context_must_cover_forward_and_backward": True,
            "trainable": [
                {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
                for name, parameter in trainable
            ],
            "base_parameters_frozen": self.base_parameters_frozen(),
            "natural_action_claim": False,
        }
        return {**payload, "digest": _object_sha256(payload)}

    def restore(self) -> None:
        if self.restored:
            raise RAMPRouteError("adapter handle has already been restored")
        if active_route() is not None:
            raise RAMPRouteError("cannot restore while a route context is active")
        self.transformer.patch_embedding = self.original_patch_embedding
        blocks = tuple(getattr(self.transformer, "blocks"))
        if len(blocks) != len(self.original_queries):
            raise RAMPRouteError("transformer block count changed before restore")
        for block, original in zip(blocks, self.original_queries):
            block.attn1.to_q = original
        self.restored = True


def install_ramp_adapter(
    transformer: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 8.0,
) -> RAMPAdapterHandle:
    """Install role/program embedding and target-row Q LoRA into Bernini."""

    if not isinstance(transformer, nn.Module):
        raise RAMPRouteError("transformer must be an nn.Module")
    original_patch = getattr(transformer, "patch_embedding", None)
    blocks = tuple(getattr(transformer, "blocks", ()))
    if not isinstance(original_patch, nn.Conv3d) or not blocks:
        raise RAMPRouteError("transformer lacks Bernini Conv3d/blocks structure")
    if tuple(int(item) for item in original_patch.kernel_size) != PATCH_SHAPE:
        raise RAMPRouteError("Bernini patch kernel differs from (1,2,2)")
    hidden_size = int(original_patch.out_channels)
    originals: list[nn.Module] = []
    wrappers: list[TargetQueryLoRA] = []
    for index, block in enumerate(blocks):
        attention = getattr(block, "attn1", None)
        query = getattr(attention, "to_q", None)
        if isinstance(query, TargetQueryLoRA):
            raise RAMPRouteError(f"block {index} is already wrapped")
        if not isinstance(query, nn.Linear):
            raise RAMPRouteError(f"block {index} attn1.to_q is not nn.Linear")
        if query.in_features != hidden_size or query.out_features != hidden_size:
            raise RAMPRouteError(f"block {index} query width differs from patch width")
        originals.append(query)

    patch_wrapper = RoleAwarePatchEmbedding(original_patch, hidden_size=hidden_size)
    # Match device but deliberately keep adapter parameters FP32.
    device = original_patch.weight.device
    patch_wrapper.role_embedding.to(device=device, dtype=torch.float32)
    patch_wrapper.program_projector.to(device=device, dtype=torch.float32)
    transformer.patch_embedding = patch_wrapper
    for block, query in zip(blocks, originals):
        wrapper = TargetQueryLoRA(query, rank=rank, alpha=alpha).to(device=device)
        block.attn1.to_q = wrapper
        wrappers.append(wrapper)

    handle = RAMPAdapterHandle(
        transformer=transformer,
        patch_wrapper=patch_wrapper,
        query_wrappers=tuple(wrappers),
        original_patch_embedding=original_patch,
        original_queries=tuple(originals),
    )
    if not handle.base_parameters_frozen():
        # Restore before failing so installation never leaves a half-authorized
        # training path behind.
        handle.restore()
        raise RAMPRouteError("freeze the complete Bernini transformer before installing RAMP")
    return handle


__all__ = [
    "LATENT_PHASES",
    "PATCH_CHANNELS",
    "PATCH_SHAPE",
    "ROLE_IDENTITY",
    "ROLE_MOTION",
    "ROLE_TARGET",
    "RAMPAdapterHandle",
    "RAMPRouteError",
    "RouteInvocation",
    "TargetQueryLoRA",
    "TokenRoleLayout",
    "activate_route",
    "active_route",
    "install_ramp_adapter",
    "latent_phase_transport",
    "oracle_program_patches",
    "recover_oracle_transport",
]
