#!/usr/bin/env python3
"""Minimal fail-closed ELAL-3 representation and all-block C0 injection.

This module implements only the representation ABI and the causal/gradient
path needed by the synthetic C0 canary.  It does not implement or qualify the
offline action tokenizer, ActionPredictor, data contracts, or a trainer.

The native Bernini visual layout is treated as one contiguous source prefix
followed by one target suffix.  ELAL-3 residuals are written only to target
rows.  Sequence-parallel append-padding rows are never written.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import math
from typing import Any, Iterator, Mapping, Optional, Sequence

import torch
from torch import nn


SCHEMA_VERSION = "bernini-elal3-c0-v1"
LATENT_PHASES = 21
LOCAL_WIDTH = 64
ENTITY_SLOTS = 3
ENTITY_WIDTH = 256
RELATION_SLOTS = 6
RELATION_WIDTH = 128
PHASE_WIDTH = 128
TERMINAL_SLOTS = 9
TERMINAL_WIDTH = 256
CAMERA_WIDTH = 128
MEMORY_WIDTH = 256
MEMORY_TOKENS = (ENTITY_SLOTS + RELATION_SLOTS + 1) * LATENT_PHASES
BERNINI_BLOCKS = 30
BERNINI_HIDDEN = 1536
BERNINI_PATCH_SIZE = (1, 2, 2)
ALLOWED_SP_SIZES = (1, 4)
ALLOWED_VARIANTS = ("full", "no_relation")
RELATION_EDGES = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))


class ELAL3C0Error(RuntimeError):
    """Raised instead of accepting an ambiguous ELAL-3 C0 state."""


def _finite_float_tensor(
    value: Any, *, label: str, shape: Sequence[Optional[int]]
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or not value.is_floating_point()
        or value.device.type == "meta"
        or value.ndim != len(shape)
        or any(
            expected is not None and int(value.shape[index]) != expected
            for index, expected in enumerate(shape)
        )
        or not value.is_contiguous()
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise ELAL3C0Error(f"{label} must be contiguous finite float with shape {tuple(shape)}")
    return value


def _bool_tensor(
    value: Any, *, label: str, shape: Sequence[Optional[int]], device: torch.device
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.dtype != torch.bool
        or value.device != device
        or value.ndim != len(shape)
        or any(
            expected is not None and int(value.shape[index]) != expected
            for index, expected in enumerate(shape)
        )
        or not value.is_contiguous()
        or value.requires_grad
        or value.grad_fn is not None
    ):
        raise ELAL3C0Error(f"{label} must be detached contiguous bool with shape {tuple(shape)}")
    return value


def _tensor_bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare floating tensors byte-for-byte, including signed zero and NaN bits."""

    if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
        return False
    return bool(
        torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
    )


@dataclass(frozen=True)
class ELAL3LatentV1:
    """Fixed-capacity ELAL-3 latent, including non-injected terminal readout."""

    q_local: torch.Tensor
    q_entity: torch.Tensor
    q_relation: torch.Tensor
    q_phase: torch.Tensor
    q_terminal: torch.Tensor
    q_camera: torch.Tensor
    entity_presence: torch.Tensor
    temporal_valid: torch.Tensor
    relation_valid: torch.Tensor
    phase_valid: torch.Tensor

    def validate(self) -> None:
        local = _finite_float_tensor(
            self.q_local,
            label="q_local",
            shape=(None, LATENT_PHASES, None, None, LOCAL_WIDTH),
        )
        batch, _, height, width, _ = map(int, local.shape)
        if batch <= 0 or height <= 0 or width <= 0:
            raise ELAL3C0Error("q_local batch/grid must be positive")
        tensors = (
            (self.q_entity, "q_entity", (batch, ENTITY_SLOTS, LATENT_PHASES, ENTITY_WIDTH)),
            (self.q_relation, "q_relation", (batch, RELATION_SLOTS, LATENT_PHASES, RELATION_WIDTH)),
            (self.q_phase, "q_phase", (batch, LATENT_PHASES, PHASE_WIDTH)),
            (self.q_terminal, "q_terminal", (batch, TERMINAL_SLOTS, TERMINAL_WIDTH)),
            (self.q_camera, "q_camera", (batch, LATENT_PHASES, CAMERA_WIDTH)),
        )
        for value, label, shape in tensors:
            checked = _finite_float_tensor(value, label=label, shape=shape)
            if checked.device != local.device or checked.dtype != local.dtype:
                raise ELAL3C0Error(f"{label} device/dtype differs from q_local")
        presence = _bool_tensor(
            self.entity_presence,
            label="entity_presence",
            shape=(batch, ENTITY_SLOTS),
            device=local.device,
        )
        temporal = _bool_tensor(
            self.temporal_valid,
            label="temporal_valid",
            shape=(batch, ENTITY_SLOTS, LATENT_PHASES),
            device=local.device,
        )
        relation = _bool_tensor(
            self.relation_valid,
            label="relation_valid",
            shape=(batch, RELATION_SLOTS, LATENT_PHASES),
            device=local.device,
        )
        phase = _bool_tensor(
            self.phase_valid,
            label="phase_valid",
            shape=(batch, LATENT_PHASES),
            device=local.device,
        )
        if not bool(presence[:, 0].all().item()):
            raise ELAL3C0Error("designated-agent slot 0 must be present")
        if bool((temporal & ~presence[:, :, None]).any().item()):
            raise ELAL3C0Error("absent entity slots cannot have valid phases")
        if bool((presence & ~temporal.any(dim=2)).any().item()):
            raise ELAL3C0Error("every present entity needs at least one valid phase")
        for edge_index, (source, target) in enumerate(RELATION_EDGES):
            possible = temporal[:, source] & temporal[:, target]
            if bool((relation[:, edge_index] & ~possible).any().item()):
                raise ELAL3C0Error("relation validity exceeds endpoint validity")
        if not bool(phase.any(dim=1).all().item()):
            raise ELAL3C0Error("each row needs at least one valid phase")

    @property
    def batch_size(self) -> int:
        return int(self.q_local.shape[0])

    @property
    def local_grid(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.q_local.shape[1:4])

    @property
    def local_token_count(self) -> int:
        return math.prod(self.local_grid)


def intervene_elal3_v1(latent: ELAL3LatentV1, intervention: str) -> ELAL3LatentV1:
    """Apply a deterministic C0 intervention without changing tensor ranks."""

    latent.validate()
    if intervention == "correct":
        return latent
    if intervention == "zero":
        return ELAL3LatentV1(
            q_local=torch.zeros_like(latent.q_local),
            q_entity=torch.zeros_like(latent.q_entity),
            q_relation=torch.zeros_like(latent.q_relation),
            q_phase=torch.zeros_like(latent.q_phase),
            q_terminal=torch.zeros_like(latent.q_terminal),
            q_camera=latent.q_camera,
            entity_presence=latent.entity_presence,
            temporal_valid=latent.temporal_valid,
            relation_valid=latent.relation_valid,
            phase_valid=latent.phase_valid,
        )
    if intervention == "phase_reverse":
        return ELAL3LatentV1(
            q_local=latent.q_local.flip(1).contiguous(),
            q_entity=latent.q_entity.flip(2).contiguous(),
            q_relation=latent.q_relation.flip(2).contiguous(),
            q_phase=latent.q_phase.flip(1).contiguous(),
            q_terminal=latent.q_terminal,
            q_camera=latent.q_camera,
            entity_presence=latent.entity_presence,
            temporal_valid=latent.temporal_valid.flip(2).contiguous(),
            relation_valid=latent.relation_valid.flip(2).contiguous(),
            phase_valid=latent.phase_valid.flip(1).contiguous(),
        )
    if intervention == "relation_zero":
        return ELAL3LatentV1(
            q_local=latent.q_local,
            q_entity=latent.q_entity,
            q_relation=torch.zeros_like(latent.q_relation),
            q_phase=latent.q_phase,
            q_terminal=latent.q_terminal,
            q_camera=latent.q_camera,
            entity_presence=latent.entity_presence,
            temporal_valid=latent.temporal_valid,
            relation_valid=latent.relation_valid,
            phase_valid=latent.phase_valid,
        )
    if intervention != "role_slot_swap":
        raise ELAL3C0Error(f"unknown intervention: {intervention!r}")
    permutation = torch.tensor((1, 0, 2), dtype=torch.int64, device=latent.q_local.device)
    edge_lookup = {edge: index for index, edge in enumerate(RELATION_EDGES)}
    edge_permutation = torch.tensor(
        [
            edge_lookup[(int(permutation[source]), int(permutation[target]))]
            for source, target in RELATION_EDGES
        ],
        dtype=torch.int64,
        device=latent.q_local.device,
    )
    terminal = torch.cat(
        (
            latent.q_terminal.index_select(1, permutation),
            latent.q_terminal[:, ENTITY_SLOTS:].index_select(1, edge_permutation),
        ),
        dim=1,
    ).contiguous()
    return ELAL3LatentV1(
        q_local=latent.q_local,
        q_entity=latent.q_entity.index_select(1, permutation).contiguous(),
        q_relation=latent.q_relation.index_select(1, edge_permutation).contiguous(),
        q_phase=latent.q_phase,
        q_terminal=terminal,
        q_camera=latent.q_camera,
        entity_presence=latent.entity_presence.index_select(1, permutation).contiguous(),
        temporal_valid=latent.temporal_valid.index_select(1, permutation).contiguous(),
        relation_valid=latent.relation_valid.index_select(1, edge_permutation).contiguous(),
        phase_valid=latent.phase_valid,
    )


@dataclass(frozen=True)
class ELAL3ActionMemoryV1:
    tokens: torch.Tensor
    valid: torch.Tensor
    local_tokens: torch.Tensor
    local_grid: tuple[int, int, int]
    variant: str

    def validate(self) -> None:
        if self.variant not in ALLOWED_VARIANTS:
            raise ELAL3C0Error("action-memory variant differs")
        tokens = _finite_float_tensor(
            self.tokens,
            label="action memory",
            shape=(None, MEMORY_TOKENS, MEMORY_WIDTH),
        )
        batch = int(tokens.shape[0])
        valid = _bool_tensor(
            self.valid,
            label="action memory validity",
            shape=(batch, MEMORY_TOKENS),
            device=tokens.device,
        )
        local = _finite_float_tensor(
            self.local_tokens,
            label="q_local flattened tokens",
            shape=(batch, None, LOCAL_WIDTH),
        )
        if local.device != tokens.device or local.dtype != tokens.dtype:
            raise ELAL3C0Error("local/action memory device or dtype differs")
        if tuple(self.local_grid)[:1] != (LATENT_PHASES,) or math.prod(self.local_grid) != int(local.shape[1]):
            raise ELAL3C0Error("local action grid differs")
        if not bool(valid.any(dim=1).all().item()):
            raise ELAL3C0Error("every row needs at least one valid action token")
        relation_slice = valid[:, ENTITY_SLOTS * LATENT_PHASES : (ENTITY_SLOTS + RELATION_SLOTS) * LATENT_PHASES]
        if self.variant == "no_relation" and bool(relation_slice.any().item()):
            raise ELAL3C0Error("no_relation memory exposed relation tokens")


class ELAL3ActionMemoryBuilderV1(nn.Module):
    """Project fixed ELAL-3 fields into a 210-token action memory."""

    def __init__(self, *, variant: str) -> None:
        super().__init__()
        if variant not in ALLOWED_VARIANTS:
            raise ELAL3C0Error("unknown ELAL-3 variant")
        self.variant = variant
        self.entity_projection = nn.Linear(ENTITY_WIDTH, MEMORY_WIDTH, bias=False)
        self.relation_projection = (
            nn.Linear(RELATION_WIDTH, MEMORY_WIDTH, bias=False)
            if variant == "full"
            else None
        )
        self.phase_projection = nn.Linear(PHASE_WIDTH, MEMORY_WIDTH, bias=False)
        self.entity_slot = nn.Parameter(torch.empty(ENTITY_SLOTS, MEMORY_WIDTH))
        self.entity_time = nn.Parameter(torch.empty(LATENT_PHASES, MEMORY_WIDTH))
        self.relation_edge = (
            nn.Parameter(torch.empty(RELATION_SLOTS, MEMORY_WIDTH))
            if variant == "full"
            else None
        )
        self.relation_time = (
            nn.Parameter(torch.empty(LATENT_PHASES, MEMORY_WIDTH))
            if variant == "full"
            else None
        )
        self.phase_time = nn.Parameter(torch.empty(LATENT_PHASES, MEMORY_WIDTH))
        for module in (self.entity_projection, self.phase_projection):
            nn.init.xavier_uniform_(module.weight)
        if self.relation_projection is not None:
            nn.init.xavier_uniform_(self.relation_projection.weight)
        projection_parameter_ids = {
            id(parameter)
            for module in (
                self.entity_projection,
                self.phase_projection,
                self.relation_projection,
            )
            if module is not None
            for parameter in module.parameters()
        }
        for parameter in self.parameters():
            if (
                id(parameter) not in projection_parameter_ids
                and parameter.ndim == 2
                and parameter.shape[-1] == MEMORY_WIDTH
            ):
                nn.init.normal_(parameter, mean=0.0, std=MEMORY_WIDTH ** -0.5)

    def forward(self, latent: ELAL3LatentV1) -> ELAL3ActionMemoryV1:
        latent.validate()
        batch = latent.batch_size
        entity = self.entity_projection(latent.q_entity.float())
        entity = entity + self.entity_slot[None, :, None, :] + self.entity_time[None, None, :, :]
        if self.variant == "full":
            if self.relation_projection is None or self.relation_edge is None or self.relation_time is None:
                raise ELAL3C0Error("full relation modules are absent")
            relation = self.relation_projection(latent.q_relation.float())
            relation = relation + self.relation_edge[None, :, None, :] + self.relation_time[None, None, :, :]
            relation_valid = latent.relation_valid
        else:
            relation = entity.new_zeros((batch, RELATION_SLOTS, LATENT_PHASES, MEMORY_WIDTH))
            relation_valid = torch.zeros_like(latent.relation_valid)
        phase = self.phase_projection(latent.q_phase.float()) + self.phase_time[None, :, :]
        tokens = torch.cat(
            (
                entity.reshape(batch, ENTITY_SLOTS * LATENT_PHASES, MEMORY_WIDTH),
                relation.reshape(batch, RELATION_SLOTS * LATENT_PHASES, MEMORY_WIDTH),
                phase,
            ),
            dim=1,
        ).contiguous()
        entity_valid = (latent.entity_presence[:, :, None] & latent.temporal_valid).reshape(batch, -1)
        valid = torch.cat(
            (entity_valid, relation_valid.reshape(batch, -1), latent.phase_valid), dim=1
        ).contiguous()
        local = latent.q_local.reshape(batch, latent.local_token_count, LOCAL_WIDTH).float().contiguous()
        memory = ELAL3ActionMemoryV1(
            tokens=tokens,
            valid=valid,
            local_tokens=local,
            local_grid=latent.local_grid,
            variant=self.variant,
        )
        memory.validate()
        return memory


@dataclass(frozen=True)
class ELAL3RouteV1:
    total_tokens: int
    condition_tokens: int
    sequence_parallel_rank: int
    sequence_parallel_size: int
    memory: ELAL3ActionMemoryV1
    route_identity: str

    def __post_init__(self) -> None:
        if type(self.total_tokens) is not int or self.total_tokens <= 0:
            raise ELAL3C0Error("total_tokens must be positive")
        if type(self.condition_tokens) is not int or not 0 < self.condition_tokens < self.total_tokens:
            raise ELAL3C0Error("condition_tokens must define a strict source prefix")
        if self.sequence_parallel_size not in ALLOWED_SP_SIZES:
            raise ELAL3C0Error("only SP1 and SP4 are supported")
        if type(self.sequence_parallel_rank) is not int or not 0 <= self.sequence_parallel_rank < self.sequence_parallel_size:
            raise ELAL3C0Error("SP rank differs")
        if type(self.route_identity) is not str or not self.route_identity:
            raise ELAL3C0Error("route identity is empty")
        self.memory.validate()
        if self.target_tokens != int(self.memory.local_tokens.shape[1]):
            raise ELAL3C0Error("target suffix and q_local token counts differ")

    @property
    def target_tokens(self) -> int:
        return self.total_tokens - self.condition_tokens

    @property
    def local_length(self) -> int:
        return math.ceil(self.total_tokens / self.sequence_parallel_size)

    def local_global_indices(self, *, device: torch.device) -> torch.Tensor:
        start = self.sequence_parallel_rank * self.local_length
        return torch.arange(start, start + self.local_length, device=device, dtype=torch.int64)

    def local_target_indices(self, *, device: torch.device) -> torch.Tensor:
        global_indices = self.local_global_indices(device=device)
        result = global_indices - self.condition_tokens
        valid = (global_indices >= self.condition_tokens) & (global_indices < self.total_tokens)
        return torch.where(valid, result, torch.full_like(result, -1)).contiguous()

    def local_source_selector(self, *, device: torch.device) -> torch.Tensor:
        indices = self.local_global_indices(device=device)
        return ((indices >= 0) & (indices < self.condition_tokens)).contiguous()

    def local_padding_selector(self, *, device: torch.device) -> torch.Tensor:
        return (self.local_global_indices(device=device) >= self.total_tokens).contiguous()


_ACTIVE_ROUTE: ContextVar[Optional[ELAL3RouteV1]] = ContextVar(
    "bernini_elal3_route_v1", default=None
)


def active_elal3_route_v1() -> Optional[ELAL3RouteV1]:
    return _ACTIVE_ROUTE.get()


@contextmanager
def activate_elal3_route_v1(route: ELAL3RouteV1) -> Iterator[None]:
    if not isinstance(route, ELAL3RouteV1):
        raise ELAL3C0Error("route type differs")
    if active_elal3_route_v1() is not None:
        raise ELAL3C0Error("nested ELAL-3 routes are forbidden")
    token: Token[Optional[ELAL3RouteV1]] = _ACTIVE_ROUTE.set(route)
    try:
        yield
    finally:
        _ACTIVE_ROUTE.reset(token)


@contextmanager
def _replay_elal3_route_v1(route: ELAL3RouteV1) -> Iterator[None]:
    current = active_elal3_route_v1()
    if current is route:
        yield
        return
    if current is not None:
        raise ELAL3C0Error("checkpoint recomputation entered a different ELAL-3 route")
    with activate_elal3_route_v1(route):
        yield


def elal3_checkpoint_context_fn_v1() -> tuple[Any, Any]:
    route = active_elal3_route_v1()
    if route is None:
        raise ELAL3C0Error("checkpoint created without an ELAL-3 route")
    return _replay_elal3_route_v1(route), _replay_elal3_route_v1(route)


class ELAL3BlockInjectionV1(nn.Module):
    """One target-only action-memory attention plus aligned q_local residual."""

    def __init__(self, *, hidden_size: int, attention_width: int) -> None:
        super().__init__()
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ELAL3C0Error("hidden size must be positive")
        if attention_width not in (64, 128):
            raise ELAL3C0Error("attention width must be 64 or 128")
        self.hidden_size = hidden_size
        self.attention_width = attention_width
        self.num_heads = 8
        self.head_dim = attention_width // self.num_heads
        self.query_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.memory_norm = nn.LayerNorm(MEMORY_WIDTH, elementwise_affine=False)
        self.query = nn.Linear(hidden_size, attention_width, bias=False)
        self.key = nn.Linear(MEMORY_WIDTH, attention_width, bias=False)
        self.value = nn.Linear(MEMORY_WIDTH, attention_width, bias=False)
        self.output = nn.Linear(attention_width, hidden_size, bias=False)
        self.local_projection = nn.Linear(LOCAL_WIDTH, hidden_size, bias=False)
        self.residual_gain = nn.Parameter(torch.ones((), dtype=torch.float32))
        for projection in (self.query, self.key, self.value, self.output, self.local_projection):
            nn.init.xavier_uniform_(projection.weight)

    def adapter_delta(self, query_states: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(query_states, torch.Tensor)
            or query_states.layout != torch.strided
            or query_states.ndim != 3
            or int(query_states.shape[2]) != self.hidden_size
            or not query_states.is_floating_point()
        ):
            raise ELAL3C0Error("block input must be float [B,local_N,hidden]")
        route = active_elal3_route_v1()
        if route is None:
            raise ELAL3C0Error("ELAL-3 block ran without an authenticated route")
        memory = route.memory
        if int(query_states.shape[0]) != int(memory.tokens.shape[0]) or query_states.device != memory.tokens.device:
            raise ELAL3C0Error("query/action-memory batch or device differs")
        indices = route.local_target_indices(device=query_states.device)
        if int(query_states.shape[1]) != int(indices.numel()):
            raise ELAL3C0Error("local query length differs from SP route")
        graph_zero = self.residual_gain.to(query_states.dtype) * query_states.new_zeros(())
        result = query_states * graph_zero
        selector = indices >= 0
        if not bool(selector.any().item()):
            return result
        target_indices = indices[selector]
        target = query_states[:, selector, :].float()
        tokens = memory.tokens.float()
        valid = memory.valid
        q = self.query(self.query_norm(target)).view(
            int(target.shape[0]), -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.key(self.memory_norm(tokens)).view(
            int(tokens.shape[0]), -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.value(self.memory_norm(tokens)).view(
            int(tokens.shape[0]), -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim ** -0.5)
        logits = logits.masked_fill(~valid[:, None, None, :], torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(
            int(target.shape[0]), -1, self.attention_width
        )
        action_delta = self.output(attended)
        local_delta = self.local_projection(memory.local_tokens.index_select(1, target_indices))
        delta = (action_delta + local_delta) * self.residual_gain
        result[:, selector, :] = delta.to(query_states.dtype)
        return result.contiguous()

    def forward(self, query_states: torch.Tensor, native_output: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(native_output, torch.Tensor)
            or native_output.shape != query_states.shape
            or native_output.device != query_states.device
            or native_output.dtype != query_states.dtype
        ):
            raise ELAL3C0Error("native block input/output geometry differs")
        route = active_elal3_route_v1()
        if route is None:
            raise ELAL3C0Error("ELAL-3 block ran without an authenticated route")
        target_selector = route.local_target_indices(device=native_output.device) >= 0
        result = native_output.clone()
        if bool(target_selector.any().item()):
            delta = self.adapter_delta(query_states).to(native_output.dtype)
            result[:, target_selector, :] = (
                native_output[:, target_selector, :] + delta[:, target_selector, :]
            )
        return result.contiguous()


class _ELAL3ComponentsV1(nn.Module):
    def __init__(self, *, variant: str, hidden_size: int, attention_width: int) -> None:
        super().__init__()
        self.memory_builder = ELAL3ActionMemoryBuilderV1(variant=variant)
        self.injections = nn.ModuleList(
            ELAL3BlockInjectionV1(hidden_size=hidden_size, attention_width=attention_width)
            for _ in range(BERNINI_BLOCKS)
        )


def _output_tensor(output: Any) -> tuple[torch.Tensor, Any]:
    if isinstance(output, torch.Tensor):
        return output, lambda value: value
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0], lambda value: (value, *output[1:])
    raise ELAL3C0Error("block output must be Tensor or tensor-first tuple")


@dataclass
class ELAL3C0HandleV1:
    transformer: nn.Module
    components: _ELAL3ComponentsV1
    hooks: tuple[Any, ...]
    native_block_ids: tuple[int, ...]
    variant: str
    hidden_size: int
    attention_width: int
    test_only: bool
    audit_records: list[Mapping[str, Any]]
    restored: bool = False

    def build_memory(self, latent: ELAL3LatentV1) -> ELAL3ActionMemoryV1:
        if self.restored:
            raise ELAL3C0Error("adapter is restored")
        return self.components.memory_builder(latent)

    @contextmanager
    def route(self, route: ELAL3RouteV1) -> Iterator[None]:
        if self.restored:
            raise ELAL3C0Error("adapter is restored")
        with activate_elal3_route_v1(route):
            yield

    def trainable_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        values = tuple(self.components.named_parameters())
        if not values or any(not parameter.requires_grad for _, parameter in values):
            raise ELAL3C0Error("action module trainable scope differs")
        return values

    def clear_audit(self) -> None:
        self.audit_records.clear()

    def restore(self) -> None:
        if self.restored or active_elal3_route_v1() is not None:
            raise ELAL3C0Error("cannot restore adapter now")
        for hook in self.hooks:
            hook.remove()
        if getattr(self.transformer, "elal3_c0_v1", None) is not self.components:
            raise ELAL3C0Error("ELAL-3 component owner changed")
        delattr(self.transformer, "elal3_c0_v1")
        self.restored = True


def install_elal3_c0_v1(
    transformer: nn.Module,
    *,
    variant: str,
    attention_width: int,
    hidden_size: int = BERNINI_HIDDEN,
    test_only: bool = False,
) -> ELAL3C0HandleV1:
    """Install reversible post-block hooks on exactly all 30 blocks."""

    if not isinstance(transformer, nn.Module) or variant not in ALLOWED_VARIANTS:
        raise ELAL3C0Error("transformer or variant differs")
    if attention_width not in (64, 128) or (variant, attention_width) not in {
        ("no_relation", 64), ("full", 64), ("full", 128)
    }:
        raise ELAL3C0Error("arm must be no_relation-w64, full-w64, or full-w128")
    if hasattr(transformer, "elal3_c0_v1"):
        raise ELAL3C0Error("ELAL-3 is already installed")
    blocks = tuple(getattr(transformer, "blocks", ()))
    if len(blocks) != BERNINI_BLOCKS:
        raise ELAL3C0Error("ELAL-3 requires exactly 30 blocks")
    if not test_only:
        patch = getattr(transformer, "patch_embedding", None)
        if (
            hidden_size != BERNINI_HIDDEN
            or not isinstance(patch, nn.Conv3d)
            or patch.in_channels != 16
            or patch.out_channels != BERNINI_HIDDEN
            or tuple(patch.kernel_size) != BERNINI_PATCH_SIZE
            or tuple(patch.stride) != BERNINI_PATCH_SIZE
        ):
            raise ELAL3C0Error("production Bernini patch/hidden ABI differs")
        device = patch.weight.device
    else:
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ELAL3C0Error("test hidden size differs")
        first_parameter = next(transformer.parameters(), None)
        device = first_parameter.device if first_parameter is not None else torch.device("cpu")
    components = _ELAL3ComponentsV1(
        variant=variant, hidden_size=hidden_size, attention_width=attention_width
    ).to(device=device)
    transformer.add_module("elal3_c0_v1", components)
    audits: list[Mapping[str, Any]] = []
    hooks = []
    try:
        for block_index, (block, injection) in enumerate(zip(blocks, components.injections)):
            def callback(
                _module: nn.Module,
                args: tuple[Any, ...],
                output: Any,
                *,
                bound_index: int = block_index,
                bound_injection: ELAL3BlockInjectionV1 = injection,
            ) -> Any:
                if not args or not isinstance(args[0], torch.Tensor):
                    raise ELAL3C0Error("block input lacks hidden tensor")
                native, rebuild = _output_tensor(output)
                adapted = bound_injection(args[0], native)
                route = active_elal3_route_v1()
                if route is None:
                    raise ELAL3C0Error("hook audit has no route")
                source = route.local_source_selector(device=native.device)
                padding = route.local_padding_selector(device=native.device)
                audits.append(
                    {
                        "block_index": bound_index,
                        "route_identity": route.route_identity,
                        "source_bit_exact": _tensor_bits_equal(adapted[:, source], native[:, source]),
                        "padding_bit_exact": _tensor_bits_equal(adapted[:, padding], native[:, padding]),
                        "source_rows": int(source.sum().item()),
                        "padding_rows": int(padding.sum().item()),
                    }
                )
                return rebuild(adapted)

            hooks.append(block.register_forward_hook(callback))
    except Exception:
        for hook in hooks:
            hook.remove()
        delattr(transformer, "elal3_c0_v1")
        raise
    return ELAL3C0HandleV1(
        transformer=transformer,
        components=components,
        hooks=tuple(hooks),
        native_block_ids=tuple(id(block) for block in blocks),
        variant=variant,
        hidden_size=hidden_size,
        attention_width=attention_width,
        test_only=bool(test_only),
        audit_records=audits,
    )


__all__ = [
    "ALLOWED_VARIANTS",
    "BERNINI_BLOCKS",
    "BERNINI_HIDDEN",
    "CAMERA_WIDTH",
    "ELAL3ActionMemoryBuilderV1",
    "ELAL3ActionMemoryV1",
    "ELAL3BlockInjectionV1",
    "ELAL3C0Error",
    "ELAL3C0HandleV1",
    "ELAL3LatentV1",
    "ELAL3RouteV1",
    "LATENT_PHASES",
    "MEMORY_TOKENS",
    "RELATION_EDGES",
    "SCHEMA_VERSION",
    "activate_elal3_route_v1",
    "active_elal3_route_v1",
    "elal3_checkpoint_context_fn_v1",
    "install_elal3_c0_v1",
    "intervene_elal3_v1",
]
