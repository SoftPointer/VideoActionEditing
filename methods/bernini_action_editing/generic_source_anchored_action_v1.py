#!/usr/bin/env python3
"""Core modules and fail-closed contract for generic source-anchored editing.

This module contains the trainable architecture shared by the two registered
canaries:

``joint_source_anchored_v1``
    Source carrier R64, language planner P24, then action operator O16.

``action_only_no_carrier_v1``
    The byte-identical planner/operator initialization and P24/O16 row order,
    but no carrier module is installed or optimized.

The self-generated videos that author the detached action quotient are not
runtime inputs.  The optimizer-facing API accepts only frozen UMT5 states,
the reviewed 21x32 quotient, and current real-source hidden states.  Generated
RGB, latent, noise, velocity, reference media, action IDs, masks, tracks,
poses, and flow fields are rejected before an optimizer can be constructed.

This is an engineering primitive.  It does not itself claim decoded action or
preservation success.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterator, Mapping, Optional, Sequence
import weakref

import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from . import clean_source_visual_context_adapter_v1 as carrier_core
else:
    import clean_source_visual_context_adapter_v1 as carrier_core


SCHEMA_VERSION = "bernini-generic-source-anchored-action-core-v1"
TRAINING_RECEIPT_SCHEMA = (
    "bernini-generic-source-anchored-action-training-receipt-v1"
)
WORLD_SIZE = 4
SP_SIZE = 4
DP_SIZE = 1
TOPOLOGY = "world4-dp1-sp4"
FRAME_COUNT = 81
LATENT_PHASES = 21
TEXT_WIDTH = 4096
PLANNER_WIDTH = 256
PHASE_CODE_WIDTH = 32
ACTION_OPERATOR_RANK = 8
TOTAL_BLOCKS_1P3B = 30
HIDDEN_SIZE_1P3B = 1536
CARRIER_BLOCK_INDICES = (8, 12, 16, 20)
ACTION_BLOCK_INDICES = tuple(range(23))
TRAIN_SIGMA_INDICES = (4, 12, 20, 28, 35, 38)
R_SIGMA_COUNTS = (11, 11, 11, 11, 10, 10)
O_SIGMA_COUNTS = (3, 3, 3, 3, 2, 2)
STAGE_UPDATES = {"R": 64, "P": 24, "O": 16}
EXPERIMENTS = (
    "joint_source_anchored_v1",
    "action_only_no_carrier_v1",
)
DEFAULT_LEARNING_RATE = 1.0e-4
DEFAULT_MAX_GRAD_NORM = 1.0
OPERATOR_ZERO_INIT_COSINE_EPS = 1.0e-8
DEFAULT_SEED = 20260814
GPU_MEMORY_LIMIT_GIB = 52.0
HOST_MEMORY_LIMIT_GIB = 60.0
P32_SEED = 2026081401
PHI_BLOCK_INDEX = 22
PHI_TEACHER_SCHEDULE_INDEX = 29
EXPECTED_SOURCE_ONLY_MANIFEST_SHA256 = (
    "128064fd335c4e48c567217c6e7bae43555a904875625c9d1e21178e6f7fcc3d"
)
EXACT_NOOP_INSTRUCTION = (
    "Keep the source video exactly unchanged, including every subject, "
    "appearance, action, camera motion, background, timing, and composition."
)
EXACT_NOOP_INSTRUCTION_SHA256 = (
    "fb5f23b5b9de175696cff019f035e81eb1ee6a1123db7e3b63afb604b88daf3a"
)

PLANNER_PARAMETER_COUNT = 1_584_160
ACTION_OPERATOR_PARAMETER_COUNT = 1_142_272
CARRIER_PARAMETER_COUNT = 2_036_996
JOINT_PARAMETER_COUNT = (
    PLANNER_PARAMETER_COUNT
    + ACTION_OPERATOR_PARAMETER_COUNT
    + CARRIER_PARAMETER_COUNT
)


class GenericSourceAnchoredActionError(RuntimeError):
    """Raised rather than accepting an ambiguous or leaking training route."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise GenericSourceAnchoredActionError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GenericSourceAnchoredActionError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def validate_noop_instruction(instruction: Any, digest: Any) -> None:
    """Require the one canonical no-op byte string and its independent hash."""

    if (
        instruction != EXACT_NOOP_INSTRUCTION
        or digest != EXACT_NOOP_INSTRUCTION_SHA256
        or hashlib.sha256(str(instruction).encode("utf-8")).hexdigest() != digest
    ):
        raise GenericSourceAnchoredActionError(
            "canonical no-op instruction/string hash differs"
        )


_FORBIDDEN_OPTIMIZER_KEYS = frozenset(
    {
        "action_id",
        "action_family_id",
        "family_id",
        "one_hot_action",
        "generated_rgb",
        "generated_rgb_path",
        "generated_video",
        "generated_video_path",
        "generated_mp4",
        "generated_frames",
        "generated_pixels",
        "generated_latent",
        "generated_latents",
        "clean_latent",
        "clean_latents",
        "target_latent",
        "target_latents",
        "target_velocity",
        "generated_velocity",
        "teacher_velocity",
        "noise",
        "gaussian",
        "epsilon",
        "t2v_reference",
        "t2v_references",
        "anchor_pixels",
        "actor_pixels",
        "background_pixels",
        "camera_pixels",
        "pose",
        "poses",
        "flow",
        "optical_flow",
        "track",
        "tracks",
        "mask",
        "masks",
        "trajectory",
    }
)


def assert_optimizer_payload_safe(value: Any, *, path: str = "row") -> None:
    """Recursively reject privileged/generated optimizer inputs.

    Provenance may retain hashes and review receipts, but the optimizer row
    cannot expose media paths or tensors from the self-generated teacher.  A
    separate manifest validator owns the exact admitted field set; this guard
    remains intentionally redundant at the trainer boundary.
    """

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if type(raw_key) is not str:
                raise GenericSourceAnchoredActionError(
                    f"{path} contains a non-text field name"
                )
            key = raw_key.casefold()
            if key in _FORBIDDEN_OPTIMIZER_KEYS:
                raise GenericSourceAnchoredActionError(
                    f"optimizer payload exposes forbidden field {path}.{raw_key}"
                )
            assert_optimizer_payload_safe(child, path=f"{path}.{raw_key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_optimizer_payload_safe(child, path=f"{path}[{index}]")
        return
    if isinstance(value, torch.Tensor) and path != "row.quotient_tensor":
        raise GenericSourceAnchoredActionError(
            f"optimizer payload embeds an unregistered tensor at {path}"
        )


def stage_sequence(experiment: str) -> tuple[str, ...]:
    if experiment == "joint_source_anchored_v1":
        return ("R", "P", "O")
    if experiment == "action_only_no_carrier_v1":
        return ("P", "O")
    raise GenericSourceAnchoredActionError("experiment is not registered")


def stage_update_count(experiment: str) -> int:
    return sum(STAGE_UPDATES[stage] for stage in stage_sequence(experiment))


def component_initialization_seed(base_seed: int, component: str) -> int:
    if (
        type(base_seed) is not int
        or not 0 <= base_seed < 2**63
        or component not in {"carrier", "planner", "operator"}
    ):
        raise GenericSourceAnchoredActionError(
            "component initialization seed request differs"
        )
    payload = (
        f"{base_seed}\0generic-source-anchored-action-v1\0{component}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


@contextmanager
def _component_rng(transformer: nn.Module, seed: int) -> Iterator[None]:
    device = next(transformer.parameters()).device
    devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        yield


def fixed_sigma_schedule(stage: str) -> tuple[int, ...]:
    if stage == "R":
        counts = R_SIGMA_COUNTS
    elif stage == "O":
        counts = O_SIGMA_COUNTS
    else:
        raise GenericSourceAnchoredActionError(
            "only R/O have diffusion sigma schedules"
        )
    values: list[int] = []
    remaining = list(counts)
    while any(remaining):
        for position, schedule_index in enumerate(TRAIN_SIGMA_INDICES):
            if remaining[position]:
                values.append(schedule_index)
                remaining[position] -= 1
    expected = STAGE_UPDATES[stage]
    if len(values) != expected:
        raise GenericSourceAnchoredActionError("fixed sigma schedule differs")
    return tuple(values)


def deterministic_row_order_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    keys = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise GenericSourceAnchoredActionError("row order contains non-object")
        canonical = row.get("canonical_sha256")
        keys.append(_sha256(canonical, label="row canonical_sha256"))
    if len(set(keys)) != len(keys):
        raise GenericSourceAnchoredActionError("row order contains duplicate SHA")
    return object_sha256(keys)


def sinusoidal_phase_queries(
    phases: int = LATENT_PHASES, width: int = PLANNER_WIDTH
) -> torch.Tensor:
    if type(phases) is not int or phases <= 0 or type(width) is not int or width <= 0:
        raise GenericSourceAnchoredActionError("phase-query geometry differs")
    half = max(width // 2, 1)
    positions = torch.arange(phases, dtype=torch.float32).unsqueeze(1)
    denominator = max(half - 1, 1)
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32)
        / denominator
    ).unsqueeze(0)
    encoded = torch.cat(
        (torch.sin(positions * frequencies), torch.cos(positions * frequencies)),
        dim=1,
    )
    if int(encoded.shape[1]) < width:
        encoded = F.pad(encoded, (0, width - int(encoded.shape[1])))
    return encoded[:, :width].contiguous()


class NaturalLanguagePhasePlanner(nn.Module):
    """Frozen-UMT5 tokens to one shared 21x32 phase program."""

    def __init__(self) -> None:
        super().__init__()
        self.text_projection = nn.Linear(TEXT_WIDTH, PLANNER_WIDTH, dtype=torch.float32)
        self.cross_attention = nn.MultiheadAttention(
            PLANNER_WIDTH,
            4,
            dropout=0.0,
            batch_first=True,
            dtype=torch.float32,
        )
        self.attention_norm = nn.LayerNorm(PLANNER_WIDTH, dtype=torch.float32)
        self.feedforward_norm = nn.LayerNorm(PLANNER_WIDTH, dtype=torch.float32)
        self.feedforward = nn.Sequential(
            nn.Linear(PLANNER_WIDTH, 512, dtype=torch.float32),
            nn.GELU(approximate="tanh"),
            nn.Linear(512, PLANNER_WIDTH, dtype=torch.float32),
        )
        self.output = nn.Linear(PLANNER_WIDTH, PHASE_CODE_WIDTH, dtype=torch.float32)
        self.register_buffer(
            "phase_queries",
            sinusoidal_phase_queries(),
            persistent=True,
        )

    def trainable_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        values = tuple(self.named_parameters())
        if sum(parameter.numel() for _, parameter in values) != PLANNER_PARAMETER_COUNT:
            raise GenericSourceAnchoredActionError(
                "natural-language planner parameter count differs"
            )
        return values

    def forward(
        self,
        instruction_tokens: torch.Tensor,
        *,
        instruction: str,
        instruction_sha256: str,
        is_noop: bool = False,
    ) -> torch.Tensor:
        if type(is_noop) is not bool:
            raise GenericSourceAnchoredActionError("is_noop must be boolean")
        if is_noop:
            validate_noop_instruction(instruction, instruction_sha256)
            if not isinstance(instruction_tokens, torch.Tensor) or instruction_tokens.ndim != 3:
                raise GenericSourceAnchoredActionError(
                    "no-op batch geometry cannot be established"
                )
            return instruction_tokens.new_zeros(
                (int(instruction_tokens.shape[0]), LATENT_PHASES, PHASE_CODE_WIDTH),
                dtype=torch.float32,
            )
        if instruction == EXACT_NOOP_INSTRUCTION or instruction_sha256 == EXACT_NOOP_INSTRUCTION_SHA256:
            raise GenericSourceAnchoredActionError(
                "canonical no-op must declare is_noop and take the hard bypass"
            )
        _sha256(instruction_sha256, label="instruction_sha256")
        if hashlib.sha256(instruction.encode("utf-8")).hexdigest() != instruction_sha256:
            raise GenericSourceAnchoredActionError("instruction SHA differs")
        if (
            not isinstance(instruction_tokens, torch.Tensor)
            or instruction_tokens.ndim != 3
            or int(instruction_tokens.shape[0]) <= 0
            or int(instruction_tokens.shape[1]) <= 0
            or int(instruction_tokens.shape[2]) != TEXT_WIDTH
            or instruction_tokens.requires_grad
            or not bool(torch.isfinite(instruction_tokens).all().item())
        ):
            raise GenericSourceAnchoredActionError(
                "instruction tokens must be detached finite [B,L,4096]"
            )
        with torch.autocast(
            device_type=instruction_tokens.device.type, enabled=False
        ):
            tokens = self.text_projection(instruction_tokens.float())
            queries = self.phase_queries.to(tokens.device).unsqueeze(0).expand(
                int(tokens.shape[0]), -1, -1
            )
            attended, _ = self.cross_attention(
                self.attention_norm(queries), tokens, tokens, need_weights=False
            )
            states = queries + attended
            states = states + self.feedforward(
                self.feedforward_norm(states)
            )
            code = self.output(states)
        if tuple(code.shape[1:]) != (LATENT_PHASES, PHASE_CODE_WIDTH):
            raise GenericSourceAnchoredActionError("planner output geometry differs")
        return code.float().contiguous()


@dataclass(frozen=True)
class ActionRoute:
    """Authenticated target suffix and phase program for one native SP shard."""

    total_tokens: int
    condition_tokens: int
    sequence_parallel_rank: int
    sequence_parallel_size: int
    phase_code: torch.Tensor
    schedule_index: int
    is_noop: bool = False
    enabled: bool = True

    def __post_init__(self) -> None:
        if (
            type(self.total_tokens) is not int
            or self.total_tokens <= 0
            or type(self.condition_tokens) is not int
            or not 0 <= self.condition_tokens < self.total_tokens
            or type(self.sequence_parallel_size) is not int
            or self.sequence_parallel_size not in (1, SP_SIZE)
            or type(self.sequence_parallel_rank) is not int
            or not 0 <= self.sequence_parallel_rank < self.sequence_parallel_size
            or type(self.schedule_index) is not int
            or not 0 <= self.schedule_index < 40
            or type(self.is_noop) is not bool
            or type(self.enabled) is not bool
        ):
            raise GenericSourceAnchoredActionError("action route metadata differs")
        if (
            not isinstance(self.phase_code, torch.Tensor)
            or self.phase_code.ndim != 3
            or tuple(self.phase_code.shape) != (1, LATENT_PHASES, PHASE_CODE_WIDTH)
            or self.phase_code.dtype != torch.float32
            or not bool(torch.isfinite(self.phase_code.detach()).all().item())
        ):
            raise GenericSourceAnchoredActionError(
                "action route requires finite FP32 [1,21,32] phase code"
            )
        if self.is_noop and bool(torch.count_nonzero(self.phase_code.detach()).item()):
            raise GenericSourceAnchoredActionError(
                "no-op action route phase code must be exact zero"
            )

    @property
    def target_tokens(self) -> int:
        return self.total_tokens - self.condition_tokens

    @property
    def patch_positions(self) -> int:
        if self.target_tokens % LATENT_PHASES:
            raise GenericSourceAnchoredActionError(
                "target suffix is not an exact 21-phase geometry"
            )
        return self.target_tokens // LATENT_PHASES

    @property
    def local_length(self) -> int:
        return math.ceil(self.total_tokens / self.sequence_parallel_size)

    def local_phase_indices(self, *, device: torch.device) -> torch.Tensor:
        phases = torch.cat(
            (
                torch.full(
                    (self.condition_tokens,), -1, dtype=torch.int64, device=device
                ),
                torch.arange(LATENT_PHASES, dtype=torch.int64, device=device)
                .repeat_interleave(self.patch_positions),
            )
        )
        padded = self.local_length * self.sequence_parallel_size
        if padded > self.total_tokens:
            phases = F.pad(phases, (0, padded - self.total_tokens), value=-1)
        start = self.sequence_parallel_rank * self.local_length
        return phases[start : start + self.local_length].contiguous()

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "total_tokens": self.total_tokens,
            "condition_tokens": self.condition_tokens,
            "target_tokens": self.target_tokens,
            "patch_positions_per_phase": self.patch_positions,
            "sequence_parallel_rank": self.sequence_parallel_rank,
            "sequence_parallel_size": self.sequence_parallel_size,
            "schedule_index": self.schedule_index,
            "all_exact40_operator_available": True,
            "is_noop": self.is_noop,
            "enabled": self.enabled,
            "source_reference_padding_rows_written": False,
            "phase0_hard_bypass": True,
        }
        return {**value, "digest": object_sha256(value)}


_ACTIVE_ACTION_ROUTE: ContextVar[Optional[ActionRoute]] = ContextVar(
    "bernini_generic_source_anchored_action_route_v1", default=None
)


def active_action_route() -> Optional[ActionRoute]:
    return _ACTIVE_ACTION_ROUTE.get()


@contextmanager
def activate_action_route(route: ActionRoute) -> Iterator[None]:
    if not isinstance(route, ActionRoute):
        raise GenericSourceAnchoredActionError("route must be ActionRoute")
    if active_action_route() is not None:
        raise GenericSourceAnchoredActionError("nested action routes are forbidden")
    token: Token[Optional[ActionRoute]] = _ACTIVE_ACTION_ROUTE.set(route)
    try:
        yield
    finally:
        _ACTIVE_ACTION_ROUTE.reset(token)


@contextmanager
def _replay_composite_checkpoint_routes(
    carrier_route: Optional[carrier_core.VisualContextRoute],
    action_route: ActionRoute,
) -> Iterator[None]:
    """Restore the exact carrier/action pair during checkpoint recompute.

    Bernini uses non-reentrant activation checkpointing.  A plain ContextVar
    route is no longer active when the backward pass asks a block to recompute
    its forward, so relying on the lexical ``with composite_route(...)`` alone
    silently turns the learned action residual off during recomputation.  This
    context is identity-safe when the original forward is still active and
    reinstalls only the captured objects during backward recomputation.
    """

    current_carrier = carrier_core.active_route()
    current_action = active_action_route()
    if current_carrier is not None and current_carrier is not carrier_route:
        raise GenericSourceAnchoredActionError(
            "checkpoint recomputation entered a different carrier route"
        )
    if current_action is not None and current_action is not action_route:
        raise GenericSourceAnchoredActionError(
            "checkpoint recomputation entered a different action route"
        )
    with ExitStack() as stack:
        if carrier_route is not None and current_carrier is None:
            stack.enter_context(carrier_core.activate_route(carrier_route))
        if current_action is None:
            stack.enter_context(activate_action_route(action_route))
        yield


def composite_checkpoint_route_context_fn() -> tuple[Any, Any]:
    """Capture both routes for Bernini's non-reentrant checkpoint replay."""

    action_route = active_action_route()
    if action_route is None:
        raise GenericSourceAnchoredActionError(
            "checkpoint was created without an active action route"
        )
    carrier_route = carrier_core.active_route()
    return (
        _replay_composite_checkpoint_routes(carrier_route, action_route),
        _replay_composite_checkpoint_routes(carrier_route, action_route),
    )


class CurrentHiddenPhaseResidual(nn.Module):
    """Bias-free rank-8 Q/O residual conditioned on current hidden and phase."""

    def __init__(self, base: nn.Linear, *, projection: str) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear) or projection not in {"to_q", "to_out.0"}:
            raise GenericSourceAnchoredActionError(
                "operator may wrap only linear attn2 Q/O"
            )
        if any(parameter.requires_grad for parameter in base.parameters()):
            raise GenericSourceAnchoredActionError("operator base must be frozen")
        self.base = base
        self.projection = projection
        self.state_down = nn.Linear(
            base.in_features, ACTION_OPERATOR_RANK, bias=False, dtype=torch.float32
        )
        self.phase_gate = nn.Linear(
            PHASE_CODE_WIDTH, ACTION_OPERATOR_RANK, bias=False, dtype=torch.float32
        )
        self.output_up = nn.Linear(
            ACTION_OPERATOR_RANK, base.out_features, bias=False, dtype=torch.float32
        )
        nn.init.kaiming_uniform_(self.state_down.weight, a=math.sqrt(5.0))
        nn.init.kaiming_uniform_(self.phase_gate.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.output_up.weight)
        # Ephemeral runtime evidence only.  This is deliberately not a
        # parameter/buffer and therefore never enters a checkpoint.  Stage O
        # consumes and clears it immediately after each native forward.
        self._last_runtime_audit: Optional[Mapping[str, Any]] = None

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    def _delta(
        self, hidden_states: torch.Tensor, phases: torch.Tensor, route: ActionRoute
    ) -> torch.Tensor:
        selected = phases > 0  # phase 0 is architectural frozen-base bypass.
        rows = hidden_states[:, selected, :]
        row_phases = phases[selected]
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            codes = route.phase_code.to(hidden_states.device).index_select(
                1, row_phases
            )
            state = F.silu(self.state_down(rows.float()))
            gate = torch.tanh(self.phase_gate(codes.float()))
            delta = self.output_up(state * gate)
        return delta.to(hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base = self.base(hidden_states)
        route = active_action_route()
        if route is None:
            self._last_runtime_audit = {
                "projection": self.projection,
                "route_present": False,
                "route_enabled": False,
                "is_noop": False,
                "hard_bypass": True,
            }
            return base
        if (
            not isinstance(hidden_states, torch.Tensor)
            or hidden_states.ndim != 3
            or int(hidden_states.shape[0]) != 1
        ):
            raise GenericSourceAnchoredActionError(
                "operator hidden states must be [1,local_N,D]"
            )
        phases = route.local_phase_indices(device=hidden_states.device)
        if int(phases.numel()) != int(hidden_states.shape[1]):
            raise GenericSourceAnchoredActionError(
                "operator local sequence differs from append-pad/SP route"
            )
        protected = phases <= 0
        phase_zero = phases == 0
        source_reference_padding = phases < 0
        if not route.enabled or route.is_noop:
            self._last_runtime_audit = {
                "projection": self.projection,
                "route_present": True,
                "route_enabled": route.enabled,
                "is_noop": route.is_noop,
                "schedule_index": route.schedule_index,
                "local_rows": int(phases.numel()),
                "source_reference_padding_rows": int(
                    source_reference_padding.sum().item()
                ),
                "phase0_rows": int(phase_zero.sum().item()),
                "positive_phase_target_rows": int((phases > 0).sum().item()),
                "protected_rows_bit_exact": True,
                "hard_bypass": True,
                "selected_delta_l2": 0.0,
                "selected_delta_nonzero": False,
            }
            return base
        selected = phases > 0
        if not bool(selected.any().item()):
            self._last_runtime_audit = {
                "projection": self.projection,
                "route_present": True,
                "route_enabled": True,
                "is_noop": False,
                "schedule_index": route.schedule_index,
                "local_rows": int(phases.numel()),
                "source_reference_padding_rows": int(
                    source_reference_padding.sum().item()
                ),
                "phase0_rows": int(phase_zero.sum().item()),
                "positive_phase_target_rows": 0,
                "protected_rows_bit_exact": True,
                "hard_bypass": False,
                "selected_delta_l2": 0.0,
                "selected_delta_nonzero": False,
            }
            return base
        result = base.clone()
        delta = self._delta(hidden_states, phases, route)
        result[:, selected, :] = base[:, selected, :] + delta
        protected_exact = torch.equal(result[:, protected, :], base[:, protected, :])
        delta_l2 = float(delta.detach().float().norm().item())
        delta_nonzero = bool(torch.count_nonzero(delta.detach()).item())
        self._last_runtime_audit = {
            "projection": self.projection,
            "route_present": True,
            "route_enabled": True,
            "is_noop": False,
            "schedule_index": route.schedule_index,
            "local_rows": int(phases.numel()),
            "source_reference_padding_rows": int(
                source_reference_padding.sum().item()
            ),
            "phase0_rows": int(phase_zero.sum().item()),
            "positive_phase_target_rows": int(selected.sum().item()),
            "protected_rows_bit_exact": protected_exact,
            "hard_bypass": False,
            "selected_delta_l2": delta_l2,
            "selected_delta_nonzero": delta_nonzero,
        }
        if not protected_exact:
            raise GenericSourceAnchoredActionError(
                "operator directly changed source/reference/padding or phase-0 rows"
            )
        return result


@dataclass
class ActionOperatorHandle:
    transformer: nn.Module
    wrappers: tuple[tuple[str, CurrentHiddenPhaseResidual], ...]
    originals: tuple[tuple[str, nn.Linear], ...]
    restored: bool = False

    def trainable_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        if self.restored:
            raise GenericSourceAnchoredActionError("action operator is restored")
        rows: list[tuple[str, nn.Parameter]] = []
        for prefix, wrapper in self.wrappers:
            rows.extend(
                (
                    (f"{prefix}.state_down.weight", wrapper.state_down.weight),
                    (f"{prefix}.phase_gate.weight", wrapper.phase_gate.weight),
                    (f"{prefix}.output_up.weight", wrapper.output_up.weight),
                )
            )
        if len(rows) != len(ACTION_BLOCK_INDICES) * 2 * 3:
            raise GenericSourceAnchoredActionError("operator key count differs")
        if len({id(parameter) for _, parameter in rows}) != len(rows):
            raise GenericSourceAnchoredActionError("operator parameter aliases")
        return tuple(rows)

    @contextmanager
    def route(self, route: ActionRoute) -> Iterator[None]:
        if self.restored:
            raise GenericSourceAnchoredActionError("action operator is restored")
        with activate_action_route(route):
            yield

    def state_dict_for_save(self) -> Mapping[str, torch.Tensor]:
        return {
            name: parameter.detach().float().cpu().contiguous().clone()
            for name, parameter in self.trainable_named_parameters()
        }

    def pop_runtime_audits(self) -> tuple[Mapping[str, Any], ...]:
        """Consume exact per-wrapper row-write evidence from one forward."""

        rows: list[Mapping[str, Any]] = []
        for prefix, wrapper in self.wrappers:
            audit = wrapper._last_runtime_audit
            wrapper._last_runtime_audit = None
            if not isinstance(audit, Mapping):
                raise GenericSourceAnchoredActionError(
                    f"operator wrapper {prefix} did not execute in the native forward"
                )
            rows.append({"wrapper": prefix, **dict(audit)})
        if len(rows) != len(ACTION_BLOCK_INDICES) * 2:
            raise GenericSourceAnchoredActionError(
                "operator runtime audit wrapper closure differs"
            )
        return tuple(rows)

    def clear_runtime_audits(self) -> None:
        """Discard checkpoint-recompute audits before the next sentinel."""

        for _, wrapper in self.wrappers:
            wrapper._last_runtime_audit = None

    def restore(self) -> None:
        if self.restored or active_action_route() is not None:
            raise GenericSourceAnchoredActionError(
                "action operator cannot be restored now"
            )
        blocks = tuple(getattr(self.transformer, "blocks", ()))
        for name, original in self.originals:
            parts = name.split(".")
            index = int(parts[1])
            if parts[-1] == "to_q":
                blocks[index].attn2.to_q = original
            else:
                blocks[index].attn2.to_out[0] = original
        self.restored = True


def install_action_operator_v1(
    transformer: nn.Module, *, strict_production_shape: bool = True
) -> ActionOperatorHandle:
    if not isinstance(transformer, nn.Module):
        raise GenericSourceAnchoredActionError("transformer must be nn.Module")
    if any(parameter.requires_grad for parameter in transformer.parameters()):
        raise GenericSourceAnchoredActionError(
            "freeze transformer and any installed carrier before operator install"
        )
    blocks = tuple(getattr(transformer, "blocks", ()))
    if len(blocks) != TOTAL_BLOCKS_1P3B:
        raise GenericSourceAnchoredActionError("Bernini block count differs")
    originals: list[tuple[str, nn.Linear]] = []
    wrappers: list[tuple[str, CurrentHiddenPhaseResidual]] = []
    try:
        for index in ACTION_BLOCK_INDICES:
            attention = getattr(blocks[index], "attn2", None)
            output = getattr(attention, "to_out", None)
            query = getattr(attention, "to_q", None)
            if (
                not isinstance(query, nn.Linear)
                or not isinstance(output, nn.ModuleList)
                or len(output) != 2
                or not isinstance(output[0], nn.Linear)
                or query.in_features != query.out_features
                or output[0].in_features != output[0].out_features
                or query.in_features != output[0].in_features
                or (strict_production_shape and query.in_features != HIDDEN_SIZE_1P3B)
            ):
                raise GenericSourceAnchoredActionError(
                    f"block {index} native attn2 Q/O differs"
                )
            q_name = f"blocks.{index}.attn2.to_q"
            o_name = f"blocks.{index}.attn2.to_out.0"
            originals.extend(((q_name, query), (o_name, output[0])))
            q_wrapper = CurrentHiddenPhaseResidual(query, projection="to_q").to(
                device=query.weight.device
            )
            o_wrapper = CurrentHiddenPhaseResidual(
                output[0], projection="to_out.0"
            ).to(device=output[0].weight.device)
            attention.to_q = q_wrapper
            output[0] = o_wrapper
            wrappers.extend(((q_name, q_wrapper), (o_name, o_wrapper)))
    except Exception:
        for name, original in originals:
            parts = name.split(".")
            block = blocks[int(parts[1])]
            if parts[-1] == "to_q":
                block.attn2.to_q = original
            else:
                block.attn2.to_out[0] = original
        raise
    handle = ActionOperatorHandle(transformer, tuple(wrappers), tuple(originals))
    if strict_production_shape:
        count = sum(
            parameter.numel() for _, parameter in handle.trainable_named_parameters()
        )
        if count != ACTION_OPERATOR_PARAMETER_COUNT:
            handle.restore()
            raise GenericSourceAnchoredActionError(
                "production action operator parameter count differs"
            )
    return handle


def fixed_p32() -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(P32_SEED)
    raw = torch.randn(
        (HIDDEN_SIZE_1P3B, PHASE_CODE_WIDTH),
        generator=generator,
        dtype=torch.float32,
    )
    q, r = torch.linalg.qr(raw, mode="reduced")
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return (q * signs.unsqueeze(0)).float().contiguous()


def _project_off(code: torch.Tensor, nuisance: Optional[torch.Tensor]) -> torch.Tensor:
    if nuisance is None:
        return code
    if (
        not isinstance(nuisance, torch.Tensor)
        or nuisance.shape != code.shape
        or not bool(torch.isfinite(nuisance.detach()).all().item())
    ):
        raise GenericSourceAnchoredActionError("nuisance coordinate differs")
    direction = nuisance.to(device=code.device, dtype=torch.float32).reshape(-1)
    norm = direction.square().sum()
    if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 1.0e-12:
        raise GenericSourceAnchoredActionError("nuisance direction is degenerate")
    flat = code.float().reshape(-1)
    projected = flat - (flat @ direction / norm) * direction
    return projected.reshape_as(code)


def _gram_schmidt_nuisances(
    code: torch.Tensor,
    camera: Optional[torch.Tensor],
    appearance: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    directions: list[Optional[torch.Tensor]] = []
    camera_unit: Optional[torch.Tensor] = None
    if camera is not None:
        if camera.shape != code.shape or not bool(torch.isfinite(camera).all().item()):
            raise GenericSourceAnchoredActionError("camera nuisance coordinate differs")
        camera_flat = camera.to(device=code.device, dtype=torch.float32).reshape(-1)
        camera_norm = camera_flat.norm()
        if not bool(torch.isfinite(camera_norm).item()) or float(camera_norm.item()) <= 1.0e-6:
            raise GenericSourceAnchoredActionError("camera nuisance direction is degenerate")
        camera_unit = (camera_flat / camera_norm).reshape_as(code)
    directions.append(camera_unit)

    appearance_unit: Optional[torch.Tensor] = None
    if appearance is not None:
        if appearance.shape != code.shape or not bool(torch.isfinite(appearance).all().item()):
            raise GenericSourceAnchoredActionError(
                "appearance nuisance coordinate differs"
            )
        appearance_flat = appearance.to(
            device=code.device, dtype=torch.float32
        ).reshape(-1)
        appearance_original_norm = appearance_flat.norm()
        if (
            not bool(torch.isfinite(appearance_original_norm).item())
            or float(appearance_original_norm.item()) <= 1.0e-6
        ):
            raise GenericSourceAnchoredActionError(
                "appearance nuisance direction is degenerate"
            )
        if camera_unit is not None:
            camera_flat = camera_unit.reshape(-1)
            appearance_flat = appearance_flat - (
                appearance_flat @ camera_flat
            ) * camera_flat
        appearance_norm = appearance_flat.norm()
        if (
            not bool(torch.isfinite(appearance_norm).item())
            or float(appearance_norm.item())
            <= max(1.0e-6, 1.0e-5 * float(appearance_original_norm.item()))
        ):
            raise GenericSourceAnchoredActionError(
                "appearance nuisance direction is degenerate after Gram-Schmidt"
            )
        appearance_unit = (appearance_flat / appearance_norm).reshape_as(code)
    directions.append(appearance_unit)
    return directions[0], directions[1]


def phi_v1_from_global_hidden_delta(
    hidden_delta: torch.Tensor,
    *,
    condition_tokens: int,
    p32: torch.Tensor,
    camera_nuisance: Optional[torch.Tensor] = None,
    appearance_nuisance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply the registered block-22 spatial-orderless temporal quotient."""

    if (
        not isinstance(hidden_delta, torch.Tensor)
        or hidden_delta.ndim != 3
        or int(hidden_delta.shape[0]) != 1
        or int(hidden_delta.shape[2]) != HIDDEN_SIZE_1P3B
        or type(condition_tokens) is not int
        or not 0 <= condition_tokens < int(hidden_delta.shape[1])
        or not isinstance(p32, torch.Tensor)
        or tuple(p32.shape) != (HIDDEN_SIZE_1P3B, PHASE_CODE_WIDTH)
        or p32.dtype != torch.float32
    ):
        raise GenericSourceAnchoredActionError("Phi_v1 input geometry differs")
    target = hidden_delta[:, condition_tokens:, :]
    if int(target.shape[1]) % LATENT_PHASES:
        raise GenericSourceAnchoredActionError("Phi_v1 target suffix differs")
    spatial = int(target.shape[1]) // LATENT_PHASES
    pooled = target.reshape(1, LATENT_PHASES, spatial, HIDDEN_SIZE_1P3B).float().mean(2)
    temporal = pooled.clone()
    temporal[:, 0, :] = 0.0
    temporal[:, 1:, :] = temporal[:, 1:, :] - temporal[:, 1:, :].mean(
        dim=1, keepdim=True
    )
    code = torch.matmul(temporal, p32.to(temporal.device))
    camera_direction, appearance_direction = _gram_schmidt_nuisances(
        code, camera_nuisance, appearance_nuisance
    )
    code = _project_off(code, camera_direction)
    code = _project_off(code, appearance_direction)
    return code.float().contiguous()


def phi_v1_from_sp_hidden_delta(
    local_hidden_delta: torch.Tensor,
    *,
    route: ActionRoute,
    p32: torch.Tensor,
    sp_group: Any = None,
    camera_nuisance: Optional[torch.Tensor] = None,
    appearance_nuisance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply ``Phi_v1`` to one contiguous SP shard without losing gradients.

    Only per-phase sums are communicated.  This is equivalent to gathering the
    append-padded global block-22 hidden sequence, removing condition/padding
    rows, and taking the registered fixed spatial mean, but avoids materializing
    four copies of the full ``[N,1536]`` tensor.  ``torch.distributed.nn`` is
    required for SP4 so the all-reduce remains in the autograd graph.
    """

    if (
        not isinstance(local_hidden_delta, torch.Tensor)
        or local_hidden_delta.ndim != 3
        or tuple(local_hidden_delta.shape[:1]) != (1,)
        or int(local_hidden_delta.shape[1]) != route.local_length
        or int(local_hidden_delta.shape[2]) != HIDDEN_SIZE_1P3B
        or not bool(torch.isfinite(local_hidden_delta.detach()).all().item())
        or route.sequence_parallel_size not in (1, SP_SIZE)
    ):
        raise GenericSourceAnchoredActionError(
            "Phi_v1 local SP hidden geometry differs"
        )
    phases = route.local_phase_indices(device=local_hidden_delta.device)
    selected = phases >= 0
    selected_phases = phases[selected]
    local_sums = local_hidden_delta.new_zeros(
        (LATENT_PHASES, HIDDEN_SIZE_1P3B), dtype=torch.float32
    )
    local_sums = local_sums.index_add(
        0, selected_phases, local_hidden_delta[0, selected, :].float()
    )
    local_counts = torch.bincount(
        selected_phases, minlength=LATENT_PHASES
    ).to(device=local_hidden_delta.device, dtype=torch.float32)
    if route.sequence_parallel_size == 1:
        global_sums = local_sums
        global_counts = local_counts
    else:
        if sp_group is None:
            raise GenericSourceAnchoredActionError(
                "Phi_v1 SP4 requires the authenticated SP process group"
            )
        try:
            from torch.distributed.nn.functional import all_reduce as autograd_all_reduce
            import torch.distributed as dist
        except (ImportError, AttributeError) as error:
            raise GenericSourceAnchoredActionError(
                "autograd-aware SP4 all-reduce is unavailable"
            ) from error
        global_sums = autograd_all_reduce(
            local_sums, op=dist.ReduceOp.SUM, group=sp_group
        )
        global_counts = local_counts.clone()
        dist.all_reduce(global_counts, op=dist.ReduceOp.SUM, group=sp_group)
    expected_counts = torch.full_like(global_counts, float(route.patch_positions))
    if not torch.equal(global_counts, expected_counts):
        raise GenericSourceAnchoredActionError(
            "Phi_v1 SP4 phase/spatial population differs"
        )
    pooled = (global_sums / global_counts.unsqueeze(1)).unsqueeze(0)
    temporal = pooled.clone()
    temporal[:, 0, :] = 0.0
    temporal[:, 1:, :] = temporal[:, 1:, :] - temporal[:, 1:, :].mean(
        dim=1, keepdim=True
    )
    if (
        not isinstance(p32, torch.Tensor)
        or tuple(p32.shape) != (HIDDEN_SIZE_1P3B, PHASE_CODE_WIDTH)
        or p32.dtype != torch.float32
    ):
        raise GenericSourceAnchoredActionError("Phi_v1 P32 coordinate differs")
    code = torch.matmul(temporal, p32.to(temporal.device))
    camera_direction, appearance_direction = _gram_schmidt_nuisances(
        code, camera_nuisance, appearance_nuisance
    )
    code = _project_off(code, camera_direction)
    code = _project_off(code, appearance_direction)
    return code.float().contiguous()


def cosine_quotient_loss(prediction: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(prediction, torch.Tensor)
        or not isinstance(teacher, torch.Tensor)
        or prediction.shape != teacher.shape
        or tuple(prediction.shape[-2:]) != (LATENT_PHASES, PHASE_CODE_WIDTH)
        or teacher.requires_grad
        or not bool(torch.isfinite(prediction.detach()).all().item())
        or not bool(torch.isfinite(teacher.detach()).all().item())
    ):
        raise GenericSourceAnchoredActionError("quotient loss inputs differ")
    pred_flat = prediction.float().reshape(int(prediction.shape[0]), -1)
    teacher_flat = teacher.float().reshape(int(teacher.shape[0]), -1)
    if bool((pred_flat.norm(dim=1) <= 1.0e-12).any().item()) or bool(
        (teacher_flat.norm(dim=1) <= 1.0e-12).any().item()
    ):
        raise GenericSourceAnchoredActionError(
            "non-noop quotient loss received a zero vector"
        )
    loss = 1.0 - F.cosine_similarity(pred_flat, teacher_flat, dim=1).mean()
    if not bool(torch.isfinite(loss.detach()).item()):
        raise GenericSourceAnchoredActionError("quotient loss is non-finite")
    return loss


def zero_init_operator_cosine_quotient_loss(
    prediction: torch.Tensor,
    teacher: torch.Tensor,
    *,
    stage_update: int,
) -> torch.Tensor:
    """Stage-O cosine with one explicit zero-initialization exception.

    Every action ``output_up`` is architecturally zero at the sealed R64
    boundary.  The frozen-base action/noop contrast can make the first Phi
    prediction nonzero, but an exact-zero first prediction remains possible
    after the registered quotient/projections even though it has a real
    autograd path.  PyTorch's fixed-epsilon cosine denominator supplies the
    intended first output-up gradient in only that case.  This exception is
    legal only at O update 1; it never weakens the strict planner loss and
    never permits a zero teacher or a later collapsed prediction.
    """

    if type(stage_update) is not int or stage_update <= 0:
        raise GenericSourceAnchoredActionError(
            "Stage-O cosine requires a positive stage update"
        )
    if (
        not isinstance(prediction, torch.Tensor)
        or not isinstance(teacher, torch.Tensor)
        or prediction.shape != teacher.shape
        or tuple(prediction.shape[-2:]) != (LATENT_PHASES, PHASE_CODE_WIDTH)
        or teacher.requires_grad
        or not bool(torch.isfinite(prediction.detach()).all().item())
        or not bool(torch.isfinite(teacher.detach()).all().item())
    ):
        raise GenericSourceAnchoredActionError(
            "Stage-O quotient loss inputs differ"
        )
    pred_flat = prediction.float().reshape(int(prediction.shape[0]), -1)
    teacher_flat = teacher.float().reshape(int(teacher.shape[0]), -1)
    prediction_zero = pred_flat.norm(dim=1) <= 1.0e-12
    if bool((teacher_flat.norm(dim=1) <= 1.0e-12).any().item()):
        raise GenericSourceAnchoredActionError(
            "Stage-O quotient loss received a zero teacher"
        )
    if stage_update != 1 and bool(prediction_zero.any().item()):
        raise GenericSourceAnchoredActionError(
            "Stage-O prediction collapsed after the zero-init update"
        )
    loss = 1.0 - F.cosine_similarity(
        pred_flat,
        teacher_flat,
        dim=1,
        eps=OPERATOR_ZERO_INIT_COSINE_EPS,
    ).mean()
    if not bool(torch.isfinite(loss.detach()).item()):
        raise GenericSourceAnchoredActionError(
            "Stage-O quotient loss is non-finite"
        )
    return loss


class BlockOutputCapture:
    """Capture one block output without detaching its optimizer graph."""

    def __init__(self, transformer: nn.Module, block_index: int = PHI_BLOCK_INDEX):
        blocks = tuple(getattr(transformer, "blocks", ()))
        if len(blocks) != TOTAL_BLOCKS_1P3B or block_index != PHI_BLOCK_INDEX:
            raise GenericSourceAnchoredActionError("Phi capture block differs")
        self.value: Optional[torch.Tensor] = None

        def callback(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            candidate = output[0] if isinstance(output, tuple) else output
            if not isinstance(candidate, torch.Tensor):
                raise GenericSourceAnchoredActionError(
                    "Phi block output is not a tensor"
                )
            self.value = candidate

        self._hook = blocks[block_index].register_forward_hook(callback)

    def pop(self) -> torch.Tensor:
        value = self.value
        self.value = None
        if value is None:
            raise GenericSourceAnchoredActionError("Phi block output was not captured")
        return value

    def close(self) -> None:
        self._hook.remove()


@dataclass
class CompositeHandle:
    transformer: nn.Module
    planner: NaturalLanguagePhasePlanner
    operator: ActionOperatorHandle
    carrier: Optional[carrier_core.CleanSourceVisualContextHandle]
    carrier_parameter_rows: tuple[tuple[str, nn.Parameter], ...]

    def named_parameter_groups(self) -> Mapping[str, tuple[tuple[str, nn.Parameter], ...]]:
        if self.carrier is None:
            if self.carrier_parameter_rows:
                raise GenericSourceAnchoredActionError(
                    "carrier parameter registry exists without a carrier"
                )
            carrier = ()
        else:
            observed = tuple(self.carrier.components.named_parameters())
            if (
                not self.carrier_parameter_rows
                or len(observed) != len(self.carrier_parameter_rows)
                or any(
                    observed_name != cached_name
                    or observed_parameter is not cached_parameter
                    for (observed_name, observed_parameter),
                    (cached_name, cached_parameter) in zip(
                        observed, self.carrier_parameter_rows
                    )
                )
            ):
                raise GenericSourceAnchoredActionError(
                    "carrier parameter registry changed after installation"
                )
            carrier = tuple(
                (f"carrier.{name}", parameter)
                for name, parameter in self.carrier_parameter_rows
            )
        planner = tuple(
            (f"planner.{name}", parameter)
            for name, parameter in self.planner.trainable_named_parameters()
        )
        operator = tuple(
            (f"operator.{name}", parameter)
            for name, parameter in self.operator.trainable_named_parameters()
        )
        groups = {"R": carrier, "P": planner, "O": operator}
        all_ids = [id(parameter) for rows in groups.values() for _, parameter in rows]
        if len(all_ids) != len(set(all_ids)):
            raise GenericSourceAnchoredActionError("composite trainable aliases")
        return groups

    def set_active_stage(self, stage: str) -> tuple[tuple[str, nn.Parameter], ...]:
        if stage not in stage_sequence(
            "joint_source_anchored_v1"
            if self.carrier is not None
            else "action_only_no_carrier_v1"
        ):
            raise GenericSourceAnchoredActionError("stage is unavailable")
        groups = self.named_parameter_groups()
        for candidate, rows in groups.items():
            for _, parameter in rows:
                parameter.requires_grad_(candidate == stage)
                if candidate != stage:
                    parameter.grad = None
        active = groups[stage]
        observed = {
            id(parameter)
            for parameter in self.transformer.parameters()
            if parameter.requires_grad
        } | {
            id(parameter)
            for parameter in self.planner.parameters()
            if parameter.requires_grad
        }
        if observed != {id(parameter) for _, parameter in active}:
            raise GenericSourceAnchoredActionError(
                "active stage parameter scope differs"
            )
        return active

    def parameter_count_receipt(self) -> Mapping[str, Any]:
        groups = self.named_parameter_groups()
        counts = {
            stage: sum(parameter.numel() for _, parameter in rows)
            for stage, rows in groups.items()
        }
        expected_r = CARRIER_PARAMETER_COUNT if self.carrier is not None else 0
        if (
            counts["R"] != expected_r
            or counts["P"] != PLANNER_PARAMETER_COUNT
            or counts["O"] != ACTION_OPERATOR_PARAMETER_COUNT
        ):
            raise GenericSourceAnchoredActionError(
                "composite parameter count differs"
            )
        value = {
            "carrier": counts["R"],
            "planner": counts["P"],
            "operator": counts["O"],
            "total": sum(counts.values()),
        }
        return {**value, "digest": object_sha256(value)}


def install_composite_v1(
    transformer: nn.Module,
    *,
    experiment: str,
    runtime_source_commit: str,
    model_revision: str,
    checkpoint_manifest_sha256: str,
    initialization_seed: int = DEFAULT_SEED,
) -> CompositeHandle:
    stages = stage_sequence(experiment)
    if any(parameter.requires_grad for parameter in transformer.parameters()):
        raise GenericSourceAnchoredActionError("base transformer must be frozen")
    carrier: Optional[carrier_core.CleanSourceVisualContextHandle] = None
    carrier_parameter_rows: tuple[tuple[str, nn.Parameter], ...] = ()
    if "R" in stages:
        with _component_rng(
            transformer,
            component_initialization_seed(initialization_seed, "carrier"),
        ):
            carrier = carrier_core.install_clean_source_visual_context_adapter_v1(
                transformer,
                runtime_source_commit=runtime_source_commit,
                model_revision=model_revision,
                checkpoint_manifest_sha256=checkpoint_manifest_sha256,
                block_indices=CARRIER_BLOCK_INDICES,
            )
        carrier_parameter_rows = tuple(carrier.trainable_named_parameters())
        carrier.components.requires_grad_(False)
    with _component_rng(
        transformer,
        component_initialization_seed(initialization_seed, "operator"),
    ):
        operator = install_action_operator_v1(transformer)
    with _component_rng(
        transformer,
        component_initialization_seed(initialization_seed, "planner"),
    ):
        planner = NaturalLanguagePhasePlanner().to(
            device=next(transformer.parameters()).device
        )
    planner.requires_grad_(False)
    operator_params = operator.trainable_named_parameters()
    for _, parameter in operator_params:
        parameter.requires_grad_(False)
    handle = CompositeHandle(
        transformer,
        planner,
        operator,
        carrier,
        carrier_parameter_rows,
    )
    handle.parameter_count_receipt()
    return handle


@contextmanager
def composite_route(
    handle: CompositeHandle,
    *,
    carrier_route: Optional[carrier_core.VisualContextRoute],
    action_route: ActionRoute,
) -> Iterator[None]:
    if (handle.carrier is None) != (carrier_route is None):
        raise GenericSourceAnchoredActionError(
            "carrier installation and route presence differ"
        )
    with ExitStack() as stack:
        if handle.carrier is not None:
            assert carrier_route is not None
            stack.enter_context(handle.carrier.route(carrier_route))
        stack.enter_context(handle.operator.route(action_route))
        yield


class StageOptimizerController:
    """One AdamW whose inactive groups have LR zero and requires_grad false."""

    def __init__(
        self,
        handle: CompositeHandle,
        *,
        learning_rate: float = DEFAULT_LEARNING_RATE,
    ) -> None:
        if learning_rate != DEFAULT_LEARNING_RATE:
            raise GenericSourceAnchoredActionError("learning rate is fixed at 1e-4")
        self.handle = handle
        groups = handle.named_parameter_groups()
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": [parameter for _, parameter in groups[stage]],
                    "lr": 0.0,
                    "name": stage,
                }
                for stage in ("R", "P", "O")
                if groups[stage]
            ],
            lr=0.0,
            betas=(0.9, 0.95),
            eps=1.0e-8,
            weight_decay=0.0,
        )
        self.active_stage: Optional[str] = None

    def activate(self, stage: str) -> tuple[tuple[str, nn.Parameter], ...]:
        active = self.handle.set_active_stage(stage)
        for group in self.optimizer.param_groups:
            group["lr"] = DEFAULT_LEARNING_RATE if group["name"] == stage else 0.0
        self.active_stage = stage
        return active

    def assert_inactive_unchanged(
        self, before: Mapping[str, torch.Tensor]
    ) -> None:
        if self.active_stage is None:
            raise GenericSourceAnchoredActionError("optimizer stage is inactive")
        groups = self.handle.named_parameter_groups()
        for stage, rows in groups.items():
            if stage == self.active_stage:
                continue
            for name, parameter in rows:
                key = f"{stage}:{name}"
                if key not in before or not torch.equal(
                    before[key], parameter.detach().cpu()
                ):
                    raise GenericSourceAnchoredActionError(
                        "inactive parameter changed across optimizer step"
                    )


def frozen_inactive_snapshot(handle: CompositeHandle, active_stage: str) -> Mapping[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for stage, rows in handle.named_parameter_groups().items():
        if stage != active_stage:
            for name, parameter in rows:
                result[f"{stage}:{name}"] = parameter.detach().cpu().clone()
    return result


def training_contract_receipt(experiment: str) -> Mapping[str, Any]:
    stages = stage_sequence(experiment)
    value = {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "topology": TOPOLOGY,
        "world_size": WORLD_SIZE,
        "dp_size": DP_SIZE,
        "sp_size": SP_SIZE,
        "one_shared_model": True,
        "rank_or_gpu_action_family_partition": False,
        "stages": list(stages),
        "stage_updates": {stage: STAGE_UPDATES[stage] for stage in stages},
        "total_updates": stage_update_count(experiment),
        "carrier_policy": "installed_blocks_8_12_16_20_same_noise" if "R" in stages else "not_installed",
        "planner": "frozen_UMT5_full_tokens_to_21x32_no_action_id",
        "operator": "blocks_0_22_target_row_attn2_QO_rank8_current_hidden",
        "noop_hard_bypass": True,
        "phase0_hard_bypass": True,
        "all_exact40_operator_available": True,
        "training_sigma_indices": list(TRAIN_SIGMA_INDICES),
        "r_sigma_counts": list(R_SIGMA_COUNTS) if "R" in stages else None,
        "o_sigma_counts": list(O_SIGMA_COUNTS),
        "self_generated_rgb_latent_noise_velocity_target": False,
        "action_family_id_consumed": False,
        "gpu_memory_reserved_hard_limit_gib": GPU_MEMORY_LIMIT_GIB,
        "host_memory_rss_hard_limit_gib": HOST_MEMORY_LIMIT_GIB,
        "scientific_success_claimed": False,
    }
    return {**value, "digest": object_sha256(value)}


__all__ = [
    "ACTION_BLOCK_INDICES",
    "ACTION_OPERATOR_PARAMETER_COUNT",
    "ACTION_OPERATOR_RANK",
    "ActionOperatorHandle",
    "ActionRoute",
    "BlockOutputCapture",
    "CARRIER_BLOCK_INDICES",
    "CARRIER_PARAMETER_COUNT",
    "CompositeHandle",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_MAX_GRAD_NORM",
    "DEFAULT_SEED",
    "DP_SIZE",
    "EXACT_NOOP_INSTRUCTION",
    "EXACT_NOOP_INSTRUCTION_SHA256",
    "EXPERIMENTS",
    "EXPECTED_SOURCE_ONLY_MANIFEST_SHA256",
    "GPU_MEMORY_LIMIT_GIB",
    "GenericSourceAnchoredActionError",
    "HOST_MEMORY_LIMIT_GIB",
    "HIDDEN_SIZE_1P3B",
    "LATENT_PHASES",
    "NaturalLanguagePhasePlanner",
    "O_SIGMA_COUNTS",
    "OPERATOR_ZERO_INIT_COSINE_EPS",
    "P32_SEED",
    "PHASE_CODE_WIDTH",
    "PHI_BLOCK_INDEX",
    "PHI_TEACHER_SCHEDULE_INDEX",
    "PLANNER_PARAMETER_COUNT",
    "R_SIGMA_COUNTS",
    "SCHEMA_VERSION",
    "SP_SIZE",
    "STAGE_UPDATES",
    "StageOptimizerController",
    "TEXT_WIDTH",
    "TOPOLOGY",
    "TRAINING_RECEIPT_SCHEMA",
    "TRAIN_SIGMA_INDICES",
    "WORLD_SIZE",
    "activate_action_route",
    "active_action_route",
    "assert_optimizer_payload_safe",
    "canonical_json_bytes",
    "composite_checkpoint_route_context_fn",
    "composite_route",
    "component_initialization_seed",
    "cosine_quotient_loss",
    "deterministic_row_order_sha256",
    "fixed_p32",
    "fixed_sigma_schedule",
    "frozen_inactive_snapshot",
    "install_action_operator_v1",
    "install_composite_v1",
    "object_sha256",
    "phi_v1_from_global_hidden_delta",
    "phi_v1_from_sp_hidden_delta",
    "sinusoidal_phase_queries",
    "stage_sequence",
    "stage_update_count",
    "training_contract_receipt",
    "validate_noop_instruction",
    "zero_init_operator_cosine_quotient_loss",
]
