#!/usr/bin/env python3
"""Typed, state-aware action operator for frozen Bernini-R 1.3B.

SAIC (Source-Anchored Inverse-Cycle Action Operator) represents an edit as a typed
``initial_state -> terminal_state`` arrow.  The arrow is injected only into
the noisy-target suffix of Bernini's native visual sequence, and only through
cross-attention ``attn2.to_q`` and ``attn2.to_out[0]`` in blocks 0..22.
Source/reference rows, append-padding rows, K/V projections, self-attention,
late blocks, the patch embedding, and the complete Bernini base stay frozen.

For selected target rows in sigma stratum ``s`` the residual is

    up_s(silu(state_down_s(h)) * tanh(arrow_gate_s(r))) * sigma_gate

where ``s`` is either ``high`` or ``mid`` and the two parameter sets are
disjoint.  Every learned projection is bias-free and rank 8.  Both ``up_s``
heads are zero initialized, so installation is an exact function-preserving
operation.  The odd ``tanh`` gate makes the pre-up feature of the sign-reversed
arrow exactly the negative of the forward arrow for the same hidden state.
No-op arrows and the two registered low-sigma steps return the native base
projection directly without evaluating either learned path.  Sign reversal of
a typed arrow only guarantees same-hidden-state oddness of this pre-up feature;
it is not a claim that either routed network evaluation undoes the other.

The operator consumes only an inference-available typed action code.  It does
not consume a proposal video, target video, mask, flow, pose, track, or
trajectory.  This module provides routing, installation, restoration, and a
closed FP32 checkpoint surface; it makes no semantic-success claim by itself.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Optional
import weakref

import torch
from torch import nn
from torch.nn import functional as F

try:  # Package import used by trainers and tests launched from the repository.
    from . import inference_sigma_strata as sigma_strata
except ImportError:  # Direct ``python saic_typed_action_operator_v1.py`` fallback.
    if __package__ not in (None, ""):
        raise
    import inference_sigma_strata as sigma_strata


SCHEMA_VERSION = "bernini-saic-typed-action-operator-v1"
CHECKPOINT_SCHEMA_VERSION = "bernini-saic-typed-action-checkpoint-v1"
TOTAL_BLOCKS_1P3B = 30
ACTION_BLOCK_INDICES = tuple(range(23))
ACTION_OPERATOR_RANK = 8
ARROW_CODE_DIM = 32
ALLOWED_SP_SIZES = frozenset({1, 4})
NATIVE_BRANCHES = ("none", "V", "I", "VI")

# Pinned Bernini-R RV2V-4 visual concatenation descriptors.  The runtime route
# factory observes these fields on the already-built native branch; callers do
# not report a branch label or token geometry separately.
NATIVE_BRANCH_ORDERED_DESCRIPTORS = {
    "none": (("target", "00000000"),),
    "V": (("video", "3f800000"), ("target", "00000000")),
    "I": (
        ("ref0", "3f800000"),
        ("ref1", "40000000"),
        ("ref2", "40400000"),
        ("ref3", "40800000"),
        ("target", "00000000"),
    ),
    "VI": (
        ("video", "3f800000"),
        ("ref0", "40000000"),
        ("ref1", "40400000"),
        ("ref2", "40800000"),
        ("ref3", "40a00000"),
        ("target", "00000000"),
    ),
}

HIGH_SIGMA_INDICES = tuple(range(33))
MID_SIGMA_INDICES = tuple(range(33, 38))
LOW_SIGMA_INDICES = tuple(range(38, 40))
HIGH_SIGMA_WEIGHT = 1.0
MID_SIGMA_WEIGHT = 0.5
LOW_SIGMA_WEIGHT = 0.0

_STATE_TYPE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


class SAICTypedActionOperatorError(RuntimeError):
    """Raised before an ambiguous SAIC route, module, or state is used."""


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
        raise SAICTypedActionOperatorError(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SAICTypedActionOperatorError(f"{label} must be a positive integer")
    return value


def _validate_state_type(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _STATE_TYPE.fullmatch(value) is None:
        raise SAICTypedActionOperatorError(
            f"{label} must match [a-z][a-z0-9_]{{0,63}}"
        )
    return value


def _quantize_float32(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SAICTypedActionOperatorError(f"{label} must be a real scalar")
    try:
        result = struct.unpack(">f", struct.pack(">f", float(value)))[0]
    except (OverflowError, struct.error) as error:
        raise SAICTypedActionOperatorError(f"{label} is outside finite FP32") from error
    if not math.isfinite(result):
        raise SAICTypedActionOperatorError(f"{label} must be finite FP32")
    return result


def _require_exact_float32(value: Any, *, label: str) -> float:
    result = _quantize_float32(value, label=label)
    if float(value) != result:
        raise SAICTypedActionOperatorError(
            f"{label} is not an exact FP32 scalar; use SAICArrowCode.quantized"
        )
    return result


def _float32_be_hex(value: float) -> str:
    return struct.pack(">f", value).hex()


def _bool_mask_sha256(value: torch.Tensor, *, label: str) -> str:
    if (
        type(value) is not torch.Tensor
        or value.dtype != torch.bool
        or value.ndim != 1
        or value.requires_grad
        or value.grad_fn is not None
    ):
        raise SAICTypedActionOperatorError(
            f"{label} must be a detached rank-1 torch.bool tensor"
        )
    cpu = value.detach().to(device="cpu").contiguous()
    payload = {
        "schema": "saic-bool-mask-v1",
        "length": int(cpu.numel()),
        "true_count": int(cpu.sum().item()),
        "uint8_sha256": hashlib.sha256(
            cpu.to(dtype=torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }
    return _object_sha256(payload)


def _gradient_checkpointing_observations(transformer: nn.Module) -> tuple[str, ...]:
    """Return enabled checkpoint flags without mistaking capability for state."""

    enabled: list[str] = []
    try:
        aggregate = getattr(transformer, "is_gradient_checkpointing", False)
    except Exception as error:
        raise SAICTypedActionOperatorError(
            f"cannot audit transformer.is_gradient_checkpointing: {error}"
        ) from error
    if callable(aggregate):
        try:
            aggregate = aggregate()
        except Exception as error:
            raise SAICTypedActionOperatorError(
                f"cannot call transformer.is_gradient_checkpointing: {error}"
            ) from error
    if isinstance(aggregate, torch.Tensor):
        if aggregate.numel() != 1:
            raise SAICTypedActionOperatorError(
                "transformer.is_gradient_checkpointing is not scalar"
            )
        aggregate = bool(aggregate.detach().cpu().item())
    if bool(aggregate):
        enabled.append("transformer.is_gradient_checkpointing")

    for module_name, module in transformer.named_modules():
        display = module_name or "<root>"
        for attribute in ("gradient_checkpointing", "_gradient_checkpointing"):
            try:
                observed = getattr(module, attribute, False)
            except Exception as error:
                raise SAICTypedActionOperatorError(
                    f"cannot audit {display}.{attribute}: {error}"
                ) from error
            # A callable checkpoint implementation is not an enabled-state
            # flag.  Bernini/Diffusers use bools on blocks plus the aggregate
            # property above.
            if isinstance(observed, bool) and observed:
                enabled.append(f"{display}.{attribute}")
    return tuple(sorted(set(enabled)))


def _assert_gradient_checkpointing_disabled(transformer: nn.Module) -> None:
    enabled = _gradient_checkpointing_observations(transformer)
    if enabled:
        raise SAICTypedActionOperatorError(
            "gradient checkpointing must remain disabled for routed SAIC "
            f"execution; enabled={list(enabled)[:4]}"
        )


def _query_sequence_parallel_coordinate(group: Any) -> tuple[int, int]:
    """Read the actual group-local coordinate; never accept reported ints."""

    distributed = torch.distributed
    if not distributed.is_available() or not distributed.is_initialized():
        if group is not None:
            raise SAICTypedActionOperatorError(
                "sequence-parallel group supplied before torch.distributed init"
            )
        return 0, 1
    try:
        rank = int(distributed.get_rank(group=group))
        size = int(distributed.get_world_size(group=group))
    except Exception as error:
        raise SAICTypedActionOperatorError(
            f"cannot query actual sequence-parallel coordinate: {error}"
        ) from error
    return rank, size


@dataclass(frozen=True)
class _SAICRouteRuntimeBinding:
    """Factory-only evidence binding one route to observed native runtime state."""

    install_authority: object = field(repr=False, compare=False)
    branch_ordered_descriptor: tuple[tuple[str, str], ...]
    global_target_mask_sha256: str
    local_target_mask_sha256: str
    local_target_mask_bits: tuple[bool, ...] = field(repr=False)
    parallel_rank: int
    parallel_size: int
    actual_sigma_float32_be_hex: str
    pinned_schedule_sha256: str
    descriptor_digest: str

    def public_value(self) -> Mapping[str, Any]:
        return {
            "branch_ordered_descriptor": [
                {"role": role, "source_id_float32_be_hex": source_id}
                for role, source_id in self.branch_ordered_descriptor
            ],
            "global_target_mask_sha256": self.global_target_mask_sha256,
            "local_target_mask_sha256": self.local_target_mask_sha256,
            "parallel_rank": self.parallel_rank,
            "parallel_size": self.parallel_size,
            "actual_sigma_float32_be_hex": self.actual_sigma_float32_be_hex,
            "pinned_schedule_sha256": self.pinned_schedule_sha256,
        }

    def validate(self) -> None:
        value = self.public_value()
        if (
            self.pinned_schedule_sha256 != sigma_strata.SCHEDULE_SHA256
            or _object_sha256(value) != self.descriptor_digest
            or self.parallel_size not in ALLOWED_SP_SIZES
            or not 0 <= self.parallel_rank < self.parallel_size
            or len(self.local_target_mask_bits) <= 0
        ):
            raise SAICTypedActionOperatorError(
                "factory-bound runtime descriptor validation failed"
            )


def _validate_registered_schedule() -> None:
    complete = HIGH_SIGMA_INDICES + MID_SIGMA_INDICES + LOW_SIGMA_INDICES
    if complete != tuple(range(sigma_strata.NUM_INFERENCE_STEPS)):
        raise RuntimeError("SAIC sigma partition is not exact40")
    if any(
        sigma_strata.PINNED_POSITIVE_SIGMAS[index] < 0.55
        for index in HIGH_SIGMA_INDICES
    ):
        raise RuntimeError("SAIC high-sigma indices differ from the pinned threshold")
    if any(
        not 0.25 <= sigma_strata.PINNED_POSITIVE_SIGMAS[index] < 0.55
        for index in MID_SIGMA_INDICES
    ):
        raise RuntimeError("SAIC mid-sigma indices differ from the pinned thresholds")
    if any(
        sigma_strata.PINNED_POSITIVE_SIGMAS[index] >= 0.25
        for index in LOW_SIGMA_INDICES
    ):
        raise RuntimeError("SAIC low-sigma indices differ from the pinned threshold")


_validate_registered_schedule()


def sigma_gate(schedule_index: Any) -> tuple[str, float]:
    """Return the pinned exact40 phase name and scalar gate."""

    if (
        isinstance(schedule_index, bool)
        or not isinstance(schedule_index, int)
        or not 0 <= schedule_index < sigma_strata.NUM_INFERENCE_STEPS
    ):
        raise SAICTypedActionOperatorError(
            "sigma_schedule_index must be an exact integer in [0,39]"
        )
    if schedule_index in HIGH_SIGMA_INDICES:
        return "high", HIGH_SIGMA_WEIGHT
    if schedule_index in MID_SIGMA_INDICES:
        return "mid", MID_SIGMA_WEIGHT
    if schedule_index in LOW_SIGMA_INDICES:
        return "low_base_only", LOW_SIGMA_WEIGHT
    raise SAICTypedActionOperatorError("sigma_schedule_index is not preregistered")


@dataclass(frozen=True)
class SAICArrowCode:
    """An immutable, typed, exact-FP32 initial-to-terminal action arrow."""

    initial_state_type: str
    terminal_state_type: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        initial = _validate_state_type(
            self.initial_state_type, label="initial_state_type"
        )
        terminal = _validate_state_type(
            self.terminal_state_type, label="terminal_state_type"
        )
        if not isinstance(self.values, tuple) or len(self.values) != ARROW_CODE_DIM:
            raise SAICTypedActionOperatorError(
                f"arrow values must be an exact tuple of length {ARROW_CODE_DIM}"
            )
        normalized = tuple(
            _require_exact_float32(value, label=f"arrow.values[{index}]")
            for index, value in enumerate(self.values)
        )
        object.__setattr__(self, "values", normalized)
        zero = all(value == 0.0 for value in normalized)
        if zero != (initial == terminal):
            raise SAICTypedActionOperatorError(
                "zero arrow iff initial_state_type equals terminal_state_type"
            )

    @classmethod
    def quantized(
        cls,
        initial_state_type: str,
        terminal_state_type: str,
        values: Iterable[Any],
    ) -> "SAICArrowCode":
        """Deliberately quantize a planner vector once at the route boundary."""

        try:
            raw = tuple(values)
        except TypeError as error:
            raise SAICTypedActionOperatorError("arrow values must be iterable") from error
        normalized = tuple(
            _quantize_float32(value, label=f"arrow.values[{index}]")
            for index, value in enumerate(raw)
        )
        return cls(initial_state_type, terminal_state_type, normalized)

    @classmethod
    def from_tensor(
        cls,
        initial_state_type: str,
        terminal_state_type: str,
        value: torch.Tensor,
    ) -> "SAICArrowCode":
        if (
            type(value) is not torch.Tensor
            or value.dtype != torch.float32
            or value.ndim != 1
            or int(value.numel()) != ARROW_CODE_DIM
            or value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())
        ):
            raise SAICTypedActionOperatorError(
                "arrow tensor must be detached finite rank-1 FP32 length 32"
            )
        return cls(
            initial_state_type,
            terminal_state_type,
            tuple(float(item) for item in value.detach().cpu()),
        )

    @classmethod
    def between(
        cls,
        initial_state_type: str,
        initial_code: torch.Tensor,
        terminal_state_type: str,
        terminal_code: torch.Tensor,
    ) -> "SAICArrowCode":
        """Create the FP32 arrow ``terminal_code - initial_code``."""

        for label, value in (
            ("initial_code", initial_code),
            ("terminal_code", terminal_code),
        ):
            if (
                type(value) is not torch.Tensor
                or value.dtype != torch.float32
                or value.ndim != 1
                or int(value.numel()) != ARROW_CODE_DIM
                or value.requires_grad
                or value.grad_fn is not None
                or not bool(torch.isfinite(value).all().item())
            ):
                raise SAICTypedActionOperatorError(
                    f"{label} must be detached finite rank-1 FP32 length 32"
                )
        delta = terminal_code.detach().cpu() - initial_code.detach().cpu()
        if not bool(torch.isfinite(delta).all().item()):
            raise SAICTypedActionOperatorError("endpoint subtraction overflowed FP32")
        return cls.from_tensor(initial_state_type, terminal_state_type, delta)

    @classmethod
    def noop(cls, state_type: str) -> "SAICArrowCode":
        return cls(state_type, state_type, (0.0,) * ARROW_CODE_DIM)

    @property
    def is_noop(self) -> bool:
        return self.initial_state_type == self.terminal_state_type

    def sign_reversed(self) -> "SAICArrowCode":
        """Swap endpoint types and negate the code without claiming inversion."""

        values = tuple(
            _quantize_float32(-value, label=f"sign_reversed.values[{index}]")
            for index, value in enumerate(self.values)
        )
        return SAICArrowCode(
            self.terminal_state_type, self.initial_state_type, values
        )

    def tensor(self, *, device: torch.device) -> torch.Tensor:
        return torch.tensor(self.values, dtype=torch.float32, device=device)

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "initial_state_type": self.initial_state_type,
            "terminal_state_type": self.terminal_state_type,
            "dimension": ARROW_CODE_DIM,
            "float32_be_hex": [_float32_be_hex(item) for item in self.values],
            "is_noop": self.is_noop,
        }
        return {**value, "digest": _object_sha256(value)}


@dataclass(frozen=True)
class SAICTypedActionRoute:
    """Native route value; routed execution requires a factory runtime binding."""

    total_tokens: int
    condition_tokens: int
    sequence_parallel_rank: int
    sequence_parallel_size: int
    branch_name: str
    sigma_schedule_index: int
    arrow: SAICArrowCode
    _runtime_binding: Optional[_SAICRouteRuntimeBinding] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        total = _positive_int(self.total_tokens, label="total_tokens")
        if (
            isinstance(self.condition_tokens, bool)
            or not isinstance(self.condition_tokens, int)
            or not 0 <= self.condition_tokens < total
        ):
            raise SAICTypedActionOperatorError(
                "condition_tokens must identify a strict noisy-target suffix"
            )
        if self.branch_name not in NATIVE_BRANCHES:
            raise SAICTypedActionOperatorError("branch_name is not a native visual branch")
        if self.branch_name == "none" and self.condition_tokens != 0:
            raise SAICTypedActionOperatorError(
                "native none branch cannot contain condition rows"
            )
        if self.branch_name != "none" and self.condition_tokens == 0:
            raise SAICTypedActionOperatorError(
                "native conditioned branches must contain condition rows"
            )
        size = _positive_int(
            self.sequence_parallel_size, label="sequence_parallel_size"
        )
        rank = self.sequence_parallel_rank
        if size not in ALLOWED_SP_SIZES:
            raise SAICTypedActionOperatorError(
                "only SP1 tests and native Ulysses SP4 are supported"
            )
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < size:
            raise SAICTypedActionOperatorError("SP rank lies outside its group")
        sigma_gate(self.sigma_schedule_index)
        if type(self.arrow) is not SAICArrowCode:
            raise SAICTypedActionOperatorError("arrow must be an exact SAICArrowCode")

    def _require_runtime_binding(
        self, *, install_authority: Optional[object] = None
    ) -> _SAICRouteRuntimeBinding:
        binding = self._runtime_binding
        if type(binding) is not _SAICRouteRuntimeBinding:
            raise SAICTypedActionOperatorError(
                "routed execution requires a factory-bound runtime descriptor; "
                "direct SAICTypedActionRoute construction is non-executable"
            )
        binding.validate()
        if install_authority is not None and binding.install_authority is not install_authority:
            raise SAICTypedActionOperatorError(
                "runtime route belongs to a different installed SAIC operator"
            )
        if (
            binding.parallel_rank != self.sequence_parallel_rank
            or binding.parallel_size != self.sequence_parallel_size
            or binding.actual_sigma_float32_be_hex
            != sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[
                self.sigma_schedule_index
            ]
            or binding.branch_ordered_descriptor
            != NATIVE_BRANCH_ORDERED_DESCRIPTORS[self.branch_name]
        ):
            raise SAICTypedActionOperatorError(
                "runtime binding and route value differ"
            )
        expected = tuple(
            bool(item)
            for item in self._geometric_local_target_selector(
                device=torch.device("cpu")
            ).tolist()
        )
        expected_global_tensor = self.global_target_selector(
            device=torch.device("cpu")
        )
        expected_local_tensor = torch.tensor(expected, dtype=torch.bool)
        if (
            expected != binding.local_target_mask_bits
            or _bool_mask_sha256(
                expected_global_tensor, label="bound_global_target_mask"
            )
            != binding.global_target_mask_sha256
            or _bool_mask_sha256(
                expected_local_tensor, label="bound_local_target_mask"
            )
            != binding.local_target_mask_sha256
        ):
            raise SAICTypedActionOperatorError(
                "runtime local target mask differs from route geometry"
            )
        return binding

    @property
    def target_tokens(self) -> int:
        return self.total_tokens - self.condition_tokens

    @property
    def local_length(self) -> int:
        return math.ceil(self.total_tokens / self.sequence_parallel_size)

    @property
    def gate_name(self) -> str:
        if self.arrow.is_noop:
            return "noop_base_only"
        return sigma_gate(self.sigma_schedule_index)[0]

    @property
    def gate_weight(self) -> float:
        if self.arrow.is_noop:
            return 0.0
        return sigma_gate(self.sigma_schedule_index)[1]

    @property
    def operator_active(self) -> bool:
        return not self.arrow.is_noop and self.gate_weight > 0.0

    def global_target_selector(self, *, device: torch.device) -> torch.Tensor:
        return torch.cat(
            (
                torch.zeros(self.condition_tokens, dtype=torch.bool, device=device),
                torch.ones(self.target_tokens, dtype=torch.bool, device=device),
            )
        )

    def _geometric_local_target_selector(
        self, *, device: torch.device
    ) -> torch.Tensor:
        selector = self.global_target_selector(device=device)
        padded_length = self.local_length * self.sequence_parallel_size
        if padded_length > self.total_tokens:
            selector = torch.cat(
                (
                    selector,
                    torch.zeros(
                        padded_length - self.total_tokens,
                        dtype=torch.bool,
                        device=device,
                    ),
                )
            )
        start = self.sequence_parallel_rank * self.local_length
        return selector[start : start + self.local_length].contiguous()

    def local_target_selector(self, *, device: torch.device) -> torch.Tensor:
        binding = self._runtime_binding
        if type(binding) is _SAICRouteRuntimeBinding:
            binding.validate()
            return torch.tensor(
                binding.local_target_mask_bits, dtype=torch.bool, device=device
            )
        return self._geometric_local_target_selector(device=device)

    def receipt(self) -> Mapping[str, Any]:
        binding = self._require_runtime_binding()
        value = {
            "branch_name": self.branch_name,
            "total_tokens": self.total_tokens,
            "condition_tokens": self.condition_tokens,
            "target_tokens": self.target_tokens,
            "sequence_parallel_rank": self.sequence_parallel_rank,
            "sequence_parallel_size": self.sequence_parallel_size,
            "padding_policy": "append_false_then_contiguous_rank_chunk",
            "sigma_schedule_index": self.sigma_schedule_index,
            "sigma_float32_be_hex": sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[
                self.sigma_schedule_index
            ],
            "sigma_gate": self.gate_name,
            "sigma_gate_weight": self.gate_weight,
            "operator_active": self.operator_active,
            "arrow": dict(self.arrow.receipt()),
            "runtime_descriptor": dict(binding.public_value()),
            "runtime_descriptor_digest": binding.descriptor_digest,
            "route_factory_bound": True,
        }
        return {**value, "digest": _object_sha256(value)}


_ACTIVE_ROUTE: ContextVar[Optional[SAICTypedActionRoute]] = ContextVar(
    "bernini_saic_typed_action_route", default=None
)


def active_route() -> Optional[SAICTypedActionRoute]:
    return _ACTIVE_ROUTE.get()


@contextmanager
def activate_route(route: SAICTypedActionRoute) -> Iterator[None]:
    if type(route) is not SAICTypedActionRoute:
        raise SAICTypedActionOperatorError("route must be an exact SAICTypedActionRoute")
    route._require_runtime_binding()
    if active_route() is not None:
        raise SAICTypedActionOperatorError("nested SAIC routes are forbidden")
    token: Token[Optional[SAICTypedActionRoute]] = _ACTIVE_ROUTE.set(route)
    try:
        yield
    finally:
        _ACTIVE_ROUTE.reset(token)


class SAICTargetRowTypedActionOperator(nn.Module):
    """Sigma-partitioned rank-8 typed residual around one frozen Q/O."""

    def __init__(
        self,
        base: nn.Module,
        *,
        projection: str,
        transformer: nn.Module,
        install_authority: object,
    ):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise SAICTypedActionOperatorError(f"{projection} base must be nn.Linear")
        if projection not in {"to_q", "to_out.0"}:
            raise SAICTypedActionOperatorError("only cross-attention Q/O may be wrapped")
        if any(parameter.requires_grad for parameter in base.parameters()):
            raise SAICTypedActionOperatorError("wrapped base projection must be frozen")
        self.base = base
        self.projection = projection
        self.rank = ACTION_OPERATOR_RANK
        object.__setattr__(self, "_transformer_ref", weakref.ref(transformer))
        object.__setattr__(self, "_install_authority", install_authority)
        for stratum in ("high", "mid"):
            state_down = nn.Linear(
                base.in_features, self.rank, bias=False, dtype=torch.float32
            )
            arrow_gate = nn.Linear(
                ARROW_CODE_DIM, self.rank, bias=False, dtype=torch.float32
            )
            output_up = nn.Linear(
                self.rank, base.out_features, bias=False, dtype=torch.float32
            )
            nn.init.kaiming_uniform_(state_down.weight, a=math.sqrt(5.0))
            nn.init.kaiming_uniform_(arrow_gate.weight, a=math.sqrt(5.0))
            nn.init.zeros_(output_up.weight)
            setattr(self, f"state_down_{stratum}", state_down)
            setattr(self, f"arrow_gate_{stratum}", arrow_gate)
            setattr(self, f"output_up_{stratum}", output_up)

    # Compatibility views deliberately name the high-sigma partition.  They
    # are not additional registered modules or checkpoint aliases.
    @property
    def state_down(self) -> nn.Linear:
        return self.state_down_high

    @property
    def arrow_gate(self) -> nn.Linear:
        return self.arrow_gate_high

    @property
    def output_up(self) -> nn.Linear:
        return self.output_up_high

    def _assert_routed_runtime(self, route: SAICTypedActionRoute) -> None:
        route._require_runtime_binding(install_authority=self._install_authority)
        transformer = self._transformer_ref()
        if transformer is None:
            raise SAICTypedActionOperatorError(
                "installed transformer no longer exists"
            )
        _assert_gradient_checkpointing_disabled(transformer)

    @property
    def weight(self) -> Any:
        return self.base.weight

    @property
    def bias(self) -> Any:
        return self.base.bias

    @staticmethod
    def _selector(
        hidden_states: torch.Tensor, route: SAICTypedActionRoute
    ) -> torch.Tensor:
        if hidden_states.ndim != 3 or int(hidden_states.shape[0]) != 1:
            raise SAICTypedActionOperatorError(
                "native SAIC operator expects hidden states [1,N,D]"
            )
        selector = route.local_target_selector(device=hidden_states.device)
        if int(hidden_states.shape[1]) != int(selector.numel()):
            raise SAICTypedActionOperatorError(
                "local hidden sequence differs from native append-pad/SP slice"
            )
        return selector

    def _pre_up(
        self,
        selected_hidden: torch.Tensor,
        arrow: SAICArrowCode,
        sigma_stratum: str,
    ) -> torch.Tensor:
        if sigma_stratum not in {"high", "mid"}:
            raise SAICTypedActionOperatorError(
                "active typed operator requires high or mid sigma stratum"
            )
        state_down = getattr(self, f"state_down_{sigma_stratum}")
        arrow_gate = getattr(self, f"arrow_gate_{sigma_stratum}")
        with torch.autocast(device_type=selected_hidden.device.type, enabled=False):
            state_feature = F.silu(state_down(selected_hidden.float()))
            arrow_tensor = arrow.tensor(device=selected_hidden.device)
            arrow_feature = torch.tanh(arrow_gate(arrow_tensor)).view(
                1, 1, self.rank
            )
            return state_feature * arrow_feature

    def selected_pre_up(
        self, hidden_states: torch.Tensor, route: SAICTypedActionRoute
    ) -> torch.Tensor:
        """Expose the typed bilinear feature for losses and oddness audits."""

        if type(route) is not SAICTypedActionRoute:
            raise SAICTypedActionOperatorError("route must be exact for pre-up audit")
        self._assert_routed_runtime(route)
        selector = self._selector(hidden_states, route)
        if not route.operator_active:
            return torch.zeros(
                (1, int(selector.sum().item()), self.rank),
                dtype=torch.float32,
                device=hidden_states.device,
            )
        return self._pre_up(
            hidden_states[:, selector, :], route.arrow, route.gate_name
        )

    def _selected_delta(
        self,
        hidden_states: torch.Tensor,
        selector: torch.Tensor,
        route: SAICTypedActionRoute,
    ) -> torch.Tensor:
        # Empty selected tensors are intentionally evaluated on source-only SP
        # shards so every rank retains the same distributed autograd topology.
        selected = hidden_states[:, selector, :]
        stratum = route.gate_name
        if stratum not in {"high", "mid"}:
            raise SAICTypedActionOperatorError(
                "active route did not select a trainable sigma partition"
            )
        dormant = "mid" if stratum == "high" else "high"
        output_up = getattr(self, f"output_up_{stratum}")
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            delta = output_up(self._pre_up(selected, route.arrow, stratum))
            # The distributed reducer requires a gradient tensor for every
            # registered parameter.  Dormant-stratum zero links keep that
            # topology closed without allowing dormant values to affect the
            # forward result.  A non-finite dormant parameter still fails the
            # active forward instead of being silently hidden.
            dormant_zero = sum(
                getattr(self, f"{name}_{dormant}").weight.sum()
                for name in ("state_down", "arrow_gate", "output_up")
            ) * 0.0
            delta = delta + dormant_zero
            delta = delta * route.gate_weight
        return delta.to(hidden_states.dtype)

    def adapter_delta(self, hidden_states: torch.Tensor) -> torch.Tensor:
        route = active_route()
        result = torch.zeros(
            (*hidden_states.shape[:-1], self.base.out_features),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if route is None:
            return result
        self._assert_routed_runtime(route)
        selector = self._selector(hidden_states, route)
        if not route.operator_active:
            return result
        result[:, selector, :] = self._selected_delta(hidden_states, selector, route)
        return result

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        route = active_route()
        if route is not None:
            # Guard before even the frozen projection is evaluated so a
            # checkpoint-enabled routed forward has no partial side effect.
            self._assert_routed_runtime(route)
        base = self.base(hidden_states)
        # Direct return is semantically important: no-op and low sigma must not
        # merely evaluate a possibly non-finite residual and multiply by zero.
        if route is None:
            return base
        selector = self._selector(hidden_states, route)
        if not route.operator_active:
            return base
        result = base.clone()
        result[:, selector, :] = base[:, selector, :] + self._selected_delta(
            hidden_states, selector, route
        ).to(base.dtype)
        return result


def trainable_state_digest(state: Mapping[str, torch.Tensor]) -> str:
    """Digest a closed detached CPU-FP32 operator state."""

    if not isinstance(state, Mapping):
        raise SAICTypedActionOperatorError("operator state must be a mapping")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or name.encode("ascii", "strict").decode("ascii") != name:
            raise SAICTypedActionOperatorError("operator state names must be ASCII")
        if (
            type(value) is not torch.Tensor
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or value.layout != torch.strided
            or value.requires_grad
            or value.grad_fn is not None
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all().item())
        ):
            raise SAICTypedActionOperatorError(
                f"operator state {name} must be detached finite contiguous CPU FP32"
            )
        digest.update(name.encode("ascii"))
        digest.update(_canonical_json(list(value.shape)))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@dataclass
class SAICTypedActionOperatorHandle:
    transformer: nn.Module
    q_wrappers: tuple[tuple[int, SAICTargetRowTypedActionOperator], ...]
    o_wrappers: tuple[tuple[int, SAICTargetRowTypedActionOperator], ...]
    original_q: tuple[tuple[int, nn.Module], ...]
    original_o: tuple[tuple[int, nn.Module], ...]
    original_patch_embedding_id: int
    protected_attention_ids: tuple[tuple[int, ...], ...]
    route_binding_authority: object = field(repr=False)
    restored: bool = False

    def trainable_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        if self.restored:
            raise SAICTypedActionOperatorError("SAIC operator has been restored")
        result: list[tuple[str, nn.Parameter]] = []
        for index, wrapper in self.q_wrappers:
            for stratum in ("high", "mid"):
                result.extend(
                    (
                        (f"blocks.{index}.attn2.to_q.state_down_{stratum}.weight", getattr(wrapper, f"state_down_{stratum}").weight),
                        (f"blocks.{index}.attn2.to_q.arrow_gate_{stratum}.weight", getattr(wrapper, f"arrow_gate_{stratum}").weight),
                        (f"blocks.{index}.attn2.to_q.output_up_{stratum}.weight", getattr(wrapper, f"output_up_{stratum}").weight),
                    )
                )
        for index, wrapper in self.o_wrappers:
            for stratum in ("high", "mid"):
                result.extend(
                    (
                        (f"blocks.{index}.attn2.to_out.0.state_down_{stratum}.weight", getattr(wrapper, f"state_down_{stratum}").weight),
                        (f"blocks.{index}.attn2.to_out.0.arrow_gate_{stratum}.weight", getattr(wrapper, f"arrow_gate_{stratum}").weight),
                        (f"blocks.{index}.attn2.to_out.0.output_up_{stratum}.weight", getattr(wrapper, f"output_up_{stratum}").weight),
                    )
                )
        if len({id(parameter) for _, parameter in result}) != len(result):
            raise SAICTypedActionOperatorError("SAIC trainable parameter alias detected")
        if any(not parameter.requires_grad for _, parameter in result):
            raise SAICTypedActionOperatorError("SAIC parameter is unexpectedly frozen")
        return tuple(result)

    def trainable_named_parameters_for_sigma(
        self, sigma_stratum: str
    ) -> tuple[tuple[str, nn.Parameter], ...]:
        """Return one complete, parameter-disjoint optimizer partition."""

        if sigma_stratum not in {"high", "mid"}:
            raise SAICTypedActionOperatorError(
                "optimizer sigma stratum must be high or mid"
            )
        suffix = f"_{sigma_stratum}.weight"
        selected = tuple(
            (name, parameter)
            for name, parameter in self.trainable_named_parameters()
            if name.endswith(suffix)
        )
        if (
            len(selected) != len(ACTION_BLOCK_INDICES) * 2 * 3
            or len({id(parameter) for _, parameter in selected}) != len(selected)
        ):
            raise SAICTypedActionOperatorError(
                "sigma optimizer parameter partition differs"
            )
        return selected

    def base_parameters_frozen(self) -> bool:
        trainable_ids = {
            id(parameter) for _, parameter in self.trainable_named_parameters()
        }
        observed = {
            id(parameter)
            for parameter in self.transformer.parameters()
            if parameter.requires_grad
        }
        return observed == trainable_ids

    def protected_attention_untouched(self) -> bool:
        blocks = tuple(getattr(self.transformer, "blocks", ()))
        if len(blocks) != TOTAL_BLOCKS_1P3B:
            return False
        return _capture_protected_attention_ids(blocks) == self.protected_attention_ids

    @contextmanager
    def route(self, route: SAICTypedActionRoute) -> Iterator[None]:
        if self.restored:
            raise SAICTypedActionOperatorError("cannot route a restored SAIC operator")
        route._require_runtime_binding(
            install_authority=self.route_binding_authority
        )
        _assert_gradient_checkpointing_disabled(self.transformer)
        with activate_route(route):
            yield

    def bind_runtime_route(
        self,
        *,
        native_branch: Any,
        actual_local_target_mask: torch.Tensor,
        actual_sigma: torch.Tensor,
        arrow: SAICArrowCode,
        sequence_parallel_group: Any = None,
    ) -> SAICTypedActionRoute:
        """Bind one route to observed native branch/SP/sigma runtime state."""

        return bind_saic_runtime_route(
            handle=self,
            native_branch=native_branch,
            actual_local_target_mask=actual_local_target_mask,
            actual_sigma=actual_sigma,
            arrow=arrow,
            sequence_parallel_group=sequence_parallel_group,
        )

    def state_dict_for_save(self) -> Mapping[str, torch.Tensor]:
        state = {
            name: parameter.detach().float().cpu().contiguous().clone()
            for name, parameter in self.trainable_named_parameters()
        }
        trainable_state_digest(state)
        return state

    def trainable_state_digest(self) -> str:
        return trainable_state_digest(self.state_dict_for_save())

    def load_trainable_state_dict(
        self, state: Mapping[str, torch.Tensor]
    ) -> Mapping[str, Any]:
        if self.restored:
            raise SAICTypedActionOperatorError("cannot load a restored SAIC operator")
        if not isinstance(state, Mapping):
            raise SAICTypedActionOperatorError("SAIC state must be a mapping")
        expected = dict(self.trainable_named_parameters())
        actual_keys = set(state)
        if actual_keys != set(expected):
            missing = sorted(set(expected) - actual_keys)
            unexpected = sorted(actual_keys - set(expected))
            raise SAICTypedActionOperatorError(
                "SAIC state key closure differs: "
                f"missing={missing[:2]} unexpected={unexpected[:2]}"
            )
        normalized: dict[str, torch.Tensor] = {}
        for name in sorted(expected):
            value = state[name]
            parameter = expected[name]
            if (
                type(value) is not torch.Tensor
                or value.dtype != torch.float32
                or value.device.type != "cpu"
                or value.layout != torch.strided
                or value.requires_grad
                or value.grad_fn is not None
                or not value.is_contiguous()
                or tuple(value.shape) != tuple(parameter.shape)
                or not bool(torch.isfinite(value).all().item())
            ):
                raise SAICTypedActionOperatorError(
                    f"SAIC state {name} must be detached finite contiguous CPU FP32 with exact shape"
                )
            normalized[name] = value
        digest = trainable_state_digest(normalized)
        with torch.no_grad():
            for name, parameter in expected.items():
                parameter.copy_(normalized[name].to(device=parameter.device))
        value = {
            "schema_version": SCHEMA_VERSION,
            "state_key_count": len(normalized),
            "state_key_sha256": _object_sha256(sorted(normalized)),
            "state_tensor_sha256": digest,
            "closed_exact_key_set": True,
        }
        return {**value, "digest": _object_sha256(value)}

    def save_checkpoint(self, path: os.PathLike[str] | str) -> Mapping[str, Any]:
        destination = Path(path)
        if not destination.parent.is_dir() or destination.is_dir():
            raise SAICTypedActionOperatorError(
                "checkpoint parent must exist and destination must not be a directory"
            )
        state = dict(self.state_dict_for_save())
        state_digest = trainable_state_digest(state)
        payload = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "operator_schema_version": SCHEMA_VERSION,
            "state_tensor_sha256": state_digest,
            "state": state,
        }
        temporary_name: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                torch.save(payload, temporary)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, destination)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
        value = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "path": str(destination),
            "state_key_count": len(state),
            "state_tensor_sha256": state_digest,
        }
        return {**value, "digest": _object_sha256(value)}

    def load_checkpoint(self, path: os.PathLike[str] | str) -> Mapping[str, Any]:
        source = Path(path)
        if not source.is_file():
            raise SAICTypedActionOperatorError("SAIC checkpoint is not a regular file")
        try:
            payload = torch.load(source, map_location="cpu", weights_only=True)
        except Exception as error:
            raise SAICTypedActionOperatorError(
                f"failed to read weights-only SAIC checkpoint: {error}"
            ) from error
        if not isinstance(payload, Mapping) or set(payload) != {
            "checkpoint_schema_version",
            "operator_schema_version",
            "state_tensor_sha256",
            "state",
        }:
            raise SAICTypedActionOperatorError("SAIC checkpoint envelope differs")
        if (
            payload["checkpoint_schema_version"] != CHECKPOINT_SCHEMA_VERSION
            or payload["operator_schema_version"] != SCHEMA_VERSION
            or not isinstance(payload["state_tensor_sha256"], str)
            or not isinstance(payload["state"], Mapping)
        ):
            raise SAICTypedActionOperatorError("SAIC checkpoint schema differs")
        state_digest = trainable_state_digest(payload["state"])
        if state_digest != payload["state_tensor_sha256"]:
            raise SAICTypedActionOperatorError("SAIC checkpoint tensor digest differs")
        load_receipt = dict(self.load_trainable_state_dict(payload["state"]))
        value = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "path": str(source),
            "state_tensor_sha256": state_digest,
            "load_receipt_digest": load_receipt["digest"],
        }
        return {**value, "digest": _object_sha256(value)}

    def receipt(self) -> Mapping[str, Any]:
        patch = getattr(self.transformer, "patch_embedding", None)
        trainable = self.trainable_named_parameters()
        value = {
            "schema_version": SCHEMA_VERSION,
            "block_indices": list(ACTION_BLOCK_INDICES),
            "projections": ["attn2.to_q", "attn2.to_out.0"],
            "operator": "up_s(silu(state_down_s(h))*tanh(arrow_gate_s(r)))",
            "rank": ACTION_OPERATOR_RANK,
            "arrow_code_dimension": ARROW_CODE_DIM,
            "bias": False,
            "output_up_zero_initialized_at_install": True,
            "sigma_parameter_partition": {
                "strata": ["high", "mid"],
                "complete_parameter_disjoint_heads": True,
                "shared_trainable_parameters": False,
                "optimizer_must_step_only_active_stratum": True,
            },
            "target_suffix_only": True,
            "condition_and_padding_rows_exact_base": True,
            "sign_reversed_pre_up_same_hidden_is_odd": True,
            "routed_function_undo_claim": False,
            "noop_direct_base_return": True,
            "low_sigma_direct_base_return": True,
            "native_branches": list(NATIVE_BRANCHES),
            "sp_selector": "append_false_then_contiguous_rank_chunk",
            "exact40_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
            "sigma_gate_indices": {
                "high_weight_1": list(HIGH_SIGMA_INDICES),
                "mid_weight_0.5": list(MID_SIGMA_INDICES),
                "low_base_only_weight_0": list(LOW_SIGMA_INDICES),
            },
            "patch_embedding_untouched": id(patch) == self.original_patch_embedding_id,
            "self_attention_and_cross_attention_kv_untouched": self.protected_attention_untouched(),
            "key_value_trainable": False,
            "self_attention_trainable": False,
            "late_blocks_trainable": False,
            "base_parameters_frozen": self.base_parameters_frozen(),
            "trainable_state_closed": True,
            "trainable_state_key_sha256": _object_sha256(
                sorted(name for name, _ in trainable)
            ),
            "trainable": [
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                }
                for name, parameter in trainable
            ],
            "proposal_or_target_video_consumed": False,
            "mask_flow_pose_track_trajectory_consumed": False,
            "gradient_checkpointing_required_disabled": True,
            "routes_require_factory_runtime_binding": True,
            "semantic_action_claim": False,
        }
        return {**value, "digest": _object_sha256(value)}

    def restore(self) -> None:
        if self.restored or active_route() is not None:
            raise SAICTypedActionOperatorError("SAIC operator cannot be restored now")
        blocks = tuple(getattr(self.transformer, "blocks", ()))
        if len(blocks) != TOTAL_BLOCKS_1P3B:
            raise SAICTypedActionOperatorError("transformer block count changed")
        if id(getattr(self.transformer, "patch_embedding", None)) != self.original_patch_embedding_id:
            raise SAICTypedActionOperatorError(
                "native patch embedding changed while SAIC was active"
            )
        if not self.protected_attention_untouched():
            raise SAICTypedActionOperatorError(
                "self-attention or cross-attention K/V changed while SAIC was active"
            )
        for index, original in self.original_q:
            blocks[index].attn2.to_q = original
        for index, original in self.original_o:
            blocks[index].attn2.to_out[0] = original
        self.restored = True


def bind_saic_runtime_route(
    *,
    handle: SAICTypedActionOperatorHandle,
    native_branch: Any,
    actual_local_target_mask: torch.Tensor,
    actual_sigma: torch.Tensor,
    arrow: SAICArrowCode,
    sequence_parallel_group: Any = None,
) -> SAICTypedActionRoute:
    """Create an executable route from observed Bernini runtime descriptors.

    ``total_tokens``, ``condition_tokens``, the native branch name/order, the
    local source/target partition, SP rank/size, and the exact40 index are all
    derived here.  They are intentionally absent as caller-reported scalar
    arguments.  A directly constructed :class:`SAICTypedActionRoute` remains
    useful for pure geometry validation but cannot enter routed execution.
    """

    if type(handle) is not SAICTypedActionOperatorHandle or handle.restored:
        raise SAICTypedActionOperatorError(
            "runtime route factory requires one live exact SAIC handle"
        )
    if type(arrow) is not SAICArrowCode:
        raise SAICTypedActionOperatorError("runtime route arrow type differs")
    _assert_gradient_checkpointing_disabled(handle.transformer)

    try:
        branch_name = native_branch.name
        latents = native_branch.latents
        global_target_mask = native_branch.target_mask
        total_tokens = native_branch.total_tokens
        condition_tokens = native_branch.condition_tokens
        concat_order = native_branch.concat_order
        source_ids = native_branch.source_ids
    except AttributeError as error:
        raise SAICTypedActionOperatorError(
            "native branch runtime descriptor is incomplete"
        ) from error
    if branch_name not in NATIVE_BRANCHES:
        raise SAICTypedActionOperatorError(
            "native branch runtime descriptor name differs"
        )
    total_tokens = _positive_int(total_tokens, label="native_branch.total_tokens")
    if (
        isinstance(condition_tokens, bool)
        or not isinstance(condition_tokens, int)
        or not 0 <= condition_tokens < total_tokens
    ):
        raise SAICTypedActionOperatorError(
            "native branch condition geometry differs"
        )
    if (branch_name == "none") != (condition_tokens == 0):
        raise SAICTypedActionOperatorError(
            "native branch name and condition prefix disagree"
        )
    if (
        type(latents) is not torch.Tensor
        or latents.ndim != 3
        or int(latents.shape[0]) != 1
        or int(latents.shape[1]) != total_tokens
    ):
        raise SAICTypedActionOperatorError(
            "native branch latent sequence does not bind total_tokens"
        )
    _bool_mask_sha256(global_target_mask, label="native_branch.target_mask")
    if int(global_target_mask.numel()) != total_tokens:
        raise SAICTypedActionOperatorError(
            "native branch target mask length differs"
        )
    expected_global_mask = torch.cat(
        (
            torch.zeros(condition_tokens, dtype=torch.bool),
            torch.ones(total_tokens - condition_tokens, dtype=torch.bool),
        )
    )
    observed_global_mask = global_target_mask.detach().to(device="cpu").contiguous()
    if not torch.equal(observed_global_mask, expected_global_mask):
        raise SAICTypedActionOperatorError(
            "native branch target mask is not the actual target suffix"
        )

    if not isinstance(concat_order, tuple) or not isinstance(source_ids, tuple):
        raise SAICTypedActionOperatorError(
            "native branch ordered descriptor must use immutable tuples"
        )
    if len(concat_order) != len(source_ids) or any(
        not isinstance(role, str) for role in concat_order
    ):
        raise SAICTypedActionOperatorError(
            "native branch concat/source-id order closure differs"
        )
    observed_order = tuple(
        (
            role,
            _float32_be_hex(
                _require_exact_float32(
                    source_id,
                    label=f"native_branch.source_ids[{index}]",
                )
            ),
        )
        for index, (role, source_id) in enumerate(zip(concat_order, source_ids))
    )
    if observed_order != NATIVE_BRANCH_ORDERED_DESCRIPTORS[branch_name]:
        raise SAICTypedActionOperatorError(
            "native branch ordered concat/source-id descriptor differs"
        )

    sp_rank, sp_size = _query_sequence_parallel_coordinate(
        sequence_parallel_group
    )
    if sp_size not in ALLOWED_SP_SIZES or not 0 <= sp_rank < sp_size:
        raise SAICTypedActionOperatorError(
            "actual sequence-parallel coordinate is outside SP1/SP4"
        )
    local_length = math.ceil(total_tokens / sp_size)
    padded = expected_global_mask
    if local_length * sp_size > total_tokens:
        padded = torch.cat(
            (
                padded,
                torch.zeros(
                    local_length * sp_size - total_tokens, dtype=torch.bool
                ),
            )
        )
    expected_local_mask = padded[
        sp_rank * local_length : (sp_rank + 1) * local_length
    ].contiguous()
    _bool_mask_sha256(
        actual_local_target_mask, label="actual_local_target_mask"
    )
    observed_local_mask = (
        actual_local_target_mask.detach().to(device="cpu").contiguous()
    )
    if not torch.equal(observed_local_mask, expected_local_mask):
        raise SAICTypedActionOperatorError(
            "actual local target/source mask differs from native branch and SP slice"
        )

    if (
        type(actual_sigma) is not torch.Tensor
        or actual_sigma.dtype != torch.float32
        or actual_sigma.numel() != 1
        or actual_sigma.requires_grad
        or actual_sigma.grad_fn is not None
        or not bool(torch.isfinite(actual_sigma).all().item())
    ):
        raise SAICTypedActionOperatorError(
            "actual sigma must be a detached finite device-local FP32 scalar tensor"
        )
    sigma_hex = _float32_be_hex(float(actual_sigma.detach().cpu().item()))
    try:
        sigma_schedule_index = (
            sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX.index(sigma_hex)
        )
    except ValueError as error:
        raise SAICTypedActionOperatorError(
            "actual sigma is outside the pinned Bernini exact40 schedule"
        ) from error

    public_binding = {
        "branch_ordered_descriptor": [
            {"role": role, "source_id_float32_be_hex": source_id}
            for role, source_id in observed_order
        ],
        "global_target_mask_sha256": _bool_mask_sha256(
            observed_global_mask, label="observed_global_target_mask"
        ),
        "local_target_mask_sha256": _bool_mask_sha256(
            observed_local_mask, label="observed_local_target_mask"
        ),
        "parallel_rank": sp_rank,
        "parallel_size": sp_size,
        "actual_sigma_float32_be_hex": sigma_hex,
        "pinned_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
    }
    binding = _SAICRouteRuntimeBinding(
        install_authority=handle.route_binding_authority,
        branch_ordered_descriptor=observed_order,
        global_target_mask_sha256=public_binding["global_target_mask_sha256"],
        local_target_mask_sha256=public_binding["local_target_mask_sha256"],
        local_target_mask_bits=tuple(bool(item) for item in observed_local_mask.tolist()),
        parallel_rank=sp_rank,
        parallel_size=sp_size,
        actual_sigma_float32_be_hex=sigma_hex,
        pinned_schedule_sha256=sigma_strata.SCHEDULE_SHA256,
        descriptor_digest=_object_sha256(public_binding),
    )
    binding.validate()
    route = SAICTypedActionRoute(
        total_tokens=total_tokens,
        condition_tokens=condition_tokens,
        sequence_parallel_rank=sp_rank,
        sequence_parallel_size=sp_size,
        branch_name=branch_name,
        sigma_schedule_index=sigma_schedule_index,
        arrow=arrow,
        _runtime_binding=binding,
    )
    route._require_runtime_binding(
        install_authority=handle.route_binding_authority
    )
    return route


def _capture_protected_attention_ids(
    blocks: tuple[nn.Module, ...]
) -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = []
    for index, block in enumerate(blocks):
        attn1 = getattr(block, "attn1", None)
        attn2 = getattr(block, "attn2", None)
        self_out = getattr(attn1, "to_out", None)
        cross_out = getattr(attn2, "to_out", None)
        if (
            self_out is None
            or len(self_out) < 1
            or cross_out is None
            or len(cross_out) < 2
            or getattr(attn1, "to_q", None) is None
            or getattr(attn1, "to_k", None) is None
            or getattr(attn1, "to_v", None) is None
            or getattr(attn2, "to_k", None) is None
            or getattr(attn2, "to_v", None) is None
        ):
            raise SAICTypedActionOperatorError(
                f"block {index} protected attention structure differs"
            )
        result.append(
            (
                id(attn1),
                id(attn1.to_q),
                id(attn1.to_k),
                id(attn1.to_v),
                id(self_out[0]),
                id(attn2.to_k),
                id(attn2.to_v),
                id(cross_out[1]),
            )
        )
    return tuple(result)


def install_saic_typed_action_operator(
    transformer: nn.Module,
) -> SAICTypedActionOperatorHandle:
    """Install function-preserving rank-8 SAIC wrappers on Bernini Q/O."""

    if not isinstance(transformer, nn.Module):
        raise SAICTypedActionOperatorError("transformer must be nn.Module")
    if any(parameter.requires_grad for parameter in transformer.parameters()):
        raise SAICTypedActionOperatorError(
            "freeze the complete Bernini base before SAIC installation"
        )
    _assert_gradient_checkpointing_disabled(transformer)
    blocks = tuple(getattr(transformer, "blocks", ()))
    patch = getattr(transformer, "patch_embedding", None)
    if (
        len(blocks) != TOTAL_BLOCKS_1P3B
        or not isinstance(patch, nn.Conv3d)
        or not callable(getattr(transformer, "patch_vae_latent", None))
    ):
        raise SAICTypedActionOperatorError(
            "Bernini-R 1.3B native transformer structure differs"
        )

    hidden = int(patch.out_channels)
    protected_attention_ids = _capture_protected_attention_ids(blocks)
    original_q: list[tuple[int, nn.Module]] = []
    original_o: list[tuple[int, nn.Module]] = []
    for index in ACTION_BLOCK_INDICES:
        attention = getattr(blocks[index], "attn2", None)
        query = getattr(attention, "to_q", None)
        output = getattr(attention, "to_out", None)
        if (
            not isinstance(query, nn.Linear)
            or not isinstance(output, nn.ModuleList)
            or len(output) != 2
            or not isinstance(output[0], nn.Linear)
            or query.in_features != hidden
            or query.out_features != hidden
            or output[0].in_features != hidden
            or output[0].out_features != hidden
        ):
            raise SAICTypedActionOperatorError(
                f"block {index} native cross-attention Q/O differs"
            )
        original_q.append((index, query))
        original_o.append((index, output[0]))

    device = patch.weight.device
    q_wrappers: list[tuple[int, SAICTargetRowTypedActionOperator]] = []
    o_wrappers: list[tuple[int, SAICTargetRowTypedActionOperator]] = []
    route_binding_authority = object()
    try:
        for (index, query), (_, output) in zip(original_q, original_o):
            q_wrapper = SAICTargetRowTypedActionOperator(
                query,
                projection="to_q",
                transformer=transformer,
                install_authority=route_binding_authority,
            ).to(device=device)
            o_wrapper = SAICTargetRowTypedActionOperator(
                output,
                projection="to_out.0",
                transformer=transformer,
                install_authority=route_binding_authority,
            ).to(device=device)
            blocks[index].attn2.to_q = q_wrapper
            blocks[index].attn2.to_out[0] = o_wrapper
            q_wrappers.append((index, q_wrapper))
            o_wrappers.append((index, o_wrapper))
    except Exception:
        for index, original in original_q:
            blocks[index].attn2.to_q = original
        for index, original in original_o:
            blocks[index].attn2.to_out[0] = original
        raise

    handle = SAICTypedActionOperatorHandle(
        transformer=transformer,
        q_wrappers=tuple(q_wrappers),
        o_wrappers=tuple(o_wrappers),
        original_q=tuple(original_q),
        original_o=tuple(original_o),
        original_patch_embedding_id=id(patch),
        protected_attention_ids=protected_attention_ids,
        route_binding_authority=route_binding_authority,
    )
    receipt = handle.receipt()
    if (
        not handle.base_parameters_frozen()
        or receipt["patch_embedding_untouched"] is not True
        or receipt["self_attention_and_cross_attention_kv_untouched"] is not True
    ):
        handle.restore()
        raise SAICTypedActionOperatorError("SAIC operator scope closure failed")
    return handle


__all__ = [
    "ACTION_BLOCK_INDICES",
    "ACTION_OPERATOR_RANK",
    "ARROW_CODE_DIM",
    "CHECKPOINT_SCHEMA_VERSION",
    "HIGH_SIGMA_INDICES",
    "LOW_SIGMA_INDICES",
    "MID_SIGMA_INDICES",
    "NATIVE_BRANCH_ORDERED_DESCRIPTORS",
    "SAICArrowCode",
    "SAICTargetRowTypedActionOperator",
    "SAICTypedActionOperatorError",
    "SAICTypedActionOperatorHandle",
    "SAICTypedActionRoute",
    "SCHEMA_VERSION",
    "activate_route",
    "active_route",
    "bind_saic_runtime_route",
    "install_saic_typed_action_operator",
    "sigma_gate",
    "trainable_state_digest",
]
