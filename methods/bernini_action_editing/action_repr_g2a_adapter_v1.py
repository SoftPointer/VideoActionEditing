#!/usr/bin/env python3
"""G2a-safe action-representation residuals for a frozen Bernini transformer.

This module is the small trainable route required by the 2026-08-24 action
representation design.  It deliberately does not implement an optimizer or
accept target pixels/latents.  A route contains only detached, source-aligned
flow and optional action-minus-noop middle residuals.

The native transformer is frozen before four post-block residuals are
installed.  Their output projections are positive-byte zero.  At optimizer
step zero an autograd primitive returns the native tensor bit-for-bit while
still exposing the output projection to the first backward pass.  Route-off
and the explicit zero route are permanent hard bypasses.

The patch handle exposes an exact three-role parameter allowlist, a deep
frozen-base byte audit, a create-only G2a receipt, and per-block student
residual traces shaped ``[B,P,N,D]`` for
``action_representation_joint_objective_v1``.  It is an implementation/safety
primitive, not evidence that G0/G1 or Stage-B training is authorized.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import types
from typing import Any, Iterator, Mapping, MutableMapping, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F


SCHEMA_VERSION = "bernini-action-repr-g2a-adapter-v1"
RECEIPT_SCHEMA_VERSION = "bernini-action-repr-g2a-receipt-v1"
STEP0_AUDIT_SCHEMA_VERSION = "bernini-action-repr-g2a-step0-audit-v1"
STATE_SCHEMA_VERSION = "bernini-action-repr-g2a-state-v1"
MODULE_NAME = "action_repr_g2a"

EXPECTED_BLOCK_COUNT = 30
DEFAULT_BLOCK_INDICES = (6, 12, 18, 24)
DEFAULT_HIDDEN_WIDTH = 1536
DEFAULT_FLOW_WIDTH = 12
DEFAULT_BOTTLENECK_WIDTH = 256

ROUTE_KINDS = (
    "route_off",
    "correct",
    "zero",
    "temporal_shuffle",
    "reverse",
    "incomplete",
    "wrong_action",
)
ACTIVE_ROUTE_KINDS = frozenset(
    {"correct", "temporal_shuffle", "reverse", "incomplete", "wrong_action"}
)
STEP0_REQUIRED_ROUTES = (
    "correct",
    "zero",
    "temporal_shuffle",
    "reverse",
    "incomplete",
    "wrong_action",
)
REPRESENTATION_ORIGINS = frozenset(
    {
        "real_target_frozen_extractor",
        "selfgen_decoded_video_reencode",
        "selfgen_native_trajectory",
        "counterfactual_control",
    }
)
MIDDLE_VALUE_KINDS = frozenset(
    {"post_attention_residual", "predicted_velocity_residual"}
)
TRAINABLE_ROLES = (
    "zero_initialized_motion_adapter",
    "middle_projector",
    "source_copy_adapter_if_preservation",
)


class G2AAdapterError(RuntimeError):
    """Raised before an ambiguous route, parameter, or receipt is accepted."""


def _fail(message: str) -> None:
    raise G2AAdapterError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise G2AAdapterError(
            f"G2a value is not canonical finite ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256")
    return value


def _exact_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{label} must be a nonnegative integer")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    result = _exact_nonnegative_int(value, label=label)
    if result == 0:
        _fail(f"{label} must be positive")
    return result


def _owned_cpu_bytes(value: torch.Tensor) -> bytes:
    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        _fail("tensor byte audit requires a materialized tensor")
    owned = value.detach().to(device="cpu").clone(
        memory_format=torch.contiguous_format
    ).contiguous()
    return owned.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")


def tensor_sha256(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        _fail("tensor digest requires a materialized tensor")
    header = canonical_json_bytes(
        {"dtype": str(value.dtype), "shape": list(map(int, value.shape))}
    )
    return hashlib.sha256(header + b"\0" + _owned_cpu_bytes(value)).hexdigest()


def tensor_bits_equal(left: Any, right: Any) -> bool:
    if (
        not isinstance(left, torch.Tensor)
        or not isinstance(right, torch.Tensor)
        or left.shape != right.shape
        or left.dtype != right.dtype
        or left.device != right.device
    ):
        return False
    return _owned_cpu_bytes(left) == _owned_cpu_bytes(right)


def _finite_detached_tensor(
    value: Any, *, label: str, ndim: int, width: Optional[int] = None
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != ndim
        or not value.is_floating_point()
        or value.device.type == "meta"
        or any(int(size) <= 0 for size in value.shape)
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        _fail(f"{label} must be a finite floating rank-{ndim} tensor")
    if value.requires_grad or value.grad_fn is not None:
        _fail(f"{label} must be a detached representation cache")
    if width is not None and int(value.shape[-1]) != int(width):
        _fail(f"{label} feature width differs")
    return value


def _byte_positive_zero(value: torch.Tensor) -> bool:
    return bool(value.is_floating_point()) and not any(_owned_cpu_bytes(value))


def _numeric_zero(value: torch.Tensor) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.is_floating_point()
        and bool(torch.isfinite(value.detach()).all().item())
        and int(torch.count_nonzero(value.detach()).item()) == 0
    )


@dataclass(frozen=True)
class TokenLayout:
    """Global native visual-token layout before append-pad/SP slicing."""

    total_tokens: int
    source_tokens: int
    phase_count: int

    def validate(self) -> None:
        total = _positive_int(self.total_tokens, label="layout.total_tokens")
        source = _exact_nonnegative_int(
            self.source_tokens, label="layout.source_tokens"
        )
        phases = _positive_int(self.phase_count, label="layout.phase_count")
        target = total - source
        if target <= 0 or target % phases:
            _fail("layout target suffix must factor exactly into P phases")

    @property
    def target_tokens(self) -> int:
        self.validate()
        return int(self.total_tokens) - int(self.source_tokens)

    @property
    def tokens_per_phase(self) -> int:
        return self.target_tokens // int(self.phase_count)

    def receipt(self) -> Mapping[str, Any]:
        self.validate()
        return {
            "total_tokens": int(self.total_tokens),
            "source_tokens": int(self.source_tokens),
            "target_tokens": self.target_tokens,
            "phase_count": int(self.phase_count),
            "tokens_per_phase": self.tokens_per_phase,
        }


@dataclass
class ResidualTraceCollector:
    """Collect differentiable per-block target residuals in ``[B,P,N,D]``.

    ``feature_projection`` is an optional fixed, detached matrix with shape
    ``[native_hidden_width, teacher_width]``.  When supplied, the native
    differentiable residual is projected before the target suffix is reshaped
    into BPND.  The collector owns a clone so later mutation of the caller's
    tensor cannot change the training feature map.  Projection provenance is
    deliberately left to the runner that authenticates the cached teacher.

    Full global sequences need no collective.  A sequence-parallel caller must
    opt into differentiable gathering explicitly; otherwise a local/global
    mismatch fails instead of silently assigning the wrong phase or token.
    """

    expected_blocks: tuple[int, ...]
    gather_sequence_parallel: bool = False
    feature_projection: Optional[torch.Tensor] = field(
        default=None, repr=False, compare=False
    )
    _values: MutableMapping[int, torch.Tensor] = field(default_factory=dict)
    _activities: MutableMapping[int, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        indices = tuple(self.expected_blocks)
        if not indices or indices != tuple(sorted(set(indices))):
            _fail("trace expected_blocks must be increasing and unique")
        if not isinstance(self.gather_sequence_parallel, bool):
            _fail("trace gather_sequence_parallel must be boolean")
        if self.feature_projection is not None:
            projection = _finite_detached_tensor(
                self.feature_projection,
                label="trace.feature_projection",
                ndim=2,
            )
            # Own the immutable-at-construction feature map rather than retain
            # an alias to runner memory.  This clone remains detached and is
            # never registered as a trainable parameter.
            self.feature_projection = projection.detach().clone(
                memory_format=torch.contiguous_format
            ).contiguous()

    def _global(self, value: torch.Tensor, *, total_tokens: int) -> torch.Tensor:
        if int(value.shape[1]) == int(total_tokens):
            return value
        if not self.gather_sequence_parallel:
            _fail(
                "rank-local residual trace requires explicit differentiable SP gather"
            )
        try:
            import torch.distributed as dist
            from torch.distributed.nn.functional import all_gather
        except ImportError as error:  # pragma: no cover - runtime dependent
            raise G2AAdapterError(
                "differentiable SP trace gather is unavailable"
            ) from error
        if not dist.is_initialized() or int(dist.get_world_size()) <= 1:
            _fail("rank-local residual trace lacks an initialized SP group")
        gathered = all_gather(value.contiguous())
        if not isinstance(gathered, (tuple, list)):
            _fail("differentiable SP all_gather returned an unsupported value")
        result = torch.cat(tuple(gathered), dim=1)
        if int(result.shape[1]) < int(total_tokens):
            _fail("gathered residual is shorter than the declared global layout")
        return result[:, : int(total_tokens)].contiguous()

    def record(
        self,
        *,
        block_index: int,
        residual: torch.Tensor,
        activity: torch.Tensor,
        layout: TokenLayout,
    ) -> None:
        if block_index not in self.expected_blocks or block_index in self._values:
            _fail("residual trace block closure differs or was recorded twice")
        layout.validate()
        if (
            not isinstance(residual, torch.Tensor)
            or residual.ndim != 3
            or not residual.is_floating_point()
            or not isinstance(activity, torch.Tensor)
            or activity.dtype != torch.bool
            or activity.ndim != 3
            or int(activity.shape[2]) != 1
        ):
            _fail("residual trace tensors must be [B,L,D] and bool [B,L,1]")
        global_residual = self._global(
            residual, total_tokens=int(layout.total_tokens)
        )
        if self.feature_projection is not None:
            projection = _finite_detached_tensor(
                self.feature_projection,
                label="trace.feature_projection",
                ndim=2,
            )
            if int(global_residual.shape[-1]) != int(projection.shape[0]):
                _fail(
                    "trace feature projection input width differs from "
                    "the native residual"
                )
            # Casting the residual preserves its autograd edge to the adapter
            # output gate; only the fixed projection remains detached.
            global_residual = torch.matmul(
                global_residual.float(),
                projection.to(
                    device=global_residual.device,
                    dtype=torch.float32,
                ),
            )
            if not bool(torch.isfinite(global_residual.detach()).all().item()):
                _fail("trace feature projection produced a non-finite residual")
        # Activity is detached routing metadata.  Use the native Bernini
        # padding/slicing helper only for adapter execution; trace gathering
        # starts from its already-global cache authority.
        if int(activity.shape[1]) != int(layout.total_tokens):
            _fail("trace activity must retain the global representation layout")
        if int(global_residual.shape[0]) != int(activity.shape[0]):
            _fail("trace residual/activity batch geometry differs")
        target = global_residual[:, int(layout.source_tokens) :]
        target_activity = activity[:, int(layout.source_tokens) :]
        batch, _, width = map(int, target.shape)
        value = target.reshape(
            batch,
            int(layout.phase_count),
            int(layout.tokens_per_phase),
            width,
        )
        mask = target_activity.reshape(
            batch,
            int(layout.phase_count),
            int(layout.tokens_per_phase),
            1,
        )
        self._values[int(block_index)] = value
        self._activities[int(block_index)] = mask

    def for_block(self, block_index: int) -> torch.Tensor:
        if block_index not in self._values:
            _fail(f"residual trace lacks block {block_index}")
        return self._values[block_index]

    def activity_for_block(self, block_index: int) -> torch.Tensor:
        if block_index not in self._activities:
            _fail(f"residual trace lacks activity for block {block_index}")
        return self._activities[block_index]

    def require_complete(self) -> Mapping[int, torch.Tensor]:
        if set(self._values) != set(self.expected_blocks):
            _fail("residual trace did not close every installed block")
        return {index: self._values[index] for index in self.expected_blocks}


@dataclass(frozen=True)
class ActionRepresentationRoute:
    """One authenticated, detached action route for a native forward."""

    kind: str
    optimizer_step: int
    layout: TokenLayout
    flow: Optional[torch.Tensor] = field(default=None, repr=False, compare=False)
    activity: Optional[torch.Tensor] = field(
        default=None, repr=False, compare=False
    )
    middle_by_block: Mapping[int, torch.Tensor] = field(
        default_factory=dict, repr=False, compare=False
    )
    representation_origin: Optional[str] = None
    representation_cache_sha256: Optional[str] = None
    middle_value_kind: Optional[str] = None
    matched_noise_timestep_rotary: Optional[bool] = None
    trace: Optional[ResidualTraceCollector] = field(
        default=None, repr=False, compare=False
    )

    def validate_basic(self) -> None:
        if self.kind not in ROUTE_KINDS:
            _fail("action representation route kind differs")
        _exact_nonnegative_int(self.optimizer_step, label="optimizer_step")
        self.layout.validate()
        if self.trace is not None and not isinstance(
            self.trace, ResidualTraceCollector
        ):
            _fail("route trace collector has the wrong type")
        if self.kind in {"route_off", "zero"}:
            if (
                self.flow is not None
                or self.activity is not None
                or bool(self.middle_by_block)
                or self.representation_origin is not None
                or self.representation_cache_sha256 is not None
                or self.middle_value_kind is not None
                or self.matched_noise_timestep_rotary is not None
            ):
                _fail("route-off/zero must not carry an action representation")
            return
        flow = _finite_detached_tensor(
            self.flow,
            label="route.flow",
            ndim=3,
            width=DEFAULT_FLOW_WIDTH,
        )
        if (
            not isinstance(self.activity, torch.Tensor)
            or self.activity.dtype != torch.bool
            or tuple(self.activity.shape)
            != (int(flow.shape[0]), int(flow.shape[1]), 1)
            or self.activity.device != flow.device
        ):
            _fail("route.activity must be global bool [B,L,1]")
        if int(flow.shape[1]) != int(self.layout.total_tokens):
            _fail("route flow token count differs from its global layout")
        if not bool(self.activity.any().item()):
            _fail("active action route has no active token")
        if bool(self.activity[:, : int(self.layout.source_tokens)].any().item()):
            _fail("source/reference rows cannot be action-active")
        if self.representation_origin not in REPRESENTATION_ORIGINS:
            _fail("representation origin differs from the detached ABI")
        _sha256(
            self.representation_cache_sha256,
            label="representation_cache_sha256",
        )
        if not isinstance(self.middle_by_block, Mapping):
            _fail("middle_by_block must be a mapping")
        for raw_index, value in self.middle_by_block.items():
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                _fail("middle residual block key must be an integer")
            middle = _finite_detached_tensor(
                value, label=f"middle_by_block.{raw_index}", ndim=3
            )
            if (
                tuple(middle.shape[:2]) != tuple(flow.shape[:2])
                or middle.device != flow.device
            ):
                _fail("middle residual layout/device differs from flow")
        if self.middle_by_block:
            if self.middle_value_kind not in MIDDLE_VALUE_KINDS:
                _fail("middle residual value kind differs")
            if self.matched_noise_timestep_rotary is not True:
                _fail("middle residual lacks matched noise/timestep/rotary")
        elif (
            self.middle_value_kind is not None
            or self.matched_noise_timestep_rotary is not None
        ):
            _fail("middle provenance was supplied without middle residuals")

    def receipt(self) -> Mapping[str, Any]:
        self.validate_basic()
        return {
            "kind": self.kind,
            "optimizer_step": int(self.optimizer_step),
            "layout": self.layout.receipt(),
            "active_representation": self.kind in ACTIVE_ROUTE_KINDS,
            "representation_origin": self.representation_origin,
            "representation_cache_sha256": self.representation_cache_sha256,
            "middle_blocks": sorted(map(int, self.middle_by_block)),
            "middle_value_kind": self.middle_value_kind,
            "matched_noise_timestep_rotary": self.matched_noise_timestep_rotary,
            "target_rgb_or_latent_present": False,
            "raw_qk_or_absolute_hidden_present": False,
        }


_CURRENT_ROUTE: contextvars.ContextVar[Optional[ActionRepresentationRoute]] = (
    contextvars.ContextVar("bernini_action_repr_g2a_route", default=None)
)


def current_action_representation_route() -> Optional[ActionRepresentationRoute]:
    return _CURRENT_ROUTE.get()


@contextlib.contextmanager
def action_representation_route(
    route: ActionRepresentationRoute,
) -> Iterator[ActionRepresentationRoute]:
    if not isinstance(route, ActionRepresentationRoute):
        _fail("G2a context received the wrong route type")
    route.validate_basic()
    if _CURRENT_ROUTE.get() is not None:
        _fail("nested G2a routes are forbidden")
    token = _CURRENT_ROUTE.set(route)
    try:
        yield route
    finally:
        _CURRENT_ROUTE.reset(token)


class _ExactZeroResidualAdd(torch.autograd.Function):
    """Bit-exact forward identity with the derivative of ``native + delta``."""

    @staticmethod
    def forward(ctx: Any, native: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        del ctx
        if not _numeric_zero(delta):
            _fail("step-zero residual is not exact numeric zero")
        # Returning the native value directly avoids BF16/FP16 add-rounding and
        # preserves the sign bit of negative zero.  backward() still sends the
        # upstream gradient to the zero output projection.
        return native.clone(memory_format=torch.preserve_format)

    @staticmethod
    def backward(
        ctx: Any, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del ctx
        return gradient, gradient


class MiddleProjector(nn.Module):
    def __init__(self, *, middle_width: int, bottleneck_width: int) -> None:
        super().__init__()
        self.middle_width = _positive_int(middle_width, label="middle_width")
        self.bottleneck_width = _positive_int(
            bottleneck_width, label="bottleneck_width"
        )
        self.norm = nn.LayerNorm(self.middle_width, elementwise_affine=False)
        self.projection = nn.Linear(
            self.middle_width, self.bottleneck_width, bias=False, dtype=torch.float32
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if (
            value.ndim != 3
            or int(value.shape[-1]) != self.middle_width
            or not bool(torch.isfinite(value.detach()).all().item())
        ):
            _fail("middle projector input geometry/finiteness differs")
        return self.projection(self.norm(value.float()))


class ZeroInitializedMotionAdapter(nn.Module):
    def __init__(
        self,
        *,
        hidden_width: int,
        flow_width: int,
        bottleneck_width: int,
    ) -> None:
        super().__init__()
        self.hidden_width = _positive_int(hidden_width, label="hidden_width")
        self.flow_width = _positive_int(flow_width, label="flow_width")
        self.bottleneck_width = _positive_int(
            bottleneck_width, label="bottleneck_width"
        )
        self.hidden_norm = nn.LayerNorm(
            self.hidden_width, elementwise_affine=False
        )
        self.hidden_down = nn.Linear(
            self.hidden_width, self.bottleneck_width, bias=False, dtype=torch.float32
        )
        self.flow_in = nn.Linear(
            self.flow_width, self.bottleneck_width, bias=False, dtype=torch.float32
        )
        self.output = nn.Linear(
            self.bottleneck_width, self.hidden_width, bias=False, dtype=torch.float32
        )
        nn.init.zeros_(self.output.weight)

    def delta(
        self,
        hidden: torch.Tensor,
        flow: torch.Tensor,
        activity: torch.Tensor,
        middle: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (
            hidden.ndim != 3
            or int(hidden.shape[-1]) != self.hidden_width
            or flow.ndim != 3
            or tuple(flow.shape[:2]) != tuple(hidden.shape[:2])
            or int(flow.shape[-1]) != self.flow_width
            or activity.dtype != torch.bool
            or tuple(activity.shape) != (*hidden.shape[:2], 1)
            or (middle is not None and tuple(middle.shape[:2]) != tuple(hidden.shape[:2]))
        ):
            _fail("motion adapter tensor geometry differs")
        fused = self.hidden_down(self.hidden_norm(hidden.float())) + self.flow_in(
            flow.float()
        )
        if middle is not None:
            if int(middle.shape[-1]) != self.bottleneck_width:
                _fail("projected middle width differs from motion bottleneck")
            fused = fused + middle.float()
        delta = self.output(F.silu(fused))
        return torch.where(activity, delta, torch.zeros_like(delta)).to(hidden.dtype)

    def output_is_byte_zero(self) -> bool:
        return _byte_positive_zero(self.output.weight)


class SourceCopyAdapter(nn.Module):
    """Small source-owned carrier branch; no target/anchor value is copied."""

    def __init__(self, *, hidden_width: int, bottleneck_width: int) -> None:
        super().__init__()
        self.hidden_width = _positive_int(hidden_width, label="hidden_width")
        self.bottleneck_width = _positive_int(
            bottleneck_width, label="bottleneck_width"
        )
        self.target_norm = nn.LayerNorm(
            self.hidden_width, elementwise_affine=False
        )
        self.source_norm = nn.LayerNorm(
            self.hidden_width, elementwise_affine=False
        )
        self.target_down = nn.Linear(
            self.hidden_width, self.bottleneck_width, bias=False, dtype=torch.float32
        )
        self.source_down = nn.Linear(
            self.hidden_width, self.bottleneck_width, bias=False, dtype=torch.float32
        )
        self.output = nn.Linear(
            self.bottleneck_width, self.hidden_width, bias=False, dtype=torch.float32
        )
        nn.init.zeros_(self.output.weight)

    def delta(
        self,
        hidden: torch.Tensor,
        source_carrier: torch.Tensor,
        activity: torch.Tensor,
    ) -> torch.Tensor:
        if (
            hidden.ndim != 3
            or tuple(source_carrier.shape) != tuple(hidden.shape)
            or activity.dtype != torch.bool
            or tuple(activity.shape) != (*hidden.shape[:2], 1)
        ):
            _fail("source-copy adapter tensor geometry differs")
        fused = self.target_down(self.target_norm(hidden.float())) + self.source_down(
            self.source_norm(source_carrier.float())
        )
        delta = self.output(F.silu(fused))
        return torch.where(activity, delta, torch.zeros_like(delta)).to(hidden.dtype)

    def output_is_byte_zero(self) -> bool:
        return _byte_positive_zero(self.output.weight)


class _G2ABlockBundle(nn.Module):
    def __init__(
        self,
        *,
        hidden_width: int,
        flow_width: int,
        bottleneck_width: int,
        middle_width: Optional[int],
        source_copy: bool,
    ) -> None:
        super().__init__()
        self.motion_adapter = ZeroInitializedMotionAdapter(
            hidden_width=hidden_width,
            flow_width=flow_width,
            bottleneck_width=bottleneck_width,
        )
        self.middle_projector = (
            MiddleProjector(
                middle_width=middle_width, bottleneck_width=bottleneck_width
            )
            if middle_width is not None
            else None
        )
        self.source_copy_adapter = (
            SourceCopyAdapter(
                hidden_width=hidden_width, bottleneck_width=bottleneck_width
            )
            if source_copy
            else None
        )

    def output_gates_are_byte_zero(self) -> bool:
        return self.motion_adapter.output_is_byte_zero() and (
            self.source_copy_adapter is None
            or self.source_copy_adapter.output_is_byte_zero()
        )


def _resolve_transformer(model: Any) -> nn.Module:
    queue = [model]
    seen: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        blocks = getattr(candidate, "blocks", None)
        if isinstance(blocks, (nn.ModuleList, list, tuple)):
            if len(blocks) != EXPECTED_BLOCK_COUNT:
                _fail("Bernini transformer must expose exactly 30 blocks")
            return candidate
        getter = getattr(candidate, "get_base_model", None)
        if callable(getter):
            try:
                queue.append(getter())
            except Exception:
                pass
        for name in ("diff_dec", "transformer", "base_model", "model", "module"):
            nested = getattr(candidate, name, None)
            if nested is not None and nested is not candidate:
                queue.append(nested)
    _fail("could not resolve the Bernini 30-block transformer")


def _named_parameters_all(module: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    try:
        return tuple(module.named_parameters(remove_duplicate=False))
    except TypeError:  # pragma: no cover - old torch fallback
        return tuple(module.named_parameters())


def _base_manifest(
    named: Sequence[tuple[str, nn.Parameter]],
) -> tuple[tuple[Mapping[str, Any], ...], str]:
    rows = tuple(
        {
            "name": name,
            "shape": list(map(int, parameter.shape)),
            "dtype": str(parameter.dtype),
            "sha256": tensor_sha256(parameter),
        }
        for name, parameter in named
    )
    return rows, object_sha256(rows)


def _local_representation_tensor(
    value: torch.Tensor, hidden: torch.Tensor, *, label: str
) -> torch.Tensor:
    value = value.to(device=hidden.device)
    if int(value.shape[1]) == int(hidden.shape[1]):
        return value
    try:
        from bernini.parallel import (
            padding_tensor_for_seqeunce_parallel,
            slice_input_tensor,
        )
    except ImportError as error:
        raise G2AAdapterError(
            f"global {label} requires Bernini sequence-parallel helpers"
        ) from error
    local = slice_input_tensor(
        padding_tensor_for_seqeunce_parallel(value, dim=1), dim=1
    )
    if tuple(local.shape[:2]) != tuple(hidden.shape[:2]):
        _fail(f"rank-local {label} differs from hidden states")
    return local


def _global_hidden_detached(
    hidden: torch.Tensor, *, global_tokens: int
) -> torch.Tensor:
    if int(hidden.shape[1]) == int(global_tokens):
        return hidden.detach()
    try:
        import torch.distributed as dist
    except ImportError as error:  # pragma: no cover - runtime dependent
        raise G2AAdapterError("source-copy SP gather requires torch.distributed") from error
    if not dist.is_initialized() or int(dist.get_world_size()) <= 1:
        _fail("rank-local source-copy hidden lacks an initialized SP group")
    gathered = [torch.empty_like(hidden) for _ in range(int(dist.get_world_size()))]
    dist.all_gather(gathered, hidden.detach().contiguous())
    result = torch.cat(gathered, dim=1)
    if int(result.shape[1]) < int(global_tokens):
        _fail("gathered source-copy hidden is shorter than global tokens")
    return result[:, : int(global_tokens)].contiguous()


def _source_carrier(
    hidden: torch.Tensor, route: ActionRepresentationRoute
) -> torch.Tensor:
    layout = route.layout
    if int(layout.source_tokens) != int(layout.target_tokens):
        _fail("source-copy requires equal source and target token extents")
    global_hidden = _global_hidden_detached(
        hidden, global_tokens=int(layout.total_tokens)
    )
    source = global_hidden[:, : int(layout.source_tokens)]
    global_carrier = torch.cat((torch.zeros_like(source), source), dim=1)
    return _local_representation_tensor(
        global_carrier, hidden, label="source-copy carrier"
    ).to(dtype=hidden.dtype)


@dataclass
class G2APatchHandle:
    transformer: nn.Module
    block_indices: tuple[int, ...]
    bundles: tuple[_G2ABlockBundle, ...]
    original_forwards: tuple[Any, ...] = field(repr=False)
    hidden_width: int = DEFAULT_HIDDEN_WIDTH
    flow_width: int = DEFAULT_FLOW_WIDTH
    bottleneck_width: int = DEFAULT_BOTTLENECK_WIDTH
    middle_width: Optional[int] = None
    source_copy_enabled: bool = False
    base_parameter_names: tuple[str, ...] = field(default_factory=tuple)
    base_parameter_ids: tuple[int, ...] = field(default_factory=tuple, repr=False)
    base_manifest_at_install: tuple[Mapping[str, Any], ...] = field(
        default_factory=tuple, repr=False
    )
    base_digest_at_install: str = ""
    restored: bool = False

    def _require_live(self) -> None:
        if self.restored:
            _fail("G2a patch has already been restored")

    def parameter_allowlist(
        self,
    ) -> Mapping[str, tuple[tuple[str, nn.Parameter], ...]]:
        self._require_live()
        result: dict[str, list[tuple[str, nn.Parameter]]] = {
            role: [] for role in TRAINABLE_ROLES
        }
        for index, bundle in zip(self.block_indices, self.bundles):
            prefix = f"blocks.{index}.{MODULE_NAME}"
            for name, parameter in bundle.motion_adapter.named_parameters():
                result[TRAINABLE_ROLES[0]].append(
                    (f"{prefix}.motion_adapter.{name}", parameter)
                )
            if bundle.middle_projector is not None:
                for name, parameter in bundle.middle_projector.named_parameters():
                    result[TRAINABLE_ROLES[1]].append(
                        (f"{prefix}.middle_projector.{name}", parameter)
                    )
            if bundle.source_copy_adapter is not None:
                for name, parameter in bundle.source_copy_adapter.named_parameters():
                    result[TRAINABLE_ROLES[2]].append(
                        (f"{prefix}.source_copy_adapter.{name}", parameter)
                    )
        flat = [row for role in TRAINABLE_ROLES for row in result[role]]
        if not result[TRAINABLE_ROLES[0]]:
            _fail("motion adapter allowlist is empty")
        if bool(result[TRAINABLE_ROLES[1]]) != (self.middle_width is not None):
            _fail("middle-projector allowlist closure differs")
        if bool(result[TRAINABLE_ROLES[2]]) != bool(self.source_copy_enabled):
            _fail("source-copy allowlist closure differs")
        if len({name for name, _ in flat}) != len(flat) or len(
            {id(parameter) for _, parameter in flat}
        ) != len(flat):
            _fail("G2a allowlist contains aliases or duplicate names")
        return {role: tuple(result[role]) for role in TRAINABLE_ROLES}

    def trainable_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        allowlist = self.parameter_allowlist()
        return tuple(row for role in TRAINABLE_ROLES for row in allowlist[role])

    def output_gates_are_byte_zero(self) -> bool:
        self._require_live()
        return all(bundle.output_gates_are_byte_zero() for bundle in self.bundles)

    def _current_base_named(self) -> tuple[tuple[str, nn.Parameter], ...]:
        allowed_ids = {id(value) for _, value in self.trainable_named_parameters()}
        return tuple(
            (name, value)
            for name, value in _named_parameters_all(self.transformer)
            if id(value) not in allowed_ids
        )

    def audit_parameters(self, *, deep_base_bytes: bool = True) -> Mapping[str, Any]:
        self._require_live()
        allowlist = self.parameter_allowlist()
        allowed = tuple(
            row for role in TRAINABLE_ROLES for row in allowlist[role]
        )
        allowed_ids = {id(parameter) for _, parameter in allowed}
        current_all = _named_parameters_all(self.transformer)
        current_ids = [id(parameter) for _, parameter in current_all]
        # Duplicate IDs mean a parameter can escape by a second unallowlisted
        # path; reject rather than trusting named_parameters deduplication.
        if len(current_ids) != len(set(current_ids)):
            _fail("transformer parameter aliases make the allowlist ambiguous")
        observed_trainable = {
            id(parameter) for _, parameter in current_all if parameter.requires_grad
        }
        if observed_trainable != allowed_ids:
            _fail("trainable parameter set escaped or undershot the G2a allowlist")
        current_base = tuple(
            (name, parameter)
            for name, parameter in current_all
            if id(parameter) not in allowed_ids
        )
        if (
            tuple(name for name, _ in current_base) != self.base_parameter_names
            or tuple(id(parameter) for _, parameter in current_base)
            != self.base_parameter_ids
            or any(parameter.requires_grad for _, parameter in current_base)
        ):
            _fail("frozen base parameter identity/name/requires_grad closure differs")
        current_digest: Optional[str] = None
        if deep_base_bytes:
            manifest, current_digest = _base_manifest(current_base)
            if (
                manifest != self.base_manifest_at_install
                or current_digest != self.base_digest_at_install
            ):
                _fail("frozen base parameter bytes changed after G2a installation")
        roles = {
            role: {
                "parameter_count": sum(
                    int(parameter.numel()) for _, parameter in allowlist[role]
                ),
                "tensor_count": len(allowlist[role]),
                "names": [name for name, _ in allowlist[role]],
            }
            for role in TRAINABLE_ROLES
        }
        return {
            "trainable_roles_exact": list(TRAINABLE_ROLES),
            "roles": roles,
            "trainable_tensor_count": len(allowed),
            "trainable_parameter_count": sum(
                int(parameter.numel()) for _, parameter in allowed
            ),
            "no_parameter_aliases": True,
            "observed_trainable_equals_allowlist": True,
            "base_parameter_tensor_count": len(current_base),
            "base_requires_grad_false": True,
            "deep_base_byte_audit": bool(deep_base_bytes),
            "base_digest_at_install": self.base_digest_at_install,
            "base_digest_current": current_digest,
            "base_bytes_unchanged": bool(deep_base_bytes),
        }

    def state_dict_cpu(self) -> Mapping[str, torch.Tensor]:
        self._require_live()
        return {
            name: parameter.detach().float().cpu().contiguous()
            for name, parameter in self.trainable_named_parameters()
        }

    def load_state_dict_strict(self, state: Mapping[str, torch.Tensor]) -> None:
        self._require_live()
        expected = dict(self.trainable_named_parameters())
        if not isinstance(state, Mapping) or set(state) != set(expected):
            _fail("G2a state-key closure differs")
        with torch.no_grad():
            for name, parameter in expected.items():
                value = state[name]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != parameter.shape
                    or not value.is_floating_point()
                    or not bool(torch.isfinite(value).all().item())
                ):
                    _fail(f"G2a state tensor differs: {name}")
                parameter.copy_(value.to(parameter.device, parameter.dtype))

    def architecture_receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": SCHEMA_VERSION,
            "insertion": "post_native_block_residual",
            "block_indices": list(self.block_indices),
            "hidden_width": int(self.hidden_width),
            "flow_width": int(self.flow_width),
            "bottleneck_width": int(self.bottleneck_width),
            "middle_width": self.middle_width,
            "middle_projector_enabled": self.middle_width is not None,
            "source_copy_adapter_enabled": bool(self.source_copy_enabled),
            "source_copy_reads": "detached_source_hidden_prefix_only",
            "route_kinds": list(ROUTE_KINDS),
            "step0_required_routes": list(STEP0_REQUIRED_ROUTES),
            "step0_addition": "bit_exact_forward_autograd_identity_to_delta",
            "route_off_hard_bypass": True,
            "zero_route_hard_bypass": True,
            "target_rgb_vae_clean_latent_accepted": False,
            "target_absolute_hidden_value_raw_qk_accepted": False,
            "graph_route_enabled": False,
            "trace_abi": "per_block_student_residual_BPND",
        }
        return {**value, "architecture_digest": object_sha256(value)}

    def build_g2a_receipt(
        self,
        *,
        native_output: torch.Tensor,
        routed_outputs: Mapping[str, torch.Tensor],
        matched_input_sha256: str,
        forward_scope: str,
    ) -> Mapping[str, Any]:
        self._require_live()
        matched = _sha256(matched_input_sha256, label="matched_input_sha256")
        if (
            not isinstance(forward_scope, str)
            or not forward_scope
            or not forward_scope.isascii()
        ):
            _fail("forward_scope must be non-empty ASCII")
        if set(routed_outputs) != set(STEP0_REQUIRED_ROUTES):
            _fail("G2a step-zero audit lacks the exact six required routes")
        if not self.output_gates_are_byte_zero():
            _fail("G2a receipt cannot be issued after zero-init output changed")
        if not isinstance(native_output, torch.Tensor):
            _fail("native step-zero output must be a tensor")
        output_rows: dict[str, Any] = {}
        native_digest = tensor_sha256(native_output)
        for kind in STEP0_REQUIRED_ROUTES:
            value = routed_outputs[kind]
            if not tensor_bits_equal(native_output, value):
                _fail(f"step-zero route {kind} is not exact native no-op")
            output_rows[kind] = {
                "tensor_sha256": tensor_sha256(value),
                "exact_native_bits": True,
            }
        architecture = self.architecture_receipt()
        parameter_audit = self.audit_parameters(deep_base_bytes=True)
        step0 = {
            "schema_version": STEP0_AUDIT_SCHEMA_VERSION,
            "optimizer_step": 0,
            "matched_input_sha256": matched,
            "forward_scope": forward_scope,
            "native_tensor_sha256": native_digest,
            "required_routes": list(STEP0_REQUIRED_ROUTES),
            "route_outputs": output_rows,
            "all_routes_exact_native_bits": True,
        }
        step0["audit_digest"] = object_sha256(step0)
        receipt: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "gate": "G2a_zero_init_noop",
            "passed": True,
            "optimizer_created": False,
            "optimizer_step": 0,
            "optimizer_authorized_by_this_receipt": False,
            "stage_b_training_started": False,
            "G0_asserted_by_this_receipt": False,
            "G1_asserted_by_this_receipt": False,
            "architecture": architecture,
            "parameter_audit": parameter_audit,
            "step0_noop_audit": step0,
            "information_firewall": {
                "detached_representation_cache_only": True,
                "target_rgb_to_adapter": False,
                "target_vae_or_clean_latent_to_adapter": False,
                "target_absolute_hidden_value_or_raw_qk_to_adapter": False,
                "source_copy_reads_source_hidden_only": bool(
                    self.source_copy_enabled
                ),
            },
            "claim_scope": "implementation_safety_gate_only_not_method_success",
        }
        receipt["receipt_digest"] = object_sha256(receipt)
        validate_g2a_receipt(receipt)
        return receipt

    def restore(self) -> None:
        self._require_live()
        if current_action_representation_route() is not None:
            _fail("cannot restore G2a patch inside an active route")
        for index, bundle, original in zip(
            self.block_indices, self.bundles, self.original_forwards
        ):
            block = self.transformer.blocks[index]
            if getattr(block, MODULE_NAME, None) is not bundle:
                _fail("G2a block module changed behind its patch handle")
            block.forward = original
            delattr(block, MODULE_NAME)
        self.restored = True


def _validate_route_for_handle(
    route: ActionRepresentationRoute, handle: G2APatchHandle
) -> None:
    route.validate_basic()
    if route.trace is not None and tuple(route.trace.expected_blocks) != tuple(
        handle.block_indices
    ):
        _fail("route trace block closure differs from installed G2a blocks")
    if route.kind not in ACTIVE_ROUTE_KINDS:
        return
    assert route.flow is not None and route.activity is not None
    if int(route.flow.shape[-1]) != int(handle.flow_width):
        _fail("route flow width differs from installed G2a adapter")
    observed_middle = set(map(int, route.middle_by_block))
    expected_middle = set(handle.block_indices) if handle.middle_width is not None else set()
    if observed_middle != expected_middle:
        _fail("route middle residual block closure differs from installation")
    if handle.middle_width is not None:
        for index in handle.block_indices:
            if int(route.middle_by_block[index].shape[-1]) != int(handle.middle_width):
                _fail(f"route middle width differs at block {index}")
    if handle.source_copy_enabled and int(route.layout.source_tokens) <= 0:
        _fail("source-copy installation requires source-owned prefix tokens")


def _record_trace(
    *,
    block_index: int,
    route: ActionRepresentationRoute,
    native: torch.Tensor,
    result: torch.Tensor,
) -> None:
    if route.trace is None:
        return
    if route.activity is None:
        activity = torch.zeros(
            (int(native.shape[0]), int(route.layout.total_tokens), 1),
            dtype=torch.bool,
            device=native.device,
        )
    else:
        activity = route.activity.to(device=native.device)
    # This subtraction retains the adapter gradient.  Under the special
    # step-zero add its numeric value is exact zero while d(result)/d(delta)=1.
    residual = result - native
    route.trace.record(
        block_index=block_index,
        residual=residual,
        activity=activity,
        layout=route.layout,
    )


def install_action_repr_g2a_adapter(
    model: Any,
    *,
    block_indices: Sequence[int] = DEFAULT_BLOCK_INDICES,
    hidden_width: int = DEFAULT_HIDDEN_WIDTH,
    flow_width: int = DEFAULT_FLOW_WIDTH,
    bottleneck_width: int = DEFAULT_BOTTLENECK_WIDTH,
    middle_width: Optional[int] = None,
    enable_source_copy_adapter: bool = False,
) -> G2APatchHandle:
    """Freeze the Bernini transformer and install the G2a residual route."""

    transformer = _resolve_transformer(model)
    indices = tuple(int(index) for index in block_indices)
    if (
        not indices
        or indices != tuple(sorted(set(indices)))
        or any(index < 0 or index >= EXPECTED_BLOCK_COUNT for index in indices)
    ):
        _fail("G2a block indices must be increasing unique values in [0,29]")
    if int(flow_width) != DEFAULT_FLOW_WIDTH:
        _fail("20260824 G2a flow ABI requires width 12")
    _positive_int(hidden_width, label="hidden_width")
    _positive_int(bottleneck_width, label="bottleneck_width")
    if middle_width is not None:
        _positive_int(middle_width, label="middle_width")
    if not isinstance(enable_source_copy_adapter, bool):
        _fail("enable_source_copy_adapter must be boolean")

    original_base = _named_parameters_all(transformer)
    original_ids = [id(parameter) for _, parameter in original_base]
    if len(original_ids) != len(set(original_ids)):
        _fail("base transformer has ambiguous parameter aliases")
    base_manifest, base_digest = _base_manifest(original_base)
    transformer.requires_grad_(False)

    bundles: list[_G2ABlockBundle] = []
    originals: list[Any] = []
    installed: list[int] = []
    handle_cell: list[Optional[G2APatchHandle]] = [None]
    base_device = (
        original_base[0][1].device if original_base else torch.device("cpu")
    )
    try:
        for index in indices:
            block = transformer.blocks[index]
            if hasattr(block, MODULE_NAME):
                _fail(f"block {index} already owns a G2a adapter")
            bundle = _G2ABlockBundle(
                hidden_width=int(hidden_width),
                flow_width=int(flow_width),
                bottleneck_width=int(bottleneck_width),
                middle_width=(int(middle_width) if middle_width is not None else None),
                source_copy=enable_source_copy_adapter,
            ).to(device=base_device, dtype=torch.float32)
            block.add_module(MODULE_NAME, bundle)
            original = block.forward

            def wrapped_forward(
                self: Any,
                *args: Any,
                _original: Any = original,
                _bundle: _G2ABlockBundle = bundle,
                _index: int = index,
                **kwargs: Any,
            ) -> torch.Tensor:
                native = _original(*args, **kwargs)
                if not isinstance(native, torch.Tensor) or native.ndim != 3:
                    _fail("G2a patched Bernini block must return [B,L,D]")
                route = current_action_representation_route()
                if route is None:
                    return native
                if handle_cell[0] is None:
                    _fail("G2a wrapper executed before installation closure")
                handle = handle_cell[0]
                _validate_route_for_handle(route, handle)
                if route.kind in {"route_off", "zero"}:
                    _record_trace(
                        block_index=_index,
                        route=route,
                        native=native,
                        result=native,
                    )
                    return native
                assert route.flow is not None and route.activity is not None
                flow = _local_representation_tensor(
                    route.flow, native, label="action flow"
                ).to(dtype=torch.float32)
                activity = _local_representation_tensor(
                    route.activity, native, label="action activity"
                ).bool()
                projected_middle: Optional[torch.Tensor] = None
                if _bundle.middle_projector is not None:
                    middle = _local_representation_tensor(
                        route.middle_by_block[_index],
                        native,
                        label=f"middle residual block {_index}",
                    )
                    projected_middle = _bundle.middle_projector(middle)
                motion_delta = _bundle.motion_adapter.delta(
                    native, flow, activity, projected_middle
                )
                if int(route.optimizer_step) == 0:
                    if not _bundle.output_gates_are_byte_zero():
                        _fail("optimizer step zero encountered a nonzero output gate")
                    result = _ExactZeroResidualAdd.apply(native, motion_delta)
                else:
                    result = native + motion_delta
                if _bundle.source_copy_adapter is not None:
                    carrier = _source_carrier(result, route)
                    source_delta = _bundle.source_copy_adapter.delta(
                        result, carrier, activity
                    )
                    if int(route.optimizer_step) == 0:
                        result = _ExactZeroResidualAdd.apply(result, source_delta)
                    else:
                        result = result + source_delta
                _record_trace(
                    block_index=_index,
                    route=route,
                    native=native,
                    result=result,
                )
                return result

            block.forward = types.MethodType(wrapped_forward, block)
            bundles.append(bundle)
            originals.append(original)
            installed.append(index)
    except Exception:
        for index, bundle, original in zip(
            reversed(installed), reversed(bundles), reversed(originals)
        ):
            block = transformer.blocks[index]
            block.forward = original
            if getattr(block, MODULE_NAME, None) is bundle:
                delattr(block, MODULE_NAME)
        raise

    # A one-element closure lets wrappers resolve the handle after all patches
    # are installed without capturing a partially constructed object.
    handle = G2APatchHandle(
        transformer=transformer,
        block_indices=indices,
        bundles=tuple(bundles),
        original_forwards=tuple(originals),
        hidden_width=int(hidden_width),
        flow_width=int(flow_width),
        bottleneck_width=int(bottleneck_width),
        middle_width=(int(middle_width) if middle_width is not None else None),
        source_copy_enabled=enable_source_copy_adapter,
        base_parameter_names=tuple(name for name, _ in original_base),
        base_parameter_ids=tuple(id(parameter) for _, parameter in original_base),
        base_manifest_at_install=base_manifest,
        base_digest_at_install=base_digest,
    )
    handle_cell[0] = handle
    if not handle.output_gates_are_byte_zero():
        handle.restore()
        _fail("G2a output projection lost positive-byte-zero initialization")
    handle.audit_parameters(deep_base_bytes=True)
    return handle


def validate_g2a_receipt(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("G2a receipt must be a mapping")
    row = dict(value)
    declared = row.pop("receipt_digest", None)
    _sha256(declared, label="receipt_digest")
    if object_sha256(row) != declared:
        _fail("G2a receipt digest differs")
    step0 = row.get("step0_noop_audit")
    audit = dict(step0) if isinstance(step0, Mapping) else {}
    audit_declared = audit.pop("audit_digest", None)
    _sha256(audit_declared, label="step0 audit digest")
    parameter = row.get("parameter_audit")
    architecture = row.get("architecture")
    architecture_unsigned = (
        dict(architecture) if isinstance(architecture, Mapping) else {}
    )
    architecture_digest = architecture_unsigned.pop("architecture_digest", None)
    route_outputs = audit.get("route_outputs")
    information = row.get("information_firewall")
    parameter_roles = parameter.get("roles") if isinstance(parameter, Mapping) else None
    if isinstance(parameter, Mapping):
        _sha256(
            parameter.get("base_digest_at_install"),
            label="base digest at install",
        )
        _sha256(
            parameter.get("base_digest_current"),
            label="current base digest",
        )
    _sha256(audit.get("matched_input_sha256"), label="matched input digest")
    _sha256(audit.get("native_tensor_sha256"), label="native tensor digest")
    _sha256(architecture_digest, label="architecture digest")
    if (
        row.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or row.get("gate") != "G2a_zero_init_noop"
        or row.get("passed") is not True
        or row.get("optimizer_created") is not False
        or row.get("optimizer_step") != 0
        or row.get("optimizer_authorized_by_this_receipt") is not False
        or row.get("stage_b_training_started") is not False
        or row.get("G0_asserted_by_this_receipt") is not False
        or row.get("G1_asserted_by_this_receipt") is not False
        or not isinstance(parameter, Mapping)
        or parameter.get("trainable_roles_exact") != list(TRAINABLE_ROLES)
        or not isinstance(parameter_roles, Mapping)
        or set(parameter_roles) != set(TRAINABLE_ROLES)
        or parameter.get("observed_trainable_equals_allowlist") is not True
        or parameter.get("base_requires_grad_false") is not True
        or parameter.get("deep_base_byte_audit") is not True
        or parameter.get("base_bytes_unchanged") is not True
        or parameter.get("base_digest_at_install")
        != parameter.get("base_digest_current")
        or not isinstance(architecture, Mapping)
        or architecture.get("schema_version") != SCHEMA_VERSION
        or object_sha256(architecture_unsigned) != architecture_digest
        or architecture.get("graph_route_enabled") is not False
        or architecture.get("route_off_hard_bypass") is not True
        or architecture.get("zero_route_hard_bypass") is not True
        or audit.get("schema_version") != STEP0_AUDIT_SCHEMA_VERSION
        or audit.get("optimizer_step") != 0
        or audit.get("required_routes") != list(STEP0_REQUIRED_ROUTES)
        or audit.get("all_routes_exact_native_bits") is not True
        or not isinstance(route_outputs, Mapping)
        or set(route_outputs) != set(STEP0_REQUIRED_ROUTES)
        or any(
            not isinstance(route_outputs[kind], Mapping)
            or route_outputs[kind].get("exact_native_bits") is not True
            or route_outputs[kind].get("tensor_sha256")
            != audit.get("native_tensor_sha256")
            for kind in STEP0_REQUIRED_ROUTES
        )
        or not isinstance(information, Mapping)
        or information.get("detached_representation_cache_only") is not True
        or information.get("target_rgb_to_adapter") is not False
        or information.get("target_vae_or_clean_latent_to_adapter") is not False
        or information.get("target_absolute_hidden_value_or_raw_qk_to_adapter")
        is not False
        or row.get("claim_scope")
        != "implementation_safety_gate_only_not_method_success"
        or object_sha256(audit) != audit_declared
    ):
        _fail("G2a receipt safety closure differs")
    return value


def write_receipt_create_only(path: str | Path, value: Mapping[str, Any]) -> None:
    validate_g2a_receipt(value)
    target = Path(path).expanduser()
    if not target.is_absolute():
        _fail("G2a receipt path must be absolute")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    try:
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as error:
        raise G2AAdapterError("G2a receipt publication is create-only") from error
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                _fail("G2a receipt publication made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "ACTIVE_ROUTE_KINDS",
    "ActionRepresentationRoute",
    "DEFAULT_BLOCK_INDICES",
    "DEFAULT_BOTTLENECK_WIDTH",
    "DEFAULT_FLOW_WIDTH",
    "DEFAULT_HIDDEN_WIDTH",
    "G2AAdapterError",
    "G2APatchHandle",
    "MiddleProjector",
    "RECEIPT_SCHEMA_VERSION",
    "REPRESENTATION_ORIGINS",
    "ROUTE_KINDS",
    "ResidualTraceCollector",
    "SCHEMA_VERSION",
    "STATE_SCHEMA_VERSION",
    "STEP0_REQUIRED_ROUTES",
    "SourceCopyAdapter",
    "TRAINABLE_ROLES",
    "TokenLayout",
    "ZeroInitializedMotionAdapter",
    "action_representation_route",
    "canonical_json_bytes",
    "current_action_representation_route",
    "install_action_repr_g2a_adapter",
    "object_sha256",
    "tensor_bits_equal",
    "tensor_sha256",
    "validate_g2a_receipt",
    "write_receipt_create_only",
]
