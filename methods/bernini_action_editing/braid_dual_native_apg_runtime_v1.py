#!/usr/bin/env python3
"""Stage-0 dual-native-APG runtime contracts for pinned Bernini V2V.

This module is deliberately narrower than a BRAID editor.  It runs the stock
source-conditioned no-op APG branch and a second source-conditioned action APG
branch on the same solver state, observes a single block-15 source co-state
reset, and passes the *original stock no-op APG tensor object* to the untouched
UniPC scheduler exactly once.  It creates no optimizer, performs no backward,
decodes no video, and reads or writes no checkpoint.

``reference_4f`` is the Stage-0 reference call graph::

    base negative -> base positive -> action negative -> action positive

``shared_negative_3f_diagnostic`` is an explicitly opt-in diagnostic that
reuses the base negative prediction.  It is not described as two independent
native APG trajectories.

Block 15 is an infrastructure canary chosen because its SP4 layout has already
been audited.  This file makes no claim that block 15 is an effective or
authorized BRAID intervention boundary.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import hashlib
import importlib
import inspect
import json
import math
from numbers import Real
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import torch

import self_guided_action_field_v1 as sgaf
import tri_branch_unipc as bernini_contract


SCHEMA_VERSION = "bernini-braid-dual-native-apg-stage0-runtime-v1"
LAYOUT_SCHEMA_VERSION = "bernini-braid-sp4-role-layout-v1"
BLOCK15_SCHEMA_VERSION = "bernini-braid-block15-source-costate-canary-v1"
PINNED_BERNINI_COMMIT = bernini_contract.PINNED_BERNINI_COMMIT
PINNED_WAN_DIFFUSION_SHA256 = bernini_contract.PINNED_WAN_DIFFUSION_SHA256
VENDOR_APG_MODULE = "bernini.models.wan_diffusion"

REFERENCE_4F = "reference_4f"
SHARED_NEGATIVE_3F_DIAGNOSTIC = "shared_negative_3f_diagnostic"
FORWARD_MODES = (REFERENCE_4F, SHARED_NEGATIVE_3F_DIAGNOSTIC)
REFERENCE_4F_ORDER = (
    "base_negative",
    "base_positive",
    "action_negative",
    "action_positive",
)
SHARED_NEGATIVE_3F_ORDER = (
    "base_negative",
    "base_positive",
    "action_positive",
)
BLOCK_INDEX = 15
EXPECTED_TRANSFORMER_BLOCKS = 30
BLOCK15_AUTHORITY = (
    "infrastructure_canary_only_not_an_authorized_braid_reset_boundary"
)


class BraidDualNativeAPGRuntimeError(RuntimeError):
    """Raised before UniPC integration when a Stage-0 contract differs."""


def _object_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise BraidDualNativeAPGRuntimeError(
            "runtime receipt is not canonical finite ASCII JSON"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _scalar(value: Any, *, label: str) -> float:
    try:
        return float(sgaf._coerce_scalar(value, label=label))
    except Exception as error:
        raise BraidDualNativeAPGRuntimeError(str(error)) from error


def _shape(value: Any, *, label: str) -> tuple[int, ...]:
    try:
        return tuple(sgaf._shape(value, label=label))
    except Exception as error:
        raise BraidDualNativeAPGRuntimeError(str(error)) from error


def _bind(
    callable_object: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        return dict(sgaf._bind_call(callable_object, args, kwargs))
    except Exception as error:
        raise BraidDualNativeAPGRuntimeError(str(error)) from error


def _metadata(value: Any, *, label: str) -> tuple[int, ...]:
    try:
        return tuple(sgaf._metadata_tuple(value, label=label))
    except Exception as error:
        raise BraidDualNativeAPGRuntimeError(str(error)) from error


def _same(left: Any, right: Any, *, label: str) -> None:
    if left is not right:
        raise BraidDualNativeAPGRuntimeError(f"{label} object identity differs")


def _raw_bytes_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if (
        type(left) is not torch.Tensor
        or type(right) is not torch.Tensor
        or tuple(left.shape) != tuple(right.shape)
        or left.dtype != right.dtype
        or left.device != right.device
        or left.layout != torch.strided
        or right.layout != torch.strided
    ):
        return False
    return bool(
        torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
    )


def _mismatch_count(left: torch.Tensor, right: torch.Tensor) -> int:
    if tuple(left.shape) != tuple(right.shape):
        raise BraidDualNativeAPGRuntimeError("mismatch tensors have different shapes")
    if left.numel() == 0:
        return 0
    left_bytes = left.detach().contiguous().view(torch.uint8)
    right_bytes = right.detach().contiguous().view(torch.uint8)
    return int(torch.count_nonzero(left_bytes != right_bytes).item())


def _finite_tensor(value: Any, *, label: str) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.layout != torch.strided
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise BraidDualNativeAPGRuntimeError(
            f"{label} must be an exact-type finite no-grad strided tensor"
        )
    return value


@dataclass(frozen=True)
class BraidSP4RoleLayout:
    """Rank-local source/target/append-padding roles for contiguous SP4.

    The global index of local row ``j`` on SP rank ``r`` is
    ``g = r * ceil(total_tokens / 4) + j``.  Source rows precede target rows;
    only rows at or beyond ``total_tokens`` are append padding.
    """

    total_tokens: int
    condition_tokens: int
    target_tokens: int
    sp_rank: int
    sp_size: int
    local_length: int
    shard_global_start: int
    shard_global_stop_padded: int
    source_local_indices: torch.Tensor = field(repr=False, compare=False)
    target_local_indices: torch.Tensor = field(repr=False, compare=False)
    padding_local_indices: torch.Tensor = field(repr=False, compare=False)

    @classmethod
    def build(
        cls,
        *,
        total_tokens: int,
        condition_tokens: int,
        sp_rank: int,
        sp_size: int = 4,
        observed_local_length: Optional[int] = None,
    ) -> "BraidSP4RoleLayout":
        for value, label in (
            (total_tokens, "total_tokens"),
            (condition_tokens, "condition_tokens"),
            (sp_rank, "sp_rank"),
            (sp_size, "sp_size"),
        ):
            if type(value) is not int:
                raise BraidDualNativeAPGRuntimeError(f"{label} must be an integer")
        if total_tokens <= 1 or not 0 < condition_tokens < total_tokens:
            raise BraidDualNativeAPGRuntimeError(
                "source-prefix/target-suffix token geometry differs"
            )
        if sp_size != 4 or not 0 <= sp_rank < sp_size:
            raise BraidDualNativeAPGRuntimeError(
                "BraidSP4RoleLayout requires one rank of an SP4 group"
            )
        local_length = math.ceil(total_tokens / sp_size)
        if (
            observed_local_length is not None
            and observed_local_length != local_length
        ):
            raise BraidDualNativeAPGRuntimeError(
                "observed rank-local hidden length differs from contiguous SP4"
            )
        start = sp_rank * local_length
        local = torch.arange(local_length, dtype=torch.int64)
        global_indices = start + local
        source = global_indices < condition_tokens
        target = (global_indices >= condition_tokens) & (
            global_indices < total_tokens
        )
        padding = global_indices >= total_tokens
        if not bool(torch.all((source.to(torch.int8) + target.to(torch.int8) + padding.to(torch.int8)) == 1).item()):
            raise BraidDualNativeAPGRuntimeError("SP4 row-role partition is not closed")
        result = cls(
            total_tokens=total_tokens,
            condition_tokens=condition_tokens,
            target_tokens=total_tokens - condition_tokens,
            sp_rank=sp_rank,
            sp_size=sp_size,
            local_length=local_length,
            shard_global_start=start,
            shard_global_stop_padded=start + local_length,
            source_local_indices=local[source].contiguous(),
            target_local_indices=local[target].contiguous(),
            padding_local_indices=local[padding].contiguous(),
        )
        result.validate()
        return result

    def validate(self) -> None:
        parts = (
            self.source_local_indices,
            self.target_local_indices,
            self.padding_local_indices,
        )
        if (
            self.sp_size != 4
            or not 0 <= self.sp_rank < 4
            or self.target_tokens != self.total_tokens - self.condition_tokens
            or self.local_length != math.ceil(self.total_tokens / 4)
            or any(
                type(value) is not torch.Tensor
                or value.device.type != "cpu"
                or value.dtype != torch.int64
                or value.ndim != 1
                for value in parts
            )
        ):
            raise BraidDualNativeAPGRuntimeError("SP4 role layout changed")
        joined = torch.cat(parts).sort().values
        if not torch.equal(joined, torch.arange(self.local_length, dtype=torch.int64)):
            raise BraidDualNativeAPGRuntimeError("SP4 local roles do not partition the shard")

    def indices(self, role: str, *, device: torch.device) -> torch.Tensor:
        values = {
            "source": self.source_local_indices,
            "target": self.target_local_indices,
            "padding": self.padding_local_indices,
        }
        if role not in values:
            raise BraidDualNativeAPGRuntimeError(f"unknown SP4 row role {role!r}")
        return values[role].to(device=device)

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "sp_rank": self.sp_rank,
            "sp_size": self.sp_size,
            "total_tokens": self.total_tokens,
            "condition_tokens": self.condition_tokens,
            "target_tokens": self.target_tokens,
            "local_length_ceil": self.local_length,
            "shard_global_start": self.shard_global_start,
            "shard_global_stop_padded": self.shard_global_stop_padded,
            "source_rows": int(self.source_local_indices.numel()),
            "target_rows": int(self.target_local_indices.numel()),
            "padding_rows": int(self.padding_local_indices.numel()),
            "global_index_formula": (
                "g=sp_rank*ceil(total_tokens/4)+local_index;"
                "source iff g<condition_tokens;target iff "
                "condition_tokens<=g<total_tokens;padding iff g>=total_tokens"
            ),
            "cross_rank_hidden_gather_or_reinjection": False,
        }


@dataclass(frozen=True)
class BraidBlock15StepRecord:
    step_index: int
    block_index: int
    forward_order: tuple[str, ...]
    reset_enabled: bool
    source_rows: int
    target_rows: int
    padding_rows: int
    source_pre_reset_mismatch_bytes: int
    source_post_reset_mismatch_bytes: int
    target_post_reset_mismatch_bytes: int
    padding_post_reset_mismatch_bytes: int
    reset_returned_new_object: bool
    reset_off_returned_original_object: bool
    cache_created_once: bool
    cache_consumed_once: bool


class BraidBlock15SourceCoStateHook:
    """One rank-local post-block source reset at block 15.

    This is a Stage-0 infrastructure canary, not evidence that block 15 is an
    effective motion-editing boundary.  No collective is performed by the
    hook.  The no-op source cache is detached, cloned, consumed once, and then
    released at the end of the denoising step.
    """

    def __init__(
        self,
        transformer: Any,
        *,
        reset_enabled: bool,
        block_index: int = BLOCK_INDEX,
    ) -> None:
        if type(reset_enabled) is not bool:
            raise BraidDualNativeAPGRuntimeError("reset_enabled must be bool")
        blocks = tuple(getattr(transformer, "blocks", ()))
        if block_index != BLOCK_INDEX or len(blocks) != EXPECTED_TRANSFORMER_BLOCKS:
            raise BraidDualNativeAPGRuntimeError(
                "block15 canary requires the pinned 30-block 1.3B transformer"
            )
        block = blocks[block_index]
        registry = getattr(block, "_forward_hooks", None)
        if not callable(getattr(block, "register_forward_hook", None)) or not isinstance(
            registry, Mapping
        ):
            raise BraidDualNativeAPGRuntimeError("block15 is not auditable/hookable")
        if registry:
            raise BraidDualNativeAPGRuntimeError(
                "block15 already has forward hooks; intervention order is ambiguous"
            )
        self.transformer = transformer
        self.block = block
        self.block_index = block_index
        self.reset_enabled = reset_enabled
        self._handle: Any = None
        self._hook_id: Optional[int] = None
        self._registered_callback = self._hook
        self._pending_leg: Optional[str] = None
        self._pending_calls = 0
        self._step_index: Optional[int] = None
        self._layout: Optional[BraidSP4RoleLayout] = None
        self._expected_order: tuple[str, ...] = ()
        self._observed_order: list[str] = []
        self._anchor: Optional[torch.Tensor] = None
        self._cache_creations = 0
        self._cache_consumptions = 0
        self._action_record: Optional[dict[str, Any]] = None
        self.records: list[BraidBlock15StepRecord] = []

    @property
    def installed(self) -> bool:
        return self._handle is not None

    def install(self) -> None:
        if self._handle is not None:
            raise BraidDualNativeAPGRuntimeError("block15 hook is already installed")
        handle = self.block.register_forward_hook(self._registered_callback)
        identifier = getattr(handle, "id", None)
        registry = getattr(self.block, "_forward_hooks", None)
        if (
            type(identifier) is not int
            or not isinstance(registry, Mapping)
            or len(registry) != 1
            or registry.get(identifier) is not self._registered_callback
        ):
            handle.remove()
            raise BraidDualNativeAPGRuntimeError(
                "installed block15 hook identity cannot be authenticated"
            )
        self._handle = handle
        self._hook_id = identifier

    def _audit(self) -> None:
        registry = getattr(self.block, "_forward_hooks", None)
        if (
            self._handle is None
            or self._hook_id is None
            or not isinstance(registry, Mapping)
            or len(registry) != 1
            or registry.get(self._hook_id) is not self._registered_callback
        ):
            raise BraidDualNativeAPGRuntimeError("block15 hook registry/order changed")

    def remove(self) -> None:
        if self._step_index is not None or self._pending_leg is not None:
            raise BraidDualNativeAPGRuntimeError("cannot remove an active block15 hook")
        self._audit()
        assert self._handle is not None
        self._handle.remove()
        self._handle = None
        self._hook_id = None

    def begin_step(
        self,
        *,
        step_index: int,
        layout: BraidSP4RoleLayout,
        forward_order: Sequence[str],
    ) -> None:
        self._audit()
        layout.validate()
        order = tuple(forward_order)
        if (
            self._step_index is not None
            or type(step_index) is not int
            or step_index != len(self.records)
            or order not in (REFERENCE_4F_ORDER, SHARED_NEGATIVE_3F_ORDER)
        ):
            raise BraidDualNativeAPGRuntimeError("block15 step lifecycle/order differs")
        self._step_index = step_index
        self._layout = layout
        self._expected_order = order
        self._observed_order = []
        self._anchor = None
        self._cache_creations = 0
        self._cache_consumptions = 0
        self._action_record = None

    @contextmanager
    def leg(self, name: str) -> Iterator[None]:
        if (
            self._step_index is None
            or self._pending_leg is not None
            or name not in self._expected_order
            or len(self._observed_order) >= len(self._expected_order)
            or self._expected_order[len(self._observed_order)] != name
        ):
            raise BraidDualNativeAPGRuntimeError(
                f"unexpected or out-of-order block15 leg {name!r}"
            )
        self._pending_leg = name
        before = self._pending_calls
        try:
            yield
            if self._pending_calls != before + 1:
                raise BraidDualNativeAPGRuntimeError(
                    f"block15 hook did not fire exactly once for {name}"
                )
            self._observed_order.append(name)
        finally:
            self._pending_leg = None

    def _validate_hidden(self, output: Any) -> torch.Tensor:
        layout = self._layout
        if layout is None:
            raise BraidDualNativeAPGRuntimeError("block15 layout is unavailable")
        hidden = _finite_tensor(output, label="block15 output")
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != layout.local_length:
            raise BraidDualNativeAPGRuntimeError(
                "block15 rank-local [1,ceil(total/SP4),hidden] geometry differs"
            )
        return hidden

    def _hook(self, module: Any, inputs: Any, output: Any) -> Optional[Any]:
        del inputs
        self._audit()
        if module is not self.block or self._pending_leg is None:
            raise BraidDualNativeAPGRuntimeError(
                "block15 fired outside an authenticated branch leg"
            )
        hidden = self._validate_hidden(output)
        assert self._layout is not None
        source = self._layout.indices("source", device=hidden.device)
        target = self._layout.indices("target", device=hidden.device)
        padding = self._layout.indices("padding", device=hidden.device)
        leg = self._pending_leg
        self._pending_calls += 1
        if leg == "base_positive":
            if self._anchor is not None or self._cache_creations != 0:
                raise BraidDualNativeAPGRuntimeError(
                    "block15 no-op source cache was created twice"
                )
            self._anchor = hidden.index_select(1, source).detach().clone().contiguous()
            self._cache_creations = 1
            return None
        if leg != "action_positive":
            return None
        if self._anchor is None or self._cache_consumptions != 0:
            raise BraidDualNativeAPGRuntimeError(
                "block15 action leg lacks a single-use no-op source cache"
            )
        current_source = hidden.index_select(1, source)
        current_target = hidden.index_select(1, target).detach().clone().contiguous()
        current_padding = hidden.index_select(1, padding).detach().clone().contiguous()
        pre_mismatch = _mismatch_count(current_source, self._anchor)
        returned_new = False
        reset_off_identity = False
        if self.reset_enabled and source.numel() > 0:
            result = hidden.clone()
            result.index_copy_(1, source, self._anchor)
            returned_new = result is not hidden
            returned: Optional[torch.Tensor] = result
            checked = result
        else:
            returned = None
            checked = hidden
            reset_off_identity = not self.reset_enabled
        post_source = checked.index_select(1, source)
        post_target = checked.index_select(1, target)
        post_padding = checked.index_select(1, padding)
        source_mismatch = _mismatch_count(post_source, self._anchor)
        target_mismatch = _mismatch_count(post_target, current_target)
        padding_mismatch = _mismatch_count(post_padding, current_padding)
        if (
            (self.reset_enabled and source_mismatch != 0)
            or target_mismatch != 0
            or padding_mismatch != 0
            or (not self.reset_enabled and returned is not None)
        ):
            raise BraidDualNativeAPGRuntimeError(
                "block15 source/target/padding reset postcondition failed"
            )
        self._cache_consumptions = 1
        self._action_record = {
            "source_pre_reset_mismatch_bytes": pre_mismatch,
            "source_post_reset_mismatch_bytes": source_mismatch,
            "target_post_reset_mismatch_bytes": target_mismatch,
            "padding_post_reset_mismatch_bytes": padding_mismatch,
            "reset_returned_new_object": returned_new,
            "reset_off_returned_original_object": reset_off_identity,
        }
        return returned

    def finish_step(self) -> BraidBlock15StepRecord:
        if (
            self._step_index is None
            or self._layout is None
            or self._pending_leg is not None
            or tuple(self._observed_order) != self._expected_order
            or self._cache_creations != 1
            or self._cache_consumptions != 1
            or self._action_record is None
        ):
            raise BraidDualNativeAPGRuntimeError(
                "block15 source co-state step did not close exactly once"
            )
        record = BraidBlock15StepRecord(
            step_index=self._step_index,
            block_index=self.block_index,
            forward_order=self._expected_order,
            reset_enabled=self.reset_enabled,
            source_rows=int(self._layout.source_local_indices.numel()),
            target_rows=int(self._layout.target_local_indices.numel()),
            padding_rows=int(self._layout.padding_local_indices.numel()),
            cache_created_once=True,
            cache_consumed_once=True,
            **self._action_record,
        )
        self.records.append(record)
        self._step_index = None
        self._layout = None
        self._expected_order = ()
        self._observed_order = []
        self._anchor = None
        self._action_record = None
        return record

    def abort_step(self) -> None:
        self._pending_leg = None
        self._step_index = None
        self._layout = None
        self._expected_order = ()
        self._observed_order = []
        self._anchor = None
        self._cache_creations = 0
        self._cache_consumptions = 0
        self._action_record = None

    def receipt(self) -> dict[str, Any]:
        if self._step_index is not None:
            raise BraidDualNativeAPGRuntimeError("block15 receipt requested mid-step")
        return {
            "schema_version": BLOCK15_SCHEMA_VERSION,
            "block_index": self.block_index,
            "selection_authority": BLOCK15_AUTHORITY,
            "reset_enabled": self.reset_enabled,
            "rank_local_only": True,
            "hidden_collective_or_reinjection": False,
            "records": [asdict(row) for row in self.records],
            "semantic_action_editing_claim": False,
            "training_authorized": False,
        }


@dataclass
class BraidVendorAPGStateBinding:
    """Strong-reference binding to one exact vendor ``MomentumBuffer``."""

    branch: str
    buffer: Any
    vendor_class: type[Any]
    expected_momentum: float
    call_count: int = 0
    initial_state_authenticated: bool = False

    @classmethod
    def bind_fresh(
        cls,
        *,
        branch: str,
        buffer: Any,
        vendor_class: type[Any],
        expected_momentum: float,
        observed_initial_running_average: Any,
    ) -> "BraidVendorAPGStateBinding":
        if type(buffer) is not vendor_class:
            raise BraidDualNativeAPGRuntimeError(
                f"{branch} APG buffer is not the exact vendor type"
            )
        if _scalar(getattr(buffer, "momentum", None), label=f"{branch} momentum") != expected_momentum:
            raise BraidDualNativeAPGRuntimeError(f"{branch} APG momentum differs")
        initial = observed_initial_running_average
        if type(initial) is not int or initial != 0:
            raise BraidDualNativeAPGRuntimeError(
                f"{branch} APG initial running_average is not pinned integer zero"
            )
        return cls(
            branch=branch,
            buffer=buffer,
            vendor_class=vendor_class,
            expected_momentum=expected_momentum,
            initial_state_authenticated=True,
        )

    def observe_completed_call(
        self,
        *,
        pred_cond: torch.Tensor,
        pred_uncond: torch.Tensor,
    ) -> None:
        if type(self.buffer) is not self.vendor_class:
            raise BraidDualNativeAPGRuntimeError(
                f"{self.branch} APG buffer type changed"
            )
        running = getattr(self.buffer, "running_average", None)
        expected = pred_cond - pred_uncond
        if (
            type(running) is not torch.Tensor
            or tuple(running.shape) != tuple(expected.shape)
            or running.dtype != expected.dtype
            or running.device != expected.device
            or not torch.equal(running, expected)
        ):
            raise BraidDualNativeAPGRuntimeError(
                f"{self.branch} zero-momentum APG state did not receive this call once"
            )
        self.call_count += 1

    def receipt(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "vendor_type": (
                f"{self.vendor_class.__module__}.{self.vendor_class.__name__}"
            ),
            "buffer_object_id": id(self.buffer),
            "momentum": self.expected_momentum,
            "initial_integer_zero_authenticated": self.initial_state_authenticated,
            "normalized_guidance_calls": self.call_count,
        }


@dataclass(frozen=True)
class _VendorAPGObservation:
    pred_cond: torch.Tensor
    pred_uncond: torch.Tensor
    guided_clean: torch.Tensor
    binding: BraidVendorAPGStateBinding


class BraidVendorAPGStateObserver:
    """Reversible observer around the exact vendor APG function."""

    def __init__(
        self,
        *,
        vendor_module: Any,
        vendor_function: Callable[..., Any],
        vendor_momentum_class: type[Any],
        handler: Callable[..., None],
    ) -> None:
        self.vendor_module = vendor_module
        self.vendor_function = vendor_function
        self.vendor_momentum_class = vendor_momentum_class
        self.handler = handler
        self.wrapper: Optional[Callable[..., Any]] = None
        self.installed = False

    def install(self) -> None:
        if self.installed or getattr(self.vendor_module, "normalized_guidance", None) is not self.vendor_function:
            raise BraidDualNativeAPGRuntimeError(
                "vendor normalized_guidance observer cannot be stacked"
            )

        def wrapper(
            pred_cond: Any,
            pred_uncond: Any,
            guidance_scale: Any,
            momentum_buffer: Any = None,
            eta: Any = 1.0,
            norm_threshold: Any = 0.0,
        ) -> Any:
            before = getattr(momentum_buffer, "running_average", None)
            result = self.vendor_function(
                pred_cond=pred_cond,
                pred_uncond=pred_uncond,
                guidance_scale=guidance_scale,
                momentum_buffer=momentum_buffer,
                eta=eta,
                norm_threshold=norm_threshold,
            )
            self.handler(
                pred_cond=pred_cond,
                pred_uncond=pred_uncond,
                guidance_scale=guidance_scale,
                momentum_buffer=momentum_buffer,
                eta=eta,
                norm_threshold=norm_threshold,
                before_running_average=before,
                result=result,
            )
            return result

        setattr(wrapper, "_bernini_braid_vendor_apg_observer_v1", self)
        self.vendor_module.normalized_guidance = wrapper
        self.wrapper = wrapper
        self.installed = True

    def restore(self) -> None:
        if not self.installed or self.wrapper is None:
            raise BraidDualNativeAPGRuntimeError("vendor APG observer is not installed")
        if getattr(self.vendor_module, "normalized_guidance", None) is not self.wrapper:
            raise BraidDualNativeAPGRuntimeError(
                "vendor normalized_guidance changed behind observer"
            )
        self.vendor_module.normalized_guidance = self.vendor_function
        self.wrapper = None
        self.installed = False


@dataclass(frozen=True)
class BraidDualNativeAPGConfig:
    """Pinned exact81 Stage-0 structural canary configuration."""

    target_latent_shape: tuple[int, int, int, int, int]
    sp_rank: int
    reset_source_costate: bool
    forward_mode: str = REFERENCE_4F
    allow_shared_negative_diagnostic: bool = False
    expected_steps: int = 40
    expected_num_frames: int = 81
    expected_flow_shift: float = 5.0
    omega_text: float = 4.0
    omega_scale: float = 0.75
    eta: float = 0.5
    norm_threshold: float = 50.0
    momentum: float = 0.0
    expected_hidden_dim: int = 1536
    expected_text_dim: int = 4096
    expected_model_id: str = "transformer_1"
    expected_guidance_mode: str = "v2v_apg"
    block_index: int = BLOCK_INDEX

    @property
    def target_patch_tokens(self) -> int:
        _, _, phases, height, width = self.target_latent_shape
        return int(phases * (height // 2) * (width // 2))

    @property
    def total_v2v_tokens(self) -> int:
        return 2 * self.target_patch_tokens

    @property
    def forward_order(self) -> tuple[str, ...]:
        return (
            REFERENCE_4F_ORDER
            if self.forward_mode == REFERENCE_4F
            else SHARED_NEGATIVE_3F_ORDER
        )

    @property
    def forwards_per_step(self) -> int:
        return len(self.forward_order)

    def validate(self) -> None:
        shape = tuple(self.target_latent_shape)
        if (
            len(shape) != 5
            or any(type(value) is not int or value <= 0 for value in shape)
            or shape[0] != 1
            or shape[1] != 16
            or shape[2] != 21
            or shape[3] % 2
            or shape[4] % 2
        ):
            raise BraidDualNativeAPGRuntimeError(
                "target latent must be exact81 Bernini [1,16,21,even,even]"
            )
        if self.forward_mode not in FORWARD_MODES:
            raise BraidDualNativeAPGRuntimeError("unknown dual-native forward mode")
        if (
            self.forward_mode == SHARED_NEGATIVE_3F_DIAGNOSTIC
            and self.allow_shared_negative_diagnostic is not True
        ):
            raise BraidDualNativeAPGRuntimeError(
                "three-forward shared-negative diagnostic requires explicit opt-in"
            )
        if type(self.reset_source_costate) is not bool:
            raise BraidDualNativeAPGRuntimeError("reset_source_costate must be bool")
        if (
            type(self.sp_rank) is not int
            or not 0 <= self.sp_rank < 4
            or self.block_index != BLOCK_INDEX
            or self.expected_steps != 40
            or self.expected_num_frames != 81
            or self.expected_hidden_dim != 1536
            or self.expected_text_dim != 4096
            or self.expected_model_id != "transformer_1"
            or self.expected_guidance_mode != "v2v_apg"
        ):
            raise BraidDualNativeAPGRuntimeError(
                "Stage-0 canary is pinned to exact81/40, SP4, block15, Bernini-R 1.3B"
            )
        exact = {
            "expected_flow_shift": (self.expected_flow_shift, 5.0),
            "omega_text": (self.omega_text, 4.0),
            "omega_scale": (self.omega_scale, 0.75),
            "eta": (self.eta, 0.5),
            "norm_threshold": (self.norm_threshold, 50.0),
            "momentum": (self.momentum, 0.0),
        }
        for label, (value, expected) in exact.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or float(value) != expected
            ):
                raise BraidDualNativeAPGRuntimeError(
                    f"{label} must be exactly {expected}"
                )


@dataclass(frozen=True)
class _ForwardResult:
    name: str
    values: Mapping[str, Any]
    prediction: torch.Tensor
    target_tail: torch.Tensor


@dataclass
class _ActiveSample:
    base_prompt: torch.Tensor
    negative_prompt: torch.Tensor
    action_prompt: torch.Tensor
    action_binding: BraidVendorAPGStateBinding
    completed_steps: int = 0
    base_forwards: list[_ForwardResult] = field(default_factory=list)
    action_forwards: list[_ForwardResult] = field(default_factory=list)
    base_apg: Optional[_VendorAPGObservation] = None


def _resolve_vendor_apg_symbols() -> tuple[Any, Callable[..., Any], type[Any]]:
    try:
        module = importlib.import_module(VENDOR_APG_MODULE)
    except Exception as error:
        raise BraidDualNativeAPGRuntimeError(
            f"cannot import pinned {VENDOR_APG_MODULE}"
        ) from error
    function = getattr(module, "normalized_guidance", None)
    momentum_class = getattr(module, "MomentumBuffer", None)
    if not callable(function) or not isinstance(momentum_class, type):
        raise BraidDualNativeAPGRuntimeError("pinned vendor APG symbols unavailable")
    for value, name in (
        (function, "normalized_guidance"),
        (momentum_class, "MomentumBuffer"),
    ):
        if (
            getattr(value, "__module__", None) != VENDOR_APG_MODULE
            or getattr(value, "__name__", None) != name
            or inspect.getmodule(value) is not module
            or getattr(module, name) is not value
        ):
            raise BraidDualNativeAPGRuntimeError(
                "vendor APG module/symbol identity differs"
            )
    try:
        function_parameters = tuple(inspect.signature(function).parameters)
        momentum_parameters = tuple(inspect.signature(momentum_class).parameters)
    except (TypeError, ValueError) as error:
        raise BraidDualNativeAPGRuntimeError(
            "vendor APG signatures are not inspectable"
        ) from error
    if function_parameters != (
        "pred_cond",
        "pred_uncond",
        "guidance_scale",
        "momentum_buffer",
        "eta",
        "norm_threshold",
    ) or momentum_parameters != ("momentum",):
        raise BraidDualNativeAPGRuntimeError("vendor APG signature differs")
    return module, function, momentum_class


class BraidDualNativeAPGRuntimePatch:
    """Reversible one-sample Stage-0 dual-native APG runtime core."""

    def __init__(
        self,
        diffusion: Any,
        *,
        action_prompt_embeds: torch.Tensor,
        config: BraidDualNativeAPGConfig,
        expected_bernini_commit: str = PINNED_BERNINI_COMMIT,
        observed_wan_diffusion_sha256: str = PINNED_WAN_DIFFUSION_SHA256,
    ) -> None:
        config.validate()
        if expected_bernini_commit != PINNED_BERNINI_COMMIT:
            raise BraidDualNativeAPGRuntimeError("Bernini revision differs")
        if observed_wan_diffusion_sha256 != PINNED_WAN_DIFFUSION_SHA256:
            raise BraidDualNativeAPGRuntimeError("wan_diffusion.py bytes differ")
        try:
            core = bernini_contract.resolve_diffusion_core(diffusion)
        except Exception as error:
            raise BraidDualNativeAPGRuntimeError(str(error)) from error
        transformer = getattr(core, "transformer", None)
        scheduler = getattr(core, "scheduler", None)
        originals = {
            "sample": getattr(core, "sample", None),
            "shared_step": getattr(core, "shared_step", None),
            "scheduler.step": getattr(scheduler, "step", None),
        }
        if any(not callable(value) for value in originals.values()):
            raise BraidDualNativeAPGRuntimeError(
                "pinned Bernini sampler call surface differs"
            )
        if getattr(core, "use_unipc", None) is not True:
            raise BraidDualNativeAPGRuntimeError("runtime requires native UniPC")
        if getattr(core, "transformer_2", None) is not None:
            raise BraidDualNativeAPGRuntimeError(
                "runtime supports only single-expert Bernini-R 1.3B"
            )
        transformer_config = getattr(transformer, "config", None)
        if transformer_config is None:
            raise BraidDualNativeAPGRuntimeError("transformer config unavailable")

        def config_value(name: str) -> Any:
            value = getattr(transformer_config, name, None)
            if value is None and isinstance(transformer_config, Mapping):
                value = transformer_config.get(name)
            return value

        heads = config_value("num_attention_heads")
        head_dim = config_value("attention_head_dim")
        if (
            type(heads) is not int
            or type(head_dim) is not int
            or heads * head_dim != config.expected_hidden_dim
            or config_value("in_channels") != 16
            or config_value("text_dim") != config.expected_text_dim
        ):
            raise BraidDualNativeAPGRuntimeError(
                "transformer hidden/text/input geometry differs"
            )
        try:
            bernini_contract._validate_scheduler_contract(
                scheduler, expected_flow_shift=config.expected_flow_shift
            )
        except Exception as error:
            raise BraidDualNativeAPGRuntimeError(str(error)) from error
        if getattr(transformer, "training", False):
            raise BraidDualNativeAPGRuntimeError("transformer must remain in eval mode")
        for owner, name in ((core, "sample"), (core, "shared_step"), (scheduler, "step")):
            try:
                if name in vars(owner):
                    raise BraidDualNativeAPGRuntimeError(
                        f"refusing stacked instance override on {name}"
                    )
            except TypeError as error:
                raise BraidDualNativeAPGRuntimeError(
                    f"cannot inspect {name} owner"
                ) from error
        named_parameters = getattr(transformer, "named_parameters", None)
        named_buffers = getattr(transformer, "named_buffers", None)
        if not callable(named_parameters) or not callable(named_buffers):
            raise BraidDualNativeAPGRuntimeError(
                "transformer freeze surface unavailable"
            )
        for name, parameter in named_parameters():
            if bool(parameter.requires_grad) or parameter.grad is not None:
                raise BraidDualNativeAPGRuntimeError(
                    f"transformer parameter {name} is not freeze-safe"
                )
        self._state_versions_before = self._state_versions(transformer)
        self._validate_prompt_shape(action_prompt_embeds, config, label="action")
        vendor_module, vendor_function, vendor_class = _resolve_vendor_apg_symbols()
        self.diffusion = core
        self.transformer = transformer
        self.scheduler = scheduler
        self.action_prompt_embeds = action_prompt_embeds
        self.config = config
        self.layout = BraidSP4RoleLayout.build(
            total_tokens=config.total_v2v_tokens,
            condition_tokens=config.target_patch_tokens,
            sp_rank=config.sp_rank,
        )
        self.vendor_module = vendor_module
        self.vendor_function = vendor_function
        self.vendor_momentum_class = vendor_class
        self.original_sample = originals["sample"]
        self.original_shared_step = originals["shared_step"]
        self.original_scheduler_step = originals["scheduler.step"]
        self.block15 = BraidBlock15SourceCoStateHook(
            transformer,
            reset_enabled=config.reset_source_costate,
            block_index=config.block_index,
        )
        self.apg_observer = BraidVendorAPGStateObserver(
            vendor_module=vendor_module,
            vendor_function=vendor_function,
            vendor_momentum_class=vendor_class,
            handler=self._observe_base_apg,
        )
        self._patches: list[tuple[Any, str, bool, Any, Any]] = []
        self._active: Optional[_ActiveSample] = None
        self._base_binding: Optional[BraidVendorAPGStateBinding] = None
        self._action_binding: Optional[BraidVendorAPGStateBinding] = None
        self.installed = False
        self.restored = False
        self.finalized = False
        self.sample_call_count = 0
        self.transformer_forward_count = 0
        self.base_forward_count = 0
        self.action_forward_count = 0
        self.base_apg_call_count = 0
        self.action_apg_call_count = 0
        self.original_scheduler_call_count = 0
        self.trace: list[dict[str, Any]] = []

    @staticmethod
    def _state_versions(transformer: Any) -> tuple[tuple[Any, ...], ...]:
        rows = []
        for kind, iterator in (
            ("parameter", transformer.named_parameters()),
            ("buffer", transformer.named_buffers()),
        ):
            for name, value in iterator:
                rows.append(
                    (
                        kind,
                        name,
                        id(value),
                        int(getattr(value, "_version", -1)),
                        bool(getattr(value, "requires_grad", False)),
                        getattr(value, "grad", None) is None,
                    )
                )
        return tuple(rows)

    @staticmethod
    def _validate_prompt_shape(
        value: Any, config: BraidDualNativeAPGConfig, *, label: str
    ) -> torch.Tensor:
        tensor = _finite_tensor(value, label=f"{label} prompt")
        if tuple(tensor.shape) != (1, 512, config.expected_text_dim):
            raise BraidDualNativeAPGRuntimeError(f"{label} prompt geometry differs")
        return tensor

    def _set_patch(self, owner: Any, name: str, value: Any) -> None:
        try:
            instance = vars(owner)
        except TypeError as error:
            raise BraidDualNativeAPGRuntimeError(
                f"cannot reversibly patch {name} owner"
            ) from error
        had_instance = name in instance
        previous = instance.get(name)
        resolved_before = getattr(owner, name)
        setattr(owner, name, value)
        self._patches.append((owner, name, had_instance, previous, resolved_before))

    def install(self) -> None:
        if self.installed or self.restored or self.finalized:
            raise BraidDualNativeAPGRuntimeError("runtime patch lifecycle differs")

        def sample_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_sample(*args, **kwargs)

        def shared_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_shared_step(*args, **kwargs)

        def scheduler_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_scheduler_step(*args, **kwargs)

        for wrapper in (sample_wrapper, shared_wrapper, scheduler_wrapper):
            setattr(wrapper, "_bernini_braid_dual_native_apg_v1", self)
        try:
            self.block15.install()
            self.apg_observer.install()
            self._set_patch(self.diffusion, "shared_step", shared_wrapper)
            self._set_patch(self.scheduler, "step", scheduler_wrapper)
            self._set_patch(self.diffusion, "sample", sample_wrapper)
        except BaseException:
            self._restore_partial(require_identity=False)
            raise
        self.installed = True

    def _restore_partial(self, *, require_identity: bool) -> None:
        errors: list[BaseException] = []
        while self._patches:
            owner, name, had_instance, previous, resolved_before = self._patches.pop()
            try:
                current = getattr(owner, name, None)
                if require_identity and getattr(
                    current, "_bernini_braid_dual_native_apg_v1", None
                ) is not self:
                    raise BraidDualNativeAPGRuntimeError(f"{name} wrapper changed")
                if had_instance:
                    setattr(owner, name, previous)
                else:
                    delattr(owner, name)
                if getattr(owner, name, None) != resolved_before:
                    raise BraidDualNativeAPGRuntimeError(f"{name} restoration failed")
            except BaseException as error:
                errors.append(error)
        try:
            if self.apg_observer.installed:
                self.apg_observer.restore()
        except BaseException as error:
            errors.append(error)
        try:
            if self.block15.installed:
                if self._active is not None:
                    self.block15.abort_step()
                self.block15.remove()
        except BaseException as error:
            errors.append(error)
        self._active = None
        if errors:
            raise BraidDualNativeAPGRuntimeError(
                f"failed to restore {len(errors)} BRAID runtime hook(s)"
            ) from errors[0]

    def restore(self) -> None:
        if not self.installed or self.restored:
            raise BraidDualNativeAPGRuntimeError("runtime restore lifecycle differs")
        try:
            self._restore_partial(require_identity=True)
        finally:
            self.installed = False
            self.restored = not self._patches and not self.block15.installed and not self.apg_observer.installed

    def _normalize_threshold(self, value: Any) -> float:
        if isinstance(value, (tuple, list)):
            if not value or any(
                _scalar(item, label="norm_threshold") != self.config.norm_threshold
                for item in value
            ):
                raise BraidDualNativeAPGRuntimeError(
                    "sample norm_threshold must contain only 50"
                )
            return self.config.norm_threshold
        observed = _scalar(value, label="norm_threshold")
        if observed != self.config.norm_threshold:
            raise BraidDualNativeAPGRuntimeError("sample norm_threshold differs")
        return observed

    def _validate_sample(self, values: Mapping[str, Any]) -> _ActiveSample:
        videos = values.get("multi_video_vae_latents")
        if isinstance(videos, torch.Tensor):
            videos = [videos]
        if (
            values.get("guidance_mode") != self.config.expected_guidance_mode
            or values.get("num_frames") != self.config.expected_num_frames
            or values.get("num_inference_steps") != self.config.expected_steps
            or _scalar(values.get("flow_shift"), label="flow_shift")
            != self.config.expected_flow_shift
            or _scalar(values.get("omega_txt"), label="omega_txt")
            != self.config.omega_text
            or _scalar(values.get("omega_scale"), label="omega_scale")
            != self.config.omega_scale
            or _scalar(values.get("eta"), label="eta") != self.config.eta
            or _scalar(values.get("momentum"), label="momentum")
            != self.config.momentum
            or values.get("prompt_embeds_t2") is not None
            or values.get("uncond_embeds_t2") is not None
            or not isinstance(videos, (list, tuple))
            or len(videos) != 1
            or values.get("image_vae_latents") is not None
            or values.get("multi_image_vae_latents") is not None
        ):
            raise BraidDualNativeAPGRuntimeError(
                "source-video-only v2v_apg sample contract differs"
            )
        self._normalize_threshold(values.get("norm_threshold"))
        source = _finite_tensor(videos[0], label="source video latent")
        if tuple(source.shape) != tuple(self.config.target_latent_shape):
            raise BraidDualNativeAPGRuntimeError("source video latent geometry differs")
        base = self._validate_prompt_shape(values.get("prompt_embeds"), self.config, label="base/noop")
        negative = self._validate_prompt_shape(values.get("uncond_prompt_embeds"), self.config, label="negative")
        action = self._validate_prompt_shape(self.action_prompt_embeds, self.config, label="action")
        if base.device != action.device or base.dtype != action.dtype or negative.device != action.device or negative.dtype != action.dtype:
            raise BraidDualNativeAPGRuntimeError(
                "base/action/negative prompt dtype or device differs"
            )
        action_buffer = self.vendor_momentum_class(self.config.momentum)
        action_binding = BraidVendorAPGStateBinding.bind_fresh(
            branch="action",
            buffer=action_buffer,
            vendor_class=self.vendor_momentum_class,
            expected_momentum=self.config.momentum,
            observed_initial_running_average=getattr(
                action_buffer, "running_average", None
            ),
        )
        self._action_binding = action_binding
        return _ActiveSample(
            base_prompt=base,
            negative_prompt=negative,
            action_prompt=action,
            action_binding=action_binding,
        )

    def _wrapped_sample(self, *args: Any, **kwargs: Any) -> Any:
        if self._active is not None or self.sample_call_count != 0:
            raise BraidDualNativeAPGRuntimeError(
                "runtime permits exactly one non-nested sample"
            )
        values = _bind(self.original_sample, args, kwargs)
        state = self._validate_sample(values)
        self._active = state
        try:
            result = self.original_sample(*args, **kwargs)
            if (
                state.completed_steps != self.config.expected_steps
                or state.base_forwards
                or state.action_forwards
                or state.base_apg is not None
            ):
                raise BraidDualNativeAPGRuntimeError(
                    "sample returned with an incomplete dual-native step"
                )
            result = _finite_tensor(result, label="sample result")
            if tuple(result.shape) != tuple(self.config.target_latent_shape) or result.dtype != torch.float32:
                raise BraidDualNativeAPGRuntimeError(
                    "sample returned non-native exact81 fp32 latent geometry"
                )
            self.sample_call_count += 1
            return result
        finally:
            if self.block15._step_index is not None:
                self.block15.abort_step()
            self._active = None

    def _validate_forward_call(
        self,
        state: _ActiveSample,
        values: Mapping[str, Any],
        *,
        name: str,
        prompt: torch.Tensor,
        reference: Optional[Mapping[str, Any]],
    ) -> None:
        if values.get("model_id") != self.config.expected_model_id or values.get("cond_embeds") is not prompt:
            raise BraidDualNativeAPGRuntimeError(f"{name} model/prompt differs")
        noisy = _finite_tensor(values.get("noisy_latents"), label=f"{name} noisy")
        timestep = _finite_tensor(values.get("timesteps"), label=f"{name} timestep")
        rotary = _finite_tensor(values.get("rotary_embs"), label=f"{name} rotary")
        if (
            tuple(noisy.shape)
            != (1, self.config.total_v2v_tokens, self.config.expected_hidden_dim)
            or timestep.shape != (1,)
            or rotary.ndim != 4
            or rotary.shape[0] != 1
            or rotary.shape[2] != self.config.total_v2v_tokens
            or _metadata(values.get("batch_vae_seqlen"), label=f"{name} VAE length")
            != (self.config.total_v2v_tokens,)
            or _metadata(values.get("batch_text_seqlen"), label=f"{name} text length")
            != (512,)
        ):
            raise BraidDualNativeAPGRuntimeError(
                f"{name} same-state source+target geometry differs"
            )
        if reference is not None:
            for field_name in ("noisy_latents", "timesteps", "rotary_embs"):
                _same(reference.get(field_name), values.get(field_name), label=f"{name} {field_name}")

    def _validate_prediction(
        self, prediction: Any, *, name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = _finite_tensor(prediction, label=f"{name} prediction")
        expected_channels = self.config.target_latent_shape[1] * 4
        if tuple(tensor.shape) != (
            1,
            self.config.total_v2v_tokens,
            expected_channels,
        ):
            raise BraidDualNativeAPGRuntimeError(f"{name} prediction geometry differs")
        tail = tensor[:, -self.config.target_patch_tokens :, :]
        return tensor, tail

    def _run_forward(
        self,
        *,
        state: _ActiveSample,
        name: str,
        values: Mapping[str, Any],
        prompt: torch.Tensor,
        reference: Mapping[str, Any],
    ) -> _ForwardResult:
        call_values = dict(values)
        call_values["cond_embeds"] = prompt
        bound = _bind(self.original_shared_step, (), call_values)
        self._validate_forward_call(
            state,
            bound,
            name=name,
            prompt=prompt,
            reference=reference,
        )
        with self.block15.leg(name):
            prediction = self.original_shared_step(**call_values)
        checked, tail = self._validate_prediction(prediction, name=name)
        self.transformer_forward_count += 1
        self.action_forward_count += 1
        return _ForwardResult(name=name, values=bound, prediction=checked, target_tail=tail)

    def _wrapped_shared_step(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise BraidDualNativeAPGRuntimeError(
                "shared_step ran outside authenticated sample"
            )
        index = len(state.base_forwards)
        if index >= 2 or state.action_forwards or state.base_apg is not None:
            raise BraidDualNativeAPGRuntimeError(
                "unexpected official shared_step call graph"
            )
        values = _bind(self.original_shared_step, args, kwargs)
        name = "base_negative" if index == 0 else "base_positive"
        prompt = state.negative_prompt if index == 0 else state.base_prompt
        reference = None if index == 0 else state.base_forwards[0].values
        self._validate_forward_call(
            state, values, name=name, prompt=prompt, reference=reference
        )
        if index == 0:
            self.block15.begin_step(
                step_index=state.completed_steps,
                layout=self.layout,
                forward_order=self.config.forward_order,
            )
        with self.block15.leg(name):
            prediction = self.original_shared_step(*args, **kwargs)
        checked, tail = self._validate_prediction(prediction, name=name)
        state.base_forwards.append(
            _ForwardResult(name=name, values=values, prediction=checked, target_tail=tail)
        )
        self.transformer_forward_count += 1
        self.base_forward_count += 1
        if index == 1:
            if self.config.forward_mode == REFERENCE_4F:
                state.action_forwards.append(
                    self._run_forward(
                        state=state,
                        name="action_negative",
                        values=values,
                        prompt=state.negative_prompt,
                        reference=state.base_forwards[0].values,
                    )
                )
            state.action_forwards.append(
                self._run_forward(
                    state=state,
                    name="action_positive",
                    values=values,
                    prompt=state.action_prompt,
                    reference=state.base_forwards[0].values,
                )
            )
        return prediction

    def _observe_base_apg(self, **values: Any) -> None:
        state = self._active
        if (
            state is None
            or len(state.base_forwards) != 2
            or len(state.action_forwards) != self.config.forwards_per_step - 2
            or state.base_apg is not None
        ):
            raise BraidDualNativeAPGRuntimeError(
                "stock APG call occurred outside four/three-forward closure"
            )
        if (
            _scalar(values["guidance_scale"], label="base guidance scale")
            != self.config.omega_text
            or _scalar(values["eta"], label="base eta") != self.config.eta
            or _scalar(values["norm_threshold"], label="base norm threshold")
            != self.config.norm_threshold
        ):
            raise BraidDualNativeAPGRuntimeError("stock base APG parameters differ")
        buffer = values["momentum_buffer"]
        if self._base_binding is None:
            self._base_binding = BraidVendorAPGStateBinding.bind_fresh(
                branch="base",
                buffer=buffer,
                vendor_class=self.vendor_momentum_class,
                expected_momentum=self.config.momentum,
                observed_initial_running_average=values[
                    "before_running_average"
                ],
            )
            if buffer is state.action_binding.buffer:
                raise BraidDualNativeAPGRuntimeError(
                    "base and action APG buffers alias"
                )
        elif buffer is not self._base_binding.buffer:
            raise BraidDualNativeAPGRuntimeError(
                "stock sample replaced its APG buffer mid-trajectory"
            )
        pred_cond = _finite_tensor(values["pred_cond"], label="base APG conditional")
        pred_uncond = _finite_tensor(values["pred_uncond"], label="base APG negative")
        result = _finite_tensor(values["result"], label="base APG result")
        assert self._base_binding is not None
        self._base_binding.observe_completed_call(
            pred_cond=pred_cond, pred_uncond=pred_uncond
        )
        state.base_apg = _VendorAPGObservation(
            pred_cond=pred_cond,
            pred_uncond=pred_uncond,
            guided_clean=result,
            binding=self._base_binding,
        )
        self.base_apg_call_count += 1

    def _clean_from_forward(
        self,
        forward: _ForwardResult,
        *,
        sample_spatial: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        velocity = sgaf._packed_to_spatial(
            forward.target_tail, self.config.target_latent_shape
        )
        return sample_spatial - sigma * velocity

    def _velocity_from_clean(
        self,
        clean: torch.Tensor,
        *,
        sample_spatial: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        return sgaf._spatial_to_packed(
            (sample_spatial - clean) / sigma,
            self.config.target_latent_shape,
        )

    def _wrapped_scheduler_step(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise BraidDualNativeAPGRuntimeError(
                "scheduler.step ran outside authenticated sample"
            )
        expected_action_forwards = self.config.forwards_per_step - 2
        if (
            len(state.base_forwards) != 2
            or len(state.action_forwards) != expected_action_forwards
            or state.base_apg is None
        ):
            raise BraidDualNativeAPGRuntimeError(
                "scheduler.step arrived before dual-native APG closure"
            )
        official = sgaf._extract_argument(args, kwargs, index=0, name="model_output")
        timestep = sgaf._extract_argument(args, kwargs, index=1, name="timestep")
        sample = sgaf._extract_argument(args, kwargs, index=2, name="sample")
        official = _finite_tensor(official, label="official model_output")
        sample = _finite_tensor(sample, label="scheduler sample")
        expected_shape = (
            1,
            self.config.target_patch_tokens,
            self.config.target_latent_shape[1] * 4,
        )
        if tuple(official.shape) != expected_shape or tuple(sample.shape) != expected_shape or official.device != sample.device:
            raise BraidDualNativeAPGRuntimeError(
                "scheduler-bound target packed geometry differs"
            )
        try:
            sgaf._certify_expanded_timestep(
                state.base_forwards[1].values["timesteps"], timestep
            )
            step_index, sigma, sigma_float = sgaf._resolve_sigma(
                self.scheduler, timestep
            )
        except Exception as error:
            raise BraidDualNativeAPGRuntimeError(str(error)) from error
        if step_index != state.completed_steps:
            raise BraidDualNativeAPGRuntimeError("scheduler step index differs")
        if (
            type(sigma) is not torch.Tensor
            or sigma.ndim != 0
            or sigma.device.type != "cpu"
            or sigma.dtype != torch.float32
        ):
            raise BraidDualNativeAPGRuntimeError(
                "active UniPC sigma must remain CPU fp32 scalar"
            )
        sample_spatial = sgaf._packed_to_spatial(
            sample, self.config.target_latent_shape
        )
        base_negative_clean = self._clean_from_forward(
            state.base_forwards[0], sample_spatial=sample_spatial, sigma=sigma
        )
        base_positive_clean = self._clean_from_forward(
            state.base_forwards[1], sample_spatial=sample_spatial, sigma=sigma
        )
        if (
            not torch.equal(base_negative_clean, state.base_apg.pred_uncond)
            or not torch.equal(base_positive_clean, state.base_apg.pred_cond)
        ):
            raise BraidDualNativeAPGRuntimeError(
                "observed stock APG inputs differ from captured native forwards"
            )
        rebuilt_base = self._velocity_from_clean(
            state.base_apg.guided_clean,
            sample_spatial=sample_spatial,
            sigma=sigma,
        )
        parity = rebuilt_base.float() - official.float()
        parity_max = float(parity.abs().max().item())
        parity_rms = float(sgaf._tensor_rms(parity).item())
        if not torch.equal(rebuilt_base, official):
            raise BraidDualNativeAPGRuntimeError(
                "captured vendor base APG differs from stock scheduler output: "
                f"max_abs={parity_max:.9g} rms={parity_rms:.9g}"
            )
        action_negative = (
            state.action_forwards[0]
            if self.config.forward_mode == REFERENCE_4F
            else state.base_forwards[0]
        )
        action_positive = state.action_forwards[-1]
        action_negative_clean = self._clean_from_forward(
            action_negative, sample_spatial=sample_spatial, sigma=sigma
        )
        action_positive_clean = self._clean_from_forward(
            action_positive, sample_spatial=sample_spatial, sigma=sigma
        )
        action_guided = self.vendor_function(
            pred_cond=action_positive_clean,
            pred_uncond=action_negative_clean,
            guidance_scale=self.config.omega_text,
            momentum_buffer=state.action_binding.buffer,
            eta=self.config.eta,
            norm_threshold=self.config.norm_threshold,
        )
        action_guided = _finite_tensor(action_guided, label="action APG result")
        state.action_binding.observe_completed_call(
            pred_cond=action_positive_clean,
            pred_uncond=action_negative_clean,
        )
        self.action_apg_call_count += 1
        action_velocity = self._velocity_from_clean(
            action_guided, sample_spatial=sample_spatial, sigma=sigma
        )
        if (
            tuple(action_velocity.shape) != tuple(official.shape)
            or action_velocity.dtype != official.dtype
            or action_velocity.device != official.device
        ):
            raise BraidDualNativeAPGRuntimeError(
                "action APG scheduler-space geometry differs"
            )
        block_record = self.block15.finish_step()
        # Stage-0 has no authorized mixer.  Preserve both values and object
        # identity by calling the original scheduler with the untouched args.
        result = self.original_scheduler_step(*args, **kwargs)
        self.original_scheduler_call_count += 1
        state.completed_steps += 1
        repeated_negative = action_negative.target_tail
        base_negative = state.base_forwards[0].target_tail
        action_delta = action_velocity.float() - official.float()
        self.trace.append(
            {
                "schema_version": SCHEMA_VERSION,
                "step_index": step_index,
                "timestep": _scalar(timestep, label="timestep"),
                "sigma": sigma_float,
                "forward_mode": self.config.forward_mode,
                "forward_order": list(self.config.forward_order),
                "transformer_forwards": self.config.forwards_per_step,
                "base_forwards": 2,
                "action_forwards": expected_action_forwards,
                "shared_negative": self.config.forward_mode == SHARED_NEGATIVE_3F_DIAGNOSTIC,
                "independent_complete_native_apg_pairs": self.config.forward_mode == REFERENCE_4F,
                "vendor_base_apg_calls": 1,
                "vendor_action_apg_calls": 1,
                "base_action_buffers_distinct": (
                    state.base_apg.binding.buffer is not state.action_binding.buffer
                ),
                "base_stock_apg_exact_parity": True,
                "base_stock_apg_parity_max_abs": parity_max,
                "base_stock_apg_parity_rms": parity_rms,
                "negative_repeat_exact_parity": _raw_bytes_equal(
                    repeated_negative, base_negative
                ),
                "negative_repeat_mismatch_bytes": _mismatch_count(
                    repeated_negative, base_negative
                ),
                "action_base_velocity_delta_rms": float(
                    sgaf._tensor_rms(action_delta).item()
                ),
                "original_scheduler_calls": 1,
                "scheduler_received_stock_base_object": True,
                "block15": asdict(block_record),
            }
        )
        state.base_forwards.clear()
        state.action_forwards.clear()
        state.base_apg = None
        return result

    def finalize(self) -> Mapping[str, Any]:
        if not self.restored or self.finalized:
            raise BraidDualNativeAPGRuntimeError("runtime finalize lifecycle differs")
        steps = self.config.expected_steps
        if (
            self.sample_call_count != 1
            or self.transformer_forward_count != self.config.forwards_per_step * steps
            or self.base_forward_count != 2 * steps
            or self.action_forward_count != (self.config.forwards_per_step - 2) * steps
            or self.base_apg_call_count != steps
            or self.action_apg_call_count != steps
            or self.original_scheduler_call_count != steps
            or len(self.trace) != steps
            or len(self.block15.records) != steps
            or self._base_binding is None
            or self._base_binding.call_count != steps
            or self._action_binding is None
            or self._action_binding.call_count != steps
        ):
            raise BraidDualNativeAPGRuntimeError(
                "dual-native runtime call-count certificate differs"
            )
        action_receipts = {
            row["base_action_buffers_distinct"] for row in self.trace
        }
        if action_receipts != {True}:
            raise BraidDualNativeAPGRuntimeError("APG buffer separation certificate differs")
        # The action binding is retained through the last active sample only in
        # the per-step trace.  Its exact call count is already checked at every
        # call; expose the aggregate without weakening object separation.
        if self._state_versions(self.transformer) != self._state_versions_before:
            raise BraidDualNativeAPGRuntimeError(
                "transformer parameter/buffer identity or version changed"
            )
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "method": "BRAID Stage-0 dual-native APG structural canary",
            "pinned_bernini_commit": PINNED_BERNINI_COMMIT,
            "pinned_wan_diffusion_sha256": PINNED_WAN_DIFFUSION_SHA256,
            "forward_mode": self.config.forward_mode,
            "forward_mode_authority": (
                "four_forward_reference"
                if self.config.forward_mode == REFERENCE_4F
                else "shared_negative_diagnostic_only"
            ),
            "per_step_forward_order": list(self.config.forward_order),
            "steps": steps,
            "transformer_forwards": self.transformer_forward_count,
            "base_forwards": self.base_forward_count,
            "action_forwards": self.action_forward_count,
            "vendor_base_apg_calls": self.base_apg_call_count,
            "vendor_action_apg_calls": self.action_apg_call_count,
            "original_scheduler_calls": self.original_scheduler_call_count,
            "scheduler_execution": "stock_base_V0_exact_object_only",
            "vendor_apg_function": (
                f"{self.vendor_function.__module__}.{self.vendor_function.__name__}"
            ),
            "base_apg_binding": self._base_binding.receipt(),
            "action_apg_binding": self._action_binding.receipt(),
            "layout": self.layout.receipt(),
            "block15": self.block15.receipt(),
            "trace": list(self.trace),
            "parameter_and_buffer_versions_unchanged": True,
            "optimizer_created": False,
            "backward_executed": False,
            "video_decoded": False,
            "checkpoint_read_or_written_by_runtime": False,
            "semantic_action_editing_claim": False,
            "training_authorized": False,
            "runtime_source_identity_enforcement": "external_canary_required",
        }
        self.finalized = True
        return {**receipt, "runtime_digest": _object_sha256(receipt)}


__all__ = [
    "BLOCK15_AUTHORITY",
    "BLOCK_INDEX",
    "BraidBlock15SourceCoStateHook",
    "BraidBlock15StepRecord",
    "BraidDualNativeAPGConfig",
    "BraidDualNativeAPGRuntimeError",
    "BraidDualNativeAPGRuntimePatch",
    "BraidSP4RoleLayout",
    "BraidVendorAPGStateBinding",
    "BraidVendorAPGStateObserver",
    "FORWARD_MODES",
    "PINNED_BERNINI_COMMIT",
    "PINNED_WAN_DIFFUSION_SHA256",
    "REFERENCE_4F",
    "REFERENCE_4F_ORDER",
    "SCHEMA_VERSION",
    "SHARED_NEGATIVE_3F_DIAGNOSTIC",
    "SHARED_NEGATIVE_3F_ORDER",
    "VENDOR_APG_MODULE",
]
