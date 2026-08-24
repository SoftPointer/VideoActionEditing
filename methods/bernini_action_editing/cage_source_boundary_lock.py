#!/usr/bin/env python3
"""Same-state/same-action post-block source boundary lock for CAGE.

The lock is installed with ordinary PyTorch forward hooks; it does not patch
official Bernini source.  For one explicit ``VI_cond`` route and one exact
state/source/action/sigma key, a frozen base forward first captures detached
rank-local non-target rows at every selected block boundary.  One subsequent
student forward consumes those rows once: source and append-padding rows are
replaced by the cached base bits while target rows retain the student graph.

Capture must run with no active CAGE Action-LoRA route.  Student replay must
run under the exact route object declared to this lock.  This makes the extra
base pass same-state and same-action rather than using ``VI_uncond`` as a
surrogate.  The cache is create-once, consume-once, and released after a
complete student pass.  Partial or ambiguous execution poisons and clears it.

The guarantee is limited to the selected post-block boundaries.  It is not a
claim that an arbitrary partial band globally locks all downstream source
memory.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

import torch
from torch import nn

import cage_branch_locked_action_adapter as cage
import inference_sigma_strata as sigma_strata


SCHEMA_VERSION = "bernini-cage-source-boundary-lock-v1"
KEY_SCHEMA_VERSION = "bernini-cage-same-state-action-key-v1"
CACHE_SCHEMA_VERSION = "bernini-cage-source-boundary-cache-v1"

CAPTURE_BASE = "capture_base"
LOCK_STUDENT = "lock_student"
INVOCATION_MODES = (CAPTURE_BASE, LOCK_STUDENT)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class CAGESourceBoundaryLockError(RuntimeError):
    """Raised instead of using an incomplete, stale, or ambiguous cache."""


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
        raise CAGESourceBoundaryLockError(
            f"receipt is not canonical finite ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _closed(value: Any, fields: Iterable[str], *, label: str) -> dict[str, Any]:
    expected = set(fields)
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise CAGESourceBoundaryLockError(
            f"{label} closure differs; missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )
    return dict(value)


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CAGESourceBoundaryLockError(f"{label} must be lowercase SHA-256")
    return value


def _seal(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(unsigned)
    return {**row, "digest": object_sha256(row)}


def _verify_seal(value: Any, fields: Iterable[str], *, label: str) -> dict[str, Any]:
    row = _closed(value, {*fields, "digest"}, label=label)
    unsigned = dict(row)
    digest = _sha256(unsigned.pop("digest"), label=f"{label} digest")
    if object_sha256(unsigned) != digest:
        raise CAGESourceBoundaryLockError(f"{label} digest differs")
    return row


def _tensor_raw_sha256(value: torch.Tensor) -> str:
    work = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(work.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(work.shape)))
    digest.update(work.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate_selected_blocks(value: Any) -> tuple[int, ...]:
    try:
        blocks, _ = cage.validate_registered_module_band(
            value, ("attn2.to_q",)
        )
    except cage.CAGEBranchLockError as error:
        raise CAGESourceBoundaryLockError(
            f"selected boundary block band differs: {error}"
        ) from error
    return blocks


@dataclass(frozen=True)
class CAGESameStateActionKey:
    """Caller-bound identity for one noisy state and the same action condition."""

    state_sha256: str
    source_sha256: str
    action_prompt_sha256: str
    sigma_schedule_index: int

    def __post_init__(self) -> None:
        _sha256(self.state_sha256, label="state SHA-256")
        _sha256(self.source_sha256, label="source SHA-256")
        _sha256(self.action_prompt_sha256, label="action prompt SHA-256")
        try:
            cage.sigma_gate(self.sigma_schedule_index)
        except cage.CAGEBranchLockError as error:
            raise CAGESourceBoundaryLockError(str(error)) from error

    def receipt(self) -> dict[str, Any]:
        unsigned = {
            "schema_version": KEY_SCHEMA_VERSION,
            "state_sha256": self.state_sha256,
            "source_sha256": self.source_sha256,
            "action_prompt_sha256": self.action_prompt_sha256,
            "sigma_schedule_index": self.sigma_schedule_index,
            "sigma_float32_be_hex": sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[
                self.sigma_schedule_index
            ],
            "same_state_required": True,
            "same_source_required": True,
            "same_action_prompt_required": True,
        }
        return _seal(unsigned)

    @property
    def digest(self) -> str:
        return self.receipt()["digest"]


_KEY_FIELDS = {
    "schema_version",
    "state_sha256",
    "source_sha256",
    "action_prompt_sha256",
    "sigma_schedule_index",
    "sigma_float32_be_hex",
    "same_state_required",
    "same_source_required",
    "same_action_prompt_required",
}


def validate_same_state_action_key_receipt(value: Any) -> dict[str, Any]:
    row = _verify_seal(value, _KEY_FIELDS, label="CAGE state/action key")
    key = CAGESameStateActionKey(
        state_sha256=row["state_sha256"],
        source_sha256=row["source_sha256"],
        action_prompt_sha256=row["action_prompt_sha256"],
        sigma_schedule_index=row["sigma_schedule_index"],
    )
    expected = key.receipt()
    if row != expected:
        raise CAGESourceBoundaryLockError(
            "CAGE state/action key does not replay exactly"
        )
    return row


def _validate_route(
    key: CAGESameStateActionKey, route: cage.CAGEBranchLockedRoute
) -> dict[str, Any]:
    if not isinstance(key, CAGESameStateActionKey):
        raise CAGESourceBoundaryLockError("key must be CAGESameStateActionKey")
    if not isinstance(route, cage.CAGEBranchLockedRoute):
        raise CAGESourceBoundaryLockError("route must be CAGEBranchLockedRoute")
    if route.guidance_row != cage.DELTA_GUIDANCE_ROW:
        raise CAGESourceBoundaryLockError(
            "source-boundary capture/lock is VI_cond-only"
        )
    if route.sigma_schedule_index != key.sigma_schedule_index:
        raise CAGESourceBoundaryLockError(
            "route sigma differs from same-state/action key"
        )
    return route.receipt()


def _extract_hidden(output: Any) -> tuple[torch.Tensor, Callable[[torch.Tensor], Any]]:
    if isinstance(output, torch.Tensor):
        return output, lambda replacement: replacement
    if type(output) is tuple and output and isinstance(output[0], torch.Tensor):
        suffix = output[1:]
        return output[0], lambda replacement: (replacement, *suffix)
    raise CAGESourceBoundaryLockError(
        "selected block output must be a Tensor or plain tuple beginning with Tensor"
    )


def _validate_hidden(
    hidden: Any,
    *,
    route: cage.CAGEBranchLockedRoute,
    label: str,
    frozen: bool,
) -> torch.Tensor:
    if (
        not isinstance(hidden, torch.Tensor)
        or hidden.layout != torch.strided
        or hidden.device.type == "meta"
        or not hidden.is_floating_point()
        or hidden.ndim != 3
        or int(hidden.shape[0]) != 1
        or int(hidden.shape[1]) != route.local_length
        or int(hidden.shape[2]) <= 0
    ):
        raise CAGESourceBoundaryLockError(
            f"{label} must be floating [1,local_sequence,hidden]"
        )
    if frozen and (hidden.requires_grad or hidden.grad_fn is not None):
        raise CAGESourceBoundaryLockError(
            f"{label} capture must run under a frozen no-grad base forward"
        )
    return hidden


@dataclass(frozen=True)
class _CachedBoundary:
    block_index: int
    full_shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    non_target_selector: tuple[bool, ...]
    cached_non_target: torch.Tensor
    cached_raw_sha256: str

    def metadata(self) -> dict[str, Any]:
        return {
            "block_index": self.block_index,
            "full_shape": list(self.full_shape),
            "dtype": str(self.dtype),
            "device": str(self.device),
            "non_target_row_count": sum(self.non_target_selector),
            "cached_shape": list(self.cached_non_target.shape),
            "cached_raw_sha256": self.cached_raw_sha256,
            "detached_contiguous": bool(
                not self.cached_non_target.requires_grad
                and self.cached_non_target.grad_fn is None
                and self.cached_non_target.is_contiguous()
            ),
        }


@dataclass(frozen=True)
class CAGESourceBoundaryInvocation:
    mode: str
    key: CAGESameStateActionKey
    route: cage.CAGEBranchLockedRoute
    cache_bank: "CAGESourceBoundaryCacheBank"


_ACTIVE_INVOCATION: ContextVar[Optional[CAGESourceBoundaryInvocation]] = (
    ContextVar("bernini_cage_source_boundary_invocation", default=None)
)


def active_invocation() -> Optional[CAGESourceBoundaryInvocation]:
    return _ACTIVE_INVOCATION.get()


class CAGESourceBoundaryCacheBank:
    """One-use cache for selected local post-block source/padding rows."""

    def __init__(self, selected_block_indices: Sequence[int]) -> None:
        self.selected_block_indices = _validate_selected_blocks(
            selected_block_indices
        )
        self._key: Optional[CAGESameStateActionKey] = None
        self._route: Optional[cage.CAGEBranchLockedRoute] = None
        self._route_receipt: Optional[dict[str, Any]] = None
        self._entries: dict[int, _CachedBoundary] = {}
        self._capture_seen: set[int] = set()
        self._consume_seen: set[int] = set()
        self._phase = "empty"
        self._poisoned = False
        self._retired_key_digests: set[str] = set()
        self._last_success: Optional[dict[str, Any]] = None
        self.completed_pair_count = 0
        self.capture_hook_calls = 0
        self.student_hook_calls = 0

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def cache_released(self) -> bool:
        return self._key is None and self._route is None and not self._entries

    def _require_current(
        self, invocation: CAGESourceBoundaryInvocation, *, mode: str
    ) -> None:
        if active_invocation() is not invocation:
            raise CAGESourceBoundaryLockError(
                "boundary cache access is outside its exact invocation"
            )
        if invocation.cache_bank is not self or invocation.mode != mode:
            raise CAGESourceBoundaryLockError(
                "boundary cache invocation mode/bank differs"
            )
        if self._key != invocation.key or self._route is not invocation.route:
            raise CAGESourceBoundaryLockError(
                "boundary cache state/source/action/sigma/route key differs"
            )
        if self._poisoned:
            raise CAGESourceBoundaryLockError("boundary cache is poisoned")

    def _begin(self, invocation: CAGESourceBoundaryInvocation) -> None:
        route_receipt = _validate_route(invocation.key, invocation.route)
        if invocation.cache_bank is not self:
            raise CAGESourceBoundaryLockError("invocation refers to another cache")
        if self._poisoned:
            raise CAGESourceBoundaryLockError(
                "poisoned boundary cache must be discarded"
            )
        if invocation.mode == CAPTURE_BASE:
            if cage.active_route() is not None:
                raise CAGESourceBoundaryLockError(
                    "capture_base must run with Action-LoRA route absent"
                )
            if invocation.key.digest in self._retired_key_digests:
                raise CAGESourceBoundaryLockError(
                    "same-state/action key was already consumed"
                )
            if self._phase != "empty" or not self.cache_released:
                raise CAGESourceBoundaryLockError(
                    "boundary cache is occupied; overwrite is forbidden"
                )
            self._key = invocation.key
            self._route = invocation.route
            self._route_receipt = route_receipt
            self._entries.clear()
            self._capture_seen.clear()
            self._consume_seen.clear()
            self._phase = "capturing"
            return
        if invocation.mode != LOCK_STUDENT:
            raise CAGESourceBoundaryLockError("unknown boundary invocation mode")
        if self._phase != "captured":
            raise CAGESourceBoundaryLockError(
                "student lock requires one complete base capture"
            )
        if self._key != invocation.key:
            raise CAGESourceBoundaryLockError(
                "student lock is cross-state/source/action/sigma"
            )
        if self._route is not invocation.route or self._route_receipt != route_receipt:
            raise CAGESourceBoundaryLockError(
                "student lock route/SP selector differs from capture"
            )
        if self._consume_seen:
            raise CAGESourceBoundaryLockError(
                "student lock cannot resume a partial consumption"
            )
        self._phase = "locking"

    def _finish(self, invocation: CAGESourceBoundaryInvocation) -> None:
        expected = set(self.selected_block_indices)
        if invocation.mode == CAPTURE_BASE:
            if self._capture_seen != expected or set(self._entries) != expected:
                missing = sorted(expected - self._capture_seen)
                raise CAGESourceBoundaryLockError(
                    f"base capture missed selected blocks {missing}"
                )
            self._phase = "captured"
            return
        if self._consume_seen != expected:
            missing = sorted(expected - self._consume_seen)
            raise CAGESourceBoundaryLockError(
                f"student lock missed selected blocks {missing}"
            )
        assert self._key is not None and self._route_receipt is not None
        entry_metadata = [
            self._entries[index].metadata()
            for index in self.selected_block_indices
        ]
        self._last_success = {
            "key_digest": self._key.digest,
            "route_digest": self._route_receipt["digest"],
            "selected_block_indices": list(self.selected_block_indices),
            "entry_metadata": entry_metadata,
            "all_blocks_captured_once": True,
            "all_blocks_consumed_once": True,
            "source_and_padding_rows_base_raw_byte_exact": True,
            "target_rows_student_graph_preserved": True,
        }
        self._retired_key_digests.add(self._key.digest)
        self.completed_pair_count += 1
        self._clear_current()

    def _clear_current(self) -> None:
        self._entries.clear()
        self._capture_seen.clear()
        self._consume_seen.clear()
        self._key = None
        self._route = None
        self._route_receipt = None
        self._phase = "empty"

    def _abort(self) -> None:
        if self._key is not None:
            self._retired_key_digests.add(self._key.digest)
        self._clear_current()
        self._poisoned = True

    @contextmanager
    def invocation(
        self,
        *,
        mode: str,
        key: CAGESameStateActionKey,
        route: cage.CAGEBranchLockedRoute,
    ) -> Iterator[CAGESourceBoundaryInvocation]:
        if active_invocation() is not None:
            raise CAGESourceBoundaryLockError(
                "nested source-boundary invocations are forbidden"
            )
        invocation = CAGESourceBoundaryInvocation(
            mode=mode, key=key, route=route, cache_bank=self
        )
        self._begin(invocation)
        token: Token[Optional[CAGESourceBoundaryInvocation]] = (
            _ACTIVE_INVOCATION.set(invocation)
        )
        try:
            try:
                yield invocation
                self._finish(invocation)
            except BaseException:
                self._abort()
                raise
        finally:
            _ACTIVE_INVOCATION.reset(token)

    def _validate_block(self, block_index: Any) -> int:
        if (
            isinstance(block_index, bool)
            or not isinstance(block_index, int)
            or block_index not in self.selected_block_indices
        ):
            raise CAGESourceBoundaryLockError(
                "boundary hook block lies outside selected continuous band"
            )
        return block_index

    def capture_boundary(
        self,
        *,
        invocation: CAGESourceBoundaryInvocation,
        block_index: int,
        hidden: torch.Tensor,
    ) -> None:
        self._require_current(invocation, mode=CAPTURE_BASE)
        if cage.active_route() is not None:
            raise CAGESourceBoundaryLockError(
                "base capture observed an active Action-LoRA route"
            )
        index = self._validate_block(block_index)
        if index in self._capture_seen or index in self._entries:
            raise CAGESourceBoundaryLockError(
                f"block {index} base boundary was captured twice"
            )
        checked = _validate_hidden(
            hidden,
            route=invocation.route,
            label=f"block {index} base boundary",
            frozen=True,
        )
        target_selector = invocation.route.local_target_selector(
            device=checked.device
        )
        non_target_selector = ~target_selector
        cached = checked[:, non_target_selector, :].detach().clone().contiguous()
        if cached.requires_grad or cached.grad_fn is not None:
            raise CAGESourceBoundaryLockError("captured boundary retains autograd")
        if not cage.tensors_byte_exact(
            cached, checked[:, non_target_selector, :]
        ):
            raise CAGESourceBoundaryLockError(
                "detached source/padding capture changed raw bits"
            )
        self._entries[index] = _CachedBoundary(
            block_index=index,
            full_shape=tuple(int(item) for item in checked.shape),
            dtype=checked.dtype,
            device=checked.device,
            non_target_selector=tuple(
                bool(item) for item in non_target_selector.cpu().tolist()
            ),
            cached_non_target=cached,
            cached_raw_sha256=_tensor_raw_sha256(cached),
        )
        self._capture_seen.add(index)
        self.capture_hook_calls += 1

    def lock_boundary(
        self,
        *,
        invocation: CAGESourceBoundaryInvocation,
        block_index: int,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        self._require_current(invocation, mode=LOCK_STUDENT)
        if cage.active_route() is not invocation.route:
            raise CAGESourceBoundaryLockError(
                "student boundary must share the exact active CAGE route object"
            )
        index = self._validate_block(block_index)
        if index in self._consume_seen:
            raise CAGESourceBoundaryLockError(
                f"block {index} student boundary was consumed twice"
            )
        checked = _validate_hidden(
            hidden,
            route=invocation.route,
            label=f"block {index} student boundary",
            frozen=False,
        )
        entry = self._entries.get(index)
        if entry is None:
            raise CAGESourceBoundaryLockError(
                f"block {index} has no captured base boundary"
            )
        if (
            tuple(int(item) for item in checked.shape) != entry.full_shape
            or checked.dtype != entry.dtype
            or checked.device != entry.device
        ):
            raise CAGESourceBoundaryLockError(
                f"block {index} base/student shape, dtype, or device differs"
            )
        target_selector = invocation.route.local_target_selector(
            device=checked.device
        )
        non_target_selector = ~target_selector
        selector_tuple = tuple(
            bool(item) for item in non_target_selector.cpu().tolist()
        )
        if selector_tuple != entry.non_target_selector:
            raise CAGESourceBoundaryLockError(
                f"block {index} source/padding selector differs"
            )
        current_target = checked[:, target_selector, :]
        result = checked.clone()
        result[:, non_target_selector, :] = entry.cached_non_target
        if not cage.tensors_byte_exact(
            result[:, non_target_selector, :], entry.cached_non_target
        ):
            raise CAGESourceBoundaryLockError(
                f"block {index} source/padding reinjection changed base bits"
            )
        if not cage.tensors_byte_exact(
            result[:, target_selector, :], current_target
        ):
            raise CAGESourceBoundaryLockError(
                f"block {index} reinjection changed student target bits"
            )
        self._consume_seen.add(index)
        self.student_hook_calls += 1
        return result

    def discard(self) -> None:
        if active_invocation() is not None:
            raise CAGESourceBoundaryLockError(
                "cannot discard cache during an active invocation"
            )
        if self._key is not None:
            self._retired_key_digests.add(self._key.digest)
        self._clear_current()
        self._poisoned = False

    def receipt(self) -> dict[str, Any]:
        unsigned = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "selected_block_indices": list(self.selected_block_indices),
            "phase": self._phase,
            "poisoned": self._poisoned,
            "cache_entry_count": len(self._entries),
            "cache_released": self.cache_released,
            "capture_hook_calls": self.capture_hook_calls,
            "student_hook_calls": self.student_hook_calls,
            "completed_pair_count": self.completed_pair_count,
            "retired_key_count": len(self._retired_key_digests),
            "last_success": self._last_success,
            "create_once_per_block": True,
            "consume_once_per_block": True,
            "successful_pair_releases_cache": True,
        }
        return _seal(unsigned)


class _BoundaryHook:
    def __init__(
        self, *, block_index: int, cache_bank: CAGESourceBoundaryCacheBank
    ) -> None:
        self.block_index = cache_bank._validate_block(block_index)
        self.cache_bank = cache_bank
        self.capture_calls = 0
        self.lock_calls = 0
        self.passthrough_calls = 0
        self._installed_hook_id: Optional[int] = None

    def bind(self, block: nn.Module, handle: Any) -> None:
        identifier = getattr(handle, "id", None)
        registry = getattr(block, "_forward_hooks", None)
        if (
            self._installed_hook_id is not None
            or type(identifier) is not int
            or not isinstance(registry, Mapping)
            or len(registry) != 1
            or registry.get(identifier) is not self
        ):
            raise CAGESourceBoundaryLockError(
                "installed boundary hook identity cannot be proven"
            )
        self._installed_hook_id = identifier

    def audit(self, block: nn.Module) -> None:
        registry = getattr(block, "_forward_hooks", None)
        if (
            self._installed_hook_id is None
            or not isinstance(registry, Mapping)
            or len(registry) != 1
            or registry.get(self._installed_hook_id) is not self
        ):
            raise CAGESourceBoundaryLockError(
                "boundary hook registry/order differs"
            )

    def __call__(self, module: nn.Module, inputs: Any, output: Any) -> Any:
        self.audit(module)
        del inputs
        invocation = active_invocation()
        if invocation is None:
            self.passthrough_calls += 1
            return None
        if invocation.cache_bank is not self.cache_bank:
            raise CAGESourceBoundaryLockError(
                "boundary hook and invocation cache differ"
            )
        hidden, rebuild = _extract_hidden(output)
        if invocation.mode == CAPTURE_BASE:
            self.cache_bank.capture_boundary(
                invocation=invocation,
                block_index=self.block_index,
                hidden=hidden,
            )
            self.capture_calls += 1
            return None
        if invocation.mode != LOCK_STUDENT:
            raise CAGESourceBoundaryLockError("boundary hook mode differs")
        locked = self.cache_bank.lock_boundary(
            invocation=invocation,
            block_index=self.block_index,
            hidden=hidden,
        )
        self.lock_calls += 1
        return rebuild(locked)

    def statistics(self) -> dict[str, Any]:
        return {
            "block_index": self.block_index,
            "capture_calls": self.capture_calls,
            "lock_calls": self.lock_calls,
            "passthrough_calls": self.passthrough_calls,
            "installed_hook_id": self._installed_hook_id,
        }


def _parameter_closure(module: nn.Module) -> dict[str, Any]:
    parameters = [
        {
            "name": name,
            "object_id": id(parameter),
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in module.named_parameters()
    ]
    state_keys = list(module.state_dict().keys())
    value = {"parameters": parameters, "state_dict_keys": state_keys}
    return {**value, "digest": object_sha256(value)}


@dataclass
class CAGESourceBoundaryLockHandle:
    transformer: nn.Module
    selected_block_indices: tuple[int, ...]
    cache_bank: CAGESourceBoundaryCacheBank
    hooks: tuple[_BoundaryHook, ...]
    hook_handles: tuple[Any, ...]
    original_block_ids: tuple[tuple[int, int], ...]
    parameter_closure: dict[str, Any]
    restored: bool = False

    def assert_scope(self) -> None:
        if self.restored:
            raise CAGESourceBoundaryLockError("source-boundary hooks are restored")
        blocks = getattr(self.transformer, "blocks", None)
        if blocks is None:
            raise CAGESourceBoundaryLockError("transformer blocks disappeared")
        if _parameter_closure(self.transformer) != self.parameter_closure:
            raise CAGESourceBoundaryLockError(
                "hook installation changed parameter/state_dict closure"
            )
        block_ids = dict(self.original_block_ids)
        for index, hook in zip(self.selected_block_indices, self.hooks):
            block = blocks[index]
            if id(block) != block_ids[index]:
                raise CAGESourceBoundaryLockError(
                    f"selected block {index} identity changed"
                )
            hook.audit(block)

    @contextmanager
    def capture_base(
        self,
        *,
        key: CAGESameStateActionKey,
        route: cage.CAGEBranchLockedRoute,
    ) -> Iterator[CAGESourceBoundaryInvocation]:
        self.assert_scope()
        with self.cache_bank.invocation(
            mode=CAPTURE_BASE, key=key, route=route
        ) as invocation:
            yield invocation

    @contextmanager
    def lock_student(
        self,
        *,
        key: CAGESameStateActionKey,
        route: cage.CAGEBranchLockedRoute,
    ) -> Iterator[CAGESourceBoundaryInvocation]:
        self.assert_scope()
        with self.cache_bank.invocation(
            mode=LOCK_STUDENT, key=key, route=route
        ) as invocation:
            yield invocation

    def discard_cache(self) -> None:
        self.cache_bank.discard()

    def restore(self) -> None:
        if self.restored or active_invocation() is not None:
            raise CAGESourceBoundaryLockError(
                "source-boundary hooks cannot be restored now"
            )
        if not self.cache_bank.cache_released or self.cache_bank.poisoned:
            raise CAGESourceBoundaryLockError(
                "release/discard source-boundary cache before restore"
            )
        self.assert_scope()
        for hook_handle in reversed(self.hook_handles):
            hook_handle.remove()
        self.restored = True

    def receipt(self) -> dict[str, Any]:
        self.assert_scope()
        unsigned = {
            "schema_version": SCHEMA_VERSION,
            "selected_block_indices": list(self.selected_block_indices),
            "selected_blocks_are_continuous": True,
            "integration": {
                "official_bernini_source_modified": False,
                "mechanism": "torch_post_block_forward_hooks",
                "hook_count": len(self.hooks),
                "selected_boundary_scope_only": True,
                "global_source_memory_lock_claim": False,
            },
            "same_state_action_contract": {
                "key_fields": [
                    "state_sha256",
                    "source_sha256",
                    "action_prompt_sha256",
                    "sigma_schedule_index",
                ],
                "capture_declared_guidance_row": "VI_cond",
                "capture_action_lora_route_active": False,
                "student_guidance_row": "VI_cond",
                "student_requires_exact_active_route_object": True,
                "route_receipt_binds_SP_selector_and_token_geometry": True,
            },
            "row_contract": {
                "cached_rows": "local_source_and_append_padding_non_target_rows",
                "capture_tensor": "detach_clone_contiguous",
                "student_source_and_padding": "base_raw_byte_exact",
                "student_target": "unchanged_bits_and_autograd_graph",
                "same_shape_dtype_device_required": True,
            },
            "cache_contract": {
                "create_once_per_selected_block": True,
                "consume_once_per_selected_block": True,
                "complete_block_coverage_required": True,
                "successful_pair_releases_cache": True,
                "reuse_and_cross_key_fail_closed": True,
                "gradient_checkpoint_recomputation_supported": False,
            },
            "training_inference_forward_contract": {
                "same_contract_in_training_and_inference": True,
                "native_guidance_forward_count": 4,
                "additional_same_action_base_VI_cond_forward_count": 1,
                "total_denoiser_forward_count": 5,
                "execution_requires_base_capture_before_student_VI_cond": True,
                "denoiser_forward_compute_multiplier": 1.25,
                "cache_tensor_count_per_rank": len(self.selected_block_indices),
                "cache_elements": (
                    "sum_selected_blocks(local_non_target_rows*hidden_width)"
                ),
            },
            "parameter_closure": self.parameter_closure,
            "hook_statistics": [hook.statistics() for hook in self.hooks],
            "cache": self.cache_bank.receipt(),
            "optimizer_authorized": False,
            "semantic_action_editing_claim": False,
        }
        return _seal(unsigned)


def install_cage_source_boundary_lock(
    transformer: nn.Module,
    *,
    selected_block_indices: Sequence[int],
) -> CAGESourceBoundaryLockHandle:
    """Install auditable post-block hooks on one explicit continuous band."""

    if not isinstance(transformer, nn.Module):
        raise CAGESourceBoundaryLockError("transformer must be nn.Module")
    indices = _validate_selected_blocks(selected_block_indices)
    blocks = getattr(transformer, "blocks", None)
    if blocks is None or indices[-1] >= len(blocks):
        raise CAGESourceBoundaryLockError(
            "selected boundary block lies outside transformer"
        )
    parameter_closure = _parameter_closure(transformer)
    bank = CAGESourceBoundaryCacheBank(indices)
    hooks: list[_BoundaryHook] = []
    handles: list[Any] = []
    block_ids: list[tuple[int, int]] = []
    try:
        for index in indices:
            block = blocks[index]
            register_hook = getattr(block, "register_forward_hook", None)
            registry = getattr(block, "_forward_hooks", None)
            if not callable(register_hook) or not isinstance(registry, Mapping):
                raise CAGESourceBoundaryLockError(
                    f"block {index} is not auditable/hookable"
                )
            if registry:
                raise CAGESourceBoundaryLockError(
                    f"block {index} already has forward hooks"
                )
            hook = _BoundaryHook(block_index=index, cache_bank=bank)
            handle = register_hook(hook)
            try:
                hook.bind(block, handle)
            except BaseException:
                handle.remove()
                raise
            hooks.append(hook)
            handles.append(handle)
            block_ids.append((index, id(block)))
    except BaseException:
        for handle in reversed(handles):
            handle.remove()
        raise

    result = CAGESourceBoundaryLockHandle(
        transformer=transformer,
        selected_block_indices=indices,
        cache_bank=bank,
        hooks=tuple(hooks),
        hook_handles=tuple(handles),
        original_block_ids=tuple(block_ids),
        parameter_closure=parameter_closure,
    )
    try:
        result.assert_scope()
    except BaseException:
        for handle in reversed(handles):
            handle.remove()
        result.restored = True
        raise
    return result


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CAPTURE_BASE",
    "CAGESameStateActionKey",
    "CAGESourceBoundaryCacheBank",
    "CAGESourceBoundaryInvocation",
    "CAGESourceBoundaryLockError",
    "CAGESourceBoundaryLockHandle",
    "INVOCATION_MODES",
    "KEY_SCHEMA_VERSION",
    "LOCK_STUDENT",
    "SCHEMA_VERSION",
    "active_invocation",
    "canonical_json_bytes",
    "install_cage_source_boundary_lock",
    "object_sha256",
    "validate_same_state_action_key_receipt",
]
