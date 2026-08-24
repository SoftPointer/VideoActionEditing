#!/usr/bin/env python3
"""CAGE Action-LoRA with an explicit native-guidance branch lock.

The older :mod:`pair_v5_action_adapter` gates an Action-LoRA by target row,
sigma, and module site, but its route identifies only a native visual pack.
CAGE adds the missing four-way guidance-row coordinate.  A LoRA residual is
authorized by one closed conjunction only::

    guidance_row == "VI_cond"
    and row_kind == "target"
    and sigma_schedule_index in the registered high/mid band
    and module_site in an explicit post-probe continuous block/projection band

``empty_uncond``, ``V_uncond``, and ``VI_uncond`` return the base projection
directly.  The same direct return is used for low sigma and modules outside
the registered band.  Within ``VI_cond``, source/reference and append-padding
rows remain byte-exact copies of the base output.

That guarantee is projection-local.  It does not claim that downstream joint
self-attention cannot mix changed target states back into source memory, and
it is not a substitute for source-memory capture/re-injection.

The installer has no default module band: a caller must supply a non-empty
continuous block interval and an explicit Q/O projection subset selected by
the external all-30-block probe.  The low-level wrapper itself accepts every
block index 0..29 so the probe does not inherit a hidden 0..22 prior.

This is a minimal routing/adapter primitive.  It does not authorize an
optimizer update or make a semantic action-editing claim.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

import torch
from torch import nn

import inference_sigma_strata as sigma_strata
import pair_v5_action_adapter as pair_v5


SCHEMA_VERSION = "bernini-cage-branch-locked-action-adapter-v1"
ROUTE_SCHEMA_VERSION = "bernini-cage-branch-locked-route-v1"
AUDIT_SCHEMA_VERSION = "bernini-cage-branch-lock-audit-v1"

TOTAL_BLOCKS_1P3B = pair_v5.TOTAL_BLOCKS_1P3B
PROBE_BLOCK_INDICES = tuple(range(TOTAL_BLOCKS_1P3B))
CAGE_PROJECTIONS = ("attn2.to_q", "attn2.to_out.0")
CAGE_LORA_RANK = pair_v5.ACTION_LORA_RANK
CAGE_LORA_ALPHA = pair_v5.ACTION_LORA_ALPHA
CAGE_LORA_DROPOUT = pair_v5.ACTION_LORA_DROPOUT
ALLOWED_SP_SIZES = pair_v5.ALLOWED_SP_SIZES

GUIDANCE_ROWS = ("empty_uncond", "V_uncond", "VI_uncond", "VI_cond")
DELTA_GUIDANCE_ROW = "VI_cond"
LOCKED_GUIDANCE_ROWS = GUIDANCE_ROWS[:-1]
GUIDANCE_TO_NATIVE_BRANCH = {
    "empty_uncond": "none",
    "V_uncond": "V",
    "VI_uncond": "VI",
    "VI_cond": "VI",
}

HIGH_SIGMA_INDICES = pair_v5.HIGH_SIGMA_INDICES
MID_SIGMA_INDICES = pair_v5.MID_SIGMA_INDICES
LOW_SIGMA_INDICES = pair_v5.LOW_SIGMA_INDICES
DELTA_SIGMA_INDICES = HIGH_SIGMA_INDICES + MID_SIGMA_INDICES


class CAGEBranchLockError(RuntimeError):
    """Raised before an ambiguous or branch-leaking CAGE route is used."""


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
        raise CAGEBranchLockError(
            f"audit value is not canonical finite ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_registered_module_band(
    registered_block_indices: Any,
    registered_projections: Any,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Validate one explicit, non-empty, continuous post-probe module band."""

    if not isinstance(registered_block_indices, (tuple, list)):
        raise CAGEBranchLockError(
            "registered_block_indices must be an explicit tuple/list"
        )
    blocks = tuple(registered_block_indices)
    if not blocks:
        raise CAGEBranchLockError("registered block band cannot be empty")
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < TOTAL_BLOCKS_1P3B
        for index in blocks
    ):
        raise CAGEBranchLockError(
            f"registered block index must lie in [0,{TOTAL_BLOCKS_1P3B - 1}]"
        )
    if blocks != tuple(sorted(set(blocks))):
        raise CAGEBranchLockError(
            "registered block band must be unique and ascending"
        )
    if blocks != tuple(range(blocks[0], blocks[-1] + 1)):
        raise CAGEBranchLockError("registered block band must be continuous")

    if not isinstance(registered_projections, (tuple, list)):
        raise CAGEBranchLockError(
            "registered_projections must be an explicit tuple/list"
        )
    projections = tuple(registered_projections)
    if not projections:
        raise CAGEBranchLockError("registered projection band cannot be empty")
    if any(projection not in CAGE_PROJECTIONS for projection in projections):
        raise CAGEBranchLockError(
            "registered projection band may contain only attn2 Q/O"
        )
    canonical = tuple(
        projection for projection in CAGE_PROJECTIONS if projection in projections
    )
    if projections != canonical or len(set(projections)) != len(projections):
        raise CAGEBranchLockError(
            "registered projection band must be unique and canonical"
        )
    return blocks, projections


def registered_module_band_receipt(
    registered_block_indices: Sequence[int],
    registered_projections: Sequence[str],
) -> dict[str, Any]:
    blocks, projections = validate_registered_module_band(
        registered_block_indices, registered_projections
    )
    sites = [
        f"blocks.{index}.{projection}"
        for index in blocks
        for projection in projections
    ]
    unsigned = {
        "block_indices": list(blocks),
        "projections": list(projections),
        "module_sites": sites,
        "continuous_block_band": True,
        "explicit_no_default": True,
    }
    return {**unsigned, "digest": object_sha256(unsigned)}


def validate_registered_module_band_receipt(value: Any) -> dict[str, Any]:
    row = _closed(
        value,
        {
            "block_indices",
            "projections",
            "module_sites",
            "continuous_block_band",
            "explicit_no_default",
            "digest",
        },
        label="CAGE registered module band",
    )
    _verify_digest(row, label="CAGE registered module band")
    expected = registered_module_band_receipt(
        row["block_indices"], row["projections"]
    )
    if row != expected:
        raise CAGEBranchLockError("CAGE registered module band does not replay")
    return row


def _closed(value: Any, fields: Iterable[str], *, label: str) -> dict[str, Any]:
    expected = set(fields)
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise CAGEBranchLockError(
            f"{label} closure differs; missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )
    return dict(value)


def _verify_digest(row: Mapping[str, Any], *, label: str) -> str:
    unsigned = dict(row)
    digest = unsigned.pop("digest", None)
    if not isinstance(digest, str) or len(digest) != 64:
        raise CAGEBranchLockError(f"{label} digest differs")
    if object_sha256(unsigned) != digest:
        raise CAGEBranchLockError(f"{label} digest differs")
    return digest


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CAGEBranchLockError(f"{label} must be a positive integer")
    return value


def sigma_gate(schedule_index: Any) -> tuple[str, float]:
    """CAGE branch lock: unit gate at 0..37 and direct base at 38..39."""

    try:
        pair_v5.sigma_gate(schedule_index)
    except pair_v5.PairV5ActionAdapterError as error:
        raise CAGEBranchLockError(str(error)) from error
    if schedule_index in DELTA_SIGMA_INDICES:
        return "allowed_unit", 1.0
    if schedule_index in LOW_SIGMA_INDICES:
        return "low_base_only", 0.0
    raise CAGEBranchLockError("sigma_schedule_index is not preregistered")


def module_in_delta_band(
    block_index: Any,
    projection: Any,
    *,
    registered_block_indices: Sequence[int],
    registered_projections: Sequence[str],
) -> bool:
    """Return whether a Bernini module coordinate is CAGE-trainable."""

    if (
        isinstance(block_index, bool)
        or not isinstance(block_index, int)
        or not 0 <= block_index < TOTAL_BLOCKS_1P3B
    ):
        raise CAGEBranchLockError(
            f"block_index must be an exact integer in [0,{TOTAL_BLOCKS_1P3B - 1}]"
        )
    if not isinstance(projection, str) or not projection:
        raise CAGEBranchLockError("projection must be a non-empty string")
    blocks, projections = validate_registered_module_band(
        registered_block_indices, registered_projections
    )
    return block_index in blocks and projection in projections


def tensors_byte_exact(left: Any, right: Any) -> bool:
    """Compare tensor metadata and contiguous storage bytes, including -0/+0."""

    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return False
    if (
        left.dtype != right.dtype
        or left.device != right.device
        or left.layout != torch.strided
        or right.layout != torch.strided
        or tuple(left.shape) != tuple(right.shape)
    ):
        return False
    left_bytes = left.detach().contiguous().view(torch.uint8)
    right_bytes = right.detach().contiguous().view(torch.uint8)
    return bool(torch.equal(left_bytes, right_bytes))


def assert_tensors_byte_exact(left: Any, right: Any, *, label: str) -> None:
    if not tensors_byte_exact(left, right):
        raise CAGEBranchLockError(f"{label} is not byte-exact base output")


@dataclass(frozen=True)
class CAGEBranchLockedRoute:
    """One explicit row of Bernini's four-forward guidance decomposition."""

    total_tokens: int
    condition_tokens: int
    sequence_parallel_rank: int
    sequence_parallel_size: int
    guidance_row: str
    sigma_schedule_index: int

    def __post_init__(self) -> None:
        total = _positive_int(self.total_tokens, label="total_tokens")
        if (
            isinstance(self.condition_tokens, bool)
            or not isinstance(self.condition_tokens, int)
            or not 0 <= self.condition_tokens < total
        ):
            raise CAGEBranchLockError(
                "condition_tokens must identify a strict noisy-target suffix"
            )
        if self.guidance_row not in GUIDANCE_ROWS:
            raise CAGEBranchLockError(
                "guidance_row must be exactly empty_uncond/V_uncond/VI_uncond/VI_cond"
            )
        if self.guidance_row == "empty_uncond" and self.condition_tokens != 0:
            raise CAGEBranchLockError(
                "empty_uncond cannot contain source/reference condition rows"
            )
        if self.guidance_row != "empty_uncond" and self.condition_tokens == 0:
            raise CAGEBranchLockError(
                "conditioned guidance rows require source/reference condition rows"
            )
        size = _positive_int(
            self.sequence_parallel_size, label="sequence_parallel_size"
        )
        if size not in ALLOWED_SP_SIZES:
            raise CAGEBranchLockError("only SP1 tests and native SP4 are supported")
        rank = self.sequence_parallel_rank
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < size:
            raise CAGEBranchLockError("SP rank lies outside its group")
        sigma_gate(self.sigma_schedule_index)

    @property
    def native_branch(self) -> str:
        return GUIDANCE_TO_NATIVE_BRANCH[self.guidance_row]

    @property
    def target_tokens(self) -> int:
        return self.total_tokens - self.condition_tokens

    @property
    def local_length(self) -> int:
        return math.ceil(self.total_tokens / self.sequence_parallel_size)

    @property
    def gate_name(self) -> str:
        return sigma_gate(self.sigma_schedule_index)[0]

    @property
    def gate_weight(self) -> float:
        if self.guidance_row != DELTA_GUIDANCE_ROW:
            return 0.0
        return sigma_gate(self.sigma_schedule_index)[1]

    @property
    def branch_and_sigma_active(self) -> bool:
        return self.gate_weight > 0.0

    def authorizes_module(
        self,
        block_index: int,
        projection: str,
        *,
        registered_block_indices: Sequence[int],
        registered_projections: Sequence[str],
    ) -> bool:
        return self.branch_and_sigma_active and module_in_delta_band(
            block_index,
            projection,
            registered_block_indices=registered_block_indices,
            registered_projections=registered_projections,
        )

    def global_target_selector(self, *, device: torch.device) -> torch.Tensor:
        return torch.cat(
            (
                torch.zeros(self.condition_tokens, dtype=torch.bool, device=device),
                torch.ones(self.target_tokens, dtype=torch.bool, device=device),
            )
        )

    def local_target_selector(self, *, device: torch.device) -> torch.Tensor:
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

    def receipt(self) -> dict[str, Any]:
        unsigned = {
            "schema_version": ROUTE_SCHEMA_VERSION,
            "guidance_row": self.guidance_row,
            "guidance_row_order": list(GUIDANCE_ROWS),
            "derived_native_branch": self.native_branch,
            "delta_guidance_row": DELTA_GUIDANCE_ROW,
            "locked_guidance_rows": list(LOCKED_GUIDANCE_ROWS),
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
            "sigma_gate_weight_after_branch_lock": self.gate_weight,
            "branch_and_sigma_active": self.branch_and_sigma_active,
            "source_rows_delta_authorized": False,
            "target_rows_only": True,
        }
        return {**unsigned, "digest": object_sha256(unsigned)}


_ROUTE_RECEIPT_FIELDS = {
    "schema_version",
    "guidance_row",
    "guidance_row_order",
    "derived_native_branch",
    "delta_guidance_row",
    "locked_guidance_rows",
    "total_tokens",
    "condition_tokens",
    "target_tokens",
    "sequence_parallel_rank",
    "sequence_parallel_size",
    "padding_policy",
    "sigma_schedule_index",
    "sigma_float32_be_hex",
    "sigma_gate",
    "sigma_gate_weight_after_branch_lock",
    "branch_and_sigma_active",
    "source_rows_delta_authorized",
    "target_rows_only",
    "digest",
}


def validate_route_receipt(value: Any) -> dict[str, Any]:
    row = _closed(value, _ROUTE_RECEIPT_FIELDS, label="CAGE route receipt")
    _verify_digest(row, label="CAGE route receipt")
    try:
        route = CAGEBranchLockedRoute(
            total_tokens=row["total_tokens"],
            condition_tokens=row["condition_tokens"],
            sequence_parallel_rank=row["sequence_parallel_rank"],
            sequence_parallel_size=row["sequence_parallel_size"],
            guidance_row=row["guidance_row"],
            sigma_schedule_index=row["sigma_schedule_index"],
        )
    except (KeyError, CAGEBranchLockError) as error:
        raise CAGEBranchLockError(f"CAGE route receipt differs: {error}") from error
    expected = route.receipt()
    if row != expected:
        raise CAGEBranchLockError("CAGE route receipt does not replay exactly")
    return row


def make_branch_lock_audit_receipt(
    route: CAGEBranchLockedRoute,
    *,
    block_index: int,
    projection: str,
    registered_block_indices: Sequence[int],
    registered_projections: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(route, CAGEBranchLockedRoute):
        raise CAGEBranchLockError("audit route must be CAGEBranchLockedRoute")
    band = registered_module_band_receipt(
        registered_block_indices, registered_projections
    )
    module_allowed = module_in_delta_band(
        block_index,
        projection,
        registered_block_indices=band["block_indices"],
        registered_projections=band["projections"],
    )
    predicates = {
        "guidance_row_is_VI_cond": route.guidance_row == DELTA_GUIDANCE_ROW,
        "sigma_band_allows_delta": route.sigma_schedule_index in DELTA_SIGMA_INDICES,
        "module_band_allows_delta": module_allowed,
        "target_row_selector_nonempty": route.target_tokens > 0,
    }
    target_authorized = all(predicates.values())
    unsigned = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "route_receipt": route.receipt(),
        "block_index": block_index,
        "projection": projection,
        "registered_module_band": band,
        "authorization_conjunction": [
            "guidance_row_is_VI_cond",
            "sigma_band_allows_delta",
            "module_band_allows_delta",
            "target_row_selector_nonempty",
        ],
        "predicates": predicates,
        "source_rows_delta_authorized": False,
        "padding_rows_delta_authorized": False,
        "target_rows_delta_authorized": target_authorized,
        "full_projection_direct_base_required": not target_authorized,
        "locked_guidance_rows_direct_base": True,
        "low_sigma_direct_base": True,
        "outside_module_band_direct_base": True,
        "optimizer_authorized": False,
        "semantic_action_claim": False,
    }
    return {**unsigned, "digest": object_sha256(unsigned)}


_AUDIT_FIELDS = {
    "schema_version",
    "route_receipt",
    "block_index",
    "projection",
    "registered_module_band",
    "authorization_conjunction",
    "predicates",
    "source_rows_delta_authorized",
    "padding_rows_delta_authorized",
    "target_rows_delta_authorized",
    "full_projection_direct_base_required",
    "locked_guidance_rows_direct_base",
    "low_sigma_direct_base",
    "outside_module_band_direct_base",
    "optimizer_authorized",
    "semantic_action_claim",
    "digest",
}


def validate_branch_lock_audit_receipt(value: Any) -> dict[str, Any]:
    row = _closed(value, _AUDIT_FIELDS, label="CAGE branch-lock audit receipt")
    _verify_digest(row, label="CAGE branch-lock audit receipt")
    route_row = validate_route_receipt(row["route_receipt"])
    route = CAGEBranchLockedRoute(
        total_tokens=route_row["total_tokens"],
        condition_tokens=route_row["condition_tokens"],
        sequence_parallel_rank=route_row["sequence_parallel_rank"],
        sequence_parallel_size=route_row["sequence_parallel_size"],
        guidance_row=route_row["guidance_row"],
        sigma_schedule_index=route_row["sigma_schedule_index"],
    )
    band = validate_registered_module_band_receipt(
        row["registered_module_band"]
    )
    expected = make_branch_lock_audit_receipt(
        route,
        block_index=row["block_index"],
        projection=row["projection"],
        registered_block_indices=band["block_indices"],
        registered_projections=band["projections"],
    )
    if row != expected:
        raise CAGEBranchLockError(
            "CAGE branch-lock audit receipt does not replay exactly"
        )
    return row


_ACTIVE_ROUTE: ContextVar[Optional[CAGEBranchLockedRoute]] = ContextVar(
    "bernini_cage_branch_locked_route", default=None
)


def active_route() -> Optional[CAGEBranchLockedRoute]:
    return _ACTIVE_ROUTE.get()


@contextmanager
def activate_route(route: CAGEBranchLockedRoute) -> Iterator[None]:
    if not isinstance(route, CAGEBranchLockedRoute):
        raise CAGEBranchLockError("route must be CAGEBranchLockedRoute")
    if active_route() is not None:
        raise CAGEBranchLockError("nested CAGE branch routes are forbidden")
    token: Token[Optional[CAGEBranchLockedRoute]] = _ACTIVE_ROUTE.set(route)
    try:
        yield
    finally:
        _ACTIVE_ROUTE.reset(token)


class CAGEBranchLockedActionLoRA(nn.Module):
    """Cross-attention LoRA whose forward implements the CAGE conjunction."""

    def __init__(
        self,
        base: nn.Module,
        *,
        block_index: int,
        projection: str,
        registered_block_indices: Sequence[int],
        registered_projections: Sequence[str],
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise CAGEBranchLockError(f"{projection} base must be nn.Linear")
        # An out-of-band late block is accepted for a fail-closed audit probe,
        # but installation below never adds trainable modules there.
        band = registered_module_band_receipt(
            registered_block_indices, registered_projections
        )
        module_in_delta_band(
            block_index,
            projection,
            registered_block_indices=band["block_indices"],
            registered_projections=band["projections"],
        )
        if projection not in CAGE_PROJECTIONS:
            raise CAGEBranchLockError("only cross-attention Q/O may be wrapped")
        self.base = base
        self.block_index = block_index
        self.projection = projection
        self.registered_block_indices = tuple(band["block_indices"])
        self.registered_projections = tuple(band["projections"])
        self.registered_module_band_digest = band["digest"]
        self.rank = CAGE_LORA_RANK
        self.alpha = CAGE_LORA_ALPHA
        self.dropout = CAGE_LORA_DROPOUT
        self.cage_lora_a = nn.Linear(
            base.in_features, self.rank, bias=False, dtype=torch.float32
        )
        self.cage_lora_b = nn.Linear(
            self.rank, base.out_features, bias=False, dtype=torch.float32
        )
        nn.init.kaiming_uniform_(self.cage_lora_a.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.cage_lora_b.weight)

    @property
    def scale(self) -> float:
        return self.alpha / float(self.rank)

    @property
    def weight(self) -> Any:
        return self.base.weight

    @property
    def bias(self) -> Any:
        return self.base.bias

    def _module_authorized(self, route: CAGEBranchLockedRoute) -> bool:
        return route.authorizes_module(
            self.block_index,
            self.projection,
            registered_block_indices=self.registered_block_indices,
            registered_projections=self.registered_projections,
        )

    @staticmethod
    def _selector(
        hidden_states: torch.Tensor, route: CAGEBranchLockedRoute
    ) -> torch.Tensor:
        if hidden_states.ndim != 3 or int(hidden_states.shape[0]) != 1:
            raise CAGEBranchLockError(
                "native CAGE Action-LoRA expects hidden states [1,N,D]"
            )
        selector = route.local_target_selector(device=hidden_states.device)
        if int(hidden_states.shape[1]) != int(selector.numel()):
            raise CAGEBranchLockError(
                "local hidden sequence differs from append-pad/SP slice"
            )
        return selector

    def _selected_delta(
        self,
        hidden_states: torch.Tensor,
        selector: torch.Tensor,
        gate_weight: float,
    ) -> torch.Tensor:
        selected = hidden_states[:, selector, :]
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            delta = self.cage_lora_b(self.cage_lora_a(selected.float()))
            delta = delta * (self.scale * gate_weight)
        return delta.to(hidden_states.dtype)

    def adapter_delta(self, hidden_states: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(
            (*hidden_states.shape[:-1], self.base.out_features),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        route = active_route()
        if route is None or not self._module_authorized(route):
            return result
        selector = self._selector(hidden_states, route)
        if not bool(selector.any().item()):
            return result
        result[:, selector, :] = self._selected_delta(
            hidden_states, selector, route.gate_weight
        )
        return result

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base = self.base(hidden_states)
        route = active_route()
        # Crucially, locked branches, low sigma, and out-of-band modules return
        # before selector or LoRA evaluation.  The returned object is exactly
        # the computed base projection, not ``base + 0 * delta``.
        if route is None or not self._module_authorized(route):
            return base
        selector = self._selector(hidden_states, route)
        if not bool(selector.any().item()):
            return base
        result = base.clone()
        result[:, selector, :] = base[:, selector, :] + self._selected_delta(
            hidden_states, selector, route.gate_weight
        ).to(base.dtype)
        return result


@dataclass
class CAGEBranchLockHandle:
    transformer: nn.Module
    registered_block_indices: tuple[int, ...]
    registered_projections: tuple[str, ...]
    registered_module_band_digest: str
    q_wrappers: tuple[tuple[int, CAGEBranchLockedActionLoRA], ...]
    o_wrappers: tuple[tuple[int, CAGEBranchLockedActionLoRA], ...]
    original_q: tuple[tuple[int, nn.Module], ...]
    original_o: tuple[tuple[int, nn.Module], ...]
    original_patch_embedding_id: int
    original_self_attention_ids: tuple[tuple[int, int, int], ...]
    restored: bool = False

    def trainable_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        if self.restored:
            raise CAGEBranchLockError("CAGE adapter has been restored")
        result: list[tuple[str, nn.Parameter]] = []
        for index, wrapper in self.q_wrappers:
            result.extend(
                (
                    (
                        f"blocks.{index}.attn2.to_q.cage_lora_a.weight",
                        wrapper.cage_lora_a.weight,
                    ),
                    (
                        f"blocks.{index}.attn2.to_q.cage_lora_b.weight",
                        wrapper.cage_lora_b.weight,
                    ),
                )
            )
        for index, wrapper in self.o_wrappers:
            result.extend(
                (
                    (
                        f"blocks.{index}.attn2.to_out.0.cage_lora_a.weight",
                        wrapper.cage_lora_a.weight,
                    ),
                    (
                        f"blocks.{index}.attn2.to_out.0.cage_lora_b.weight",
                        wrapper.cage_lora_b.weight,
                    ),
                )
            )
        if len({id(parameter) for _, parameter in result}) != len(result):
            raise CAGEBranchLockError("CAGE LoRA parameter aliases another")
        if any(not parameter.requires_grad for _, parameter in result):
            raise CAGEBranchLockError("CAGE LoRA parameter is unexpectedly frozen")
        return tuple(result)

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

    def self_attention_untouched(self) -> bool:
        blocks = tuple(getattr(self.transformer, "blocks", ()))
        if len(blocks) != TOTAL_BLOCKS_1P3B:
            return False
        current: list[tuple[int, int, int]] = []
        for block in blocks:
            attention = getattr(block, "attn1", None)
            output = getattr(attention, "to_out", None)
            if output is None or len(output) < 1:
                return False
            current.append(
                (id(attention), id(getattr(attention, "to_q", None)), id(output[0]))
            )
        return tuple(current) == self.original_self_attention_ids

    def assert_scope(self) -> None:
        if self.restored:
            raise CAGEBranchLockError("CAGE adapter has been restored")
        blocks = tuple(getattr(self.transformer, "blocks", ()))
        if len(blocks) != TOTAL_BLOCKS_1P3B:
            raise CAGEBranchLockError("transformer block count changed")
        if (
            id(getattr(self.transformer, "patch_embedding", None))
            != self.original_patch_embedding_id
        ):
            raise CAGEBranchLockError("patch embedding changed under CAGE")
        if not self.self_attention_untouched():
            raise CAGEBranchLockError("self-attention/CIO scope changed under CAGE")
        q_by_index = dict(self.q_wrappers)
        o_by_index = dict(self.o_wrappers)
        original_q = dict(self.original_q)
        original_o = dict(self.original_o)
        band = registered_module_band_receipt(
            self.registered_block_indices, self.registered_projections
        )
        if band["digest"] != self.registered_module_band_digest:
            raise CAGEBranchLockError("CAGE registered module band digest differs")
        expected_q = (
            set(self.registered_block_indices)
            if "attn2.to_q" in self.registered_projections
            else set()
        )
        expected_o = (
            set(self.registered_block_indices)
            if "attn2.to_out.0" in self.registered_projections
            else set()
        )
        if set(q_by_index) != expected_q or set(o_by_index) != expected_o:
            raise CAGEBranchLockError("CAGE wrapper block inventory differs")
        for index, block in enumerate(blocks):
            query = block.attn2.to_q
            output = block.attn2.to_out[0]
            if index in expected_q:
                if query is not q_by_index[index]:
                    raise CAGEBranchLockError(
                        f"block {index} CAGE query identity differs"
                    )
                if (
                    query.block_index != index
                    or query.projection != "attn2.to_q"
                    or query.registered_module_band_digest != band["digest"]
                ):
                    raise CAGEBranchLockError(
                        f"block {index} CAGE query coordinate differs"
                    )
            elif query is not original_q[index]:
                raise CAGEBranchLockError(
                    f"block {index} query changed outside CAGE module band"
                )
            if index in expected_o:
                if output is not o_by_index[index]:
                    raise CAGEBranchLockError(
                        f"block {index} CAGE output identity differs"
                    )
                if (
                    output.block_index != index
                    or output.projection != "attn2.to_out.0"
                    or output.registered_module_band_digest != band["digest"]
                ):
                    raise CAGEBranchLockError(
                        f"block {index} CAGE output coordinate differs"
                    )
            elif output is not original_o[index]:
                raise CAGEBranchLockError(
                    f"block {index} output changed outside CAGE module band"
                )
        if not self.base_parameters_frozen():
            raise CAGEBranchLockError("base/trainable parameter closure differs")

    @contextmanager
    def route(self, route: CAGEBranchLockedRoute) -> Iterator[None]:
        self.assert_scope()
        with activate_route(route):
            yield

    def receipt(self) -> dict[str, Any]:
        self.assert_scope()
        trainable = self.trainable_named_parameters()
        band = registered_module_band_receipt(
            self.registered_block_indices, self.registered_projections
        )
        unsigned = {
            "schema_version": SCHEMA_VERSION,
            "parent_adapter_schema": pair_v5.SCHEMA_VERSION,
            "guidance_rows": list(GUIDANCE_ROWS),
            "delta_guidance_row": DELTA_GUIDANCE_ROW,
            "locked_guidance_rows": list(LOCKED_GUIDANCE_ROWS),
            "branch_lock_is_explicit_route_coordinate": True,
            "row_policy": {
                "source_delta_authorized": False,
                "target_delta_authorized_only_after_all_gates": True,
                "padding_delta_authorized": False,
            },
            "module_band": {
                **band,
                "outside_band_direct_base": True,
                "all_30_blocks_probeable_by_single_wrapper": True,
                "band_requires_external_probe_selection": True,
            },
            "sigma_band": {
                "exact40_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
                "allowed_unit_weight_1": list(DELTA_SIGMA_INDICES),
                "low_direct_base": list(LOW_SIGMA_INDICES),
                "condition_annealing_is_separate_factorial": True,
            },
            "rank": CAGE_LORA_RANK,
            "alpha": CAGE_LORA_ALPHA,
            "dropout": CAGE_LORA_DROPOUT,
            "sp_selector": "append_false_then_contiguous_rank_chunk",
            "locked_branches_direct_base_return": True,
            "low_sigma_direct_base_return": True,
            "wrapped_projection_source_and_padding_rows_byte_exact_base": True,
            "full_source_memory_reinjection_implemented": False,
            "global_source_memory_lock_claim": False,
            "patch_embedding_untouched": True,
            "self_attention_and_frozen_cio_untouched": True,
            "key_value_trainable": False,
            "unregistered_modules_trainable": False,
            "base_and_frozen_cio_parameters_frozen": True,
            "gradient_checkpointing_supported": False,
            "trainable_key_sha256": object_sha256(
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
            "optimizer_authorized": False,
            "semantic_action_claim": False,
        }
        return {**unsigned, "digest": object_sha256(unsigned)}

    def restore(self) -> None:
        if self.restored or active_route() is not None:
            raise CAGEBranchLockError("CAGE adapter cannot be restored now")
        self.assert_scope()
        blocks = tuple(self.transformer.blocks)
        original_q = dict(self.original_q)
        original_o = dict(self.original_o)
        for index in self.registered_block_indices:
            if "attn2.to_q" in self.registered_projections:
                blocks[index].attn2.to_q = original_q[index]
            if "attn2.to_out.0" in self.registered_projections:
                blocks[index].attn2.to_out[0] = original_o[index]
        self.restored = True


_ADAPTER_RECEIPT_FIELDS = {
    "schema_version",
    "parent_adapter_schema",
    "guidance_rows",
    "delta_guidance_row",
    "locked_guidance_rows",
    "branch_lock_is_explicit_route_coordinate",
    "row_policy",
    "module_band",
    "sigma_band",
    "rank",
    "alpha",
    "dropout",
    "sp_selector",
    "locked_branches_direct_base_return",
    "low_sigma_direct_base_return",
    "wrapped_projection_source_and_padding_rows_byte_exact_base",
    "full_source_memory_reinjection_implemented",
    "global_source_memory_lock_claim",
    "patch_embedding_untouched",
    "self_attention_and_frozen_cio_untouched",
    "key_value_trainable",
    "unregistered_modules_trainable",
    "base_and_frozen_cio_parameters_frozen",
    "gradient_checkpointing_supported",
    "trainable_key_sha256",
    "trainable",
    "optimizer_authorized",
    "semantic_action_claim",
    "digest",
}


def _expected_trainable_names(
    registered_block_indices: Sequence[int],
    registered_projections: Sequence[str],
) -> list[str]:
    blocks, projections = validate_registered_module_band(
        registered_block_indices, registered_projections
    )
    names: list[str] = []
    for full_projection in projections:
        projection = full_projection[len("attn2.") :]
        for index in blocks:
            for factor in ("a", "b"):
                names.append(
                    f"blocks.{index}.attn2.{projection}.cage_lora_{factor}.weight"
                )
    return names


def validate_adapter_receipt(value: Any) -> dict[str, Any]:
    row = _closed(value, _ADAPTER_RECEIPT_FIELDS, label="CAGE adapter receipt")
    _verify_digest(row, label="CAGE adapter receipt")
    module_band = _closed(
        row["module_band"],
        {
            "block_indices",
            "projections",
            "module_sites",
            "continuous_block_band",
            "explicit_no_default",
            "digest",
            "outside_band_direct_base",
            "all_30_blocks_probeable_by_single_wrapper",
            "band_requires_external_probe_selection",
        },
        label="CAGE adapter module band",
    )
    band = validate_registered_module_band_receipt(
        {
            key: module_band[key]
            for key in (
                "block_indices",
                "projections",
                "module_sites",
                "continuous_block_band",
                "explicit_no_default",
                "digest",
            )
        }
    )
    expected_static = {
        "schema_version": SCHEMA_VERSION,
        "parent_adapter_schema": pair_v5.SCHEMA_VERSION,
        "guidance_rows": list(GUIDANCE_ROWS),
        "delta_guidance_row": DELTA_GUIDANCE_ROW,
        "locked_guidance_rows": list(LOCKED_GUIDANCE_ROWS),
        "branch_lock_is_explicit_route_coordinate": True,
        "row_policy": {
            "source_delta_authorized": False,
            "target_delta_authorized_only_after_all_gates": True,
            "padding_delta_authorized": False,
        },
        "module_band": {
            **band,
            "outside_band_direct_base": True,
            "all_30_blocks_probeable_by_single_wrapper": True,
            "band_requires_external_probe_selection": True,
        },
        "sigma_band": {
            "exact40_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
            "allowed_unit_weight_1": list(DELTA_SIGMA_INDICES),
            "low_direct_base": list(LOW_SIGMA_INDICES),
            "condition_annealing_is_separate_factorial": True,
        },
        "rank": CAGE_LORA_RANK,
        "alpha": CAGE_LORA_ALPHA,
        "dropout": CAGE_LORA_DROPOUT,
        "sp_selector": "append_false_then_contiguous_rank_chunk",
        "locked_branches_direct_base_return": True,
        "low_sigma_direct_base_return": True,
        "wrapped_projection_source_and_padding_rows_byte_exact_base": True,
        "full_source_memory_reinjection_implemented": False,
        "global_source_memory_lock_claim": False,
        "patch_embedding_untouched": True,
        "self_attention_and_frozen_cio_untouched": True,
        "key_value_trainable": False,
        "unregistered_modules_trainable": False,
        "base_and_frozen_cio_parameters_frozen": True,
        "gradient_checkpointing_supported": False,
        "optimizer_authorized": False,
        "semantic_action_claim": False,
    }
    for key, expected in expected_static.items():
        if row.get(key) != expected:
            raise CAGEBranchLockError(f"CAGE adapter receipt {key} differs")
    trainable = row["trainable"]
    expected_names = _expected_trainable_names(
        band["block_indices"], band["projections"]
    )
    if not isinstance(trainable, list) or len(trainable) != len(
        expected_names
    ):
        raise CAGEBranchLockError("CAGE trainable inventory differs")
    observed_names: list[str] = []
    hidden: Optional[int] = None
    for item in trainable:
        entry = _closed(item, {"name", "shape", "dtype"}, label="CAGE trainable")
        name = entry["name"]
        shape = entry["shape"]
        if not isinstance(name, str) or entry["dtype"] != "torch.float32":
            raise CAGEBranchLockError("CAGE trainable metadata differs")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
        ):
            raise CAGEBranchLockError("CAGE trainable shape differs")
        if name.endswith("cage_lora_a.weight"):
            if shape[0] != CAGE_LORA_RANK or shape[1] <= 0:
                raise CAGEBranchLockError("CAGE LoRA-A shape differs")
            current_hidden = shape[1]
        elif name.endswith("cage_lora_b.weight"):
            if shape[1] != CAGE_LORA_RANK or shape[0] <= 0:
                raise CAGEBranchLockError("CAGE LoRA-B shape differs")
            current_hidden = shape[0]
        else:
            raise CAGEBranchLockError("CAGE trainable factor name differs")
        hidden = current_hidden if hidden is None else hidden
        if current_hidden != hidden:
            raise CAGEBranchLockError("CAGE hidden width differs across modules")
        observed_names.append(name)
    if observed_names != expected_names:
        raise CAGEBranchLockError("CAGE trainable key order differs")
    if row["trainable_key_sha256"] != object_sha256(sorted(observed_names)):
        raise CAGEBranchLockError("CAGE trainable key digest differs")
    return row


def install_cage_branch_locked_action_adapter(
    transformer: nn.Module,
    *,
    registered_block_indices: Sequence[int],
    registered_projections: Sequence[str],
) -> CAGEBranchLockHandle:
    """Install CAGE only at an explicit continuous post-probe module band."""

    if not isinstance(transformer, nn.Module):
        raise CAGEBranchLockError("transformer must be nn.Module")
    band = registered_module_band_receipt(
        registered_block_indices, registered_projections
    )
    registered_blocks = tuple(band["block_indices"])
    registered_projection_set = set(band["projections"])
    if any(parameter.requires_grad for parameter in transformer.parameters()):
        raise CAGEBranchLockError(
            "freeze the complete Bernini base and any CIO adapter before installation"
        )
    blocks = tuple(getattr(transformer, "blocks", ()))
    patch = getattr(transformer, "patch_embedding", None)
    if (
        len(blocks) != TOTAL_BLOCKS_1P3B
        or not isinstance(patch, nn.Conv3d)
        or not callable(getattr(transformer, "patch_vae_latent", None))
    ):
        raise CAGEBranchLockError("Bernini-R 1.3B native structure differs")
    hidden = int(patch.out_channels)
    original_q: list[tuple[int, nn.Module]] = []
    original_o: list[tuple[int, nn.Module]] = []
    original_self_attention_ids: list[tuple[int, int, int]] = []
    for index, block in enumerate(blocks):
        self_attention = getattr(block, "attn1", None)
        self_output = getattr(self_attention, "to_out", None)
        cross_attention = getattr(block, "attn2", None)
        query = getattr(cross_attention, "to_q", None)
        output = getattr(cross_attention, "to_out", None)
        if self_output is None or len(self_output) < 1:
            raise CAGEBranchLockError(
                f"block {index} self-attention structure differs"
            )
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
            raise CAGEBranchLockError(
                f"block {index} cross-attention Q/O structure differs"
            )
        original_q.append((index, query))
        original_o.append((index, output[0]))
        original_self_attention_ids.append(
            (
                id(self_attention),
                id(getattr(self_attention, "to_q", None)),
                id(self_output[0]),
            )
        )

    device = patch.weight.device
    q_wrappers: list[tuple[int, CAGEBranchLockedActionLoRA]] = []
    o_wrappers: list[tuple[int, CAGEBranchLockedActionLoRA]] = []
    original_q_by_index = dict(original_q)
    original_o_by_index = dict(original_o)
    try:
        for index in registered_blocks:
            if "attn2.to_q" in registered_projection_set:
                q_wrapper = CAGEBranchLockedActionLoRA(
                    original_q_by_index[index],
                    block_index=index,
                    projection="attn2.to_q",
                    registered_block_indices=registered_blocks,
                    registered_projections=band["projections"],
                ).to(device=device)
                blocks[index].attn2.to_q = q_wrapper
                q_wrappers.append((index, q_wrapper))
            if "attn2.to_out.0" in registered_projection_set:
                o_wrapper = CAGEBranchLockedActionLoRA(
                    original_o_by_index[index],
                    block_index=index,
                    projection="attn2.to_out.0",
                    registered_block_indices=registered_blocks,
                    registered_projections=band["projections"],
                ).to(device=device)
                blocks[index].attn2.to_out[0] = o_wrapper
                o_wrappers.append((index, o_wrapper))
    except Exception:
        for index in registered_blocks:
            if "attn2.to_q" in registered_projection_set:
                blocks[index].attn2.to_q = original_q_by_index[index]
            if "attn2.to_out.0" in registered_projection_set:
                blocks[index].attn2.to_out[0] = original_o_by_index[index]
        raise

    handle = CAGEBranchLockHandle(
        transformer=transformer,
        registered_block_indices=registered_blocks,
        registered_projections=tuple(band["projections"]),
        registered_module_band_digest=band["digest"],
        q_wrappers=tuple(q_wrappers),
        o_wrappers=tuple(o_wrappers),
        original_q=tuple(original_q),
        original_o=tuple(original_o),
        original_patch_embedding_id=id(patch),
        original_self_attention_ids=tuple(original_self_attention_ids),
    )
    try:
        handle.assert_scope()
        validate_adapter_receipt(handle.receipt())
    except Exception:
        if not handle.restored:
            for index in registered_blocks:
                if "attn2.to_q" in registered_projection_set:
                    blocks[index].attn2.to_q = original_q_by_index[index]
                if "attn2.to_out.0" in registered_projection_set:
                    blocks[index].attn2.to_out[0] = original_o_by_index[index]
            handle.restored = True
        raise
    return handle


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "CAGE_LORA_ALPHA",
    "CAGE_LORA_DROPOUT",
    "CAGE_LORA_RANK",
    "CAGE_PROJECTIONS",
    "CAGEBranchLockError",
    "CAGEBranchLockHandle",
    "CAGEBranchLockedActionLoRA",
    "CAGEBranchLockedRoute",
    "DELTA_GUIDANCE_ROW",
    "DELTA_SIGMA_INDICES",
    "GUIDANCE_TO_NATIVE_BRANCH",
    "GUIDANCE_ROWS",
    "HIGH_SIGMA_INDICES",
    "LOCKED_GUIDANCE_ROWS",
    "LOW_SIGMA_INDICES",
    "MID_SIGMA_INDICES",
    "PROBE_BLOCK_INDICES",
    "ROUTE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "TOTAL_BLOCKS_1P3B",
    "activate_route",
    "active_route",
    "assert_tensors_byte_exact",
    "canonical_json_bytes",
    "install_cage_branch_locked_action_adapter",
    "make_branch_lock_audit_receipt",
    "module_in_delta_band",
    "object_sha256",
    "registered_module_band_receipt",
    "sigma_gate",
    "tensors_byte_exact",
    "validate_adapter_receipt",
    "validate_branch_lock_audit_receipt",
    "validate_registered_module_band",
    "validate_registered_module_band_receipt",
    "validate_route_receipt",
]
