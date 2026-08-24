#!/usr/bin/env python3
"""CAPER rank-8 capacity probe for frozen Bernini-R 1.3B.

CAPER installs a deliberately closed LoRA capacity probe at exactly the
cross-attention query and output projections

``diff_dec.transformer.blocks.{0..29}.attn2.to_q`` and
``diff_dec.transformer.blocks.{0..29}.attn2.to_out.0``.

The adapter is projection-local and target-row-only.  An explicit global
boolean selector is bound to the canonical preference pack ``[S,y+,S,y-]``
and sliced only from a parallel-state authority snapshot.  Source/reference
rows and append-padding rows are copied byte-for-byte from each wrapped frozen
base projection.  This is not a global activation guarantee: a later joint
``attn1`` can mix changed target activations back into source rows.  The adapter
is active only on a hash-pinned exact40 UniPC high/mid-sigma coordinate.  A
low-sigma, disabled, or absent route returns the base projection directly and
never evaluates either LoRA factor.

This is a capacity primitive, not a training loop and not evidence of semantic
action editing.  The injector accepts only the native Bernini namespace; it
does not search aliases or fall back to PEFT suffix matching.  It also emits a
replayable freeze/checksum certificate over the complete frozen Wan
transformer parameter state.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
import hashlib
import importlib
import json
import math
import struct
from typing import Any, Iterator, Mapping, Optional, Sequence

import torch
from torch import nn

import inference_sigma_strata as sigma_strata


SCHEMA_VERSION = "bernini-caper-target-row-sigma-gated-lora-v2"
CERTIFICATE_SCHEMA = "bernini-caper-freeze-checksum-certificate-v2"
PACK_RECEIPT_SCHEMA = "bernini-caper-preference-pack-receipt-v1"
PARALLEL_RECEIPT_SCHEMA = "bernini-caper-parallel-state-receipt-v1"
ROOT_NAMESPACE = "diff_dec.transformer"
TOTAL_BLOCKS_1P3B = 30
BERNINI_1P3B_HIDDEN_SIZE = 1536
CAPER_BLOCK_INDICES = tuple(range(TOTAL_BLOCKS_1P3B))
CAPER_PROJECTIONS = ("attn2.to_q", "attn2.to_out.0")
CAPER_RANK = 8
CAPER_ALPHA = 8.0
CAPER_DROPOUT = 0.0
ALLOWED_SP_SIZES = frozenset({1, 4})
SP1_TEST_AUTHORITY_ID = "caper-explicit-sp1-unit-test-authority"

PREFERENCE_PACK_LAYOUT = (
    ("source_for_winner", False),
    ("winner_target", True),
    ("source_for_loser", False),
    ("loser_target", True),
)

HIGH_SIGMA_INDICES = tuple(range(33))
MID_SIGMA_INDICES = tuple(range(33, 38))
LOW_SIGMA_INDICES = tuple(range(38, 40))
HIGH_SIGMA_WEIGHT = 1.0
MID_SIGMA_WEIGHT = 0.5
LOW_SIGMA_WEIGHT = 0.0


class CAPERContractError(RuntimeError):
    """Raised before an ambiguous route, module scope, or state can be used."""


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
        raise CAPERContractError(
            f"certificate value is not canonical finite ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CAPERContractError(f"{label} must be a positive integer")
    return value


def _validate_block_indices(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise CAPERContractError("block_indices must be an explicit tuple/list")
    blocks = tuple(value)
    if not blocks:
        raise CAPERContractError("CAPER block scope cannot be empty")
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < TOTAL_BLOCKS_1P3B
        for index in blocks
    ):
        raise CAPERContractError("CAPER block index must lie in [0,29]")
    if blocks != tuple(sorted(set(blocks))):
        raise CAPERContractError("CAPER block indices must be unique and ascending")
    return blocks


def canonical_target_module_names(
    block_indices: Sequence[int] = CAPER_BLOCK_INDICES,
) -> tuple[str, ...]:
    blocks = _validate_block_indices(block_indices)
    return tuple(
        f"{ROOT_NAMESPACE}.blocks.{block}.{projection}"
        for block in blocks
        for projection in CAPER_PROJECTIONS
    )


CAPER_TARGET_MODULES = canonical_target_module_names()
CAPER_TARGET_MODULES_SHA256 = (
    "1b36a863a385fb69c660b6b24b10e5a2b6ed906ec20017bd5e418998b6783695"
)
if object_sha256(list(CAPER_TARGET_MODULES)) != CAPER_TARGET_MODULES_SHA256:
    raise RuntimeError("CAPER canonical all-30 Q/O module allowlist changed")


def _validate_registered_schedule() -> None:
    if (
        HIGH_SIGMA_INDICES + MID_SIGMA_INDICES + LOW_SIGMA_INDICES
        != tuple(range(sigma_strata.NUM_INFERENCE_STEPS))
    ):
        raise RuntimeError("CAPER sigma partition is not exact40")
    if any(
        sigma_strata.PINNED_POSITIVE_SIGMAS[index] < 0.55
        for index in HIGH_SIGMA_INDICES
    ):
        raise RuntimeError("CAPER high-sigma threshold differs")
    if any(
        not 0.25 <= sigma_strata.PINNED_POSITIVE_SIGMAS[index] < 0.55
        for index in MID_SIGMA_INDICES
    ):
        raise RuntimeError("CAPER mid-sigma thresholds differ")
    if any(
        sigma_strata.PINNED_POSITIVE_SIGMAS[index] >= 0.25
        for index in LOW_SIGMA_INDICES
    ):
        raise RuntimeError("CAPER low-sigma threshold differs")


_validate_registered_schedule()


def _runtime_float32_hex(value: Any, *, label: str) -> str:
    try:
        candidate = value.detach() if hasattr(value, "detach") else value
        if hasattr(candidate, "numel") and int(candidate.numel()) != 1:
            raise CAPERContractError(f"{label} must be scalar")
        if hasattr(candidate, "cpu"):
            candidate = candidate.cpu()
        if hasattr(candidate, "item"):
            candidate = candidate.item()
        numeric = float(candidate)
        encoded = struct.pack(">f", numeric)
    except CAPERContractError:
        raise
    except (TypeError, ValueError, OverflowError, struct.error) as error:
        raise CAPERContractError(f"{label} must be finite float32") from error
    if not math.isfinite(numeric):
        raise CAPERContractError(f"{label} must be finite float32")
    return encoded.hex()


def sigma_gate(
    schedule_index: Any, sigma_float32_be_hex: Any
) -> tuple[str, float]:
    """Validate one exact runtime sigma and return its explicit CAPER gate."""

    if (
        isinstance(schedule_index, bool)
        or not isinstance(schedule_index, int)
        or not 0 <= schedule_index < sigma_strata.NUM_INFERENCE_STEPS
    ):
        raise CAPERContractError("sigma_schedule_index must be an integer in [0,39]")
    expected_hex = sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[schedule_index]
    if type(sigma_float32_be_hex) is not str or sigma_float32_be_hex != expected_hex:
        raise CAPERContractError("runtime sigma differs from the pinned schedule index")
    if schedule_index in HIGH_SIGMA_INDICES:
        return "high", HIGH_SIGMA_WEIGHT
    if schedule_index in MID_SIGMA_INDICES:
        return "mid", MID_SIGMA_WEIGHT
    if schedule_index in LOW_SIGMA_INDICES:
        return "low_base_only", LOW_SIGMA_WEIGHT
    raise CAPERContractError("sigma schedule index is not preregistered")


def _strict_bool_tuple(value: Any, *, label: str) -> tuple[bool, ...]:
    if not isinstance(value, (tuple, list)):
        raise CAPERContractError(f"{label} must be an explicit tuple/list of booleans")
    result = tuple(value)
    if not result or any(type(item) is not bool for item in result):
        raise CAPERContractError(f"{label} must contain only explicit booleans")
    return result


def target_selector_sha256(selector: Sequence[bool]) -> str:
    """Hash selector length and every F/T bit without serializing a huge list."""

    bits = _strict_bool_tuple(selector, label="global_target_selector")
    digest = hashlib.sha256()
    digest.update(SCHEMA_VERSION.encode("ascii"))
    digest.update(len(bits).to_bytes(8, "big"))
    digest.update(bytes(1 if item else 0 for item in bits))
    return digest.hexdigest()


@dataclass(frozen=True)
class CAPERPackSegment:
    """One explicit segment in the canonical ``[S,y+,S,y-]`` pack."""

    role: str
    tokens: int
    is_target: bool

    def __post_init__(self) -> None:
        if type(self.role) is not str or not self.role:
            raise CAPERContractError("pack segment role must be a non-empty string")
        _positive_int(self.tokens, label=f"pack segment {self.role} tokens")
        if type(self.is_target) is not bool:
            raise CAPERContractError("pack segment is_target must be boolean")

    def receipt(self) -> Mapping[str, Any]:
        return {
            "role": self.role,
            "tokens": self.tokens,
            "is_target": self.is_target,
        }


def preference_pack_segments(
    *, source_tokens: int, target_tokens: int
) -> tuple[CAPERPackSegment, ...]:
    """Create the canonical equal-geometry preference pack declaration."""

    source = _positive_int(source_tokens, label="source_tokens")
    target = _positive_int(target_tokens, label="target_tokens")
    lengths = (source, target, source, target)
    return tuple(
        CAPERPackSegment(role=role, tokens=tokens, is_target=is_target)
        for (role, is_target), tokens in zip(PREFERENCE_PACK_LAYOUT, lengths)
    )


def preference_pack_target_selector(
    *, source_tokens: int, target_tokens: int
) -> tuple[bool, ...]:
    segments = preference_pack_segments(
        source_tokens=source_tokens, target_tokens=target_tokens
    )
    return tuple(
        is_target
        for segment in segments
        for is_target in (segment.is_target,) * segment.tokens
    )


def _validated_preference_pack(
    selector: Any, segments: Any
) -> tuple[tuple[bool, ...], tuple[CAPERPackSegment, ...]]:
    bits = _strict_bool_tuple(selector, label="global_target_selector")
    if not isinstance(segments, (tuple, list)):
        raise CAPERContractError("pack_segments must be an explicit tuple/list")
    packed = tuple(segments)
    if len(packed) != len(PREFERENCE_PACK_LAYOUT) or any(
        type(segment) is not CAPERPackSegment for segment in packed
    ):
        raise CAPERContractError("pack_segments must declare exact [S,y+,S,y-]")
    observed_layout = tuple(
        (segment.role, segment.is_target) for segment in packed
    )
    if observed_layout != PREFERENCE_PACK_LAYOUT:
        raise CAPERContractError("pack segment roles/selector must be F,T,F,T")
    if packed[0].tokens != packed[2].tokens:
        raise CAPERContractError("the repeated source S must have equal token geometry")
    if packed[1].tokens != packed[3].tokens:
        raise CAPERContractError("winner/loser targets must have equal token geometry")
    expanded = tuple(
        is_target
        for segment in packed
        for is_target in (segment.is_target,) * segment.tokens
    )
    if expanded != bits:
        raise CAPERContractError(
            "global_target_selector is not bound to the declared preference pack"
        )
    return bits, packed


def preference_pack_receipt(
    selector: Sequence[bool], segments: Sequence[CAPERPackSegment]
) -> Mapping[str, Any]:
    bits, packed = _validated_preference_pack(selector, segments)
    value = {
        "schema_version": PACK_RECEIPT_SCHEMA,
        "layout": "[S,y+,S,y-]",
        "segments": [dict(segment.receipt()) for segment in packed],
        "total_tokens": len(bits),
        "source_tokens": sum(not item for item in bits),
        "target_tokens": sum(bits),
        "target_intervals": 2,
        "target_selector_sha256": target_selector_sha256(bits),
    }
    return {**value, "digest": object_sha256(value)}


@dataclass(frozen=True)
class CAPERParallelState:
    """Snapshot returned by the injected Bernini parallel-state authority.

    Group membership, not a caller supplied size, defines the SP coordinate.
    The authority must obtain these values from the initialized distributed
    runtime.  ``test_only`` exists solely for deterministic logic oracles.
    """

    world_size: int
    world_rank: int
    sequence_parallel_group_ranks: tuple[int, ...]
    sequence_parallel_rank: int
    authority_id: str
    test_only: bool = False

    def __post_init__(self) -> None:
        world = _positive_int(self.world_size, label="parallel world_size")
        rank = self.world_rank
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world:
            raise CAPERContractError("parallel world_rank lies outside world")
        if not isinstance(self.sequence_parallel_group_ranks, (tuple, list)):
            raise CAPERContractError("SP group ranks must be an explicit tuple/list")
        group = tuple(self.sequence_parallel_group_ranks)
        if len(group) not in ALLOWED_SP_SIZES:
            raise CAPERContractError("only explicit SP1 tests and native SP4 are supported")
        if any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or not 0 <= item < world
            for item in group
        ) or len(set(group)) != len(group):
            raise CAPERContractError("SP group ranks are invalid or aliased")
        sp_rank = self.sequence_parallel_rank
        if (
            isinstance(sp_rank, bool)
            or not isinstance(sp_rank, int)
            or not 0 <= sp_rank < len(group)
            or group[sp_rank] != rank
        ):
            raise CAPERContractError("SP rank does not index world_rank in its group")
        if type(self.authority_id) is not str or not self.authority_id:
            raise CAPERContractError("parallel authority_id must be non-empty")
        try:
            self.authority_id.encode("ascii")
        except UnicodeEncodeError as error:
            raise CAPERContractError("parallel authority_id must be ASCII") from error
        if type(self.test_only) is not bool:
            raise CAPERContractError("parallel test_only must be boolean")
        if len(group) == 1 and not (
            self.test_only
            and self.authority_id == SP1_TEST_AUTHORITY_ID
            and world == 1
            and rank == 0
            and group == (0,)
            and sp_rank == 0
        ):
            raise CAPERContractError("SP1 is permitted only by explicit unit-test authority")
        if len(group) == 4 and (world < 4 or world % 4 != 0):
            raise CAPERContractError("SP4 requires a world size divisible by four")
        object.__setattr__(self, "sequence_parallel_group_ranks", group)

    @property
    def sequence_parallel_size(self) -> int:
        return len(self.sequence_parallel_group_ranks)

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": PARALLEL_RECEIPT_SCHEMA,
            "world_size": self.world_size,
            "world_rank": self.world_rank,
            "sequence_parallel_group_ranks": list(
                self.sequence_parallel_group_ranks
            ),
            "sequence_parallel_size": self.sequence_parallel_size,
            "sequence_parallel_rank": self.sequence_parallel_rank,
            "authority_id": self.authority_id,
            "test_only": self.test_only,
        }
        return {**value, "digest": object_sha256(value)}


def _parallel_state_from_authority(authority: Any) -> CAPERParallelState:
    """Call one trusted parallel-state boundary; never accept raw rank/size."""

    if authority is None:
        raise CAPERContractError("parallel-state authority is required (fail closed)")
    snapshot = getattr(authority, "snapshot", None)
    if callable(snapshot):
        value = snapshot()
    elif callable(authority):
        value = authority()
    else:
        raise CAPERContractError("parallel-state authority must be callable or expose snapshot()")
    if type(value) is not CAPERParallelState:
        raise CAPERContractError("parallel-state authority returned an untyped snapshot")
    # Reconstruct to run validation even if a hostile object bypassed dataclass init.
    return CAPERParallelState(
        world_size=value.world_size,
        world_rank=value.world_rank,
        sequence_parallel_group_ranks=value.sequence_parallel_group_ranks,
        sequence_parallel_rank=value.sequence_parallel_rank,
        authority_id=value.authority_id,
        test_only=value.test_only,
    )


def snapshot_live_bernini_parallel_state() -> CAPERParallelState:
    """Authenticate the live torch.distributed/Bernini SP coordinate.

    This is the production authority used by the training boundary.  It has no
    rank/size arguments: world rank/size come from the default process group,
    while SP rank/size and group identity come from Bernini's installed
    parallel state.  Ordered membership is observed with a real group
    collective rather than inferred from caller input.
    """

    dist = torch.distributed
    try:
        bernini_parallel = importlib.import_module("bernini.parallel")
        state = bernini_parallel.get_parallel_state()
    except Exception as error:
        raise CAPERContractError("live Bernini parallel state is unavailable") from error
    if not dist.is_available() or not dist.is_initialized():
        raise CAPERContractError("torch.distributed must be initialized")
    world_size = dist.get_world_size()
    world_rank = dist.get_rank()
    group = getattr(state, "ulysses_group", None)
    sp_size = getattr(state, "ulysses_size", None)
    sp_rank = getattr(state, "ulysses_rank", None)
    state_type = type(state)
    if (
        not state_type.__module__.startswith("bernini.parallel")
        or getattr(state, "ulysses_enabled", None) is not True
        or type(sp_size) is not int
        or sp_size != 4
        or type(sp_rank) is not int
        or group is None
        or type(getattr(state, "world_size", None)) is not int
        or state.world_size != world_size
        or type(getattr(state, "rank", None)) is not int
        or state.rank != world_rank
        or dist.get_world_size(group) != sp_size
        or dist.get_rank(group) != sp_rank
    ):
        raise CAPERContractError(
            "live torch.distributed/Bernini WORLD/SP/rank coordinate differs"
        )
    members: list[Any] = [None] * sp_size
    dist.all_gather_object(members, world_rank, group=group)
    if (
        any(type(item) is not int for item in members)
        or len(set(members)) != sp_size
        or tuple(members) != tuple(range(members[0], members[0] + sp_size))
        or members[0] % sp_size != 0
        or members[sp_rank] != world_rank
    ):
        raise CAPERContractError("live Bernini SP4 ordered group membership differs")
    return CAPERParallelState(
        world_size=world_size,
        world_rank=world_rank,
        sequence_parallel_group_ranks=tuple(members),
        sequence_parallel_rank=sp_rank,
        authority_id="torch.distributed+bernini.parallel.get_parallel_state",
        test_only=False,
    )


_AUTHORITY_VERIFIED_TOKEN = object()


@dataclass(frozen=True)
class CAPERRoute:
    """One explicit preference pack after authority-derived native SP slicing."""

    global_target_selector: tuple[bool, ...]
    pack_segments: tuple[CAPERPackSegment, ...]
    parallel_state: CAPERParallelState
    sigma_schedule_index: int
    sigma_float32_be_hex: str
    _authority_verified_token: Any = field(repr=False, compare=False)
    enabled: bool = True

    def __post_init__(self) -> None:
        if self._authority_verified_token is not _AUTHORITY_VERIFIED_TOKEN:
            raise CAPERContractError("construct CAPERRoute only through parallel authority")
        bits, segments = _validated_preference_pack(
            self.global_target_selector, self.pack_segments
        )
        if type(self.parallel_state) is not CAPERParallelState:
            raise CAPERContractError("route parallel state is not an authority snapshot")
        if type(self.enabled) is not bool:
            raise CAPERContractError("enabled must be boolean")
        sigma_gate(self.sigma_schedule_index, self.sigma_float32_be_hex)
        object.__setattr__(self, "global_target_selector", bits)
        object.__setattr__(self, "pack_segments", segments)

    @classmethod
    def from_runtime_sigma(
        cls,
        *,
        global_target_selector: Sequence[bool],
        pack_segments: Sequence[CAPERPackSegment],
        parallel_state_authority: Any = None,
        sigma_schedule_index: int,
        sigma: Any,
        enabled: bool = True,
    ) -> "CAPERRoute":
        """Bind selector, pack, runtime sigma, and authoritative SP snapshot."""

        observed_hex = _runtime_float32_hex(sigma, label="runtime sigma")
        parallel_state = _parallel_state_from_authority(parallel_state_authority)
        return cls(
            global_target_selector=tuple(global_target_selector),
            pack_segments=tuple(pack_segments),
            parallel_state=parallel_state,
            sigma_schedule_index=sigma_schedule_index,
            sigma_float32_be_hex=observed_hex,
            _authority_verified_token=_AUTHORITY_VERIFIED_TOKEN,
            enabled=enabled,
        )

    @classmethod
    def from_live_runtime_sigma(
        cls,
        *,
        global_target_selector: Sequence[bool],
        pack_segments: Sequence[CAPERPackSegment],
        sigma_schedule_index: int,
        sigma: Any,
        enabled: bool = True,
    ) -> "CAPERRoute":
        """Construct a production route solely from the live runtime."""

        observed_hex = _runtime_float32_hex(sigma, label="runtime sigma")
        return cls(
            global_target_selector=tuple(global_target_selector),
            pack_segments=tuple(pack_segments),
            parallel_state=snapshot_live_bernini_parallel_state(),
            sigma_schedule_index=sigma_schedule_index,
            sigma_float32_be_hex=observed_hex,
            _authority_verified_token=_AUTHORITY_VERIFIED_TOKEN,
            enabled=enabled,
        )

    @property
    def total_tokens(self) -> int:
        return len(self.global_target_selector)

    @property
    def target_tokens(self) -> int:
        return sum(self.global_target_selector)

    @property
    def source_tokens(self) -> int:
        return self.total_tokens - self.target_tokens

    @property
    def sequence_parallel_rank(self) -> int:
        return self.parallel_state.sequence_parallel_rank

    @property
    def sequence_parallel_size(self) -> int:
        return self.parallel_state.sequence_parallel_size

    @property
    def local_length(self) -> int:
        return math.ceil(self.total_tokens / self.sequence_parallel_size)

    @property
    def padded_tokens(self) -> int:
        return self.local_length * self.sequence_parallel_size

    @property
    def gate_name(self) -> str:
        return sigma_gate(
            self.sigma_schedule_index, self.sigma_float32_be_hex
        )[0]

    @property
    def gate_weight(self) -> float:
        if not self.enabled:
            return 0.0
        return sigma_gate(
            self.sigma_schedule_index, self.sigma_float32_be_hex
        )[1]

    @property
    def adapter_active(self) -> bool:
        return self.enabled and self.gate_weight > 0.0

    def global_target_selector_tensor(self, *, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            self.global_target_selector, dtype=torch.bool, device=device
        )

    def local_target_selector(self, *, device: torch.device) -> torch.Tensor:
        selector = self.global_target_selector_tensor(device=device)
        if self.padded_tokens > self.total_tokens:
            selector = torch.cat(
                (
                    selector,
                    torch.zeros(
                        self.padded_tokens - self.total_tokens,
                        dtype=torch.bool,
                        device=device,
                    ),
                )
            )
        start = self.sequence_parallel_rank * self.local_length
        return selector[start : start + self.local_length].contiguous()

    def receipt(self) -> Mapping[str, Any]:
        pack = dict(
            preference_pack_receipt(
                self.global_target_selector, self.pack_segments
            )
        )
        parallel = dict(self.parallel_state.receipt())
        value = {
            "schema_version": SCHEMA_VERSION,
            "total_tokens": self.total_tokens,
            "source_tokens": self.source_tokens,
            "target_tokens": self.target_tokens,
            "target_selector_sha256": target_selector_sha256(
                self.global_target_selector
            ),
            "preference_pack_receipt": pack,
            "parallel_state_receipt": parallel,
            "padding_policy": "append_false_then_contiguous_rank_chunk",
            "padded_tokens": self.padded_tokens,
            "sigma_schedule_index": self.sigma_schedule_index,
            "sigma_float32_be_hex": self.sigma_float32_be_hex,
            "sigma_gate": self.gate_name,
            "sigma_gate_weight": self.gate_weight,
            "enabled": self.enabled,
            "adapter_active": self.adapter_active,
        }
        return {**value, "digest": object_sha256(value)}


_ACTIVE_ROUTE: ContextVar[Optional[CAPERRoute]] = ContextVar(
    "bernini_caper_route", default=None
)


def active_route() -> Optional[CAPERRoute]:
    return _ACTIVE_ROUTE.get()


@contextmanager
def activate_route(route: CAPERRoute) -> Iterator[None]:
    if not isinstance(route, CAPERRoute):
        raise CAPERContractError("route must be CAPERRoute")
    if active_route() is not None:
        raise CAPERContractError("nested CAPER routes are forbidden")
    token: Token[Optional[CAPERRoute]] = _ACTIVE_ROUTE.set(route)
    try:
        yield
    finally:
        _ACTIVE_ROUTE.reset(token)


class CAPERTargetRowLoRA(nn.Module):
    """Rank-8 Q/O residual evaluated only for selected target rows."""

    def __init__(
        self,
        base: nn.Module,
        *,
        block_index: int,
        projection: str,
        expected_hidden_size: int,
    ) -> None:
        super().__init__()
        hidden = _positive_int(expected_hidden_size, label="expected_hidden_size")
        if type(base) is not nn.Linear:
            raise CAPERContractError(f"{projection} base must be exact nn.Linear")
        if projection not in CAPER_PROJECTIONS:
            raise CAPERContractError("CAPER may wrap only attn2 Q/O")
        if (
            isinstance(block_index, bool)
            or not isinstance(block_index, int)
            or block_index not in CAPER_BLOCK_INDICES
        ):
            raise CAPERContractError("CAPER block index must lie in [0,29]")
        if (base.in_features, base.out_features) != (hidden, hidden):
            raise CAPERContractError(
                f"block {block_index} {projection} shape is not ({hidden},{hidden})"
            )
        if base.weight.device.type == "meta" or base.weight.layout != torch.strided:
            raise CAPERContractError("CAPER base weight must have materialized strided storage")
        if any(parameter.requires_grad for parameter in base.parameters()):
            raise CAPERContractError("CAPER base projection must already be frozen")

        self.base = base
        self.block_index = block_index
        self.projection = projection
        self.rank = CAPER_RANK
        self.alpha = CAPER_ALPHA
        self.dropout = CAPER_DROPOUT
        self.caper_lora_A = nn.Linear(
            hidden, CAPER_RANK, bias=False, dtype=torch.float32
        )
        self.caper_lora_B = nn.Linear(
            CAPER_RANK, hidden, bias=False, dtype=torch.float32
        )
        nn.init.kaiming_uniform_(self.caper_lora_A.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.caper_lora_B.weight)

    @property
    def scale(self) -> float:
        return self.alpha / float(self.rank)

    @property
    def weight(self) -> Any:
        return self.base.weight

    @property
    def bias(self) -> Any:
        return self.base.bias

    @staticmethod
    def _selector(hidden_states: torch.Tensor, route: CAPERRoute) -> torch.Tensor:
        if hidden_states.ndim != 3 or int(hidden_states.shape[0]) != 1:
            raise CAPERContractError("CAPER projection expects hidden states [1,N,D]")
        selector = route.local_target_selector(device=hidden_states.device)
        if int(hidden_states.shape[1]) != int(selector.numel()):
            raise CAPERContractError(
                "local hidden sequence differs from append-pad/SP selector"
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
            delta = self.caper_lora_B(self.caper_lora_A(selected.float()))
            delta = delta * (self.scale * gate_weight)
        return delta.to(hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_output = self.base(hidden_states)
        route = active_route()
        # Exact-off means no selector construction and no A/B evaluation.
        if route is None or not route.adapter_active:
            return base_output
        selector = self._selector(hidden_states, route)
        if not bool(selector.any().item()):
            return base_output
        result = base_output.clone()
        result[:, selector, :] = (
            base_output[:, selector, :]
            + self._selected_delta(hidden_states, selector, route.gate_weight).to(
                base_output.dtype
            )
        )
        return result


def _named_modules_no_dedup(module: nn.Module) -> tuple[tuple[str, nn.Module], ...]:
    try:
        return tuple(module.named_modules(remove_duplicate=False))
    except TypeError as error:  # pragma: no cover - old torch is unsupported
        raise CAPERContractError(
            "PyTorch named_modules(remove_duplicate=False) is required for alias audit"
        ) from error


def _named_parameters_no_dedup(
    module: nn.Module,
) -> tuple[tuple[str, nn.Parameter], ...]:
    try:
        return tuple(module.named_parameters(remove_duplicate=False))
    except TypeError as error:  # pragma: no cover - old torch is unsupported
        raise CAPERContractError(
            "PyTorch named_parameters(remove_duplicate=False) is required for alias audit"
        ) from error


def _require_unique_paths_for_objects(
    rows: Sequence[tuple[str, Any]],
    required: Mapping[str, Any],
    *,
    label: str,
) -> None:
    paths_by_id: dict[int, list[str]] = {}
    object_by_path: dict[str, Any] = {}
    for name, value in rows:
        object_by_path[name] = value
        paths_by_id.setdefault(id(value), []).append(name)
    for name, expected in required.items():
        if object_by_path.get(name) is not expected:
            raise CAPERContractError(f"{label} path differs: {name}")
        aliases = paths_by_id.get(id(expected), [])
        if aliases != [name]:
            raise CAPERContractError(
                f"{label} module/parameter alias at {name}: {aliases}"
            )


def _named_tensor_sha256(
    rows: Sequence[tuple[str, nn.Parameter]], *, label: str
) -> str:
    digest = hashlib.sha256()
    seen_names: set[str] = set()
    seen_ids: set[int] = set()
    for name, parameter in sorted(rows, key=lambda item: item[0]):
        if name in seen_names or id(parameter) in seen_ids:
            raise CAPERContractError(f"{label} contains a name/storage alias")
        seen_names.add(name)
        seen_ids.add(id(parameter))
        if (
            not isinstance(parameter, nn.Parameter)
            or parameter.device.type == "meta"
            or parameter.layout != torch.strided
        ):
            raise CAPERContractError(f"{label} parameter {name} is not materialized strided")
        value = parameter.detach().contiguous()
        metadata = canonical_json_bytes(
            {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        )
        byte_view = value.view(torch.uint8).cpu()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(byte_view.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class _BaseParameterBinding:
    canonical_name: str
    registered_name: str
    parameter: nn.Parameter


def _canonical_adapter_parameter_rows(
    block_indices: Sequence[int],
    q_wrappers: Sequence[tuple[int, CAPERTargetRowLoRA]],
    o_wrappers: Sequence[tuple[int, CAPERTargetRowLoRA]],
) -> tuple[tuple[str, nn.Parameter], ...]:
    result: list[tuple[str, nn.Parameter]] = []
    q = dict(q_wrappers)
    o = dict(o_wrappers)
    for index in block_indices:
        for projection, wrapper in (
            ("attn2.to_q", q[index]),
            ("attn2.to_out.0", o[index]),
        ):
            prefix = f"{ROOT_NAMESPACE}.blocks.{index}.{projection}"
            result.extend(
                (
                    (f"{prefix}.caper_lora_A.weight", wrapper.caper_lora_A.weight),
                    (f"{prefix}.caper_lora_B.weight", wrapper.caper_lora_B.weight),
                )
            )
    if len({name for name, _ in result}) != len(result) or len(
        {id(parameter) for _, parameter in result}
    ) != len(result):
        raise CAPERContractError("CAPER A/B inventory contains an alias")
    if any(
        not parameter.requires_grad
        or parameter.dtype != torch.float32
        or parameter.device.type == "meta"
        for _, parameter in result
    ):
        raise CAPERContractError("CAPER A/B trainability or dtype differs")
    return tuple(result)


@dataclass
class CAPERHandle:
    renderer: nn.Module
    diffusion: nn.Module
    transformer: nn.Module
    block_indices: tuple[int, ...]
    expected_hidden_size: int
    q_wrappers: tuple[tuple[int, CAPERTargetRowLoRA], ...]
    o_wrappers: tuple[tuple[int, CAPERTargetRowLoRA], ...]
    original_q: tuple[tuple[int, nn.Linear], ...]
    original_o: tuple[tuple[int, nn.Linear], ...]
    base_parameter_bindings: tuple[_BaseParameterBinding, ...]
    initial_base_parameter_sha256: str
    initial_trainable_parameter_sha256: str
    protected_modules: tuple[tuple[str, nn.Module], ...]
    block_objects: tuple[nn.Module, ...]
    restored: bool = False

    def trainable_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        if self.restored:
            raise CAPERContractError("CAPER adapter has been restored")
        return _canonical_adapter_parameter_rows(
            self.block_indices, self.q_wrappers, self.o_wrappers
        )

    @contextmanager
    def route(self, route: CAPERRoute) -> Iterator[None]:
        self.assert_scope()
        with activate_route(route):
            yield

    def assert_scope(self) -> None:
        if self.restored:
            raise CAPERContractError("CAPER adapter has been restored")
        if (
            getattr(self.renderer, "diff_dec", None) is not self.diffusion
            or getattr(self.diffusion, "transformer", None) is not self.transformer
            or getattr(self.diffusion, "transformer_2", None) is not None
        ):
            raise CAPERContractError("native diff_dec.transformer namespace changed")
        if bool(getattr(self.transformer, "gradient_checkpointing", False)):
            raise CAPERContractError(
                "CAPER requires gradient checkpointing disabled so its route covers backward"
            )
        blocks = tuple(getattr(self.transformer, "blocks", ()))
        if len(blocks) != len(self.block_objects) or any(
            current is not expected
            for current, expected in zip(blocks, self.block_objects)
        ):
            raise CAPERContractError("Bernini block inventory or identity changed")

        current_modules = _named_modules_no_dedup(self.renderer)
        protected = dict(self.protected_modules)
        _require_unique_paths_for_objects(
            current_modules, protected, label="protected frozen module"
        )

        q = dict(self.q_wrappers)
        o = dict(self.o_wrappers)
        original_q = dict(self.original_q)
        original_o = dict(self.original_o)
        wrappers: dict[str, nn.Module] = {}
        for index in range(TOTAL_BLOCKS_1P3B):
            block = blocks[index]
            query = block.attn2.to_q
            output = block.attn2.to_out[0]
            if index in self.block_indices:
                if query is not q[index] or output is not o[index]:
                    raise CAPERContractError(f"block {index} CAPER wrapper identity differs")
                if query.base is not original_q[index] or output.base is not original_o[index]:
                    raise CAPERContractError(f"block {index} CAPER frozen base differs")
                wrappers[f"{ROOT_NAMESPACE}.blocks.{index}.attn2.to_q"] = query
                wrappers[f"{ROOT_NAMESPACE}.blocks.{index}.attn2.to_out.0"] = output
            elif query is not original_q[index] or output is not original_o[index]:
                raise CAPERContractError(
                    f"block {index} changed outside the registered CAPER scope"
                )
        _require_unique_paths_for_objects(
            current_modules, wrappers, label="CAPER wrapper allowlist"
        )

        current_parameters = _named_parameters_no_dedup(self.renderer)
        base_required = {
            binding.registered_name: binding.parameter
            for binding in self.base_parameter_bindings
        }
        _require_unique_paths_for_objects(
            current_parameters, base_required, label="frozen base parameter"
        )
        if any(binding.parameter.requires_grad for binding in self.base_parameter_bindings):
            raise CAPERContractError("a frozen Bernini transformer parameter became trainable")

        trainable = self.trainable_named_parameters()
        expected_trainable = dict(trainable)
        _require_unique_paths_for_objects(
            current_parameters, expected_trainable, label="CAPER trainable parameter"
        )
        observed_trainable = {
            name: parameter
            for name, parameter in current_parameters
            if parameter.requires_grad
        }
        if set(observed_trainable) != set(expected_trainable) or any(
            observed_trainable[name] is not parameter
            for name, parameter in expected_trainable.items()
        ):
            leaked = sorted(set(observed_trainable) - set(expected_trainable))
            missing = sorted(set(expected_trainable) - set(observed_trainable))
            raise CAPERContractError(
                f"trainable parameter leakage: leaked={leaked[:4]} missing={missing[:4]}"
            )

    def _current_base_sha256(self) -> str:
        rows = tuple(
            (binding.canonical_name, binding.parameter)
            for binding in self.base_parameter_bindings
        )
        return _named_tensor_sha256(rows, label="frozen Bernini transformer state")

    def trainable_parameter_values_sha256(self) -> str:
        """Return the current, name/shape/dtype/value-bound A/B state hash."""

        self.assert_scope()
        return _named_tensor_sha256(
            self.trainable_named_parameters(), label="CAPER A/B trainable state"
        )

    def freeze_checksum_certificate(self) -> Mapping[str, Any]:
        """Hash and certify the exact frozen-transformer/trainable closure."""

        self.assert_scope()
        current_sha = self._current_base_sha256()
        if current_sha != self.initial_base_parameter_sha256:
            raise CAPERContractError("frozen Bernini transformer checksum changed")
        trainable = self.trainable_named_parameters()
        current_trainable_sha = _named_tensor_sha256(
            trainable, label="CAPER A/B trainable state"
        )
        target_modules = canonical_target_module_names(self.block_indices)
        protected_names = [name for name, _ in self.protected_modules]
        value = {
            "schema_version": CERTIFICATE_SCHEMA,
            "adapter_schema_version": SCHEMA_VERSION,
            "root_namespace": ROOT_NAMESPACE,
            "block_indices": list(self.block_indices),
            "default_all_30_blocks": self.block_indices == CAPER_BLOCK_INDICES,
            "target_modules": list(target_modules),
            "target_module_count": len(target_modules),
            "target_modules_sha256": object_sha256(list(target_modules)),
            "rank": CAPER_RANK,
            "alpha": CAPER_ALPHA,
            "dropout": CAPER_DROPOUT,
            "expected_hidden_size": self.expected_hidden_size,
            "target_row_only": True,
            "wrapped_projection_source_and_padding_rows_byte_exact": True,
            "source_exactness_scope": "wrapped_attn2_q_and_o_projection_output_only",
            "global_source_activation_byte_exact_after_joint_attn1": False,
            "global_source_activation_disclaimer": (
                "later joint attn1 may mix target-row changes back into source rows"
            ),
            "exact40_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
            "sigma_gate": {
                "high_weight_1": list(HIGH_SIGMA_INDICES),
                "mid_weight_0.5": list(MID_SIGMA_INDICES),
                "low_direct_base": list(LOW_SIGMA_INDICES),
            },
            "low_sigma_evaluates_lora": False,
            "frozen_transformer_parameter_tensor_count": len(
                self.base_parameter_bindings
            ),
            "frozen_transformer_parameter_elements": sum(
                int(binding.parameter.numel())
                for binding in self.base_parameter_bindings
            ),
            "frozen_transformer_initial_sha256": self.initial_base_parameter_sha256,
            "frozen_transformer_current_sha256": current_sha,
            "frozen_transformer_byte_exact": True,
            "protected_module_names": protected_names,
            "protected_modules_sha256": object_sha256(protected_names),
            "every_unregistered_transformer_module_identity_exact": True,
            "key_value_trainable": False,
            "attn1_trainable": False,
            "ffn_trainable": False,
            "patch_embedding_trainable": False,
            "proj_out_trainable": False,
            "trainable_parameter_names": [name for name, _ in trainable],
            "trainable_parameter_tensor_count": len(trainable),
            "trainable_parameter_elements": sum(
                int(parameter.numel()) for _, parameter in trainable
            ),
            "trainable_parameter_names_sha256": object_sha256(
                [name for name, _ in trainable]
            ),
            "trainable_parameter_initial_values_sha256": (
                self.initial_trainable_parameter_sha256
            ),
            "trainable_parameter_current_values_sha256": current_trainable_sha,
            "trainable_parameter_values_hashed": True,
            "trainable_parameter_value_change_allowed": True,
            "trainable_scope_closed": True,
            "gradient_checkpointing_must_be_disabled": True,
            "route_context_must_cover_forward_and_backward": True,
            "semantic_action_editing_claim": False,
        }
        return {**value, "digest": object_sha256(value)}

    def restore(self) -> None:
        if self.restored or active_route() is not None:
            raise CAPERContractError("CAPER adapter cannot be restored now")
        self.assert_scope()
        blocks = tuple(self.transformer.blocks)
        for index, original in self.original_q:
            if index in self.block_indices:
                blocks[index].attn2.to_q = original
        for index, original in self.original_o:
            if index in self.block_indices:
                blocks[index].attn2.to_out[0] = original
        self.restored = True


def _registered_name_after_wrap(
    canonical_name: str, target_modules: set[str]
) -> str:
    for target in target_modules:
        prefix = f"{target}."
        if canonical_name.startswith(prefix):
            return f"{target}.base.{canonical_name[len(prefix):]}"
    return canonical_name


def install_caper_capacity_probe(
    renderer: nn.Module,
    *,
    block_indices: Sequence[int] = CAPER_BLOCK_INDICES,
    expected_hidden_size: int = BERNINI_1P3B_HIDDEN_SIZE,
) -> CAPERHandle:
    """Install the closed rank-8 target-row Q/O CAPER capacity probe.

    ``expected_hidden_size`` defaults to the released 1.3B width.  An explicit
    smaller value exists only so the same fail-closed injector can be exercised
    with lightweight structural test doubles.
    """

    if not isinstance(renderer, nn.Module):
        raise CAPERContractError("renderer must be nn.Module")
    blocks_to_wrap = _validate_block_indices(block_indices)
    hidden = _positive_int(expected_hidden_size, label="expected_hidden_size")
    if any(parameter.requires_grad for parameter in renderer.parameters()):
        raise CAPERContractError("freeze the complete renderer before CAPER injection")

    diffusion = getattr(renderer, "diff_dec", None)
    if not isinstance(diffusion, nn.Module):
        raise CAPERContractError("renderer.diff_dec must be one registered nn.Module")
    transformer = getattr(diffusion, "transformer", None)
    if not isinstance(transformer, nn.Module) or getattr(diffusion, "transformer_2", None) is not None:
        raise CAPERContractError("CAPER requires only diff_dec.transformer_1")
    if bool(getattr(transformer, "gradient_checkpointing", False)):
        raise CAPERContractError(
            "disable gradient checkpointing before CAPER injection"
        )
    blocks = tuple(getattr(transformer, "blocks", ()))
    patch = getattr(transformer, "patch_embedding", None)
    proj_out = getattr(transformer, "proj_out", None)
    if (
        not isinstance(getattr(transformer, "blocks", None), nn.ModuleList)
        or len(blocks) != TOTAL_BLOCKS_1P3B
        or type(patch) is not nn.Conv3d
        or int(patch.out_channels) != hidden
        or tuple(int(item) for item in patch.kernel_size) != (1, 2, 2)
        or not isinstance(proj_out, nn.Module)
        or not callable(getattr(transformer, "patch_vae_latent", None))
    ):
        raise CAPERContractError("Bernini-R 1.3B transformer structure differs")

    before_modules = _named_modules_no_dedup(renderer)
    module_by_name = dict(before_modules)
    if module_by_name.get("diff_dec") is not diffusion or module_by_name.get(
        ROOT_NAMESPACE
    ) is not transformer:
        raise CAPERContractError("native diff_dec.transformer registration differs")

    originals_q: list[tuple[int, nn.Linear]] = []
    originals_o: list[tuple[int, nn.Linear]] = []
    all_targets: dict[str, nn.Module] = {}
    protected: dict[str, nn.Module] = {
        f"{ROOT_NAMESPACE}.patch_embedding": patch,
        f"{ROOT_NAMESPACE}.proj_out": proj_out,
    }
    for index, block in enumerate(blocks):
        self_attention = getattr(block, "attn1", None)
        cross_attention = getattr(block, "attn2", None)
        ffn = getattr(block, "ffn", None)
        query = getattr(cross_attention, "to_q", None)
        key = getattr(cross_attention, "to_k", None)
        value = getattr(cross_attention, "to_v", None)
        output = getattr(cross_attention, "to_out", None)
        if (
            not isinstance(self_attention, nn.Module)
            or not isinstance(cross_attention, nn.Module)
            or not isinstance(ffn, nn.Module)
            or type(query) is not nn.Linear
            or not isinstance(key, nn.Module)
            or not isinstance(value, nn.Module)
            or type(output) is not nn.ModuleList
            or len(output) != 2
            or type(output[0]) is not nn.Linear
            or (query.in_features, query.out_features) != (hidden, hidden)
            or (output[0].in_features, output[0].out_features) != (hidden, hidden)
            or query.weight.device.type == "meta"
            or output[0].weight.device.type == "meta"
        ):
            raise CAPERContractError(f"block {index} native attn2 Q/O structure differs")
        originals_q.append((index, query))
        originals_o.append((index, output[0]))
        q_name = f"{ROOT_NAMESPACE}.blocks.{index}.attn2.to_q"
        o_name = f"{ROOT_NAMESPACE}.blocks.{index}.attn2.to_out.0"
        all_targets[q_name] = query
        all_targets[o_name] = output[0]
        protected.update(
            {
                f"{ROOT_NAMESPACE}.blocks.{index}.attn1": self_attention,
                f"{ROOT_NAMESPACE}.blocks.{index}.attn2.to_k": key,
                f"{ROOT_NAMESPACE}.blocks.{index}.attn2.to_v": value,
                f"{ROOT_NAMESPACE}.blocks.{index}.ffn": ffn,
            }
        )

    if tuple(all_targets) != CAPER_TARGET_MODULES:
        raise CAPERContractError("all-30 Bernini attn2 Q/O module allowlist differs")
    # Close the structural universe, not only the named forbidden families:
    # every transformer module except the 60 replaceable Q/O leaves must keep
    # its exact registered path and Python object identity after injection.
    for name, module in before_modules:
        if (
            name == ROOT_NAMESPACE or name.startswith(f"{ROOT_NAMESPACE}.")
        ) and name not in all_targets:
            protected.setdefault(name, module)
    _require_unique_paths_for_objects(
        before_modules, all_targets, label="native target module"
    )
    _require_unique_paths_for_objects(
        before_modules, protected, label="protected frozen module"
    )

    transformer_parameters = _named_parameters_no_dedup(transformer)
    canonical_base_rows = tuple(
        (f"{ROOT_NAMESPACE}.{name}", parameter)
        for name, parameter in transformer_parameters
    )
    if len({name for name, _ in canonical_base_rows}) != len(canonical_base_rows) or len(
        {id(parameter) for _, parameter in canonical_base_rows}
    ) != len(canonical_base_rows):
        raise CAPERContractError("Wan transformer contains a parameter alias")
    if any(parameter.requires_grad for _, parameter in canonical_base_rows):
        raise CAPERContractError("Wan transformer base is not completely frozen")
    initial_base_sha = _named_tensor_sha256(
        canonical_base_rows, label="frozen Bernini transformer state"
    )

    target_modules = set(canonical_target_module_names(blocks_to_wrap))
    bindings = tuple(
        _BaseParameterBinding(
            canonical_name=name,
            registered_name=_registered_name_after_wrap(name, target_modules),
            parameter=parameter,
        )
        for name, parameter in canonical_base_rows
    )
    original_q_by_index = dict(originals_q)
    original_o_by_index = dict(originals_o)
    q_wrappers: list[tuple[int, CAPERTargetRowLoRA]] = []
    o_wrappers: list[tuple[int, CAPERTargetRowLoRA]] = []
    try:
        for index in blocks_to_wrap:
            query = original_q_by_index[index]
            output = original_o_by_index[index]
            q_wrapper = CAPERTargetRowLoRA(
                query,
                block_index=index,
                projection="attn2.to_q",
                expected_hidden_size=hidden,
            ).to(device=query.weight.device)
            o_wrapper = CAPERTargetRowLoRA(
                output,
                block_index=index,
                projection="attn2.to_out.0",
                expected_hidden_size=hidden,
            ).to(device=output.weight.device)
            blocks[index].attn2.to_q = q_wrapper
            blocks[index].attn2.to_out[0] = o_wrapper
            q_wrappers.append((index, q_wrapper))
            o_wrappers.append((index, o_wrapper))
        initial_trainable_sha = _named_tensor_sha256(
            _canonical_adapter_parameter_rows(
                blocks_to_wrap, tuple(q_wrappers), tuple(o_wrappers)
            ),
            label="initial CAPER A/B trainable state",
        )
    except Exception:
        for index in blocks_to_wrap:
            blocks[index].attn2.to_q = original_q_by_index[index]
            blocks[index].attn2.to_out[0] = original_o_by_index[index]
        raise

    handle = CAPERHandle(
        renderer=renderer,
        diffusion=diffusion,
        transformer=transformer,
        block_indices=blocks_to_wrap,
        expected_hidden_size=hidden,
        q_wrappers=tuple(q_wrappers),
        o_wrappers=tuple(o_wrappers),
        original_q=tuple(originals_q),
        original_o=tuple(originals_o),
        base_parameter_bindings=bindings,
        initial_base_parameter_sha256=initial_base_sha,
        initial_trainable_parameter_sha256=initial_trainable_sha,
        protected_modules=tuple(protected.items()),
        block_objects=blocks,
    )
    try:
        handle.assert_scope()
    except Exception:
        for index in blocks_to_wrap:
            blocks[index].attn2.to_q = original_q_by_index[index]
            blocks[index].attn2.to_out[0] = original_o_by_index[index]
        handle.restored = True
        raise
    return handle


__all__ = [
    "ALLOWED_SP_SIZES",
    "BERNINI_1P3B_HIDDEN_SIZE",
    "CAPER_ALPHA",
    "CAPER_BLOCK_INDICES",
    "CAPER_DROPOUT",
    "CAPER_PROJECTIONS",
    "CAPER_RANK",
    "CAPER_TARGET_MODULES",
    "CAPER_TARGET_MODULES_SHA256",
    "CAPERContractError",
    "CAPERHandle",
    "CAPERPackSegment",
    "CAPERParallelState",
    "CAPERRoute",
    "CAPERTargetRowLoRA",
    "CERTIFICATE_SCHEMA",
    "HIGH_SIGMA_INDICES",
    "LOW_SIGMA_INDICES",
    "MID_SIGMA_INDICES",
    "PACK_RECEIPT_SCHEMA",
    "PARALLEL_RECEIPT_SCHEMA",
    "PREFERENCE_PACK_LAYOUT",
    "ROOT_NAMESPACE",
    "SCHEMA_VERSION",
    "SP1_TEST_AUTHORITY_ID",
    "activate_route",
    "active_route",
    "canonical_target_module_names",
    "install_caper_capacity_probe",
    "preference_pack_receipt",
    "preference_pack_segments",
    "preference_pack_target_selector",
    "sigma_gate",
    "snapshot_live_bernini_parallel_state",
    "target_selector_sha256",
]
