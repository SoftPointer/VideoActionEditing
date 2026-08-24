#!/usr/bin/env python3
"""Fail-closed contracts for EPMC K=2 episode parallelism.

This module does not launch distributed workers and does not import PyTorch.
It describes the only eight-rank schedule that preserves the scientific
meaning of the four-rank serial EPMC experiment:

* ranks 0..3 independently invert support 1 with Ulysses-4;
* ranks 4..7 independently invert support 2 with Ulysses-4;
* gradients are reduced only inside the owning Ulysses group and divided by 4;
* detached support artifacts may cross the DP groups only after the two inner
  loops have finished; and
* the K=2 prototype, reference probes, and held-noise statistics retain their
  original serial definitions.

The collective surface is deliberately dependency-injected.  CPU unit tests
can therefore reject a WORLD-group reduction, division by eight, a shuffled or
duplicated episode, or a support-2 probe without initializing a process group.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
import re
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence


WORLD_SIZE = 8
DATA_PARALLEL_SIZE = 2
ULYSSES_SIZE = 4
SUPPORT_COUNT = 2
SUPPORT_INDICES = (1, 2)
REFERENCE_PROBE_SUPPORT_INDEX = 1
REFERENCE_PROBE_FAMILIES = ("phase_only", "block_only")
HELD_CONTROL_NAMES = ("zero", "correct", "reverse", "shuffle")
GRADIENT_DIVISOR = ULYSSES_SIZE
AGGREGATION_RULE = "exact_arithmetic_midpoint_in_decoded_fp32_gate_space"

_SHA256 = re.compile(r"[0-9a-f]{64}")


class FewShotEpisodeParallelError(RuntimeError):
    """Raised before an eight-rank schedule can change K=2 semantics."""


def _exact_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FewShotEpisodeParallelError(f"{label} must be an exact integer")
    return value


def _rank(value: Any) -> int:
    rank = _exact_int(value, label="rank")
    if not 0 <= rank < WORLD_SIZE:
        raise FewShotEpisodeParallelError(
            f"rank must lie in [0,{WORLD_SIZE - 1}]"
        )
    return rank


def _support_index(value: Any) -> int:
    index = _exact_int(value, label="support_index")
    if index not in SUPPORT_INDICES:
        raise FewShotEpisodeParallelError("support_index must be exactly 1 or 2")
    return index


def _iid(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise FewShotEpisodeParallelError("IID must be a non-empty stripped string")
    if "\x00" in value:
        raise FewShotEpisodeParallelError("IID cannot contain NUL")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FewShotEpisodeParallelError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True)
class GroupSpec:
    """Auditable logical group passed to a dependency-injected collective."""

    kind: str
    index: int
    ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.kind not in {"world", "ulysses", "data_parallel"}:
            raise FewShotEpisodeParallelError(f"unknown group kind {self.kind!r}")
        _exact_int(self.index, label="group index")
        if (
            not isinstance(self.ranks, tuple)
            or not self.ranks
            or any(isinstance(rank, bool) or not isinstance(rank, int) for rank in self.ranks)
            or len(set(self.ranks)) != len(self.ranks)
            or tuple(sorted(self.ranks)) != self.ranks
            or any(not 0 <= rank < WORLD_SIZE for rank in self.ranks)
        ):
            raise FewShotEpisodeParallelError(
                "group ranks must be unique ordered world-8 ranks"
            )

    @property
    def name(self) -> str:
        return f"{self.kind}:{self.index}"

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "index": self.index, "ranks": list(self.ranks)}


@dataclass(frozen=True)
class RankAssignment:
    rank: int
    dp_rank: int
    ulysses_rank: int
    support_index: int
    ulysses_group: GroupSpec
    data_parallel_group: GroupSpec

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "dp_rank": self.dp_rank,
            "ulysses_rank": self.ulysses_rank,
            "support_index": self.support_index,
            "ulysses_group": self.ulysses_group.as_dict(),
            "data_parallel_group": self.data_parallel_group.as_dict(),
        }


@dataclass(frozen=True)
class EpisodeParallelTopology:
    """The one supported DP=2 x Ulysses=4 topology."""

    world_size: int = WORLD_SIZE
    data_parallel_size: int = DATA_PARALLEL_SIZE
    ulysses_size: int = ULYSSES_SIZE

    def __post_init__(self) -> None:
        observed = (self.world_size, self.data_parallel_size, self.ulysses_size)
        expected = (WORLD_SIZE, DATA_PARALLEL_SIZE, ULYSSES_SIZE)
        if observed != expected:
            raise FewShotEpisodeParallelError(
                "EPMC episode parallelism is frozen to world=8, DP=2, Ulysses=4"
            )
        if self.data_parallel_size * self.ulysses_size != self.world_size:
            raise FewShotEpisodeParallelError("parallel dimensions do not multiply to 8")

    @property
    def world_group(self) -> GroupSpec:
        return GroupSpec("world", 0, tuple(range(WORLD_SIZE)))

    @property
    def ulysses_groups(self) -> tuple[GroupSpec, GroupSpec]:
        return (
            GroupSpec("ulysses", 0, (0, 1, 2, 3)),
            GroupSpec("ulysses", 1, (4, 5, 6, 7)),
        )

    @property
    def data_parallel_groups(self) -> tuple[GroupSpec, ...]:
        return tuple(
            GroupSpec("data_parallel", index, (index, index + ULYSSES_SIZE))
            for index in range(ULYSSES_SIZE)
        )

    def assignment(self, rank: int) -> RankAssignment:
        rank = _rank(rank)
        dp_rank = rank // ULYSSES_SIZE
        ulysses_rank = rank % ULYSSES_SIZE
        return RankAssignment(
            rank=rank,
            dp_rank=dp_rank,
            ulysses_rank=ulysses_rank,
            support_index=dp_rank + 1,
            ulysses_group=self.ulysses_groups[dp_rank],
            data_parallel_group=self.data_parallel_groups[ulysses_rank],
        )

    def receipt(self) -> dict[str, Any]:
        return {
            "world_size": self.world_size,
            "data_parallel_size": self.data_parallel_size,
            "ulysses_size": self.ulysses_size,
            "semantics": "two_independent_support_inner_loops",
            "gradient_group": "support_owning_ulysses_group_only",
            "gradient_divisor": GRADIENT_DIVISOR,
            "cross_dp_gradient_sync": False,
            "rank_assignments": [self.assignment(rank).as_dict() for rank in range(WORLD_SIZE)],
        }


DEFAULT_TOPOLOGY = EpisodeParallelTopology()


class Collective(Protocol):
    """Minimal mockable collective API; logical ``GroupSpec`` is mandatory."""

    def all_reduce_sum(self, value: Any, *, group: GroupSpec) -> Any:
        ...

    def all_gather_object(self, value: Any, *, group: GroupSpec) -> Sequence[Any]:
        ...


def mean_support_gradient(
    local_gradient: Any,
    *,
    rank: int,
    support_index: int,
    group: GroupSpec,
    divisor: int,
    collective: Collective,
    topology: EpisodeParallelTopology = DEFAULT_TOPOLOGY,
) -> Any:
    """Reduce one support code gradient without ever crossing the DP axis."""

    assignment = topology.assignment(rank)
    support_index = _support_index(support_index)
    if support_index != assignment.support_index:
        raise FewShotEpisodeParallelError(
            "rank-to-support assignment differs from the frozen episode schedule"
        )
    if group != assignment.ulysses_group:
        raise FewShotEpisodeParallelError(
            "support gradient must use its exact Ulysses-4 group, never WORLD/DP"
        )
    divisor = _exact_int(divisor, label="gradient divisor")
    if divisor != GRADIENT_DIVISOR or divisor != len(group.ranks):
        raise FewShotEpisodeParallelError(
            "support gradient divisor must be exactly 4, never DP=2 or world=8"
        )
    reducer = getattr(collective, "all_reduce_sum", None)
    if not callable(reducer):
        raise FewShotEpisodeParallelError("collective lacks callable all_reduce_sum")
    reduced = reducer(local_gradient, group=group)
    try:
        return reduced / divisor
    except Exception as error:
        raise FewShotEpisodeParallelError(
            "reduced gradient does not support exact division by four"
        ) from error


def _assert_detached(value: Any, *, path: str = "code") -> None:
    """Recursively reject an autograd-bearing payload without importing torch."""

    if value is None:
        # ``grad_fn=None`` is the canonical detached state and commonly appears
        # as a field on tensor-like test doubles and immutable code containers.
        return
    requires_grad = getattr(value, "requires_grad", False)
    try:
        attached = bool(requires_grad)
    except Exception as error:
        raise FewShotEpisodeParallelError(
            f"cannot establish detached state for {path}"
        ) from error
    if attached:
        raise FewShotEpisodeParallelError(f"{path} still requires gradients")
    if getattr(value, "grad_fn", None) is not None:
        raise FewShotEpisodeParallelError(f"{path} still has an autograd grad_fn")

    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _assert_detached(getattr(value, field.name), path=f"{path}.{field.name}")
    elif isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: repr(item)):
            _assert_detached(value[key], path=f"{path}[{key!r}]")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_detached(item, path=f"{path}[{index}]")


def _held_losses(value: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, Mapping) or set(value) != set(HELD_CONTROL_NAMES):
        raise FewShotEpisodeParallelError(
            "held losses must contain exactly zero/correct/reverse/shuffle"
        )
    result: list[tuple[str, float]] = []
    for name in HELD_CONTROL_NAMES:
        raw = value[name]
        if isinstance(raw, bool):
            raise FewShotEpisodeParallelError(f"held loss {name} must be numeric")
        try:
            numeric = float(raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise FewShotEpisodeParallelError(
                f"held loss {name} must be numeric"
            ) from error
        if not math.isfinite(numeric) or numeric < 0.0:
            raise FewShotEpisodeParallelError(
                f"held loss {name} must be finite and nonnegative"
            )
        result.append((name, numeric))
    return tuple(result)


@dataclass(frozen=True)
class DetachedSupportPayload:
    """One immutable support result allowed to cross the DP axis."""

    support_index: int
    iid: str
    producer_rank: int
    code: Any
    code_sha256: str
    held_losses: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        _support_index(self.support_index)
        _iid(self.iid)
        _rank(self.producer_rank)
        _sha256(self.code_sha256, label="code_sha256")
        _assert_detached(self.code)
        canonical = _held_losses(dict(self.held_losses))
        if canonical != self.held_losses:
            raise FewShotEpisodeParallelError(
                "held losses are not in the frozen canonical order"
            )

    @property
    def held_loss_map(self) -> dict[str, float]:
        return dict(self.held_losses)

    @property
    def semantic_sha256(self) -> str:
        """Digest excluding producer rank so four Ulysses replicas can agree."""

        value = {
            "support_index": self.support_index,
            "iid": self.iid,
            "code_sha256": self.code_sha256,
            "held_losses_hex": [
                [name, float(loss).hex()] for name, loss in self.held_losses
            ],
        }
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def freeze_support_payload(
    *,
    support_index: int,
    iid: str,
    producer_rank: int,
    code: Any,
    code_sha256: str,
    held_losses: Mapping[str, Any],
    detach_code: Optional[Callable[[Any], Any]] = None,
    topology: EpisodeParallelTopology = DEFAULT_TOPOLOGY,
) -> DetachedSupportPayload:
    """Detach and bind one support result to its owning episode rank."""

    assignment = topology.assignment(producer_rank)
    support_index = _support_index(support_index)
    if assignment.support_index != support_index:
        raise FewShotEpisodeParallelError(
            "payload producer rank belongs to the other support"
        )
    detached = detach_code(code) if detach_code is not None else code
    if detached is None:
        raise FewShotEpisodeParallelError("detached code cannot be None")
    _assert_detached(detached)
    return DetachedSupportPayload(
        support_index=support_index,
        iid=_iid(iid),
        producer_rank=assignment.rank,
        code=detached,
        code_sha256=_sha256(code_sha256, label="code_sha256"),
        held_losses=_held_losses(held_losses),
    )


@dataclass(frozen=True)
class SupportReplicaCertificate:
    support_index: int
    iid: str
    semantic_sha256: str
    group: GroupSpec


def certify_support_replicas(
    payload: DetachedSupportPayload,
    *,
    rank: int,
    group: GroupSpec,
    collective: Collective,
    topology: EpisodeParallelTopology = DEFAULT_TOPOLOGY,
) -> SupportReplicaCertificate:
    """Require all four ranks of one Ulysses worker to hold one exact result."""

    assignment = topology.assignment(rank)
    if payload.producer_rank != assignment.rank:
        raise FewShotEpisodeParallelError("local payload producer rank differs")
    if payload.support_index != assignment.support_index:
        raise FewShotEpisodeParallelError("local payload belongs to another support")
    if group != assignment.ulysses_group:
        raise FewShotEpisodeParallelError(
            "support replica certification requires the exact Ulysses group"
        )
    gather = getattr(collective, "all_gather_object", None)
    if not callable(gather):
        raise FewShotEpisodeParallelError("collective lacks callable all_gather_object")
    observed = tuple(gather(payload.semantic_sha256, group=group))
    if len(observed) != ULYSSES_SIZE or any(
        item != payload.semantic_sha256 for item in observed
    ):
        raise FewShotEpisodeParallelError(
            "Ulysses ranks do not hold byte-identical detached support evidence"
        )
    return SupportReplicaCertificate(
        support_index=payload.support_index,
        iid=payload.iid,
        semantic_sha256=payload.semantic_sha256,
        group=group,
    )


def canonical_two_support_payloads(
    payloads: Sequence[DetachedSupportPayload],
) -> tuple[DetachedSupportPayload, DetachedSupportPayload]:
    """Validate K=2 membership and return support-index order, never arrival order."""

    values = tuple(payloads)
    if len(values) != SUPPORT_COUNT or any(
        not isinstance(item, DetachedSupportPayload) for item in values
    ):
        raise FewShotEpisodeParallelError("exchange must contain exactly two support payloads")
    if {item.support_index for item in values} != set(SUPPORT_INDICES):
        raise FewShotEpisodeParallelError(
            "exchange must contain support 1 and support 2 exactly once"
        )
    iids = [item.iid for item in values]
    if len(set(iids)) != SUPPORT_COUNT:
        raise FewShotEpisodeParallelError("the two supports must have distinct IIDs")
    ordered = tuple(sorted(values, key=lambda item: item.support_index))
    return ordered[0], ordered[1]


def canonical_exchange_support_payloads(
    payload: DetachedSupportPayload,
    certificate: SupportReplicaCertificate,
    *,
    rank: int,
    group: GroupSpec,
    collective: Collective,
    topology: EpisodeParallelTopology = DEFAULT_TOPOLOGY,
) -> tuple[DetachedSupportPayload, DetachedSupportPayload]:
    """Exchange only frozen support results across one DP column."""

    assignment = topology.assignment(rank)
    if payload.producer_rank != assignment.rank:
        raise FewShotEpisodeParallelError("local payload producer rank differs")
    if payload.support_index != assignment.support_index:
        raise FewShotEpisodeParallelError("local payload belongs to another support")
    if (
        certificate.support_index != payload.support_index
        or certificate.iid != payload.iid
        or certificate.semantic_sha256 != payload.semantic_sha256
        or certificate.group != assignment.ulysses_group
    ):
        raise FewShotEpisodeParallelError(
            "support payload lacks a matching Ulysses replica certificate"
        )
    if group != assignment.data_parallel_group:
        raise FewShotEpisodeParallelError(
            "detached support exchange requires the exact two-rank DP column"
        )
    gather = getattr(collective, "all_gather_object", None)
    if not callable(gather):
        raise FewShotEpisodeParallelError("collective lacks callable all_gather_object")
    observed = tuple(gather(payload, group=group))
    ordered = canonical_two_support_payloads(observed)
    for item in ordered:
        if item.producer_rank not in group.ranks:
            raise FewShotEpisodeParallelError(
                "exchanged payload producer is outside the DP column"
            )
        expected = topology.assignment(item.producer_rank)
        if expected.support_index != item.support_index:
            raise FewShotEpisodeParallelError(
                "exchanged payload rank-to-support binding differs"
            )
    if not any(
        item.producer_rank == assignment.rank
        and item.semantic_sha256 == payload.semantic_sha256
        for item in ordered
    ):
        raise FewShotEpisodeParallelError("DP exchange did not return the local payload")
    return ordered


@dataclass(frozen=True)
class CanonicalPrototype:
    code: Any
    support_iids: tuple[str, str]
    support_code_sha256: tuple[str, str]
    aggregation_rule: str = AGGREGATION_RULE

    def __post_init__(self) -> None:
        _assert_detached(self.code, path="prototype")


def build_canonical_prototype(
    payloads: Sequence[DetachedSupportPayload],
    *,
    midpoint: Callable[[Any, Any], Any],
) -> CanonicalPrototype:
    """Call the injected FP32 gate midpoint in support-1/support-2 order."""

    if not callable(midpoint):
        raise FewShotEpisodeParallelError("midpoint must be callable")
    first, second = canonical_two_support_payloads(payloads)
    code = midpoint(first.code, second.code)
    if code is None:
        raise FewShotEpisodeParallelError("midpoint returned no prototype code")
    _assert_detached(code, path="prototype")
    return CanonicalPrototype(
        code=code,
        support_iids=(first.iid, second.iid),
        support_code_sha256=(first.code_sha256, second.code_sha256),
    )


@dataclass(frozen=True)
class ProbeEvidence:
    support_index: int
    iid: str
    family: str
    passed: bool

    def __post_init__(self) -> None:
        _support_index(self.support_index)
        _iid(self.iid)
        if self.family not in REFERENCE_PROBE_FAMILIES:
            raise FewShotEpisodeParallelError(f"unknown probe family {self.family!r}")
        if not isinstance(self.passed, bool):
            raise FewShotEpisodeParallelError("probe passed must be boolean")


def canonical_reference_probes(
    probes: Sequence[ProbeEvidence],
    payloads: Sequence[DetachedSupportPayload],
) -> tuple[ProbeEvidence, ProbeEvidence]:
    """Keep the serial reference: phase/block probes come from support 1 only."""

    support_one, _ = canonical_two_support_payloads(payloads)
    values = tuple(probes)
    if len(values) != len(REFERENCE_PROBE_FAMILIES):
        raise FewShotEpisodeParallelError("exactly two reference probes are required")
    if any(not isinstance(item, ProbeEvidence) for item in values):
        raise FewShotEpisodeParallelError("probe evidence has the wrong type")
    if any(
        item.support_index != REFERENCE_PROBE_SUPPORT_INDEX
        or item.iid != support_one.iid
        for item in values
    ):
        raise FewShotEpisodeParallelError(
            "only support 1 may supply the serial-reference gradient probes"
        )
    by_family = {item.family: item for item in values}
    if len(by_family) != len(REFERENCE_PROBE_FAMILIES) or set(by_family) != set(
        REFERENCE_PROBE_FAMILIES
    ):
        raise FewShotEpisodeParallelError(
            "reference probes must contain phase_only and block_only exactly once"
        )
    return tuple(by_family[name] for name in REFERENCE_PROBE_FAMILIES)  # type: ignore[return-value]


@dataclass(frozen=True)
class HeldControlAggregate:
    ordered_iids: tuple[str, str]
    mean_losses: tuple[tuple[str, float], ...]
    aggregation: str = "sort_by_iid_then_arithmetic_mean_losses_before_ratios"

    @property
    def mean_loss_map(self) -> dict[str, float]:
        return dict(self.mean_losses)


def aggregate_held_control_losses(
    payloads: Sequence[DetachedSupportPayload],
) -> HeldControlAggregate:
    """Sort by IID, average losses, and leave nonlinear GO ratios for later."""

    first, second = canonical_two_support_payloads(payloads)
    by_iid = tuple(sorted((first, second), key=lambda item: item.iid))
    means: list[tuple[str, float]] = []
    for name in HELD_CONTROL_NAMES:
        values = [item.held_loss_map[name] for item in by_iid]
        # Match the serial runner's Python arithmetic: sum two scalar losses,
        # then divide once.  Do not average per-support improvement ratios.
        mean = sum(float(value) for value in values) / SUPPORT_COUNT
        if not math.isfinite(mean) or mean < 0.0:
            raise FewShotEpisodeParallelError("held control mean is invalid")
        means.append((name, mean))
    return HeldControlAggregate(
        ordered_iids=(by_iid[0].iid, by_iid[1].iid),
        mean_losses=tuple(means),
    )


__all__ = [
    "AGGREGATION_RULE",
    "CanonicalPrototype",
    "Collective",
    "DATA_PARALLEL_SIZE",
    "DEFAULT_TOPOLOGY",
    "DetachedSupportPayload",
    "EpisodeParallelTopology",
    "FewShotEpisodeParallelError",
    "GRADIENT_DIVISOR",
    "GroupSpec",
    "HELD_CONTROL_NAMES",
    "HeldControlAggregate",
    "ProbeEvidence",
    "REFERENCE_PROBE_FAMILIES",
    "REFERENCE_PROBE_SUPPORT_INDEX",
    "RankAssignment",
    "SUPPORT_COUNT",
    "SUPPORT_INDICES",
    "SupportReplicaCertificate",
    "ULYSSES_SIZE",
    "WORLD_SIZE",
    "aggregate_held_control_losses",
    "build_canonical_prototype",
    "canonical_exchange_support_payloads",
    "canonical_reference_probes",
    "canonical_two_support_payloads",
    "certify_support_replicas",
    "freeze_support_payload",
    "mean_support_gradient",
]
