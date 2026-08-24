#!/usr/bin/env python3
"""Production streaming checkpoint/resume for full-30 action learning.

This is schema v2 despite the stable import filename.  The implementation is
purposefully tied to ``Full30ActionFirstOptimizerV1`` and the exact full-30
FP32 capacity.  Tensor files are scanned and written in bounded chunks; no
rank constructs a whole tensor-container payload, and only rank zero performs
filesystem writes.  Every load rank validates every byte locally before one
fixed status-consensus exchange.  State itself is never broadcast.
"""

from __future__ import annotations

import base64
from collections import Counter, defaultdict
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence


SCHEMA_VERSION = "bernini-full30-action-checkpoint-v2"
TENSOR_SCHEMA_VERSION = "bernini-full30-action-streaming-fp32-v2"
SCHEDULE_SCHEMA_VERSION = "bernini-full30-action-schedule-v2"
HISTORY_SCHEMA_VERSION = "bernini-full30-action-history-row-v2"
RNG_SCHEMA_VERSION = "bernini-full30-action-rng-state-v2"
OPTIMIZER_SCHEMA_VERSION = "bernini-full30-action-first-optimizer-state-v1"
OPTIMIZER_RECEIPT_SCHEMA_VERSION = "bernini-full30-action-first-update-receipt-v1"
OPTIMIZER_CLASS = "Full30ActionFirstOptimizerV1"
STATE_KIND = "canonical-per-parameter-v_t-fp32"

WORLD_SIZE = 8
DP_SIZE = 2
SP_SIZE = 4
GLOBAL_BATCH = 8
MAX_UPDATES = 160
FLAT_SCHEDULE_ROWS = GLOBAL_BATCH * MAX_UPDATES
EPOCHS = 10
UPDATES_PER_EPOCH = MAX_UPDATES // EPOCHS
TRAIN_SOURCES = 64
SIGMA_INDICES = (4, 12, 20, 28, 35, 38)
EXACT_TRAINABLE_NUMEL = 188_946_432
EXACT_LORA_TENSORS = 480
EXACT_TYPED_TENSORS = 5
EXACT_TRAINABLE_TENSORS = EXACT_LORA_TENSORS + EXACT_TYPED_TENSORS
LORA_RANK = 256
HIDDEN_SIZE = 1536
TYPED_NUMEL = 202_752

STREAM_CHUNK_ELEMENTS = 16_777_216
STREAM_CHUNK_BYTES = STREAM_CHUNK_ELEMENTS * 4
DIRECTORY_MODE = 0o750
FILE_MODE = 0o600
ARTIFACT_NAMES = (
    "trainables.f32",
    "optimizer_v.f32",
    "schedule.json",
    "history.json",
    "rng.json",
)
ALL_FILE_NAMES = frozenset((*ARTIFACT_NAMES, "manifest.json"))

_MAGIC = b"F30STRM2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,511}$")
_SAFE_OUTPUT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_LORA = re.compile(
    r"(?:^|\.)blocks\.(?P<block>\d+)\.attn(?P<attention>[12])\."
    r"(?P<projection>to_q|to_k|to_v|to_out\.0)\."
    r"lora_(?P<factor>[AB])(?:\.default)?\.weight$"
)
_TYPED = re.compile(
    r"(?:^|\.)(?P<group>source_delta|target_delta)\."
    r"(?P<part>weight|bias)$"
)
_ROLE = re.compile(r"(?:^|\.)role_embedding$")
_BINDING_FIELDS = frozenset(
    {
        "arm",
        "release_sha256",
        "model_sha256",
        "data_sha256",
        "teacher_sha256",
        "nuisance_sha256",
        "noise_sha256",
        "runtime_sha256",
        "objective_sha256",
    }
)
_REFERENCE_FIELDS = frozenset(
    {
        "checkpoint_sequence",
        "completed_updates",
        "manifest_sha256",
        "manifest_digest",
        "history_sha256",
        "rng_sha256",
        "schedule_prefix_sha256",
        "trainable_state_sha256",
        "optimizer_v_state_sha256",
    }
)
_HISTORY_FIELDS = frozenset(
    {
        "schema_version",
        "update_count",
        "optimizer_class",
        "optimizer_receipt_schema_version",
        "optimizer_receipt_digest",
        "parameters_before_sha256",
        "parameters_after_sha256",
        "optimizer_v_before_sha256",
        "optimizer_v_after_sha256",
        "rng_before_sha256",
        "rng_after_sha256",
        "schedule_prefix_sha256",
    }
)


class Full30CheckpointError(RuntimeError):
    """Raised before accepting or mutating ambiguous checkpoint state."""


class Full30CheckpointTransactionError(Full30CheckpointError):
    """Rank-zero publication definitely did not commit."""


class Full30CheckpointCommitIndeterminate(Full30CheckpointError):
    """A complete target may exist; exact-existing recovery is required."""


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
        raise Full30CheckpointError(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise Full30CheckpointError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_json(raw: bytes, *, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise Full30CheckpointError(f"{label} contains {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Full30CheckpointError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("ascii"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Full30CheckpointError(f"cannot decode {label}: {error}") from error


def _json_value(value: Any, *, label: str) -> Any:
    raw = canonical_json_bytes(value)
    parsed = _strict_json(raw, label=label)
    if canonical_json_bytes(parsed) != raw:
        raise Full30CheckpointError(f"{label} canonical roundtrip differs")
    return parsed


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _product(shape: Sequence[int]) -> int:
    result = 1
    for item in shape:
        result *= int(item)
    return result


def _shape(value: Any, *, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        raise Full30CheckpointError(f"{label} must contain positive dimensions")
    result = tuple(int(item) for item in value)
    if _product(result) > (1 << 63) - 1:
        raise Full30CheckpointError(f"{label} overflows int64")
    return result


@dataclass(frozen=True)
class CheckpointBindings:
    arm: str
    release_sha256: str
    model_sha256: str
    data_sha256: str
    teacher_sha256: str
    nuisance_sha256: str
    noise_sha256: str
    runtime_sha256: str
    objective_sha256: str

    def __post_init__(self) -> None:
        if self.arm not in {"action+retain", "action-only"}:
            raise Full30CheckpointError(
                "checkpoint arm must be action+retain or action-only"
            )
        for name in sorted(_BINDING_FIELDS - {"arm"}):
            _sha256(getattr(self, name), label=name)

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in sorted(_BINDING_FIELDS)}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CheckpointBindings":
        if not isinstance(value, Mapping) or set(value) != _BINDING_FIELDS:
            raise Full30CheckpointError("checkpoint binding field set differs")
        return cls(**dict(value))


@dataclass(frozen=True)
class CheckpointReference:
    checkpoint_sequence: int
    completed_updates: int
    manifest_sha256: str
    manifest_digest: str
    history_sha256: str
    rng_sha256: str
    schedule_prefix_sha256: str
    trainable_state_sha256: str
    optimizer_v_state_sha256: str

    def __post_init__(self) -> None:
        if type(self.checkpoint_sequence) is not int or self.checkpoint_sequence < 0:
            raise Full30CheckpointError("checkpoint sequence differs")
        if (
            type(self.completed_updates) is not int
            or not 0 <= self.completed_updates <= MAX_UPDATES
        ):
            raise Full30CheckpointError("checkpoint completed update count differs")
        for name in sorted(_REFERENCE_FIELDS - {"checkpoint_sequence", "completed_updates"}):
            _sha256(getattr(self, name), label=f"checkpoint reference {name}")

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in sorted(_REFERENCE_FIELDS)}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CheckpointReference":
        if not isinstance(value, Mapping) or set(value) != _REFERENCE_FIELDS:
            raise Full30CheckpointError("checkpoint reference field set differs")
        return cls(**dict(value))


@dataclass(frozen=True)
class TensorFileIndex:
    path: Path
    kind: str
    payload_offset: int
    rows: tuple[Mapping[str, Any], ...]
    state_sha256: str
    file_sha256: str
    byte_size: int
    file_identity: tuple[int, ...]


@dataclass(frozen=True)
class LoadedCheckpoint:
    root: Path
    bindings: CheckpointBindings
    completed_updates: int
    next_cursor: Mapping[str, Any]
    schedule: tuple[Mapping[str, Any], ...]
    history: tuple[Mapping[str, Any], ...]
    rng_state: Mapping[str, Any]
    previous_checkpoint: Optional[CheckpointReference]
    reference: CheckpointReference
    manifest: Mapping[str, Any]
    manifest_bytes: bytes
    trainable_index: TensorFileIndex
    optimizer_v_index: TensorFileIndex
    local_validation_digest: str


@dataclass(frozen=True)
class StreamingAllocationPlan:
    tensor_count: int
    total_numel: int
    configured_chunk_elements: int
    maximum_chunk_elements: int
    maximum_chunk_bytes: int
    whole_state_payload_materialized: bool
    per_rank_tensor_file_bytes_retained: int


def streaming_allocation_plan_v2(
    named_numels: Mapping[str, int] | Iterable[tuple[str, int]],
    *,
    chunk_elements: int = STREAM_CHUNK_ELEMENTS,
) -> StreamingAllocationPlan:
    if type(chunk_elements) is not int or not 0 < chunk_elements <= STREAM_CHUNK_ELEMENTS:
        raise Full30CheckpointError("stream chunk element bound differs")
    rows = list(named_numels.items()) if isinstance(named_numels, Mapping) else list(named_numels)
    if not rows:
        raise Full30CheckpointError("streaming plan inventory is empty")
    total = 0
    seen: set[str] = set()
    largest = 0
    for name, count in rows:
        if type(name) is not str or _SAFE_NAME.fullmatch(name) is None or name in seen:
            raise Full30CheckpointError("streaming plan parameter name differs")
        if type(count) is not int or count <= 0:
            raise Full30CheckpointError("streaming plan element count differs")
        seen.add(name)
        total += count
        largest = max(largest, min(count, chunk_elements))
    return StreamingAllocationPlan(
        tensor_count=len(rows),
        total_numel=total,
        configured_chunk_elements=chunk_elements,
        maximum_chunk_elements=largest,
        maximum_chunk_bytes=largest * 4,
        whole_state_payload_materialized=False,
        per_rank_tensor_file_bytes_retained=0,
    )


def _normalise_bindings(value: CheckpointBindings | Mapping[str, Any]) -> CheckpointBindings:
    return value if isinstance(value, CheckpointBindings) else CheckpointBindings.from_mapping(value)


def _dtype_text(value: Any) -> str:
    return str(getattr(value, "dtype", ""))


def _is_contiguous(value: Any) -> bool:
    probe = getattr(value, "is_contiguous", None)
    return bool(probe() if callable(probe) else probe)


def _numel(value: Any) -> int:
    probe = getattr(value, "numel", None)
    if callable(probe):
        return int(probe())
    return _product(tuple(int(item) for item in getattr(value, "shape", ())))


def _named_optimizer_state(
    optimizer: Any, *, test_only_allow_small_capacity: bool
) -> tuple[tuple[tuple[str, Any], ...], tuple[tuple[str, Any], ...], int]:
    if not test_only_allow_small_capacity:
        module_name = optimizer.__class__.__module__
        try:
            module = importlib.import_module(module_name)
            registered_class = getattr(module, OPTIMIZER_CLASS)
            registered_schema = getattr(module, "STATE_SCHEMA_VERSION")
            module_path = Path(module.__file__).resolve(strict=True)
        except (AttributeError, ImportError, OSError, TypeError) as error:
            raise Full30CheckpointError(
                "cannot resolve actual Full30ActionFirstOptimizerV1 type"
            ) from error
        if module_path.suffix == ".pyc":
            try:
                module_path = Path(importlib.util.source_from_cache(str(module_path))).resolve(
                    strict=True
                )
            except (OSError, ValueError) as error:
                raise Full30CheckpointError(
                    "cannot resolve optimizer source identity"
                ) from error
        if (
            optimizer.__class__ is not registered_class
            or module_path != Path(__file__).with_name("full30_action_optimizer_v1.py").resolve(strict=True)
            or registered_schema != OPTIMIZER_SCHEMA_VERSION
        ):
            raise Full30CheckpointError(
                "checkpoint requires actual Full30ActionFirstOptimizerV1"
            )
    names = getattr(optimizer, "canonical_parameter_names", None)
    parameters = getattr(optimizer, "_parameters", None)
    moments = getattr(optimizer, "_second_moments", None)
    update_count = getattr(optimizer, "update_count", None)
    if (
        not isinstance(names, tuple)
        or not names
        or tuple(sorted(names, key=lambda item: item.encode("utf-8"))) != names
        or not isinstance(parameters, Mapping)
        or not isinstance(moments, Mapping)
        or set(parameters) != set(names)
        or set(moments) != set(names)
        or type(update_count) is not int
        or update_count < 0
    ):
        raise Full30CheckpointError("optimizer canonical state contract differs")
    return (
        tuple((name, parameters[name]) for name in names),
        tuple((name, moments[name]) for name in names),
        update_count,
    )


def _inventory_rows(named: Sequence[tuple[str, Any]]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, tensor in named:
        if type(name) is not str or _SAFE_NAME.fullmatch(name) is None or name in seen:
            raise Full30CheckpointError("trainable canonical name differs")
        seen.add(name)
        shape = _shape(tuple(int(item) for item in getattr(tensor, "shape", ())), label=f"{name} shape")
        if (
            _dtype_text(tensor) != "torch.float32"
            or not _is_contiguous(tensor)
            or getattr(tensor, "requires_grad", True) is False
            or _numel(tensor) != _product(shape)
        ):
            raise Full30CheckpointError(f"trainable FP32 tensor contract differs: {name}")
        rows.append(
            {
                "name": name,
                "shape": list(shape),
                "runtime_dtype": "torch.float32",
                "stored_dtype": "float32-le",
                "numel": _product(shape),
            }
        )
    if tuple(row["name"] for row in rows) != tuple(
        sorted((row["name"] for row in rows), key=lambda item: item.encode("utf-8"))
    ):
        raise Full30CheckpointError("trainable inventory order differs")
    return tuple(rows)


def _validate_production_capacity(rows: Sequence[Mapping[str, Any]]) -> None:
    if (
        len(rows) != EXACT_TRAINABLE_TENSORS
        or sum(int(row["numel"]) for row in rows) != EXACT_TRAINABLE_NUMEL
    ):
        raise Full30CheckpointError("full30 authoritative trainable capacity differs")
    lora: dict[tuple[int, int, str, str], Mapping[str, Any]] = {}
    typed: dict[tuple[str, str], Mapping[str, Any]] = {}
    role: list[Mapping[str, Any]] = []
    for row in rows:
        name = str(row["name"])
        match = _LORA.search(name)
        typed_match = _TYPED.search(name)
        if match is not None:
            key = (
                int(match.group("block")),
                int(match.group("attention")),
                match.group("projection"),
                match.group("factor"),
            )
            if key in lora:
                raise Full30CheckpointError("duplicate LoRA affine factor")
            expected = (LORA_RANK, HIDDEN_SIZE) if key[3] == "A" else (HIDDEN_SIZE, LORA_RANK)
            if tuple(row["shape"]) != expected:
                raise Full30CheckpointError("LoRA affine factor shape differs")
            lora[key] = row
        elif typed_match is not None:
            key = (typed_match.group("group"), typed_match.group("part"))
            if key in typed:
                raise Full30CheckpointError("duplicate typed patch tensor")
            expected = (HIDDEN_SIZE, 16, 1, 2, 2) if key[1] == "weight" else (HIDDEN_SIZE,)
            if tuple(row["shape"]) != expected:
                raise Full30CheckpointError("typed patch tensor shape differs")
            typed[key] = row
        elif _ROLE.search(name) is not None:
            if tuple(row["shape"]) != (2, HIDDEN_SIZE):
                raise Full30CheckpointError("typed role embedding shape differs")
            role.append(row)
        else:
            raise Full30CheckpointError("non-full30 trainable leaked into checkpoint")
    expected_lora = {
        (block, attention, projection, factor)
        for block in range(30)
        for attention in (1, 2)
        for projection in ("to_q", "to_k", "to_v", "to_out.0")
        for factor in ("A", "B")
    }
    if set(lora) != expected_lora or len(typed) != 4 or len(role) != 1:
        raise Full30CheckpointError("full30 affine/typed inventory closure differs")
    if sum(int(row["numel"]) for row in (*typed.values(), *role)) != TYPED_NUMEL:
        raise Full30CheckpointError("typed patch/role parameter count differs")


def inventory_identity_v2(
    optimizer: Any, *, test_only_allow_small_capacity: bool = False
) -> Mapping[str, Any]:
    parameters, _moments, _count = _named_optimizer_state(
        optimizer, test_only_allow_small_capacity=test_only_allow_small_capacity
    )
    rows = _inventory_rows(parameters)
    if not test_only_allow_small_capacity:
        _validate_production_capacity(rows)
    value = {
        "rows": list(rows),
        "tensor_count": len(rows),
        "total_numel": sum(row["numel"] for row in rows),
        "production_capacity_authorized": not test_only_allow_small_capacity,
    }
    return {**value, "inventory_sha256": object_sha256(value)}


def _schedule_row(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        row = dict(value)
        nested = row.get("row")
        if isinstance(nested, Mapping):
            row["row"] = dict(nested)
    else:
        action_row = getattr(value, "row", None)
        row = {
            "global_index": getattr(value, "global_index", None),
            "epoch": getattr(value, "epoch", None),
            "update": getattr(value, "update", None),
            "microbatch": getattr(value, "microbatch", None),
            "dp_rank": getattr(value, "dp_rank", None),
            "sigma_index": getattr(value, "sigma_index", None),
            "noise_seed": getattr(value, "noise_seed", None),
            "row": {
                "row_id": getattr(action_row, "row_id", None),
                "source_id": getattr(action_row, "source_id", None),
                "branch": getattr(action_row, "branch", None),
                "teacher_cell_id": getattr(action_row, "teacher_cell_id", None),
            },
        }
    fields = {
        "global_index",
        "epoch",
        "update",
        "microbatch",
        "dp_rank",
        "sigma_index",
        "noise_seed",
        "row",
    }
    row_fields = {"row_id", "source_id", "branch", "teacher_cell_id"}
    if set(row) != fields or not isinstance(row.get("row"), Mapping) or set(row["row"]) != row_fields:
        raise Full30CheckpointError("formal schedule row fields differ")
    for name in ("global_index", "epoch", "update", "microbatch", "dp_rank", "sigma_index", "noise_seed"):
        if type(row[name]) is not int or row[name] < 0:
            raise Full30CheckpointError(f"formal schedule {name} differs")
    for name in row_fields:
        item = row["row"][name]
        if type(item) is not str or not item or "\x00" in item:
            raise Full30CheckpointError(f"formal schedule {name} differs")
    if row["row"]["branch"] not in {"action", "incomplete"}:
        raise Full30CheckpointError("formal schedule branch differs")
    return _json_value(row, label="formal schedule row")


def canonical_schedule_v2(schedule: Sequence[Any]) -> tuple[Mapping[str, Any], ...]:
    rows = tuple(_schedule_row(item) for item in schedule)
    if len(rows) != FLAT_SCHEDULE_ROWS:
        raise Full30CheckpointError("formal schedule requires exactly 1280 flat rows")
    row_contract: dict[str, tuple[str, str, str]] = {}
    source_rows: dict[str, set[str]] = defaultdict(set)
    source_teachers: dict[str, set[str]] = defaultdict(set)
    teacher_sources: dict[str, set[str]] = defaultdict(set)
    source_epoch_counts: Counter[tuple[int, str]] = Counter()
    for update in range(MAX_UPDATES):
        group = rows[update * GLOBAL_BATCH : (update + 1) * GLOBAL_BATCH]
        expected_positions = [
            (microbatch, dp_rank)
            for microbatch in range(4)
            for dp_rank in range(2)
        ]
        if (
            [row["global_index"] for row in group]
            != list(range(update * GLOBAL_BATCH, (update + 1) * GLOBAL_BATCH))
            or any(row["update"] != update for row in group)
            or any(row["epoch"] != update // UPDATES_PER_EPOCH for row in group)
            or [(row["microbatch"], row["dp_rank"]) for row in group]
            != expected_positions
            or sum(row["row"]["branch"] == "action" for row in group) != 4
            or sum(row["row"]["branch"] == "incomplete" for row in group) != 4
        ):
            raise Full30CheckpointError("formal schedule DP2 update group differs")
        for microbatch in range(4):
            pair = group[microbatch * 2 : microbatch * 2 + 2]
            expected_branches = (
                ("action", "incomplete")
                if (pair[0]["epoch"] + microbatch) % 2 == 0
                else ("incomplete", "action")
            )
            if (
                pair[0]["row"]["source_id"] != pair[1]["row"]["source_id"]
                or pair[0]["row"]["teacher_cell_id"] != pair[1]["row"]["teacher_cell_id"]
                or tuple(item["row"]["branch"] for item in pair) != expected_branches
                or pair[0]["sigma_index"] != pair[1]["sigma_index"]
                or pair[0]["sigma_index"] not in SIGMA_INDICES
                or pair[0]["sigma_index"]
                != SIGMA_INDICES[
                    (pair[0]["epoch"] * TRAIN_SOURCES + (update % UPDATES_PER_EPOCH) * 4 + microbatch)
                    % len(SIGMA_INDICES)
                ]
                or pair[0]["noise_seed"] != pair[1]["noise_seed"]
                or pair[0]["noise_seed"] >= 2**64
            ):
                raise Full30CheckpointError("formal paired source/noise row differs")
            epoch = pair[0]["epoch"]
            source_id = pair[0]["row"]["source_id"]
            source_epoch_counts[(epoch, source_id)] += 1
            for item in pair:
                nested = item["row"]
                contract = (
                    nested["source_id"],
                    nested["branch"],
                    nested["teacher_cell_id"],
                )
                if nested["row_id"] in row_contract and row_contract[nested["row_id"]] != contract:
                    raise Full30CheckpointError("formal row identity changes across epochs")
                row_contract[nested["row_id"]] = contract
                source_rows[source_id].add(nested["branch"])
                source_teachers[source_id].add(nested["teacher_cell_id"])
                teacher_sources[nested["teacher_cell_id"]].add(source_id)
    if (
        len(source_rows) != TRAIN_SOURCES
        or len(row_contract) != TRAIN_SOURCES * 2
        or any(branches != {"action", "incomplete"} for branches in source_rows.values())
        or any(len(teachers) != 1 for teachers in source_teachers.values())
        or len(teacher_sources) != 8
        or any(len(sources) != 8 for sources in teacher_sources.values())
        or any(
            source_epoch_counts[(epoch, source_id)] != 1
            for epoch in range(EPOCHS)
            for source_id in source_rows
        )
        or Counter(row["sigma_index"] for row in rows)
        != Counter({4: 214, 12: 214, 20: 214, 28: 214, 35: 212, 38: 212})
        or len({row["noise_seed"] for row in rows}) != EPOCHS * TRAIN_SOURCES
    ):
        raise Full30CheckpointError("formal 10-epoch source/sigma/noise closure differs")
    return tuple(MappingProxyType(dict(row)) for row in rows)


def schedule_digests_v2(
    schedule: Sequence[Any], completed_updates: int
) -> tuple[str, str]:
    rows = canonical_schedule_v2(schedule)
    if type(completed_updates) is not int or not 0 <= completed_updates <= MAX_UPDATES:
        raise Full30CheckpointError("completed updates differ")
    serial = [dict(row) for row in rows]
    full = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "flat_rows": serial,
    }
    prefix = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "completed_updates": completed_updates,
        "flat_row_count": GLOBAL_BATCH * completed_updates,
        "flat_rows": serial[: GLOBAL_BATCH * completed_updates],
    }
    return object_sha256(full), object_sha256(prefix)


def next_cursor_v2(schedule: Sequence[Any], completed_updates: int) -> Mapping[str, Any]:
    rows = canonical_schedule_v2(schedule)
    _full, prefix = schedule_digests_v2(rows, completed_updates)
    next_index = GLOBAL_BATCH * completed_updates
    value = {
        "schema_version": "bernini-full30-action-data-cursor-v2",
        "completed_updates": completed_updates,
        "completed_flat_rows": next_index,
        "next_update": completed_updates,
        "next_global_index": next_index,
        "next_epoch": completed_updates // UPDATES_PER_EPOCH,
        "next_microbatch": 0,
        "next_dp_rank": 0,
        "terminal": completed_updates == MAX_UPDATES,
        "schedule_prefix_sha256": prefix,
    }
    return MappingProxyType(value)


def _validate_rng_state(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "python_rank_state_b64",
        "torch_cpu_rank_state_b64",
        "torch_cuda_rank_state_b64",
    }
    row = _json_value(dict(value), label="RNG state")
    if not isinstance(row, dict) or set(row) != expected or row["schema_version"] != RNG_SCHEMA_VERSION:
        raise Full30CheckpointError("RNG state schema differs")
    rank_fields = (
        row["python_rank_state_b64"],
        row["torch_cpu_rank_state_b64"],
        row["torch_cuda_rank_state_b64"],
    )
    if any(not isinstance(ranks, list) or len(ranks) != WORLD_SIZE for ranks in rank_fields):
        raise Full30CheckpointError("RNG state must bind Python/CPU/CUDA state for every WORLD8 rank")
    encoded = [item for ranks in rank_fields for item in ranks]
    decoded_total = 0
    for item in encoded:
        if type(item) is not str or not item:
            raise Full30CheckpointError("RNG state base64 field differs")
        try:
            decoded = base64.b64decode(item.encode("ascii"), validate=True)
        except (ValueError, UnicodeError) as error:
            raise Full30CheckpointError("RNG state base64 is invalid") from error
        if not decoded or base64.b64encode(decoded).decode("ascii") != item:
            raise Full30CheckpointError("RNG state base64 is not canonical")
        decoded_total += len(decoded)
    if decoded_total > 16 * 1024 * 1024:
        raise Full30CheckpointError("RNG state payload is unexpectedly large")
    return row


def _history_prefix_sha(history: Sequence[Mapping[str, Any]], count: int) -> str:
    return hashlib.sha256(
        canonical_json_bytes([dict(row) for row in history[:count]]) + b"\n"
    ).hexdigest()


def _history_rows(
    history: Sequence[Mapping[str, Any]],
    *,
    completed_updates: int,
    schedule: Sequence[Any],
    current_parameter_sha256: str,
    current_optimizer_v_sha256: str,
    current_rng_sha256: str,
    previous: Optional[CheckpointReference],
    previous_history: Optional[Sequence[Mapping[str, Any]]],
) -> tuple[Mapping[str, Any], ...]:
    rows = _json_value(
        [dict(row) if isinstance(row, Mapping) else row for row in history],
        label="optimizer history",
    )
    if not isinstance(rows, list) or len(rows) != completed_updates:
        raise Full30CheckpointError("history must contain exactly one row per update")
    checked: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != _HISTORY_FIELDS:
            raise Full30CheckpointError("history row field set differs")
        if row["schema_version"] != HISTORY_SCHEMA_VERSION or row["update_count"] != index + 1:
            raise Full30CheckpointError("history optimizer update count differs")
        if (
            row["optimizer_class"] != OPTIMIZER_CLASS
            or row["optimizer_receipt_schema_version"]
            != OPTIMIZER_RECEIPT_SCHEMA_VERSION
        ):
            raise Full30CheckpointError("history typed optimizer receipt differs")
        for name in _HISTORY_FIELDS - {
            "schema_version",
            "update_count",
            "optimizer_class",
            "optimizer_receipt_schema_version",
        }:
            _sha256(row[name], label=f"history {name}")
        _full, expected_prefix = schedule_digests_v2(schedule, index + 1)
        if row["schedule_prefix_sha256"] != expected_prefix:
            raise Full30CheckpointError("history schedule prefix differs")
        if index:
            before_after = (
                ("parameters_before_sha256", "parameters_after_sha256"),
                ("optimizer_v_before_sha256", "optimizer_v_after_sha256"),
                ("rng_before_sha256", "rng_after_sha256"),
            )
            for before, after in before_after:
                if row[before] != rows[index - 1][after]:
                    raise Full30CheckpointError("history state chain is not contiguous")
        checked.append(MappingProxyType(dict(row)))
    if len({row["optimizer_receipt_digest"] for row in rows}) != len(rows):
        raise Full30CheckpointError("history reuses an optimizer update receipt")
    if completed_updates:
        final = rows[-1]
        if (
            final["parameters_after_sha256"] != current_parameter_sha256
            or final["optimizer_v_after_sha256"] != current_optimizer_v_sha256
            or final["rng_after_sha256"] != current_rng_sha256
        ):
            raise Full30CheckpointError("history final state does not bind checkpoint")
    if completed_updates == 0:
        if previous is not None:
            raise Full30CheckpointError("zero-update checkpoint cannot have predecessor")
    else:
        if previous is None or previous.completed_updates >= completed_updates:
            raise Full30CheckpointError("nonzero checkpoint requires earlier predecessor")
        if _history_prefix_sha(checked, previous.completed_updates) != previous.history_sha256:
            raise Full30CheckpointError("predecessor history prefix SHA differs")
        if previous_history is not None and list(checked[: previous.completed_updates]) != [
            dict(item) for item in previous_history
        ]:
            raise Full30CheckpointError("predecessor history object prefix differs")
        predecessor_state = {
            "parameters": previous.trainable_state_sha256,
            "optimizer_v": previous.optimizer_v_state_sha256,
            "rng": previous.rng_sha256,
        }
        if previous.completed_updates:
            predecessor_final = checked[previous.completed_updates - 1]
            if (
                predecessor_final["parameters_after_sha256"] != predecessor_state["parameters"]
                or predecessor_final["optimizer_v_after_sha256"] != predecessor_state["optimizer_v"]
                or predecessor_final["rng_after_sha256"] != predecessor_state["rng"]
            ):
                raise Full30CheckpointError("predecessor reference does not bind its history state")
        first_new = checked[previous.completed_updates]
        if (
            first_new["parameters_before_sha256"] != predecessor_state["parameters"]
            or first_new["optimizer_v_before_sha256"] != predecessor_state["optimizer_v"]
            or first_new["rng_before_sha256"] != predecessor_state["rng"]
        ):
            raise Full30CheckpointError("new history segment does not resume predecessor exact state")
    return tuple(checked)


def _previous_parts(
    previous: Optional[LoadedCheckpoint | CheckpointReference],
) -> tuple[Optional[CheckpointReference], Optional[Sequence[Mapping[str, Any]]]]:
    if previous is None:
        return None, None
    if isinstance(previous, LoadedCheckpoint):
        return previous.reference, previous.history
    if isinstance(previous, CheckpointReference):
        return previous, None
    raise Full30CheckpointError("previous checkpoint object differs")


StreamObserver = Callable[[str, int], None]


@dataclass(frozen=True)
class _ScannedState:
    kind: str
    rows: tuple[Mapping[str, Any], ...]
    state_sha256: str
    all_zero: bool


def _observe(observer: Optional[StreamObserver], event: str, size: int) -> None:
    if size < 0 or size > STREAM_CHUNK_BYTES:
        raise Full30CheckpointError("stream operation exceeded registered chunk bound")
    if observer is not None:
        observer(event, size)


def _raw_values_valid(raw: bytes, *, nonnegative: bool) -> tuple[bool, bool]:
    if not raw or len(raw) % 4:
        raise Full30CheckpointError("FP32 stream chunk byte count differs")
    try:
        import torch
    except ModuleNotFoundError:
        finite = True
        all_zero = True
        for (number,) in struct.iter_unpack("<f", raw):
            finite = finite and math.isfinite(number)
            all_zero = all_zero and number == 0.0
            if not finite or (nonnegative and number < 0.0):
                return False, all_zero
        return True, all_zero
    values = torch.frombuffer(bytearray(raw), dtype=torch.float32)
    valid = bool(torch.isfinite(values).all().item())
    if nonnegative:
        valid = valid and not bool((values < 0).any().item())
    return valid, bool((values == 0).all().item())


def _iter_tensor_chunks(
    tensor: Any,
    *,
    observer: Optional[StreamObserver],
    event: str,
) -> Iterator[bytes]:
    custom = getattr(tensor, "iter_checkpoint_fp32_chunks", None)
    if callable(custom):
        emitted = 0
        for raw in custom(STREAM_CHUNK_ELEMENTS):
            if type(raw) is not bytes or not raw or len(raw) > STREAM_CHUNK_BYTES or len(raw) % 4:
                raise Full30CheckpointError("custom tensor stream chunk differs")
            emitted += len(raw) // 4
            _observe(observer, event, len(raw))
            yield raw
        if emitted != _numel(tensor):
            raise Full30CheckpointError("custom tensor stream element count differs")
        return
    try:
        import torch
    except ModuleNotFoundError as error:
        raise Full30CheckpointError(
            "tensor streaming requires torch or iter_checkpoint_fp32_chunks"
        ) from error
    if not isinstance(tensor, torch.Tensor) or tensor.device.type == "meta":
        raise Full30CheckpointError("stream source is not one materialized tensor")
    flat = tensor.detach().reshape(-1)
    for offset in range(0, int(flat.numel()), STREAM_CHUNK_ELEMENTS):
        count = min(STREAM_CHUNK_ELEMENTS, int(flat.numel()) - offset)
        chunk = flat.narrow(0, offset, count).to(device="cpu").contiguous()
        storage = chunk.untyped_storage()
        if chunk.storage_offset() != 0 or int(storage.nbytes()) != count * 4:
            # A contiguous CPU narrow may still retain its base tensor's full
            # storage.  Clone only that case so bytes(storage) can never leak
            # adjacent values or defeat the chunk bound.  This path does not
            # depend on NumPy being installed or ABI-compatible.
            chunk = chunk.clone(memory_format=torch.contiguous_format)
            storage = chunk.untyped_storage()
        raw = bytes(storage)
        if len(raw) != count * 4:
            raise Full30CheckpointError("torch tensor stream byte count differs")
        _observe(observer, event, len(raw))
        yield raw


def _scan_named_state(
    named: Sequence[tuple[str, Any]],
    *,
    kind: str,
    inventory: Sequence[Mapping[str, Any]],
    nonnegative: bool,
    require_all_zero: bool,
    observer: Optional[StreamObserver],
) -> _ScannedState:
    if kind not in {"trainables", "optimizer_v"}:
        raise Full30CheckpointError("stream state kind differs")
    if tuple(name for name, _ in named) != tuple(row["name"] for row in inventory):
        raise Full30CheckpointError(f"{kind} name order differs from inventory")
    rows: list[Mapping[str, Any]] = []
    payload_offset = 0
    state_all_zero = True
    for (name, tensor), inventory_row in zip(named, inventory):
        shape = tuple(int(item) for item in getattr(tensor, "shape", ()))
        if (
            shape != tuple(inventory_row["shape"])
            or _dtype_text(tensor) != "torch.float32"
            or not _is_contiguous(tensor)
            or _numel(tensor) != inventory_row["numel"]
        ):
            raise Full30CheckpointError(f"{kind} tensor contract differs: {name}")
        digest = hashlib.sha256()
        byte_count = 0
        tensor_all_zero = True
        for raw in _iter_tensor_chunks(
            tensor, observer=observer, event=f"scan:{kind}"
        ):
            valid, chunk_zero = _raw_values_valid(raw, nonnegative=nonnegative)
            if not valid:
                raise Full30CheckpointError(f"{kind} contains invalid FP32: {name}")
            tensor_all_zero = tensor_all_zero and chunk_zero
            digest.update(raw)
            byte_count += len(raw)
        expected_bytes = int(inventory_row["numel"]) * 4
        if byte_count != expected_bytes:
            raise Full30CheckpointError(f"{kind} tensor byte count differs: {name}")
        rows.append(
            {
                **dict(inventory_row),
                "payload_offset": payload_offset,
                "nbytes": byte_count,
                "tensor_sha256": digest.hexdigest(),
            }
        )
        payload_offset += byte_count
        state_all_zero = state_all_zero and tensor_all_zero
    if require_all_zero and not state_all_zero:
        raise Full30CheckpointError("u0 optimizer v_t must be exactly all-zero FP32")
    state_payload = {
        "schema_version": TENSOR_SCHEMA_VERSION,
        "kind": kind,
        "rows": rows,
    }
    return _ScannedState(
        kind=kind,
        rows=tuple(MappingProxyType(dict(row)) for row in rows),
        state_sha256=object_sha256(state_payload),
        all_zero=state_all_zero,
    )


def _tensor_header(scan: _ScannedState) -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": TENSOR_SCHEMA_VERSION,
            "kind": scan.kind,
            "rows": [dict(row) for row in scan.rows],
        }
    )


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            raise OSError(errno.EIO, "short checkpoint write")
        offset += written


def _write_tensor_file(
    path: Path,
    *,
    scan: _ScannedState,
    named: Sequence[tuple[str, Any]],
    observer: Optional[StreamObserver],
) -> tuple[str, int]:
    header = _tensor_header(scan)
    prefix = _MAGIC + len(header).to_bytes(8, "big") + header
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, FILE_MODE)
    file_digest = hashlib.sha256()
    total = 0
    try:
        _write_all(descriptor, prefix)
        file_digest.update(prefix)
        total += len(prefix)
        for (name, tensor), row in zip(named, scan.rows):
            digest = hashlib.sha256()
            emitted = 0
            for raw in _iter_tensor_chunks(
                tensor, observer=observer, event=f"write:{scan.kind}"
            ):
                _write_all(descriptor, raw)
                file_digest.update(raw)
                digest.update(raw)
                emitted += len(raw)
                total += len(raw)
            if emitted != row["nbytes"] or digest.hexdigest() != row["tensor_sha256"]:
                raise Full30CheckpointError(
                    f"{scan.kind} changed between preflight and rank-zero write: {name}"
                )
        os.fchmod(descriptor, FILE_MODE)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return file_digest.hexdigest(), total


def _write_plain_file(path: Path, raw: bytes) -> tuple[str, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, FILE_MODE)
    try:
        _write_all(descriptor, raw)
        os.fchmod(descriptor, FILE_MODE)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _read_exact(descriptor: int, count: int, *, observer: Optional[StreamObserver], event: str) -> bytes:
    if count < 0 or count > STREAM_CHUNK_BYTES:
        raise Full30CheckpointError("bounded checkpoint read size differs")
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        block = os.read(descriptor, remaining)
        if not block:
            raise Full30CheckpointError("checkpoint tensor file is truncated")
        chunks.append(block)
        remaining -= len(block)
    raw = b"".join(chunks)
    _observe(observer, event, len(raw))
    return raw


def _regular_file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _plain_file_bytes(path: Path, *, label: str, maximum: int = 32 * 1024 * 1024) -> bytes:
    before = os.lstat(path)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != FILE_MODE
        or before.st_nlink != 1
        or before.st_size < 0
        or before.st_size > maximum
    ):
        raise Full30CheckpointError(f"{label} type/mode/size differs")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        raw = b""
        while len(raw) < before.st_size:
            block = os.read(descriptor, min(1024 * 1024, before.st_size - len(raw)))
            if not block:
                break
            raw += block
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    identity = _regular_file_identity(before)
    if (
        identity != _regular_file_identity(opened)
        or identity != _regular_file_identity(after)
        or len(raw) != before.st_size
    ):
        raise Full30CheckpointError(f"{label} changed while reading")
    return raw


def _read_tensor_index(
    path: Path,
    *,
    expected_kind: str,
    inventory: Sequence[Mapping[str, Any]],
    expected_file_sha256: str,
    expected_file_bytes: int,
    nonnegative: bool,
    require_all_zero: bool,
    observer: Optional[StreamObserver],
) -> TensorFileIndex:
    before = os.lstat(path)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != FILE_MODE
        or before.st_nlink != 1
    ):
        raise Full30CheckpointError(f"{expected_kind} tensor file type/mode differs")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    file_digest = hashlib.sha256()
    try:
        prefix = _read_exact(
            descriptor, 16, observer=observer, event=f"read:{expected_kind}:prefix"
        )
        if prefix[:8] != _MAGIC:
            raise Full30CheckpointError(f"{expected_kind} tensor magic differs")
        header_size = int.from_bytes(prefix[8:], "big")
        if header_size <= 0 or header_size > 16 * 1024 * 1024:
            raise Full30CheckpointError(f"{expected_kind} tensor header size differs")
        header_raw = _read_exact(
            descriptor,
            header_size,
            observer=observer,
            event=f"read:{expected_kind}:header",
        )
        file_digest.update(prefix)
        file_digest.update(header_raw)
        header = _strict_json(header_raw, label=f"{expected_kind} tensor header")
        if canonical_json_bytes(header) != header_raw:
            raise Full30CheckpointError(f"{expected_kind} tensor header is not canonical")
        if (
            not isinstance(header, dict)
            or set(header) != {"schema_version", "kind", "rows"}
            or header["schema_version"] != TENSOR_SCHEMA_VERSION
            or header["kind"] != expected_kind
            or not isinstance(header["rows"], list)
            or len(header["rows"]) != len(inventory)
        ):
            raise Full30CheckpointError(f"{expected_kind} tensor header fields differ")
        expected_row_fields = {
            "name",
            "shape",
            "runtime_dtype",
            "stored_dtype",
            "numel",
            "payload_offset",
            "nbytes",
            "tensor_sha256",
        }
        checked_rows: list[Mapping[str, Any]] = []
        payload_position = 0
        state_all_zero = True
        for source_row, expected_inventory in zip(header["rows"], inventory):
            if not isinstance(source_row, dict) or set(source_row) != expected_row_fields:
                raise Full30CheckpointError(f"{expected_kind} tensor row fields differ")
            inventory_part = {
                key: source_row[key]
                for key in ("name", "shape", "runtime_dtype", "stored_dtype", "numel")
            }
            if inventory_part != dict(expected_inventory):
                raise Full30CheckpointError(f"{expected_kind} inventory differs")
            expected_nbytes = int(expected_inventory["numel"]) * 4
            if (
                source_row["payload_offset"] != payload_position
                or source_row["nbytes"] != expected_nbytes
            ):
                raise Full30CheckpointError(f"{expected_kind} payload layout differs")
            tensor_expected_sha = _sha256(
                source_row["tensor_sha256"], label=f"{expected_kind} tensor SHA"
            )
            tensor_digest = hashlib.sha256()
            remaining = expected_nbytes
            tensor_all_zero = True
            while remaining:
                count = min(remaining, STREAM_CHUNK_BYTES)
                raw = _read_exact(
                    descriptor,
                    count,
                    observer=observer,
                    event=f"read:{expected_kind}:payload",
                )
                valid, chunk_zero = _raw_values_valid(raw, nonnegative=nonnegative)
                if not valid:
                    raise Full30CheckpointError(
                        f"{expected_kind} tensor contains invalid FP32"
                    )
                tensor_all_zero = tensor_all_zero and chunk_zero
                tensor_digest.update(raw)
                file_digest.update(raw)
                remaining -= count
            if tensor_digest.hexdigest() != tensor_expected_sha:
                raise Full30CheckpointError(f"{expected_kind} tensor digest differs")
            checked_rows.append(MappingProxyType(dict(source_row)))
            payload_position += expected_nbytes
            state_all_zero = state_all_zero and tensor_all_zero
        if os.read(descriptor, 1):
            raise Full30CheckpointError(f"{expected_kind} tensor file has trailing bytes")
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    identity = _regular_file_identity(before)
    if (
        identity != _regular_file_identity(opened)
        or identity != _regular_file_identity(after)
    ):
        raise Full30CheckpointError(f"{expected_kind} tensor file changed while reading")
    actual_file_sha = file_digest.hexdigest()
    if before.st_size != expected_file_bytes or actual_file_sha != expected_file_sha256:
        raise Full30CheckpointError(f"{expected_kind} tensor file artifact digest differs")
    if require_all_zero and not state_all_zero:
        raise Full30CheckpointError("u0 optimizer v_t file is not all-zero")
    state_payload = {
        "schema_version": TENSOR_SCHEMA_VERSION,
        "kind": expected_kind,
        "rows": [dict(row) for row in checked_rows],
    }
    return TensorFileIndex(
        path=path,
        kind=expected_kind,
        payload_offset=16 + header_size,
        rows=tuple(checked_rows),
        state_sha256=object_sha256(state_payload),
        file_sha256=actual_file_sha,
        byte_size=int(before.st_size),
        file_identity=identity,
    )


def _canonical_path(path: str | Path, *, must_exist: bool) -> Path:
    requested = Path(path).expanduser()
    if not requested.is_absolute() or requested == Path("/") or _SAFE_OUTPUT.fullmatch(requested.name) is None:
        raise Full30CheckpointError("checkpoint path must be an absolute safe directory")
    try:
        parent = requested.parent.resolve(strict=True)
    except OSError as error:
        raise Full30CheckpointError(f"checkpoint parent differs: {error}") from error
    if not parent.is_dir():
        raise Full30CheckpointError("checkpoint parent is not a directory")
    result = parent / requested.name
    if must_exist:
        root_stat = os.lstat(result)
        if not stat.S_ISDIR(root_stat.st_mode) or stat.S_IMODE(root_stat.st_mode) != DIRECTORY_MODE:
            raise Full30CheckpointError("checkpoint directory type/mode differs")
    return result


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux") and getattr(libc, "renameat2", None) is not None:
        function = libc.renameat2
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    elif getattr(libc, "renamex_np", None) is not None:
        function = libc.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(os.fsencode(source), os.fsencode(destination), 0x00000004)
    else:
        raise Full30CheckpointError("atomic create-only rename is unavailable")
    if result:
        number = ctypes.get_errno()
        if number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise Full30CheckpointError("checkpoint already exists")
        raise OSError(number, os.strerror(number), str(destination))


def _cleanup_stage(stage: Path, identity: tuple[int, int]) -> None:
    try:
        current = os.lstat(stage)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != identity:
        return
    entries = list(os.scandir(stage))
    if any(not item.is_file(follow_symlinks=False) for item in entries):
        return
    for item in entries:
        os.unlink(item.path)
    os.rmdir(stage)


@dataclass(frozen=True)
class _Preflight:
    parameters: tuple[tuple[str, Any], ...]
    moments: tuple[tuple[str, Any], ...]
    inventory: tuple[Mapping[str, Any], ...]
    inventory_sha256: str
    production_capacity_authorized: bool
    parameter_scan: _ScannedState
    optimizer_v_scan: _ScannedState
    completed_updates: int
    schedule: tuple[Mapping[str, Any], ...]
    schedule_full_sha256: str
    schedule_prefix_sha256: str
    cursor: Mapping[str, Any]
    history: tuple[Mapping[str, Any], ...]
    history_bytes: bytes
    rng_state: Mapping[str, Any]
    rng_bytes: bytes
    bindings: CheckpointBindings
    previous: Optional[CheckpointReference]
    checkpoint_sequence: int
    local_state_digest: str


def _build_preflight(
    *,
    optimizer: Any,
    completed_updates: int,
    full_schedule: Sequence[Any],
    history: Sequence[Mapping[str, Any]],
    rng_state: Mapping[str, Any],
    bindings: CheckpointBindings | Mapping[str, Any],
    previous_checkpoint: Optional[LoadedCheckpoint | CheckpointReference],
    authoritative_inventory_sha256: str,
    test_only_allow_small_capacity: bool,
    observer: Optional[StreamObserver],
) -> _Preflight:
    expected_inventory_sha = _sha256(
        authoritative_inventory_sha256, label="authoritative inventory SHA"
    )
    parameters, moments, optimizer_update_count = _named_optimizer_state(
        optimizer, test_only_allow_small_capacity=test_only_allow_small_capacity
    )
    if any(getattr(parameter, "grad", None) is not None for _, parameter in parameters):
        raise Full30CheckpointError(
            "checkpoint segment boundary requires every live parameter gradient absent"
        )
    if optimizer_update_count != completed_updates:
        raise Full30CheckpointError(
            "optimizer update_count must equal completed_updates"
        )
    inventory_receipt = inventory_identity_v2(
        optimizer, test_only_allow_small_capacity=test_only_allow_small_capacity
    )
    if inventory_receipt["inventory_sha256"] != expected_inventory_sha:
        raise Full30CheckpointError("authoritative trainable inventory SHA differs")
    inventory = tuple(MappingProxyType(dict(row)) for row in inventory_receipt["rows"])
    for (name, moment), row in zip(moments, inventory):
        if (
            name != row["name"]
            or tuple(int(item) for item in getattr(moment, "shape", ())) != tuple(row["shape"])
            or _dtype_text(moment) != "torch.float32"
            or not _is_contiguous(moment)
            or getattr(moment, "requires_grad", False) is not False
        ):
            raise Full30CheckpointError(f"optimizer v_t tensor contract differs: {name}")
    parameter_scan = _scan_named_state(
        parameters,
        kind="trainables",
        inventory=inventory,
        nonnegative=False,
        require_all_zero=False,
        observer=observer,
    )
    optimizer_v_scan = _scan_named_state(
        moments,
        kind="optimizer_v",
        inventory=inventory,
        nonnegative=True,
        require_all_zero=completed_updates == 0,
        observer=observer,
    )
    schedule = canonical_schedule_v2(full_schedule)
    schedule_full, schedule_prefix = schedule_digests_v2(schedule, completed_updates)
    cursor = next_cursor_v2(schedule, completed_updates)
    rng = _validate_rng_state(rng_state)
    rng_bytes = canonical_json_bytes(rng) + b"\n"
    rng_sha = hashlib.sha256(rng_bytes).hexdigest()
    previous, previous_history = _previous_parts(previous_checkpoint)
    checked_history = _history_rows(
        history,
        completed_updates=completed_updates,
        schedule=schedule,
        current_parameter_sha256=parameter_scan.state_sha256,
        current_optimizer_v_sha256=optimizer_v_scan.state_sha256,
        current_rng_sha256=rng_sha,
        previous=previous,
        previous_history=previous_history,
    )
    if previous is not None and previous.schedule_prefix_sha256 != schedule_digests_v2(
        schedule, previous.completed_updates
    )[1]:
        raise Full30CheckpointError("predecessor schedule prefix differs")
    sequence = 0 if previous is None else previous.checkpoint_sequence + 1
    history_bytes = canonical_json_bytes([dict(row) for row in checked_history]) + b"\n"
    canonical_bindings = _normalise_bindings(bindings)
    local_payload = {
        "schema_version": SCHEMA_VERSION,
        "bindings": canonical_bindings.as_dict(),
        "completed_updates": completed_updates,
        "checkpoint_sequence": sequence,
        "cursor": dict(cursor),
        "schedule_full_sha256": schedule_full,
        "schedule_prefix_sha256": schedule_prefix,
        "history_sha256": hashlib.sha256(history_bytes).hexdigest(),
        "rng_sha256": rng_sha,
        "inventory_sha256": expected_inventory_sha,
        "trainable_state_sha256": parameter_scan.state_sha256,
        "optimizer_v_state_sha256": optimizer_v_scan.state_sha256,
        "previous_checkpoint": None if previous is None else previous.as_dict(),
        "production_capacity_authorized": not test_only_allow_small_capacity,
    }
    return _Preflight(
        parameters=parameters,
        moments=moments,
        inventory=inventory,
        inventory_sha256=expected_inventory_sha,
        production_capacity_authorized=not test_only_allow_small_capacity,
        parameter_scan=parameter_scan,
        optimizer_v_scan=optimizer_v_scan,
        completed_updates=completed_updates,
        schedule=schedule,
        schedule_full_sha256=schedule_full,
        schedule_prefix_sha256=schedule_prefix,
        cursor=cursor,
        history=checked_history,
        history_bytes=history_bytes,
        rng_state=MappingProxyType(rng),
        rng_bytes=rng_bytes,
        bindings=canonical_bindings,
        previous=previous,
        checkpoint_sequence=sequence,
        local_state_digest=object_sha256(local_payload),
    )


StatusConsensus = Callable[[Mapping[str, Any]], Mapping[str, Any]]
ResultBroadcast = Callable[[Optional[Mapping[str, Any]]], Mapping[str, Any]]


def _exchange_local_status(
    local: Mapping[str, Any],
    *,
    world_size: int,
    consensus_callback: Optional[StatusConsensus],
    label: str,
) -> Mapping[str, Any]:
    if world_size == 1:
        result = dict(local)
        result["participant_count"] = 1
    else:
        if consensus_callback is None:
            raise Full30CheckpointError(f"distributed {label} requires status consensus")
        try:
            result = consensus_callback(dict(local))
        except Exception as error:
            raise Full30CheckpointError(f"{label} status consensus failed: {error}") from error
    if not isinstance(result, Mapping) or result.get("participant_count") != world_size or type(result.get("ok")) is not bool:
        raise Full30CheckpointError(f"{label} consensus result is malformed")
    if result["ok"]:
        if set(result) != {"ok", "digest", "participant_count"}:
            raise Full30CheckpointError(f"{label} success consensus fields differ")
        _sha256(result.get("digest"), label=f"{label} consensus digest")
    else:
        if type(result.get("error")) is not str:
            raise Full30CheckpointError(f"{label} failure consensus fields differ")
    return result


def _artifact_row(sha: str, size: int) -> dict[str, Any]:
    return {"sha256": sha, "bytes": size, "mode": "0600"}


def _build_manifest(
    preflight: _Preflight, artifacts: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "directory_mode": "0750",
        "file_mode": "0600",
        "bindings": preflight.bindings.as_dict(),
        "capacity": {
            "authoritative_inventory_sha256": preflight.inventory_sha256,
            "tensor_count": len(preflight.inventory),
            "total_numel": sum(int(row["numel"]) for row in preflight.inventory),
            "production_capacity_authorized": preflight.production_capacity_authorized,
        },
        "trainables": {
            "artifact": "trainables.f32",
            "inventory": [dict(row) for row in preflight.inventory],
            "state_sha256": preflight.parameter_scan.state_sha256,
            "stored_dtype": "float32-le",
        },
        "optimizer": {
            "artifact": "optimizer_v.f32",
            "class": OPTIMIZER_CLASS,
            "schema_version": OPTIMIZER_SCHEMA_VERSION,
            "state_kind": STATE_KIND,
            "update_count": preflight.completed_updates,
            "state_sha256": preflight.optimizer_v_scan.state_sha256,
            "nonnegative": True,
            "u0_all_zero": preflight.completed_updates == 0,
        },
        "progress": {
            "completed_updates": preflight.completed_updates,
            "next_cursor": dict(preflight.cursor),
        },
        "schedule": {
            "schema_version": SCHEDULE_SCHEMA_VERSION,
            "artifact": "schedule.json",
            "flat_row_count": FLAT_SCHEDULE_ROWS,
            "completed_prefix_flat_rows": GLOBAL_BATCH * preflight.completed_updates,
            "full_sha256": preflight.schedule_full_sha256,
            "prefix_sha256": preflight.schedule_prefix_sha256,
        },
        "history": {
            "schema_version": HISTORY_SCHEMA_VERSION,
            "artifact": "history.json",
            "row_count": preflight.completed_updates,
            "sha256": artifacts["history.json"]["sha256"],
        },
        "rng": {
            "schema_version": RNG_SCHEMA_VERSION,
            "artifact": "rng.json",
            "sha256": artifacts["rng.json"]["sha256"],
            "exact_resume_required": True,
        },
        "checkpoint_sequence": preflight.checkpoint_sequence,
        "previous_checkpoint": None if preflight.previous is None else preflight.previous.as_dict(),
        "artifacts": {name: dict(artifacts[name]) for name in ARTIFACT_NAMES},
    }
    value["manifest_digest"] = object_sha256(value)
    return value


def _reference_from_manifest(manifest: Mapping[str, Any], manifest_bytes: bytes) -> CheckpointReference:
    return CheckpointReference(
        checkpoint_sequence=manifest["checkpoint_sequence"],
        completed_updates=manifest["progress"]["completed_updates"],
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        manifest_digest=manifest["manifest_digest"],
        history_sha256=manifest["history"]["sha256"],
        rng_sha256=manifest["rng"]["sha256"],
        schedule_prefix_sha256=manifest["schedule"]["prefix_sha256"],
        trainable_state_sha256=manifest["trainables"]["state_sha256"],
        optimizer_v_state_sha256=manifest["optimizer"]["state_sha256"],
    )


def _commit_rank_zero(
    target: Path,
    *,
    preflight: _Preflight,
    observer: Optional[StreamObserver],
) -> Mapping[str, Any]:
    try:
        os.lstat(target)
    except FileNotFoundError:
        pass
    else:
        try:
            recovered = _validate_existing_against_preflight(
                target, preflight=preflight, observer=observer
            )
            return {
                "ok": True,
                "status": "recovered_exact_existing",
                "reference": recovered.as_dict(),
            }
        except Exception as error:
            return {
                "ok": False,
                "status": "not_committed",
                "error_type": type(error).__name__,
                "error": f"existing checkpoint is not the exact transaction: {error}",
            }
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    stage_stat = os.lstat(stage)
    identity = (int(stage_stat.st_dev), int(stage_stat.st_ino))
    renamed = False
    reference: Optional[CheckpointReference] = None
    try:
        artifacts: dict[str, Mapping[str, Any]] = {}
        trainable_sha, trainable_size = _write_tensor_file(
            stage / "trainables.f32",
            scan=preflight.parameter_scan,
            named=preflight.parameters,
            observer=observer,
        )
        artifacts["trainables.f32"] = _artifact_row(trainable_sha, trainable_size)
        optimizer_sha, optimizer_size = _write_tensor_file(
            stage / "optimizer_v.f32",
            scan=preflight.optimizer_v_scan,
            named=preflight.moments,
            observer=observer,
        )
        artifacts["optimizer_v.f32"] = _artifact_row(optimizer_sha, optimizer_size)
        schedule_bytes = canonical_json_bytes([dict(row) for row in preflight.schedule]) + b"\n"
        schedule_sha, schedule_size = _write_plain_file(
            stage / "schedule.json", schedule_bytes
        )
        artifacts["schedule.json"] = _artifact_row(schedule_sha, schedule_size)
        history_sha, history_size = _write_plain_file(
            stage / "history.json", preflight.history_bytes
        )
        artifacts["history.json"] = _artifact_row(history_sha, history_size)
        rng_sha, rng_size = _write_plain_file(stage / "rng.json", preflight.rng_bytes)
        artifacts["rng.json"] = _artifact_row(rng_sha, rng_size)
        manifest = _build_manifest(preflight, artifacts)
        manifest_bytes = canonical_json_bytes(manifest) + b"\n"
        _write_plain_file(stage / "manifest.json", manifest_bytes)
        reference = _reference_from_manifest(manifest, manifest_bytes)
        os.chmod(stage, DIRECTORY_MODE)
        _fsync_directory(stage)
        _rename_noreplace(stage, target)
        renamed = True
        _fsync_directory(target.parent)
        return {
            "ok": True,
            "status": "committed",
            "reference": reference.as_dict(),
        }
    except Exception as error:
        if not renamed:
            _cleanup_stage(stage, identity)
            return {
                "ok": False,
                "status": "not_committed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        return {
            "ok": False,
            "status": "commit_indeterminate",
            "reference": None if reference is None else reference.as_dict(),
            "error_type": type(error).__name__,
            "error": str(error),
            "recovery": "call save_checkpoint again with identical state for exact-existing validation",
        }


def save_checkpoint(
    path: str | Path,
    *,
    optimizer: Any,
    completed_updates: int,
    full_schedule: Sequence[Any],
    history: Sequence[Mapping[str, Any]],
    rng_state: Mapping[str, Any],
    bindings: CheckpointBindings | Mapping[str, Any],
    authoritative_inventory_sha256: str,
    previous_checkpoint: Optional[LoadedCheckpoint | CheckpointReference] = None,
    rank: int = 0,
    world_size: int = 1,
    status_consensus: Optional[StatusConsensus] = None,
    result_broadcast: Optional[ResultBroadcast] = None,
    stream_observer: Optional[StreamObserver] = None,
    test_only_allow_small_capacity: bool = False,
) -> CheckpointReference:
    """Stream-hash on every rank, then stream-write only on rank zero.

    The caller invokes this only at a completed-update segment boundary, after
    sealing the optimizer receipt/history row, data cursor, and all WORLD8 RNG
    states.  An absent-gradient gate and the all-rank state digest make that
    boundary explicit; this function never substitutes an untyped barrier.

    In WORLD8 the status callback performs exactly one all-rank exchange even
    when local preflight fails.  The result callback then broadcasts rank
    zero's structured commit outcome.  Neither callback may broadcast tensors.
    """

    if (
        type(rank) is not int
        or type(world_size) is not int
        or world_size not in {1, WORLD_SIZE}
        or not 0 <= rank < world_size
    ):
        raise Full30CheckpointError("checkpoint rank/world contract differs")
    preflight: Optional[_Preflight] = None
    target: Optional[Path] = None
    local_digest: Optional[str] = None
    local_error: Optional[Exception] = None
    try:
        target = _canonical_path(path, must_exist=False)
        preflight = _build_preflight(
            optimizer=optimizer,
            completed_updates=completed_updates,
            full_schedule=full_schedule,
            history=history,
            rng_state=rng_state,
            bindings=bindings,
            previous_checkpoint=previous_checkpoint,
            authoritative_inventory_sha256=authoritative_inventory_sha256,
            test_only_allow_small_capacity=test_only_allow_small_capacity,
            observer=stream_observer,
        )
        local_digest = object_sha256(
            {
                "checkpoint_target": os.fsdecode(os.fsencode(target)),
                "state_digest": preflight.local_state_digest,
            }
        )
        local_status: Mapping[str, Any] = {
            "ok": True,
            "digest": local_digest,
        }
    except Exception as error:
        local_error = error
        local_status = {
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
            "error_sentinel": object_sha256(
                {"phase": "save-preflight", "error_type": type(error).__name__}
            ),
        }
    consensus = _exchange_local_status(
        local_status,
        world_size=world_size,
        consensus_callback=status_consensus,
        label="save preflight",
    )
    if consensus["ok"] is not True:
        raise Full30CheckpointError(
            f"save preflight failed consistently across ranks: {consensus['error']}"
        )
    locally_ready = (
        preflight is not None
        and target is not None
        and local_error is None
        and local_digest is not None
        and consensus["digest"] == local_digest
    )
    local_result: Optional[Mapping[str, Any]] = None
    if rank == 0:
        if not locally_ready:
            local_result = {
                "ok": False,
                "status": "not_committed",
                "error_type": "Full30CheckpointError",
                "error": "save preflight state differs across ranks",
            }
        else:
            try:
                local_result = _commit_rank_zero(
                    target, preflight=preflight, observer=stream_observer
                )
            except Exception as error:
                # All operations which can publish are handled inside
                # _commit_rank_zero.  This guard covers pre-publication setup
                # failures and, critically, still reaches result_broadcast.
                local_result = {
                    "ok": False,
                    "status": "not_committed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
    if world_size == 1:
        result = local_result
    else:
        if result_broadcast is None:
            raise Full30CheckpointCommitIndeterminate(
                "rank-zero result broadcast is absent; commit state is indeterminate"
            )
        try:
            result = result_broadcast(local_result)
        except Exception as error:
            raise Full30CheckpointCommitIndeterminate(
                "rank-zero result broadcast failed; use exact-existing recovery: "
                f"{error}"
            ) from error
    if not isinstance(result, Mapping) or type(result.get("ok")) is not bool:
        raise Full30CheckpointCommitIndeterminate(
            "rank-zero result is malformed; commit state is indeterminate"
        )
    status = result.get("status")
    if result["ok"] is True:
        if not locally_ready:
            raise Full30CheckpointError("save preflight state differs across ranks")
        if status not in {"committed", "recovered_exact_existing"}:
            raise Full30CheckpointCommitIndeterminate("rank-zero success status differs")
        reference = CheckpointReference.from_mapping(result.get("reference"))
        return reference
    if status == "commit_indeterminate":
        raise Full30CheckpointCommitIndeterminate(
            "rank-zero checkpoint commit is indeterminate; exact-existing recovery required: "
            f"{result.get('error')}"
        )
    if status != "not_committed":
        raise Full30CheckpointCommitIndeterminate("rank-zero failure status differs")
    raise Full30CheckpointTransactionError(
        f"rank-zero checkpoint save did not commit [{result.get('error_type')}]: "
        f"{result.get('error')}"
    )


def _canonical_json_file(
    root: Path,
    name: str,
    *,
    artifact: Mapping[str, Any],
    label: str,
) -> tuple[Any, bytes]:
    if not isinstance(artifact, Mapping) or set(artifact) != {"sha256", "bytes", "mode"}:
        raise Full30CheckpointError(f"{label} artifact row differs")
    expected_sha = _sha256(artifact["sha256"], label=f"{label} artifact SHA")
    if type(artifact["bytes"]) is not int or artifact["bytes"] < 0 or artifact["mode"] != "0600":
        raise Full30CheckpointError(f"{label} artifact size/mode differs")
    raw = _plain_file_bytes(root / name, label=label)
    if len(raw) != artifact["bytes"] or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise Full30CheckpointError(f"{label} artifact bytes differ")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise Full30CheckpointError(f"{label} newline framing differs")
    value = _strict_json(raw[:-1], label=label)
    if canonical_json_bytes(value) + b"\n" != raw:
        raise Full30CheckpointError(f"{label} is not canonical")
    return value, raw


def _manifest_file(root: Path) -> tuple[dict[str, Any], bytes]:
    raw = _plain_file_bytes(root / "manifest.json", label="checkpoint manifest")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise Full30CheckpointError("manifest newline framing differs")
    value = _strict_json(raw[:-1], label="checkpoint manifest")
    if not isinstance(value, dict) or canonical_json_bytes(value) + b"\n" != raw:
        raise Full30CheckpointError("checkpoint manifest is not canonical")
    fields = {
        "schema_version",
        "directory_mode",
        "file_mode",
        "bindings",
        "capacity",
        "trainables",
        "optimizer",
        "progress",
        "schedule",
        "history",
        "rng",
        "checkpoint_sequence",
        "previous_checkpoint",
        "artifacts",
        "manifest_digest",
    }
    if set(value) != fields or value["schema_version"] != SCHEMA_VERSION:
        raise Full30CheckpointError("manifest field/schema set differs")
    declared = _sha256(value["manifest_digest"], label="manifest digest")
    unsigned = dict(value)
    unsigned.pop("manifest_digest")
    if object_sha256(unsigned) != declared:
        raise Full30CheckpointError("manifest embedded digest differs")
    if value["directory_mode"] != "0750" or value["file_mode"] != "0600":
        raise Full30CheckpointError("manifest mode contract differs")
    return value, raw


def _validate_checkpoint_local(
    root: Path,
    *,
    inventory: Sequence[Mapping[str, Any]],
    authoritative_inventory_sha256: str,
    production_capacity_authorized: bool,
    expected_bindings: CheckpointBindings,
    expected_schedule: Sequence[Any],
    expected_completed_updates: int,
    expected_previous: Optional[LoadedCheckpoint | CheckpointReference],
    expected_reference: Optional[CheckpointReference],
    observer: Optional[StreamObserver],
) -> LoadedCheckpoint:
    root = _canonical_path(root, must_exist=True)
    root_before = os.lstat(root)
    entries = list(os.scandir(root))
    if {entry.name for entry in entries} != ALL_FILE_NAMES or any(
        not entry.is_file(follow_symlinks=False) for entry in entries
    ):
        raise Full30CheckpointError("checkpoint artifact closure differs")
    manifest, manifest_bytes = _manifest_file(root)
    bindings = CheckpointBindings.from_mapping(manifest["bindings"])
    if bindings != expected_bindings:
        raise Full30CheckpointError("checkpoint arm/SHA bindings differ")
    capacity = manifest.get("capacity")
    expected_capacity = {
        "authoritative_inventory_sha256": authoritative_inventory_sha256,
        "tensor_count": len(inventory),
        "total_numel": sum(int(row["numel"]) for row in inventory),
        "production_capacity_authorized": production_capacity_authorized,
    }
    if capacity != expected_capacity:
        raise Full30CheckpointError("checkpoint authoritative capacity differs")
    if production_capacity_authorized:
        _validate_production_capacity(inventory)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_NAMES):
        raise Full30CheckpointError("checkpoint artifact manifest differs")

    schedule_value, _schedule_bytes = _canonical_json_file(
        root,
        "schedule.json",
        artifact=artifacts["schedule.json"],
        label="checkpoint schedule",
    )
    schedule = canonical_schedule_v2(schedule_value)
    expected_schedule_rows = canonical_schedule_v2(expected_schedule)
    if [dict(row) for row in schedule] != [dict(row) for row in expected_schedule_rows]:
        raise Full30CheckpointError("checkpoint full schedule differs")
    if type(expected_completed_updates) is not int or not 0 <= expected_completed_updates <= MAX_UPDATES:
        raise Full30CheckpointError("expected completed updates differ")
    full_sha, prefix_sha = schedule_digests_v2(schedule, expected_completed_updates)
    expected_cursor = dict(next_cursor_v2(schedule, expected_completed_updates))
    if manifest.get("progress") != {
        "completed_updates": expected_completed_updates,
        "next_cursor": expected_cursor,
    }:
        raise Full30CheckpointError("checkpoint mechanical next cursor differs")
    if manifest.get("schedule") != {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "artifact": "schedule.json",
        "flat_row_count": FLAT_SCHEDULE_ROWS,
        "completed_prefix_flat_rows": GLOBAL_BATCH * expected_completed_updates,
        "full_sha256": full_sha,
        "prefix_sha256": prefix_sha,
    }:
        raise Full30CheckpointError("checkpoint full/prefix schedule digest differs")

    history_value, history_bytes = _canonical_json_file(
        root,
        "history.json",
        artifact=artifacts["history.json"],
        label="checkpoint history",
    )
    rng_value, rng_bytes = _canonical_json_file(
        root,
        "rng.json",
        artifact=artifacts["rng.json"],
        label="checkpoint RNG state",
    )
    rng = _validate_rng_state(rng_value)
    rng_sha = hashlib.sha256(rng_bytes).hexdigest()
    if manifest.get("rng") != {
        "schema_version": RNG_SCHEMA_VERSION,
        "artifact": "rng.json",
        "sha256": rng_sha,
        "exact_resume_required": True,
    }:
        raise Full30CheckpointError("checkpoint RNG manifest differs")

    trainables_row = manifest.get("trainables")
    if (
        not isinstance(trainables_row, dict)
        or set(trainables_row)
        != {"artifact", "inventory", "state_sha256", "stored_dtype"}
        or trainables_row["artifact"] != "trainables.f32"
        or trainables_row["inventory"] != [dict(row) for row in inventory]
        or trainables_row["stored_dtype"] != "float32-le"
    ):
        raise Full30CheckpointError("checkpoint trainable manifest differs")
    trainable_artifact = artifacts["trainables.f32"]
    trainable_index = _read_tensor_index(
        root / "trainables.f32",
        expected_kind="trainables",
        inventory=inventory,
        expected_file_sha256=_sha256(trainable_artifact["sha256"], label="trainable file SHA"),
        expected_file_bytes=trainable_artifact["bytes"],
        nonnegative=False,
        require_all_zero=False,
        observer=observer,
    )
    if trainable_index.state_sha256 != trainables_row["state_sha256"]:
        raise Full30CheckpointError("checkpoint trainable state digest differs")

    optimizer_row = manifest.get("optimizer")
    expected_optimizer_fields = {
        "artifact",
        "class",
        "schema_version",
        "state_kind",
        "update_count",
        "state_sha256",
        "nonnegative",
        "u0_all_zero",
    }
    if (
        not isinstance(optimizer_row, dict)
        or set(optimizer_row) != expected_optimizer_fields
        or optimizer_row["artifact"] != "optimizer_v.f32"
        or optimizer_row["class"] != OPTIMIZER_CLASS
        or optimizer_row["schema_version"] != OPTIMIZER_SCHEMA_VERSION
        or optimizer_row["state_kind"] != STATE_KIND
        or optimizer_row["update_count"] != expected_completed_updates
        or optimizer_row["nonnegative"] is not True
        or optimizer_row["u0_all_zero"] is not (expected_completed_updates == 0)
    ):
        raise Full30CheckpointError("checkpoint typed optimizer state differs")
    optimizer_artifact = artifacts["optimizer_v.f32"]
    optimizer_index = _read_tensor_index(
        root / "optimizer_v.f32",
        expected_kind="optimizer_v",
        inventory=inventory,
        expected_file_sha256=_sha256(optimizer_artifact["sha256"], label="optimizer file SHA"),
        expected_file_bytes=optimizer_artifact["bytes"],
        nonnegative=True,
        require_all_zero=expected_completed_updates == 0,
        observer=observer,
    )
    if optimizer_index.state_sha256 != optimizer_row["state_sha256"]:
        raise Full30CheckpointError("checkpoint optimizer v_t digest differs")

    previous, previous_history = _previous_parts(expected_previous)
    recorded_previous = manifest.get("previous_checkpoint")
    if recorded_previous != (None if previous is None else previous.as_dict()):
        raise Full30CheckpointError("checkpoint previous reference differs")
    expected_sequence = 0 if previous is None else previous.checkpoint_sequence + 1
    if manifest.get("checkpoint_sequence") != expected_sequence:
        raise Full30CheckpointError("checkpoint does not follow immediately published predecessor")
    checked_history = _history_rows(
        history_value,
        completed_updates=expected_completed_updates,
        schedule=schedule,
        current_parameter_sha256=trainable_index.state_sha256,
        current_optimizer_v_sha256=optimizer_index.state_sha256,
        current_rng_sha256=rng_sha,
        previous=previous,
        previous_history=previous_history,
    )
    history_sha = hashlib.sha256(history_bytes).hexdigest()
    if manifest.get("history") != {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "artifact": "history.json",
        "row_count": expected_completed_updates,
        "sha256": history_sha,
    }:
        raise Full30CheckpointError("checkpoint history manifest differs")
    reference = _reference_from_manifest(manifest, manifest_bytes)
    if expected_reference is not None and reference != expected_reference:
        raise Full30CheckpointError("current checkpoint reference differs")
    root_after = os.lstat(root)
    if _directory_identity(root_before) != _directory_identity(root_after):
        raise Full30CheckpointError("checkpoint directory changed while validating")
    local_digest = object_sha256(
        {
            "reference": reference.as_dict(),
            "bindings": bindings.as_dict(),
            "inventory_sha256": authoritative_inventory_sha256,
            "schedule_full_sha256": full_sha,
            "local_all_bytes_validated": True,
        }
    )
    return LoadedCheckpoint(
        root=root,
        bindings=bindings,
        completed_updates=expected_completed_updates,
        next_cursor=_deep_freeze(expected_cursor),
        schedule=tuple(_deep_freeze(dict(row)) for row in schedule),
        history=tuple(_deep_freeze(dict(row)) for row in checked_history),
        rng_state=_deep_freeze(rng),
        previous_checkpoint=previous,
        reference=reference,
        manifest=_deep_freeze(manifest),
        manifest_bytes=manifest_bytes,
        trainable_index=trainable_index,
        optimizer_v_index=optimizer_index,
        local_validation_digest=local_digest,
    )


def _validate_existing_against_preflight(
    target: Path,
    *,
    preflight: _Preflight,
    observer: Optional[StreamObserver],
) -> CheckpointReference:
    loaded = _validate_checkpoint_local(
        target,
        inventory=preflight.inventory,
        authoritative_inventory_sha256=preflight.inventory_sha256,
        production_capacity_authorized=preflight.production_capacity_authorized,
        expected_bindings=preflight.bindings,
        expected_schedule=preflight.schedule,
        expected_completed_updates=preflight.completed_updates,
        expected_previous=preflight.previous,
        expected_reference=None,
        observer=observer,
    )
    if (
        loaded.reference.trainable_state_sha256
        != preflight.parameter_scan.state_sha256
        or loaded.reference.optimizer_v_state_sha256
        != preflight.optimizer_v_scan.state_sha256
        or loaded.reference.history_sha256
        != hashlib.sha256(preflight.history_bytes).hexdigest()
        or loaded.reference.rng_sha256
        != hashlib.sha256(preflight.rng_bytes).hexdigest()
        or loaded.reference.schedule_prefix_sha256
        != preflight.schedule_prefix_sha256
        or loaded.reference.checkpoint_sequence != preflight.checkpoint_sequence
    ):
        raise Full30CheckpointError("existing checkpoint logical state differs")
    return loaded.reference


def load_checkpoint(
    path: str | Path,
    *,
    optimizer: Any,
    expected_bindings: CheckpointBindings | Mapping[str, Any],
    expected_full_schedule: Sequence[Any],
    expected_completed_updates: int,
    expected_previous_checkpoint: Optional[LoadedCheckpoint | CheckpointReference],
    expected_reference: Optional[CheckpointReference],
    authoritative_inventory_sha256: str,
    rank: int = 0,
    world_size: int = 1,
    status_consensus: Optional[StatusConsensus] = None,
    stream_observer: Optional[StreamObserver] = None,
    test_only_allow_small_capacity: bool = False,
) -> LoadedCheckpoint:
    """Validate every local byte, then perform one fixed status consensus."""

    if (
        type(rank) is not int
        or type(world_size) is not int
        or world_size not in {1, WORLD_SIZE}
        or not 0 <= rank < world_size
    ):
        raise Full30CheckpointError("checkpoint load rank/world contract differs")
    expected_inventory_sha: Optional[str] = None
    loaded: Optional[LoadedCheckpoint] = None
    local_error: Optional[Exception] = None
    try:
        if expected_reference is None and not test_only_allow_small_capacity:
            raise Full30CheckpointError(
                "production resume requires the expected current checkpoint reference"
            )
        expected_inventory_sha = _sha256(
            authoritative_inventory_sha256, label="authoritative inventory SHA"
        )
        inventory_receipt = inventory_identity_v2(
            optimizer, test_only_allow_small_capacity=test_only_allow_small_capacity
        )
        if inventory_receipt["inventory_sha256"] != expected_inventory_sha:
            raise Full30CheckpointError("live authoritative inventory SHA differs")
        inventory = tuple(
            MappingProxyType(dict(row)) for row in inventory_receipt["rows"]
        )
        loaded = _validate_checkpoint_local(
            _canonical_path(path, must_exist=True),
            inventory=inventory,
            authoritative_inventory_sha256=expected_inventory_sha,
            production_capacity_authorized=not test_only_allow_small_capacity,
            expected_bindings=_normalise_bindings(expected_bindings),
            expected_schedule=expected_full_schedule,
            expected_completed_updates=expected_completed_updates,
            expected_previous=expected_previous_checkpoint,
            expected_reference=expected_reference,
            observer=stream_observer,
        )
        local_status: Mapping[str, Any] = {
            "ok": True,
            "digest": loaded.local_validation_digest,
        }
    except Exception as error:
        local_error = error
        local_status = {
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
            "error_sentinel": object_sha256(
                {"phase": "load-validation", "error_type": type(error).__name__}
            ),
        }
    consensus = _exchange_local_status(
        local_status,
        world_size=world_size,
        consensus_callback=status_consensus,
        label="load validation",
    )
    if consensus["ok"] is not True:
        raise Full30CheckpointError(
            f"load validation failed consistently across ranks: {consensus['error']}"
        )
    if loaded is None or local_error is not None or consensus["digest"] != loaded.local_validation_digest:
        raise Full30CheckpointError("load checkpoint digest differs across ranks")
    return loaded


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _load_tensor_candidates(
    index: TensorFileIndex,
    *,
    named: Sequence[tuple[str, Any]],
    nonnegative: bool,
    observer: Optional[StreamObserver],
) -> Mapping[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise Full30CheckpointError("exact-state restore requires PyTorch") from error
    current = os.lstat(index.path)
    identity = _regular_file_identity(current)
    if (
        identity != index.file_identity
        or not stat.S_ISREG(current.st_mode)
        or stat.S_IMODE(current.st_mode) != FILE_MODE
        or current.st_nlink != 1
    ):
        raise Full30CheckpointError(f"{index.kind} file changed after load validation")
    if tuple(name for name, _ in named) != tuple(row["name"] for row in index.rows):
        raise Full30CheckpointError(f"{index.kind} restore name order differs")
    descriptor = os.open(index.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    candidates: dict[str, Any] = {}
    try:
        os.lseek(descriptor, index.payload_offset, os.SEEK_SET)
        for (name, reference), row in zip(named, index.rows):
            if _dtype_text(reference) != "torch.float32" or not _is_contiguous(reference):
                raise Full30CheckpointError(f"live FP32 destination differs: {name}")
            candidate = torch.empty(
                tuple(row["shape"]), dtype=torch.float32, device=reference.device
            ).contiguous()
            flat = candidate.view(-1)
            offset = 0
            digest = hashlib.sha256()
            while offset < int(row["numel"]):
                count = min(STREAM_CHUNK_ELEMENTS, int(row["numel"]) - offset)
                raw = _read_exact(
                    descriptor,
                    count * 4,
                    observer=observer,
                    event=f"restore:{index.kind}",
                )
                valid, _zero = _raw_values_valid(raw, nonnegative=nonnegative)
                if not valid:
                    raise Full30CheckpointError(
                        f"{index.kind} restore chunk contains invalid FP32"
                    )
                source = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
                flat.narrow(0, offset, count).copy_(source.to(device=reference.device))
                digest.update(raw)
                offset += count
            if digest.hexdigest() != row["tensor_sha256"]:
                raise Full30CheckpointError(f"{index.kind} restore tensor SHA differs")
            # Re-read the candidate bytes through the production bounded path.
            candidate_digest = hashlib.sha256()
            for raw in _iter_tensor_chunks(
                candidate,
                observer=observer,
                event=f"roundtrip:{index.kind}",
            ):
                candidate_digest.update(raw)
            if candidate_digest.hexdigest() != row["tensor_sha256"]:
                raise Full30CheckpointError(
                    f"{index.kind} candidate roundtrip is not exact FP32"
                )
            candidates[name] = candidate
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.lstat(index.path)
    if (
        _regular_file_identity(after) != index.file_identity
        or _regular_file_identity(opened) != index.file_identity
    ):
        raise Full30CheckpointError(f"{index.kind} file changed during restore staging")
    return candidates


def restore_checkpoint_state(
    checkpoint: LoadedCheckpoint,
    *,
    optimizer: Any,
    rng_transaction_factory: Callable[[Mapping[str, Any]], Any],
    stream_observer: Optional[StreamObserver] = None,
    test_only_allow_small_capacity: bool = False,
    test_only_commit_hook: Optional[Callable[[str, int], None]] = None,
) -> Mapping[str, Any]:
    """Transactionally swap validated FP32 state into live Parameters/optimizer.

    ``rng_transaction_factory`` must be non-mutating and return an object with
    ``commit() -> rng_artifact_sha256`` and ``rollback()``.  Parameter objects
    are retained; only their ``.data`` storage references are swapped.
    """

    if not isinstance(checkpoint, LoadedCheckpoint):
        raise Full30CheckpointError("loaded checkpoint object differs")
    parameters, moments, old_update_count = _named_optimizer_state(
        optimizer, test_only_allow_small_capacity=test_only_allow_small_capacity
    )
    inventory_receipt = inventory_identity_v2(
        optimizer, test_only_allow_small_capacity=test_only_allow_small_capacity
    )
    if inventory_receipt["inventory_sha256"] != checkpoint.manifest["capacity"][
        "authoritative_inventory_sha256"
    ]:
        raise Full30CheckpointError("restore live inventory SHA differs")
    inventory = tuple(MappingProxyType(dict(row)) for row in inventory_receipt["rows"])
    if [dict(row) for row in inventory] != [
        dict(row) for row in checkpoint.trainable_index.rows
    ]:
        # Tensor rows contain offsets/digests in addition to inventory.
        stripped = [
            {
                key: row[key]
                for key in ("name", "shape", "runtime_dtype", "stored_dtype", "numel")
            }
            for row in checkpoint.trainable_index.rows
        ]
        if [dict(row) for row in inventory] != stripped:
            raise Full30CheckpointError("restore trainable inventory differs")
    if any(getattr(parameter, "grad", None) is not None for _, parameter in parameters):
        raise Full30CheckpointError("restore requires every live parameter gradient absent")
    before_parameters = _scan_named_state(
        parameters,
        kind="trainables",
        inventory=inventory,
        nonnegative=False,
        require_all_zero=False,
        observer=stream_observer,
    )
    before_moments = _scan_named_state(
        moments,
        kind="optimizer_v",
        inventory=inventory,
        nonnegative=True,
        require_all_zero=old_update_count == 0,
        observer=stream_observer,
    )
    candidate_parameters = _load_tensor_candidates(
        checkpoint.trainable_index,
        named=parameters,
        nonnegative=False,
        observer=stream_observer,
    )
    candidate_moments = _load_tensor_candidates(
        checkpoint.optimizer_v_index,
        named=moments,
        nonnegative=True,
        observer=stream_observer,
    )
    try:
        rng_transaction = rng_transaction_factory(_thaw_json(checkpoint.rng_state))
    except Exception as error:
        raise Full30CheckpointError(f"cannot prepare RNG restore transaction: {error}") from error
    if not callable(getattr(rng_transaction, "commit", None)) or not callable(
        getattr(rng_transaction, "rollback", None)
    ):
        raise Full30CheckpointError("RNG restore transaction interface differs")
    parameter_ids = {name: id(parameter) for name, parameter in parameters}
    hook_ids = {
        name: id(getattr(parameter, "_backward_hooks", None))
        for name, parameter in parameters
    }
    old_data = {name: parameter.data for name, parameter in parameters}
    old_moment_mapping = getattr(optimizer, "_second_moments")
    rng_commit_attempted = False
    swapped = 0
    try:
        for index_value, (name, parameter) in enumerate(parameters):
            parameter.data = candidate_parameters[name]
            swapped += 1
            if test_only_commit_hook is not None:
                test_only_commit_hook(name, index_value)
        optimizer._second_moments = dict(candidate_moments)
        optimizer._update_count = checkpoint.completed_updates
        rng_commit_attempted = True
        rng_digest = rng_transaction.commit()
        if rng_digest != checkpoint.reference.rng_sha256:
            raise Full30CheckpointError("restored RNG state digest differs")
        after_parameters = _scan_named_state(
            parameters,
            kind="trainables",
            inventory=inventory,
            nonnegative=False,
            require_all_zero=False,
            observer=stream_observer,
        )
        restored_moments = tuple(
            (name, optimizer._second_moments[name]) for name, _ in parameters
        )
        after_moments = _scan_named_state(
            restored_moments,
            kind="optimizer_v",
            inventory=inventory,
            nonnegative=True,
            require_all_zero=checkpoint.completed_updates == 0,
            observer=stream_observer,
        )
        if (
            after_parameters.state_sha256 != checkpoint.reference.trainable_state_sha256
            or after_moments.state_sha256
            != checkpoint.reference.optimizer_v_state_sha256
            or optimizer.update_count != checkpoint.completed_updates
            or any(id(parameter) != parameter_ids[name] for name, parameter in parameters)
            or any(
                id(getattr(parameter, "_backward_hooks", None)) != hook_ids[name]
                for name, parameter in parameters
            )
        ):
            raise Full30CheckpointError("restored optimizer/Parameter exact state differs")
        value = {
            "schema_version": "bernini-full30-action-restore-receipt-v2",
            "status": "committed",
            "completed_updates": checkpoint.completed_updates,
            "parameter_objects_preserved": True,
            "parameter_hooks_preserved": True,
            "trainable_state_sha256": after_parameters.state_sha256,
            "optimizer_v_state_sha256": after_moments.state_sha256,
            "rng_sha256": rng_digest,
            "transactional_storage_swap": True,
        }
        return {**value, "receipt_digest": object_sha256(value)}
    except Exception as error:
        rollback_errors: list[str] = []
        try:
            if rng_commit_attempted:
                rng_transaction.rollback()
        except Exception as rollback_error:
            rollback_errors.append(f"rng:{rollback_error}")
        try:
            for name, parameter in parameters:
                parameter.data = old_data[name]
            optimizer._second_moments = old_moment_mapping
            optimizer._update_count = old_update_count
        except Exception as rollback_error:
            rollback_errors.append(f"optimizer:{rollback_error}")
        if not rollback_errors:
            try:
                rollback_parameters = _scan_named_state(
                    parameters,
                    kind="trainables",
                    inventory=inventory,
                    nonnegative=False,
                    require_all_zero=False,
                    observer=stream_observer,
                )
                rollback_moments = _scan_named_state(
                    tuple((name, optimizer._second_moments[name]) for name, _ in parameters),
                    kind="optimizer_v",
                    inventory=inventory,
                    nonnegative=True,
                    require_all_zero=old_update_count == 0,
                    observer=stream_observer,
                )
                if (
                    rollback_parameters.state_sha256 != before_parameters.state_sha256
                    or rollback_moments.state_sha256 != before_moments.state_sha256
                    or optimizer.update_count != old_update_count
                    or any(id(parameter) != parameter_ids[name] for name, parameter in parameters)
                    or any(
                        id(getattr(parameter, "_backward_hooks", None)) != hook_ids[name]
                        for name, parameter in parameters
                    )
                ):
                    rollback_errors.append("rollback digest/object identity differs")
            except Exception as rollback_error:
                rollback_errors.append(f"verification:{rollback_error}")
        if rollback_errors:
            raise Full30CheckpointError(
                "restore failed and rollback could not be proven: "
                + "; ".join(rollback_errors)
            ) from error
        raise Full30CheckpointError(
            f"restore transaction failed and rolled back after {swapped} swaps: {error}"
        ) from error


def optimizer_state_identity_v2(
    optimizer: Any,
    *,
    test_only_allow_small_capacity: bool = False,
    stream_observer: Optional[StreamObserver] = None,
) -> Mapping[str, Any]:
    parameters, moments, update_count = _named_optimizer_state(
        optimizer, test_only_allow_small_capacity=test_only_allow_small_capacity
    )
    inventory_receipt = inventory_identity_v2(
        optimizer, test_only_allow_small_capacity=test_only_allow_small_capacity
    )
    inventory = tuple(MappingProxyType(dict(row)) for row in inventory_receipt["rows"])
    parameter_scan = _scan_named_state(
        parameters,
        kind="trainables",
        inventory=inventory,
        nonnegative=False,
        require_all_zero=False,
        observer=stream_observer,
    )
    moment_scan = _scan_named_state(
        moments,
        kind="optimizer_v",
        inventory=inventory,
        nonnegative=True,
        require_all_zero=update_count == 0,
        observer=stream_observer,
    )
    value = {
        "optimizer_schema_version": OPTIMIZER_SCHEMA_VERSION,
        "optimizer_class": OPTIMIZER_CLASS,
        "update_count": update_count,
        "inventory_sha256": inventory_receipt["inventory_sha256"],
        "trainable_state_sha256": parameter_scan.state_sha256,
        "optimizer_v_state_sha256": moment_scan.state_sha256,
    }
    return {**value, "identity_digest": object_sha256(value)}


def build_history_row_v2(
    *,
    update_count: int,
    optimizer_receipt_digest: str,
    parameters_before_sha256: str,
    parameters_after_sha256: str,
    optimizer_v_before_sha256: str,
    optimizer_v_after_sha256: str,
    rng_before_sha256: str,
    rng_after_sha256: str,
    schedule_prefix_sha256: str,
) -> Mapping[str, Any]:
    value = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "update_count": update_count,
        "optimizer_class": OPTIMIZER_CLASS,
        "optimizer_receipt_schema_version": OPTIMIZER_RECEIPT_SCHEMA_VERSION,
        "optimizer_receipt_digest": optimizer_receipt_digest,
        "parameters_before_sha256": parameters_before_sha256,
        "parameters_after_sha256": parameters_after_sha256,
        "optimizer_v_before_sha256": optimizer_v_before_sha256,
        "optimizer_v_after_sha256": optimizer_v_after_sha256,
        "rng_before_sha256": rng_before_sha256,
        "rng_after_sha256": rng_after_sha256,
        "schedule_prefix_sha256": schedule_prefix_sha256,
    }
    if type(update_count) is not int or not 1 <= update_count <= MAX_UPDATES:
        raise Full30CheckpointError("history update_count differs")
    for name in _HISTORY_FIELDS - {
        "schema_version",
        "update_count",
        "optimizer_class",
        "optimizer_receipt_schema_version",
    }:
        _sha256(value[name], label=f"history {name}")
    return MappingProxyType(value)


__all__ = [
    "ALL_FILE_NAMES",
    "ARTIFACT_NAMES",
    "CheckpointBindings",
    "CheckpointReference",
    "DIRECTORY_MODE",
    "EXACT_TRAINABLE_NUMEL",
    "EXACT_TRAINABLE_TENSORS",
    "FILE_MODE",
    "FLAT_SCHEDULE_ROWS",
    "Full30CheckpointCommitIndeterminate",
    "Full30CheckpointError",
    "Full30CheckpointTransactionError",
    "GLOBAL_BATCH",
    "HISTORY_SCHEMA_VERSION",
    "LoadedCheckpoint",
    "MAX_UPDATES",
    "OPTIMIZER_CLASS",
    "OPTIMIZER_RECEIPT_SCHEMA_VERSION",
    "OPTIMIZER_SCHEMA_VERSION",
    "RNG_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SCHEDULE_SCHEMA_VERSION",
    "STREAM_CHUNK_BYTES",
    "STREAM_CHUNK_ELEMENTS",
    "StreamingAllocationPlan",
    "TensorFileIndex",
    "build_history_row_v2",
    "canonical_json_bytes",
    "canonical_schedule_v2",
    "inventory_identity_v2",
    "load_checkpoint",
    "next_cursor_v2",
    "object_sha256",
    "optimizer_state_identity_v2",
    "restore_checkpoint_state",
    "save_checkpoint",
    "schedule_digests_v2",
    "streaming_allocation_plan_v2",
]
