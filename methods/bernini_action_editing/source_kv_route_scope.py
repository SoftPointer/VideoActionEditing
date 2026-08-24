#!/usr/bin/env python3
"""Fail-closed Bernini-R 1.3B LoRA scope contract for V9.

The V9 main arm starts a new adapter from the frozen Bernini base.  It adapts
the query and output projections that can route source evidence into the
target stream:

* ``attn2.to_q`` and ``attn2.to_out.0`` in every block 0 through 29; and
* ``attn1.to_q`` and ``attn1.to_out.0`` in blocks 7 through 22.

This module contains no training loop and imports no tensor framework.  It
provides the immutable name universe plus validators for the real Bernini
module inventory, PEFT adapter tensors, V9-main fresh initialization, and a
serialization-ready receipt manifest.  Tensor-like objects need only expose a
``shape`` attribute, so all contract tests run without PyTorch.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence


METHOD_NAME = "bernini-source-kv-route-v9"
RECEIPT_MANIFEST_SCHEMA = "bernini-source-kv-route-lora-scope-manifest-v9"
SCOPE_NAME = "all_cross_q_out_plus_mid_self_q_out"

TRANSFORMER_BLOCK_COUNT = 30
MIDDLE_SELF_BLOCKS_INCLUSIVE = (7, 22)
HIDDEN_SIZE = 1536
LORA_RANK = 8
LORA_ALPHA = 8
LORA_DROPOUT = 0.0
LORA_BIAS = "none"

EXPECTED_TARGET_MODULE_COUNT = 92
EXPECTED_ADAPTER_TENSOR_COUNT = 184
EXPECTED_TRAINABLE_PARAMETER_COUNT = 2_260_992
EXPECTED_TARGET_MODULES_SHA256 = (
    "16e5dc87ca134419841e2e9af6d26091141aa473aa4cc11ae53d2e4e28e0e4b5"
)

MAIN_RUN_ROLE = "v9_main"
FRESH_INITIALIZATION_SOURCE = "fresh_base_plus_new_peft_lora"

_TARGET_CANDIDATE_RE = re.compile(
    r"^diff_dec\.transformer\.blocks\.(?P<block>\d+)\."
    r"attn(?P<attention>[12])\.(?P<projection>to_q|to_out\.0)$"
)


class SourceKVRouteScopeError(RuntimeError):
    """Raised before an ambiguous V9 LoRA scope or artifact is accepted."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a receipt value using the repository's deterministic JSON form."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SourceKVRouteScopeError(
            f"scope value is not canonical JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _build_canonical_target_modules() -> tuple[str, ...]:
    targets = {
        f"diff_dec.transformer.blocks.{block}.attn2.{projection}"
        for block in range(TRANSFORMER_BLOCK_COUNT)
        for projection in ("to_q", "to_out.0")
    }
    first, last = MIDDLE_SELF_BLOCKS_INCLUSIVE
    targets.update(
        f"diff_dec.transformer.blocks.{block}.attn1.{projection}"
        for block in range(first, last + 1)
        for projection in ("to_q", "to_out.0")
    )
    return tuple(sorted(targets))


_CANONICAL_TARGET_MODULES = _build_canonical_target_modules()
if (
    len(_CANONICAL_TARGET_MODULES) != EXPECTED_TARGET_MODULE_COUNT
    or object_sha256(list(_CANONICAL_TARGET_MODULES))
    != EXPECTED_TARGET_MODULES_SHA256
):
    raise RuntimeError("V9 canonical target-module identity changed")


def canonical_target_modules() -> list[str]:
    """Return a fresh copy of the immutable, lexicographically ordered scope."""

    return list(_CANONICAL_TARGET_MODULES)


def canonical_adapter_state_keys() -> list[str]:
    """Return the exact PEFT safetensors key set in canonical order."""

    return sorted(
        f"base_model.model.{target}.lora_{factor}.weight"
        for target in _CANONICAL_TARGET_MODULES
        for factor in ("A", "B")
    )


def expected_adapter_shapes(
    *, rank: int = LORA_RANK, hidden_size: int = HIDDEN_SIZE
) -> dict[str, tuple[int, int]]:
    """Return exact LoRA A/B shapes after validating the immutable geometry."""

    _validate_rank_and_hidden_size(rank=rank, hidden_size=hidden_size)
    shapes: dict[str, tuple[int, int]] = {}
    for target in _CANONICAL_TARGET_MODULES:
        prefix = f"base_model.model.{target}"
        shapes[f"{prefix}.lora_A.weight"] = (rank, hidden_size)
        shapes[f"{prefix}.lora_B.weight"] = (hidden_size, rank)
    return dict(sorted(shapes.items()))


def _validate_rank_and_hidden_size(*, rank: Any, hidden_size: Any) -> None:
    if type(rank) is not int or rank != LORA_RANK:
        raise SourceKVRouteScopeError(f"V9 LoRA rank must be {LORA_RANK}")
    if type(hidden_size) is not int or hidden_size != HIDDEN_SIZE:
        raise SourceKVRouteScopeError(
            f"Bernini-R 1.3B hidden size must be {HIDDEN_SIZE}"
        )


def validate_lora_hyperparameters(
    *,
    rank: Any,
    alpha: Any,
    hidden_size: Any = HIDDEN_SIZE,
    dropout: Any = LORA_DROPOUT,
    bias: Any = LORA_BIAS,
) -> dict[str, Any]:
    """Validate the complete V9 LoRA geometry and return its receipt form."""

    _validate_rank_and_hidden_size(rank=rank, hidden_size=hidden_size)
    if type(alpha) is not int or alpha != LORA_ALPHA:
        raise SourceKVRouteScopeError(f"V9 LoRA alpha must be {LORA_ALPHA}")
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not math.isfinite(float(dropout))
        or float(dropout) != LORA_DROPOUT
    ):
        raise SourceKVRouteScopeError("V9 LoRA dropout must be exactly zero")
    if bias != LORA_BIAS:
        raise SourceKVRouteScopeError("V9 LoRA bias must be 'none'")
    return {
        "rank": rank,
        "alpha": alpha,
        "dropout": float(dropout),
        "bias": bias,
        "hidden_size": hidden_size,
    }


def validate_target_module_names(names: Sequence[str]) -> list[str]:
    """Require one exact copy of every canonical fully-qualified module name."""

    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
        raise SourceKVRouteScopeError("target modules must be a sequence of names")
    values = list(names)
    if not all(isinstance(name, str) and name for name in values):
        raise SourceKVRouteScopeError("target modules contain a non-name value")
    if len(values) != len(set(values)):
        raise SourceKVRouteScopeError("target modules contain duplicates")
    actual = sorted(values)
    expected = list(_CANONICAL_TARGET_MODULES)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise SourceKVRouteScopeError(
            "V9 target-module scope differs: "
            f"missing={missing[:4]} unexpected={unexpected[:4]}"
        )
    if object_sha256(actual) != EXPECTED_TARGET_MODULES_SHA256:
        raise SourceKVRouteScopeError("V9 target-module digest differs")
    return actual


def _shape_tuple(value: Any, *, label: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None or isinstance(shape, (str, bytes)):
        raise SourceKVRouteScopeError(f"{label} has no tensor-like shape")
    try:
        dimensions = tuple(shape)
    except TypeError as error:
        raise SourceKVRouteScopeError(f"{label} shape is not iterable") from error
    result: list[int] = []
    for dimension in dimensions:
        if isinstance(dimension, bool):
            raise SourceKVRouteScopeError(f"{label} shape is not concrete")
        try:
            integer = int(dimension)
        except (TypeError, ValueError, OverflowError) as error:
            raise SourceKVRouteScopeError(
                f"{label} shape is not concrete"
            ) from error
        try:
            numeric = float(dimension)
        except (TypeError, ValueError, OverflowError) as error:
            raise SourceKVRouteScopeError(
                f"{label} shape is not numeric"
            ) from error
        if not math.isfinite(numeric) or numeric != float(integer) or integer <= 0:
            raise SourceKVRouteScopeError(f"{label} shape is invalid")
        result.append(integer)
    return tuple(result)


def _module_weight_shape(module: Any, *, name: str) -> tuple[int, ...]:
    weight = getattr(module, "weight", None)
    if weight is None:
        raise SourceKVRouteScopeError(f"V9 target is not affine: {name}")
    return _shape_tuple(weight, label=f"weight for {name}")


def _inventory_entries(inventory_or_model: Any) -> Iterable[tuple[Any, Any]]:
    named_modules = getattr(inventory_or_model, "named_modules", None)
    if callable(named_modules):
        try:
            return named_modules()
        except Exception as error:
            raise SourceKVRouteScopeError(
                f"cannot enumerate runtime named_modules: {error}"
            ) from error
    if isinstance(inventory_or_model, Mapping):
        return inventory_or_model.items()
    if isinstance(inventory_or_model, (str, bytes)):
        raise SourceKVRouteScopeError("runtime module inventory is not iterable")
    try:
        return iter(inventory_or_model)
    except TypeError as error:
        raise SourceKVRouteScopeError(
            "runtime model does not expose named_modules()"
        ) from error


def validate_runtime_target_modules(inventory_or_model: Any) -> list[str]:
    """Enumerate and geometrically validate the exact V9 scope on a real model.

    Ordinary non-target model modules and the intentionally frozen self-attn
    q/out projections outside blocks 7..22 are ignored.  Any duplicate name or
    out-of-range Bernini q/out candidate is rejected.
    """

    selected: dict[str, Any] = {}
    seen: set[str] = set()
    for entry in _inventory_entries(inventory_or_model):
        if (
            not isinstance(entry, Sequence)
            or isinstance(entry, (str, bytes))
            or len(entry) != 2
        ):
            raise SourceKVRouteScopeError(
                "named_modules() must yield (name, module) pairs"
            )
        name, module = entry
        if not isinstance(name, str):
            raise SourceKVRouteScopeError("runtime module name is not text")
        if name in seen:
            raise SourceKVRouteScopeError(f"duplicate runtime module name: {name}")
        seen.add(name)
        match = _TARGET_CANDIDATE_RE.fullmatch(name)
        if match is None:
            continue
        block = int(match.group("block"))
        attention = int(match.group("attention"))
        if not 0 <= block < TRANSFORMER_BLOCK_COUNT:
            raise SourceKVRouteScopeError(
                f"out-of-range Bernini attention block in runtime inventory: {name}"
            )
        first, last = MIDDLE_SELF_BLOCKS_INCLUSIVE
        if attention == 2 or first <= block <= last:
            selected[name] = module

    names = validate_target_module_names(list(selected))
    expected_shape = (HIDDEN_SIZE, HIDDEN_SIZE)
    for name in names:
        actual_shape = _module_weight_shape(selected[name], name=name)
        if actual_shape != expected_shape:
            raise SourceKVRouteScopeError(
                f"V9 target weight shape differs for {name}: "
                f"{actual_shape} != {expected_shape}"
            )
    return names


def validate_adapter_state(
    state: Mapping[str, Any],
    *,
    rank: Any = LORA_RANK,
    hidden_size: Any = HIDDEN_SIZE,
) -> dict[str, Any]:
    """Validate exact PEFT keys, A/B shapes, tensor count, and parameter count."""

    _validate_rank_and_hidden_size(rank=rank, hidden_size=hidden_size)
    if not isinstance(state, Mapping):
        raise SourceKVRouteScopeError("adapter state must be a key-to-tensor mapping")
    if not all(isinstance(key, str) and key for key in state):
        raise SourceKVRouteScopeError("adapter state contains a non-name key")
    expected_shapes = expected_adapter_shapes(rank=rank, hidden_size=hidden_size)
    actual_keys = set(state)
    expected_keys = set(expected_shapes)
    if actual_keys != expected_keys or len(state) != EXPECTED_ADAPTER_TENSOR_COUNT:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise SourceKVRouteScopeError(
            "V9 adapter tensor scope differs: "
            f"count={len(state)} missing={missing[:4]} unexpected={unexpected[:4]}"
        )

    parameter_count = 0
    for key in sorted(expected_keys):
        actual_shape = _shape_tuple(state[key], label=f"adapter tensor {key}")
        expected_shape = expected_shapes[key]
        if actual_shape != expected_shape:
            raise SourceKVRouteScopeError(
                f"V9 adapter tensor shape differs for {key}: "
                f"{actual_shape} != {expected_shape}"
            )
        parameter_count += math.prod(actual_shape)
    if parameter_count != EXPECTED_TRAINABLE_PARAMETER_COUNT:
        raise SourceKVRouteScopeError(
            "V9 adapter parameter count differs: "
            f"{parameter_count} != {EXPECTED_TRAINABLE_PARAMETER_COUNT}"
        )
    keys = canonical_adapter_state_keys()
    return {
        "validated": True,
        "adapter_tensor_count": len(keys),
        "adapter_state_keys_sha256": object_sha256(keys),
        "trainable_parameter_count": parameter_count,
        "rank": rank,
        "hidden_size": hidden_size,
    }


def fresh_initialization_declaration() -> dict[str, Any]:
    """Return the only legal initialization declaration for the V9 main arm."""

    return {
        "run_role": MAIN_RUN_ROLE,
        "initialization_source": FRESH_INITIALIZATION_SOURCE,
        "peft_adapter_created_after_frozen_base_load": True,
        "adapter_checkpoint_loaded": False,
        "warm_start": False,
        "warm_start_method": None,
        "warm_start_adapter_sha256": None,
        "v8_warm_start": False,
    }


def validate_fresh_initialization(value: Mapping[str, Any]) -> dict[str, Any]:
    """Forbid V8 or any other adapter warm start for a claimed V9 main run."""

    if not isinstance(value, Mapping):
        raise SourceKVRouteScopeError(
            "V9 main initialization declaration must be a mapping"
        )
    candidate = dict(value)
    expected = fresh_initialization_declaration()
    if candidate.get("run_role") == MAIN_RUN_ROLE and (
        candidate.get("v8_warm_start") is not False
        or candidate.get("warm_start") is not False
        or candidate.get("adapter_checkpoint_loaded") is not False
        or candidate.get("warm_start_method") is not None
        or candidate.get("warm_start_adapter_sha256") is not None
    ):
        raise SourceKVRouteScopeError(
            "V9 main forbids V8 and all other adapter warm starts"
        )
    if candidate != expected:
        raise SourceKVRouteScopeError(
            "V9 main requires the exact fresh-adapter initialization contract"
        )
    return dict(expected)


def build_receipt_manifest(
    *,
    runtime_module_inventory: Any,
    adapter_state: Mapping[str, Any],
    initialization: Mapping[str, Any],
    rank: Any = LORA_RANK,
    alpha: Any = LORA_ALPHA,
    hidden_size: Any = HIDDEN_SIZE,
    dropout: Any = LORA_DROPOUT,
    bias: Any = LORA_BIAS,
) -> dict[str, Any]:
    """Build a digest-bound receipt manifest from validated runtime evidence."""

    targets = validate_runtime_target_modules(runtime_module_inventory)
    hyperparameters = validate_lora_hyperparameters(
        rank=rank,
        alpha=alpha,
        hidden_size=hidden_size,
        dropout=dropout,
        bias=bias,
    )
    adapter = validate_adapter_state(
        adapter_state,
        rank=rank,
        hidden_size=hidden_size,
    )
    fresh = validate_fresh_initialization(initialization)
    manifest: dict[str, Any] = {
        "schema_version": RECEIPT_MANIFEST_SCHEMA,
        "method": METHOD_NAME,
        "scope": SCOPE_NAME,
        "base_model": {
            "model": "Bernini-R-1.3B-Diffusers",
            "transformer_block_count": TRANSFORMER_BLOCK_COUNT,
            "hidden_size": HIDDEN_SIZE,
        },
        "lora": {
            **hyperparameters,
            "target_modules": targets,
            "target_module_count": len(targets),
            "target_modules_sha256": object_sha256(targets),
            "adapter_tensor_count": adapter["adapter_tensor_count"],
            "adapter_state_keys_sha256": adapter[
                "adapter_state_keys_sha256"
            ],
            "trainable_parameter_count": adapter[
                "trainable_parameter_count"
            ],
            "middle_self_blocks_inclusive": list(
                MIDDLE_SELF_BLOCKS_INCLUSIVE
            ),
            "cross_attention_blocks_inclusive": [
                0,
                TRANSFORMER_BLOCK_COUNT - 1,
            ],
        },
        "initialization": fresh,
        "validation": {
            "runtime_module_inventory_exact": True,
            "runtime_base_weight_shapes_exact": True,
            "adapter_state_scope_and_shapes_exact": adapter["validated"],
            "fresh_initialization_exact": True,
            "v8_warm_start_forbidden_for_main": True,
        },
    }
    manifest["manifest_digest"] = object_sha256(manifest)
    return manifest


__all__ = [
    "EXPECTED_ADAPTER_TENSOR_COUNT",
    "EXPECTED_TARGET_MODULE_COUNT",
    "EXPECTED_TARGET_MODULES_SHA256",
    "EXPECTED_TRAINABLE_PARAMETER_COUNT",
    "FRESH_INITIALIZATION_SOURCE",
    "HIDDEN_SIZE",
    "LORA_ALPHA",
    "LORA_BIAS",
    "LORA_DROPOUT",
    "LORA_RANK",
    "MAIN_RUN_ROLE",
    "METHOD_NAME",
    "MIDDLE_SELF_BLOCKS_INCLUSIVE",
    "RECEIPT_MANIFEST_SCHEMA",
    "SCOPE_NAME",
    "SourceKVRouteScopeError",
    "TRANSFORMER_BLOCK_COUNT",
    "build_receipt_manifest",
    "canonical_adapter_state_keys",
    "canonical_json_bytes",
    "canonical_target_modules",
    "expected_adapter_shapes",
    "fresh_initialization_declaration",
    "object_sha256",
    "validate_adapter_state",
    "validate_fresh_initialization",
    "validate_lora_hyperparameters",
    "validate_runtime_target_modules",
    "validate_target_module_names",
]
