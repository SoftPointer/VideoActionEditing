#!/usr/bin/env python3
"""Authenticated orchestration core for the two-cell GRAFT Phase-A pilot.

This module owns only the short-training state machine.  It consumes the
path-free two-update/two-confirmation routing minted by the pinned A-lite
consumer, accepts an exact v2 native closure for each fixed update cell, and
owns gradient synchronization, clipping, AdamW updates, and tensor-field
confirmation accounting.

It deliberately does *not* load a model, decode or sample media, construct a
native closure from raw inputs, save a checkpoint, or publish an artifact.
Consequently it cannot prove the source-bytes-to-latent binding, useful action
editing, decoded quality, or checkpoint correctness.  Those facts remain the
responsibility of a later hash-pinned GPU runner.  The public orchestration API
has no proposal-bank, selector, free-form conditioning, generated endpoint,
or auxiliary-video input.

The fixed execution is WORLD8 = DP2 x SP4:

* update cell 1 is exact40 index 29 and must be v2 ``bootstrap``;
* update cell 2 is exact40 index 38 and must be v2 ``post_bootstrap``;
* absent local gradients are materialized as true zeros, then averaged by
  SP4 SUM/4 followed by orthogonal DP2 SUM/2;
* AdamW is fixed at lr=1e-3, weight_decay=0, with global norm clipping at 1;
* each DP arm owns one update row and one disjoint confirmation row;
* confirmation applies noncompensating no-op FM and action-delta gates at
  both indices 29 and 38 under ``torch.no_grad()``.

Receipts are diagnostic only.  Every action, quality, training, checkpoint,
publication, and scientific-success authority remains false.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

import graft_a_lite_source_release_consumer_v1 as source_consumer
import graft_phase_a_native_training_closure_v2 as native_v2


SCHEMA_VERSION = "bernini-graft-phase-a-a-lite-short-training-v1"
FAILURE_SCHEMA_VERSION = (
    "bernini-graft-phase-a-a-lite-short-training-failure-v1"
)
BACKEND_SCHEMA_VERSION = "bernini-graft-world8-dp2-sp4-backend-v1"
TEST_ROUTING_SCHEMA_VERSION = "bernini-graft-a-lite-test-routing-v1"
PLAN_SCHEMA_VERSION = "bernini-graft-phase-a-short-cell-plan-v1"

WORLD_SIZE = 8
DP_SIZE = 2
SP_SIZE = 4
UPDATE_SCHEDULE_INDICES = (29, 38)
UPDATE_REGIMES = ("bootstrap", "post_bootstrap")
OPTIMIZER_LEARNING_RATE = 1.0e-3
OPTIMIZER_WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
OPTIMIZER_BETAS = (0.9, 0.999)
OPTIMIZER_EPS = 1.0e-8

PINNED_CONSUMER_COMMIT = "6bae78c40b531310ad89a9b47418a8ff4a81ce05"
PINNED_CONSUMER_SOURCE_SHA256 = (
    "13ecb082ab3cff6f809b056c35715123be302a5c8d82a6760a7367861920ee75"
)
PINNED_NATIVE_V2_SOURCE_SHA256 = (
    # Single update point.  These stable bytes include the Python 3.12 AUH
    # runtime contract; launcher authority still requires independent audit.
    "bf6a1d438183de5aa0460e729a39382e4597b3e43a4b9f1b3cdff5457439f20f"
)
# Source-independent runtime-surface record.  This is updated together with
# the single v2 source pin after postfix audit.
PINNED_CRITICAL_RUNTIME_SURFACE_SHA256 = (
    "1bdf3747b598c577d73132b8bc278d8a3f911da35d19236d12d706b4d476cac1"
)
# Source-stable complete execution contract for this trainer module itself.
# It intentionally excludes this digest value from its own record.
PINNED_TRAINER_EXECUTION_RUNTIME_SHA256 = "f35f621938e7a6b8bd3b9b5a6b0fb782f5ebf483939f585f561e3908f993af3c"

CONFIRMATION_SCHEDULE_INDICES = UPDATE_SCHEDULE_INDICES
CONFIRMATION_FIELD_ROLES = (
    "source_noop_target_velocity",
    "correct_atlas_noop_velocity",
    "wrong_atlas_noop_velocity",
    "dropped_atlas_noop_velocity",
    "correct_atlas_action_velocity",
    "dropped_atlas_action_velocity",
)
MIN_CONFIRMATION_RELATIVE_GAIN = 1.0e-4
MIN_ACTION_DELTA_CORRECT_DROP_NORM_RATIO = 0.5
MIN_ACTION_DELTA_COSINE = 0.0
FLOAT64_TINY = 2.2250738585072014e-308
RUNNER_ADAPTER_OFF_PARITY_INDICES = (0, 25)
RAW_DIGEST_CHUNK_SIZE_BYTES = 64 * 1024 * 1024
_PINNED_CTYPES_STRING_AT_SOURCE_SHA256_BY_PYTHON_MINOR = MappingProxyType(
    {
        (3, 8): (
            "73d993d6188f0c66dfd3e0f8d34561000a607e37db6533726ae1ee5bb40cd3a7"
        ),
        (3, 10): (
            "73d993d6188f0c66dfd3e0f8d34561000a607e37db6533726ae1ee5bb40cd3a7"
        ),
        (3, 12): (
            "30e5adafeec1e9a07cd4848465995c49f0124ddedd31ab3ee627b218086839eb"
        ),
    }
)
_PINNED_ADAMW_STEP_RUNTIME_BY_TORCH_VERSION = MappingProxyType(
    {
        "2.2.1": (
            "torch.optim.adamw",
            "AdamW.step",
            ("optim", "adamw.py"),
            "0a974170c9297c6d1d3a65d472e245012cc3dca7342b78cd373a2712588daaad",
        ),
        "2.4.0": (
            "torch.optim.adamw",
            "AdamW.step",
            ("optim", "adamw.py"),
            "bad2ab4efb8a3ffd8e383d46dd73f3cacd4136d64bfb3b190e4549d79ed3671a",
        ),
        "2.7.1+rocm6.3": (
            "torch.optim.adam",
            "Adam.step",
            ("optim", "adam.py"),
            "a30c184cf93b5e1b06bf6abf2ef3642eb1e6c2a03992e4914f538239fd28a5f2",
        ),
    }
)

AUTHORITY_FIELDS = (
    "action_authority",
    "identity_authority",
    "cross_clip_identity_authority",
    "quality_authority",
    "training_authority",
    "checkpoint_authority",
    "publication_authority",
    "production_authority",
    "data_governance_authority",
    "data_license_authority",
    "scientific_success_claimed",
    "semantic_action_editing_success_claimed",
)

_FORBIDDEN_PUBLIC_INPUT_FRAGMENTS = (
    "proposal",
    "asga",
    "selector",
    "retelling",
    "caption",
    "target_video",
    "generated_video",
)

_PINNED_BINDING_TYPE = native_v2.AuthenticatedNativeBindings
_PINNED_CELL_TYPE = native_v2.PhaseANativeTrainingClosure
_PINNED_RESULT_TYPE = native_v2.NativeTrainingClosureResult
_PINNED_ROUTING_TYPE = source_consumer.TrainerRouting
_PINNED_ROW_TYPE = source_consumer.TrainerOwnedSourceRow
_PINNED_EXECUTE = native_v2.execute_phase_a_native_training_closure
_PINNED_ADAMW = torch.optim.AdamW
# Optimizer.__init__ lazily wraps ``step`` for hook dispatch in some torch
# releases.  Pin the unwrapped class implementation; each session separately
# pins the post-construction live wrapper identity.
_PINNED_ADAMW_INIT = inspect.unwrap(torch.optim.AdamW.__init__)
_PINNED_ADAMW_STEP = inspect.unwrap(torch.optim.AdamW.step)
_PINNED_ADAMW_ZERO_GRAD = inspect.unwrap(torch.optim.AdamW.zero_grad)
_PINNED_TORCH_VERSION = torch.__version__
_PINNED_TORCH_VERSION_STRING = str(torch.__version__)
_PINNED_CLIP_GRAD_NORM = torch.nn.utils.clip_grad_norm_
_PINNED_CTYPES_STRING_AT = ctypes.string_at
_PINNED_CTYPES_STRING_AT_CODE = ctypes.string_at.__code__
_PINNED_CTYPES_RAW_STRING_AT = ctypes._string_at
_PINNED_CTYPES_RAW_STRING_AT_TYPE = type(ctypes._string_at)
_PINNED_CTYPES_C_VOID_P = ctypes.c_void_p
_PINNED_CTYPES_C_INT = ctypes.c_int
_PINNED_CTYPES_PY_OBJECT = ctypes.py_object
_PINNED_PYTHON_RUNTIME_MINOR = (
    sys.version_info.major,
    sys.version_info.minor,
)


class GraftPhaseAShortTrainingError(RuntimeError):
    """Raised before a non-conforming update or confirmation is accepted."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise GraftPhaseAShortTrainingError(
            "value is not canonical finite ASCII JSON"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GraftPhaseAShortTrainingError(f"{label} must be canonical SHA256")
    return value


def _seal(value: Mapping[str, Any]) -> Mapping[str, Any]:
    plain = dict(value)
    if "digest" in plain:
        raise GraftPhaseAShortTrainingError("sealed payload already contains digest")
    plain["digest"] = _object_sha256(plain)
    return MappingProxyType(plain)


def _validated_sealed_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GraftPhaseAShortTrainingError(f"{label} must be a mapping")
    plain = dict(value)
    digest = plain.pop("digest", None)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
        or _object_sha256(plain) != digest
    ):
        raise GraftPhaseAShortTrainingError(f"{label} digest differs")
    return {**plain, "digest": digest}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except (OSError, TypeError) as error:
        raise GraftPhaseAShortTrainingError(
            "pinned dependency source cannot be read"
        ) from error
    return digest.hexdigest()


def _runtime_callable_contract(value: Any) -> Mapping[str, Any]:
    function = value
    if isinstance(value, (classmethod, staticmethod)):
        function = value.__func__
    elif isinstance(value, property):
        function = value.fget
    if not inspect.isfunction(function):
        return {
            "descriptor_type": (
                f"{type(value).__module__}.{type(value).__qualname__}"
            )
        }
    try:
        source = inspect.getsource(function).encode("utf-8")
        source_path = Path(inspect.getsourcefile(function) or "").resolve()
        code_path = Path(function.__code__.co_filename).resolve()
        signature = str(inspect.signature(function))
    except (OSError, TypeError, ValueError) as error:
        if (
            inspect.isfunction(function)
            and (
                function.__code__.co_filename == "<string>"
                or Path(function.__code__.co_filename).name == "dataclasses.py"
            )
        ):
            return {
                "descriptor_type": (
                    f"{type(value).__module__}.{type(value).__qualname__}"
                ),
                "function_module": function.__module__,
                "function_qualname": function.__qualname__,
                "generated_dataclass_function": True,
                "code_file": Path(function.__code__.co_filename).name,
                "signature": str(inspect.signature(function)),
            }
        raise GraftPhaseAShortTrainingError(
            "critical runtime callable provenance cannot be inspected"
        ) from error
    return {
        "descriptor_type": f"{type(value).__module__}.{type(value).__qualname__}",
        "function_module": function.__module__,
        "function_qualname": function.__qualname__,
        "source_file": source_path.name,
        "code_file": code_path.name,
        "firstlineno": function.__code__.co_firstlineno,
        "signature": signature,
        "source_sha256": hashlib.sha256(source).hexdigest(),
    }


def _runtime_object_contract(value: Any) -> Mapping[str, Any]:
    if inspect.isfunction(value):
        return _runtime_callable_contract(value)
    if not inspect.isclass(value):
        raise GraftPhaseAShortTrainingError(
            "critical runtime object must be a class or function"
        )
    try:
        source_lines, source_start = inspect.getsourcelines(value)
        class_offset = next(
            index
            for index, line in enumerate(source_lines)
            if line.lstrip().startswith("class ")
        )
        # Python 3.8's inspect omits class decorators while 3.10+ includes
        # them.  Hash from the actual ``class`` line in every runtime.
        class_source = "".join(source_lines[class_offset:]).encode("utf-8")
        class_firstlineno = source_start + class_offset
        source_path = Path(inspect.getsourcefile(value) or "").resolve()
    except (OSError, StopIteration, TypeError, ValueError) as error:
        raise GraftPhaseAShortTrainingError(
            "critical runtime class provenance cannot be inspected"
        ) from error
    inventory = {}
    # Dataclass-generated dunder methods/metadata differ across the pinned
    # Python 3.8/3.10/3.12 runtimes (for example ``__match_args__`` is absent
    # on 3.8).  Retain every ordinary member plus dunder descriptors whose
    # implementation is explicitly defined in this class's source file.  This
    # removes generated runtime noise while still pinning execution-relevant
    # methods such as v2 ``__getattribute__``.
    def explicitly_source_defined_dunder(name: str) -> bool:
        if not name.startswith("__"):
            return True
        descriptor = inspect.getattr_static(value, name)
        function = descriptor
        if isinstance(descriptor, (classmethod, staticmethod)):
            function = descriptor.__func__
        elif isinstance(descriptor, property):
            function = descriptor.fget
        return inspect.isfunction(function) and (
            Path(function.__code__.co_filename).resolve() == source_path
        )

    for name in sorted(
        member
        for member in vars(value)
        if explicitly_source_defined_dunder(member)
    ):
        descriptor = inspect.getattr_static(value, name)
        if (
            inspect.isfunction(descriptor)
            or isinstance(descriptor, (classmethod, staticmethod, property))
        ):
            inventory[name] = _runtime_callable_contract(descriptor)
        else:
            inventory[name] = {
                "descriptor_type": (
                    f"{type(descriptor).__module__}."
                    f"{type(descriptor).__qualname__}"
                )
            }
    dataclass_fields = getattr(value, "__dataclass_fields__", None)
    return {
        "class_module": value.__module__,
        "class_qualname": value.__qualname__,
        "source_file": source_path.name,
        "class_firstlineno": class_firstlineno,
        "source_sha256": hashlib.sha256(class_source).hexdigest(),
        "mro": [
            f"{base.__module__}.{base.__qualname__}" for base in value.__mro__
        ],
        "inventory": inventory,
        "dataclass_field_names": (
            [] if dataclass_fields is None else list(dataclass_fields)
        ),
    }


def _expected_adamw_step_runtime(
    torch_version: str,
) -> tuple[str, str, tuple[str, str], str]:
    runtime_pins = _PINNED_ADAMW_STEP_RUNTIME_BY_TORCH_VERSION
    if (
        type(runtime_pins) is not MappingProxyType
        or type(torch_version) is not str
    ):
        raise GraftPhaseAShortTrainingError(
            "AdamW.step torch version contract differs"
        )
    expected = runtime_pins.get(torch_version)
    if expected is None:
        raise GraftPhaseAShortTrainingError(
            "AdamW.step torch version is unsupported"
        )
    if (
        type(expected) is not tuple
        or len(expected) != 4
        or type(expected[0]) is not str
        or type(expected[1]) is not str
        or type(expected[2]) is not tuple
        or len(expected[2]) != 2
        or any(type(part) is not str for part in expected[2])
        or type(expected[3]) is not str
        or len(expected[3]) != 64
        or expected[3] != expected[3].lower()
        or any(
            character not in "0123456789abcdef"
            for character in expected[3]
        )
    ):
        raise GraftPhaseAShortTrainingError(
            "AdamW.step runtime pin differs"
        )
    return expected


def _assert_pinned_adamw_step_runtime_contract(
    *,
    torch_version: str,
    observed_module: str,
    observed_qualname: str,
    observed_source_path_suffix: tuple[str, str],
    observed_source_sha256: str,
) -> None:
    expected = _expected_adamw_step_runtime(torch_version)
    observed = (
        observed_module,
        observed_qualname,
        observed_source_path_suffix,
        observed_source_sha256,
    )
    if (
        type(observed_module) is not str
        or type(observed_qualname) is not str
        or type(observed_source_path_suffix) is not tuple
        or len(observed_source_path_suffix) != 2
        or any(type(part) is not str for part in observed_source_path_suffix)
        or type(observed_source_sha256) is not str
        or observed != expected
    ):
        raise GraftPhaseAShortTrainingError(
            "live AdamW.step runtime provenance differs"
        )


def _assert_pinned_adamw_step_runtime(function: Any) -> None:
    try:
        live_torch_version = torch.__version__
        source = inspect.getsource(function).encode("utf-8")
        source_path = Path(inspect.getsourcefile(function) or "").resolve()
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise GraftPhaseAShortTrainingError(
            "AdamW.step runtime provenance cannot be inspected"
        ) from error
    if (
        not isinstance(live_torch_version, str)
        or live_torch_version is not _PINNED_TORCH_VERSION
        or str(live_torch_version) != _PINNED_TORCH_VERSION_STRING
        or function is not _PINNED_ADAMW_STEP
        or not inspect.isfunction(function)
    ):
        raise GraftPhaseAShortTrainingError(
            "live AdamW.step identity or torch version differs"
        )
    _assert_pinned_adamw_step_runtime_contract(
        torch_version=_PINNED_TORCH_VERSION_STRING,
        observed_module=function.__module__,
        observed_qualname=function.__qualname__,
        observed_source_path_suffix=tuple(source_path.parts[-2:]),
        observed_source_sha256=hashlib.sha256(source).hexdigest(),
    )


def _expected_ctypes_string_at_source_sha256(
    python_minor: tuple[int, int],
) -> str:
    source_pins = _PINNED_CTYPES_STRING_AT_SOURCE_SHA256_BY_PYTHON_MINOR
    if (
        type(source_pins) is not MappingProxyType
        or type(python_minor) is not tuple
        or len(python_minor) != 2
        or any(type(value) is not int for value in python_minor)
    ):
        raise GraftPhaseAShortTrainingError(
            "ctypes.string_at Python minor contract differs"
        )
    expected = source_pins.get(python_minor)
    if expected is None:
        raise GraftPhaseAShortTrainingError(
            "ctypes.string_at Python minor is unsupported"
        )
    if (
        type(expected) is not str
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise GraftPhaseAShortTrainingError(
            "ctypes.string_at source pin differs"
        )
    return expected


def _assert_pinned_ctypes_string_at_source_sha256(
    *,
    python_minor: tuple[int, int],
    observed_sha256: str,
) -> None:
    expected = _expected_ctypes_string_at_source_sha256(python_minor)
    if type(observed_sha256) is not str or observed_sha256 != expected:
        raise GraftPhaseAShortTrainingError(
            "live ctypes.string_at source differs"
        )


def _assert_pinned_ctypes_runtime() -> None:
    try:
        live_python_minor = (
            sys.version_info.major,
            sys.version_info.minor,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise GraftPhaseAShortTrainingError(
            "ctypes.string_at Python minor cannot be inspected"
        ) from error
    if (
        type(live_python_minor[0]) is not int
        or type(live_python_minor[1]) is not int
        or live_python_minor != _PINNED_PYTHON_RUNTIME_MINOR
    ):
        raise GraftPhaseAShortTrainingError(
            "live ctypes.string_at Python minor differs"
        )
    try:
        source = inspect.getsource(_PINNED_CTYPES_STRING_AT).encode("utf-8")
        source_path = Path(
            inspect.getsourcefile(_PINNED_CTYPES_STRING_AT) or ""
        ).resolve()
        signature = str(inspect.signature(_PINNED_CTYPES_STRING_AT))
        raw_argtypes = tuple(_PINNED_CTYPES_RAW_STRING_AT.argtypes)
        raw_flags = int(_PINNED_CTYPES_RAW_STRING_AT._flags_)
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise GraftPhaseAShortTrainingError(
            "ctypes.string_at runtime provenance cannot be inspected"
        ) from error
    _assert_pinned_ctypes_string_at_source_sha256(
        python_minor=live_python_minor,
        observed_sha256=hashlib.sha256(source).hexdigest(),
    )
    if (
        ctypes.string_at is not _PINNED_CTYPES_STRING_AT
        or getattr(ctypes, "_string_at", None)
        is not _PINNED_CTYPES_RAW_STRING_AT
        or not inspect.isfunction(_PINNED_CTYPES_STRING_AT)
        or _PINNED_CTYPES_STRING_AT.__code__
        is not _PINNED_CTYPES_STRING_AT_CODE
        or _PINNED_CTYPES_STRING_AT.__module__ != "ctypes"
        or _PINNED_CTYPES_STRING_AT.__qualname__ != "string_at"
        or _PINNED_CTYPES_STRING_AT.__globals__ is not vars(ctypes)
        or _PINNED_CTYPES_STRING_AT.__globals__.get("_string_at")
        is not _PINNED_CTYPES_RAW_STRING_AT
        or source_path.name != "__init__.py"
        or source_path.parent.name != "ctypes"
        or signature != "(ptr, size=-1)"
        or type(_PINNED_CTYPES_RAW_STRING_AT)
        is not _PINNED_CTYPES_RAW_STRING_AT_TYPE
        or _PINNED_CTYPES_RAW_STRING_AT.__module__ != "ctypes"
        or type(_PINNED_CTYPES_RAW_STRING_AT).__module__ != "ctypes"
        or type(_PINNED_CTYPES_RAW_STRING_AT).__qualname__
        != "PYFUNCTYPE.<locals>.CFunctionType"
        or raw_argtypes
        != (_PINNED_CTYPES_C_VOID_P, _PINNED_CTYPES_C_INT)
        or _PINNED_CTYPES_RAW_STRING_AT.restype
        is not _PINNED_CTYPES_PY_OBJECT
        or raw_flags != 5
        or ctypes.c_void_p is not _PINNED_CTYPES_C_VOID_P
        or ctypes.c_int is not _PINNED_CTYPES_C_INT
        or ctypes.py_object is not _PINNED_CTYPES_PY_OBJECT
    ):
        raise GraftPhaseAShortTrainingError(
            "live ctypes.string_at runtime differs"
        )


_PINNED_ASSERT_CTYPES_RUNTIME = _assert_pinned_ctypes_runtime


def _critical_runtime_surface_sha256() -> str:
    record = {
        "consumer": {
            "TrainerRouting": _runtime_object_contract(
                source_consumer.TrainerRouting
            ),
            "TrainerOwnedSourceRow": _runtime_object_contract(
                source_consumer.TrainerOwnedSourceRow
            ),
            "AuthorityBoundary": _runtime_object_contract(
                source_consumer.AuthorityBoundary
            ),
            "OwnedSourceMedia": _runtime_object_contract(
                source_consumer.OwnedSourceMedia
            ),
            "validate_for_training": _runtime_object_contract(
                source_consumer.validate_for_training
            ),
            "noop_instruction": source_consumer.NOOP_INSTRUCTION,
            "canary4": [list(row) for row in source_consumer.CANARY4],
        },
        "native_v2": {
            "AuthenticatedNativeBindings": _runtime_object_contract(
                native_v2.AuthenticatedNativeBindings
            ),
            "PhaseANativeTrainingClosure": _runtime_object_contract(
                native_v2.PhaseANativeTrainingClosure
            ),
            "NativeTrainingClosureResult": _runtime_object_contract(
                native_v2.NativeTrainingClosureResult
            ),
            "execute": _runtime_object_contract(
                native_v2.execute_phase_a_native_training_closure
            ),
            "schema": native_v2.SCHEMA_VERSION,
            "gradient_categories": list(native_v2.GRADIENT_CATEGORIES),
            "active_schedule_indices": list(
                native_v2.PHASE_A_ACTIVE_SCHEDULE_INDICES
            ),
        },
    }
    return _object_sha256(record)


def _assert_pinned_dependencies() -> None:
    _PINNED_TRAINER_EXECUTION_GATE()
    consumer_path = Path(source_consumer.__file__).resolve()
    v2_path = Path(native_v2.__file__).resolve()
    try:
        live_adamw_init = inspect.unwrap(torch.optim.AdamW.__init__)
        live_adamw_step = inspect.unwrap(torch.optim.AdamW.step)
        live_adamw_zero_grad = inspect.unwrap(torch.optim.AdamW.zero_grad)
    except (TypeError, ValueError) as error:
        raise GraftPhaseAShortTrainingError(
            "live AdamW callable chain differs"
        ) from error
    if (
        not consumer_path.is_file()
        or _file_sha256(consumer_path) != PINNED_CONSUMER_SOURCE_SHA256
        or not v2_path.is_file()
        or _file_sha256(v2_path) != PINNED_NATIVE_V2_SOURCE_SHA256
    ):
        raise GraftPhaseAShortTrainingError("pinned dependency source differs")
    if (
        native_v2.AuthenticatedNativeBindings is not _PINNED_BINDING_TYPE
        or native_v2.PhaseANativeTrainingClosure is not _PINNED_CELL_TYPE
        or native_v2.PhaseANativeTrainingClosureV2 is not _PINNED_CELL_TYPE
        or native_v2.NativeTrainingClosureResult is not _PINNED_RESULT_TYPE
        or native_v2.execute_phase_a_native_training_closure is not _PINNED_EXECUTE
        or source_consumer.TrainerRouting is not _PINNED_ROUTING_TYPE
        or source_consumer.TrainerOwnedSourceRow is not _PINNED_ROW_TYPE
        or torch.optim.AdamW is not _PINNED_ADAMW
        or live_adamw_init is not _PINNED_ADAMW_INIT
        or live_adamw_step is not _PINNED_ADAMW_STEP
        or live_adamw_zero_grad is not _PINNED_ADAMW_ZERO_GRAD
        or torch.nn.utils.clip_grad_norm_ is not _PINNED_CLIP_GRAD_NORM
        or _PINNED_CELL_TYPE.__module__ != native_v2.__name__
        or _PINNED_EXECUTE.__module__ != native_v2.__name__
        or _PINNED_ADAMW.__module__ != "torch.optim.adamw"
        or _PINNED_ADAMW.__name__ != "AdamW"
        or _PINNED_ADAMW_INIT.__module__ != "torch.optim.adamw"
        or _PINNED_ADAMW_INIT.__qualname__ != "AdamW.__init__"
        or _PINNED_ADAMW_ZERO_GRAD.__module__ != "torch.optim.optimizer"
        or _PINNED_ADAMW_ZERO_GRAD.__qualname__ != "Optimizer.zero_grad"
        or _PINNED_CLIP_GRAD_NORM.__module__ != "torch.nn.utils.clip_grad"
        or _PINNED_CLIP_GRAD_NORM.__name__ != "clip_grad_norm_"
    ):
        raise GraftPhaseAShortTrainingError("live dependency namespace differs")
    _assert_pinned_adamw_step_runtime(live_adamw_step)
    try:
        adamw_source = Path(inspect.getsourcefile(_PINNED_ADAMW) or "").resolve()
        adamw_init_source = Path(
            inspect.getsourcefile(_PINNED_ADAMW_INIT) or ""
        ).resolve()
        adamw_zero_grad_source = Path(
            inspect.getsourcefile(_PINNED_ADAMW_ZERO_GRAD) or ""
        ).resolve()
        clip_source = Path(
            inspect.getsourcefile(_PINNED_CLIP_GRAD_NORM) or ""
        ).resolve()
    except (OSError, TypeError) as error:
        raise GraftPhaseAShortTrainingError(
            "optimizer runtime provenance cannot be resolved"
        ) from error
    if (
        adamw_source.name != "adamw.py"
        or tuple(adamw_source.parts[-2:]) != ("optim", "adamw.py")
        or tuple(adamw_init_source.parts[-2:]) != ("optim", "adamw.py")
        or tuple(adamw_zero_grad_source.parts[-2:])
        != ("optim", "optimizer.py")
        or clip_source.name != "clip_grad.py"
        or tuple(clip_source.parts[-3:]) != ("nn", "utils", "clip_grad.py")
    ):
        raise GraftPhaseAShortTrainingError(
            "optimizer runtime provenance differs"
        )
    execute_fields = tuple(inspect.signature(_PINNED_EXECUTE).parameters)
    if execute_fields != tuple(native_v2.v1.PHASE_A_EXECUTE_API_FIELDS):
        raise GraftPhaseAShortTrainingError("native v2 execute API differs")
    lowered = tuple(name.lower() for name in execute_fields)
    if any(
        fragment in name
        for fragment in _FORBIDDEN_PUBLIC_INPUT_FRAGMENTS
        for name in lowered
    ):
        raise GraftPhaseAShortTrainingError(
            "native v2 execute API acquired a forbidden orchestration input"
        )
    if (
        tuple(native_v2.GRADIENT_CATEGORIES)
        != (
            "atlas_encoder",
            "query_projection",
            "key_projection",
            "value_projection",
            "output_projection",
        )
        or any(index not in native_v2.PHASE_A_ACTIVE_SCHEDULE_INDICES for index in UPDATE_SCHEDULE_INDICES)
        or tuple(source_consumer.CANARY4)
        != (
            ("7b88a1ca1f804f41", "optimizer_train", True, False),
            ("a35b590961d24694", "optimizer_train", True, False),
            ("841b5e0080a1441d", "optimizer_confirmation", False, True),
            ("a66e6818e4144928", "optimizer_confirmation", False, True),
        )
    ):
        raise GraftPhaseAShortTrainingError("fixed Phase-A dependency contract differs")
    if (
        _critical_runtime_surface_sha256()
        != PINNED_CRITICAL_RUNTIME_SURFACE_SHA256
    ):
        raise GraftPhaseAShortTrainingError(
            "critical consumer/native runtime surface differs"
        )


def _tensor_bytes_sha256(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor):
        raise GraftPhaseAShortTrainingError("digest input must be a tensor")
    owned = value.detach().cpu().contiguous().clone()
    expected = int(owned.numel()) * int(owned.element_size())
    storage = owned.untyped_storage()
    if int(storage.nbytes()) != expected:
        raise GraftPhaseAShortTrainingError(
            "tensor digest storage exceeds its logical tensor"
        )
    pointer = int(storage.data_ptr())
    if expected == 0:
        return hashlib.sha256(b"").hexdigest()
    if pointer <= 0:
        raise GraftPhaseAShortTrainingError(
            "non-empty tensor digest storage has a null pointer"
        )
    if (
        type(RAW_DIGEST_CHUNK_SIZE_BYTES) is not int
        or not 0 < RAW_DIGEST_CHUNK_SIZE_BYTES < 2**31
    ):
        raise GraftPhaseAShortTrainingError(
            "tensor digest ctypes chunk size differs"
        )
    digest = hashlib.sha256()
    observed = 0
    while observed < expected:
        chunk_size = min(
            RAW_DIGEST_CHUNK_SIZE_BYTES,
            expected - observed,
        )
        raw = _PINNED_CTYPES_STRING_AT(pointer + observed, chunk_size)
        if type(raw) is not bytes or len(raw) != chunk_size:
            raise GraftPhaseAShortTrainingError(
                "tensor digest byte length differs"
            )
        digest.update(raw)
        observed += len(raw)
    if observed != expected:
        raise GraftPhaseAShortTrainingError(
            "tensor digest byte length differs"
        )
    return digest.hexdigest()


_PINNED_TENSOR_BYTES_SHA256 = _tensor_bytes_sha256


def _named_tensor_digest(named: Sequence[tuple[str, torch.Tensor]]) -> str:
    _PINNED_TRAINER_EXECUTION_GATE()
    rows = []
    for name, tensor in named:
        rows.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "tensor_sha256": _PINNED_TENSOR_BYTES_SHA256(tensor),
            }
        )
    return _object_sha256(rows)


def _gradient_category(name: str) -> str:
    if name.startswith("atlas_encoder."):
        return "atlas_encoder"
    for projection in ("query", "key", "value", "output"):
        if name.endswith(f".identity_rebinder.{projection}.weight"):
            return f"{projection}_projection"
    raise GraftPhaseAShortTrainingError(
        f"trainable gradient category is unknown: {name}"
    )


def _finite_scalar(value: Any, *, label: str) -> float:
    if type(value) is not torch.Tensor or value.numel() != 1:
        raise GraftPhaseAShortTrainingError(f"{label} must be one exact tensor scalar")
    if (
        not value.is_floating_point()
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).item())
    ):
        raise GraftPhaseAShortTrainingError(
            f"{label} must be finite, floating, and detached"
        )
    result = float(value.item())
    if not math.isfinite(result):
        raise GraftPhaseAShortTrainingError(f"{label} is non-finite")
    return result


def _confirmation_field_metrics(
    *,
    schedule_index: int,
    source_noop_target_velocity: torch.Tensor,
    correct_atlas_noop_velocity: torch.Tensor,
    wrong_atlas_noop_velocity: torch.Tensor,
    dropped_atlas_noop_velocity: torch.Tensor,
    correct_atlas_action_velocity: torch.Tensor,
    dropped_atlas_action_velocity: torch.Tensor,
) -> Mapping[str, Any]:
    """Compute the fixed no-op FM and action-delta confirmation gates."""

    _PINNED_TRAINER_EXECUTION_GATE()

    fields = {
        "source_noop_target_velocity": source_noop_target_velocity,
        "correct_atlas_noop_velocity": correct_atlas_noop_velocity,
        "wrong_atlas_noop_velocity": wrong_atlas_noop_velocity,
        "dropped_atlas_noop_velocity": dropped_atlas_noop_velocity,
        "correct_atlas_action_velocity": correct_atlas_action_velocity,
        "dropped_atlas_action_velocity": dropped_atlas_action_velocity,
    }
    if (
        type(schedule_index) is not int
        or schedule_index not in CONFIRMATION_SCHEDULE_INDICES
        or tuple(fields) != CONFIRMATION_FIELD_ROLES
    ):
        raise GraftPhaseAShortTrainingError("confirmation coordinate differs")
    reference = source_noop_target_velocity
    if (
        type(reference) is not torch.Tensor
        or reference.dtype != torch.float32
        or reference.numel() < 2
    ):
        raise GraftPhaseAShortTrainingError(
            "confirmation fields must be nontrivial exact FP32 tensors"
        )
    identities = []
    for role, value in fields.items():
        if (
            type(value) is not torch.Tensor
            or value.dtype != torch.float32
            or value.shape != reference.shape
            or value.device != reference.device
            or value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())
        ):
            raise GraftPhaseAShortTrainingError(
                f"confirmation field contract differs: {role}"
            )
        identities.append(
            (
                value.device.type,
                value.device.index,
                int(value.untyped_storage().data_ptr()),
            )
        )
    if len(set(identities)) != len(fields):
        raise GraftPhaseAShortTrainingError(
            "confirmation fields must have pairwise-distinct storage"
        )

    def mse(value: torch.Tensor) -> float:
        result = float(
            (value.double() - source_noop_target_velocity.double())
            .square()
            .mean()
            .item()
        )
        if not math.isfinite(result) or result < 0.0:
            raise GraftPhaseAShortTrainingError(
                "confirmation no-op FM loss is non-finite or negative"
            )
        return result

    correct_loss = mse(correct_atlas_noop_velocity)
    wrong_loss = mse(wrong_atlas_noop_velocity)
    dropped_loss = mse(dropped_atlas_noop_velocity)
    wrong_gain = (wrong_loss - correct_loss) / max(wrong_loss, FLOAT64_TINY)
    dropped_gain = (dropped_loss - correct_loss) / max(
        dropped_loss, FLOAT64_TINY
    )

    correct_delta = correct_atlas_action_velocity.double() - (
        correct_atlas_noop_velocity.double()
    )
    dropped_delta = dropped_atlas_action_velocity.double() - (
        dropped_atlas_noop_velocity.double()
    )
    correct_norm = float(correct_delta.square().sum().sqrt().item())
    dropped_norm = float(dropped_delta.square().sum().sqrt().item())
    if (
        not math.isfinite(correct_norm)
        or not math.isfinite(dropped_norm)
        or correct_norm <= 0.0
        or dropped_norm <= 0.0
    ):
        raise GraftPhaseAShortTrainingError(
            "both confirmation action deltas must have finite positive norm"
        )
    norm_ratio = correct_norm / max(dropped_norm, FLOAT64_TINY)
    cosine = float(
        (correct_delta * dropped_delta).sum().item()
        / (correct_norm * dropped_norm)
    )
    if not all(
        math.isfinite(value)
        for value in (wrong_gain, dropped_gain, norm_ratio, cosine)
    ):
        raise GraftPhaseAShortTrainingError(
            "confirmation derived metric is non-finite"
        )
    gates = {
        "correct_vs_wrong_noop_relative_gain": (
            wrong_gain >= MIN_CONFIRMATION_RELATIVE_GAIN
        ),
        "correct_vs_drop_noop_relative_gain": (
            dropped_gain >= MIN_CONFIRMATION_RELATIVE_GAIN
        ),
        "action_delta_correct_drop_norm_ratio": (
            norm_ratio >= MIN_ACTION_DELTA_CORRECT_DROP_NORM_RATIO
        ),
        "action_delta_correct_drop_cosine": cosine >= MIN_ACTION_DELTA_COSINE,
    }
    return _seal(
        {
            "schema_version": "bernini-graft-phase-a-confirmation-metrics-v1",
            "schedule_index": schedule_index,
            "field_roles": list(CONFIRMATION_FIELD_ROLES),
            "field_shape": list(reference.shape),
            "field_dtype": "torch.float32",
            "field_device_type": reference.device.type,
            "field_tensor_sha256": {
                role: _tensor_bytes_sha256(value) for role, value in fields.items()
            },
            "noop_fm_loss_float64_hex": {
                "correct_atlas": correct_loss.hex(),
                "wrong_atlas": wrong_loss.hex(),
                "dropped_atlas": dropped_loss.hex(),
            },
            "relative_gain_formula": (
                "(L_control-L_correct)/max(L_control,float64_tiny)"
            ),
            "relative_gain_float64_hex": {
                "correct_vs_wrong": wrong_gain.hex(),
                "correct_vs_drop": dropped_gain.hex(),
            },
            "minimum_relative_gain_float64_hex": (
                MIN_CONFIRMATION_RELATIVE_GAIN.hex()
            ),
            "action_delta_formula": "v_action-v_noop",
            "action_delta_norm_float64_hex": {
                "correct_atlas": correct_norm.hex(),
                "dropped_atlas": dropped_norm.hex(),
            },
            "action_delta_correct_drop_norm_ratio_formula": (
                "norm(delta_correct)/max(norm(delta_drop),float64_tiny)"
            ),
            "action_delta_correct_drop_norm_ratio_float64_hex": (
                norm_ratio.hex()
            ),
            "minimum_action_delta_correct_drop_norm_ratio_float64_hex": (
                MIN_ACTION_DELTA_CORRECT_DROP_NORM_RATIO.hex()
            ),
            "action_delta_correct_drop_cosine_float64_hex": cosine.hex(),
            "minimum_action_delta_cosine_float64_hex": (
                MIN_ACTION_DELTA_COSINE.hex()
            ),
            "float64_tiny_hex": FLOAT64_TINY.hex(),
            "noncompensating_gates": gates,
            "noncompensating_all_pass": all(gates.values()),
            "metrics_computed_from_six_detached_fields_by_this_core": True,
            "field_origin_same_noise_state_coordinate_verified_by_this_core": False,
            **_false_authority(),
        }
    )


def _false_authority() -> dict[str, bool]:
    return {name: False for name in AUTHORITY_FIELDS}


@dataclass(frozen=True)
class _TestSourceRow:
    iid: str
    split: str
    optimizer_update_allowed: bool
    optimizer_confirmation_only: bool
    source_sha256: str
    source_bytes: bytes = field(repr=False, compare=False)
    noop_instruction: str = source_consumer.NOOP_INSTRUCTION


@dataclass(frozen=True)
class _TestRouting:
    update_rows: tuple[_TestSourceRow, _TestSourceRow]
    confirmation_rows: tuple[_TestSourceRow, _TestSourceRow]
    source_release_result_digest: str
    pinset_digest: str
    routing_digest: str
    authority: Mapping[str, bool]
    test_only: bool = True


def authenticate_cpu_test_routing(
    *, test_name: str = "cpu_fake:phase_a_short_routing"
) -> _TestRouting:
    """Mint a path-free four-row routing usable only with other CPU fakes."""

    _PINNED_TRAINER_EXECUTION_GATE()

    if (
        not isinstance(test_name, str)
        or not test_name.startswith("cpu_fake:")
        or len(test_name) > 160
    ):
        raise GraftPhaseAShortTrainingError(
            "CPU routing name must use the cpu_fake namespace"
        )
    rows = []
    for index, (iid, split, update, confirmation) in enumerate(
        source_consumer.CANARY4
    ):
        raw = f"{test_name}\0{index}\0{iid}".encode("ascii")
        rows.append(
            _TestSourceRow(
                iid=iid,
                split=split,
                optimizer_update_allowed=update,
                optimizer_confirmation_only=confirmation,
                source_sha256=hashlib.sha256(raw).hexdigest(),
                source_bytes=raw,
            )
        )
    release_digest = _object_sha256(
        {"schema_version": TEST_ROUTING_SCHEMA_VERSION, "test_name": test_name}
    )
    pinset_digest = _object_sha256(
        {"schema_version": TEST_ROUTING_SCHEMA_VERSION, "pinset": test_name}
    )
    record = {
        "schema_version": TEST_ROUTING_SCHEMA_VERSION,
        "source_release_result_digest": release_digest,
        "pinset_digest": pinset_digest,
        "rows": [
            {
                "iid": row.iid,
                "split": row.split,
                "optimizer_update_allowed": row.optimizer_update_allowed,
                "optimizer_confirmation_only": row.optimizer_confirmation_only,
                "source_sha256": row.source_sha256,
                "source_size_bytes": len(row.source_bytes),
            }
            for row in rows
        ],
        "path_reopen_allowed": False,
        "owned_bytes_only": True,
        "test_only": True,
        "authority": _false_authority(),
    }
    return _TestRouting(
        update_rows=(rows[0], rows[1]),
        confirmation_rows=(rows[2], rows[3]),
        source_release_result_digest=release_digest,
        pinset_digest=pinset_digest,
        routing_digest=_object_sha256(record),
        authority=MappingProxyType(_false_authority()),
    )


_BACKEND_TOKEN = object()


class AuthenticatedDP2SP4Backend:
    """Opaque collective backend with fixed WORLD8 group geometry."""

    __slots__ = (
        "_rank",
        "_test_only",
        "_kind",
        "_world_group",
        "_sp_group",
        "_dp_group",
        "_test_reduce",
        "_test_consensus",
        "_test_reduce_identity",
        "_test_consensus_identity",
        "_receipt",
        "_token",
        "_locked",
    )

    def __init__(self, *_: Any, **__: Any) -> None:
        raise GraftPhaseAShortTrainingError(
            "collective backend must be minted by an authenticator"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("AuthenticatedDP2SP4Backend is immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def _mint(
        cls,
        *,
        token: object,
        rank: int,
        test_only: bool,
        kind: str,
        world_group: Any,
        sp_group: Any,
        dp_group: Any,
        test_reduce: Optional[Callable[[torch.Tensor, str, str, int], None]],
        test_consensus: Optional[Callable[[str, str, str], str]],
        receipt: Mapping[str, Any],
    ) -> "AuthenticatedDP2SP4Backend":
        _PINNED_TRAINER_EXECUTION_GATE()
        if token is not _BACKEND_TOKEN:
            raise GraftPhaseAShortTrainingError("collective backend mint differs")
        instance = object.__new__(cls)
        object.__setattr__(instance, "_rank", rank)
        object.__setattr__(instance, "_test_only", test_only)
        object.__setattr__(instance, "_kind", kind)
        object.__setattr__(instance, "_world_group", world_group)
        object.__setattr__(instance, "_sp_group", sp_group)
        object.__setattr__(instance, "_dp_group", dp_group)
        object.__setattr__(instance, "_test_reduce", test_reduce)
        object.__setattr__(instance, "_test_consensus", test_consensus)
        object.__setattr__(instance, "_test_reduce_identity", test_reduce)
        object.__setattr__(instance, "_test_consensus_identity", test_consensus)
        object.__setattr__(instance, "_receipt", receipt)
        object.__setattr__(instance, "_token", _BACKEND_TOKEN)
        object.__setattr__(instance, "_locked", True)
        _PINNED_BACKEND_ASSERT_LIVE(instance)
        return instance

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def dp_arm(self) -> int:
        return self._rank // SP_SIZE

    @property
    def sp_rank(self) -> int:
        return self._rank % SP_SIZE

    @property
    def test_only(self) -> bool:
        return self._test_only

    def assert_live(self) -> None:
        _PINNED_TRAINER_EXECUTION_GATE()
        if (
            self._token is not _BACKEND_TOKEN
            or type(self._rank) is not int
            or not 0 <= self._rank < WORLD_SIZE
            or type(self._test_only) is not bool
            or self._kind not in {"torch_distributed", "cpu_test_fake"}
        ):
            raise GraftPhaseAShortTrainingError("collective backend changed")
        receipt = _validated_sealed_mapping(
            self._receipt, label="collective backend receipt"
        )
        if (
            receipt.get("schema_version") != BACKEND_SCHEMA_VERSION
            or receipt.get("world_size") != WORLD_SIZE
            or receipt.get("data_parallel_size") != DP_SIZE
            or receipt.get("sequence_parallel_size") != SP_SIZE
            or receipt.get("rank") != self._rank
            or receipt.get("dp_arm") != self.dp_arm
            or receipt.get("sp_rank") != self.sp_rank
            or receipt.get("test_only") is not self._test_only
        ):
            raise GraftPhaseAShortTrainingError(
                "collective backend receipt differs"
            )
        if self._kind == "cpu_test_fake":
            if (
                self._test_only is not True
                or any(
                    group is not None
                    for group in (
                        self._world_group,
                        self._sp_group,
                        self._dp_group,
                    )
                )
                or self._test_reduce is not self._test_reduce_identity
                or self._test_consensus is not self._test_consensus_identity
                or not callable(self._test_reduce)
                or not callable(self._test_consensus)
            ):
                raise GraftPhaseAShortTrainingError(
                    "CPU collective fake changed"
                )
            return
        if (
            self._test_only is not False
            or self._test_reduce is not None
            or self._test_consensus is not None
            or self._test_reduce_identity is not None
            or self._test_consensus_identity is not None
            or any(
                group is None
                for group in (
                    self._world_group,
                    self._sp_group,
                    self._dp_group,
                )
            )
        ):
            raise GraftPhaseAShortTrainingError(
                "production collective backend changed"
            )
        import torch.distributed as dist

        if (
            not dist.is_available()
            or not dist.is_initialized()
            or int(dist.get_rank(group=self._world_group)) != self._rank
            or int(dist.get_world_size(group=self._world_group)) != WORLD_SIZE
            or int(dist.get_world_size(group=self._sp_group)) != SP_SIZE
            or int(dist.get_world_size(group=self._dp_group)) != DP_SIZE
        ):
            raise GraftPhaseAShortTrainingError(
                "live production collective geometry differs"
            )

        def members(group: Any, count: int) -> tuple[int, ...]:
            gathered: list[Any] = [None] * count
            dist.all_gather_object(gathered, dist.get_rank(), group=group)
            if any(type(item) is not int for item in gathered):
                raise GraftPhaseAShortTrainingError(
                    "live process-group member differs"
                )
            return tuple(gathered)

        expected_sp = tuple(
            range(
                self.dp_arm * SP_SIZE,
                (self.dp_arm + 1) * SP_SIZE,
            )
        )
        expected_dp = (self.sp_rank, self.sp_rank + SP_SIZE)
        if (
            members(self._world_group, WORLD_SIZE) != tuple(range(WORLD_SIZE))
            or members(self._sp_group, SP_SIZE) != expected_sp
            or members(self._dp_group, DP_SIZE) != expected_dp
        ):
            raise GraftPhaseAShortTrainingError(
                "live DP2xSP4 process-group membership differs"
            )

    def reduce_gradient_(
        self,
        value: torch.Tensor,
        *,
        axis: str,
        parameter_name: str,
        update_number: int,
    ) -> None:
        _PINNED_TRAINER_EXECUTION_GATE()
        _PINNED_BACKEND_ASSERT_LIVE(self)
        if axis not in {"sp", "dp"}:
            raise GraftPhaseAShortTrainingError("collective axis differs")
        if self._kind == "cpu_test_fake":
            assert self._test_reduce is not None
            self._test_reduce(value, axis, parameter_name, update_number)
        else:
            import torch.distributed as dist

            group = self._sp_group if axis == "sp" else self._dp_group
            dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)
        if not bool(torch.isfinite(value).all().item()):
            raise GraftPhaseAShortTrainingError(
                f"{axis} collective produced a non-finite gradient"
            )

    def consensus(self, value: str, *, scope: str, label: str) -> str:
        _PINNED_TRAINER_EXECUTION_GATE()
        _PINNED_BACKEND_ASSERT_LIVE(self)
        if scope not in {"world", "sp"}:
            raise GraftPhaseAShortTrainingError("consensus scope differs")
        if self._kind == "cpu_test_fake":
            assert self._test_consensus is not None
            observed = self._test_consensus(value, scope, label)
        else:
            import torch.distributed as dist

            group = self._world_group if scope == "world" else self._sp_group
            count = WORLD_SIZE if scope == "world" else SP_SIZE
            gathered: list[Any] = [None] * count
            dist.all_gather_object(gathered, value, group=group)
            if any(item != value for item in gathered):
                raise GraftPhaseAShortTrainingError(
                    f"{label} differs across {scope}"
                )
            observed = value
        if observed != value:
            raise GraftPhaseAShortTrainingError(f"{label} consensus differs")
        return observed

    def receipt(self) -> Mapping[str, Any]:
        _PINNED_TRAINER_EXECUTION_GATE()
        _PINNED_BACKEND_ASSERT_LIVE(self)
        return self._receipt


def authenticate_torch_distributed_world8_dp2sp4(
    *, world_group: Any, sp_group: Any, dp_group: Any
) -> AuthenticatedDP2SP4Backend:
    """Bind live torch.distributed groups to exact WORLD8/DP2xSP4 geometry."""

    _PINNED_TRAINER_EXECUTION_GATE()

    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise GraftPhaseAShortTrainingError("torch.distributed is not initialized")
    rank = int(dist.get_rank(group=world_group))
    if int(dist.get_world_size(group=world_group)) != WORLD_SIZE:
        raise GraftPhaseAShortTrainingError("world group is not WORLD8")
    expected_sp = tuple(range((rank // SP_SIZE) * SP_SIZE, (rank // SP_SIZE + 1) * SP_SIZE))
    expected_dp = (rank % SP_SIZE, rank % SP_SIZE + SP_SIZE)

    def members(group: Any, count: int) -> tuple[int, ...]:
        gathered: list[Any] = [None] * count
        dist.all_gather_object(gathered, dist.get_rank(), group=group)
        if any(type(item) is not int for item in gathered):
            raise GraftPhaseAShortTrainingError("process-group member differs")
        return tuple(gathered)

    if (
        members(world_group, WORLD_SIZE) != tuple(range(WORLD_SIZE))
        or members(sp_group, SP_SIZE) != expected_sp
        or members(dp_group, DP_SIZE) != expected_dp
    ):
        raise GraftPhaseAShortTrainingError("DP2xSP4 group membership differs")

    receipt = _seal(
        {
            "schema_version": BACKEND_SCHEMA_VERSION,
            "world_size": WORLD_SIZE,
            "data_parallel_size": DP_SIZE,
            "sequence_parallel_size": SP_SIZE,
            "rank": rank,
            "dp_arm": rank // SP_SIZE,
            "sp_rank": rank % SP_SIZE,
            "sp_members": list(expected_sp),
            "dp_members": list(expected_dp),
            "gradient_collective_order": ["SP4_SUM_div_4", "DP2_SUM_div_2"],
            "test_only": False,
            "distributed_groups_live_verified": True,
            **_false_authority(),
        }
    )
    return AuthenticatedDP2SP4Backend._mint(
        token=_BACKEND_TOKEN,
        rank=rank,
        test_only=False,
        kind="torch_distributed",
        world_group=world_group,
        sp_group=sp_group,
        dp_group=dp_group,
        test_reduce=None,
        test_consensus=None,
        receipt=receipt,
    )


def authenticate_cpu_test_collectives(
    *,
    rank: int,
    gradient_reducer: Optional[
        Callable[[torch.Tensor, str, str, int], None]
    ] = None,
    digest_consensus: Optional[Callable[[str, str, str], str]] = None,
) -> AuthenticatedDP2SP4Backend:
    """Mint explicit CPU fakes; they can never authorize a production claim."""

    _PINNED_TRAINER_EXECUTION_GATE()

    if type(rank) is not int or not 0 <= rank < WORLD_SIZE:
        raise GraftPhaseAShortTrainingError("CPU fake rank differs")

    def default_reduce(
        value: torch.Tensor, axis: str, _name: str, _update_number: int
    ) -> None:
        value.mul_(float(SP_SIZE if axis == "sp" else DP_SIZE))

    def default_consensus(value: str, _scope: str, _label: str) -> str:
        return value

    reduce = default_reduce if gradient_reducer is None else gradient_reducer
    consensus = default_consensus if digest_consensus is None else digest_consensus
    if not callable(reduce) or not callable(consensus):
        raise GraftPhaseAShortTrainingError("CPU collective fake is not callable")
    receipt = _seal(
        {
            "schema_version": BACKEND_SCHEMA_VERSION,
            "world_size": WORLD_SIZE,
            "data_parallel_size": DP_SIZE,
            "sequence_parallel_size": SP_SIZE,
            "rank": rank,
            "dp_arm": rank // SP_SIZE,
            "sp_rank": rank % SP_SIZE,
            "sp_members": list(
                range((rank // SP_SIZE) * SP_SIZE, (rank // SP_SIZE + 1) * SP_SIZE)
            ),
            "dp_members": [rank % SP_SIZE, rank % SP_SIZE + SP_SIZE],
            "gradient_collective_order": ["SP4_SUM_div_4", "DP2_SUM_div_2"],
            "test_only": True,
            "distributed_groups_live_verified": False,
            **_false_authority(),
        }
    )
    return AuthenticatedDP2SP4Backend._mint(
        token=_BACKEND_TOKEN,
        rank=rank,
        test_only=True,
        kind="cpu_test_fake",
        world_group=None,
        sp_group=None,
        dp_group=None,
        test_reduce=reduce,
        test_consensus=consensus,
        receipt=receipt,
    )


_PLAN_TOKEN = object()


@dataclass(frozen=True, init=False)
class UpdateCellPlan:
    """Opaque binding of one routed update row to one exact v2 cell."""

    update_number: int
    schedule_index: int
    expected_regime: str
    dp_arm: int
    row: Any = field(repr=False, compare=False)
    row_iid: str
    row_source_sha256: str
    routing_digest: str
    plan_digest: str
    _session_token: object = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "UpdateCellPlan":
        raise GraftPhaseAShortTrainingError("update plans are minted by a session")

    @classmethod
    def _mint(cls, *, session_token: object, **values: Any) -> "UpdateCellPlan":
        _PINNED_TRAINER_EXECUTION_GATE()
        instance = object.__new__(cls)
        expected = set(cls.__dataclass_fields__) - {"_session_token", "_token"}
        if set(values) != expected:
            raise GraftPhaseAShortTrainingError("update-plan fields differ")
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        object.__setattr__(instance, "_session_token", session_token)
        object.__setattr__(instance, "_token", _PLAN_TOKEN)
        return instance


@dataclass(frozen=True, init=False)
class ConfirmationPlan:
    """Opaque local confirmation row plus its fixed cross-owner intervention."""

    dp_arm: int
    row: Any = field(repr=False, compare=False)
    wrong_owner_row: Any = field(repr=False, compare=False)
    row_iid: str
    wrong_owner_iid: str
    schedule_indices: tuple[int, ...]
    field_roles: tuple[str, ...]
    parameter_digest: str
    plan_digest: str
    _session_token: object = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "ConfirmationPlan":
        raise GraftPhaseAShortTrainingError(
            "confirmation plans are minted by a session"
        )

    @classmethod
    def _mint(cls, *, session_token: object, **values: Any) -> "ConfirmationPlan":
        _PINNED_TRAINER_EXECUTION_GATE()
        instance = object.__new__(cls)
        expected = set(cls.__dataclass_fields__) - {"_session_token", "_token"}
        if set(values) != expected:
            raise GraftPhaseAShortTrainingError("confirmation-plan fields differ")
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        object.__setattr__(instance, "_session_token", session_token)
        object.__setattr__(instance, "_token", _PLAN_TOKEN)
        return instance


@dataclass(frozen=True)
class ShortTrainingResult:
    receipt: Mapping[str, Any]
    checkpoint_payload: None = None
    publication_payload: None = None


def _routing_record(routing: Any) -> tuple[tuple[Any, ...], tuple[Any, ...], bool]:
    _PINNED_TRAINER_EXECUTION_GATE()
    if type(routing) is _PINNED_ROUTING_TYPE:
        test_only = False
        if (
            type(routing.authority) is not source_consumer.AuthorityBoundary
            or any(vars(routing.authority).values())
        ):
            raise GraftPhaseAShortTrainingError("source routing authority differs")
        update_rows = routing.update_rows
        confirmation_rows = routing.confirmation_rows
        if any(type(row) is not _PINNED_ROW_TYPE for row in update_rows + confirmation_rows):
            raise GraftPhaseAShortTrainingError("source routing row type differs")
    elif type(routing) is _TestRouting:
        test_only = True
        if routing.test_only is not True or any(routing.authority.values()):
            raise GraftPhaseAShortTrainingError("CPU routing authority differs")
        update_rows = routing.update_rows
        confirmation_rows = routing.confirmation_rows
        if any(type(row) is not _TestSourceRow for row in update_rows + confirmation_rows):
            raise GraftPhaseAShortTrainingError("CPU routing row type differs")
    else:
        raise GraftPhaseAShortTrainingError(
            "training requires an exact path-free consumer routing"
        )
    if len(update_rows) != 2 or len(confirmation_rows) != 2:
        raise GraftPhaseAShortTrainingError("routing must contain 2+2 rows")
    rows = update_rows + confirmation_rows
    for index, (row, expected) in enumerate(zip(rows, source_consumer.CANARY4)):
        iid, split, update_allowed, confirmation_only = expected
        raw = row.source_bytes
        if (
            row.iid != iid
            or row.split != split
            or row.optimizer_update_allowed is not update_allowed
            or row.optimizer_confirmation_only is not confirmation_only
            or type(raw) is not bytes
            or not raw
            or hashlib.sha256(raw).hexdigest() != row.source_sha256
            or row.noop_instruction != source_consumer.NOOP_INSTRUCTION
        ):
            raise GraftPhaseAShortTrainingError(f"routed source row {index} differs")
    if len({row.source_sha256 for row in rows}) != 4:
        raise GraftPhaseAShortTrainingError("routing digest closure differs")
    _require_sha256(routing.routing_digest, label="routing digest")
    _require_sha256(
        routing.source_release_result_digest, label="source release result digest"
    )
    _require_sha256(routing.pinset_digest, label="source pinset digest")
    if test_only:
        recomputed_record = {
            "schema_version": TEST_ROUTING_SCHEMA_VERSION,
            "source_release_result_digest": routing.source_release_result_digest,
            "pinset_digest": routing.pinset_digest,
            "rows": [
                {
                    "iid": row.iid,
                    "split": row.split,
                    "optimizer_update_allowed": row.optimizer_update_allowed,
                    "optimizer_confirmation_only": row.optimizer_confirmation_only,
                    "source_sha256": row.source_sha256,
                    "source_size_bytes": len(row.source_bytes),
                }
                for row in rows
            ],
            "path_reopen_allowed": False,
            "owned_bytes_only": True,
            "test_only": True,
            "authority": _false_authority(),
        }
    else:
        media_rows = []
        for index, row in enumerate(rows):
            media = row.media
            if type(media) is not source_consumer.OwnedSourceMedia or (
                media.frame_count != source_consumer.FRAME_COUNT
                or media.fps_numerator != source_consumer.FPS_NUMERATOR
                or media.fps_denominator != source_consumer.FPS_DENOMINATOR
                or isinstance(media.width, bool)
                or isinstance(media.height, bool)
                or not isinstance(media.width, int)
                or not isinstance(media.height, int)
                or min(media.width, media.height) != source_consumer.SHORT_SIDE
                or media.rgb_or_codec_content_not_interpreted is not True
            ):
                raise GraftPhaseAShortTrainingError(
                    f"routed source row {index} media differs"
                )
            media_rows.append(
                {
                    "frame_count": media.frame_count,
                    "fps_numerator": media.fps_numerator,
                    "fps_denominator": media.fps_denominator,
                    "width": media.width,
                    "height": media.height,
                    "rgb_or_codec_content_not_interpreted": (
                        media.rgb_or_codec_content_not_interpreted
                    ),
                }
            )
        recomputed_record = {
            "schema_version": "bernini-graft-a-lite-owned-trainer-routing-v1",
            "source_release_result_digest": routing.source_release_result_digest,
            "pinset_digest": routing.pinset_digest,
            "rows": [
                {
                    "iid": row.iid,
                    "split": row.split,
                    "optimizer_update_allowed": row.optimizer_update_allowed,
                    "optimizer_confirmation_only": row.optimizer_confirmation_only,
                    "source_sha256": row.source_sha256,
                    "owned_source_bytes_sha256": hashlib.sha256(
                        row.source_bytes
                    ).hexdigest(),
                    "owned_source_bytes_size": len(row.source_bytes),
                    "media": media_rows[index],
                    "noop_instruction": row.noop_instruction,
                }
                for index, row in enumerate(rows)
            ],
            "path_reopen_allowed": False,
            "owned_bytes_only": True,
            "authority": vars(routing.authority),
        }
    if _object_sha256(recomputed_record) != routing.routing_digest:
        raise GraftPhaseAShortTrainingError("routing canonical digest differs")
    return tuple(update_rows), tuple(confirmation_rows), test_only


def _routing_live_fingerprint(routing: Any) -> str:
    _PINNED_TRAINER_EXECUTION_GATE()
    update_rows, confirmation_rows, test_only = _PINNED_ROUTING_RECORD(
        routing
    )
    rows = update_rows + confirmation_rows
    return _object_sha256(
        {
            "routing_type": f"{type(routing).__module__}.{type(routing).__qualname__}",
            "test_only": test_only,
            "source_release_result_digest": routing.source_release_result_digest,
            "pinset_digest": routing.pinset_digest,
            "routing_digest": routing.routing_digest,
            "rows": [
                {
                    "row_type": f"{type(row).__module__}.{type(row).__qualname__}",
                    "object_identity": id(row),
                    "iid": row.iid,
                    "split": row.split,
                    "optimizer_update_allowed": row.optimizer_update_allowed,
                    "optimizer_confirmation_only": row.optimizer_confirmation_only,
                    "source_sha256": row.source_sha256,
                    "source_bytes_sha256": hashlib.sha256(
                        row.source_bytes
                    ).hexdigest(),
                    "source_bytes_size": len(row.source_bytes),
                    "noop_instruction": row.noop_instruction,
                }
                for row in rows
            ],
        }
    )


def _frozen_base_registry(
    bindings: native_v2.AuthenticatedNativeBindings,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    _PINNED_TRAINER_EXECUTION_GATE()
    trainable_ids = {id(parameter) for _, parameter in bindings.named_trainable_parameters}
    seen: set[int] = set()
    result: list[tuple[str, torch.nn.Parameter]] = []
    owners = (
        ("diffusion", bindings.diffusion),
        ("transformer", bindings.transformer),
        *bindings.external_trainable_owner_modules,
    )
    for owner_name, module in owners:
        for name, parameter in module.named_parameters(recurse=True):
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            if identity in trainable_ids:
                continue
            if parameter.requires_grad or parameter.grad is not None:
                raise GraftPhaseAShortTrainingError(
                    f"frozen base parameter is trainable or has a gradient: {owner_name}.{name}"
                )
            result.append((f"{owner_name}.{name}", parameter))
    if not result:
        raise GraftPhaseAShortTrainingError("authenticated frozen base registry is empty")
    return tuple(result)


def _registry(
    bindings: native_v2.AuthenticatedNativeBindings,
) -> tuple[
    tuple[tuple[str, torch.nn.Parameter], ...],
    Mapping[str, tuple[tuple[str, torch.nn.Parameter], ...]],
]:
    _PINNED_TRAINER_EXECUTION_GATE()
    rows = tuple(bindings.named_trainable_parameters)
    if not rows or len({name for name, _ in rows}) != len(rows) or len({id(parameter) for _, parameter in rows}) != len(rows):
        raise GraftPhaseAShortTrainingError("trainable registry differs")
    categories: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        category: [] for category in native_v2.GRADIENT_CATEGORIES
    }
    for name, parameter in rows:
        category = _gradient_category(name)
        if not parameter.requires_grad or not parameter.is_floating_point():
            raise GraftPhaseAShortTrainingError(
                f"trainable parameter contract differs: {name}"
            )
        if parameter.grad is not None:
            raise GraftPhaseAShortTrainingError(
                f"trainable gradient must be empty before opening: {name}"
            )
        categories[category].append((name, parameter))
    if any(not categories[category] for category in native_v2.GRADIENT_CATEGORIES):
        raise GraftPhaseAShortTrainingError("five-category registry is incomplete")
    return rows, MappingProxyType(
        {category: tuple(values) for category, values in categories.items()}
    )


def _category_norms(
    categories: Mapping[str, Sequence[tuple[str, torch.nn.Parameter]]]
) -> dict[str, float]:
    result = {}
    for category, rows in categories.items():
        squared = 0.0
        for _, parameter in rows:
            if parameter.grad is None:
                raise GraftPhaseAShortTrainingError(
                    "gradient remained absent after materialization"
                )
            value = float(parameter.grad.detach().float().square().sum().item())
            if not math.isfinite(value):
                raise GraftPhaseAShortTrainingError("gradient norm is non-finite")
            squared += value
        result[category] = math.sqrt(squared)
    return result


def _synchronize_dp2_sp4_gradients(
    *,
    named: Sequence[tuple[str, torch.nn.Parameter]],
    categories: Mapping[str, Sequence[tuple[str, torch.nn.Parameter]]],
    backend: AuthenticatedDP2SP4Backend,
    update_number: int,
    expected_regime: str,
) -> Mapping[str, Any]:
    """Materialize None, then apply the one allowed SP->DP averaging order."""

    _PINNED_TRAINER_EXECUTION_GATE()

    if expected_regime not in UPDATE_REGIMES:
        raise GraftPhaseAShortTrainingError("gradient regime differs")
    materialized = []
    for name, parameter in named:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(
                parameter, memory_format=torch.preserve_format
            )
            materialized.append(name)
        if (
            type(parameter.grad) is not torch.Tensor
            or parameter.grad.shape != parameter.shape
            or parameter.grad.dtype != parameter.dtype
            or parameter.grad.device != parameter.device
            or not bool(torch.isfinite(parameter.grad).all().item())
        ):
            raise GraftPhaseAShortTrainingError(
                f"local gradient differs before collective: {name}"
            )
        backend.reduce_gradient_(
            parameter.grad,
            axis="sp",
            parameter_name=name,
            update_number=update_number,
        )
        parameter.grad.div_(float(SP_SIZE))
        backend.reduce_gradient_(
            parameter.grad,
            axis="dp",
            parameter_name=name,
            update_number=update_number,
        )
        parameter.grad.div_(float(DP_SIZE))
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise GraftPhaseAShortTrainingError(
                f"averaged gradient is non-finite: {name}"
            )
    norms = _category_norms(categories)
    if expected_regime == "bootstrap":
        if norms["output_projection"] <= 0.0 or any(
            norms[category] != 0.0
            for category in native_v2.UPSTREAM_GRADIENT_CATEGORIES
        ):
            raise GraftPhaseAShortTrainingError(
                "WORLD8 bootstrap gradient is not output-only"
            )
        gate = "world8_bootstrap_output_projection_only_nonzero"
    else:
        if any(norms[category] <= 0.0 for category in native_v2.GRADIENT_CATEGORIES):
            raise GraftPhaseAShortTrainingError(
                "WORLD8 post-bootstrap gradient lacks a nonzero category"
            )
        gate = "world8_post_bootstrap_all_five_categories_nonzero"
    squared = sum(value * value for value in norms.values())
    total = math.sqrt(squared)
    if not math.isfinite(total) or total <= 0.0:
        raise GraftPhaseAShortTrainingError("WORLD8 gradient norm is zero/non-finite")
    return _seal(
        {
            "schema_version": "bernini-graft-dp2-sp4-gradient-sync-v1",
            "update_number": update_number,
            "training_regime": expected_regime,
            "none_materialized_as_true_zero_count": len(materialized),
            "none_materialized_parameter_names": materialized,
            "collective_order": ["SP4_SUM", "divide_by_4", "DP2_SUM", "divide_by_2"],
            "category_l2_float64_hex": {
                category: norms[category].hex()
                for category in native_v2.GRADIENT_CATEGORIES
            },
            "preclip_l2_float64_hex": total.hex(),
            "gate": gate,
            "finite": True,
        }
    )


def _validate_native_result(
    *,
    result: Any,
    bindings: native_v2.AuthenticatedNativeBindings,
    backend: AuthenticatedDP2SP4Backend,
    plan: UpdateCellPlan,
) -> Mapping[str, Any]:
    _PINNED_TRAINER_EXECUTION_GATE()
    if type(result) is not _PINNED_RESULT_TYPE:
        raise GraftPhaseAShortTrainingError("native v2 result type differs")
    loss = _finite_scalar(result.flow_matching_loss, label="native v2 FM loss")
    if loss < 0.0:
        raise GraftPhaseAShortTrainingError("native v2 FM loss is negative")
    receipt = _validated_sealed_mapping(result.receipt, label="native v2 receipt")
    expected_target = backend.sp_rank >= 2
    local_target_rows = receipt.get("local_target_rows")
    denied = (
        "optimizer_created",
        "parameters_updated",
        "scheduler_step_called",
        "outer_clean_state_transport_used",
        "external_guided_clean_cotangent_accepted",
        "target_video_used",
        "mask_used",
        "pose_used",
        "track_used",
        "optical_flow_used",
        "motion_donor_used",
        "full_sampler_trajectory_verified",
        "training_quality_claim_authorized",
        "scientific_action_editing_claim_authorized",
        "optimizer_step_verified_by_this_core",
        "two_consecutive_steps_verified_by_this_core",
        "post_bootstrap_cuda_short_training_verified_by_this_core",
        "short_training_claim_authorized",
    )
    if (
        receipt.get("schema_version") != native_v2.SCHEMA_VERSION
        or receipt.get("binding_receipt_digest") != bindings.receipt()["digest"]
        or receipt.get("schedule_index") != plan.schedule_index
        or receipt.get("training_regime") != plan.expected_regime
        or receipt.get("schedule_cell_active_for_training") is not True
        or receipt.get("schedule_cell_counted_as_trained") is not True
        or receipt.get("phase_a_objective") != native_v2.FLOW_MATCHING_OBJECTIVE
        or receipt.get("supervision_pair") != "same_source_video_noop"
        or receipt.get("frame_count") != native_v2.EXPECTED_FRAMES
        or receipt.get("local_sequence_parallel_rank") != backend.sp_rank
        or receipt.get("local_sequence_parallel_size") != SP_SIZE
        or type(local_target_rows) is not int
        or local_target_rows < 0
        or (local_target_rows > 0) is not expected_target
        or receipt.get("local_adapter_graph_bearing") is not expected_target
        or receipt.get("trainable_registry_values_unchanged") is not True
        or receipt.get("exclusive_trainable_scope_is_exact_authenticated_registry") is not True
        or any(receipt.get(key) is not False for key in denied)
    ):
        raise GraftPhaseAShortTrainingError("native v2 receipt contract differs")
    if not expected_target:
        local_gradient_gate = (
            "source_only_sp_rank_all_five_categories_exact_zero"
        )
        expected_gradient_gates = {
            "bootstrap_output_only_gate_verified": False,
            "post_bootstrap_five_category_local_gate_verified": False,
            "source_only_sp_all_five_categories_exact_zero_verified": True,
        }
    elif plan.expected_regime == "bootstrap":
        local_gradient_gate = (
            "bootstrap_target_rows_output_projection_only_nonzero"
        )
        expected_gradient_gates = {
            "bootstrap_output_only_gate_verified": True,
            "post_bootstrap_five_category_local_gate_verified": False,
            "source_only_sp_all_five_categories_exact_zero_verified": False,
        }
    else:
        local_gradient_gate = (
            "post_bootstrap_target_rows_all_five_categories_finite_nonzero"
        )
        expected_gradient_gates = {
            "bootstrap_output_only_gate_verified": False,
            "post_bootstrap_five_category_local_gate_verified": True,
            "source_only_sp_all_five_categories_exact_zero_verified": False,
        }
    if any(
        receipt.get(field) is not expected
        for field, expected in expected_gradient_gates.items()
    ):
        label = (
            "bootstrap"
            if plan.expected_regime == "bootstrap"
            else "post-bootstrap"
        )
        raise GraftPhaseAShortTrainingError(f"{label} v2 gate differs")
    return _seal(
        {
            "schema_version": "bernini-graft-native-v2-cell-admission-v2",
            "update_number": plan.update_number,
            "schedule_index": plan.schedule_index,
            "training_regime": plan.expected_regime,
            "row_iid": plan.row_iid,
            "row_source_sha256": plan.row_source_sha256,
            "v2_receipt_digest": receipt["digest"],
            "flow_matching_loss_float64_hex": loss.hex(),
            "local_sp_rank": backend.sp_rank,
            "local_target_owner": expected_target,
            "local_gradient_gate": local_gradient_gate,
            "source_bytes_to_latent_binding_verified_by_this_core": False,
            "full_sampler_used": False,
            "decoded_media_used": False,
            **_false_authority(),
        }
    )


_SESSION_TOKEN = object()


class PhaseAShortTrainingSession:
    """Single-use two-update then confirmation state machine."""

    __slots__ = (
        "_token",
        "_routing",
        "_opened_routing_live_fingerprint",
        "_bindings",
        "_backend",
        "_update_rows",
        "_confirmation_rows",
        "_named",
        "_categories",
        "_base",
        "_initial_trainable",
        "_initial_parameter_digest",
        "_initial_base_digest",
        "_optimizer",
        "_optimizer_step_live",
        "_optimizer_zero_grad_live",
        "_optimizer_steps",
        "_phase",
        "_update_receipts",
        "_confirmation_metrics",
        "_confirmation_consensus",
        "_failed_confirmation_metrics",
        "_post_training_parameter_digest",
        "_post_training_base_digest",
        "_post_training_optimizer_digest",
        "_failure_reason",
    )

    def __init_subclass__(cls, **_kwargs: Any) -> None:
        raise GraftPhaseAShortTrainingError("short-training session is exact-type only")

    def __init__(self, *_: Any, **__: Any) -> None:
        raise GraftPhaseAShortTrainingError(
            "short-training sessions are opened by the authenticated factory"
        )

    @classmethod
    def _mint(
        cls,
        *,
        routing: Any,
        bindings: native_v2.AuthenticatedNativeBindings,
        backend: AuthenticatedDP2SP4Backend,
    ) -> "PhaseAShortTrainingSession":
        _PINNED_TRAINER_EXECUTION_GATE()
        _assert_pinned_dependencies()
        if type(bindings) is not _PINNED_BINDING_TYPE:
            raise GraftPhaseAShortTrainingError("native binding type differs")
        bindings.assert_live()
        _PINNED_BACKEND_ASSERT_LIVE(backend)
        update_rows, confirmation_rows, routing_test_only = _PINNED_ROUTING_RECORD(
            routing
        )
        if not (
            routing_test_only is backend.test_only is bindings.test_only
        ):
            raise GraftPhaseAShortTrainingError(
                "production and test-only evidence cannot be mixed"
            )
        named, categories = _PINNED_REGISTRY(bindings)
        base = _PINNED_FROZEN_BASE_REGISTRY(bindings)
        instance = object.__new__(cls)
        instance._token = _SESSION_TOKEN
        instance._routing = routing
        instance._opened_routing_live_fingerprint = (
            _PINNED_ROUTING_LIVE_FINGERPRINT(routing)
        )
        instance._bindings = bindings
        instance._backend = backend
        instance._update_rows = update_rows
        instance._confirmation_rows = confirmation_rows
        instance._named = named
        instance._categories = categories
        instance._base = base
        instance._initial_trainable = tuple(
            (parameter, parameter.detach().clone()) for _, parameter in named
        )
        instance._initial_parameter_digest = _named_tensor_digest(named)
        instance._initial_base_digest = _named_tensor_digest(base)
        backend.consensus(
            instance._initial_parameter_digest,
            scope="world",
            label="initial trainable parameters",
        )
        backend.consensus(
            instance._initial_base_digest,
            scope="world",
            label="initial frozen base",
        )
        instance._optimizer = _PINNED_ADAMW(
            [parameter for _, parameter in named],
            lr=OPTIMIZER_LEARNING_RATE,
            betas=OPTIMIZER_BETAS,
            eps=OPTIMIZER_EPS,
            weight_decay=OPTIMIZER_WEIGHT_DECAY,
            foreach=False,
        )
        instance._optimizer_step_live = type(instance._optimizer).step
        instance._optimizer_zero_grad_live = type(instance._optimizer).zero_grad
        instance._optimizer_steps = 0
        instance._phase = "update_1_pending"
        instance._update_receipts: list[Mapping[str, Any]] = []
        instance._confirmation_metrics: dict[int, Mapping[str, Any]] = {}
        instance._confirmation_consensus: dict[int, str] = {}
        instance._failed_confirmation_metrics: Optional[Mapping[str, Any]] = None
        instance._post_training_parameter_digest: Optional[str] = None
        instance._post_training_base_digest: Optional[str] = None
        instance._post_training_optimizer_digest: Optional[str] = None
        instance._failure_reason: Optional[str] = None
        _PINNED_SESSION_ASSERT_OPTIMIZER_LIVE(instance)
        return instance

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def optimizer_steps(self) -> int:
        return self._optimizer_steps

    def _assert_live(self) -> None:
        _PINNED_TRAINER_EXECUTION_GATE()
        if (
            self._token is not _SESSION_TOKEN
            or type(self) is not PhaseAShortTrainingSession
            or type(self._backend) is not AuthenticatedDP2SP4Backend
            or type(self._bindings) is not _PINNED_BINDING_TYPE
        ):
            raise GraftPhaseAShortTrainingError("short-training session changed")
        _assert_pinned_dependencies()
        self._bindings.assert_live()
        _PINNED_BACKEND_ASSERT_LIVE(self._backend)
        _PINNED_SESSION_ASSERT_OPTIMIZER_LIVE(self)
        for index, receipt in enumerate(self._update_receipts):
            _validated_sealed_mapping(
                receipt, label=f"stored update receipt {index + 1}"
            )
        for index, metrics in self._confirmation_metrics.items():
            _validated_sealed_mapping(
                metrics, label=f"stored confirmation metrics {index}"
            )
        if (
            _PINNED_ROUTING_LIVE_FINGERPRINT(self._routing)
            != self._opened_routing_live_fingerprint
        ):
            raise GraftPhaseAShortTrainingError(
                "path-free routing changed after session open"
            )

    def _restore_trainable_snapshot(self) -> None:
        cleanup_error: Optional[Exception] = None
        for parameter, snapshot in self._initial_trainable:
            try:
                with torch.no_grad():
                    parameter.copy_(snapshot)
                parameter.grad = None
            except Exception as error:  # pragma: no cover - catastrophic runtime
                cleanup_error = error
        self._optimizer.state.clear()
        if cleanup_error is not None:
            raise GraftPhaseAShortTrainingError(
                "failed session could not restore trainable parameters"
            ) from cleanup_error

    def _fail(self, error: Exception) -> None:
        try:
            _PINNED_SESSION_RESTORE_TRAINABLE_SNAPSHOT(self)
        finally:
            self._failure_reason = f"{type(error).__name__}:{error}"
            self._phase = "failed"

    def next_update_plan(self) -> UpdateCellPlan:
        try:
            _PINNED_TRAINER_EXECUTION_GATE()
            _PINNED_SESSION_ASSERT_LIVE(self)
            if self._phase not in {"update_1_pending", "update_2_pending"}:
                raise GraftPhaseAShortTrainingError("no update cell is pending")
            index = self._optimizer_steps
            row = self._update_rows[self._backend.dp_arm]
            if not row.optimizer_update_allowed or row.optimizer_confirmation_only:
                raise GraftPhaseAShortTrainingError(
                    "confirmation row cannot be promoted to an update"
                )
            record = {
                "schema_version": PLAN_SCHEMA_VERSION,
                "update_number": index + 1,
                "schedule_index": UPDATE_SCHEDULE_INDICES[index],
                "expected_regime": UPDATE_REGIMES[index],
                "dp_arm": self._backend.dp_arm,
                "row_iid": row.iid,
                "row_source_sha256": row.source_sha256,
                "routing_digest": self._routing.routing_digest,
                "source_owned_bytes_only": True,
                "source_path_input_accepted": False,
            }
            return UpdateCellPlan._mint(
                session_token=self._token,
                update_number=index + 1,
                schedule_index=UPDATE_SCHEDULE_INDICES[index],
                expected_regime=UPDATE_REGIMES[index],
                dp_arm=self._backend.dp_arm,
                row=row,
                row_iid=row.iid,
                row_source_sha256=row.source_sha256,
                routing_digest=self._routing.routing_digest,
                plan_digest=_object_sha256(record),
            )
        except Exception as error:
            _PINNED_SESSION_FAIL(self, error)
            raise

    def _validate_update_plan(self, plan: Any) -> None:
        _PINNED_TRAINER_EXECUTION_GATE()
        expected_record = None
        if type(plan) is UpdateCellPlan:
            expected_record = {
                "schema_version": PLAN_SCHEMA_VERSION,
                "update_number": plan.update_number,
                "schedule_index": plan.schedule_index,
                "expected_regime": plan.expected_regime,
                "dp_arm": plan.dp_arm,
                "row_iid": plan.row_iid,
                "row_source_sha256": plan.row_source_sha256,
                "routing_digest": plan.routing_digest,
                "source_owned_bytes_only": True,
                "source_path_input_accepted": False,
            }
        if (
            type(plan) is not UpdateCellPlan
            or plan._token is not _PLAN_TOKEN
            or plan._session_token is not self._token
            or plan.update_number != self._optimizer_steps + 1
            or plan.schedule_index != UPDATE_SCHEDULE_INDICES[self._optimizer_steps]
            or plan.expected_regime != UPDATE_REGIMES[self._optimizer_steps]
            or plan.dp_arm != self._backend.dp_arm
            or plan.row is not self._update_rows[self._backend.dp_arm]
            or plan.row_iid != plan.row.iid
            or plan.row_source_sha256 != plan.row.source_sha256
            or plan.routing_digest != self._routing.routing_digest
            or not plan.row.optimizer_update_allowed
            or plan.row.optimizer_confirmation_only
            or expected_record is None
            or plan.plan_digest != _object_sha256(expected_record)
        ):
            raise GraftPhaseAShortTrainingError("update plan differs or uses wrong split")

    def run_update(
        self, *, plan: UpdateCellPlan, cell: native_v2.PhaseANativeTrainingClosure
    ) -> Mapping[str, Any]:
        """Run one exact v2 cell, synchronize its gradients, and update AdamW."""

        try:
            _PINNED_TRAINER_EXECUTION_GATE()
            _PINNED_SESSION_ASSERT_LIVE(self)
            if self._phase not in {"update_1_pending", "update_2_pending"}:
                raise GraftPhaseAShortTrainingError("update state differs")
            _PINNED_SESSION_VALIDATE_UPDATE_PLAN(self, plan)
            if (
                type(cell) is not _PINNED_CELL_TYPE
                or cell.bindings is not self._bindings
                or cell.schedule_index != plan.schedule_index
                or cell.training_regime != plan.expected_regime
                or cell.phase != "new"
            ):
                raise GraftPhaseAShortTrainingError("native v2 cell binding differs")
            if any(parameter.grad is not None for _, parameter in self._named):
                raise GraftPhaseAShortTrainingError(
                    "trainable gradients are stale before v2 execution"
                )
            before_parameters = _named_tensor_digest(self._named)
            before_base = _named_tensor_digest(self._base)
            if before_base != self._initial_base_digest:
                raise GraftPhaseAShortTrainingError("frozen base changed before update")
            cell.measure()
            cell.derive_phase_a_flow_matching_vjp()
            result = cell.replay_and_backward()
            admission = _PINNED_VALIDATE_NATIVE_RESULT(
                result=result,
                bindings=self._bindings,
                backend=self._backend,
                plan=plan,
            )
            if any(parameter.grad is not None for _, parameter in self._base):
                raise GraftPhaseAShortTrainingError(
                    "frozen base acquired a gradient"
                )
            synchronization = _PINNED_SYNCHRONIZE_DP2_SP4_GRADIENTS(
                named=self._named,
                categories=self._categories,
                backend=self._backend,
                update_number=plan.update_number,
                expected_regime=plan.expected_regime,
            )
            clipped = _PINNED_CLIP_GRAD_NORM(
                [parameter for _, parameter in self._named],
                MAX_GRAD_NORM,
            )
            clipped_float = float(clipped.item())
            if not math.isfinite(clipped_float) or clipped_float <= 0.0:
                raise GraftPhaseAShortTrainingError("gradient clipping norm differs")
            postclip_norms = _category_norms(self._categories)
            postclip_total = math.sqrt(sum(value * value for value in postclip_norms.values()))
            if not math.isfinite(postclip_total) or postclip_total > MAX_GRAD_NORM + 1.0e-5:
                raise GraftPhaseAShortTrainingError("post-clip norm exceeds one")
            self._optimizer.step()
            self._optimizer_steps += 1
            after_parameters = _named_tensor_digest(self._named)
            after_base = _named_tensor_digest(self._base)
            if after_parameters == before_parameters:
                raise GraftPhaseAShortTrainingError("optimizer did not change parameters")
            if after_base != before_base or after_base != self._initial_base_digest:
                raise GraftPhaseAShortTrainingError("frozen base changed during update")
            self._backend.consensus(
                after_parameters,
                scope="world",
                label=f"trainable parameters after update {plan.update_number}",
            )
            self._backend.consensus(
                after_base,
                scope="world",
                label=f"frozen base after update {plan.update_number}",
            )
            self._optimizer.zero_grad(set_to_none=True)
            if any(parameter.grad is not None for _, parameter in self._named):
                raise GraftPhaseAShortTrainingError(
                    "optimizer zero_grad did not restore None gradients"
                )
            update_receipt = _seal(
                {
                    "schema_version": "bernini-graft-phase-a-short-update-v1",
                    "update_number": plan.update_number,
                    "schedule_index": plan.schedule_index,
                    "training_regime": plan.expected_regime,
                    "dp_arm": plan.dp_arm,
                    "sp_rank": self._backend.sp_rank,
                    "row_iid": plan.row_iid,
                    "row_source_sha256": plan.row_source_sha256,
                    "plan_digest": plan.plan_digest,
                    "native_admission_digest": admission["digest"],
                    "gradient_synchronization_digest": synchronization["digest"],
                    "gradient_sync": dict(synchronization),
                    "preclip_norm_float64_hex": clipped_float.hex(),
                    "postclip_norm_float64_hex": postclip_total.hex(),
                    "max_grad_norm": MAX_GRAD_NORM,
                    "optimizer": {
                        "kind": "torch.optim.AdamW",
                        "learning_rate": OPTIMIZER_LEARNING_RATE,
                        "weight_decay": OPTIMIZER_WEIGHT_DECAY,
                        "betas": list(OPTIMIZER_BETAS),
                        "eps": OPTIMIZER_EPS,
                        "foreach": False,
                    },
                    "parameter_digest_before": before_parameters,
                    "parameter_digest_after": after_parameters,
                    "frozen_base_digest": after_base,
                    "parameter_world_consensus": True,
                    "frozen_base_world_consensus": True,
                    "gradients_reset_to_none_after_step": True,
                    "checkpoint_written": False,
                    **_false_authority(),
                }
            )
            self._update_receipts.append(update_receipt)
            if self._optimizer_steps == 1:
                output_nonzero = any(
                    bool(torch.count_nonzero(parameter.detach()).item())
                    for _, parameter in self._categories["output_projection"]
                )
                if not output_nonzero:
                    raise GraftPhaseAShortTrainingError(
                        "bootstrap did not create a post-bootstrap output state"
                    )
                self._phase = "update_2_pending"
            elif self._optimizer_steps == 2:
                self._post_training_parameter_digest = after_parameters
                self._post_training_base_digest = after_base
                self._post_training_optimizer_digest = (
                    _PINNED_SESSION_OPTIMIZER_STATE_DIGEST(self)
                )
                self._phase = "confirmation_pending"
            else:  # pragma: no cover - protected by the state gate
                raise GraftPhaseAShortTrainingError("optimizer step count exceeded two")
            return update_receipt
        except Exception as error:
            _PINNED_SESSION_FAIL(self, error)
            raise

    def _assert_optimizer_live(self) -> None:
        _PINNED_TRAINER_EXECUTION_GATE()
        if (
            type(self._optimizer) is not _PINNED_ADAMW
            or type(self._optimizer).step is not self._optimizer_step_live
            or type(self._optimizer).zero_grad
            is not self._optimizer_zero_grad_live
            or inspect.unwrap(self._optimizer.step) is not _PINNED_ADAMW_STEP
            or inspect.unwrap(self._optimizer.zero_grad)
            is not _PINNED_ADAMW_ZERO_GRAD
            or len(self._optimizer.param_groups) != 1
        ):
            raise GraftPhaseAShortTrainingError("live AdamW object differs")
        group = self._optimizer.param_groups[0]
        expected_parameters = tuple(parameter for _, parameter in self._named)
        observed_parameters = tuple(group.get("params", ()))
        if (
            len(observed_parameters) != len(expected_parameters)
            or any(
                observed is not expected
                for observed, expected in zip(
                    observed_parameters, expected_parameters
                )
            )
            or group.get("lr") != OPTIMIZER_LEARNING_RATE
            or tuple(group.get("betas", ())) != OPTIMIZER_BETAS
            or group.get("eps") != OPTIMIZER_EPS
            or group.get("weight_decay") != OPTIMIZER_WEIGHT_DECAY
            or group.get("amsgrad") is not False
            or group.get("maximize", False) is not False
            or group.get("foreach") is not False
            or group.get("capturable", False) is not False
            or group.get("differentiable", False) is not False
            or group.get("fused", None) not in {None, False}
        ):
            raise GraftPhaseAShortTrainingError(
                "live AdamW parameter group or hyperparameters differ"
            )
        defaults = self._optimizer.defaults
        if (
            defaults.get("lr") != OPTIMIZER_LEARNING_RATE
            or tuple(defaults.get("betas", ())) != OPTIMIZER_BETAS
            or defaults.get("eps") != OPTIMIZER_EPS
            or defaults.get("weight_decay") != OPTIMIZER_WEIGHT_DECAY
            or defaults.get("amsgrad") is not False
            or defaults.get("maximize", False) is not False
            or defaults.get("foreach") is not False
            or defaults.get("capturable", False) is not False
            or defaults.get("differentiable", False) is not False
            or defaults.get("fused", None) not in {None, False}
        ):
            raise GraftPhaseAShortTrainingError("live AdamW defaults differ")
        state = self._optimizer.state
        if set(state) - set(expected_parameters) or (
            self._optimizer_steps == 0 and state
        ) or (
            self._optimizer_steps > 0 and set(state) != set(expected_parameters)
        ):
            raise GraftPhaseAShortTrainingError("live AdamW state registry differs")
        for parameter in expected_parameters:
            if self._optimizer_steps == 0:
                continue
            row = state[parameter]
            if set(row) != {"step", "exp_avg", "exp_avg_sq"}:
                raise GraftPhaseAShortTrainingError(
                    "live AdamW state fields differ"
                )
            step = row["step"]
            if (
                type(step) is not torch.Tensor
                or step.numel() != 1
                or step.requires_grad
                or step.grad_fn is not None
                or not bool(torch.isfinite(step).item())
                or float(step.item()) != float(self._optimizer_steps)
            ):
                raise GraftPhaseAShortTrainingError("live AdamW step differs")
            state_storage_pointers = []
            for name in ("exp_avg", "exp_avg_sq"):
                value = row[name]
                if (
                    type(value) is not torch.Tensor
                    or value.shape != parameter.shape
                    or value.dtype != parameter.dtype
                    or value.device != parameter.device
                    or value.requires_grad
                    or value.grad_fn is not None
                    or not bool(torch.isfinite(value).all().item())
                ):
                    raise GraftPhaseAShortTrainingError(
                        f"live AdamW {name} state differs"
                    )
                state_storage_pointers.append(
                    (
                        value.device.type,
                        value.device.index,
                        int(value.untyped_storage().data_ptr()),
                    )
                )
            parameter_pointer = (
                parameter.device.type,
                parameter.device.index,
                int(parameter.untyped_storage().data_ptr()),
            )
            if (
                len(set(state_storage_pointers)) != 2
                or parameter_pointer in state_storage_pointers
            ):
                raise GraftPhaseAShortTrainingError(
                    "live AdamW moment storage aliases"
                )

    def _optimizer_state_digest(self) -> str:
        _PINNED_TRAINER_EXECUTION_GATE()
        _PINNED_SESSION_ASSERT_OPTIMIZER_LIVE(self)
        rows = []
        parameter_index = {id(parameter): index for index, (_, parameter) in enumerate(self._named)}
        for parameter, state in self._optimizer.state.items():
            state_row: dict[str, Any] = {"parameter_index": parameter_index[id(parameter)]}
            for key in sorted(state):
                value = state[key]
                if type(value) is torch.Tensor:
                    state_row[str(key)] = {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "sha256": _tensor_bytes_sha256(value),
                    }
                elif isinstance(value, (bool, int, float)):
                    state_row[str(key)] = value
                else:
                    state_row[str(key)] = str(value)
            rows.append(state_row)
        return _object_sha256(
            {
                "state": sorted(rows, key=lambda row: row["parameter_index"]),
                "step_count": self._optimizer_steps,
                "lr": OPTIMIZER_LEARNING_RATE,
                "weight_decay": OPTIMIZER_WEIGHT_DECAY,
                "betas": list(OPTIMIZER_BETAS),
                "eps": OPTIMIZER_EPS,
            }
        )

    def confirmation_plan(self) -> ConfirmationPlan:
        try:
            _PINNED_TRAINER_EXECUTION_GATE()
            _PINNED_SESSION_ASSERT_LIVE(self)
            if self._phase != "confirmation_pending":
                raise GraftPhaseAShortTrainingError("confirmation is not pending")
            row = self._confirmation_rows[self._backend.dp_arm]
            # The preregistered wrong-owner control stays within family: the held
            # out dog uses the fit dog atlas and the held-out human uses the fit
            # human atlas.  A cross-family dog<->human swap is not this control.
            wrong = self._update_rows[self._backend.dp_arm]
            if (
                row.optimizer_update_allowed
                or not row.optimizer_confirmation_only
                or not wrong.optimizer_update_allowed
                or wrong.optimizer_confirmation_only
                or row.source_sha256 == wrong.source_sha256
            ):
                raise GraftPhaseAShortTrainingError("confirmation routing differs")
            record = {
                "schema_version": "bernini-graft-phase-a-short-confirmation-plan-v1",
                "dp_arm": self._backend.dp_arm,
                "row_iid": row.iid,
                "wrong_owner_iid": wrong.iid,
                "schedule_indices": list(CONFIRMATION_SCHEDULE_INDICES),
                "field_roles": list(CONFIRMATION_FIELD_ROLES),
                "parameter_digest": self._post_training_parameter_digest,
                "optimizer_updates_allowed": False,
                "decoded_sampler_used": False,
            }
            return ConfirmationPlan._mint(
                session_token=self._token,
                dp_arm=self._backend.dp_arm,
                row=row,
                wrong_owner_row=wrong,
                row_iid=row.iid,
                wrong_owner_iid=wrong.iid,
                schedule_indices=CONFIRMATION_SCHEDULE_INDICES,
                field_roles=CONFIRMATION_FIELD_ROLES,
                parameter_digest=str(self._post_training_parameter_digest),
                plan_digest=_object_sha256(record),
            )
        except Exception as error:
            _PINNED_SESSION_FAIL(self, error)
            raise

    def _validate_confirmation_plan(self, plan: Any) -> None:
        _PINNED_TRAINER_EXECUTION_GATE()
        expected_record = None
        if type(plan) is ConfirmationPlan:
            expected_record = {
                "schema_version": "bernini-graft-phase-a-short-confirmation-plan-v1",
                "dp_arm": plan.dp_arm,
                "row_iid": plan.row_iid,
                "wrong_owner_iid": plan.wrong_owner_iid,
                "schedule_indices": list(plan.schedule_indices),
                "field_roles": list(plan.field_roles),
                "parameter_digest": plan.parameter_digest,
                "optimizer_updates_allowed": False,
                "decoded_sampler_used": False,
            }
        if (
            type(plan) is not ConfirmationPlan
            or plan._token is not _PLAN_TOKEN
            or plan._session_token is not self._token
            or plan.dp_arm != self._backend.dp_arm
            or plan.row is not self._confirmation_rows[self._backend.dp_arm]
            or plan.wrong_owner_row is not self._update_rows[self._backend.dp_arm]
            or plan.row_iid != plan.row.iid
            or plan.wrong_owner_iid != plan.wrong_owner_row.iid
            or plan.schedule_indices != CONFIRMATION_SCHEDULE_INDICES
            or plan.field_roles != CONFIRMATION_FIELD_ROLES
            or plan.parameter_digest != self._post_training_parameter_digest
            or plan.row.optimizer_update_allowed
            or not plan.row.optimizer_confirmation_only
            or not plan.wrong_owner_row.optimizer_update_allowed
            or plan.wrong_owner_row.optimizer_confirmation_only
            or expected_record is None
            or plan.plan_digest != _object_sha256(expected_record)
        ):
            raise GraftPhaseAShortTrainingError("confirmation plan differs")

    def record_confirmation_fields(
        self,
        *,
        plan: ConfirmationPlan,
        schedule_index: int,
        source_noop_target_velocity: torch.Tensor,
        correct_atlas_noop_velocity: torch.Tensor,
        wrong_atlas_noop_velocity: torch.Tensor,
        dropped_atlas_noop_velocity: torch.Tensor,
        correct_atlas_action_velocity: torch.Tensor,
        dropped_atlas_action_velocity: torch.Tensor,
    ) -> Mapping[str, Any]:
        """Authenticate six detached fields and apply one index-local gate."""

        try:
            _PINNED_TRAINER_EXECUTION_GATE()
            _PINNED_SESSION_ASSERT_LIVE(self)
            if self._phase != "confirmation_pending":
                raise GraftPhaseAShortTrainingError("confirmation state differs")
            _PINNED_SESSION_VALIDATE_CONFIRMATION_PLAN(self, plan)
            if (
                type(schedule_index) is not int
                or schedule_index not in CONFIRMATION_SCHEDULE_INDICES
                or schedule_index in self._confirmation_metrics
            ):
                raise GraftPhaseAShortTrainingError(
                    "confirmation schedule index is unknown or duplicated"
                )
            if torch.is_grad_enabled():
                raise GraftPhaseAShortTrainingError(
                    "confirmation must execute under torch.no_grad"
                )
            if any(parameter.grad is not None for _, parameter in self._named):
                raise GraftPhaseAShortTrainingError(
                    "confirmation cannot inherit a training gradient"
                )
            parameter_digest = _named_tensor_digest(self._named)
            base_digest = _named_tensor_digest(self._base)
            optimizer_digest = _PINNED_SESSION_OPTIMIZER_STATE_DIGEST(self)
            if (
                parameter_digest != self._post_training_parameter_digest
                or base_digest != self._post_training_base_digest
                or optimizer_digest != self._post_training_optimizer_digest
            ):
                raise GraftPhaseAShortTrainingError(
                    "confirmation mutated parameters, base, or optimizer"
                )
            metrics = _PINNED_CONFIRMATION_FIELD_METRICS(
                schedule_index=schedule_index,
                source_noop_target_velocity=source_noop_target_velocity,
                correct_atlas_noop_velocity=correct_atlas_noop_velocity,
                wrong_atlas_noop_velocity=wrong_atlas_noop_velocity,
                dropped_atlas_noop_velocity=dropped_atlas_noop_velocity,
                correct_atlas_action_velocity=correct_atlas_action_velocity,
                dropped_atlas_action_velocity=dropped_atlas_action_velocity,
            )
            record = {
                "row_iid": plan.row_iid,
                "wrong_owner_iid": plan.wrong_owner_iid,
                "schedule_index": schedule_index,
                "metrics_digest": metrics["digest"],
                "parameter_digest": parameter_digest,
                "base_digest": base_digest,
                "optimizer_digest": optimizer_digest,
            }
            consensus_digest = _object_sha256(record)
            self._backend.consensus(
                consensus_digest,
                scope="sp",
                label=f"confirmation index {schedule_index}",
            )
            if metrics["noncompensating_all_pass"] is not True:
                self._failed_confirmation_metrics = metrics
                raise GraftPhaseAShortTrainingError(
                    f"confirmation index {schedule_index} failed a noncompensating gate"
                )
            self._confirmation_metrics[schedule_index] = metrics
            self._confirmation_consensus[schedule_index] = consensus_digest
            if len(self._confirmation_metrics) == len(
                CONFIRMATION_SCHEDULE_INDICES
            ):
                self._phase = "ready_to_finalize"
            return _seal(
                {
                    "schema_version": (
                        "bernini-graft-phase-a-confirmation-field-admission-v1"
                    ),
                    **record,
                    "metrics": dict(metrics),
                    "sp4_consensus_digest": consensus_digest,
                    "no_grad": True,
                    "optimizer_update_performed": False,
                    "checkpoint_written": False,
                    **_false_authority(),
                }
            )
        except Exception as error:
            _PINNED_SESSION_FAIL(self, error)
            raise

    def finish(self) -> ShortTrainingResult:
        try:
            _PINNED_TRAINER_EXECUTION_GATE()
            _PINNED_SESSION_ASSERT_LIVE(self)
            validated_updates = [
                _validated_sealed_mapping(
                    row, label=f"stored update receipt {index + 1}"
                )
                for index, row in enumerate(self._update_receipts)
            ]
            validated_confirmation_metrics = {
                index: _validated_sealed_mapping(
                    self._confirmation_metrics[index],
                    label=f"stored confirmation metrics {index}",
                )
                for index in CONFIRMATION_SCHEDULE_INDICES
                if index in self._confirmation_metrics
            }
            if (
                self._phase != "ready_to_finalize"
                or len(validated_updates) != 2
                or set(self._confirmation_metrics)
                != set(CONFIRMATION_SCHEDULE_INDICES)
                or set(validated_confirmation_metrics)
                != set(CONFIRMATION_SCHEDULE_INDICES)
                or any(
                    row.get("noncompensating_all_pass") is not True
                    for row in validated_confirmation_metrics.values()
                )
            ):
                raise GraftPhaseAShortTrainingError("confirmation coverage is incomplete")
            parameter_digest = _named_tensor_digest(self._named)
            base_digest = _named_tensor_digest(self._base)
            optimizer_digest = _PINNED_SESSION_OPTIMIZER_STATE_DIGEST(self)
            if (
                parameter_digest != self._post_training_parameter_digest
                or base_digest != self._post_training_base_digest
                or base_digest != self._initial_base_digest
                or optimizer_digest != self._post_training_optimizer_digest
                or self._optimizer_steps != 2
                or any(parameter.grad is not None for _, parameter in self._named)
            ):
                raise GraftPhaseAShortTrainingError("final trainer state differs")
            self._backend.consensus(
                parameter_digest,
                scope="world",
                label="final trainable parameters",
            )
            self._backend.consensus(
                base_digest,
                scope="world",
                label="final frozen base",
            )
            routing_authority = (
                vars(self._routing.authority)
                if type(self._routing.authority) is source_consumer.AuthorityBoundary
                else dict(self._routing.authority)
            )
            receipt = _seal(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "completed_in_memory_orchestration",
                    "topology": {
                        "world_size": WORLD_SIZE,
                        "data_parallel_size": DP_SIZE,
                        "sequence_parallel_size": SP_SIZE,
                        "rank": self._backend.rank,
                        "dp_arm": self._backend.dp_arm,
                        "sp_rank": self._backend.sp_rank,
                    },
                    "dependency_pins": {
                        "consumer_commit": PINNED_CONSUMER_COMMIT,
                        "consumer_source_sha256": PINNED_CONSUMER_SOURCE_SHA256,
                        "native_v2_source_sha256": PINNED_NATIVE_V2_SOURCE_SHA256,
                        "native_v2_schema": native_v2.SCHEMA_VERSION,
                        "dependency_runtime_surface_sha256": (
                            PINNED_CRITICAL_RUNTIME_SURFACE_SHA256
                        ),
                        "trainer_execution_runtime_sha256": (
                            PINNED_TRAINER_EXECUTION_RUNTIME_SHA256
                        ),
                    },
                    "source_routing": {
                        "routing_digest": self._routing.routing_digest,
                        "source_release_result_digest": self._routing.source_release_result_digest,
                        "pinset_digest": self._routing.pinset_digest,
                        "path_free_owned_bytes_only": True,
                        "update_rows": 2,
                        "confirmation_rows": 2,
                        "local_update_iid": self._update_rows[self._backend.dp_arm].iid,
                        "local_confirmation_iid": self._confirmation_rows[self._backend.dp_arm].iid,
                        "confirmation_rows_consumed_by_optimizer": False,
                        "authority": routing_authority,
                    },
                    "optimizer_contract": {
                        "kind": "torch.optim.AdamW",
                        "learning_rate": OPTIMIZER_LEARNING_RATE,
                        "weight_decay": OPTIMIZER_WEIGHT_DECAY,
                        "betas": list(OPTIMIZER_BETAS),
                        "eps": OPTIMIZER_EPS,
                        "max_grad_norm": MAX_GRAD_NORM,
                        "steps": 2,
                        "schedule_indices": list(UPDATE_SCHEDULE_INDICES),
                        "regimes": list(UPDATE_REGIMES),
                        "gradient_collective_order": ["SP4_SUM_div_4", "DP2_SUM_div_2"],
                    },
                    "updates": validated_updates,
                    "confirmation": {
                        "plan": (
                            "per_row_per_index_noop_fm_relative_gain_plus_"
                            "action_delta_geometry"
                        ),
                        "row_iid": self._confirmation_rows[self._backend.dp_arm].iid,
                        "wrong_owner_iid": self._update_rows[self._backend.dp_arm].iid,
                        "wrong_owner_is_same_family_fit_row": True,
                        "schedule_indices": list(CONFIRMATION_SCHEDULE_INDICES),
                        "field_roles": list(CONFIRMATION_FIELD_ROLES),
                        "thresholds": {
                            "minimum_noop_fm_relative_gain": (
                                MIN_CONFIRMATION_RELATIVE_GAIN
                            ),
                            "minimum_action_delta_correct_drop_norm_ratio": (
                                MIN_ACTION_DELTA_CORRECT_DROP_NORM_RATIO
                            ),
                            "minimum_action_delta_cosine": MIN_ACTION_DELTA_COSINE,
                        },
                        "per_index_metrics": {
                            str(index): validated_confirmation_metrics[index]
                            for index in CONFIRMATION_SCHEDULE_INDICES
                        },
                        "sp4_consensus_digest": {
                            str(index): self._confirmation_consensus[index]
                            for index in CONFIRMATION_SCHEDULE_INDICES
                        },
                        "all_indices_noncompensating_hard_gate_passed": True,
                        "evaluated_under_no_grad": True,
                        "optimizer_state_unchanged": True,
                        "parameters_unchanged": True,
                        "frozen_base_unchanged": True,
                        "six_field_tensor_contract_and_metrics_authenticated_by_this_core": True,
                        "same_noise_x_sigma_coordinate_authenticated_by_this_core": False,
                        "field_origin_runtime_authenticated_by_this_core": False,
                        "runner_adapter_off_parity_indices": list(
                            RUNNER_ADAPTER_OFF_PARITY_INDICES
                        ),
                        "runner_adapter_off_parity_verified_by_this_core": False,
                        "runner_must_block_checkpoint_without_adapter_off_parity": True,
                    },
                    "initial_parameter_digest": self._initial_parameter_digest,
                    "final_parameter_digest": parameter_digest,
                    "initial_frozen_base_digest": self._initial_base_digest,
                    "final_frozen_base_digest": base_digest,
                    "optimizer_state_digest_after_training_and_confirmation": optimizer_digest,
                    "parameter_world_consensus": True,
                    "frozen_base_world_consensus": True,
                    "test_only": self._backend.test_only,
                    "real_model_loaded_by_this_core": False,
                    "source_bytes_to_latent_binding_verified_by_this_core": False,
                    "trainer_execution_runtime_live_verified": True,
                    "same_process_execution_integrity_formally_proven_by_this_core": False,
                    "same_process_formal_security_proven_by_this_core": False,
                    "full_sampler_used": False,
                    "decoded_media_used": False,
                    "checkpoint_written": False,
                    "checkpoint_payload_returned": False,
                    "publication_performed": False,
                    **_false_authority(),
                }
            )
            self._phase = "closed"
            return ShortTrainingResult(receipt=receipt)
        except Exception as error:
            _PINNED_SESSION_FAIL(self, error)
            raise

    def failure_receipt(self) -> Mapping[str, Any]:
        _PINNED_TRAINER_EXECUTION_GATE()
        if self._phase != "failed" or self._failure_reason is None:
            raise GraftPhaseAShortTrainingError("session has not failed")
        return _seal(
            {
                "schema_version": FAILURE_SCHEMA_VERSION,
                "status": "failed_rolled_back_no_checkpoint",
                "rank": self._backend.rank,
                "dp_arm": self._backend.dp_arm,
                "sp_rank": self._backend.sp_rank,
                "completed_optimizer_steps_before_failure": self._optimizer_steps,
                "failure_reason": self._failure_reason,
                "failed_confirmation_metrics": (
                    None
                    if self._failed_confirmation_metrics is None
                    else dict(self._failed_confirmation_metrics)
                ),
                "trainable_parameters_restored_to_initial_snapshot": (
                    _named_tensor_digest(self._named)
                    == self._initial_parameter_digest
                ),
                "dependency_runtime_surface_sha256": (
                    PINNED_CRITICAL_RUNTIME_SURFACE_SHA256
                ),
                "trainer_execution_runtime_sha256": (
                    PINNED_TRAINER_EXECUTION_RUNTIME_SHA256
                ),
                "trainer_execution_runtime_live_verified": True,
                "same_process_execution_integrity_formally_proven_by_this_core": False,
                "same_process_formal_security_proven_by_this_core": False,
                "checkpoint_written": False,
                "checkpoint_payload_returned": False,
                "publication_performed": False,
                **_false_authority(),
            }
        )


# These aliases are captured once from the source-defined exact classes.  They
# are used only after the module-level execution gate has checked the complete
# live trainer namespace, and keep cleanup off instance attribute lookup.
_PINNED_SESSION_ASSERT_LIVE = PhaseAShortTrainingSession._assert_live
_PINNED_SESSION_ASSERT_OPTIMIZER_LIVE = (
    PhaseAShortTrainingSession._assert_optimizer_live
)
_PINNED_SESSION_RESTORE_TRAINABLE_SNAPSHOT = (
    PhaseAShortTrainingSession._restore_trainable_snapshot
)
_PINNED_SESSION_FAIL = PhaseAShortTrainingSession._fail
_PINNED_SESSION_VALIDATE_UPDATE_PLAN = (
    PhaseAShortTrainingSession._validate_update_plan
)
_PINNED_SESSION_OPTIMIZER_STATE_DIGEST = (
    PhaseAShortTrainingSession._optimizer_state_digest
)
_PINNED_SESSION_VALIDATE_CONFIRMATION_PLAN = (
    PhaseAShortTrainingSession._validate_confirmation_plan
)
_PINNED_BACKEND_ASSERT_LIVE = AuthenticatedDP2SP4Backend.assert_live
_PINNED_ROUTING_RECORD = _routing_record
_PINNED_ROUTING_LIVE_FINGERPRINT = _routing_live_fingerprint
_PINNED_FROZEN_BASE_REGISTRY = _frozen_base_registry
_PINNED_REGISTRY = _registry
_PINNED_SYNCHRONIZE_DP2_SP4_GRADIENTS = _synchronize_dp2_sp4_gradients
_PINNED_VALIDATE_NATIVE_RESULT = _validate_native_result
_PINNED_CONFIRMATION_FIELD_METRICS = _confirmation_field_metrics


def _trainer_execution_critical_constants_record() -> Mapping[str, Any]:
    return {
        "schemas": {
            "trainer": SCHEMA_VERSION,
            "failure": FAILURE_SCHEMA_VERSION,
            "backend": BACKEND_SCHEMA_VERSION,
            "test_routing": TEST_ROUTING_SCHEMA_VERSION,
            "plan": PLAN_SCHEMA_VERSION,
        },
        "topology": {
            "world": WORLD_SIZE,
            "dp": DP_SIZE,
            "sp": SP_SIZE,
        },
        "updates": {
            "schedule_indices": list(UPDATE_SCHEDULE_INDICES),
            "regimes": list(UPDATE_REGIMES),
        },
        "optimizer": {
            "kind": "torch.optim.AdamW",
            "lr_hex": OPTIMIZER_LEARNING_RATE.hex(),
            "weight_decay_hex": OPTIMIZER_WEIGHT_DECAY.hex(),
            "max_grad_norm_hex": MAX_GRAD_NORM.hex(),
            "betas_hex": [value.hex() for value in OPTIMIZER_BETAS],
            "eps_hex": OPTIMIZER_EPS.hex(),
            "step_runtime_by_torch_version": [
                {
                    "torch_version": torch_version,
                    "module": runtime[0],
                    "qualname": runtime[1],
                    "source_path_suffix": list(runtime[2]),
                    "source_sha256": runtime[3],
                }
                for torch_version, runtime in sorted(
                    _PINNED_ADAMW_STEP_RUNTIME_BY_TORCH_VERSION.items()
                )
            ],
        },
        "tensor_digest": {
            "raw_reader": "pinned_ctypes.string_at",
            "ctypes_string_at_source_sha256_by_python_minor": [
                {
                    "python_minor": [major, minor],
                    "sha256": source_sha256,
                }
                for (major, minor), source_sha256 in sorted(
                    _PINNED_CTYPES_STRING_AT_SOURCE_SHA256_BY_PYTHON_MINOR.items()
                )
            ],
            "chunk_size_bytes": RAW_DIGEST_CHUNK_SIZE_BYTES,
            "empty_tensor_raw_reader_calls": 0,
            "owned_cpu_contiguous_clone": True,
            "storage_nbytes_must_equal_logical_bytes": True,
        },
        "dependency_pins": {
            "consumer_commit": PINNED_CONSUMER_COMMIT,
            "consumer_source_sha256": PINNED_CONSUMER_SOURCE_SHA256,
            "native_v2_source_sha256": PINNED_NATIVE_V2_SOURCE_SHA256,
            "dependency_runtime_surface_sha256": (
                PINNED_CRITICAL_RUNTIME_SURFACE_SHA256
            ),
        },
        "confirmation": {
            "schedule_indices": list(CONFIRMATION_SCHEDULE_INDICES),
            "field_roles": list(CONFIRMATION_FIELD_ROLES),
            "minimum_relative_gain_hex": (
                MIN_CONFIRMATION_RELATIVE_GAIN.hex()
            ),
            "minimum_norm_ratio_hex": (
                MIN_ACTION_DELTA_CORRECT_DROP_NORM_RATIO.hex()
            ),
            "minimum_cosine_hex": MIN_ACTION_DELTA_COSINE.hex(),
            "float64_tiny_hex": FLOAT64_TINY.hex(),
            "runner_adapter_off_parity_indices": list(
                RUNNER_ADAPTER_OFF_PARITY_INDICES
            ),
        },
        "authority_fields": list(AUTHORITY_FIELDS),
        "forbidden_public_input_fragments": list(
            _FORBIDDEN_PUBLIC_INPUT_FRAGMENTS
        ),
        "opaque_capabilities": {
            "backend_token_type": (
                f"{type(_BACKEND_TOKEN).__module__}."
                f"{type(_BACKEND_TOKEN).__qualname__}"
            ),
            "plan_token_type": (
                f"{type(_PLAN_TOKEN).__module__}."
                f"{type(_PLAN_TOKEN).__qualname__}"
            ),
            "session_token_type": (
                f"{type(_SESSION_TOKEN).__module__}."
                f"{type(_SESSION_TOKEN).__qualname__}"
            ),
            "pairwise_distinct": (
                len({_BACKEND_TOKEN, _PLAN_TOKEN, _SESSION_TOKEN}) == 3
            ),
        },
        "process_local_identity_contract_names": {
            "imports": sorted(
                globals().get("_PINNED_TRAINER_IMPORT_IDENTITIES", {})
            ),
            "module_functions": sorted(
                globals().get(
                    "_PINNED_TRAINER_MODULE_FUNCTION_IDENTITIES", {}
                )
            ),
            "module_classes": sorted(
                globals().get("_PINNED_TRAINER_MODULE_CLASS_IDENTITIES", {})
            ),
        },
        "public_exports": list(globals().get("__all__", ())),
    }


def _trainer_execution_runtime_record() -> Mapping[str, Any]:
    module_name = __name__
    module_functions = {
        name: _runtime_object_contract(value)
        for name, value in sorted(globals().items())
        if inspect.isfunction(value) and value.__module__ == module_name
    }
    module_classes = {
        name: _runtime_object_contract(value)
        for name, value in sorted(globals().items())
        if inspect.isclass(value) and value.__module__ == module_name
    }
    execution_classes = {
        cls.__name__: {
            "contract": _runtime_object_contract(cls),
            "class_dict_names": sorted(cls.__dict__),
            "mro": [
                f"{base.__module__}.{base.__qualname__}"
                for base in cls.__mro__
            ],
        }
        for cls in (
            AuthenticatedDP2SP4Backend,
            PhaseAShortTrainingSession,
        )
    }
    return {
        "schema_version": (
            "bernini-graft-phase-a-short-trainer-execution-runtime-v1"
        ),
        "module_name": module_name,
        "module_source_file": Path(__file__).resolve().name,
        "module_function_names": sorted(module_functions),
        "module_functions": module_functions,
        "module_class_names": sorted(module_classes),
        "module_classes": module_classes,
        "exact_execution_classes": execution_classes,
        "critical_constants": _trainer_execution_critical_constants_record(),
    }


def _trainer_execution_runtime_sha256() -> str:
    return _object_sha256(_trainer_execution_runtime_record())


def _execution_descriptor_code_identities(cls: type) -> Mapping[str, tuple[Any, ...]]:
    result = {}
    for name in sorted(cls.__dict__):
        descriptor = inspect.getattr_static(cls, name)
        functions = ()
        if inspect.isfunction(descriptor):
            functions = (descriptor,)
        elif isinstance(descriptor, (classmethod, staticmethod)):
            functions = (descriptor.__func__,)
        elif isinstance(descriptor, property):
            functions = tuple(
                function
                for function in (
                    descriptor.fget,
                    descriptor.fset,
                    descriptor.fdel,
                )
                if function is not None
            )
        if functions:
            result[name] = tuple(function.__code__ for function in functions)
    return MappingProxyType(result)


def _assert_pinned_trainer_execution_runtime() -> None:
    expected_imports = globals().get("_PINNED_TRAINER_IMPORT_IDENTITIES")
    expected_functions = globals().get(
        "_PINNED_TRAINER_MODULE_FUNCTION_IDENTITIES"
    )
    expected_classes = globals().get("_PINNED_TRAINER_MODULE_CLASS_IDENTITIES")
    expected_function_codes = globals().get(
        "_PINNED_TRAINER_MODULE_FUNCTION_CODE_IDENTITIES"
    )
    expected_session_dict = globals().get(
        "_PINNED_TRAINER_SESSION_CLASS_DICT_IDENTITIES"
    )
    expected_backend_dict = globals().get(
        "_PINNED_TRAINER_BACKEND_CLASS_DICT_IDENTITIES"
    )
    expected_session_codes = globals().get(
        "_PINNED_TRAINER_SESSION_DESCRIPTOR_CODE_IDENTITIES"
    )
    expected_backend_codes = globals().get(
        "_PINNED_TRAINER_BACKEND_DESCRIPTOR_CODE_IDENTITIES"
    )
    expected_session_mro = globals().get("_PINNED_TRAINER_SESSION_MRO")
    expected_backend_mro = globals().get("_PINNED_TRAINER_BACKEND_MRO")
    module_name = __name__
    live_functions = {
        name: value
        for name, value in globals().items()
        if inspect.isfunction(value) and value.__module__ == module_name
    }
    live_classes = {
        name: value
        for name, value in globals().items()
        if inspect.isclass(value) and value.__module__ == module_name
    }
    identity_contracts = (
        expected_imports,
        expected_functions,
        expected_classes,
        expected_function_codes,
        expected_session_dict,
        expected_backend_dict,
        expected_session_codes,
        expected_backend_codes,
    )
    if (
        any(type(value) is not MappingProxyType for value in identity_contracts)
        or any(
            globals().get(name) is not value
            for name, value in expected_imports.items()
        )
        or set(live_functions) != set(expected_functions)
        or any(
            live_functions.get(name) is not value
            for name, value in expected_functions.items()
        )
        or set(live_functions) != set(expected_function_codes)
        or any(
            live_functions[name].__code__ is not code
            for name, code in expected_function_codes.items()
        )
        or set(live_classes) != set(expected_classes)
        or any(
            live_classes.get(name) is not value
            for name, value in expected_classes.items()
        )
        or PhaseAShortTrainingSession.__mro__ != expected_session_mro
        or AuthenticatedDP2SP4Backend.__mro__ != expected_backend_mro
        or set(PhaseAShortTrainingSession.__dict__) != set(expected_session_dict)
        or any(
            PhaseAShortTrainingSession.__dict__.get(name) is not value
            for name, value in expected_session_dict.items()
        )
        or set(AuthenticatedDP2SP4Backend.__dict__) != set(expected_backend_dict)
        or any(
            AuthenticatedDP2SP4Backend.__dict__.get(name) is not value
            for name, value in expected_backend_dict.items()
        )
        or _execution_descriptor_code_identities(PhaseAShortTrainingSession)
        != expected_session_codes
        or _execution_descriptor_code_identities(AuthenticatedDP2SP4Backend)
        != expected_backend_codes
    ):
        raise GraftPhaseAShortTrainingError(
            "live short-trainer execution runtime differs"
        )
    _PINNED_ASSERT_CTYPES_RUNTIME()
    if (
        _trainer_execution_runtime_sha256()
        != PINNED_TRAINER_EXECUTION_RUNTIME_SHA256
    ):
        raise GraftPhaseAShortTrainingError(
            "live short-trainer execution runtime differs"
        )


_PINNED_TRAINER_EXECUTION_GATE = _assert_pinned_trainer_execution_runtime


def open_authenticated_short_training(
    *,
    routing: source_consumer.TrainerRouting,
    bindings: native_v2.AuthenticatedNativeBindings,
    collectives: AuthenticatedDP2SP4Backend,
) -> PhaseAShortTrainingSession:
    """Open the in-memory core; no model loading or artifact path is accepted."""

    _PINNED_TRAINER_EXECUTION_GATE()
    return PhaseAShortTrainingSession._mint(
        routing=routing,
        bindings=bindings,
        backend=collectives,
    )


__all__ = [
    "AUTHORITY_FIELDS",
    "AuthenticatedDP2SP4Backend",
    "BACKEND_SCHEMA_VERSION",
    "CONFIRMATION_FIELD_ROLES",
    "CONFIRMATION_SCHEDULE_INDICES",
    "ConfirmationPlan",
    "DP_SIZE",
    "FAILURE_SCHEMA_VERSION",
    "GraftPhaseAShortTrainingError",
    "MAX_GRAD_NORM",
    "MIN_ACTION_DELTA_CORRECT_DROP_NORM_RATIO",
    "MIN_ACTION_DELTA_COSINE",
    "MIN_CONFIRMATION_RELATIVE_GAIN",
    "OPTIMIZER_LEARNING_RATE",
    "OPTIMIZER_WEIGHT_DECAY",
    "PINNED_CONSUMER_COMMIT",
    "PINNED_CONSUMER_SOURCE_SHA256",
    "PINNED_CRITICAL_RUNTIME_SURFACE_SHA256",
    "PINNED_NATIVE_V2_SOURCE_SHA256",
    "PINNED_TRAINER_EXECUTION_RUNTIME_SHA256",
    "PhaseAShortTrainingSession",
    "SCHEMA_VERSION",
    "SP_SIZE",
    "ShortTrainingResult",
    "RUNNER_ADAPTER_OFF_PARITY_INDICES",
    "UPDATE_REGIMES",
    "UPDATE_SCHEDULE_INDICES",
    "UpdateCellPlan",
    "WORLD_SIZE",
    "authenticate_cpu_test_collectives",
    "authenticate_cpu_test_routing",
    "authenticate_torch_distributed_world8_dp2sp4",
    "open_authenticated_short_training",
]


# Live object identity complements the cross-runtime structural digest.  The
# digest is portable across the pinned Python minors; these immutable maps are
# process-local and ensure that an equivalent-looking replacement is rejected
# before a transition calls it.  Receipts nevertheless make no formal claim
# about integrity against an actor able to rewrite arbitrary process memory.
_PINNED_TRAINER_IMPORT_IDENTITIES = MappingProxyType(
    {
        "dataclass": dataclass,
        "ctypes": ctypes,
        "field": field,
        "hashlib": hashlib,
        "inspect": inspect,
        "json": json,
        "math": math,
        "Path": Path,
        "sys": sys,
        "MappingProxyType": MappingProxyType,
        "Any": Any,
        "Callable": Callable,
        "Mapping": Mapping,
        "Optional": Optional,
        "Sequence": Sequence,
        "torch": torch,
        "source_consumer": source_consumer,
        "native_v2": native_v2,
        "_PINNED_CTYPES_STRING_AT": _PINNED_CTYPES_STRING_AT,
        "_PINNED_CTYPES_STRING_AT_CODE": _PINNED_CTYPES_STRING_AT_CODE,
        "_PINNED_CTYPES_RAW_STRING_AT": _PINNED_CTYPES_RAW_STRING_AT,
        "_PINNED_CTYPES_RAW_STRING_AT_TYPE": (
            _PINNED_CTYPES_RAW_STRING_AT_TYPE
        ),
        "_PINNED_CTYPES_C_VOID_P": _PINNED_CTYPES_C_VOID_P,
        "_PINNED_CTYPES_C_INT": _PINNED_CTYPES_C_INT,
        "_PINNED_CTYPES_PY_OBJECT": _PINNED_CTYPES_PY_OBJECT,
        "_PINNED_CTYPES_STRING_AT_SOURCE_SHA256_BY_PYTHON_MINOR": (
            _PINNED_CTYPES_STRING_AT_SOURCE_SHA256_BY_PYTHON_MINOR
        ),
        "_PINNED_ADAMW_STEP_RUNTIME_BY_TORCH_VERSION": (
            _PINNED_ADAMW_STEP_RUNTIME_BY_TORCH_VERSION
        ),
        "_PINNED_TORCH_VERSION": _PINNED_TORCH_VERSION,
        "_PINNED_TORCH_VERSION_STRING": _PINNED_TORCH_VERSION_STRING,
        "_PINNED_PYTHON_RUNTIME_MINOR": _PINNED_PYTHON_RUNTIME_MINOR,
    }
)
_PINNED_TRAINER_MODULE_FUNCTION_IDENTITIES = MappingProxyType(
    {
        name: value
        for name, value in globals().items()
        if inspect.isfunction(value) and value.__module__ == __name__
    }
)
_PINNED_TRAINER_MODULE_CLASS_IDENTITIES = MappingProxyType(
    {
        name: value
        for name, value in globals().items()
        if inspect.isclass(value) and value.__module__ == __name__
    }
)
_PINNED_TRAINER_MODULE_FUNCTION_CODE_IDENTITIES = MappingProxyType(
    {
        name: function.__code__
        for name, function in _PINNED_TRAINER_MODULE_FUNCTION_IDENTITIES.items()
    }
)
_PINNED_TRAINER_SESSION_MRO = PhaseAShortTrainingSession.__mro__
_PINNED_TRAINER_BACKEND_MRO = AuthenticatedDP2SP4Backend.__mro__
_PINNED_TRAINER_SESSION_CLASS_DICT_IDENTITIES = MappingProxyType(
    dict(PhaseAShortTrainingSession.__dict__)
)
_PINNED_TRAINER_BACKEND_CLASS_DICT_IDENTITIES = MappingProxyType(
    dict(AuthenticatedDP2SP4Backend.__dict__)
)
_PINNED_TRAINER_SESSION_DESCRIPTOR_CODE_IDENTITIES = (
    _execution_descriptor_code_identities(PhaseAShortTrainingSession)
)
_PINNED_TRAINER_BACKEND_DESCRIPTOR_CODE_IDENTITIES = (
    _execution_descriptor_code_identities(AuthenticatedDP2SP4Backend)
)
