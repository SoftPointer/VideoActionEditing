#!/usr/bin/env python3
"""Post-bootstrap extension of the pinned Phase-A native closure.

Version 2 deliberately reuses version 1 for the native measurement, vendor
APG, FP32 flow-matching objective, BF16 raw cotangents, serial replay, pack
leaf handling, immutable-input checks, and failure rollback.  The only
execution seam replaced here is the rank-local trainable-gradient gate:

* an exact all-zero live output-projection byte state is ``bootstrap`` and
  continues to require output-only gradients;
* every other finite, numerically non-zero live output-projection state is
  ``post_bootstrap`` and requires finite non-zero atlas, query, key, value,
  and output gradient deltas on every target-owning replay branch;
* an active source-only SP rank requires exact-zero deltas in all five
  categories in either regime.

The regime is derived only from parameters in the authenticated v1 trainable
registry.  It is not a constructor, execute-API, CLI, environment, or receipt
input.  Every negative-zero element is rejected, including a negative zero
mixed into an otherwise non-zero output tensor.

This module wraps a private v1 extension seam, so the complete v1 source file
is SHA256 pinned.  A source drift fails closed before a v2 session is opened.
The v2 receipt retains the underlying v1 receipt digest and every v1 denial of
optimizer, CUDA, quality, and scientific authority.

CPU authenticated fakes can close the algebra and state machine.  They do not
establish these remaining GPU facts: official checkpoint/runtime binding,
post-update CUDA replay parity, real SP4 local-before-collective gradients,
optimizer-state/update correctness, exact40 coverage, or useful training.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import struct
import sys
from types import MappingProxyType
import types
import typing
from typing import Any, Mapping

import torch

import graft_phase_a_native_training_closure_v1 as v1
import inference_sigma_strata as pinned_sigma_strata


SCHEMA_VERSION = "bernini-graft-phase-a-native-training-closure-v2"
LOCAL_GRADIENT_SCHEMA_VERSION = (
    "bernini-graft-phase-a-local-trainable-gradient-delta-v2"
)
PINNED_V1_SOURCE_SHA256 = (
    "36861e8670fc77d65b469f86aa472a10d453f9ba5fc227a62303329c4c38409a"
)
PINNED_SIGMA_STRATA_SOURCE_SHA256 = (
    "e3782a22130c09a48dc3ea27fa219af6caca445e1fce2c8f3bca7cde6058afd3"
)
PINNED_SIGMA_SCHEDULE_SHA256 = (
    "3e5ad4473d133318026cc9e8f32399782bf06313691b58870c89d9c4c87c3d03"
)
PINNED_SIGMA_STRATA_EXPORTS = (
    "FLOW_SHIFT",
    "InferenceSigmaStrataError",
    "NUM_INFERENCE_STEPS",
    "NUM_TRAIN_TIMESTEPS",
    "PINNED_POSITIVE_SIGMAS",
    "PINNED_POSITIVE_SIGMA_FLOAT32_HEX",
    "PINNED_TIMESTEPS",
    "RECEIPT_SCHEMA",
    "SCHEDULE_SCHEMA",
    "SCHEDULE_SHA256",
    "SigmaStratum",
    "assert_selected_timestep_sigma",
    "audit_runtime_unipc_schedule",
    "build_sigma_strata_receipt",
    "histogram_for_optimizer_range",
    "select_sigma_stratum",
)
# Filled from the deterministic complete-namespace contract below, never from
# the already-imported live v1 objects.  The bytecode portion is necessarily
# Python-minor-specific.  Unsupported interpreter minors fail closed.
# The 3.12 rows were independently reproduced twice with AUH vace Python
# 3.12.13, executable SHA256
# 8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a.
PINNED_V1_RUNTIME_NAMESPACE_SHA256 = MappingProxyType(
    {
        (3, 8): "271bf16176d35041f120e2ea32fc32834ae57d656ffd28934a3a4f3e6367594e",
        (3, 10): "8b392138361154bbbf71d72061a6d27b7a932534c047a831363cf563d7e955b2",
        (3, 12): "1af7b8ec856fc50adc0b2d1b938eb5b60650809f60cdabe08fdbf040b0731913",
    }
)
PINNED_SIGMA_RUNTIME_NAMESPACE_SHA256 = MappingProxyType(
    {
        (3, 8): "93d7c91ca5ad15b9b68ed033672e0fb1ee0886e198e1c94ceaa4b27bf42362dd",
        (3, 10): "c1c2d85d5e77b88661e6963347b8a2325e786203f7519bd03863eca0de60c961",
        (3, 12): "a8ee8f4b5b4ad1d15e2d340c71a15e2e8808d0143661ebb51b41bc3a6189c50c",
    }
)
TRAINING_REGIMES = ("bootstrap", "post_bootstrap")
GRADIENT_CATEGORIES = (
    "atlas_encoder",
    "query_projection",
    "key_projection",
    "value_projection",
    "output_projection",
)
UPSTREAM_GRADIENT_CATEGORIES = GRADIENT_CATEGORIES[:-1]
REMAINING_GPU_ASSUMPTIONS = (
    "official_checkpoint_and_runtime_content_bound_by_a_gpu_runner",
    "post_update_cuda_measurement_replay_raw_bf16_parity",
    "real_sp4_rank_local_gradients_observed_before_any_collective",
    "optimizer_state_update_and_next_step_input_freshness",
    "exact40_schedule_coverage_outside_this_one_cell_core",
    "training_quality_and_action_editing_utility",
)

GraftPhaseANativeTrainingClosureError = v1.GraftPhaseANativeTrainingClosureError
AuthenticatedNativeBindings = v1.AuthenticatedNativeBindings
NativeFieldMeasurement = v1.NativeFieldMeasurement
NativePhaseAFlowMatchingVJP = v1.NativePhaseAFlowMatchingVJP
NativeForwardContextObservation = v1.NativeForwardContextObservation
NativeForwardContextRequest = v1.NativeForwardContextRequest
NativeTrainingClosureResult = v1.NativeTrainingClosureResult
APG_ETA = v1.APG_ETA
APG_MOMENTUM = v1.APG_MOMENTUM
APG_NORM_THRESHOLD = v1.APG_NORM_THRESHOLD
BRANCH_ORDER = v1.BRANCH_ORDER
EXPECTED_FRAMES = v1.EXPECTED_FRAMES
EXPECTED_LATENT_PHASES = v1.EXPECTED_LATENT_PHASES
EXPECTED_PATCH_SOURCE_IDS = v1.EXPECTED_PATCH_SOURCE_IDS
FLOW_MATCHING_OBJECTIVE = v1.FLOW_MATCHING_OBJECTIVE
FLOW_MATCHING_REDUCTION = v1.FLOW_MATCHING_REDUCTION
FORWARD_ROUTE_SCHEMA_VERSION = v1.FORWARD_ROUTE_SCHEMA_VERSION
GUIDANCE_MODE = v1.GUIDANCE_MODE
GUIDANCE_SCALE = v1.GUIDANCE_SCALE
PHASE_A_ACTIVE_SCHEDULE_INDICES = v1.PHASE_A_ACTIVE_SCHEDULE_INDICES
PINNED_BERNINI_COMMIT = v1.PINNED_BERNINI_COMMIT
PINNED_TRANSFORMER_WAN_SHA256 = v1.PINNED_TRANSFORMER_WAN_SHA256
PINNED_WAN_DIFFUSION_SHA256 = v1.PINNED_WAN_DIFFUSION_SHA256
authenticate_cpu_test_fakes = v1.authenticate_cpu_test_fakes
authenticate_pinned_native_bindings = v1.authenticate_pinned_native_bindings
build_native_forward_context_observation = (
    v1.build_native_forward_context_observation
)
unpack_wan_target_velocity = v1.unpack_wan_target_velocity


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise GraftPhaseANativeTrainingClosureError(
            "pinned v1 closure source cannot be read"
        ) from error
    return digest.hexdigest()


def _code_contract(code: types.CodeType) -> Mapping[str, Any]:
    def constant(value: Any) -> Any:
        if isinstance(value, types.CodeType):
            return {"code": _code_contract(value)}
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            return {"float_hex": value.hex()}
        if isinstance(value, complex):
            return {
                "complex_hex": [value.real.hex(), value.imag.hex()]
            }
        if isinstance(value, bytes):
            return {"bytes_hex": value.hex()}
        if isinstance(value, tuple):
            return {"tuple": [constant(item) for item in value]}
        if value is Ellipsis:
            return {"ellipsis": True}
        return {
            "opaque_constant_type": (
                f"{type(value).__module__}.{type(value).__qualname__}"
            )
        }

    return {
        "argcount": code.co_argcount,
        "posonlyargcount": getattr(code, "co_posonlyargcount", 0),
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "code_hex": code.co_code.hex(),
        "constants": [constant(value) for value in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "name": code.co_name,
        "qualname": getattr(code, "co_qualname", code.co_name),
        "firstlineno": code.co_firstlineno,
        "line_table_hex": getattr(code, "co_linetable", code.co_lnotab).hex(),
        "exception_table_hex": getattr(code, "co_exceptiontable", b"").hex(),
    }


def _plain_runtime_value(value: Any, *, module_name: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, complex):
        return {"complex_hex": [value.real.hex(), value.imag.hex()]}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, tuple):
        return {
            "tuple": [
                _plain_runtime_value(item, module_name=module_name)
                for item in value
            ]
        }
    if isinstance(value, list):
        return {
            "list": [
                _plain_runtime_value(item, module_name=module_name)
                for item in value
            ]
        }
    if isinstance(value, (dict, MappingProxyType)):
        return {
            "mapping": [
                [
                    str(key),
                    _plain_runtime_value(value[key], module_name=module_name),
                ]
                for key in sorted(value, key=str)
            ]
        }
    if isinstance(value, types.ModuleType):
        return {"module": value.__name__}
    if isinstance(value, types.FunctionType):
        if value.__module__ == module_name:
            return {"defined_function": _function_contract(value, module_name)}
        return {
            "external_function": [value.__module__, value.__qualname__]
        }
    if isinstance(value, type):
        if value.__module__ == module_name:
            return {"defined_class": _class_contract(value, module_name)}
        return {"external_class": [value.__module__, value.__qualname__]}
    return {
        "external_object": [
            type(value).__module__,
            type(value).__qualname__,
            str(value),
        ]
    }


def _function_contract(
    function: types.FunctionType, module_name: str
) -> Mapping[str, Any]:
    closure_rows = []
    for cell in function.__closure__ or ():
        try:
            cell_value = cell.cell_contents
        except ValueError:
            closure_rows.append({"empty_cell": True})
            continue
        if isinstance(cell_value, types.FunctionType):
            closure_rows.append(
                {
                    "function_code": _code_contract(cell_value.__code__),
                    "name": cell_value.__qualname__,
                }
            )
        elif isinstance(cell_value, type):
            closure_rows.append(
                {
                    "class_reference": [
                        (
                            "defined_module_namespace"
                            if cell_value.__module__ == module_name
                            else cell_value.__module__
                        ),
                        cell_value.__qualname__,
                    ]
                }
            )
        else:
            closure_rows.append(
                _plain_runtime_value(cell_value, module_name=module_name)
            )
    return {
        "name": function.__name__,
        "qualname": function.__qualname__,
        "code": _code_contract(function.__code__),
        "defaults": _plain_runtime_value(
            function.__defaults__, module_name=module_name
        ),
        "kwdefaults": _plain_runtime_value(
            function.__kwdefaults__, module_name=module_name
        ),
        "annotations": _plain_runtime_value(
            function.__annotations__, module_name=module_name
        ),
        "signature": str(inspect.signature(function)),
        "globals_owner": (
            "defined_module_namespace"
            if function.__globals__.get("__name__") == module_name
            else str(function.__globals__.get("__name__"))
        ),
        "closure": closure_rows,
    }


def _class_contract(cls: type, module_name: str) -> Mapping[str, Any]:
    attributes = {}
    for name in sorted(cls.__dict__):
        value = inspect.getattr_static(cls, name)
        if isinstance(value, types.FunctionType):
            attributes[name] = {
                "function": _function_contract(value, module_name)
            }
        elif isinstance(value, staticmethod):
            attributes[name] = {
                "staticmethod": _function_contract(value.__func__, module_name)
            }
        elif isinstance(value, classmethod):
            attributes[name] = {
                "classmethod": _function_contract(value.__func__, module_name)
            }
        elif isinstance(value, property):
            attributes[name] = {
                "property": {
                    label: (
                        _function_contract(function, module_name)
                        if function is not None
                        else None
                    )
                    for label, function in (
                        ("get", value.fget),
                        ("set", value.fset),
                        ("delete", value.fdel),
                    )
                }
            }
        elif name == "__dataclass_fields__":
            attributes[name] = {
                field_name: {
                    "type": str(field_value.type),
                    "init": field_value.init,
                    "repr": field_value.repr,
                    "hash": field_value.hash,
                    "compare": field_value.compare,
                    "kw_only": getattr(field_value, "kw_only", False),
                    "default_type": (
                        f"{type(field_value.default).__module__}."
                        f"{type(field_value.default).__qualname__}"
                    ),
                    "factory_type": (
                        f"{type(field_value.default_factory).__module__}."
                        f"{type(field_value.default_factory).__qualname__}"
                    ),
                }
                for field_name, field_value in sorted(value.items())
            }
        elif name in ("__dict__", "__weakref__"):
            attributes[name] = {
                "descriptor_type": (
                    f"{type(value).__module__}.{type(value).__qualname__}"
                )
            }
        elif name == "__module__":
            attributes[name] = "defined_module_namespace"
        else:
            attributes[name] = _plain_runtime_value(
                value, module_name=module_name
            )
    return {
        "name": cls.__name__,
        "qualname": cls.__qualname__,
        "bases": [
            [base.__module__, base.__qualname__] for base in cls.__bases__
        ],
        "attributes": attributes,
    }


def _runtime_namespace_sha256(module: types.ModuleType) -> str:
    namespace = vars(module)
    module_name = module.__name__
    names = sorted(name for name in namespace if not name.startswith("__"))
    rows = {}
    for name in names:
        value = namespace[name]
        if name == "_BINDING_TOKEN":
            rows[name] = {
                "opaque_capability_type": (
                    f"{type(value).__module__}.{type(value).__qualname__}"
                )
            }
        else:
            rows[name] = _plain_runtime_value(value, module_name=module_name)
    encoded = json.dumps(
        {"names": names, "rows": rows},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _v1_runtime_namespace_sha256() -> str:
    return _runtime_namespace_sha256(v1)


def _sigma_runtime_namespace_sha256() -> str:
    return _runtime_namespace_sha256(pinned_sigma_strata)


def _expected_runtime_namespace_sha256(
    pins: Mapping[tuple[int, int], str], *, label: str
) -> str:
    python_minor = (sys.version_info.major, sys.version_info.minor)
    expected = pins.get(python_minor)
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or expected != expected.lower()
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise GraftPhaseANativeTrainingClosureError(
            f"unsupported Python runtime minor for {label}: "
            f"{python_minor[0]}.{python_minor[1]}"
        )
    return expected


def _assert_execution_import_identities() -> None:
    expected = {
        "hashlib": hashlib,
        "importlib": importlib,
        "inspect": inspect,
        "json": json,
        "math": math,
        "sigma_strata": pinned_sigma_strata,
        "struct": struct,
        "torch": torch,
        "nullcontext": contextlib.nullcontext,
        "dataclass": dataclasses.dataclass,
        "field": dataclasses.field,
        "Path": Path,
        "MappingProxyType": MappingProxyType,
        "Any": typing.Any,
        "Callable": typing.Callable,
        "ContextManager": typing.ContextManager,
        "Mapping": typing.Mapping,
        "Optional": typing.Optional,
        "Sequence": typing.Sequence,
    }
    if any(getattr(v1, name, None) is not value for name, value in expected.items()):
        raise GraftPhaseANativeTrainingClosureError(
            "live v1 execution import identity differs"
        )


def _assert_pinned_sigma_runtime() -> None:
    source = Path(pinned_sigma_strata.__file__).resolve()
    expected_runtime = _expected_runtime_namespace_sha256(
        PINNED_SIGMA_RUNTIME_NAMESPACE_SHA256,
        label="local sigma",
    )
    try:
        positive_hex = tuple(
            pinned_sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX
        )
        timesteps = tuple(pinned_sigma_strata.PINNED_TIMESTEPS)
        positive_sigmas = tuple(pinned_sigma_strata.PINNED_POSITIVE_SIGMAS)
        terminal_hex = pinned_sigma_strata.TERMINAL_SIGMA_FLOAT32_HEX
        schedule_payload = {
            "schema_version": pinned_sigma_strata.SCHEDULE_SCHEMA,
            "scheduler_class": pinned_sigma_strata.SCHEDULER_CLASS,
            "num_train_timesteps": pinned_sigma_strata.NUM_TRAIN_TIMESTEPS,
            "num_inference_steps": pinned_sigma_strata.NUM_INFERENCE_STEPS,
            "flow_shift_float64_hex": pinned_sigma_strata.FLOW_SHIFT.hex(),
            "timesteps_int64": list(timesteps),
            "positive_sigmas_float32_be_hex": list(positive_hex),
            "terminal_sigma_float32_be_hex": terminal_hex,
        }
        recomputed_schedule = hashlib.sha256(
            _canonical_json_bytes(schedule_payload)
        ).hexdigest()
        decoded_sigmas = tuple(
            float(struct.unpack(">f", bytes.fromhex(value))[0])
            for value in positive_hex
        )
    except Exception as error:
        raise GraftPhaseANativeTrainingClosureError(
            "pinned local sigma runtime is incomplete"
        ) from error
    if (
        v1.sigma_strata is not pinned_sigma_strata
        or not source.is_file()
        or _source_sha256(source) != PINNED_SIGMA_STRATA_SOURCE_SHA256
        or _sigma_runtime_namespace_sha256() != expected_runtime
        or tuple(pinned_sigma_strata.__all__) != PINNED_SIGMA_STRATA_EXPORTS
        or pinned_sigma_strata.SCHEDULE_SHA256 != PINNED_SIGMA_SCHEDULE_SHA256
        or recomputed_schedule != PINNED_SIGMA_SCHEDULE_SHA256
        or len(timesteps) != 40
        or len(positive_hex) != 40
        or positive_sigmas != decoded_sigmas
        or terminal_hex != "00000000"
    ):
        raise GraftPhaseANativeTrainingClosureError(
            "local sigma source/runtime/export/schedule contract differs"
        )


def _assert_pinned_v1_kernel() -> None:
    source = Path(v1.__file__).resolve()
    expected_runtime = _expected_runtime_namespace_sha256(
        PINNED_V1_RUNTIME_NAMESPACE_SHA256,
        label="wrapped v1",
    )
    _assert_execution_import_identities()
    _assert_pinned_sigma_runtime()
    if (
        not source.is_file()
        or _source_sha256(source) != PINNED_V1_SOURCE_SHA256
        or _v1_runtime_namespace_sha256() != expected_runtime
    ):
        raise GraftPhaseANativeTrainingClosureError(
            "live v1 execution namespace differs from the pinned runtime contract"
        )


def _gradient_category_from_name(name: str) -> str:
    if name.startswith("atlas_encoder."):
        return "atlas_encoder"
    for projection in ("query", "key", "value", "output"):
        if name.endswith(f".identity_rebinder.{projection}.weight"):
            return f"{projection}_projection"
    raise GraftPhaseANativeTrainingClosureError(
        f"v2 trainable gradient category is unknown: {name}"
    )


def _owned_raw_bytes(value: torch.Tensor) -> bytes:
    owned = value.detach().cpu().contiguous().clone()
    if type(owned) is not torch.Tensor or int(owned.numel()) <= 0:
        raise GraftPhaseANativeTrainingClosureError(
            "output projection is not a non-empty exact torch.Tensor"
        )
    storage = owned.untyped_storage()
    expected = int(owned.numel()) * int(owned.element_size())
    if int(storage.nbytes()) != expected:
        raise GraftPhaseANativeTrainingClosureError(
            "output projection storage exceeds its logical tensor"
        )
    raw = bytes(storage)
    if len(raw) != expected:
        raise GraftPhaseANativeTrainingClosureError(
            "output projection raw-byte length differs"
        )
    return raw


def _live_output_weight_state(
    bindings: AuthenticatedNativeBindings,
) -> Mapping[str, Any]:
    if type(bindings) is not AuthenticatedNativeBindings:
        raise GraftPhaseANativeTrainingClosureError(
            "v2 requires a v1-authenticated native binding"
        )
    bindings.assert_live()
    category_counts = {category: 0 for category in GRADIENT_CATEGORIES}
    output_rows = []
    output_raw_all_zero = True
    output_numeric_nonzero = False
    for name, parameter in bindings.named_trainable_parameters:
        category = _gradient_category_from_name(name)
        category_counts[category] += 1
        if category != "output_projection":
            continue
        if (
            not parameter.is_floating_point()
            or not bool(torch.isfinite(parameter.detach()).all().item())
        ):
            raise GraftPhaseANativeTrainingClosureError(
                f"live output projection is not finite floating point: {name}"
            )
        raw = _owned_raw_bytes(parameter)
        raw_all_zero = not any(raw)
        numeric_nonzero = bool(torch.count_nonzero(parameter.detach()).item())
        negative_zero_count = int(
            torch.count_nonzero(
                torch.eq(parameter.detach(), 0)
                & torch.signbit(parameter.detach())
            ).item()
        )
        if negative_zero_count:
            raise GraftPhaseANativeTrainingClosureError(
                f"signed negative-zero output bytes cannot name a regime: {name}"
            )
        output_raw_all_zero = output_raw_all_zero and raw_all_zero
        output_numeric_nonzero = output_numeric_nonzero or numeric_nonzero
        output_rows.append(
            {
                "name": name,
                "dtype": str(parameter.dtype),
                "shape": list(parameter.shape),
                "tensor_sha256": v1._tensor_bytes_sha256(parameter),
                "raw_bytes_all_zero": raw_all_zero,
                "numeric_nonzero": numeric_nonzero,
                "negative_zero_count": negative_zero_count,
            }
        )
    if any(category_counts[category] <= 0 for category in GRADIENT_CATEGORIES):
        raise GraftPhaseANativeTrainingClosureError(
            "v2 authenticated registry must contain all five trainable categories"
        )
    if output_raw_all_zero:
        if output_numeric_nonzero:
            raise GraftPhaseANativeTrainingClosureError(
                "raw-zero output state disagrees with its numeric state"
            )
        regime = "bootstrap"
    else:
        if not output_numeric_nonzero:
            raise GraftPhaseANativeTrainingClosureError(
                "non-canonical numerical-zero output bytes cannot name a regime"
            )
        regime = "post_bootstrap"
    return v1._seal(
        {
            "schema_version": (
                "bernini-graft-phase-a-live-output-weight-state-v2"
            ),
            "binding_receipt_digest": bindings.receipt()["digest"],
            "state_source": "live_authenticated_trainable_registry_raw_bytes",
            "category_parameter_counts": category_counts,
            "output_rows": output_rows,
            "output_raw_bytes_all_zero": output_raw_all_zero,
            "output_numeric_nonzero": output_numeric_nonzero,
            "derived_training_regime": regime,
            "caller_reported_regime_accepted": False,
        }
    )


def _construction_registry_snapshots(
    bindings: Any,
) -> tuple[tuple[torch.nn.Parameter, torch.Tensor, bool], ...]:
    if type(bindings) is not AuthenticatedNativeBindings:
        return ()
    try:
        return tuple(
            (parameter, parameter.detach().clone(), parameter.requires_grad)
            for _, parameter in bindings.named_trainable_parameters
        )
    except Exception as error:
        raise GraftPhaseANativeTrainingClosureError(
            "v2 construction could not snapshot its authenticated registry"
        ) from error


def _restore_construction_registry(
    snapshots: tuple[tuple[torch.nn.Parameter, torch.Tensor, bool], ...]
) -> None:
    cleanup_error = None
    for parameter, snapshot, requires_grad in snapshots:
        try:
            if not parameter.requires_grad:
                parameter.requires_grad_(True)
            with torch.no_grad():
                parameter.copy_(snapshot)
            parameter.requires_grad_(requires_grad)
        except Exception as error:  # pragma: no cover - catastrophic mutation
            cleanup_error = error
        try:
            parameter.grad = None
        except Exception as error:  # pragma: no cover - catastrophic mutation
            cleanup_error = error
    if cleanup_error is not None:
        raise GraftPhaseANativeTrainingClosureError(
            "v2 construction failed and could not restore its registry"
        ) from cleanup_error


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise GraftPhaseANativeTrainingClosureError(
            f"v2 receipt is not canonical finite ASCII JSON: {error}"
        ) from error


def _validated_sealed_mapping(
    value: Any, *, label: str
) -> tuple[dict[str, Any], str]:
    if not isinstance(value, Mapping):
        raise GraftPhaseANativeTrainingClosureError(f"{label} is not a mapping")
    plain = dict(value)
    digest = plain.pop("digest", None)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
        or hashlib.sha256(_canonical_json_bytes(plain)).hexdigest() != digest
    ):
        raise GraftPhaseANativeTrainingClosureError(
            f"{label} digest is not canonical lowercase SHA256"
        )
    return plain, digest


def _normalized_finite_nonnegative_hex(value: Any, *, label: str) -> float:
    if not isinstance(value, str):
        raise GraftPhaseANativeTrainingClosureError(
            f"{label} is not a hexadecimal float64"
        )
    try:
        parsed = float.fromhex(value)
    except ValueError as error:
        raise GraftPhaseANativeTrainingClosureError(
            f"{label} is not a hexadecimal float64"
        ) from error
    if not math.isfinite(parsed) or parsed < 0.0 or parsed.hex() != value:
        raise GraftPhaseANativeTrainingClosureError(
            f"{label} is not normalized finite non-negative float64"
        )
    return parsed


def _verified_branch_gradient_gates(
    *,
    base_receipt: Mapping[str, Any],
    bindings: AuthenticatedNativeBindings,
    cell_active: bool,
    training_regime: str,
    output_weight_state: Mapping[str, Any],
) -> Mapping[str, bool]:
    branches = base_receipt.get("per_branch_local_trainable_gradient_gate")
    local_target_rows = base_receipt.get("local_target_rows")
    local_graph_bearing = base_receipt.get("local_adapter_graph_bearing")
    if (
        not isinstance(branches, list)
        or len(branches) != len(BRANCH_ORDER)
        or type(local_target_rows) is not int
        or local_target_rows < 0
        or type(local_graph_bearing) is not bool
    ):
        raise GraftPhaseANativeTrainingClosureError(
            "underlying v1 branch-gradient inventory differs"
        )
    expected_graph_bearing = cell_active and local_target_rows > 0
    if (
        local_graph_bearing is not expected_graph_bearing
        or base_receipt.get("schedule_cell_active_for_training") is not cell_active
    ):
        raise GraftPhaseANativeTrainingClosureError(
            "underlying v1 branch ownership/activity fields differ"
        )
    if not cell_active:
        expected_gate = "inactive_cell_all_five_categories_exact_zero"
        expected_nonzero = []
    elif local_target_rows == 0:
        expected_gate = "source_only_sp_rank_all_five_categories_exact_zero"
        expected_nonzero = []
    elif training_regime == "bootstrap":
        expected_gate = "bootstrap_target_rows_output_projection_only_nonzero"
        expected_nonzero = ["output_projection"]
    elif training_regime == "post_bootstrap":
        expected_gate = (
            "post_bootstrap_target_rows_all_five_categories_finite_nonzero"
        )
        expected_nonzero = list(GRADIENT_CATEGORIES)
    else:
        raise GraftPhaseANativeTrainingClosureError(
            "verified branch gate received an unknown regime"
        )
    expected_counts = dict(output_weight_state["category_parameter_counts"])
    expected_names = [name for name, _ in bindings.named_trainable_parameters]
    for expected_role, sealed_branch in zip(BRANCH_ORDER, branches):
        branch, _ = _validated_sealed_mapping(
            sealed_branch, label=f"{expected_role} v2 branch receipt"
        )
        if (
            branch.get("schema_version") != LOCAL_GRADIENT_SCHEMA_VERSION
            or branch.get("role") != expected_role
            or branch.get("training_regime") != training_regime
            or branch.get("regime_source")
            != "live_authenticated_output_weight_bytes"
            or branch.get("local_target_rows") != local_target_rows
            or branch.get("adapter_graph_bearing") is not local_graph_bearing
            or branch.get("category_parameter_counts") != expected_counts
            or branch.get("finite_nonzero_categories") != expected_nonzero
            or branch.get("gate") != expected_gate
        ):
            raise GraftPhaseANativeTrainingClosureError(
                f"{expected_role} v2 branch receipt critical fields differ"
            )
        category_hex = branch.get("category_delta_l2_float64_hex")
        rows = branch.get("rows")
        if (
            not isinstance(category_hex, Mapping)
            or set(category_hex) != set(GRADIENT_CATEGORIES)
            or not isinstance(rows, list)
            or len(rows) != len(expected_names)
        ):
            raise GraftPhaseANativeTrainingClosureError(
                f"{expected_role} v2 branch category/row inventory differs"
            )
        category_squared = {category: 0.0 for category in GRADIENT_CATEGORIES}
        for expected_name, row in zip(expected_names, rows):
            if not isinstance(row, Mapping):
                raise GraftPhaseANativeTrainingClosureError(
                    f"{expected_role} v2 branch gradient row is not a mapping"
                )
            category = _gradient_category_from_name(expected_name)
            norm = _normalized_finite_nonnegative_hex(
                row.get("delta_l2_float64_hex"),
                label=f"{expected_role} {expected_name} delta",
            )
            if (
                row.get("name") != expected_name
                or row.get("category") != category
                or row.get("delta_finite") is not True
                or row.get("delta_nonzero") is not (norm > 0.0)
                or type(row.get("gradient_present_before")) is not bool
                or type(row.get("gradient_present_after")) is not bool
                or type(row.get("gradient_after_absent_or_exact_zero"))
                is not bool
                or (
                    norm > 0.0
                    and (
                        row.get("gradient_present_after") is not True
                        or row.get("gradient_after_absent_or_exact_zero")
                        is not False
                    )
                )
                or (
                    expected_nonzero == []
                    and row.get("gradient_after_absent_or_exact_zero") is not True
                )
            ):
                raise GraftPhaseANativeTrainingClosureError(
                    f"{expected_role} v2 branch gradient row differs: {expected_name}"
                )
            category_squared[category] += norm * norm
        observed_category = {
            category: _normalized_finite_nonnegative_hex(
                category_hex[category],
                label=f"{expected_role} {category} aggregate",
            )
            for category in GRADIENT_CATEGORIES
        }
        recomputed_category = {
            category: math.sqrt(category_squared[category])
            for category in GRADIENT_CATEGORIES
        }
        if any(
            observed_category[category].hex()
            != recomputed_category[category].hex()
            or ((category in expected_nonzero) is not (observed_category[category] > 0.0))
            for category in GRADIENT_CATEGORIES
        ):
            raise GraftPhaseANativeTrainingClosureError(
                f"{expected_role} v2 branch aggregate differs from its rows"
            )
    return MappingProxyType(
        {
            "bootstrap_output_only_gate_verified": (
                expected_gate
                == "bootstrap_target_rows_output_projection_only_nonzero"
            ),
            "post_bootstrap_five_category_local_gate_verified": (
                expected_gate
                == (
                    "post_bootstrap_target_rows_all_five_categories_"
                    "finite_nonzero"
                )
            ),
            "source_only_sp_all_five_categories_exact_zero_verified": (
                expected_gate
                == "source_only_sp_rank_all_five_categories_exact_zero"
            ),
        }
    )


def _assert_v2_seam(instance: Any) -> None:
    expected = globals().get("_PINNED_V2_CLASS_DICT_IDENTITIES")
    inherited = globals().get("_PINNED_V1_EXECUTION_DESCRIPTOR_IDENTITIES")
    protected = globals().get(
        "_PINNED_INSTANCE_PROTECTED_EXECUTION_DESCRIPTOR_NAMES"
    )
    cls = globals().get("_PINNED_V2_EXACT_CLASS")
    base = globals().get("_PINNED_V1_EXECUTION_CLASS")
    expected_protected = frozenset()
    if isinstance(expected, Mapping) and isinstance(inherited, Mapping):
        expected_protected = frozenset(
            name
            for inventory in (expected, inherited)
            for name, descriptor in inventory.items()
            if inspect.isfunction(descriptor)
            or isinstance(
                descriptor,
                (classmethod, staticmethod, property),
            )
        )
    try:
        instance_namespace = object.__getattribute__(instance, "__dict__")
    except (AttributeError, TypeError):
        instance_namespace = None
    shadowed = (
        tuple(
            sorted(
                name
                for name in protected
                if isinstance(instance_namespace, dict)
                and name in instance_namespace
            )
        )
        if isinstance(protected, frozenset)
        else ()
    )
    if (
        type(instance) is not cls
        or globals().get("PhaseANativeTrainingClosure") is not cls
        or globals().get("PhaseANativeTrainingClosureV2", cls) is not cls
        or not isinstance(expected, Mapping)
        or not isinstance(inherited, Mapping)
        or not isinstance(protected, frozenset)
        or protected != expected_protected
        or not isinstance(instance_namespace, dict)
        or bool(shadowed)
        or cls.__mro__ != globals().get("_PINNED_V2_EXACT_MRO")
        or v1.PhaseANativeTrainingClosure is not base
        or base.__mro__ != globals().get("_PINNED_V1_EXECUTION_MRO")
        or set(cls.__dict__) != set(expected)
        or any(
            cls.__dict__.get(name) is not descriptor
            for name, descriptor in expected.items()
        )
        or set(base.__dict__) != set(inherited)
        or any(
            base.__dict__.get(name) is not descriptor
            for name, descriptor in inherited.items()
        )
        or any(
            name not in expected
            and inspect.getattr_static(cls, name, None) is not descriptor
            for name, descriptor in inherited.items()
        )
    ):
        if isinstance(instance_namespace, dict):
            for name in shadowed:
                instance_namespace.pop(name, None)
        raise GraftPhaseANativeTrainingClosureError(
            "v2 exact-type or fixed gradient-gate seam differs"
        )


class PhaseANativeTrainingClosure(v1.PhaseANativeTrainingClosure):
    """Pinned v1 closure with an automatic post-bootstrap gradient gate."""

    def __getattribute__(self, name: str) -> Any:
        """Fail and poison before an instance-level execution shadow is used."""

        protected = globals().get(
            "_PINNED_INSTANCE_PROTECTED_EXECUTION_DESCRIPTOR_NAMES"
        )
        namespace = object.__getattribute__(self, "__dict__")
        if not isinstance(protected, frozenset):
            raise GraftPhaseANativeTrainingClosureError(
                "v2 exact-type or fixed gradient-gate seam differs"
            )
        shadowed = tuple(
            sorted(member for member in protected if member in namespace)
        )
        if shadowed:
            for member in shadowed:
                namespace.pop(member, None)
            error = GraftPhaseANativeTrainingClosureError(
                "v2 exact-type or fixed gradient-gate seam differs"
            )
            snapshots = namespace.get("_initial_parameter_snapshots")
            inherited = globals().get(
                "_PINNED_V1_EXECUTION_DESCRIPTOR_IDENTITIES"
            )
            poison = (
                inherited.get("_poison")
                if isinstance(inherited, Mapping)
                else None
            )
            if isinstance(snapshots, tuple) and inspect.isfunction(poison):
                try:
                    poison(self)
                except Exception as cleanup_error:
                    raise cleanup_error from error
            else:
                namespace["_phase"] = "failed"
            raise error
        return object.__getattribute__(self, name)

    def __init_subclass__(cls, **_kwargs: Any) -> None:
        raise GraftPhaseANativeTrainingClosureError(
            "v2 Phase-A closure does not permit subclassing"
        )

    def __init__(
        self,
        *,
        bindings: AuthenticatedNativeBindings,
        source_video: torch.Tensor,
        noisy_target: torch.Tensor,
        negative_condition: torch.Tensor,
        positive_condition: torch.Tensor,
        schedule_index: int,
        sigma: torch.Tensor,
        timestep: torch.Tensor,
    ) -> None:
        self._phase = "constructing"
        construction_snapshots = _construction_registry_snapshots(bindings)
        try:
            _assert_v2_seam(self)
            _assert_pinned_v1_kernel()
            self._output_weight_state = _live_output_weight_state(bindings)
            self._training_regime = str(
                self._output_weight_state["derived_training_regime"]
            )
            super().__init__(
                bindings=bindings,
                source_video=source_video,
                noisy_target=noisy_target,
                negative_condition=negative_condition,
                positive_condition=positive_condition,
                schedule_index=schedule_index,
                sigma=sigma,
                timestep=timestep,
            )
            _assert_pinned_v1_kernel()
            _assert_v2_seam(self)
        except Exception as error:
            try:
                _restore_construction_registry(construction_snapshots)
            except Exception as cleanup_error:
                self._phase = "failed"
                raise cleanup_error from error
            self._phase = "failed"
            raise

    @property
    def training_regime(self) -> str:
        """Read the live-byte-derived regime; callers cannot set it."""

        return self._training_regime

    def _assert_live_regime_unchanged(self) -> None:
        observed = _live_output_weight_state(self.bindings)
        if dict(observed) != dict(self._output_weight_state):
            raise GraftPhaseANativeTrainingClosureError(
                "live output-weight bytes changed after v2 regime derivation"
            )

    def _gradient_category(self, name: str) -> str:
        return _gradient_category_from_name(name)

    def _local_trainable_delta_receipt(
        self,
        *,
        role: str,
        before: Mapping[str, torch.Tensor],
        after: Mapping[str, torch.Tensor],
        before_presence: Mapping[str, bool],
        observation: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        _assert_v2_seam(self)
        _assert_pinned_v1_kernel()
        self._assert_live_regime_unchanged()
        category_squared = {category: 0.0 for category in GRADIENT_CATEGORIES}
        category_counts = {category: 0 for category in GRADIENT_CATEGORIES}
        rows = []
        for name, parameter in self.bindings.named_trainable_parameters:
            delta = after[name] - before[name]
            if not bool(torch.isfinite(delta).all().item()):
                raise GraftPhaseANativeTrainingClosureError(
                    f"{role} local v2 gradient delta is non-finite: {name}"
                )
            norm = float(delta.norm().item())
            if not math.isfinite(norm):
                raise GraftPhaseANativeTrainingClosureError(
                    f"{role} local v2 gradient norm is non-finite: {name}"
                )
            category = self._gradient_category(name)
            category_squared[category] += norm * norm
            category_counts[category] += 1
            current = parameter.grad
            current_absent_or_zero = current is None or not bool(
                torch.count_nonzero(current.detach()).item()
            )
            rows.append(
                {
                    "name": name,
                    "category": category,
                    "gradient_present_before": bool(before_presence[name]),
                    "gradient_present_after": current is not None,
                    "gradient_after_absent_or_exact_zero": current_absent_or_zero,
                    "delta_l2_float64_hex": norm.hex(),
                    "delta_finite": True,
                    "delta_nonzero": norm > 0.0,
                }
            )
        category_l2 = {
            category: math.sqrt(squared)
            for category, squared in category_squared.items()
        }
        if any(
            category_counts[category] <= 0
            or not math.isfinite(category_l2[category])
            for category in GRADIENT_CATEGORIES
        ):
            raise GraftPhaseANativeTrainingClosureError(
                f"{role} v2 five-category gradient inventory differs"
            )
        local_target_rows = int(observation["local_target_rows"])
        adapter_graph_bearing = bool(observation["adapter_graph_bearing"])
        nonzero_categories = tuple(
            category
            for category in GRADIENT_CATEGORIES
            if category_l2[category] > 0.0
        )
        if not self._cell_active:
            if adapter_graph_bearing or nonzero_categories:
                raise GraftPhaseANativeTrainingClosureError(
                    f"{role} inactive cell produced a v2 trainable gradient"
                )
            gate = "inactive_cell_all_five_categories_exact_zero"
        elif local_target_rows == 0:
            if adapter_graph_bearing or nonzero_categories:
                raise GraftPhaseANativeTrainingClosureError(
                    f"{role} source-only SP rank produced a v2 trainable gradient"
                )
            if any(not row["gradient_after_absent_or_exact_zero"] for row in rows):
                raise GraftPhaseANativeTrainingClosureError(
                    f"{role} source-only SP rank retained a nonzero v2 gradient"
                )
            gate = "source_only_sp_rank_all_five_categories_exact_zero"
        elif self._training_regime == "bootstrap":
            if (
                not adapter_graph_bearing
                or category_l2["output_projection"] <= 0.0
                or any(
                    category_l2[category] != 0.0
                    for category in UPSTREAM_GRADIENT_CATEGORIES
                )
            ):
                raise GraftPhaseANativeTrainingClosureError(
                    f"{role} target-owning rank failed the v2 bootstrap output-only gradient gate"
                )
            gate = "bootstrap_target_rows_output_projection_only_nonzero"
        elif self._training_regime == "post_bootstrap":
            if not adapter_graph_bearing or any(
                category_l2[category] <= 0.0
                for category in GRADIENT_CATEGORIES
            ):
                raise GraftPhaseANativeTrainingClosureError(
                    f"{role} target-owning rank failed the v2 post-bootstrap five-category gradient gate"
                )
            gate = "post_bootstrap_target_rows_all_five_categories_finite_nonzero"
        else:  # pragma: no cover - protected by the live-state derivation
            raise GraftPhaseANativeTrainingClosureError(
                "v2 derived training regime is unknown"
            )
        return v1._seal(
            {
                "schema_version": LOCAL_GRADIENT_SCHEMA_VERSION,
                "role": role,
                "training_regime": self._training_regime,
                "regime_source": "live_authenticated_output_weight_bytes",
                "local_target_rows": local_target_rows,
                "adapter_graph_bearing": adapter_graph_bearing,
                "category_parameter_counts": category_counts,
                "category_delta_l2_float64_hex": {
                    category: value.hex()
                    for category, value in category_l2.items()
                },
                "finite_nonzero_categories": list(nonzero_categories),
                "rows": rows,
                "gate": gate,
            }
        )

    def replay_and_backward(self) -> NativeTrainingClosureResult:
        try:
            _assert_v2_seam(self)
            _assert_pinned_v1_kernel()
            self._assert_live_regime_unchanged()
            underlying = super().replay_and_backward()
            _assert_pinned_v1_kernel()
            _assert_v2_seam(self)
            self._assert_live_regime_unchanged()
            base, underlying_digest = _validated_sealed_mapping(
                underlying.receipt, label="underlying v1 closure receipt"
            )
            if base.get("schema_version") != v1.SCHEMA_VERSION:
                raise GraftPhaseANativeTrainingClosureError(
                    "underlying v1 closure receipt schema differs"
                )
            denied_authority = (
                "exact40_coverage_verified_by_this_closure",
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
                "checkpoint_weight_content_verified_by_this_core",
                "official_cuda_closure_verified_by_this_core",
                "forward_route_semantics_verified_by_this_core",
                "packed_raw_to_apg_registry_chain_verified_by_this_core",
                "sp4_collective_parity_verified",
                "full_sampler_trajectory_verified",
                "training_quality_claim_authorized",
                "scientific_action_editing_claim_authorized",
            )
            if any(base.get(key) is not False for key in denied_authority):
                raise GraftPhaseANativeTrainingClosureError(
                    "underlying v1 receipt authority was unexpectedly elevated"
                )
            verified_gates = _verified_branch_gradient_gates(
                base_receipt=base,
                bindings=self.bindings,
                cell_active=self._cell_active,
                training_regime=self._training_regime,
                output_weight_state=self._output_weight_state,
            )
            base.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "wrapped_v1_schema_version": v1.SCHEMA_VERSION,
                    "wrapped_v1_source_sha256": PINNED_V1_SOURCE_SHA256,
                    "wrapped_v1_runtime_namespace_sha256": (
                        PINNED_V1_RUNTIME_NAMESPACE_SHA256[
                            (sys.version_info.major, sys.version_info.minor)
                        ]
                    ),
                    "wrapped_v1_runtime_python_minor": [
                        sys.version_info.major,
                        sys.version_info.minor,
                    ],
                    "pinned_sigma_strata_source_sha256": (
                        PINNED_SIGMA_STRATA_SOURCE_SHA256
                    ),
                    "pinned_sigma_runtime_namespace_sha256": (
                        PINNED_SIGMA_RUNTIME_NAMESPACE_SHA256[
                            (sys.version_info.major, sys.version_info.minor)
                        ]
                    ),
                    "pinned_sigma_schedule_sha256": (
                        PINNED_SIGMA_SCHEDULE_SHA256
                    ),
                    "execution_import_identities_verified": True,
                    "wrapped_v1_receipt_digest": underlying_digest,
                    "training_regime": self._training_regime,
                    "training_regime_source": (
                        "live_authenticated_output_weight_raw_bytes"
                    ),
                    "training_regime_caller_or_cli_input_accepted": False,
                    "live_output_weight_state": dict(self._output_weight_state),
                    "gradient_categories": list(GRADIENT_CATEGORIES),
                    **dict(verified_gates),
                    "optimizer_step_verified_by_this_core": False,
                    "two_consecutive_steps_verified_by_this_core": False,
                    "post_bootstrap_cuda_short_training_verified_by_this_core": False,
                    "short_training_claim_authorized": False,
                    "remaining_gpu_assumptions": list(REMAINING_GPU_ASSUMPTIONS),
                }
            )
            self._receipt = v1._seal(base)
            final_plain, _ = _validated_sealed_mapping(
                self._receipt, label="final v2 closure receipt"
            )
            if final_plain.get("schema_version") != SCHEMA_VERSION:
                raise GraftPhaseANativeTrainingClosureError(
                    "final v2 closure receipt schema differs"
                )
            _assert_pinned_v1_kernel()
            _assert_v2_seam(self)
            return NativeTrainingClosureResult(
                guided_clean=underlying.guided_clean.detach().clone(),
                flow_matching_loss=underlying.flow_matching_loss.detach().clone(),
                receipt=self._receipt,
            )
        except Exception as error:
            self._fail(error)
            raise


_PINNED_V2_EXACT_CLASS = PhaseANativeTrainingClosure
_PINNED_V2_EXACT_MRO = PhaseANativeTrainingClosure.__mro__
_PINNED_V2_CLASS_DICT_IDENTITIES = MappingProxyType(
    dict(PhaseANativeTrainingClosure.__dict__)
)
_PINNED_V1_EXECUTION_CLASS = v1.PhaseANativeTrainingClosure
_PINNED_V1_EXECUTION_MRO = v1.PhaseANativeTrainingClosure.__mro__
_PINNED_V1_EXECUTION_DESCRIPTOR_IDENTITIES = MappingProxyType(
    dict(v1.PhaseANativeTrainingClosure.__dict__)
)
_PINNED_INSTANCE_PROTECTED_EXECUTION_DESCRIPTOR_NAMES = frozenset(
    name
    for inventory in (
        _PINNED_V2_CLASS_DICT_IDENTITIES,
        _PINNED_V1_EXECUTION_DESCRIPTOR_IDENTITIES,
    )
    for name, descriptor in inventory.items()
    if inspect.isfunction(descriptor)
    or isinstance(descriptor, (classmethod, staticmethod, property))
)
PhaseANativeTrainingClosureV2 = PhaseANativeTrainingClosure


def execute_phase_a_native_training_closure(
    *,
    bindings: AuthenticatedNativeBindings,
    source_video: torch.Tensor,
    noisy_target: torch.Tensor,
    negative_condition: torch.Tensor,
    positive_condition: torch.Tensor,
    schedule_index: int,
    sigma: torch.Tensor,
    timestep: torch.Tensor,
) -> NativeTrainingClosureResult:
    """Execute one v2 cell; no caller-reported regime is accepted."""

    closure = PhaseANativeTrainingClosure(
        bindings=bindings,
        source_video=source_video,
        noisy_target=noisy_target,
        negative_condition=negative_condition,
        positive_condition=positive_condition,
        schedule_index=schedule_index,
        sigma=sigma,
        timestep=timestep,
    )
    closure.measure()
    closure.derive_phase_a_flow_matching_vjp()
    return closure.replay_and_backward()


__all__ = [
    "APG_ETA",
    "APG_MOMENTUM",
    "APG_NORM_THRESHOLD",
    "AuthenticatedNativeBindings",
    "BRANCH_ORDER",
    "EXPECTED_FRAMES",
    "EXPECTED_LATENT_PHASES",
    "EXPECTED_PATCH_SOURCE_IDS",
    "FLOW_MATCHING_OBJECTIVE",
    "FLOW_MATCHING_REDUCTION",
    "FORWARD_ROUTE_SCHEMA_VERSION",
    "GRADIENT_CATEGORIES",
    "GraftPhaseANativeTrainingClosureError",
    "GUIDANCE_MODE",
    "GUIDANCE_SCALE",
    "LOCAL_GRADIENT_SCHEMA_VERSION",
    "NativeFieldMeasurement",
    "NativeForwardContextObservation",
    "NativeForwardContextRequest",
    "NativePhaseAFlowMatchingVJP",
    "NativeTrainingClosureResult",
    "PHASE_A_ACTIVE_SCHEDULE_INDICES",
    "PINNED_BERNINI_COMMIT",
    "PINNED_TRANSFORMER_WAN_SHA256",
    "PINNED_V1_RUNTIME_NAMESPACE_SHA256",
    "PINNED_V1_SOURCE_SHA256",
    "PINNED_WAN_DIFFUSION_SHA256",
    "PhaseANativeTrainingClosure",
    "PhaseANativeTrainingClosureV2",
    "REMAINING_GPU_ASSUMPTIONS",
    "SCHEMA_VERSION",
    "TRAINING_REGIMES",
    "authenticate_cpu_test_fakes",
    "authenticate_pinned_native_bindings",
    "build_native_forward_context_observation",
    "execute_phase_a_native_training_closure",
    "unpack_wan_target_velocity",
]
