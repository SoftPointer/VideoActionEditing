#!/usr/bin/env python3
"""One-cell training-mode closure for Bernini source-only ``v2v_apg``.

This module is deliberately narrower than a sampler or a trainer.  It closes
one local field cell in the exact deployed Bernini-R 1.3B coordinate:

``source video (source_id=1) ; noisy target (source_id=0)``
    -> one shared visual/rotary/timestep pack
    -> detached negative then positive measurements
    -> target-suffix Wan unpacking
    -> ``x_t - sigma * v`` in FP32
    -> the pinned vendor ``normalized_guidance`` on fresh FP32 leaves
    -> fixed same-source/no-op FM mean-MSE (no caller-supplied cotangent)
    -> loss-to-guided-to-clean VJPs and exact BF16 raw cotangents
    -> negative then positive graph-enabled serial replay/backward.

The implementation was audited against these files from Bernini commit
``2d2b4591ac053ec25c6371b01a5a6746679e5793``:

* ``bernini/models/wan_diffusion.py`` SHA256
  ``59e860ba3490a83f06bd4be75697490f49a118ee5ca969e85eea4dd7fa122512``;
* ``bernini/models/transformer_wan.py`` SHA256
  ``9fb579611e79e0f534d5d6ccdcd956c35e57b4513c15267e8533ff3832a1f223``.

The pinned source uses the einops order
``b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)`` with
``pt=1, ph=pw=2``.  :func:`unpack_wan_target_velocity` spells that inverse
without adding an einops dependency and preserves autograd.

There is intentionally no scheduler step, solver state transport, external
target video/cotangent, image reference, mask, pose, track, optical flow, or
motion donor in this API.  High-sigma cells outside the route's pinned active
index registry are exact zero-updates and are never counted as trained.  A
successful CPU fake run proves the state machine and algebra; only a separate
GPU runner can bind checkpoint content, route semantics, packed chain, SP4,
and exact40 coverage.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import struct
from types import MappingProxyType
from typing import Any, Callable, ContextManager, Mapping, Optional, Sequence

import torch

import inference_sigma_strata as sigma_strata


SCHEMA_VERSION = "bernini-graft-phase-a-native-training-closure-v1"
BINDING_SCHEMA_VERSION = "bernini-graft-phase-a-native-binding-v1"
PINNED_BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
PINNED_WAN_DIFFUSION_SHA256 = (
    "59e860ba3490a83f06bd4be75697490f49a118ee5ca969e85eea4dd7fa122512"
)
PINNED_TRANSFORMER_WAN_SHA256 = (
    "9fb579611e79e0f534d5d6ccdcd956c35e57b4513c15267e8533ff3832a1f223"
)
EXPECTED_FRAMES = 81
EXPECTED_LATENT_PHASES = 21
EXPECTED_LATENT_CHANNELS = 16
EXPECTED_PATCH_SIZE = (1, 2, 2)
EXPECTED_PACKED_CHANNELS = 64
EXPECTED_HIDDEN_DIM = 1536
EXPECTED_TEXT_TOKENS = 512
EXPECTED_TEXT_DIM = 4096
EXPECTED_STEPS = 40
EXPECTED_MODEL_ID = "transformer_1"
EXPECTED_PATCH_SOURCE_IDS = (1.0, 0.0)
BRANCH_ORDER = ("negative", "positive")
GUIDANCE_MODE = "v2v_apg"
GUIDANCE_SCALE = 4.0
APG_ETA = 0.5
APG_NORM_THRESHOLD = 50.0
APG_MOMENTUM = 0.0
FLOW_MATCHING_REDUCTION = "mean"
FLOW_MATCHING_OBJECTIVE = "same_source_noop_velocity_mean_mse"
FORWARD_ROUTE_SCHEMA_VERSION = "bernini-graft-phase-a-forward-route-v1"
# IdentityRebinderV1 is exactly off at sigma >= 0.75.  The first release
# therefore authorizes only the exact40 cells for which the pinned route has a
# non-zero mid/low-sigma gate.  A runner must account for all forty cells; this
# one-cell closure must never silently relabel an inactive high-sigma cell as
# a trained example.
PHASE_A_ACTIVE_SCHEDULE_INDICES = tuple(range(26, EXPECTED_STEPS))
PHASE_A_EXECUTE_API_FIELDS = (
    "bindings",
    "source_video",
    "noisy_target",
    "negative_condition",
    "positive_condition",
    "schedule_index",
    "sigma",
    "timestep",
)
FORBIDDEN_ORACLE_API_FIELDS = (
    "target_video",
    "guided_clean_cotangent",
    "mask",
    "pose",
    "track",
    "flow",
    "optical_flow",
    "donor",
    "motion_donor",
    "scheduler",
    "solver_state",
)
_BINDING_TOKEN = object()


class GraftPhaseANativeTrainingClosureError(RuntimeError):
    """Raised before an unauthenticated or ambiguous closure can continue."""


@dataclass(frozen=True)
class NativeForwardContextRequest:
    """Exact native V-pack coordinate handed to a trainable forward route."""

    phase: str
    role: str
    visual_pack: torch.Tensor = field(repr=False, compare=False)
    rotary_pack: torch.Tensor = field(repr=False, compare=False)
    timestep: torch.Tensor = field(repr=False, compare=False)
    schedule_index: int
    sigma: float
    total_tokens: int
    condition_tokens: int
    target_tokens: int


@dataclass(frozen=True)
class NativeForwardContextObservation:
    """Authenticated rank-local geometry exposed by one entered route."""

    sequence_parallel_rank: int
    sequence_parallel_size: int
    global_total_tokens: int
    global_condition_tokens: int
    global_target_tokens: int
    local_shard_start: int
    local_shard_stop_exclusive: int
    local_shard_rows: int
    local_valid_rows: int
    local_padding_rows: int
    local_target_rows: int
    local_target_selector_sha256: str
    route_gate: float
    adapter_graph_bearing: bool


def _cpu_fake_global_context(
    *, request: NativeForwardContextRequest
) -> ContextManager[Any]:
    if not isinstance(request, NativeForwardContextRequest):
        raise GraftPhaseANativeTrainingClosureError(
            "CPU fake route received a non-native request"
        )
    local_selector = torch.cat(
        (
            torch.zeros(request.condition_tokens, dtype=torch.bool),
            torch.ones(request.target_tokens, dtype=torch.bool),
        )
    )
    active = request.schedule_index in PHASE_A_ACTIVE_SCHEDULE_INDICES
    observation = build_native_forward_context_observation(
        request=request,
        sequence_parallel_rank=0,
        sequence_parallel_size=1,
        local_target_selector=local_selector,
        route_gate=(1.0 if active else 0.0),
        adapter_graph_bearing=(
            request.phase == "replay" and active and torch.is_grad_enabled()
        ),
    )
    return nullcontext(observation)


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
            f"receipt is not canonical finite ASCII JSON: {error}"
        ) from error


def _seal(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    plain = dict(payload)
    if "digest" in plain:
        raise GraftPhaseANativeTrainingClosureError("receipt payload already has digest")
    digest = hashlib.sha256(_canonical_json_bytes(plain)).hexdigest()
    return MappingProxyType({**plain, "digest": digest})


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_bytes_sha256(value: torch.Tensor) -> str:
    # Clone after moving to CPU so the storage is owned, offset-free, and
    # exactly sized to the logical tensor.  ``bytes(untyped_storage())`` is
    # independent of NumPy and works in both the tc environment and the vd
    # environment whose PyTorch wheel cannot bridge to NumPy 2.x.
    if not isinstance(value, torch.Tensor):
        raise GraftPhaseANativeTrainingClosureError(
            "tensor digest input is not a torch tensor"
        )
    detached = value.detach()
    if type(detached) is not torch.Tensor:
        raise GraftPhaseANativeTrainingClosureError(
            "tensor detach did not return an exact torch.Tensor"
        )
    owned = detached.cpu().contiguous().clone()
    if type(owned) is not torch.Tensor:
        raise GraftPhaseANativeTrainingClosureError(
            "owned digest value is not an exact torch.Tensor"
        )
    storage = owned.untyped_storage()
    expected_bytes = int(owned.numel()) * int(owned.element_size())
    if int(storage.nbytes()) != expected_bytes:
        raise GraftPhaseANativeTrainingClosureError(
            "owned tensor storage contains bytes outside the logical tensor"
        )
    # This is intentionally the PyTorch storage protocol, not ctypes, NumPy,
    # pickle, a data pointer, or a tensor-subclass hook.  It is slower than a
    # native pointer copy but leaves no unbounded memory-read primitive in the
    # receipt path.
    raw = bytes(storage)
    if len(raw) != expected_bytes:
        raise GraftPhaseANativeTrainingClosureError(
            "owned tensor storage byte length differs"
        )
    header = json.dumps(
        {"dtype": str(owned.dtype), "shape": list(owned.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + raw).hexdigest()


def build_native_forward_context_observation(
    *,
    request: NativeForwardContextRequest,
    sequence_parallel_rank: int,
    sequence_parallel_size: int,
    local_target_selector: torch.Tensor,
    route_gate: float,
    adapter_graph_bearing: bool,
) -> NativeForwardContextObservation:
    """Own one exact SP1/SP4 selector observation from an authenticated route."""

    if (
        not isinstance(request, NativeForwardContextRequest)
        or type(sequence_parallel_rank) is not int
        or type(sequence_parallel_size) is not int
        or sequence_parallel_size not in (1, 4)
        or not 0 <= sequence_parallel_rank < sequence_parallel_size
        or not isinstance(local_target_selector, torch.Tensor)
        or local_target_selector.dtype != torch.bool
        or local_target_selector.ndim != 1
        or not local_target_selector.is_contiguous()
        or isinstance(route_gate, bool)
        or not isinstance(route_gate, (int, float))
        or not math.isfinite(float(route_gate))
        or not 0.0 <= float(route_gate) <= 1.0
        or type(adapter_graph_bearing) is not bool
    ):
        raise GraftPhaseANativeTrainingClosureError(
            "native forward-context observation inputs differ"
        )
    local_rows = math.ceil(request.total_tokens / sequence_parallel_size)
    if int(local_target_selector.numel()) != local_rows:
        raise GraftPhaseANativeTrainingClosureError(
            "rank-local target selector length differs from append-pad shard"
        )
    padded = local_rows * sequence_parallel_size
    expected_global = torch.cat(
        (
            torch.zeros(request.condition_tokens, dtype=torch.bool),
            torch.ones(request.target_tokens, dtype=torch.bool),
            torch.zeros(
                padded - request.total_tokens,
                dtype=torch.bool,
            ),
        )
    )
    start = sequence_parallel_rank * local_rows
    expected = expected_global[start : start + local_rows].to(
        device=local_target_selector.device
    ).contiguous()
    if not torch.equal(local_target_selector, expected):
        raise GraftPhaseANativeTrainingClosureError(
            "rank-local target selector differs from the global target suffix"
        )
    stop = min(start + local_rows, request.total_tokens)
    valid_rows = max(0, stop - start)
    padding_rows = local_rows - valid_rows
    target_rows = int(torch.count_nonzero(expected).item())
    active = request.schedule_index in PHASE_A_ACTIVE_SCHEDULE_INDICES
    if (active and float(route_gate) <= 0.0) or (
        not active and float(route_gate) != 0.0
    ):
        raise GraftPhaseANativeTrainingClosureError(
            "route gate differs from the authenticated active-cell envelope"
        )
    expected_adapter_graph = (
        request.phase == "replay"
        and active
        and target_rows > 0
        and float(route_gate) > 0.0
        and torch.is_grad_enabled()
    )
    if adapter_graph_bearing is not expected_adapter_graph:
        raise GraftPhaseANativeTrainingClosureError(
            "adapter graph-bearing claim differs from local target ownership"
        )
    return NativeForwardContextObservation(
        sequence_parallel_rank=sequence_parallel_rank,
        sequence_parallel_size=sequence_parallel_size,
        global_total_tokens=request.total_tokens,
        global_condition_tokens=request.condition_tokens,
        global_target_tokens=request.target_tokens,
        local_shard_start=start,
        local_shard_stop_exclusive=stop,
        local_shard_rows=local_rows,
        local_valid_rows=valid_rows,
        local_padding_rows=padding_rows,
        local_target_rows=target_rows,
        local_target_selector_sha256=_tensor_bytes_sha256(expected),
        route_gate=float(route_gate),
        adapter_graph_bearing=adapter_graph_bearing,
    )


def _validated_forward_context_observation(
    *,
    request: NativeForwardContextRequest,
    value: Any,
) -> NativeForwardContextObservation:
    if type(value) is not NativeForwardContextObservation:
        raise GraftPhaseANativeTrainingClosureError(
            "entered forward context omitted its authenticated observation"
        )
    if (
        type(value.sequence_parallel_rank) is not int
        or type(value.sequence_parallel_size) is not int
        or value.sequence_parallel_size not in (1, 4)
        or not 0 <= value.sequence_parallel_rank < value.sequence_parallel_size
    ):
        raise GraftPhaseANativeTrainingClosureError(
            "forward-context observation rank/size differs"
        )
    local_rows = math.ceil(request.total_tokens / value.sequence_parallel_size)
    padded = local_rows * value.sequence_parallel_size
    selector = torch.cat(
        (
            torch.zeros(request.condition_tokens, dtype=torch.bool),
            torch.ones(request.target_tokens, dtype=torch.bool),
            torch.zeros(
                padded - request.total_tokens,
                dtype=torch.bool,
            ),
        )
    )[
        value.sequence_parallel_rank
        * local_rows : (value.sequence_parallel_rank + 1)
        * local_rows
    ].contiguous()
    expected = build_native_forward_context_observation(
        request=request,
        sequence_parallel_rank=value.sequence_parallel_rank,
        sequence_parallel_size=value.sequence_parallel_size,
        local_target_selector=selector,
        route_gate=value.route_gate,
        adapter_graph_bearing=value.adapter_graph_bearing,
    )
    if value != expected:
        raise GraftPhaseANativeTrainingClosureError(
            "forward-context observation fields differ from recomputation"
        )
    return value


def _forward_context_observation_receipt(
    *,
    request: NativeForwardContextRequest,
    observation: NativeForwardContextObservation,
) -> Mapping[str, Any]:
    return _seal(
        {
            "schema_version": "bernini-graft-phase-a-forward-context-observation-v1",
            "phase": request.phase,
            "role": request.role,
            "schedule_index": request.schedule_index,
            "sequence_parallel_rank": observation.sequence_parallel_rank,
            "sequence_parallel_size": observation.sequence_parallel_size,
            "global_total_tokens": observation.global_total_tokens,
            "global_condition_tokens": observation.global_condition_tokens,
            "global_target_tokens": observation.global_target_tokens,
            "local_shard_start": observation.local_shard_start,
            "local_shard_stop_exclusive": observation.local_shard_stop_exclusive,
            "local_shard_rows": observation.local_shard_rows,
            "local_valid_rows": observation.local_valid_rows,
            "local_padding_rows": observation.local_padding_rows,
            "local_target_rows": observation.local_target_rows,
            "local_target_selector_sha256": (
                observation.local_target_selector_sha256
            ),
            "route_gate_float64_hex": observation.route_gate.hex(),
            "adapter_graph_bearing": observation.adapter_graph_bearing,
        }
    )


def _finite_tensor(value: Any, *, label: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or not (value.is_floating_point() or value.is_complex())
        or not bool(torch.isfinite(value).all().item())
    ):
        raise GraftPhaseANativeTrainingClosureError(
            f"{label} must be a finite materialized torch tensor"
        )
    return value


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    try:
        return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    except (AttributeError, RuntimeError):
        return left.data_ptr() == right.data_ptr()


def _validate_callable_parameters(
    function: Callable[..., Any], *, required: Sequence[str], label: str
) -> None:
    if not callable(function):
        raise GraftPhaseANativeTrainingClosureError(f"{label} must be callable")
    try:
        names = tuple(inspect.signature(function).parameters)
    except (TypeError, ValueError) as error:
        raise GraftPhaseANativeTrainingClosureError(
            f"{label} signature is unavailable"
        ) from error
    if any(name not in names for name in required):
        raise GraftPhaseANativeTrainingClosureError(
            f"{label} signature differs from the pinned interface"
        )


def _objective_api_structure_receipt() -> Mapping[str, Any]:
    execute = globals().get("execute_phase_a_native_training_closure")
    derive = getattr(
        globals().get("PhaseANativeTrainingClosure"),
        "derive_phase_a_flow_matching_vjp",
        None,
    )
    if not callable(execute) or not callable(derive):
        raise GraftPhaseANativeTrainingClosureError(
            "Phase-A objective public API is unavailable"
        )
    execute_fields = tuple(inspect.signature(execute).parameters)
    derive_fields = tuple(inspect.signature(derive).parameters)
    forbidden_present = sorted(
        set(execute_fields).intersection(FORBIDDEN_ORACLE_API_FIELDS)
        | set(derive_fields).intersection(FORBIDDEN_ORACLE_API_FIELDS)
    )
    if (
        execute_fields != PHASE_A_EXECUTE_API_FIELDS
        or derive_fields != ("self",)
        or forbidden_present
    ):
        raise GraftPhaseANativeTrainingClosureError(
            "Phase-A objective API gained an unauthenticated input"
        )
    return MappingProxyType(
        {
            "execute_parameters": execute_fields,
            "derive_parameters": derive_fields,
            "forbidden_oracle_parameters_present": tuple(forbidden_present),
            "oracle_inputs_absent": True,
        }
    )


def _named_trainable_parameters(
    value: Sequence[tuple[str, torch.nn.Parameter]],
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    try:
        rows = tuple(value)
    except TypeError as error:
        raise GraftPhaseANativeTrainingClosureError(
            "named_trainable_parameters must be a finite sequence"
        ) from error
    if not rows:
        raise GraftPhaseANativeTrainingClosureError(
            "at least one trainable-registry parameter is required"
        )
    names: set[str] = set()
    identities: set[int] = set()
    for row in rows:
        if not isinstance(row, tuple) or len(row) != 2:
            raise GraftPhaseANativeTrainingClosureError(
                "trainable parameter rows must be (name, parameter) tuples"
            )
        name, parameter = row
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or not isinstance(parameter, torch.nn.Parameter)
            or id(parameter) in identities
            or not parameter.requires_grad
            or parameter.device.type == "meta"
            or not bool(torch.isfinite(parameter.detach()).all().item())
        ):
            raise GraftPhaseANativeTrainingClosureError(
                "trainable parameter registry differs"
            )
        names.add(name)
        identities.add(id(parameter))
    return rows


def _external_trainable_owners(
    value: Mapping[str, torch.nn.Module],
    *,
    diffusion: torch.nn.Module,
    transformer: torch.nn.Module,
) -> tuple[tuple[str, torch.nn.Module], ...]:
    if not isinstance(value, Mapping):
        raise GraftPhaseANativeTrainingClosureError(
            "external_trainable_owner_modules must be a mapping"
        )
    rows: list[tuple[str, torch.nn.Module]] = []
    identities = {id(diffusion), id(transformer)}
    for name in sorted(value):
        module = value[name]
        if (
            not isinstance(name, str)
            or not name
            or not name.isascii()
            or name in ("diffusion", "transformer")
            or not isinstance(module, torch.nn.Module)
            or id(module) in identities
            or module.training
        ):
            raise GraftPhaseANativeTrainingClosureError(
                "external trainable-owner registry differs"
            )
        identities.add(id(module))
        rows.append((name, module))
    return tuple(rows)


def _validate_exclusive_trainable_scope(
    diffusion: torch.nn.Module,
    transformer: torch.nn.Module,
    external_owner_modules: Sequence[tuple[str, torch.nn.Module]],
    rows: Sequence[tuple[str, torch.nn.Parameter]],
) -> None:
    expected = {id(parameter) for _, parameter in rows}
    observed: set[int] = set()
    for module in (
        diffusion,
        transformer,
        *(module for _, module in external_owner_modules),
    ):
        for parameter in module.parameters():
            if parameter.requires_grad:
                observed.add(id(parameter))
    if observed != expected:
        raise GraftPhaseANativeTrainingClosureError(
            "live trainable scope is not exactly the authenticated registry"
        )


def _trainable_registry_digest(
    rows: Sequence[tuple[str, torch.nn.Parameter]],
) -> str:
    payload = {
        "rows": [
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "device_type": parameter.device.type,
            }
            for name, parameter in rows
        ]
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _owner_registry_digest(
    external_owner_modules: Sequence[tuple[str, torch.nn.Module]],
) -> str:
    payload = {
        "fixed_owners": ["diffusion", "transformer"],
        "external_owners": [name for name, _ in external_owner_modules],
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _validated_route_receipt(
    value: Mapping[str, Any], *, expected_route_kind: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GraftPhaseANativeTrainingClosureError(
            "forward route receipt must be a canonical mapping"
        )
    plain = dict(value)
    digest = plain.pop("digest", None)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or hashlib.sha256(_canonical_json_bytes(plain)).hexdigest() != digest
    ):
        raise GraftPhaseANativeTrainingClosureError(
            "forward route receipt digest differs from its canonical mapping"
        )
    required = {
        "schema_version": FORWARD_ROUTE_SCHEMA_VERSION,
        "route_kind": expected_route_kind,
        "inactive_schedule_policy": "exact_zero_update_not_trained",
        "target_queries_only": True,
        "condition_rows_written": False,
        "external_oracle_inputs": False,
    }
    active = plain.get("phase_a_active_schedule_indices")
    if (
        not isinstance(active, (list, tuple))
        or tuple(active) != PHASE_A_ACTIVE_SCHEDULE_INDICES
        or any(plain.get(key) != expected for key, expected in required.items())
    ):
        raise GraftPhaseANativeTrainingClosureError(
            "forward route receipt critical fields differ"
        )
    # Store a fresh JSON-owned copy so mutation of the caller's mapping or
    # nested active-index list cannot alter the authenticated capability.
    cloned = json.loads(_canonical_json_bytes({**plain, "digest": digest}))
    cloned["phase_a_active_schedule_indices"] = tuple(
        cloned["phase_a_active_schedule_indices"]
    )
    return MappingProxyType(cloned)


def _cpu_fake_route_receipt() -> Mapping[str, Any]:
    return _seal(
        {
            "schema_version": FORWARD_ROUTE_SCHEMA_VERSION,
            "route_kind": "cpu_fake_global_context",
            "phase_a_active_schedule_indices": list(
                PHASE_A_ACTIVE_SCHEDULE_INDICES
            ),
            "inactive_schedule_policy": "exact_zero_update_not_trained",
            "target_queries_only": True,
            "condition_rows_written": False,
            "external_oracle_inputs": False,
        }
    )


def _parameter_value_digest(
    rows: Sequence[tuple[str, torch.nn.Parameter]],
) -> str:
    payload = hashlib.sha256()
    for name, parameter in rows:
        payload.update(name.encode("utf-8"))
        payload.update(b"\0")
        payload.update(_tensor_bytes_sha256(parameter).encode("ascii"))
        payload.update(b"\n")
    return payload.hexdigest()


def _validate_binding_surface(
    *,
    diffusion: torch.nn.Module,
    transformer: torch.nn.Module,
    vendor_normalized_guidance: Callable[..., Any],
    momentum_buffer_factory: Callable[[float], Any],
    named_trainable_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    external_trainable_owner_modules: Mapping[str, torch.nn.Module],
) -> tuple[
    tuple[tuple[str, torch.nn.Parameter], ...],
    tuple[tuple[str, torch.nn.Module], ...],
]:
    if not isinstance(diffusion, torch.nn.Module) or not isinstance(
        transformer, torch.nn.Module
    ):
        raise GraftPhaseANativeTrainingClosureError(
            "diffusion and transformer must be torch modules"
        )
    if diffusion.training or transformer.training:
        raise GraftPhaseANativeTrainingClosureError(
            "native closure requires deterministic eval-mode modules"
        )
    if bool(getattr(transformer, "gradient_checkpointing", False)):
        raise GraftPhaseANativeTrainingClosureError(
            "native closure requires gradient checkpointing disabled for exact replay"
        )
    patch = getattr(transformer, "patch_vae_latent", None)
    shared = getattr(diffusion, "shared_step", None)
    _validate_callable_parameters(
        patch, required=("hidden_states", "source_id"), label="patch_vae_latent"
    )
    _validate_callable_parameters(
        shared,
        required=(
            "model_id",
            "noisy_latents",
            "timesteps",
            "cond_embeds",
            "rotary_embs",
            "batch_vae_seqlen",
            "batch_text_seqlen",
        ),
        label="shared_step",
    )
    _validate_callable_parameters(
        vendor_normalized_guidance,
        required=(
            "pred_cond",
            "pred_uncond",
            "guidance_scale",
            "momentum_buffer",
            "eta",
            "norm_threshold",
        ),
        label="normalized_guidance",
    )
    _validate_callable_parameters(
        momentum_buffer_factory, required=("momentum",), label="MomentumBuffer"
    )
    rows = _named_trainable_parameters(named_trainable_parameters)
    external_owners = _external_trainable_owners(
        external_trainable_owner_modules,
        diffusion=diffusion,
        transformer=transformer,
    )
    _validate_exclusive_trainable_scope(
        diffusion, transformer, external_owners, rows
    )
    return rows, external_owners


@dataclass(frozen=True, init=False)
class AuthenticatedNativeBindings:
    """Closed callable/owner registry, not a same-process security boundary.

    Construction is intentionally unavailable through the public dataclass
    constructor (and consequently through :func:`dataclasses.replace`).  The
    authenticators below are the only supported minting boundary.  Python code
    in this same process can still rewrite objects with reflection; receipts
    are provenance/fail-closed checks, not a cryptographic capability system.
    """

    diffusion: torch.nn.Module = field(repr=False, compare=False)
    transformer: torch.nn.Module = field(repr=False, compare=False)
    vendor_normalized_guidance: Callable[..., Any] = field(repr=False, compare=False)
    momentum_buffer_factory: Callable[[float], Any] = field(repr=False, compare=False)
    forward_context_factory: Callable[..., ContextManager[Any]] = field(
        repr=False, compare=False
    )
    named_trainable_parameters: tuple[tuple[str, torch.nn.Parameter], ...] = field(
        repr=False, compare=False
    )
    external_trainable_owner_modules: tuple[tuple[str, torch.nn.Module], ...] = field(
        repr=False, compare=False
    )
    forward_route_receipt: Mapping[str, Any] = field(repr=False, compare=False)
    official_pinned_code: bool
    test_only: bool
    binding_label: str
    code_receipt: Mapping[str, Any]
    _shared_step_function: Any = field(repr=False, compare=False)
    _shared_step_owner: Any = field(repr=False, compare=False)
    _patch_function: Any = field(repr=False, compare=False)
    _patch_owner: Any = field(repr=False, compare=False)
    _guidance_function: Any = field(repr=False, compare=False)
    _guidance_owner: Any = field(repr=False, compare=False)
    _momentum_function: Any = field(repr=False, compare=False)
    _momentum_owner: Any = field(repr=False, compare=False)
    _official_vendor_module: Any = field(repr=False, compare=False)
    _forward_context_function: Any = field(repr=False, compare=False)
    _forward_context_owner: Any = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "AuthenticatedNativeBindings":
        raise GraftPhaseANativeTrainingClosureError(
            "AuthenticatedNativeBindings has no public constructor"
        )

    @classmethod
    def _mint(cls, *, token: object, **values: Any) -> "AuthenticatedNativeBindings":
        if token is not _BINDING_TOKEN:
            raise GraftPhaseANativeTrainingClosureError(
                "native binding mint token differs"
            )
        instance = object.__new__(cls)
        if set(values) != set(cls.__dataclass_fields__):
            raise GraftPhaseANativeTrainingClosureError(
                "native binding mint fields differ from the closed schema"
            )
        for name in cls.__dataclass_fields__:
            if name not in values:
                raise GraftPhaseANativeTrainingClosureError(
                    f"native binding mint omitted {name}"
                )
            object.__setattr__(instance, name, values[name])
        instance.assert_live()
        return instance

    @property
    def forward_route_receipt_digest(self) -> str:
        return str(self.forward_route_receipt["digest"])

    @property
    def active_schedule_indices(self) -> tuple[int, ...]:
        return tuple(self.forward_route_receipt["phase_a_active_schedule_indices"])

    @staticmethod
    def _callable_pair(value: Callable[..., Any]) -> tuple[Any, Any]:
        return getattr(value, "__func__", value), getattr(value, "__self__", None)

    def _assert_code_receipt(self) -> None:
        plain = dict(self.code_receipt)
        digest = plain.pop("digest", None)
        if (
            not isinstance(digest, str)
            or hashlib.sha256(_canonical_json_bytes(plain)).hexdigest() != digest
        ):
            raise GraftPhaseANativeTrainingClosureError(
                "native binding code receipt digest differs"
            )
        expected = {
            "schema_version": BINDING_SCHEMA_VERSION,
            "binding_label": self.binding_label,
            "official_pinned_code": self.official_pinned_code,
            "test_only": self.test_only,
            "forward_route_receipt_digest": self.forward_route_receipt_digest,
            "trainable_registry_digest": _trainable_registry_digest(
                self.named_trainable_parameters
            ),
            "trainable_owner_registry_digest": _owner_registry_digest(
                self.external_trainable_owner_modules
            ),
            "trainable_owner_names": [
                "diffusion",
                "transformer",
                *(name for name, _ in self.external_trainable_owner_modules),
            ],
            "trainable_scope_claim": "exact_registry_closure",
            "python_same_process_security_boundary": False,
        }
        if any(plain.get(key) != value for key, value in expected.items()):
            raise GraftPhaseANativeTrainingClosureError(
                "native binding receipt fields do not match live bindings"
            )

    def assert_live(self) -> None:
        try:
            if self._token is not _BINDING_TOKEN:
                raise GraftPhaseANativeTrainingClosureError(
                    "native binding capability was not minted by an authenticator"
                )
            shared = getattr(self.diffusion, "shared_step", None)
            patch = getattr(self.transformer, "patch_vae_latent", None)
            callable_rows = (
                (shared, self._shared_step_function, self._shared_step_owner),
                (patch, self._patch_function, self._patch_owner),
                (
                    self.vendor_normalized_guidance,
                    self._guidance_function,
                    self._guidance_owner,
                ),
                (
                    self.momentum_buffer_factory,
                    self._momentum_function,
                    self._momentum_owner,
                ),
                (
                    self.forward_context_factory,
                    self._forward_context_function,
                    self._forward_context_owner,
                ),
            )
            if (
                any(self._callable_pair(value) != (function, owner) for value, function, owner in callable_rows)
                or (
                    self.official_pinned_code
                    and (
                        self._official_vendor_module is None
                        or getattr(
                            self._official_vendor_module,
                            "normalized_guidance",
                            None,
                        )
                        is not self.vendor_normalized_guidance
                        or getattr(
                            self._official_vendor_module, "MomentumBuffer", None
                        )
                        is not self.momentum_buffer_factory
                    )
                )
                or self.diffusion.training
                or self.transformer.training
                or any(
                    module.training
                    for _, module in self.external_trainable_owner_modules
                )
                or bool(getattr(self.transformer, "gradient_checkpointing", False))
            ):
                raise GraftPhaseANativeTrainingClosureError(
                    "authenticated native callable/eval closure changed"
                )
            expected_route_kind = (
                "cpu_fake_global_context"
                if self.test_only
                else "identity_rebinder_v1"
            )
            _validated_route_receipt(
                self.forward_route_receipt,
                expected_route_kind=expected_route_kind,
            )
            _validate_exclusive_trainable_scope(
                self.diffusion,
                self.transformer,
                self.external_trainable_owner_modules,
                self.named_trainable_parameters,
            )
            self._assert_code_receipt()
        except GraftPhaseANativeTrainingClosureError:
            raise
        except Exception as error:
            raise GraftPhaseANativeTrainingClosureError(
                "native binding object is incomplete or forged"
            ) from error

    def receipt(self) -> Mapping[str, Any]:
        self.assert_live()
        return self.code_receipt


def authenticate_cpu_test_fakes(
    *,
    diffusion: torch.nn.Module,
    transformer: torch.nn.Module,
    vendor_normalized_guidance: Callable[..., Any],
    momentum_buffer_factory: Callable[[float], Any],
    named_trainable_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    external_trainable_owner_modules: Mapping[str, torch.nn.Module],
    test_name: str,
    forward_context_factory: Optional[Callable[..., ContextManager[Any]]] = None,
) -> AuthenticatedNativeBindings:
    """Authenticate injectable CPU fakes while permanently denying official claims."""

    if (
        not isinstance(test_name, str)
        or not test_name.startswith("cpu_fake:")
        or len(test_name) > 160
    ):
        raise GraftPhaseANativeTrainingClosureError(
            "CPU fake name must use the explicit 'cpu_fake:' namespace"
        )
    rows, external_owners = _validate_binding_surface(
        diffusion=diffusion,
        transformer=transformer,
        vendor_normalized_guidance=vendor_normalized_guidance,
        momentum_buffer_factory=momentum_buffer_factory,
        named_trainable_parameters=named_trainable_parameters,
        external_trainable_owner_modules=external_trainable_owner_modules,
    )
    route_factory = (
        _cpu_fake_global_context
        if forward_context_factory is None
        else forward_context_factory
    )
    _validate_callable_parameters(
        route_factory, required=("request",), label="CPU fake forward context factory"
    )
    devices = {
        parameter.device.type
        for module in (
            diffusion,
            transformer,
            *(module for _, module in external_owners),
        )
        for parameter in module.parameters()
    }
    if devices - {"cpu"}:
        raise GraftPhaseANativeTrainingClosureError("CPU fake contains non-CPU parameters")
    route_receipt = _validated_route_receipt(
        _cpu_fake_route_receipt(), expected_route_kind="cpu_fake_global_context"
    )
    receipt = _seal(
        {
            "schema_version": BINDING_SCHEMA_VERSION,
            "binding_label": test_name,
            "official_pinned_code": False,
            "test_only": True,
            "pinned_bernini_commit": PINNED_BERNINI_COMMIT,
            "pinned_wan_diffusion_sha256": PINNED_WAN_DIFFUSION_SHA256,
            "pinned_transformer_wan_sha256": PINNED_TRANSFORMER_WAN_SHA256,
            "file_hashes_verified": False,
            "real_checkpoint_loaded": False,
            "gpu_execution_authorized": False,
            "forward_route_receipt_digest": route_receipt["digest"],
            "trainable_registry_digest": _trainable_registry_digest(rows),
            "trainable_owner_registry_digest": _owner_registry_digest(
                external_owners
            ),
            "trainable_owner_names": [
                "diffusion",
                "transformer",
                *(name for name, _ in external_owners),
            ],
            "trainable_scope_claim": "exact_registry_closure",
            "python_same_process_security_boundary": False,
            "scientific_claim_authorized": False,
        }
    )
    pairs = {
        "shared": AuthenticatedNativeBindings._callable_pair(diffusion.shared_step),
        "patch": AuthenticatedNativeBindings._callable_pair(
            transformer.patch_vae_latent
        ),
        "guidance": AuthenticatedNativeBindings._callable_pair(
            vendor_normalized_guidance
        ),
        "momentum": AuthenticatedNativeBindings._callable_pair(
            momentum_buffer_factory
        ),
        "route": AuthenticatedNativeBindings._callable_pair(route_factory),
    }
    return AuthenticatedNativeBindings._mint(
        token=_BINDING_TOKEN,
        diffusion=diffusion,
        transformer=transformer,
        vendor_normalized_guidance=vendor_normalized_guidance,
        momentum_buffer_factory=momentum_buffer_factory,
        forward_context_factory=route_factory,
        named_trainable_parameters=rows,
        external_trainable_owner_modules=external_owners,
        forward_route_receipt=route_receipt,
        official_pinned_code=False,
        test_only=True,
        binding_label=test_name,
        code_receipt=receipt,
        _shared_step_function=pairs["shared"][0],
        _shared_step_owner=pairs["shared"][1],
        _patch_function=pairs["patch"][0],
        _patch_owner=pairs["patch"][1],
        _guidance_function=pairs["guidance"][0],
        _guidance_owner=pairs["guidance"][1],
        _momentum_function=pairs["momentum"][0],
        _momentum_owner=pairs["momentum"][1],
        _official_vendor_module=None,
        _forward_context_function=pairs["route"][0],
        _forward_context_owner=pairs["route"][1],
        _token=_BINDING_TOKEN,
    )


def authenticate_pinned_native_bindings(
    *,
    diffusion: torch.nn.Module,
    transformer: torch.nn.Module,
    named_trainable_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    external_trainable_owner_modules: Mapping[str, torch.nn.Module],
    wan_diffusion_path: Path | str,
    transformer_wan_path: Path | str,
    bernini_commit: str,
    forward_context_factory: Optional[Callable[..., ContextManager[Any]]] = None,
    forward_route_receipt: Optional[Mapping[str, Any]] = None,
) -> AuthenticatedNativeBindings:
    """Hash-bind the live official symbols before a CUDA closure is opened."""

    if bernini_commit != PINNED_BERNINI_COMMIT:
        raise GraftPhaseANativeTrainingClosureError("Bernini commit differs")
    wan_path = Path(wan_diffusion_path).resolve(strict=True)
    transformer_path = Path(transformer_wan_path).resolve(strict=True)
    if (
        _file_sha256(wan_path) != PINNED_WAN_DIFFUSION_SHA256
        or _file_sha256(transformer_path) != PINNED_TRANSFORMER_WAN_SHA256
    ):
        raise GraftPhaseANativeTrainingClosureError("pinned Bernini source hash differs")
    try:
        wan_module = importlib.import_module("bernini.models.wan_diffusion")
        transformer_module = importlib.import_module("bernini.models.transformer_wan")
    except Exception as error:
        raise GraftPhaseANativeTrainingClosureError(
            "pinned Bernini modules are not importable"
        ) from error
    if (
        Path(wan_module.__file__).resolve(strict=True) != wan_path
        or Path(transformer_module.__file__).resolve(strict=True) != transformer_path
        or type(diffusion).__module__ != "bernini.models.wan_diffusion"
        or type(diffusion).__name__ != "GEN_Wanx22"
        or type(transformer).__module__ != "bernini.models.transformer_wan"
        or type(transformer).__name__ != "WanTransformer3DModel"
        or getattr(diffusion, "transformer", None) is not transformer
        or getattr(diffusion, "transformer_2", None) is not None
        or getattr(wan_module, "normalized_guidance", None) is None
        or getattr(wan_module, "MomentumBuffer", None) is None
        or getattr(diffusion.shared_step, "__func__", diffusion.shared_step)
        is not wan_module.GEN_Wanx22.shared_step
        or getattr(transformer.patch_vae_latent, "__func__", transformer.patch_vae_latent)
        is not transformer_module.WanTransformer3DModel.patch_vae_latent
    ):
        raise GraftPhaseANativeTrainingClosureError(
            "live Bernini symbols do not belong to the hash-bound source"
        )
    rows, external_owners = _validate_binding_surface(
        diffusion=diffusion,
        transformer=transformer,
        vendor_normalized_guidance=wan_module.normalized_guidance,
        momentum_buffer_factory=wan_module.MomentumBuffer,
        named_trainable_parameters=named_trainable_parameters,
        external_trainable_owner_modules=external_trainable_owner_modules,
    )
    if forward_context_factory is None or forward_route_receipt is None:
        raise GraftPhaseANativeTrainingClosureError(
            "official binding requires an externally authenticated forward route"
        )
    _validate_callable_parameters(
        forward_context_factory,
        required=("request",),
        label="official forward context factory",
    )
    route_receipt = _validated_route_receipt(
        forward_route_receipt, expected_route_kind="identity_rebinder_v1"
    )
    receipt = _seal(
        {
            "schema_version": BINDING_SCHEMA_VERSION,
            "binding_label": "official_pinned_bernini_r_1p3b",
            "official_pinned_code": True,
            "test_only": False,
            "pinned_bernini_commit": PINNED_BERNINI_COMMIT,
            "pinned_wan_diffusion_sha256": PINNED_WAN_DIFFUSION_SHA256,
            "pinned_transformer_wan_sha256": PINNED_TRANSFORMER_WAN_SHA256,
            "file_hashes_verified": True,
            "checkpoint_weight_content_verified_by_authenticator": False,
            "checkpoint_content_binding_required_from_gpu_runner": True,
            "single_expert": EXPECTED_MODEL_ID,
            "forward_route_receipt_digest": route_receipt["digest"],
            "trainable_registry_digest": _trainable_registry_digest(rows),
            "trainable_owner_registry_digest": _owner_registry_digest(
                external_owners
            ),
            "trainable_owner_names": [
                "diffusion",
                "transformer",
                *(name for name, _ in external_owners),
            ],
            "trainable_scope_claim": "exact_registry_closure",
            "python_same_process_security_boundary": False,
            "scientific_claim_authorized": False,
        }
    )
    pairs = {
        "shared": AuthenticatedNativeBindings._callable_pair(diffusion.shared_step),
        "patch": AuthenticatedNativeBindings._callable_pair(
            transformer.patch_vae_latent
        ),
        "guidance": AuthenticatedNativeBindings._callable_pair(
            wan_module.normalized_guidance
        ),
        "momentum": AuthenticatedNativeBindings._callable_pair(
            wan_module.MomentumBuffer
        ),
        "route": AuthenticatedNativeBindings._callable_pair(
            forward_context_factory
        ),
    }
    return AuthenticatedNativeBindings._mint(
        token=_BINDING_TOKEN,
        diffusion=diffusion,
        transformer=transformer,
        vendor_normalized_guidance=wan_module.normalized_guidance,
        momentum_buffer_factory=wan_module.MomentumBuffer,
        forward_context_factory=forward_context_factory,
        named_trainable_parameters=rows,
        external_trainable_owner_modules=external_owners,
        forward_route_receipt=route_receipt,
        official_pinned_code=True,
        test_only=False,
        binding_label="official_pinned_bernini_r_1p3b",
        code_receipt=receipt,
        _shared_step_function=pairs["shared"][0],
        _shared_step_owner=pairs["shared"][1],
        _patch_function=pairs["patch"][0],
        _patch_owner=pairs["patch"][1],
        _guidance_function=pairs["guidance"][0],
        _guidance_owner=pairs["guidance"][1],
        _momentum_function=pairs["momentum"][0],
        _momentum_owner=pairs["momentum"][1],
        _official_vendor_module=wan_module,
        _forward_context_function=pairs["route"][0],
        _forward_context_owner=pairs["route"][1],
        _token=_BINDING_TOKEN,
    )


def unpack_wan_target_velocity(
    packed: torch.Tensor, *, spatial_shape: Sequence[int]
) -> torch.Tensor:
    """Invert the pinned Wan ``(t,h,w),(pt,ph,pw,c)`` order, preserving graph."""

    shape = tuple(int(item) for item in spatial_shape)
    if (
        len(shape) != 5
        or shape[:3] != (1, EXPECTED_LATENT_CHANNELS, EXPECTED_LATENT_PHASES)
        or shape[3] <= 0
        or shape[4] <= 0
        or shape[3] % EXPECTED_PATCH_SIZE[1]
        or shape[4] % EXPECTED_PATCH_SIZE[2]
    ):
        raise GraftPhaseANativeTrainingClosureError(
            "spatial_shape must be exact81 [1,16,21,H,W] with positive even H/W"
        )
    batch, channels, phases, height, width = shape
    patch_h, patch_w = height // 2, width // 2
    tokens = phases * patch_h * patch_w
    value = _finite_tensor(packed, label="packed target velocity")
    if tuple(value.shape) != (batch, tokens, EXPECTED_PACKED_CHANNELS):
        raise GraftPhaseANativeTrainingClosureError(
            "packed target velocity differs from official Wan geometry"
        )
    patches = value.reshape(batch, phases, patch_h, patch_w, 1, 2, 2, channels)
    result = (
        patches.permute(0, 7, 1, 4, 2, 5, 3, 6)
        .reshape(batch, channels, phases, height, width)
        .contiguous()
    )
    if not bool(torch.isfinite(result).all().item()):
        raise GraftPhaseANativeTrainingClosureError(
            "Wan unpack produced a non-finite spatial velocity"
        )
    return result


@dataclass(frozen=True)
class NativeFieldMeasurement:
    negative_full_raw: torch.Tensor = field(repr=False, compare=False)
    positive_full_raw: torch.Tensor = field(repr=False, compare=False)
    negative_spatial_raw: torch.Tensor = field(repr=False, compare=False)
    positive_spatial_raw: torch.Tensor = field(repr=False, compare=False)
    negative_clean: torch.Tensor = field(repr=False, compare=False)
    positive_clean: torch.Tensor = field(repr=False, compare=False)


@dataclass(frozen=True)
class NativePhaseAFlowMatchingVJP:
    guided_clean: torch.Tensor = field(repr=False, compare=False)
    predicted_velocity: torch.Tensor = field(repr=False, compare=False)
    same_source_target_velocity: torch.Tensor = field(repr=False, compare=False)
    flow_matching_loss: torch.Tensor = field(repr=False, compare=False)
    guided_clean_cotangent: torch.Tensor = field(repr=False, compare=False)
    negative_clean_cotangent: torch.Tensor = field(repr=False, compare=False)
    positive_clean_cotangent: torch.Tensor = field(repr=False, compare=False)
    negative_raw_cotangent: torch.Tensor = field(repr=False, compare=False)
    positive_raw_cotangent: torch.Tensor = field(repr=False, compare=False)


@dataclass(frozen=True)
class NativeTrainingClosureResult:
    guided_clean: torch.Tensor = field(repr=False, compare=False)
    flow_matching_loss: torch.Tensor = field(repr=False, compare=False)
    receipt: Mapping[str, Any]


class PhaseANativeTrainingClosure:
    """Single-use measurement -> fixed FM-VJP -> serial-replay state machine."""

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
        self._call_trace: list[tuple[str, str]] = []
        self._measurement: Optional[NativeFieldMeasurement] = None
        self._leaf_vjp: Optional[NativePhaseAFlowMatchingVJP] = None
        self._receipt: Optional[Mapping[str, Any]] = None
        self._forward_observation_receipts: list[Mapping[str, Any]] = []
        self._replay_pack_gradient_receipts: list[Mapping[str, Any]] = []
        self._initial_parameter_snapshots: tuple[
            tuple[torch.nn.Parameter, torch.Tensor], ...
        ] = ()
        try:
            if type(bindings) is not AuthenticatedNativeBindings:
                raise GraftPhaseANativeTrainingClosureError(
                    "bindings must be minted by an explicit authenticator"
                )
            self.bindings = bindings
            self._initial_parameter_snapshots = tuple(
                (parameter, parameter.detach().clone())
                for _, parameter in bindings.named_trainable_parameters
            )
            bindings.assert_live()
            if any(
                parameter.grad is not None
                for _, parameter in bindings.named_trainable_parameters
            ):
                raise GraftPhaseANativeTrainingClosureError(
                    "trainable-registry gradients must be empty when a closure is opened"
                )
            self.source_video = self._validate_spatial_latent(
                source_video, label="source video latent"
            )
            self.noisy_target = self._validate_spatial_latent(
                noisy_target, label="noisy target state"
            )
            if (
                self.source_video.shape != self.noisy_target.shape
                or self.source_video.device != self.noisy_target.device
                or self.source_video is self.noisy_target
                or _shares_storage(self.source_video, self.noisy_target)
            ):
                raise GraftPhaseANativeTrainingClosureError(
                    "source video and noisy target must be separate same-geometry tensors"
                )
            if any(
                parameter.device != self.noisy_target.device
                for _, parameter in bindings.named_trainable_parameters
            ):
                raise GraftPhaseANativeTrainingClosureError(
                    "trainable registry and native target must share one device"
                )
            self.negative_condition = self._validate_condition(
                negative_condition, label="negative condition"
            )
            self.positive_condition = self._validate_condition(
                positive_condition, label="positive condition"
            )
            if (
                self.negative_condition.device != self.noisy_target.device
                or self.positive_condition.device != self.noisy_target.device
                or self.negative_condition is self.positive_condition
                or _shares_storage(self.negative_condition, self.positive_condition)
                or torch.equal(self.negative_condition, self.positive_condition)
            ):
                raise GraftPhaseANativeTrainingClosureError(
                    "negative and positive prompt conditions are not distinct on-device objects"
                )
            self.schedule_index, self.sigma, self.timestep = self._validate_coordinate(
                schedule_index=schedule_index, sigma=sigma, timestep=timestep
            )
            self._cell_active = self.schedule_index in bindings.active_schedule_indices
            self._parameter_digest = _parameter_value_digest(
                self.bindings.named_trainable_parameters
            )
            self._immutable_tensors = (
                self.source_video,
                self.noisy_target,
                self.negative_condition,
                self.positive_condition,
                self.sigma,
                self.timestep,
            )
            labels = (
                "source_video",
                "noisy_target",
                "negative_condition",
                "positive_condition",
                "sigma",
                "timestep",
            )
            self._immutable_digests = tuple(
                _tensor_bytes_sha256(value) for value in self._immutable_tensors
            )
            self._input_digests = dict(zip(labels, self._immutable_digests))
            self._immutable_snapshots = tuple(
                value.detach().clone() for value in self._immutable_tensors
            )
            self._immutable_object_ids = tuple(
                id(value) for value in self._immutable_tensors
            )
            self._build_pack()
            self._assert_immutable()
            self._phase = "new"
        except Exception:
            self._poison()
            raise

    def _poison(self) -> None:
        cleanup_error: Optional[Exception] = None
        visual_pack = getattr(self, "_visual_pack", None)
        if isinstance(visual_pack, torch.Tensor):
            try:
                visual_pack.grad = None
                if visual_pack.is_leaf and visual_pack.requires_grad:
                    visual_pack.requires_grad_(False)
            except Exception as error:  # pragma: no cover - catastrophic mutation
                cleanup_error = error
        for parameter, snapshot in self._initial_parameter_snapshots:
            try:
                if not parameter.requires_grad:
                    parameter.requires_grad_(True)
                with torch.no_grad():
                    parameter.copy_(snapshot)
            except Exception as error:  # pragma: no cover - catastrophic mutation
                cleanup_error = error
            try:
                parameter.grad = None
            except Exception as error:  # pragma: no cover - catastrophic mutation
                cleanup_error = error
        self._phase = "failed"
        if cleanup_error is not None:
            raise GraftPhaseANativeTrainingClosureError(
                "closure failed and could not restore its trainable registry snapshot"
            ) from cleanup_error

    def _fail(self, error: Exception) -> None:
        try:
            self._poison()
        except Exception as cleanup_error:
            raise cleanup_error from error

    @staticmethod
    def _validate_spatial_latent(value: Any, *, label: str) -> torch.Tensor:
        tensor = _finite_tensor(value, label=label)
        if (
            tensor.dtype != torch.float32
            or tensor.ndim != 5
            or tuple(tensor.shape[:3])
            != (1, EXPECTED_LATENT_CHANNELS, EXPECTED_LATENT_PHASES)
            or int(tensor.shape[3]) <= 0
            or int(tensor.shape[4]) <= 0
            or int(tensor.shape[3]) % 2
            or int(tensor.shape[4]) % 2
            or tensor.requires_grad
            or tensor.grad_fn is not None
        ):
            raise GraftPhaseANativeTrainingClosureError(
                f"{label} must be detached FP32 exact81 [1,16,21,H,W]"
            )
        return tensor

    def _validate_condition(self, value: Any, *, label: str) -> torch.Tensor:
        tensor = _finite_tensor(value, label=label)
        official_shape = (1, EXPECTED_TEXT_TOKENS, EXPECTED_TEXT_DIM)
        cpu_fake_shape = (1, 2, 4)
        if (
            tuple(tensor.shape)
            not in (
                official_shape,
                *((cpu_fake_shape,) if self.bindings.test_only else ()),
            )
            or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or tensor.requires_grad
            or tensor.grad_fn is not None
        ):
            allowed = (
                "[1,512,4096]"
                if not self.bindings.test_only
                else "[1,512,4096] or compact [1,2,4] CPU fake"
            )
            raise GraftPhaseANativeTrainingClosureError(
                f"{label} must be detached {allowed} floating text embeddings"
            )
        return tensor

    def _validate_coordinate(
        self, *, schedule_index: int, sigma: Any, timestep: Any
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        if type(schedule_index) is not int or schedule_index not in range(EXPECTED_STEPS):
            raise GraftPhaseANativeTrainingClosureError(
                "schedule_index must select one of the 40 native forward cells"
            )
        sigma_tensor = _finite_tensor(sigma, label="native sigma")
        timestep_tensor = timestep
        if (
            sigma_tensor.dtype != torch.float32
            or sigma_tensor.ndim != 0
            or sigma_tensor.device.type != "cpu"
            or sigma_tensor.requires_grad
            or sigma_tensor.grad_fn is not None
            or not isinstance(timestep_tensor, torch.Tensor)
            or timestep_tensor.device.type == "meta"
            or timestep_tensor.dtype != torch.int64
            or tuple(timestep_tensor.shape) != (1,)
            or timestep_tensor.device != self.noisy_target.device
            or timestep_tensor.requires_grad
        ):
            raise GraftPhaseANativeTrainingClosureError(
                "native coordinate requires CPU scalar FP32 sigma and device-local INT64 [1] timestep"
            )
        expected_sigma_hex = sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[
            schedule_index
        ]
        observed_sigma_hex = struct.pack(">f", float(sigma_tensor.item())).hex()
        if (
            observed_sigma_hex != expected_sigma_hex
            or int(timestep_tensor.item())
            != sigma_strata.PINNED_TIMESTEPS[schedule_index]
        ):
            raise GraftPhaseANativeTrainingClosureError(
                "sigma/timestep do not name the same pinned exact40 cell"
            )
        return schedule_index, sigma_tensor, timestep_tensor

    def _assert_immutable(self) -> None:
        self.bindings.assert_live()
        if any(
            not torch.equal(parameter.detach(), snapshot)
            for parameter, snapshot in self._initial_parameter_snapshots
        ):
            raise GraftPhaseANativeTrainingClosureError(
                "trainable-registry parameter values changed inside the closure"
            )
        if (
            tuple(id(value) for value in self._immutable_tensors)
            != self._immutable_object_ids
            or any(
                not torch.equal(value, snapshot)
                for value, snapshot in zip(
                    self._immutable_tensors, self._immutable_snapshots
                )
            )
        ):
            raise GraftPhaseANativeTrainingClosureError(
                "source/state/prompt/schedule inputs mutated inside the closure"
            )
        for value, identity, snapshot in (
            (
                self._visual_pack,
                self._visual_pack_object_id,
                self._visual_pack_snapshot,
            ),
            (
                self._rotary_pack,
                self._rotary_pack_object_id,
                self._rotary_pack_snapshot,
            ),
        ):
            if id(value) != identity or not torch.equal(value, snapshot):
                raise GraftPhaseANativeTrainingClosureError(
                    "shared native visual or rotary pack mutated"
                )

    def _build_pack(self) -> None:
        patch = self.bindings.transformer.patch_vae_latent
        patch_rows: list[tuple[float, torch.Tensor, torch.Tensor]] = []
        with torch.no_grad(), self._autocast_context():
            for source_id, value in zip(
                EXPECTED_PATCH_SOURCE_IDS, (self.source_video, self.noisy_target)
            ):
                result = patch(
                    hidden_states=value.to(dtype=torch.bfloat16), source_id=source_id
                )
                if not isinstance(result, tuple) or len(result) != 2:
                    raise GraftPhaseANativeTrainingClosureError(
                        "patch_vae_latent must return (tokens, rotary)"
                    )
                tokens = _finite_tensor(result[0], label="native patch tokens")
                rotary = _finite_tensor(result[1], label="native patch rotary")
                patch_rows.append((source_id, tokens.detach(), rotary.detach()))
        source_tokens, target_tokens = patch_rows[0][1], patch_rows[1][1]
        source_rotary, target_rotary = patch_rows[0][2], patch_rows[1][2]
        height, width = map(int, self.noisy_target.shape[-2:])
        token_count = EXPECTED_LATENT_PHASES * (height // 2) * (width // 2)
        for label, tokens in (("source", source_tokens), ("target", target_tokens)):
            if (
                tuple(tokens.shape) != (1, token_count, EXPECTED_HIDDEN_DIM)
                or tokens.dtype != torch.bfloat16
                or tokens.device != self.noisy_target.device
                or tokens.requires_grad
                or tokens.grad_fn is not None
            ):
                raise GraftPhaseANativeTrainingClosureError(
                    f"{label} patch differs from Bernini-R 1.3B BF16 geometry"
                )
        for label, rotary in (("source", source_rotary), ("target", target_rotary)):
            if (
                rotary.ndim != 4
                or tuple(rotary.shape[:3]) != (1, 1, token_count)
                or int(rotary.shape[3]) <= 0
                or rotary.device != self.noisy_target.device
                or rotary.requires_grad
                or rotary.grad_fn is not None
            ):
                raise GraftPhaseANativeTrainingClosureError(
                    f"{label} rotary differs from Bernini patch geometry"
                )
        if source_rotary.dtype != target_rotary.dtype or source_rotary.shape[3] != target_rotary.shape[3]:
            raise GraftPhaseANativeTrainingClosureError(
                "source/target rotary coordinates are incompatible"
            )
        self._source_tokens = token_count
        self._target_tokens = token_count
        self._visual_pack = torch.cat((source_tokens, target_tokens), dim=1).detach()
        self._rotary_pack = torch.cat((source_rotary, target_rotary), dim=2).detach()
        if (
            tuple(self._visual_pack.shape)
            != (1, token_count * 2, EXPECTED_HIDDEN_DIM)
            or tuple(self._rotary_pack.shape[:3]) != (1, 1, token_count * 2)
            or self._visual_pack.requires_grad
            or self._rotary_pack.requires_grad
        ):
            raise GraftPhaseANativeTrainingClosureError(
                "source-prefix/target-suffix pack geometry differs"
            )
        self._visual_pack_digest = _tensor_bytes_sha256(self._visual_pack)
        self._rotary_pack_digest = _tensor_bytes_sha256(self._rotary_pack)
        self._visual_pack_object_id = id(self._visual_pack)
        self._rotary_pack_object_id = id(self._rotary_pack)
        self._visual_pack_snapshot = self._visual_pack.detach().clone()
        self._rotary_pack_snapshot = self._rotary_pack.detach().clone()

    def _autocast_context(self) -> ContextManager[Any]:
        if self.noisy_target.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _shared_forward(self, *, phase: str, role: str, graph_enabled: bool) -> torch.Tensor:
        if role not in BRANCH_ORDER or phase not in ("measurement", "replay"):
            raise GraftPhaseANativeTrainingClosureError("native branch call label differs")
        expected = (
            ("measurement", "negative"),
            ("measurement", "positive"),
            ("replay", "negative"),
            ("replay", "positive"),
        )
        position = len(self._call_trace)
        if position >= len(expected) or expected[position] != (phase, role):
            raise GraftPhaseANativeTrainingClosureError(
                "native shared-step branch order differs"
            )
        self._assert_immutable()
        condition = (
            self.negative_condition if role == "negative" else self.positive_condition
        )
        route_request = NativeForwardContextRequest(
            phase=phase,
            role=role,
            visual_pack=self._visual_pack,
            rotary_pack=self._rotary_pack,
            timestep=self.timestep,
            schedule_index=self.schedule_index,
            sigma=float(self.sigma.item()),
            total_tokens=self._source_tokens + self._target_tokens,
            condition_tokens=self._source_tokens,
            target_tokens=self._target_tokens,
        )
        grad_context = torch.enable_grad() if graph_enabled else torch.no_grad()
        # Route construction itself may rebuild a graph-bearing identity atlas.
        # It therefore has to occur *inside* the branch's grad/autocast mode,
        # not merely have its returned context entered there.
        with grad_context, self._autocast_context():
            route_context = self.bindings.forward_context_factory(
                request=route_request
            )
            if not (
                callable(getattr(route_context, "__enter__", None))
                and callable(getattr(route_context, "__exit__", None))
            ):
                raise GraftPhaseANativeTrainingClosureError(
                    "authenticated forward route did not return a context manager"
                )
            with route_context as observed_context:
                observation = _validated_forward_context_observation(
                    request=route_request,
                    value=observed_context,
                )
                self._forward_observation_receipts.append(
                    _forward_context_observation_receipt(
                        request=route_request,
                        observation=observation,
                    )
                )
                raw = self.bindings.diffusion.shared_step(
                    model_id=EXPECTED_MODEL_ID,
                    noisy_latents=self._visual_pack,
                    timesteps=self.timestep,
                    cond_embeds=condition,
                    rotary_embs=self._rotary_pack,
                    batch_vae_seqlen=[self._source_tokens + self._target_tokens],
                    batch_text_seqlen=[int(condition.shape[1])],
                )
        value = _finite_tensor(raw, label=f"{phase} {role} raw output")
        if (
            tuple(value.shape)
            != (1, self._source_tokens + self._target_tokens, EXPECTED_PACKED_CHANNELS)
            or value.dtype != torch.bfloat16
            or value.device != self.noisy_target.device
        ):
            raise GraftPhaseANativeTrainingClosureError(
                f"{phase} {role} raw output is not full packed BF16 Bernini velocity"
            )
        if graph_enabled and self._cell_active:
            if not value.requires_grad or value.grad_fn is None:
                raise GraftPhaseANativeTrainingClosureError(
                    f"{role} active-cell replay raw output is detached from the native pack/route graph"
                )
        elif value.requires_grad or value.grad_fn is not None:
            raise GraftPhaseANativeTrainingClosureError(
                f"{role} measurement/inactive replay unexpectedly retained a graph"
            )
        self._call_trace.append((phase, role))
        self._assert_immutable()
        return value

    def measure(self) -> NativeFieldMeasurement:
        """Run exactly two detached shared-step measurements on one native pack."""

        try:
            if self._phase != "new":
                raise GraftPhaseANativeTrainingClosureError(
                    "measurement is single-use and must be the first phase"
                )
            full: list[torch.Tensor] = []
            spatial: list[torch.Tensor] = []
            clean: list[torch.Tensor] = []
            for role in BRANCH_ORDER:
                raw = self._shared_forward(
                    phase="measurement", role=role, graph_enabled=False
                ).detach().clone()
                tail = raw[:, -self._target_tokens :, :]
                unpacked = unpack_wan_target_velocity(
                    tail, spatial_shape=self.noisy_target.shape
                ).detach().contiguous()
                clean_value = (
                    self.noisy_target - self.sigma * unpacked
                ).detach().contiguous()
                if clean_value.dtype != torch.float32:
                    raise GraftPhaseANativeTrainingClosureError(
                        "pinned x_t - sigma * raw order did not produce FP32 clean state"
                    )
                full.append(raw)
                spatial.append(unpacked)
                clean.append(clean_value)
            self._measurement = NativeFieldMeasurement(
                negative_full_raw=full[0],
                positive_full_raw=full[1],
                negative_spatial_raw=spatial[0],
                positive_spatial_raw=spatial[1],
                negative_clean=clean[0],
                positive_clean=clean[1],
            )
            self._measurement_digests = tuple(
                _tensor_bytes_sha256(value)
                for value in (
                    self._measurement.negative_full_raw,
                    self._measurement.positive_full_raw,
                    self._measurement.negative_spatial_raw,
                    self._measurement.positive_spatial_raw,
                    self._measurement.negative_clean,
                    self._measurement.positive_clean,
                )
            )
            self._measurement_snapshots = tuple(
                value.detach().clone()
                for value in (
                    self._measurement.negative_full_raw,
                    self._measurement.positive_full_raw,
                    self._measurement.negative_spatial_raw,
                    self._measurement.positive_spatial_raw,
                    self._measurement.negative_clean,
                    self._measurement.positive_clean,
                )
            )
            self._phase = "measured"
            return self._clone_measurement(self._measurement)
        except Exception as error:
            self._fail(error)
            raise

    @staticmethod
    def _fresh_zero_momentum(factory: Callable[[float], Any]) -> Any:
        momentum = factory(APG_MOMENTUM)
        numeric = getattr(momentum, "momentum", None)
        running = getattr(momentum, "running_average", None)
        if (
            isinstance(numeric, bool)
            or not isinstance(numeric, (int, float))
            or float(numeric) != 0.0
            or not isinstance(running, (int, float))
            or isinstance(running, bool)
            or float(running) != 0.0
        ):
            raise GraftPhaseANativeTrainingClosureError(
                "APG requires a newly constructed momentum=0 history"
            )
        return momentum

    @staticmethod
    def _clone_measurement(value: NativeFieldMeasurement) -> NativeFieldMeasurement:
        return NativeFieldMeasurement(
            negative_full_raw=value.negative_full_raw.detach().clone(),
            positive_full_raw=value.positive_full_raw.detach().clone(),
            negative_spatial_raw=value.negative_spatial_raw.detach().clone(),
            positive_spatial_raw=value.positive_spatial_raw.detach().clone(),
            negative_clean=value.negative_clean.detach().clone(),
            positive_clean=value.positive_clean.detach().clone(),
        )

    def _assert_measurement_immutable(self) -> None:
        if self._measurement is None or not hasattr(self, "_measurement_digests"):
            raise GraftPhaseANativeTrainingClosureError(
                "native field measurement seal is absent"
            )
        values = (
            self._measurement.negative_full_raw,
            self._measurement.positive_full_raw,
            self._measurement.negative_spatial_raw,
            self._measurement.positive_spatial_raw,
            self._measurement.negative_clean,
            self._measurement.positive_clean,
        )
        if not hasattr(self, "_measurement_snapshots") or any(
            not torch.equal(value, snapshot)
            for value, snapshot in zip(values, self._measurement_snapshots)
        ):
            raise GraftPhaseANativeTrainingClosureError(
                "native field measurement changed after capture"
            )

    def derive_phase_a_flow_matching_vjp(self) -> NativePhaseAFlowMatchingVJP:
        """Close the fixed same-source/no-op FM objective through vendor APG."""

        try:
            if self._phase != "measured" or self._measurement is None:
                raise GraftPhaseANativeTrainingClosureError(
                    "Phase-A FM-VJP requires one completed measurement pair"
                )
            self._assert_immutable()
            self._assert_measurement_immutable()
            negative_leaf = (
                self._measurement.negative_clean.detach().clone().requires_grad_(True)
            )
            positive_leaf = (
                self._measurement.positive_clean.detach().clone().requires_grad_(True)
            )
            if not (
                negative_leaf.is_leaf
                and positive_leaf.is_leaf
                and negative_leaf.dtype == torch.float32
                and positive_leaf.dtype == torch.float32
            ):
                raise GraftPhaseANativeTrainingClosureError(
                    "APG clean predictions are not fresh FP32 leaves"
                )
            momentum = self._fresh_zero_momentum(
                self.bindings.momentum_buffer_factory
            )
            with torch.enable_grad():
                guided = self.bindings.vendor_normalized_guidance(
                    pred_cond=positive_leaf,
                    pred_uncond=negative_leaf,
                    guidance_scale=GUIDANCE_SCALE,
                    momentum_buffer=momentum,
                    eta=APG_ETA,
                    norm_threshold=APG_NORM_THRESHOLD,
                )
                guided_tensor = _finite_tensor(guided, label="vendor guided clean")
                if (
                    guided_tensor.dtype != torch.float32
                    or guided_tensor.shape != self.noisy_target.shape
                    or not guided_tensor.requires_grad
                    or guided_tensor.grad_fn is None
                ):
                    raise GraftPhaseANativeTrainingClosureError(
                        "vendor normalized_guidance detached or changed FP32 geometry"
                    )
                predicted_velocity = (
                    self.noisy_target - guided_tensor
                ) / self.sigma
                same_source_target_velocity = (
                    self.noisy_target - self.source_video
                ) / self.sigma
                if (
                    predicted_velocity.dtype != torch.float32
                    or same_source_target_velocity.dtype != torch.float32
                    or predicted_velocity.shape != self.noisy_target.shape
                    or same_source_target_velocity.shape != self.noisy_target.shape
                    or not predicted_velocity.requires_grad
                    or same_source_target_velocity.requires_grad
                ):
                    raise GraftPhaseANativeTrainingClosureError(
                        "same-source flow-matching velocity geometry differs"
                    )
                flow_matching_loss = torch.mean(
                    (predicted_velocity - same_source_target_velocity) ** 2
                )
                if (
                    type(flow_matching_loss) is not torch.Tensor
                    or flow_matching_loss.dtype != torch.float32
                    or flow_matching_loss.ndim != 0
                    or not flow_matching_loss.requires_grad
                    or not bool(torch.isfinite(flow_matching_loss).item())
                ):
                    raise GraftPhaseANativeTrainingClosureError(
                        "fixed mean flow-matching loss differs"
                    )
                (guided_clean_cotangent,) = torch.autograd.grad(
                    flow_matching_loss,
                    (guided_tensor,),
                    retain_graph=True,
                    create_graph=False,
                )
                clean_cotangents = torch.autograd.grad(
                    guided_tensor,
                    (negative_leaf, positive_leaf),
                    grad_outputs=guided_clean_cotangent,
                    retain_graph=False,
                    create_graph=False,
                )
            if (
                guided_clean_cotangent.dtype != torch.float32
                or guided_clean_cotangent.shape != self.noisy_target.shape
                or not bool(torch.isfinite(guided_clean_cotangent).all().item())
            ):
                raise GraftPhaseANativeTrainingClosureError(
                    "internal FM loss produced an invalid guided-clean cotangent"
                )
            # Do not replace this with ``(-sigma * clean_vjp).to(bfloat16)``.
            # In the pinned source ``sigma * raw_bf16`` is evaluated before
            # subtraction from x_t.  Its backward rounding order is therefore
            # ``clean_vjp -> BF16 product output -> multiply by sigma``.  An
            # isolated leaf VJP through that exact expression is both clearer
            # and byte-faithful across supported PyTorch builds.
            raw_cotangent_rows: list[torch.Tensor] = []
            for raw_measured, clean_measured, clean_cotangent in zip(
                (
                    self._measurement.negative_spatial_raw,
                    self._measurement.positive_spatial_raw,
                ),
                (
                    self._measurement.negative_clean,
                    self._measurement.positive_clean,
                ),
                clean_cotangents,
            ):
                raw_leaf = raw_measured.detach().clone().requires_grad_(True)
                rebuilt_clean = self.noisy_target - self.sigma * raw_leaf
                if not torch.equal(rebuilt_clean.detach(), clean_measured):
                    raise GraftPhaseANativeTrainingClosureError(
                        "isolated BF16 raw-to-clean replay changed pinned forward bytes"
                    )
                (raw_cotangent,) = torch.autograd.grad(
                    rebuilt_clean,
                    (raw_leaf,),
                    grad_outputs=clean_cotangent,
                    retain_graph=False,
                    create_graph=False,
                )
                raw_cotangent_rows.append(raw_cotangent.detach().contiguous())
            raw_cotangents = tuple(raw_cotangent_rows)
            for label, clean_cotangent, raw_cotangent in zip(
                BRANCH_ORDER, clean_cotangents, raw_cotangents
            ):
                if (
                    clean_cotangent.dtype != torch.float32
                    or raw_cotangent.dtype != torch.bfloat16
                    or not bool(torch.isfinite(clean_cotangent).all().item())
                    or not bool(torch.isfinite(raw_cotangent).all().item())
                ):
                    raise GraftPhaseANativeTrainingClosureError(
                        f"{label} Phase-A FM-VJP cotangent is non-finite or changed dtype"
                    )
            self._leaf_vjp = NativePhaseAFlowMatchingVJP(
                guided_clean=guided_tensor.detach().contiguous(),
                predicted_velocity=predicted_velocity.detach().contiguous(),
                same_source_target_velocity=(
                    same_source_target_velocity.detach().contiguous()
                ),
                flow_matching_loss=flow_matching_loss.detach().clone(),
                guided_clean_cotangent=(
                    guided_clean_cotangent.detach().contiguous()
                ),
                negative_clean_cotangent=clean_cotangents[0].detach().contiguous(),
                positive_clean_cotangent=clean_cotangents[1].detach().contiguous(),
                negative_raw_cotangent=raw_cotangents[0],
                positive_raw_cotangent=raw_cotangents[1],
            )
            self._leaf_vjp_digests = tuple(
                _tensor_bytes_sha256(value)
                for value in (
                    self._leaf_vjp.guided_clean,
                    self._leaf_vjp.predicted_velocity,
                    self._leaf_vjp.same_source_target_velocity,
                    self._leaf_vjp.flow_matching_loss,
                    self._leaf_vjp.guided_clean_cotangent,
                    self._leaf_vjp.negative_clean_cotangent,
                    self._leaf_vjp.positive_clean_cotangent,
                    self._leaf_vjp.negative_raw_cotangent,
                    self._leaf_vjp.positive_raw_cotangent,
                )
            )
            self._leaf_vjp_snapshots = tuple(
                value.detach().clone()
                for value in (
                    self._leaf_vjp.guided_clean,
                    self._leaf_vjp.predicted_velocity,
                    self._leaf_vjp.same_source_target_velocity,
                    self._leaf_vjp.flow_matching_loss,
                    self._leaf_vjp.guided_clean_cotangent,
                    self._leaf_vjp.negative_clean_cotangent,
                    self._leaf_vjp.positive_clean_cotangent,
                    self._leaf_vjp.negative_raw_cotangent,
                    self._leaf_vjp.positive_raw_cotangent,
                )
            )
            self._assert_immutable()
            self._phase = "vjp_ready"
            return NativePhaseAFlowMatchingVJP(
                guided_clean=self._leaf_vjp.guided_clean.detach().clone(),
                predicted_velocity=(
                    self._leaf_vjp.predicted_velocity.detach().clone()
                ),
                same_source_target_velocity=(
                    self._leaf_vjp.same_source_target_velocity.detach().clone()
                ),
                flow_matching_loss=(
                    self._leaf_vjp.flow_matching_loss.detach().clone()
                ),
                guided_clean_cotangent=(
                    self._leaf_vjp.guided_clean_cotangent.detach().clone()
                ),
                negative_clean_cotangent=(
                    self._leaf_vjp.negative_clean_cotangent.detach().clone()
                ),
                positive_clean_cotangent=(
                    self._leaf_vjp.positive_clean_cotangent.detach().clone()
                ),
                negative_raw_cotangent=(
                    self._leaf_vjp.negative_raw_cotangent.detach().clone()
                ),
                positive_raw_cotangent=(
                    self._leaf_vjp.positive_raw_cotangent.detach().clone()
                ),
            )
        except Exception as error:
            self._fail(error)
            raise

    def _gradient_snapshot(self) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for name, parameter in self.bindings.named_trainable_parameters:
            if parameter.grad is None:
                result[name] = torch.zeros_like(parameter.detach(), dtype=torch.float64)
            else:
                if not bool(torch.isfinite(parameter.grad).all().item()):
                    raise GraftPhaseANativeTrainingClosureError(
                        f"trainable-registry gradient {name} is non-finite"
                    )
                result[name] = parameter.grad.detach().double().clone()
        return result

    def _gradient_category(self, name: str) -> str:
        if name.startswith("atlas_encoder."):
            return "atlas_encoder"
        if name.endswith(".identity_rebinder.output.weight"):
            return "output_projection"
        if any(
            name.endswith(f".identity_rebinder.{projection}.weight")
            for projection in ("query", "key", "value")
        ):
            return "query_key_value"
        if self.bindings.test_only:
            return "output_projection"
        raise GraftPhaseANativeTrainingClosureError(
            f"official trainable gradient category is unknown: {name}"
        )

    def _local_trainable_delta_receipt(
        self,
        *,
        role: str,
        before: Mapping[str, torch.Tensor],
        after: Mapping[str, torch.Tensor],
        before_presence: Mapping[str, bool],
        observation: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        category_squared = {
            "atlas_encoder": 0.0,
            "query_key_value": 0.0,
            "output_projection": 0.0,
        }
        category_counts = {name: 0 for name in category_squared}
        rows: list[Mapping[str, Any]] = []
        for name, parameter in self.bindings.named_trainable_parameters:
            delta = after[name] - before[name]
            if not bool(torch.isfinite(delta).all().item()):
                raise GraftPhaseANativeTrainingClosureError(
                    f"{role} local trainable gradient delta is non-finite: {name}"
                )
            norm = float(delta.norm().item())
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
                    "delta_nonzero": norm > 0.0,
                }
            )
        category_l2 = {
            name: math.sqrt(value) for name, value in category_squared.items()
        }
        local_target_rows = int(observation["local_target_rows"])
        adapter_graph_bearing = bool(observation["adapter_graph_bearing"])
        if not self._cell_active:
            if adapter_graph_bearing or any(
                value != 0.0 for value in category_l2.values()
            ):
                raise GraftPhaseANativeTrainingClosureError(
                    f"{role} inactive cell produced an adapter gradient"
                )
            gate = "inactive_cell_adapter_absent_or_zero"
        elif local_target_rows == 0:
            if adapter_graph_bearing or any(value != 0.0 for value in category_l2.values()):
                raise GraftPhaseANativeTrainingClosureError(
                    f"{role} zero-target rank produced an adapter gradient"
                )
            if any(not row["gradient_after_absent_or_exact_zero"] for row in rows):
                raise GraftPhaseANativeTrainingClosureError(
                    f"{role} zero-target rank retained a nonzero adapter gradient"
                )
            gate = "zero_target_rows_adapter_absent_or_zero"
        else:
            if (
                not adapter_graph_bearing
                or category_counts["output_projection"] <= 0
                or category_l2["output_projection"] <= 0.0
                or category_l2["query_key_value"] != 0.0
                or category_l2["atlas_encoder"] != 0.0
            ):
                raise GraftPhaseANativeTrainingClosureError(
                    f"{role} target-owning rank failed the zero-init output-only gradient gate"
                )
            gate = "target_rows_output_projection_only_nonzero"
        return _seal(
            {
                "schema_version": (
                    "bernini-graft-phase-a-local-trainable-gradient-delta-v1"
                ),
                "role": role,
                "local_target_rows": local_target_rows,
                "adapter_graph_bearing": adapter_graph_bearing,
                "category_parameter_counts": category_counts,
                "category_delta_l2_float64_hex": {
                    name: value.hex() for name, value in category_l2.items()
                },
                "rows": rows,
                "gate": gate,
            }
        )

    def _capture_and_clear_pack_gradient(
        self, *, role: str
    ) -> Mapping[str, Any]:
        gradient = self._visual_pack.grad
        if (
            not isinstance(gradient, torch.Tensor)
            or gradient.shape != self._visual_pack.shape
            or gradient.device != self._visual_pack.device
            or not bool(torch.isfinite(gradient).all().item())
        ):
            raise GraftPhaseANativeTrainingClosureError(
                f"{role} replay visual-pack leaf gradient is absent/non-finite"
            )
        norm = float(gradient.detach().float().norm().item())
        if not math.isfinite(norm) or norm <= 0.0:
            raise GraftPhaseANativeTrainingClosureError(
                f"{role} replay visual-pack leaf gradient is zero/non-finite"
            )
        receipt = _seal(
            {
                "schema_version": (
                    "bernini-graft-phase-a-replay-pack-leaf-gradient-v1"
                ),
                "role": role,
                "pack_is_leaf": self._visual_pack.is_leaf,
                "pack_requires_grad_during_replay": (
                    self._visual_pack.requires_grad
                ),
                "gradient_dtype": str(gradient.dtype),
                "gradient_shape": list(gradient.shape),
                "gradient_sha256": _tensor_bytes_sha256(gradient),
                "gradient_l2_float64_hex": norm.hex(),
                "gradient_finite_nonzero": True,
                "cleared_after_branch": True,
            }
        )
        self._visual_pack.grad = None
        if self._visual_pack.grad is not None:
            raise GraftPhaseANativeTrainingClosureError(
                f"{role} replay visual-pack leaf gradient did not clear"
            )
        return receipt

    def _assert_leaf_vjp_immutable(self) -> None:
        if self._leaf_vjp is None or not hasattr(self, "_leaf_vjp_digests"):
            raise GraftPhaseANativeTrainingClosureError("Phase-A FM-VJP seal is absent")
        values = (
            self._leaf_vjp.guided_clean,
            self._leaf_vjp.predicted_velocity,
            self._leaf_vjp.same_source_target_velocity,
            self._leaf_vjp.flow_matching_loss,
            self._leaf_vjp.guided_clean_cotangent,
            self._leaf_vjp.negative_clean_cotangent,
            self._leaf_vjp.positive_clean_cotangent,
            self._leaf_vjp.negative_raw_cotangent,
            self._leaf_vjp.positive_raw_cotangent,
        )
        if not hasattr(self, "_leaf_vjp_snapshots") or any(
            not torch.equal(value, snapshot)
            for value, snapshot in zip(values, self._leaf_vjp_snapshots)
        ):
            raise GraftPhaseANativeTrainingClosureError(
                "Phase-A FM-VJP changed before serial replay"
            )

    def replay_and_backward(self) -> NativeTrainingClosureResult:
        """Serially replay negative/positive and apply their raw cotangents."""

        try:
            if (
                self._phase != "vjp_ready"
                or self._measurement is None
                or self._leaf_vjp is None
            ):
                raise GraftPhaseANativeTrainingClosureError(
                    "serial replay requires one completed Phase-A FM-VJP"
                )
            self._assert_measurement_immutable()
            self._assert_leaf_vjp_immutable()
            if any(
                parameter.grad is not None
                for _, parameter in self.bindings.named_trainable_parameters
            ):
                raise GraftPhaseANativeTrainingClosureError(
                    "trainable-registry gradients appeared before serial replay"
                )
            if self._visual_pack.grad is not None or self._visual_pack.requires_grad:
                raise GraftPhaseANativeTrainingClosureError(
                    "visual pack was not a fresh detached leaf before replay"
                )
            if self._cell_active:
                self._visual_pack.requires_grad_(True)
                if not self._visual_pack.is_leaf or self._visual_pack.grad_fn is not None:
                    raise GraftPhaseANativeTrainingClosureError(
                        "active replay visual pack is not a detached native leaf"
                    )
            measured = (
                self._measurement.negative_full_raw,
                self._measurement.positive_full_raw,
            )
            cotangents = (
                self._leaf_vjp.negative_raw_cotangent,
                self._leaf_vjp.positive_raw_cotangent,
            )
            replay_hashes: list[str] = []
            gradient_delta_l2: dict[str, float] = {}
            local_gradient_delta_receipts: list[Mapping[str, Any]] = []
            for role, expected, cotangent in zip(BRANCH_ORDER, measured, cotangents):
                before = self._gradient_snapshot()
                before_presence = {
                    name: parameter.grad is not None
                    for name, parameter in self.bindings.named_trainable_parameters
                }
                if self._visual_pack.grad is not None:
                    raise GraftPhaseANativeTrainingClosureError(
                        f"{role} replay visual-pack gradient was not clear"
                    )
                replay = self._shared_forward(
                    phase="replay", role=role, graph_enabled=True
                )
                observation = self._forward_observation_receipts[-1]
                if not torch.equal(replay.detach(), expected):
                    raise GraftPhaseANativeTrainingClosureError(
                        f"{role} graph replay raw BF16 bytes differ from measurement"
                    )
                replay_hashes.append(_tensor_bytes_sha256(replay))
                spatial = unpack_wan_target_velocity(
                    replay[:, -self._target_tokens :, :],
                    spatial_shape=self.noisy_target.shape,
                )
                if self._cell_active:
                    if not spatial.requires_grad or spatial.grad_fn is None:
                        raise GraftPhaseANativeTrainingClosureError(
                            f"{role} active-cell Wan unpack lost the trainable-registry graph"
                        )
                    torch.autograd.backward(
                        spatial, grad_tensors=cotangent.to(dtype=spatial.dtype)
                    )
                    self._replay_pack_gradient_receipts.append(
                        self._capture_and_clear_pack_gradient(role=role)
                    )
                elif spatial.requires_grad or spatial.grad_fn is not None:
                    raise GraftPhaseANativeTrainingClosureError(
                        f"{role} inactive-cell route was not an exact zero-update"
                    )
                after = self._gradient_snapshot()
                squared = 0.0
                for name in before:
                    delta = after[name] - before[name]
                    squared += float((delta * delta).sum().item())
                norm = math.sqrt(squared)
                if not math.isfinite(norm):
                    raise GraftPhaseANativeTrainingClosureError(
                        f"{role} serial replay gradient delta is non-finite"
                    )
                gradient_delta_l2[role] = norm
                local_gradient_delta_receipts.append(
                    self._local_trainable_delta_receipt(
                        role=role,
                        before=before,
                        after=after,
                        before_presence=before_presence,
                        observation=observation,
                    )
                )
                del spatial, replay
            if self._cell_active:
                if self._visual_pack.grad is not None:
                    raise GraftPhaseANativeTrainingClosureError(
                        "replay visual-pack gradient remained after serial branches"
                    )
                self._visual_pack.requires_grad_(False)
            self._assert_immutable()
            if tuple(self._call_trace) != (
                ("measurement", "negative"),
                ("measurement", "positive"),
                ("replay", "negative"),
                ("replay", "positive"),
            ):
                raise GraftPhaseANativeTrainingClosureError(
                    "completed closure call trace differs"
                )
            final_gradients = self._gradient_snapshot()
            final_gradient_nonzero = any(
                bool(torch.count_nonzero(value).item())
                for value in final_gradients.values()
            )
            replay_observations = self._forward_observation_receipts[2:]
            if len(replay_observations) != 2 or any(
                row["local_target_rows"] != replay_observations[0]["local_target_rows"]
                for row in replay_observations
            ):
                raise GraftPhaseANativeTrainingClosureError(
                    "serial replay local target ownership changed"
                )
            local_target_rows = int(replay_observations[0]["local_target_rows"])
            if self._cell_active and (
                (local_target_rows > 0) is not final_gradient_nonzero
            ):
                raise GraftPhaseANativeTrainingClosureError(
                    "active local trainable gradient differs from target ownership"
                )
            if not self._cell_active and (
                final_gradient_nonzero
                or any(
                    parameter.grad is not None
                    for _, parameter in self.bindings.named_trainable_parameters
                )
            ):
                raise GraftPhaseANativeTrainingClosureError(
                    "inactive schedule cell produced a trainable-registry gradient"
                )
            device_type = self.noisy_target.device.type
            hash_pinned_code_on_cuda = (
                self.bindings.official_pinned_code and device_type == "cuda"
            )
            objective_api = _objective_api_structure_receipt()
            payload = {
                "schema_version": SCHEMA_VERSION,
                "binding_receipt_digest": self.bindings.receipt()["digest"],
                "official_pinned_code": self.bindings.official_pinned_code,
                "test_only_binding": self.bindings.test_only,
                "pinned_bernini_commit": PINNED_BERNINI_COMMIT,
                "pinned_wan_diffusion_sha256": PINNED_WAN_DIFFUSION_SHA256,
                "pinned_transformer_wan_sha256": PINNED_TRANSFORMER_WAN_SHA256,
                "guidance_mode": GUIDANCE_MODE,
                "frame_count": EXPECTED_FRAMES,
                "latent_shape": list(self.noisy_target.shape),
                "latent_phases": EXPECTED_LATENT_PHASES,
                "text_condition_shape": list(self.positive_condition.shape),
                "official_text_condition_geometry_exercised": tuple(
                    self.positive_condition.shape
                )
                == (1, EXPECTED_TEXT_TOKENS, EXPECTED_TEXT_DIM),
                "schedule_steps": EXPECTED_STEPS,
                "schedule_index": self.schedule_index,
                "phase_a_active_schedule_indices": list(
                    self.bindings.active_schedule_indices
                ),
                "schedule_cell_active_for_training": self._cell_active,
                "schedule_cell_counted_as_trained": self._cell_active,
                "inactive_schedule_policy": "exact_zero_update_not_trained",
                "exact40_coverage_verified_by_this_closure": False,
                "exact40_coverage_required_from_gpu_runner": True,
                "timestep": int(self.timestep.item()),
                "sigma_float32_be_hex": struct.pack(">f", float(self.sigma.item())).hex(),
                "exact40_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
                "patch_source_ids": list(EXPECTED_PATCH_SOURCE_IDS),
                "pack_layout": "source_id_1_prefix_then_noisy_target_id_0_suffix",
                "source_tokens": self._source_tokens,
                "target_tokens": self._target_tokens,
                "shared_visual_pack_sha256": self._visual_pack_digest,
                "shared_rotary_pack_sha256": self._rotary_pack_digest,
                "same_pack_timestep_rotary_objects_all_four_forwards": True,
                "measurement_visual_pack_requires_grad": False,
                "replay_visual_pack_detached_leaf": self._cell_active,
                "replay_visual_pack_requires_grad_only_during_replay": (
                    self._cell_active
                ),
                "replay_pack_gradient_cleared_after_each_branch": (
                    len(self._replay_pack_gradient_receipts) == 2
                    if self._cell_active
                    else True
                ),
                "forward_route_receipt_digest": (
                    self.bindings.forward_route_receipt_digest
                ),
                "forward_route_context_opened_per_forward": True,
                "branch_order": list(BRANCH_ORDER),
                "call_trace": [list(item) for item in self._call_trace],
                "forward_context_observations": [
                    dict(row) for row in self._forward_observation_receipts
                ],
                "measurement_grad_enabled": False,
                "replay_autograd_context_enabled": True,
                "replay_backward_applied": self._cell_active,
                "raw_output_dtype": "torch.bfloat16",
                "target_selection": "last_target_tokens_without_mask",
                "wan_unpack_order": "b_(t_h_w)_(pt_ph_pw_c)_to_b_c_(t_pt)_(h_ph)_(w_pw)",
                "wan_patch_size": list(EXPECTED_PATCH_SIZE),
                "official_unpack_source_audited": True,
                "clean_formula": "x_t_fp32 - sigma_fp32 * raw_velocity_bf16",
                "apg_guidance_scale": GUIDANCE_SCALE,
                "apg_eta": APG_ETA,
                "apg_norm_threshold": APG_NORM_THRESHOLD,
                "apg_momentum": APG_MOMENTUM,
                "fresh_apg_momentum_instance": True,
                "apg_input_kind": "fresh_detached_fp32_clean_leaves",
                "phase_a_objective": FLOW_MATCHING_OBJECTIVE,
                "supervision_pair": "same_source_video_noop",
                "predicted_velocity_formula": "(x_t_fp32-guided_clean_fp32)/sigma_fp32",
                "same_source_target_velocity_formula": "(x_t_fp32-source_video_fp32)/sigma_fp32",
                "flow_matching_loss_formula": "mean((v_pred-v_target)**2)",
                "flow_matching_loss_reduction": FLOW_MATCHING_REDUCTION,
                "flow_matching_loss_float32_be_hex": struct.pack(
                    ">f", float(self._leaf_vjp.flow_matching_loss.item())
                ).hex(),
                "raw_cotangent_formula": (
                    "internal_mean_fm_loss_to_vendor_apg_clean_leaves_then_"
                    "autograd_vjp_of_x_t_fp32_minus_sigma_fp32_times_raw_bf16"
                ),
                "raw_cotangent_dtype": "torch.bfloat16",
                "input_tensor_sha256": dict(self._input_digests),
                "negative_measurement_raw_sha256": _tensor_bytes_sha256(measured[0]),
                "positive_measurement_raw_sha256": _tensor_bytes_sha256(measured[1]),
                "negative_measurement_clean_sha256": _tensor_bytes_sha256(
                    self._measurement.negative_clean
                ),
                "positive_measurement_clean_sha256": _tensor_bytes_sha256(
                    self._measurement.positive_clean
                ),
                "guided_clean_sha256": _tensor_bytes_sha256(
                    self._leaf_vjp.guided_clean
                ),
                "predicted_velocity_sha256": _tensor_bytes_sha256(
                    self._leaf_vjp.predicted_velocity
                ),
                "same_source_target_velocity_sha256": _tensor_bytes_sha256(
                    self._leaf_vjp.same_source_target_velocity
                ),
                "flow_matching_loss_sha256": _tensor_bytes_sha256(
                    self._leaf_vjp.flow_matching_loss
                ),
                "negative_clean_cotangent_sha256": _tensor_bytes_sha256(
                    self._leaf_vjp.negative_clean_cotangent
                ),
                "positive_clean_cotangent_sha256": _tensor_bytes_sha256(
                    self._leaf_vjp.positive_clean_cotangent
                ),
                "guided_clean_cotangent_sha256": _tensor_bytes_sha256(
                    self._leaf_vjp.guided_clean_cotangent
                ),
                "negative_raw_cotangent_sha256": _tensor_bytes_sha256(
                    self._leaf_vjp.negative_raw_cotangent
                ),
                "positive_raw_cotangent_sha256": _tensor_bytes_sha256(
                    self._leaf_vjp.positive_raw_cotangent
                ),
                "negative_replay_raw_sha256": replay_hashes[0],
                "positive_replay_raw_sha256": replay_hashes[1],
                "per_branch_raw_replay_exact": [True, True],
                "per_branch_gradient_delta_l2": gradient_delta_l2,
                "per_branch_nonzero_gradient_delta": {
                    role: gradient_delta_l2[role] > 0.0 for role in BRANCH_ORDER
                },
                "per_branch_local_trainable_gradient_gate": (
                    [dict(row) for row in local_gradient_delta_receipts]
                ),
                "per_branch_replay_pack_leaf_gradient": [
                    dict(row) for row in self._replay_pack_gradient_receipts
                ],
                "local_sequence_parallel_rank": replay_observations[0][
                    "sequence_parallel_rank"
                ],
                "local_sequence_parallel_size": replay_observations[0][
                    "sequence_parallel_size"
                ],
                "local_shard_rows": replay_observations[0]["local_shard_rows"],
                "local_target_rows": local_target_rows,
                "local_adapter_graph_bearing": replay_observations[0][
                    "adapter_graph_bearing"
                ],
                "trainable_registry_values_unchanged": True,
                "trainable_registry_value_sha256": self._parameter_digest,
                "trainable_registry_final_gradient_nonzero": final_gradient_nonzero,
                "local_trainable_registry_final_gradient_matches_target_ownership": True,
                "exclusive_trainable_scope_is_exact_authenticated_registry": True,
                "external_trainable_owner_names": [
                    name
                    for name, _ in self.bindings.external_trainable_owner_modules
                ],
                "frozen_base_requires_grad": False,
                "optimizer_created": False,
                "parameters_updated": False,
                "scheduler_step_called": False,
                "outer_clean_state_transport_used": False,
                "external_guided_clean_cotangent_accepted": False,
                "objective_execute_api_parameters": list(
                    objective_api["execute_parameters"]
                ),
                "objective_derive_api_parameters": list(
                    objective_api["derive_parameters"]
                ),
                "forbidden_oracle_api_parameters_present": list(
                    objective_api["forbidden_oracle_parameters_present"]
                ),
                "oracle_inputs_absent_by_public_api": objective_api[
                    "oracle_inputs_absent"
                ],
                "target_video_used": False,
                "mask_used": False,
                "pose_used": False,
                "track_used": False,
                "optical_flow_used": False,
                "motion_donor_used": False,
                "checkpoint_weight_content_verified_by_this_core": False,
                "checkpoint_content_binding_required_from_gpu_runner": True,
                "hash_pinned_code_executed_on_cuda": hash_pinned_code_on_cuda,
                "official_cuda_closure_verified_by_this_core": False,
                "forward_route_semantics_verified_by_this_core": False,
                "packed_raw_to_apg_registry_chain_verified_by_this_core": False,
                "checkpoint_route_chain_claim_requires_gpu_runner": True,
                "python_same_process_security_boundary": False,
                "sp4_collective_parity_verified": False,
                "full_sampler_trajectory_verified": False,
                "training_quality_claim_authorized": False,
                "scientific_action_editing_claim_authorized": False,
            }
            self._receipt = _seal(payload)
            self._phase = "closed"
            return NativeTrainingClosureResult(
                guided_clean=self._leaf_vjp.guided_clean.detach().clone(),
                flow_matching_loss=(
                    self._leaf_vjp.flow_matching_loss.detach().clone()
                ),
                receipt=self._receipt,
            )
        except Exception as error:
            self._fail(error)
            raise

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def call_trace(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._call_trace)

    def forward_context_observation_receipt(self) -> Mapping[str, Any]:
        """Expose sealed measurement locality before any replay/backward."""

        if self._phase not in ("measured", "vjp_ready"):
            raise GraftPhaseANativeTrainingClosureError(
                "pre-backward context observation requires completed measurement"
            )
        if (
            tuple(self._call_trace)
            != (("measurement", "negative"), ("measurement", "positive"))
            or len(self._forward_observation_receipts) != 2
            or any(
                row["phase"] != "measurement"
                or row["adapter_graph_bearing"] is not False
                for row in self._forward_observation_receipts
            )
        ):
            raise GraftPhaseANativeTrainingClosureError(
                "pre-backward measurement-context inventory differs"
            )
        first = self._forward_observation_receipts[0]
        if any(
            row[key] != first[key]
            for row in self._forward_observation_receipts[1:]
            for key in (
                "sequence_parallel_rank",
                "sequence_parallel_size",
                "global_total_tokens",
                "global_condition_tokens",
                "global_target_tokens",
                "local_shard_start",
                "local_shard_stop_exclusive",
                "local_shard_rows",
                "local_valid_rows",
                "local_padding_rows",
                "local_target_rows",
                "local_target_selector_sha256",
                "route_gate_float64_hex",
            )
        ):
            raise GraftPhaseANativeTrainingClosureError(
                "negative/positive measurement locality differs"
            )
        return _seal(
            {
                "schema_version": (
                    "bernini-graft-phase-a-pre-backward-context-v1"
                ),
                "measurement_complete": True,
                "backward_started": False,
                "measurement_grad_enabled": False,
                "adapter_graph_bearing": False,
                "sequence_parallel_rank": first["sequence_parallel_rank"],
                "sequence_parallel_size": first["sequence_parallel_size"],
                "global_total_tokens": first["global_total_tokens"],
                "global_condition_tokens": first["global_condition_tokens"],
                "global_target_tokens": first["global_target_tokens"],
                "local_shard_start": first["local_shard_start"],
                "local_shard_stop_exclusive": first[
                    "local_shard_stop_exclusive"
                ],
                "local_shard_rows": first["local_shard_rows"],
                "local_valid_rows": first["local_valid_rows"],
                "local_padding_rows": first["local_padding_rows"],
                "local_target_rows": first["local_target_rows"],
                "local_target_selector_sha256": first[
                    "local_target_selector_sha256"
                ],
                "route_gate_float64_hex": first["route_gate_float64_hex"],
                "measurement_observations": [
                    dict(row) for row in self._forward_observation_receipts
                ],
            }
        )

    def receipt(self) -> Mapping[str, Any]:
        try:
            if self._phase != "closed" or self._receipt is None:
                raise GraftPhaseANativeTrainingClosureError(
                    "unfinished native closure has no receipt"
                )
            return self._receipt
        except Exception as error:
            self._fail(error)
            raise


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
    """Convenience entry point that executes the single-use closure completely."""

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
    "GraftPhaseANativeTrainingClosureError",
    "GUIDANCE_MODE",
    "GUIDANCE_SCALE",
    "FLOW_MATCHING_OBJECTIVE",
    "FLOW_MATCHING_REDUCTION",
    "FORWARD_ROUTE_SCHEMA_VERSION",
    "NativePhaseAFlowMatchingVJP",
    "NativeFieldMeasurement",
    "NativeForwardContextObservation",
    "NativeForwardContextRequest",
    "NativeTrainingClosureResult",
    "PINNED_BERNINI_COMMIT",
    "PINNED_TRANSFORMER_WAN_SHA256",
    "PINNED_WAN_DIFFUSION_SHA256",
    "PHASE_A_ACTIVE_SCHEDULE_INDICES",
    "PhaseANativeTrainingClosure",
    "SCHEMA_VERSION",
    "authenticate_cpu_test_fakes",
    "authenticate_pinned_native_bindings",
    "build_native_forward_context_observation",
    "execute_phase_a_native_training_closure",
    "unpack_wan_target_velocity",
]
