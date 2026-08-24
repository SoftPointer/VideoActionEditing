#!/usr/bin/env python3
"""Frozen Bernini APG field adapter for SAIC source-state transport.

This is a narrow native-model boundary, not a sampler or a training path.  It
accepts the immutable guided-velocity requests emitted by
``saic_source_state_flow_transport_v1`` and implements exactly two registered
visual regimes:

* ``t2v_apg``: target-only query, patched with ``source_id=0``;
* ``r2v_apg_source_i0``: one independently encoded RGB frame-0 latent,
  ``[1,16,1,H,W]`` and ``source_id=1``, followed by the target query with
  ``source_id=0``.

Every guided query patches its target state once; R2V also repatches the same
sealed I0 bytes first, preserving the native physical ``[source_id=1,
source_id=0]`` call order.  The resulting packed visual tokens, rotary tensor,
and native timestep object are then reused unchanged. T2V performs the native
two-forward text APG. R2V performs the native three-forward chain: target-only
negative, I0 negative, and I0 role caption, followed by vendor
``normalized_guidance_chain`` with scales ``[4.5,4.0]``. The returned value is
a detached, owned FP32 spatial velocity. There is no optimizer, training,
evaluator, or semantic authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import inspect
import math
import re
import struct
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence

import torch

import dclr_runtime_contract as runtime_contract
import saic_source_state_flow_transport_v1 as transport


SCHEMA_VERSION = "bernini-saic-native-source-state-field-v1"
VENDOR_APG_MODULE = "bernini.models.wan_diffusion"
EXPECTED_REQUEST_SCHEMA_VERSION = (
    "bernini-saic-source-state-flow-transport-v1/guided-velocity-request-v1"
)
REGISTERED_REGIMES = ("t2v_apg", "r2v_apg_source_i0")
EXPECTED_STEPS = 40
REGISTERED_K1_SCHEDULE = (1,) * EXPECTED_STEPS
REGISTERED_K5_EARLY_SCHEDULE = (5, 5, 5) + (1,) * (EXPECTED_STEPS - 3)
REGISTERED_CANDIDATE_SCHEDULES = (
    REGISTERED_K1_SCHEDULE,
    REGISTERED_K5_EARLY_SCHEDULE,
)
TARGET_SOURCE_ID = 0
REFERENCE_SOURCE_ID = 1
GUIDANCE_SCALE = 4.0
IMAGE_GUIDANCE_SCALE = 4.5
APG_ETA = 0.5
APG_NORM_THRESHOLD = 50.0
APG_MOMENTUM = 0.0
PATCH_CHANNELS = 64
TEXT_TOKENS = 512
TEXT_DIM = 4096
LATENT_SHAPE_PREFIX = (1, 16, 21)
REFERENCE_SHAPE_PREFIX = (1, 16, 1)
ROLE_ORDER = ("target", "source")
CONDITION_KEYS = ("negative", "target", "source")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_VELOCITY_QUERY_REQUEST_TYPE = transport.VelocityQueryRequest
_FLOW_TRANSPORT_STEP_BINDING_TYPE = transport.FlowTransportStepBinding
_NATIVE_GUIDANCE_BINDING_TYPE = transport.NativeGuidanceBinding


class SAICNativeSourceStateFieldError(RuntimeError):
    """The frozen native field closure or call protocol was violated."""


@dataclass(frozen=True)
class NativeFieldProvenance:
    """Externally sealed identities that cannot be inferred from tensors."""

    model_id: str
    checkpoint_sha256: str
    model_receipt_sha256: str
    guidance_contract_sha256: str
    negative_prompt_sha256: str
    native_schedule_sha256: str
    noise_generator_id: str
    master_seed: int
    noise_bank_sha256: str
    reference_encoder_sha256: str
    reference_frame0_latent_sha256: str
    prompt_utf8_sha256_by_role: Mapping[str, str]
    prompt_condition_sha256_by_key: Mapping[str, str]

    def validate(self, *, regime: str) -> "NativeFieldProvenance":
        if _text(self.model_id, label="model_id") != "transformer_1":
            raise SAICNativeSourceStateFieldError(
                "model_id must equal the pinned Bernini single expert transformer_1"
            )
        for label, value in (
            ("checkpoint_sha256", self.checkpoint_sha256),
            ("model_receipt_sha256", self.model_receipt_sha256),
            ("guidance_contract_sha256", self.guidance_contract_sha256),
            ("negative_prompt_sha256", self.negative_prompt_sha256),
            ("native_schedule_sha256", self.native_schedule_sha256),
            ("noise_bank_sha256", self.noise_bank_sha256),
            ("reference_encoder_sha256", self.reference_encoder_sha256),
            (
                "reference_frame0_latent_sha256",
                self.reference_frame0_latent_sha256,
            ),
        ):
            _sha256(value, label=label)
        _text(self.noise_generator_id, label="noise_generator_id")
        if type(self.master_seed) is not int or self.master_seed < 0:
            raise SAICNativeSourceStateFieldError(
                "master_seed must be a nonnegative integer"
            )
        if set(self.prompt_utf8_sha256_by_role) != {"target", "source"}:
            raise SAICNativeSourceStateFieldError(
                "prompt UTF-8 registry must contain exactly target and source"
            )
        if set(self.prompt_condition_sha256_by_key) != {
            "negative",
            "target",
            "source",
        }:
            raise SAICNativeSourceStateFieldError(
                "condition registry must contain exactly negative, target, and source"
            )
        for key, value in self.prompt_utf8_sha256_by_role.items():
            _sha256(value, label=f"prompt_utf8_sha256_by_role[{key}]")
        for key, value in self.prompt_condition_sha256_by_key.items():
            _sha256(value, label=f"prompt_condition_sha256_by_key[{key}]")
        if len(set(self.prompt_condition_sha256_by_key.values())) != 3:
            raise SAICNativeSourceStateFieldError(
                "negative/source/target condition tensors must have distinct digests"
            )
        if regime not in ("t2v_apg", "r2v_apg_source_i0"):
            raise SAICNativeSourceStateFieldError("unregistered native field regime")
        if regime == "t2v_apg" and (
            self.reference_encoder_sha256 != "0" * 64
            or self.reference_frame0_latent_sha256 != "0" * 64
        ):
            raise SAICNativeSourceStateFieldError(
                "target-only T2V must bind all-zero no-reference digests"
            )
        if regime == "r2v_apg_source_i0" and (
            self.reference_encoder_sha256 == "0" * 64
            or self.reference_frame0_latent_sha256 == "0" * 64
        ):
            raise SAICNativeSourceStateFieldError(
                "source-I0 R2V requires nonzero independent-reference digests"
            )
        return self


@dataclass(frozen=True)
class NativeFieldDiagnostics:
    """Tensor-free execution facts; never a quality or update receipt.

    ``raw_transformer_forward_count`` and both legacy patch counts are physical
    attempts for backward-compatible runner reporting.  The explicit
    ``*_attempt_count``/``*_success_count`` fields are authoritative on failed
    paths; a finalized rollout requires equality at every boundary.
    ``guided_query_count`` is the legacy successful-query count.
    """

    field_regime: str
    guided_query_count: int
    raw_transformer_forward_count: int
    patch_query_count: int
    patch_reference_count: int
    guided_query_attempt_count: int
    guided_query_success_count: int
    raw_transformer_forward_attempt_count: int
    raw_transformer_forward_success_count: int
    patch_query_attempt_count: int
    patch_query_success_count: int
    patch_reference_attempt_count: int
    patch_reference_success_count: int
    vendor_single_attempt_count: int
    vendor_single_success_count: int
    vendor_chain_attempt_count: int
    vendor_chain_success_count: int
    expected_guided_query_count: int
    expected_raw_transformer_forward_count: int
    next_step_index: int
    next_candidate_index: int
    next_role: str
    initial_full_model_content_audit: bool
    final_full_model_content_audit: bool
    rollout_complete: bool
    initial_model_content_seal_sha256_by_module: tuple[tuple[str, str], ...]
    final_model_content_seal_sha256_by_module: tuple[tuple[str, str], ...]
    model_receipt_sha256: str
    native_schedule_sha256: str
    reference_encoder_sha256: str
    provenance_seal_sha256: str
    adapter_failed: bool
    failure_stage: Optional[str]
    raw_transformer_forward_count_verified: bool
    native_request_execution_verified: bool
    vendor_apg_execution_verified: bool
    model_checkpoint_use_verified: bool = False
    target_tail_direct_view: bool = True
    optimizer_step_allowed: bool = False
    training_update_allowed: bool = False
    semantic_action_success: bool = False


_NATIVE_FIELD_PROVENANCE_TYPE = NativeFieldProvenance
_NATIVE_FIELD_DIAGNOSTICS_TYPE = NativeFieldDiagnostics
_PROVENANCE_VALIDATE_FUNCTION = NativeFieldProvenance.validate


def _text(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise SAICNativeSourceStateFieldError(
            f"{label} must be a nonempty stripped string"
        )
    return value


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SAICNativeSourceStateFieldError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _tensor_bytes_sha256(value: torch.Tensor) -> str:
    # ``view(dtype)`` rejects a zero-dimensional tensor, so establish one
    # explicit element axis before reinterpreting bytes.
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).to("cpu")
    try:
        payload = raw.numpy().tobytes()
    except RuntimeError:  # pragma: no cover - numpy-less AUH fallback
        payload = bytes(raw.tolist())
    return hashlib.sha256(payload).hexdigest()


def _tensor_snapshot(value: torch.Tensor) -> tuple[Any, ...]:
    return (
        id(value),
        _tensor_version(value),
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.stride()),
        int(value.storage_offset()),
        value.dtype,
        value.device,
        value.layout,
        _storage_ptr(value),
        _tensor_bytes_sha256(value),
    )


def _tensor_version(value: torch.Tensor) -> Optional[int]:
    # Tensors created inside ``torch.inference_mode`` intentionally expose no
    # version counter.  Their byte digest remains part of the snapshot.
    try:
        return int(value._version)
    except RuntimeError:
        return None


def _storage_ptr(value: torch.Tensor) -> int:
    return int(value.untyped_storage().data_ptr())


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.device == right.device and _storage_ptr(left) == _storage_ptr(right)


def _finite_detached(value: Any, *, label: str) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.layout != torch.strided
        or not value.is_floating_point()
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise SAICNativeSourceStateFieldError(
            f"{label} must be a finite detached strided floating tensor"
        )
    return value


def _registered_module_tensors(module: Any) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    """Enumerate through torch's canonical implementation, not replaceable methods."""

    if not isinstance(module, torch.nn.Module):
        raise SAICNativeSourceStateFieldError(
            "native Bernini owners must be torch.nn.Module instances"
        )
    return (
        (
            "parameter",
            tuple(
                torch.nn.Module.named_parameters(
                    module, recurse=True, remove_duplicate=False
                )
            ),
        ),
        (
            "buffer",
            tuple(
                torch.nn.Module.named_buffers(
                    module, recurse=True, remove_duplicate=False
                )
            ),
        ),
    )


def _registered_module_topology(module: torch.nn.Module) -> tuple[tuple[Any, ...], ...]:
    rows: list[tuple[Any, ...]] = []
    for name, child in torch.nn.Module.named_modules(
        module, remove_duplicate=False
    ):
        if child.training is not False:
            raise SAICNativeSourceStateFieldError(
                f"registered submodule {name or '<root>'} must remain in eval mode"
            )
        rows.append((name, id(child), type(child)))
    return tuple(rows)


def _module_content_seal(
    module: Any,
    *,
    label: str,
    tensor_digest_cache: Optional[MutableMapping[tuple[Any, ...], str]] = None,
) -> str:
    if not isinstance(module, torch.nn.Module):
        raise SAICNativeSourceStateFieldError(
            f"{label} must be a torch.nn.Module"
        )
    topology = _registered_module_topology(module)
    digest = hashlib.sha256()
    for name, _identity, module_type in topology:
        digest.update(b"module\0" + name.encode("utf-8") + b"\0")
        digest.update(
            f"{module_type.__module__}.{module_type.__qualname__}".encode("utf-8")
            + b"\0"
        )
    seen: set[int] = set()
    cache = tensor_digest_cache if tensor_digest_cache is not None else {}
    for kind, rows in _registered_module_tensors(module):
        for name, value in rows:
            if not isinstance(value, torch.Tensor):
                raise SAICNativeSourceStateFieldError(
                    f"{label} {kind} {name} is not a tensor"
                )
            if id(value) in seen:
                raise SAICNativeSourceStateFieldError(
                    f"{label} exposes aliased registered tensors"
                )
            seen.add(id(value))
            if value.requires_grad or value.grad_fn is not None:
                raise SAICNativeSourceStateFieldError(
                    f"{label} is trainable at {kind} {name}"
                )
            digest.update(kind.encode("utf-8") + b"\0" + name.encode("utf-8") + b"\0")
            digest.update(str(value.dtype).encode("ascii") + b"\0")
            digest.update(str(tuple(int(x) for x in value.shape)).encode("ascii") + b"\0")
            tensor_key = (
                value.device,
                _storage_ptr(value),
                int(value.storage_offset()),
                tuple(int(x) for x in value.shape),
                tuple(int(x) for x in value.stride()),
                value.dtype,
            )
            value_digest = cache.get(tensor_key)
            if value_digest is None:
                value_digest = _tensor_bytes_sha256(value)
                cache[tensor_key] = value_digest
            digest.update(bytes.fromhex(value_digest))
    if getattr(module, "training", None) is not False:
        raise SAICNativeSourceStateFieldError(f"{label} must remain in eval mode")
    return digest.hexdigest()


def _module_structure_seal(module: Any, *, label: str) -> tuple[Any, ...]:
    """Cheap per-query seal: no parameter bytes leave the accelerator.

    This catches rebinding, storage changes, ordinary in-place updates,
    trainability, and mode changes.  A same-process ``parameter.data`` write
    can evade PyTorch's version counter, so :meth:`finalize` performs the
    second full content audit before a rollout may be called complete.
    """

    if getattr(module, "training", None) is not False:
        raise SAICNativeSourceStateFieldError(f"{label} must remain in eval mode")
    rows: list[Any] = [("module_topology", _registered_module_topology(module))]
    seen: set[int] = set()
    for kind, values in _registered_module_tensors(module):
        for name, value in values:
            if not isinstance(value, torch.Tensor) or id(value) in seen:
                raise SAICNativeSourceStateFieldError(
                    f"{label} registered tensor structure differs"
                )
            seen.add(id(value))
            if value.requires_grad or value.grad_fn is not None:
                raise SAICNativeSourceStateFieldError(
                    f"{label} became trainable at {kind} {name}"
                )
            rows.append(
                (
                    kind,
                    name,
                    id(value),
                    tuple(int(x) for x in value.shape),
                    value.dtype,
                    value.device,
                    value.layout,
                    _storage_ptr(value),
                    _tensor_version(value),
                )
            )
    return tuple(rows)


def _capture_bound_method(owner: Any, name: str) -> tuple[Any, Any]:
    """Capture one canonical bound method and its immutable function identity."""

    value = getattr(owner, name, None)
    function = getattr(value, "__func__", None)
    if not callable(value) or getattr(value, "__self__", None) is not owner or not callable(function):
        raise SAICNativeSourceStateFieldError(
            f"{name} must be a canonical method bound to its Bernini owner"
        )
    return value, function


def _bound_method_is_current(owner: Any, name: str, function: Any) -> bool:
    current = getattr(owner, name, None)
    return (
        callable(current)
        and getattr(current, "__self__", None) is owner
        and getattr(current, "__func__", None) is function
    )


def _resolve_vendor_apg_symbols() -> tuple[Any, Callable[..., Any], Callable[..., Any], type[Any]]:
    """Authenticate the exact Bernini single/chain APG implementation."""

    try:
        module = importlib.import_module("bernini.models.wan_diffusion")
    except Exception as error:
        raise SAICNativeSourceStateFieldError(
            "cannot import pinned bernini.models.wan_diffusion APG symbols"
        ) from error
    single = getattr(module, "normalized_guidance", None)
    chain = getattr(module, "normalized_guidance_chain", None)
    momentum_class = getattr(module, "MomentumBuffer", None)
    for value, name, expected_type in (
        (single, "normalized_guidance", "callable"),
        (chain, "normalized_guidance_chain", "callable"),
        (momentum_class, "MomentumBuffer", "type"),
    ):
        valid_kind = isinstance(value, type) if expected_type == "type" else callable(value)
        if (
            not valid_kind
            or getattr(value, "__module__", None)
            != "bernini.models.wan_diffusion"
            or getattr(value, "__name__", None) != name
            or inspect.getmodule(value) is not module
            or getattr(module, name, None) is not value
        ):
            raise SAICNativeSourceStateFieldError(
                f"vendor APG symbol identity differs at {name}"
            )
    try:
        single_parameters = tuple(inspect.signature(single).parameters)
        chain_parameters = tuple(inspect.signature(chain).parameters)
        momentum_parameters = tuple(inspect.signature(momentum_class).parameters)
    except (TypeError, ValueError) as error:
        raise SAICNativeSourceStateFieldError(
            "vendor APG signatures are not inspectable"
        ) from error
    if single_parameters != (
        "pred_cond",
        "pred_uncond",
        "guidance_scale",
        "momentum_buffer",
        "eta",
        "norm_threshold",
    ) or chain_parameters != (
        "pred_uncond",
        "preds",
        "scales",
        "momentum_buffers",
        "eta",
        "norm_thresholds",
    ) or momentum_parameters != ("momentum",):
        raise SAICNativeSourceStateFieldError("vendor APG signatures differ")
    return module, single, chain, momentum_class


def _validate_latent(value: Any, *, label: str, reference: bool) -> torch.Tensor:
    tensor = _finite_detached(value, label=label)
    prefix = (1, 16, 1) if reference else (1, 16, 21)
    if (
        tensor.dtype != torch.float32
        or tensor.ndim != 5
        or tuple(int(x) for x in tensor.shape[:3]) != prefix
        or int(tensor.shape[3]) <= 0
        or int(tensor.shape[4]) <= 0
        or int(tensor.shape[3]) % 2
        or int(tensor.shape[4]) % 2
    ):
        raise SAICNativeSourceStateFieldError(
            f"{label} has wrong Bernini latent geometry"
        )
    return tensor


def _validate_condition(value: Any, *, label: str) -> torch.Tensor:
    condition = _finite_detached(value, label=label)
    if (
        condition.dtype != torch.bfloat16
        or tuple(int(x) for x in condition.shape) != (1, 512, 4096)
    ):
        raise SAICNativeSourceStateFieldError(
            f"{label} must be detached BF16 [1,512,4096]"
        )
    return condition


def _validate_patch(
    value: Any, *, label: str, expected_tokens: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise SAICNativeSourceStateFieldError(
            f"{label} patch must return tokens and rotary"
        )
    tokens = _finite_detached(value[0], label=f"{label} tokens")
    rotary = value[1]
    if (
        type(rotary) is not torch.Tensor
        or rotary.layout != torch.strided
        or not rotary.is_complex()
        or rotary.requires_grad
        or rotary.grad_fn is not None
        or not bool(torch.isfinite(rotary).all().item())
    ):
        raise SAICNativeSourceStateFieldError(
            f"{label} rotary must be a finite detached complex tensor"
        )
    if (
        tuple(int(x) for x in tokens.shape)
        != (1, expected_tokens, 1536)
        or tokens.dtype != torch.bfloat16
        or tokens.device != device
    ):
        raise SAICNativeSourceStateFieldError(f"{label} token geometry differs")
    if (
        tuple(int(x) for x in rotary.shape)
        != (1, 1, expected_tokens, 64)
        or rotary.dtype != torch.complex128
        or rotary.device != device
    ):
        raise SAICNativeSourceStateFieldError(f"{label} rotary geometry differs")
    return tokens, rotary


def _unpack_target_velocity(
    packed: torch.Tensor, *, video_shape: tuple[int, ...]
) -> torch.Tensor:
    batch, channels, phases, height, width = video_shape
    token_count = phases * (height // 2) * (width // 2)
    if tuple(int(x) for x in packed.shape) != (batch, token_count, 64):
        raise SAICNativeSourceStateFieldError("target-tail velocity geometry differs")
    patches = packed.reshape(batch, phases, height // 2, width // 2, 2, 2, channels)
    return (
        patches.permute(0, 6, 1, 2, 4, 3, 5)
        .reshape(batch, channels, phases, height, width)
        .contiguous()
    )


def native_schedule_sha256(
    sigma_scalars: Sequence[torch.Tensor],
    next_sigmas: Sequence[float],
    timestep_tensors: Sequence[torch.Tensor],
    candidate_schedule: Sequence[int],
    aggregation_mode: str,
    temperature: Optional[float],
) -> str:
    """Hash exact FP32 sigma and official INT64 timestep cells."""

    if not (len(sigma_scalars) == len(next_sigmas) == len(timestep_tensors) == 40):
        raise SAICNativeSourceStateFieldError(
            "native schedule digest requires exactly 40 complete cells"
        )
    candidates = tuple(candidate_schedule)
    if (
        len(candidates) != 40
        or any(type(value) is not int for value in candidates)
        or candidates not in ((1,) * 40, (5, 5, 5) + (1,) * 37)
    ):
        raise SAICNativeSourceStateFieldError(
            "candidate schedule must be registered K1 or K5-early exact40"
        )
    if aggregation_mode == "uniform":
        if temperature is not None:
            raise SAICNativeSourceStateFieldError(
                "uniform aggregation requires temperature=None"
            )
    elif aggregation_mode == "source_similarity_softmax":
        if (
            type(temperature) not in (int, float)
            or isinstance(temperature, bool)
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0
        ):
            raise SAICNativeSourceStateFieldError(
                "source-similarity aggregation requires positive temperature"
            )
    else:
        raise SAICNativeSourceStateFieldError("unregistered aggregation mode")
    if candidates == (1,) * 40 and aggregation_mode != "uniform":
        raise SAICNativeSourceStateFieldError("K1 requires uniform aggregation")
    payload = bytearray(b"bernini-saic-native-source-state-field-v1/schedule\0")
    payload.extend(aggregation_mode.encode("ascii") + b"\0")
    payload.extend(
        b"none" if temperature is None else struct.pack(">d", float(temperature))
    )
    for index, (sigma, next_sigma, timestep) in enumerate(
        zip(sigma_scalars, next_sigmas, timestep_tensors)
    ):
        payload.extend(struct.pack(">I", index))
        payload.extend(struct.pack(">f", float(sigma.item())))
        payload.extend(struct.pack(">f", float(next_sigma)))
        payload.extend(struct.pack(">q", int(timestep.item())))
        payload.extend(struct.pack(">I", candidates[index]))
    return hashlib.sha256(bytes(payload)).hexdigest()


def _core_sigma_schedule_sha256(value: Sequence[float]) -> str:
    if len(value) != 41:
        raise SAICNativeSourceStateFieldError(
            "core sigma schedule must contain 41 values"
        )
    schedule = tuple(float(item) for item in value)
    if (
        schedule[0] <= 0.0
        or schedule[-1] != 0.0
        or any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in schedule)
        or any(right >= left for left, right in zip(schedule, schedule[1:]))
    ):
        raise SAICNativeSourceStateFieldError("core sigma schedule is invalid")
    digest = hashlib.sha256(b"saic-exact40-sigma-schedule-v1\0")
    for index, sigma in enumerate(schedule):
        digest.update(struct.pack(">Id", index, sigma))
    return digest.hexdigest()


class NativeSourceStateFieldAdapter:
    """Stateful, fail-closed callback for one complete exact40 SSFT rollout."""

    def __init__(
        self,
        *,
        diffusion: Any,
        transformer: Any,
        field_regime: str,
        conditions: Mapping[str, torch.Tensor],
        captions: Mapping[str, str],
        sigma_scalars: Sequence[torch.Tensor],
        next_sigmas: Sequence[float],
        timestep_tensors: Sequence[torch.Tensor],
        candidate_schedule: tuple[int, ...],
        aggregation_mode: str,
        temperature: Optional[float],
        provenance: NativeFieldProvenance,
        reference_frame0_latent: Optional[torch.Tensor] = None,
    ) -> None:
        if field_regime not in ("t2v_apg", "r2v_apg_source_i0"):
            raise SAICNativeSourceStateFieldError(
                "field_regime must be t2v_apg or r2v_apg_source_i0"
            )
        if not isinstance(diffusion, torch.nn.Module) or not isinstance(
            transformer, torch.nn.Module
        ):
            raise SAICNativeSourceStateFieldError(
                "diffusion and transformer must be torch.nn.Module instances"
            )
        if getattr(diffusion, "transformer", None) is not transformer or getattr(
            diffusion, "transformer_2", None
        ) is not None:
            raise SAICNativeSourceStateFieldError(
                "diffusion must own this exact Bernini single transformer"
            )
        if getattr(diffusion, "use_unipc", None) is not True:
            raise SAICNativeSourceStateFieldError(
                "diffusion must use the pinned UniPC inference path"
            )
        if getattr(transformer, "dtype", None) is not torch.bfloat16:
            raise SAICNativeSourceStateFieldError(
                "Bernini-R transformer dtype must be exactly torch.bfloat16"
            )
        shared_step, shared_step_function = _capture_bound_method(
            diffusion, "shared_step"
        )
        patch_vae_latent, patch_vae_latent_function = _capture_bound_method(
            transformer, "patch_vae_latent"
        )
        self._diffusion = diffusion
        self._transformer = transformer
        self._shared_step = shared_step
        self._shared_step_function = shared_step_function
        self._patch_vae_latent = patch_vae_latent
        self._patch_vae_latent_function = patch_vae_latent_function
        (
            self._vendor_apg_module,
            self._vendor_single,
            self._vendor_chain,
            self._momentum_class,
        ) = _resolve_vendor_apg_symbols()
        self._text_momentum = self._momentum_class(momentum=0.0)
        self._image_momentum = (
            self._momentum_class(momentum=0.0)
            if field_regime == "r2v_apg_source_i0"
            else None
        )
        for label, value in (
            ("text", self._text_momentum),
            ("image", self._image_momentum),
        ):
            if value is not None and (
                type(value) is not self._momentum_class
                or type(getattr(value, "momentum", None)) not in (int, float)
                or float(value.momentum) != 0.0
            ):
                raise SAICNativeSourceStateFieldError(
                    f"vendor {label} momentum buffer is not exact zero momentum"
                )
        self._field_regime = field_regime
        if type(provenance) is not _NATIVE_FIELD_PROVENANCE_TYPE:
            raise SAICNativeSourceStateFieldError(
                "provenance must be the exact NativeFieldProvenance type"
            )
        # Seal caller-owned mappings exactly once before validation.  A frozen
        # dataclass does not make nested Mapping objects immutable.
        prompt_utf8_by_role = dict(provenance.prompt_utf8_sha256_by_role)
        prompt_condition_by_key = dict(provenance.prompt_condition_sha256_by_key)
        sealed_provenance = _NATIVE_FIELD_PROVENANCE_TYPE(
            model_id=provenance.model_id,
            checkpoint_sha256=provenance.checkpoint_sha256,
            model_receipt_sha256=provenance.model_receipt_sha256,
            guidance_contract_sha256=provenance.guidance_contract_sha256,
            negative_prompt_sha256=provenance.negative_prompt_sha256,
            native_schedule_sha256=provenance.native_schedule_sha256,
            noise_generator_id=provenance.noise_generator_id,
            master_seed=provenance.master_seed,
            noise_bank_sha256=provenance.noise_bank_sha256,
            reference_encoder_sha256=provenance.reference_encoder_sha256,
            reference_frame0_latent_sha256=provenance.reference_frame0_latent_sha256,
            prompt_utf8_sha256_by_role=prompt_utf8_by_role,
            prompt_condition_sha256_by_key=prompt_condition_by_key,
        )
        _PROVENANCE_VALIDATE_FUNCTION(sealed_provenance, regime=field_regime)
        # Copy every consumed provenance leaf into a private primitive.  The
        # frozen dataclass may contain caller-owned Mapping objects, which are
        # deliberately never trusted again after construction.
        self._model_id = sealed_provenance.model_id
        self._checkpoint_sha256 = sealed_provenance.checkpoint_sha256
        self._model_receipt_sha256 = sealed_provenance.model_receipt_sha256
        self._guidance_contract_sha256 = sealed_provenance.guidance_contract_sha256
        self._negative_prompt_sha256 = sealed_provenance.negative_prompt_sha256
        self._native_schedule_sha256 = sealed_provenance.native_schedule_sha256
        self._noise_generator_id = sealed_provenance.noise_generator_id
        self._master_seed = sealed_provenance.master_seed
        self._noise_bank_sha256 = sealed_provenance.noise_bank_sha256
        self._reference_encoder_sha256 = sealed_provenance.reference_encoder_sha256
        self._reference_frame0_latent_sha256 = (
            sealed_provenance.reference_frame0_latent_sha256
        )
        self._prompt_utf8_sha256_by_role = prompt_utf8_by_role
        self._prompt_condition_sha256_by_key = prompt_condition_by_key
        provenance_payload = (
            self._model_id,
            self._checkpoint_sha256,
            self._model_receipt_sha256,
            self._guidance_contract_sha256,
            self._negative_prompt_sha256,
            self._native_schedule_sha256,
            self._noise_generator_id,
            str(self._master_seed),
            self._noise_bank_sha256,
            self._reference_encoder_sha256,
            self._reference_frame0_latent_sha256,
            *(f"{key}:{self._prompt_utf8_sha256_by_role[key]}" for key in ("target", "source")),
            *(
                f"{key}:{self._prompt_condition_sha256_by_key[key]}"
                for key in ("negative", "target", "source")
            ),
        )
        provenance_digest = hashlib.sha256(
            b"bernini-saic-native-field-provenance-v1\0"
        )
        for value in provenance_payload:
            encoded = value.encode("utf-8")
            provenance_digest.update(struct.pack(">Q", len(encoded)))
            provenance_digest.update(encoded)
        self._provenance_seal_sha256 = provenance_digest.hexdigest()
        initial_digest_cache: dict[tuple[Any, ...], str] = {}
        self._module_seals = {
            "diffusion": _module_content_seal(
                diffusion,
                label="diffusion",
                tensor_digest_cache=initial_digest_cache,
            ),
            "transformer": _module_content_seal(
                transformer,
                label="transformer",
                tensor_digest_cache=initial_digest_cache,
            ),
        }
        self._module_structure_seals = {
            "diffusion": _module_structure_seal(diffusion, label="diffusion"),
            "transformer": _module_structure_seal(transformer, label="transformer"),
        }

        condition_inputs = dict(conditions)
        if set(condition_inputs) != {"negative", "target", "source"}:
            raise SAICNativeSourceStateFieldError(
                "conditions must contain exactly negative, target, and source"
            )
        self._conditions = {
            key: _validate_condition(condition_inputs[key], label=f"condition[{key}]")
            for key in ("negative", "target", "source")
        }
        condition_values = tuple(self._conditions.values())
        if any(
            _shares_storage(left, right)
            for index, left in enumerate(condition_values)
            for right in condition_values[index + 1 :]
        ):
            raise SAICNativeSourceStateFieldError("condition tensors may not alias")
        for key, condition in self._conditions.items():
            if _tensor_bytes_sha256(condition) != self._prompt_condition_sha256_by_key[key]:
                raise SAICNativeSourceStateFieldError(
                    f"condition[{key}] bytes differ from provenance"
                )
        self._condition_snapshots = {
            key: _tensor_snapshot(value) for key, value in self._conditions.items()
        }

        caption_inputs = dict(captions)
        if set(caption_inputs) != {"target", "source"}:
            raise SAICNativeSourceStateFieldError(
                "captions must contain exactly target and source"
            )
        self._captions = {
            key: _text(caption_inputs[key], label=f"caption[{key}]")
            for key in ("target", "source")
        }
        if self._captions["target"] == self._captions["source"]:
            raise SAICNativeSourceStateFieldError(
                "adapter is unnecessary for an exact caption no-op"
            )
        for role, caption in self._captions.items():
            digest = hashlib.sha256(caption.encode("utf-8")).hexdigest()
            if digest != self._prompt_utf8_sha256_by_role[role]:
                raise SAICNativeSourceStateFieldError(
                    f"caption[{role}] bytes differ from provenance"
                )

        self._sigma_scalars, self._next_sigmas, self._timesteps = self._validate_schedule(
            tuple(sigma_scalars), tuple(next_sigmas), tuple(timestep_tensors)
        )
        if (
            type(candidate_schedule) is not tuple
            or len(candidate_schedule) != 40
            or any(type(value) is not int for value in candidate_schedule)
            or candidate_schedule not in (
                (1,) * 40,
                (5, 5, 5) + (1,) * 37,
            )
        ):
            raise SAICNativeSourceStateFieldError(
                "candidate_schedule must be exact registered K1 or K5-early tuple"
            )
        self._candidate_schedule = candidate_schedule
        # Reuse the same literal policy as the core, but seal a private copy so
        # module-global rebinding cannot change this adapter instance.
        if aggregation_mode == "uniform":
            if temperature is not None:
                raise SAICNativeSourceStateFieldError(
                    "uniform aggregation requires temperature=None"
                )
            self._temperature = None
        elif aggregation_mode == "source_similarity_softmax":
            if (
                type(temperature) not in (int, float)
                or isinstance(temperature, bool)
                or not math.isfinite(float(temperature))
                or float(temperature) <= 0.0
            ):
                raise SAICNativeSourceStateFieldError(
                    "source-similarity aggregation requires positive temperature"
                )
            self._temperature = float(temperature)
        else:
            raise SAICNativeSourceStateFieldError("unregistered aggregation mode")
        if self._candidate_schedule == (1,) * 40 and aggregation_mode != "uniform":
            raise SAICNativeSourceStateFieldError("K1 requires uniform aggregation")
        self._aggregation_mode = aggregation_mode
        if native_schedule_sha256(
            self._sigma_scalars,
            self._next_sigmas,
            self._timesteps,
            self._candidate_schedule,
            self._aggregation_mode,
            self._temperature,
        ) != self._native_schedule_sha256:
            raise SAICNativeSourceStateFieldError(
                "native exact40 schedule bytes differ from provenance"
            )
        self._sigma_schedule = tuple(
            float(value.item()) for value in self._sigma_scalars
        ) + (self._next_sigmas[-1],)
        self._core_sigma_schedule_sha256 = _core_sigma_schedule_sha256(
            self._sigma_schedule
        )
        self._schedule_snapshots = tuple(
            (_tensor_snapshot(sigma), _tensor_snapshot(timestep))
            for sigma, timestep in zip(self._sigma_scalars, self._timesteps)
        )
        if len({id(value) for value in self._sigma_scalars}) != 40 or len(
            {id(value) for value in self._timesteps}
        ) != 40:
            raise SAICNativeSourceStateFieldError(
                "native schedule cells must be 40 distinct tensor view objects"
            )
        if len(
            {(_storage_ptr(value), int(value.storage_offset())) for value in self._sigma_scalars}
        ) != 40 or len(
            {(_storage_ptr(value), int(value.storage_offset())) for value in self._timesteps}
        ) != 40:
            raise SAICNativeSourceStateFieldError(
                "native schedule cells must identify 40 distinct storage offsets"
            )

        if field_regime == "t2v_apg":
            if reference_frame0_latent is not None:
                raise SAICNativeSourceStateFieldError(
                    "target-only T2V cannot consume a visual reference"
                )
            self._reference = None
            self._reference_snapshot = None
        else:
            reference = _validate_latent(
                reference_frame0_latent, label="reference_frame0_latent", reference=True
            )
            self._reference = reference
            self._reference_snapshot = _tensor_snapshot(reference)
            if (
                _tensor_bytes_sha256(reference)
                != self._reference_frame0_latent_sha256
            ):
                raise SAICNativeSourceStateFieldError(
                    "source frame-0 latent bytes differ from provenance"
                )

        self._expected_step = 0
        self._expected_candidate = 0
        self._expected_role = "target"
        self._guided_attempts = 0
        self._guided_successes = 0
        self._raw_attempts = 0
        self._raw_successes = 0
        self._target_tail_views = 0
        self._query_patch_attempts = 0
        self._query_patch_successes = 0
        self._reference_patch_attempts = 0
        self._reference_patch_successes = 0
        self._vendor_single_attempts = 0
        self._vendor_single_successes = 0
        self._vendor_chain_attempts = 0
        self._vendor_chain_successes = 0
        self._finalized = False
        self._poisoned = False
        self._failure_stage: Optional[str] = None
        self._active_stage = "idle"
        self._final_content_seals: Optional[dict[str, str]] = None
        self._final_diagnostics: Optional[NativeFieldDiagnostics] = None

    @property
    def diffusion(self) -> torch.nn.Module:
        return self._diffusion

    @property
    def transformer(self) -> torch.nn.Module:
        return self._transformer

    @property
    def field_regime(self) -> str:
        return self._field_regime

    @staticmethod
    def _validate_schedule(
        sigma_scalars: Sequence[torch.Tensor],
        next_sigmas: Sequence[float],
        timestep_tensors: Sequence[torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, ...], tuple[float, ...], tuple[torch.Tensor, ...]]:
        if any(len(value) != 40 for value in (sigma_scalars, next_sigmas, timestep_tensors)):
            raise SAICNativeSourceStateFieldError("native schedule must contain exactly 40 cells")
        sigmas: list[torch.Tensor] = []
        next_values: list[float] = []
        times: list[torch.Tensor] = []
        previous_timestep: Optional[int] = None
        for index, (sigma, next_sigma, timestep) in enumerate(
            zip(sigma_scalars, next_sigmas, timestep_tensors)
        ):
            if (
                type(sigma) is not torch.Tensor
                or sigma.dtype != torch.float32
                or sigma.device.type != "cpu"
                or sigma.ndim != 0
                or sigma.layout != torch.strided
                or sigma.requires_grad
                or sigma.grad_fn is not None
                or not bool(torch.isfinite(sigma).item())
                or not 0.0 < float(sigma.item()) <= 1.0
            ):
                raise SAICNativeSourceStateFieldError(
                    f"sigma_scalars[{index}] must be a positive CPU FP32 scalar"
                )
            if type(next_sigma) not in (int, float) or not math.isfinite(float(next_sigma)):
                raise SAICNativeSourceStateFieldError(f"next_sigmas[{index}] is invalid")
            next_value = float(next_sigma)
            if not 0.0 <= next_value < float(sigma.item()):
                raise SAICNativeSourceStateFieldError(
                    f"next_sigmas[{index}] must be smaller and nonnegative"
                )
            if index and float(sigma.item()) != next_values[-1]:
                raise SAICNativeSourceStateFieldError(
                    "native sigma cells are not one contiguous exact40 chain"
                )
            if (
                type(timestep) is not torch.Tensor
                or timestep.dtype != torch.int64
                or tuple(timestep.shape) != (1,)
                or timestep.layout != torch.strided
                or timestep.device.type == "meta"
                or timestep.requires_grad
                or timestep.grad_fn is not None
                or not 0 < int(timestep.item()) < 1000
                or (
                    previous_timestep is not None
                    and int(timestep.item()) >= previous_timestep
                )
            ):
                raise SAICNativeSourceStateFieldError(
                    f"timestep_tensors[{index}] must be official descending INT64"
                )
            previous_timestep = int(timestep.item())
            sigmas.append(sigma)
            next_values.append(next_value)
            times.append(timestep)
        if next_values[-1] != 0.0:
            raise SAICNativeSourceStateFieldError("exact40 native schedule must terminate at zero")
        if struct.pack(">d", next_values[-1]) != struct.pack(">d", 0.0):
            raise SAICNativeSourceStateFieldError(
                "exact40 native schedule must terminate at bit-exact positive zero"
            )
        return tuple(sigmas), tuple(next_values), tuple(times)

    def _audit_closure(self) -> None:
        if (
            getattr(self.diffusion, "transformer", None) is not self.transformer
            or getattr(self.diffusion, "transformer_2", None) is not None
            or getattr(self.diffusion, "use_unipc", None) is not True
            or getattr(self.transformer, "dtype", None) is not torch.bfloat16
            or not _bound_method_is_current(
                self.diffusion, "shared_step", self._shared_step_function
            )
            or not _bound_method_is_current(
                self.transformer,
                "patch_vae_latent",
                self._patch_vae_latent_function,
            )
            or getattr(
                self._vendor_apg_module, "normalized_guidance", None
            ) is not self._vendor_single
            or getattr(
                self._vendor_apg_module, "normalized_guidance_chain", None
            ) is not self._vendor_chain
            or getattr(self._vendor_apg_module, "MomentumBuffer", None)
            is not self._momentum_class
            or type(self._text_momentum) is not self._momentum_class
            or float(getattr(self._text_momentum, "momentum", math.nan)) != 0.0
            or (
                self._image_momentum is not None
                and (
                    type(self._image_momentum) is not self._momentum_class
                    or float(
                        getattr(self._image_momentum, "momentum", math.nan)
                    )
                    != 0.0
                )
            )
        ):
            raise SAICNativeSourceStateFieldError(
                "native owner topology/dtype/callable identity was replaced"
            )
        for key, value in self._conditions.items():
            if _tensor_snapshot(value) != self._condition_snapshots[key]:
                raise SAICNativeSourceStateFieldError(f"condition[{key}] was mutated")
        if self._reference is not None and _tensor_snapshot(self._reference) != self._reference_snapshot:
            raise SAICNativeSourceStateFieldError("source frame-0 reference was mutated")
        for index, (sigma, timestep) in enumerate(zip(self._sigma_scalars, self._timesteps)):
            expected_sigma, expected_time = self._schedule_snapshots[index]
            if _tensor_snapshot(sigma) != expected_sigma or _tensor_snapshot(timestep) != expected_time:
                raise SAICNativeSourceStateFieldError(
                    f"native schedule cell {index} was mutated"
                )
        for label, module in (("diffusion", self.diffusion), ("transformer", self.transformer)):
            if _module_structure_seal(module, label=label) != self._module_structure_seals[label]:
                raise SAICNativeSourceStateFieldError(
                    f"frozen {label} structure/version state was mutated"
                )

    def _validate_request(self, request: Any) -> torch.Tensor:
        if type(request) is not _VELOCITY_QUERY_REQUEST_TYPE:
            raise SAICNativeSourceStateFieldError(
                "callback accepts only the exact SSFT VelocityQueryRequest"
            )
        if (
            type(request.step) is not _FLOW_TRANSPORT_STEP_BINDING_TYPE
            or type(request.step.native) is not _NATIVE_GUIDANCE_BINDING_TYPE
        ):
            raise SAICNativeSourceStateFieldError(
                "request step/native binding types differ"
            )
        if request.step.native.field_regime != self.field_regime:
            raise SAICNativeSourceStateFieldError(
                "request native model/APG regime closure differs"
            )
        expected_raw_per_guided = 2 if self.field_regime == "t2v_apg" else 3
        expected_raw_per_candidate = 2 * expected_raw_per_guided
        if (
            request.request_schema
            != "bernini-saic-source-state-flow-transport-v1/guided-velocity-request-v1"
            or type(request.expected_raw_transformer_forwards) is not int
            or request.expected_raw_transformer_forwards
            != expected_raw_per_guided
        ):
            raise SAICNativeSourceStateFieldError("guided request schema/forward count differs")
        if request.role != self._expected_role or request.role not in ("target", "source"):
            raise SAICNativeSourceStateFieldError(
                f"call-order drift: expected role {self._expected_role}, observed {request.role}"
            )
        if (
            type(request.step.step_index) is not int
            or type(request.candidate_index) is not int
            or request.step.step_index != self._expected_step
            or request.candidate_index != self._expected_candidate
        ):
            raise SAICNativeSourceStateFieldError("step/candidate call-order drift")
        native = request.step.native
        expected_guidance_mode = "t2v_apg" if self.field_regime == "t2v_apg" else "r2v_apg"
        expected_branch_order = (
            (
                "target_negative",
                "target_condition",
                "source_negative",
                "source_condition",
            )
            if self.field_regime == "t2v_apg"
            else (
                "target_none_negative",
                "target_i0_negative",
                "target_i0_condition",
                "source_none_negative",
                "source_i0_negative",
                "source_i0_condition",
            )
        )
        expected_image_scale = 0.0 if self.field_regime == "t2v_apg" else 4.5
        expected_chain_scales = (
            (4.0,) if self.field_regime == "t2v_apg" else (4.5, 4.0)
        )
        expected_norm_thresholds = (
            (50.0,) if self.field_regime == "t2v_apg" else (50.0, 50.0)
        )
        expected_momenta = (
            (0.0,) if self.field_regime == "t2v_apg" else (0.0, 0.0)
        )
        if (
            native.field_regime != self.field_regime
            or native.guidance_mode != expected_guidance_mode
            or native.model_id != self._model_id
            or native.checkpoint_sha256 != self._checkpoint_sha256
            or native.guidance_contract_sha256 != self._guidance_contract_sha256
            or native.negative_prompt_sha256 != self._negative_prompt_sha256
            or type(native.guidance_scale) not in (int, float)
            or float(native.guidance_scale) != 4.0
            or type(native.image_guidance_scale) not in (int, float)
            or float(native.image_guidance_scale) != expected_image_scale
            or type(native.guidance_chain_scales) is not tuple
            or native.guidance_chain_scales != expected_chain_scales
            or type(native.apg_eta) not in (int, float)
            or float(native.apg_eta) != 0.5
            or type(native.apg_norm_threshold) not in (int, float)
            or float(native.apg_norm_threshold) != 50.0
            or type(native.apg_norm_thresholds) is not tuple
            or native.apg_norm_thresholds != expected_norm_thresholds
            or type(native.apg_momentum) not in (int, float)
            or float(native.apg_momentum) != 0.0
            or type(native.apg_momenta) is not tuple
            or native.apg_momenta != expected_momenta
            or native.branch_order != expected_branch_order
            or type(native.raw_transformer_forwards_per_candidate) is not int
            or native.raw_transformer_forwards_per_candidate
            != expected_raw_per_candidate
            or request.step.noise_generator_id != self._noise_generator_id
            or request.step.master_seed != self._master_seed
            or request.step.noise_bank_sha256 != self._noise_bank_sha256
        ):
            raise SAICNativeSourceStateFieldError("request native model/APG regime closure differs")
        index = self._expected_step
        if (
            type(request.step.candidate_count) is not int
            or type(request.step.guided_velocity_queries_per_candidate) is not int
            or type(request.step.raw_transformer_forwards_per_candidate) is not int
            or type(request.step.master_seed) is not int
            or
            float(request.step.sigma) != float(self._sigma_scalars[index].item())
            or float(request.step.time) != float(self._sigma_scalars[index].item())
            or float(request.step.next_sigma) != self._next_sigmas[index]
            or float(request.step.next_time) != self._next_sigmas[index]
            or request.step.candidate_schedule != self._candidate_schedule
            or request.step.candidate_count != self._candidate_schedule[index]
            or request.step.aggregation_mode != self._aggregation_mode
            or request.step.temperature != self._temperature
            or request.step.sigma_schedule != self._sigma_schedule
            or request.step.sigma_schedule_sha256
            != self._core_sigma_schedule_sha256
            or request.step.candidate_continuation != "candidate_zero"
            or request.step.time_parameterization != "flow_time_equals_sigma"
            or request.step.guided_velocity_queries_per_candidate != 2
            or request.step.raw_transformer_forwards_per_candidate
            != expected_raw_per_candidate
        ):
            raise SAICNativeSourceStateFieldError("request exact40 sigma/time cell differs")
        if request.caption != self._captions[request.role] or hashlib.sha256(
            request.caption.encode("utf-8")
        ).hexdigest() != self._prompt_utf8_sha256_by_role[request.role]:
            raise SAICNativeSourceStateFieldError("request role/caption binding differs")
        state = _validate_latent(request.state, label="request.state", reference=False)
        if _tensor_bytes_sha256(state) != request.state_sha256:
            raise SAICNativeSourceStateFieldError("request state digest differs")
        if self._reference is not None and (
            state.device != self._reference.device
            or tuple(state.shape[3:]) != tuple(self._reference.shape[3:])
            or _shares_storage(state, self._reference)
        ):
            raise SAICNativeSourceStateFieldError("query/reference geometry or alias closure differs")
        for condition in self._conditions.values():
            if condition.device != state.device or _shares_storage(state, condition):
                raise SAICNativeSourceStateFieldError("query/condition device or alias closure differs")
        if self._timesteps[index].device != state.device:
            raise SAICNativeSourceStateFieldError("native timestep/query device differs")
        if any(
            _shares_storage(state, value)
            for value in (*self._sigma_scalars, *self._timesteps)
        ):
            raise SAICNativeSourceStateFieldError(
                "query state may not alias native schedule storage"
            )
        return state

    def _ensure_reference_patch(self, *, state: torch.Tensor) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if self._reference is None:
            return None
        # Match the physical pinned Bernini R2V path: every independent guided
        # query repatches the sealed I0 first, then patches the target.  The
        # negative/conditional shared_step pair alone reuses one exact pack.
        self._active_stage = "reference_patch"
        with torch.inference_mode():
            patch_input = self._reference.to(dtype=torch.bfloat16)
            input_snapshot = _tensor_snapshot(patch_input)
            self._reference_patch_attempts += 1
            raw = self._patch_vae_latent(
                hidden_states=patch_input, source_id=1.0
            )
        expected = (int(state.shape[3]) // 2) * (int(state.shape[4]) // 2)
        reference_patch = _validate_patch(
            raw, label="source frame-0", expected_tokens=expected, device=state.device
        )
        if (
            _tensor_snapshot(patch_input) != input_snapshot
            or any(
                _shares_storage(value, patch_input) for value in reference_patch
            )
            or _shares_storage(reference_patch[0], reference_patch[1])
        ):
            raise SAICNativeSourceStateFieldError(
                "source frame-0 patch mutated/aliased its native input"
            )
        self._reference_patch_successes += 1
        return reference_patch

    def _advance(self, request: transport.VelocityQueryRequest) -> None:
        if request.role == "target":
            self._expected_role = "source"
            return
        self._expected_role = "target"
        if self._expected_candidate + 1 < request.step.candidate_count:
            self._expected_candidate += 1
        else:
            self._expected_candidate = 0
            self._expected_step += 1

    def _execute(self, request: transport.VelocityQueryRequest) -> torch.Tensor:
        self._active_stage = "query_preflight"
        self._audit_closure()
        state = self._validate_request(request)
        state_snapshot = _tensor_snapshot(state)
        reference_patch = self._ensure_reference_patch(state=state)

        self._active_stage = "target_patch"
        with torch.inference_mode():
            query_patch_input = state.to(dtype=torch.bfloat16)
            query_patch_input_snapshot = _tensor_snapshot(query_patch_input)
            self._query_patch_attempts += 1
            raw_query_patch = self._patch_vae_latent(
                hidden_states=query_patch_input, source_id=0.0
            )
        target_tokens_expected = int(state.shape[2]) * (int(state.shape[3]) // 2) * (
            int(state.shape[4]) // 2
        )
        query_tokens, query_rotary = _validate_patch(
            raw_query_patch,
            label="target query",
            expected_tokens=target_tokens_expected,
            device=state.device,
        )
        if (
            _tensor_snapshot(query_patch_input) != query_patch_input_snapshot
            or _shares_storage(query_tokens, query_patch_input)
            or _shares_storage(query_rotary, query_patch_input)
            or _shares_storage(query_tokens, query_rotary)
        ):
            raise SAICNativeSourceStateFieldError(
                "target patch mutated/aliased its native input"
            )
        self._query_patch_successes += 1
        target_count = int(query_tokens.shape[1])
        if reference_patch is None:
            packed_tokens = query_tokens
            packed_rotary = query_rotary
            branch_calls = (
                ("t2v_negative", query_tokens, query_rotary, "negative"),
                ("t2v_role", query_tokens, query_rotary, request.role),
            )
        else:
            reference_tokens, reference_rotary = reference_patch
            packed_tokens = torch.cat((reference_tokens, query_tokens), dim=1)
            packed_rotary = torch.cat((reference_rotary, query_rotary), dim=2)
            branch_calls = (
                ("r2v_none_negative", query_tokens, query_rotary, "negative"),
                ("r2v_i0_negative", packed_tokens, packed_rotary, "negative"),
                ("r2v_i0_role", packed_tokens, packed_rotary, request.role),
            )
        total_count = int(packed_tokens.shape[1])
        expected_total = target_count + (0 if reference_patch is None else int(reference_patch[0].shape[1]))
        if total_count != expected_total:
            raise SAICNativeSourceStateFieldError("packed visual token count differs")
        timestep = self._timesteps[request.step.step_index]
        tracked = (
            (state, query_tokens, query_rotary, timestep)
            if reference_patch is None
            else (
                state,
                query_tokens,
                query_rotary,
                packed_tokens,
                packed_rotary,
                timestep,
            )
        )
        tracked_snapshots = tuple(_tensor_snapshot(value) for value in tracked)
        object_ids = tuple(id(value) for value in tracked[1:])

        raw_spatial: list[torch.Tensor] = []
        for branch_name, branch_tokens, branch_rotary, condition_key in branch_calls:
            branch_total = int(branch_tokens.shape[1])
            self._active_stage = f"shared_step:{branch_name}"
            with torch.inference_mode():
                self._raw_attempts += 1
                prediction = self._shared_step(
                    model_id=self._model_id,
                    noisy_latents=branch_tokens,
                    timesteps=timestep,
                    cond_embeds=self._conditions[condition_key],
                    rotary_embs=branch_rotary,
                    batch_vae_seqlen=[branch_total],
                    batch_text_seqlen=[512],
                )
            if (
                type(prediction) is not torch.Tensor
                or tuple(int(x) for x in prediction.shape)
                != (1, branch_total, 64)
                or prediction.device != state.device
                or prediction.dtype != torch.bfloat16
                or not prediction.is_contiguous()
                or prediction.requires_grad
                or prediction.grad_fn is not None
                or not bool(torch.isfinite(prediction).all().item())
            ):
                raise SAICNativeSourceStateFieldError(
                    "frozen shared_step must return detached BF16 [1,total,64] velocity"
                )
            if any(_shares_storage(prediction, value) for value in tracked):
                raise SAICNativeSourceStateFieldError(
                    "shared_step prediction may not alias query/packed/timestep storage"
                )
            target_tail = prediction[:, -target_count:, :]
            if _storage_ptr(target_tail) != _storage_ptr(prediction):
                raise SAICNativeSourceStateFieldError("target tail is not a direct storage view")
            self._target_tail_views += 1
            raw_spatial.append(
                _unpack_target_velocity(
                    target_tail, video_shape=tuple(int(x) for x in state.shape)
                )
            )
            if tuple(id(value) for value in tracked[1:]) != object_ids or tuple(
                _tensor_snapshot(value) for value in tracked
            ) != tracked_snapshots:
                raise SAICNativeSourceStateFieldError(
                    "shared query/pack/rotary/timestep bytes or objects changed across APG program"
                )
            self._raw_successes += 1

        sigma = self._sigma_scalars[request.step.step_index]
        self._active_stage = "clean_space_apg"
        clean_states = tuple(state - sigma * value for value in raw_spatial)
        clean_snapshots = tuple(_tensor_snapshot(value) for value in clean_states)
        with torch.inference_mode():
            if reference_patch is None:
                self._vendor_single_attempts += 1
                guided_clean = self._vendor_single(
                    pred_cond=clean_states[1],
                    pred_uncond=clean_states[0],
                    guidance_scale=4.0,
                    momentum_buffer=self._text_momentum,
                    eta=0.5,
                    norm_threshold=50.0,
                )
            else:
                assert self._image_momentum is not None
                self._vendor_chain_attempts += 1
                guided_clean = self._vendor_chain(
                    pred_uncond=clean_states[0],
                    preds=[clean_states[1], clean_states[2]],
                    scales=[4.5, 4.0],
                    momentum_buffers=[
                        self._image_momentum,
                        self._text_momentum,
                    ],
                    eta=0.5,
                    norm_thresholds=[50.0, 50.0],
                )
        momentum_expectations = (
            ((self._text_momentum, clean_states[1] - clean_states[0]),)
            if reference_patch is None
            else (
                (self._image_momentum, clean_states[1] - clean_states[0]),
                (self._text_momentum, clean_states[2] - clean_states[1]),
            )
        )
        for momentum_buffer, expected_average in momentum_expectations:
            running_average = getattr(momentum_buffer, "running_average", None)
            if (
                type(running_average) is not torch.Tensor
                or tuple(running_average.shape) != tuple(expected_average.shape)
                or running_average.dtype != expected_average.dtype
                or running_average.device != expected_average.device
                or running_average.requires_grad
                or running_average.grad_fn is not None
                or not torch.equal(running_average, expected_average)
                or _shares_storage(running_average, expected_average)
            ):
                raise SAICNativeSourceStateFieldError(
                    "vendor zero-momentum APG buffer update differs"
                )
        if (
            type(guided_clean) is not torch.Tensor
            or tuple(guided_clean.shape) != tuple(state.shape)
            or guided_clean.dtype != torch.float32
            or guided_clean.device != state.device
            or guided_clean.requires_grad
            or guided_clean.grad_fn is not None
            or not bool(torch.isfinite(guided_clean).all().item())
            or any(_shares_storage(guided_clean, value) for value in clean_states)
            or tuple(_tensor_snapshot(value) for value in clean_states)
            != clean_snapshots
        ):
            raise SAICNativeSourceStateFieldError(
                "vendor clean-space APG output/input closure differs"
            )
        if reference_patch is None:
            self._vendor_single_successes += 1
        else:
            self._vendor_chain_successes += 1
        guided_velocity = ((state - guided_clean) / sigma).float().detach().clone().contiguous()
        if (
            tuple(guided_velocity.shape) != tuple(state.shape)
            or not bool(torch.isfinite(guided_velocity).all().item())
            or _shares_storage(guided_velocity, state)
            or (self._reference is not None and _shares_storage(guided_velocity, self._reference))
        ):
            raise SAICNativeSourceStateFieldError("guided spatial velocity closure differs")
        if _tensor_snapshot(state) != state_snapshot:
            raise SAICNativeSourceStateFieldError("request state was mutated")
        self._guided_successes += 1
        self._advance(request)
        self._active_stage = "post_query_audit"
        self._audit_closure()
        self._active_stage = "idle"
        return guided_velocity

    def __call__(self, request: transport.VelocityQueryRequest) -> torch.Tensor:
        if self._finalized:
            raise SAICNativeSourceStateFieldError(
                "a finalized native field adapter cannot execute more queries"
            )
        if self._poisoned:
            raise SAICNativeSourceStateFieldError(
                "a failed native field adapter cannot be resumed"
            )
        try:
            self._guided_attempts += 1
            return self._execute(request)
        except Exception as error:
            self._poisoned = True
            self._failure_stage = self._active_stage
            if isinstance(error, SAICNativeSourceStateFieldError):
                raise
            raise SAICNativeSourceStateFieldError(
                "native guided query failed; adapter is permanently poisoned"
            ) from error

    def finalize(self) -> NativeFieldDiagnostics:
        """Perform the mandatory second full model-content audit.

        The per-query seal deliberately does not read 1.3B parameter bytes.
        Consequently only this successful end audit permits
        ``rollout_complete=True``. K1/K5 use 80/104 guided queries: T2V has
        160/208 raw forwards, while native R2V-I0 has 240/312.
        Arbitrary same-process reflection remains outside the authority of a
        Python callback; a runner must also bind immutable source-tree and
        checkpoint receipts in its external execution record.
        """

        if self._finalized:
            assert self._final_diagnostics is not None
            return self._final_diagnostics
        if self._poisoned:
            raise SAICNativeSourceStateFieldError(
                "a failed native field adapter cannot be finalized"
            )
        try:
            self._audit_closure()
            expected_guided = 2 * sum(self._candidate_schedule)
            expected_raw = (
                2 if self.field_regime == "t2v_apg" else 3
            ) * expected_guided
            final_seals: dict[str, str] = {}
            final_digest_cache: dict[tuple[Any, ...], str] = {}
            for label, module in (
                ("diffusion", self.diffusion),
                ("transformer", self.transformer),
            ):
                final_seals[label] = _module_content_seal(
                    module,
                    label=label,
                    tensor_digest_cache=final_digest_cache,
                )
                if final_seals[label] != self._module_seals[label]:
                    raise SAICNativeSourceStateFieldError(
                        f"frozen {label} content was mutated before finalization"
                    )
            if (
                self._expected_step != 40
                or self._expected_candidate != 0
                or self._expected_role != "target"
                or self._guided_attempts != expected_guided
                or self._guided_successes != expected_guided
                or self._raw_attempts != expected_raw
                or self._raw_successes != expected_raw
                or self._query_patch_attempts != expected_guided
                or self._query_patch_successes != expected_guided
                or self._reference_patch_attempts
                != (0 if self.field_regime == "t2v_apg" else expected_guided)
                or self._reference_patch_successes
                != (0 if self.field_regime == "t2v_apg" else expected_guided)
                or self._vendor_single_attempts
                != (expected_guided if self.field_regime == "t2v_apg" else 0)
                or self._vendor_single_successes
                != (expected_guided if self.field_regime == "t2v_apg" else 0)
                or self._vendor_chain_attempts
                != (expected_guided if self.field_regime != "t2v_apg" else 0)
                or self._vendor_chain_successes
                != (expected_guided if self.field_regime != "t2v_apg" else 0)
            ):
                raise SAICNativeSourceStateFieldError(
                    "cannot finalize before the complete exact40 native call protocol"
                )
            self._final_content_seals = final_seals
            self._finalized = True
            self._final_diagnostics = self._build_diagnostics()
            return self._final_diagnostics
        except Exception as error:
            self._poisoned = True
            self._failure_stage = f"finalize:{type(error).__name__}"
            if isinstance(error, SAICNativeSourceStateFieldError):
                raise
            raise SAICNativeSourceStateFieldError(
                "native finalization failed; adapter is permanently poisoned"
            ) from error

    close = finalize

    def _build_diagnostics(self) -> NativeFieldDiagnostics:
        return _NATIVE_FIELD_DIAGNOSTICS_TYPE(
            field_regime=self.field_regime,
            guided_query_count=self._guided_successes,
            raw_transformer_forward_count=self._raw_attempts,
            patch_query_count=self._query_patch_attempts,
            patch_reference_count=self._reference_patch_attempts,
            guided_query_attempt_count=self._guided_attempts,
            guided_query_success_count=self._guided_successes,
            raw_transformer_forward_attempt_count=self._raw_attempts,
            raw_transformer_forward_success_count=self._raw_successes,
            patch_query_attempt_count=self._query_patch_attempts,
            patch_query_success_count=self._query_patch_successes,
            patch_reference_attempt_count=self._reference_patch_attempts,
            patch_reference_success_count=self._reference_patch_successes,
            vendor_single_attempt_count=self._vendor_single_attempts,
            vendor_single_success_count=self._vendor_single_successes,
            vendor_chain_attempt_count=self._vendor_chain_attempts,
            vendor_chain_success_count=self._vendor_chain_successes,
            expected_guided_query_count=2 * sum(self._candidate_schedule),
            expected_raw_transformer_forward_count=(
                (4 if self.field_regime == "t2v_apg" else 6)
                * sum(self._candidate_schedule)
            ),
            next_step_index=self._expected_step,
            next_candidate_index=self._expected_candidate,
            next_role=self._expected_role,
            initial_full_model_content_audit=True,
            final_full_model_content_audit=self._finalized,
            rollout_complete=self._finalized,
            initial_model_content_seal_sha256_by_module=tuple(
                sorted(self._module_seals.items())
            ),
            final_model_content_seal_sha256_by_module=tuple(
                sorted((self._final_content_seals or {}).items())
            ),
            model_receipt_sha256=self._model_receipt_sha256,
            native_schedule_sha256=self._native_schedule_sha256,
            reference_encoder_sha256=self._reference_encoder_sha256,
            provenance_seal_sha256=self._provenance_seal_sha256,
            adapter_failed=self._poisoned,
            failure_stage=self._failure_stage,
            raw_transformer_forward_count_verified=self._finalized,
            native_request_execution_verified=self._finalized,
            vendor_apg_execution_verified=self._finalized,
            target_tail_direct_view=(
                self._raw_successes > 0
                and self._target_tail_views == self._raw_successes
            ),
        )

    @property
    def diagnostics(self) -> NativeFieldDiagnostics:
        if self._finalized:
            assert self._final_diagnostics is not None
            return self._final_diagnostics
        return self._build_diagnostics()


__all__ = [
    "APG_ETA",
    "APG_MOMENTUM",
    "APG_NORM_THRESHOLD",
    "GUIDANCE_SCALE",
    "IMAGE_GUIDANCE_SCALE",
    "EXPECTED_REQUEST_SCHEMA_VERSION",
    "NativeFieldDiagnostics",
    "NativeFieldProvenance",
    "NativeSourceStateFieldAdapter",
    "REGISTERED_CANDIDATE_SCHEDULES",
    "REGISTERED_K1_SCHEDULE",
    "REGISTERED_K5_EARLY_SCHEDULE",
    "REGISTERED_REGIMES",
    "SAICNativeSourceStateFieldError",
    "SCHEMA_VERSION",
    "VENDOR_APG_MODULE",
    "native_schedule_sha256",
]
