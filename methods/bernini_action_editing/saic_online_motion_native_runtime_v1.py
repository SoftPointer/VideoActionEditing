#!/usr/bin/env python3
"""Bind SAIC's online motion field to Bernini's real RV2V sample seam.

The pinned Bernini sampler exposes the required current raw state without
replacing the sampler.  In every denoising cell it calls, in this order::

    transformer.patch_vae_latent(unpacked_noisy_latent, source_id=0)
    diffusion.shared_step(..., uncond_prompt)       # native negative
    diffusion.shared_step(..., action_prompt)       # native action
    scheduler.step(...)

``unpacked_noisy_latent`` is the exact current ``[1,16,21,H,W]`` tensor.  This
one-shot adapter authenticates that vendor call, derives a V/VI
``NativeRV2VBranch`` from the later official packed ``shared_step`` arguments,
and changes the action cell to the following order::

    native negative                         (temporal operator route absent)
    target-only T2V action teacher          (temporal operator route absent)
    target-only T2V no-op teacher           (temporal operator route absent)
    build_online_motion_field(action, noop)
    native RV2V action under operator route (the only routed forward)
    original scheduler.step exactly once

The teacher queries reuse the native target token/rotary suffix and native
expanded timestep.  Their packed 64-channel velocities are unpacked back to
the exact raw-state geometry before the fixed 21x32 online representation is
built.  The native public timestep is int64; the online-field/operator
primitive requires a device-local FP32 coordinate, so the adapter derives one
exact-valued FP32 scalar after authenticating the int64 coordinate against the
pinned schedule.  No caller can provide a mask, branch, phase code, schedule
index, sigma, SP rank, or action ID.

The implementation follows and reuses the already GPU-exercised reversible
``NativeRV2VActionFieldPatch`` seam and the pinned scheduler/class audits in
``t_qmosaic_bernini_unipc_runtime_adapter_v1``.  It owns neither a sampler nor
a scheduler and is inference-only.  A successful receipt is engineering
evidence, not semantic action-editing evidence or training authority.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import importlib
import inspect
import json
import math
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Optional, Sequence

import torch

if __package__:
    from . import inference_sigma_strata as sigma_strata
    from . import saic_online_motion_field_v1 as online_motion
    from . import saic_temporal_action_operator_v2 as temporal_operator
    from . import self_guided_action_field_v1 as sgaf
    from . import source_self_native_ref_contrastive_v3 as native_pack
else:  # Direct import from methods/bernini_action_editing.
    import inference_sigma_strata as sigma_strata
    import saic_online_motion_field_v1 as online_motion
    import saic_temporal_action_operator_v2 as temporal_operator
    import self_guided_action_field_v1 as sgaf
    import source_self_native_ref_contrastive_v3 as native_pack


SCHEMA_VERSION = "bernini-saic-online-motion-native-runtime-v1"
CLASSIFICATION = "native_inference_route_primitive/no_training_authority"
EXPECTED_FRAMES = 81
EXPECTED_LATENT_PHASES = 21
EXPECTED_LATENT_CHANNELS = 16
EXPECTED_STEPS = 40
EXPECTED_FLOW_SHIFT = 5.0
EXPECTED_GUIDANCE_MODE = "v2v_apg"
EXPECTED_MODEL_ID = "transformer_1"
EXPECTED_HIDDEN_DIM = 1536
EXPECTED_TEXT_TOKENS = 512
EXPECTED_TEXT_DIM = 4096
PACK_SPATIAL = 2
PACKED_VELOCITY_CHANNELS = EXPECTED_LATENT_CHANNELS * PACK_SPATIAL**2
ALLOWED_REFERENCE_COUNTS = frozenset({0, native_pack.REFERENCE_COUNT})

# Narrow import closure from t_qmosaic_bernini_unipc_runtime_adapter_v1.
# Importing that full replay adapter also imports its trajectory intervention,
# whose module-level Python 3.10 ``zip(strict=True)`` construction is unrelated
# to these three read-only Bernini audits and breaks the supported local
# Python 3.8 audit host before a test can start.  Keep the values, normalization
# rules, and digest bit-identical to that pinned adapter without importing its
# mutation-bearing replay stack.
TQMOSAIC_AUDIT_SOURCE_SCHEMA = "bernini-t-qmosaic-unipc-runtime-adapter-v1"
PINNED_BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
PINNED_WAN_DIFFUSION_SHA256 = (
    "59e860ba3490a83f06bd4be75697490f49a118ee5ca969e85eea4dd7fa122512"
)
PINNED_DIFFUSION_CLASS = ("bernini.models.wan_diffusion", "GEN_Wanx22")
PINNED_SCHEDULER_CLASS = (
    "diffusers.schedulers.scheduling_unipc_multistep",
    "UniPCMultistepScheduler",
)
PINNED_STEP_PARAMETER_NAMES = (
    "model_output",
    "timestep",
    "sample",
    "return_dict",
)
PINNED_SCHEDULER_CONFIG = MappingProxyType(
    {
        "_class_name": "UniPCMultistepScheduler",
        "_diffusers_version": "0.33.0.dev0",
        "beta_end": 0.02,
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "disable_corrector": (),
        "dynamic_thresholding_ratio": 0.995,
        "final_sigmas_type": "zero",
        "flow_shift": 5.0,
        "lower_order_final": True,
        "num_train_timesteps": 1000,
        "predict_x0": True,
        "prediction_type": "flow_prediction",
        "rescale_betas_zero_snr": False,
        "sample_max_value": 1.0,
        "shift_terminal": None,
        "sigma_max": None,
        "sigma_min": None,
        "solver_order": 2,
        "solver_p": None,
        "solver_type": "bh2",
        "steps_offset": 0,
        "thresholding": False,
        "time_shift_type": "exponential",
        "timestep_spacing": "linspace",
        "trained_betas": None,
        "use_beta_sigmas": False,
        "use_dynamic_shifting": False,
        "use_exponential_sigmas": False,
        "use_flow_sigmas": True,
        "use_karras_sigmas": False,
    }
)
PINNED_SCHEDULER_CONFIG_DIGEST = (
    "376b2bc18f8801411e1a7bf7005c4734a9cbf52565b1a1d84fd3a86e34c6e595"
)


class SAICOnlineMotionNativeRuntimeError(RuntimeError):
    """The authenticated Bernini online-motion call graph differed."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise SAICOnlineMotionNativeRuntimeError(
            "runtime receipt is not canonical finite ASCII JSON"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _normalized_scheduler_config_value(value: Any, *, label: str) -> Any:
    """Exact narrow mirror of the pinned t-QMoSAIC config normalizer."""

    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise SAICOnlineMotionNativeRuntimeError(
                f"scheduler config {label} is non-finite"
            )
        return value
    if isinstance(value, (list, tuple)):
        return [
            _normalized_scheduler_config_value(
                item, label=f"{label}[{index}]"
            )
            for index, item in enumerate(value)
        ]
    raise SAICOnlineMotionNativeRuntimeError(
        f"scheduler config {label} has unsupported type {type(value).__name__}"
    )


def _scheduler_config_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        if name not in config:
            raise SAICOnlineMotionNativeRuntimeError(
                f"scheduler config is missing required field {name}"
            )
        return config[name]
    if not hasattr(config, name):
        raise SAICOnlineMotionNativeRuntimeError(
            f"scheduler config is missing required field {name}"
        )
    return getattr(config, name)


def _expected_scheduler_config_snapshot() -> dict[str, Any]:
    return {
        name: _normalized_scheduler_config_value(value, label=name)
        for name, value in PINNED_SCHEDULER_CONFIG.items()
    }


if _object_sha256(_expected_scheduler_config_snapshot()) != PINNED_SCHEDULER_CONFIG_DIGEST:
    raise RuntimeError("pinned Bernini UniPC config constants differ from their hash")


def _audit_scheduler_config(scheduler: Any) -> dict[str, Any]:
    """Closed config audit extracted from the pinned t-QMoSAIC adapter."""

    config = getattr(scheduler, "config", None)
    if config is None:
        raise SAICOnlineMotionNativeRuntimeError(
            "UniPC scheduler must expose config"
        )
    expected = _expected_scheduler_config_snapshot()
    observed: dict[str, Any] = {}
    for name, expected_value in expected.items():
        value = _normalized_scheduler_config_value(
            _scheduler_config_value(config, name), label=name
        )
        if type(expected_value) is bool:
            matches = value is expected_value
        elif type(expected_value) is int:
            matches = type(value) is int and value == expected_value
        elif type(expected_value) is float:
            matches = type(value) in (int, float) and float(value) == expected_value
        else:
            matches = value == expected_value
        if not matches:
            raise SAICOnlineMotionNativeRuntimeError(
                f"scheduler config {name} differs: expected {expected_value!r}, "
                f"got {value!r}"
            )
        observed[name] = value
    if _object_sha256(observed) != PINNED_SCHEDULER_CONFIG_DIGEST:
        raise SAICOnlineMotionNativeRuntimeError(
            "scheduler config digest differs"
        )
    return observed


def _audit_step_signature(step: Any) -> dict[str, Any]:
    """Exact narrow mirror of the pinned t-QMoSAIC step audit."""

    try:
        parameters = tuple(inspect.signature(step).parameters.values())
    except (TypeError, ValueError) as error:
        raise SAICOnlineMotionNativeRuntimeError(
            "cannot inspect original UniPC step signature"
        ) from error
    if tuple(parameter.name for parameter in parameters) != PINNED_STEP_PARAMETER_NAMES:
        raise SAICOnlineMotionNativeRuntimeError(
            "original UniPC step parameter names differ"
        )
    positional_kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    if any(parameter.kind not in positional_kinds for parameter in parameters[:3]):
        raise SAICOnlineMotionNativeRuntimeError(
            "original UniPC tensor arguments are not positional"
        )
    final = parameters[3]
    if (
        final.kind not in positional_kinds
        or final.default is not True
        or any(
            parameter.default is not inspect.Parameter.empty
            for parameter in parameters[:3]
        )
    ):
        raise SAICOnlineMotionNativeRuntimeError(
            "original UniPC return_dict/default signature differs"
        )
    return {
        "parameter_names": list(PINNED_STEP_PARAMETER_NAMES),
        "three_required_tensor_arguments": True,
        "return_dict_default": True,
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    if type(value) is not torch.Tensor:
        raise SAICOnlineMotionNativeRuntimeError("tensor digest input differs")
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    digest = hashlib.sha256()
    metadata = {
        "dtype": str(value.dtype),
        "shape": list(map(int, value.shape)),
    }
    digest.update(_canonical_json(metadata))
    try:
        payload = raw.numpy().tobytes(order="C")
    except RuntimeError:
        # Some lightweight audit hosts deliberately have Torch without a
        # working NumPy ABI.  The uint8 value stream is byte-identical and
        # keeps the receipt codec independent of that optional bridge.
        payload = bytes(raw.tolist())
    digest.update(payload)
    return digest.hexdigest()


def _flatten_bound_arguments(
    callable_object: Any, args: Sequence[Any], kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind a pinned call and flatten its sole ``**kwargs`` payload."""

    try:
        signature = inspect.signature(callable_object)
        bound = signature.bind(*args, **dict(kwargs))
        bound.apply_defaults()
    except (TypeError, ValueError) as error:
        raise SAICOnlineMotionNativeRuntimeError(
            "call does not match the pinned Bernini signature"
        ) from error
    values = dict(bound.arguments)
    for parameter in signature.parameters.values():
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD:
            continue
        extras = values.pop(parameter.name, {})
        if not isinstance(extras, Mapping):
            raise SAICOnlineMotionNativeRuntimeError(
                "variadic keyword payload differs"
            )
        for name, value in extras.items():
            if name in values:
                raise SAICOnlineMotionNativeRuntimeError(
                    f"duplicate variadic keyword {name}"
                )
            values[name] = value
    return values


def _replace_argument(
    callable_object: Any,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    name: str,
    value: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Reuse SGAF's audited positional/keyword replacement primitive."""

    try:
        return sgaf._replace_argument(  # noqa: SLF001 - shared native seam
            callable_object, args, kwargs, name=name, value=value
        )
    except sgaf.SelfGuidedActionFieldError as error:
        raise SAICOnlineMotionNativeRuntimeError(str(error)) from error


def _shape(value: Any, *, label: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.shape)
    except Exception as error:
        raise SAICOnlineMotionNativeRuntimeError(
            f"{label} must expose an integer tensor shape"
        ) from error


def _detached_finite_tensor(value: Any, *, label: str) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.device.type == "meta"
        or value.layout != torch.strided
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise SAICOnlineMotionNativeRuntimeError(
            f"{label} must be a detached finite materialized torch tensor"
        )
    return value


def _plain_scalar(value: Any, *, label: str) -> float:
    try:
        candidate = value.detach() if hasattr(value, "detach") else value
        if hasattr(candidate, "numel") and int(candidate.numel()) != 1:
            raise SAICOnlineMotionNativeRuntimeError(f"{label} must be scalar")
        if hasattr(candidate, "cpu"):
            candidate = candidate.cpu()
        if hasattr(candidate, "item"):
            candidate = candidate.item()
        result = float(candidate)
    except SAICOnlineMotionNativeRuntimeError:
        raise
    except Exception as error:
        raise SAICOnlineMotionNativeRuntimeError(f"{label} must be scalar") from error
    if not math.isfinite(result):
        raise SAICOnlineMotionNativeRuntimeError(f"{label} must be finite")
    return result


def _plain_schedule_index(timestep: torch.Tensor, *, expected: int) -> int:
    tensor = _detached_finite_tensor(timestep, label="native shared timestep")
    if tensor.shape != (1,) or tensor.dtype != torch.int64:
        raise SAICOnlineMotionNativeRuntimeError(
            "native shared timestep must be the public int64 expand(1) view"
        )
    numeric = _plain_scalar(tensor, label="native shared timestep")
    if numeric != float(int(numeric)):
        raise SAICOnlineMotionNativeRuntimeError(
            "native shared timestep is not integer-valued"
        )
    matches = [
        index
        for index, registered in enumerate(sigma_strata.PINNED_TIMESTEPS)
        if registered == int(numeric)
    ]
    if matches != [expected]:
        raise SAICOnlineMotionNativeRuntimeError(
            f"native exact40 coordinate differs: expected {expected}, got {matches}"
        )
    return expected


def _metadata_tuple(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or any(
        type(item) is not int or item <= 0 for item in value
    ):
        raise SAICOnlineMotionNativeRuntimeError(
            f"{label} must be positive integer list/tuple metadata"
        )
    return tuple(value)


def _natural_caption(value: Any, *, label: str) -> str:
    try:
        return online_motion._validate_natural_caption(value, label=label)  # noqa: SLF001
    except online_motion.SAICOnlineMotionFieldError as error:
        raise SAICOnlineMotionNativeRuntimeError(str(error)) from error


def _packed_to_spatial(
    value: torch.Tensor, *, like: torch.Tensor, label: str
) -> torch.Tensor:
    """Invert Bernini's exact ``_to_packed(..., ph=2, pw=2)`` layout."""

    packed = _detached_finite_tensor(value, label=label)
    state = _detached_finite_tensor(like, label="current raw target")
    batch, channels, phases, height, width = map(int, state.shape)
    if height % PACK_SPATIAL or width % PACK_SPATIAL:
        raise SAICOnlineMotionNativeRuntimeError(
            "current raw target spatial geometry is not patch divisible"
        )
    patch_h, patch_w = height // PACK_SPATIAL, width // PACK_SPATIAL
    expected = (
        batch,
        phases * patch_h * patch_w,
        PACK_SPATIAL * PACK_SPATIAL * channels,
    )
    if tuple(map(int, packed.shape)) != expected:
        raise SAICOnlineMotionNativeRuntimeError(
            f"{label} differs from exact Bernini packed velocity geometry"
        )
    result = (
        packed.reshape(
            batch,
            phases,
            patch_h,
            patch_w,
            PACK_SPATIAL,
            PACK_SPATIAL,
            channels,
        )
        .permute(0, 6, 1, 2, 4, 3, 5)
        .reshape(batch, channels, phases, height, width)
        .contiguous()
    )
    if not bool(torch.isfinite(result).all().item()):
        raise SAICOnlineMotionNativeRuntimeError(
            f"{label} became non-finite while unpacking"
        )
    return result.detach()


@dataclass(frozen=True)
class _TargetPatch:
    raw_state: torch.Tensor = field(repr=False, compare=False)
    packed_tokens: torch.Tensor = field(repr=False, compare=False)
    rotary: torch.Tensor = field(repr=False, compare=False)


@dataclass(frozen=True)
class _NativeForward:
    role: str
    noisy_latents: torch.Tensor = field(repr=False, compare=False)
    timestep: torch.Tensor = field(repr=False, compare=False)
    rotary: torch.Tensor = field(repr=False, compare=False)
    batch_vae_seqlen: tuple[int, ...]


@dataclass
class _ActiveSample:
    action_embed: torch.Tensor = field(repr=False)
    negative_embed: torch.Tensor = field(repr=False)
    reference_count: int
    expected_patch_source_ids: tuple[float, ...]
    completed_steps: int = 0
    patch_source_ids: list[float] = field(default_factory=list)
    target_patch: Optional[_TargetPatch] = None
    pending_negative: Optional[_NativeForward] = None
    pending_action: Optional[_NativeForward] = None
    teacher_target_tokens: Optional[torch.Tensor] = None
    teacher_target_rotary: Optional[torch.Tensor] = None
    pending_motion_field: Optional[online_motion.SAICOnlineMotionField] = None
    pending_route_receipt: Optional[Mapping[str, Any]] = None


class SAICOnlineMotionNativeRuntimeV1:
    """One-shot official Bernini sample/shared-step temporal-operator adapter."""

    _MARKER = "_bernini_saic_online_motion_native_runtime_v1"

    def __init__(
        self,
        diffusion: Any,
        *,
        action_handle: temporal_operator.SAICTemporalActionOperatorHandle,
        action_prompt: str,
        noop_prompt: str,
        action_t2v_embeds: torch.Tensor,
        noop_t2v_embeds: torch.Tensor,
    ) -> None:
        self.action_prompt = _natural_caption(action_prompt, label="action_prompt")
        self.noop_prompt = _natural_caption(noop_prompt, label="noop_prompt")
        if self.action_prompt == self.noop_prompt:
            raise SAICOnlineMotionNativeRuntimeError(
                "action and no-op natural captions must differ for action editing"
            )
        teacher_embeds: list[torch.Tensor] = []
        for label, value in (
            ("action T2V prompt embedding", action_t2v_embeds),
            ("no-op T2V prompt embedding", noop_t2v_embeds),
        ):
            embed = _detached_finite_tensor(value, label=label)
            if tuple(map(int, embed.shape)) != (
                1,
                EXPECTED_TEXT_TOKENS,
                EXPECTED_TEXT_DIM,
            ):
                raise SAICOnlineMotionNativeRuntimeError(
                    f"{label} differs from Bernini text geometry"
                )
            teacher_embeds.append(embed)
        action_t2v_embed, noop_embed = teacher_embeds
        if action_t2v_embed is noop_embed or action_t2v_embed.device != noop_embed.device:
            raise SAICOnlineMotionNativeRuntimeError(
                "action/no-op T2V prompt objects must be distinct on one device"
            )
        observed_diffusion_class = (type(diffusion).__module__, type(diffusion).__name__)
        if observed_diffusion_class != PINNED_DIFFUSION_CLASS:
            raise SAICOnlineMotionNativeRuntimeError(
                "diffusion class is not pinned bernini.models.wan_diffusion.GEN_Wanx22"
            )
        scheduler = getattr(diffusion, "scheduler", None)
        observed_scheduler_class = (type(scheduler).__module__, type(scheduler).__name__)
        if observed_scheduler_class != PINNED_SCHEDULER_CLASS:
            raise SAICOnlineMotionNativeRuntimeError(
                "scheduler class is not pinned Diffusers UniPC"
            )
        if getattr(diffusion, "use_unipc", None) is not True:
            raise SAICOnlineMotionNativeRuntimeError("native runtime requires UniPC")
        if getattr(diffusion, "transformer_2", None) is not None:
            raise SAICOnlineMotionNativeRuntimeError(
                "native runtime is pinned to single-expert Bernini-R 1.3B"
            )
        if _plain_scalar(
            getattr(diffusion, "switch_dit_boundary", None),
            label="switch_dit_boundary",
        ) != 0.0:
            raise SAICOnlineMotionNativeRuntimeError(
                "single-expert Bernini-R requires switch_dit_boundary=0"
            )
        transformer = getattr(diffusion, "transformer", None)
        if (
            type(action_handle) is not temporal_operator.SAICTemporalActionOperatorHandle
            or action_handle.transformer is not transformer
            or action_handle.restored
        ):
            raise SAICOnlineMotionNativeRuntimeError(
                "action_handle is not the live SAIC temporal operator on this transformer"
            )
        if not action_handle.base_parameters_frozen() or not action_handle.scope_untouched():
            raise SAICOnlineMotionNativeRuntimeError(
                "temporal operator/base freeze scope differs"
            )
        if temporal_operator.active_route() is not None:
            raise SAICOnlineMotionNativeRuntimeError(
                "a temporal action route is already active"
            )
        transformer_config = getattr(transformer, "config", None)

        def config_value(name: str) -> Any:
            if isinstance(transformer_config, Mapping):
                return transformer_config.get(name)
            return getattr(transformer_config, name, None)

        heads = config_value("num_attention_heads")
        head_dim = config_value("attention_head_dim")
        if (
            type(heads) is not int
            or type(head_dim) is not int
            or heads * head_dim != EXPECTED_HIDDEN_DIM
            or config_value("in_channels") != EXPECTED_LATENT_CHANNELS
            or config_value("out_channels") != EXPECTED_LATENT_CHANNELS
            or tuple(config_value("patch_size") or ()) != (1, 2, 2)
            or config_value("text_dim") != EXPECTED_TEXT_DIM
        ):
            raise SAICOnlineMotionNativeRuntimeError(
                "transformer is not the pinned Bernini-R 1.3B geometry"
            )
        originals = {
            "sample": getattr(diffusion, "sample", None),
            "shared_step": getattr(diffusion, "shared_step", None),
            "patch_vae_latent": getattr(transformer, "patch_vae_latent", None),
            "scheduler.step": getattr(scheduler, "step", None),
        }
        if any(not callable(value) for value in originals.values()):
            raise SAICOnlineMotionNativeRuntimeError(
                "pinned Bernini sampler call surface differs"
            )
        try:
            instance_overrides = {
                "sample": "sample" in vars(diffusion),
                "shared_step": "shared_step" in vars(diffusion),
                "patch_vae_latent": "patch_vae_latent" in vars(transformer),
                "scheduler.step": "step" in vars(scheduler),
            }
        except TypeError as error:
            raise SAICOnlineMotionNativeRuntimeError(
                "cannot inspect sampler instance overrides"
            ) from error
        # The late source anchor is deliberately the one allowed inner
        # wrapper.  It routes only full-source calls and directly passes the
        # target-only teachers.  Require one exact marker object across all
        # three sampler surfaces; partial/arbitrary wrapper stacking remains
        # forbidden.  patch_vae_latent itself must still be the vendor method.
        inner_markers = (
            getattr(originals["sample"], "_bernini_saic_source_anchor_native_runtime_v1", None),
            getattr(originals["shared_step"], "_bernini_saic_source_anchor_native_runtime_v1", None),
            getattr(originals["scheduler.step"], "_bernini_saic_source_anchor_native_runtime_v1", None),
        )
        source_anchor_inner = None
        if any(marker is not None for marker in inner_markers):
            if (
                any(marker is None for marker in inner_markers)
                or len({id(marker) for marker in inner_markers}) != 1
                or not all(
                    instance_overrides[name]
                    for name in ("sample", "shared_step", "scheduler.step")
                )
                or instance_overrides["patch_vae_latent"]
            ):
                raise SAICOnlineMotionNativeRuntimeError(
                    "partial or ambiguous source-anchor inner runtime"
                )
            source_anchor_inner = inner_markers[0]
            module_name = (
                f"{__package__}.saic_source_anchor_native_runtime_v1"
                if __package__
                else "saic_source_anchor_native_runtime_v1"
            )
            try:
                source_runtime_module = importlib.import_module(module_name)
                source_runtime_type = getattr(
                    source_runtime_module,
                    "SAICSourceAnchorNativeRuntimePatch",
                )
            except (ImportError, AttributeError) as error:
                raise SAICOnlineMotionNativeRuntimeError(
                    "cannot authenticate source-anchor inner runtime class"
                ) from error
            if (
                type(source_anchor_inner) is not source_runtime_type
                or getattr(source_anchor_inner, "diffusion", None) is not diffusion
                or getattr(source_anchor_inner, "transformer", None) is not transformer
                or getattr(source_anchor_inner, "scheduler", None) is not scheduler
                or getattr(source_anchor_inner, "installed", None) is not True
                or getattr(source_anchor_inner, "restored", None) is not False
                or not callable(getattr(source_anchor_inner, "original_sample", None))
                or not callable(
                    getattr(source_anchor_inner, "original_shared_step", None)
                )
                or not callable(
                    getattr(source_anchor_inner, "original_scheduler_step", None)
                )
            ):
                raise SAICOnlineMotionNativeRuntimeError(
                    "source-anchor inner runtime identity/lifecycle differs"
                )
        elif any(instance_overrides.values()):
            raise SAICOnlineMotionNativeRuntimeError(
                "refusing non-source-anchor stacked instance override"
            )
        if any(getattr(value, self._MARKER, None) is not None for value in originals.values()):
            raise SAICOnlineMotionNativeRuntimeError(
                "online-motion native wrapper is already installed"
            )
        scheduler_config = _audit_scheduler_config(scheduler)
        step_signature = _audit_step_signature(
            (
                source_anchor_inner.original_scheduler_step
                if source_anchor_inner is not None
                else originals["scheduler.step"]
            )
        )

        self.diffusion = diffusion
        self.transformer = transformer
        self.scheduler = scheduler
        self.action_handle = action_handle
        self.action_t2v_embeds = action_t2v_embed
        self.noop_t2v_embeds = noop_embed
        self.original_sample = originals["sample"]
        self.original_shared_step = originals["shared_step"]
        self.original_patch_vae_latent = originals["patch_vae_latent"]
        self.original_scheduler_step = originals["scheduler.step"]
        self.vendor_sample_signature = (
            source_anchor_inner.original_sample
            if source_anchor_inner is not None
            else originals["sample"]
        )
        self.vendor_shared_step_signature = (
            source_anchor_inner.original_shared_step
            if source_anchor_inner is not None
            else originals["shared_step"]
        )
        self.source_anchor_inner = source_anchor_inner
        self.scheduler_config = scheduler_config
        self.scheduler_step_signature = step_signature
        self._patches: list[tuple[Any, str, bool, Any, Any]] = []
        self._active: Optional[_ActiveSample] = None
        self.installed = False
        self.restored = False
        self.finalized = False
        self.schedule_audit: Optional[Mapping[str, Any]] = None
        self.sample_calls = 0
        self.successful_sample_calls = 0
        self.official_negative_calls = 0
        self.official_action_calls = 0
        self.teacher_action_calls = 0
        self.teacher_noop_calls = 0
        self.original_scheduler_calls = 0
        self.trace: list[dict[str, Any]] = []
        self.sample_contract: Optional[dict[str, Any]] = None
        self.output_sha256: Optional[str] = None

    def _set_patch(self, owner: Any, name: str, value: Any) -> None:
        try:
            instance = vars(owner)
        except TypeError as error:
            raise SAICOnlineMotionNativeRuntimeError(
                f"cannot reversibly patch {name} owner"
            ) from error
        had_instance = name in instance
        previous = instance.get(name)
        resolved_before = getattr(owner, name)
        setattr(owner, name, value)
        self._patches.append((owner, name, had_instance, previous, resolved_before))

    def install(self) -> None:
        if self.installed or self.restored or self.finalized:
            raise SAICOnlineMotionNativeRuntimeError("native runtime is one-shot")

        def sample_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_sample(*args, **kwargs)

        def shared_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_shared_step(*args, **kwargs)

        def patch_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_patch_vae_latent(*args, **kwargs)

        def scheduler_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_scheduler_step(*args, **kwargs)

        for wrapper in (sample_wrapper, shared_wrapper, patch_wrapper, scheduler_wrapper):
            setattr(wrapper, self._MARKER, self)
        try:
            self._set_patch(self.scheduler, "step", scheduler_wrapper)
            self._set_patch(self.transformer, "patch_vae_latent", patch_wrapper)
            self._set_patch(self.diffusion, "shared_step", shared_wrapper)
            self._set_patch(self.diffusion, "sample", sample_wrapper)
        except Exception:
            self._restore_patches(require_wrapper_identity=False)
            raise
        self.installed = True

    def _restore_patches(self, *, require_wrapper_identity: bool) -> None:
        errors: list[Exception] = []
        while self._patches:
            owner, name, had_instance, previous, resolved_before = self._patches.pop()
            try:
                current = getattr(owner, name, None)
                if require_wrapper_identity and getattr(current, self._MARKER, None) is not self:
                    errors.append(
                        SAICOnlineMotionNativeRuntimeError(
                            f"{name} changed while native runtime was active"
                        )
                    )
                if had_instance:
                    setattr(owner, name, previous)
                else:
                    delattr(owner, name)
                if getattr(owner, name, None) != resolved_before:
                    errors.append(
                        SAICOnlineMotionNativeRuntimeError(f"{name} restoration failed")
                    )
            except Exception as error:
                errors.append(error)
        self._active = None
        if errors:
            raise SAICOnlineMotionNativeRuntimeError(
                f"failed to restore {len(errors)} native runtime wrapper(s)"
            ) from errors[0]

    def restore(self) -> None:
        if not self.installed or self.restored:
            raise SAICOnlineMotionNativeRuntimeError("native runtime restore differs")
        try:
            self._restore_patches(require_wrapper_identity=True)
        finally:
            self.installed = False
            self.restored = not self._patches

    def _validate_prompt_embed(self, value: Any, *, label: str) -> torch.Tensor:
        tensor = _detached_finite_tensor(value, label=label)
        if tuple(map(int, tensor.shape)) != (
            1,
            EXPECTED_TEXT_TOKENS,
            EXPECTED_TEXT_DIM,
        ):
            raise SAICOnlineMotionNativeRuntimeError(
                f"{label} differs from Bernini text geometry"
            )
        return tensor

    def _validate_sample_contract(self, values: Mapping[str, Any]) -> _ActiveSample:
        if (
            values.get("guidance_mode") != EXPECTED_GUIDANCE_MODE
            or values.get("num_frames") != EXPECTED_FRAMES
            or values.get("num_inference_steps") != EXPECTED_STEPS
            or _plain_scalar(values.get("flow_shift"), label="flow_shift")
            != EXPECTED_FLOW_SHIFT
            or values.get("prompt_embeds_t2") is not None
            or values.get("uncond_embeds_t2") is not None
            or values.get("image_vae_latents") is not None
        ):
            raise SAICOnlineMotionNativeRuntimeError(
                "official exact81/exact40 v2v_apg sample contract differs"
            )
        width, height = values.get("width"), values.get("height")
        if (
            type(width) is not int
            or type(height) is not int
            or width <= 0
            or height <= 0
            or width % 16
            or height % 16
        ):
            raise SAICOnlineMotionNativeRuntimeError(
                "decoded width/height must be positive multiples of 16"
            )
        latent_shape = (
            1,
            EXPECTED_LATENT_CHANNELS,
            EXPECTED_LATENT_PHASES,
            height // 8,
            width // 8,
        )
        videos = values.get("multi_video_vae_latents")
        if type(videos) is torch.Tensor:
            videos = (videos,)
        if not isinstance(videos, (list, tuple)) or len(videos) != 1:
            raise SAICOnlineMotionNativeRuntimeError(
                "native online motion requires exactly one source-video latent"
            )
        source = _detached_finite_tensor(videos[0], label="source-video latent")
        if tuple(map(int, source.shape)) != latent_shape:
            raise SAICOnlineMotionNativeRuntimeError(
                "source-video latent differs from exact81 target geometry"
            )
        references = values.get("multi_image_vae_latents")
        refs = () if references is None else tuple(references)
        if len(refs) not in ALLOWED_REFERENCE_COUNTS:
            raise SAICOnlineMotionNativeRuntimeError(
                "native route supports V-only or exactly four source references"
            )
        for index, reference in enumerate(refs):
            tensor = _detached_finite_tensor(reference, label=f"reference {index}")
            if tuple(map(int, tensor.shape)) != (
                1,
                EXPECTED_LATENT_CHANNELS,
                1,
                height // 8,
                width // 8,
            ):
                raise SAICOnlineMotionNativeRuntimeError(
                    f"reference {index} differs from one-phase native geometry"
                )
        action = self._validate_prompt_embed(values.get("prompt_embeds"), label="action")
        negative = self._validate_prompt_embed(
            values.get("uncond_prompt_embeds"), label="negative"
        )
        if (
            action is negative
            or action is self.action_t2v_embeds
            or action is self.noop_t2v_embeds
            or negative is self.action_t2v_embeds
            or negative is self.noop_t2v_embeds
            or self.action_t2v_embeds is self.noop_t2v_embeds
            or action.device != negative.device
            or action.device != self.action_t2v_embeds.device
            or action.device != self.noop_t2v_embeds.device
        ):
            raise SAICOnlineMotionNativeRuntimeError(
                "native action/negative/no-op prompt object or device binding differs"
            )
        expected_patch_ids = (
            native_pack.VI_VIDEO_SOURCE_IDS + (0.0,)
            if not refs
            else native_pack.PATCH_CALL_SOURCE_IDS
        )
        self.sample_contract = {
            "num_frames": EXPECTED_FRAMES,
            "latent_phases": EXPECTED_LATENT_PHASES,
            "num_inference_steps": EXPECTED_STEPS,
            "flow_shift": EXPECTED_FLOW_SHIFT,
            "guidance_mode": EXPECTED_GUIDANCE_MODE,
            "source_video_count": 1,
            "source_reference_count": len(refs),
            "native_branch": "V" if not refs else "VI",
            "latent_shape": list(latent_shape),
            "action_embed_sha256": _tensor_sha256(action),
            "negative_embed_sha256": _tensor_sha256(negative),
            "action_t2v_embed_sha256": _tensor_sha256(self.action_t2v_embeds),
            "noop_embed_sha256": _tensor_sha256(self.noop_t2v_embeds),
            "action_prompt_sha256": hashlib.sha256(
                self.action_prompt.encode("utf-8")
            ).hexdigest(),
            "noop_prompt_sha256": hashlib.sha256(
                self.noop_prompt.encode("utf-8")
            ).hexdigest(),
        }
        return _ActiveSample(
            action_embed=action,
            negative_embed=negative,
            reference_count=len(refs),
            expected_patch_source_ids=tuple(map(float, expected_patch_ids)),
        )

    def _wrapped_sample(self, *args: Any, **kwargs: Any) -> Any:
        if (
            self._active is not None
            or self.sample_calls != 0
            or temporal_operator.active_route() is not None
        ):
            raise SAICOnlineMotionNativeRuntimeError(
                "native runtime permits exactly one non-nested sample"
            )
        if (
            self.diffusion.scheduler is not self.scheduler
            or self.action_handle.transformer is not self.transformer
            or self.action_handle.restored
        ):
            raise SAICOnlineMotionNativeRuntimeError("sample ownership changed")
        values = _flatten_bound_arguments(self.vendor_sample_signature, args, kwargs)
        state = self._validate_sample_contract(values)
        self._active = state
        self.sample_calls += 1
        try:
            result = self.original_sample(*args, **kwargs)
            if (
                state.completed_steps != EXPECTED_STEPS
                or state.patch_source_ids
                or state.target_patch is not None
                or state.pending_negative is not None
                or state.pending_action is not None
                or state.teacher_target_tokens is not None
                or state.teacher_target_rotary is not None
                or state.pending_motion_field is not None
                or len(self.trace) != EXPECTED_STEPS
            ):
                raise SAICOnlineMotionNativeRuntimeError(
                    "sample returned without exact40 native cell closure"
                )
            output = _detached_finite_tensor(result, label="native sample output")
            expected_shape = tuple(self.sample_contract["latent_shape"])
            if tuple(map(int, output.shape)) != expected_shape:
                raise SAICOnlineMotionNativeRuntimeError(
                    "native sample output differs from exact81 geometry"
                )
            self.output_sha256 = _tensor_sha256(output)
            self.successful_sample_calls += 1
            return result
        finally:
            self._active = None

    def _wrapped_patch_vae_latent(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise SAICOnlineMotionNativeRuntimeError(
                "patch_vae_latent ran outside the authenticated sample"
            )
        if state.pending_negative is not None or state.pending_action is not None:
            raise SAICOnlineMotionNativeRuntimeError(
                "patch_vae_latent ran after native forwards began"
            )
        values = _flatten_bound_arguments(
            self.original_patch_vae_latent, args, kwargs
        )
        raw = _detached_finite_tensor(
            values.get("hidden_states"), label="patch_vae_latent hidden_states"
        )
        source_id = _plain_scalar(values.get("source_id"), label="source_id")
        if len(state.patch_source_ids) >= len(state.expected_patch_source_ids):
            raise SAICOnlineMotionNativeRuntimeError(
                "too many patch_vae_latent calls before native negative"
            )
        expected_source_id = state.expected_patch_source_ids[len(state.patch_source_ids)]
        if source_id != expected_source_id:
            raise SAICOnlineMotionNativeRuntimeError(
                "vendor patch_vae_latent source-id order differs"
            )
        result = self.original_patch_vae_latent(*args, **kwargs)
        if type(result) is not tuple or len(result) != 2:
            raise SAICOnlineMotionNativeRuntimeError(
                "patch_vae_latent must return one built-in (tokens, rotary) tuple"
            )
        tokens = _detached_finite_tensor(result[0], label="patched VAE tokens")
        rotary = _detached_finite_tensor(result[1], label="patched VAE rotary")
        state.patch_source_ids.append(source_id)
        if source_id == 0.0:
            if (
                state.target_patch is not None
                or tuple(state.patch_source_ids) != state.expected_patch_source_ids
                or tuple(map(int, raw.shape[:3]))
                != (1, EXPECTED_LATENT_CHANNELS, EXPECTED_LATENT_PHASES)
                or raw.ndim != 5
            ):
                raise SAICOnlineMotionNativeRuntimeError(
                    "source_id=0 call is not the unique final exact81 target patch"
                )
            height, width = map(int, raw.shape[-2:])
            expected_tokens = EXPECTED_LATENT_PHASES * (height // 2) * (width // 2)
            if (
                height <= 0
                or width <= 0
                or height % 2
                or width % 2
                or tuple(map(int, tokens.shape))
                != (1, expected_tokens, EXPECTED_HIDDEN_DIM)
                or rotary.ndim != 4
                or tuple(map(int, rotary.shape[:3]))
                != (1, 1, expected_tokens)
            ):
                raise SAICOnlineMotionNativeRuntimeError(
                    "captured native target patch geometry differs"
                )
            state.target_patch = _TargetPatch(raw, tokens, rotary)
        return result

    def _ensure_schedule(self) -> None:
        if self.schedule_audit is not None:
            return
        try:
            audit = sigma_strata.audit_runtime_unipc_schedule(
                self.scheduler, initialize=False
            )
        except Exception as error:
            raise SAICOnlineMotionNativeRuntimeError(
                f"live exact40 UniPC schedule differs: {error}"
            ) from error
        if audit.get("schedule_sha256") != sigma_strata.SCHEDULE_SHA256:
            raise SAICOnlineMotionNativeRuntimeError(
                "live exact40 UniPC schedule digest differs"
            )
        self.schedule_audit = dict(audit)

    def _branch_name(self, state: _ActiveSample) -> str:
        return "V" if state.reference_count == 0 else "VI"

    def _validate_shared_call(
        self,
        values: Mapping[str, Any],
        *,
        state: _ActiveSample,
        role: str,
    ) -> _NativeForward:
        target = state.target_patch
        if (
            target is None
            or tuple(state.patch_source_ids) != state.expected_patch_source_ids
        ):
            raise SAICOnlineMotionNativeRuntimeError(
                "shared_step arrived before the authenticated raw target patch"
            )
        if values.get("model_id") != EXPECTED_MODEL_ID:
            raise SAICOnlineMotionNativeRuntimeError("shared_step model route differs")
        expected_prompt = state.negative_embed if role == "negative" else state.action_embed
        if values.get("cond_embeds") is not expected_prompt:
            raise SAICOnlineMotionNativeRuntimeError(
                f"native {role} prompt is not the exact sample object"
            )
        noisy = _detached_finite_tensor(
            values.get("noisy_latents"), label=f"native {role} packed state"
        )
        timestep = _detached_finite_tensor(
            values.get("timesteps"), label=f"native {role} timestep"
        )
        rotary = _detached_finite_tensor(
            values.get("rotary_embs"), label=f"native {role} rotary"
        )
        _plain_schedule_index(timestep, expected=state.completed_steps)
        target_tokens = int(target.packed_tokens.shape[1])
        patch_positions = target_tokens // EXPECTED_LATENT_PHASES
        condition_tokens = (
            target_tokens
            if state.reference_count == 0
            else target_tokens + native_pack.REFERENCE_COUNT * patch_positions
        )
        total_tokens = condition_tokens + target_tokens
        if (
            tuple(map(int, noisy.shape))
            != (1, total_tokens, EXPECTED_HIDDEN_DIM)
            or rotary.ndim != 4
            or tuple(map(int, rotary.shape[:3])) != (1, 1, total_tokens)
            or _metadata_tuple(
                values.get("batch_vae_seqlen"),
                label=f"native {role} batch_vae_seqlen",
            )
            != (total_tokens,)
            or _metadata_tuple(
                values.get("batch_text_seqlen"),
                label=f"native {role} batch_text_seqlen",
            )
            != (EXPECTED_TEXT_TOKENS,)
        ):
            raise SAICOnlineMotionNativeRuntimeError(
                f"native {role} V/VI packed geometry differs"
            )
        if not torch.equal(noisy[:, -target_tokens:, :], target.packed_tokens):
            raise SAICOnlineMotionNativeRuntimeError(
                "official shared_step target suffix differs from captured target patch"
            )
        if not torch.equal(rotary[:, :, -target_tokens:, :], target.rotary):
            raise SAICOnlineMotionNativeRuntimeError(
                "official shared_step target rotary differs from captured target patch"
            )
        if state.pending_negative is not None:
            first = state.pending_negative
            for label, left, right in (
                ("noisy_latents", first.noisy_latents, noisy),
                ("timesteps", first.timestep, timestep),
                ("rotary_embs", first.rotary, rotary),
            ):
                if left is not right:
                    raise SAICOnlineMotionNativeRuntimeError(
                        f"negative/action {label} is not the same native object"
                    )
            if first.batch_vae_seqlen != (total_tokens,):
                raise SAICOnlineMotionNativeRuntimeError(
                    "negative/action VAE sequence metadata differ"
                )
        return _NativeForward(
            role=role,
            noisy_latents=noisy,
            timestep=timestep,
            rotary=rotary,
            batch_vae_seqlen=(total_tokens,),
        )

    def _validate_shared_result(
        self, value: Any, *, tokens: int, label: str
    ) -> torch.Tensor:
        result = _detached_finite_tensor(value, label=label)
        if tuple(map(int, result.shape)) != (
            1,
            tokens,
            PACKED_VELOCITY_CHANNELS,
        ):
            raise SAICOnlineMotionNativeRuntimeError(
                f"{label} differs from Bernini packed velocity geometry"
            )
        return result

    def _teacher_query(
        self,
        base_args: Sequence[Any],
        base_kwargs: Mapping[str, Any],
        *,
        state: _ActiveSample,
        request: online_motion.FrozenT2VVelocityRequest,
    ) -> torch.Tensor:
        target = state.target_patch
        if target is None:
            raise SAICOnlineMotionNativeRuntimeError("teacher target patch is absent")
        if temporal_operator.active_route() is not None:
            raise SAICOnlineMotionNativeRuntimeError(
                "T2V teacher query attempted under the temporal operator route"
            )
        if request.current_noisy_target is not target.raw_state:
            raise SAICOnlineMotionNativeRuntimeError(
                "online field teacher request changed the current raw target object"
            )
        if request.natural_language_prompt != (
            self.action_prompt if request.branch == "action" else self.noop_prompt
        ):
            raise SAICOnlineMotionNativeRuntimeError(
                "online field teacher natural caption differs"
            )
        expected_sigma = sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[
            state.completed_steps
        ]
        if (
            sigma_strata._float32_hex(  # noqa: SLF001 - pinned bit audit
                request.actual_sigma.item(), label="teacher actual sigma"
            )
            != expected_sigma
        ):
            raise SAICOnlineMotionNativeRuntimeError(
                "online field teacher actual sigma differs"
            )
        prompt = (
            self.action_t2v_embeds
            if request.branch == "action"
            else self.noop_t2v_embeds
        )
        target_tokens_tensor = state.teacher_target_tokens
        target_rotary_tensor = state.teacher_target_rotary
        if target_tokens_tensor is None or target_rotary_tensor is None:
            raise SAICOnlineMotionNativeRuntimeError(
                "official target suffix views were not minted before teacher query"
            )
        target_tokens = int(target_tokens_tensor.shape[1])
        call_args, call_kwargs = tuple(base_args), dict(base_kwargs)
        replacements = {
            "noisy_latents": target_tokens_tensor,
            # The actual teacher forward uses the native int64 expand(1)
            # object.  request.timestep is its authenticated FP32 coordinate
            # used solely by the online-field/operator primitive.
            "timesteps": state.pending_negative.timestep,
            "cond_embeds": prompt,
            "rotary_embs": target_rotary_tensor,
            "batch_vae_seqlen": [target_tokens],
            "batch_text_seqlen": [EXPECTED_TEXT_TOKENS],
        }
        for name, replacement in replacements.items():
            call_args, call_kwargs = _replace_argument(
                self.vendor_shared_step_signature,
                call_args,
                call_kwargs,
                name=name,
                value=replacement,
            )
        bound = _flatten_bound_arguments(
            self.vendor_shared_step_signature, call_args, call_kwargs
        )
        for name, expected_object in (
            ("noisy_latents", target_tokens_tensor),
            ("timesteps", state.pending_negative.timestep),
            ("cond_embeds", prompt),
            ("rotary_embs", target_rotary_tensor),
        ):
            if bound.get(name) is not expected_object:
                raise SAICOnlineMotionNativeRuntimeError(
                    f"teacher {request.branch} {name} object binding differs"
                )
        result = self.original_shared_step(*call_args, **call_kwargs)
        packed = self._validate_shared_result(
            result,
            tokens=target_tokens,
            label=f"T2V {request.branch} teacher output",
        )
        if request.branch == "action":
            self.teacher_action_calls += 1
        else:
            self.teacher_noop_calls += 1
        return _packed_to_spatial(
            packed,
            like=target.raw_state,
            label=f"T2V {request.branch} teacher output",
        )

    def _build_native_branch(
        self, values: Mapping[str, Any], *, state: _ActiveSample
    ) -> native_pack.NativeRV2VBranch:
        target = state.target_patch
        if target is None:
            raise SAICOnlineMotionNativeRuntimeError("native branch target is absent")
        noisy = values["noisy_latents"]
        total_tokens = int(noisy.shape[1])
        target_tokens = int(target.packed_tokens.shape[1])
        condition_tokens = total_tokens - target_tokens
        mask = torch.zeros(total_tokens, dtype=torch.bool, device=noisy.device)
        mask[condition_tokens:] = True
        branch_name = self._branch_name(state)
        source_ids = (
            native_pack.VI_VIDEO_SOURCE_IDS + (0.0,)
            if branch_name == "V"
            else native_pack.VI_VIDEO_SOURCE_IDS
            + native_pack.VI_IMAGE_SOURCE_IDS
            + (0.0,)
        )
        try:
            return native_pack.NativeRV2VBranch(
                name=branch_name,
                latents=noisy,
                rotary=values["rotary_embs"],
                target_mask=mask,
                total_tokens=total_tokens,
                condition_tokens=condition_tokens,
                source_ids=tuple(map(float, source_ids)),
                concat_order=native_pack.BRANCH_CONCAT_ORDER[branch_name],
            )
        except Exception as error:
            raise SAICOnlineMotionNativeRuntimeError(
                f"cannot derive native RV2V branch: {error}"
            ) from error

    def _wrapped_shared_step(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise SAICOnlineMotionNativeRuntimeError(
                "shared_step ran outside the authenticated sample"
            )
        self._ensure_schedule()
        if state.pending_negative is None:
            if state.pending_action is not None:
                raise SAICOnlineMotionNativeRuntimeError(
                    "native forward state is internally inconsistent"
                )
            values = _flatten_bound_arguments(
                self.vendor_shared_step_signature, args, kwargs
            )
            observation = self._validate_shared_call(
                values, state=state, role="negative"
            )
            if temporal_operator.active_route() is not None:
                raise SAICOnlineMotionNativeRuntimeError(
                    "native negative attempted under temporal operator route"
                )
            result = self.original_shared_step(*args, **kwargs)
            self._validate_shared_result(
                result,
                tokens=int(observation.noisy_latents.shape[1]),
                label="native negative output",
            )
            self.official_negative_calls += 1
            state.pending_negative = observation
            return result

        if state.pending_action is not None:
            raise SAICOnlineMotionNativeRuntimeError(
                "more than two official forwards occurred before scheduler.step"
            )
        values = _flatten_bound_arguments(
            self.vendor_shared_step_signature, args, kwargs
        )
        observation = self._validate_shared_call(
            values, state=state, role="action"
        )
        target = state.target_patch
        if target is None:
            raise SAICOnlineMotionNativeRuntimeError("current raw target disappeared")
        # Native timesteps are int64.  Mint the exact-valued, device-local FP32
        # coordinate required by build_online_motion_field/operator v2 only
        # after the native coordinate has passed the exact40 audit above.
        operator_timestep = torch.tensor(
            [float(sigma_strata.PINNED_TIMESTEPS[state.completed_steps])],
            dtype=torch.float32,
            device=target.raw_state.device,
        )
        target_tokens = int(target.packed_tokens.shape[1])
        state.teacher_target_tokens = observation.noisy_latents[
            :, -target_tokens:, :
        ]
        state.teacher_target_rotary = observation.rotary[
            :, :, -target_tokens:, :
        ]
        if (
            not torch.equal(state.teacher_target_tokens, target.packed_tokens)
            or not torch.equal(state.teacher_target_rotary, target.rotary)
        ):
            raise SAICOnlineMotionNativeRuntimeError(
                "official target suffix views differ from captured target patch"
            )

        def frozen_t2v_velocity(
            request: online_motion.FrozenT2VVelocityRequest,
        ) -> torch.Tensor:
            return self._teacher_query(
                args, kwargs, state=state, request=request
            )

        try:
            motion_field = online_motion.build_online_motion_field(
                current_noisy_target=target.raw_state,
                action_prompt=self.action_prompt,
                noop_prompt=self.noop_prompt,
                scheduler=self.scheduler,
                timestep=operator_timestep,
                frozen_t2v_velocity=frozen_t2v_velocity,
            )
        except Exception as error:
            raise SAICOnlineMotionNativeRuntimeError(
                f"cannot build same-state online motion field: {error}"
            ) from error
        if temporal_operator.active_route() is not None:
            raise SAICOnlineMotionNativeRuntimeError(
                "temporal operator route leaked from teacher queries"
            )
        branch = self._build_native_branch(values, state=state)
        try:
            with self.action_handle.route(
                branch=branch,
                scheduler=self.scheduler,
                timestep=operator_timestep,
                current_noisy_target=target.raw_state,
                motion_field=motion_field,
            ) as route:
                if temporal_operator.active_route() is not route:
                    raise SAICOnlineMotionNativeRuntimeError(
                        "action handle did not activate its minted route"
                    )
                result = self.original_shared_step(*args, **kwargs)
                route_receipt = dict(route.receipt())
        except SAICOnlineMotionNativeRuntimeError:
            raise
        except Exception as error:
            raise SAICOnlineMotionNativeRuntimeError(
                f"native action routed forward failed: {error}"
            ) from error
        if temporal_operator.active_route() is not None:
            raise SAICOnlineMotionNativeRuntimeError(
                "temporal action route leaked after native action"
            )
        self._validate_shared_result(
            result,
            tokens=int(observation.noisy_latents.shape[1]),
            label="native action output",
        )
        self.official_action_calls += 1
        state.pending_action = observation
        state.pending_motion_field = motion_field
        state.pending_route_receipt = route_receipt
        return result

    def _wrapped_scheduler_step(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise SAICOnlineMotionNativeRuntimeError(
                "scheduler.step ran outside the authenticated sample"
            )
        if (
            state.pending_negative is None
            or state.pending_action is None
            or state.pending_motion_field is None
            or state.pending_route_receipt is None
        ):
            raise SAICOnlineMotionNativeRuntimeError(
                "scheduler.step arrived before negative/teacher/action closure"
            )
        if temporal_operator.active_route() is not None:
            raise SAICOnlineMotionNativeRuntimeError(
                "scheduler.step attempted under temporal operator route"
            )
        if len(args) > 1:
            scheduler_timestep = args[1]
        elif "timestep" in kwargs:
            scheduler_timestep = kwargs["timestep"]
        else:
            raise SAICOnlineMotionNativeRuntimeError(
                "scheduler.step is missing timestep"
            )
        try:
            sgaf._certify_expanded_timestep(  # noqa: SLF001 - audited native seam
                state.pending_negative.timestep, scheduler_timestep
            )
        except sgaf.SelfGuidedActionFieldError as error:
            raise SAICOnlineMotionNativeRuntimeError(str(error)) from error
        _plain_schedule_index(
            state.pending_negative.timestep, expected=state.completed_steps
        )
        before = getattr(self.scheduler, "step_index", None)
        if before is not None and int(before) != state.completed_steps:
            raise SAICOnlineMotionNativeRuntimeError(
                "live scheduler cursor differs before original step"
            )
        result = self.original_scheduler_step(*args, **kwargs)
        self.original_scheduler_calls += 1
        if type(result) is not tuple or len(result) != 1:
            raise SAICOnlineMotionNativeRuntimeError(
                "native return_dict=False scheduler result must be one tuple"
            )
        after = getattr(self.scheduler, "step_index", None)
        if after is None or int(after) != state.completed_steps + 1:
            raise SAICOnlineMotionNativeRuntimeError(
                "live scheduler cursor did not advance exactly once"
            )
        route_receipt = dict(state.pending_route_receipt)
        field_receipt = dict(state.pending_motion_field.receipt())
        self.trace.append(
            {
                "schedule_index": state.completed_steps,
                "timestep": sigma_strata.PINNED_TIMESTEPS[state.completed_steps],
                "sigma_float32_be_hex": sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[
                    state.completed_steps
                ],
                "native_patch_source_ids": list(state.patch_source_ids),
                "native_branch": self._branch_name(state),
                "call_order": [
                    "native-negative",
                    "t2v-action-target-only",
                    "t2v-noop-target-only",
                    "native-action-routed",
                    "original-scheduler-step",
                ],
                "operator_active": route_receipt["operator_active"],
                "operator_gate": route_receipt["sigma_gate"],
                "motion_field_digest": field_receipt["digest"],
                "route_receipt_digest": route_receipt["digest"],
                "raw_state_source": (
                    "native_transformer.patch_vae_latent_source_id_0_argument"
                ),
                "native_timestep_dtype": "torch.int64",
                "operator_coordinate_derivation": (
                    "authenticated_native_int64_to_exact_device_fp32"
                ),
                "negative_operator_route_absent": True,
                "teacher_operator_route_absent": True,
                "scheduler_operator_route_absent": True,
                "caller_supplied_mask_branch_code_action_id_index_sigma_or_sp": False,
            }
        )
        state.completed_steps += 1
        state.patch_source_ids.clear()
        state.target_patch = None
        state.pending_negative = None
        state.pending_action = None
        state.teacher_target_tokens = None
        state.teacher_target_rotary = None
        state.pending_motion_field = None
        state.pending_route_receipt = None
        return result

    def finalize(self) -> Mapping[str, Any]:
        if not self.restored or self.finalized:
            raise SAICOnlineMotionNativeRuntimeError(
                "native runtime finalize lifecycle differs"
            )
        if (
            self.schedule_audit is None
            or self.sample_contract is None
            or self.output_sha256 is None
            or self.sample_calls != 1
            or self.successful_sample_calls != 1
            or self.official_negative_calls != EXPECTED_STEPS
            or self.official_action_calls != EXPECTED_STEPS
            or self.teacher_action_calls != EXPECTED_STEPS
            or self.teacher_noop_calls != EXPECTED_STEPS
            or self.original_scheduler_calls != EXPECTED_STEPS
            or len(self.trace) != EXPECTED_STEPS
            or [row["schedule_index"] for row in self.trace]
            != list(range(EXPECTED_STEPS))
        ):
            raise SAICOnlineMotionNativeRuntimeError(
                "native runtime exact40 call-count certificate differs"
            )
        unsigned = {
            "schema_version": SCHEMA_VERSION,
            "classification": CLASSIFICATION,
            "official_sample_calls": self.sample_calls,
            "exact81": True,
            "exact40": True,
            "official_negative_calls": self.official_negative_calls,
            "official_action_calls": self.official_action_calls,
            "target_only_t2v_action_calls": self.teacher_action_calls,
            "target_only_t2v_noop_calls": self.teacher_noop_calls,
            "original_scheduler_calls": self.original_scheduler_calls,
            "native_raw_state_seam": (
                "transformer.patch_vae_latent(unpacked_noisy_latent,source_id=0)"
            ),
            "native_raw_state_captured_before_official_shared_steps": True,
            "online_motion_field_built_before_native_action_forward": True,
            "temporal_operator_only_active_for_native_action_forward": True,
            "negative_noop_and_t2v_teacher_learned_path_executed": False,
            "teacher_conditioning": "target_tokens_and_rotary_only",
            "teacher_same_native_target_state_and_timestep": True,
            "native_timestep_to_operator_coordinate": (
                "exact_int64_schedule_value_to_device_fp32_after_audit"
            ),
            "native_branch_and_target_mask_derived_inside_runtime": True,
            "source_anchor_inner_runtime_present": self.source_anchor_inner is not None,
            "caller_supplied_mask_branch_code_action_id_index_sigma_or_sp": False,
            "allowed_native_branches": ["V", "VI"],
            "sample_contract": dict(self.sample_contract),
            "sample_contract_digest": _object_sha256(self.sample_contract),
            "exact40_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
            "scheduler_config_digest": PINNED_SCHEDULER_CONFIG_DIGEST,
            "scheduler_step_signature": dict(self.scheduler_step_signature),
            "sgaf_seam_reused": (
                "NativeRV2VActionFieldPatch reversible sample/shared/scheduler pattern"
            ),
            "t_qmosaic_runtime_audits_reused": [
                "pinned diffusion/scheduler class",
                "closed scheduler config",
                "scheduler.step signature",
            ],
            "t_qmosaic_audit_source_schema": TQMOSAIC_AUDIT_SOURCE_SCHEMA,
            "pinned_bernini_commit": PINNED_BERNINI_COMMIT,
            "pinned_wan_diffusion_sha256": PINNED_WAN_DIFFUSION_SHA256,
            "official_output_sha256": self.output_sha256,
            "operator_checkpoint_loaded_by_this_runtime": False,
            "optimizer_created": False,
            "parameters_updated": False,
            "training_authorized": False,
            "semantic_action_success_claim": False,
            "trace": list(self.trace),
        }
        self.finalized = True
        return {**unsigned, "receipt_digest": _object_sha256(unsigned)}


@contextmanager
def saic_online_motion_native_runtime(
    diffusion: Any,
    *,
    action_handle: temporal_operator.SAICTemporalActionOperatorHandle,
    action_prompt: str,
    noop_prompt: str,
    action_t2v_embeds: torch.Tensor,
    noop_t2v_embeds: torch.Tensor,
) -> Iterator[SAICOnlineMotionNativeRuntimeV1]:
    """Install the one-shot native runtime and always restore it."""

    runtime = SAICOnlineMotionNativeRuntimeV1(
        diffusion,
        action_handle=action_handle,
        action_prompt=action_prompt,
        noop_prompt=noop_prompt,
        action_t2v_embeds=action_t2v_embeds,
        noop_t2v_embeds=noop_t2v_embeds,
    )
    runtime.install()
    try:
        yield runtime
    finally:
        runtime.restore()


__all__ = [
    "CLASSIFICATION",
    "EXPECTED_FRAMES",
    "EXPECTED_STEPS",
    "SAICOnlineMotionNativeRuntimeError",
    "SAICOnlineMotionNativeRuntimeV1",
    "SCHEMA_VERSION",
    "saic_online_motion_native_runtime",
]
