#!/usr/bin/env python3
"""Same-state T2V-APG action field at Bernini's native UniPC boundary.

The pinned Bernini ``v2v_apg`` sampler performs exactly two official
``shared_step`` calls per solver step: negative RV2V text followed by action
RV2V text.  APG is then completed inside the untouched vendor sampler and the
resulting target-only velocity is passed to ``scheduler.step``.

This module preserves that call graph.  It observes the two official calls and
adds three *target-only* frozen T2V queries on the exact same target noisy
state: negative, target-action, and source-action.  At the scheduler boundary
it uses the exact active UniPC sigma and the pinned Bernini APG program to form
two independently guided T2V velocities::

    delta_action = v_t2v_apg(target | negative)
                 - v_t2v_apg(source | negative)

The final velocity is composed only after the official RV2V APG output exists::

    v_final = v_official_rv2v_apg + scale * gate(sigma) * delta_action

There is no division by ``omega_txt``: the injected quantity is already an
APG-guided velocity, not a raw conditional prediction.  The original UniPC
``scheduler.step`` is invoked exactly once.  Scale zero is an object-exact
no-op and performs no teacher forwards.

The wrapper is intentionally fail-closed.  It authenticates the official
negative/action order, model route, prompt object identities, shared noisy /
timestep / rotary objects, sequence metadata, target-tail geometry, exact
native APG parity, scheduler sigma, and reversible wrapper lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import math
from typing import Any, Callable, Mapping, Optional, Sequence


class SelfGuidedActionFieldError(RuntimeError):
    """Raised before UniPC integration when the pinned SGAF contract differs."""


@dataclass(frozen=True)
class ActionFieldConfig:
    """Pinned geometry, APG, and schedule for exactly one sampling call."""

    target_patch_tokens: int
    effective_scale: float
    target_latent_shape: tuple[int, int, int, int, int]
    expected_condition_prefix_tokens: int
    expected_steps: int = 40
    native_text_guidance_scale: float = 4.0
    sigma_zero_below: float = 0.20
    sigma_full_above: float = 0.55
    maximum_delta_to_native_text_rms: float = 1.50
    expected_hidden_dim: int = 1536
    expected_model_id: str = "transformer_1"
    expected_guidance_mode: str = "v2v_apg"

    def validate(self) -> None:
        if type(self.target_patch_tokens) is not int or self.target_patch_tokens <= 0:
            raise SelfGuidedActionFieldError("target_patch_tokens must be positive")
        if type(self.expected_steps) is not int or self.expected_steps <= 0:
            raise SelfGuidedActionFieldError("expected_steps must be positive")
        shape = tuple(self.target_latent_shape)
        if len(shape) != 5 or any(type(value) is not int or value <= 0 for value in shape):
            raise SelfGuidedActionFieldError(
                "target_latent_shape must be positive [B,C,T,H,W] integers"
            )
        batch, channels, frames, height, width = shape
        if batch != 1 or channels != 16 or height % 2 or width % 2:
            raise SelfGuidedActionFieldError(
                "pinned Bernini target latent must be [1,16,T,even,even]"
            )
        expected_tokens = frames * (height // 2) * (width // 2)
        if expected_tokens != self.target_patch_tokens:
            raise SelfGuidedActionFieldError(
                "target_patch_tokens differs from target_latent_shape packing"
            )
        if (
            type(self.expected_condition_prefix_tokens) is not int
            or self.expected_condition_prefix_tokens <= 0
        ):
            raise SelfGuidedActionFieldError(
                "expected_condition_prefix_tokens must be positive"
            )
        if self.target_patch_tokens % frames:
            raise SelfGuidedActionFieldError(
                "target token count must divide into exact latent phases"
            )
        reference_tokens = self.target_patch_tokens // frames
        pinned_prefix = self.target_patch_tokens + 4 * reference_tokens
        if self.expected_condition_prefix_tokens != pinned_prefix:
            raise SelfGuidedActionFieldError(
                "condition prefix must encode one exact source video plus four "
                "one-frame references"
            )
        if self.expected_model_id != "transformer_1":
            raise SelfGuidedActionFieldError(
                "SGAF is pinned to Bernini-R 1.3B transformer_1"
            )
        if self.expected_hidden_dim != 1536:
            raise SelfGuidedActionFieldError(
                "SGAF is pinned to Bernini-R 1.3B hidden_dim=1536"
            )
        if self.expected_guidance_mode != "v2v_apg":
            raise SelfGuidedActionFieldError("SGAF requires guidance_mode='v2v_apg'")
        for name in (
            "effective_scale",
            "native_text_guidance_scale",
            "sigma_zero_below",
            "sigma_full_above",
            "maximum_delta_to_native_text_rms",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise SelfGuidedActionFieldError(f"{name} must be finite")
        if self.effective_scale < 0.0:
            raise SelfGuidedActionFieldError("effective_scale must be non-negative")
        if self.native_text_guidance_scale <= 0.0:
            raise SelfGuidedActionFieldError(
                "native_text_guidance_scale must be positive"
            )
        if not 0.0 <= self.sigma_zero_below < self.sigma_full_above <= 1.0:
            raise SelfGuidedActionFieldError("sigma gate must lie inside [0,1]")
        if self.maximum_delta_to_native_text_rms <= 0.0:
            raise SelfGuidedActionFieldError(
                "maximum_delta_to_native_text_rms must be positive"
            )


@dataclass(frozen=True)
class _APGParameters:
    guidance_scale: float
    eta: float
    norm_threshold: float
    momentum: float


class _MomentumBuffer:
    """Branch-local copy of Bernini's APG momentum accumulator."""

    def __init__(self, momentum: float, *, branch: str) -> None:
        self.momentum = float(momentum)
        self.branch = branch
        self.running_average: Any = 0
        self.update_count = 0

    def update(self, value: Any) -> None:
        self.running_average = value + self.momentum * self.running_average
        self.update_count += 1


@dataclass
class _PendingStep:
    model_id: str
    noisy_latents: Any
    timesteps: Any
    rotary_embs: Any
    batch_vae_seqlen: tuple[int, ...]
    negative_tail: Any
    action_tail: Any
    teacher_uncond: Optional[Any]
    teacher_target: Optional[Any]
    teacher_source: Optional[Any]
    target_noisy: Optional[Any]
    target_rotary: Optional[Any]


@dataclass
class _ActiveSample:
    action_prompt: Any
    negative_prompt: Any
    apg: _APGParameters
    native_momentum: _MomentumBuffer
    teacher_target_momentum: _MomentumBuffer
    teacher_source_momentum: _MomentumBuffer
    pending_negative: Optional[tuple[dict[str, Any], Any]] = None
    pending_step: Optional[_PendingStep] = None
    completed_steps: int = 0


def smooth_action_gate(
    sigma: float,
    *,
    zero_below: float,
    full_above: float,
) -> float:
    """C1 smoothstep: zero at low noise and one at high/mid noise."""

    values = (sigma, zero_below, full_above)
    if any(not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in values):
        raise SelfGuidedActionFieldError("sigma gate arguments must be finite")
    sigma_f, lo, hi = (float(v) for v in values)
    if not 0.0 <= lo < hi <= 1.0 or not 0.0 <= sigma_f <= 1.0 + 1e-6:
        raise SelfGuidedActionFieldError("sigma gate coordinate differs")
    if sigma_f <= lo:
        return 0.0
    if sigma_f >= hi:
        return 1.0
    u = (sigma_f - lo) / (hi - lo)
    return float(u * u * (3.0 - 2.0 * u))


def _tensor_rms(value: Any) -> Any:
    import torch

    if not isinstance(value, torch.Tensor) or value.numel() <= 0:
        raise SelfGuidedActionFieldError("RMS input must be a non-empty tensor")
    return value.float().square().mean().sqrt()


def clip_action_delta_by_native_text_rms(
    action_delta: Any,
    native_text_delta: Any,
    *,
    maximum_ratio: float,
    epsilon: float = 1e-8,
) -> tuple[Any, float, float, float]:
    """Clip the APG teacher delta without normalizing away its confidence."""

    import torch

    if (
        not isinstance(action_delta, torch.Tensor)
        or not isinstance(native_text_delta, torch.Tensor)
        or tuple(action_delta.shape) != tuple(native_text_delta.shape)
        or action_delta.device != native_text_delta.device
    ):
        raise SelfGuidedActionFieldError("action/native text delta geometry differs")
    if not math.isfinite(float(maximum_ratio)) or maximum_ratio <= 0.0:
        raise SelfGuidedActionFieldError("maximum_ratio must be positive and finite")
    if not bool(torch.isfinite(action_delta).all().item()) or not bool(
        torch.isfinite(native_text_delta).all().item()
    ):
        raise SelfGuidedActionFieldError("action/native text delta is non-finite")
    action_rms_tensor = _tensor_rms(action_delta)
    native_rms_tensor = _tensor_rms(native_text_delta)
    action_rms = float(action_rms_tensor.item())
    native_rms = float(native_rms_tensor.item())
    ceiling = float(maximum_ratio) * native_rms
    if action_rms <= epsilon or ceiling <= epsilon:
        multiplier = 0.0
    else:
        multiplier = min(1.0, ceiling / action_rms)
    return action_delta * multiplier, action_rms, native_rms, float(multiplier)


def _shape(value: Any, *, label: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.shape)
    except Exception as error:
        raise SelfGuidedActionFieldError(f"{label} must expose an integer shape") from error


def _metadata_tuple(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise SelfGuidedActionFieldError(f"{label} must be list/tuple metadata")
    if any(type(item) is not int or item <= 0 for item in value):
        raise SelfGuidedActionFieldError(f"{label} must contain positive integers")
    return tuple(value)


def _coerce_scalar(value: Any, *, label: str) -> float:
    try:
        candidate = value.detach() if hasattr(value, "detach") else value
        if hasattr(candidate, "numel") and int(candidate.numel()) != 1:
            raise SelfGuidedActionFieldError(f"{label} must be scalar")
        if hasattr(candidate, "cpu"):
            candidate = candidate.cpu()
        if hasattr(candidate, "item"):
            candidate = candidate.item()
        result = float(candidate)
    except SelfGuidedActionFieldError:
        raise
    except Exception as error:
        raise SelfGuidedActionFieldError(f"{label} must be scalar") from error
    if not math.isfinite(result):
        raise SelfGuidedActionFieldError(f"{label} must be finite")
    return result


def _coerce_index(value: Any, *, label: str) -> int:
    numeric = _coerce_scalar(value, label=label)
    integer = int(numeric)
    if numeric != float(integer) or integer < 0:
        raise SelfGuidedActionFieldError(f"{label} must be a non-negative integer")
    return integer


def _bind_call(
    callable_object: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        bound = inspect.signature(callable_object).bind(*args, **kwargs)
        bound.apply_defaults()
    except (TypeError, ValueError) as error:
        raise SelfGuidedActionFieldError(
            "call does not match the pinned Bernini signature"
        ) from error
    return dict(bound.arguments)


def _extract_argument(
    args: Sequence[Any], kwargs: Mapping[str, Any], *, index: int, name: str
) -> Any:
    if len(args) > index and name in kwargs:
        raise SelfGuidedActionFieldError(f"call received duplicate {name}")
    if len(args) > index:
        return args[index]
    if name in kwargs:
        return kwargs[name]
    raise SelfGuidedActionFieldError(f"call is missing {name}")


def _replace_argument(
    callable_object: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    name: str,
    value: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    new_args = list(args)
    new_kwargs = dict(kwargs)
    if name in new_kwargs:
        new_kwargs[name] = value
        return tuple(new_args), new_kwargs
    try:
        parameters = list(inspect.signature(callable_object).parameters.values())
    except (TypeError, ValueError) as error:
        raise SelfGuidedActionFieldError(
            "cannot inspect pinned callable signature"
        ) from error
    positional_names = [
        parameter.name
        for parameter in parameters
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    if name in positional_names:
        position = positional_names.index(name)
        if position < len(new_args):
            new_args[position] = value
            return tuple(new_args), new_kwargs
    new_kwargs[name] = value
    return tuple(new_args), new_kwargs


def _same_object(left: Any, right: Any, *, label: str) -> None:
    if left is not right:
        raise SelfGuidedActionFieldError(f"{label} must be the exact same object")


def _certify_expanded_timestep(shared_timestep: Any, scheduler_timestep: Any) -> None:
    """Prove vendor ``t.expand(1)`` is a view of scheduler scalar ``t``.

    Pinned Bernini passes ``t.expand(1)`` to both transformer calls but sends
    scalar ``t`` to UniPC.  They cannot be the same Python object.  Requiring
    the same dtype/device/value, storage, offset, data pointer, and zero-stride
    expansion authenticates the real relationship without accepting a copied
    equal-valued timestep.
    """

    import torch

    if (
        not isinstance(shared_timestep, torch.Tensor)
        or not isinstance(scheduler_timestep, torch.Tensor)
        or shared_timestep.shape != (1,)
        or scheduler_timestep.ndim != 0
        or shared_timestep.dtype != scheduler_timestep.dtype
        or shared_timestep.device != scheduler_timestep.device
        or shared_timestep.stride() != (0,)
        or int(shared_timestep.storage_offset())
        != int(scheduler_timestep.storage_offset())
        or int(shared_timestep.data_ptr()) != int(scheduler_timestep.data_ptr())
        or int(shared_timestep.untyped_storage().data_ptr())
        != int(scheduler_timestep.untyped_storage().data_ptr())
        or not torch.equal(shared_timestep.reshape(()), scheduler_timestep)
    ):
        raise SelfGuidedActionFieldError(
            "shared timestep is not the authenticated zero-stride expand(1) "
            "view of the scheduler scalar"
        )


def _resolve_sigma(scheduler: Any, timestep: Any) -> tuple[int, Any, float]:
    """Read the exact active UniPC sigma without advancing the scheduler."""

    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is None:
        raise SelfGuidedActionFieldError("scheduler must expose sigmas")
    current = getattr(scheduler, "step_index", None)
    if current is not None:
        index = _coerce_index(current, label="scheduler.step_index")
    else:
        begin = getattr(scheduler, "begin_index", None)
        if begin is None:
            begin = getattr(scheduler, "_begin_index", None)
        if begin is not None:
            index = _coerce_index(begin, label="scheduler.begin_index")
        else:
            resolver = getattr(scheduler, "index_for_timestep", None)
            if callable(resolver):
                lookup = timestep
                timeline = getattr(scheduler, "timesteps", None)
                timeline_device = getattr(timeline, "device", None)
                if timeline_device is not None and hasattr(lookup, "to"):
                    lookup = lookup.to(device=timeline_device)
                try:
                    index = _coerce_index(
                        resolver(lookup), label="scheduler timestep index"
                    )
                except SelfGuidedActionFieldError:
                    raise
                except Exception as error:
                    raise SelfGuidedActionFieldError(
                        "scheduler.index_for_timestep failed"
                    ) from error
            else:
                timeline = getattr(scheduler, "timesteps", None)
                if timeline is None:
                    raise SelfGuidedActionFieldError(
                        "scheduler lacks timestep lookup state"
                    )
                query = _coerce_scalar(timestep, label="timestep")
                matches = [
                    position
                    for position, value in enumerate(timeline)
                    if _coerce_scalar(value, label="scheduler timestep") == query
                ]
                if not matches:
                    raise SelfGuidedActionFieldError(
                        "timestep is absent from scheduler.timesteps"
                    )
                index = matches[1] if len(matches) > 1 else matches[0]
    try:
        sigma = sigmas[index]
    except Exception as error:
        raise SelfGuidedActionFieldError("scheduler sigma index is invalid") from error
    sigma_float = _coerce_scalar(sigma, label="scheduler sigma")
    if not 0.0 < sigma_float <= 1.0 + 1e-6:
        raise SelfGuidedActionFieldError(
            "active flow sigma must be finite in (0,1]"
        )
    return index, sigma, sigma_float


def _packed_to_spatial(packed: Any, shape: tuple[int, int, int, int, int]) -> Any:
    batch, channels, frames, height, width = shape
    expected = (batch, frames * (height // 2) * (width // 2), channels * 4)
    if _shape(packed, label="packed tensor") != expected:
        raise SelfGuidedActionFieldError(
            f"packed tensor geometry differs from {expected}"
        )
    return (
        packed.reshape(batch, frames, height // 2, width // 2, 2, 2, channels)
        .permute(0, 6, 1, 2, 4, 3, 5)
        .reshape(shape)
    )


def _spatial_to_packed(spatial: Any, shape: tuple[int, int, int, int, int]) -> Any:
    batch, channels, frames, height, width = shape
    if _shape(spatial, label="spatial tensor") != shape:
        raise SelfGuidedActionFieldError("spatial tensor geometry differs")
    return (
        spatial.reshape(
            batch,
            channels,
            frames,
            height // 2,
            2,
            width // 2,
            2,
        )
        .permute(0, 2, 3, 5, 4, 6, 1)
        .reshape(batch, frames * (height // 2) * (width // 2), channels * 4)
    )


def _normalized_guidance(
    pred_cond: Any,
    pred_uncond: Any,
    *,
    parameters: _APGParameters,
    momentum_buffer: _MomentumBuffer,
) -> Any:
    """Exact local equivalent of pinned Bernini ``normalized_guidance``."""

    import torch
    import torch.nn.functional as torch_f

    diff = pred_cond - pred_uncond
    momentum_buffer.update(diff)
    diff = momentum_buffer.running_average
    if parameters.norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -4], keepdim=True)
        scale_factor = torch.minimum(ones, parameters.norm_threshold / diff_norm)
        diff = diff * scale_factor
    v0, v1 = diff.double(), pred_cond.double()
    v1 = torch_f.normalize(v1, dim=[-1, -2, -4])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -4], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    normalized = v0_orthogonal.to(diff.dtype) + parameters.eta * v0_parallel.to(
        diff.dtype
    )
    return pred_uncond + parameters.guidance_scale * normalized


def _guided_velocity(
    sample_packed: Any,
    uncond_velocity_packed: Any,
    cond_velocity_packed: Any,
    sigma: Any,
    *,
    shape: tuple[int, int, int, int, int],
    parameters: _APGParameters,
    momentum_buffer: _MomentumBuffer,
    output_like: Any,
) -> Any:
    """Run pinned clean-space APG and return scheduler-bound packed velocity."""

    import torch

    for label, value in (
        ("sample", sample_packed),
        ("uncond velocity", uncond_velocity_packed),
        ("conditional velocity", cond_velocity_packed),
        ("output reference", output_like),
    ):
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise SelfGuidedActionFieldError(f"{label} must be a finite tensor")
    if not isinstance(sigma, torch.Tensor) or sigma.ndim != 0:
        raise SelfGuidedActionFieldError("pinned UniPC sigma must be a 0-d tensor")
    if sigma.device.type != "cpu" or sigma.dtype != torch.float32:
        raise SelfGuidedActionFieldError(
            "pinned UniPC sigma must remain a CPU fp32 scalar"
        )
    sample = _packed_to_spatial(sample_packed, shape)
    uncond = _packed_to_spatial(uncond_velocity_packed, shape)
    conditional = _packed_to_spatial(cond_velocity_packed, shape)
    if sample.device != uncond.device or sample.device != conditional.device:
        raise SelfGuidedActionFieldError("APG sample/branch devices differ")
    uncond_clean = sample - sigma * uncond
    conditional_clean = sample - sigma * conditional
    guided_clean = _normalized_guidance(
        conditional_clean,
        uncond_clean,
        parameters=parameters,
        momentum_buffer=momentum_buffer,
    )
    return _spatial_to_packed((sample - guided_clean) / sigma, shape).to(
        device=output_like.device,
        dtype=output_like.dtype,
    )


def module_state_hash_certificate(module: Any) -> dict[str, Any]:
    """Hash every named parameter and buffer, including raw BF16 bytes.

    The certificate is device-independent and streams one tensor at a time.
    Comparing the complete certificate before and after SGAF proves both
    parameters *and buffers* remained frozen; checking ``requires_grad`` alone
    is not a state-integrity certificate.
    """

    import torch

    if not callable(getattr(module, "named_parameters", None)) or not callable(
        getattr(module, "named_buffers", None)
    ):
        raise SelfGuidedActionFieldError("freeze certificate requires a torch module")
    digest = hashlib.sha256()
    rows: list[tuple[str, str, Any]] = []
    rows.extend(("parameter", name, value) for name, value in module.named_parameters())
    rows.extend(("buffer", name, value) for name, value in module.named_buffers())
    names = [(kind, name) for kind, name, _ in rows]
    if len(names) != len(set(names)):
        raise SelfGuidedActionFieldError("module state contains duplicate names")
    parameter_tensors = parameter_elements = 0
    buffer_tensors = buffer_elements = 0
    trainable_tensors = trainable_elements = 0
    for kind, name, value in sorted(rows, key=lambda row: (row[0], row[1])):
        if not isinstance(value, torch.Tensor):
            raise SelfGuidedActionFieldError("module state entry is not a tensor")
        if kind == "parameter":
            parameter_tensors += 1
            parameter_elements += int(value.numel())
            if bool(value.requires_grad):
                trainable_tensors += 1
                trainable_elements += int(value.numel())
        else:
            buffer_tensors += 1
            buffer_elements += int(value.numel())
        metadata = (
            f"{kind}\0{name}\0{value.dtype}\0"
            + ",".join(str(int(item)) for item in value.shape)
            + "\0"
        ).encode("utf-8")
        # dtype-view rejects zero-dimensional tensors when element sizes
        # differ; flatten first so scalar parameters/buffers are certified too.
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(int(raw.numel()).to_bytes(8, "big"))
        digest.update(raw.numpy().tobytes(order="C"))
    if trainable_tensors:
        raise SelfGuidedActionFieldError("SGAF model contains trainable parameters")
    return {
        "schema_version": "torch-module-parameters-buffers-sha256-v1",
        "parameters_and_buffers_sha256": digest.hexdigest(),
        "parameter_tensors": parameter_tensors,
        "parameter_elements": parameter_elements,
        "buffer_tensors": buffer_tensors,
        "buffer_elements": buffer_elements,
        "trainable_parameter_tensors": trainable_tensors,
        "trainable_parameter_elements": trainable_elements,
        "all_parameters_frozen": True,
    }


class NativeRV2VActionFieldPatch:
    """Reversible five-forward SGAF adapter for one pinned Bernini sample."""

    def __init__(
        self,
        diffusion: Any,
        *,
        target_t2v_embeds: Any,
        source_t2v_embeds: Any,
        config: ActionFieldConfig,
    ) -> None:
        import torch

        config.validate()
        for label, value in (
            ("target_t2v_embeds", target_t2v_embeds),
            ("source_t2v_embeds", source_t2v_embeds),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 3
                or value.shape[0] != 1
                or value.numel() <= 0
                or value.requires_grad
                or value.grad_fn is not None
                or not bool(torch.isfinite(value).all().item())
            ):
                raise SelfGuidedActionFieldError(f"{label} differs")
        if target_t2v_embeds is source_t2v_embeds:
            raise SelfGuidedActionFieldError("target/source T2V prompts must be distinct")
        if target_t2v_embeds.device != source_t2v_embeds.device:
            raise SelfGuidedActionFieldError("teacher prompt devices differ")
        scheduler = getattr(diffusion, "scheduler", None)
        originals = {
            "sample": getattr(diffusion, "sample", None),
            "shared_step": getattr(diffusion, "shared_step", None),
            "scheduler.step": getattr(scheduler, "step", None),
        }
        if any(not callable(value) for value in originals.values()):
            raise SelfGuidedActionFieldError(
                "diffusion must expose callable sample/shared_step/scheduler.step"
            )
        if getattr(diffusion, "use_unipc", None) is not True:
            raise SelfGuidedActionFieldError("SGAF requires diffusion.use_unipc is True")
        if getattr(diffusion, "transformer_2", None) is not None:
            raise SelfGuidedActionFieldError(
                "SGAF is pinned to the single-expert Bernini-R 1.3B model"
            )
        transformer_config = getattr(getattr(diffusion, "transformer", None), "config", None)
        if transformer_config is None:
            raise SelfGuidedActionFieldError(
                "SGAF requires the pinned transformer config"
            )

        def config_value(name: str) -> Any:
            value = getattr(transformer_config, name, None)
            if value is None and isinstance(transformer_config, Mapping):
                value = transformer_config.get(name)
            return value

        heads = config_value("num_attention_heads")
        head_dim = config_value("attention_head_dim")
        if (
            type(heads) is not int
            or type(head_dim) is not int
            or heads * head_dim != config.expected_hidden_dim
        ):
            raise SelfGuidedActionFieldError(
                "transformer attention geometry does not authenticate hidden_dim=1536"
            )
        for owner, name in (
            (diffusion, "sample"),
            (diffusion, "shared_step"),
            (scheduler, "step"),
        ):
            try:
                if name in vars(owner):
                    raise SelfGuidedActionFieldError(
                        f"refusing to stack over instance-level {name}"
                    )
            except TypeError as error:
                raise SelfGuidedActionFieldError(
                    f"cannot inspect {name} owner"
                ) from error
        for original in originals.values():
            if getattr(original, "_bernini_sgaf_v1", None) is not None:
                raise SelfGuidedActionFieldError("SGAF wrapper is already installed")

        self.diffusion = diffusion
        self.scheduler = scheduler
        self.target_t2v_embeds = target_t2v_embeds
        self.source_t2v_embeds = source_t2v_embeds
        self.config = config
        self.original_sample = originals["sample"]
        self.original_shared_step = originals["shared_step"]
        self.original_scheduler_step = originals["scheduler.step"]
        self.installed = False
        self.restored = False
        self.finalized = False
        self._patches: list[tuple[Any, str, bool, Any, Any]] = []
        self._active: Optional[_ActiveSample] = None
        self.sample_call_count = 0
        self.native_call_count = 0
        self.teacher_call_count = 0
        self.original_scheduler_call_count = 0
        self.trace: list[dict[str, Any]] = []

    def _validate_prompt_tensor(self, value: Any, *, label: str) -> int:
        import torch

        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 3
            or value.shape[0] != 1
            or value.shape[1] <= 0
            or value.device != self.target_t2v_embeds.device
            or not bool(torch.isfinite(value).all())
        ):
            raise SelfGuidedActionFieldError(f"{label} prompt geometry differs")
        return int(value.shape[1])

    def _validate_shared_geometry(
        self,
        values: Mapping[str, Any],
        *,
        prompt: Any,
        branch: str,
    ) -> int:
        import torch

        model_id = str(values.get("model_id"))
        if model_id != self.config.expected_model_id:
            raise SelfGuidedActionFieldError(f"{branch} model_id differs")
        if values.get("cond_embeds") is not prompt:
            raise SelfGuidedActionFieldError(
                f"{branch} prompt is not the exact authenticated object"
            )
        prompt_length = self._validate_prompt_tensor(prompt, label=branch)
        noisy = values.get("noisy_latents")
        noisy_shape = _shape(noisy, label=f"{branch} noisy_latents")
        packed_channels = int(self.config.target_latent_shape[1]) * 4
        expected_full_tokens = (
            self.config.expected_condition_prefix_tokens
            + self.config.target_patch_tokens
        )
        if (
            not isinstance(noisy, torch.Tensor)
            or len(noisy_shape) != 3
            or noisy_shape[0] != 1
            or noisy_shape[1] != expected_full_tokens
            or noisy_shape[2] != self.config.expected_hidden_dim
            or not bool(torch.isfinite(noisy).all())
        ):
            raise SelfGuidedActionFieldError(f"{branch} noisy geometry differs")
        full_tokens = noisy_shape[1]
        if _metadata_tuple(
            values.get("batch_vae_seqlen"), label=f"{branch} batch_vae_seqlen"
        ) != (full_tokens,):
            raise SelfGuidedActionFieldError(
                f"{branch} batch_vae_seqlen differs from noisy sequence"
            )
        if _metadata_tuple(
            values.get("batch_text_seqlen"), label=f"{branch} batch_text_seqlen"
        ) != (prompt_length,):
            raise SelfGuidedActionFieldError(
                f"{branch} batch_text_seqlen differs from prompt"
            )
        timestep = values.get("timesteps")
        if (
            not isinstance(timestep, torch.Tensor)
            or timestep.numel() != 1
            or not bool(torch.isfinite(timestep).all())
        ):
            raise SelfGuidedActionFieldError(f"{branch} timestep differs")
        rotary = values.get("rotary_embs")
        rotary_shape = _shape(rotary, label=f"{branch} rotary_embs")
        if (
            not isinstance(rotary, torch.Tensor)
            or len(rotary_shape) != 4
            or rotary_shape[0] != 1
            or rotary_shape[1] != 1
            or rotary_shape[2] != full_tokens
            or rotary_shape[3] <= 0
            or not bool(torch.isfinite(rotary).all())
        ):
            raise SelfGuidedActionFieldError(f"{branch} rotary geometry differs")
        return full_tokens

    def _validate_prediction(
        self, prediction: Any, values: Mapping[str, Any], *, branch: str
    ) -> Any:
        import torch

        noisy = values["noisy_latents"]
        expected_prediction_shape = (
            1,
            int(noisy.shape[1]),
            int(self.config.target_latent_shape[1]) * 4,
        )
        if (
            not isinstance(prediction, torch.Tensor)
            or _shape(prediction, label=f"{branch} prediction")
            != expected_prediction_shape
            or prediction.device != noisy.device
            or not bool(torch.isfinite(prediction).all())
        ):
            raise SelfGuidedActionFieldError(f"{branch} shared_step output differs")
        tail = prediction[:, -self.config.target_patch_tokens :, :]
        expected = (
            1,
            self.config.target_patch_tokens,
            int(self.config.target_latent_shape[1]) * 4,
        )
        if _shape(tail, label=f"{branch} target tail") != expected:
            raise SelfGuidedActionFieldError(f"{branch} target tail length differs")
        return tail

    def _teacher_query(
        self,
        base_args: Sequence[Any],
        base_kwargs: Mapping[str, Any],
        *,
        prompt: Any,
        target_noisy: Any,
        target_rotary: Any,
        timestep: Any,
        branch: str,
    ) -> Any:
        call_args, call_kwargs = tuple(base_args), dict(base_kwargs)
        replacements = {
            "noisy_latents": target_noisy,
            "timesteps": timestep,
            "cond_embeds": prompt,
            "rotary_embs": target_rotary,
            "batch_vae_seqlen": [self.config.target_patch_tokens],
            "batch_text_seqlen": [self._validate_prompt_tensor(prompt, label=branch)],
        }
        for name, value in replacements.items():
            call_args, call_kwargs = _replace_argument(
                self.original_shared_step,
                call_args,
                call_kwargs,
                name=name,
                value=value,
            )
        values = _bind_call(self.original_shared_step, call_args, call_kwargs)
        if str(values.get("model_id")) != self.config.expected_model_id:
            raise SelfGuidedActionFieldError(f"{branch} teacher model_id differs")
        for name, expected in (
            ("noisy_latents", target_noisy),
            ("timesteps", timestep),
            ("rotary_embs", target_rotary),
            ("cond_embeds", prompt),
        ):
            _same_object(values.get(name), expected, label=f"{branch} teacher {name}")
        if _metadata_tuple(
            values.get("batch_vae_seqlen"), label=f"{branch} teacher VAE length"
        ) != (self.config.target_patch_tokens,):
            raise SelfGuidedActionFieldError(f"{branch} teacher VAE length differs")
        prediction = self.original_shared_step(*call_args, **call_kwargs)
        self.teacher_call_count += 1
        expected_shape = (
            1,
            self.config.target_patch_tokens,
            int(self.config.target_latent_shape[1]) * 4,
        )
        if (
            _shape(prediction, label=f"{branch} teacher prediction") != expected_shape
            or prediction.device != target_noisy.device
        ):
            raise SelfGuidedActionFieldError(f"{branch} T2V teacher output differs")
        import torch

        if not bool(torch.isfinite(prediction).all()):
            raise SelfGuidedActionFieldError(f"{branch} T2V teacher output is non-finite")
        return prediction

    def _wrapped_shared_step(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise SelfGuidedActionFieldError(
                "shared_step ran outside the authenticated sample"
            )
        values = _bind_call(self.original_shared_step, args, kwargs)
        if state.pending_negative is None:
            if state.pending_step is not None:
                raise SelfGuidedActionFieldError(
                    "new native negative arrived before scheduler integration"
                )
            self._validate_shared_geometry(
                values, prompt=state.negative_prompt, branch="native negative"
            )
            prediction = self.original_shared_step(*args, **kwargs)
            self.native_call_count += 1
            negative_tail = self._validate_prediction(
                prediction, values, branch="native negative"
            )
            state.pending_negative = (dict(values), negative_tail)
            return prediction

        if state.pending_step is not None:
            raise SelfGuidedActionFieldError(
                "more than two official shared_step calls occurred before scheduler.step"
            )
        negative_values, negative_tail = state.pending_negative
        full_tokens = self._validate_shared_geometry(
            values, prompt=state.action_prompt, branch="native action"
        )
        if str(negative_values.get("model_id")) != str(values.get("model_id")):
            raise SelfGuidedActionFieldError("negative/action model_id differ")
        for name in ("noisy_latents", "timesteps", "rotary_embs"):
            _same_object(
                negative_values.get(name),
                values.get(name),
                label=f"negative/action {name}",
            )
        if _metadata_tuple(
            negative_values.get("batch_vae_seqlen"), label="negative VAE length"
        ) != _metadata_tuple(
            values.get("batch_vae_seqlen"), label="action VAE length"
        ):
            raise SelfGuidedActionFieldError(
                "negative/action batch_vae_seqlen differ"
            )
        prediction = self.original_shared_step(*args, **kwargs)
        self.native_call_count += 1
        action_tail = self._validate_prediction(
            prediction, values, branch="native action"
        )

        teacher_uncond = teacher_target = teacher_source = None
        target_noisy = target_rotary = None
        if self.config.effective_scale > 0.0:
            target_noisy = values["noisy_latents"][:, -self.config.target_patch_tokens :, :]
            target_rotary = values["rotary_embs"][
                :, :, -self.config.target_patch_tokens :, :
            ]
            if _shape(target_noisy, label="teacher target noisy") != (
                1,
                self.config.target_patch_tokens,
                self.config.expected_hidden_dim,
            ):
                raise SelfGuidedActionFieldError("teacher target-noisy tail differs")
            teacher_uncond = self._teacher_query(
                args,
                kwargs,
                prompt=state.negative_prompt,
                target_noisy=target_noisy,
                target_rotary=target_rotary,
                timestep=values["timesteps"],
                branch="T2V uncond",
            )
            teacher_target = self._teacher_query(
                args,
                kwargs,
                prompt=self.target_t2v_embeds,
                target_noisy=target_noisy,
                target_rotary=target_rotary,
                timestep=values["timesteps"],
                branch="T2V target",
            )
            teacher_source = self._teacher_query(
                args,
                kwargs,
                prompt=self.source_t2v_embeds,
                target_noisy=target_noisy,
                target_rotary=target_rotary,
                timestep=values["timesteps"],
                branch="T2V source",
            )
        state.pending_negative = None
        state.pending_step = _PendingStep(
            model_id=str(values["model_id"]),
            noisy_latents=values["noisy_latents"],
            timesteps=values["timesteps"],
            rotary_embs=values["rotary_embs"],
            batch_vae_seqlen=(full_tokens,),
            negative_tail=negative_tail,
            action_tail=action_tail,
            teacher_uncond=teacher_uncond,
            teacher_target=teacher_target,
            teacher_source=teacher_source,
            target_noisy=target_noisy,
            target_rotary=target_rotary,
        )
        return prediction

    def _wrapped_scheduler_step(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        state = self._active
        if state is None:
            raise SelfGuidedActionFieldError(
                "scheduler.step ran outside the authenticated sample"
            )
        pending = state.pending_step
        if pending is None or state.pending_negative is not None:
            raise SelfGuidedActionFieldError(
                "scheduler.step arrived before one complete native pair"
            )
        official = _extract_argument(args, kwargs, index=0, name="model_output")
        timestep = _extract_argument(args, kwargs, index=1, name="timestep")
        sample = _extract_argument(args, kwargs, index=2, name="sample")
        _certify_expanded_timestep(pending.timesteps, timestep)
        expected_shape = (
            1,
            self.config.target_patch_tokens,
            int(self.config.target_latent_shape[1]) * 4,
        )
        for label, value in (("official model_output", official), ("scheduler sample", sample)):
            if (
                not isinstance(value, torch.Tensor)
                or _shape(value, label=label) != expected_shape
                or not bool(torch.isfinite(value).all())
            ):
                raise SelfGuidedActionFieldError(f"{label} geometry differs")
        if official.device != sample.device:
            raise SelfGuidedActionFieldError("official output/sample devices differ")
        step_index, sigma, sigma_float = _resolve_sigma(self.scheduler, timestep)
        if step_index != state.completed_steps:
            raise SelfGuidedActionFieldError("scheduler step index differs from SGAF state")

        locally_rebuilt_native = _guided_velocity(
            sample,
            pending.negative_tail,
            pending.action_tail,
            sigma,
            shape=self.config.target_latent_shape,
            parameters=state.apg,
            momentum_buffer=state.native_momentum,
            output_like=official,
        )
        native_parity_error = locally_rebuilt_native.float() - official.float()
        native_parity_rms = float(_tensor_rms(native_parity_error).item())
        native_parity_max = float(native_parity_error.abs().max().item())
        if not torch.equal(locally_rebuilt_native, official):
            raise SelfGuidedActionFieldError(
                "locally rebuilt native APG differs from official model_output: "
                f"max_abs={native_parity_max:.9g} rms={native_parity_rms:.9g}"
            )

        gate = smooth_action_gate(
            sigma_float,
            zero_below=self.config.sigma_zero_below,
            full_above=self.config.sigma_full_above,
        )
        multiplier = float(self.config.effective_scale) * gate
        raw_rms = native_rms = 0.0
        clip_multiplier = 0.0
        guided_delta_rms = applied_rms = 0.0
        executed = official
        if self.config.effective_scale > 0.0:
            if any(
                value is None
                for value in (
                    pending.teacher_uncond,
                    pending.teacher_target,
                    pending.teacher_source,
                    pending.target_noisy,
                    pending.target_rotary,
                )
            ):
                raise SelfGuidedActionFieldError("T2V teacher triplet is incomplete")
            target_guided = _guided_velocity(
                sample,
                pending.teacher_uncond,
                pending.teacher_target,
                sigma,
                shape=self.config.target_latent_shape,
                parameters=state.apg,
                momentum_buffer=state.teacher_target_momentum,
                output_like=official,
            )
            source_guided = _guided_velocity(
                sample,
                pending.teacher_uncond,
                pending.teacher_source,
                sigma,
                shape=self.config.target_latent_shape,
                parameters=state.apg,
                momentum_buffer=state.teacher_source_momentum,
                output_like=official,
            )
            guided_delta = target_guided - source_guided
            guided_delta_rms = float(_tensor_rms(guided_delta).item())
            native_text_delta = pending.action_tail - pending.negative_tail
            clipped, raw_rms, native_rms, clip_multiplier = (
                clip_action_delta_by_native_text_rms(
                    guided_delta,
                    native_text_delta,
                    maximum_ratio=self.config.maximum_delta_to_native_text_rms,
                )
            )
            if multiplier != 0.0:
                correction = clipped.to(dtype=official.dtype) * multiplier
                executed = official + correction
                applied_rms = float(_tensor_rms(correction).item())
                if not bool(torch.isfinite(executed).all()):
                    raise SelfGuidedActionFieldError(
                        "composed scheduler velocity is non-finite"
                    )

        call_args, call_kwargs = _replace_argument(
            self.original_scheduler_step,
            args,
            kwargs,
            name="model_output",
            value=executed,
        )
        result = self.original_scheduler_step(*call_args, **call_kwargs)
        self.original_scheduler_call_count += 1
        state.completed_steps += 1
        self.trace.append(
            {
                "step_index": step_index,
                "timestep": _coerce_scalar(timestep, label="timestep"),
                "sigma": sigma_float,
                "gate": gate,
                "effective_scale": float(self.config.effective_scale),
                "native_official_apg_exact_parity": True,
                "native_official_apg_parity_rms": native_parity_rms,
                "native_official_apg_parity_max_abs": native_parity_max,
                "t2v_teacher_forwards": 3 if self.config.effective_scale > 0 else 0,
                "t2v_target_source_guided_delta_rms": guided_delta_rms,
                "unclipped_guided_action_delta_rms": raw_rms,
                "native_rv2v_raw_text_delta_rms": native_rms,
                "clip_multiplier": clip_multiplier,
                "applied_scheduler_velocity_delta_rms": applied_rms,
                "injection_divided_by_omega_txt": False,
                "original_scheduler_calls": 1,
                "scale_zero_exact_model_output_object": (
                    self.config.effective_scale == 0.0 and executed is official
                ),
            }
        )
        state.pending_step = None
        return result

    def _wrapped_sample(self, *args: Any, **kwargs: Any) -> Any:
        if self._active is not None or self.sample_call_count != 0:
            raise SelfGuidedActionFieldError(
                "SGAF permits exactly one non-nested diffusion.sample call"
            )
        if self.diffusion.scheduler is not self.scheduler:
            raise SelfGuidedActionFieldError("diffusion.scheduler changed after install")
        values = _bind_call(self.original_sample, args, kwargs)
        if values.get("guidance_mode") != self.config.expected_guidance_mode:
            raise SelfGuidedActionFieldError("sample guidance_mode differs")
        if int(values.get("num_inference_steps")) != self.config.expected_steps:
            raise SelfGuidedActionFieldError("sample inference-step count differs")
        action_prompt = values.get("prompt_embeds")
        negative_prompt = values.get("uncond_prompt_embeds")
        self._validate_prompt_tensor(action_prompt, label="native action")
        self._validate_prompt_tensor(negative_prompt, label="native negative")
        if any(
            left is right
            for left, right in (
                (action_prompt, negative_prompt),
                (action_prompt, self.target_t2v_embeds),
                (action_prompt, self.source_t2v_embeds),
                (negative_prompt, self.target_t2v_embeds),
                (negative_prompt, self.source_t2v_embeds),
            )
        ):
            raise SelfGuidedActionFieldError(
                "native and T2V prompt objects must be independently authenticated"
            )
        guidance_scale = _coerce_scalar(values.get("omega_txt"), label="omega_txt")
        if not math.isclose(
            guidance_scale,
            float(self.config.native_text_guidance_scale),
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise SelfGuidedActionFieldError("sample omega_txt differs from config")
        eta = _coerce_scalar(values.get("eta"), label="eta")
        momentum = _coerce_scalar(values.get("momentum"), label="momentum")
        thresholds = values.get("norm_threshold")
        threshold_value = (
            thresholds[0] if isinstance(thresholds, (list, tuple)) else thresholds
        )
        norm_threshold = _coerce_scalar(
            threshold_value, label="norm_threshold[transformer_1]"
        )
        if eta < 0.0 or momentum < 0.0 or norm_threshold < 0.0:
            raise SelfGuidedActionFieldError("APG parameters must be non-negative")
        apg = _APGParameters(
            guidance_scale=guidance_scale,
            eta=eta,
            norm_threshold=norm_threshold,
            momentum=momentum,
        )
        state = _ActiveSample(
            action_prompt=action_prompt,
            negative_prompt=negative_prompt,
            apg=apg,
            native_momentum=_MomentumBuffer(momentum, branch="native-rv2v"),
            teacher_target_momentum=_MomentumBuffer(
                momentum, branch="t2v-target"
            ),
            teacher_source_momentum=_MomentumBuffer(
                momentum, branch="t2v-source"
            ),
        )
        if len(
            {
                id(state.native_momentum),
                id(state.teacher_target_momentum),
                id(state.teacher_source_momentum),
            }
        ) != 3:
            raise SelfGuidedActionFieldError("APG momentum buffers are not independent")
        self._active = state
        try:
            result = self.original_sample(*args, **kwargs)
            if state.pending_negative is not None or state.pending_step is not None:
                raise SelfGuidedActionFieldError(
                    "sample returned with an incomplete native/scheduler step"
                )
            if state.completed_steps != self.config.expected_steps:
                raise SelfGuidedActionFieldError("sample completed-step count differs")
            expected_teacher_updates = (
                self.config.expected_steps if self.config.effective_scale > 0 else 0
            )
            if state.native_momentum.update_count != self.config.expected_steps:
                raise SelfGuidedActionFieldError("native APG certificate count differs")
            if (
                state.teacher_target_momentum.update_count != expected_teacher_updates
                or state.teacher_source_momentum.update_count != expected_teacher_updates
            ):
                raise SelfGuidedActionFieldError("teacher APG momentum counts differ")
            self.sample_call_count += 1
            return result
        finally:
            self._active = None

    def _set_patch(self, owner: Any, name: str, value: Any) -> None:
        try:
            instance_values = vars(owner)
        except TypeError as error:
            raise SelfGuidedActionFieldError(
                f"cannot reversibly patch {name} owner"
            ) from error
        had_instance = name in instance_values
        previous = instance_values.get(name)
        resolved_before = getattr(owner, name)
        setattr(owner, name, value)
        self._patches.append((owner, name, had_instance, previous, resolved_before))

    def install(self) -> None:
        if self.installed or self.restored or self.finalized:
            raise SelfGuidedActionFieldError("action-field patch lifecycle differs")

        def sample_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_sample(*args, **kwargs)

        def shared_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_shared_step(*args, **kwargs)

        def scheduler_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_scheduler_step(*args, **kwargs)

        for wrapper in (sample_wrapper, shared_wrapper, scheduler_wrapper):
            setattr(wrapper, "_bernini_sgaf_v1", self)
        try:
            self._set_patch(self.diffusion, "sample", sample_wrapper)
            self._set_patch(self.diffusion, "shared_step", shared_wrapper)
            self._set_patch(self.scheduler, "step", scheduler_wrapper)
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
                if require_wrapper_identity and getattr(
                    current, "_bernini_sgaf_v1", None
                ) is not self:
                    errors.append(
                        SelfGuidedActionFieldError(f"{name} changed during SGAF patch")
                    )
                if had_instance:
                    setattr(owner, name, previous)
                else:
                    delattr(owner, name)
                if getattr(owner, name, None) != resolved_before:
                    errors.append(
                        SelfGuidedActionFieldError(f"{name} restoration failed")
                    )
            except Exception as error:
                errors.append(error)
        self._active = None
        if errors:
            raise SelfGuidedActionFieldError(
                f"failed to restore {len(errors)} SGAF wrapper(s)"
            ) from errors[0]

    def restore(self) -> None:
        if not self.installed or self.restored:
            raise SelfGuidedActionFieldError("action-field patch restore differs")
        try:
            self._restore_patches(require_wrapper_identity=True)
        finally:
            self.installed = False
            self.restored = not self._patches

    def finalize(self) -> Mapping[str, Any]:
        if not self.restored or self.finalized:
            raise SelfGuidedActionFieldError("action-field patch finalize differs")
        expected_teacher = (
            3 * self.config.expected_steps if self.config.effective_scale > 0 else 0
        )
        if self.sample_call_count != 1:
            raise SelfGuidedActionFieldError("sample call count differs")
        if self.native_call_count != 2 * self.config.expected_steps:
            raise SelfGuidedActionFieldError("official native forward count differs")
        if self.teacher_call_count != expected_teacher:
            raise SelfGuidedActionFieldError("T2V teacher forward count differs")
        if self.original_scheduler_call_count != self.config.expected_steps:
            raise SelfGuidedActionFieldError("original scheduler call count differs")
        if len(self.trace) != self.config.expected_steps:
            raise SelfGuidedActionFieldError("action-field trace length differs")
        if any(row["native_official_apg_exact_parity"] is not True for row in self.trace):
            raise SelfGuidedActionFieldError("native APG parity certificate differs")
        if self.config.effective_scale == 0.0 and any(
            row["scale_zero_exact_model_output_object"] is not True
            for row in self.trace
        ):
            raise SelfGuidedActionFieldError("scale-zero no-op certificate differs")
        self.finalized = True
        return {
            "sample_calls": self.sample_call_count,
            "native_rv2v_steps": self.original_scheduler_call_count,
            "native_rv2v_forwards": self.native_call_count,
            "frozen_t2v_teacher_forwards": self.teacher_call_count,
            "per_step_call_graph": (
                ["native-negative", "native-action"]
                if self.config.effective_scale == 0.0
                else [
                    "native-negative",
                    "native-action",
                    "t2v-uncond",
                    "t2v-target",
                    "t2v-source",
                ]
            ),
            "original_scheduler_calls": self.original_scheduler_call_count,
            "target_patch_tokens": self.config.target_patch_tokens,
            "target_latent_shape": list(self.config.target_latent_shape),
            "expected_condition_prefix_tokens": (
                self.config.expected_condition_prefix_tokens
            ),
            "expected_hidden_dim": self.config.expected_hidden_dim,
            "effective_scale": self.config.effective_scale,
            "native_text_guidance_scale": self.config.native_text_guidance_scale,
            "teacher_field": "t2v_apg_target_minus_t2v_apg_source",
            "teacher_apg_unconditional": "target_only_negative_prompt",
            "teacher_apg_momentum": "independent_target_and_source_buffers",
            "composition_boundary": "after_official_apg_before_original_scheduler_step",
            "injection_divided_by_omega_txt": False,
            "sigma_source": "exact_active_unipc_scheduler_sigma",
            "scale_zero_exact_noop": self.config.effective_scale == 0.0,
            "sigma_zero_below": self.config.sigma_zero_below,
            "sigma_full_above": self.config.sigma_full_above,
            "maximum_delta_to_native_text_rms": (
                self.config.maximum_delta_to_native_text_rms
            ),
            "trace": list(self.trace),
        }


__all__ = [
    "ActionFieldConfig",
    "NativeRV2VActionFieldPatch",
    "SelfGuidedActionFieldError",
    "clip_action_delta_by_native_text_rms",
    "module_state_hash_certificate",
    "smooth_action_gate",
]
