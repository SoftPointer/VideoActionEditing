#!/usr/bin/env python3
"""Reversible native Bernini R2V-4 / V2V-APG homotopy hook.

The pinned Bernini ``v2v_apg`` loop already materializes the full-source plus
four-reference (``VI``) pack and, while doing so, also materializes the exact
reference-only (``I``) patches.  This hook observes the official two VI
forwards, reuses those same patch results for three additional native
``none/I/TI`` forwards, reconstructs both clean-space APG fields, and replaces
only the velocity delivered to the original UniPC step::

    v_exec = (1 - h(sigma)) * v_official_v2v_apg
             + h(sigma) * v_reference_only_r2v4_apg

``h`` is the pinned FP32 smoothstep in :mod:`native_branch_homotopy_v1`.
Every denoising step therefore has exactly five transformer forwards and one
call to the untouched scheduler.  The hook edits no vendor file, creates no
optimizer, and restores all four instance-level wrappers in reverse order.

The implementation is deliberately fail closed.  It authenticates the
official patch source-id order, generic target/reference token geometry
(``P``/``R``), prompt object identities, timestep/rotary sharing, exact local
parity with the official low APG output, the pinned vendor
``normalized_guidance_chain`` function identity, zero momentum, and exact40
UniPC flow-shift configuration before an alternate velocity can be integrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import inspect
import math
from typing import Any, Callable, Mapping, Optional, Sequence

import native_branch_homotopy_v1 as homotopy
import self_guided_action_field_v1 as sgaf
import tri_branch_unipc as sampler_contract


SCHEMA_VERSION = "bernini-native-branch-homotopy-runtime-v1"
PINNED_BERNINI_COMMIT = sampler_contract.PINNED_BERNINI_COMMIT
PINNED_WAN_DIFFUSION_SHA256 = sampler_contract.PINNED_WAN_DIFFUSION_SHA256
VENDOR_APG_MODULE = "bernini.models.wan_diffusion"
EXPECTED_PATCH_SOURCE_IDS = (1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 0.0)
PER_STEP_FORWARD_ORDER = (
    "low-vi-negative",
    "low-vi-action",
    "high-none-negative",
    "high-i-negative",
    "high-i-action",
)


class NativeBranchHomotopyRuntimeError(RuntimeError):
    """Raised before integration when the native homotopy contract differs."""


def _raise_from_sgaf(error: Exception) -> NativeBranchHomotopyRuntimeError:
    return NativeBranchHomotopyRuntimeError(str(error))


def _scalar(value: Any, *, label: str) -> float:
    try:
        return sgaf._coerce_scalar(value, label=label)
    except Exception as error:
        raise _raise_from_sgaf(error) from error


def _shape(value: Any, *, label: str) -> tuple[int, ...]:
    try:
        return sgaf._shape(value, label=label)
    except Exception as error:
        raise _raise_from_sgaf(error) from error


def _bind(
    callable_object: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        return sgaf._bind_call(callable_object, args, kwargs)
    except Exception as error:
        raise _raise_from_sgaf(error) from error


def _replace(
    callable_object: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    name: str,
    value: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    try:
        return sgaf._replace_argument(
            callable_object,
            args,
            kwargs,
            name=name,
            value=value,
        )
    except Exception as error:
        raise _raise_from_sgaf(error) from error


def _metadata(value: Any, *, label: str) -> tuple[int, ...]:
    try:
        return sgaf._metadata_tuple(value, label=label)
    except Exception as error:
        raise _raise_from_sgaf(error) from error


def _same(left: Any, right: Any, *, label: str) -> None:
    try:
        sgaf._same_object(left, right, label=label)
    except Exception as error:
        raise _raise_from_sgaf(error) from error


def _tensor_rms(value: Any) -> float:
    try:
        return float(sgaf._tensor_rms(value).item())
    except Exception as error:
        raise _raise_from_sgaf(error) from error


@dataclass(frozen=True)
class NativeBranchHomotopyRuntimeConfig:
    """Pinned sampler and generic ``P``/``R`` latent geometry."""

    target_latent_shape: tuple[int, int, int, int, int]
    expected_steps: int = 40
    expected_num_frames: int = 81
    expected_flow_shift: float = 5.0
    omega_image: float = 4.5
    omega_text: float = 4.0
    eta: float = 0.5
    image_norm_threshold: float = 50.0
    text_norm_threshold: float = 50.0
    momentum: float = 0.0
    expected_hidden_dim: int = 1536
    expected_model_id: str = "transformer_1"
    expected_guidance_mode: str = "v2v_apg"

    @property
    def target_patch_tokens(self) -> int:
        _, _, phases, height, width = self.target_latent_shape
        return int(phases * (height // 2) * (width // 2))

    @property
    def reference_patch_tokens(self) -> int:
        _, _, _, height, width = self.target_latent_shape
        return int((height // 2) * (width // 2))

    @property
    def low_vi_tokens(self) -> int:
        return 2 * self.target_patch_tokens + 4 * self.reference_patch_tokens

    @property
    def high_i_tokens(self) -> int:
        return self.target_patch_tokens + 4 * self.reference_patch_tokens

    def validate(self) -> None:
        shape = tuple(self.target_latent_shape)
        if (
            len(shape) != 5
            or any(type(value) is not int or value <= 0 for value in shape)
            or shape[0] != 1
            or shape[1] != 16
            or shape[2] != 21
            or shape[3] % 2
            or shape[4] % 2
        ):
            raise NativeBranchHomotopyRuntimeError(
                "target latent must be exact81 Bernini [1,16,21,even,even]"
            )
        if type(self.expected_steps) is not int or self.expected_steps != 40:
            raise NativeBranchHomotopyRuntimeError("runtime is pinned to exact40")
        if type(self.expected_num_frames) is not int or self.expected_num_frames != 81:
            raise NativeBranchHomotopyRuntimeError("runtime is pinned to exact81")
        if self.expected_hidden_dim != 1536 or self.expected_model_id != "transformer_1":
            raise NativeBranchHomotopyRuntimeError(
                "runtime is pinned to Bernini-R 1.3B transformer_1"
            )
        if self.expected_guidance_mode != "v2v_apg":
            raise NativeBranchHomotopyRuntimeError(
                "runtime requires guidance_mode='v2v_apg'"
            )
        exact = {
            "expected_flow_shift": (self.expected_flow_shift, 5.0),
            "omega_image": (self.omega_image, 4.5),
            "omega_text": (self.omega_text, 4.0),
            "eta": (self.eta, 0.5),
            "image_norm_threshold": (self.image_norm_threshold, 50.0),
            "text_norm_threshold": (self.text_norm_threshold, 50.0),
            "momentum": (self.momentum, 0.0),
        }
        for label, (observed, wanted) in exact.items():
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not math.isfinite(float(observed))
                or float(observed) != wanted
            ):
                raise NativeBranchHomotopyRuntimeError(
                    f"{label} must be exactly {wanted}"
                )


@dataclass(frozen=True)
class _PatchResult:
    source_id: float
    input_value: Any
    latent: Any
    rotary: Any


@dataclass(frozen=True)
class _ForwardResult:
    name: str
    values: Mapping[str, Any]
    prediction: Any
    target_tail: Any


@dataclass
class _ActiveSample:
    low_action_prompt: Any
    low_negative_prompt: Any
    high_action_prompt: Any
    source_video: Any
    references: tuple[Any, ...]
    source_patch_value: Any
    reference_patch_values: tuple[Any, ...]
    low_momentum: Any
    high_image_momentum: Any
    high_text_momentum: Any
    completed_steps: int = 0
    schedule_preflight: Optional[Mapping[str, Any]] = None
    patch_results: list[_PatchResult] = field(default_factory=list)
    low_forwards: list[_ForwardResult] = field(default_factory=list)
    high_forwards: list[_ForwardResult] = field(default_factory=list)


def _resolve_vendor_apg_symbols() -> tuple[Callable[..., Any], type[Any]]:
    """Resolve exact pinned vendor objects and reject aliases or copies."""

    try:
        module = importlib.import_module(VENDOR_APG_MODULE)
    except Exception as error:
        raise NativeBranchHomotopyRuntimeError(
            f"cannot import pinned {VENDOR_APG_MODULE}"
        ) from error
    helper = getattr(module, "normalized_guidance_chain", None)
    momentum_class = getattr(module, "MomentumBuffer", None)
    if not callable(helper) or not isinstance(momentum_class, type):
        raise NativeBranchHomotopyRuntimeError(
            "pinned vendor APG symbols are unavailable"
        )
    if (
        getattr(helper, "__module__", None) != VENDOR_APG_MODULE
        or getattr(helper, "__name__", None) != "normalized_guidance_chain"
        or inspect.getmodule(helper) is not module
        or getattr(module, "normalized_guidance_chain") is not helper
        or getattr(momentum_class, "__module__", None) != VENDOR_APG_MODULE
        or getattr(momentum_class, "__name__", None) != "MomentumBuffer"
        or inspect.getmodule(momentum_class) is not module
        or getattr(module, "MomentumBuffer") is not momentum_class
    ):
        raise NativeBranchHomotopyRuntimeError(
            "vendor APG module/function identity differs"
        )
    try:
        helper_parameters = tuple(inspect.signature(helper).parameters)
        momentum_parameters = tuple(inspect.signature(momentum_class).parameters)
    except (TypeError, ValueError) as error:
        raise NativeBranchHomotopyRuntimeError(
            "vendor APG signatures are not inspectable"
        ) from error
    if helper_parameters != (
        "pred_uncond",
        "preds",
        "scales",
        "momentum_buffers",
        "eta",
        "norm_thresholds",
    ) or momentum_parameters != ("momentum",):
        raise NativeBranchHomotopyRuntimeError("vendor APG signature differs")
    return helper, momentum_class


class NativeBranchHomotopyRuntimePatch:
    """One-sample reversible five-forward native homotopy adapter."""

    def __init__(
        self,
        diffusion: Any,
        *,
        r2v_action_prompt_embeds: Any,
        config: NativeBranchHomotopyRuntimeConfig,
        expected_bernini_commit: str = PINNED_BERNINI_COMMIT,
        observed_wan_diffusion_sha256: str = PINNED_WAN_DIFFUSION_SHA256,
    ) -> None:
        import torch

        config.validate()
        if expected_bernini_commit != PINNED_BERNINI_COMMIT:
            raise NativeBranchHomotopyRuntimeError("Bernini revision differs")
        if observed_wan_diffusion_sha256 != PINNED_WAN_DIFFUSION_SHA256:
            raise NativeBranchHomotopyRuntimeError("wan_diffusion.py bytes differ")
        if (
            not isinstance(r2v_action_prompt_embeds, torch.Tensor)
            or r2v_action_prompt_embeds.ndim != 3
            or r2v_action_prompt_embeds.shape[0] != 1
            or r2v_action_prompt_embeds.shape[1] <= 0
            or r2v_action_prompt_embeds.numel() <= 0
            or r2v_action_prompt_embeds.requires_grad
            or r2v_action_prompt_embeds.grad_fn is not None
            or not bool(torch.isfinite(r2v_action_prompt_embeds).all().item())
        ):
            raise NativeBranchHomotopyRuntimeError(
                "official R2V action embedding geometry differs"
            )
        try:
            core = sampler_contract.resolve_diffusion_core(diffusion)
        except Exception as error:
            raise NativeBranchHomotopyRuntimeError(str(error)) from error
        transformer = getattr(core, "transformer", None)
        scheduler = getattr(core, "scheduler", None)
        originals = {
            "sample": getattr(core, "sample", None),
            "shared_step": getattr(core, "shared_step", None),
            "patch_vae_latent": getattr(transformer, "patch_vae_latent", None),
            "scheduler.step": getattr(scheduler, "step", None),
        }
        if any(not callable(value) for value in originals.values()):
            raise NativeBranchHomotopyRuntimeError(
                "pinned Bernini sampler call surface differs"
            )
        if getattr(core, "use_unipc", None) is not True:
            raise NativeBranchHomotopyRuntimeError("runtime requires native UniPC")
        if getattr(core, "transformer_2", None) is not None:
            raise NativeBranchHomotopyRuntimeError(
                "runtime supports only single-expert Bernini-R 1.3B"
            )
        transformer_config = getattr(transformer, "config", None)
        if transformer_config is None:
            raise NativeBranchHomotopyRuntimeError("transformer config is unavailable")

        def config_value(name: str) -> Any:
            value = getattr(transformer_config, name, None)
            if value is None and isinstance(transformer_config, Mapping):
                value = transformer_config.get(name)
            return value

        heads = config_value("num_attention_heads")
        head_dim = config_value("attention_head_dim")
        in_channels = config_value("in_channels")
        if (
            type(heads) is not int
            or type(head_dim) is not int
            or heads * head_dim != config.expected_hidden_dim
            or in_channels != 16
        ):
            raise NativeBranchHomotopyRuntimeError(
                "transformer hidden/input geometry differs"
            )
        try:
            sampler_contract._validate_scheduler_contract(
                scheduler,
                expected_flow_shift=config.expected_flow_shift,
            )
        except Exception as error:
            raise NativeBranchHomotopyRuntimeError(str(error)) from error
        for owner, name in (
            (core, "sample"),
            (core, "shared_step"),
            (transformer, "patch_vae_latent"),
            (scheduler, "step"),
        ):
            try:
                if name in vars(owner):
                    raise NativeBranchHomotopyRuntimeError(
                        f"refusing stacked instance override on {name}"
                    )
            except TypeError as error:
                raise NativeBranchHomotopyRuntimeError(
                    f"cannot inspect {name} owner"
                ) from error
        for original in originals.values():
            if getattr(original, "_bernini_native_branch_homotopy_v1", None) is not None:
                raise NativeBranchHomotopyRuntimeError(
                    "native homotopy wrapper is already installed"
                )
        if getattr(transformer, "training", False) is not False:
            raise NativeBranchHomotopyRuntimeError("transformer must remain in eval mode")
        named_parameters = getattr(transformer, "named_parameters", None)
        if not callable(named_parameters):
            raise NativeBranchHomotopyRuntimeError(
                "transformer freeze surface is unavailable"
            )
        for name, parameter in named_parameters():
            if bool(getattr(parameter, "requires_grad", False)) or getattr(
                parameter, "grad", None
            ) is not None:
                raise NativeBranchHomotopyRuntimeError(
                    f"transformer parameter {name} is not freeze-safe"
                )

        vendor_chain, vendor_momentum_class = _resolve_vendor_apg_symbols()
        self.diffusion = core
        self.transformer = transformer
        self.scheduler = scheduler
        self.r2v_action_prompt_embeds = r2v_action_prompt_embeds
        self.config = config
        self.vendor_chain = vendor_chain
        self.vendor_momentum_class = vendor_momentum_class
        self.original_sample = originals["sample"]
        self.original_shared_step = originals["shared_step"]
        self.original_patch_vae_latent = originals["patch_vae_latent"]
        self.original_scheduler_step = originals["scheduler.step"]
        self._patches: list[tuple[Any, str, bool, Any, Any]] = []
        self._active: Optional[_ActiveSample] = None
        self.installed = False
        self.restored = False
        self.finalized = False
        self.sample_call_count = 0
        self.patch_call_count = 0
        self.low_forward_count = 0
        self.high_forward_count = 0
        self.original_scheduler_call_count = 0
        self.schedule_preflight: Optional[Mapping[str, Any]] = None
        self.trace: list[dict[str, Any]] = []

    def _validate_prompt(self, value: Any, *, label: str) -> int:
        import torch

        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 3
            or value.shape[0] != 1
            or value.shape[1] <= 0
            or value.device != self.r2v_action_prompt_embeds.device
            or value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())
        ):
            raise NativeBranchHomotopyRuntimeError(f"{label} prompt geometry differs")
        return int(value.shape[1])

    def _set_patch(self, owner: Any, name: str, value: Any) -> None:
        try:
            instance = vars(owner)
        except TypeError as error:
            raise NativeBranchHomotopyRuntimeError(
                f"cannot reversibly patch {name} owner"
            ) from error
        had_instance = name in instance
        previous = instance.get(name)
        resolved_before = getattr(owner, name)
        setattr(owner, name, value)
        self._patches.append((owner, name, had_instance, previous, resolved_before))

    def install(self) -> None:
        if self.installed or self.restored or self.finalized:
            raise NativeBranchHomotopyRuntimeError("homotopy patch lifecycle differs")

        def sample_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_sample(*args, **kwargs)

        def shared_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_shared_step(*args, **kwargs)

        def patch_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_patch_vae_latent(*args, **kwargs)

        def scheduler_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_scheduler_step(*args, **kwargs)

        for wrapper in (sample_wrapper, shared_wrapper, patch_wrapper, scheduler_wrapper):
            setattr(wrapper, "_bernini_native_branch_homotopy_v1", self)
        try:
            self._set_patch(self.transformer, "patch_vae_latent", patch_wrapper)
            self._set_patch(self.diffusion, "shared_step", shared_wrapper)
            self._set_patch(self.scheduler, "step", scheduler_wrapper)
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
                if require_wrapper_identity and getattr(
                    current, "_bernini_native_branch_homotopy_v1", None
                ) is not self:
                    errors.append(
                        NativeBranchHomotopyRuntimeError(
                            f"{name} changed during homotopy patch"
                        )
                    )
                if had_instance:
                    setattr(owner, name, previous)
                else:
                    delattr(owner, name)
                if getattr(owner, name, None) != resolved_before:
                    errors.append(
                        NativeBranchHomotopyRuntimeError(
                            f"{name} restoration failed"
                        )
                    )
            except Exception as error:
                errors.append(error)
        self._active = None
        if errors:
            raise NativeBranchHomotopyRuntimeError(
                f"failed to restore {len(errors)} homotopy wrapper(s)"
            ) from errors[0]

    def restore(self) -> None:
        if not self.installed or self.restored:
            raise NativeBranchHomotopyRuntimeError("homotopy patch restore differs")
        try:
            self._restore_patches(require_wrapper_identity=True)
        finally:
            self.installed = False
            self.restored = not self._patches

    def _validate_sample_contract(self, values: Mapping[str, Any]) -> _ActiveSample:
        import torch

        refs_value = values.get("multi_image_vae_latents")
        references = tuple(refs_value) if isinstance(refs_value, (list, tuple)) else ()
        videos = values.get("multi_video_vae_latents")
        thresholds = values.get("norm_threshold")
        if not isinstance(thresholds, (list, tuple)) or len(thresholds) != 2:
            raise NativeBranchHomotopyRuntimeError(
                "sample norm_threshold must be explicit (50, 50)"
            )
        if (
            values.get("guidance_mode") != self.config.expected_guidance_mode
            or values.get("num_frames") != self.config.expected_num_frames
            or values.get("num_inference_steps") != self.config.expected_steps
            or _scalar(values.get("flow_shift"), label="flow_shift")
            != self.config.expected_flow_shift
            or _scalar(values.get("omega_img"), label="omega_img")
            != self.config.omega_image
            or _scalar(values.get("omega_txt"), label="omega_txt")
            != self.config.omega_text
            or _scalar(values.get("eta"), label="eta") != self.config.eta
            or _scalar(values.get("momentum"), label="momentum")
            != self.config.momentum
            or _scalar(thresholds[0], label="norm_threshold[0]")
            != self.config.image_norm_threshold
            or _scalar(thresholds[1], label="norm_threshold[1]")
            != self.config.text_norm_threshold
            or not isinstance(videos, (list, tuple))
            or len(videos) != 1
            or len(references) != 4
            or values.get("image_vae_latents") is not None
        ):
            raise NativeBranchHomotopyRuntimeError(
                "native v2v_apg sample/condition contract differs"
            )
        source_video = videos[0]
        conditioning_values = (source_video, *references)
        if any(
            not isinstance(value, torch.Tensor)
            or value.ndim != 5
            or value.shape[0] != 1
            or value.shape[1] != 16
            or value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())
            for value in conditioning_values
        ):
            raise NativeBranchHomotopyRuntimeError(
                "source/reference latent geometry differs"
            )
        if tuple(int(value) for value in source_video.shape) != self.config.target_latent_shape:
            raise NativeBranchHomotopyRuntimeError(
                "source video does not share target latent geometry"
            )
        expected_reference_shape = (
            1,
            16,
            1,
            self.config.target_latent_shape[3],
            self.config.target_latent_shape[4],
        )
        if any(
            tuple(int(value) for value in reference.shape)
            != expected_reference_shape
            for reference in references
        ):
            raise NativeBranchHomotopyRuntimeError(
                "four references do not share generic R geometry"
            )
        transformer_dtype = getattr(self.transformer, "dtype", None)
        if transformer_dtype is None:
            raise NativeBranchHomotopyRuntimeError("transformer dtype is unavailable")
        source_patch_value = source_video.to(dtype=transformer_dtype)
        reference_patch_values = tuple(
            reference.to(dtype=transformer_dtype) for reference in references
        )
        low_action = values.get("prompt_embeds")
        low_negative = values.get("uncond_prompt_embeds")
        self._validate_prompt(low_action, label="low V2V action")
        self._validate_prompt(low_negative, label="low V2V negative")
        self._validate_prompt(self.r2v_action_prompt_embeds, label="high R2V action")
        if (
            low_action is low_negative
            or low_action is self.r2v_action_prompt_embeds
            or low_negative is self.r2v_action_prompt_embeds
        ):
            raise NativeBranchHomotopyRuntimeError(
                "low/high prompt objects must be independently authenticated"
            )
        low_momentum = sgaf._MomentumBuffer(0.0, branch="low-official-v2v-apg")
        high_image_momentum = self.vendor_momentum_class(0.0)
        high_text_momentum = self.vendor_momentum_class(0.0)
        if (
            high_image_momentum is high_text_momentum
            or _scalar(getattr(high_image_momentum, "momentum", None), label="high I momentum")
            != 0.0
            or _scalar(getattr(high_text_momentum, "momentum", None), label="high T momentum")
            != 0.0
        ):
            raise NativeBranchHomotopyRuntimeError(
                "high R2V APG momentum buffers differ"
            )
        return _ActiveSample(
            low_action_prompt=low_action,
            low_negative_prompt=low_negative,
            high_action_prompt=self.r2v_action_prompt_embeds,
            source_video=source_video,
            references=references,
            source_patch_value=source_patch_value,
            reference_patch_values=reference_patch_values,
            low_momentum=low_momentum,
            high_image_momentum=high_image_momentum,
            high_text_momentum=high_text_momentum,
        )

    def _wrapped_sample(self, *args: Any, **kwargs: Any) -> Any:
        if self._active is not None or self.sample_call_count != 0:
            raise NativeBranchHomotopyRuntimeError(
                "homotopy patch permits exactly one non-nested sample"
            )
        if self.diffusion.scheduler is not self.scheduler:
            raise NativeBranchHomotopyRuntimeError("diffusion.scheduler changed")
        values = _bind(self.original_sample, args, kwargs)
        state = self._validate_sample_contract(values)
        self._active = state
        try:
            result = self.original_sample(*args, **kwargs)
            if (
                state.completed_steps != self.config.expected_steps
                or state.schedule_preflight is None
                or state.patch_results
                or state.low_forwards
                or state.high_forwards
            ):
                raise NativeBranchHomotopyRuntimeError(
                    "sample returned with an incomplete homotopy step"
                )
            if state.low_momentum.update_count != self.config.expected_steps:
                raise NativeBranchHomotopyRuntimeError(
                    "low APG parity momentum count differs"
                )
            self.schedule_preflight = dict(state.schedule_preflight)
            self.sample_call_count += 1
            return result
        finally:
            self._active = None

    def _validate_live_exact40_schedule(self) -> Mapping[str, Any]:
        """Bind the live scheduler to the preregistered shift-5 coordinates.

        This executes on the first ``patch_vae_latent`` call, after pinned
        Bernini has called ``scheduler.set_timesteps(40)`` but before the first
        transformer forward or original scheduler integration.
        """

        import torch

        try:
            import source_self_native_ref_contrastive_v3 as schedule_contract
        except Exception as error:
            raise NativeBranchHomotopyRuntimeError(
                "cannot import pinned native UniPC40 schedule contract"
            ) from error
        expected_timesteps = tuple(schedule_contract.NATIVE_UNIPC40_TIMESTEPS)
        expected_forward_sigmas = tuple(schedule_contract.NATIVE_UNIPC40_SIGMAS)
        pinned_digest = schedule_contract.PINNED_NATIVE_UNIPC40_SCHEDULE_DIGEST
        if (
            len(expected_timesteps) != 40
            or len(expected_forward_sigmas) != 40
            or schedule_contract.native_unipc40_schedule_receipt().get("digest")
            != pinned_digest
        ):
            raise NativeBranchHomotopyRuntimeError(
                "pinned native UniPC40 schedule registry differs"
            )
        timesteps = getattr(self.scheduler, "timesteps", None)
        sigmas = getattr(self.scheduler, "sigmas", None)
        if (
            not isinstance(timesteps, torch.Tensor)
            or timesteps.ndim != 1
            or int(timesteps.numel()) != 40
            or timesteps.device.type != "cpu"
            or not isinstance(sigmas, torch.Tensor)
            or sigmas.ndim != 1
            or int(sigmas.numel()) != 41
            or sigmas.device.type != "cpu"
            or sigmas.dtype != torch.float32
            or not bool(torch.isfinite(timesteps).all().item())
            or not bool(torch.isfinite(sigmas).all().item())
        ):
            raise NativeBranchHomotopyRuntimeError(
                "live exact40 timestep/sigma tensor geometry differs"
            )
        live_timesteps = tuple(
            int(_scalar(value, label="live scheduler timestep"))
            for value in timesteps
        )
        if any(
            _scalar(value, label="live scheduler timestep") != float(integer)
            for value, integer in zip(timesteps, live_timesteps)
        ):
            raise NativeBranchHomotopyRuntimeError(
                "live scheduler timesteps are not exact integers"
            )
        live_sigmas = tuple(float(value.item()) for value in sigmas[:40])
        terminal_sigma = float(sigmas[40].item())
        if (
            live_timesteps != expected_timesteps
            or live_sigmas != expected_forward_sigmas
            or terminal_sigma != 0.0
        ):
            raise NativeBranchHomotopyRuntimeError(
                "live exact40 shift-5 timestep/sigma schedule differs"
            )
        live_value = {
            "scheduler": "UniPCMultistepScheduler",
            "prediction_type": "flow_prediction",
            "use_flow_sigmas": True,
            "flow_shift": 5.0,
            "num_inference_steps": 40,
            "model_forward_count": 40,
            "terminal_sigma_excluded_from_training": terminal_sigma,
            "timesteps": list(live_timesteps),
            "sigma_float64_hex": [float(value).hex() for value in live_sigmas],
            "sampling_distribution": "uniform_without_replacement_over_exact40_per_cycle",
        }
        live_digest = schedule_contract.object_sha256(live_value)
        if live_digest != pinned_digest:
            raise NativeBranchHomotopyRuntimeError(
                "live exact40 shift-5 schedule digest differs"
            )
        return {
            "validated_before_first_transformer_forward": True,
            "pinned_schedule_digest": pinned_digest,
            "live_schedule_digest": live_digest,
            "timestep_count": len(live_timesteps),
            "model_forward_sigma_count": len(live_sigmas),
            "sigma_count_including_terminal": int(sigmas.numel()),
            "terminal_sigma": terminal_sigma,
            "flow_shift": self.config.expected_flow_shift,
        }

    def _wrapped_patch_vae_latent(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        state = self._active
        if state is None:
            raise NativeBranchHomotopyRuntimeError(
                "patch_vae_latent ran outside authenticated sample"
            )
        if state.schedule_preflight is None:
            state.schedule_preflight = self._validate_live_exact40_schedule()
        if state.low_forwards or state.high_forwards:
            raise NativeBranchHomotopyRuntimeError(
                "native patch call arrived after transformer forwarding began"
            )
        index = len(state.patch_results)
        if index >= len(EXPECTED_PATCH_SOURCE_IDS):
            raise NativeBranchHomotopyRuntimeError(
                "too many patch_vae_latent calls before scheduler.step"
            )
        values = _bind(self.original_patch_vae_latent, args, kwargs)
        source_id = _scalar(values.get("source_id"), label="patch source_id")
        if source_id != EXPECTED_PATCH_SOURCE_IDS[index]:
            raise NativeBranchHomotopyRuntimeError(
                "native patch_vae_latent source-id order differs"
            )
        input_value = values.get("value")
        if input_value is None:
            # The pinned method calls the first argument ``vae_latent``.  Fakes
            # often call it ``value``; accept either only through signature bind.
            input_value = values.get("vae_latent")
        if input_value is None:
            input_value = next(
                (
                    value
                    for name, value in values.items()
                    if name not in ("self", "source_id")
                ),
                None,
            )
        if index == 0:
            expected_input = state.source_patch_value
            if (
                not isinstance(input_value, torch.Tensor)
                or input_value.shape != expected_input.shape
                or input_value.device != expected_input.device
                or input_value.dtype != expected_input.dtype
                or not torch.equal(input_value, expected_input)
            ):
                raise NativeBranchHomotopyRuntimeError(
                    "source-video patch input differs after native dtype conversion"
                )
        if index in (1, 2, 3, 4, 5, 6, 7, 8):
            reference_index = (index - 1) // 2
            expected_input = state.reference_patch_values[reference_index]
            if (
                not isinstance(input_value, torch.Tensor)
                or input_value.shape != expected_input.shape
                or input_value.device != expected_input.device
                or input_value.dtype != expected_input.dtype
                or not torch.equal(input_value, expected_input)
            ):
                raise NativeBranchHomotopyRuntimeError(
                    "reference patch input/order differs after native dtype conversion"
                )
            if index % 2 == 0:
                previous_input = state.patch_results[-1].input_value
                if (
                    input_value.shape != previous_input.shape
                    or input_value.device != previous_input.device
                    or input_value.dtype != previous_input.dtype
                    or not torch.equal(input_value, previous_input)
                ):
                    raise NativeBranchHomotopyRuntimeError(
                        "paired VI/I reference patch inputs differ"
                    )
        result = self.original_patch_vae_latent(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise NativeBranchHomotopyRuntimeError(
                "patch_vae_latent must return (latent, rotary)"
            )
        latent, rotary = result
        expected_tokens = (
            self.config.target_patch_tokens
            if index in (0, 9)
            else self.config.reference_patch_tokens
        )
        if (
            not isinstance(latent, torch.Tensor)
            or _shape(latent, label="patched latent")
            != (1, expected_tokens, self.config.expected_hidden_dim)
            or latent.requires_grad
            or latent.grad_fn is not None
            or not bool(torch.isfinite(latent).all().item())
            or not isinstance(rotary, torch.Tensor)
            or rotary.ndim != 4
            or rotary.shape[0] != 1
            or rotary.shape[2] != expected_tokens
            or rotary.requires_grad
            or rotary.grad_fn is not None
            or not bool(torch.isfinite(rotary).all().item())
        ):
            raise NativeBranchHomotopyRuntimeError(
                "patch_vae_latent output P/R geometry differs"
            )
        state.patch_results.append(
            _PatchResult(source_id, input_value, latent, rotary)
        )
        self.patch_call_count += 1
        return result

    def _captured_packs(self, state: _ActiveSample) -> Mapping[str, tuple[Any, Any]]:
        import torch

        if len(state.patch_results) != len(EXPECTED_PATCH_SOURCE_IDS):
            raise NativeBranchHomotopyRuntimeError(
                "transformer forward began before ten native patches closed"
            )
        parts = state.patch_results
        video = parts[0]
        vi_refs = [parts[index] for index in (1, 3, 5, 7)]
        i_refs = [parts[index] for index in (2, 4, 6, 8)]
        target = parts[9]

        def assemble(items: Sequence[_PatchResult]) -> tuple[Any, Any]:
            latent = torch.cat([item.latent for item in (*items, target)], dim=1)
            rotary = torch.cat([item.rotary for item in (*items, target)], dim=2)
            return latent, rotary

        return {
            "none": assemble(()),
            "i": assemble(i_refs),
            "vi": assemble((video, *vi_refs)),
        }

    def _validate_low_call(
        self,
        state: _ActiveSample,
        values: Mapping[str, Any],
        *,
        index: int,
        vi_pack: tuple[Any, Any],
    ) -> None:
        import torch

        branch = PER_STEP_FORWARD_ORDER[index]
        prompt = state.low_negative_prompt if index == 0 else state.low_action_prompt
        if values.get("model_id") != self.config.expected_model_id:
            raise NativeBranchHomotopyRuntimeError(f"{branch} model_id differs")
        if values.get("cond_embeds") is not prompt:
            raise NativeBranchHomotopyRuntimeError(
                f"{branch} prompt object differs"
            )
        prompt_length = self._validate_prompt(prompt, label=branch)
        noisy = values.get("noisy_latents")
        rotary = values.get("rotary_embs")
        timestep = values.get("timesteps")
        if (
            not isinstance(noisy, torch.Tensor)
            or _shape(noisy, label=f"{branch} noisy")
            != (1, self.config.low_vi_tokens, self.config.expected_hidden_dim)
            or not torch.equal(noisy, vi_pack[0])
            or not isinstance(rotary, torch.Tensor)
            or rotary.ndim != 4
            or rotary.shape[0] != 1
            or rotary.shape[2] != self.config.low_vi_tokens
            or not torch.equal(rotary, vi_pack[1])
            or _metadata(values.get("batch_vae_seqlen"), label=f"{branch} VAE length")
            != (self.config.low_vi_tokens,)
            or _metadata(values.get("batch_text_seqlen"), label=f"{branch} text length")
            != (prompt_length,)
            or not isinstance(timestep, torch.Tensor)
            or timestep.shape != (1,)
            or not bool(torch.isfinite(timestep).all().item())
        ):
            raise NativeBranchHomotopyRuntimeError(
                f"{branch} VI token/timestep/rotary geometry differs"
            )
        if index == 1:
            first = state.low_forwards[0].values
            for name in ("noisy_latents", "timesteps", "rotary_embs"):
                _same(first.get(name), values.get(name), label=f"low negative/action {name}")

    def _validate_prediction(
        self,
        prediction: Any,
        *,
        total_tokens: int,
        target_tokens: int,
        label: str,
    ) -> Any:
        import torch

        expected_channels = self.config.target_latent_shape[1] * 4
        if (
            not isinstance(prediction, torch.Tensor)
            or _shape(prediction, label=f"{label} prediction")
            != (1, total_tokens, expected_channels)
            or prediction.requires_grad
            or prediction.grad_fn is not None
            or not bool(torch.isfinite(prediction).all().item())
        ):
            raise NativeBranchHomotopyRuntimeError(
                f"{label} transformer prediction geometry differs"
            )
        tail = prediction[:, -target_tokens:, :]
        if _shape(tail, label=f"{label} target tail") != (
            1,
            target_tokens,
            expected_channels,
        ):
            raise NativeBranchHomotopyRuntimeError(
                f"{label} target-tail geometry differs"
            )
        return tail

    def _high_query(
        self,
        state: _ActiveSample,
        *,
        name: str,
        pack: tuple[Any, Any],
        prompt: Any,
        timestep: Any,
    ) -> _ForwardResult:
        total_tokens = int(pack[0].shape[1])
        prompt_length = self._validate_prompt(prompt, label=name)
        call_kwargs = {
            "model_id": self.config.expected_model_id,
            "noisy_latents": pack[0],
            "timesteps": timestep,
            "cond_embeds": prompt,
            "rotary_embs": pack[1],
            "batch_vae_seqlen": [total_tokens],
            "batch_text_seqlen": [prompt_length],
        }
        bound = _bind(self.original_shared_step, (), call_kwargs)
        for key, expected in (
            ("noisy_latents", pack[0]),
            ("timesteps", timestep),
            ("cond_embeds", prompt),
            ("rotary_embs", pack[1]),
        ):
            _same(bound.get(key), expected, label=f"{name} {key}")
        prediction = self.original_shared_step(**call_kwargs)
        tail = self._validate_prediction(
            prediction,
            total_tokens=total_tokens,
            target_tokens=self.config.target_patch_tokens,
            label=name,
        )
        self.high_forward_count += 1
        return _ForwardResult(name, bound, prediction, tail)

    def _wrapped_shared_step(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise NativeBranchHomotopyRuntimeError(
                "shared_step ran outside authenticated sample"
            )
        if state.high_forwards:
            raise NativeBranchHomotopyRuntimeError(
                "unexpected shared_step after high R2V closure"
            )
        index = len(state.low_forwards)
        if index >= 2:
            raise NativeBranchHomotopyRuntimeError(
                "more than two official VI forwards occurred"
            )
        packs = self._captured_packs(state)
        values = _bind(self.original_shared_step, args, kwargs)
        self._validate_low_call(state, values, index=index, vi_pack=packs["vi"])
        prediction = self.original_shared_step(*args, **kwargs)
        tail = self._validate_prediction(
            prediction,
            total_tokens=self.config.low_vi_tokens,
            target_tokens=self.config.target_patch_tokens,
            label=PER_STEP_FORWARD_ORDER[index],
        )
        state.low_forwards.append(
            _ForwardResult(PER_STEP_FORWARD_ORDER[index], values, prediction, tail)
        )
        self.low_forward_count += 1
        if index == 1:
            timestep = values["timesteps"]
            state.high_forwards.extend(
                (
                    self._high_query(
                        state,
                        name=PER_STEP_FORWARD_ORDER[2],
                        pack=packs["none"],
                        prompt=state.low_negative_prompt,
                        timestep=timestep,
                    ),
                    self._high_query(
                        state,
                        name=PER_STEP_FORWARD_ORDER[3],
                        pack=packs["i"],
                        prompt=state.low_negative_prompt,
                        timestep=timestep,
                    ),
                    self._high_query(
                        state,
                        name=PER_STEP_FORWARD_ORDER[4],
                        pack=packs["i"],
                        prompt=state.high_action_prompt,
                        timestep=timestep,
                    ),
                )
            )
        return prediction

    def _high_r2v4_velocity(
        self,
        state: _ActiveSample,
        *,
        sample: Any,
        sigma: Any,
        official: Any,
    ) -> Any:
        import torch

        if len(state.high_forwards) != 3:
            raise NativeBranchHomotopyRuntimeError("high R2V forward triplet is incomplete")
        sample_spatial = sgaf._packed_to_spatial(
            sample,
            self.config.target_latent_shape,
        )
        clean = []
        for branch in state.high_forwards:
            velocity = sgaf._packed_to_spatial(
                branch.target_tail,
                self.config.target_latent_shape,
            )
            clean.append(sample_spatial - sigma * velocity)
        guided_clean = self.vendor_chain(
            pred_uncond=clean[0],
            preds=[clean[1], clean[2]],
            scales=[self.config.omega_image, self.config.omega_text],
            momentum_buffers=[state.high_image_momentum, state.high_text_momentum],
            eta=self.config.eta,
            norm_thresholds=[
                self.config.image_norm_threshold,
                self.config.text_norm_threshold,
            ],
        )
        if (
            not isinstance(guided_clean, torch.Tensor)
            or tuple(guided_clean.shape) != tuple(sample_spatial.shape)
            or guided_clean.device != sample_spatial.device
            or not bool(torch.isfinite(guided_clean).all().item())
        ):
            raise NativeBranchHomotopyRuntimeError(
                "vendor normalized_guidance_chain output differs"
            )
        high = sgaf._spatial_to_packed(
            (sample_spatial - guided_clean) / sigma,
            self.config.target_latent_shape,
        )
        if high.device != official.device or high.dtype != official.dtype:
            raise NativeBranchHomotopyRuntimeError(
                "high R2V-4 APG scheduler-bound dtype/device differs"
            )
        return high

    def _wrapped_scheduler_step(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        state = self._active
        if state is None:
            raise NativeBranchHomotopyRuntimeError(
                "scheduler.step ran outside authenticated sample"
            )
        if (
            len(state.patch_results) != 10
            or tuple(item.source_id for item in state.patch_results)
            != EXPECTED_PATCH_SOURCE_IDS
            or len(state.low_forwards) != 2
            or len(state.high_forwards) != 3
        ):
            raise NativeBranchHomotopyRuntimeError(
                "scheduler.step arrived before five-forward closure"
            )
        official = sgaf._extract_argument(args, kwargs, index=0, name="model_output")
        timestep = sgaf._extract_argument(args, kwargs, index=1, name="timestep")
        sample = sgaf._extract_argument(args, kwargs, index=2, name="sample")
        try:
            sgaf._certify_expanded_timestep(
                state.low_forwards[1].values["timesteps"],
                timestep,
            )
        except Exception as error:
            raise _raise_from_sgaf(error) from error
        expected_shape = (
            1,
            self.config.target_patch_tokens,
            self.config.target_latent_shape[1] * 4,
        )
        for label, value in (("official model_output", official), ("scheduler sample", sample)):
            if (
                not isinstance(value, torch.Tensor)
                or _shape(value, label=label) != expected_shape
                or not bool(torch.isfinite(value).all().item())
            ):
                raise NativeBranchHomotopyRuntimeError(f"{label} geometry differs")
        if official.device != sample.device:
            raise NativeBranchHomotopyRuntimeError("official output/sample devices differ")
        expected_target_patch_input = sgaf._packed_to_spatial(
            sample,
            self.config.target_latent_shape,
        ).to(dtype=self.transformer.dtype)
        observed_target_patch_input = state.patch_results[9].input_value
        if (
            not isinstance(observed_target_patch_input, torch.Tensor)
            or observed_target_patch_input.shape != expected_target_patch_input.shape
            or observed_target_patch_input.device != expected_target_patch_input.device
            or observed_target_patch_input.dtype != expected_target_patch_input.dtype
            or not torch.equal(observed_target_patch_input, expected_target_patch_input)
        ):
            raise NativeBranchHomotopyRuntimeError(
                "captured target patch input differs from scheduler sample"
            )
        try:
            step_index, sigma, sigma_float = sgaf._resolve_sigma(
                self.scheduler,
                timestep,
            )
        except Exception as error:
            raise _raise_from_sgaf(error) from error
        if step_index != state.completed_steps:
            raise NativeBranchHomotopyRuntimeError(
                "scheduler step index differs from homotopy state"
            )
        if (
            not isinstance(sigma, torch.Tensor)
            or sigma.ndim != 0
            or sigma.device.type != "cpu"
            or sigma.dtype != torch.float32
        ):
            raise NativeBranchHomotopyRuntimeError(
                "active UniPC sigma must remain a CPU fp32 scalar"
            )
        low_parameters = sgaf._APGParameters(
            guidance_scale=self.config.omega_text,
            eta=self.config.eta,
            norm_threshold=self.config.image_norm_threshold,
            momentum=0.0,
        )
        rebuilt_low = sgaf._guided_velocity(
            sample,
            state.low_forwards[0].target_tail,
            state.low_forwards[1].target_tail,
            sigma,
            shape=self.config.target_latent_shape,
            parameters=low_parameters,
            momentum_buffer=state.low_momentum,
            output_like=official,
        )
        parity_delta = rebuilt_low.float() - official.float()
        parity_rms = _tensor_rms(parity_delta)
        parity_max = float(parity_delta.abs().max().item())
        if not torch.equal(rebuilt_low, official):
            raise NativeBranchHomotopyRuntimeError(
                "locally rebuilt low V2V APG differs from official model_output: "
                f"max_abs={parity_max:.9g} rms={parity_rms:.9g}"
            )

        high = self._high_r2v4_velocity(
            state,
            sample=sample,
            sigma=sigma,
            official=official,
        )
        try:
            combined = homotopy.native_branch_homotopy_step(
                sample,
                high,
                official,
                sigma,
                high_r2v4_momentum=0.0,
                low_official_v2v_apg_momentum=0.0,
            )
        except Exception as error:
            raise NativeBranchHomotopyRuntimeError(str(error)) from error
        executed = combined.velocity
        if combined.endpoint == "low_official_v2v_apg" and executed is not official:
            raise NativeBranchHomotopyRuntimeError(
                "low endpoint did not directly return official tensor"
            )
        if combined.endpoint == "high_r2v4_apg" and executed is not high:
            raise NativeBranchHomotopyRuntimeError(
                "high endpoint did not directly return R2V-4 tensor"
            )
        if executed is official:
            call_args, call_kwargs = tuple(args), dict(kwargs)
        else:
            call_args, call_kwargs = _replace(
                self.original_scheduler_step,
                args,
                kwargs,
                name="model_output",
                value=executed,
            )
        result = self.original_scheduler_step(*call_args, **call_kwargs)
        self.original_scheduler_call_count += 1
        state.completed_steps += 1
        high_low_rms = _tensor_rms(high.float() - official.float())
        self.trace.append(
            {
                "schema_version": SCHEMA_VERSION,
                "step_index": step_index,
                "timestep": _scalar(timestep, label="timestep"),
                "sigma": sigma_float,
                "forward_order": list(PER_STEP_FORWARD_ORDER),
                "transformer_forwards": 5,
                "low_vi_forwards": 2,
                "high_r2v4_forwards": 3,
                "original_scheduler_calls": 1,
                "patch_call_count": 10,
                "patch_source_ids": list(EXPECTED_PATCH_SOURCE_IDS),
                "target_patch_tokens_P": self.config.target_patch_tokens,
                "reference_patch_tokens_R": self.config.reference_patch_tokens,
                "low_vi_total_tokens": self.config.low_vi_tokens,
                "high_i_total_tokens": self.config.high_i_tokens,
                "low_official_apg_exact_parity": True,
                "low_official_apg_parity_rms": parity_rms,
                "low_official_apg_parity_max_abs": parity_max,
                "high_low_velocity_delta_rms": high_low_rms,
                "vendor_high_apg_function": (
                    f"{self.vendor_chain.__module__}.{self.vendor_chain.__name__}"
                ),
                **combined.trace_dict(),
                "scheduler_received_original_model_output_object": executed is official,
                "endpoint_direct_return_verified": combined.endpoint != "transition",
                "freeze_safe_no_grad_outputs": all(
                    not value.requires_grad and value.grad_fn is None
                    for value in (official, high, executed)
                ),
            }
        )
        state.patch_results.clear()
        state.low_forwards.clear()
        state.high_forwards.clear()
        return result

    def finalize(self) -> Mapping[str, Any]:
        if not self.restored or self.finalized:
            raise NativeBranchHomotopyRuntimeError("homotopy patch finalize differs")
        steps = self.config.expected_steps
        if (
            self.sample_call_count != 1
            or self.schedule_preflight is None
            or self.patch_call_count != 10 * steps
            or self.low_forward_count != 2 * steps
            or self.high_forward_count != 3 * steps
            or self.original_scheduler_call_count != steps
            or len(self.trace) != steps
        ):
            raise NativeBranchHomotopyRuntimeError(
                "homotopy runtime call-count certificate differs"
            )
        if any(
            row["low_official_apg_exact_parity"] is not True
            or row["transformer_forwards"] != 5
            or row["original_scheduler_calls"] != 1
            or row["patch_source_ids"] != list(EXPECTED_PATCH_SOURCE_IDS)
            for row in self.trace
        ):
            raise NativeBranchHomotopyRuntimeError(
                "homotopy per-step certificate differs"
            )
        expected_endpoints = (
            ["high_r2v4_apg"] * 15
            + ["transition"] * 16
            + ["low_official_v2v_apg"] * 9
        )
        observed_endpoints = [str(row["endpoint"]) for row in self.trace]
        if observed_endpoints != expected_endpoints:
            raise NativeBranchHomotopyRuntimeError(
                "exact40 shift-5 homotopy endpoint partition differs"
            )
        if any(
            row["endpoint_direct_return_verified"] is not True
            for row in self.trace[:15] + self.trace[31:]
        ) or any(
            row["endpoint_direct_return_verified"] is not False
            for row in self.trace[15:31]
        ):
            raise NativeBranchHomotopyRuntimeError(
                "homotopy endpoint direct-return certificate differs"
            )
        self.finalized = True
        return {
            "schema_version": SCHEMA_VERSION,
            "sample_calls": 1,
            "steps": steps,
            "transformer_forwards": self.low_forward_count + self.high_forward_count,
            "low_vi_forwards": self.low_forward_count,
            "high_r2v4_forwards": self.high_forward_count,
            "patch_vae_latent_calls": self.patch_call_count,
            "original_scheduler_calls": self.original_scheduler_call_count,
            "per_step_forward_order": list(PER_STEP_FORWARD_ORDER),
            "per_step_patch_source_ids": list(EXPECTED_PATCH_SOURCE_IDS),
            "schedule_preflight": dict(self.schedule_preflight),
            "target_patch_tokens_P": self.config.target_patch_tokens,
            "reference_patch_tokens_R": self.config.reference_patch_tokens,
            "low_vi_total_tokens": self.config.low_vi_tokens,
            "high_i_total_tokens": self.config.high_i_tokens,
            "high_apg": {
                "function": f"{VENDOR_APG_MODULE}.normalized_guidance_chain",
                "omega_image": self.config.omega_image,
                "omega_text": self.config.omega_text,
                "eta": self.config.eta,
                "norm_thresholds": [
                    self.config.image_norm_threshold,
                    self.config.text_norm_threshold,
                ],
                "momentum": self.config.momentum,
            },
            "low_official_apg_exact_parity_all_steps": True,
            "smoothstep_sigma_low": homotopy.SIGMA_LOW,
            "smoothstep_sigma_high": homotopy.SIGMA_HIGH,
            "exact40_endpoint_partition": {
                "high_r2v4_apg_indices": list(range(0, 15)),
                "transition_indices": list(range(15, 31)),
                "low_official_v2v_apg_indices": list(range(31, 40)),
            },
            "scheduler_mutation_surface": "model_output_argument_only",
            "vendor_source_modified": False,
            "optimizer_created": False,
            "parameters_updated": False,
            "trace": list(self.trace),
        }


__all__ = [
    "EXPECTED_PATCH_SOURCE_IDS",
    "NativeBranchHomotopyRuntimeConfig",
    "NativeBranchHomotopyRuntimeError",
    "NativeBranchHomotopyRuntimePatch",
    "PER_STEP_FORWARD_ORDER",
    "PINNED_BERNINI_COMMIT",
    "PINNED_WAN_DIFFUSION_SHA256",
    "SCHEMA_VERSION",
]
