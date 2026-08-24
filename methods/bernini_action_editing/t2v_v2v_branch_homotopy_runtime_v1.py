#!/usr/bin/env python3
"""Reversible pure-T2V / source-only-V2V Bernini homotopy hook.

The host call is pinned stock ``v2v_apg`` with exactly one full source video
and no image references.  Its two official source+target forwards are observed
unchanged.  On the same step, the hook reuses the already-patched noisy target
to run exactly two additional target-only forwards with the shared negative
embedding and a separately authenticated T2V-native action embedding.  Each
pair is independently guided in clean space by the pinned vendor
``normalized_guidance`` (text scale 4, eta .5, norm 50, momentum zero), then
the scheduler-bound velocities are combined by
``t2v_v2v_branch_homotopy_v1`` immediately before the untouched UniPC step.

Every denoising step therefore has four transformer forwards and exactly one
call to the original scheduler.  The adapter is inference-only, creates no
optimizer, edits no vendor file, and restores all instance hooks in reverse
order.  It fails closed on source packing, prompt/state/timestep identity,
stock-low APG parity, the complete pinned exact40 shift-5 timeline (including
the terminal sigma), and endpoint object parity before integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import inspect
import math
from typing import Any, Callable, Mapping, Optional, Sequence

import self_guided_action_field_v1 as sgaf
import source_self_native_ref_contrastive_v3 as schedule_contract
import t2v_v2v_branch_homotopy_v1 as homotopy
import tri_branch_unipc as sampler_contract


SCHEMA_VERSION = "bernini-t2v-v2v-branch-homotopy-runtime-v1"
PINNED_BERNINI_COMMIT = sampler_contract.PINNED_BERNINI_COMMIT
PINNED_WAN_DIFFUSION_SHA256 = sampler_contract.PINNED_WAN_DIFFUSION_SHA256
PINNED_SCHEDULE_DIGEST = (
    schedule_contract.PINNED_NATIVE_UNIPC40_SCHEDULE_DIGEST
)
VENDOR_APG_MODULE = "bernini.models.wan_diffusion"
EXPECTED_PATCH_SOURCE_IDS = (1.0, 0.0)
PER_STEP_FORWARD_ORDER = (
    "low-source-v2v-negative",
    "low-source-v2v-action",
    "high-pure-t2v-negative",
    "high-pure-t2v-action",
)
HIGH_ENDPOINT_STEP_INDICES = tuple(range(0, 9))
TRANSITION_STEP_INDICES = tuple(range(9, 26))
LOW_ENDPOINT_STEP_INDICES = tuple(range(26, 40))


class T2VV2VBranchHomotopyRuntimeError(RuntimeError):
    """Raised before integration when the pinned runtime contract differs."""


def _raise_from_sgaf(error: Exception) -> T2VV2VBranchHomotopyRuntimeError:
    return T2VV2VBranchHomotopyRuntimeError(str(error))


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
class T2VV2VBranchHomotopyRuntimeConfig:
    """Pinned exact81/single-DiT sampler and generic latent geometry."""

    target_latent_shape: tuple[int, int, int, int, int]
    expected_steps: int = 40
    expected_num_frames: int = 81
    expected_flow_shift: float = 5.0
    omega_text: float = 4.0
    eta: float = 0.5
    norm_threshold: float = 50.0
    momentum: float = 0.0
    expected_hidden_dim: int = 1536
    expected_text_dim: int = 4096
    expected_model_id: str = "transformer_1"
    expected_guidance_mode: str = "v2v_apg"

    @property
    def target_patch_tokens(self) -> int:
        _, _, phases, height, width = self.target_latent_shape
        return int(phases * (height // 2) * (width // 2))

    @property
    def low_source_v2v_tokens(self) -> int:
        return 2 * self.target_patch_tokens

    @property
    def high_pure_t2v_tokens(self) -> int:
        return self.target_patch_tokens

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
            raise T2VV2VBranchHomotopyRuntimeError(
                "target latent must be exact81 Bernini [1,16,21,even,even]"
            )
        if type(self.expected_steps) is not int or self.expected_steps != 40:
            raise T2VV2VBranchHomotopyRuntimeError("runtime is pinned to exact40")
        if (
            type(self.expected_num_frames) is not int
            or self.expected_num_frames != 81
        ):
            raise T2VV2VBranchHomotopyRuntimeError("runtime is pinned to exact81")
        if (
            self.expected_hidden_dim != 1536
            or self.expected_text_dim != 4096
            or self.expected_model_id != "transformer_1"
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
                "runtime is pinned to Bernini-R 1.3B transformer_1"
            )
        if self.expected_guidance_mode != "v2v_apg":
            raise T2VV2VBranchHomotopyRuntimeError(
                "runtime requires guidance_mode='v2v_apg'"
            )
        exact = {
            "expected_flow_shift": (self.expected_flow_shift, 5.0),
            "omega_text": (self.omega_text, 4.0),
            "eta": (self.eta, 0.5),
            "norm_threshold": (self.norm_threshold, 50.0),
            "momentum": (self.momentum, 0.0),
        }
        for label, (observed, wanted) in exact.items():
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not math.isfinite(float(observed))
                or float(observed) != wanted
            ):
                raise T2VV2VBranchHomotopyRuntimeError(
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
    shared_negative_prompt: Any
    high_t2v_action_prompt: Any
    source_video: Any
    source_patch_value: Any
    low_momentum: Any
    high_momentum: Any
    completed_steps: int = 0
    patch_results: list[_PatchResult] = field(default_factory=list)
    low_forwards: list[_ForwardResult] = field(default_factory=list)
    high_forwards: list[_ForwardResult] = field(default_factory=list)


def _resolve_vendor_apg_symbols() -> tuple[Callable[..., Any], type[Any]]:
    """Resolve and authenticate the exact APG symbols used by T2V/V2V."""

    try:
        module = importlib.import_module(VENDOR_APG_MODULE)
    except Exception as error:
        raise T2VV2VBranchHomotopyRuntimeError(
            f"cannot import pinned {VENDOR_APG_MODULE}"
        ) from error
    single = getattr(module, "normalized_guidance", None)
    momentum_class = getattr(module, "MomentumBuffer", None)
    if not callable(single) or not isinstance(momentum_class, type):
        raise T2VV2VBranchHomotopyRuntimeError(
            "pinned vendor APG symbols are unavailable"
        )
    for value, name in (
        (single, "normalized_guidance"),
        (momentum_class, "MomentumBuffer"),
    ):
        if (
            getattr(value, "__module__", None) != VENDOR_APG_MODULE
            or getattr(value, "__name__", None) != name
            or inspect.getmodule(value) is not module
            or getattr(module, name) is not value
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
                "vendor APG module/function identity differs"
            )
    try:
        single_parameters = tuple(inspect.signature(single).parameters)
        momentum_parameters = tuple(inspect.signature(momentum_class).parameters)
    except (TypeError, ValueError) as error:
        raise T2VV2VBranchHomotopyRuntimeError(
            "vendor APG signatures are not inspectable"
        ) from error
    if single_parameters != (
        "pred_cond",
        "pred_uncond",
        "guidance_scale",
        "momentum_buffer",
        "eta",
        "norm_threshold",
    ) or momentum_parameters != ("momentum",):
        raise T2VV2VBranchHomotopyRuntimeError("vendor APG signature differs")
    return single, momentum_class


class T2VV2VBranchHomotopyRuntimePatch:
    """One-sample reversible four-forward T2V/V2V homotopy adapter."""

    def __init__(
        self,
        diffusion: Any,
        *,
        t2v_action_prompt_embeds: Any,
        config: T2VV2VBranchHomotopyRuntimeConfig,
        expected_bernini_commit: str = PINNED_BERNINI_COMMIT,
        observed_wan_diffusion_sha256: str = PINNED_WAN_DIFFUSION_SHA256,
    ) -> None:
        import torch

        config.validate()
        if expected_bernini_commit != PINNED_BERNINI_COMMIT:
            raise T2VV2VBranchHomotopyRuntimeError("Bernini revision differs")
        if observed_wan_diffusion_sha256 != PINNED_WAN_DIFFUSION_SHA256:
            raise T2VV2VBranchHomotopyRuntimeError("wan_diffusion.py bytes differ")
        if (
            not isinstance(t2v_action_prompt_embeds, torch.Tensor)
            or tuple(t2v_action_prompt_embeds.shape)
            != (1, 512, config.expected_text_dim)
            or t2v_action_prompt_embeds.requires_grad
            or t2v_action_prompt_embeds.grad_fn is not None
            or not bool(torch.isfinite(t2v_action_prompt_embeds).all().item())
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
                "official T2V action embedding geometry differs"
            )
        try:
            core = sampler_contract.resolve_diffusion_core(diffusion)
        except Exception as error:
            raise T2VV2VBranchHomotopyRuntimeError(str(error)) from error
        transformer = getattr(core, "transformer", None)
        scheduler = getattr(core, "scheduler", None)
        originals = {
            "sample": getattr(core, "sample", None),
            "shared_step": getattr(core, "shared_step", None),
            "patch_vae_latent": getattr(transformer, "patch_vae_latent", None),
            "scheduler.step": getattr(scheduler, "step", None),
        }
        if any(not callable(value) for value in originals.values()):
            raise T2VV2VBranchHomotopyRuntimeError(
                "pinned Bernini sampler call surface differs"
            )
        if getattr(core, "use_unipc", None) is not True:
            raise T2VV2VBranchHomotopyRuntimeError("runtime requires native UniPC")
        if getattr(core, "transformer_2", None) is not None:
            raise T2VV2VBranchHomotopyRuntimeError(
                "runtime supports only single-expert Bernini-R 1.3B"
            )
        transformer_config = getattr(transformer, "config", None)
        if transformer_config is None:
            raise T2VV2VBranchHomotopyRuntimeError(
                "transformer config is unavailable"
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
            or config_value("in_channels") != 16
            or config_value("text_dim") != config.expected_text_dim
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
                "transformer hidden/text/input geometry differs"
            )
        try:
            sampler_contract._validate_scheduler_contract(
                scheduler,
                expected_flow_shift=config.expected_flow_shift,
            )
        except Exception as error:
            raise T2VV2VBranchHomotopyRuntimeError(str(error)) from error
        for owner, name in (
            (core, "sample"),
            (core, "shared_step"),
            (transformer, "patch_vae_latent"),
            (scheduler, "step"),
        ):
            try:
                if name in vars(owner):
                    raise T2VV2VBranchHomotopyRuntimeError(
                        f"refusing stacked instance override on {name}"
                    )
            except TypeError as error:
                raise T2VV2VBranchHomotopyRuntimeError(
                    f"cannot inspect {name} owner"
                ) from error
        for original in originals.values():
            if getattr(original, "_bernini_t2v_v2v_homotopy_v1", None) is not None:
                raise T2VV2VBranchHomotopyRuntimeError(
                    "T2V/V2V homotopy wrapper is already installed"
                )
        if getattr(transformer, "training", False) is not False:
            raise T2VV2VBranchHomotopyRuntimeError(
                "transformer must remain in eval mode"
            )
        named_parameters = getattr(transformer, "named_parameters", None)
        if not callable(named_parameters):
            raise T2VV2VBranchHomotopyRuntimeError(
                "transformer freeze surface is unavailable"
            )
        for name, parameter in named_parameters():
            if bool(getattr(parameter, "requires_grad", False)) or getattr(
                parameter, "grad", None
            ) is not None:
                raise T2VV2VBranchHomotopyRuntimeError(
                    f"transformer parameter {name} is not freeze-safe"
                )

        vendor_single, vendor_momentum_class = _resolve_vendor_apg_symbols()
        self.diffusion = core
        self.transformer = transformer
        self.scheduler = scheduler
        self.t2v_action_prompt_embeds = t2v_action_prompt_embeds
        self.config = config
        self.vendor_single = vendor_single
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
        self.schedule_validated = False
        self.sample_call_count = 0
        self.patch_call_count = 0
        self.low_forward_count = 0
        self.high_forward_count = 0
        self.low_apg_call_count = 0
        self.high_apg_call_count = 0
        self.original_scheduler_call_count = 0
        self.trace: list[dict[str, Any]] = []

    def _validate_prompt(self, value: Any, *, label: str) -> None:
        import torch

        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != (1, 512, self.config.expected_text_dim)
            or value.device != self.t2v_action_prompt_embeds.device
            or value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
                f"{label} prompt geometry differs"
            )

    def _set_patch(self, owner: Any, name: str, value: Any) -> None:
        try:
            instance = vars(owner)
        except TypeError as error:
            raise T2VV2VBranchHomotopyRuntimeError(
                f"cannot reversibly patch {name} owner"
            ) from error
        had_instance = name in instance
        previous = instance.get(name)
        resolved_before = getattr(owner, name)
        setattr(owner, name, value)
        self._patches.append((owner, name, had_instance, previous, resolved_before))

    def install(self) -> None:
        if self.installed or self.restored or self.finalized:
            raise T2VV2VBranchHomotopyRuntimeError(
                "T2V/V2V homotopy patch lifecycle differs"
            )

        def sample_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_sample(*args, **kwargs)

        def shared_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_shared_step(*args, **kwargs)

        def patch_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_patch_vae_latent(*args, **kwargs)

        def scheduler_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_scheduler_step(*args, **kwargs)

        for wrapper in (sample_wrapper, shared_wrapper, patch_wrapper, scheduler_wrapper):
            setattr(wrapper, "_bernini_t2v_v2v_homotopy_v1", self)
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
                    current, "_bernini_t2v_v2v_homotopy_v1", None
                ) is not self:
                    errors.append(
                        T2VV2VBranchHomotopyRuntimeError(
                            f"{name} changed during T2V/V2V homotopy patch"
                        )
                    )
                if had_instance:
                    setattr(owner, name, previous)
                else:
                    delattr(owner, name)
                if getattr(owner, name, None) != resolved_before:
                    errors.append(
                        T2VV2VBranchHomotopyRuntimeError(
                            f"{name} restoration failed"
                        )
                    )
            except Exception as error:
                errors.append(error)
        self._active = None
        if errors:
            raise T2VV2VBranchHomotopyRuntimeError(
                f"failed to restore {len(errors)} T2V/V2V wrapper(s)"
            ) from errors[0]

    def restore(self) -> None:
        if not self.installed or self.restored:
            raise T2VV2VBranchHomotopyRuntimeError(
                "T2V/V2V homotopy patch restore differs"
            )
        try:
            self._restore_patches(require_wrapper_identity=True)
        finally:
            self.installed = False
            self.restored = not self._patches

    def _normalize_threshold(self, value: Any) -> float:
        if isinstance(value, (list, tuple)):
            if not value or any(
                _scalar(item, label="norm_threshold") != self.config.norm_threshold
                for item in value
            ):
                raise T2VV2VBranchHomotopyRuntimeError(
                    "sample norm_threshold must contain only 50"
                )
            return self.config.norm_threshold
        observed = _scalar(value, label="norm_threshold")
        if observed != self.config.norm_threshold:
            raise T2VV2VBranchHomotopyRuntimeError(
                "sample norm_threshold must equal 50"
            )
        return observed

    def _validate_sample_contract(self, values: Mapping[str, Any]) -> _ActiveSample:
        import torch

        videos = values.get("multi_video_vae_latents")
        if isinstance(videos, torch.Tensor):
            videos = [videos]
        if (
            values.get("guidance_mode") != self.config.expected_guidance_mode
            or values.get("num_frames") != self.config.expected_num_frames
            or values.get("num_inference_steps") != self.config.expected_steps
            or _scalar(values.get("flow_shift"), label="flow_shift")
            != self.config.expected_flow_shift
            or _scalar(values.get("omega_txt"), label="omega_txt")
            != self.config.omega_text
            or _scalar(values.get("eta"), label="eta") != self.config.eta
            or _scalar(values.get("momentum"), label="momentum")
            != self.config.momentum
            or values.get("prompt_embeds_t2") is not None
            or values.get("uncond_embeds_t2") is not None
            or not isinstance(videos, (list, tuple))
            or len(videos) != 1
            or values.get("image_vae_latents") is not None
            or values.get("multi_image_vae_latents") is not None
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
                "source-video-only v2v_apg sample/condition contract differs"
            )
        self._normalize_threshold(values.get("norm_threshold"))
        source_video = videos[0]
        if (
            not isinstance(source_video, torch.Tensor)
            or tuple(source_video.shape) != self.config.target_latent_shape
            or source_video.requires_grad
            or source_video.grad_fn is not None
            or not bool(torch.isfinite(source_video).all().item())
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
                "full source video latent geometry differs"
            )
        transformer_dtype = getattr(self.transformer, "dtype", None)
        if transformer_dtype is None:
            raise T2VV2VBranchHomotopyRuntimeError(
                "transformer dtype is unavailable"
            )
        source_patch_value = source_video.to(dtype=transformer_dtype)
        low_action = values.get("prompt_embeds")
        negative = values.get("uncond_prompt_embeds")
        self._validate_prompt(low_action, label="low MV2V action")
        self._validate_prompt(negative, label="shared negative")
        self._validate_prompt(self.t2v_action_prompt_embeds, label="high T2V action")
        if (
            low_action is negative
            or low_action is self.t2v_action_prompt_embeds
            or negative is self.t2v_action_prompt_embeds
            or torch.equal(low_action, self.t2v_action_prompt_embeds)
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
                "T2V and MV2V action prompts must be distinct; negative is shared"
            )
        low_momentum = self.vendor_momentum_class(0.0)
        high_momentum = self.vendor_momentum_class(0.0)
        if (
            low_momentum is high_momentum
            or _scalar(getattr(low_momentum, "momentum", None), label="low momentum")
            != 0.0
            or _scalar(getattr(high_momentum, "momentum", None), label="high momentum")
            != 0.0
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
                "independent zero-momentum APG buffers differ"
            )
        return _ActiveSample(
            low_action_prompt=low_action,
            shared_negative_prompt=negative,
            high_t2v_action_prompt=self.t2v_action_prompt_embeds,
            source_video=source_video,
            source_patch_value=source_patch_value,
            low_momentum=low_momentum,
            high_momentum=high_momentum,
        )

    def _wrapped_sample(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        if self._active is not None or self.sample_call_count != 0:
            raise T2VV2VBranchHomotopyRuntimeError(
                "homotopy patch permits exactly one non-nested sample"
            )
        if self.diffusion.scheduler is not self.scheduler:
            raise T2VV2VBranchHomotopyRuntimeError("diffusion.scheduler changed")
        values = _bind(self.original_sample, args, kwargs)
        state = self._validate_sample_contract(values)
        self._active = state
        try:
            result = self.original_sample(*args, **kwargs)
            if (
                state.completed_steps != self.config.expected_steps
                or state.patch_results
                or state.low_forwards
                or state.high_forwards
            ):
                raise T2VV2VBranchHomotopyRuntimeError(
                    "sample returned with an incomplete T2V/V2V step"
                )
            if (
                not isinstance(result, torch.Tensor)
                or tuple(result.shape) != self.config.target_latent_shape
                or result.dtype != torch.float32
                or result.requires_grad
                or result.grad_fn is not None
                or not bool(torch.isfinite(result).all().item())
            ):
                raise T2VV2VBranchHomotopyRuntimeError(
                    "sample returned non-native exact81 latent geometry"
                )
            self.sample_call_count += 1
            return result
        finally:
            self._active = None

    def _validate_live_exact40_schedule(self) -> None:
        import torch

        if self.schedule_validated:
            return
        receipt = schedule_contract.native_unipc40_schedule_receipt()
        if receipt.get("digest") != PINNED_SCHEDULE_DIGEST:
            raise T2VV2VBranchHomotopyRuntimeError(
                "pinned exact40 schedule digest differs"
            )
        timesteps = getattr(self.scheduler, "timesteps", None)
        sigmas = getattr(self.scheduler, "sigmas", None)
        if (
            not isinstance(timesteps, torch.Tensor)
            or timesteps.device.type != "cpu"
            or timesteps.ndim != 1
            or int(timesteps.numel()) != 40
            or not isinstance(sigmas, torch.Tensor)
            or sigmas.device.type != "cpu"
            or sigmas.dtype != torch.float32
            or sigmas.ndim != 1
            or int(sigmas.numel()) != 41
            or not bool(torch.isfinite(timesteps).all().item())
            or not bool(torch.isfinite(sigmas).all().item())
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
                "live exact40 shift-5 timeline geometry differs"
            )
        observed_timestep_integers = tuple(
            int(_scalar(value, label="live scheduler timestep"))
            for value in timesteps
        )
        if any(
            _scalar(value, label="live scheduler timestep") != float(integer)
            for value, integer in zip(timesteps, observed_timestep_integers)
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
                "live exact40 scheduler timesteps are not exact integers"
            )
        observed_sigmas = tuple(float(value.item()) for value in sigmas[:40])
        expected_timesteps = tuple(schedule_contract.NATIVE_UNIPC40_TIMESTEPS)
        expected_sigmas = tuple(schedule_contract.NATIVE_UNIPC40_SIGMAS)
        if (
            observed_timestep_integers != expected_timesteps
            or observed_sigmas != expected_sigmas
            or float(sigmas[40].item()) != 0.0
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
                "live exact40 shift-5 timesteps/sigmas differ"
            )
        regions = tuple(
            "high"
            if value >= homotopy.SIGMA_HIGH
            else "low" if value <= homotopy.SIGMA_LOW else "transition"
            for value in observed_sigmas
        )
        expected_regions = (
            ("high",) * 9 + ("transition",) * 17 + ("low",) * 14
        )
        if regions != expected_regions:
            raise T2VV2VBranchHomotopyRuntimeError(
                "exact40 T2V/V2V endpoint partition differs"
            )
        self.schedule_validated = True

    def _wrapped_patch_vae_latent(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        state = self._active
        if state is None:
            raise T2VV2VBranchHomotopyRuntimeError(
                "patch_vae_latent ran outside authenticated sample"
            )
        self._validate_live_exact40_schedule()
        if state.low_forwards or state.high_forwards:
            raise T2VV2VBranchHomotopyRuntimeError(
                "native patch call arrived after transformer forwarding began"
            )
        index = len(state.patch_results)
        if index >= 2:
            raise T2VV2VBranchHomotopyRuntimeError(
                "too many patch_vae_latent calls before scheduler.step"
            )
        values = _bind(self.original_patch_vae_latent, args, kwargs)
        source_id = _scalar(values.get("source_id"), label="patch source_id")
        if source_id != EXPECTED_PATCH_SOURCE_IDS[index]:
            raise T2VV2VBranchHomotopyRuntimeError(
                "native patch_vae_latent source-id order differs"
            )
        # The pinned Bernini renderer names the first patch input
        # ``hidden_states``.  Keep this binding exact: accepting guessed aliases
        # made the CPU fake pass while the real checkpoint failed before its
        # first homotopy forward.
        input_value = values.get("hidden_states")
        if not isinstance(input_value, torch.Tensor):
            raise T2VV2VBranchHomotopyRuntimeError(
                "patch_vae_latent canonical hidden_states input differs"
            )
        if index == 0 and (
            input_value.shape != state.source_patch_value.shape
            or input_value.device != state.source_patch_value.device
            or input_value.dtype != state.source_patch_value.dtype
            or not torch.equal(input_value, state.source_patch_value)
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
                "source patch input differs from authenticated full source"
            )
        result = self.original_patch_vae_latent(*args, **kwargs)
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise T2VV2VBranchHomotopyRuntimeError(
                "patch_vae_latent return geometry differs"
            )
        latent, rotary = result
        expected_tokens = self.config.target_patch_tokens
        if (
            not isinstance(latent, torch.Tensor)
            or _shape(latent, label="patched latent")
            != (1, expected_tokens, self.config.expected_hidden_dim)
            or not isinstance(rotary, torch.Tensor)
            or rotary.ndim != 4
            or rotary.shape[0] != 1
            or rotary.shape[2] != expected_tokens
            or latent.requires_grad
            or latent.grad_fn is not None
            or rotary.requires_grad
            or rotary.grad_fn is not None
            or not bool(torch.isfinite(latent).all().item())
            or not bool(torch.isfinite(rotary).all().item())
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
                "patched source/target token geometry differs"
            )
        state.patch_results.append(_PatchResult(source_id, input_value, latent, rotary))
        self.patch_call_count += 1
        return result

    def _captured_packs(self, state: _ActiveSample) -> Mapping[str, tuple[Any, Any]]:
        import torch

        if len(state.patch_results) != 2:
            raise T2VV2VBranchHomotopyRuntimeError(
                "source/target patch pair is incomplete"
            )
        source, target = state.patch_results
        low = (
            torch.cat([source.latent, target.latent], dim=1).to(
                dtype=self.transformer.dtype
            ),
            torch.cat([source.rotary, target.rotary], dim=2),
        )
        high = (
            torch.cat([target.latent], dim=1).to(dtype=self.transformer.dtype),
            torch.cat([target.rotary], dim=2),
        )
        return {"low": low, "high": high}

    def _validate_low_call(
        self,
        state: _ActiveSample,
        values: Mapping[str, Any],
        *,
        index: int,
        low_pack: tuple[Any, Any],
    ) -> None:
        import torch

        branch = PER_STEP_FORWARD_ORDER[index]
        prompt = state.shared_negative_prompt if index == 0 else state.low_action_prompt
        if values.get("model_id") != self.config.expected_model_id:
            raise T2VV2VBranchHomotopyRuntimeError(f"{branch} model_id differs")
        if values.get("cond_embeds") is not prompt:
            raise T2VV2VBranchHomotopyRuntimeError(f"{branch} prompt object differs")
        noisy = values.get("noisy_latents")
        rotary = values.get("rotary_embs")
        timestep = values.get("timesteps")
        if (
            not isinstance(noisy, torch.Tensor)
            or _shape(noisy, label=f"{branch} noisy")
            != (1, self.config.low_source_v2v_tokens, self.config.expected_hidden_dim)
            or not torch.equal(noisy, low_pack[0])
            or not isinstance(rotary, torch.Tensor)
            or rotary.ndim != 4
            or rotary.shape[0] != 1
            or rotary.shape[2] != self.config.low_source_v2v_tokens
            or not torch.equal(rotary, low_pack[1])
            or _metadata(values.get("batch_vae_seqlen"), label=f"{branch} VAE length")
            != (self.config.low_source_v2v_tokens,)
            or _metadata(values.get("batch_text_seqlen"), label=f"{branch} text length")
            != (512,)
            or not isinstance(timestep, torch.Tensor)
            or timestep.shape != (1,)
            or not bool(torch.isfinite(timestep).all().item())
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
                f"{branch} source+target token/timestep/rotary geometry differs"
            )
        if index == 1:
            first = state.low_forwards[0].values
            for name in ("noisy_latents", "timesteps", "rotary_embs"):
                _same(first.get(name), values.get(name), label=f"low neg/action {name}")

    def _validate_prediction(
        self,
        prediction: Any,
        *,
        total_tokens: int,
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
            raise T2VV2VBranchHomotopyRuntimeError(
                f"{label} transformer prediction geometry differs"
            )
        tail = prediction[:, -self.config.target_patch_tokens :, :]
        if _shape(tail, label=f"{label} target tail") != (
            1,
            self.config.target_patch_tokens,
            expected_channels,
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
                f"{label} target-tail geometry differs"
            )
        return tail

    def _high_query(
        self,
        *,
        name: str,
        pack: tuple[Any, Any],
        prompt: Any,
        timestep: Any,
    ) -> _ForwardResult:
        call_kwargs = {
            "model_id": self.config.expected_model_id,
            "noisy_latents": pack[0],
            "timesteps": timestep,
            "cond_embeds": prompt,
            "rotary_embs": pack[1],
            "batch_vae_seqlen": [self.config.high_pure_t2v_tokens],
            "batch_text_seqlen": [512],
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
            total_tokens=self.config.high_pure_t2v_tokens,
            label=name,
        )
        self.high_forward_count += 1
        return _ForwardResult(name, bound, prediction, tail)

    def _wrapped_shared_step(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise T2VV2VBranchHomotopyRuntimeError(
                "shared_step ran outside authenticated sample"
            )
        if state.high_forwards:
            raise T2VV2VBranchHomotopyRuntimeError(
                "unexpected shared_step after pure-T2V closure"
            )
        index = len(state.low_forwards)
        if index >= 2:
            raise T2VV2VBranchHomotopyRuntimeError(
                "more than two official source-V2V forwards occurred"
            )
        packs = self._captured_packs(state)
        values = _bind(self.original_shared_step, args, kwargs)
        self._validate_low_call(state, values, index=index, low_pack=packs["low"])
        prediction = self.original_shared_step(*args, **kwargs)
        tail = self._validate_prediction(
            prediction,
            total_tokens=self.config.low_source_v2v_tokens,
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
                        name=PER_STEP_FORWARD_ORDER[2],
                        pack=packs["high"],
                        prompt=state.shared_negative_prompt,
                        timestep=timestep,
                    ),
                    self._high_query(
                        name=PER_STEP_FORWARD_ORDER[3],
                        pack=packs["high"],
                        prompt=state.high_t2v_action_prompt,
                        timestep=timestep,
                    ),
                )
            )
        return prediction

    def _branch_apg_velocity(
        self,
        forwards: Sequence[_ForwardResult],
        *,
        sample: Any,
        sigma: Any,
        momentum_buffer: Any,
        output_like: Any,
        branch: str,
    ) -> Any:
        import torch

        if len(forwards) != 2:
            raise T2VV2VBranchHomotopyRuntimeError(
                f"{branch} forward pair is incomplete"
            )
        sample_spatial = sgaf._packed_to_spatial(
            sample,
            self.config.target_latent_shape,
        )
        clean = []
        for forward in forwards:
            velocity = sgaf._packed_to_spatial(
                forward.target_tail,
                self.config.target_latent_shape,
            )
            clean.append(sample_spatial - sigma * velocity)
        guided_clean = self.vendor_single(
            pred_cond=clean[1],
            pred_uncond=clean[0],
            guidance_scale=self.config.omega_text,
            momentum_buffer=momentum_buffer,
            eta=self.config.eta,
            norm_threshold=self.config.norm_threshold,
        )
        if branch == "low":
            self.low_apg_call_count += 1
        else:
            self.high_apg_call_count += 1
        if (
            not isinstance(guided_clean, torch.Tensor)
            or tuple(guided_clean.shape) != tuple(sample_spatial.shape)
            or guided_clean.device != sample_spatial.device
            or not bool(torch.isfinite(guided_clean).all().item())
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
                f"vendor normalized_guidance {branch} output differs"
            )
        velocity = sgaf._spatial_to_packed(
            (sample_spatial - guided_clean) / sigma,
            self.config.target_latent_shape,
        )
        if (
            velocity.shape != output_like.shape
            or velocity.device != output_like.device
            or velocity.dtype != output_like.dtype
            or velocity.requires_grad
            or velocity.grad_fn is not None
            or not bool(torch.isfinite(velocity).all().item())
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
                f"{branch} APG scheduler-bound geometry differs"
            )
        return velocity

    def _wrapped_scheduler_step(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        state = self._active
        if state is None:
            raise T2VV2VBranchHomotopyRuntimeError(
                "scheduler.step ran outside authenticated sample"
            )
        self._validate_live_exact40_schedule()
        if (
            len(state.patch_results) != 2
            or tuple(item.source_id for item in state.patch_results)
            != EXPECTED_PATCH_SOURCE_IDS
            or len(state.low_forwards) != 2
            or len(state.high_forwards) != 2
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
                "scheduler.step arrived before four-forward closure"
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
        for label, value in (
            ("official model_output", official),
            ("scheduler sample", sample),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or _shape(value, label=label) != expected_shape
                or not bool(torch.isfinite(value).all().item())
            ):
                raise T2VV2VBranchHomotopyRuntimeError(
                    f"{label} geometry differs"
                )
        if official.device != sample.device:
            raise T2VV2VBranchHomotopyRuntimeError(
                "official output/sample devices differ"
            )
        expected_target_patch_input = sgaf._packed_to_spatial(
            sample,
            self.config.target_latent_shape,
        ).to(dtype=self.transformer.dtype)
        observed_target_patch_input = state.patch_results[1].input_value
        if (
            observed_target_patch_input.shape != expected_target_patch_input.shape
            or observed_target_patch_input.device != expected_target_patch_input.device
            or observed_target_patch_input.dtype != expected_target_patch_input.dtype
            or not torch.equal(observed_target_patch_input, expected_target_patch_input)
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
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
            raise T2VV2VBranchHomotopyRuntimeError(
                "scheduler step index differs from homotopy state"
            )
        if (
            not isinstance(sigma, torch.Tensor)
            or sigma.ndim != 0
            or sigma.device.type != "cpu"
            or sigma.dtype != torch.float32
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
                "active UniPC sigma must remain a CPU fp32 scalar"
            )
        low = self._branch_apg_velocity(
            state.low_forwards,
            sample=sample,
            sigma=sigma,
            momentum_buffer=state.low_momentum,
            output_like=official,
            branch="low",
        )
        parity_delta = low.float() - official.float()
        parity_rms = _tensor_rms(parity_delta)
        parity_max = float(parity_delta.abs().max().item())
        if not torch.equal(low, official):
            raise T2VV2VBranchHomotopyRuntimeError(
                "vendor-single rebuilt low source-V2V APG differs from stock output: "
                f"max_abs={parity_max:.9g} rms={parity_rms:.9g}"
            )
        high = self._branch_apg_velocity(
            state.high_forwards,
            sample=sample,
            sigma=sigma,
            momentum_buffer=state.high_momentum,
            output_like=official,
            branch="high",
        )
        try:
            combined = homotopy.t2v_v2v_branch_homotopy_step(
                sample,
                high,
                official,
                sigma,
                high_pure_t2v_momentum=0.0,
                low_source_v2v_momentum=0.0,
            )
        except Exception as error:
            raise T2VV2VBranchHomotopyRuntimeError(str(error)) from error
        executed = combined.velocity
        if combined.endpoint == "low_source_v2v_apg" and executed is not official:
            raise T2VV2VBranchHomotopyRuntimeError(
                "low endpoint did not directly return stock V2V tensor"
            )
        if combined.endpoint == "high_pure_t2v_apg" and executed is not high:
            raise T2VV2VBranchHomotopyRuntimeError(
                "high endpoint did not directly return pure-T2V tensor"
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
        self.trace.append(
            {
                "schema_version": SCHEMA_VERSION,
                "step_index": step_index,
                "timestep": _scalar(timestep, label="timestep"),
                "sigma": sigma_float,
                "forward_order": list(PER_STEP_FORWARD_ORDER),
                "transformer_forwards": 4,
                "low_source_v2v_forwards": 2,
                "high_pure_t2v_forwards": 2,
                "original_scheduler_calls": 1,
                "patch_call_count": 2,
                "patch_source_ids": list(EXPECTED_PATCH_SOURCE_IDS),
                "target_patch_tokens": self.config.target_patch_tokens,
                "low_source_v2v_total_tokens": self.config.low_source_v2v_tokens,
                "high_pure_t2v_total_tokens": self.config.high_pure_t2v_tokens,
                "low_stock_apg_exact_parity": True,
                "low_stock_apg_parity_rms": parity_rms,
                "low_stock_apg_parity_max_abs": parity_max,
                "high_low_velocity_delta_rms": _tensor_rms(
                    high.float() - official.float()
                ),
                "vendor_apg_function": (
                    f"{self.vendor_single.__module__}.{self.vendor_single.__name__}"
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
            raise T2VV2VBranchHomotopyRuntimeError(
                "T2V/V2V homotopy patch finalize differs"
            )
        steps = self.config.expected_steps
        if (
            not self.schedule_validated
            or self.sample_call_count != 1
            or self.patch_call_count != 2 * steps
            or self.low_forward_count != 2 * steps
            or self.high_forward_count != 2 * steps
            or self.low_apg_call_count != steps
            or self.high_apg_call_count != steps
            or self.original_scheduler_call_count != steps
            or len(self.trace) != steps
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
                "T2V/V2V homotopy runtime call-count certificate differs"
            )
        if any(
            row["low_stock_apg_exact_parity"] is not True
            or row["transformer_forwards"] != 4
            or row["original_scheduler_calls"] != 1
            or row["patch_source_ids"] != list(EXPECTED_PATCH_SOURCE_IDS)
            or row["freeze_safe_no_grad_outputs"] is not True
            for row in self.trace
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
                "T2V/V2V homotopy per-step certificate differs"
            )
        expected_endpoints = (
            ["high_pure_t2v_apg"] * 9
            + ["transition"] * 17
            + ["low_source_v2v_apg"] * 14
        )
        if [str(row["endpoint"]) for row in self.trace] != expected_endpoints:
            raise T2VV2VBranchHomotopyRuntimeError(
                "exact40 shift-5 T2V/V2V endpoint partition differs"
            )
        if any(
            row["endpoint_direct_return_verified"] is not True
            for row in self.trace[:9] + self.trace[26:]
        ) or any(
            row["endpoint_direct_return_verified"] is not False
            for row in self.trace[9:26]
        ):
            raise T2VV2VBranchHomotopyRuntimeError(
                "T2V/V2V endpoint direct-return certificate differs"
            )
        self.finalized = True
        return {
            "schema_version": SCHEMA_VERSION,
            "sample_calls": 1,
            "steps": steps,
            "transformer_forwards": self.low_forward_count + self.high_forward_count,
            "low_source_v2v_forwards": self.low_forward_count,
            "high_pure_t2v_forwards": self.high_forward_count,
            "patch_vae_latent_calls": self.patch_call_count,
            "original_scheduler_calls": self.original_scheduler_call_count,
            "per_step_forward_order": list(PER_STEP_FORWARD_ORDER),
            "per_step_patch_source_ids": list(EXPECTED_PATCH_SOURCE_IDS),
            "target_patch_tokens": self.config.target_patch_tokens,
            "low_source_v2v_total_tokens": self.config.low_source_v2v_tokens,
            "high_pure_t2v_total_tokens": self.config.high_pure_t2v_tokens,
            "branch_apg": {
                "function": f"{VENDOR_APG_MODULE}.normalized_guidance",
                "one_condition_per_branch": True,
                "omega_text": self.config.omega_text,
                "eta": self.config.eta,
                "norm_threshold": self.config.norm_threshold,
                "independent_momentum": self.config.momentum,
            },
            "shared_negative_embedding_object": True,
            "low_stock_apg_exact_parity_all_steps": True,
            "smoothstep_sigma_low": homotopy.SIGMA_LOW,
            "smoothstep_sigma_high": homotopy.SIGMA_HIGH,
            "exact40_shift5_schedule_digest": PINNED_SCHEDULE_DIGEST,
            "terminal_sigma": 0.0,
            "exact40_endpoint_partition": {
                "high_pure_t2v_indices": list(HIGH_ENDPOINT_STEP_INDICES),
                "transition_indices": list(TRANSITION_STEP_INDICES),
                "low_source_v2v_indices": list(LOW_ENDPOINT_STEP_INDICES),
            },
            "scheduler_mutation_surface": "model_output_argument_only",
            "runtime_source_identity_enforcement": "external_canary_required",
            "vendor_source_modified": False,
            "optimizer_created": False,
            "parameters_updated": False,
            "trace": list(self.trace),
        }


__all__ = [
    "EXPECTED_PATCH_SOURCE_IDS",
    "HIGH_ENDPOINT_STEP_INDICES",
    "LOW_ENDPOINT_STEP_INDICES",
    "PER_STEP_FORWARD_ORDER",
    "PINNED_BERNINI_COMMIT",
    "PINNED_SCHEDULE_DIGEST",
    "PINNED_WAN_DIFFUSION_SHA256",
    "SCHEMA_VERSION",
    "T2VV2VBranchHomotopyRuntimeConfig",
    "T2VV2VBranchHomotopyRuntimeError",
    "T2VV2VBranchHomotopyRuntimePatch",
    "TRANSITION_STEP_INDICES",
]
