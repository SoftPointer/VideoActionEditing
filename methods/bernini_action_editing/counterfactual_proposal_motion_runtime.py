"""Full-step routing bridge for the frozen Bernini CPMR motion branch.

The V11 tensor and processor modules intentionally do not guess which of the
two APG calls is positive.  This module binds the *actual* pinned ``GEN_Wanx22``
``sample``/``shared_step`` boundary and creates one authenticated invocation per
transformer call.  It is deliberately limited to one 81-frame, 40-step,
Ulysses-compatible final-render sample.

No prompt string matching or call-number-only routing is used: the first call
of each official APG pair must carry the exact negative embedding object and
the second the exact positive embedding object supplied to ``sample``.  The
positive call receives a fresh conditioned-encoder binding covering every
installed CPMR block.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

import counterfactual_proposal_motion_branch as branch


EXPECTED_STEPS = 40
EXPECTED_FRAMES = 81
EXPECTED_GLOBAL_TOKENS = 39_060
EXPECTED_SOURCE_TOKENS = 19_530
EXPECTED_FLOW_SHIFT = 5.0
EXPECTED_GUIDANCE_MODE = "v2v_apg"


class CPMRRuntimeContractError(RuntimeError):
    """Raised when the pinned final-render call graph differs."""


def _bind_call(function: Any, args: Sequence[Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        bound = inspect.signature(function).bind(*args, **dict(kwargs))
        bound.apply_defaults()
    except (TypeError, ValueError) as error:
        raise CPMRRuntimeContractError("pinned Bernini call signature differs") from error
    return dict(bound.arguments)


def _shape(value: Any, *, label: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise CPMRRuntimeContractError(f"{label} has no tensor shape")
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError, OverflowError) as error:
        raise CPMRRuntimeContractError(f"{label} shape is not integral") from error


def _lengths(value: Any, *, label: str) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CPMRRuntimeContractError(f"{label} must be a sequence")
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError, OverflowError) as error:
        raise CPMRRuntimeContractError(f"{label} must contain integers") from error
    if any(item <= 0 for item in result):
        raise CPMRRuntimeContractError(f"{label} must contain positive lengths")
    return result


def _timestep_token(value: Any) -> str:
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise CPMRRuntimeContractError("shared_step timestep must be one tensor scalar")
    scalar = value.detach().to(device="cpu").reshape(()).item()
    if isinstance(scalar, bool):
        raise CPMRRuntimeContractError("shared_step timestep must not be bool")
    if isinstance(scalar, int):
        return f"i:{scalar}"
    numeric = float(scalar)
    if not math.isfinite(numeric):
        raise CPMRRuntimeContractError("shared_step timestep is not finite")
    return f"f:{numeric.hex()}"


def resolve_diffusion_core(renderer_or_diffusion: Any) -> Any:
    """Resolve the pinned single-expert ``GEN_Wanx22`` instance."""

    queue = [renderer_or_diffusion]
    seen: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if all(callable(getattr(candidate, name, None)) for name in ("sample", "shared_step")):
            if getattr(candidate, "transformer_2", None) is not None:
                raise CPMRRuntimeContractError("CPMR supports only the 1.3B single expert")
            if getattr(candidate, "transformer", None) is not None:
                return candidate
        get_base_model = getattr(candidate, "get_base_model", None)
        if callable(get_base_model):
            try:
                queue.append(get_base_model())
            except Exception:
                pass
        for name in ("diff_dec", "base_model", "model", "module"):
            nested = getattr(candidate, name, None)
            if nested is not None and nested is not candidate:
                queue.append(nested)
    raise CPMRRuntimeContractError("could not resolve pinned Bernini diffusion core")


@dataclass(frozen=True)
class CPMRStepRecord:
    step_index: int
    timestep_token: str
    negative_prompt_identity_exact: bool
    positive_prompt_identity_exact: bool
    paired_state_identity_exact: bool
    conditioned_encoder_binding: Mapping[str, Any]


@dataclass
class CPMRRuntimeTrace:
    records: list[CPMRStepRecord] = field(default_factory=list)
    sample_calls: int = 0
    shared_step_calls: int = 0

    def receipt(self) -> dict[str, Any]:
        return {
            "sample_calls": self.sample_calls,
            "shared_step_calls": self.shared_step_calls,
            "completed_steps": len(self.records),
            "all_prompt_identity_exact": all(
                item.negative_prompt_identity_exact
                and item.positive_prompt_identity_exact
                for item in self.records
            ),
            "all_paired_state_identity_exact": all(
                item.paired_state_identity_exact for item in self.records
            ),
            "all_bindings_complete": all(
                item.conditioned_encoder_binding.get("completed") is True
                and item.conditioned_encoder_binding.get("consumed") is True
                and item.conditioned_encoder_binding.get("aborted") is False
                and item.conditioned_encoder_binding.get("bound_tensor_released") is True
                for item in self.records
            ),
            "records": [
                {
                    "step_index": item.step_index,
                    "timestep_token": item.timestep_token,
                    "negative_prompt_identity_exact": item.negative_prompt_identity_exact,
                    "positive_prompt_identity_exact": item.positive_prompt_identity_exact,
                    "paired_state_identity_exact": item.paired_state_identity_exact,
                    "conditioned_encoder_binding": dict(item.conditioned_encoder_binding),
                }
                for item in self.records
            ],
        }


@dataclass
class _PendingPair:
    noisy_latents: Any
    timesteps: Any
    rotary_embs: Any
    batch_vae_seqlen: tuple[int, ...]
    timestep_token: str


@dataclass
class _ActiveSample:
    positive_prompt: torch.Tensor
    negative_prompt: torch.Tensor
    completed_steps: int = 0
    pending: Optional[_PendingPair] = None


class InstalledCPMRFinalRenderHook:
    """Reversible one-sample hook for the official negative/positive APG pair."""

    def __init__(
        self,
        renderer_or_diffusion: Any,
        *,
        patch_handle: Any,
        carrier: torch.Tensor,
        activity: torch.Tensor,
        gate: float,
        expected_steps: int = EXPECTED_STEPS,
        binding_factory: Optional[Callable[[], branch.CPMRConditionedEncoderBinding]] = None,
    ) -> None:
        self.diffusion = resolve_diffusion_core(renderer_or_diffusion)
        self.patch_handle = patch_handle
        self.carrier = carrier
        self.activity = activity
        self.gate = float(gate)
        self.expected_steps = int(expected_steps)
        self.trace = CPMRRuntimeTrace()
        self.restored = False
        self._active: Optional[_ActiveSample] = None
        self._patches: list[tuple[Any, str, bool, Any]] = []
        self._original_sample = getattr(self.diffusion, "sample", None)
        self._original_shared_step = getattr(self.diffusion, "shared_step", None)
        if not callable(self._original_sample) or not callable(self._original_shared_step):
            raise CPMRRuntimeContractError("diffusion sample/shared_step must be callable")
        if self.expected_steps != EXPECTED_STEPS:
            raise CPMRRuntimeContractError("CPMR final render is fixed to 40 steps")
        if self.gate not in branch.FROZEN_GATES:
            raise CPMRRuntimeContractError("gate is outside the frozen CPMR registry")
        if not isinstance(carrier, torch.Tensor) or tuple(carrier.shape) != (
            1,
            branch.CARRIER_TOKENS,
            branch.HIDDEN_SIZE,
        ):
            raise CPMRRuntimeContractError("carrier must be [1,1344,1536]")
        if (
            not isinstance(activity, torch.Tensor)
            or tuple(activity.shape) != (1, branch.LATENT_PHASES)
            or activity.dtype != torch.bool
        ):
            raise CPMRRuntimeContractError("activity must be bool [1,21]")
        if carrier.device != activity.device:
            raise CPMRRuntimeContractError("carrier and activity devices differ")
        if not bool(torch.isfinite(carrier).all().item()):
            raise CPMRRuntimeContractError("carrier contains non-finite values")
        branch._require_phase_zero_positive_zero(carrier)
        phase_nonzero = torch.count_nonzero(
            carrier.reshape(1, branch.LATENT_PHASES, 64, branch.HIDDEN_SIZE),
            dim=(2, 3),
        ).ne(0)
        if not torch.equal(phase_nonzero, activity):
            raise CPMRRuntimeContractError("activity differs from carrier nonzero phases")
        if binding_factory is None:
            if not isinstance(patch_handle, branch.CPMRMotionPatchHandle):
                raise CPMRRuntimeContractError("patch_handle must be a CPMR patch handle")
            binding_factory = patch_handle.new_conditioned_encoder_binding
        self._binding_factory = binding_factory
        if "sample" in vars(self.diffusion) or "shared_step" in vars(self.diffusion):
            raise CPMRRuntimeContractError("refusing to stack on instance call overrides")

    def _set_patch(self, name: str, value: Any) -> None:
        instance = vars(self.diffusion)
        had_instance = name in instance
        previous = instance.get(name)
        setattr(self.diffusion, name, value)
        self._patches.append((self.diffusion, name, had_instance, previous))

    def install(self) -> None:
        if self._patches:
            raise CPMRRuntimeContractError("CPMR final-render hook is already installed")

        def sample_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_sample(*args, **kwargs)

        def shared_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_shared_step(*args, **kwargs)

        setattr(sample_wrapper, "_bernini_cpmr_final_render", self)
        setattr(shared_wrapper, "_bernini_cpmr_final_render", self)
        try:
            self._set_patch("shared_step", shared_wrapper)
            self._set_patch("sample", sample_wrapper)
        except Exception:
            self.restore()
            raise
        self.restored = False

    def restore(self) -> None:
        errors: list[Exception] = []
        while self._patches:
            owner, name, had_instance, previous = self._patches.pop()
            try:
                if had_instance:
                    setattr(owner, name, previous)
                else:
                    delattr(owner, name)
            except Exception as error:
                errors.append(error)
        self._active = None
        self.restored = not errors
        if errors:
            raise CPMRRuntimeContractError(
                f"failed to restore {len(errors)} CPMR runtime hook(s)"
            ) from errors[0]

    def _validate_sample(self, values: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if values.get("guidance_mode") != EXPECTED_GUIDANCE_MODE:
            raise CPMRRuntimeContractError("final render requires v2v_apg")
        if int(values.get("num_inference_steps")) != EXPECTED_STEPS:
            raise CPMRRuntimeContractError("final render requires exactly 40 steps")
        if int(values.get("num_frames")) != EXPECTED_FRAMES:
            raise CPMRRuntimeContractError("final render requires exactly 81 frames")
        if not math.isclose(
            float(values.get("flow_shift")), EXPECTED_FLOW_SHIFT, rel_tol=0.0, abs_tol=1.0e-8
        ):
            raise CPMRRuntimeContractError("final render flow shift differs from 5")
        if values.get("image_vae_latents") is not None or values.get("multi_image_vae_latents") is not None:
            raise CPMRRuntimeContractError("CPMR final render forbids image references")
        videos = values.get("multi_video_vae_latents")
        if not isinstance(videos, (list, tuple)) or len(videos) != 1:
            raise CPMRRuntimeContractError("CPMR requires exactly one source video")
        positive = values.get("prompt_embeds")
        negative = values.get("uncond_prompt_embeds")
        if not isinstance(positive, torch.Tensor) or not isinstance(negative, torch.Tensor):
            raise CPMRRuntimeContractError("positive and negative prompt embeddings are required")
        if positive is negative:
            raise CPMRRuntimeContractError("positive and negative prompt objects must differ")
        return positive, negative

    def _wrapped_sample(self, *args: Any, **kwargs: Any) -> Any:
        if self._active is not None or self.trace.sample_calls:
            raise CPMRRuntimeContractError("hook permits one non-nested final sample")
        values = _bind_call(self._original_sample, args, kwargs)
        positive, negative = self._validate_sample(values)
        self._active = _ActiveSample(positive_prompt=positive, negative_prompt=negative)
        try:
            result = self._original_sample(*args, **kwargs)
            state = self._active
            if state is None or state.pending is not None:
                raise CPMRRuntimeContractError("final sample ended inside an APG pair")
            if state.completed_steps != EXPECTED_STEPS:
                raise CPMRRuntimeContractError("final sample did not execute 40 APG pairs")
            if len(self.trace.records) != EXPECTED_STEPS:
                raise CPMRRuntimeContractError("trace length differs from completed steps")
            self.trace.sample_calls = 1
            return result
        finally:
            self._active = None

    def _validate_geometry(self, values: Mapping[str, Any]) -> None:
        noisy = values.get("noisy_latents")
        if _shape(noisy, label="paired noisy latents")[:2] != (1, EXPECTED_GLOBAL_TOKENS):
            raise CPMRRuntimeContractError("paired latent tokens must equal [1,39060,...]")
        if _lengths(values.get("batch_vae_seqlen"), label="batch_vae_seqlen") != (
            EXPECTED_GLOBAL_TOKENS,
        ):
            raise CPMRRuntimeContractError("batch_vae_seqlen must equal [39060]")

    def _invoke(
        self,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        *,
        state: _ActiveSample,
        prompt: torch.Tensor,
        polarity: str,
        binding: Optional[branch.CPMRConditionedEncoderBinding],
    ) -> Any:
        invocation = branch.CPMRMotionInvocation(
            trajectory=branch.FINAL_RENDER,
            polarity=polarity,
            prompt_object=prompt,
            positive_noop_prompt_object=state.positive_prompt,
            conditioned_encoder_binding=binding,
            gate=self.gate,
            carrier=self.carrier,
            activity=self.activity,
        )
        with branch.cpmr_motion_invocation(
            invocation, encoder_hidden_states=prompt
        ):
            return self._original_shared_step(*args, **dict(kwargs))

    def _wrapped_shared_step(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise CPMRRuntimeContractError("shared_step ran outside final sample")
        values = _bind_call(self._original_shared_step, args, kwargs)
        if str(values.get("model_id")) != "transformer_1":
            raise CPMRRuntimeContractError("CPMR observed a non-1.3B transformer route")
        self._validate_geometry(values)
        prompt = values.get("cond_embeds")
        if not isinstance(prompt, torch.Tensor):
            raise CPMRRuntimeContractError("shared_step cond_embeds must be a tensor")
        token = _timestep_token(values.get("timesteps"))
        self.trace.shared_step_calls += 1

        if state.pending is None:
            if prompt is not state.negative_prompt:
                raise CPMRRuntimeContractError("first APG call is not the exact negative prompt")
            result = self._invoke(
                args,
                kwargs,
                state=state,
                prompt=prompt,
                polarity=branch.UNCONDITIONAL,
                binding=None,
            )
            state.pending = _PendingPair(
                noisy_latents=values.get("noisy_latents"),
                timesteps=values.get("timesteps"),
                rotary_embs=values.get("rotary_embs"),
                batch_vae_seqlen=(EXPECTED_GLOBAL_TOKENS,),
                timestep_token=token,
            )
            return result

        pending = state.pending
        if prompt is not state.positive_prompt:
            raise CPMRRuntimeContractError("second APG call is not the exact positive no-op prompt")
        if token != pending.timestep_token:
            raise CPMRRuntimeContractError("negative/positive timestep value differs")
        for name in ("noisy_latents", "timesteps", "rotary_embs"):
            if values.get(name) is not getattr(pending, name):
                raise CPMRRuntimeContractError(f"negative/positive {name} object differs")
        binding = self._binding_factory()
        if not isinstance(binding, branch.CPMRConditionedEncoderBinding):
            raise CPMRRuntimeContractError("binding factory returned the wrong type")
        result = self._invoke(
            args,
            kwargs,
            state=state,
            prompt=prompt,
            polarity=branch.POSITIVE,
            binding=binding,
        )
        binding_receipt = binding.receipt()
        if not (
            binding_receipt.get("completed") is True
            and binding_receipt.get("consumed") is True
            and binding_receipt.get("aborted") is False
            and binding_receipt.get("bound_tensor_released") is True
        ):
            raise CPMRRuntimeContractError("positive binding did not complete exactly")
        self.trace.records.append(
            CPMRStepRecord(
                step_index=state.completed_steps,
                timestep_token=token,
                negative_prompt_identity_exact=True,
                positive_prompt_identity_exact=True,
                paired_state_identity_exact=True,
                conditioned_encoder_binding=binding_receipt,
            )
        )
        state.completed_steps += 1
        state.pending = None
        return result

    def __enter__(self) -> "InstalledCPMRFinalRenderHook":
        self.install()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.restore()


def cpmr_final_render_hook(
    renderer_or_diffusion: Any,
    *,
    patch_handle: Any,
    carrier: torch.Tensor,
    activity: torch.Tensor,
    gate: float,
) -> InstalledCPMRFinalRenderHook:
    return InstalledCPMRFinalRenderHook(
        renderer_or_diffusion,
        patch_handle=patch_handle,
        carrier=carrier,
        activity=activity,
        gate=gate,
    )


__all__ = [
    "CPMRRuntimeContractError",
    "CPMRRuntimeTrace",
    "CPMRStepRecord",
    "InstalledCPMRFinalRenderHook",
    "cpmr_final_render_hook",
    "resolve_diffusion_core",
]
