#!/usr/bin/env python3
"""Compose a no-op-trained preservation residual with native RV2V action.

The official Bernini RV2V action path remains unchanged.  Immediately before
each UniPC step, this reversible patch evaluates the same current target state
twice under the exact no-op prompt and a source-only carrier pack:

    delta_pres = v_adapted_noop - v_frozen_noop
    v_edit      = v_native_action + delta_pres

There is no scale, sigma gate, clipping, feature score, VLM score, target
video, or synthetic target.  The adapter never receives the action prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from typing import Any, Mapping, Sequence

import inference_sigma_strata as exact40
import preservation_source_role_v1 as role
import source_noised_ladder_v1 as ladder
import train_source_self_role_repaint as packer
from self_guided_action_field_v1 import (
    _coerce_scalar,
    _extract_argument,
    _packed_to_spatial,
    _replace_argument,
    _resolve_sigma,
    _shape,
    _tensor_rms,
)


class PreservationResidualPatchError(RuntimeError):
    """Raised before an ambiguous scheduler composition is executed."""


@dataclass(frozen=True)
class PreservationPatchConfig:
    target_latent_shape: tuple[int, int, int, int, int]
    sequence_parallel_size: int = 4
    expected_steps: int = 40
    expected_model_id: str = "transformer_1"
    expected_guidance_mode: str = "v2v_apg"

    def validate(self) -> None:
        shape = tuple(self.target_latent_shape)
        if (
            len(shape) != 5
            or shape[0] != 1
            or shape[1] != 16
            or any(type(item) is not int or item <= 0 for item in shape)
            or shape[3] % 2
            or shape[4] % 2
        ):
            raise PreservationResidualPatchError(
                "target latent must be positive [1,16,T,even,even]"
            )
        if self.expected_steps != len(exact40.PINNED_TIMESTEPS) or self.expected_steps != 40:
            raise PreservationResidualPatchError("patch requires the exact40 sampler")
        if self.sequence_parallel_size not in {2, 4}:
            raise PreservationResidualPatchError("preservation inference requires SP2 or SP4")
        if self.expected_model_id != "transformer_1" or self.expected_guidance_mode != "v2v_apg":
            raise PreservationResidualPatchError("Bernini action path contract differs")

    @property
    def target_tokens(self) -> int:
        _, _, phases, height, width = self.target_latent_shape
        return phases * (height // 2) * (width // 2)


def _bind(callable_object: Any, args: Sequence[Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = inspect.signature(callable_object).bind(*args, **kwargs)
        value.apply_defaults()
        return dict(value.arguments)
    except (TypeError, ValueError) as error:
        raise PreservationResidualPatchError("call signature differs") from error


def _packed_rotary(
    rope: Any,
    donor: Any,
    references: Sequence[Any],
    target: Any,
    *,
    expected_tokens: int,
) -> Any:
    """Return Bernini's pre-transformer ``[1,1,S,D/2]`` rotary pack.

    ``WanTransformer3DModel.forward`` transposes axes 1 and 2 immediately
    before self-attention.  Supplying the post-transpose layout here changes
    that into ``[1,1,S,D/2]`` inside attention and makes the head axis collide
    with ``S`` under Ulysses4.  Training uses this exact pre-transpose layout.
    """

    import torch

    if type(expected_tokens) is not int or expected_tokens <= 0:
        raise PreservationResidualPatchError("rotary token count differs")
    rotary = torch.cat(
        (
            rope(donor, source_id=1),
            *(rope(item, source_id=index + 2) for index, item in enumerate(references)),
            rope(target, source_id=0),
        ),
        dim=2,
    ).contiguous()
    if (
        rotary.ndim != 4
        or tuple(rotary.shape[:3]) != (1, 1, expected_tokens)
        or int(rotary.shape[3]) <= 0
        or not bool(torch.isfinite(rotary).all())
    ):
        raise PreservationResidualPatchError("packed rotary geometry differs")
    return rotary


class NativeRV2VPreservationResidualPatch:
    """One-sample, all-exact40, unit-gain preservation composition."""

    def __init__(
        self,
        diffusion: Any,
        *,
        adapter: role.SourceSelfAdapterHandle,
        noop_prompt_embeds: Any,
        noop_text_lens: Any,
        source_latent: Any,
        source_references: Sequence[Any],
        rope: Any,
        config: PreservationPatchConfig,
    ) -> None:
        import torch

        config.validate()
        if len(source_references) != role.REFERENCE_COUNT:
            raise PreservationResidualPatchError("three source references are required")
        if not isinstance(source_latent, torch.Tensor) or tuple(source_latent.shape) != config.target_latent_shape:
            raise PreservationResidualPatchError("source latent geometry differs")
        reference_shape = (
            1,
            16,
            1,
            config.target_latent_shape[3],
            config.target_latent_shape[4],
        )
        if any(not isinstance(item, torch.Tensor) or tuple(item.shape) != reference_shape for item in source_references):
            raise PreservationResidualPatchError("source reference geometry differs")
        if (
            not isinstance(noop_prompt_embeds, torch.Tensor)
            or noop_prompt_embeds.ndim != 3
            or int(noop_prompt_embeds.shape[0]) != 1
            or noop_prompt_embeds.requires_grad
            or not bool(torch.isfinite(noop_prompt_embeds).all())
        ):
            raise PreservationResidualPatchError("no-op prompt embedding differs")
        if noop_text_lens is None:
            raise PreservationResidualPatchError("no-op text lengths are required")
        scheduler = getattr(diffusion, "scheduler", None)
        originals = {
            "sample": getattr(diffusion, "sample", None),
            "scheduler_step": getattr(scheduler, "step", None),
            "shared_step": getattr(diffusion, "shared_step", None),
        }
        if any(not callable(item) for item in originals.values()):
            raise PreservationResidualPatchError("diffusion call graph differs")
        if getattr(diffusion, "use_unipc", None) is not True or getattr(diffusion, "transformer_2", None) is not None:
            raise PreservationResidualPatchError("patch requires single-expert UniPC")
        if adapter.transformer is not getattr(diffusion, "transformer", None):
            raise PreservationResidualPatchError("adapter/diffusion transformer identity differs")
        if role.active_route() is not None:
            raise PreservationResidualPatchError("a source role route is already active")

        self.diffusion = diffusion
        self.scheduler = scheduler
        self.adapter = adapter
        self.noop_prompt_embeds = noop_prompt_embeds
        self.noop_text_lens = noop_text_lens
        self.source_latent = source_latent.detach().float().contiguous()
        self.source_references = tuple(item.detach().float().contiguous() for item in source_references)
        self.rope = rope
        self.config = config
        self.original_sample = originals["sample"]
        self.original_scheduler_step = originals["scheduler_step"]
        self.original_shared_step = originals["shared_step"]
        self.installed = False
        self.restored = False
        self.finalized = False
        self.sample_calls = 0
        self.scheduler_calls = 0
        self.noop_forwards = 0
        self.initial_epsilon: Any = None
        self.trace: list[dict[str, Any]] = []

    def _query(self, sample: Any, timestep: Any, sigma: float, *, enabled: bool) -> Any:
        import torch
        import torch.distributed as dist

        target = _packed_to_spatial(sample, self.config.target_latent_shape).float()
        if self.initial_epsilon is None:
            self.initial_epsilon = target.detach().clone().contiguous()
        epsilon = self.initial_epsilon
        source_state = ladder.shared_noise_source_state(
            self.source_latent, epsilon, sigma
        )
        donor_patches = packer.pack_latent_patches(
            source_state[0], phases=self.config.target_latent_shape[2]
        )
        reference_patches = [
            packer.pack_latent_patches(item[0], phases=1)
            for item in self.source_references
        ]
        target_patches = packer.pack_latent_patches(
            target[0], phases=self.config.target_latent_shape[2]
        )
        layout = role.TokenRoleLayout.contiguous(
            donor_tokens=int(donor_patches.shape[0]),
            reference_tokens=[int(item.shape[0]) for item in reference_patches],
            target_tokens=int(target_patches.shape[0]),
        )
        patches = torch.cat((donor_patches, *reference_patches, target_patches), dim=0).to(sample.device)
        invocation = role.RouteInvocation(
            layout,
            sequence_parallel_rank=int(dist.get_rank()) % self.config.sequence_parallel_size,
            sequence_parallel_size=self.config.sequence_parallel_size,
            enabled=enabled,
        )
        transformer = self.adapter.transformer
        rotary = _packed_rotary(
            self.rope,
            source_state,
            self.source_references,
            target,
            expected_tokens=layout.total_tokens,
        )
        with self.adapter.route(invocation), torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            embedded = transformer.patch_embedding(patches).flatten(1).unsqueeze(0)
            prediction = self.original_shared_step(
                model_id=self.config.expected_model_id,
                noisy_latents=embedded,
                timesteps=timestep.expand(1),
                cond_embeds=self.noop_prompt_embeds,
                rotary_embs=rotary,
                batch_vae_seqlen=[layout.total_tokens],
                batch_text_seqlen=self.noop_text_lens,
            )
        self.noop_forwards += 1
        tail = prediction[:, layout.condition_tokens :, :]
        expected = (1, self.config.target_tokens, 64)
        if tuple(tail.shape) != expected or not bool(torch.isfinite(tail).all()):
            raise PreservationResidualPatchError("no-op target prediction differs")
        return tail

    def _wrapped_scheduler_step(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        official = _extract_argument(args, kwargs, index=0, name="model_output")
        timestep = _extract_argument(args, kwargs, index=1, name="timestep")
        sample = _extract_argument(args, kwargs, index=2, name="sample")
        expected = (1, self.config.target_tokens, 64)
        if any(not isinstance(item, torch.Tensor) or tuple(item.shape) != expected for item in (official, sample)):
            raise PreservationResidualPatchError("scheduler tensor geometry differs")
        step_index, _, sigma = _resolve_sigma(self.scheduler, timestep)
        if step_index != self.scheduler_calls or step_index >= self.config.expected_steps:
            raise PreservationResidualPatchError("scheduler coordinate differs")
        expected_timestep = exact40.PINNED_TIMESTEPS[step_index]
        if int(_coerce_scalar(timestep, label="timestep")) != expected_timestep:
            raise PreservationResidualPatchError("exact40 timestep differs")
        frozen = self._query(sample, timestep, sigma, enabled=False)
        adapted = self._query(sample, timestep, sigma, enabled=True)
        correction = (adapted.float() - frozen.float()).to(official.dtype)
        edited = official + correction
        if not bool(torch.isfinite(edited).all()):
            raise PreservationResidualPatchError("composed action velocity is non-finite")
        call_args, call_kwargs = _replace_argument(
            self.original_scheduler_step,
            args,
            kwargs,
            name="model_output",
            value=edited,
        )
        result = self.original_scheduler_step(*call_args, **call_kwargs)
        self.trace.append(
            {
                "step_index": step_index,
                "timestep": expected_timestep,
                "sigma": sigma,
                "unit_gain": True,
                "frozen_noop_rms": float(_tensor_rms(frozen).item()),
                "adapted_noop_rms": float(_tensor_rms(adapted).item()),
                "preservation_residual_rms": float(_tensor_rms(correction).item()),
                "native_action_rms": float(_tensor_rms(official).item()),
                "feature_reward": False,
            }
        )
        self.scheduler_calls += 1
        return result

    def _wrapped_sample(self, *args: Any, **kwargs: Any) -> Any:
        if self.sample_calls or self.initial_epsilon is not None:
            raise PreservationResidualPatchError("patch permits one sample call")
        values = _bind(self.original_sample, args, kwargs)
        if values.get("guidance_mode") != self.config.expected_guidance_mode:
            raise PreservationResidualPatchError("native guidance mode differs")
        if int(values.get("num_inference_steps")) != self.config.expected_steps:
            raise PreservationResidualPatchError("native step count differs")
        self.sample_calls += 1
        return self.original_sample(*args, **kwargs)

    def install(self) -> None:
        if self.installed or self.restored or self.finalized:
            raise PreservationResidualPatchError("patch lifecycle differs")

        def sample_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_sample(*args, **kwargs)

        def scheduler_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_scheduler_step(*args, **kwargs)

        setattr(self.diffusion, "sample", sample_wrapper)
        setattr(self.scheduler, "step", scheduler_wrapper)
        self.installed = True

    def restore(self) -> None:
        if not self.installed or self.restored:
            raise PreservationResidualPatchError("restore lifecycle differs")
        setattr(self.diffusion, "sample", self.original_sample)
        setattr(self.scheduler, "step", self.original_scheduler_step)
        self.installed = False
        self.restored = True

    def finalize(self) -> Mapping[str, Any]:
        if not self.restored or self.finalized:
            raise PreservationResidualPatchError("finalize lifecycle differs")
        if (
            self.sample_calls != 1
            or self.scheduler_calls != self.config.expected_steps
            or self.noop_forwards != 2 * self.config.expected_steps
            or len(self.trace) != self.config.expected_steps
        ):
            raise PreservationResidualPatchError("patch execution closure differs")
        self.finalized = True
        return {
            "schema_version": "bernini-preservation-residual-action-patch-v1",
            "native_action_path_frozen": True,
            "adapter_action_text_input": False,
            "composition": "v_native_action+(v_adapted_noop-v_frozen_noop)",
            "unit_gain": True,
            "sigma_gate": False,
            "clipping": False,
            "feature_reward": False,
            "vlm_reward": False,
            "scheduler_steps": self.scheduler_calls,
            "noop_forwards": self.noop_forwards,
            "trace": list(self.trace),
        }


__all__ = [
    "NativeRV2VPreservationResidualPatch",
    "PreservationPatchConfig",
    "PreservationResidualPatchError",
]
