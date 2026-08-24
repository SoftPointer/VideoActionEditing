#!/usr/bin/env python3
"""DynaEdit-inspired source-state differential flow for Bernini-R.

This module deliberately does *not* patch the Bernini source tree.  It uses the
publicly reachable internals of either ``BerniniRendererModel`` (``diff_dec``)
or ``GEN_Wanx22`` (``shared_step``, ``scheduler`` and ``transformer``).

The inference-time condition set is closed:

* one clean source VAE latent ``S``;
* an action-prompt embedding; and
* a semantic no-op-prompt embedding.

There is no target video, mask, track, swept tube, pose, trajectory, optical
flow, or first-frame anchor.  With one fixed Gaussian ``w`` and a descending
flow-noise schedule, the sampler evaluates

``z_src(sigma) = (1 - sigma) * S + sigma * w``

``z_tar(sigma) = z_edit + z_src(sigma) - S``

``delta_v = V(z_tar, action | S) - V(z_src, no-op | S)``

and integrates the clean-like edit state with explicit Euler,

``z_edit <- z_edit + motion_scale * (sigma_next - sigma) * delta_v``.

This is a DynaEdit-inspired *field-difference sampler*, not an implementation
of DynaEdit's complete method.  In particular, there is no ANC: no fresh probe
noise is drawn inside the loop.  Bernini's normal UniPC path starts from one
Gaussian, so importing ANC's inter-step probe-noise correlation here would be
the wrong abstraction.

The two transformer calls at every step assemble tokens exactly like the
official Bernini video-editing sampler: clean source video tokens use
``source_id=1`` and the queried state uses ``source_id=0``; the prediction is
selected only at the query-token mask.  Full token sequences are passed on
every rank.  Bernini's transformer owns Ulysses padding, slicing, attention
collectives, and output gathering, which makes this assembly compatible with
the audited four-rank path without private rank-wise slicing in this module.

PyTorch is imported lazily so the pure contract helpers and their tests remain
usable in lightweight environments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Optional, Sequence


SOURCE_ID = 1.0
QUERY_ID = 0.0
EXPECTED_ULYSSES_WORLD_SIZE = 4
PACK_PATCH_TEMPORAL = 1
PACK_PATCH_HEIGHT = 2
PACK_PATCH_WIDTH = 2


class DifferentialSamplerContractError(RuntimeError):
    """Raised before or during sampling when a Bernini contract is violated."""


@dataclass(frozen=True)
class DifferentialFlowConfig:
    """Runtime controls for :func:`sample_differential_flow`.

    ``motion_scale`` scales only the source-relative velocity difference.  A
    value of zero is an exact early identity bypass.  Values above one are
    useful as a motion-strength ablation but may amplify model errors.
    """

    num_inference_steps: int = 40
    flow_shift: float = 5.0
    seed: int = 20260806
    motion_scale: float = 1.0

    def validate(self) -> "DifferentialFlowConfig":
        if type(self.num_inference_steps) is not int or self.num_inference_steps <= 0:
            raise DifferentialSamplerContractError(
                "num_inference_steps must be a positive integer"
            )
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise DifferentialSamplerContractError("seed must be in [0, 2^63)")
        if (
            isinstance(self.flow_shift, bool)
            or not isinstance(self.flow_shift, (int, float))
            or not math.isfinite(float(self.flow_shift))
            or float(self.flow_shift) <= 0.0
        ):
            raise DifferentialSamplerContractError("flow_shift must be finite and positive")
        if (
            isinstance(self.motion_scale, bool)
            or not isinstance(self.motion_scale, (int, float))
            or not math.isfinite(float(self.motion_scale))
            or float(self.motion_scale) < 0.0
        ):
            raise DifferentialSamplerContractError(
                "motion_scale must be finite and non-negative"
            )
        return self


@dataclass(frozen=True)
class LatentLayout:
    """Wan latent and packed-token geometry for one source video."""

    batch: int
    channels: int
    frames: int
    height: int
    width: int
    tokens: int
    packed_channels: int


@dataclass(frozen=True)
class DifferentialFlowTrace:
    """Optional diagnostics; values are rank-local but deterministic by contract."""

    identity_bypassed: bool
    sigmas: tuple[float, ...]
    delta_rms: tuple[float, ...]


def sampler_contract() -> dict[str, Any]:
    """Return a serialisable statement of the scientific/runtime contract."""

    return {
        "method": "bernini_source_state_differential_flow",
        "status": "dynaedit_inspired_field_difference_not_full_dynaedit",
        "inference_conditions": [
            "clean_source_vae_latent",
            "action_prompt_embedding",
            "noop_prompt_embedding",
        ],
        "forbidden_conditions": [
            "target_video",
            "mask",
            "track",
            "swept_tube",
            "pose",
            "trajectory",
            "optical_flow",
            "first_frame_anchor",
        ],
        "source_path": "z_src=(1-sigma)*source+sigma*fixed_noise",
        "target_query_path": "z_tar=z_edit+z_src-source",
        "velocity_field": "V(action,z_tar|source)-V(noop,z_src|source)",
        "integrator": "descending_sigma_explicit_euler",
        "noise_policy": "one_fixed_shared_gaussian",
        "fresh_per_step_probe_noise": False,
        "anc": False,
        "source_id": SOURCE_ID,
        "query_id": QUERY_ID,
        "ulysses_world_size_tested": EXPECTED_ULYSSES_WORLD_SIZE,
        "sequence_parallel_owner": "official_bernini_transformer",
    }


def validate_latent_shape(shape: Sequence[int]) -> LatentLayout:
    """Validate ``[1,C,T,H,W]`` and compute official Wan 1x2x2 packing."""

    values = tuple(shape)
    if len(values) != 5 or any(type(value) is not int for value in values):
        raise DifferentialSamplerContractError(
            f"source latent shape must be five integers [1,C,T,H,W], got {values!r}"
        )
    batch, channels, frames, height, width = values
    if batch != 1:
        raise DifferentialSamplerContractError(
            f"official Bernini sampling supports batch size one, got {batch}"
        )
    if min(channels, frames, height, width) <= 0:
        raise DifferentialSamplerContractError("source latent dimensions must be positive")
    if height % PACK_PATCH_HEIGHT or width % PACK_PATCH_WIDTH:
        raise DifferentialSamplerContractError(
            "source latent height and width must be divisible by Wan's 2x2 patch"
        )
    tokens = frames * (height // PACK_PATCH_HEIGHT) * (width // PACK_PATCH_WIDTH)
    packed_channels = (
        channels * PACK_PATCH_TEMPORAL * PACK_PATCH_HEIGHT * PACK_PATCH_WIDTH
    )
    return LatentLayout(
        batch=batch,
        channels=channels,
        frames=frames,
        height=height,
        width=width,
        tokens=tokens,
        packed_channels=packed_channels,
    )


def descending_sigma_intervals(
    sigmas: Iterable[float], *, expected_steps: int
) -> tuple[tuple[float, float], ...]:
    """Normalise a scheduler sigma list into exactly ``expected_steps`` pairs.

    Diffusers UniPC exposes ``steps + 1`` sigmas including the terminal zero.
    Bernini's small ``FlowMatchScheduler`` exposes ``steps`` sigmas and treats
    its implicit next value after the last step as zero.  Both forms are
    accepted, and every interval is required to be non-increasing in [0, 1].
    """

    if type(expected_steps) is not int or expected_steps <= 0:
        raise DifferentialSamplerContractError("expected_steps must be positive")
    try:
        values = tuple(float(value) for value in sigmas)
    except (TypeError, ValueError) as error:
        raise DifferentialSamplerContractError("scheduler sigmas are not numeric") from error
    if len(values) == expected_steps:
        values = values + (0.0,)
    elif len(values) != expected_steps + 1:
        raise DifferentialSamplerContractError(
            "scheduler must expose either steps or steps+1 sigmas; "
            f"got {len(values)} for {expected_steps} steps"
        )
    tolerance = 1.0e-6
    for index, value in enumerate(values):
        if not math.isfinite(value) or value < -tolerance or value > 1.0 + tolerance:
            raise DifferentialSamplerContractError(
                f"scheduler sigma {index} is outside finite [0,1]: {value!r}"
            )
    for index, (current, following) in enumerate(zip(values, values[1:])):
        if following > current + tolerance:
            raise DifferentialSamplerContractError(
                f"scheduler sigmas ascend at interval {index}: {current} -> {following}"
            )
    if abs(values[-1]) > tolerance:
        raise DifferentialSamplerContractError(
            f"descending flow integration must terminate at sigma=0, got {values[-1]}"
        )
    return tuple(zip(values, values[1:]))


def prompts_are_exactly_identical(
    action_prompt_embeds: Any,
    noop_prompt_embeds: Any,
    *,
    tensor_equal: Optional[Callable[[Any, Any], bool]] = None,
) -> bool:
    """Return true only for identity or exact tensor equality.

    The injectable comparator keeps this helper independently testable without
    importing PyTorch.  Shape and dtype must agree before any comparator runs.
    """

    if action_prompt_embeds is noop_prompt_embeds:
        return True
    action_shape = getattr(action_prompt_embeds, "shape", None)
    noop_shape = getattr(noop_prompt_embeds, "shape", None)
    if action_shape is None or noop_shape is None or tuple(action_shape) != tuple(noop_shape):
        return False
    if getattr(action_prompt_embeds, "dtype", None) != getattr(
        noop_prompt_embeds, "dtype", None
    ):
        return False
    action_device = getattr(action_prompt_embeds, "device", None)
    noop_device = getattr(noop_prompt_embeds, "device", None)
    if action_device is not None and noop_device is not None and action_device != noop_device:
        return False
    if tensor_equal is not None:
        return bool(tensor_equal(action_prompt_embeds, noop_prompt_embeds))
    try:
        import torch
    except Exception:
        return False
    if not isinstance(action_prompt_embeds, torch.Tensor) or not isinstance(
        noop_prompt_embeds, torch.Tensor
    ):
        return False
    return bool(torch.equal(action_prompt_embeds, noop_prompt_embeds))


def resolve_diffusion_core(renderer_or_diffusion: Any) -> Any:
    """Resolve ``GEN_Wanx22`` from an official renderer or PEFT wrapper."""

    queue = [renderer_or_diffusion]
    seen: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if all(hasattr(candidate, name) for name in ("shared_step", "scheduler", "transformer")):
            return candidate
        diff_dec = getattr(candidate, "diff_dec", None)
        if diff_dec is not None:
            queue.append(diff_dec)
        get_base_model = getattr(candidate, "get_base_model", None)
        if callable(get_base_model):
            try:
                queue.append(get_base_model())
            except Exception:
                # A wrapper may expose the method before it is fully loaded;
                # other resolution paths remain valid.
                pass
        for name in ("base_model", "model", "module"):
            nested = getattr(candidate, name, None)
            if nested is not None:
                queue.append(nested)
    raise DifferentialSamplerContractError(
        "could not resolve official GEN_Wanx22 internals from renderer/model"
    )


def _require_torch() -> Any:
    try:
        import torch
    except Exception as error:  # pragma: no cover - exercised on AUH
        raise DifferentialSamplerContractError(
            "PyTorch is required for differential-flow sampling"
        ) from error
    return torch


def _pack_spatial_latent(latent: Any, layout: Optional[LatentLayout] = None) -> Any:
    """Official ``b c t (h ph) (w pw) -> b (t h w) (ph pw c)`` packing."""

    if layout is None:
        layout = validate_latent_shape(tuple(int(value) for value in latent.shape))
    # [B,C,T,Hg,ph,Wg,pw] -> [B,T,Hg,Wg,ph,pw,C]
    return (
        latent.reshape(
            layout.batch,
            layout.channels,
            layout.frames,
            layout.height // PACK_PATCH_HEIGHT,
            PACK_PATCH_HEIGHT,
            layout.width // PACK_PATCH_WIDTH,
            PACK_PATCH_WIDTH,
        )
        .permute(0, 2, 3, 5, 4, 6, 1)
        .reshape(layout.batch, layout.tokens, layout.packed_channels)
    )


def _unpack_spatial_latent(packed: Any, layout: LatentLayout) -> Any:
    """Inverse of :func:`_pack_spatial_latent` using official Wan ordering."""

    actual = tuple(int(value) for value in packed.shape)
    expected = (layout.batch, layout.tokens, layout.packed_channels)
    if actual != expected:
        raise DifferentialSamplerContractError(
            f"packed prediction shape differs from Wan layout: {actual} != {expected}"
        )
    # [B,T,Hg,Wg,ph,pw,C] -> [B,C,T,Hg,ph,Wg,pw]
    return (
        packed.reshape(
            layout.batch,
            layout.frames,
            layout.height // PACK_PATCH_HEIGHT,
            layout.width // PACK_PATCH_WIDTH,
            PACK_PATCH_HEIGHT,
            PACK_PATCH_WIDTH,
            layout.channels,
        )
        .permute(0, 6, 1, 2, 4, 3, 5)
        .reshape(
            layout.batch,
            layout.channels,
            layout.frames,
            layout.height,
            layout.width,
        )
    )


def _module_dtype(module: Any, fallback: Any) -> Any:
    dtype = getattr(module, "dtype", None)
    if dtype is not None:
        return dtype
    try:
        return next(module.parameters()).dtype
    except (AttributeError, StopIteration, TypeError):
        return fallback


def _validate_runtime_inputs(
    diffusion: Any,
    source_latent: Any,
    action_prompt_embeds: Any,
    noop_prompt_embeds: Any,
) -> tuple[LatentLayout, Any]:
    torch = _require_torch()
    if not isinstance(source_latent, torch.Tensor) or not source_latent.is_floating_point():
        raise DifferentialSamplerContractError("source_latent must be a floating torch tensor")
    layout = validate_latent_shape(tuple(int(value) for value in source_latent.shape))
    transformer = getattr(diffusion, "transformer", None)
    transformer_2 = getattr(diffusion, "transformer_2", None)
    if transformer is None or transformer_2 is not None:
        raise DifferentialSamplerContractError(
            "this sampler currently supports Bernini-R 1.3B's single transformer_1 only"
        )
    in_channels = getattr(getattr(transformer, "config", None), "in_channels", None)
    if in_channels is not None and int(in_channels) != layout.channels:
        raise DifferentialSamplerContractError(
            f"source latent channels {layout.channels} != transformer in_channels {in_channels}"
        )
    for label, embeds in (
        ("action_prompt_embeds", action_prompt_embeds),
        ("noop_prompt_embeds", noop_prompt_embeds),
    ):
        if not isinstance(embeds, torch.Tensor) or embeds.ndim != 3:
            raise DifferentialSamplerContractError(f"{label} must be a [1,L,D] torch tensor")
        if int(embeds.shape[0]) != 1 or int(embeds.shape[1]) <= 0 or int(embeds.shape[2]) <= 0:
            raise DifferentialSamplerContractError(f"{label} must have non-empty shape [1,L,D]")
    if int(action_prompt_embeds.shape[2]) != int(noop_prompt_embeds.shape[2]):
        raise DifferentialSamplerContractError("action/no-op embedding dimensions differ")
    text_dim = getattr(getattr(transformer, "config", None), "text_dim", None)
    if text_dim is not None and int(action_prompt_embeds.shape[2]) != int(text_dim):
        raise DifferentialSamplerContractError(
            f"prompt embedding dimension {action_prompt_embeds.shape[2]} != text_dim {text_dim}"
        )
    return layout, transformer


def _set_scheduler_timesteps(diffusion: Any, config: DifferentialFlowConfig, device: Any) -> Any:
    scheduler = diffusion.scheduler
    if bool(getattr(diffusion, "use_unipc", False)):
        scheduler_config = getattr(scheduler, "config", None)
        configured_shift = getattr(scheduler_config, "flow_shift", None)
        if configured_shift is None and isinstance(scheduler_config, dict):
            configured_shift = scheduler_config.get("flow_shift")
        if configured_shift is not None and not math.isclose(
            float(configured_shift), float(config.flow_shift), rel_tol=0.0, abs_tol=1.0e-8
        ):
            raise DifferentialSamplerContractError(
                "the existing UniPC scheduler flow_shift differs from the requested "
                f"contract: {configured_shift} != {config.flow_shift}; construct the "
                "Bernini renderer with the intended shift before sampling"
            )
        scheduler.set_timesteps(config.num_inference_steps)
    else:
        # Bernini's local FlowMatchScheduler accepts these arguments; retain a
        # narrow fallback for scheduler versions without device/dtype kwargs.
        try:
            scheduler.set_timesteps(
                config.num_inference_steps,
                shift=float(config.flow_shift),
                device=str(device),
            )
        except TypeError:
            scheduler.set_timesteps(
                config.num_inference_steps, shift=float(config.flow_shift)
            )
    timesteps = scheduler.timesteps.to(device)
    raw_sigmas = scheduler.sigmas.detach().to(device="cpu", dtype=_require_torch().float64).tolist()
    intervals = descending_sigma_intervals(
        raw_sigmas, expected_steps=config.num_inference_steps
    )
    if len(timesteps) != len(intervals):
        raise DifferentialSamplerContractError(
            f"scheduler timestep/sigma interval mismatch: {len(timesteps)} != {len(intervals)}"
        )
    return timesteps, intervals


def _make_fixed_noise(source_latent: Any, *, seed: int) -> Any:
    torch = _require_torch()
    # Generate on CPU exactly once.  Every Ulysses rank receives the same seed
    # and therefore the same full tensor before Bernini performs its own token
    # sharding.  There is intentionally no random draw inside the solver loop.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        tuple(int(value) for value in source_latent.shape),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(device=source_latent.device)


def _patch_source_condition(transformer: Any, source_latent: Any) -> tuple[Any, Any, Any]:
    torch = _require_torch()
    compute_dtype = _module_dtype(transformer, source_latent.dtype)
    source_tokens, source_rotary = transformer.patch_vae_latent(
        source_latent.to(dtype=compute_dtype), source_id=SOURCE_ID
    )
    source_mask = torch.zeros(
        int(source_tokens.shape[1]), device=source_tokens.device, dtype=torch.bool
    )
    return source_tokens, source_rotary, source_mask


def _predict_source_conditioned_velocity(
    *,
    diffusion: Any,
    transformer: Any,
    source_condition: tuple[Any, Any, Any],
    query_latent: Any,
    prompt_embeds: Any,
    timestep: Any,
) -> Any:
    """Assemble ``[clean source, query]`` and return query-token velocity."""

    torch = _require_torch()
    source_tokens, source_rotary, source_mask = source_condition
    compute_dtype = _module_dtype(transformer, query_latent.dtype)
    query_tokens, query_rotary = transformer.patch_vae_latent(
        query_latent.to(dtype=compute_dtype), source_id=QUERY_ID
    )
    query_mask = torch.ones(
        int(query_tokens.shape[1]), device=query_tokens.device, dtype=torch.bool
    )
    latent_input = torch.cat([source_tokens, query_tokens], dim=1).to(compute_dtype)
    rotary_input = torch.cat([source_rotary, query_rotary], dim=2)
    target_mask = torch.cat([source_mask, query_mask], dim=0)
    text = prompt_embeds.to(device=query_latent.device, dtype=compute_dtype)
    prediction = diffusion.shared_step(
        model_id="transformer_1",
        noisy_latents=latent_input,
        timesteps=timestep.expand(1),
        cond_embeds=text,
        rotary_embs=rotary_input,
        batch_vae_seqlen=[int(latent_input.shape[1])],
        batch_text_seqlen=[int(text.shape[1])],
    )
    return prediction[:, target_mask, :]


def sample_differential_flow(
    renderer_or_diffusion: Any,
    *,
    source_latent: Any,
    action_prompt_embeds: Any,
    noop_prompt_embeds: Any,
    config: Optional[DifferentialFlowConfig] = None,
    return_trace: bool = False,
) -> Any:
    """Sample an action edit through a source-relative Bernini velocity field.

    ``action_prompt_embeds`` and ``noop_prompt_embeds`` are expected to be the
    outputs of the same official renderer ``encode_prompt`` path (including
    whatever shared mv2v system prefix the caller uses); this function does not
    run or alter T5.

    All Ulysses ranks must call this function with identical inputs, config and
    call order.  The function passes each *full* assembled sequence into the
    official transformer; Bernini handles four-rank sequence parallelism and
    gathers the full query prediction before the Euler update.

    Exact action/no-op embeddings (or ``motion_scale == 0``) return the exact
    input ``source_latent`` object before PyTorch/model/scheduler work.  This is
    stronger than numerical cancellation and makes the identity control a
    bitwise bypass.
    """

    runtime = (config or DifferentialFlowConfig()).validate()
    identity = float(runtime.motion_scale) == 0.0 or prompts_are_exactly_identical(
        action_prompt_embeds, noop_prompt_embeds
    )
    if identity:
        if return_trace:
            return source_latent, DifferentialFlowTrace(True, (), ())
        return source_latent

    torch = _require_torch()
    diffusion = resolve_diffusion_core(renderer_or_diffusion)
    layout, transformer = _validate_runtime_inputs(
        diffusion, source_latent, action_prompt_embeds, noop_prompt_embeds
    )
    timesteps, intervals = _set_scheduler_timesteps(
        diffusion, runtime, source_latent.device
    )
    fixed_noise = _make_fixed_noise(source_latent, seed=runtime.seed)
    source_clean = source_latent.detach().to(dtype=torch.float32)
    source_packed = _pack_spatial_latent(source_clean, layout)
    noise_packed = _pack_spatial_latent(fixed_noise, layout)
    edit_packed = source_packed.clone()

    delta_rms: list[float] = []
    with torch.no_grad():
        source_condition = _patch_source_condition(transformer, source_clean)
        for index, (sigma_value, next_sigma_value) in enumerate(intervals):
            sigma = torch.as_tensor(
                sigma_value, device=source_clean.device, dtype=torch.float32
            )
            source_state_packed = (1.0 - sigma) * source_packed + sigma * noise_packed
            target_state_packed = edit_packed + source_state_packed - source_packed
            source_state = _unpack_spatial_latent(source_state_packed, layout)
            target_state = _unpack_spatial_latent(target_state_packed, layout)
            timestep = timesteps[index]

            action_velocity = _predict_source_conditioned_velocity(
                diffusion=diffusion,
                transformer=transformer,
                source_condition=source_condition,
                query_latent=target_state,
                prompt_embeds=action_prompt_embeds,
                timestep=timestep,
            )
            noop_velocity = _predict_source_conditioned_velocity(
                diffusion=diffusion,
                transformer=transformer,
                source_condition=source_condition,
                query_latent=source_state,
                prompt_embeds=noop_prompt_embeds,
                timestep=timestep,
            )
            delta_velocity = action_velocity.float() - noop_velocity.float()
            actual_shape = tuple(int(value) for value in delta_velocity.shape)
            expected_shape = (layout.batch, layout.tokens, layout.packed_channels)
            if actual_shape != expected_shape:
                raise DifferentialSamplerContractError(
                    "Bernini query prediction shape differs from packed source: "
                    f"{actual_shape} != {expected_shape}"
                )
            delta_sigma = float(next_sigma_value - sigma_value)
            edit_packed = edit_packed + (
                float(runtime.motion_scale) * delta_sigma * delta_velocity
            )
            if return_trace:
                delta_rms.append(
                    float(delta_velocity.square().mean().sqrt().detach().cpu().item())
                )

    result = _unpack_spatial_latent(edit_packed, layout)
    if return_trace:
        trace_sigmas = tuple(intervals[0][:1]) + tuple(pair[1] for pair in intervals)
        return result, DifferentialFlowTrace(False, trace_sigmas, tuple(delta_rms))
    return result


__all__ = [
    "DifferentialFlowConfig",
    "DifferentialFlowTrace",
    "DifferentialSamplerContractError",
    "EXPECTED_ULYSSES_WORLD_SIZE",
    "LatentLayout",
    "QUERY_ID",
    "SOURCE_ID",
    "descending_sigma_intervals",
    "prompts_are_exactly_identical",
    "resolve_diffusion_core",
    "sample_differential_flow",
    "sampler_contract",
    "validate_latent_shape",
]
