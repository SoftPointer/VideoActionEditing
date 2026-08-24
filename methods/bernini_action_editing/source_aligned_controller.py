#!/usr/bin/env python3
"""Source-aligned full-video differential controller for Bernini-R.

This module is a deliberately narrow, DynaEdit-inspired *Bernini adaptation*.
It is not an implementation claim for the unpublished DynaEdit code:

* Bernini-R is used as a full-source MV2V editor rather than the paper's I2V
  backbone.  The clean source video is the visual condition on every query.
* The public interface remains source video plus edit instruction.  Internally,
  the instruction and a fixed semantic no-op are encoded by the same Bernini
  prompt path; no source caption or target first-frame image is required.
* DynaEdit does not fully specify the feature space used by its SGA cosine.  We
  use the raw packed VAE latent and expose that choice in :func:`controller_contract`.
* When the early SGA bank changes from K noise chains to one ANC chain, the
  paper's public description does not specify the collapse.  Here the candidate
  noises are combined with the same SGA weights and divided by
  ``sqrt(sum(weight**2))``.  This preserves unit marginal variance only under
  independent Gaussian candidates with exogenous weights; because SGA weights
  depend on the candidates, it is an explicit approximation.

For each descending flow interval and candidate correlated noise ``w_j`` the
controller evaluates the source-relative FlowEdit field

``z_src_j = (1 - sigma) * source + sigma * w_j``

``z_tar_j = edit + z_src_j - source``

``delta_j = V(z_tar_j, action | clean_source)
           - V(z_src_j, no_op | clean_source)``.

During the first configurable steps, SGA projects each candidate toward the
clean endpoint, scores source similarity, and soft-aggregates the directions.
ANC is a real timestep-varying Markov process, not a fixed-noise alias:

``w_i = sqrt(a_i) * w_(i-1) + sqrt(1-a_i) * fresh_i``.

``a_i`` grows linearly from zero at ``sigma=1`` to one at ``sigma=0.25`` and
then remains one.  It controls retained *variance*; the nominal adjacent-noise
Pearson coefficient is ``sqrt(a_i)``.

The enforced clip contract is exactly 81 RGB frames / 21 Wan VAE phases.  At
inference there is no target video, mask, tracking, swept tube, pose,
trajectory, optical flow, or first-frame anchor.  This controller supplies no
local-edit guarantee and no semantic correspondence mechanism; SGA is only a
global source-similarity bias and may suppress a desired large action.

PyTorch is imported lazily so configuration and schedule contracts can be
tested in lightweight environments.  The heavyweight renderer integration is
intentionally limited to the already-audited helpers in ``differential_sampler``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import differential_sampler as cdf


EXPECTED_RGB_FRAMES = 81
EXPECTED_LATENT_PHASES = 21
DEFAULT_ANC_LOCK_SIGMA = 0.25


class SourceAlignedControllerError(RuntimeError):
    """Raised when the controller cannot satisfy its scientific contract."""


@dataclass(frozen=True)
class SourceAlignedControllerConfig:
    """Inference controls for :func:`sample_source_aligned_controller`.

    ``sga_steps=3`` and ``sga_candidates=5`` follow the public DynaEdit
    description.  Setting ``sga_steps=0`` is allowed only as an explicit
    ablation.  ``motion_scale=0`` remains an exact identity bypass.
    """

    num_inference_steps: int = 40
    flow_shift: float = 5.0
    seed: int = 20260807
    motion_scale: float = 1.0
    sga_steps: int = 3
    sga_candidates: int = 5
    sga_temperature: float = 0.01
    anc_lock_sigma: float = DEFAULT_ANC_LOCK_SIGMA

    def validate(self) -> "SourceAlignedControllerConfig":
        try:
            cdf.DifferentialFlowConfig(
                num_inference_steps=self.num_inference_steps,
                flow_shift=self.flow_shift,
                seed=self.seed,
                motion_scale=self.motion_scale,
            ).validate()
        except cdf.DifferentialSamplerContractError as error:
            raise SourceAlignedControllerError(str(error)) from error
        if type(self.sga_steps) is not int or not 0 <= self.sga_steps <= self.num_inference_steps:
            raise SourceAlignedControllerError(
                "sga_steps must be an integer in [0, num_inference_steps]"
            )
        if type(self.sga_candidates) is not int or self.sga_candidates < 2:
            raise SourceAlignedControllerError("sga_candidates must be an integer >= 2")
        if (
            isinstance(self.sga_temperature, bool)
            or not isinstance(self.sga_temperature, (int, float))
            or not math.isfinite(float(self.sga_temperature))
            or float(self.sga_temperature) <= 0.0
        ):
            raise SourceAlignedControllerError("sga_temperature must be finite and positive")
        if (
            isinstance(self.anc_lock_sigma, bool)
            or not isinstance(self.anc_lock_sigma, (int, float))
            or not math.isfinite(float(self.anc_lock_sigma))
            or not 0.0 <= float(self.anc_lock_sigma) < 1.0
        ):
            raise SourceAlignedControllerError("anc_lock_sigma must be in [0, 1)")
        return self


@dataclass(frozen=True)
class SourceAlignedControllerTrace:
    """Auditable controller diagnostics in scheduler order."""

    identity_bypassed: bool
    sigmas: tuple[float, ...]
    candidate_counts: tuple[int, ...]
    anc_retained_variance: tuple[float, ...]
    anc_nominal_correlation: tuple[float, ...]
    sga_scores: tuple[tuple[float, ...], ...]
    sga_weights: tuple[tuple[float, ...], ...]
    delta_rms: tuple[float, ...]
    update_rms: tuple[float, ...]
    noise_state_change_rms: tuple[float, ...]
    fresh_noise_draws: int


def controller_contract() -> dict[str, Any]:
    """Return a serialisable, deliberately conservative method statement."""

    return {
        "method": "bernini_source_aligned_sga_anc_controller",
        "status": "dynaedit_inspired_bernini_adaptation_not_official_reproduction",
        "clip_geometry": {
            "rgb_frames": EXPECTED_RGB_FRAMES,
            "wan_vae_phases": EXPECTED_LATENT_PHASES,
        },
        "user_inputs": ["source_video", "edit_instruction"],
        "internal_conditions": [
            "clean_source_vae_latent",
            "action_prompt_embedding",
            "fixed_semantic_noop_prompt_embedding",
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
        "flowedit_field": "V(action,z_tar|source)-V(noop,z_src|source)",
        "source_state": "z_src=(1-sigma)*source+sigma*correlated_noise",
        "target_state": "z_tar=edit+z_src-source",
        "sga": {
            "enabled": True,
            "default_steps": 3,
            "default_candidates": 5,
            "similarity_space": "raw_packed_vae_latent_cosine",
            "scope": "global_soft_source_bias_no_local_preservation_guarantee",
        },
        "anc": {
            "enabled": True,
            "process": "markov_noise_across_flow_timesteps",
            "retained_variance": "linear_0_at_sigma_1_to_1_at_lock_sigma",
            "nominal_correlation": "sqrt(retained_variance)",
            "fixed_noise": False,
        },
        "candidate_chain_collapse": (
            "sga_weighted_noise_divided_by_sqrt_sum_squared_weights; "
            "explicit_approximation_because_weights_are_noise_dependent"
        ),
        "known_limitations": [
            "sga_representation_space_is_underspecified_by_the_paper",
            "bernini_mv2v_conditioning_differs_from_dynaedit_i2v_conditioning",
            "instruction_noop_pair_differs_from_source_target_caption_pair",
            "global_similarity_can_suppress_required_large_motion",
            "no_local_edit_or_identity_guarantee",
        ],
        "sequence_parallel_owner": "official_bernini_transformer",
    }


def anc_retained_variance(
    sigma: float, *, lock_sigma: float = DEFAULT_ANC_LOCK_SIGMA
) -> float:
    """Return ANC's retained-variance coefficient ``a(sigma)``.

    Sampling descends from sigma one to zero.  Noise is independent at the
    full-noise endpoint, becomes progressively correlated, and is locked once
    ``sigma <= lock_sigma``.
    """

    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
        raise SourceAlignedControllerError("sigma must be a finite scalar")
    value = float(sigma)
    if isinstance(lock_sigma, bool) or not isinstance(lock_sigma, (int, float)):
        raise SourceAlignedControllerError("lock_sigma must be in [0, 1)")
    lock = float(lock_sigma)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise SourceAlignedControllerError("sigma must be finite and in [0, 1]")
    if not math.isfinite(lock) or not 0.0 <= lock < 1.0:
        raise SourceAlignedControllerError("lock_sigma must be in [0, 1)")
    if value >= 1.0:
        return 0.0
    if value <= lock:
        return 1.0
    return (1.0 - value) / (1.0 - lock)


def _require_torch() -> Any:
    try:
        import torch
    except Exception as error:  # pragma: no cover - exercised on AUH
        raise SourceAlignedControllerError("PyTorch is required by the controller") from error
    return torch


def flowedit_source_target_states(
    source: Any, edit: Any, correlated_noise: Any, *, sigma: float
) -> tuple[Any, Any]:
    """Construct the exact source-relative FlowEdit query pair."""

    torch = _require_torch()
    for label, tensor in (("source", source), ("edit", edit), ("correlated_noise", correlated_noise)):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise SourceAlignedControllerError(f"{label} must be a floating torch tensor")
    if source.shape != edit.shape or source.shape != correlated_noise.shape:
        raise SourceAlignedControllerError("source, edit and correlated_noise shapes must match")
    if source.device != edit.device or source.device != correlated_noise.device:
        raise SourceAlignedControllerError("FlowEdit state tensors must share a device")
    sigma_value = float(sigma)
    if not math.isfinite(sigma_value) or not 0.0 <= sigma_value <= 1.0:
        raise SourceAlignedControllerError("sigma must be finite and in [0, 1]")
    source_state = (1.0 - sigma_value) * source + sigma_value * correlated_noise
    target_state = edit + source_state - source
    return source_state, target_state


def advance_anc_noise(previous: Any, fresh: Any, *, retained_variance: float) -> Any:
    """Advance one ANC Markov chain with the public paper formula."""

    torch = _require_torch()
    if not isinstance(previous, torch.Tensor) or not isinstance(fresh, torch.Tensor):
        raise SourceAlignedControllerError("ANC states must be torch tensors")
    if not previous.is_floating_point() or not fresh.is_floating_point():
        raise SourceAlignedControllerError("ANC states must be floating tensors")
    if previous.shape != fresh.shape or previous.device != fresh.device:
        raise SourceAlignedControllerError("previous and fresh ANC states must match")
    value = float(retained_variance)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise SourceAlignedControllerError("retained_variance must be in [0, 1]")
    return math.sqrt(value) * previous + math.sqrt(1.0 - value) * fresh


def similarity_guided_aggregate(
    *,
    source: Any,
    edit: Any,
    candidate_deltas: Any,
    sigma: float,
    temperature: float,
) -> tuple[Any, Any, Any, Any]:
    """Soft-aggregate SGA candidates using projected clean-state cosine.

    ``candidate_deltas`` has shape ``[K,B,...]`` while ``source`` and ``edit``
    have ``[B,...]``.  Returned values are aggregate delta, candidate weights
    ``[K,B]``, scores ``[K,B]``, and projected clean states ``[K,B,...]``.
    """

    torch = _require_torch()
    if not all(isinstance(value, torch.Tensor) for value in (source, edit, candidate_deltas)):
        raise SourceAlignedControllerError("SGA inputs must be torch tensors")
    if not source.is_floating_point() or not edit.is_floating_point() or not candidate_deltas.is_floating_point():
        raise SourceAlignedControllerError("SGA inputs must be floating tensors")
    if source.shape != edit.shape or candidate_deltas.ndim != source.ndim + 1:
        raise SourceAlignedControllerError("SGA candidate dimensions do not match source/edit")
    if tuple(candidate_deltas.shape[1:]) != tuple(source.shape):
        raise SourceAlignedControllerError("each SGA candidate must match source/edit shape")
    if int(candidate_deltas.shape[0]) < 2:
        raise SourceAlignedControllerError("SGA requires at least two candidates")
    if source.device != edit.device or source.device != candidate_deltas.device:
        raise SourceAlignedControllerError("SGA inputs must share a device")
    if not all(bool(torch.isfinite(value).all().item()) for value in (source, edit, candidate_deltas)):
        raise SourceAlignedControllerError("SGA inputs must be finite")
    sigma_value = float(sigma)
    tau = float(temperature)
    if not math.isfinite(sigma_value) or not 0.0 < sigma_value <= 1.0:
        raise SourceAlignedControllerError("SGA sigma must be finite and in (0, 1]")
    if not math.isfinite(tau) or tau <= 0.0:
        raise SourceAlignedControllerError("SGA temperature must be finite and positive")

    projected = edit.unsqueeze(0) - sigma_value * candidate_deltas
    candidates = projected.reshape(int(projected.shape[0]), int(projected.shape[1]), -1).float()
    source_flat = source.reshape(int(source.shape[0]), -1).float()
    source_norm = source_flat.square().sum(dim=-1).sqrt()
    if bool((source_norm <= 1.0e-12).any().item()):
        raise SourceAlignedControllerError("SGA source cosine is undefined for zero-norm source")
    candidate_norm = candidates.square().sum(dim=-1).sqrt()
    if bool((candidate_norm <= 1.0e-12).any().item()):
        raise SourceAlignedControllerError("SGA projected cosine is undefined for zero-norm candidate")
    scores = (candidates * source_flat.unsqueeze(0)).sum(dim=-1) / (
        candidate_norm * source_norm.unsqueeze(0)
    )
    weights = torch.softmax(scores / tau, dim=0)
    broadcast = weights.reshape(
        int(weights.shape[0]), int(weights.shape[1]), *([1] * (source.ndim - 1))
    )
    aggregate_projection = (broadcast.to(projected.dtype) * projected).sum(dim=0)
    aggregate_delta = (edit - aggregate_projection) / sigma_value
    return aggregate_delta, weights, scores, projected


def collapse_sga_noise_chains(candidate_noise: Any, weights: Any) -> Any:
    """Collapse ``[K,B,...]`` noise chains to one variance-normalised state."""

    torch = _require_torch()
    if not isinstance(candidate_noise, torch.Tensor) or not isinstance(weights, torch.Tensor):
        raise SourceAlignedControllerError("candidate_noise and weights must be tensors")
    if candidate_noise.ndim < 3 or weights.ndim != 2:
        raise SourceAlignedControllerError("noise must be [K,B,...] and weights [K,B]")
    if tuple(candidate_noise.shape[:2]) != tuple(weights.shape):
        raise SourceAlignedControllerError("noise candidate and weight axes differ")
    if int(candidate_noise.shape[0]) < 2:
        raise SourceAlignedControllerError("noise-chain collapse requires >=2 candidates")
    if not candidate_noise.is_floating_point() or not weights.is_floating_point():
        raise SourceAlignedControllerError("noise-chain collapse requires floating tensors")
    if candidate_noise.device != weights.device:
        raise SourceAlignedControllerError("noise chains and weights must share a device")
    if not bool(torch.isfinite(weights).all().item()) or bool((weights < 0).any().item()):
        raise SourceAlignedControllerError("SGA weights must be finite and non-negative")
    sums = weights.sum(dim=0)
    if not bool(torch.allclose(sums, torch.ones_like(sums), atol=1.0e-5, rtol=1.0e-5)):
        raise SourceAlignedControllerError("SGA weights must sum to one per batch item")
    broadcast = weights.reshape(
        int(weights.shape[0]), int(weights.shape[1]), *([1] * (candidate_noise.ndim - 2))
    )
    weighted = (broadcast.to(candidate_noise.dtype) * candidate_noise).sum(dim=0)
    denominator = weights.square().sum(dim=0).sqrt().reshape(
        int(weights.shape[1]), *([1] * (candidate_noise.ndim - 2))
    )
    if bool((denominator <= 0).any().item()):
        raise SourceAlignedControllerError("noise-chain collapse has zero variance denominator")
    return weighted / denominator.to(weighted.dtype)


def _validate_exact_geometry(layout: cdf.LatentLayout, *, source_rgb_frames: int) -> None:
    if type(source_rgb_frames) is not int or source_rgb_frames != EXPECTED_RGB_FRAMES:
        raise SourceAlignedControllerError(
            f"source_rgb_frames must equal {EXPECTED_RGB_FRAMES}, got {source_rgb_frames!r}"
        )
    if layout.frames != EXPECTED_LATENT_PHASES:
        raise SourceAlignedControllerError(
            f"81-frame Wan input must have exactly {EXPECTED_LATENT_PHASES} latent phases, "
            f"got {layout.frames}"
        )


def _validate_full_noise_intervals(intervals: Any) -> None:
    if not intervals:
        raise SourceAlignedControllerError("flow schedule is empty")
    if not math.isclose(float(intervals[0][0]), 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise SourceAlignedControllerError("controller requires full-noise sigma=1 start")
    if not math.isclose(float(intervals[-1][1]), 0.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise SourceAlignedControllerError("controller requires sigma=0 endpoint")


def _draw_fresh_packed_noise(generator: Any, source_latent: Any, layout: cdf.LatentLayout) -> Any:
    torch = _require_torch()
    fresh = torch.randn(
        tuple(int(value) for value in source_latent.shape),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(device=source_latent.device)
    return cdf._pack_spatial_latent(fresh, layout)


def _identity_trace() -> SourceAlignedControllerTrace:
    return SourceAlignedControllerTrace(
        True, (), (), (), (), (), (), (), (), (), 0
    )


def sample_source_aligned_controller(
    renderer_or_diffusion: Any,
    *,
    source_latent: Any,
    source_rgb_frames: int,
    action_prompt_embeds: Any,
    noop_prompt_embeds: Any,
    config: Optional[SourceAlignedControllerConfig] = None,
    return_trace: bool = False,
) -> Any:
    """Run the full-source Bernini SGA+ANC differential controller.

    The caller must derive both prompt embeddings from the same official
    Bernini text path.  ``noop_prompt_embeds`` is an internal fixed semantic
    no-op, not a user-supplied source caption.  All Ulysses ranks must enter
    with identical tensors, seed, configuration and call order.
    """

    runtime = (config or SourceAlignedControllerConfig()).validate()
    if type(source_rgb_frames) is not int or source_rgb_frames != EXPECTED_RGB_FRAMES:
        raise SourceAlignedControllerError(
            f"source_rgb_frames must equal {EXPECTED_RGB_FRAMES}, got {source_rgb_frames!r}"
        )
    latent_shape = getattr(source_latent, "shape", None)
    if latent_shape is not None:
        shape_values = tuple(int(value) for value in latent_shape)
        if len(shape_values) != 5 or shape_values[2] != EXPECTED_LATENT_PHASES:
            raise SourceAlignedControllerError(
                f"source latent must expose exactly {EXPECTED_LATENT_PHASES} temporal phases"
            )
    identity = float(runtime.motion_scale) == 0.0 or cdf.prompts_are_exactly_identical(
        action_prompt_embeds, noop_prompt_embeds
    )
    if identity:
        trace = _identity_trace()
        return (source_latent, trace) if return_trace else source_latent

    torch = _require_torch()
    diffusion = cdf.resolve_diffusion_core(renderer_or_diffusion)
    layout, transformer = cdf._validate_runtime_inputs(
        diffusion, source_latent, action_prompt_embeds, noop_prompt_embeds
    )
    _validate_exact_geometry(layout, source_rgb_frames=source_rgb_frames)
    cdf_runtime = cdf.DifferentialFlowConfig(
        num_inference_steps=runtime.num_inference_steps,
        flow_shift=runtime.flow_shift,
        seed=runtime.seed,
        motion_scale=runtime.motion_scale,
    )
    timesteps, intervals = cdf._set_scheduler_timesteps(
        diffusion, cdf_runtime, source_latent.device
    )
    _validate_full_noise_intervals(intervals)

    source_clean = source_latent.detach().to(dtype=torch.float32)
    source_packed = cdf._pack_spatial_latent(source_clean, layout)
    edit_packed = source_packed.clone()
    initial_chain_count = runtime.sga_candidates if runtime.sga_steps > 0 else 1
    previous_noises = [torch.zeros_like(source_packed) for _ in range(initial_chain_count)]
    generator = torch.Generator(device="cpu").manual_seed(runtime.seed)

    candidate_counts: list[int] = []
    retention_trace: list[float] = []
    correlation_trace: list[float] = []
    score_trace: list[tuple[float, ...]] = []
    weight_trace: list[tuple[float, ...]] = []
    delta_trace: list[float] = []
    update_trace: list[float] = []
    noise_change_trace: list[float] = []
    fresh_noise_draws = 0

    with torch.no_grad():
        source_condition = cdf._patch_source_condition(transformer, source_clean)
        for index, (sigma_value, next_sigma_value) in enumerate(intervals):
            sga_active = index < runtime.sga_steps
            candidate_count = runtime.sga_candidates if sga_active else 1
            if len(previous_noises) != candidate_count:
                raise SourceAlignedControllerError(
                    "internal ANC chain count changed without the audited SGA collapse"
                )
            retained = anc_retained_variance(
                float(sigma_value), lock_sigma=float(runtime.anc_lock_sigma)
            )
            candidate_noises = []
            candidate_deltas = []
            per_candidate_change = []
            for candidate_index in range(candidate_count):
                fresh = _draw_fresh_packed_noise(generator, source_clean, layout)
                fresh_noise_draws += 1
                correlated = advance_anc_noise(
                    previous_noises[candidate_index],
                    fresh,
                    retained_variance=retained,
                )
                per_candidate_change.append(
                    (correlated - previous_noises[candidate_index]).float().square().mean().sqrt()
                )
                candidate_noises.append(correlated)
                source_state_packed, target_state_packed = flowedit_source_target_states(
                    source_packed,
                    edit_packed,
                    correlated,
                    sigma=float(sigma_value),
                )
                source_state = cdf._unpack_spatial_latent(source_state_packed, layout)
                target_state = cdf._unpack_spatial_latent(target_state_packed, layout)
                timestep = timesteps[index]
                action_velocity = cdf._predict_source_conditioned_velocity(
                    diffusion=diffusion,
                    transformer=transformer,
                    source_condition=source_condition,
                    query_latent=target_state,
                    prompt_embeds=action_prompt_embeds,
                    timestep=timestep,
                )
                noop_velocity = cdf._predict_source_conditioned_velocity(
                    diffusion=diffusion,
                    transformer=transformer,
                    source_condition=source_condition,
                    query_latent=source_state,
                    prompt_embeds=noop_prompt_embeds,
                    timestep=timestep,
                )
                delta = action_velocity.float() - noop_velocity.float()
                expected_shape = (layout.batch, layout.tokens, layout.packed_channels)
                if tuple(int(value) for value in delta.shape) != expected_shape:
                    raise SourceAlignedControllerError(
                        "Bernini query prediction shape differs from packed source"
                    )
                candidate_deltas.append(delta)

            noise_bank = torch.stack(candidate_noises, dim=0)
            delta_bank = torch.stack(candidate_deltas, dim=0)
            if sga_active:
                aggregate_delta, weights, scores, _ = similarity_guided_aggregate(
                    source=source_packed,
                    edit=edit_packed,
                    candidate_deltas=delta_bank,
                    sigma=float(sigma_value),
                    temperature=float(runtime.sga_temperature),
                )
                score_trace.append(tuple(float(x) for x in scores[:, 0].detach().cpu().tolist()))
                weight_trace.append(tuple(float(x) for x in weights[:, 0].detach().cpu().tolist()))
                if index == runtime.sga_steps - 1:
                    previous_noises = [collapse_sga_noise_chains(noise_bank, weights)]
                else:
                    previous_noises = list(noise_bank.unbind(dim=0))
            else:
                aggregate_delta = delta_bank[0]
                score_trace.append(())
                weight_trace.append((1.0,))
                previous_noises = [noise_bank[0]]

            delta_sigma = float(next_sigma_value - sigma_value)
            update = float(runtime.motion_scale) * delta_sigma * aggregate_delta
            edit_packed = edit_packed + update
            candidate_counts.append(candidate_count)
            retention_trace.append(float(retained))
            correlation_trace.append(math.sqrt(float(retained)))
            delta_trace.append(float(aggregate_delta.square().mean().sqrt().cpu().item()))
            update_trace.append(float(update.square().mean().sqrt().cpu().item()))
            noise_change_trace.append(
                float(torch.stack(per_candidate_change).mean().cpu().item())
            )

    result = cdf._unpack_spatial_latent(edit_packed, layout)
    if not return_trace:
        return result
    sigmas = tuple(intervals[0][:1]) + tuple(pair[1] for pair in intervals)
    trace = SourceAlignedControllerTrace(
        False,
        sigmas,
        tuple(candidate_counts),
        tuple(retention_trace),
        tuple(correlation_trace),
        tuple(score_trace),
        tuple(weight_trace),
        tuple(delta_trace),
        tuple(update_trace),
        tuple(noise_change_trace),
        fresh_noise_draws,
    )
    return result, trace


__all__ = [
    "DEFAULT_ANC_LOCK_SIGMA",
    "EXPECTED_LATENT_PHASES",
    "EXPECTED_RGB_FRAMES",
    "SourceAlignedControllerConfig",
    "SourceAlignedControllerError",
    "SourceAlignedControllerTrace",
    "advance_anc_noise",
    "anc_retained_variance",
    "collapse_sga_noise_chains",
    "controller_contract",
    "flowedit_source_target_states",
    "sample_source_aligned_controller",
    "similarity_guided_aggregate",
]
