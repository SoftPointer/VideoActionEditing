"""Auditable training primitives for OmniVideo2 PACT fine-tuning.

The functions in this module do not import OmniVideo.  This keeps the data and
objective contract testable on CPU before a multi-gigabyte checkpoint is
loaded.  Latent/video tensors use ``[B, C, T, H, W]`` throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .conditioning import (
    SourceLatentBudgetMetadata,
    budget_source_latent,
    erase_source_motion,
)
from .flow import reconstruct_x0, shared_noise_local_latent_splice, velocity_target
from .losses import (
    area_normalized_masked_loss,
    boundary_consistency_loss,
    outside_temporal_difference_loss,
)
from .lora import lora_scope_target_regex
from .masks import (
    boundary_ring,
    dilate_and_feather,
    source_target_tube_union,
    validate_video_mask,
)
from .router import router_loss_components


TRAINING_CONFIG_FORMAT = "pact-omnivideo2-training-v2"
DIFFSYNTH_WAN_TRAINING_BINS = 1000
DIFFSYNTH_WAN_TIMESTEP_SCALE = 1000.0


@dataclass(frozen=True)
class DiffSynthWanTrainingSample:
    """One batch-shared draw from DiffSynth's discrete Wan SFT schedule.

    ``sigma`` and ``timestep`` have shape ``[batch_size]`` but every element is
    identical, matching DiffSynth's single ``torch.randint(..., (1,))`` draw
    per batch. ``flow_weight`` is a scalar FP32 tensor containing the exact
    BSMNTW table entry for ``timestep_id``.
    """

    timestep_id: int
    sigma: Tensor
    timestep: Tensor
    flow_weight: Tensor


class DiffSynthWanTrainingScheduler:
    """FP32 reproduction of DiffSynth's intended Wan SFT time schedule.

    DiffSynth constructs 1000 training bins from
    ``linspace(1, 0, 1001)[:-1]``, applies Wan's rational shift, and feeds
    ``1000 * sigma`` to the model.  Its BSMNTW loss weights are derived from
    that same shifted table.  The table construction is bit-exact to
    DiffSynth-Studio revision ``ab12bf4``.

    The reference Wan training entry subsequently casts the sampled timestep
    to BF16 and performs a nearest-table lookup.  That implementation detail
    collapses many nominal bins.  PACT deliberately keeps the selected
    sigma/timestep/path in FP32 and records this deviation in every run receipt;
    it reproduces DiffSynth's table and objective, not that BF16 quantization.
    """

    num_training_bins = DIFFSYNTH_WAN_TRAINING_BINS
    num_train_timesteps = int(DIFFSYNTH_WAN_TIMESTEP_SCALE)

    def __init__(self, shift: float = 5.0) -> None:
        if (
            not isinstance(shift, (int, float))
            or isinstance(shift, bool)
            or not math.isfinite(float(shift))
            or float(shift) <= 0.0
        ):
            raise ValueError("shift must be a finite positive number")
        self.shift = float(shift)

        # Keep operation order and FP32 defaults identical to DiffSynth-Studio
        # diffsynth/diffusion/flow_match.py at revision ab12bf4.
        unshifted = torch.linspace(
            1.0,
            0.0,
            self.num_training_bins + 1,
            dtype=torch.float32,
            device="cpu",
        )[:-1]
        self.sigmas = shifted_rectified_flow_sigma(unshifted, self.shift)
        self.timesteps = self.sigmas * DIFFSYNTH_WAN_TIMESTEP_SCALE

        profile = torch.exp(
            -2.0
            * (
                (self.timesteps - DIFFSYNTH_WAN_TIMESTEP_SCALE / 2.0)
                / DIFFSYNTH_WAN_TIMESTEP_SCALE
            )
            ** 2
        )
        shifted_profile = profile - profile.min()
        normalizer = shifted_profile.sum()
        if not bool(torch.isfinite(normalizer)) or not bool(normalizer > 0):
            raise RuntimeError("invalid DiffSynth BSMNTW normalization")
        self.flow_weights = shifted_profile * (
            self.num_training_bins / normalizer
        )

    @staticmethod
    def _validate_batch_size(batch_size: int) -> None:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")

    @staticmethod
    def _validate_dtype(dtype: torch.dtype) -> None:
        if not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("dtype must be floating point")

    def at(
        self,
        timestep_id: int,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> DiffSynthWanTrainingSample:
        """Return one validated table entry expanded across a whole batch."""

        if (
            not isinstance(timestep_id, int)
            or isinstance(timestep_id, bool)
            or not 0 <= timestep_id < self.num_training_bins
        ):
            raise ValueError(
                f"timestep_id must lie in [0, {self.num_training_bins})"
            )
        self._validate_batch_size(batch_size)
        self._validate_dtype(dtype)
        target_device = torch.device(device)
        sigma = self.sigmas[timestep_id].to(
            device=target_device, dtype=dtype
        ).repeat(batch_size)
        timestep = self.timesteps[timestep_id].to(
            device=target_device, dtype=dtype
        ).repeat(batch_size)
        flow_weight = self.flow_weights[timestep_id].to(
            device=target_device, dtype=torch.float32
        )
        if flow_weight.ndim != 0:
            raise RuntimeError("internal error: DiffSynth flow weight is not scalar")
        return DiffSynthWanTrainingSample(
            timestep_id=timestep_id,
            sigma=sigma,
            timestep=timestep,
            flow_weight=flow_weight,
        )

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> DiffSynthWanTrainingSample:
        """Uniformly sample one discrete bin and share it across the batch.

        DiffSynth samples the index on CPU, so a supplied generator must also
        be a CPU generator. This explicit restriction prevents a caller from
        silently changing seeded parity by using a device-specific RNG.
        """

        self._validate_batch_size(batch_size)
        self._validate_dtype(dtype)
        if generator is not None:
            if not isinstance(generator, torch.Generator):
                raise TypeError("generator must be a torch.Generator or None")
            if torch.device(generator.device).type != "cpu":
                raise ValueError("DiffSynth-compatible sampling requires a CPU generator")
        timestep_id = int(
            torch.randint(
                0,
                self.num_training_bins,
                (1,),
                generator=generator,
                device="cpu",
            ).item()
        )
        return self.at(
            timestep_id,
            batch_size,
            device=device,
            dtype=dtype,
        )


@dataclass(frozen=True)
class PactFlowBatch:
    """Prepared shared-noise local-flow sample and its auditable targets."""

    source_x0: Tensor
    target_x0: Tensor
    local_x0: Tensor
    edit_mask: Tensor
    source_erase_mask: Tensor
    source_condition: Tensor
    x_t: Tensor
    noise: Tensor
    local_velocity: Tensor
    source_velocity: Tensor
    sigma: Tensor


def shifted_rectified_flow_sigma(uniform: Tensor, shift: float) -> Tensor:
    """Apply the Wan/flow-scheduler rational timestep shift to ``U[0, 1]``.

    ``shift=1`` is the identity and values greater than one bias training toward
    noisier states while preserving the exact endpoints.
    """

    if not isinstance(uniform, Tensor) or not uniform.is_floating_point():
        raise TypeError("uniform must be a floating torch.Tensor")
    if not bool(torch.isfinite(uniform.detach()).all()):
        raise ValueError("uniform must be finite")
    if not bool(((uniform.detach() >= 0) & (uniform.detach() <= 1)).all()):
        raise ValueError("uniform values must lie in [0, 1]")
    shift = float(shift)
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("shift must be finite and positive")
    return shift * uniform / (1.0 + (shift - 1.0) * uniform)


def build_edit_support(
    source_component_mask: Tensor,
    target_component_mask: Tensor,
    *,
    dilation_radius: int | Sequence[int] = (0, 1, 1),
    feather_radius: int | Sequence[int] = (0, 1, 1),
) -> tuple[Tensor, Tensor]:
    """Build target-aware edit support and source-only erasure support.

    The edit endpoint uses the union of source and counterfactual target actor
    tubes, so both the actor's old and new locations may change.  Source-motion
    erasure uses only the source tube and therefore does not erase unrelated
    content at a target-only future location.
    """

    union = source_target_tube_union(source_component_mask, target_component_mask)
    edit_mask = dilate_and_feather(
        union,
        dilation_radius=dilation_radius,
        feather_radius=feather_radius,
    )
    source_erase_mask = dilate_and_feather(
        source_component_mask,
        dilation_radius=dilation_radius,
        feather_radius=feather_radius,
    )
    reduce_dims = tuple(range(1, edit_mask.ndim))
    if not bool((edit_mask.sum(dim=reduce_dims) > 0).all()):
        raise ValueError("every sample must contain a non-empty selected actor tube")
    if not bool(((1.0 - edit_mask).sum(dim=reduce_dims) > 0).all()):
        raise ValueError("every sample must retain non-empty preservation support")
    return edit_mask, source_erase_mask


def prepare_pact_flow_batch(
    source_x0: Tensor,
    global_target_x0: Tensor,
    source_component_mask: Tensor,
    target_component_mask: Tensor,
    sigma: Tensor,
    *,
    dilation_radius: int | Sequence[int] = (0, 1, 1),
    feather_radius: int | Sequence[int] = (0, 1, 1),
    source_erasure_mode: str = "zero",
    noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> PactFlowBatch:
    """Prepare actor-selective local endpoint and shared-noise flow targets."""

    if not isinstance(source_x0, Tensor) or source_x0.ndim != 5:
        raise ValueError("source_x0 must have shape [B, C, T, H, W]")
    if not source_x0.is_floating_point():
        raise TypeError("source_x0 must have a floating dtype")
    if not isinstance(global_target_x0, Tensor) or global_target_x0.shape != source_x0.shape:
        raise ValueError("global_target_x0 must match source_x0 shape")
    if global_target_x0.dtype != source_x0.dtype or global_target_x0.device != source_x0.device:
        raise ValueError("source and target latents must share dtype and device")
    validate_video_mask(
        source_component_mask,
        name="source_component_mask",
        batch_size=source_x0.shape[0],
        frames=source_x0.shape[2],
        height=source_x0.shape[3],
        width=source_x0.shape[4],
    )
    validate_video_mask(
        target_component_mask,
        name="target_component_mask",
        batch_size=source_x0.shape[0],
        frames=source_x0.shape[2],
        height=source_x0.shape[3],
        width=source_x0.shape[4],
    )
    if source_component_mask.device != source_x0.device or target_component_mask.device != source_x0.device:
        raise ValueError("component masks and latents must share a device")

    edit_mask, source_erase_mask = build_edit_support(
        source_component_mask,
        target_component_mask,
        dilation_radius=dilation_radius,
        feather_radius=feather_radius,
    )
    source_condition = erase_source_motion(
        source_x0,
        source_erase_mask,
        mode=source_erasure_mode,
        keep_first_frame=True,
    )
    splice = shared_noise_local_latent_splice(
        source_x0,
        global_target_x0,
        edit_mask,
        sigma,
        noise=noise,
        generator=generator,
    )
    return PactFlowBatch(
        source_x0=source_x0,
        target_x0=global_target_x0,
        local_x0=splice.local_x0,
        edit_mask=edit_mask,
        source_erase_mask=source_erase_mask,
        source_condition=source_condition,
        x_t=splice.x_t,
        noise=splice.noise,
        local_velocity=splice.target_velocity,
        source_velocity=velocity_target(source_x0, splice.noise),
        sigma=sigma,
    )


def budget_source_condition_preserving_first_frame(
    source_condition: Tensor,
    *,
    max_context_len: int,
    nonvisual_tokens: int,
    visual_patch_size: Sequence[int] = (1, 4, 4),
) -> tuple[Tensor, SourceLatentBudgetMetadata]:
    """Budget visual tokens while retaining the exact first latent frame.

    OmniVideo2-1.3B uses a temporal visual-adapter patch size of one.  We fail
    for any other temporal patch size because a multi-frame convolution cannot
    offer an exact first-frame identity guarantee.
    """

    patch_size = tuple(int(item) for item in visual_patch_size)
    if len(patch_size) != 3 or patch_size[0] != 1:
        raise ValueError("first-frame-preserving budgeting requires temporal patch size 1")
    latent_grid = source_condition.shape[2:]
    if any(size % stride for size, stride in zip(latent_grid, patch_size)):
        raise ValueError(
            f"source latent grid {latent_grid} must be divisible by visual patch size "
            f"{patch_size}; no-padding Conv3d token counts would otherwise differ"
        )
    pooled, metadata = budget_source_latent(
        source_condition,
        max_context_len=max_context_len,
        nonvisual_tokens=nonvisual_tokens,
        visual_patch_size=patch_size,
    )
    if not metadata.compressed:
        return pooled, metadata

    output_frames = metadata.output_shape[2]
    if output_frames == 1:
        output = source_condition[:, :, :1]
    else:
        tail = F.adaptive_avg_pool3d(
            source_condition[:, :, 1:],
            output_size=(output_frames - 1, source_condition.shape[3], source_condition.shape[4]),
        )
        output = torch.cat((source_condition[:, :, :1], tail), dim=2)
    if not torch.equal(output[:, :, 0], source_condition[:, :, 0]):
        raise RuntimeError("internal error: source first frame changed during budgeting")
    return output, metadata


def nonvisual_token_counts(
    text_context: Sequence[Tensor],
    vlm_context: Sequence[Tensor],
    *,
    special_token_count: int,
) -> list[int]:
    """Count the exact non-visual prefix used by tight concatenation."""

    if len(text_context) != len(vlm_context) or not text_context:
        raise ValueError("text_context and vlm_context must be non-empty equal-length sequences")
    if not isinstance(special_token_count, int) or isinstance(special_token_count, bool):
        raise TypeError("special_token_count must be an integer")
    if special_token_count < 0:
        raise ValueError("special_token_count must be non-negative")
    counts: list[int] = []
    for index, (text, vlm) in enumerate(zip(text_context, vlm_context)):
        if not isinstance(text, Tensor) or text.ndim != 2 or text.shape[1] != 4096:
            raise ValueError(f"text_context[{index}] must have shape [L, 4096]")
        if not isinstance(vlm, Tensor) or vlm.ndim != 2 or vlm.shape[1] != 2048:
            raise ValueError(f"vlm_context[{index}] must have shape [L, 2048]")
        counts.append(int(text.shape[0] + vlm.shape[0] + special_token_count))
    return counts


def wan_sequence_length(
    latent: Tensor | Sequence[int], patch_size: Sequence[int] = (1, 2, 2)
) -> int:
    """Return the exact Wan patch sequence length, rejecting cropped shapes."""

    shape = tuple(latent.shape) if isinstance(latent, Tensor) else tuple(latent)
    if len(shape) != 5 or any(not isinstance(item, int) or item <= 0 for item in shape):
        raise ValueError("latent shape must be [B, C, T, H, W] with positive dimensions")
    patch = tuple(patch_size)
    if len(patch) != 3 or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in patch):
        raise ValueError("patch_size must contain three positive integers")
    grid = shape[2:]
    if any(size % stride for size, stride in zip(grid, patch)):
        raise ValueError(
            f"latent grid {grid} must be divisible by Wan patch size {patch}; "
            "implicit Conv3d cropping is forbidden"
        )
    return math.prod(size // stride for size, stride in zip(grid, patch))


def _loss_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    configured = {
        "velocity_edit": 1.0,
        "velocity_preserve": 1.0,
        "x0_boundary": 0.25,
        "x0_temporal_outside": 0.25,
        "router": 0.1,
    }
    if weights is not None:
        unknown = set(weights) - set(configured)
        if unknown:
            raise ValueError(f"unknown PACT loss weights: {sorted(unknown)}")
        for name, value in weights.items():
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError(f"loss weight {name!r} must be finite and non-negative")
            configured[name] = numeric
    return configured


def pact_training_losses(
    prediction_velocity: Tensor,
    prepared: PactFlowBatch,
    *,
    router_logits: Tensor | None = None,
    router_target_mask: Tensor | None = None,
    weights: Mapping[str, float] | None = None,
    loss_type: str = "mse",
    flow_weight: float | Tensor | None = None,
) -> dict[str, Tensor]:
    """Combine velocity supervision with reconstructed-x0 preservation losses.

    ``flow_weight`` is an optional scalar timestep weight, such as the BSMNTW
    value emitted by :class:`DiffSynthWanTrainingScheduler`.  It scales the
    four diffusion/reconstruction terms in ``total`` and deliberately never
    scales router supervision. Returned component values remain unweighted for
    transparent logging.
    """

    if not isinstance(prediction_velocity, Tensor) or prediction_velocity.ndim != 5:
        raise ValueError("prediction_velocity must have shape [B, C, T, H, W]")
    if not prediction_velocity.is_floating_point():
        raise TypeError("prediction_velocity must have a floating dtype")
    if prediction_velocity.shape != prepared.x_t.shape or prediction_velocity.device != prepared.x_t.device:
        raise ValueError("prediction_velocity must match the prepared latent shape and device")
    dtype = prediction_velocity.dtype
    x_t = prepared.x_t.to(dtype=dtype)
    source = prepared.source_x0.to(dtype=dtype)
    target = prepared.target_x0.to(dtype=dtype)
    local_velocity = prepared.local_velocity.to(dtype=dtype)
    edit_mask = prepared.edit_mask.to(dtype=dtype)

    ring = boundary_ring(edit_mask)
    preservation_exclusion = torch.maximum(edit_mask, ring)
    preserve_mask = (1.0 - preservation_exclusion).clamp(0.0, 1.0)
    velocity_edit = area_normalized_masked_loss(
        prediction_velocity,
        local_velocity,
        edit_mask,
        loss_type=loss_type,
    )
    # One local endpoint defines the RF target everywhere.  Using that same
    # target avoids contradictory supervision in soft feather voxels.
    velocity_preserve = area_normalized_masked_loss(
        prediction_velocity,
        local_velocity,
        preserve_mask,
        loss_type=loss_type,
    )
    reconstructed = reconstruct_x0(x_t, prediction_velocity, prepared.sigma)
    components = {
        "velocity_edit": velocity_edit,
        "velocity_preserve": velocity_preserve,
        "x0_boundary": boundary_consistency_loss(
            reconstructed,
            target,
            source,
            edit_mask,
            loss_type=loss_type,
        ),
        "x0_temporal_outside": outside_temporal_difference_loss(
            reconstructed,
            source,
            preservation_exclusion,
            loss_type=loss_type,
        ),
    }

    if (router_logits is None) != (router_target_mask is None):
        raise ValueError("router_logits and router_target_mask must be provided together")
    if router_logits is None:
        components["router"] = prediction_velocity.sum() * 0.0
    else:
        router = router_loss_components(router_logits, router_target_mask)
        components["router"] = router["total"]
        components["router_bce"] = router["bce"]
        components["router_dice"] = router["dice"]

    if flow_weight is None:
        scalar_flow_weight = prediction_velocity.new_tensor(1.0)
    elif isinstance(flow_weight, Tensor):
        if flow_weight.ndim != 0:
            raise ValueError("flow_weight tensor must be scalar")
        if flow_weight.requires_grad:
            raise ValueError("flow_weight must not require gradients")
        scalar_flow_weight = flow_weight.to(
            device=prediction_velocity.device,
            dtype=prediction_velocity.dtype,
        )
    elif isinstance(flow_weight, (int, float)) and not isinstance(flow_weight, bool):
        scalar_flow_weight = prediction_velocity.new_tensor(float(flow_weight))
    else:
        raise TypeError("flow_weight must be a scalar number, scalar tensor, or None")
    if not bool(torch.isfinite(scalar_flow_weight.detach())):
        raise ValueError("flow_weight must be finite")
    if not bool(scalar_flow_weight.detach() >= 0):
        raise ValueError("flow_weight must be non-negative")

    configured = _loss_weights(weights)
    flow_component_names = (
        "velocity_edit",
        "velocity_preserve",
        "x0_boundary",
        "x0_temporal_outside",
    )
    flow_total = sum(
        configured[name] * components[name] for name in flow_component_names
    )
    total = (
        scalar_flow_weight * flow_total
        + configured["router"] * components["router"]
    )
    return {"total": total, **components}


def validate_training_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the closed JSON configuration consumed by the training entry."""

    if not isinstance(value, Mapping):
        raise ValueError("training config must be a JSON object")
    config = dict(value)
    expected_top = {
        "format",
        "seed",
        "model",
        "mask",
        "flow",
        "lora",
        "router",
        "loss_weights",
        "optimizer",
        "training",
    }
    if set(config) != expected_top:
        raise ValueError(
            f"training config fields differ: missing={sorted(expected_top - set(config))}, "
            f"unknown={sorted(set(config) - expected_top)}"
        )
    if config["format"] != TRAINING_CONFIG_FORMAT:
        raise ValueError(f"training config format must be {TRAINING_CONFIG_FORMAT}")
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool) or config["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")

    schemas: dict[str, set[str]] = {
        "model": {
            "max_context_len",
            "visual_patch_size",
            "wan_patch_size",
            "source_erasure_mode",
            "require_special_tokens",
            "train_visual_adapter",
            "train_vlm_projection",
            "gradient_checkpointing",
        },
        "mask": {"dilation_radius", "feather_radius"},
        "flow": {
            "num_train_timesteps",
            "shift",
            "timestep_sampling",
            "loss_weighting",
        },
        "lora": {"scope", "rank", "alpha", "dropout"},
        "router": {"hidden_channels", "depth"},
        "loss_weights": {
            "velocity_edit",
            "velocity_preserve",
            "x0_boundary",
            "x0_temporal_outside",
            "router",
        },
        "optimizer": {
            "learning_rate",
            "pretrained_adapter_learning_rate",
            "weight_decay",
            "beta1",
            "beta2",
            "eps",
            "max_grad_norm",
        },
        "training": {
            "epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "num_workers",
            "max_steps",
            "checkpoint_every",
            "log_every",
        },
    }
    for section, fields in schemas.items():
        section_value = config[section]
        if not isinstance(section_value, Mapping) or set(section_value) != fields:
            actual = set(section_value) if isinstance(section_value, Mapping) else set()
            raise ValueError(
                f"config section {section!r} differs: missing={sorted(fields - actual)}, "
                f"unknown={sorted(actual - fields)}"
            )

    model = config["model"]
    positive_ints = {
        "model.max_context_len": model["max_context_len"],
        "flow.num_train_timesteps": config["flow"]["num_train_timesteps"],
        "lora.rank": config["lora"]["rank"],
        "router.hidden_channels": config["router"]["hidden_channels"],
        "training.epochs": config["training"]["epochs"],
        "training.batch_size": config["training"]["batch_size"],
        "training.gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
        "training.checkpoint_every": config["training"]["checkpoint_every"],
        "training.log_every": config["training"]["log_every"],
    }
    for name, item in positive_ints.items():
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("visual_patch_size", "wan_patch_size"):
        patch = model[name]
        if not isinstance(patch, list) or len(patch) != 3 or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in patch
        ):
            raise ValueError(f"model.{name} must be three positive integers")
    for name in ("dilation_radius", "feather_radius"):
        radius = config["mask"][name]
        if not isinstance(radius, list) or len(radius) != 3 or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in radius
        ):
            raise ValueError(f"mask.{name} must be three non-negative integers")
    if model["source_erasure_mode"] not in {"zero", "temporal_mean"}:
        raise ValueError("model.source_erasure_mode must be zero or temporal_mean")
    for name in (
        "require_special_tokens",
        "train_visual_adapter",
        "train_vlm_projection",
        "gradient_checkpointing",
    ):
        if not isinstance(model[name], bool):
            raise ValueError(f"model.{name} must be boolean")
    if config["flow"]["num_train_timesteps"] != DIFFSYNTH_WAN_TRAINING_BINS:
        raise ValueError(
            f"flow.num_train_timesteps must equal {DIFFSYNTH_WAN_TRAINING_BINS} "
            "for DiffSynth/Wan parity"
        )
    if config["flow"]["timestep_sampling"] != "uniform_discrete_batch_shared":
        raise ValueError(
            "flow.timestep_sampling must be uniform_discrete_batch_shared"
        )
    if config["flow"]["loss_weighting"] != "diffsynth_bsmntw":
        raise ValueError("flow.loss_weighting must be diffsynth_bsmntw")
    lora_scope_target_regex(config["lora"]["scope"])
    if not isinstance(config["router"]["depth"], int) or config["router"]["depth"] < 0:
        raise ValueError("router.depth must be a non-negative integer")
    if not isinstance(config["training"]["num_workers"], int) or config["training"]["num_workers"] < 0:
        raise ValueError("training.num_workers must be a non-negative integer")
    for name in ("max_steps",):
        item = config["training"][name]
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ValueError(f"training.{name} must be a non-negative integer")

    finite_positive = {
        "flow.shift": config["flow"]["shift"],
        "lora.alpha": config["lora"]["alpha"],
        "optimizer.learning_rate": config["optimizer"]["learning_rate"],
        "optimizer.pretrained_adapter_learning_rate": config["optimizer"][
            "pretrained_adapter_learning_rate"
        ],
        "optimizer.eps": config["optimizer"]["eps"],
        "optimizer.max_grad_norm": config["optimizer"]["max_grad_norm"],
    }
    for name, item in finite_positive.items():
        if not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(float(item)) or item <= 0:
            raise ValueError(f"{name} must be finite and positive")
    bounded = {
        "lora.dropout": (config["lora"]["dropout"], 0.0, 1.0),
        "optimizer.beta1": (config["optimizer"]["beta1"], 0.0, 1.0),
        "optimizer.beta2": (config["optimizer"]["beta2"], 0.0, 1.0),
    }
    for name, (item, lower, upper) in bounded.items():
        if not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(float(item)) or not lower <= item < upper:
            raise ValueError(f"{name} must lie in [{lower}, {upper})")
    weight_decay = config["optimizer"]["weight_decay"]
    if not isinstance(weight_decay, (int, float)) or isinstance(weight_decay, bool) or not math.isfinite(float(weight_decay)) or weight_decay < 0:
        raise ValueError("optimizer.weight_decay must be finite and non-negative")
    _loss_weights(config["loss_weights"])
    return config
