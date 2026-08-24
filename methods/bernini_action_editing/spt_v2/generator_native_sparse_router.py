#!/usr/bin/env python3
"""Inference-only sparse P/G routing from Bernini's action/no-op clean field.

The tri-branch Bernini hook exposes raw conditional action and semantic-noop
clean predictions at the *same* noisy state.  Their difference is therefore
the exact field supervised by same-state C2FR LoRA training.  This module converts
that proposal from Wan's spatial latent layout to SPT's phase-cell layout,
removes its temporal DC component, and ranks cells with multi-lag temporal
energy.  A deterministic exact top-k router then emits a hard student
``PhasePlan`` containing only preserve and generate gates.  The executed edit
field is the temporal quotient of that proposal, so a temporally constant
appearance replacement is not injected through an objective that never
supervised it.  A denoising-step EMA of generator saliency stabilizes support
selection without introducing any external condition.

There is no paired target, mask, tracker, optical flow, pose, trajectory, or
first-frame anchor API.  Execution is delegated to
``counterfactual_clean_field.execute_counterfactual_clean_plan`` so the clean
field is ``source + alpha * (action - noop)`` and G is applied exactly once.
Outside the selected hard support the result is checked to be bit-exactly the
source phase tensor.

This is deliberately a generator-native routing baseline, not a learned
localizer.  Its activity floors, energy coverage, and 12% capacity are exposed
as auditable heuristics rather than hidden train-time inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import tri_branch_unipc as tri

from . import counterfactual_clean_field as counterfactual
from . import phase_transport as spt


METHOD_NAME = "generator-native-quotient-energy-sparse-pg-router-v2"
MAX_ALLOWED_GENERATE_FRACTION = 0.12
WAN_PACKED_CHANNELS = 64


class GeneratorNativeSparseRouterError(spt.PhaseTransportError):
    """Raised before execution when a sparse-router invariant is violated."""


@dataclass(frozen=True)
class GeneratorNativeSparseRouterConfig:
    """Auditable inference heuristics for generator-native sparse routing."""

    max_generate_fraction_per_phase: float = MAX_ALLOWED_GENERATE_FRACTION
    temporal_lags: tuple[int, ...] = (1, 2, 4)
    temporal_lag_weights: tuple[float, ...] = (1.0, 0.5, 0.25)
    centered_energy_weight: float = 1.0
    multi_lag_energy_weight: float = 1.0
    activity_energy_floor: float = 1e-8
    relative_phase_activity_floor: float = 0.05
    energy_coverage: float = 0.85
    static_delta_retention: float = 0.0
    denoise_saliency_ema_decay: float = 0.8

    def validate(self) -> None:
        fraction = _finite_float(
            self.max_generate_fraction_per_phase,
            label="max_generate_fraction_per_phase",
        )
        if not 0.0 < fraction <= MAX_ALLOWED_GENERATE_FRACTION:
            raise GeneratorNativeSparseRouterError(
                "max_generate_fraction_per_phase must lie in (0,0.12]"
            )
        if not self.temporal_lags:
            raise GeneratorNativeSparseRouterError("temporal_lags cannot be empty")
        if len(self.temporal_lags) != len(self.temporal_lag_weights):
            raise GeneratorNativeSparseRouterError(
                "temporal_lags and temporal_lag_weights must have equal length"
            )
        if len(set(self.temporal_lags)) != len(self.temporal_lags):
            raise GeneratorNativeSparseRouterError("temporal_lags must be unique")
        for lag in self.temporal_lags:
            if type(lag) is not int or not 0 < lag < spt.LATENT_PHASES:
                raise GeneratorNativeSparseRouterError(
                    f"temporal lags must be integers in [1,{spt.LATENT_PHASES - 1}]"
                )
        for weight in self.temporal_lag_weights:
            if _finite_float(weight, label="temporal_lag_weight") <= 0.0:
                raise GeneratorNativeSparseRouterError(
                    "temporal_lag_weights must be positive"
                )
        saliency_weights = {}
        for name in ("centered_energy_weight", "multi_lag_energy_weight"):
            saliency_weights[name] = _finite_float(getattr(self, name), label=name)
            if saliency_weights[name] < 0.0:
                raise GeneratorNativeSparseRouterError(f"{name} must be non-negative")
        if all(weight == 0.0 for weight in saliency_weights.values()):
            raise GeneratorNativeSparseRouterError(
                "at least one saliency energy weight must be positive"
            )
        if _finite_float(self.activity_energy_floor, label="activity_energy_floor") < 0.0:
            raise GeneratorNativeSparseRouterError(
                "activity_energy_floor must be non-negative"
            )
        relative = _finite_float(
            self.relative_phase_activity_floor,
            label="relative_phase_activity_floor",
        )
        if not 0.0 <= relative <= 1.0:
            raise GeneratorNativeSparseRouterError(
                "relative_phase_activity_floor must lie in [0,1]"
            )
        coverage = _finite_float(self.energy_coverage, label="energy_coverage")
        if not 0.0 < coverage <= 1.0:
            raise GeneratorNativeSparseRouterError(
                "energy_coverage must lie in (0,1]"
            )
        retention = _finite_float(
            self.static_delta_retention, label="static_delta_retention"
        )
        if not 0.0 <= retention <= 1.0:
            raise GeneratorNativeSparseRouterError(
                "static_delta_retention must lie in [0,1]"
            )
        decay = _finite_float(
            self.denoise_saliency_ema_decay,
            label="denoise_saliency_ema_decay",
        )
        if not 0.0 <= decay < 1.0:
            raise GeneratorNativeSparseRouterError(
                "denoise_saliency_ema_decay must lie in [0,1)"
            )


@dataclass(frozen=True)
class SparseCleanExecution:
    """One clean callback result plus its hard student routing decision."""

    executed_clean_spatial: Any
    executed_clean_phase: Any
    plan: spt.PhasePlan


def runtime_contract(
    config: Optional[GeneratorNativeSparseRouterConfig] = None,
) -> dict[str, Any]:
    """Return the train/test-boundary and numerical routing contract.

    The frozen-base v2 control keeps the historical temporal quotient.  A
    causal-boundary-trained adapter passes ``static_delta_retention=1`` and
    therefore executes its identified raw field.  Making the choice an input
    to the receipt prevents a raw/quotient switch from being hidden behind one
    static contract blob.
    """

    cfg = config or GeneratorNativeSparseRouterConfig()
    cfg.validate()
    retention = float(cfg.static_delta_retention)
    execution_representation = (
        "raw_boundary_gauged_clean_delta"
        if retention == 1.0
        else "temporal_mean_quotient_with_static_retention"
    )

    return {
        "method": METHOD_NAME,
        "status": "inference-only-generator-native-router",
        "external_inference_conditions": [
            "source_video",
            "action_instruction",
        ],
        "internal_fixed_controls": [
            "semantic_noop_instruction",
            "negative_prompt",
        ],
        "same_state_input": (
            "raw_action_condition_clean_minus_raw_noop_condition_clean"
        ),
        "training_alignment": "exact_-sigma*(v_action-v_noop)_clean_delta",
        "official_apg_role": "parity_certificate_only_not_routed_delta",
        "saliency_formula": (
            "mean_channel((delta-mean_time(delta))^2)"
            "+weighted_endpoint_mean_multilag_delta_difference_energy"
        ),
        "activity_abstain": (
            "phase_mean_saliency>absolute_floor_and_"
            "phase_mean_saliency>=relative_floor*sequence_peak"
        ),
        "selection": (
            "stable_descending_energy_coverage_topk_with_canonical_flat_ties"
        ),
        "generate_fraction_hard_cap": MAX_ALLOWED_GENERATE_FRACTION,
        "gates": ["hard_preserve", "zero_transport", "hard_generate"],
        "execution_field_representation": execution_representation,
        "execution_field": (
            "raw_boundary_gauged_action_minus_noop_clean_field"
            if retention == 1.0
            else "temporal_quotient_action_minus_noop_clean_field"
        ),
        "execution_field_formula": (
            "delta_motion=delta-mean_time(delta)+"
            "static_delta_retention*mean_time(delta)"
        ),
        "static_delta_retention": retention,
        "raw_field_execution": retention == 1.0,
        "temporal_mean_subtraction_at_execution": retention != 1.0,
        "counterfactual_formula": "source_clean+alpha*delta_motion",
        "denoise_support_memory": "causal_saliency_ema",
        "execution_formula": "P*source_clean+G*counterfactual_clean",
        "generate_gate_application_count": 1,
        "outside_generate_support": "bit_exact_source_phase_tensor",
        "spatial_layout": "B,C,T,H,W",
        "packed_layout": "B,T*(H/2)*(W/2),64",
        "phase_layout": "B,T,H/2,W/2,64",
        "latent_phases": spt.LATENT_PHASES,
        "forbidden_conditions": [
            "target_video",
            "paired_target",
            "mask",
            "track",
            "pose",
            "optical_flow",
            "trajectory",
            "first_frame_anchor",
        ],
        "learned_parameters": False,
    }


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise GeneratorNativeSparseRouterError(f"{label} must be a finite scalar")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise GeneratorNativeSparseRouterError(
            f"{label} must be a finite scalar"
        ) from error
    if not math.isfinite(numeric):
        raise GeneratorNativeSparseRouterError(f"{label} must be finite")
    return numeric


def _validate_layout(layout: tri.PackedLatentLayout) -> None:
    if not isinstance(layout, tri.PackedLatentLayout):
        raise GeneratorNativeSparseRouterError(
            "layout must be tri_branch_unipc.PackedLatentLayout"
        )
    if layout.frames != spt.LATENT_PHASES:
        raise GeneratorNativeSparseRouterError(
            f"Wan spatial latent must have exactly {spt.LATENT_PHASES} phases"
        )
    if layout.packed_channels != WAN_PACKED_CHANNELS:
        raise GeneratorNativeSparseRouterError(
            f"Wan 1x2x2 packing must produce {WAN_PACKED_CHANNELS} channels"
        )


def _validate_floating_tensor(value: Any, *, label: str) -> None:
    try:
        import torch
    except Exception as error:  # pragma: no cover - exercised on AUH
        raise GeneratorNativeSparseRouterError("PyTorch is required") from error
    if not isinstance(value, torch.Tensor):
        raise GeneratorNativeSparseRouterError(f"{label} must be a torch.Tensor")
    if not torch.is_floating_point(value):
        raise GeneratorNativeSparseRouterError(f"{label} must be floating point")
    if not bool(torch.isfinite(value).all()):
        raise GeneratorNativeSparseRouterError(f"{label} contains non-finite values")


def spatial_to_packed(
    spatial: Any, *, layout: tri.PackedLatentLayout
) -> Any:
    """Pack ``[B,C,T,H,W]`` with the audited Wan 1x2x2 permutation."""

    _validate_layout(layout)
    _validate_floating_tensor(spatial, label="spatial clean latent")
    try:
        return tri._spatial_to_packed(spatial, layout)
    except tri.TriBranchHookError as error:
        raise GeneratorNativeSparseRouterError(str(error)) from error


def packed_to_spatial(
    packed: Any, *, layout: tri.PackedLatentLayout
) -> Any:
    """Unpack Wan ``[B,N,64]`` into ``[B,C,T,H,W]``."""

    _validate_layout(layout)
    _validate_floating_tensor(packed, label="packed clean latent")
    try:
        return tri._packed_to_spatial(packed, layout)
    except tri.TriBranchHookError as error:
        raise GeneratorNativeSparseRouterError(str(error)) from error


def packed_to_phase_video(
    packed: Any, *, layout: tri.PackedLatentLayout
) -> Any:
    """View Wan packed tokens as SPT ``[B,T,H/2,W/2,64]`` cells."""

    _validate_layout(layout)
    _validate_floating_tensor(packed, label="packed clean latent")
    actual = tuple(int(size) for size in packed.shape)
    if actual != layout.packed_shape:
        raise GeneratorNativeSparseRouterError(
            f"packed clean latent shape {actual} differs from {layout.packed_shape}"
        )
    try:
        return spt.packed_to_video(
            packed,
            height=layout.height // tri.PACK_PATCH_HEIGHT,
            width=layout.width // tri.PACK_PATCH_WIDTH,
        )
    except spt.PhaseTransportError as error:
        raise GeneratorNativeSparseRouterError(str(error)) from error


def phase_video_to_packed(
    phase_video: Any, *, layout: tri.PackedLatentLayout
) -> Any:
    """Flatten SPT phase cells back to Wan ``[B,N,64]`` tokens."""

    _validate_layout(layout)
    _validate_phase_video(phase_video, label="phase clean latent", layout=layout)
    try:
        return spt.video_to_packed(phase_video)
    except spt.PhaseTransportError as error:
        raise GeneratorNativeSparseRouterError(str(error)) from error


def spatial_to_phase_video(
    spatial: Any, *, layout: tri.PackedLatentLayout
) -> Any:
    """Convert tri-branch spatial clean fields to SPT phase cells."""

    return packed_to_phase_video(spatial_to_packed(spatial, layout=layout), layout=layout)


def phase_video_to_spatial(
    phase_video: Any, *, layout: tri.PackedLatentLayout
) -> Any:
    """Convert SPT phase cells back to tri-branch spatial clean fields."""

    return packed_to_spatial(
        phase_video_to_packed(phase_video, layout=layout), layout=layout
    )


def _validate_phase_video(
    value: Any,
    *,
    label: str,
    layout: Optional[tri.PackedLatentLayout] = None,
) -> None:
    _validate_floating_tensor(value, label=label)
    if value.ndim != 5 or int(value.shape[1]) != spt.LATENT_PHASES:
        raise GeneratorNativeSparseRouterError(
            f"{label} must be [B,{spt.LATENT_PHASES},H,W,D]"
        )
    if any(int(size) <= 0 for size in value.shape):
        raise GeneratorNativeSparseRouterError(f"{label} has an empty dimension")
    if int(value.shape[-1]) != WAN_PACKED_CHANNELS:
        raise GeneratorNativeSparseRouterError(
            f"{label} must contain {WAN_PACKED_CHANNELS} packed channels"
        )
    if layout is not None:
        expected = (
            layout.batch,
            layout.frames,
            layout.height // tri.PACK_PATCH_HEIGHT,
            layout.width // tri.PACK_PATCH_WIDTH,
            layout.packed_channels,
        )
        if tuple(int(size) for size in value.shape) != expected:
            raise GeneratorNativeSparseRouterError(
                f"{label} shape {tuple(value.shape)} differs from {expected}"
            )


def source_to_phase_video(
    source_clean: Any, *, layout: tri.PackedLatentLayout
) -> Any:
    """Normalize a spatial, packed, or phase source to the SPT phase layout."""

    _validate_layout(layout)
    _validate_floating_tensor(source_clean, label="source_clean")
    shape = tuple(int(size) for size in source_clean.shape)
    spatial_shape = (
        layout.batch,
        layout.channels,
        layout.frames,
        layout.height,
        layout.width,
    )
    phase_shape = (
        layout.batch,
        layout.frames,
        layout.height // tri.PACK_PATCH_HEIGHT,
        layout.width // tri.PACK_PATCH_WIDTH,
        layout.packed_channels,
    )
    if shape == spatial_shape:
        return spatial_to_phase_video(source_clean, layout=layout)
    if shape == layout.packed_shape:
        return packed_to_phase_video(source_clean, layout=layout)
    if shape == phase_shape:
        return source_clean
    raise GeneratorNativeSparseRouterError(
        "source_clean must match the tri spatial, Wan packed, or SPT phase layout"
    )


def generator_native_motion_saliency(
    guided_action_noop_delta_phase: Any,
    *,
    config: Optional[GeneratorNativeSparseRouterConfig] = None,
) -> Any:
    """Compute per-phase/cell motion energy after temporal DC removal.

    For ``d[t,i,c] = action_clean - noop_clean``, the first term is
    ``mean_c((d - mean_t(d))**2)``.  For every configured lag, squared
    same-cell temporal differences contribute to both endpoints; contributions
    are normalized by the sum of incident lag weights.  This deliberately
    suppresses a temporally constant appearance-only action/no-op difference.
    """

    cfg = config or GeneratorNativeSparseRouterConfig()
    cfg.validate()
    _validate_phase_video(
        guided_action_noop_delta_phase,
        label="guided_action_noop_delta_phase",
    )
    import torch

    delta = guided_action_noop_delta_phase.float()
    centered = delta - delta.mean(dim=1, keepdim=True)
    centered_energy = centered.square().mean(dim=-1)
    lag_sum = torch.zeros_like(centered_energy)
    lag_weight_sum = torch.zeros_like(centered_energy)
    for lag, raw_weight in zip(cfg.temporal_lags, cfg.temporal_lag_weights):
        weight = float(raw_weight)
        pair_energy = (centered[:, lag:] - centered[:, :-lag]).square().mean(dim=-1)
        lag_sum[:, lag:] += weight * pair_energy
        lag_sum[:, :-lag] += weight * pair_energy
        lag_weight_sum[:, lag:] += weight
        lag_weight_sum[:, :-lag] += weight
    multi_lag = torch.where(
        lag_weight_sum > 0.0,
        lag_sum / lag_weight_sum.clamp_min(torch.finfo(lag_sum.dtype).tiny),
        torch.zeros_like(lag_sum),
    )
    saliency = (
        float(cfg.centered_energy_weight) * centered_energy
        + float(cfg.multi_lag_energy_weight) * multi_lag
    )
    if not bool(torch.isfinite(saliency).all()) or bool((saliency < 0.0).any()):
        raise GeneratorNativeSparseRouterError("motion saliency is invalid")
    return saliency


def temporal_static_quotient(
    guided_action_noop_delta_phase: Any,
    *,
    static_delta_retention: float = 0.0,
) -> Any:
    """Remove the unsupervised temporal-DC appearance component.

    The quotient/multi-lag LoRA objective never identifies a constant-in-time
    action/no-op offset.  Executing that null-space component would therefore
    reintroduce the very appearance replacement (for example, a different
    dog) that the representation was designed to reject.
    """

    _validate_phase_video(
        guided_action_noop_delta_phase,
        label="guided_action_noop_delta_phase",
    )
    retention = _finite_float(
        static_delta_retention, label="static_delta_retention"
    )
    if not 0.0 <= retention <= 1.0:
        raise GeneratorNativeSparseRouterError(
            "static_delta_retention must lie in [0,1]"
        )
    import torch

    value = guided_action_noop_delta_phase.float()
    temporal_mean = value.mean(dim=1, keepdim=True)
    result = value - temporal_mean + retention * temporal_mean
    if not bool(torch.isfinite(result).all()):
        raise GeneratorNativeSparseRouterError("temporal quotient is non-finite")
    return result


def causal_boundary_projection(guided_action_noop_delta_phase: Any) -> Any:
    """Project a field relative to its first latent phase.

    This is the executable representation for causal-boundary LoRA.  Unlike a
    temporal-mean quotient, a persistent step action remains zero before its
    onset and keeps its full terminal amplitude.  The first phase is exactly
    zero by construction, so residual raw-field DC cannot leak appearance into
    inference even when the auxiliary boundary loss has not fully converged.
    """

    _validate_phase_video(
        guided_action_noop_delta_phase,
        label="guided_action_noop_delta_phase",
    )
    import torch

    value = guided_action_noop_delta_phase.float()
    result = value - value[:, :1]
    if not bool(torch.isfinite(result).all()):
        raise GeneratorNativeSparseRouterError(
            "causal-boundary projection is non-finite"
        )
    if not bool(torch.equal(result[:, :1], torch.zeros_like(result[:, :1]))):
        raise GeneratorNativeSparseRouterError(
            "causal-boundary projection did not zero the first phase exactly"
        )
    return result


def causal_ema_boundary_projection(
    guided_action_noop_delta_phase: Any,
    *,
    decay: float = 0.5,
) -> Any:
    """Apply an inference-ablation causal low-pass followed by Q0.

    The formal v4 path intentionally does not call this operator: filtering
    along latent video time attenuates motion onset and produces a decaying
    tail.  It remains public only for a named EMA ablation.
    """

    _validate_phase_video(
        guided_action_noop_delta_phase,
        label="guided_action_noop_delta_phase",
    )
    if (
        isinstance(decay, bool)
        or not isinstance(decay, (int, float))
        or not math.isfinite(float(decay))
        or not 0.0 <= float(decay) < 1.0
    ):
        raise GeneratorNativeSparseRouterError(
            "causal EMA decay must lie in [0,1)"
        )
    import torch

    value = guided_action_noop_delta_phase.float()
    phases = [value[:, 0]]
    for phase_index in range(1, int(value.shape[1])):
        phases.append(
            float(decay) * phases[-1]
            + (1.0 - float(decay)) * value[:, phase_index]
        )
    filtered = torch.stack(phases, dim=1)
    result = filtered - filtered[:, :1]
    if not bool(torch.isfinite(result).all()):
        raise GeneratorNativeSparseRouterError(
            "causal EMA boundary projection is non-finite"
        )
    if not bool(torch.equal(result[:, :1], torch.zeros_like(result[:, :1]))):
        raise GeneratorNativeSparseRouterError(
            "causal EMA boundary projection did not zero phase zero exactly"
        )
    return result


def _canonical_sparse_support(
    saliency: Any,
    *,
    config: GeneratorNativeSparseRouterConfig,
) -> tuple[Any, Any, Any, int]:
    """Select exact stable top-k cells, with activity-conditioned abstention."""

    import torch

    if saliency.ndim != 4:
        raise GeneratorNativeSparseRouterError("saliency must be [B,T,H,W]")
    batch, phases, height, width = map(int, saliency.shape)
    if phases != spt.LATENT_PHASES:
        raise GeneratorNativeSparseRouterError(
            f"saliency must contain {spt.LATENT_PHASES} phases"
        )
    cells = height * width
    cap = int(math.floor(float(config.max_generate_fraction_per_phase) * cells))
    if cap > 0 and cap / cells > float(config.max_generate_fraction_per_phase):
        raise GeneratorNativeSparseRouterError("integer top-k exceeds configured cap")
    activity_energy = saliency.mean(dim=(-1, -2))
    sequence_peak = activity_energy.amax(dim=1, keepdim=True)
    absolute = float(config.activity_energy_floor)
    threshold = torch.maximum(
        torch.full_like(activity_energy, absolute),
        float(config.relative_phase_activity_floor) * sequence_peak,
    )
    active = (
        (sequence_peak > absolute)
        & (activity_energy >= threshold)
        & (saliency.sum(dim=(-1, -2)) > 0.0)
    )
    support = torch.zeros_like(saliency, dtype=torch.bool)
    counts = torch.zeros(
        batch, phases, device=saliency.device, dtype=torch.int64
    )
    if cap == 0:
        return support, activity_energy, active & False, cap

    flat_saliency = saliency.reshape(batch, phases, cells)
    flat_support = support.reshape(batch, phases, cells)
    for batch_index in range(batch):
        for phase_index in range(phases):
            if not bool(active[batch_index, phase_index].item()):
                continue
            scores = flat_saliency[batch_index, phase_index]
            positive_count = int((scores > 0.0).sum().item())
            if positive_count == 0:
                active[batch_index, phase_index] = False
                continue
            # Stable descending sort makes equal scores choose the smaller
            # canonical row-major flat index.  No epsilon perturbs real scores.
            order = torch.argsort(scores, dim=0, descending=True, stable=True)
            ranked = scores[order]
            total = ranked[:positive_count].sum()
            if not bool(total > 0.0):
                active[batch_index, phase_index] = False
                continue
            required_energy = float(config.energy_coverage) * total
            cumulative = ranked[:positive_count].cumsum(dim=0)
            needed = int(
                torch.searchsorted(cumulative, required_energy, right=False).item()
            ) + 1
            selected = min(cap, positive_count, needed)
            flat_support[batch_index, phase_index, order[:selected]] = True
            counts[batch_index, phase_index] = selected
    if bool((counts > cap).any()):
        raise GeneratorNativeSparseRouterError("exact top-k count exceeds cap")
    return support, activity_energy, active, cap


def generator_native_phase_plan(
    guided_action_noop_delta_phase: Any,
    *,
    config: Optional[GeneratorNativeSparseRouterConfig] = None,
    routing_saliency: Optional[Any] = None,
) -> spt.PhasePlan:
    """Return a hard source/instruction-time ``student`` P/G ``PhasePlan``."""

    cfg = config or GeneratorNativeSparseRouterConfig()
    cfg.validate()
    raw_saliency = generator_native_motion_saliency(
        guided_action_noop_delta_phase, config=cfg
    )
    saliency = raw_saliency if routing_saliency is None else routing_saliency
    if (
        getattr(saliency, "dtype", None) is None
        or tuple(getattr(saliency, "shape", ())) != tuple(raw_saliency.shape)
    ):
        raise GeneratorNativeSparseRouterError(
            "routing saliency must match the generator-native saliency shape"
        )
    _validate_floating_tensor(saliency, label="routing saliency")
    saliency = saliency.float()
    support, activity, active, cap = _canonical_sparse_support(
        saliency, config=cfg
    )
    import torch

    batch, phases, height, width = map(int, support.shape)
    gates = torch.zeros(
        batch,
        3,
        phases,
        height,
        width,
        device=support.device,
        dtype=torch.float32,
    )
    gates[:, spt.GATE_GENERATE] = support.float()
    gates[:, spt.GATE_PRESERVE] = (~support).float()
    offsets = torch.zeros_like(gates)
    plan = spt.PhasePlan(
        offsets=offsets,
        gate_probs=gates,
        provenance="student",
        diagnostics={
            "router": METHOD_NAME,
            "motion_saliency": saliency,
            "raw_motion_saliency": raw_saliency,
            "phase_activity_energy": activity,
            "active_phases": active,
            "selected_support": support,
            "selected_counts": support.sum(dim=(-1, -2)),
            "integer_capacity_per_phase": cap,
        },
    )
    plan.validate(guided_action_noop_delta_phase)
    return plan


def execute_generator_native_sparse_clean(
    fields: tri.CleanFieldStep,
    *,
    source_clean: Any,
    layout: tri.PackedLatentLayout,
    config: Optional[GeneratorNativeSparseRouterConfig] = None,
    alpha: Any = 1.0,
    routing_saliency: Optional[Any] = None,
) -> SparseCleanExecution:
    """Bridge one tri-branch clean step through SPT's counterfactual executor."""

    if not isinstance(fields, tri.CleanFieldStep):
        raise GeneratorNativeSparseRouterError(
            "fields must be tri_branch_unipc.CleanFieldStep"
        )
    _validate_layout(layout)
    source_phase = source_to_phase_video(source_clean, layout=layout)
    action_phase = spatial_to_phase_video(
        fields.action_condition_clean, layout=layout
    )
    noop_phase = spatial_to_phase_video(
        fields.noop_condition_clean, layout=layout
    )
    computed_delta_phase = action_phase.float() - noop_phase.float()
    import torch
    supplied_delta_phase = computed_delta_phase
    cfg = config or GeneratorNativeSparseRouterConfig()
    cfg.validate()
    plan = generator_native_phase_plan(
        supplied_delta_phase,
        config=cfg,
        routing_saliency=routing_saliency,
    )
    motion_delta_phase = temporal_static_quotient(
        supplied_delta_phase,
        static_delta_retention=cfg.static_delta_retention,
    )
    # Feed the represented residual against an exact zero control so the
    # generic counterfactual executor applies ``motion_delta_phase`` without a
    # lossy ``(noop + delta) - noop`` round trip.
    executable_noop_phase = torch.zeros_like(motion_delta_phase)
    try:
        executed_phase = counterfactual.execute_counterfactual_clean_plan(
            source=source_phase,
            action_clean=motion_delta_phase,
            noop_clean=executable_noop_phase,
            plan=plan,
            alpha=alpha,
            detach_source_bank=True,
        )
    except spt.PhaseTransportError as error:
        raise GeneratorNativeSparseRouterError(str(error)) from error

    support = plan.gate_probs[:, spt.GATE_GENERATE].bool()
    outside = (~support).unsqueeze(-1).expand_as(executed_phase)
    source_float = source_phase.float()
    if not torch.equal(executed_phase[outside], source_float[outside]):
        raise GeneratorNativeSparseRouterError(
            "preserve region is not bit-exactly the source"
        )
    executed_spatial = phase_video_to_spatial(executed_phase, layout=layout)
    return SparseCleanExecution(
        executed_clean_spatial=executed_spatial,
        executed_clean_phase=executed_phase,
        plan=plan,
    )


class GeneratorNativeSparseCleanCallback:
    """State-light callable suitable for ``tri_branch_unipc_hook``."""

    def __init__(
        self,
        *,
        source_clean: Any,
        layout: tri.PackedLatentLayout,
        config: Optional[GeneratorNativeSparseRouterConfig] = None,
        alpha: Any = 1.0,
    ) -> None:
        _validate_layout(layout)
        self.source_clean = source_clean
        self.layout = layout
        self.config = config or GeneratorNativeSparseRouterConfig()
        self.config.validate()
        self.alpha = alpha
        self.last_execution: Optional[SparseCleanExecution] = None
        self._saliency_ema: Optional[Any] = None

    def __call__(self, fields: tri.CleanFieldStep) -> Any:
        action_phase = spatial_to_phase_video(
            fields.action_condition_clean, layout=self.layout
        )
        noop_phase = spatial_to_phase_video(
            fields.noop_condition_clean, layout=self.layout
        )
        delta_phase = action_phase.float() - noop_phase.float()
        current_saliency = generator_native_motion_saliency(
            delta_phase, config=self.config
        )
        if self._saliency_ema is None:
            routing_saliency = current_saliency
        else:
            decay = float(self.config.denoise_saliency_ema_decay)
            routing_saliency = (
                decay * self._saliency_ema
                + (1.0 - decay) * current_saliency.float()
            )
        # Support memory is causal and detached across inference steps; no
        # training graph or future denoising state is consulted.
        self._saliency_ema = routing_saliency.detach()
        execution = execute_generator_native_sparse_clean(
            fields,
            source_clean=self.source_clean,
            layout=self.layout,
            config=self.config,
            alpha=self.alpha,
            routing_saliency=routing_saliency,
        )
        self.last_execution = execution
        return execution.executed_clean_spatial


__all__ = [
    "GeneratorNativeSparseCleanCallback",
    "GeneratorNativeSparseRouterConfig",
    "GeneratorNativeSparseRouterError",
    "MAX_ALLOWED_GENERATE_FRACTION",
    "METHOD_NAME",
    "SparseCleanExecution",
    "causal_boundary_projection",
    "causal_ema_boundary_projection",
    "execute_generator_native_sparse_clean",
    "generator_native_motion_saliency",
    "generator_native_phase_plan",
    "packed_to_phase_video",
    "packed_to_spatial",
    "phase_video_to_packed",
    "phase_video_to_spatial",
    "runtime_contract",
    "source_to_phase_video",
    "spatial_to_packed",
    "spatial_to_phase_video",
    "temporal_static_quotient",
]
