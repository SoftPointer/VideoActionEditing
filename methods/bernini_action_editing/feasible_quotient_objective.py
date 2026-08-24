#!/usr/bin/env python3
"""Projection-consistent RS-FQT objective for Bernini action editing v8.

The objective trains a complete action/no-op quotient while the frozen no-op
field remains the deployment appearance section::

    q_theta = Q0(A_theta - N_theta)
    z_theta = FIR(D q_theta)
    z_star  = Project_{0.95 r}(FIR(D Q0(T-N0)))
    F_hat   = N0 + rho * Integrate(Project_r(z_theta))

The motion loss differentiates the complete deployed quotient, including the
shared-adapter no-op branch.  ``N_theta`` is also constrained by a separate
all-sigma no-op preservation loss.
The target quotient is relative to the frozen no-op reconstruction section on
the *same diffusion query*.  This prevents a target-noised training query from
counting ``T-S`` a second time when ``N0`` already reconstructs part of the
edit.  The source-only radius is identical to deployment and has no
target-dependent term.  No mask, flow, pose, track, trajectory, first-frame
anchor, or T2V generator enters this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

try:  # Package import.
    from . import gauge_anchored_commutator as gauge
    from . import motion_commutator as commutator
except ImportError:  # Direct import with METHOD_ROOT on sys.path.
    import gauge_anchored_commutator as gauge
    import motion_commutator as commutator


METHOD_NAME = "bernini-reconstruction-section-feasible-quotient-objective-v8"
RECEIPT_SCHEMA = "bernini-rs-fqt-objective-receipt-v8"
EXPECTED_PHASES = 21
FORWARD_BRANCH_ORDER = (
    "frozen_editor_negative_full_source",
    "frozen_editor_noop_full_source",
    "frozen_editor_action_full_source",
    "adapted_editor_noop_full_source",
    "adapted_editor_action_full_source",
)
GRAPH_BRANCHES = (
    "adapted_editor_noop_full_source",
    "adapted_editor_action_full_source",
)
FORBIDDEN_INFERENCE_CONDITIONS = (
    "paired_target_video",
    "generator_branch",
    "mask",
    "track",
    "swept_tube",
    "pose",
    "trajectory",
    "optical_flow",
    "first_frame_anchor",
)


class FeasibleQuotientObjectiveError(RuntimeError):
    """Raised before an invalid v8 candidate can create an update."""


@dataclass(frozen=True)
class FeasibleQuotientLossConfig:
    canonical_weight: float = 1.0
    executed_weight: float = 0.25
    noop_preservation_weight: float = 0.5
    margin_weight: float = 0.05
    temporal_jitter_weight: float = 0.02
    target_interior_ratio: float = 0.95
    occupancy_margin: float = 0.98
    charbonnier_epsilon: float = 1.0e-3
    normalization_floor: float = 1.0e-4
    feasible_config: gauge.FeasibleQuotientConfig = field(
        default_factory=gauge.FeasibleQuotientConfig
    )

    def validate(self) -> None:
        for name in (
            "canonical_weight",
            "executed_weight",
            "noop_preservation_weight",
            "margin_weight",
            "temporal_jitter_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise FeasibleQuotientObjectiveError(
                    f"{name} must be finite and nonnegative"
                )
        if (
            float(self.canonical_weight) <= 0.0
            or float(self.noop_preservation_weight) <= 0.0
        ):
            raise FeasibleQuotientObjectiveError(
                "canonical and no-op weights must be strictly positive"
            )
        for name in ("target_interior_ratio", "occupancy_margin"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 < float(value) < 1.0
            ):
                raise FeasibleQuotientObjectiveError(
                    f"{name} must lie strictly between zero and one"
                )
        for name in ("charbonnier_epsilon", "normalization_floor"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise FeasibleQuotientObjectiveError(
                    f"{name} must be finite and strictly positive"
                )
        try:
            self.feasible_config.validate()
        except gauge.GaugeAnchoredCommutatorError as error:
            raise FeasibleQuotientObjectiveError(str(error)) from error


@dataclass(frozen=True)
class FiveBranchCleanFields:
    """Post-guidance clean fields plus paired offline supervision.

    Every tensor is float32 ``[B,21,S,D]``.  Only the two adapted editor
    fields retain graphs.  The target is a training label, never a model input.
    """

    frozen_editor_noop: Any
    frozen_editor_action: Any
    adapted_editor_noop: Any
    adapted_editor_action: Any
    source_clean: Any
    target_clean: Any


@dataclass(frozen=True)
class FeasibleQuotientDiagnostics:
    frozen_quotient: Any
    adapted_quotient: Any
    target_quotient: Any
    frozen_smoothed_increments: Any
    predicted_smoothed_increments: Any
    target_unsmoothed_increments: Any
    target_raw_smoothed_increments: Any
    canonical_target_increments: Any
    executed_predicted_increments: Any
    source_only_radius: Any
    predicted_projection_scale: Any
    target_projection_scale: Any
    predicted_execution: Any
    target_execution: Any
    execution_normalization_radius: Any
    frozen_noop_to_source_rms_per_sample: Any
    frozen_noop_to_target_rms_per_sample: Any
    target_frozen_prior_cosine_per_phase: Any
    canonical_per_sample: Any
    executed_per_sample: Any
    noop_per_sample: Any
    margin_per_sample: Any
    jitter_per_sample: Any
    motion_noop_full_gradient: bool


@dataclass(frozen=True)
class FeasibleQuotientLossResult:
    total: Any
    canonical: Any
    executed: Any
    noop_preservation: Any
    margin: Any
    temporal_jitter: Any
    rho: float
    diagnostics: FeasibleQuotientDiagnostics


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependent
        raise FeasibleQuotientObjectiveError(
            "v8 objective tensor operations require PyTorch"
        ) from error
    return torch


def _validate_fields(fields: FiveBranchCleanFields) -> None:
    torch = _require_torch()
    names = tuple(FiveBranchCleanFields.__dataclass_fields__)
    tensors = tuple(getattr(fields, name) for name in names)
    reference = tensors[0]
    if (
        not isinstance(reference, torch.Tensor)
        or reference.ndim != 4
        or int(reference.shape[0]) <= 0
        or int(reference.shape[1]) != EXPECTED_PHASES
        or int(reference.shape[2]) <= 0
        or int(reference.shape[3]) <= 0
        or reference.dtype != torch.float32
    ):
        raise FeasibleQuotientObjectiveError(
            "clean fields must be float32 [B,21,S,D] tensors"
        )
    for name, tensor in zip(names, tensors):
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != tuple(reference.shape)
            or tensor.dtype != reference.dtype
            or tensor.device != reference.device
            or not bool(torch.isfinite(tensor).all())
        ):
            raise FeasibleQuotientObjectiveError(
                f"{name} shape, dtype, device, or finiteness differs"
            )
    graph_names = {"adapted_editor_noop", "adapted_editor_action"}
    for name in names:
        requires_grad = bool(getattr(fields, name).requires_grad)
        if (name in graph_names) is not requires_grad:
            expected = "retain" if name in graph_names else "exclude"
            raise FeasibleQuotientObjectiveError(
                f"{name} must {expected} the adapter graph"
            )


def _causal_gauge(value: Any) -> Any:
    try:
        return commutator.causal_gauge(value)
    except commutator.MotionCommutatorError as error:
        raise FeasibleQuotientObjectiveError(str(error)) from error


def _phase_increments(value: Any) -> Any:
    try:
        return commutator.phase_increments(value)
    except commutator.MotionCommutatorError as error:
        raise FeasibleQuotientObjectiveError(str(error)) from error


def _phase_charbonnier_per_sample(
    error: Any,
    phase_scale: Any,
    *,
    epsilon: float,
) -> Any:
    normalized = error / phase_scale[..., None, None]
    return (
        (normalized.square() + float(epsilon) ** 2).sqrt()
        - float(epsilon)
    ).mean(dim=(1, 2, 3))


def _sample_charbonnier_per_sample(
    error: Any,
    sample_scale: Any,
    *,
    epsilon: float,
) -> Any:
    normalized = error / sample_scale[:, None, None, None]
    return (
        (normalized.square() + float(epsilon) ** 2).sqrt()
        - float(epsilon)
    ).mean(dim=(1, 2, 3))


def _source_only_radius(
    frozen_quotient: Any,
    frozen_noop: Any,
    config: FeasibleQuotientLossConfig,
) -> Any:
    torch = _require_torch()
    frozen_z = gauge.smooth_phase_increments(
        _phase_increments(frozen_quotient)
    )
    noop_z = gauge.smooth_phase_increments(
        _phase_increments(_causal_gauge(frozen_noop))
    )
    frozen_rms = gauge.phase_rms(frozen_z)
    noop_rms = gauge.phase_rms(noop_z)
    feasible = config.feasible_config
    floor = torch.full_like(frozen_rms, float(feasible.radius_floor))
    return torch.maximum(
        torch.maximum(
            float(feasible.frozen_quotient_radius_ratio) * frozen_rms,
            float(feasible.noop_dynamics_radius_ratio) * noop_rms,
        ),
        floor,
    ).detach()


def compute_feasible_quotient_objective(
    fields: FiveBranchCleanFields,
    *,
    step_index: int,
    config: FeasibleQuotientLossConfig = FeasibleQuotientLossConfig(),
) -> FeasibleQuotientLossResult:
    """Compute one projection-consistent five-forward v8 candidate loss."""

    torch = _require_torch()
    config.validate()
    _validate_fields(fields)
    try:
        rho = float(commutator.release_rho(step_index))
    except commutator.MotionCommutatorError as error:
        raise FeasibleQuotientObjectiveError(str(error)) from error

    frozen_quotient = _causal_gauge(
        fields.frozen_editor_action - fields.frozen_editor_noop
    ).detach()
    # Match both the forward value and the shared-parameter derivative of the
    # deployed quotient.  Detaching N_theta here would optimize J_A while
    # deployment changes as J_A-J_N.
    adapted_quotient = _causal_gauge(
        fields.adapted_editor_action - fields.adapted_editor_noop
    )
    # The paired target is a training label, while N0 is the exact frozen
    # reconstruction section used by deployment on this same query.  Defining
    # the teacher relative to N0 is essential for target-noised (beta=1)
    # training: Q0(T-S) would double-count any edit already reconstructed by
    # N0.  Detaching N0 keeps the teacher and its section query non-trainable.
    target_quotient = _causal_gauge(
        fields.target_clean - fields.frozen_editor_noop.detach()
    ).detach()
    frozen_z = gauge.smooth_phase_increments(
        _phase_increments(frozen_quotient)
    )
    predicted_z = gauge.smooth_phase_increments(
        _phase_increments(adapted_quotient)
    )
    target_unsmoothed_z = _phase_increments(target_quotient)
    target_raw_z = gauge.smooth_phase_increments(target_unsmoothed_z)
    radius = _source_only_radius(
        frozen_quotient, fields.frozen_editor_noop, config
    )
    try:
        canonical_target, target_scale = gauge.radial_project_phase_increments(
            target_raw_z,
            float(config.target_interior_ratio) * radius,
            epsilon=float(config.feasible_config.epsilon),
        )
        executed_predicted, predicted_scale = (
            gauge.radial_project_phase_increments(
                predicted_z,
                radius,
                epsilon=float(config.feasible_config.epsilon),
            )
        )
    except gauge.GaugeAnchoredCommutatorError as error:
        raise FeasibleQuotientObjectiveError(str(error)) from error

    active_radius = radius[:, 1:].clamp_min(
        float(config.normalization_floor)
    )
    canonical_error = rho * (predicted_z[:, 1:] - canonical_target[:, 1:])
    canonical_per_sample = _phase_charbonnier_per_sample(
        canonical_error,
        active_radius,
        epsilon=float(config.charbonnier_epsilon),
    )

    # The deployed object is a lifted clean field, not an increment tensor.
    # Use the exact deployment primitive here so an early phase error is
    # charged at every later phase it displaces.  The cumulative trust radius
    # is the corresponding source-only normalization in clean-field space.
    try:
        predicted_transport = gauge.execute_feasible_quotient_transport(
            fields.frozen_editor_noop.detach(),
            gauge.integrate_phase_increments(executed_predicted),
            step_index=step_index,
        )
        target_transport = gauge.execute_feasible_quotient_transport(
            fields.frozen_editor_noop.detach(),
            gauge.integrate_phase_increments(canonical_target),
            step_index=step_index,
        )
    except gauge.GaugeAnchoredCommutatorError as error:
        raise FeasibleQuotientObjectiveError(str(error)) from error
    predicted_execution = predicted_transport.executed_clean_field
    target_execution = target_transport.executed_clean_field
    execution_radius = torch.cumsum(radius[:, 1:], dim=1).clamp_min(
        float(config.normalization_floor)
    )
    executed_error = predicted_execution[:, 1:] - target_execution[:, 1:]
    executed_per_sample = _phase_charbonnier_per_sample(
        executed_error,
        execution_radius,
        epsilon=float(config.charbonnier_epsilon),
    )

    frozen_noop_to_source_rms = (
        fields.frozen_editor_noop.detach() - fields.source_clean
    ).square().mean(dim=(1, 2, 3)).sqrt()
    frozen_noop_to_target_rms = (
        fields.frozen_editor_noop.detach() - fields.target_clean
    ).square().mean(dim=(1, 2, 3)).sqrt()
    target_prior_dot = (target_raw_z * frozen_z).sum(dim=(-1, -2))
    target_prior_denominator = (
        target_raw_z.square().sum(dim=(-1, -2)).sqrt()
        * frozen_z.square().sum(dim=(-1, -2)).sqrt()
    )
    target_prior_cosine = torch.where(
        target_prior_denominator > 1.0e-12,
        target_prior_dot / target_prior_denominator.clamp_min(1.0e-12),
        torch.zeros_like(target_prior_dot),
    ).clamp(-1.0, 1.0).detach()

    noop_error = fields.adapted_editor_noop - fields.frozen_editor_noop
    noop_scale = fields.frozen_editor_noop.square().mean(
        dim=(1, 2, 3)
    ).sqrt().clamp_min(float(config.normalization_floor)).detach()
    noop_per_sample = _sample_charbonnier_per_sample(
        noop_error,
        noop_scale,
        epsilon=float(config.charbonnier_epsilon),
    )

    occupancy = gauge.phase_rms(predicted_z)[:, 1:] / active_radius
    margin_per_sample = rho * (
        occupancy - float(config.occupancy_margin)
    ).clamp_min(0.0).square().mean(dim=1)

    residual_z = predicted_z - canonical_target
    if int(residual_z.shape[1]) < 3:
        raise FeasibleQuotientObjectiveError(
            "temporal jitter requires at least two active phase intervals"
        )
    residual_jitter = rho * (
        residual_z[:, 2:] - residual_z[:, 1:-1]
    )
    jitter_scale = 0.5 * (radius[:, 2:] + radius[:, 1:-1])
    jitter_scale = jitter_scale.clamp_min(float(config.normalization_floor))
    jitter_per_sample = _phase_charbonnier_per_sample(
        residual_jitter,
        jitter_scale,
        epsilon=float(config.charbonnier_epsilon),
    )

    canonical_loss = canonical_per_sample.mean()
    executed_loss = executed_per_sample.mean()
    noop_loss = noop_per_sample.mean()
    margin_loss = margin_per_sample.mean()
    jitter_loss = jitter_per_sample.mean()
    total = (
        float(config.canonical_weight) * canonical_loss
        + float(config.executed_weight) * executed_loss
        + float(config.noop_preservation_weight) * noop_loss
        + float(config.margin_weight) * margin_loss
        + float(config.temporal_jitter_weight) * jitter_loss
    )
    for name, value in (
        ("canonical", canonical_loss),
        ("executed", executed_loss),
        ("noop", noop_loss),
        ("margin", margin_loss),
        ("jitter", jitter_loss),
        ("total", total),
    ):
        if not bool(torch.isfinite(value)):
            raise FeasibleQuotientObjectiveError(f"{name} loss is non-finite")
    if not total.requires_grad:
        raise FeasibleQuotientObjectiveError(
            "v8 total loss lost both adapted-editor graphs"
        )

    return FeasibleQuotientLossResult(
        total=total,
        canonical=canonical_loss,
        executed=executed_loss,
        noop_preservation=noop_loss,
        margin=margin_loss,
        temporal_jitter=jitter_loss,
        rho=rho,
        diagnostics=FeasibleQuotientDiagnostics(
            frozen_quotient=frozen_quotient,
            adapted_quotient=adapted_quotient,
            target_quotient=target_quotient,
            frozen_smoothed_increments=frozen_z,
            predicted_smoothed_increments=predicted_z,
            target_unsmoothed_increments=target_unsmoothed_z,
            target_raw_smoothed_increments=target_raw_z,
            canonical_target_increments=canonical_target,
            executed_predicted_increments=executed_predicted,
            source_only_radius=radius,
            predicted_projection_scale=predicted_scale,
            target_projection_scale=target_scale,
            predicted_execution=predicted_execution,
            target_execution=target_execution,
            execution_normalization_radius=execution_radius,
            frozen_noop_to_source_rms_per_sample=frozen_noop_to_source_rms,
            frozen_noop_to_target_rms_per_sample=frozen_noop_to_target_rms,
            target_frozen_prior_cosine_per_phase=target_prior_cosine,
            canonical_per_sample=canonical_per_sample,
            executed_per_sample=executed_per_sample,
            noop_per_sample=noop_per_sample,
            margin_per_sample=margin_per_sample,
            jitter_per_sample=jitter_per_sample,
            motion_noop_full_gradient=True,
        ),
    )


def detached_receipt_diagnostics(
    result: FeasibleQuotientLossResult,
) -> dict[str, Any]:
    """Return compact, JSON-ready deployment-consistency evidence."""

    torch = _require_torch()
    diagnostics = result.diagnostics

    def mean(value: Any) -> float:
        return float(value.detach().float().mean().cpu().item())

    target_active = diagnostics.target_projection_scale[:, 1:]
    predicted_active = diagnostics.predicted_projection_scale[:, 1:]
    radius_active = diagnostics.source_only_radius[:, 1:]
    target_raw_rms = gauge.phase_rms(
        diagnostics.target_raw_smoothed_increments
    )[:, 1:]
    canonical_rms = gauge.phase_rms(
        diagnostics.canonical_target_increments
    )[:, 1:]
    target_energy_retention = canonical_rms / target_raw_rms.clamp_min(1.0e-8)
    required_radius_multiplier = torch.where(
        target_raw_rms > 0.0,
        target_raw_rms / radius_active.clamp_min(1.0e-8),
        torch.zeros_like(target_raw_rms),
    ).flatten()
    target_high_frequency_rms = gauge.phase_rms(
        diagnostics.target_unsmoothed_increments
        - diagnostics.target_raw_smoothed_increments
    )[:, 1:]
    target_unsmoothed_rms = gauge.phase_rms(
        diagnostics.target_unsmoothed_increments
    )[:, 1:]
    target_high_frequency_fraction = (
        target_high_frequency_rms / target_unsmoothed_rms.clamp_min(1.0e-8)
    )
    target_prior_cosine = (
        diagnostics.target_frozen_prior_cosine_per_phase[:, 1:].flatten()
    )
    value = {
        "method": METHOD_NAME,
        "schema_version": RECEIPT_SCHEMA,
        "rho": float(result.rho),
        "losses": {
            "total": float(result.total.detach().float().cpu().item()),
            "canonical": float(result.canonical.detach().float().cpu().item()),
            "executed": float(result.executed.detach().float().cpu().item()),
            "noop_preservation": float(
                result.noop_preservation.detach().float().cpu().item()
            ),
            "margin": float(result.margin.detach().float().cpu().item()),
            "temporal_jitter": float(
                result.temporal_jitter.detach().float().cpu().item()
            ),
        },
        "source_only_radius_mean_active": mean(radius_active),
        "radius_floor_dominated_fraction_active": mean(
            (
                radius_active
                <= float(gauge.DEFAULT_RADIUS_FLOOR)
                + 8.0 * torch.finfo(torch.float32).eps
            ).float()
        ),
        "target_clipped_fraction_active": mean((target_active < 1.0).float()),
        "target_projection_scale_mean_active": mean(target_active),
        "target_energy_retention_mean_active": mean(target_energy_retention),
        "target_required_radius_multiplier_p50": float(
            torch.quantile(required_radius_multiplier, 0.5).cpu().item()
        ),
        "target_required_radius_multiplier_p90": float(
            torch.quantile(required_radius_multiplier, 0.9).cpu().item()
        ),
        "target_required_radius_multiplier_max": float(
            required_radius_multiplier.max().cpu().item()
        ),
        "predicted_saturated_fraction_active": mean(
            (predicted_active < 1.0).float()
        ),
        "predicted_projection_scale_mean_active": mean(predicted_active),
        "frozen_noop_to_source_rms": mean(
            diagnostics.frozen_noop_to_source_rms_per_sample
        ),
        "frozen_noop_to_target_rms": mean(
            diagnostics.frozen_noop_to_target_rms_per_sample
        ),
        "frozen_noop_target_over_source_error_ratio": mean(
            diagnostics.frozen_noop_to_target_rms_per_sample
            / diagnostics.frozen_noop_to_source_rms_per_sample.clamp_min(1.0e-8)
        ),
        "target_frozen_prior_cosine_mean_active": mean(target_prior_cosine),
        "target_frozen_prior_cosine_p10_active": float(
            torch.quantile(target_prior_cosine, 0.1).cpu().item()
        ),
        "target_frozen_prior_positive_cosine_fraction_active": mean(
            (target_prior_cosine > 0.0).float()
        ),
        "target_high_frequency_fraction_mean_active": mean(
            target_high_frequency_fraction
        ),
        "motion_noop_full_gradient": diagnostics.motion_noop_full_gradient,
        "target_inside_deployment_radius": bool(
            torch.all(
                canonical_rms
                <= 0.95 * radius_active
                + 8.0 * torch.finfo(torch.float32).eps
            ).item()
        ),
    }
    return value


def immutable_objective_contract(
    config: FeasibleQuotientLossConfig = FeasibleQuotientLossConfig(),
) -> dict[str, Any]:
    config.validate()
    return {
        "method": METHOD_NAME,
        "schema_version": RECEIPT_SCHEMA,
        "forward_branch_order": list(FORWARD_BRANCH_ORDER),
        "forwards_per_candidate": len(FORWARD_BRANCH_ORDER),
        "graph_branches": list(GRAPH_BRANCHES),
        "loss_config": asdict(config),
        "adapted_quotient": "Q0(Atheta-Ntheta)",
        "adapted_quotient_gradient": "exact_shared_parameter_JA_minus_JN",
        "target_quotient": "Q0(target_clean-stopgrad(frozen_noop_section))",
        "target_section_reference": (
            "same_query_frozen_noop_prevents_beta1_double_count"
        ),
        "training_operator": "FIR(Dq)->zero_centered_radial_projection",
        "deployment_operator": "N0+rho*Integrate(Project_r(FIR(Dqtheta)))",
        "deployment_velocity_precision": (
            "fp32_exact_clean_transport_with_post_boundary_radius_certificate"
        ),
        "executed_loss_space": (
            "exact_deployment_lifted_clean_field_with_cumulative_source_radius"
        ),
        "source_only_radius": (
            "max(RMS(FIR(DQ0(A0-N0))),"
            "0.25*RMS(FIR(DQ0(N0))),1e-3)"
        ),
        "target_interior_ratio": float(config.target_interior_ratio),
        "falsification_diagnostics": [
            "frozen_noop_source_vs_target_error",
            "target_frozen_prior_phase_cosine",
            "target_temporal_high_frequency_fraction",
            "target_energy_retention_and_clipping",
        ],
        "paired_target_used_as_model_condition": False,
        "generator_forwards": 0,
        "first_frame_anchor": False,
        "forbidden_inference_conditions": list(FORBIDDEN_INFERENCE_CONDITIONS),
    }


__all__ = [
    "EXPECTED_PHASES",
    "FORBIDDEN_INFERENCE_CONDITIONS",
    "FORWARD_BRANCH_ORDER",
    "FeasibleQuotientDiagnostics",
    "FeasibleQuotientLossConfig",
    "FeasibleQuotientLossResult",
    "FeasibleQuotientObjectiveError",
    "FiveBranchCleanFields",
    "GRAPH_BRANCHES",
    "METHOD_NAME",
    "RECEIPT_SCHEMA",
    "compute_feasible_quotient_objective",
    "detached_receipt_diagnostics",
    "immutable_objective_contract",
]
