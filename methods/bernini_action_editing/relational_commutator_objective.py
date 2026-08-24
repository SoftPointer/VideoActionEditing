#!/usr/bin/env python3
"""Pure-tensor v7 objective for a bounded Bernini motion commutator.

The objective binds one optimizer candidate to exactly seven model forwards:

1. frozen MV2V negative (adapter off, graph free),
2. frozen MV2V semantic no-op (adapter off, graph free),
3. frozen MV2V requested action (adapter off, graph free),
4. adapted MV2V semantic no-op (adapter on, differentiable),
5. adapted MV2V requested action (adapter on, differentiable),
6. frozen T2V negative (adapter off, graph free), and
7. frozen T2V requested action (adapter off, graph free).

The integration layer must reconstruct all seven clean fields from one shared
noisy query and one diffusion timestep using the official mode-specific
guidance rules before calling this module.  Source and paired-target clean
latents are offline supervision, never inference conditions.

The default objective is deliberately *target-only*.  Its deployable student
is the source-aligned difference-of-differences from :mod:`motion_commutator`::

    B0      = Q0(frozen_action - frozen_noop)
    C_theta = Q0((adapted_action - adapted_noop)
                 - (frozen_action - frozen_noop))
    M       = Q0(target_clean - source_clean)
    C_star  = Q0(M - B0)

Training directly supervises the signed increments of raw ``C_theta`` against
raw ``C_star``.  This prevents an overlarge residual from hiding behind a
saturated projection and avoids optimizing through clip-scale gradients.  The
temporal-only filter and hard per-increment trust projection are deployment
operations.  During training they run on detached tensors solely to report
reachability/saturation diagnostics.  The immutable forty-step rho schedule
weights raw motion supervision; rho-zero steps retain only the all-sigma no-op
preservation anchor.

The T2V pair is always audited with coordinate-invariant temporal relation
statistics.  It has zero influence by default, so an ineligible T2V teacher
cannot block target-only training.  A nonzero relational weight explicitly
opts into an experimental auxiliary; that path fails closed whenever the
frozen relation gate is ineligible.  Passing that metric gate is not evidence
that the fixed kernel captures action semantics, so receipts say exactly that.
No pointwise T2V/paired-target coordinate cosine exists in this module.

PyTorch remains a lazy dependency to keep configuration and receipt contracts
inspectable outside the AUH runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any

try:  # Package import.
    from . import cross_mode_motion_kernel as cmkd
    from . import motion_commutator as commutator
except ImportError:  # Direct import with METHOD_ROOT on sys.path.
    import cross_mode_motion_kernel as cmkd
    import motion_commutator as commutator


METHOD_NAME = "bernini-relational-motion-commutator-objective-v7"
RECEIPT_SCHEMA = "bernini-relational-motion-commutator-objective-receipt-v7"
EXPECTED_PHASES = 21
FORWARD_BRANCH_ORDER = (
    "frozen_editor_negative_full_source",
    "frozen_editor_noop_full_source",
    "frozen_editor_action_full_source",
    "adapted_editor_noop_full_source",
    "adapted_editor_action_full_source",
    "frozen_generator_negative_target_only",
    "frozen_generator_action_target_only",
)
GRAPH_BRANCHES = (
    "adapted_editor_noop_full_source",
    "adapted_editor_action_full_source",
)
GRAPH_FREE_MODEL_BRANCHES = tuple(
    name for name in FORWARD_BRANCH_ORDER if name not in GRAPH_BRANCHES
)
RELATIONAL_METRIC_LIMITATION = (
    "coordinate-invariant fixed-kernel eligibility is an experimental metric, "
    "not proof of semantic action correspondence"
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


class RelationalCommutatorObjectiveError(RuntimeError):
    """Raised before an invalid v7 objective can create an optimizer update."""


class RelationalAuxiliaryIneligible(RelationalCommutatorObjectiveError):
    """Raised when an explicitly enabled experimental teacher fails its gate."""


@dataclass(frozen=True)
class RelationalCommutatorLossConfig:
    """Loss weights and immutable projection/gate configurations.

    ``relational_auxiliary_weight == 0`` is the production-safe target-only
    arm.  Increasing it is an explicit experimental opt-in and activates a
    fail-closed T2V relation gate at rho-positive steps.
    """

    raw_target_weight: float = 1.0
    noop_preservation_weight: float = 0.20
    residual_temporal_jitter_weight: float = 0.05
    relational_auxiliary_weight: float = 0.0
    charbonnier_epsilon: float = 1.0e-3
    normalization_floor: float = 1.0e-4
    commutator_config: commutator.MotionCommutatorConfig = field(
        default_factory=lambda: commutator.MotionCommutatorConfig(
            temporal_smoothing=True
        )
    )
    relational_kernel_config: cmkd.CrossModeMotionKernelConfig = field(
        default_factory=cmkd.CrossModeMotionKernelConfig
    )

    def validate(self) -> None:
        for name in (
            "raw_target_weight",
            "noop_preservation_weight",
            "residual_temporal_jitter_weight",
            "relational_auxiliary_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise RelationalCommutatorObjectiveError(
                    f"{name} must be finite and nonnegative"
                )
        if float(self.raw_target_weight) <= 0.0:
            raise RelationalCommutatorObjectiveError(
                "raw_target_weight must be strictly positive"
            )
        if float(self.noop_preservation_weight) <= 0.0:
            raise RelationalCommutatorObjectiveError(
                "noop_preservation_weight must be strictly positive"
            )
        for name in ("charbonnier_epsilon", "normalization_floor"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise RelationalCommutatorObjectiveError(
                    f"{name} must be finite and strictly positive"
                )
        try:
            self.commutator_config.validate()
            self.relational_kernel_config.validate()
        except (
            commutator.MotionCommutatorError,
            cmkd.CrossModeMotionKernelError,
        ) as error:
            raise RelationalCommutatorObjectiveError(str(error)) from error


@dataclass(frozen=True)
class SevenBranchCleanFields:
    """Clean-field boundary for the exact seven-forward training cell.

    Every tensor is float32 ``[B,21,S,D]`` on one device.  The two adapted
    editor fields must retain graphs; the five frozen model fields and both
    offline supervision fields must be graph free.  Editor fields are official
    APG-reconstructed fields.  ``frozen_generator_action`` is the official T2V
    CFG-reconstructed action field, while ``frozen_generator_negative`` is its
    negative clean field.
    """

    frozen_editor_negative: Any
    frozen_editor_noop: Any
    frozen_editor_action: Any
    adapted_editor_noop: Any
    adapted_editor_action: Any
    frozen_generator_negative: Any
    frozen_generator_action: Any
    source_clean: Any
    target_clean: Any


@dataclass(frozen=True)
class RelationalCommutatorDiagnostics:
    """Differentiable tensors and detached gate evidence for one candidate."""

    raw_commutator: Any
    deployment_projection: Any
    target_projection: Any
    predicted_execution: Any
    target_execution: Any
    target_motion_direction: Any
    unbounded_target_correction: Any
    teacher_direction: Any
    teacher_eligibility: Any
    target_motion_increment_rms: Any
    noop_reference_rms: Any
    raw_target_per_sample: Any
    noop_preservation_per_sample: Any
    residual_temporal_jitter_per_sample: Any
    relational_auxiliary_per_sample: Any
    relational_auxiliary_enabled: bool
    relational_auxiliary_active: bool
    deployment_config: commutator.MotionCommutatorConfig


@dataclass(frozen=True)
class RelationalCommutatorLossResult:
    """Scalar raw-objective losses plus detached deployment diagnostics."""

    total: Any
    raw_target: Any
    noop_preservation: Any
    residual_temporal_jitter: Any
    relational_auxiliary: Any
    rho: float
    diagnostics: RelationalCommutatorDiagnostics


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RelationalCommutatorObjectiveError(
            "v7 objective tensor operations require PyTorch"
        ) from error
    return torch


def _validate_fields(fields: SevenBranchCleanFields) -> None:
    torch = _require_torch()
    names = tuple(SevenBranchCleanFields.__dataclass_fields__)
    tensors = tuple(getattr(fields, name) for name in names)
    reference = tensors[0]
    if not isinstance(reference, torch.Tensor):
        raise RelationalCommutatorObjectiveError("clean fields must be tensors")
    if (
        reference.ndim != 4
        or int(reference.shape[0]) <= 0
        or int(reference.shape[1]) != EXPECTED_PHASES
        or int(reference.shape[2]) <= 0
        or int(reference.shape[3]) <= 0
        or reference.dtype != torch.float32
    ):
        raise RelationalCommutatorObjectiveError(
            "clean fields must be float32 tensors with exact shape [B,21,S,D]"
        )
    for name, tensor in zip(names, tensors):
        if not isinstance(tensor, torch.Tensor):
            raise RelationalCommutatorObjectiveError(f"{name} is not a tensor")
        if (
            tuple(tensor.shape) != tuple(reference.shape)
            or tensor.dtype != reference.dtype
            or tensor.device != reference.device
        ):
            raise RelationalCommutatorObjectiveError(
                f"{name} shape, dtype, or device differs"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise RelationalCommutatorObjectiveError(f"{name} is non-finite")

    graph_names = {"adapted_editor_noop", "adapted_editor_action"}
    for name in names:
        requires_grad = bool(getattr(fields, name).requires_grad)
        if name in graph_names and not requires_grad:
            raise RelationalCommutatorObjectiveError(
                f"{name} must retain its adapter graph"
            )
        if name not in graph_names and requires_grad:
            raise RelationalCommutatorObjectiveError(
                f"{name} must be adapter-off and graph free"
            )


def _causal_gauge(field: Any) -> Any:
    try:
        return commutator.causal_gauge(field)
    except commutator.MotionCommutatorError as error:
        raise RelationalCommutatorObjectiveError(str(error)) from error


def _per_sample_rms(value: Any, *, floor_value: float) -> Any:
    return value.square().mean(dim=tuple(range(1, value.ndim))).sqrt().clamp_min(
        float(floor_value)
    )


def _normalized_charbonnier_per_sample(
    error: Any,
    scale: Any,
    *,
    epsilon: float,
) -> Any:
    normalized = error / scale.reshape((-1,) + (1,) * (error.ndim - 1))
    return (
        (normalized.square() + float(epsilon) ** 2).sqrt() - float(epsilon)
    ).mean(dim=tuple(range(1, error.ndim)))


def _project_training_target(
    *,
    target_correction: Any,
    frozen_official_direction: Any,
    config: commutator.MotionCommutatorConfig,
) -> Any:
    """Project detached ``C_star`` with the deployable commutator operator."""

    try:
        return commutator.bound_motion_commutator_correction(
            frozen_official_direction,
            target_correction,
            config=config,
        )
    except commutator.MotionCommutatorError as error:
        raise RelationalCommutatorObjectiveError(str(error)) from error


def compute_relational_commutator_objective(
    fields: SevenBranchCleanFields,
    *,
    step_index: int,
    config: RelationalCommutatorLossConfig = RelationalCommutatorLossConfig(),
) -> RelationalCommutatorLossResult:
    """Compute the v7 target-only objective and optional relational auxiliary.

    The target and T2V branches are training-only.  The adapted no-op and
    action branches are the only graph-bearing model forwards.  At rho-zero
    steps, raw motion supervision is exactly inactive and therefore supplies
    no action-branch gradient; the all-sigma no-op anchor remains active by
    construction.
    """

    torch = _require_torch()
    config.validate()
    _validate_fields(fields)
    try:
        raw_predicted = commutator.build_raw_motion_commutator(
            fields.adapted_editor_action,
            fields.adapted_editor_noop,
            fields.frozen_editor_action,
            fields.frozen_editor_noop,
        )
    except commutator.MotionCommutatorError as error:
        raise RelationalCommutatorObjectiveError(str(error)) from error

    target_motion = _causal_gauge(fields.target_clean - fields.source_clean).detach()
    target_correction = commutator.build_target_correction(
        target_motion, raw_predicted.frozen_official_direction
    ).detach()
    try:
        rho = float(commutator.release_rho(step_index))
    except commutator.MotionCommutatorError as error:
        raise RelationalCommutatorObjectiveError(str(error)) from error

    target_increments = commutator.phase_increments(target_motion)[:, 1:]
    target_scale = _per_sample_rms(
        target_increments, floor_value=float(config.normalization_floor)
    ).detach()
    predicted_raw_increments = commutator.phase_increments(
        raw_predicted.raw_commutator_correction
    )[:, 1:]
    target_raw_increments = commutator.phase_increments(target_correction)[:, 1:]
    # No action branch enters the loss at exact replay steps.  This conditional
    # also makes the expected absence of an adapted-action gradient auditable.
    residual_increments = (
        rho * (predicted_raw_increments - target_raw_increments)
        if rho > 0.0
        else torch.zeros_like(target_raw_increments)
    )
    raw_target_per_sample = _normalized_charbonnier_per_sample(
        residual_increments,
        target_scale,
        epsilon=float(config.charbonnier_epsilon),
    )

    # Hard bounding is a deployment/reachability diagnostic, never the source
    # of target gradients.  Detaching here prevents a future integration from
    # accidentally training through clip saturation.
    with torch.no_grad():
        try:
            deployment_projection = commutator.bound_motion_commutator_correction(
                raw_predicted.frozen_official_direction.detach(),
                raw_predicted.raw_commutator_correction.detach(),
                config=config.commutator_config,
            )
            target_projection = _project_training_target(
                target_correction=target_correction,
                frozen_official_direction=(
                    raw_predicted.frozen_official_direction.detach()
                ),
                config=config.commutator_config,
            )
            predicted_execution = commutator.execute_motion_commutator(
                raw_predicted.frozen_official_direction.detach(),
                deployment_projection.bounded_commutator_correction,
                step_index=step_index,
            )
            target_execution = commutator.execute_motion_commutator(
                raw_predicted.frozen_official_direction.detach(),
                target_projection.bounded_commutator_correction,
                step_index=step_index,
            )
        except commutator.MotionCommutatorError as error:
            raise RelationalCommutatorObjectiveError(str(error)) from error

    noop_difference = fields.adapted_editor_noop - fields.frozen_editor_noop
    noop_scale = _per_sample_rms(
        fields.frozen_editor_noop,
        floor_value=float(config.normalization_floor),
    ).detach()
    noop_per_sample = _normalized_charbonnier_per_sample(
        noop_difference,
        noop_scale,
        epsilon=float(config.charbonnier_epsilon),
    )

    if int(residual_increments.shape[1]) < 2:
        raise RelationalCommutatorObjectiveError(
            "residual jitter requires at least two active increments"
        )
    residual_acceleration = (
        residual_increments[:, 1:] - residual_increments[:, :-1]
    )
    jitter_per_sample = _normalized_charbonnier_per_sample(
        residual_acceleration,
        target_scale,
        epsilon=float(config.charbonnier_epsilon),
    )

    teacher_direction = _causal_gauge(
        fields.frozen_generator_action - fields.frozen_generator_negative
    ).detach()
    try:
        eligibility = cmkd.evaluate_teacher_target_eligibility(
            teacher_direction,
            target_motion,
            config=config.relational_kernel_config,
        )
    except cmkd.CrossModeMotionKernelError as error:
        raise RelationalCommutatorObjectiveError(str(error)) from error

    relational_enabled = float(config.relational_auxiliary_weight) > 0.0
    relational_active = relational_enabled and rho > 0.0
    if relational_active and not bool(eligibility.eligible.all()):
        rejected = (~eligibility.eligible).nonzero(as_tuple=False).flatten().tolist()
        raise RelationalAuxiliaryIneligible(
            "experimental relational auxiliary was enabled but its fixed-kernel "
            f"gate rejected samples={rejected}"
        )
    if relational_active:
        # Only the invariant teacher-kernel term is retained.  Source-aligned
        # pointwise target, amplitude, and residual-jitter terms are already
        # defined explicitly above and must not be counted twice.
        relational_config = replace(
            config.relational_kernel_config,
            target_direction_weight=0.0,
            teacher_kernel_weight=1.0,
            amplitude_envelope_weight=0.0,
            temporal_jitter_weight=0.0,
        )
        try:
            relational_result = cmkd.cmkd_student_loss(
                raw_predicted.unbounded_final_direction,
                target_motion,
                teacher_direction,
                config=relational_config,
            )
        except cmkd.CrossModeMotionKernelError as error:
            raise RelationalCommutatorObjectiveError(str(error)) from error
        relational_per_sample = (
            relational_result.diagnostics.teacher_kernel_loss * rho
        )
    else:
        relational_per_sample = torch.zeros_like(raw_target_per_sample)

    raw_target_loss = raw_target_per_sample.mean()
    noop_loss = noop_per_sample.mean()
    jitter_loss = jitter_per_sample.mean()
    relational_loss = relational_per_sample.mean()
    total = (
        float(config.raw_target_weight) * raw_target_loss
        + float(config.noop_preservation_weight) * noop_loss
        + float(config.residual_temporal_jitter_weight) * jitter_loss
        + float(config.relational_auxiliary_weight) * relational_loss
    )
    for name, value in (
        ("raw target", raw_target_loss),
        ("noop preservation", noop_loss),
        ("residual temporal jitter", jitter_loss),
        ("relational auxiliary", relational_loss),
        ("total", total),
    ):
        if not bool(torch.isfinite(value)):
            raise RelationalCommutatorObjectiveError(f"{name} loss is non-finite")
    if not total.requires_grad:
        raise RelationalCommutatorObjectiveError(
            "v7 total loss lost both adapted-editor graphs"
        )

    return RelationalCommutatorLossResult(
        total=total,
        raw_target=raw_target_loss,
        noop_preservation=noop_loss,
        residual_temporal_jitter=jitter_loss,
        relational_auxiliary=relational_loss,
        rho=rho,
        diagnostics=RelationalCommutatorDiagnostics(
            raw_commutator=raw_predicted,
            deployment_projection=deployment_projection,
            target_projection=target_projection,
            predicted_execution=predicted_execution,
            target_execution=target_execution,
            target_motion_direction=target_motion,
            unbounded_target_correction=target_correction,
            teacher_direction=teacher_direction,
            teacher_eligibility=eligibility,
            target_motion_increment_rms=target_scale,
            noop_reference_rms=noop_scale,
            raw_target_per_sample=raw_target_per_sample,
            noop_preservation_per_sample=noop_per_sample,
            residual_temporal_jitter_per_sample=jitter_per_sample,
            relational_auxiliary_per_sample=relational_per_sample,
            relational_auxiliary_enabled=relational_enabled,
            relational_auxiliary_active=relational_active,
            deployment_config=config.commutator_config,
        ),
    )


def _linear_quantile(sorted_values: list[float], probability: float) -> float:
    """Return a finite linearly interpolated quantile of a non-empty list."""

    if not sorted_values:
        raise RelationalCommutatorObjectiveError(
            "target-required-kappa statistics require active phases"
        )
    if not 0.0 <= float(probability) <= 1.0:
        raise RelationalCommutatorObjectiveError("quantile probability is invalid")
    if any(not math.isfinite(value) or value < 0.0 for value in sorted_values):
        raise RelationalCommutatorObjectiveError(
            "target-required-kappa statistic is non-finite"
        )
    position = (len(sorted_values) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower]
        + fraction * (sorted_values[upper] - sorted_values[lower])
    )


def detached_receipt_diagnostics(
    result: RelationalCommutatorLossResult,
) -> dict[str, Any]:
    """Return JSON-ready evidence for one accepted optimizer candidate."""

    torch = _require_torch()
    diagnostics = result.diagnostics
    bound = diagnostics.deployment_projection.diagnostics
    target_bound = diagnostics.target_projection.diagnostics
    eligibility = diagnostics.teacher_eligibility

    def scalar_mean(value: Any) -> float:
        return float(value.detach().float().mean().cpu().item())

    def scalar_max(value: Any) -> float:
        return float(value.detach().float().max().cpu().item())

    active = bound.bound_scale[:, 1:]
    target_active = target_bound.bound_scale[:, 1:]
    deployment_config = diagnostics.deployment_config
    deployment_config.validate()
    kappa = float(deployment_config.max_correction_increment_ratio)
    absolute_floor = float(deployment_config.correction_increment_rms_floor)
    near_zero_threshold = max(
        float(deployment_config.epsilon),
        8.0 * float(torch.finfo(torch.float32).eps),
    )
    frozen_active = target_bound.frozen_increment_rms[:, 1:].detach().double().cpu()
    target_candidate_active = (
        target_bound.candidate_correction_increment_rms[:, 1:]
        .detach()
        .double()
        .cpu()
    )
    if (
        frozen_active.numel() == 0
        or tuple(frozen_active.shape) != tuple(target_candidate_active.shape)
        or not bool(torch.isfinite(frozen_active).all())
        or not bool(torch.isfinite(target_candidate_active).all())
    ):
        raise RelationalCommutatorObjectiveError(
            "target-required-kappa inputs are empty, mismatched, or non-finite"
        )
    # The absolute floor already realizes targets at or below it, so their
    # required relative multiplier is exactly zero.  Above the floor, the exact
    # minimum is target_rms/frozen_rms.  Strictly positive near-zero increments
    # are evaluated exactly in float64.  At exact zero we divide by a documented
    # positive threshold to keep receipt statistics finite; that value is only
    # a lower-bound proxy, and the phase is separately marked as unreachable by
    # any finite kappa.
    comparison_tolerance = max(
        float(deployment_config.epsilon),
        8.0 * float(torch.finfo(torch.float32).eps),
    )
    floor_sufficient = target_candidate_active <= (
        absolute_floor + comparison_tolerance
    )
    frozen_near_zero = frozen_active <= near_zero_threshold
    exact_zero_unreachable = (frozen_active == 0.0) & (~floor_sufficient)
    near_zero_proxy = exact_zero_unreachable
    positive_denominator_or_proxy = torch.where(
        frozen_active > 0.0,
        frozen_active,
        torch.full_like(frozen_active, near_zero_threshold),
    )
    required_kappa = torch.where(
        floor_sufficient,
        torch.zeros_like(target_candidate_active),
        target_candidate_active / positive_denominator_or_proxy,
    )
    if not bool(torch.isfinite(required_kappa).all()):
        raise RelationalCommutatorObjectiveError(
            "target-required-kappa proxy became non-finite"
        )
    required_values = sorted(float(value) for value in required_kappa.flatten())
    required_median = _linear_quantile(required_values, 0.50)
    required_p90 = _linear_quantile(required_values, 0.90)
    required_max = _linear_quantile(required_values, 1.00)
    floor_dominated = absolute_floor >= kappa * frozen_active
    bound_violation = (
        bound.bounded_correction_increment_rms
        - bound.correction_increment_rms_cap
    ).clamp_min(0.0)
    value = {
        "method": METHOD_NAME,
        "schema_version": RECEIPT_SCHEMA,
        "forward_branch_order": list(FORWARD_BRANCH_ORDER),
        "model_forwards_per_candidate": len(FORWARD_BRANCH_ORDER),
        "graph_branches": list(GRAPH_BRANCHES),
        "graph_forwards_per_candidate": len(GRAPH_BRANCHES),
        "graph_free_model_branches": list(GRAPH_FREE_MODEL_BRANCHES),
        "rho": float(result.rho),
        "hard_bound_placement": (
            "deployment only: detached raw commutator -> optional temporal-only "
            "FIR -> per-phase increment projection -> rho release"
        ),
        "training_target_placement": (
            "rho-weighted raw signed commutator increments before smoothing/bound"
        ),
        "hard_bound_contributes_target_gradient": False,
        "target_source_alignment": "Q0(target_clean-source_clean)",
        "target_projection_matches_inference_operator": True,
        "noop_preservation_all_sigmas": True,
        "explicit_all_sigma_detail_loss": False,
        "explicit_common_mode_loss": False,
        "common_mode_control": (
            "exact action/noop commutator cancellation plus all-sigma adapted-noop "
            "anchor; no separately weighted common-mode term"
        ),
        "losses": {
            "total": float(result.total.detach().float().cpu().item()),
            "raw_target": float(
                result.raw_target.detach().float().cpu().item()
            ),
            "noop_preservation": float(
                result.noop_preservation.detach().float().cpu().item()
            ),
            "residual_temporal_jitter": float(
                result.residual_temporal_jitter.detach().float().cpu().item()
            ),
            "relational_auxiliary": float(
                result.relational_auxiliary.detach().float().cpu().item()
            ),
        },
        "commutator_bound": {
            "mean_scale_active": scalar_mean(active),
            "saturated_fraction_active": scalar_mean((active < 1.0).float()),
            "max_postprojection_violation": scalar_max(bound_violation),
            "floor_dominated_fraction_active": scalar_mean(
                floor_dominated.float()
            ),
            "target_mean_scale_active": scalar_mean(target_active),
            "target_bound_mean_scale_active": scalar_mean(target_active),
            "target_saturated_fraction_active": scalar_mean(
                (target_active < 1.0).float()
            ),
            "target_floor_sufficient_fraction_active": scalar_mean(
                floor_sufficient.float()
            ),
            "target_required_kappa_median": required_median,
            "target_required_kappa_p90": required_p90,
            "target_required_kappa_max": required_max,
            "target_required_kappa_near_zero_threshold": near_zero_threshold,
            "frozen_increment_near_zero_fraction_active": scalar_mean(
                frozen_near_zero.float()
            ),
            "target_required_kappa_near_zero_proxy_fraction_active": scalar_mean(
                near_zero_proxy.float()
            ),
            "target_required_kappa_exact_zero_unreachable_fraction_active": (
                scalar_mean(exact_zero_unreachable.float())
            ),
            "target_required_kappa_definition": (
                "0 when target candidate RMS is within the absolute floor; "
                "otherwise target candidate RMS / frozen increment RMS"
            ),
            "target_required_kappa_near_zero_handling": (
                "strictly positive near-zero denominators use exact float64 "
                "division; exact zero uses the recorded threshold as a finite "
                "lower-bound proxy and is separately counted as unreachable"
            ),
        },
        "relational_auxiliary": {
            "enabled": diagnostics.relational_auxiliary_enabled,
            "active_at_this_rho": diagnostics.relational_auxiliary_active,
            "all_samples_eligible": bool(eligibility.eligible.all().item()),
            "eligible_count": int(eligibility.eligible.sum().detach().cpu().item()),
            "sample_count": int(eligibility.eligible.numel()),
            "centered_kernel_alignment_mean": scalar_mean(
                eligibility.centered_kernel_alignment
            ),
            "off_diagonal_teacher_rms_mean": scalar_mean(
                eligibility.teacher.off_diagonal_relational_rms
            ),
            "off_diagonal_target_rms_mean": scalar_mean(
                eligibility.target.off_diagonal_relational_rms
            ),
            "envelope_cosine_mean": scalar_mean(eligibility.envelope_cosine),
            "limitation": RELATIONAL_METRIC_LIMITATION,
            "cross_video_pointwise_coordinate_cosine": False,
        },
        "inference_conditions": ["source_video", "action_instruction"],
        "forbidden_inference_conditions": list(FORBIDDEN_INFERENCE_CONDITIONS),
    }
    # Defensive JSON-scalar audit: tensors must not escape this boundary.
    if any(isinstance(item, torch.Tensor) for item in value.values()):
        raise RelationalCommutatorObjectiveError(
            "receipt diagnostics leaked a tensor"
        )
    return value


def immutable_objective_contract(
    config: RelationalCommutatorLossConfig = RelationalCommutatorLossConfig(),
) -> dict[str, Any]:
    """Return configuration/branch metadata suitable for an immutable receipt."""

    config.validate()
    return {
        "method": METHOD_NAME,
        "schema_version": RECEIPT_SCHEMA,
        "latent_phases": EXPECTED_PHASES,
        "forward_branch_order": list(FORWARD_BRANCH_ORDER),
        "model_forwards_per_candidate": len(FORWARD_BRANCH_ORDER),
        "graph_branches": list(GRAPH_BRANCHES),
        "graph_forwards_per_candidate": len(GRAPH_BRANCHES),
        "loss_config": asdict(config),
        "target_motion": "Q0(target_clean-source_clean)",
        "student_correction": (
            "Q0((adapted_action-adapted_noop)-(frozen_action-frozen_noop))"
        ),
        "target_only_default": float(config.relational_auxiliary_weight) == 0.0,
        "raw_commutator_supervision_before_deployment_bound": True,
        "hard_bound_contributes_target_gradient": False,
        "deployment_diagnostics": {
            "target_bound_mean_scale_active": True,
            "floor_dominated_fraction_active": True,
            "target_required_kappa_statistics": ["median", "p90", "max"],
            "target_required_kappa_near_zero_handling": (
                "exact float64 division for positive near-zero increments plus "
                "finite threshold proxy and unreachable fraction at exact zero"
            ),
        },
        "explicit_all_sigma_detail_loss": False,
        "explicit_common_mode_loss": False,
        "common_mode_control": (
            "exact action/noop commutator cancellation plus all-sigma adapted-noop "
            "anchor; no separately weighted common-mode term"
        ),
        "relational_auxiliary_fail_closed_when_enabled": True,
        "relational_metric_limitation": RELATIONAL_METRIC_LIMITATION,
        "cross_video_pointwise_coordinate_cosine": False,
        "inference_conditions": ["source_video", "action_instruction"],
        "forbidden_inference_conditions": list(FORBIDDEN_INFERENCE_CONDITIONS),
    }


__all__ = [
    "EXPECTED_PHASES",
    "FORBIDDEN_INFERENCE_CONDITIONS",
    "FORWARD_BRANCH_ORDER",
    "GRAPH_BRANCHES",
    "GRAPH_FREE_MODEL_BRANCHES",
    "METHOD_NAME",
    "RECEIPT_SCHEMA",
    "RELATIONAL_METRIC_LIMITATION",
    "RelationalAuxiliaryIneligible",
    "RelationalCommutatorDiagnostics",
    "RelationalCommutatorLossConfig",
    "RelationalCommutatorLossResult",
    "RelationalCommutatorObjectiveError",
    "SevenBranchCleanFields",
    "compute_relational_commutator_objective",
    "detached_receipt_diagnostics",
    "immutable_objective_contract",
]
