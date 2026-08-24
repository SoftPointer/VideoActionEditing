#!/usr/bin/env python3
"""Source-owned preservation arm for action-representation training.

The old replay/phase-0/outside losses compare zero-initialized residuals with
zero and can therefore have a structurally zero gradient.  This primitive
instead treats the current motion/middle output as a *detached disturbance*
which an independently-owned source-copy branch must correct outside the
detached action tangent.

No target RGB, target latent, clean target, or absolute target hidden state is
accepted by this API.  The only authorities are a frozen source-route hidden
state, a frozen source prefix, and a detached action representation.

Gradient ownership is deliberately asymmetric:

* the action view detaches the source-copy delta, so action loss can update
  motion/middle parameters but cannot teach the source-copy branch;
* the preservation view detaches the motion/middle delta, so preservation can
  update source-copy parameters but cannot suppress the action branch;
* the inference view composes both branches without a detach.

At exact zero initialization the preservation scalar and its gradient may be
zero.  After the first action update, ``after_first_action_update`` admission
requires a nonzero protected disturbance and a nonzero source-copy gradient;
otherwise TP fails closed rather than claiming preservation from a zero loss.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bernini-action-repr-source-preservation-v1"
GRADIENT_EXPECTATIONS = (
    "initial_step0",
    "after_first_action_update",
    "steady_state",
)


class SourcePreservationError(RuntimeError):
    """Raised before an invalid preservation gradient can be applied."""


def _fail(message: str) -> None:
    raise SourcePreservationError(message)


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependent
        raise SourcePreservationError("source preservation requires PyTorch") from error
    return torch


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        _fail(f"source-preservation contract is not canonical JSON: {error}")


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class SourcePreservationConfig:
    action_tangent_weight: float = 1.0
    outside_route_weight: float = 1.0
    phase0_weight: float = 1.0
    prefix_coordinate_weight: float = 0.25
    huber_delta: float = 0.10
    norm_floor: float = 1.0e-6
    first_update_signal_floor: float = 1.0e-12
    maximum_source_copy_gradient_norm: float = 1.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                _fail(f"{name} must be finite and strictly positive")


@dataclass(frozen=True)
class SourcePreservationInputs:
    """Four-dimensional source-coordinate fields with disjoint student arms."""

    detached_frozen_route_hidden: Any
    detached_source_prefix: Any
    detached_action_representation: Any
    student_motion_middle_delta: Any
    student_source_copy_delta: Any
    action_activity: Any


@dataclass(frozen=True)
class OwnedRouteViews:
    action: Any
    preservation: Any
    inference: Any


@dataclass(frozen=True)
class SourcePreservationResult:
    loss: Any
    components: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    views: OwnedRouteViews


def preservation_contract(
    config: SourcePreservationConfig = SourcePreservationConfig(),
) -> dict[str, Any]:
    config.validate()
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_authorities": [
            "detached_frozen_source_route_hidden",
            "detached_frozen_source_prefix",
        ],
        "action_authority": "detached_source_aligned_action_representation",
        "target_rgb_accepted": False,
        "target_vae_or_clean_latent_accepted": False,
        "target_absolute_hidden_or_value_accepted": False,
        "motion_middle_gradient_owner": "action_arm_only",
        "source_copy_gradient_owner": "preservation_arm_only",
        "preservation_motion_policy": "detached_observed_disturbance",
        "editable_subspace": "per_token_detached_action_tangent",
        "protected_subspace": (
            "action_tangent_orthogonal_complement_plus_full_inactive_and_phase0"
        ),
        "step0_zero_preservation_allowed": True,
        "after_first_action_update_nonzero_source_copy_gradient_required": True,
        "weighted_action_preservation_compensation": False,
        "config": asdict(config),
    }
    value["contract_digest"] = _object_sha256(value)
    return value


def _validate_route_field(
    value: Any, *, label: str, requires_grad: bool
) -> None:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 4
        or not bool(value.is_floating_point())
        or any(int(size) <= 0 for size in value.shape)
        or int(value.shape[1]) < 3
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        _fail(f"{label} must be finite floating [B,P>=3,N,D]")
    if bool(value.requires_grad) is not requires_grad:
        _fail(f"{label} requires_grad differs")


def _storage_identity(value: Any) -> tuple[str, int]:
    try:
        return (str(value.device), int(value.untyped_storage().data_ptr()))
    except AttributeError:  # pragma: no cover - old torch fallback
        return (str(value.device), int(value.storage().data_ptr()))


def _validate_inputs(inputs: SourcePreservationInputs) -> Any:
    torch = _torch()
    _validate_route_field(
        inputs.detached_frozen_route_hidden,
        label="detached_frozen_route_hidden",
        requires_grad=False,
    )
    _validate_route_field(
        inputs.detached_action_representation,
        label="detached_action_representation",
        requires_grad=False,
    )
    _validate_route_field(
        inputs.student_motion_middle_delta,
        label="student_motion_middle_delta",
        requires_grad=True,
    )
    _validate_route_field(
        inputs.student_source_copy_delta,
        label="student_source_copy_delta",
        requires_grad=True,
    )
    shape = tuple(inputs.detached_frozen_route_hidden.shape)
    device = inputs.detached_frozen_route_hidden.device
    for label, value in (
        ("detached_action_representation", inputs.detached_action_representation),
        ("student_motion_middle_delta", inputs.student_motion_middle_delta),
        ("student_source_copy_delta", inputs.student_source_copy_delta),
    ):
        if tuple(value.shape) != shape or value.device != device:
            _fail(f"{label} geometry/device differs from frozen source route")
    prefix = inputs.detached_source_prefix
    if (
        not isinstance(prefix, torch.Tensor)
        or prefix.ndim != 3
        or not bool(prefix.is_floating_point())
        or int(prefix.shape[0]) != int(shape[0])
        or int(prefix.shape[1]) <= 0
        or int(prefix.shape[2]) != int(shape[3])
        or prefix.device != device
        or bool(prefix.requires_grad)
        or not bool(torch.isfinite(prefix).all().item())
    ):
        _fail("detached_source_prefix must be detached finite [B,K,D]")
    if _storage_identity(inputs.student_motion_middle_delta) == _storage_identity(
        inputs.student_source_copy_delta
    ):
        _fail("motion/middle and source-copy fields must not share storage")
    activity = inputs.action_activity
    if not isinstance(activity, torch.Tensor) or activity.dtype != torch.bool:
        _fail("action_activity must be a bool tensor")
    if activity.ndim == 3:
        activity = activity.unsqueeze(-1)
    if (
        activity.ndim != 4
        or tuple(activity.shape[:3]) != tuple(shape[:3])
        or int(activity.shape[3]) != 1
        or activity.device != device
    ):
        _fail("action_activity must have shape [B,P,N] or [B,P,N,1]")
    if bool(activity[:, 0].any().item()):
        _fail("action_activity phase0 must be entirely inactive")
    if not bool(activity[:, 1:].any().item()):
        _fail("action_activity must contain post-phase0 action tokens")
    if not bool((~activity[:, 1:]).any().item()):
        _fail("action_activity must contain post-phase0 protected tokens")
    action = inputs.detached_action_representation.float()
    active_action_energy = action.square().sum(dim=-1, keepdim=True)
    if not bool((active_action_energy.masked_select(activity) > 0.0).any().item()):
        _fail("detached action representation is zero on every active token")
    if bool(torch.count_nonzero(action[:, 0]).item()):
        _fail("detached action representation phase0 must be exact zero")
    return activity


def build_owned_route_views(inputs: SourcePreservationInputs) -> OwnedRouteViews:
    """Compose three views which encode the two disjoint gradient owners."""

    _validate_inputs(inputs)
    frozen = inputs.detached_frozen_route_hidden.detach()
    motion = inputs.student_motion_middle_delta
    source_copy = inputs.student_source_copy_delta
    return OwnedRouteViews(
        action=frozen + source_copy.detach() + motion,
        preservation=frozen + source_copy + motion.detach(),
        inference=frozen + source_copy + motion,
    )


def _masked_mean(value: Any, mask: Any) -> Any:
    expanded = mask.expand_as(value)
    if not bool(expanded.any().item()):
        return value.sum().mul(0.0)
    return value.masked_select(expanded).mean()


def compute_source_preservation(
    inputs: SourcePreservationInputs,
    config: SourcePreservationConfig = SourcePreservationConfig(),
) -> SourcePreservationResult:
    """Build a source-only preservation scalar without performing backward."""

    torch = _torch()
    config.validate()
    activity = _validate_inputs(inputs)
    views = build_owned_route_views(inputs)

    # Only the source-copy field retains a preservation gradient.  Motion is
    # an observed disturbance, never a preservation target or gradient sink.
    disturbance = (
        inputs.student_source_copy_delta.float()
        + inputs.student_motion_middle_delta.detach().float()
    )
    action = inputs.detached_action_representation.detach().float()
    norm2 = action.square().sum(dim=-1, keepdim=True)
    tangent_valid = activity & (norm2 > float(config.norm_floor) ** 2)
    tangent = (
        (disturbance * action).sum(dim=-1, keepdim=True)
        / norm2.clamp_min(float(config.norm_floor) ** 2)
    ) * action
    tangent = torch.where(tangent_valid, tangent, torch.zeros_like(tangent))
    protected = disturbance - tangent

    active_orthogonal = _masked_mean(protected.square(), activity)
    phase0 = protected[:, :1].square().mean()
    outside_mask = (~activity).clone()
    outside_mask[:, 0] = False
    outside = _masked_mean(protected.square(), outside_mask)

    frozen = inputs.detached_frozen_route_hidden.detach().float()
    prefix = inputs.detached_source_prefix.detach().float()
    prefix_unit = prefix / prefix.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(
        float(config.norm_floor)
    )
    frozen_unit = frozen / frozen.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(
        float(config.norm_floor)
    )
    protected_state = frozen + protected
    protected_unit = protected_state / protected_state.square().sum(
        dim=-1, keepdim=True
    ).sqrt().clamp_min(float(config.norm_floor))
    frozen_coordinates = torch.einsum("bpnd,bkd->bpnk", frozen_unit, prefix_unit)
    protected_coordinates = torch.einsum(
        "bpnd,bkd->bpnk", protected_unit, prefix_unit
    )
    prefix_coordinate = torch.nn.functional.huber_loss(
        protected_coordinates,
        frozen_coordinates,
        delta=float(config.huber_delta),
        reduction="mean",
    )

    components = {
        "active_action_tangent_orthogonal_tether": active_orthogonal,
        "outside_route_source_tether": outside,
        "phase0_source_tether": phase0,
        "source_prefix_coordinate_tether": prefix_coordinate,
    }
    loss = (
        float(config.action_tangent_weight) * active_orthogonal
        + float(config.outside_route_weight) * outside
        + float(config.phase0_weight) * phase0
        + float(config.prefix_coordinate_weight) * prefix_coordinate
    )
    if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
        _fail("source preservation produced a non-finite scalar")
    protected_rms = protected.square().mean().sqrt()
    motion_rms = inputs.student_motion_middle_delta.detach().float().square().mean().sqrt()
    return SourcePreservationResult(
        loss=loss,
        components=components,
        diagnostics={
            "motion_disturbance_rms": motion_rms.detach(),
            "protected_disturbance_rms": protected_rms.detach(),
            "active_tangent_token_count": tangent_valid.sum().detach(),
            "active_token_count": activity.sum().detach(),
            "outside_post_phase0_token_count": outside_mask.sum().detach(),
            "motion_detached_in_preservation": True,
            "source_copy_detached_in_action_view": True,
        },
        views=views,
    )


def _parameter_tuple(
    values: Sequence[Any], *, label: str
) -> tuple[Any, ...]:
    torch = _torch()
    result = tuple(values)
    if (
        not result
        or len({id(value) for value in result}) != len(result)
        or any(
            not isinstance(value, torch.nn.Parameter) or not value.requires_grad
            for value in result
        )
    ):
        _fail(f"{label} must be a non-empty unique trainable Parameter closure")
    return result


def backward_source_copy_only(
    result: SourcePreservationResult,
    *,
    source_copy_parameters: Sequence[Any],
    motion_middle_parameters: Sequence[Any],
    expectation: str,
    config: SourcePreservationConfig = SourcePreservationConfig(),
) -> Mapping[str, Any]:
    """Assign preservation gradients only to the source-copy parameter set.

    ``after_first_action_update`` is the TP admission gate: a zero signal, a
    zero scalar, or a zero source-copy gradient is rejected.  ``steady_state``
    permits convergence back to zero after preservation has been established.
    Existing motion/middle ``.grad`` values are left byte-for-byte untouched.
    """

    torch = _torch()
    config.validate()
    if expectation not in GRADIENT_EXPECTATIONS:
        _fail(f"unknown preservation gradient expectation: {expectation}")
    source_params = _parameter_tuple(
        source_copy_parameters, label="source_copy_parameters"
    )
    motion_params = _parameter_tuple(
        motion_middle_parameters, label="motion_middle_parameters"
    )
    if {id(value) for value in source_params} & {id(value) for value in motion_params}:
        _fail("source-copy and motion/middle parameter closures overlap")
    if (
        not isinstance(result.loss, torch.Tensor)
        or result.loss.ndim != 0
        or not result.loss.requires_grad
    ):
        _fail("source preservation loss must be a differentiable scalar")

    preserved_motion_grads = tuple(
        None if value.grad is None else value.grad.detach().clone()
        for value in motion_params
    )
    motion_leaks = torch.autograd.grad(
        result.loss,
        motion_params,
        retain_graph=True,
        allow_unused=True,
    )
    if any(
        gradient is not None and bool(torch.count_nonzero(gradient).item())
        for gradient in motion_leaks
    ):
        _fail("preservation loss leaked gradient into motion/middle parameters")

    source_raw = torch.autograd.grad(
        result.loss,
        source_params,
        retain_graph=False,
        allow_unused=True,
    )
    source = tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(source_params, source_raw)
    )
    if any(not bool(torch.isfinite(value).all().item()) for value in source):
        _fail("source-copy preservation gradient is non-finite")
    source_norm = sum(value.float().square().sum() for value in source).sqrt()
    motion_signal = float(result.diagnostics["motion_disturbance_rms"].item())
    protected_signal = float(result.diagnostics["protected_disturbance_rms"].item())
    loss_value = float(result.loss.detach().item())
    floor = float(config.first_update_signal_floor)
    if expectation == "after_first_action_update":
        if motion_signal <= floor:
            _fail("first action update produced no observable motion disturbance")
        if protected_signal <= floor or loss_value <= floor:
            _fail("first action update produced no nontrivial preservation signal")
        if float(source_norm.detach().item()) <= floor:
            _fail("first action update produced no source-copy preservation gradient")

    clip_scale = torch.minimum(
        source_norm.new_ones(()),
        source_norm.new_tensor(float(config.maximum_source_copy_gradient_norm))
        / source_norm.clamp_min(float(config.first_update_signal_floor)),
    )
    for parameter, gradient in zip(source_params, source):
        parameter.grad = (gradient * clip_scale.to(gradient.dtype)).detach().clone()
    for parameter, before in zip(motion_params, preserved_motion_grads):
        if before is None:
            if parameter.grad is not None:
                _fail("preservation backward materialized a motion/middle .grad")
        elif parameter.grad is None or not torch.equal(parameter.grad, before):
            _fail("preservation backward modified an existing motion/middle .grad")
    return {
        "expectation": expectation,
        "preservation_loss": result.loss.detach(),
        "motion_disturbance_rms": result.diagnostics["motion_disturbance_rms"],
        "protected_disturbance_rms": result.diagnostics[
            "protected_disturbance_rms"
        ],
        "source_copy_gradient_norm_preclip": source_norm.detach(),
        "source_copy_gradient_nonzero": bool(
            float(source_norm.detach().item()) > floor
        ),
        "source_copy_gradient_clip_scale": clip_scale.detach(),
        "motion_middle_gradient_leak": False,
        "motion_middle_existing_grad_preserved": True,
        "establishes_tp_preservation": (
            expectation == "after_first_action_update"
            and protected_signal > floor
            and float(source_norm.detach().item()) > floor
        ),
    }


__all__ = [
    "GRADIENT_EXPECTATIONS",
    "OwnedRouteViews",
    "SCHEMA_VERSION",
    "SourcePreservationConfig",
    "SourcePreservationError",
    "SourcePreservationInputs",
    "SourcePreservationResult",
    "backward_source_copy_only",
    "build_owned_route_views",
    "compute_source_preservation",
    "preservation_contract",
]
