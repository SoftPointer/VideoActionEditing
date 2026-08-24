"""Pure-tensor preference objective for Bernini DCLR action editing.

This module is deliberately smaller than a trainer.  It owns the numerical
contracts that must remain true between rollout collection and a LoRA update:

* a winner/loser pair is queried with one detached FP32 physical sigma and one
  shared detached FP32 epsilon;
* rectified-flow states are exactly
  ``x_sigma = (1 - sigma) * y + sigma * epsilon`` and
  ``v_star = epsilon - y``;
* current-policy and frozen collection-policy energies are FP32 MSEs over the
  target tail only; and
* the loss is the reference-corrected energy preference
  ``softplus(-beta * Delta)``, where
  ``Delta = (ell_theta(y-) - ell_theta(y+))
           - (ell_ref(y-) - ell_ref(y+))``.

The energy is a denoising-error training surrogate, not a video likelihood.
This file performs no sampling, model forward, distributed collective, or
optimizer step.  The caller must patchify each ``v_star`` with the same pinned
Bernini layout used for the corresponding model prediction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

try:  # Package import.
    from . import dclr_runtime_contract as runtime_contract
except ImportError:  # Direct import with METHOD_ROOT on sys.path.
    import dclr_runtime_contract as runtime_contract


SCHEMA_VERSION = "bernini-dclr-preference-objective-v1"
ACTION_NEARMISS = "action_nearmiss"
PRESERVATION_NEARMISS = "preservation_nearmiss"
ACTION_ADAPTER = "action_adapter"
IDENTITY_ADAPTER = "identity_adapter"


class DCLRPreferenceObjectiveError(RuntimeError):
    """A preference state, energy, loss, or adapter route is invalid."""


@dataclass(frozen=True)
class SharedPairFlowState:
    """One exact shared-noise rectified-flow query for ``y+`` and ``y-``.

    Every tensor is detached FP32.  ``sigma`` and ``timestep`` have shape
    ``[1]``.  Candidate tensors use a single-candidate batch-first layout and
    all share the same shape, device, and representation.
    """

    sigma: torch.Tensor
    timestep: torch.Tensor
    epsilon: torch.Tensor
    winner_clean: torch.Tensor
    loser_clean: torch.Tensor
    winner_x_sigma: torch.Tensor
    loser_x_sigma: torch.Tensor
    winner_true_velocity: torch.Tensor
    loser_true_velocity: torch.Tensor


@dataclass(frozen=True)
class CandidatePolicyEnergies:
    """Target-tail denoising energies for one candidate.

    ``current`` retains the active adapter graph.  ``reference`` is detached
    and corresponds to the policy revision that collected this pair.
    """

    current: torch.Tensor
    reference: torch.Tensor


@dataclass(frozen=True)
class ReferenceCorrectedDPO:
    """Auditable components of one scalar reference-corrected preference."""

    loss: torch.Tensor
    delta: torch.Tensor
    current_margin: torch.Tensor
    reference_margin: torch.Tensor
    beta: torch.Tensor


@dataclass(frozen=True)
class OneSidedNearMissRoute:
    """The only adapter family authorized to receive this pair's gradient."""

    pair_type: str
    active_adapter: str
    action_adapter_trainable: bool
    identity_adapter_trainable: bool
    loser_failed_axis: str


@dataclass(frozen=True)
class RoutedPreferenceObjective:
    """A validated near-miss route paired with its scalar DPO objective."""

    route: OneSidedNearMissRoute
    objective: ReferenceCorrectedDPO


def _require_tensor(value: Any, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise DCLRPreferenceObjectiveError(f"{label} must be a torch.Tensor")
    if value.device.type == "meta":
        raise DCLRPreferenceObjectiveError(f"{label} cannot be a meta tensor")
    return value


def _require_finite_float_tensor(
    value: Any,
    *,
    label: str,
) -> torch.Tensor:
    tensor = _require_tensor(value, label=label)
    if not tensor.is_floating_point():
        raise DCLRPreferenceObjectiveError(
            f"{label} must be a floating-point tensor"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise DCLRPreferenceObjectiveError(f"{label} contains NaN or infinity")
    return tensor


def _require_detached_fp32(
    value: Any,
    *,
    label: str,
) -> torch.Tensor:
    tensor = _require_finite_float_tensor(value, label=label)
    if tensor.dtype != torch.float32:
        raise DCLRPreferenceObjectiveError(f"{label} must be exact FP32")
    if tensor.requires_grad or tensor.grad_fn is not None:
        raise DCLRPreferenceObjectiveError(
            f"{label} must be detached from every policy graph"
        )
    return tensor


def _require_same_representation(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    left_label: str,
    right_label: str,
) -> None:
    if (
        tuple(left.shape) != tuple(right.shape)
        or left.dtype != right.dtype
        or left.device != right.device
        or left.layout != right.layout
    ):
        raise DCLRPreferenceObjectiveError(
            f"{left_label} and {right_label} shape/dtype/device/layout differ"
        )


def build_shared_pair_flow_state(
    winner_clean: torch.Tensor,
    loser_clean: torch.Tensor,
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
) -> SharedPairFlowState:
    """Build a pair state without sampling or silently casting anything.

    A single shared epsilon is mandatory.  Accepting separate epsilon tensors
    would confound the winner/loser energy margin with noise variance, so this
    API intentionally has only one ``epsilon`` argument.
    """

    winner = _require_detached_fp32(winner_clean, label="winner_clean")
    if winner.ndim < 2 or int(winner.shape[0]) != 1 or winner.numel() == 0:
        raise DCLRPreferenceObjectiveError(
            "winner_clean must have nonempty single-candidate [1,...] layout"
        )
    loser = _require_detached_fp32(loser_clean, label="loser_clean")
    shared_epsilon = _require_detached_fp32(epsilon, label="epsilon")
    _require_same_representation(
        winner,
        loser,
        left_label="winner_clean",
        right_label="loser_clean",
    )
    _require_same_representation(
        winner,
        shared_epsilon,
        left_label="winner_clean",
        right_label="epsilon",
    )

    physical_sigma = _require_detached_fp32(sigma, label="sigma")
    if tuple(physical_sigma.shape) != (1,):
        raise DCLRPreferenceObjectiveError(
            "sigma must be one exact FP32 physical sigma with shape [1]"
        )
    if physical_sigma.device != winner.device:
        raise DCLRPreferenceObjectiveError(
            "sigma device differs from candidate/epsilon device"
        )
    try:
        timestep = runtime_contract.fp32_sigma_to_timestep(physical_sigma)
    except runtime_contract.DCLRRuntimeContractError as error:
        raise DCLRPreferenceObjectiveError(str(error)) from error

    broadcast_sigma = physical_sigma.reshape(
        1, *([1] * (winner.ndim - 1))
    )
    one = torch.ones_like(broadcast_sigma)
    winner_x_sigma = (
        (one - broadcast_sigma) * winner
        + broadcast_sigma * shared_epsilon
    )
    loser_x_sigma = (
        (one - broadcast_sigma) * loser
        + broadcast_sigma * shared_epsilon
    )
    winner_true_velocity = shared_epsilon - winner
    loser_true_velocity = shared_epsilon - loser

    outputs = (
        winner_x_sigma,
        loser_x_sigma,
        winner_true_velocity,
        loser_true_velocity,
    )
    if any(
        tensor.dtype != torch.float32
        or tensor.requires_grad
        or tensor.grad_fn is not None
        or not bool(torch.isfinite(tensor).all().item())
        for tensor in outputs
    ):
        raise DCLRPreferenceObjectiveError(
            "shared pair state did not remain finite detached FP32"
        )
    return SharedPairFlowState(
        sigma=physical_sigma,
        timestep=timestep,
        epsilon=shared_epsilon,
        winner_clean=winner,
        loser_clean=loser,
        winner_x_sigma=winner_x_sigma,
        loser_x_sigma=loser_x_sigma,
        winner_true_velocity=winner_true_velocity,
        loser_true_velocity=loser_true_velocity,
    )


def _validate_target_tail_inputs(
    prediction: Any,
    true_target_velocity: Any,
    target_selector: Any,
    *,
    prediction_label: str,
    require_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted = _require_finite_float_tensor(
        prediction, label=prediction_label
    )
    if (
        predicted.ndim != 3
        or int(predicted.shape[0]) != 1
        or int(predicted.shape[1]) <= 0
        or int(predicted.shape[2]) != runtime_contract.PINNED_PATCH_DIM
    ):
        raise DCLRPreferenceObjectiveError(
            f"{prediction_label} must be [1,total,"
            f"{runtime_contract.PINNED_PATCH_DIM}]"
        )
    has_graph = bool(predicted.requires_grad or predicted.grad_fn is not None)
    if require_graph and not has_graph:
        raise DCLRPreferenceObjectiveError(
            f"{prediction_label} must retain the active adapter graph"
        )
    if not require_graph and has_graph:
        raise DCLRPreferenceObjectiveError(
            f"{prediction_label} must be detached collection-policy output"
        )

    target = _require_detached_fp32(
        true_target_velocity, label="true_target_velocity"
    )
    if (
        target.ndim != 3
        or int(target.shape[0]) != 1
        or int(target.shape[1]) <= 0
        or int(target.shape[2]) != runtime_contract.PINNED_PATCH_DIM
    ):
        raise DCLRPreferenceObjectiveError(
            "true_target_velocity must be exact detached FP32 "
            f"[1,N,{runtime_contract.PINNED_PATCH_DIM}]"
        )
    if target.device != predicted.device:
        raise DCLRPreferenceObjectiveError(
            "prediction and true_target_velocity devices differ"
        )

    selector = _require_tensor(target_selector, label="target_selector")
    total = int(predicted.shape[1])
    if (
        selector.dtype != torch.bool
        or selector.ndim != 1
        or int(selector.numel()) != total
        or selector.device != predicted.device
        or selector.requires_grad
    ):
        raise DCLRPreferenceObjectiveError(
            f"target_selector must be detached bool [{total}] on prediction device"
        )
    target_count = int(selector.sum().item())
    boundary = total - target_count
    if (
        target_count <= 0
        or target_count != int(target.shape[1])
        or bool(selector[:boundary].any().item())
        or not bool(selector[boundary:].all().item())
    ):
        raise DCLRPreferenceObjectiveError(
            "target_selector must choose exactly one contiguous target tail"
        )
    return predicted, target, selector


def _target_tail_fp32_mse_with_graph(
    prediction: torch.Tensor,
    true_target_velocity: torch.Tensor,
    target_selector: torch.Tensor,
) -> torch.Tensor:
    selected = prediction[:, target_selector, :].to(dtype=torch.float32)
    energy = (selected - true_target_velocity).square().mean()
    if energy.dtype != torch.float32 or energy.ndim != 0:
        raise DCLRPreferenceObjectiveError(
            "target-tail energy did not remain an FP32 scalar"
        )
    if not bool(torch.isfinite(energy).item()):
        raise DCLRPreferenceObjectiveError("target-tail energy is not finite")
    return energy


def candidate_current_reference_target_tail_mse(
    current_prediction: torch.Tensor,
    reference_prediction: torch.Tensor,
    true_target_velocity: torch.Tensor,
    target_selector: torch.Tensor,
) -> CandidatePolicyEnergies:
    """Score one candidate under current and frozen collection policies.

    Source-prefix predictions are never scored.  The reference output must be
    detached, while the current output must retain an adapter graph.  Both
    paths are converted to FP32 only for subtraction and accumulation.
    """

    current, target, selector = _validate_target_tail_inputs(
        current_prediction,
        true_target_velocity,
        target_selector,
        prediction_label="current_prediction",
        require_graph=True,
    )
    reference, reference_target, reference_selector = (
        _validate_target_tail_inputs(
            reference_prediction,
            true_target_velocity,
            target_selector,
            prediction_label="reference_prediction",
            require_graph=False,
        )
    )
    _require_same_representation(
        current,
        reference,
        left_label="current_prediction",
        right_label="reference_prediction",
    )
    if reference_target is not target or reference_selector is not selector:
        raise DCLRPreferenceObjectiveError(
            "current/reference validation did not retain the same target state"
        )
    current_energy = _target_tail_fp32_mse_with_graph(
        current, target, selector
    )
    reference_energy = _target_tail_fp32_mse_with_graph(
        reference, target, selector
    )
    if not current_energy.requires_grad:
        raise DCLRPreferenceObjectiveError(
            "current target-tail energy lost the adapter graph"
        )
    if reference_energy.requires_grad or reference_energy.grad_fn is not None:
        raise DCLRPreferenceObjectiveError(
            "reference target-tail energy unexpectedly retained a graph"
        )
    return CandidatePolicyEnergies(
        current=current_energy,
        reference=reference_energy,
    )


def _require_policy_energies(
    value: Any,
    *,
    label: str,
) -> CandidatePolicyEnergies:
    if not isinstance(value, CandidatePolicyEnergies):
        raise DCLRPreferenceObjectiveError(
            f"{label} must be CandidatePolicyEnergies"
        )
    current = _require_finite_float_tensor(
        value.current, label=f"{label}.current"
    )
    reference = _require_finite_float_tensor(
        value.reference, label=f"{label}.reference"
    )
    if current.dtype != torch.float32 or current.ndim != 0:
        raise DCLRPreferenceObjectiveError(
            f"{label}.current must be one FP32 scalar"
        )
    if reference.dtype != torch.float32 or reference.ndim != 0:
        raise DCLRPreferenceObjectiveError(
            f"{label}.reference must be one FP32 scalar"
        )
    if current.device != reference.device:
        raise DCLRPreferenceObjectiveError(
            f"{label} current/reference energy devices differ"
        )
    if not current.requires_grad:
        raise DCLRPreferenceObjectiveError(
            f"{label}.current must retain the active adapter graph"
        )
    if reference.requires_grad or reference.grad_fn is not None:
        raise DCLRPreferenceObjectiveError(
            f"{label}.reference must be detached"
        )
    if bool((current < 0.0).item()) or bool((reference < 0.0).item()):
        raise DCLRPreferenceObjectiveError(
            f"{label} denoising energies must be nonnegative"
        )
    return value


def reference_corrected_dpo(
    winner: CandidatePolicyEnergies,
    loser: CandidatePolicyEnergies,
    *,
    beta: float,
) -> ReferenceCorrectedDPO:
    """Return ``softplus(-beta * Delta)`` for one winner/loser pair."""

    winner_energy = _require_policy_energies(winner, label="winner")
    loser_energy = _require_policy_energies(loser, label="loser")
    tensors = (
        winner_energy.current,
        winner_energy.reference,
        loser_energy.current,
        loser_energy.reference,
    )
    if any(tensor.device != tensors[0].device for tensor in tensors[1:]):
        raise DCLRPreferenceObjectiveError(
            "winner/loser policy energy devices differ"
        )
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise DCLRPreferenceObjectiveError(
            "beta must be a finite positive scalar"
        )
    beta_value = float(beta)
    if not math.isfinite(beta_value) or beta_value <= 0.0:
        raise DCLRPreferenceObjectiveError(
            "beta must be a finite positive scalar"
        )
    beta_tensor = tensors[0].new_tensor(beta_value)

    current_margin = loser_energy.current - winner_energy.current
    reference_margin = loser_energy.reference - winner_energy.reference
    delta = current_margin - reference_margin
    loss = F.softplus(-beta_tensor * delta)
    for label, value in (
        ("current margin", current_margin),
        ("reference margin", reference_margin),
        ("preference delta", delta),
        ("DPO loss", loss),
    ):
        if value.dtype != torch.float32 or value.ndim != 0:
            raise DCLRPreferenceObjectiveError(
                f"{label} did not remain one FP32 scalar"
            )
        if not bool(torch.isfinite(value).item()):
            raise DCLRPreferenceObjectiveError(f"{label} is not finite")
    if not loss.requires_grad or not delta.requires_grad:
        raise DCLRPreferenceObjectiveError(
            "reference-corrected DPO lost the current-policy graph"
        )
    if reference_margin.requires_grad or reference_margin.grad_fn is not None:
        raise DCLRPreferenceObjectiveError(
            "reference margin unexpectedly retained a graph"
        )
    return ReferenceCorrectedDPO(
        loss=loss,
        delta=delta,
        current_margin=current_margin,
        reference_margin=reference_margin,
        beta=beta_tensor,
    )


def _require_axis_map(value: Any, *, label: str) -> dict[str, bool]:
    if not isinstance(value, Mapping) or not value:
        raise DCLRPreferenceObjectiveError(
            f"{label} must be a nonempty axis-to-bool mapping"
        )
    result: dict[str, bool] = {}
    for axis, passed in value.items():
        if type(axis) is not str or not axis.strip() or axis != axis.strip():
            raise DCLRPreferenceObjectiveError(
                f"{label} axis names must be nonempty canonical strings"
            )
        if type(passed) is not bool:
            raise DCLRPreferenceObjectiveError(
                f"{label}.{axis} must be a plain bool"
            )
        if axis in result:
            raise DCLRPreferenceObjectiveError(f"{label} contains duplicate axes")
        result[axis] = passed
    return result


def route_one_sided_nearmiss(
    pair_type: str,
    *,
    winner_action_axis_pass: Mapping[str, bool],
    winner_preservation_axis_pass: Mapping[str, bool],
    loser_action_axis_pass: Mapping[str, bool],
    loser_preservation_axis_pass: Mapping[str, bool],
) -> OneSidedNearMissRoute:
    """Authorize exactly one adapter for a strict one-sided near-miss.

    The winner must pass every action and preservation sub-axis.  The loser
    must fail exactly one sub-axis in the family named by ``pair_type`` and
    pass every sub-axis in the other family.  Any ambiguous, two-sided, or
    mislabeled pair raises before an optimizer can see a loss.
    """

    if type(pair_type) is not str or pair_type not in {
        ACTION_NEARMISS,
        PRESERVATION_NEARMISS,
    }:
        raise DCLRPreferenceObjectiveError(
            f"unsupported one-sided pair_type: {pair_type!r}"
        )
    winner_action = _require_axis_map(
        winner_action_axis_pass, label="winner_action_axis_pass"
    )
    winner_preservation = _require_axis_map(
        winner_preservation_axis_pass,
        label="winner_preservation_axis_pass",
    )
    loser_action = _require_axis_map(
        loser_action_axis_pass, label="loser_action_axis_pass"
    )
    loser_preservation = _require_axis_map(
        loser_preservation_axis_pass,
        label="loser_preservation_axis_pass",
    )
    if set(winner_action) != set(loser_action):
        raise DCLRPreferenceObjectiveError(
            "winner/loser action-axis schemas differ"
        )
    if set(winner_preservation) != set(loser_preservation):
        raise DCLRPreferenceObjectiveError(
            "winner/loser preservation-axis schemas differ"
        )
    if not all(winner_action.values()) or not all(winner_preservation.values()):
        raise DCLRPreferenceObjectiveError(
            "preference winner must pass every action and preservation axis"
        )

    failed_action = tuple(
        axis for axis, passed in loser_action.items() if not passed
    )
    failed_preservation = tuple(
        axis for axis, passed in loser_preservation.items() if not passed
    )
    if pair_type == ACTION_NEARMISS:
        if len(failed_action) != 1 or failed_preservation:
            raise DCLRPreferenceObjectiveError(
                "action_nearmiss loser must fail exactly one action axis only"
            )
        return OneSidedNearMissRoute(
            pair_type=pair_type,
            active_adapter=ACTION_ADAPTER,
            action_adapter_trainable=True,
            identity_adapter_trainable=False,
            loser_failed_axis=failed_action[0],
        )
    if failed_action or len(failed_preservation) != 1:
        raise DCLRPreferenceObjectiveError(
            "preservation_nearmiss loser must fail exactly one preservation axis only"
        )
    return OneSidedNearMissRoute(
        pair_type=pair_type,
        active_adapter=IDENTITY_ADAPTER,
        action_adapter_trainable=False,
        identity_adapter_trainable=True,
        loser_failed_axis=failed_preservation[0],
    )


def _require_parameter_group(
    values: Any,
    *,
    label: str,
) -> tuple[torch.Tensor, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise DCLRPreferenceObjectiveError(
            f"{label} must be a nonempty tensor sequence"
        )
    tensors = tuple(values)
    if not tensors:
        raise DCLRPreferenceObjectiveError(
            f"{label} must be a nonempty tensor sequence"
        )
    for index, tensor in enumerate(tensors):
        _require_finite_float_tensor(tensor, label=f"{label}[{index}]")
    if len({id(tensor) for tensor in tensors}) != len(tensors):
        raise DCLRPreferenceObjectiveError(f"{label} contains duplicate tensors")
    return tensors


def validate_adapter_trainability(
    route: OneSidedNearMissRoute,
    *,
    action_adapter_parameters: Sequence[torch.Tensor],
    identity_adapter_parameters: Sequence[torch.Tensor],
) -> OneSidedNearMissRoute:
    """Prove that ``requires_grad`` agrees with the authorized route.

    This check belongs immediately before the current-policy forward.  It
    prevents a correct pair label from updating both LoRA families through a
    shared computation graph.
    """

    if not isinstance(route, OneSidedNearMissRoute):
        raise DCLRPreferenceObjectiveError(
            "route must be a validated OneSidedNearMissRoute"
        )
    expected = {
        ACTION_NEARMISS: (ACTION_ADAPTER, True, False),
        PRESERVATION_NEARMISS: (IDENTITY_ADAPTER, False, True),
    }.get(route.pair_type)
    if expected is None or (
        route.active_adapter,
        route.action_adapter_trainable,
        route.identity_adapter_trainable,
    ) != expected:
        raise DCLRPreferenceObjectiveError(
            "near-miss route fields are internally inconsistent"
        )
    action = _require_parameter_group(
        action_adapter_parameters, label="action_adapter_parameters"
    )
    identity = _require_parameter_group(
        identity_adapter_parameters, label="identity_adapter_parameters"
    )
    if {id(tensor) for tensor in action} & {id(tensor) for tensor in identity}:
        raise DCLRPreferenceObjectiveError(
            "action and identity adapter parameter groups overlap"
        )
    if any(
        bool(parameter.requires_grad) != route.action_adapter_trainable
        for parameter in action
    ):
        raise DCLRPreferenceObjectiveError(
            "action adapter requires_grad state violates near-miss route"
        )
    if any(
        bool(parameter.requires_grad) != route.identity_adapter_trainable
        for parameter in identity
    ):
        raise DCLRPreferenceObjectiveError(
            "identity adapter requires_grad state violates near-miss route"
        )
    return route


def compute_routed_reference_corrected_dpo(
    winner: CandidatePolicyEnergies,
    loser: CandidatePolicyEnergies,
    *,
    beta: float,
    pair_type: str,
    winner_action_axis_pass: Mapping[str, bool],
    winner_preservation_axis_pass: Mapping[str, bool],
    loser_action_axis_pass: Mapping[str, bool],
    loser_preservation_axis_pass: Mapping[str, bool],
) -> RoutedPreferenceObjective:
    """Fail closed on near-miss legality before constructing the DPO loss."""

    route = route_one_sided_nearmiss(
        pair_type,
        winner_action_axis_pass=winner_action_axis_pass,
        winner_preservation_axis_pass=winner_preservation_axis_pass,
        loser_action_axis_pass=loser_action_axis_pass,
        loser_preservation_axis_pass=loser_preservation_axis_pass,
    )
    objective = reference_corrected_dpo(winner, loser, beta=beta)
    return RoutedPreferenceObjective(route=route, objective=objective)


__all__ = [
    "ACTION_ADAPTER",
    "ACTION_NEARMISS",
    "CandidatePolicyEnergies",
    "DCLRPreferenceObjectiveError",
    "IDENTITY_ADAPTER",
    "OneSidedNearMissRoute",
    "PRESERVATION_NEARMISS",
    "ReferenceCorrectedDPO",
    "RoutedPreferenceObjective",
    "SCHEMA_VERSION",
    "SharedPairFlowState",
    "build_shared_pair_flow_state",
    "candidate_current_reference_target_tail_mse",
    "compute_routed_reference_corrected_dpo",
    "reference_corrected_dpo",
    "route_one_sided_nearmiss",
    "validate_adapter_trainability",
]
