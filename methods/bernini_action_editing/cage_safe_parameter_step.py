"""Small active-set QP for a source-safe CAGE parameter update.

CAGE obtains differentiable action and preservation signals in clean-candidate
space, but the LoRA update lives in parameter space.  Projecting only the
candidate cotangent is insufficient: the student Jacobian changes the metric.
This module therefore consumes gradients *after* each signal has been replayed
through the same student ``VI_cond`` branch.

For action gradient ``a = dL_action/dtheta`` and constraint gradients
``g_j = dC_j/dtheta``, it projects the proposed SGD displacement

    d0 = -eta * a

onto the registered linearized safe set

    C_j + g_j^T d <= 0.

The first canary requires every current constraint value ``C_j <= 0``.  Hence
zero displacement is feasible.  The requested trust radius is enforced by
shrinking ``eta`` before projection; Euclidean projection onto a convex set
containing zero cannot increase the displacement norm beyond ``||d0||``.

There are at most five constraints.  The exact active set is selected by
enumerating at most 31 subsets and solving only their tiny FP64 Gram systems.
No flattened copy of the trainable LoRA is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from typing import Any, Mapping, Optional, Sequence

import torch


SCHEMA_VERSION = "bernini-cage-source-safe-parameter-step-v1"
MAX_CONSTRAINTS = 5


class CAGESafeParameterStepError(RuntimeError):
    """An action/constraint gradient or QP contract is invalid."""


@dataclass(frozen=True)
class SafeParameterStep:
    """One projected displacement and the gradient that realizes it in SGD."""

    displacement: tuple[torch.Tensor, ...]
    projected_gradient: tuple[torch.Tensor, ...]
    effective_step_size: float
    action_gradient_norm: float
    unprojected_step_norm: float
    projected_step_norm: float
    retention_ratio: float
    active_constraint_indices: tuple[int, ...]
    active_constraint_names: tuple[str, ...]
    dual_coefficients: tuple[float, ...]
    linearized_constraint_values: tuple[float, ...]
    feasible: bool
    update_authorized: bool
    block_reason: Optional[str]
    receipt: Mapping[str, Any]


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise CAGESafeParameterStepError(
            "receipt is not canonical finite ASCII JSON"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _positive(value: Any, *, label: str, allow_zero: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (float(value) < 0.0 if allow_zero else float(value) <= 0.0)
    ):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise CAGESafeParameterStepError(f"{label} must be {qualifier} finite")
    return float(value)


def _gradient_tuple(value: Any, *, label: str) -> tuple[torch.Tensor, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CAGESafeParameterStepError(f"{label} must be a tensor sequence")
    result = tuple(value)
    if not result:
        raise CAGESafeParameterStepError(f"{label} cannot be empty")
    device: Optional[torch.device] = None
    for index, tensor in enumerate(result):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.float32
            or tensor.device.type == "meta"
            or tensor.layout != torch.strided
            or tensor.requires_grad
            or tensor.grad_fn is not None
            or any(int(item) <= 0 for item in tensor.shape)
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise CAGESafeParameterStepError(
                f"{label}[{index}] must be detached finite strided FP32"
            )
        if device is None:
            device = tensor.device
        elif tensor.device != device:
            raise CAGESafeParameterStepError(
                f"{label} tensors must share one device"
            )
    return result


def _same_geometry(
    reference: Sequence[torch.Tensor],
    value: Sequence[torch.Tensor],
    *,
    label: str,
) -> None:
    if len(reference) != len(value) or any(
        left.shape != right.shape or left.device != right.device
        for left, right in zip(reference, value)
    ):
        raise CAGESafeParameterStepError(
            f"{label} does not match the action-gradient parameter geometry"
        )


def _inner(
    left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]
) -> torch.Tensor:
    # Each local reduction is FP64.  A distributed trainer must all-reduce the
    # resulting scalar Gram matrix across DP/SP before calling the solver.
    value = torch.zeros((), dtype=torch.float64, device=left[0].device)
    for lhs, rhs in zip(left, right):
        value = value + torch.sum(lhs.to(torch.float64) * rhs.to(torch.float64))
    return value


def _linear_combination(
    base: Sequence[torch.Tensor],
    gradients: Sequence[Sequence[torch.Tensor]],
    coefficients: Sequence[float],
) -> tuple[torch.Tensor, ...]:
    result = [item.clone() for item in base]
    for gradient, coefficient in zip(gradients, coefficients):
        if coefficient == 0.0:
            continue
        for index, tensor in enumerate(gradient):
            result[index].add_(tensor, alpha=-float(coefficient))
    return tuple(item.detach().contiguous() for item in result)


def _zero_like(value: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(torch.zeros_like(item) for item in value)


def _blocked_result(
    *,
    action: Sequence[torch.Tensor],
    effective_step_size: float,
    action_norm: float,
    unprojected_norm: float,
    constraint_names: Sequence[str],
    constraint_values: Sequence[float],
    reason: str,
) -> SafeParameterStep:
    zero = _zero_like(action)
    value = {
        "schema_version": SCHEMA_VERSION,
        "constraint_names": list(constraint_names),
        "constraint_values": list(constraint_values),
        "effective_step_size": effective_step_size,
        "action_gradient_norm": action_norm,
        "unprojected_step_norm": unprojected_norm,
        "projected_step_norm": 0.0,
        "retention_ratio": 0.0,
        "active_constraint_indices": [],
        "dual_coefficients": [0.0 for _ in constraint_names],
        "linearized_constraint_values": list(constraint_values),
        "feasible": False,
        "update_authorized": False,
        "block_reason": reason,
        "solver": "enumerated_active_set_fp64_gram",
        "optimizer_geometry": "trainable_parameter_space_after_student_vjp",
    }
    receipt = {**value, "digest": _object_sha256(value)}
    return SafeParameterStep(
        displacement=zero,
        projected_gradient=zero,
        effective_step_size=effective_step_size,
        action_gradient_norm=action_norm,
        unprojected_step_norm=unprojected_norm,
        projected_step_norm=0.0,
        retention_ratio=0.0,
        active_constraint_indices=(),
        active_constraint_names=(),
        dual_coefficients=tuple(0.0 for _ in constraint_names),
        linearized_constraint_values=tuple(float(item) for item in constraint_values),
        feasible=False,
        update_authorized=False,
        block_reason=reason,
        receipt=receipt,
    )


def project_safe_parameter_step(
    action_gradient: Sequence[torch.Tensor],
    constraint_gradients: Sequence[Sequence[torch.Tensor]],
    constraint_values: Sequence[float],
    constraint_names: Sequence[str],
    *,
    step_size: float,
    trust_radius: float,
    minimum_retention: float = 0.2,
    feasibility_tolerance: float = 1.0e-8,
) -> SafeParameterStep:
    """Project one SGD displacement in the true trainable-parameter metric.

    ``constraint_gradients`` must be computed at the same model state, native
    branch, source, action, and sigma as ``action_gradient``.  This function is
    deliberately unaware of candidates, masks, T2V proposals, and model media.
    """

    action = _gradient_tuple(action_gradient, label="action_gradient")
    if not isinstance(constraint_gradients, Sequence) or isinstance(
        constraint_gradients, (str, bytes)
    ):
        raise CAGESafeParameterStepError(
            "constraint_gradients must be a sequence"
        )
    constraints = tuple(
        _gradient_tuple(item, label=f"constraint_gradients[{index}]")
        for index, item in enumerate(constraint_gradients)
    )
    count = len(constraints)
    if not 1 <= count <= MAX_CONSTRAINTS:
        raise CAGESafeParameterStepError(
            f"constraint count must lie in [1,{MAX_CONSTRAINTS}]"
        )
    for index, gradient in enumerate(constraints):
        _same_geometry(action, gradient, label=f"constraint gradient {index}")
    if (
        not isinstance(constraint_values, Sequence)
        or isinstance(constraint_values, (str, bytes))
        or len(constraint_values) != count
        or not isinstance(constraint_names, Sequence)
        or isinstance(constraint_names, (str, bytes))
        or len(constraint_names) != count
    ):
        raise CAGESafeParameterStepError(
            "constraint values/names must match gradient count"
        )
    names = tuple(constraint_names)
    if any(
        not isinstance(name, str)
        or not name
        or name != name.strip()
        or "\x00" in name
        for name in names
    ) or len(set(names)) != count:
        raise CAGESafeParameterStepError(
            "constraint names must be unique canonical text"
        )
    values: list[float] = []
    for index, item in enumerate(constraint_values):
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise CAGESafeParameterStepError(
                f"constraint value {index} must be finite"
            )
        values.append(float(item))

    requested_eta = _positive(step_size, label="step_size")
    radius = _positive(trust_radius, label="trust_radius")
    retention_floor = _positive(
        minimum_retention, label="minimum_retention", allow_zero=True
    )
    tolerance = _positive(
        feasibility_tolerance, label="feasibility_tolerance"
    )
    if retention_floor > 1.0:
        raise CAGESafeParameterStepError("minimum_retention cannot exceed one")

    action_norm_sq = float(_inner(action, action).item())
    if not math.isfinite(action_norm_sq) or action_norm_sq <= 0.0:
        raise CAGESafeParameterStepError(
            "action parameter gradient must be finite and nonzero"
        )
    action_norm = math.sqrt(action_norm_sq)
    effective_eta = min(requested_eta, radius / action_norm)
    base = tuple((-effective_eta * item).detach().contiguous() for item in action)
    unprojected_norm = effective_eta * action_norm

    if any(value > tolerance for value in values):
        return _blocked_result(
            action=action,
            effective_step_size=effective_eta,
            action_norm=action_norm,
            unprojected_norm=unprojected_norm,
            constraint_names=names,
            constraint_values=values,
            reason="current_constraint_violation_requires_recovery_not_canary_update",
        )

    gram = torch.empty((count, count), dtype=torch.float64, device=action[0].device)
    g_dot_d0 = torch.empty((count,), dtype=torch.float64, device=action[0].device)
    for row in range(count):
        g_dot_d0[row] = _inner(constraints[row], base)
        for column in range(row + 1):
            scalar = _inner(constraints[row], constraints[column])
            gram[row, column] = scalar
            gram[column, row] = scalar
    value_tensor = torch.tensor(values, dtype=torch.float64, device=gram.device)
    unprojected_linearized = g_dot_d0 + value_tensor

    best_coefficients: Optional[torch.Tensor] = None
    best_active: tuple[int, ...] = ()
    best_objective = math.inf
    if bool((unprojected_linearized <= tolerance).all().item()):
        best_coefficients = torch.zeros(count, dtype=torch.float64, device=gram.device)
        best_objective = 0.0
    else:
        indices = tuple(range(count))
        for subset_size in range(1, count + 1):
            for active in itertools.combinations(indices, subset_size):
                active_index = torch.tensor(active, dtype=torch.long, device=gram.device)
                matrix = gram.index_select(0, active_index).index_select(1, active_index)
                rhs = unprojected_linearized.index_select(0, active_index)
                try:
                    solution = torch.linalg.lstsq(matrix, rhs.unsqueeze(1)).solution[:, 0]
                except RuntimeError:
                    continue
                if not bool(torch.isfinite(solution).all().item()) or bool(
                    (solution < -tolerance).any().item()
                ):
                    continue
                solution = torch.where(
                    solution.abs() <= tolerance,
                    torch.zeros_like(solution),
                    solution,
                )
                coefficients = torch.zeros(count, dtype=torch.float64, device=gram.device)
                coefficients[active_index] = solution
                linearized = unprojected_linearized - gram @ coefficients
                equality_error = float(
                    linearized.index_select(0, active_index).abs().max().item()
                )
                if equality_error > 10.0 * tolerance or bool(
                    (linearized > tolerance).any().item()
                ):
                    continue
                objective = float((0.5 * coefficients @ gram @ coefficients).item())
                if not math.isfinite(objective) or objective < -tolerance:
                    continue
                if objective < best_objective:
                    best_coefficients = coefficients
                    best_active = active
                    best_objective = objective

    if best_coefficients is None:
        return _blocked_result(
            action=action,
            effective_step_size=effective_eta,
            action_norm=action_norm,
            unprojected_norm=unprojected_norm,
            constraint_names=names,
            constraint_values=values,
            reason="linearized_safe_set_solver_found_no_feasible_projection",
        )

    coefficient_values = tuple(float(item) for item in best_coefficients.tolist())
    displacement = _linear_combination(base, constraints, coefficient_values)
    projected_norm_sq = float(_inner(displacement, displacement).item())
    if not math.isfinite(projected_norm_sq) or projected_norm_sq < 0.0:
        raise CAGESafeParameterStepError("projected step norm is invalid")
    projected_norm = math.sqrt(projected_norm_sq)
    retention = projected_norm / unprojected_norm
    linearized_tensor = unprojected_linearized - gram @ best_coefficients
    linearized_values = tuple(float(item) for item in linearized_tensor.tolist())
    feasible = (
        all(item <= tolerance for item in linearized_values)
        and projected_norm <= radius + 10.0 * tolerance
        and retention <= 1.0 + 10.0 * tolerance
    )
    authorized = feasible and retention >= retention_floor
    reason = None if authorized else (
        "projected_gradient_retention_below_registered_floor"
        if feasible
        else "projected_step_failed_postsolve_feasibility"
    )
    if not authorized:
        return _blocked_result(
            action=action,
            effective_step_size=effective_eta,
            action_norm=action_norm,
            unprojected_norm=unprojected_norm,
            constraint_names=names,
            constraint_values=values,
            reason=str(reason),
        )

    projected_gradient = tuple(
        (-item / effective_eta).detach().contiguous() for item in displacement
    )
    value = {
        "schema_version": SCHEMA_VERSION,
        "constraint_names": list(names),
        "constraint_values": list(values),
        "effective_step_size": effective_eta,
        "action_gradient_norm": action_norm,
        "unprojected_step_norm": unprojected_norm,
        "projected_step_norm": projected_norm,
        "retention_ratio": retention,
        "active_constraint_indices": list(best_active),
        "dual_coefficients": list(coefficient_values),
        "linearized_constraint_values": list(linearized_values),
        "feasible": True,
        "update_authorized": True,
        "block_reason": None,
        "solver": "enumerated_active_set_fp64_gram",
        "optimizer_geometry": "trainable_parameter_space_after_student_vjp",
    }
    receipt = {**value, "digest": _object_sha256(value)}
    return SafeParameterStep(
        displacement=displacement,
        projected_gradient=projected_gradient,
        effective_step_size=effective_eta,
        action_gradient_norm=action_norm,
        unprojected_step_norm=unprojected_norm,
        projected_step_norm=projected_norm,
        retention_ratio=retention,
        active_constraint_indices=best_active,
        active_constraint_names=tuple(names[index] for index in best_active),
        dual_coefficients=coefficient_values,
        linearized_constraint_values=linearized_values,
        feasible=True,
        update_authorized=True,
        block_reason=None,
        receipt=receipt,
    )


def assign_projected_gradients(
    parameters: Sequence[torch.nn.Parameter], result: SafeParameterStep
) -> None:
    """Install the audited projected gradient before a plain SGD step."""

    if not isinstance(result, SafeParameterStep) or not result.update_authorized:
        raise CAGESafeParameterStepError(
            "cannot assign gradients from an unauthorized safe step"
        )
    if not isinstance(parameters, Sequence) or len(parameters) != len(
        result.projected_gradient
    ):
        raise CAGESafeParameterStepError(
            "parameter closure differs from projected gradient"
        )
    for index, (parameter, gradient) in enumerate(
        zip(parameters, result.projected_gradient)
    ):
        if (
            not isinstance(parameter, torch.nn.Parameter)
            or not parameter.requires_grad
            or parameter.shape != gradient.shape
            or parameter.device != gradient.device
            or parameter.dtype != torch.float32
        ):
            raise CAGESafeParameterStepError(
                f"trainable parameter {index} differs from projected gradient"
            )
    for parameter, gradient in zip(parameters, result.projected_gradient):
        parameter.grad = gradient.clone()


__all__ = [
    "CAGESafeParameterStepError",
    "MAX_CONSTRAINTS",
    "SCHEMA_VERSION",
    "SafeParameterStep",
    "assign_projected_gradients",
    "project_safe_parameter_step",
]
