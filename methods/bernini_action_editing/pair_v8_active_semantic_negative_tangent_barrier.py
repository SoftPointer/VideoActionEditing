#!/usr/bin/env python3
"""PAIR-v8 Phase-B active semantic-negative tangent barrier.

This module is an audit-only numerical core.  It does not import the PAIR-v7
runtime, does not mutate a tensor, and does not create an optimizer or a LoRA
delta.  Its purpose is narrower: test whether one common action descent
direction survives the joint tangent barrier formed by

* 64 source-identity VJP rows; and
* 9 semantic negative branches for each of 8 action conditions (72 rows).

The negative rows are *active signed-linear VJPs*.  They are measured by four
SP-rank-local ``student_output.backward(teacher_action_cotangent)`` calls,
then arithmetic-mean reduced, and never by a parity MSE
such as ``||feature(B)-feature(B=0)||^2``.  Consequently the measurement can be
non-zero at the exact B=0 parity point where the latter objective has a zero
gradient.

All measured rows enter a single FP64 compact-Gram nullspace projection.  The
prospective FP32 direction is then re-projected and re-audited in FP64.  A GO
requires full row rank and acceptable conditioning for every fixed panel,
positive descent for each of the eight action rows, and near-zero dot/cosine
against every one of the 136 barrier rows.  A failed audit exposes no usable
direction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Optional, Sequence

import torch


METHOD_NAME = "bernini-pair-v8-active-semantic-negative-tangent-barrier"
ROW_MEASUREMENT_SCHEMA = "bernini-pair-v8-tangent-row-measurement-v1"
ROW_AUTHORITY_SCHEMA = "bernini-pair-v8-tangent-row-authority-v1"
MEASUREMENT_AUTHORITY_SCHEMA = (
    "bernini-pair-v8-active-negative-measurement-authority-v1"
)
BARRIER_RECEIPT_SCHEMA = "bernini-pair-v8-active-negative-tangent-barrier-v1"

NEGATIVE_BRANCHES = (
    "noop",
    "incomplete",
    "reverse",
    "shuffle",
    "wrong_actor",
    "wrong_object",
    "camera_only",
    "appearance_only",
    "generic_wrong_motion",
)
EXPECTED_ACTION_CONDITION_COUNT = 8
IDENTITY_ROWS_PER_CONDITION = 8
EXPECTED_IDENTITY_ROW_COUNT = 64
EXPECTED_NEGATIVE_ROW_COUNT = 72
EXPECTED_BARRIER_ROW_COUNT = 136
SP4_SIZE = 4
UNIT_RMS_TOLERANCE = 2.0e-5

ACTIVE_NEGATIVE_OPERATOR = (
    "tensor.backward-teacher-action-cotangent-sp4-arithmetic-mean-vjp"
)
EXTERNAL_ACTION_OPERATOR = "externally-measured-action-objective-vjp"
EXTERNAL_IDENTITY_OPERATOR = "externally-measured-signed-linear-identity-vjp"

_SAFE_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,255}")
_SHA_RE = re.compile(r"[0-9a-f]{64}")


class PairV8NegativeBarrierError(RuntimeError):
    """The fixed tangent panel, its authority, or its geometry is ambiguous."""


@dataclass(frozen=True)
class BarrierConfig:
    """Fixed numerical policy; values may be tightened, never loosened."""

    minimum_row_norm: float = 1.0e-10
    singular_value_relative_tolerance: float = 1.0e-8
    eigenvalue_absolute_tolerance: float = 1.0e-12
    maximum_effective_condition_number: float = 1.0e4
    minimum_action_descent_gain: float = 1.0e-12
    minimum_action_descent_cosine: float = 2.0e-2
    barrier_dot_absolute_tolerance: float = 1.0e-9
    barrier_dot_relative_tolerance: float = 2.0e-5
    maximum_barrier_cosine: float = 2.0e-5
    fp32_refinement_passes: int = 3

    def validate(self) -> None:
        fields = (
            "minimum_row_norm",
            "singular_value_relative_tolerance",
            "eigenvalue_absolute_tolerance",
            "maximum_effective_condition_number",
            "minimum_action_descent_gain",
            "minimum_action_descent_cosine",
            "barrier_dot_absolute_tolerance",
            "barrier_dot_relative_tolerance",
            "maximum_barrier_cosine",
        )
        for name in fields:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise PairV8NegativeBarrierError(f"{name} must be finite positive")
        if self.maximum_effective_condition_number > 1.0e4:
            raise PairV8NegativeBarrierError("conditioning policy may not be loosened")
        if self.minimum_action_descent_cosine < 2.0e-2:
            raise PairV8NegativeBarrierError("action cosine policy may not be loosened")
        if self.minimum_action_descent_cosine > 1.0:
            raise PairV8NegativeBarrierError("action cosine must not exceed one")
        if self.barrier_dot_relative_tolerance > 2.0e-5:
            raise PairV8NegativeBarrierError("barrier dot policy may not be loosened")
        if self.maximum_barrier_cosine > 2.0e-5:
            raise PairV8NegativeBarrierError("barrier cosine policy may not be loosened")
        if (
            type(self.fp32_refinement_passes) is not int
            or not 1 <= self.fp32_refinement_passes <= 8
        ):
            raise PairV8NegativeBarrierError("FP32 refinement passes must be 1..8")


@dataclass(frozen=True)
class ActionTangentRow:
    condition_id: str
    gradient_by_parameter: Mapping[str, torch.Tensor]
    gradient_sha256: str
    row_authority_digest: str
    semantic_authority_digest: str
    upstream_computation_receipt_digest: str
    checkpoint_content_receipt_digest: str
    parameter_state_sha256: str
    panel_root_authority_digest: str
    measurement_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class IdentityTangentRow:
    condition_id: str
    identity_slot: int
    gradient_by_parameter: Mapping[str, torch.Tensor]
    gradient_sha256: str
    row_authority_digest: str
    semantic_authority_digest: str
    upstream_computation_receipt_digest: str
    checkpoint_content_receipt_digest: str
    parameter_state_sha256: str
    panel_root_authority_digest: str
    measurement_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class NegativeTangentRow:
    condition_id: str
    branch: str
    gradient_by_parameter: Mapping[str, torch.Tensor]
    gradient_sha256: str
    row_authority_digest: str
    semantic_authority_digest: str
    upstream_computation_receipt_digest: str
    checkpoint_content_receipt_digest: str
    parameter_state_sha256: str
    panel_root_authority_digest: str
    measurement_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class BarrierResult:
    """A read-only audit result; ``safe_direction`` is absent on every NO-GO."""

    geometry_authorized: bool
    safe_direction_by_parameter: Optional[Mapping[str, torch.Tensor]]
    receipt: Mapping[str, Any]


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise PairV8NegativeBarrierError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise PairV8NegativeBarrierError(f"{label} must be lowercase SHA-256")
    return value


def _safe(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_RE.fullmatch(value) is None:
        raise PairV8NegativeBarrierError(f"{label} is unsafe")
    return value


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = canonical_json_bytes(
        {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "numel": int(tensor.numel()),
        }
    )
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(header + b"\x00" + raw).hexdigest()


def _validate_gradient_mapping(
    value: Mapping[str, torch.Tensor], *, label: str, require_detached: bool
) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise PairV8NegativeBarrierError(f"{label} gradient mapping is empty")
    names = tuple(value)
    if len(set(names)) != len(names) or any(
        not isinstance(name, str) or _SAFE_RE.fullmatch(name) is None
        for name in names
    ):
        raise PairV8NegativeBarrierError(f"{label} parameter names differ")
    result: dict[str, torch.Tensor] = {}
    for name in sorted(names):
        tensor = value[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or tensor.numel() == 0
            or (require_detached and tensor.requires_grad)
            or not bool(torch.isfinite(tensor.detach()).all().item())
        ):
            raise PairV8NegativeBarrierError(
                f"{label} {name} must be materialized finite CPU FP32"
            )
        result[name] = tensor.detach().clone().contiguous()
    return result


def gradient_sha256(value: Mapping[str, torch.Tensor]) -> str:
    gradients = _validate_gradient_mapping(
        value, label="gradient digest", require_detached=True
    )
    return object_sha256(
        [
            {
                "name": name,
                "dtype": "torch.float32",
                "shape": list(gradients[name].shape),
                "tensor_sha256": _tensor_sha256(gradients[name]),
            }
            for name in sorted(gradients)
        ]
    )


def _layout_manifest(value: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "dtype": "torch.float32",
            "shape": list(value[name].shape),
        }
        for name in sorted(value)
    ]


def _row_id(*, role: str, condition_id: str, suffix: str = "") -> str:
    base = f"{role}:{condition_id}"
    return base if not suffix else f"{base}:{suffix}"


def derive_row_authority_digest(
    *,
    panel_root_authority_digest: str,
    role: str,
    row_id: str,
    semantic_authority_digest: str,
    upstream_computation_receipt_digest: str,
    checkpoint_content_receipt_digest: str,
    parameter_state_sha256: str,
) -> str:
    """Derive the fixed slot authority independently of measured values."""

    root = _sha(panel_root_authority_digest, label="panel root authority")
    semantic = _sha(semantic_authority_digest, label="semantic authority")
    upstream = _sha(
        upstream_computation_receipt_digest, label="upstream computation receipt"
    )
    checkpoint = _sha(
        checkpoint_content_receipt_digest, label="checkpoint content receipt"
    )
    state = _sha(parameter_state_sha256, label="parameter state")
    if role not in {"action", "identity", "negative"}:
        raise PairV8NegativeBarrierError("row role differs")
    _safe(row_id, label="row ID")
    return object_sha256(
        {
            "schema_version": ROW_AUTHORITY_SCHEMA,
            "panel_root_authority_digest": root,
            "role": role,
            "row_id": row_id,
            "semantic_authority_digest": semantic,
            "upstream_computation_receipt_digest": upstream,
            "checkpoint_content_receipt_digest": checkpoint,
            "parameter_state_sha256": state,
        }
    )


def _seal(unsigned: Mapping[str, Any], *, digest_name: str) -> dict[str, Any]:
    if digest_name in unsigned:
        raise PairV8NegativeBarrierError("object is already sealed")
    value = dict(unsigned)
    return {**value, digest_name: object_sha256(value)}


def _validate_seal(
    value: Any, *, schema: str, digest_name: str, label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema_version") != schema:
        raise PairV8NegativeBarrierError(f"{label} schema differs")
    unsigned = dict(value)
    digest = unsigned.pop(digest_name, None)
    if not isinstance(digest, str) or object_sha256(unsigned) != digest:
        raise PairV8NegativeBarrierError(f"{label} seal differs")
    return value


def _external_row(
    *,
    role: str,
    condition_id: str,
    suffix: str,
    gradient_by_parameter: Mapping[str, torch.Tensor],
    semantic_authority_digest: str,
    upstream_computation_receipt_digest: str,
    checkpoint_content_receipt_digest: str,
    parameter_state_sha256: str,
    panel_root_authority_digest: str,
) -> tuple[dict[str, torch.Tensor], str, str, Mapping[str, Any]]:
    condition = _safe(condition_id, label="condition ID")
    gradients = _validate_gradient_mapping(
        gradient_by_parameter, label=f"{role} {condition}", require_detached=True
    )
    norm = float(
        torch.linalg.vector_norm(
            torch.cat([gradients[name].reshape(-1) for name in sorted(gradients)])
        ).item()
    )
    if not math.isfinite(norm) or norm <= 0.0:
        raise PairV8NegativeBarrierError(f"{role} {condition} VJP is zero")
    row_id = _row_id(role=role, condition_id=condition, suffix=suffix)
    semantic = _sha(semantic_authority_digest, label="semantic authority")
    upstream = _sha(
        upstream_computation_receipt_digest, label="upstream computation receipt"
    )
    checkpoint = _sha(
        checkpoint_content_receipt_digest, label="checkpoint content receipt"
    )
    state = _sha(parameter_state_sha256, label="parameter state")
    root = _sha(panel_root_authority_digest, label="panel root authority")
    row_authority = derive_row_authority_digest(
        panel_root_authority_digest=root,
        role=role,
        row_id=row_id,
        semantic_authority_digest=semantic,
        upstream_computation_receipt_digest=upstream,
        checkpoint_content_receipt_digest=checkpoint,
        parameter_state_sha256=state,
    )
    digest = gradient_sha256(gradients)
    operator = (
        EXTERNAL_ACTION_OPERATOR if role == "action" else EXTERNAL_IDENTITY_OPERATOR
    )
    receipt = _seal(
        {
            "schema_version": ROW_MEASUREMENT_SCHEMA,
            "role": role,
            "row_id": row_id,
            "condition_id": condition,
            "slot_suffix": suffix,
            "measurement_operator": operator,
            "active_signed_linear_cotangent_vjp_required": True,
            "parity_residual_mse_objective_used": False,
            "gradient_values_origin_independently_proven_by_core": False,
            "external_measurement_must_be_verified_by_upstream_receipt": True,
            "semantic_authority_digest": semantic,
            "upstream_computation_receipt_digest": upstream,
            "checkpoint_content_receipt_digest": checkpoint,
            "parameter_state_sha256": state,
            "panel_root_authority_digest": root,
            "row_authority_digest": row_authority,
            "parameter_layout": _layout_manifest(gradients),
            "parameter_layout_digest": object_sha256(_layout_manifest(gradients)),
            "gradient_sha256": digest,
            "gradient_norm": norm,
            "raw_gradient_values_persisted_in_receipt": False,
            "parameter_mutation_performed": False,
        },
        digest_name="receipt_digest",
    )
    return gradients, digest, row_authority, receipt


def bind_action_tangent_row(
    *,
    condition_id: str,
    gradient_by_parameter: Mapping[str, torch.Tensor],
    semantic_authority_digest: str,
    upstream_computation_receipt_digest: str,
    checkpoint_content_receipt_digest: str,
    parameter_state_sha256: str,
    panel_root_authority_digest: str,
) -> ActionTangentRow:
    gradients, digest, row_authority, receipt = _external_row(
        role="action",
        condition_id=condition_id,
        suffix="",
        gradient_by_parameter=gradient_by_parameter,
        semantic_authority_digest=semantic_authority_digest,
        upstream_computation_receipt_digest=upstream_computation_receipt_digest,
        checkpoint_content_receipt_digest=checkpoint_content_receipt_digest,
        parameter_state_sha256=parameter_state_sha256,
        panel_root_authority_digest=panel_root_authority_digest,
    )
    return ActionTangentRow(
        condition_id=condition_id,
        gradient_by_parameter=gradients,
        gradient_sha256=digest,
        row_authority_digest=row_authority,
        semantic_authority_digest=semantic_authority_digest,
        upstream_computation_receipt_digest=upstream_computation_receipt_digest,
        checkpoint_content_receipt_digest=checkpoint_content_receipt_digest,
        parameter_state_sha256=parameter_state_sha256,
        panel_root_authority_digest=panel_root_authority_digest,
        measurement_receipt=receipt,
    )


def bind_identity_tangent_row(
    *,
    condition_id: str,
    identity_slot: int,
    gradient_by_parameter: Mapping[str, torch.Tensor],
    semantic_authority_digest: str,
    upstream_computation_receipt_digest: str,
    checkpoint_content_receipt_digest: str,
    parameter_state_sha256: str,
    panel_root_authority_digest: str,
) -> IdentityTangentRow:
    if type(identity_slot) is not int or not 0 <= identity_slot < IDENTITY_ROWS_PER_CONDITION:
        raise PairV8NegativeBarrierError("identity slot must be 0..7")
    suffix = f"k{identity_slot}"
    gradients, digest, row_authority, receipt = _external_row(
        role="identity",
        condition_id=condition_id,
        suffix=suffix,
        gradient_by_parameter=gradient_by_parameter,
        semantic_authority_digest=semantic_authority_digest,
        upstream_computation_receipt_digest=upstream_computation_receipt_digest,
        checkpoint_content_receipt_digest=checkpoint_content_receipt_digest,
        parameter_state_sha256=parameter_state_sha256,
        panel_root_authority_digest=panel_root_authority_digest,
    )
    return IdentityTangentRow(
        condition_id=condition_id,
        identity_slot=identity_slot,
        gradient_by_parameter=gradients,
        gradient_sha256=digest,
        row_authority_digest=row_authority,
        semantic_authority_digest=semantic_authority_digest,
        upstream_computation_receipt_digest=upstream_computation_receipt_digest,
        checkpoint_content_receipt_digest=checkpoint_content_receipt_digest,
        parameter_state_sha256=parameter_state_sha256,
        panel_root_authority_digest=panel_root_authority_digest,
        measurement_receipt=receipt,
    )


def measure_active_semantic_negative_vjp(
    *,
    condition_id: str,
    branch: str,
    feature_output: torch.Tensor,
    parity_reference_feature: torch.Tensor,
    cotangent: torch.Tensor,
    named_parameters: Mapping[str, torch.Tensor],
    semantic_authority_digest: str,
    upstream_computation_receipt_digest: str,
    checkpoint_content_receipt_digest: str,
    parameter_state_sha256: str,
    panel_root_authority_digest: str,
    retain_graph: bool = False,
) -> NegativeTangentRow:
    """Measure one non-zero semantic constraint with a real active VJP.

    ``parity_reference_feature`` is receipt evidence only.  It is deliberately
    never subtracted from ``feature_output`` in the differentiated scalar.
    Thus exact equality at B=0 does not suppress the injected cotangent.
    """

    condition = _safe(condition_id, label="condition ID")
    if branch not in NEGATIVE_BRANCHES:
        raise PairV8NegativeBarrierError("negative branch is not preregistered")
    if type(retain_graph) is not bool:
        raise PairV8NegativeBarrierError("retain_graph must be bool")
    if (
        not isinstance(feature_output, torch.Tensor)
        or feature_output.device.type != "cpu"
        or feature_output.dtype != torch.float32
        or feature_output.numel() == 0
        or not feature_output.requires_grad
        or not bool(torch.isfinite(feature_output).all().item())
    ):
        raise PairV8NegativeBarrierError(
            "feature output must be finite differentiable CPU FP32"
        )
    if (
        not isinstance(parity_reference_feature, torch.Tensor)
        or parity_reference_feature.device.type != "cpu"
        or parity_reference_feature.dtype != torch.float32
        or parity_reference_feature.requires_grad
        or tuple(parity_reference_feature.shape) != tuple(feature_output.shape)
        or not bool(torch.isfinite(parity_reference_feature).all().item())
    ):
        raise PairV8NegativeBarrierError("parity reference feature differs")
    if (
        not isinstance(cotangent, torch.Tensor)
        or cotangent.device.type != "cpu"
        or cotangent.dtype != torch.float32
        or cotangent.requires_grad
        or tuple(cotangent.shape) != tuple(feature_output.shape)
        or not bool(torch.isfinite(cotangent).all().item())
    ):
        raise PairV8NegativeBarrierError("semantic cotangent differs")
    cotangent_norm = float(torch.linalg.vector_norm(cotangent).item())
    if not math.isfinite(cotangent_norm) or cotangent_norm <= 0.0:
        raise PairV8NegativeBarrierError("semantic cotangent must be active non-zero")
    if not isinstance(named_parameters, Mapping) or not named_parameters:
        raise PairV8NegativeBarrierError("named parameter mapping is empty")
    parameter_names = tuple(sorted(named_parameters))
    if any(
        not isinstance(name, str) or _SAFE_RE.fullmatch(name) is None
        for name in parameter_names
    ):
        raise PairV8NegativeBarrierError("VJP parameter name is unsafe")
    parameter_tensors: list[torch.Tensor] = []
    object_ids: set[int] = set()
    for name in parameter_names:
        value = named_parameters[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or value.numel() == 0
            or not value.requires_grad
            or not bool(torch.isfinite(value).all().item())
            or id(value) in object_ids
        ):
            raise PairV8NegativeBarrierError(
                f"VJP parameter {name} must be unique finite CPU FP32 requiring grad"
            )
        object_ids.add(id(value))
        parameter_tensors.append(value)
    try:
        raw_gradients = torch.autograd.grad(
            outputs=feature_output,
            inputs=tuple(parameter_tensors),
            grad_outputs=cotangent,
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=False,
            materialize_grads=False,
        )
    except (RuntimeError, ValueError) as error:
        raise PairV8NegativeBarrierError("active semantic VJP failed") from error
    gradients = _validate_gradient_mapping(
        {
            name: gradient
            for name, gradient in zip(parameter_names, raw_gradients)
        },
        label=f"negative {condition}:{branch}",
        require_detached=True,
    )
    flat = torch.cat([gradients[name].reshape(-1) for name in parameter_names])
    gradient_norm = float(torch.linalg.vector_norm(flat).item())
    if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise PairV8NegativeBarrierError(
            "active semantic cotangent produced a zero VJP"
        )
    semantic = _sha(semantic_authority_digest, label="semantic authority")
    upstream = _sha(
        upstream_computation_receipt_digest, label="upstream computation receipt"
    )
    checkpoint = _sha(
        checkpoint_content_receipt_digest, label="checkpoint content receipt"
    )
    state = _sha(parameter_state_sha256, label="parameter state")
    root = _sha(panel_root_authority_digest, label="panel root authority")
    row_id = _row_id(role="negative", condition_id=condition, suffix=branch)
    row_authority = derive_row_authority_digest(
        panel_root_authority_digest=root,
        role="negative",
        row_id=row_id,
        semantic_authority_digest=semantic,
        upstream_computation_receipt_digest=upstream,
        checkpoint_content_receipt_digest=checkpoint,
        parameter_state_sha256=state,
    )
    digest = gradient_sha256(gradients)
    parity_equal = bool(
        torch.equal(feature_output.detach(), parity_reference_feature)
    )
    receipt = _seal(
        {
            "schema_version": ROW_MEASUREMENT_SCHEMA,
            "role": "negative",
            "row_id": row_id,
            "condition_id": condition,
            "slot_suffix": branch,
            "negative_branch": branch,
            "measurement_operator": ACTIVE_NEGATIVE_OPERATOR,
            "autograd_grad_outputs_used": True,
            "active_signed_linear_cotangent_vjp_required": True,
            "active_cotangent_nonzero": True,
            "parity_reference_subtracted_in_objective": False,
            "parity_residual_mse_objective_used": False,
            "feature_equals_declared_parity_reference": parity_equal,
            "gradient_values_origin_independently_proven_by_core": True,
            "feature_output_sha256": _tensor_sha256(feature_output.detach()),
            "parity_reference_feature_sha256": _tensor_sha256(
                parity_reference_feature
            ),
            "cotangent_sha256": _tensor_sha256(cotangent),
            "cotangent_norm": cotangent_norm,
            "semantic_authority_digest": semantic,
            "upstream_computation_receipt_digest": upstream,
            "checkpoint_content_receipt_digest": checkpoint,
            "parameter_state_sha256": state,
            "panel_root_authority_digest": root,
            "row_authority_digest": row_authority,
            "parameter_layout": _layout_manifest(gradients),
            "parameter_layout_digest": object_sha256(_layout_manifest(gradients)),
            "gradient_sha256": digest,
            "gradient_norm": gradient_norm,
            "input_numeric_dtype": "torch.float32",
            "solver_numeric_dtype": "torch.float64",
            "device": "cpu",
            "raw_gradient_values_persisted_in_receipt": False,
            "parameter_mutation_performed": False,
        },
        digest_name="receipt_digest",
    )
    return NegativeTangentRow(
        condition_id=condition,
        branch=branch,
        gradient_by_parameter=gradients,
        gradient_sha256=digest,
        row_authority_digest=row_authority,
        semantic_authority_digest=semantic,
        upstream_computation_receipt_digest=upstream,
        checkpoint_content_receipt_digest=checkpoint,
        parameter_state_sha256=state,
        panel_root_authority_digest=root,
        measurement_receipt=receipt,
    )


def _expected_row_receipt_fields(
    row: ActionTangentRow | IdentityTangentRow | NegativeTangentRow,
) -> tuple[str, str, str, str]:
    if isinstance(row, ActionTangentRow):
        role = "action"
        suffix = ""
        operator = EXTERNAL_ACTION_OPERATOR
    elif isinstance(row, IdentityTangentRow):
        role = "identity"
        if type(row.identity_slot) is not int or not 0 <= row.identity_slot < 8:
            raise PairV8NegativeBarrierError("identity slot differs")
        suffix = f"k{row.identity_slot}"
        operator = EXTERNAL_IDENTITY_OPERATOR
    elif isinstance(row, NegativeTangentRow):
        role = "negative"
        if row.branch not in NEGATIVE_BRANCHES:
            raise PairV8NegativeBarrierError("negative branch differs")
        suffix = row.branch
        operator = ACTIVE_NEGATIVE_OPERATOR
    else:
        raise PairV8NegativeBarrierError("tangent row type differs")
    condition = _safe(row.condition_id, label="condition ID")
    return role, suffix, operator, _row_id(
        role=role, condition_id=condition, suffix=suffix
    )


def _validate_row(
    row: ActionTangentRow | IdentityTangentRow | NegativeTangentRow,
) -> Mapping[str, Any]:
    role, suffix, operator, row_id = _expected_row_receipt_fields(row)
    gradients = _validate_gradient_mapping(
        row.gradient_by_parameter, label=row_id, require_detached=True
    )
    digest = gradient_sha256(gradients)
    if _sha(row.gradient_sha256, label=f"{row_id} gradient SHA") != digest:
        raise PairV8NegativeBarrierError(f"{row_id} gradient SHA differs")
    semantic = _sha(row.semantic_authority_digest, label=f"{row_id} semantic authority")
    upstream = _sha(
        row.upstream_computation_receipt_digest,
        label=f"{row_id} upstream receipt",
    )
    checkpoint = _sha(
        row.checkpoint_content_receipt_digest,
        label=f"{row_id} checkpoint receipt",
    )
    state = _sha(row.parameter_state_sha256, label=f"{row_id} parameter state")
    root = _sha(
        row.panel_root_authority_digest, label=f"{row_id} panel root authority"
    )
    expected_authority = derive_row_authority_digest(
        panel_root_authority_digest=root,
        role=role,
        row_id=row_id,
        semantic_authority_digest=semantic,
        upstream_computation_receipt_digest=upstream,
        checkpoint_content_receipt_digest=checkpoint,
        parameter_state_sha256=state,
    )
    if _sha(row.row_authority_digest, label=f"{row_id} row authority") != expected_authority:
        raise PairV8NegativeBarrierError(f"{row_id} row authority differs")
    receipt = _validate_seal(
        row.measurement_receipt,
        schema=ROW_MEASUREMENT_SCHEMA,
        digest_name="receipt_digest",
        label=f"{row_id} measurement receipt",
    )
    expected_common = {
        "role": role,
        "row_id": row_id,
        "condition_id": row.condition_id,
        "slot_suffix": suffix,
        "measurement_operator": operator,
        "active_signed_linear_cotangent_vjp_required": True,
        "parity_residual_mse_objective_used": False,
        "semantic_authority_digest": semantic,
        "upstream_computation_receipt_digest": upstream,
        "checkpoint_content_receipt_digest": checkpoint,
        "parameter_state_sha256": state,
        "panel_root_authority_digest": root,
        "row_authority_digest": expected_authority,
        "parameter_layout": _layout_manifest(gradients),
        "parameter_layout_digest": object_sha256(_layout_manifest(gradients)),
        "gradient_sha256": digest,
        "raw_gradient_values_persisted_in_receipt": False,
        "parameter_mutation_performed": False,
    }
    for name, expected in expected_common.items():
        if receipt.get(name) != expected:
            raise PairV8NegativeBarrierError(
                f"{row_id} measurement field {name} differs"
            )
    norm = float(
        torch.linalg.vector_norm(
            torch.cat([gradients[name].reshape(-1) for name in sorted(gradients)])
        ).item()
    )
    observed_norm = receipt.get("gradient_norm")
    if (
        isinstance(observed_norm, bool)
        or not isinstance(observed_norm, (int, float))
        or not math.isfinite(float(observed_norm))
        or float(observed_norm) != norm
        or norm <= 0.0
    ):
        raise PairV8NegativeBarrierError(f"{row_id} gradient norm differs")
    if role == "negative":
        required = {
            "negative_branch": suffix,
            "autograd_grad_outputs_used": True,
            "active_cotangent_nonzero": True,
            "parity_reference_subtracted_in_objective": False,
            "gradient_values_origin_independently_proven_by_core": True,
            "input_numeric_dtype": "torch.float32",
            "solver_numeric_dtype": "torch.float64",
            "device": "cpu",
        }
        for name, expected in required.items():
            if receipt.get(name) != expected:
                raise PairV8NegativeBarrierError(
                    f"{row_id} active VJP evidence {name} differs"
                )
        _sha(receipt.get("feature_output_sha256"), label=f"{row_id} feature SHA")
        _sha(
            receipt.get("parity_reference_feature_sha256"),
            label=f"{row_id} parity feature SHA",
        )
        _sha(receipt.get("cotangent_sha256"), label=f"{row_id} cotangent SHA")
        cotangent_norm = receipt.get("cotangent_norm")
        if (
            isinstance(cotangent_norm, bool)
            or not isinstance(cotangent_norm, (int, float))
            or not math.isfinite(float(cotangent_norm))
            or float(cotangent_norm) <= 0.0
        ):
            raise PairV8NegativeBarrierError(f"{row_id} cotangent norm differs")
    else:
        if (
            receipt.get("gradient_values_origin_independently_proven_by_core")
            is not False
            or receipt.get("external_measurement_must_be_verified_by_upstream_receipt")
            is not True
        ):
            raise PairV8NegativeBarrierError(f"{row_id} external provenance differs")
    return receipt


def _row_record(
    row: ActionTangentRow | IdentityTangentRow | NegativeTangentRow,
) -> Mapping[str, Any]:
    receipt = _validate_row(row)
    role, suffix, operator, row_id = _expected_row_receipt_fields(row)
    return {
        "role": role,
        "row_id": row_id,
        "condition_id": row.condition_id,
        "slot_suffix": suffix,
        "measurement_operator": operator,
        "gradient_sha256": row.gradient_sha256,
        "row_authority_digest": row.row_authority_digest,
        "semantic_authority_digest": row.semantic_authority_digest,
        "upstream_computation_receipt_digest": (
            row.upstream_computation_receipt_digest
        ),
        "measurement_receipt_digest": receipt["receipt_digest"],
    }


def _panel_closure(
    *,
    action_rows: Sequence[ActionTangentRow],
    identity_rows: Sequence[IdentityTangentRow],
    negative_rows: Sequence[NegativeTangentRow],
) -> tuple[
    tuple[ActionTangentRow, ...],
    tuple[IdentityTangentRow, ...],
    tuple[NegativeTangentRow, ...],
    tuple[str, ...],
    str,
    str,
    str,
]:
    actions = tuple(action_rows)
    identities = tuple(identity_rows)
    negatives = tuple(negative_rows)
    if (
        len(actions) != EXPECTED_ACTION_CONDITION_COUNT
        or len(identities) != EXPECTED_IDENTITY_ROW_COUNT
        or len(negatives) != EXPECTED_NEGATIVE_ROW_COUNT
        or any(not isinstance(row, ActionTangentRow) for row in actions)
        or any(not isinstance(row, IdentityTangentRow) for row in identities)
        or any(not isinstance(row, NegativeTangentRow) for row in negatives)
    ):
        raise PairV8NegativeBarrierError("fixed 8/64/72 row-count closure differs")
    for row in (*actions, *identities, *negatives):
        _validate_row(row)
    conditions = tuple(sorted(row.condition_id for row in actions))
    if len(set(conditions)) != EXPECTED_ACTION_CONDITION_COUNT:
        raise PairV8NegativeBarrierError("action condition IDs are not unique")
    expected_identity = {
        (condition, slot)
        for condition in conditions
        for slot in range(IDENTITY_ROWS_PER_CONDITION)
    }
    observed_identity = {(row.condition_id, row.identity_slot) for row in identities}
    expected_negative = {
        (condition, branch)
        for condition in conditions
        for branch in NEGATIVE_BRANCHES
    }
    observed_negative = {(row.condition_id, row.branch) for row in negatives}
    if observed_identity != expected_identity or len(observed_identity) != len(identities):
        raise PairV8NegativeBarrierError("identity 8-condition x K8 closure differs")
    if observed_negative != expected_negative or len(observed_negative) != len(negatives):
        raise PairV8NegativeBarrierError("negative 8-condition x 9-branch closure differs")
    checkpoints = {
        row.checkpoint_content_receipt_digest
        for row in (*actions, *identities, *negatives)
    }
    states = {row.parameter_state_sha256 for row in (*actions, *identities, *negatives)}
    roots = {
        row.panel_root_authority_digest
        for row in (*actions, *identities, *negatives)
    }
    if len(checkpoints) != 1 or len(states) != 1 or len(roots) != 1:
        raise PairV8NegativeBarrierError("checkpoint/state/root authority consensus differs")
    return (
        actions,
        identities,
        negatives,
        conditions,
        next(iter(checkpoints)),
        next(iter(states)),
        next(iter(roots)),
    )


def build_measurement_authority_receipt(
    *,
    action_rows: Sequence[ActionTangentRow],
    identity_rows: Sequence[IdentityTangentRow],
    negative_rows: Sequence[NegativeTangentRow],
) -> Mapping[str, Any]:
    """Seal the exact externally retained 8/64/72 measurement bank."""

    actions, identities, negatives, conditions, checkpoint, state, root = _panel_closure(
        action_rows=action_rows,
        identity_rows=identity_rows,
        negative_rows=negative_rows,
    )
    records = sorted(
        [_row_record(row) for row in (*actions, *identities, *negatives)],
        key=lambda value: value["row_id"],
    )
    unsigned = {
        "schema_version": MEASUREMENT_AUTHORITY_SCHEMA,
        "method_name": METHOD_NAME,
        "panel_root_authority_digest": root,
        "checkpoint_content_receipt_digest": checkpoint,
        "parameter_state_sha256": state,
        "action_condition_ids": list(conditions),
        "negative_branches": list(NEGATIVE_BRANCHES),
        "action_row_count": len(actions),
        "identity_row_count": len(identities),
        "negative_row_count": len(negatives),
        "barrier_row_count": len(identities) + len(negatives),
        "row_records": records,
        "raw_gradient_values_persisted": False,
        "optimizer_authorized": False,
        "parameter_update_authorized": False,
        "parameter_mutation_performed": False,
    }
    return _seal(unsigned, digest_name="authority_digest")


@dataclass(frozen=True)
class _Layout:
    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    offsets: tuple[tuple[int, int], ...]
    total_numel: int
    manifest: tuple[Mapping[str, Any], ...]
    digest: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, torch.Tensor]) -> "_Layout":
        gradients = _validate_gradient_mapping(
            value, label="layout", require_detached=True
        )
        names = tuple(sorted(gradients))
        shapes: list[tuple[int, ...]] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for name in names:
            shape = tuple(int(item) for item in gradients[name].shape)
            count = int(gradients[name].numel())
            shapes.append(shape)
            offsets.append((cursor, cursor + count))
            cursor += count
        manifest = tuple(_layout_manifest(gradients))
        return cls(
            names=names,
            shapes=tuple(shapes),
            offsets=tuple(offsets),
            total_numel=cursor,
            manifest=manifest,
            digest=object_sha256(list(manifest)),
        )

    def flatten(self, value: Mapping[str, torch.Tensor], *, label: str) -> torch.Tensor:
        gradients = _validate_gradient_mapping(
            value, label=label, require_detached=True
        )
        if tuple(sorted(gradients)) != self.names:
            raise PairV8NegativeBarrierError(f"{label} layout key closure differs")
        flat = torch.empty(self.total_numel, dtype=torch.float64, device="cpu")
        for name, shape, (start, stop) in zip(self.names, self.shapes, self.offsets):
            if tuple(gradients[name].shape) != shape:
                raise PairV8NegativeBarrierError(f"{label} shape closure differs")
            flat[start:stop].copy_(gradients[name].reshape(-1).double())
        return flat

    def unflatten_fp32(self, flat: torch.Tensor) -> Mapping[str, torch.Tensor]:
        if (
            not isinstance(flat, torch.Tensor)
            or flat.device.type != "cpu"
            or flat.dtype != torch.float32
            or flat.ndim != 1
            or flat.numel() != self.total_numel
            or not bool(torch.isfinite(flat).all().item())
        ):
            raise PairV8NegativeBarrierError("prospective FP32 direction differs")
        return {
            name: flat[start:stop].reshape(shape).clone()
            for name, shape, (start, stop) in zip(
                self.names, self.shapes, self.offsets
            )
        }


def _normalized_matrix(
    rows: Sequence[torch.Tensor], *, config: BarrierConfig, label: str
) -> tuple[torch.Tensor, list[float]]:
    unit: list[torch.Tensor] = []
    norms: list[float] = []
    for ordinal, row in enumerate(rows):
        norm = float(torch.linalg.vector_norm(row).item())
        if not math.isfinite(norm) or norm <= config.minimum_row_norm:
            raise PairV8NegativeBarrierError(f"{label}[{ordinal}] VJP norm is too small")
        norms.append(norm)
        unit.append(row / norm)
    return torch.stack(unit, dim=0), norms


def _matrix_rank_audit(
    matrix: torch.Tensor,
    *,
    expected_rank: int,
    label: str,
    config: BarrierConfig,
) -> tuple[Mapping[str, Any], torch.Tensor, torch.Tensor, list[str]]:
    gram = matrix @ matrix.transpose(0, 1)
    gram = 0.5 * (gram + gram.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    largest = max(float(eigenvalues[-1].item()), 0.0)
    threshold = max(
        config.eigenvalue_absolute_tolerance,
        config.singular_value_relative_tolerance**2 * largest,
    )
    retained = eigenvalues > threshold
    rank = int(retained.sum().item())
    if rank:
        smallest = float(eigenvalues[retained][0].item())
        condition = math.sqrt(largest / smallest) if smallest > 0.0 else math.inf
    else:
        condition = math.inf
    failures: list[str] = []
    tag = label.upper().replace("-", "_")
    if rank != expected_rank:
        failures.append(f"{tag}_RANK_NOT_{expected_rank}")
    if not math.isfinite(condition) or condition > config.maximum_effective_condition_number:
        failures.append(f"{tag}_ILL_CONDITIONED")
    receipt = {
        "panel": label,
        "row_count": int(matrix.shape[0]),
        "column_count": int(matrix.shape[1]),
        "expected_full_row_rank": expected_rank,
        "effective_rank": rank,
        "eigenvalue_retention_threshold": threshold,
        "effective_condition_number": condition,
        "maximum_effective_condition_number": config.maximum_effective_condition_number,
        "compact_gram_sha256": _tensor_sha256(gram),
        "compact_gram_eigenvalues": [float(item) for item in eigenvalues.tolist()],
        "passed": not failures,
    }
    return receipt, eigenvalues, eigenvectors, failures


def _remove_rowspace(
    vector: torch.Tensor,
    matrix: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    *,
    config: BarrierConfig,
) -> torch.Tensor:
    largest = max(float(eigenvalues[-1].item()), 0.0)
    threshold = max(
        config.eigenvalue_absolute_tolerance,
        config.singular_value_relative_tolerance**2 * largest,
    )
    retained = eigenvalues > threshold
    if not bool(retained.any().item()):
        return vector.clone()
    values = eigenvalues[retained]
    vectors = eigenvectors[:, retained]
    correlations = matrix @ vector
    coefficients = vectors @ ((vectors.transpose(0, 1) @ correlations) / values)
    projected = vector - matrix.transpose(0, 1) @ coefficients
    return projected


def _barrier_row_audits(
    *,
    rows: Sequence[tuple[str, torch.Tensor]],
    safe: torch.Tensor,
    config: BarrierConfig,
    failure_prefix: str,
) -> tuple[list[Mapping[str, Any]], list[str]]:
    safe_norm = float(torch.linalg.vector_norm(safe).item())
    receipts: list[Mapping[str, Any]] = []
    failures: list[str] = []
    for row_id, row in rows:
        norm = float(torch.linalg.vector_norm(row).item())
        dot = float(torch.dot(row, safe).item())
        absolute_dot = abs(dot)
        scale = norm * safe_norm
        tolerance = config.barrier_dot_absolute_tolerance + (
            config.barrier_dot_relative_tolerance * scale
        )
        cosine = absolute_dot / scale if scale > 0.0 else math.inf
        passed = absolute_dot <= tolerance and cosine <= config.maximum_barrier_cosine
        if not passed:
            failures.append(f"{failure_prefix}_TANGENT_LEAK:{row_id}")
        receipts.append(
            {
                "row_id": row_id,
                "row_norm": norm,
                "dot_with_prospective_direction": dot,
                "absolute_dot_with_prospective_direction": absolute_dot,
                "absolute_dot_tolerance": tolerance,
                "absolute_cosine_with_prospective_direction": cosine,
                "maximum_absolute_cosine": config.maximum_barrier_cosine,
                "passed": passed,
            }
        )
    return receipts, failures


def _action_row_audits(
    *,
    rows: Sequence[tuple[str, torch.Tensor]],
    safe: torch.Tensor,
    config: BarrierConfig,
) -> tuple[list[Mapping[str, Any]], list[str]]:
    safe_norm = float(torch.linalg.vector_norm(safe).item())
    receipts: list[Mapping[str, Any]] = []
    failures: list[str] = []
    for row_id, row in rows:
        norm = float(torch.linalg.vector_norm(row).item())
        dot = float(torch.dot(row, safe).item())
        cosine = dot / (norm * safe_norm) if norm > 0.0 and safe_norm > 0.0 else -math.inf
        passed = (
            dot > config.minimum_action_descent_gain
            and cosine >= config.minimum_action_descent_cosine
        )
        if not passed:
            failures.append(f"ACTION_DESCENT_FAILED:{row_id}")
        receipts.append(
            {
                "row_id": row_id,
                "gradient_norm": norm,
                "dot_with_prospective_gradient": dot,
                "descent_cosine": cosine,
                "minimum_descent_gain": config.minimum_action_descent_gain,
                "minimum_descent_cosine": config.minimum_action_descent_cosine,
                "gradient_descent_convention": "theta_next=theta-lr*prospective_gradient",
                "passed": passed,
            }
        )
    return receipts, failures


def audit_active_semantic_negative_tangent_barrier(
    *,
    action_rows: Sequence[ActionTangentRow],
    identity_rows: Sequence[IdentityTangentRow],
    negative_rows: Sequence[NegativeTangentRow],
    measurement_authority_receipt: Mapping[str, Any],
    expected_measurement_authority_digest: str,
    config: BarrierConfig = BarrierConfig(),
) -> BarrierResult:
    """Project the common action gradient through the sealed 64+72 barrier."""

    config.validate()
    actions, identities, negatives, conditions, checkpoint, state, root = _panel_closure(
        action_rows=action_rows,
        identity_rows=identity_rows,
        negative_rows=negative_rows,
    )
    authority = _validate_seal(
        measurement_authority_receipt,
        schema=MEASUREMENT_AUTHORITY_SCHEMA,
        digest_name="authority_digest",
        label="measurement authority",
    )
    expected_digest = _sha(
        expected_measurement_authority_digest,
        label="expected measurement authority digest",
    )
    if authority["authority_digest"] != expected_digest:
        raise PairV8NegativeBarrierError("measurement authority digest differs")
    rebuilt = build_measurement_authority_receipt(
        action_rows=actions,
        identity_rows=identities,
        negative_rows=negatives,
    )
    if authority != rebuilt:
        raise PairV8NegativeBarrierError(
            "live tangent values differ from sealed measurement authority"
        )

    layout = _Layout.from_mapping(actions[0].gradient_by_parameter)
    action_vectors = [
        layout.flatten(row.gradient_by_parameter, label=f"action:{row.condition_id}")
        for row in sorted(actions, key=lambda item: item.condition_id)
    ]
    identity_sorted = sorted(
        identities, key=lambda item: (item.condition_id, item.identity_slot)
    )
    identity_vectors = [
        layout.flatten(
            row.gradient_by_parameter,
            label=f"identity:{row.condition_id}:k{row.identity_slot}",
        )
        for row in identity_sorted
    ]
    negative_sorted = sorted(
        negatives,
        key=lambda item: (item.condition_id, NEGATIVE_BRANCHES.index(item.branch)),
    )
    negative_vectors = [
        layout.flatten(
            row.gradient_by_parameter,
            label=f"negative:{row.condition_id}:{row.branch}",
        )
        for row in negative_sorted
    ]
    action_matrix, _ = _normalized_matrix(
        action_vectors, config=config, label="action"
    )
    identity_matrix, _ = _normalized_matrix(
        identity_vectors, config=config, label="identity"
    )
    negative_matrix, _ = _normalized_matrix(
        negative_vectors, config=config, label="negative"
    )
    barrier_matrix = torch.cat((identity_matrix, negative_matrix), dim=0)

    action_rank, _, _, action_rank_failures = _matrix_rank_audit(
        action_matrix,
        expected_rank=EXPECTED_ACTION_CONDITION_COUNT,
        label="action-panel",
        config=config,
    )
    identity_rank, _, _, identity_rank_failures = _matrix_rank_audit(
        identity_matrix,
        expected_rank=EXPECTED_IDENTITY_ROW_COUNT,
        label="identity-panel",
        config=config,
    )
    negative_rank, _, _, negative_rank_failures = _matrix_rank_audit(
        negative_matrix,
        expected_rank=EXPECTED_NEGATIVE_ROW_COUNT,
        label="negative-panel",
        config=config,
    )
    barrier_rank, barrier_eigenvalues, barrier_eigenvectors, barrier_rank_failures = (
        _matrix_rank_audit(
            barrier_matrix,
            expected_rank=EXPECTED_BARRIER_ROW_COUNT,
            label="joint-barrier-panel",
            config=config,
        )
    )

    common_action = action_matrix.mean(dim=0)
    prospective64 = _remove_rowspace(
        common_action,
        barrier_matrix,
        barrier_eigenvalues,
        barrier_eigenvectors,
        config=config,
    )
    prospective32 = prospective64.float()
    for _ in range(config.fp32_refinement_passes):
        prospective64 = _remove_rowspace(
            prospective32.double(),
            barrier_matrix,
            barrier_eigenvalues,
            barrier_eigenvectors,
            config=config,
        )
        prospective32 = prospective64.float()
    audited = prospective32.double()
    safe_norm = float(torch.linalg.vector_norm(audited).item())

    action_ids = [f"action:{row.condition_id}" for row in sorted(actions, key=lambda item: item.condition_id)]
    identity_ids = [
        f"identity:{row.condition_id}:k{row.identity_slot}" for row in identity_sorted
    ]
    negative_ids = [
        f"negative:{row.condition_id}:{row.branch}" for row in negative_sorted
    ]
    action_audits, action_failures = _action_row_audits(
        rows=list(zip(action_ids, action_vectors)), safe=audited, config=config
    )
    identity_audits, identity_failures = _barrier_row_audits(
        rows=list(zip(identity_ids, identity_vectors)),
        safe=audited,
        config=config,
        failure_prefix="IDENTITY",
    )
    negative_audits, negative_failures = _barrier_row_audits(
        rows=list(zip(negative_ids, negative_vectors)),
        safe=audited,
        config=config,
        failure_prefix="NEGATIVE",
    )
    failures = sorted(
        set(
            action_rank_failures
            + identity_rank_failures
            + negative_rank_failures
            + barrier_rank_failures
            + action_failures
            + identity_failures
            + negative_failures
            + (["PROSPECTIVE_DIRECTION_ZERO"] if safe_norm <= config.minimum_row_norm else [])
        )
    )
    passed = not failures
    safe_named = layout.unflatten_fp32(prospective32) if passed else None
    unsigned = {
        "schema_version": BARRIER_RECEIPT_SCHEMA,
        "method_name": METHOD_NAME,
        "geometry_authorized": passed,
        "failure_codes": failures,
        "panel_root_authority_digest": root,
        "measurement_authority_digest": expected_digest,
        "checkpoint_content_receipt_digest": checkpoint,
        "parameter_state_sha256": state,
        "action_condition_ids": list(conditions),
        "negative_branches": list(NEGATIVE_BRANCHES),
        "action_row_count": len(actions),
        "identity_row_count": len(identities),
        "negative_row_count": len(negatives),
        "barrier_row_count": len(identities) + len(negatives),
        "parameter_layout": list(layout.manifest),
        "parameter_layout_digest": layout.digest,
        "parameter_count": layout.total_numel,
        "input_numeric_dtype": "torch.float32",
        "projection_numeric_dtype": "torch.float64",
        "prospective_direction_numeric_dtype": "torch.float32",
        "device": "cpu",
        "algorithm": "fp64-normalized-compact-gram-full-rank-nullspace-projection-fp32-refine-fp64-reaudit",
        "active_negative_operator": ACTIVE_NEGATIVE_OPERATOR,
        "parity_residual_mse_objective_used": False,
        "active_cotangent_required_for_all_72_negative_rows": True,
        "action_rank_audit": action_rank,
        "identity_rank_audit": identity_rank,
        "negative_rank_audit": negative_rank,
        "joint_barrier_rank_audit": barrier_rank,
        "prospective_direction_norm": safe_norm,
        "prospective_direction_sha256": (
            gradient_sha256(safe_named) if safe_named is not None else None
        ),
        "safe_direction_exposed": safe_named is not None,
        "per_action_descent": action_audits,
        "per_identity_tangent_barrier": identity_audits,
        "per_negative_tangent_barrier": negative_audits,
        "thresholds": {
            name: getattr(config, name)
            for name in config.__dataclass_fields__
        },
        "raw_gradient_values_persisted": False,
        "optimizer_created": False,
        "optimizer_authorized": False,
        "parameter_update_authorized": False,
        "parameter_mutation_performed": False,
        "gradient_or_adapter_artifact_written": False,
        "scientific_action_editing_success_claim": False,
    }
    receipt = _seal(unsigned, digest_name="receipt_digest")
    return BarrierResult(
        geometry_authorized=passed,
        safe_direction_by_parameter=safe_named,
        receipt=receipt,
    )


def validate_barrier_receipt(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate a v8 receipt without granting any update authority."""

    receipt = _validate_seal(
        value,
        schema=BARRIER_RECEIPT_SCHEMA,
        digest_name="receipt_digest",
        label="barrier receipt",
    )
    for name in (
        "optimizer_created",
        "optimizer_authorized",
        "parameter_update_authorized",
        "parameter_mutation_performed",
        "gradient_or_adapter_artifact_written",
        "scientific_action_editing_success_claim",
    ):
        if receipt.get(name) is not False:
            raise PairV8NegativeBarrierError(f"barrier receipt {name} differs")
    if receipt.get("safe_direction_exposed") is not receipt.get("geometry_authorized"):
        raise PairV8NegativeBarrierError("barrier receipt exposure gate differs")
    return receipt


__all__ = [
    "ACTIVE_NEGATIVE_OPERATOR",
    "ActionTangentRow",
    "BARRIER_RECEIPT_SCHEMA",
    "BarrierConfig",
    "BarrierResult",
    "EXPECTED_ACTION_CONDITION_COUNT",
    "EXPECTED_BARRIER_ROW_COUNT",
    "EXPECTED_IDENTITY_ROW_COUNT",
    "EXPECTED_NEGATIVE_ROW_COUNT",
    "IdentityTangentRow",
    "MEASUREMENT_AUTHORITY_SCHEMA",
    "NEGATIVE_BRANCHES",
    "NegativeTangentRow",
    "PairV8NegativeBarrierError",
    "audit_active_semantic_negative_tangent_barrier",
    "bind_action_tangent_row",
    "bind_identity_tangent_row",
    "build_measurement_authority_receipt",
    "derive_row_authority_digest",
    "gradient_sha256",
    "measure_active_semantic_negative_vjp",
    "object_sha256",
    "validate_barrier_receipt",
]
