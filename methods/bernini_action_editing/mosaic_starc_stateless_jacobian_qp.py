#!/usr/bin/env python3
"""Stateless Jacobian-QP core for the first MOSAIC-STARC editor step.

This module is a deliberately narrow mathematical/runtime primitive.  It does
not collect Jacobians, run Bernini, train a critic, or decide whether the
scientific gates in the canonical one-step contract have passed.  It accepts
only a hash-bound DP2 x SP4 declaration of already-computed FP32 Jacobian rows;
it does not independently open or authenticate the upstream receipt contents.

For action-score *ascent*, the solved problem is

    maximize_d  mean_i(g_i)^T d - lambda ||d||_2^2

subject to every (not averaged) action lower bound, every two-sided
preservation slab, one global L2 radius, and one radius for every whitelisted
Action-LoRA-B tensor.  The objective is equivalent to projecting
``mean(g)/(2 lambda)`` onto that closed convex intersection.  A fixed-order
CPU/FP64 Dykstra projection is used and the candidate is rounded to FP32 and
re-audited against the original (unguarded) constraints.

The only mutation API in this file uses ``parameter.add_(delta)`` under
``torch.no_grad()``.  It then computes the realized FP32 displacement from
``theta_after - theta_before`` and rechecks all constraints and hashes.  An
optimizer, AdamW, momentum, weight decay, loss scaler, or preconditioner is
not accepted by any API here.

Passing this core is geometry evidence only.  It is not evidence that a
critic is scientific, that an action edit improved, or that a checkpoint may
be retained without the fresh exact81 endpoint gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import re
from typing import Any, Mapping, Sequence

import torch


METHOD_NAME = "bernini-mosaic-starc-stateless-jacobian-qp"
EVIDENCE_UNION_SCHEMA = "bernini-mosaic-starc-dp2-sp4-jacobian-union-v1"
CANDIDATE_RECEIPT_SCHEMA = (
    "bernini-mosaic-starc-stateless-jacobian-qp-candidate-v1"
)
REALIZED_RECEIPT_SCHEMA = (
    "bernini-mosaic-starc-realized-fp32-displacement-audit-v1"
)
RECEIPT_JSON_SCHEMA_RELATIVE_PATH = (
    "schemas/bernini_mosaic_starc_jacobian_qp_v1.schema.json"
)

ACTION_FAMILIES = ("dog", "human")
PRESERVATION_FAMILIES = (
    "identity",
    "camera",
    "background",
    "sharpness",
    "flicker",
    "noop",
)
DP_ARM_ORDER = ACTION_FAMILIES
SP_GLOBAL_RANKS = {
    "dog": (0, 1, 2, 3),
    "human": (4, 5, 6, 7),
}


def _canonical_parameter_names() -> tuple[str, ...]:
    rows: list[str] = []
    for block in range(16):
        rows.append(f"blocks.{block}.attn2.to_q.action_lora_b.weight")
        rows.append(f"blocks.{block}.attn2.to_out.0.action_lora_b.weight")
    return tuple(rows)


CANONICAL_PARAMETER_NAMES = _canonical_parameter_names()
LORA_RANK = 8
LORA_ALPHA = 8.0
LORA_SCALE = LORA_ALPHA / float(LORA_RANK)
HIDDEN_SIZE = 1536
CANONICAL_B_SHAPE = (HIDDEN_SIZE, LORA_RANK)
CANONICAL_A_SHAPE = (LORA_RANK, HIDDEN_SIZE)
CANONICAL_PARAMETER_COUNT = (
    len(CANONICAL_PARAMETER_NAMES) * HIDDEN_SIZE * LORA_RANK
)

_SAFE_ID_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,255}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class MosaicStarcJacobianQPError(RuntimeError):
    """Malformed, unauthenticated, or numerically unsafe input."""


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
        raise MosaicStarcJacobianQPError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _seal(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_digest" in unsigned:
        raise MosaicStarcJacobianQPError("cannot seal an already sealed receipt")
    result = dict(unsigned)
    result["receipt_digest"] = object_sha256(unsigned)
    return result


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MosaicStarcJacobianQPError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise MosaicStarcJacobianQPError(f"{label} is unsafe")
    return value


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MosaicStarcJacobianQPError(f"{label} must be a finite scalar")
    result = float(value)
    if not math.isfinite(result):
        raise MosaicStarcJacobianQPError(f"{label} must be finite")
    return result


def _tensor_sha256(value: torch.Tensor) -> str:
    # Clone gives this logical tensor an exact, offset-zero storage.  Reading
    # the untyped storage avoids the optional NumPy bridge (which may be
    # unavailable or ABI-incompatible in the pinned AUH/local environments).
    tensor = value.detach().to(device="cpu").contiguous().clone()
    header = canonical_json_bytes(
        {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "numel": int(tensor.numel()),
        }
    )
    buffer = io.BytesIO()
    try:
        byte_view = tensor.view(torch.uint8)
        if hasattr(byte_view, "untyped_storage"):
            byte_storage = byte_view.untyped_storage()
        else:
            # PyTorch 1.12 exposes the same raw storage through the legacy
            # typed-storage bridge.  Keep this branch NumPy-free because the
            # pinned runtime may not have a compatible NumPy ABI.
            typed_storage = byte_view.storage()
            byte_storage = (
                typed_storage._untyped()
                if hasattr(typed_storage, "_untyped")
                else typed_storage
            )
        # PyTorch's legacy serializer uses this C++ path for raw storage bytes.
        # ``save_size=False`` omits its native-size prefix, while uint8 makes
        # the element size unambiguous.  This is both NumPy-free and orders of
        # magnitude faster than Python iteration over ``bytes(storage)``.
        byte_storage._write_file(buffer, False, False, 1)
    except Exception as error:
        raise MosaicStarcJacobianQPError(
            "cannot export exact tensor bytes without NumPy"
        ) from error
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\x00")
    digest.update(buffer.getbuffer())
    return digest.hexdigest()


def _validate_materialized_fp32_tensor(
    tensor: Any,
    *,
    label: str,
    allow_requires_grad: bool,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.device.type == "meta":
        raise MosaicStarcJacobianQPError(f"{label} must be a materialized tensor")
    if tensor.dtype != torch.float32:
        raise MosaicStarcJacobianQPError(f"{label} must be exact FP32")
    if tensor.requires_grad and not allow_requires_grad:
        raise MosaicStarcJacobianQPError(f"{label} must be detached")
    if tensor.numel() == 0:
        raise MosaicStarcJacobianQPError(f"{label} must be non-empty")
    if not bool(torch.isfinite(tensor).all().item()):
        raise MosaicStarcJacobianQPError(f"{label} is non-finite")
    return tensor


@dataclass(frozen=True)
class FixedParameterLayout:
    """The closed, canonical ordering of the 32 Action-LoRA-B tensors."""

    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    offsets: tuple[tuple[int, int], ...]
    total_numel: int
    layout_digest: str
    parameter_state_sha256: str

    @classmethod
    def from_ordered_parameters(
        cls,
        ordered_parameters: Sequence[tuple[str, torch.Tensor]],
        *,
        require_exact_zero_b: bool = True,
    ) -> "FixedParameterLayout":
        if isinstance(ordered_parameters, Mapping) or not isinstance(
            ordered_parameters, Sequence
        ):
            raise MosaicStarcJacobianQPError(
                "parameters must be an explicit ordered sequence, not a mapping"
            )
        if len(ordered_parameters) != len(CANONICAL_PARAMETER_NAMES):
            raise MosaicStarcJacobianQPError(
                "parameter layout must contain exactly 32 Action-LoRA-B tensors"
            )
        names: list[str] = []
        shapes: list[tuple[int, ...]] = []
        offsets: list[tuple[int, int]] = []
        state_rows: list[dict[str, Any]] = []
        cursor = 0
        for ordinal, item in enumerate(ordered_parameters):
            if not isinstance(item, tuple) or len(item) != 2:
                raise MosaicStarcJacobianQPError(
                    f"parameter entry {ordinal} must be a (name, tensor) tuple"
                )
            name, tensor = item
            expected = CANONICAL_PARAMETER_NAMES[ordinal]
            if name != expected:
                raise MosaicStarcJacobianQPError(
                    f"parameter ordering differs at {ordinal}: expected {expected}"
                )
            tensor = _validate_materialized_fp32_tensor(
                tensor,
                label=f"parameter {name}",
                allow_requires_grad=True,
            )
            shape = tuple(int(value) for value in tensor.shape)
            if shape != CANONICAL_B_SHAPE:
                raise MosaicStarcJacobianQPError(
                    f"parameter {name} must have canonical shape {CANONICAL_B_SHAPE}"
                )
            if require_exact_zero_b and bool(
                torch.count_nonzero(tensor.detach()).item()
            ):
                raise MosaicStarcJacobianQPError(
                    f"parameter {name} must be exact zero-init LoRA-B"
                )
            stop = cursor + int(tensor.numel())
            names.append(name)
            shapes.append(shape)
            offsets.append((cursor, stop))
            state_rows.append(
                {
                    "name": name,
                    "shape": list(shape),
                    "dtype": "torch.float32",
                    "tensor_sha256": _tensor_sha256(tensor),
                }
            )
            cursor = stop
        if cursor != CANONICAL_PARAMETER_COUNT:
            raise MosaicStarcJacobianQPError(
                "Action-LoRA-B coordinate must contain exactly 393216 FP32 values"
            )
        manifest = [
            {"name": name, "shape": list(shape), "dtype": "torch.float32"}
            for name, shape in zip(names, shapes)
        ]
        return cls(
            names=tuple(names),
            shapes=tuple(shapes),
            offsets=tuple(offsets),
            total_numel=cursor,
            layout_digest=object_sha256(manifest),
            parameter_state_sha256=object_sha256(state_rows),
        )

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {"name": name, "shape": list(shape), "dtype": "torch.float32"}
            for name, shape in zip(self.names, self.shapes)
        ]

    def validate_row(self, value: Any, *, label: str) -> torch.Tensor:
        tensor = _validate_materialized_fp32_tensor(
            value,
            label=label,
            allow_requires_grad=False,
        )
        if tensor.device.type != "cpu":
            raise MosaicStarcJacobianQPError(
                f"{label} must be gathered onto CPU before DP2 union"
            )
        if tensor.ndim != 1 or tensor.numel() != self.total_numel:
            raise MosaicStarcJacobianQPError(
                f"{label} shape differs from the fixed parameter coordinate"
            )
        if not tensor.is_contiguous():
            raise MosaicStarcJacobianQPError(f"{label} must be contiguous")
        return tensor

    def unflatten_cpu_fp32(
        self,
        flat: torch.Tensor,
        *,
        label: str,
    ) -> dict[str, torch.Tensor]:
        tensor = self.validate_row(flat, label=label)
        return {
            name: tensor[start:stop].reshape(shape).clone()
            for name, shape, (start, stop) in zip(
                self.names, self.shapes, self.offsets
            )
        }


def _ordered_parameter_state_sha256(
    layout: FixedParameterLayout,
    ordered_parameters: Sequence[tuple[str, torch.Tensor]],
) -> str:
    if isinstance(ordered_parameters, Mapping) or not isinstance(
        ordered_parameters, Sequence
    ) or len(ordered_parameters) != len(layout.names):
        raise MosaicStarcJacobianQPError("current parameter layout closure differs")
    state_rows: list[dict[str, Any]] = []
    for ordinal, (expected_name, expected_shape) in enumerate(
        zip(layout.names, layout.shapes)
    ):
        item = ordered_parameters[ordinal]
        if not isinstance(item, tuple) or len(item) != 2:
            raise MosaicStarcJacobianQPError(
                f"current parameter entry {ordinal} differs"
            )
        name, tensor = item
        if name != expected_name:
            raise MosaicStarcJacobianQPError(
                f"current parameter ordering differs at {ordinal}"
            )
        tensor = _validate_materialized_fp32_tensor(
            tensor,
            label=f"current parameter {name}",
            allow_requires_grad=True,
        )
        if tuple(int(value) for value in tensor.shape) != expected_shape:
            raise MosaicStarcJacobianQPError(
                f"current parameter {name} shape differs"
            )
        state_rows.append(
            {
                "name": name,
                "shape": list(expected_shape),
                "dtype": "torch.float32",
                "tensor_sha256": _tensor_sha256(tensor),
            }
        )
    return object_sha256(state_rows)


def _canonical_zero_b_state_sha256() -> str:
    zero = torch.zeros(CANONICAL_B_SHAPE, dtype=torch.float32, device="cpu")
    tensor_digest = _tensor_sha256(zero)
    return object_sha256(
        [
            {
                "name": name,
                "shape": list(CANONICAL_B_SHAPE),
                "dtype": "torch.float32",
                "tensor_sha256": tensor_digest,
            }
            for name in CANONICAL_PARAMETER_NAMES
        ]
    )


def _validate_fixed_layout_contract(layout: FixedParameterLayout) -> None:
    if not isinstance(layout, FixedParameterLayout):
        raise MosaicStarcJacobianQPError("fixed parameter layout type differs")
    expected_shapes = (CANONICAL_B_SHAPE,) * len(CANONICAL_PARAMETER_NAMES)
    expected_offsets: list[tuple[int, int]] = []
    cursor = 0
    for _ in CANONICAL_PARAMETER_NAMES:
        stop = cursor + HIDDEN_SIZE * LORA_RANK
        expected_offsets.append((cursor, stop))
        cursor = stop
    expected_manifest = [
        {
            "name": name,
            "shape": list(CANONICAL_B_SHAPE),
            "dtype": "torch.float32",
        }
        for name in CANONICAL_PARAMETER_NAMES
    ]
    if (
        layout.names != CANONICAL_PARAMETER_NAMES
        or layout.shapes != expected_shapes
        or layout.offsets != tuple(expected_offsets)
        or layout.total_numel != CANONICAL_PARAMETER_COUNT
        or layout.layout_digest != object_sha256(expected_manifest)
    ):
        raise MosaicStarcJacobianQPError(
            "fixed layout is not the canonical 32x(1536,8) Action-LoRA-B coordinate"
        )
    if layout.parameter_state_sha256 != _canonical_zero_b_state_sha256():
        raise MosaicStarcJacobianQPError(
            "first-step layout is not bound to byte-exact zero-init LoRA-B"
        )


@dataclass(frozen=True)
class ActionConstraintRow:
    row_id: str
    actor_family: str
    values: torch.Tensor
    minimum_dot: float
    layout_digest: str
    checkpoint_content_receipt_digest: str
    parameter_state_sha256: str
    gradient_computation_receipt_digest: str


@dataclass(frozen=True)
class PreservationConstraintRow:
    row_id: str
    family: str
    values: torch.Tensor
    maximum_absolute_dot: float
    layout_digest: str
    checkpoint_content_receipt_digest: str
    parameter_state_sha256: str
    gradient_computation_receipt_digest: str


@dataclass(frozen=True)
class SPRankEvidence:
    global_rank: int
    action_rows: tuple[ActionConstraintRow, ...]
    preservation_rows: tuple[PreservationConstraintRow, ...]
    rank_evidence_receipt_digest: str


@dataclass(frozen=True)
class DPArmEvidence:
    arm_id: str
    sp_ranks: tuple[SPRankEvidence, ...]


@dataclass(frozen=True)
class DP2SP4Evidence:
    dp_arms: tuple[DPArmEvidence, ...]
    topology_receipt_digest: str


@dataclass(frozen=True)
class LayerTrustRadius:
    """One per-tensor trust radius derived from an effective-weight norm.

    The constraint is on the *effective* weight displacement
    ``||(alpha/rank) delta_B A||_F``, not on naked ``||delta_B||_F``.  The
    absolute bound is ``maximum_relative_delta *
    reference_effective_weight_norm``.  The reference norm and frozen A are
    supplied by a separately authenticated fixed-gauge runtime because a
    zero-initialized LoRA-B tensor is not itself a meaningful denominator.
    """

    parameter_name: str
    fixed_lora_a_parameter_name: str
    fixed_lora_a: torch.Tensor
    maximum_relative_delta: float
    reference_effective_weight_norm: float
    fixed_gauge_receipt_digest: str
    reference_weight_receipt_digest: str

    @property
    def maximum_absolute_delta_norm(self) -> float:
        return float(self.maximum_relative_delta) * float(
            self.reference_effective_weight_norm
        )


@dataclass(frozen=True)
class JacobianQPConfig:
    quadratic_penalty: float = 1.0
    minimum_row_norm: float = 1.0e-12
    maximum_linear_condition_number: float = 1.0e10
    spectrum_relative_tolerance: float = 1.0e-12
    dykstra_max_cycles: int = 20_000
    dykstra_cycle_tolerance: float = 1.0e-12
    dykstra_primal_relative_tolerance: float = 1.0e-10
    dykstra_dual_relative_tolerance: float = 1.0e-10
    dykstra_complementarity_relative_tolerance: float = 1.0e-10
    fp32_interior_guard_fraction: float = 2.0e-6
    active_constraint_tolerance: float = 2.0e-6
    realized_minimum_candidate_cosine: float = 0.99999
    realized_maximum_candidate_relative_error: float = 2.0e-5

    def validate(self) -> None:
        positive = (
            "quadratic_penalty",
            "minimum_row_norm",
            "maximum_linear_condition_number",
            "spectrum_relative_tolerance",
            "dykstra_cycle_tolerance",
            "dykstra_primal_relative_tolerance",
            "dykstra_dual_relative_tolerance",
            "dykstra_complementarity_relative_tolerance",
            "fp32_interior_guard_fraction",
            "active_constraint_tolerance",
            "realized_minimum_candidate_cosine",
            "realized_maximum_candidate_relative_error",
        )
        for name in positive:
            if _finite_float(getattr(self, name), label=name) <= 0.0:
                raise MosaicStarcJacobianQPError(f"{name} must be positive")
        if type(self.dykstra_max_cycles) is not int or not (
            20_000 <= self.dykstra_max_cycles <= 100_000
        ):
            raise MosaicStarcJacobianQPError(
                "dykstra_max_cycles must be an integer in 20000..100000"
            )
        if self.fp32_interior_guard_fraction >= 1.0e-2:
            raise MosaicStarcJacobianQPError(
                "FP32 interior guard must be below one percent"
            )
        if not 0.0 < self.realized_minimum_candidate_cosine <= 1.0:
            raise MosaicStarcJacobianQPError(
                "realized candidate cosine must lie in (0,1]"
            )
        if self.realized_maximum_candidate_relative_error > 1.0e-3:
            raise MosaicStarcJacobianQPError(
                "realized candidate relative-error policy is too loose"
            )
        # The v1 numerical policy may be tightened but not loosened through a
        # caller-supplied config.  In particular, a huge cycle tolerance must
        # never turn a one-cycle feasible but non-optimal point into a PASS.
        if self.minimum_row_norm < 1.0e-12:
            raise MosaicStarcJacobianQPError("row-norm policy may not be loosened")
        if self.maximum_linear_condition_number > 1.0e10:
            raise MosaicStarcJacobianQPError(
                "linear-condition policy may not be loosened"
            )
        if self.spectrum_relative_tolerance > 1.0e-12:
            raise MosaicStarcJacobianQPError(
                "spectrum-retention policy may not be loosened"
            )
        if self.dykstra_cycle_tolerance > 1.0e-12:
            raise MosaicStarcJacobianQPError(
                "Dykstra cycle tolerance may not be loosened"
            )
        if self.dykstra_primal_relative_tolerance > 1.0e-10:
            raise MosaicStarcJacobianQPError(
                "Dykstra primal tolerance may not be loosened"
            )
        if self.dykstra_dual_relative_tolerance > 1.0e-10:
            raise MosaicStarcJacobianQPError(
                "Dykstra dual tolerance may not be loosened"
            )
        if self.dykstra_complementarity_relative_tolerance > 1.0e-10:
            raise MosaicStarcJacobianQPError(
                "Dykstra complementarity tolerance may not be loosened"
            )
        if self.fp32_interior_guard_fraction < 2.0e-6:
            raise MosaicStarcJacobianQPError(
                "FP32 interior guard policy may not be loosened"
            )
        if self.active_constraint_tolerance > 2.0e-6:
            raise MosaicStarcJacobianQPError(
                "active-constraint tolerance policy may not be loosened"
            )
        if self.realized_minimum_candidate_cosine < 0.99999:
            raise MosaicStarcJacobianQPError(
                "realized-cosine policy may not be loosened"
            )
        if self.realized_maximum_candidate_relative_error > 2.0e-5:
            raise MosaicStarcJacobianQPError(
                "realized-relative-error policy may not be loosened"
            )


@dataclass(frozen=True)
class _ValidatedEvidenceUnion:
    action_rows: tuple[ActionConstraintRow, ...]
    preservation_rows: tuple[PreservationConstraintRow, ...]
    checkpoint_content_receipt_digest: str
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class JacobianQPSolution:
    layout: FixedParameterLayout
    evidence: DP2SP4Evidence
    layer_trust_radii: tuple[LayerTrustRadius, ...]
    global_trust_radius: float
    config: JacobianQPConfig
    delta_by_parameter: Mapping[str, torch.Tensor]
    authorized: bool
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class RealizedDisplacementAudit:
    realized_geometry_safe: bool
    rolled_back: bool
    realized_delta_by_parameter: Mapping[str, torch.Tensor]
    receipt: Mapping[str, Any]


def _action_row_payload(
    row: ActionConstraintRow,
    *,
    layout: FixedParameterLayout,
    expected_actor_family: str,
    minimum_row_norm: float,
) -> tuple[dict[str, Any], float]:
    if not isinstance(row, ActionConstraintRow):
        raise MosaicStarcJacobianQPError("action evidence contains an invalid row")
    row_id = _safe_id(row.row_id, label="action row ID")
    if row.actor_family != expected_actor_family:
        raise MosaicStarcJacobianQPError(
            f"action row {row_id} actor family differs from its DP arm"
        )
    value = layout.validate_row(row.values, label=f"action row {row_id}")
    lower = _finite_float(row.minimum_dot, label=f"action row {row_id} lower bound")
    if lower <= 0.0:
        raise MosaicStarcJacobianQPError(
            f"action row {row_id} must have a strictly positive lower bound"
        )
    if row.layout_digest != layout.layout_digest:
        raise MosaicStarcJacobianQPError(f"action row {row_id} layout differs")
    checkpoint = _sha256(
        row.checkpoint_content_receipt_digest,
        label=f"action row {row_id} checkpoint receipt",
    )
    state = _sha256(
        row.parameter_state_sha256,
        label=f"action row {row_id} parameter state",
    )
    computation = _sha256(
        row.gradient_computation_receipt_digest,
        label=f"action row {row_id} computation receipt",
    )
    norm = float(torch.linalg.vector_norm(value.to(dtype=torch.float64)).item())
    if norm <= minimum_row_norm:
        raise MosaicStarcJacobianQPError(
            f"action row {row_id} is zero or below the row-norm floor"
        )
    return (
        {
            "row_id": row_id,
            "actor_family": row.actor_family,
            "minimum_dot": lower,
            "layout_digest": row.layout_digest,
            "checkpoint_content_receipt_digest": checkpoint,
            "parameter_state_sha256": state,
            "gradient_computation_receipt_digest": computation,
            "row_sha256": _tensor_sha256(value),
            "row_norm": norm,
        },
        norm,
    )


def _preservation_row_payload(
    row: PreservationConstraintRow,
    *,
    layout: FixedParameterLayout,
    minimum_row_norm: float,
) -> tuple[dict[str, Any], float]:
    if not isinstance(row, PreservationConstraintRow):
        raise MosaicStarcJacobianQPError(
            "preservation evidence contains an invalid row"
        )
    row_id = _safe_id(row.row_id, label="preservation row ID")
    if row.family not in PRESERVATION_FAMILIES:
        raise MosaicStarcJacobianQPError(
            f"preservation row {row_id} family differs"
        )
    value = layout.validate_row(row.values, label=f"preservation row {row_id}")
    bound = _finite_float(
        row.maximum_absolute_dot,
        label=f"preservation row {row_id} slab bound",
    )
    if bound < 0.0:
        raise MosaicStarcJacobianQPError(
            f"preservation row {row_id} slab bound must be non-negative"
        )
    if row.layout_digest != layout.layout_digest:
        raise MosaicStarcJacobianQPError(
            f"preservation row {row_id} layout differs"
        )
    checkpoint = _sha256(
        row.checkpoint_content_receipt_digest,
        label=f"preservation row {row_id} checkpoint receipt",
    )
    state = _sha256(
        row.parameter_state_sha256,
        label=f"preservation row {row_id} parameter state",
    )
    computation = _sha256(
        row.gradient_computation_receipt_digest,
        label=f"preservation row {row_id} computation receipt",
    )
    norm = float(torch.linalg.vector_norm(value.to(dtype=torch.float64)).item())
    if norm <= minimum_row_norm:
        raise MosaicStarcJacobianQPError(
            f"preservation row {row_id} is zero or below the row-norm floor"
        )
    return (
        {
            "row_id": row_id,
            "family": row.family,
            "maximum_absolute_dot": bound,
            "layout_digest": row.layout_digest,
            "checkpoint_content_receipt_digest": checkpoint,
            "parameter_state_sha256": state,
            "gradient_computation_receipt_digest": computation,
            "row_sha256": _tensor_sha256(value),
            "row_norm": norm,
        },
        norm,
    )


def _validate_and_union_dp2_sp4_evidence(
    *,
    layout: FixedParameterLayout,
    evidence: DP2SP4Evidence,
    minimum_row_norm: float,
) -> _ValidatedEvidenceUnion:
    if not isinstance(evidence, DP2SP4Evidence):
        raise MosaicStarcJacobianQPError("DP2 x SP4 evidence type differs")
    if type(evidence.dp_arms) is not tuple or len(evidence.dp_arms) != 2:
        raise MosaicStarcJacobianQPError(
            "DP2 evidence must contain exactly two arms in an immutable tuple"
        )
    if any(not isinstance(arm, DPArmEvidence) for arm in evidence.dp_arms):
        raise MosaicStarcJacobianQPError("DP evidence contains an invalid arm")
    topology_digest = _sha256(
        evidence.topology_receipt_digest,
        label="DP2 x SP4 topology receipt",
    )
    if tuple(arm.arm_id for arm in evidence.dp_arms) != DP_ARM_ORDER:
        raise MosaicStarcJacobianQPError(
            "DP arm ordering must be exactly dog then human"
        )

    canonical_action: list[ActionConstraintRow] = []
    canonical_preservation: list[PreservationConstraintRow] = []
    arm_receipts: list[dict[str, Any]] = []
    checkpoint_digest: str | None = None
    state_digest: str | None = None
    all_action_ids: set[str] = set()
    all_preservation_ids: set[str] = set()
    all_rank_receipt_digests: set[str] = set()

    for arm in evidence.dp_arms:
        expected_ranks = SP_GLOBAL_RANKS[arm.arm_id]
        if type(arm.sp_ranks) is not tuple or len(arm.sp_ranks) != 4:
            raise MosaicStarcJacobianQPError(
                f"{arm.arm_id} must contain exactly four SP ranks in an immutable tuple"
            )
        if any(not isinstance(rank, SPRankEvidence) for rank in arm.sp_ranks):
            raise MosaicStarcJacobianQPError(
                f"{arm.arm_id} contains invalid SP evidence"
            )
        if any(type(rank.global_rank) is not int for rank in arm.sp_ranks):
            raise MosaicStarcJacobianQPError(
                f"{arm.arm_id} SP global ranks must be exact integers"
            )
        if tuple(rank.global_rank for rank in arm.sp_ranks) != expected_ranks:
            raise MosaicStarcJacobianQPError(
                f"{arm.arm_id} SP4 global-rank ordering differs"
            )
        rank_payloads: list[dict[str, Any]] = []
        consensus_action_payload: list[dict[str, Any]] | None = None
        consensus_preservation_payload: list[dict[str, Any]] | None = None
        first_rank: SPRankEvidence | None = None
        for rank in arm.sp_ranks:
            if type(rank.action_rows) is not tuple or type(
                rank.preservation_rows
            ) is not tuple:
                raise MosaicStarcJacobianQPError(
                    f"global rank {rank.global_rank} row collections must be immutable tuples"
                )
            rank_digest = _sha256(
                rank.rank_evidence_receipt_digest,
                label=f"global rank {rank.global_rank} evidence receipt",
            )
            if rank_digest in all_rank_receipt_digests:
                raise MosaicStarcJacobianQPError(
                    "WORLD8 rank-evidence receipt digests must be unique"
                )
            all_rank_receipt_digests.add(rank_digest)
            if not rank.action_rows:
                raise MosaicStarcJacobianQPError(
                    f"global rank {rank.global_rank} has no action row"
                )
            if any(
                not isinstance(row, ActionConstraintRow)
                for row in rank.action_rows
            ):
                raise MosaicStarcJacobianQPError(
                    f"global rank {rank.global_rank} contains an invalid action row"
                )
            if any(
                not isinstance(row, PreservationConstraintRow)
                for row in rank.preservation_rows
            ):
                raise MosaicStarcJacobianQPError(
                    f"global rank {rank.global_rank} contains an invalid preservation row"
                )
            action_order = tuple(
                (row.actor_family, row.row_id) for row in rank.action_rows
            )
            if action_order != tuple(sorted(action_order)):
                raise MosaicStarcJacobianQPError(
                    f"global rank {rank.global_rank} action row ordering differs"
                )
            preservation_order = tuple(
                (PRESERVATION_FAMILIES.index(row.family), row.row_id)
                if isinstance(row, PreservationConstraintRow)
                and row.family in PRESERVATION_FAMILIES
                else (-1, "")
                for row in rank.preservation_rows
            )
            if preservation_order != tuple(sorted(preservation_order)):
                raise MosaicStarcJacobianQPError(
                    f"global rank {rank.global_rank} preservation row ordering differs"
                )
            action_payload: list[dict[str, Any]] = []
            for row in rank.action_rows:
                payload, _ = _action_row_payload(
                    row,
                    layout=layout,
                    expected_actor_family=arm.arm_id,
                    minimum_row_norm=minimum_row_norm,
                )
                action_payload.append(payload)
            preservation_payload: list[dict[str, Any]] = []
            family_counts = {family: 0 for family in PRESERVATION_FAMILIES}
            for row in rank.preservation_rows:
                payload, _ = _preservation_row_payload(
                    row,
                    layout=layout,
                    minimum_row_norm=minimum_row_norm,
                )
                preservation_payload.append(payload)
                family_counts[row.family] += 1
            missing = [
                family for family, count in family_counts.items() if count == 0
            ]
            if missing:
                raise MosaicStarcJacobianQPError(
                    f"global rank {rank.global_rank} misses preservation families: {missing}"
                )
            if len({item["row_id"] for item in action_payload}) != len(
                action_payload
            ):
                raise MosaicStarcJacobianQPError(
                    f"global rank {rank.global_rank} repeats an action row ID"
                )
            if len({item["row_id"] for item in preservation_payload}) != len(
                preservation_payload
            ):
                raise MosaicStarcJacobianQPError(
                    f"global rank {rank.global_rank} repeats a preservation row ID"
                )
            if consensus_action_payload is None:
                consensus_action_payload = action_payload
                consensus_preservation_payload = preservation_payload
                first_rank = rank
            elif (
                canonical_json_bytes(action_payload)
                != canonical_json_bytes(consensus_action_payload)
                or canonical_json_bytes(preservation_payload)
                != canonical_json_bytes(consensus_preservation_payload)
            ):
                raise MosaicStarcJacobianQPError(
                    f"{arm.arm_id} SP4 row consensus differs at global rank {rank.global_rank}"
                )
            rank_payloads.append(
                {
                    "global_rank": rank.global_rank,
                    "rank_evidence_receipt_digest": rank_digest,
                    "action_row_set_digest": object_sha256(action_payload),
                    "preservation_row_set_digest": object_sha256(
                        preservation_payload
                    ),
                }
            )
        assert first_rank is not None
        assert consensus_action_payload is not None
        assert consensus_preservation_payload is not None

        for payload in consensus_action_payload + consensus_preservation_payload:
            if checkpoint_digest is None:
                checkpoint_digest = payload[
                    "checkpoint_content_receipt_digest"
                ]
                state_digest = payload["parameter_state_sha256"]
            elif (
                payload["checkpoint_content_receipt_digest"] != checkpoint_digest
                or payload["parameter_state_sha256"] != state_digest
            ):
                raise MosaicStarcJacobianQPError(
                    "DP2 union rows do not share one checkpoint and parameter state"
                )
        if state_digest != layout.parameter_state_sha256:
            raise MosaicStarcJacobianQPError(
                "evidence parameter state differs from the fixed layout snapshot"
            )

        for row in first_rank.action_rows:
            if row.row_id in all_action_ids:
                raise MosaicStarcJacobianQPError(
                    f"DP2 union repeats action row ID {row.row_id}"
                )
            all_action_ids.add(row.row_id)
            canonical_action.append(
                ActionConstraintRow(
                    row_id=row.row_id,
                    actor_family=row.actor_family,
                    values=row.values.detach().clone(),
                    minimum_dot=float(row.minimum_dot),
                    layout_digest=row.layout_digest,
                    checkpoint_content_receipt_digest=(
                        row.checkpoint_content_receipt_digest
                    ),
                    parameter_state_sha256=row.parameter_state_sha256,
                    gradient_computation_receipt_digest=(
                        row.gradient_computation_receipt_digest
                    ),
                )
            )
        for row in first_rank.preservation_rows:
            if row.row_id in all_preservation_ids:
                raise MosaicStarcJacobianQPError(
                    f"DP2 union repeats preservation row ID {row.row_id}"
                )
            all_preservation_ids.add(row.row_id)
            canonical_preservation.append(
                PreservationConstraintRow(
                    row_id=row.row_id,
                    family=row.family,
                    values=row.values.detach().clone(),
                    maximum_absolute_dot=float(row.maximum_absolute_dot),
                    layout_digest=row.layout_digest,
                    checkpoint_content_receipt_digest=(
                        row.checkpoint_content_receipt_digest
                    ),
                    parameter_state_sha256=row.parameter_state_sha256,
                    gradient_computation_receipt_digest=(
                        row.gradient_computation_receipt_digest
                    ),
                )
            )
        arm_receipts.append(
            {
                "arm_id": arm.arm_id,
                "global_ranks": list(expected_ranks),
                "sp_size": 4,
                "sp_consensus": True,
                "action_row_count": len(consensus_action_payload),
                "rank_evidence": rank_payloads,
                "canonical_action_rows": consensus_action_payload,
                "canonical_preservation_rows": consensus_preservation_payload,
            }
        )

    assert checkpoint_digest is not None
    assert state_digest is not None
    action_counts = [item["action_row_count"] for item in arm_receipts]
    if len(set(action_counts)) != 1:
        raise MosaicStarcJacobianQPError(
            "dog/human DP arms must contribute the same number of raw action rows"
        )
    unsigned = {
        "schema_version": EVIDENCE_UNION_SCHEMA,
        "method_name": METHOD_NAME,
        "topology": "WORLD8_DP2_SP4",
        "world_size": 8,
        "dp_size": 2,
        "sp_size": 4,
        "dp_arm_order": list(DP_ARM_ORDER),
        "topology_receipt_digest": topology_digest,
        "parameter_layout_digest": layout.layout_digest,
        "parameter_state_sha256": state_digest,
        "checkpoint_content_receipt_digest": checkpoint_digest,
        "union_mode": "raw_rows_after_per_arm_sp4_consensus_no_local_projection",
        "local_projection_before_dp_union": False,
        "action_rows_averaged_before_constraints": False,
        "arms": arm_receipts,
    }
    return _ValidatedEvidenceUnion(
        action_rows=tuple(canonical_action),
        preservation_rows=tuple(canonical_preservation),
        checkpoint_content_receipt_digest=checkpoint_digest,
        receipt=_seal(unsigned),
    )


def _validate_layer_trust_radii(
    layout: FixedParameterLayout,
    layer_trust_radii: Sequence[LayerTrustRadius],
) -> tuple[LayerTrustRadius, ...]:
    if not isinstance(layer_trust_radii, Sequence) or isinstance(
        layer_trust_radii, Mapping
    ):
        raise MosaicStarcJacobianQPError(
            "per-layer trust radii must be an explicit ordered sequence"
        )
    if len(layer_trust_radii) != len(layout.names):
        raise MosaicStarcJacobianQPError(
            "one per-layer trust radius is required for every parameter tensor"
        )
    validated: list[LayerTrustRadius] = []
    for expected_name, bound in zip(layout.names, layer_trust_radii):
        if not isinstance(bound, LayerTrustRadius):
            raise MosaicStarcJacobianQPError("per-layer trust radius type differs")
        if bound.parameter_name != expected_name:
            raise MosaicStarcJacobianQPError(
                f"per-layer trust radius ordering differs at {expected_name}"
            )
        expected_a_name = expected_name.replace(
            "action_lora_b.weight", "action_lora_a.weight"
        )
        if bound.fixed_lora_a_parameter_name != expected_a_name:
            raise MosaicStarcJacobianQPError(
                f"{expected_name} fixed LoRA-A name differs"
            )
        fixed_a = _validate_materialized_fp32_tensor(
            bound.fixed_lora_a,
            label=f"{expected_name} fixed LoRA-A",
            allow_requires_grad=False,
        )
        if fixed_a.device.type != "cpu" or not fixed_a.is_contiguous():
            raise MosaicStarcJacobianQPError(
                f"{expected_name} fixed LoRA-A must be contiguous CPU FP32"
            )
        if tuple(int(value) for value in fixed_a.shape) != CANONICAL_A_SHAPE:
            raise MosaicStarcJacobianQPError(
                f"{expected_name} fixed LoRA-A must have shape {CANONICAL_A_SHAPE}"
            )
        a_gram = fixed_a.to(torch.float64) @ fixed_a.to(torch.float64).transpose(
            0, 1
        )
        try:
            a_eigenvalues = torch.linalg.eigvalsh(
                0.5 * (a_gram + a_gram.transpose(0, 1))
            )
        except RuntimeError as error:
            raise MosaicStarcJacobianQPError(
                f"{expected_name} fixed LoRA-A Gram EVD failed"
            ) from error
        maximum_a_eigenvalue = float(a_eigenvalues[-1].item())
        minimum_a_eigenvalue = float(a_eigenvalues[0].item())
        if (
            maximum_a_eigenvalue <= 0.0
            or minimum_a_eigenvalue
            <= max(1.0e-14, 1.0e-10 * maximum_a_eigenvalue)
        ):
            raise MosaicStarcJacobianQPError(
                f"{expected_name} fixed LoRA-A must have stable full row rank"
            )
        relative = _finite_float(
            bound.maximum_relative_delta,
            label=f"{expected_name} relative trust radius",
        )
        reference = _finite_float(
            bound.reference_effective_weight_norm,
            label=f"{expected_name} reference effective-weight norm",
        )
        if relative <= 0.0 or reference <= 0.0:
            raise MosaicStarcJacobianQPError(
                f"{expected_name} trust-radius factors must be positive"
            )
        absolute = relative * reference
        if not math.isfinite(absolute) or absolute <= 0.0:
            raise MosaicStarcJacobianQPError(
                f"{expected_name} absolute trust radius is invalid"
            )
        _sha256(
            bound.fixed_gauge_receipt_digest,
            label=f"{expected_name} fixed-gauge receipt",
        )
        _sha256(
            bound.reference_weight_receipt_digest,
            label=f"{expected_name} reference-weight receipt",
        )
        validated.append(
            LayerTrustRadius(
                parameter_name=bound.parameter_name,
                fixed_lora_a_parameter_name=bound.fixed_lora_a_parameter_name,
                fixed_lora_a=fixed_a.detach().clone(),
                maximum_relative_delta=relative,
                reference_effective_weight_norm=reference,
                fixed_gauge_receipt_digest=bound.fixed_gauge_receipt_digest,
                reference_weight_receipt_digest=(
                    bound.reference_weight_receipt_digest
                ),
            )
        )
    if len({item.fixed_gauge_receipt_digest for item in validated}) != 1:
        raise MosaicStarcJacobianQPError(
            "all 32 fixed LoRA-A tensors must share one fixed-gauge receipt"
        )
    return tuple(validated)


def _fixed_lora_a_state_sha256(
    layer_trust_radii: Sequence[LayerTrustRadius],
) -> str:
    return object_sha256(
        [
            {
                "name": item.fixed_lora_a_parameter_name,
                "shape": list(CANONICAL_A_SHAPE),
                "dtype": "torch.float32",
                "tensor_sha256": _tensor_sha256(item.fixed_lora_a),
            }
            for item in layer_trust_radii
        ]
    )


def _linear_spectrum(
    action_rows: Sequence[ActionConstraintRow],
    preservation_rows: Sequence[PreservationConstraintRow],
    *,
    tolerance: float,
) -> tuple[list[float], int, float | None, str]:
    vectors = [
        row.values.to(dtype=torch.float64) for row in action_rows
    ] + [row.values.to(dtype=torch.float64) for row in preservation_rows]
    units = [vector / torch.linalg.vector_norm(vector) for vector in vectors]
    matrix = torch.stack(units, dim=0)
    gram = matrix @ matrix.transpose(0, 1)
    gram = 0.5 * (gram + gram.transpose(0, 1))
    try:
        eigenvalues = torch.linalg.eigvalsh(gram)
    except RuntimeError as error:
        raise MosaicStarcJacobianQPError("linear-row compact Gram EVD failed") from error
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    maximum = float(eigenvalues[-1].item()) if eigenvalues.numel() else 0.0
    threshold = max(1.0e-15, tolerance * maximum)
    retained = eigenvalues[eigenvalues > threshold]
    rank = int(retained.numel())
    condition = None
    if rank:
        condition = math.sqrt(
            float(retained[-1].item() / retained[0].item())
        )
    return (
        [float(value) for value in eigenvalues.tolist()],
        rank,
        condition,
        _tensor_sha256(gram),
    )


def _action_dot_trust_upper_bound(
    *,
    row: ActionConstraintRow,
    layout: FixedParameterLayout,
    layer_trust_radii: Sequence[LayerTrustRadius],
    global_trust_radius: float,
) -> float:
    """Cauchy upper bound under both global-B and effective-weight radii."""

    vector = row.values.to(dtype=torch.float64)
    global_bound = global_trust_radius * float(
        torch.linalg.vector_norm(vector).item()
    )
    independent_layer_bound = 0.0
    for trust, (start, stop) in zip(layer_trust_radii, layout.offsets):
        gradient_b = vector[start:stop].reshape(CANONICAL_B_SHAPE)
        if not bool(torch.count_nonzero(gradient_b).item()):
            continue
        fixed_a = trust.fixed_lora_a.to(dtype=torch.float64)
        a_gram = fixed_a @ fixed_a.transpose(0, 1)
        try:
            dual = torch.linalg.solve(a_gram, gradient_b.transpose(0, 1))
        except RuntimeError as error:
            raise MosaicStarcJacobianQPError(
                f"{trust.parameter_name} effective trust dual solve failed"
            ) from error
        dual_squared = float(
            torch.sum(gradient_b.transpose(0, 1) * dual).item()
        )
        independent_layer_bound += (
            trust.maximum_absolute_delta_norm
            * math.sqrt(max(0.0, dual_squared))
            / LORA_SCALE
        )
    return min(global_bound, independent_layer_bound)


@dataclass
class _LinearProjectionSet:
    kind: str
    constraint_id: str
    row: torch.Tensor
    guarded_bound: float
    row_norm_squared: float
    correction_coefficient: float = 0.0


@dataclass
class _LayerBallProjectionSet:
    constraint_id: str
    start: int
    stop: int
    guarded_radius: float
    a_gram: torch.Tensor
    a_gram_eigenvalues: torch.Tensor
    a_gram_eigenvectors: torch.Tensor
    correction: torch.Tensor


def _effective_weight_delta_norm(
    flat_b: torch.Tensor,
    *,
    a_gram: torch.Tensor,
) -> float:
    b = flat_b.reshape(CANONICAL_B_SHAPE)
    b_gram = b.transpose(0, 1) @ b
    squared = (LORA_SCALE**2) * float(torch.sum(b_gram * a_gram).item())
    return math.sqrt(max(0.0, squared))


def _project_effective_weight_ellipsoid(
    value: torch.Tensor,
    *,
    radius: float,
    a_gram: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
) -> torch.Tensor:
    """Euclidean projection of B onto ||(alpha/r) B A||_F <= radius.

    In the 8-D eigenbasis of ``A A^T``, the KKT solution has columns
    ``Y_j / (1 + mu * scale^2 * lambda_j)``.  A fixed 96-round bisection finds
    the unique non-negative multiplier.  No 1536x1536 effective matrix is
    materialized.
    """

    current_norm = _effective_weight_delta_norm(value, a_gram=a_gram)
    if current_norm <= radius:
        return value
    y = value.reshape(CANONICAL_B_SHAPE)
    y_basis = y @ eigenvectors
    column_energy = torch.sum(y_basis * y_basis, dim=0)

    def norm_squared(multiplier: float) -> float:
        denominator = 1.0 + multiplier * (LORA_SCALE**2) * eigenvalues
        return (LORA_SCALE**2) * float(
            torch.sum(eigenvalues * column_energy / (denominator * denominator)).item()
        )

    radius_squared = radius * radius
    lower = 0.0
    upper = 1.0
    for _ in range(256):
        if norm_squared(upper) <= radius_squared:
            break
        upper *= 2.0
        if not math.isfinite(upper):
            raise MosaicStarcJacobianQPError(
                "effective-weight ellipsoid multiplier did not bracket"
            )
    else:
        raise MosaicStarcJacobianQPError(
            "effective-weight ellipsoid multiplier did not bracket"
        )
    for _ in range(96):
        midpoint = 0.5 * (lower + upper)
        if norm_squared(midpoint) > radius_squared:
            lower = midpoint
        else:
            upper = midpoint
    denominator = 1.0 + upper * (LORA_SCALE**2) * eigenvalues
    projected = (y_basis / denominator) @ eigenvectors.transpose(0, 1)
    return projected.reshape(-1).contiguous()


def _run_deterministic_dykstra(
    *,
    target: torch.Tensor,
    action_rows: Sequence[ActionConstraintRow],
    preservation_rows: Sequence[PreservationConstraintRow],
    layout: FixedParameterLayout,
    layer_trust_radii: Sequence[LayerTrustRadius],
    global_trust_radius: float,
    config: JacobianQPConfig,
) -> tuple[torch.Tensor, int, float, bool, Mapping[str, Any]]:
    """Project target in a fixed cycle order using compact corrections."""

    guard = config.fp32_interior_guard_fraction
    linear_sets: list[_LinearProjectionSet] = []
    for row in action_rows:
        vector = row.values.to(dtype=torch.float64)
        norm = float(torch.linalg.vector_norm(vector).item())
        scale = max(float(row.minimum_dot), norm * global_trust_radius, 1.0e-30)
        guarded = float(row.minimum_dot) + guard * scale
        linear_sets.append(
            _LinearProjectionSet(
                kind="lower",
                constraint_id=f"action:{row.row_id}",
                row=vector,
                guarded_bound=guarded,
                row_norm_squared=float(torch.dot(vector, vector).item()),
            )
        )
    for row in preservation_rows:
        vector = row.values.to(dtype=torch.float64)
        norm = float(torch.linalg.vector_norm(vector).item())
        scale = max(
            float(row.maximum_absolute_dot),
            norm * global_trust_radius,
            1.0e-30,
        )
        guarded = max(
            0.0,
            float(row.maximum_absolute_dot) - guard * scale,
        )
        linear_sets.append(
            _LinearProjectionSet(
                kind="slab",
                constraint_id=f"preservation:{row.row_id}",
                row=vector,
                guarded_bound=guarded,
                row_norm_squared=float(torch.dot(vector, vector).item()),
            )
        )

    layer_sets: list[_LayerBallProjectionSet] = []
    for bound, (start, stop) in zip(layer_trust_radii, layout.offsets):
        fixed_a = bound.fixed_lora_a.to(dtype=torch.float64)
        a_gram = fixed_a @ fixed_a.transpose(0, 1)
        a_gram = 0.5 * (a_gram + a_gram.transpose(0, 1))
        eigenvalues, eigenvectors = torch.linalg.eigh(a_gram)
        layer_sets.append(
            _LayerBallProjectionSet(
                constraint_id=f"layer:{bound.parameter_name}",
                start=start,
                stop=stop,
                guarded_radius=(
                    bound.maximum_absolute_delta_norm * (1.0 - guard)
                ),
                a_gram=a_gram,
                a_gram_eigenvalues=eigenvalues,
                a_gram_eigenvectors=eigenvectors,
                correction=torch.zeros(stop - start, dtype=torch.float64),
            )
        )
    guarded_global_radius = global_trust_radius * (1.0 - guard)
    global_correction = torch.zeros_like(target)
    x = target.clone()
    last_cycle_change = math.inf
    last_correction_state_change = math.inf
    stationary_failed_cycles = 0

    def correction_state_snapshot() -> tuple[
        tuple[float, ...], tuple[torch.Tensor, ...], torch.Tensor
    ]:
        return (
            tuple(item.correction_coefficient for item in linear_sets),
            tuple(item.correction.clone() for item in layer_sets),
            global_correction.clone(),
        )

    def correction_state_relative_change(
        before: tuple[tuple[float, ...], tuple[torch.Tensor, ...], torch.Tensor],
    ) -> float:
        before_linear, before_layers, before_global = before
        difference_squared = 0.0
        scale_squared = 0.0
        for old, projection in zip(before_linear, linear_sets):
            row_norm_squared = projection.row_norm_squared
            difference_squared += (
                (projection.correction_coefficient - old) ** 2
                * row_norm_squared
            )
            scale_squared += max(
                old * old,
                projection.correction_coefficient
                * projection.correction_coefficient,
            ) * row_norm_squared
        for old, projection in zip(before_layers, layer_sets):
            difference_squared += float(
                torch.dot(
                    projection.correction - old,
                    projection.correction - old,
                ).item()
            )
            scale_squared += max(
                float(torch.dot(old, old).item()),
                float(
                    torch.dot(
                        projection.correction,
                        projection.correction,
                    ).item()
                ),
            )
        difference_squared += float(
            torch.dot(
                global_correction - before_global,
                global_correction - before_global,
            ).item()
        )
        scale_squared += max(
            float(torch.dot(before_global, before_global).item()),
            float(torch.dot(global_correction, global_correction).item()),
        )
        return math.sqrt(max(0.0, difference_squared)) / max(
            1.0, math.sqrt(max(0.0, scale_squared))
        )

    def guarded_active_constraints() -> list[str]:
        active = [
            projection.constraint_id
            for projection in linear_sets
            if projection.correction_coefficient != 0.0
        ]
        active.extend(
            projection.constraint_id
            for projection in layer_sets
            if bool(torch.count_nonzero(projection.correction).item())
        )
        if bool(torch.count_nonzero(global_correction).item()):
            active.append("global_l2")
        return active

    def optimality_certificate() -> dict[str, Any]:
        primal_residuals: list[float] = []
        complementarity_residuals: list[float] = []
        dual_cone_residuals: list[float] = []
        correction_sum = torch.zeros_like(target)
        for projection in linear_sets:
            dot = float(torch.dot(projection.row, x).item())
            scale = max(1.0, abs(projection.guarded_bound))
            if projection.kind == "lower":
                violation = max(0.0, projection.guarded_bound - dot) / scale
                boundary = abs(dot - projection.guarded_bound) / scale
                correction_norm = abs(projection.correction_coefficient) * math.sqrt(
                    projection.row_norm_squared
                )
                dual_cone_residuals.append(
                    max(0.0, projection.correction_coefficient)
                    * math.sqrt(projection.row_norm_squared)
                    / max(1.0, correction_norm)
                )
            else:
                violation = max(0.0, abs(dot) - projection.guarded_bound) / scale
                if projection.correction_coefficient == 0.0:
                    boundary = 0.0
                else:
                    # A slab has two different normal cones.  Merely checking
                    # ``abs(dot) == bound`` can pair an upper-face correction
                    # with the lower face (or vice versa), which is not KKT.
                    signed_face = math.copysign(
                        projection.guarded_bound,
                        projection.correction_coefficient,
                    )
                    boundary = abs(dot - signed_face) / scale
            primal_residuals.append(violation)
            if projection.correction_coefficient != 0.0:
                complementarity_residuals.append(boundary)
            correction_sum.add_(
                projection.row,
                alpha=projection.correction_coefficient,
            )
        for projection in layer_sets:
            piece = x[projection.start : projection.stop]
            norm = _effective_weight_delta_norm(
                piece,
                a_gram=projection.a_gram,
            )
            scale = max(projection.guarded_radius, 1.0e-30)
            primal_residuals.append(
                max(0.0, norm - projection.guarded_radius) / scale
            )
            if bool(torch.count_nonzero(projection.correction).item()):
                complementarity_residuals.append(
                    abs(norm - projection.guarded_radius) / scale
                )
                b = piece.reshape(CANONICAL_B_SHAPE)
                normal = (
                    (LORA_SCALE**2) * (b @ projection.a_gram)
                ).reshape(-1)
                normal_squared = float(torch.dot(normal, normal).item())
                correction_norm = float(
                    torch.linalg.vector_norm(projection.correction).item()
                )
                if normal_squared <= 0.0:
                    dual_cone_residuals.append(correction_norm)
                else:
                    dot_normal = float(
                        torch.dot(projection.correction, normal).item()
                    )
                    multiplier = max(0.0, dot_normal / normal_squared)
                    cone_difference = projection.correction - multiplier * normal
                    dual_cone_residuals.append(
                        max(
                            max(0.0, -dot_normal)
                            / max(
                                1.0,
                                correction_norm * math.sqrt(normal_squared),
                            ),
                            float(
                                torch.linalg.vector_norm(cone_difference).item()
                            )
                            / max(1.0, correction_norm),
                        )
                    )
            correction_sum[
                projection.start : projection.stop
            ].add_(projection.correction)
        global_norm = float(torch.linalg.vector_norm(x).item())
        global_scale = max(guarded_global_radius, 1.0e-30)
        primal_residuals.append(
            max(0.0, global_norm - guarded_global_radius) / global_scale
        )
        if bool(torch.count_nonzero(global_correction).item()):
            complementarity_residuals.append(
                abs(global_norm - guarded_global_radius) / global_scale
            )
            normal_squared = float(torch.dot(x, x).item())
            correction_norm = float(
                torch.linalg.vector_norm(global_correction).item()
            )
            if normal_squared <= 0.0:
                dual_cone_residuals.append(correction_norm)
            else:
                dot_normal = float(torch.dot(global_correction, x).item())
                multiplier = max(0.0, dot_normal / normal_squared)
                cone_difference = global_correction - multiplier * x
                dual_cone_residuals.append(
                    max(
                        max(0.0, -dot_normal)
                        / max(
                            1.0,
                            correction_norm * math.sqrt(normal_squared),
                        ),
                        float(torch.linalg.vector_norm(cone_difference).item())
                        / max(1.0, correction_norm),
                    )
                )
        correction_sum.add_(global_correction)
        stationarity = target - x - correction_sum
        dual_denominator = max(
            1.0,
            float(torch.linalg.vector_norm(target - x).item()),
            float(torch.linalg.vector_norm(correction_sum).item()),
        )
        return {
            "guarded_primal_max_relative_violation": max(
                primal_residuals, default=0.0
            ),
            "dykstra_dual_balance_relative_residual": float(
                torch.linalg.vector_norm(stationarity).item()
            )
            / dual_denominator,
            "dykstra_correction_state_relative_residual": (
                last_correction_state_change
            ),
            "guarded_complementarity_max_relative_residual": max(
                complementarity_residuals, default=0.0
            ),
            "guarded_dual_cone_max_relative_residual": max(
                dual_cone_residuals, default=0.0
            ),
            "guarded_active_constraints": guarded_active_constraints(),
            "dual_correction_sum_sha256": _tensor_sha256(
                correction_sum.to(dtype=torch.float32)
            ),
        }

    for cycle in range(1, config.dykstra_max_cycles + 1):
        cycle_start = x.clone()
        correction_state_start = correction_state_snapshot()
        for projection in linear_sets:
            y = x + projection.correction_coefficient * projection.row
            dot = float(torch.dot(projection.row, y).item())
            if projection.kind == "lower":
                if dot < projection.guarded_bound:
                    step = (
                        projection.guarded_bound - dot
                    ) / projection.row_norm_squared
                    x = y + step * projection.row
                    projection.correction_coefficient = -step
                else:
                    x = y
                    projection.correction_coefficient = 0.0
            else:
                if dot > projection.guarded_bound:
                    step = (
                        dot - projection.guarded_bound
                    ) / projection.row_norm_squared
                    x = y - step * projection.row
                    projection.correction_coefficient = step
                elif dot < -projection.guarded_bound:
                    step = (
                        -projection.guarded_bound - dot
                    ) / projection.row_norm_squared
                    x = y + step * projection.row
                    projection.correction_coefficient = -step
                else:
                    x = y
                    projection.correction_coefficient = 0.0
        for projection in layer_sets:
            y = x[projection.start : projection.stop] + projection.correction
            norm = _effective_weight_delta_norm(
                y,
                a_gram=projection.a_gram,
            )
            if norm > projection.guarded_radius:
                projected = _project_effective_weight_ellipsoid(
                    y,
                    radius=projection.guarded_radius,
                    a_gram=projection.a_gram,
                    eigenvalues=projection.a_gram_eigenvalues,
                    eigenvectors=projection.a_gram_eigenvectors,
                )
                projection.correction = y - projected
                x[projection.start : projection.stop] = projected
            else:
                x[projection.start : projection.stop] = y
                projection.correction.zero_()
        y_global = x + global_correction
        global_norm = float(torch.linalg.vector_norm(y_global).item())
        if global_norm > guarded_global_radius:
            projected = y_global * (guarded_global_radius / global_norm)
            global_correction = y_global - projected
            x = projected
        else:
            x = y_global
            global_correction.zero_()
        last_cycle_change = float(
            torch.linalg.vector_norm(x - cycle_start).item()
        )
        last_correction_state_change = correction_state_relative_change(
            correction_state_start
        )
        if (
            last_cycle_change
            <= config.dykstra_cycle_tolerance
            * max(1.0, float(torch.linalg.vector_norm(x).item()))
            and last_correction_state_change
            <= config.dykstra_dual_relative_tolerance
        ):
            certificate = optimality_certificate()
            certificate_passed = (
                certificate["guarded_primal_max_relative_violation"]
                <= config.dykstra_primal_relative_tolerance
                and certificate["dykstra_dual_balance_relative_residual"]
                <= config.dykstra_dual_relative_tolerance
                and certificate[
                    "dykstra_correction_state_relative_residual"
                ]
                <= config.dykstra_dual_relative_tolerance
                and certificate[
                    "guarded_complementarity_max_relative_residual"
                ]
                <= config.dykstra_complementarity_relative_tolerance
                and certificate[
                    "guarded_dual_cone_max_relative_residual"
                ]
                <= config.dykstra_dual_relative_tolerance
            )
            if certificate_passed:
                return x, cycle, last_cycle_change, True, certificate
            stationary_failed_cycles += 1
            if stationary_failed_cycles >= 8:
                certificate = {
                    **certificate,
                    "stationary_infeasible_or_unresolved_cycles": (
                        stationary_failed_cycles
                    ),
                }
                return x, cycle, last_cycle_change, False, certificate
        else:
            stationary_failed_cycles = 0
    certificate = optimality_certificate()
    return (
        x,
        config.dykstra_max_cycles,
        last_cycle_change,
        False,
        certificate,
    )


def _constraint_audit(
    *,
    flat: torch.Tensor,
    action_rows: Sequence[ActionConstraintRow],
    preservation_rows: Sequence[PreservationConstraintRow],
    layout: FixedParameterLayout,
    layer_trust_radii: Sequence[LayerTrustRadius],
    global_trust_radius: float,
    active_tolerance: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[str],
]:
    vector = flat.to(dtype=torch.float64)
    failures: list[str] = []
    action_audit: list[dict[str, Any]] = []
    for row in action_rows:
        dot = float(torch.dot(row.values.to(dtype=torch.float64), vector).item())
        lower = float(row.minimum_dot)
        slack = dot - lower
        passed = dot >= lower
        if not passed:
            failures.append(f"ACTION_LOWER_BOUND_FAILED:{row.row_id}")
        action_audit.append(
            {
                "row_id": row.row_id,
                "actor_family": row.actor_family,
                "row_sha256": _tensor_sha256(row.values),
                "row_norm": float(
                    torch.linalg.vector_norm(
                        row.values.to(dtype=torch.float64)
                    ).item()
                ),
                "minimum_dot": lower,
                "actual_dot": dot,
                "slack": slack,
                "survival_over_required_lower": dot / lower,
                "active": abs(slack) <= active_tolerance,
                "passed": passed,
                "gradient_computation_receipt_digest": (
                    row.gradient_computation_receipt_digest
                ),
            }
        )
    preservation_audit: list[dict[str, Any]] = []
    for row in preservation_rows:
        dot = float(torch.dot(row.values.to(dtype=torch.float64), vector).item())
        bound = float(row.maximum_absolute_dot)
        slack = bound - abs(dot)
        passed = abs(dot) <= bound
        if not passed:
            failures.append(f"PRESERVATION_SLAB_FAILED:{row.row_id}")
        preservation_audit.append(
            {
                "row_id": row.row_id,
                "family": row.family,
                "row_sha256": _tensor_sha256(row.values),
                "row_norm": float(
                    torch.linalg.vector_norm(
                        row.values.to(dtype=torch.float64)
                    ).item()
                ),
                "maximum_absolute_dot": bound,
                "actual_dot": dot,
                "absolute_dot": abs(dot),
                "slack": slack,
                "active": abs(slack) <= active_tolerance,
                "passed": passed,
                "gradient_computation_receipt_digest": (
                    row.gradient_computation_receipt_digest
                ),
            }
        )
    layer_audit: list[dict[str, Any]] = []
    for bound, (start, stop) in zip(layer_trust_radii, layout.offsets):
        fixed_a = bound.fixed_lora_a.to(dtype=torch.float64)
        a_gram = fixed_a @ fixed_a.transpose(0, 1)
        actual_b_norm = float(
            torch.linalg.vector_norm(vector[start:stop]).item()
        )
        actual = _effective_weight_delta_norm(
            vector[start:stop],
            a_gram=a_gram,
        )
        maximum = bound.maximum_absolute_delta_norm
        slack = maximum - actual
        passed = actual <= maximum
        if not passed:
            failures.append(f"PER_LAYER_TRUST_RADIUS_FAILED:{bound.parameter_name}")
        layer_audit.append(
            {
                "parameter_name": bound.parameter_name,
                "fixed_lora_a_parameter_name": (
                    bound.fixed_lora_a_parameter_name
                ),
                "fixed_lora_a_sha256": _tensor_sha256(bound.fixed_lora_a),
                "fixed_gauge_receipt_digest": (
                    bound.fixed_gauge_receipt_digest
                ),
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "lora_scale": LORA_SCALE,
                "maximum_relative_delta": float(bound.maximum_relative_delta),
                "reference_effective_weight_norm": float(
                    bound.reference_effective_weight_norm
                ),
                "reference_weight_receipt_digest": (
                    bound.reference_weight_receipt_digest
                ),
                "maximum_absolute_delta_norm": maximum,
                "actual_lora_b_delta_norm": actual_b_norm,
                "actual_effective_weight_delta_norm": actual,
                "slack": slack,
                "active": abs(slack) <= active_tolerance,
                "passed": passed,
            }
        )
    actual_global = float(torch.linalg.vector_norm(vector).item())
    global_slack = global_trust_radius - actual_global
    global_passed = actual_global <= global_trust_radius
    if not global_passed:
        failures.append("GLOBAL_TRUST_RADIUS_FAILED")
    global_audit = {
        "maximum_delta_norm": global_trust_radius,
        "actual_delta_norm": actual_global,
        "slack": global_slack,
        "active": abs(global_slack) <= active_tolerance,
        "passed": global_passed,
    }
    return (
        action_audit,
        preservation_audit,
        layer_audit,
        global_audit,
        sorted(set(failures)),
    )


def _null_flat(layout: FixedParameterLayout) -> torch.Tensor:
    return torch.zeros(layout.total_numel, dtype=torch.float32, device="cpu")


def solve_stateless_jacobian_qp(
    *,
    layout: FixedParameterLayout,
    evidence: DP2SP4Evidence,
    global_trust_radius: float,
    layer_trust_radii: Sequence[LayerTrustRadius],
    config: JacobianQPConfig = JacobianQPConfig(),
) -> JacobianQPSolution:
    """Solve the authenticated global union QP or return an exact zero delta.

    Structurally invalid evidence raises ``MosaicStarcJacobianQPError``.  A
    well-formed but infeasible or numerically unresolved problem returns an
    unauthorized solution whose FP32 delta bytes are exactly all zero.
    """

    _validate_fixed_layout_contract(layout)
    config.validate()
    global_radius = _finite_float(
        global_trust_radius, label="global trust radius"
    )
    if global_radius <= 0.0:
        raise MosaicStarcJacobianQPError("global trust radius must be positive")
    trust = _validate_layer_trust_radii(layout, layer_trust_radii)
    union = _validate_and_union_dp2_sp4_evidence(
        layout=layout,
        evidence=evidence,
        minimum_row_norm=config.minimum_row_norm,
    )

    eigenvalues, effective_rank, condition, gram_digest = _linear_spectrum(
        union.action_rows,
        union.preservation_rows,
        tolerance=config.spectrum_relative_tolerance,
    )
    failures: list[str] = []
    if condition is None:
        failures.append("LINEAR_CONSTRAINT_BASIS_HAS_ZERO_RANK")
    elif condition > config.maximum_linear_condition_number:
        failures.append("LINEAR_CONSTRAINT_BASIS_ILL_CONDITIONED")
    for row in union.action_rows:
        trust_upper_bound = _action_dot_trust_upper_bound(
            row=row,
            layout=layout,
            layer_trust_radii=trust,
            global_trust_radius=global_radius,
        )
        if float(row.minimum_dot) > trust_upper_bound:
            failures.append(
                f"ACTION_LOWER_EXCEEDS_TRUST_REGION_UPPER_BOUND:{row.row_id}"
            )

    action_matrix = torch.stack(
        [row.values.to(dtype=torch.float64) for row in union.action_rows], dim=0
    )
    mean_action = torch.mean(action_matrix, dim=0)
    mean_action_norm = float(torch.linalg.vector_norm(mean_action).item())
    if mean_action_norm <= config.minimum_row_norm:
        # Opposing families may cancel in the objective even when their hard
        # lower bounds are individually meaningful.  The contract does not
        # authorize choosing an arbitrary tie-break direction.
        failures.append("MEAN_ACTION_OBJECTIVE_ZERO_OR_TOO_SMALL")
    target = mean_action / (2.0 * config.quadratic_penalty)

    converged = False
    cycles = 0
    cycle_residual: float | None = None
    dykstra_certificate: Mapping[str, Any] = {
        "guarded_primal_max_relative_violation": None,
        "dykstra_dual_balance_relative_residual": None,
        "dykstra_correction_state_relative_residual": None,
        "guarded_complementarity_max_relative_residual": None,
        "guarded_dual_cone_max_relative_residual": None,
        "guarded_active_constraints": [],
        "dual_correction_sum_sha256": None,
    }
    candidate_fp32 = _null_flat(layout)
    if not failures:
        (
            projected,
            cycles,
            cycle_residual_value,
            converged,
            dykstra_certificate,
        ) = (
            _run_deterministic_dykstra(
                target=target,
                action_rows=union.action_rows,
                preservation_rows=union.preservation_rows,
                layout=layout,
                layer_trust_radii=trust,
                global_trust_radius=global_radius,
                config=config,
            )
        )
        cycle_residual = cycle_residual_value
        if not converged:
            failures.append("DYKSTRA_OPTIMALITY_CERTIFICATE_FAILED")
        if not bool(torch.isfinite(projected).all().item()):
            failures.append("DYKSTRA_PRODUCED_NONFINITE_DIRECTION")
        else:
            candidate_fp32 = projected.to(dtype=torch.float32).contiguous()

    (
        action_audit,
        preservation_audit,
        layer_audit,
        global_audit,
        audit_failures,
    ) = _constraint_audit(
        flat=candidate_fp32,
        action_rows=union.action_rows,
        preservation_rows=union.preservation_rows,
        layout=layout,
        layer_trust_radii=trust,
        global_trust_radius=global_radius,
        active_tolerance=config.active_constraint_tolerance,
    )
    failures.extend(audit_failures)
    failures = sorted(set(failures))
    authorized = not failures
    if not authorized:
        # All well-formed infeasible/numerically unresolved problems collapse
        # to the same byte-identical FP32 null delta for this layout.
        candidate_fp32 = _null_flat(layout)
        (
            action_audit,
            preservation_audit,
            layer_audit,
            global_audit,
            _,
        ) = _constraint_audit(
            flat=candidate_fp32,
            action_rows=union.action_rows,
            preservation_rows=union.preservation_rows,
            layout=layout,
            layer_trust_radii=trust,
            global_trust_radius=global_radius,
            active_tolerance=config.active_constraint_tolerance,
        )

    objective_linear = float(torch.dot(mean_action, candidate_fp32.to(torch.float64)).item())
    candidate_norm_squared = float(
        torch.dot(
            candidate_fp32.to(torch.float64),
            candidate_fp32.to(torch.float64),
        ).item()
    )
    objective_value = objective_linear - (
        config.quadratic_penalty * candidate_norm_squared
    )
    active_constraints = [
        f"action:{row['row_id']}" for row in action_audit if row["active"]
    ] + [
        f"preservation:{row['row_id']}"
        for row in preservation_audit
        if row["active"]
    ] + [
        f"layer:{row['parameter_name']}" for row in layer_audit if row["active"]
    ]
    if global_audit["active"]:
        active_constraints.append("global_l2")

    thresholds = {
        name: getattr(config, name) for name in config.__dataclass_fields__
    }
    unsigned = {
        "schema_version": CANDIDATE_RECEIPT_SCHEMA,
        "method_name": METHOD_NAME,
        "mathematical_candidate_authorized": authorized,
        "failure_codes": failures,
        "algorithm": (
            "cpu-fp64-fixed-order-dykstra-projection-with-"
            "fp32-interior-guard-and-hard-reaudit"
        ),
        "optimization_problem": {
            "sense": "maximize",
            "objective": "mean_action_dot_d_minus_lambda_l2_squared",
            "quadratic_penalty": config.quadratic_penalty,
            "equivalent_projection_target": "mean_action_divided_by_2lambda",
            "action_constraints": (
                "each_raw_row_dot_d_greater_or_equal_declared_positive_lower"
            ),
            "preservation_constraints": (
                "each_raw_row_absolute_dot_d_less_or_equal_declared_slab"
            ),
            "trust_constraints": (
                "global_delta_B_l2_and_each_effective_"
                "weight_(alpha/rank)_delta_B_A_frobenius_ellipsoid"
            ),
        },
        "parameter_layout": layout.manifest(),
        "parameter_layout_digest": layout.layout_digest,
        "parameter_count": layout.total_numel,
        "pre_step_parameter_state_sha256": layout.parameter_state_sha256,
        "lora_gauge": {
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "scale": LORA_SCALE,
            "b_shape": list(CANONICAL_B_SHAPE),
            "a_shape": list(CANONICAL_A_SHAPE),
            "b_tensor_count": len(layout.names),
            "b_parameter_count": CANONICAL_PARAMETER_COUNT,
            "b_exact_zero_at_evidence_state": True,
            "a_frozen": True,
            "a_parameter_state_sha256": _fixed_lora_a_state_sha256(trust),
            "fixed_gauge_receipt_digest": trust[0].fixed_gauge_receipt_digest,
            "effective_weight_displacement": "(alpha/rank)_times_delta_B_times_frozen_A",
        },
        "checkpoint_content_receipt_digest": (
            union.checkpoint_content_receipt_digest
        ),
        "dp2_sp4_evidence_union_receipt": union.receipt,
        "dp2_sp4_evidence_union_receipt_digest": union.receipt[
            "receipt_digest"
        ],
        "action_rows": action_audit,
        "preservation_rows": preservation_audit,
        "global_trust_radius": global_audit,
        "per_layer_trust_radii": layer_audit,
        "linear_compact_gram_sha256": gram_digest,
        "linear_compact_gram_eigenvalues": eigenvalues,
        "linear_effective_rank": effective_rank,
        "linear_effective_condition_number": condition,
        "mean_action_row_sha256": _tensor_sha256(
            mean_action.to(dtype=torch.float32)
        ),
        "mean_action_row_norm": mean_action_norm,
        "objective_linear_term": objective_linear,
        "objective_quadratic_term": (
            config.quadratic_penalty * candidate_norm_squared
        ),
        "objective_value": objective_value,
        "dykstra_cycles": cycles,
        "dykstra_converged": converged,
        "dykstra_cycle_residual": cycle_residual,
        "active_constraints": active_constraints,
        "dykstra_optimality_certificate": dict(dykstra_certificate),
        "minimum_action_lower_bound_survival_ratio": min(
            row["survival_over_required_lower"] for row in action_audit
        ),
        "actual_fp32_candidate_delta_sha256": _tensor_sha256(candidate_fp32),
        "actual_fp32_candidate_delta_norm": float(
            torch.linalg.vector_norm(candidate_fp32.to(torch.float64)).item()
        ),
        "null_delta_sha256": _tensor_sha256(_null_flat(layout)),
        "infeasible_returns_byte_identical_fp32_null": True,
        "constraint_audit_uses_original_unguarded_bounds": True,
        "deterministic_solver": True,
        "thresholds": thresholds,
        "application_contract": (
            "single_direct_parameter_add_only_then_"
            "realized_displacement_reaudit"
        ),
        "optimizer_step_allowed": False,
        "adamw_momentum_weight_decay_gradscaler_or_preconditioner_allowed": False,
        "local_direct_add_probe_authorized": authorized,
        "runtime_apply_authorized": False,
        "world8_apply_authorized": False,
        "world8_two_phase_prepare_commit_required": True,
        "external_scientific_gate_bundle_required": True,
        "declared_input_receipt_contents_independently_authenticated_by_core": False,
        "checkpoint_retention_authorized": False,
        "fresh_exact81_endpoint_gate_required": True,
        "training_executed": False,
        "scientific_authority": False,
    }
    receipt = _seal(unsigned)
    validate_candidate_receipt_schema(receipt)
    return JacobianQPSolution(
        layout=layout,
        evidence=evidence,
        layer_trust_radii=trust,
        global_trust_radius=global_radius,
        config=config,
        delta_by_parameter=layout.unflatten_cpu_fp32(
            candidate_fp32,
            label="candidate FP32 delta",
        ),
        authorized=authorized,
        receipt=receipt,
    )


def _flatten_named_delta(
    layout: FixedParameterLayout,
    delta_by_parameter: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> torch.Tensor:
    if not isinstance(delta_by_parameter, Mapping) or tuple(
        delta_by_parameter.keys()
    ) != layout.names:
        raise MosaicStarcJacobianQPError(
            f"{label} parameter key ordering/closure differs"
        )
    flat = torch.empty(layout.total_numel, dtype=torch.float32, device="cpu")
    for name, shape, (start, stop) in zip(
        layout.names, layout.shapes, layout.offsets
    ):
        original = _validate_materialized_fp32_tensor(
            delta_by_parameter[name],
            label=f"{label} {name}",
            allow_requires_grad=False,
        )
        if original.device.type != "cpu":
            raise MosaicStarcJacobianQPError(f"{label} {name} must be on CPU")
        if tuple(original.shape) != shape:
            raise MosaicStarcJacobianQPError(f"{label} {name} shape differs")
        tensor = original.reshape(-1).contiguous()
        flat[start:stop].copy_(tensor)
    return flat


_CANDIDATE_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "method_name",
        "mathematical_candidate_authorized",
        "failure_codes",
        "algorithm",
        "optimization_problem",
        "parameter_layout",
        "parameter_layout_digest",
        "parameter_count",
        "pre_step_parameter_state_sha256",
        "lora_gauge",
        "checkpoint_content_receipt_digest",
        "dp2_sp4_evidence_union_receipt",
        "dp2_sp4_evidence_union_receipt_digest",
        "action_rows",
        "preservation_rows",
        "global_trust_radius",
        "per_layer_trust_radii",
        "linear_compact_gram_sha256",
        "linear_compact_gram_eigenvalues",
        "linear_effective_rank",
        "linear_effective_condition_number",
        "mean_action_row_sha256",
        "mean_action_row_norm",
        "objective_linear_term",
        "objective_quadratic_term",
        "objective_value",
        "dykstra_cycles",
        "dykstra_converged",
        "dykstra_cycle_residual",
        "active_constraints",
        "dykstra_optimality_certificate",
        "minimum_action_lower_bound_survival_ratio",
        "actual_fp32_candidate_delta_sha256",
        "actual_fp32_candidate_delta_norm",
        "null_delta_sha256",
        "infeasible_returns_byte_identical_fp32_null",
        "constraint_audit_uses_original_unguarded_bounds",
        "deterministic_solver",
        "thresholds",
        "application_contract",
        "optimizer_step_allowed",
        "adamw_momentum_weight_decay_gradscaler_or_preconditioner_allowed",
        "local_direct_add_probe_authorized",
        "runtime_apply_authorized",
        "world8_apply_authorized",
        "world8_two_phase_prepare_commit_required",
        "external_scientific_gate_bundle_required",
        "declared_input_receipt_contents_independently_authenticated_by_core",
        "checkpoint_retention_authorized",
        "fresh_exact81_endpoint_gate_required",
        "training_executed",
        "scientific_authority",
        "receipt_digest",
    }
)

_REALIZED_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "method_name",
        "candidate_receipt_digest",
        "parameter_layout_digest",
        "checkpoint_content_receipt_digest",
        "pre_step_parameter_state_sha256",
        "fixed_lora_a_state_before_sha256",
        "fixed_lora_a_state_after_sha256",
        "attempted_after_parameter_state_sha256",
        "final_parameter_state_sha256",
        "direct_parameter_add_attempted",
        "direct_parameter_add_exception",
        "actual_fp32_candidate_delta_sha256",
        "realized_fp32_displacement_sha256",
        "candidate_delta_norm",
        "realized_displacement_norm",
        "candidate_realized_difference_norm",
        "candidate_realized_relative_error",
        "candidate_realized_cosine",
        "action_rows",
        "preservation_rows",
        "global_trust_radius",
        "per_layer_trust_radii",
        "failure_codes",
        "realized_displacement_geometry_safe",
        "rolled_back",
        "rollback_byte_identical_to_pre_step",
        "safe_local_probe_also_rolled_back_pending_world8_two_phase_commit",
        "application_mechanism",
        "optimizer_instantiated_or_called_by_this_module",
        "adamw_momentum_weight_decay_gradscaler_or_preconditioner_used",
        "local_probe_only",
        "world8_apply_authorized",
        "world8_two_phase_prepare_commit_required",
        "parameter_update_applied",
        "checkpoint_retention_authorized",
        "fresh_exact81_endpoint_gate_required",
        "training_executed",
        "scientific_authority",
        "receipt_digest",
    }
)

_PARAMETER_LAYOUT_ITEM_KEYS = frozenset({"name", "shape", "dtype"})
_OPTIMIZATION_PROBLEM_KEYS = frozenset(
    {
        "sense",
        "objective",
        "quadratic_penalty",
        "equivalent_projection_target",
        "action_constraints",
        "preservation_constraints",
        "trust_constraints",
    }
)
_LORA_GAUGE_KEYS = frozenset(
    {
        "rank",
        "alpha",
        "scale",
        "b_shape",
        "a_shape",
        "b_tensor_count",
        "b_parameter_count",
        "b_exact_zero_at_evidence_state",
        "a_frozen",
        "a_parameter_state_sha256",
        "fixed_gauge_receipt_digest",
        "effective_weight_displacement",
    }
)
_ACTION_AUDIT_KEYS = frozenset(
    {
        "row_id",
        "actor_family",
        "row_sha256",
        "row_norm",
        "minimum_dot",
        "actual_dot",
        "slack",
        "survival_over_required_lower",
        "active",
        "passed",
        "gradient_computation_receipt_digest",
    }
)
_PRESERVATION_AUDIT_KEYS = frozenset(
    {
        "row_id",
        "family",
        "row_sha256",
        "row_norm",
        "maximum_absolute_dot",
        "actual_dot",
        "absolute_dot",
        "slack",
        "active",
        "passed",
        "gradient_computation_receipt_digest",
    }
)
_LAYER_AUDIT_KEYS = frozenset(
    {
        "parameter_name",
        "fixed_lora_a_parameter_name",
        "fixed_lora_a_sha256",
        "fixed_gauge_receipt_digest",
        "lora_rank",
        "lora_alpha",
        "lora_scale",
        "maximum_relative_delta",
        "reference_effective_weight_norm",
        "reference_weight_receipt_digest",
        "maximum_absolute_delta_norm",
        "actual_lora_b_delta_norm",
        "actual_effective_weight_delta_norm",
        "slack",
        "active",
        "passed",
    }
)
_GLOBAL_AUDIT_KEYS = frozenset(
    {"maximum_delta_norm", "actual_delta_norm", "slack", "active", "passed"}
)
_DYKSTRA_CERTIFICATE_KEYS = frozenset(
    {
        "guarded_primal_max_relative_violation",
        "dykstra_dual_balance_relative_residual",
        "dykstra_correction_state_relative_residual",
        "guarded_complementarity_max_relative_residual",
        "guarded_dual_cone_max_relative_residual",
        "guarded_active_constraints",
        "dual_correction_sum_sha256",
    }
)
_UNION_KEYS = frozenset(
    {
        "schema_version",
        "method_name",
        "topology",
        "world_size",
        "dp_size",
        "sp_size",
        "dp_arm_order",
        "topology_receipt_digest",
        "parameter_layout_digest",
        "parameter_state_sha256",
        "checkpoint_content_receipt_digest",
        "union_mode",
        "local_projection_before_dp_union",
        "action_rows_averaged_before_constraints",
        "arms",
        "receipt_digest",
    }
)
_UNION_ARM_KEYS = frozenset(
    {
        "arm_id",
        "global_ranks",
        "sp_size",
        "sp_consensus",
        "action_row_count",
        "rank_evidence",
        "canonical_action_rows",
        "canonical_preservation_rows",
    }
)
_UNION_RANK_KEYS = frozenset(
    {
        "global_rank",
        "rank_evidence_receipt_digest",
        "action_row_set_digest",
        "preservation_row_set_digest",
    }
)
_UNION_ACTION_ROW_KEYS = frozenset(
    {
        "row_id",
        "actor_family",
        "minimum_dot",
        "layout_digest",
        "checkpoint_content_receipt_digest",
        "parameter_state_sha256",
        "gradient_computation_receipt_digest",
        "row_sha256",
        "row_norm",
    }
)
_UNION_PRESERVATION_ROW_KEYS = frozenset(
    {
        "row_id",
        "family",
        "maximum_absolute_dot",
        "layout_digest",
        "checkpoint_content_receipt_digest",
        "parameter_state_sha256",
        "gradient_computation_receipt_digest",
        "row_sha256",
        "row_norm",
    }
)


def _validate_receipt_seal_and_keys(
    receipt: Mapping[str, Any],
    *,
    expected_keys: frozenset[str],
    label: str,
) -> None:
    if not isinstance(receipt, Mapping) or set(receipt) != expected_keys:
        actual = set(receipt) if isinstance(receipt, Mapping) else set()
        raise MosaicStarcJacobianQPError(
            f"{label} key closure differs: missing={sorted(expected_keys-actual)} "
            f"extra={sorted(actual-expected_keys)}"
        )
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest")
    if (
        not isinstance(declared, str)
        or _SHA256_RE.fullmatch(declared) is None
        or object_sha256(unsigned) != declared
    ):
        raise MosaicStarcJacobianQPError(f"{label} seal differs")


def _validate_failure_codes(value: Any, *, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or value != sorted(set(value))
    ):
        raise MosaicStarcJacobianQPError(
            f"{label} failure-code closure differs"
        )
    return value


def _mapping_with_exact_keys(
    value: Any,
    *,
    keys: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise MosaicStarcJacobianQPError(
            f"{label} key closure differs: missing={sorted(keys-actual)} "
            f"extra={sorted(actual-keys)}"
        )
    return value


def _receipt_finite_number(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
) -> float:
    result = _finite_float(value, label=label)
    if minimum is not None and result < minimum:
        raise MosaicStarcJacobianQPError(
            f"{label} must be at least {minimum}"
        )
    return result


def _same_float(left: float, right: float) -> bool:
    # These values are recomputed from the exact JSON binary64 inputs using
    # the same scalar operation that produced the receipt.  Exact equality is
    # intentional: a re-sealed one-ULP audit edit must not pass as equivalent.
    return left == right


def _validate_unique_string_list(value: Any, *, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise MosaicStarcJacobianQPError(f"{label} string-list closure differs")
    return value


def _validate_parameter_layout_receipt(
    value: Any,
    *,
    declared_digest: str,
) -> None:
    if not isinstance(value, list) or len(value) != len(CANONICAL_PARAMETER_NAMES):
        raise MosaicStarcJacobianQPError(
            "candidate canonical parameter-layout receipt differs"
        )
    expected = [
        {
            "name": name,
            "shape": list(CANONICAL_B_SHAPE),
            "dtype": "torch.float32",
        }
        for name in CANONICAL_PARAMETER_NAMES
    ]
    for ordinal, item in enumerate(value):
        _mapping_with_exact_keys(
            item,
            keys=_PARAMETER_LAYOUT_ITEM_KEYS,
            label=f"parameter-layout item {ordinal}",
        )
    if value != expected or object_sha256(value) != declared_digest:
        raise MosaicStarcJacobianQPError(
            "candidate canonical parameter-layout contents/digest differ"
        )


def _validate_constraint_audit_receipts(
    *,
    action_rows: Any,
    preservation_rows: Any,
    layer_rows: Any,
    global_row: Any,
    active_tolerance: float | None,
) -> None:
    if (
        not isinstance(action_rows, list)
        or len(action_rows) < 2
        or not isinstance(preservation_rows, list)
        or len(preservation_rows) < 12
        or not isinstance(layer_rows, list)
        or len(layer_rows) != len(CANONICAL_PARAMETER_NAMES)
    ):
        raise MosaicStarcJacobianQPError(
            "constraint-audit collection closure differs"
        )
    action_ids: set[str] = set()
    action_families: set[str] = set()
    for ordinal, row in enumerate(action_rows):
        row = _mapping_with_exact_keys(
            row,
            keys=_ACTION_AUDIT_KEYS,
            label=f"action audit {ordinal}",
        )
        row_id = _safe_id(row["row_id"], label=f"action audit {ordinal} row ID")
        if row_id in action_ids or row["actor_family"] not in ACTION_FAMILIES:
            raise MosaicStarcJacobianQPError("action audit identity closure differs")
        action_ids.add(row_id)
        action_families.add(row["actor_family"])
        _sha256(row["row_sha256"], label=f"action audit {row_id} row")
        _sha256(
            row["gradient_computation_receipt_digest"],
            label=f"action audit {row_id} computation",
        )
        row_norm = _receipt_finite_number(
            row["row_norm"], label=f"action audit {row_id} norm", minimum=0.0
        )
        lower = _receipt_finite_number(
            row["minimum_dot"], label=f"action audit {row_id} lower", minimum=0.0
        )
        actual = _receipt_finite_number(
            row["actual_dot"], label=f"action audit {row_id} actual"
        )
        slack = _receipt_finite_number(
            row["slack"], label=f"action audit {row_id} slack"
        )
        survival = _receipt_finite_number(
            row["survival_over_required_lower"],
            label=f"action audit {row_id} survival",
        )
        if row_norm <= 0.0 or lower <= 0.0:
            raise MosaicStarcJacobianQPError("action audit norm/lower differs")
        if (
            not _same_float(slack, actual - lower)
            or not _same_float(survival, actual / lower)
            or type(row["passed"]) is not bool
            or row["passed"] != (actual >= lower)
            or type(row["active"]) is not bool
        ):
            raise MosaicStarcJacobianQPError(
                f"action audit {row_id} arithmetic/policy differs"
            )
        if active_tolerance is not None and row["active"] != (
            abs(slack) <= active_tolerance
        ):
            raise MosaicStarcJacobianQPError(
                f"action audit {row_id} active status differs"
            )
    if action_families != set(ACTION_FAMILIES):
        raise MosaicStarcJacobianQPError("action audit actor closure differs")

    preservation_ids: set[str] = set()
    preservation_families: set[str] = set()
    for ordinal, row in enumerate(preservation_rows):
        row = _mapping_with_exact_keys(
            row,
            keys=_PRESERVATION_AUDIT_KEYS,
            label=f"preservation audit {ordinal}",
        )
        row_id = _safe_id(
            row["row_id"], label=f"preservation audit {ordinal} row ID"
        )
        family = row["family"]
        if row_id in preservation_ids or family not in PRESERVATION_FAMILIES:
            raise MosaicStarcJacobianQPError(
                "preservation audit identity closure differs"
            )
        preservation_ids.add(row_id)
        preservation_families.add(family)
        _sha256(row["row_sha256"], label=f"preservation audit {row_id} row")
        _sha256(
            row["gradient_computation_receipt_digest"],
            label=f"preservation audit {row_id} computation",
        )
        row_norm = _receipt_finite_number(
            row["row_norm"],
            label=f"preservation audit {row_id} norm",
            minimum=0.0,
        )
        bound = _receipt_finite_number(
            row["maximum_absolute_dot"],
            label=f"preservation audit {row_id} bound",
            minimum=0.0,
        )
        actual = _receipt_finite_number(
            row["actual_dot"], label=f"preservation audit {row_id} actual"
        )
        absolute = _receipt_finite_number(
            row["absolute_dot"],
            label=f"preservation audit {row_id} absolute",
            minimum=0.0,
        )
        slack = _receipt_finite_number(
            row["slack"], label=f"preservation audit {row_id} slack"
        )
        if (
            row_norm <= 0.0
            or not _same_float(absolute, abs(actual))
            or not _same_float(slack, bound - absolute)
            or type(row["passed"]) is not bool
            or row["passed"] != (absolute <= bound)
            or type(row["active"]) is not bool
        ):
            raise MosaicStarcJacobianQPError(
                f"preservation audit {row_id} arithmetic/policy differs"
            )
        if active_tolerance is not None and row["active"] != (
            abs(slack) <= active_tolerance
        ):
            raise MosaicStarcJacobianQPError(
                f"preservation audit {row_id} active status differs"
            )
    if preservation_families != set(PRESERVATION_FAMILIES):
        raise MosaicStarcJacobianQPError(
            "preservation audit family closure differs"
        )

    for ordinal, (expected_name, row) in enumerate(
        zip(CANONICAL_PARAMETER_NAMES, layer_rows)
    ):
        row = _mapping_with_exact_keys(
            row,
            keys=_LAYER_AUDIT_KEYS,
            label=f"layer audit {ordinal}",
        )
        expected_a_name = expected_name.replace(
            "action_lora_b.weight", "action_lora_a.weight"
        )
        if (
            row["parameter_name"] != expected_name
            or row["fixed_lora_a_parameter_name"] != expected_a_name
            or row["lora_rank"] != LORA_RANK
            or row["lora_alpha"] != LORA_ALPHA
            or row["lora_scale"] != LORA_SCALE
        ):
            raise MosaicStarcJacobianQPError(
                f"layer audit {ordinal} gauge/order differs"
            )
        for key in (
            "fixed_lora_a_sha256",
            "fixed_gauge_receipt_digest",
            "reference_weight_receipt_digest",
        ):
            _sha256(row[key], label=f"layer audit {ordinal} {key}")
        relative = _receipt_finite_number(
            row["maximum_relative_delta"],
            label=f"layer audit {ordinal} relative radius",
            minimum=0.0,
        )
        reference = _receipt_finite_number(
            row["reference_effective_weight_norm"],
            label=f"layer audit {ordinal} reference norm",
            minimum=0.0,
        )
        maximum = _receipt_finite_number(
            row["maximum_absolute_delta_norm"],
            label=f"layer audit {ordinal} absolute radius",
            minimum=0.0,
        )
        _receipt_finite_number(
            row["actual_lora_b_delta_norm"],
            label=f"layer audit {ordinal} B norm",
            minimum=0.0,
        )
        actual = _receipt_finite_number(
            row["actual_effective_weight_delta_norm"],
            label=f"layer audit {ordinal} effective norm",
            minimum=0.0,
        )
        slack = _receipt_finite_number(
            row["slack"], label=f"layer audit {ordinal} slack"
        )
        if (
            relative <= 0.0
            or reference <= 0.0
            or not _same_float(maximum, relative * reference)
            or not _same_float(slack, maximum - actual)
            or type(row["passed"]) is not bool
            or row["passed"] != (actual <= maximum)
            or type(row["active"]) is not bool
        ):
            raise MosaicStarcJacobianQPError(
                f"layer audit {ordinal} arithmetic/policy differs"
            )
        if active_tolerance is not None and row["active"] != (
            abs(slack) <= active_tolerance
        ):
            raise MosaicStarcJacobianQPError(
                f"layer audit {ordinal} active status differs"
            )

    global_row = _mapping_with_exact_keys(
        global_row,
        keys=_GLOBAL_AUDIT_KEYS,
        label="global trust audit",
    )
    maximum = _receipt_finite_number(
        global_row["maximum_delta_norm"],
        label="global trust maximum",
        minimum=0.0,
    )
    actual = _receipt_finite_number(
        global_row["actual_delta_norm"],
        label="global trust actual",
        minimum=0.0,
    )
    slack = _receipt_finite_number(
        global_row["slack"], label="global trust slack"
    )
    if (
        maximum <= 0.0
        or not _same_float(slack, maximum - actual)
        or type(global_row["passed"]) is not bool
        or global_row["passed"] != (actual <= maximum)
        or type(global_row["active"]) is not bool
    ):
        raise MosaicStarcJacobianQPError(
            "global trust audit arithmetic/policy differs"
        )
    if active_tolerance is not None and global_row["active"] != (
        abs(slack) <= active_tolerance
    ):
        raise MosaicStarcJacobianQPError(
            "global trust audit active status differs"
        )


def _validate_union_receipt(
    value: Any,
    *,
    declared_digest: str,
    parameter_layout_digest: str,
    parameter_state_sha256: str,
    checkpoint_content_receipt_digest: str,
) -> None:
    union = _mapping_with_exact_keys(
        value,
        keys=_UNION_KEYS,
        label="DP2 x SP4 union receipt",
    )
    unsigned = dict(union)
    digest = unsigned.pop("receipt_digest")
    _sha256(digest, label="DP2 x SP4 union receipt digest")
    for key in (
        "topology_receipt_digest",
        "parameter_layout_digest",
        "parameter_state_sha256",
        "checkpoint_content_receipt_digest",
    ):
        _sha256(union[key], label=f"DP2 x SP4 union {key}")
    if (
        digest != declared_digest
        or object_sha256(unsigned) != digest
        or union["schema_version"] != EVIDENCE_UNION_SCHEMA
        or union["method_name"] != METHOD_NAME
        or union["topology"] != "WORLD8_DP2_SP4"
        or type(union["world_size"]) is not int
        or union["world_size"] != 8
        or type(union["dp_size"]) is not int
        or union["dp_size"] != 2
        or type(union["sp_size"]) is not int
        or union["sp_size"] != 4
        or union["dp_arm_order"] != list(DP_ARM_ORDER)
        or union["union_mode"]
        != "raw_rows_after_per_arm_sp4_consensus_no_local_projection"
        or union["local_projection_before_dp_union"] is not False
        or union["action_rows_averaged_before_constraints"] is not False
        or union["parameter_layout_digest"] != parameter_layout_digest
        or union["parameter_state_sha256"] != parameter_state_sha256
        or union["checkpoint_content_receipt_digest"]
        != checkpoint_content_receipt_digest
    ):
        raise MosaicStarcJacobianQPError(
            "candidate DP2 x SP4 union seal/topology/binding differs"
        )
    arms = union["arms"]
    if not isinstance(arms, list) or len(arms) != 2:
        raise MosaicStarcJacobianQPError("DP2 union arm closure differs")
    action_counts: list[int] = []
    rank_receipts: set[str] = set()
    all_action_ids: set[str] = set()
    all_preservation_ids: set[str] = set()
    for arm, expected_arm in zip(arms, DP_ARM_ORDER):
        arm = _mapping_with_exact_keys(
            arm,
            keys=_UNION_ARM_KEYS,
            label=f"DP arm {expected_arm}",
        )
        expected_ranks = list(SP_GLOBAL_RANKS[expected_arm])
        if (
            arm["arm_id"] != expected_arm
            or arm["global_ranks"] != expected_ranks
            or arm["sp_size"] != 4
            or arm["sp_consensus"] is not True
            or type(arm["action_row_count"]) is not int
            or arm["action_row_count"] <= 0
        ):
            raise MosaicStarcJacobianQPError(
                f"DP arm {expected_arm} topology differs"
            )
        action_counts.append(arm["action_row_count"])
        rank_evidence = arm["rank_evidence"]
        if not isinstance(rank_evidence, list) or len(rank_evidence) != 4:
            raise MosaicStarcJacobianQPError(
                f"DP arm {expected_arm} rank closure differs"
            )
        for row, expected_rank in zip(rank_evidence, expected_ranks):
            row = _mapping_with_exact_keys(
                row,
                keys=_UNION_RANK_KEYS,
                label=f"DP arm {expected_arm} rank {expected_rank}",
            )
            if type(row["global_rank"]) is not int or row["global_rank"] != expected_rank:
                raise MosaicStarcJacobianQPError("union rank ordering differs")
            for key in (
                "rank_evidence_receipt_digest",
                "action_row_set_digest",
                "preservation_row_set_digest",
            ):
                _sha256(row[key], label=f"union rank {expected_rank} {key}")
            if row["rank_evidence_receipt_digest"] in rank_receipts:
                raise MosaicStarcJacobianQPError(
                    "union rank receipt digests are not unique"
                )
            rank_receipts.add(row["rank_evidence_receipt_digest"])
        canonical_action = arm["canonical_action_rows"]
        canonical_preservation = arm["canonical_preservation_rows"]
        if (
            not isinstance(canonical_action, list)
            or len(canonical_action) != arm["action_row_count"]
            or not isinstance(canonical_preservation, list)
            or len(canonical_preservation) < len(PRESERVATION_FAMILIES)
        ):
            raise MosaicStarcJacobianQPError(
                f"DP arm {expected_arm} canonical-row closure differs"
            )
        for ordinal, row in enumerate(canonical_action):
            row = _mapping_with_exact_keys(
                row,
                keys=_UNION_ACTION_ROW_KEYS,
                label=f"union action row {expected_arm}:{ordinal}",
            )
            _safe_id(row["row_id"], label="union action row ID")
            if row["row_id"] in all_action_ids:
                raise MosaicStarcJacobianQPError(
                    "union action row IDs are not globally unique"
                )
            all_action_ids.add(row["row_id"])
            if (
                row["actor_family"] != expected_arm
                or row["layout_digest"] != parameter_layout_digest
                or row["checkpoint_content_receipt_digest"]
                != checkpoint_content_receipt_digest
                or row["parameter_state_sha256"] != parameter_state_sha256
                or _receipt_finite_number(
                    row["minimum_dot"],
                    label="union action lower",
                    minimum=0.0,
                )
                <= 0.0
                or _receipt_finite_number(
                    row["row_norm"],
                    label="union action norm",
                    minimum=0.0,
                )
                <= 0.0
            ):
                raise MosaicStarcJacobianQPError(
                    "union action row binding differs"
                )
            for key in (
                "layout_digest",
                "checkpoint_content_receipt_digest",
                "parameter_state_sha256",
                "gradient_computation_receipt_digest",
                "row_sha256",
            ):
                _sha256(row[key], label=f"union action {key}")
        families: set[str] = set()
        for ordinal, row in enumerate(canonical_preservation):
            row = _mapping_with_exact_keys(
                row,
                keys=_UNION_PRESERVATION_ROW_KEYS,
                label=f"union preservation row {expected_arm}:{ordinal}",
            )
            _safe_id(row["row_id"], label="union preservation row ID")
            if row["row_id"] in all_preservation_ids:
                raise MosaicStarcJacobianQPError(
                    "union preservation row IDs are not globally unique"
                )
            all_preservation_ids.add(row["row_id"])
            family = row["family"]
            families.add(family)
            if (
                family not in PRESERVATION_FAMILIES
                or row["layout_digest"] != parameter_layout_digest
                or row["checkpoint_content_receipt_digest"]
                != checkpoint_content_receipt_digest
                or row["parameter_state_sha256"] != parameter_state_sha256
                or _receipt_finite_number(
                    row["maximum_absolute_dot"],
                    label="union preservation bound",
                    minimum=0.0,
                )
                < 0.0
                or _receipt_finite_number(
                    row["row_norm"],
                    label="union preservation norm",
                    minimum=0.0,
                )
                <= 0.0
            ):
                raise MosaicStarcJacobianQPError(
                    "union preservation row binding differs"
                )
            for key in (
                "layout_digest",
                "checkpoint_content_receipt_digest",
                "parameter_state_sha256",
                "gradient_computation_receipt_digest",
                "row_sha256",
            ):
                _sha256(row[key], label=f"union preservation {key}")
        if families != set(PRESERVATION_FAMILIES):
            raise MosaicStarcJacobianQPError(
                f"DP arm {expected_arm} preservation-family closure differs"
            )
        action_order = tuple(
            (row["actor_family"], row["row_id"]) for row in canonical_action
        )
        preservation_order = tuple(
            (PRESERVATION_FAMILIES.index(row["family"]), row["row_id"])
            for row in canonical_preservation
        )
        if action_order != tuple(sorted(action_order)) or (
            preservation_order != tuple(sorted(preservation_order))
        ):
            raise MosaicStarcJacobianQPError(
                f"DP arm {expected_arm} canonical-row ordering differs"
            )
        action_digest = object_sha256(canonical_action)
        preservation_digest = object_sha256(canonical_preservation)
        if any(
            row["action_row_set_digest"] != action_digest
            or row["preservation_row_set_digest"] != preservation_digest
            for row in rank_evidence
        ):
            raise MosaicStarcJacobianQPError(
                f"DP arm {expected_arm} row-set digest differs"
            )
    if len(set(action_counts)) != 1:
        raise MosaicStarcJacobianQPError("DP arm action-row counts differ")


def validate_candidate_receipt_schema(receipt: Mapping[str, Any]) -> None:
    """Strict runtime validation beyond the descriptive JSON Schema file."""

    _validate_receipt_seal_and_keys(
        receipt,
        expected_keys=_CANDIDATE_RECEIPT_KEYS,
        label="candidate receipt",
    )
    if (
        receipt["schema_version"] != CANDIDATE_RECEIPT_SCHEMA
        or receipt["method_name"] != METHOD_NAME
    ):
        raise MosaicStarcJacobianQPError("candidate receipt identity differs")
    authorized = receipt["mathematical_candidate_authorized"]
    if type(authorized) is not bool:
        raise MosaicStarcJacobianQPError(
            "candidate authorization must be an exact boolean"
        )
    failures = _validate_failure_codes(
        receipt["failure_codes"], label="candidate"
    )
    if authorized != (not failures):
        raise MosaicStarcJacobianQPError(
            "candidate authorization/failure consistency differs"
        )
    constant_booleans = {
        "infeasible_returns_byte_identical_fp32_null": True,
        "constraint_audit_uses_original_unguarded_bounds": True,
        "deterministic_solver": True,
        "optimizer_step_allowed": False,
        "adamw_momentum_weight_decay_gradscaler_or_preconditioner_allowed": False,
        "runtime_apply_authorized": False,
        "world8_apply_authorized": False,
        "world8_two_phase_prepare_commit_required": True,
        "external_scientific_gate_bundle_required": True,
        "declared_input_receipt_contents_independently_authenticated_by_core": False,
        "checkpoint_retention_authorized": False,
        "fresh_exact81_endpoint_gate_required": True,
        "training_executed": False,
        "scientific_authority": False,
    }
    for key, expected in constant_booleans.items():
        if receipt[key] is not expected:
            raise MosaicStarcJacobianQPError(
                f"candidate receipt boolean policy differs: {key}"
            )
    if receipt["local_direct_add_probe_authorized"] is not authorized:
        raise MosaicStarcJacobianQPError(
            "candidate local-probe authorization differs"
        )
    if receipt["algorithm"] != (
        "cpu-fp64-fixed-order-dykstra-projection-with-"
        "fp32-interior-guard-and-hard-reaudit"
    ) or receipt["application_contract"] != (
        "single_direct_parameter_add_only_then_"
        "realized_displacement_reaudit"
    ):
        raise MosaicStarcJacobianQPError(
            "candidate algorithm/application contract differs"
        )
    if type(receipt["parameter_count"]) is not int or (
        receipt["parameter_count"] != CANONICAL_PARAMETER_COUNT
    ):
        raise MosaicStarcJacobianQPError(
            "candidate canonical parameter-layout receipt differs"
        )
    for key in (
        "parameter_layout_digest",
        "pre_step_parameter_state_sha256",
        "checkpoint_content_receipt_digest",
        "dp2_sp4_evidence_union_receipt_digest",
        "linear_compact_gram_sha256",
        "mean_action_row_sha256",
        "actual_fp32_candidate_delta_sha256",
        "null_delta_sha256",
        "receipt_digest",
    ):
        _sha256(receipt[key], label=f"candidate {key}")
    _validate_parameter_layout_receipt(
        receipt["parameter_layout"],
        declared_digest=receipt["parameter_layout_digest"],
    )
    if receipt["pre_step_parameter_state_sha256"] != (
        _canonical_zero_b_state_sha256()
    ) or receipt["null_delta_sha256"] != _tensor_sha256(
        torch.zeros(
            CANONICAL_PARAMETER_COUNT,
            dtype=torch.float32,
            device="cpu",
        )
    ):
        raise MosaicStarcJacobianQPError(
            "candidate zero-B/null-delta byte binding differs"
        )

    thresholds = receipt["thresholds"]
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(
        JacobianQPConfig.__dataclass_fields__
    ):
        raise MosaicStarcJacobianQPError("candidate threshold closure differs")
    try:
        threshold_config = JacobianQPConfig(**dict(thresholds))
    except TypeError as error:
        raise MosaicStarcJacobianQPError(
            "candidate threshold types differ"
        ) from error
    threshold_config.validate()

    optimization = _mapping_with_exact_keys(
        receipt["optimization_problem"],
        keys=_OPTIMIZATION_PROBLEM_KEYS,
        label="candidate optimization problem",
    )
    if optimization != {
        "sense": "maximize",
        "objective": "mean_action_dot_d_minus_lambda_l2_squared",
        "quadratic_penalty": threshold_config.quadratic_penalty,
        "equivalent_projection_target": "mean_action_divided_by_2lambda",
        "action_constraints": (
            "each_raw_row_dot_d_greater_or_equal_declared_positive_lower"
        ),
        "preservation_constraints": (
            "each_raw_row_absolute_dot_d_less_or_equal_declared_slab"
        ),
        "trust_constraints": (
            "global_delta_B_l2_and_each_effective_"
            "weight_(alpha/rank)_delta_B_A_frobenius_ellipsoid"
        ),
    }:
        raise MosaicStarcJacobianQPError(
            "candidate optimization-problem contents differ"
        )

    gauge = _mapping_with_exact_keys(
        receipt["lora_gauge"],
        keys=_LORA_GAUGE_KEYS,
        label="candidate LoRA gauge",
    )
    if (
        type(gauge["rank"]) is not int
        or gauge["rank"] != LORA_RANK
        or gauge["alpha"] != LORA_ALPHA
        or gauge["scale"] != LORA_SCALE
        or gauge["b_shape"] != list(CANONICAL_B_SHAPE)
        or gauge["a_shape"] != list(CANONICAL_A_SHAPE)
        or type(gauge["b_tensor_count"]) is not int
        or gauge["b_tensor_count"] != len(CANONICAL_PARAMETER_NAMES)
        or type(gauge["b_parameter_count"]) is not int
        or gauge["b_parameter_count"] != CANONICAL_PARAMETER_COUNT
        or gauge["b_exact_zero_at_evidence_state"] is not True
        or gauge["a_frozen"] is not True
        or gauge["effective_weight_displacement"]
        != "(alpha/rank)_times_delta_B_times_frozen_A"
    ):
        raise MosaicStarcJacobianQPError("candidate fixed LoRA gauge differs")
    _sha256(
        gauge.get("a_parameter_state_sha256"),
        label="candidate fixed LoRA-A state",
    )
    _sha256(
        gauge.get("fixed_gauge_receipt_digest"),
        label="candidate fixed-gauge receipt",
    )
    _validate_union_receipt(
        receipt["dp2_sp4_evidence_union_receipt"],
        declared_digest=receipt["dp2_sp4_evidence_union_receipt_digest"],
        parameter_layout_digest=receipt["parameter_layout_digest"],
        parameter_state_sha256=receipt["pre_step_parameter_state_sha256"],
        checkpoint_content_receipt_digest=receipt[
            "checkpoint_content_receipt_digest"
        ],
    )
    action_rows = receipt["action_rows"]
    preservation_rows = receipt["preservation_rows"]
    layer_rows = receipt["per_layer_trust_radii"]
    global_row = receipt["global_trust_radius"]
    _validate_constraint_audit_receipts(
        action_rows=action_rows,
        preservation_rows=preservation_rows,
        layer_rows=layer_rows,
        global_row=global_row,
        active_tolerance=threshold_config.active_constraint_tolerance,
    )
    union = receipt["dp2_sp4_evidence_union_receipt"]
    canonical_action = [
        row for arm in union["arms"] for row in arm["canonical_action_rows"]
    ]
    canonical_preservation = [
        row
        for arm in union["arms"]
        for row in arm["canonical_preservation_rows"]
    ]
    if len(canonical_action) != len(action_rows) or any(
        (
            audit["row_id"],
            audit["actor_family"],
            audit["row_sha256"],
            audit["row_norm"],
            audit["minimum_dot"],
            audit["gradient_computation_receipt_digest"],
        )
        != (
            source["row_id"],
            source["actor_family"],
            source["row_sha256"],
            source["row_norm"],
            source["minimum_dot"],
            source["gradient_computation_receipt_digest"],
        )
        for audit, source in zip(action_rows, canonical_action)
    ):
        raise MosaicStarcJacobianQPError(
            "candidate action audits differ from the sealed evidence union"
        )
    if len(canonical_preservation) != len(preservation_rows) or any(
        (
            audit["row_id"],
            audit["family"],
            audit["row_sha256"],
            audit["row_norm"],
            audit["maximum_absolute_dot"],
            audit["gradient_computation_receipt_digest"],
        )
        != (
            source["row_id"],
            source["family"],
            source["row_sha256"],
            source["row_norm"],
            source["maximum_absolute_dot"],
            source["gradient_computation_receipt_digest"],
        )
        for audit, source in zip(preservation_rows, canonical_preservation)
    ):
        raise MosaicStarcJacobianQPError(
            "candidate preservation audits differ from the sealed evidence union"
        )
    expected_a_state = object_sha256(
        [
            {
                "name": row["fixed_lora_a_parameter_name"],
                "shape": list(CANONICAL_A_SHAPE),
                "dtype": "torch.float32",
                "tensor_sha256": row["fixed_lora_a_sha256"],
            }
            for row in layer_rows
        ]
    )
    if (
        expected_a_state != gauge["a_parameter_state_sha256"]
        or any(
            row["fixed_gauge_receipt_digest"]
            != gauge["fixed_gauge_receipt_digest"]
            for row in layer_rows
        )
    ):
        raise MosaicStarcJacobianQPError(
            "candidate layer audits differ from the fixed LoRA-A gauge"
        )
    all_audits = action_rows + preservation_rows + layer_rows + [global_row]
    if authorized and not all(row["passed"] for row in all_audits):
        raise MosaicStarcJacobianQPError(
            "authorized candidate contains a failed constraint"
        )
    numeric_fields = {
        "mean_action_row_norm": 0.0,
        "objective_linear_term": None,
        "objective_quadratic_term": 0.0,
        "objective_value": None,
        "minimum_action_lower_bound_survival_ratio": None,
        "actual_fp32_candidate_delta_norm": 0.0,
    }
    numeric = {
        key: _receipt_finite_number(
            receipt[key], label=f"candidate {key}", minimum=minimum
        )
        for key, minimum in numeric_fields.items()
    }
    if not _same_float(
        numeric["objective_value"],
        numeric["objective_linear_term"]
        - numeric["objective_quadratic_term"],
    ):
        raise MosaicStarcJacobianQPError("candidate objective arithmetic differs")
    if not _same_float(
        numeric["minimum_action_lower_bound_survival_ratio"],
        min(row["survival_over_required_lower"] for row in action_rows),
    ):
        raise MosaicStarcJacobianQPError(
            "candidate minimum action-survival ratio differs"
        )
    eigenvalues = receipt["linear_compact_gram_eigenvalues"]
    if not isinstance(eigenvalues, list) or len(eigenvalues) != (
        len(action_rows) + len(preservation_rows)
    ):
        raise MosaicStarcJacobianQPError("candidate Gram spectrum closure differs")
    for ordinal, value in enumerate(eigenvalues):
        _receipt_finite_number(
            value,
            label=f"candidate Gram eigenvalue {ordinal}",
            minimum=0.0,
        )
    rank = receipt["linear_effective_rank"]
    condition = receipt["linear_effective_condition_number"]
    if type(rank) is not int or not 0 <= rank <= len(eigenvalues):
        raise MosaicStarcJacobianQPError("candidate effective rank differs")
    if rank == 0:
        if condition is not None:
            raise MosaicStarcJacobianQPError(
                "zero-rank candidate must have null condition number"
            )
    else:
        if condition is None or _receipt_finite_number(
            condition,
            label="candidate effective condition number",
            minimum=1.0,
        ) < 1.0:
            raise MosaicStarcJacobianQPError(
                "positive-rank candidate condition number differs"
            )

    cycles = receipt["dykstra_cycles"]
    converged = receipt["dykstra_converged"]
    cycle_residual_value = receipt["dykstra_cycle_residual"]
    if (
        type(cycles) is not int
        or cycles < 0
        or cycles > threshold_config.dykstra_max_cycles
        or type(converged) is not bool
        or (cycles == 0 and (cycle_residual_value is not None or converged))
        or (cycles > 0 and cycle_residual_value is None)
    ):
        raise MosaicStarcJacobianQPError(
            "candidate Dykstra cycle/convergence linkage differs"
        )
    cycle_residual = None
    if cycle_residual_value is not None:
        cycle_residual = _receipt_finite_number(
            cycle_residual_value,
            label="candidate Dykstra cycle residual",
            minimum=0.0,
        )
    certificate = _mapping_with_exact_keys(
        receipt["dykstra_optimality_certificate"],
        keys=_DYKSTRA_CERTIFICATE_KEYS,
        label="candidate Dykstra certificate",
    )
    guarded_active = _validate_unique_string_list(
        certificate["guarded_active_constraints"],
        label="candidate guarded active constraints",
    )
    del guarded_active
    certificate_numeric_keys = (
        "guarded_primal_max_relative_violation",
        "dykstra_dual_balance_relative_residual",
        "dykstra_correction_state_relative_residual",
        "guarded_complementarity_max_relative_residual",
        "guarded_dual_cone_max_relative_residual",
    )
    if cycles == 0:
        if any(certificate[key] is not None for key in certificate_numeric_keys) or (
            certificate["dual_correction_sum_sha256"] is not None
        ):
            raise MosaicStarcJacobianQPError(
                "unexecuted Dykstra certificate must contain null metrics"
            )
    else:
        for key in certificate_numeric_keys:
            _receipt_finite_number(
                certificate[key],
                label=f"candidate certificate {key}",
                minimum=0.0,
            )
        _sha256(
            certificate["dual_correction_sum_sha256"],
            label="candidate Dykstra correction sum",
        )
    _validate_unique_string_list(
        receipt["active_constraints"],
        label="candidate active constraints",
    )
    expected_active_constraints = [
        f"action:{row['row_id']}" for row in action_rows if row["active"]
    ] + [
        f"preservation:{row['row_id']}"
        for row in preservation_rows
        if row["active"]
    ] + [
        f"layer:{row['parameter_name']}" for row in layer_rows if row["active"]
    ]
    if global_row["active"]:
        expected_active_constraints.append("global_l2")
    if receipt["active_constraints"] != expected_active_constraints:
        raise MosaicStarcJacobianQPError(
            "candidate active-constraint receipt differs from its audits"
        )

    if authorized:
        if (
            receipt["actual_fp32_candidate_delta_sha256"]
            == receipt["null_delta_sha256"]
            or converged is not True
            or cycle_residual is None
            or cycle_residual
            > threshold_config.dykstra_cycle_tolerance
            * max(1.0, global_row["maximum_delta_norm"])
        ):
            raise MosaicStarcJacobianQPError(
                "authorized candidate is null or lacks solver convergence"
            )
        if (
            certificate["guarded_primal_max_relative_violation"]
            > threshold_config.dykstra_primal_relative_tolerance
            or certificate["dykstra_dual_balance_relative_residual"]
            > threshold_config.dykstra_dual_relative_tolerance
            or certificate["dykstra_correction_state_relative_residual"]
            > threshold_config.dykstra_dual_relative_tolerance
            or certificate["guarded_complementarity_max_relative_residual"]
            > threshold_config.dykstra_complementarity_relative_tolerance
            or certificate["guarded_dual_cone_max_relative_residual"]
            > threshold_config.dykstra_dual_relative_tolerance
        ):
            raise MosaicStarcJacobianQPError(
                "authorized candidate lacks a valid Dykstra/KKT certificate"
            )
    elif (
        receipt["actual_fp32_candidate_delta_sha256"]
        != receipt["null_delta_sha256"]
    ):
        raise MosaicStarcJacobianQPError(
            "unauthorized candidate is not the byte-identical null delta"
        )


def validate_realized_receipt_schema(receipt: Mapping[str, Any]) -> None:
    _validate_receipt_seal_and_keys(
        receipt,
        expected_keys=_REALIZED_RECEIPT_KEYS,
        label="realized receipt",
    )
    if (
        receipt["schema_version"] != REALIZED_RECEIPT_SCHEMA
        or receipt["method_name"] != METHOD_NAME
    ):
        raise MosaicStarcJacobianQPError("realized receipt identity differs")
    failures = _validate_failure_codes(
        receipt["failure_codes"], label="realized"
    )
    safe = receipt["realized_displacement_geometry_safe"]
    if type(safe) is not bool or safe != (not failures):
        raise MosaicStarcJacobianQPError(
            "realized geometry/failure consistency differs"
        )
    constant_booleans = {
        "direct_parameter_add_attempted": True,
        "rolled_back": True,
        "rollback_byte_identical_to_pre_step": True,
        "optimizer_instantiated_or_called_by_this_module": False,
        "adamw_momentum_weight_decay_gradscaler_or_preconditioner_used": False,
        "local_probe_only": True,
        "world8_apply_authorized": False,
        "world8_two_phase_prepare_commit_required": True,
        "parameter_update_applied": False,
        "checkpoint_retention_authorized": False,
        "fresh_exact81_endpoint_gate_required": True,
        "training_executed": False,
        "scientific_authority": False,
    }
    for key, expected in constant_booleans.items():
        if receipt[key] is not expected:
            raise MosaicStarcJacobianQPError(
                f"realized receipt boolean policy differs: {key}"
            )
    if (
        receipt["safe_local_probe_also_rolled_back_pending_world8_two_phase_commit"]
        is not safe
        or receipt["final_parameter_state_sha256"]
        != receipt["pre_step_parameter_state_sha256"]
    ):
        raise MosaicStarcJacobianQPError(
            "realized local rollback/state policy differs"
        )
    for key in (
        "candidate_receipt_digest",
        "parameter_layout_digest",
        "checkpoint_content_receipt_digest",
        "pre_step_parameter_state_sha256",
        "fixed_lora_a_state_before_sha256",
        "fixed_lora_a_state_after_sha256",
        "final_parameter_state_sha256",
        "actual_fp32_candidate_delta_sha256",
        "realized_fp32_displacement_sha256",
        "receipt_digest",
    ):
        _sha256(receipt[key], label=f"realized {key}")
    attempted = receipt["attempted_after_parameter_state_sha256"]
    if attempted is not None:
        _sha256(attempted, label="realized attempted-after state")
    direct_exception = receipt["direct_parameter_add_exception"]
    if direct_exception is not None and (
        not isinstance(direct_exception, str) or not direct_exception
    ):
        raise MosaicStarcJacobianQPError(
            "realized direct-add exception representation differs"
        )
    _validate_constraint_audit_receipts(
        action_rows=receipt["action_rows"],
        preservation_rows=receipt["preservation_rows"],
        layer_rows=receipt["per_layer_trust_radii"],
        global_row=receipt["global_trust_radius"],
        active_tolerance=None,
    )
    numeric_minima = {
        "candidate_delta_norm": 0.0,
        "realized_displacement_norm": 0.0,
        "candidate_realized_difference_norm": 0.0,
        "candidate_realized_relative_error": 0.0,
    }
    numeric = {
        key: _receipt_finite_number(
            receipt[key], label=f"realized {key}", minimum=minimum
        )
        for key, minimum in numeric_minima.items()
    }
    expected_relative = numeric["candidate_realized_difference_norm"] / max(
        numeric["candidate_delta_norm"], 1.0e-30
    )
    if not _same_float(
        numeric["candidate_realized_relative_error"], expected_relative
    ):
        raise MosaicStarcJacobianQPError(
            "realized candidate-relative-error arithmetic differs"
        )
    cosine = receipt["candidate_realized_cosine"]
    if cosine is not None:
        cosine = _receipt_finite_number(
            cosine, label="realized candidate cosine"
        )
        if not -1.0000001 <= cosine <= 1.0000001:
            raise MosaicStarcJacobianQPError(
                "realized candidate cosine range differs"
            )
    if receipt["application_mechanism"] != (
        "torch_no_grad_direct_parameter_add_once"
    ):
        raise MosaicStarcJacobianQPError(
            "realized application mechanism differs"
        )
    if safe and (
        direct_exception is not None
        or receipt["fixed_lora_a_state_before_sha256"]
        != receipt["fixed_lora_a_state_after_sha256"]
        or not all(
            row.get("passed") is True
            for row in (
                receipt["action_rows"]
                + receipt["preservation_rows"]
                + receipt["per_layer_trust_radii"]
                + [receipt["global_trust_radius"]]
            )
        )
    ):
        raise MosaicStarcJacobianQPError(
            "safe realized receipt contains an exception/state/constraint failure"
        )


def _revalidate_solution(solution: JacobianQPSolution) -> JacobianQPSolution:
    if not isinstance(solution, JacobianQPSolution):
        raise MosaicStarcJacobianQPError("Jacobian-QP solution type differs")
    receipt = solution.receipt
    validate_candidate_receipt_schema(receipt)
    unsigned = dict(receipt)
    digest = unsigned.pop("receipt_digest", None)
    if (
        receipt.get("schema_version") != CANDIDATE_RECEIPT_SCHEMA
        or not isinstance(digest, str)
        or object_sha256(unsigned) != digest
    ):
        raise MosaicStarcJacobianQPError("candidate receipt seal differs")
    recomputed = solve_stateless_jacobian_qp(
        layout=solution.layout,
        evidence=solution.evidence,
        global_trust_radius=solution.global_trust_radius,
        layer_trust_radii=solution.layer_trust_radii,
        config=solution.config,
    )
    candidate = _flatten_named_delta(
        solution.layout,
        solution.delta_by_parameter,
        label="bound candidate delta",
    )
    if (
        recomputed.receipt["receipt_digest"] != digest
        or recomputed.authorized != solution.authorized
        or _tensor_sha256(candidate)
        != receipt.get("actual_fp32_candidate_delta_sha256")
    ):
        raise MosaicStarcJacobianQPError(
            "candidate tensors/evidence no longer match the sealed receipt"
        )
    return recomputed


def _validate_live_fixed_lora_a(
    *,
    solution: JacobianQPSolution,
    ordered_fixed_lora_a: Sequence[tuple[str, torch.Tensor]],
    require_gauge_match: bool = True,
) -> str:
    if isinstance(ordered_fixed_lora_a, Mapping) or not isinstance(
        ordered_fixed_lora_a, Sequence
    ) or len(ordered_fixed_lora_a) != len(solution.layer_trust_radii):
        raise MosaicStarcJacobianQPError(
            "live fixed LoRA-A closure must be one explicit 32-tensor sequence"
        )
    rows: list[dict[str, Any]] = []
    for ordinal, (trust, item) in enumerate(
        zip(solution.layer_trust_radii, ordered_fixed_lora_a)
    ):
        if not isinstance(item, tuple) or len(item) != 2:
            raise MosaicStarcJacobianQPError(
                f"live fixed LoRA-A entry {ordinal} differs"
            )
        name, tensor = item
        if name != trust.fixed_lora_a_parameter_name:
            raise MosaicStarcJacobianQPError(
                f"live fixed LoRA-A ordering differs at {ordinal}"
            )
        tensor = _validate_materialized_fp32_tensor(
            tensor,
            label=f"live fixed LoRA-A {name}",
            allow_requires_grad=False,
        )
        if tuple(int(value) for value in tensor.shape) != CANONICAL_A_SHAPE:
            raise MosaicStarcJacobianQPError(
                f"live fixed LoRA-A {name} shape differs"
            )
        tensor_digest = _tensor_sha256(tensor)
        if require_gauge_match and tensor_digest != _tensor_sha256(
            trust.fixed_lora_a
        ):
            raise MosaicStarcJacobianQPError(
                f"live fixed LoRA-A {name} differs from the QP gauge"
            )
        rows.append(
            {
                "name": name,
                "shape": list(CANONICAL_A_SHAPE),
                "dtype": "torch.float32",
                "tensor_sha256": tensor_digest,
            }
        )
    digest = object_sha256(rows)
    if require_gauge_match and digest != _fixed_lora_a_state_sha256(
        solution.layer_trust_radii
    ):
        raise MosaicStarcJacobianQPError(
            "live fixed LoRA-A aggregate state differs from the QP gauge"
        )
    return digest


def apply_direct_delta_and_audit(
    *,
    solution: JacobianQPSolution,
    ordered_parameters: Sequence[tuple[str, torch.Tensor]],
    ordered_fixed_lora_a: Sequence[tuple[str, torch.Tensor]],
) -> RealizedDisplacementAudit:
    """Probe one direct FP32 add and always restore the local pre-step state.

    This stateless core cannot implement a WORLD8 two-phase commit.  It
    therefore performs the exact local ``add_`` and realized-displacement
    audit required by the contract, but rolls back even a safe local result.
    A separate distributed coordinator must obtain eight matching local
    receipts before any atomic retained update.
    """

    solution = _revalidate_solution(solution)
    if not solution.authorized:
        raise MosaicStarcJacobianQPError(
            "an unauthorized/null Jacobian-QP solution cannot be applied"
        )
    if isinstance(ordered_parameters, Mapping) or not isinstance(
        ordered_parameters, Sequence
    ):
        raise MosaicStarcJacobianQPError(
            "live parameters must be one explicit ordered sequence"
        )
    if isinstance(ordered_fixed_lora_a, Mapping) or not isinstance(
        ordered_fixed_lora_a, Sequence
    ):
        raise MosaicStarcJacobianQPError(
            "live fixed LoRA-A must be one explicit ordered sequence"
        )
    # A caller-supplied mutable/custom Sequence must not be re-iterated across
    # the mutation boundary.  Freeze the exact object references once, then
    # use only these tuples for validation, mutation, snapshot, and rollback.
    live_parameters = tuple(ordered_parameters)
    live_fixed_lora_a = tuple(ordered_fixed_lora_a)
    before_state = _ordered_parameter_state_sha256(
        solution.layout, live_parameters
    )
    if before_state != solution.layout.parameter_state_sha256:
        raise MosaicStarcJacobianQPError(
            "live pre-step parameters differ from the Jacobian evidence state"
        )
    fixed_a_before_state = _validate_live_fixed_lora_a(
        solution=solution,
        ordered_fixed_lora_a=live_fixed_lora_a,
    )
    candidate = _flatten_named_delta(
        solution.layout,
        solution.delta_by_parameter,
        label="candidate delta",
    )
    # Finish all evidence parsing before the mutation boundary.  From this
    # point onward only validated, cloned canonical rows are used.
    union = _validate_and_union_dp2_sp4_evidence(
        layout=solution.layout,
        evidence=solution.evidence,
        minimum_row_norm=solution.config.minimum_row_norm,
    )
    before_tensors = [
        tensor.detach().to(device="cpu").contiguous().clone()
        for _, tensor in live_parameters
    ]
    attempted_after_state: str | None = None
    apply_exception: str | None = None
    post_add_exception: Exception | None = None
    fixed_a_after_state: str | None = None
    realized_flat = torch.zeros(
        solution.layout.total_numel, dtype=torch.float32, device="cpu"
    )
    try:
        try:
            with torch.no_grad():
                for ordinal, ((name, parameter), delta) in enumerate(
                    zip(
                        live_parameters,
                        (
                            solution.delta_by_parameter[item]
                            for item in solution.layout.names
                        ),
                    )
                ):
                    if name != solution.layout.names[ordinal]:
                        raise MosaicStarcJacobianQPError(
                            "live parameter ordering escaped the fixed layout"
                        )
                    parameter.add_(delta.to(device=parameter.device))
        except Exception as error:
            # A partial direct add is a measured unsafe probe, not permission
            # to skip the mandatory rollback.
            apply_exception = f"{type(error).__name__}:{error}"

        try:
            attempted_after_state = _ordered_parameter_state_sha256(
                solution.layout, live_parameters
            )
            # Capture the attempted displacement immediately.  All expensive
            # geometry work is intentionally deferred until after rollback.
            for before, (_, after), (start, stop) in zip(
                before_tensors,
                live_parameters,
                solution.layout.offsets,
            ):
                current = after.detach().to(device="cpu").contiguous()
                if current.dtype != torch.float32 or tuple(current.shape) != tuple(
                    before.shape
                ):
                    raise MosaicStarcJacobianQPError(
                        "post-add parameter shape/dtype differs"
                    )
                actual = (current - before).to(dtype=torch.float32)
                if not bool(torch.isfinite(actual).all().item()):
                    raise MosaicStarcJacobianQPError(
                        "post-add realized displacement is non-finite"
                    )
                realized_flat[start:stop].copy_(actual.reshape(-1))
            fixed_a_after_state = _validate_live_fixed_lora_a(
                solution=solution,
                ordered_fixed_lora_a=live_fixed_lora_a,
                require_gauge_match=False,
            )
        except Exception as error:
            post_add_exception = error
    finally:
        rollback_errors: list[str] = []
        with torch.no_grad():
            for ordinal, ((_, parameter), before) in enumerate(
                zip(live_parameters, before_tensors)
            ):
                try:
                    parameter.copy_(before.to(device=parameter.device))
                except Exception as error:
                    rollback_errors.append(
                        f"{ordinal}:{type(error).__name__}:{error}"
                    )
        try:
            final_state = _ordered_parameter_state_sha256(
                solution.layout, live_parameters
            )
        except Exception as error:
            rollback_errors.append(
                f"final-state:{type(error).__name__}:{error}"
            )
            final_state = ""
        if rollback_errors or final_state != before_state:
            details = ";".join(rollback_errors) or "final-state-digest-mismatch"
            raise MosaicStarcJacobianQPError(
                "mandatory local rollback was not byte-identical to the "
                f"pre-step state: {details}"
            )

    # From here onward every live LoRA-B tensor is already byte-identical to
    # the pre-step state.  Audit only the immutable CPU realized snapshot.
    if post_add_exception is not None:
        raise MosaicStarcJacobianQPError(
            "post-add snapshot/gauge audit failed after byte-identical rollback"
        ) from post_add_exception
    assert fixed_a_after_state is not None
    (
        action_audit,
        preservation_audit,
        layer_audit,
        global_audit,
        failures,
    ) = _constraint_audit(
        flat=realized_flat,
        action_rows=union.action_rows,
        preservation_rows=union.preservation_rows,
        layout=solution.layout,
        layer_trust_radii=solution.layer_trust_radii,
        global_trust_radius=solution.global_trust_radius,
        active_tolerance=solution.config.active_constraint_tolerance,
    )
    if apply_exception is not None:
        failures.append("DIRECT_PARAMETER_ADD_FAILED")
    if fixed_a_after_state != fixed_a_before_state:
        failures.append("FROZEN_LORA_A_CHANGED_DURING_DIRECT_ADD")
    candidate64 = candidate.to(dtype=torch.float64)
    realized64 = realized_flat.to(dtype=torch.float64)
    candidate_norm = float(torch.linalg.vector_norm(candidate64).item())
    realized_norm = float(torch.linalg.vector_norm(realized64).item())
    difference_norm = float(
        torch.linalg.vector_norm(realized64 - candidate64).item()
    )
    relative_error = difference_norm / max(candidate_norm, 1.0e-30)
    cosine = None
    if candidate_norm > 0.0 and realized_norm > 0.0:
        cosine = float(torch.dot(candidate64, realized64).item()) / (
            candidate_norm * realized_norm
        )
    if relative_error > solution.config.realized_maximum_candidate_relative_error:
        failures.append("REALIZED_CANDIDATE_RELATIVE_ERROR_EXCEEDED")
    if (
        cosine is None
        or cosine < solution.config.realized_minimum_candidate_cosine
    ):
        failures.append("REALIZED_CANDIDATE_COSINE_TOO_LOW")
    failures = sorted(set(failures))
    safe = not failures
    rolled_back = True

    unsigned = {
        "schema_version": REALIZED_RECEIPT_SCHEMA,
        "method_name": METHOD_NAME,
        "candidate_receipt_digest": solution.receipt["receipt_digest"],
        "parameter_layout_digest": solution.layout.layout_digest,
        "checkpoint_content_receipt_digest": (
            union.checkpoint_content_receipt_digest
        ),
        "pre_step_parameter_state_sha256": before_state,
        "fixed_lora_a_state_before_sha256": fixed_a_before_state,
        "fixed_lora_a_state_after_sha256": fixed_a_after_state,
        "attempted_after_parameter_state_sha256": attempted_after_state,
        "final_parameter_state_sha256": final_state,
        "direct_parameter_add_attempted": True,
        "direct_parameter_add_exception": apply_exception,
        "actual_fp32_candidate_delta_sha256": _tensor_sha256(candidate),
        "realized_fp32_displacement_sha256": _tensor_sha256(realized_flat),
        "candidate_delta_norm": candidate_norm,
        "realized_displacement_norm": realized_norm,
        "candidate_realized_difference_norm": difference_norm,
        "candidate_realized_relative_error": relative_error,
        "candidate_realized_cosine": cosine,
        "action_rows": action_audit,
        "preservation_rows": preservation_audit,
        "global_trust_radius": global_audit,
        "per_layer_trust_radii": layer_audit,
        "failure_codes": failures,
        "realized_displacement_geometry_safe": safe,
        "rolled_back": rolled_back,
        "rollback_byte_identical_to_pre_step": final_state == before_state,
        "safe_local_probe_also_rolled_back_pending_world8_two_phase_commit": safe,
        "application_mechanism": "torch_no_grad_direct_parameter_add_once",
        "optimizer_instantiated_or_called_by_this_module": False,
        "adamw_momentum_weight_decay_gradscaler_or_preconditioner_used": False,
        "local_probe_only": True,
        "world8_apply_authorized": False,
        "world8_two_phase_prepare_commit_required": True,
        "parameter_update_applied": False,
        "checkpoint_retention_authorized": False,
        "fresh_exact81_endpoint_gate_required": True,
        "training_executed": False,
        "scientific_authority": False,
    }
    receipt = _seal(unsigned)
    validate_realized_receipt_schema(receipt)
    return RealizedDisplacementAudit(
        realized_geometry_safe=safe,
        rolled_back=rolled_back,
        realized_delta_by_parameter=solution.layout.unflatten_cpu_fp32(
            realized_flat,
            label="realized FP32 displacement",
        ),
        receipt=receipt,
    )


__all__ = [
    "ACTION_FAMILIES",
    "CANONICAL_PARAMETER_NAMES",
    "CANDIDATE_RECEIPT_SCHEMA",
    "DP_ARM_ORDER",
    "DP2SP4Evidence",
    "DPArmEvidence",
    "EVIDENCE_UNION_SCHEMA",
    "FixedParameterLayout",
    "JacobianQPConfig",
    "JacobianQPSolution",
    "LayerTrustRadius",
    "METHOD_NAME",
    "MosaicStarcJacobianQPError",
    "PRESERVATION_FAMILIES",
    "PreservationConstraintRow",
    "REALIZED_RECEIPT_SCHEMA",
    "RECEIPT_JSON_SCHEMA_RELATIVE_PATH",
    "RealizedDisplacementAudit",
    "SPRankEvidence",
    "SP_GLOBAL_RANKS",
    "ActionConstraintRow",
    "apply_direct_delta_and_audit",
    "canonical_json_bytes",
    "object_sha256",
    "solve_stateless_jacobian_qp",
    "validate_candidate_receipt_schema",
    "validate_realized_receipt_schema",
]
