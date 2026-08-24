"""Action-conditioned delete/create permission and local regeneration routing.

Round-35/36 attention transport can change a target hidden value, but it has
no authority to stop source reconstruction from restoring an obsolete hand,
object, or ownership state.  This module separates two questions:

* *where/when may source evidence be discarded?* -- a learned, target-derived
  delete/create gate predicted from source + ``q_pred``;
* *what should regenerate that region?* -- a same-state source-aware proposal
  or a high-noise R2V-4 reconstruction expert.

The high R2V branch is an execution expert, not the action representation.  It
is spatially/temporally bounded by the gate.  Self-generated anchors may
regularize the action code, but they may not authorize a point gate target.

This local module does not have an external artifact/support authority, so all
active predicted/noise/high-R2V/SPT execution is fail-closed.  Only an exact
zero gate is executable, and it returns the source object without consuming an
expert tensor.  The active function signatures reserve a future ABI; they do
not authorize a GPU launch or claim that a learned gate is available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Optional, Tuple


SCHEMA_VERSION = "bernini-action-delete-create-regeneration-router-v1"
GATE_RECEIPT_SCHEMA_VERSION = "bernini-state-change-gate-provenance-receipt-v1"
TENSOR_RECEIPT_SCHEMA_VERSION = "bernini-regeneration-tensor-artifact-receipt-v1"
PHASE_PLAN_RECEIPT_SCHEMA_VERSION = "bernini-regeneration-phase-plan-execution-receipt-v1"
PHASE_COUNT = 21
ACTION_WIDTH = 256
MAX_HARD_AUTHORIZATION_FRACTION_PER_PHASE = 0.95


class ActionRegenerationRouterError(ValueError):
    """Raised when a delete/create or regeneration invariant differs."""


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ActionRegenerationRouterError("%s must be a SHA-256 hex digest" % label)
    lowered = value.lower()
    if any(character not in "0123456789abcdef" for character in lowered):
        raise ActionRegenerationRouterError("%s must be a SHA-256 hex digest" % label)
    if lowered == "0" * 64:
        raise ActionRegenerationRouterError("%s must not be the null digest" % label)
    return lowered


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _tensor_sha256(value: Any, *, label: str) -> str:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise ActionRegenerationRouterError("%s must be a tensor" % label)
    detached = value.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(detached.dtype), "shape": list(detached.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    raw = bytes(detached.view(torch.uint8).reshape(-1).tolist())
    return hashlib.sha256(header + b"\0" + raw).hexdigest()


def _validated_ids(values: Any, *, label: str, expected: Optional[int] = None) -> Tuple[str, ...]:
    if type(values) is not tuple or not values:
        raise ActionRegenerationRouterError("%s must be a nonempty tuple" % label)
    if any(type(value) is not str or not value for value in values):
        raise ActionRegenerationRouterError("%s contains an invalid ID" % label)
    if len(set(values)) != len(values):
        raise ActionRegenerationRouterError("%s must be unique" % label)
    if expected is not None and len(values) != expected:
        raise ActionRegenerationRouterError("%s batch binding differs" % label)
    return values


@dataclass(frozen=True)
class GateProvenanceReceiptV1:
    schema_version: str
    role: str
    origin: str
    sample_ids: Tuple[str, ...]
    state_payload_sha256: str
    gate_payload_sha256: str
    valid_sha256: str
    hard_support_sha256: str
    producer_artifact_sha256: str
    split_manifest_sha256: str
    dilation: int
    valid_voxels: int
    hard_support_voxels: int
    valid_per_sample_phase: Tuple[Tuple[int, ...], ...]
    hard_support_per_sample_phase: Tuple[Tuple[int, ...], ...]
    receipt_sha256: str


@dataclass(frozen=True)
class RegenerationTensorArtifactReceiptV1:
    schema_version: str
    role: str
    sample_ids: Tuple[str, ...]
    tensor_sha256: str
    producer_artifact_sha256: str
    producer_checkpoint_sha256: str
    producer_config_sha256: str
    producer_frozen: bool
    input_payload_sha256: str
    source_identity_sha256: str
    external_manifest_sha256: str
    derivation_key_sha256: str
    solver_sigma: float
    solver_step: int
    receipt_sha256: str


@dataclass(frozen=True)
class PhasePlanExecutionReceiptV1:
    schema_version: str
    sample_ids: Tuple[str, ...]
    gate_receipt_sha256: str
    offsets_sha256: str
    gate_probs_sha256: str
    receipt_sha256: str


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for regeneration routing") from error
    return torch


def _nn() -> Any:
    return _torch().nn


@dataclass(frozen=True)
class StateChangeGateV1:
    """Soft delete/create permission in ``[B,1,21,H,W]`` latent coordinates."""

    delete: Any
    create: Any
    contact_permission: Any
    regenerate: Any
    preserve: Any
    valid: Any
    hard_authorization_support: Any
    provenance: GateProvenanceReceiptV1


@dataclass(frozen=True)
class RegenerationGatePredictorConfigV1:
    source_channels: int = 16
    action_width: int = ACTION_WIDTH
    hidden_channels: int = 128
    phase_count: int = PHASE_COUNT
    initial_change_probability: float = 0.02
    output_weight_std: float = 1.0e-3

    def validate(self) -> None:
        for name in ("source_channels", "action_width", "hidden_channels", "phase_count"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ActionRegenerationRouterError(
                    "%s must be a positive integer" % name
                )
        if self.phase_count != PHASE_COUNT:
            raise ActionRegenerationRouterError("regeneration gate requires 21 phases")
        probability = float(self.initial_change_probability)
        if not math.isfinite(probability) or not 0.0 < probability < 0.5:
            raise ActionRegenerationRouterError(
                "initial_change_probability must lie in (0,0.5)"
            )
        if not math.isfinite(float(self.output_weight_std)) or self.output_weight_std <= 0.0:
            raise ActionRegenerationRouterError("output_weight_std must be positive")


@dataclass(frozen=True)
class RegenerationGateLossConfigV1:
    delete_bce_weight: float = 1.0
    create_bce_weight: float = 1.0
    contact_bce_weight: float = 1.0
    union_dice_weight: float = 1.0
    phase_mass_weight: float = 0.25

    def validate(self) -> None:
        for name, value in asdict(self).items():
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ActionRegenerationRouterError(
                    "%s must be finite and non-negative" % name
                )
        if sum(float(value) for value in asdict(self).values()) <= 0.0:
            raise ActionRegenerationRouterError("all regeneration gate losses are disabled")


@dataclass(frozen=True)
class RegenerationRouteResultV1:
    clean: Any
    gate: StateChangeGateV1
    diagnostics: Mapping[str, Any]


def _validate_volume(value: Any, *, label: str, channels: Optional[int] = None) -> None:
    torch = _torch()
    if not isinstance(value, torch.Tensor) or value.ndim != 5:
        raise ActionRegenerationRouterError("%s must be [B,C,21,H,W]" % label)
    if int(value.shape[0]) <= 0 or int(value.shape[2]) != PHASE_COUNT:
        raise ActionRegenerationRouterError("%s batch/phase geometry differs" % label)
    if channels is not None and int(value.shape[1]) != channels:
        raise ActionRegenerationRouterError("%s channel count differs" % label)
    if any(int(size) <= 0 for size in value.shape):
        raise ActionRegenerationRouterError("%s has an empty dimension" % label)
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ActionRegenerationRouterError("%s must be finite floating point" % label)


def _validate_probability_volume(value: Any, *, label: str) -> None:
    _validate_volume(value, label=label, channels=1)
    if bool((value < 0.0).any()) or bool((value > 1.0).any()):
        raise ActionRegenerationRouterError("%s must lie in [0,1]" % label)


def validate_state_change_gate_v1(
    gate: StateChangeGateV1,
    *,
    reference: Optional[Any] = None,
    require_phase_zero_preserved: bool = True,
) -> None:
    torch = _torch()
    if not isinstance(gate, StateChangeGateV1):
        raise ActionRegenerationRouterError("state-change gate type differs")
    for name in ("delete", "create", "contact_permission", "regenerate", "preserve"):
        value = getattr(gate, name)
        _validate_probability_volume(value, label=name)
        if value.requires_grad or value.grad_fn is not None:
            raise ActionRegenerationRouterError(
                "%s must be detached from teacher/predictor autograd" % name
            )
    shape = tuple(gate.delete.shape)
    if any(
        tuple(getattr(gate, name).shape) != shape
        for name in ("create", "contact_permission", "regenerate", "preserve")
    ):
        raise ActionRegenerationRouterError("state-change gate shapes differ")
    for name in ("valid", "hard_authorization_support"):
        value = getattr(gate, name)
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.bool
            or tuple(value.shape) != shape
        ):
            raise ActionRegenerationRouterError("%s must be a bool gate volume" % name)
    devices = {
        getattr(gate, name).device
        for name in (
            "delete",
            "create",
            "contact_permission",
            "regenerate",
            "preserve",
            "valid",
            "hard_authorization_support",
        )
    }
    if len(devices) != 1:
        raise ActionRegenerationRouterError("gate tensors must share a device")
    floating_dtypes = {
        getattr(gate, name).dtype
        for name in ("delete", "create", "contact_permission", "regenerate", "preserve")
    }
    if len(floating_dtypes) != 1:
        raise ActionRegenerationRouterError("gate probability tensors must share a dtype")
    if reference is not None:
        _validate_volume(reference, label="reference")
        expected = (
            int(reference.shape[0]),
            1,
            PHASE_COUNT,
            int(reference.shape[-2]),
            int(reference.shape[-1]),
        )
        if shape != expected:
            raise ActionRegenerationRouterError("gate/reference geometry differs")
    union = torch.maximum(
        torch.maximum(gate.delete, gate.create), gate.contact_permission
    )
    if not torch.equal(gate.regenerate, union):
        raise ActionRegenerationRouterError(
            "regenerate must exactly equal max(delete,create,contact_permission)"
        )
    if not torch.equal(gate.preserve, 1.0 - gate.regenerate):
        raise ActionRegenerationRouterError("preserve must complement regenerate")
    authorized = gate.hard_authorization_support & gate.valid
    if not torch.equal(gate.hard_authorization_support, authorized):
        raise ActionRegenerationRouterError("hard authorization extends outside valid")
    outside = ~gate.hard_authorization_support
    for name in ("delete", "create", "contact_permission", "regenerate"):
        if bool((getattr(gate, name)[outside] != 0.0).any()):
            raise ActionRegenerationRouterError("%s is nonzero outside hard authorization" % name)
    if require_phase_zero_preserved and bool((gate.regenerate[:, :, 0] != 0.0).any()):
        raise ActionRegenerationRouterError("phase zero must not be regenerated")
    if require_phase_zero_preserved and bool(gate.hard_authorization_support[:, :, 0].any()):
        raise ActionRegenerationRouterError("phase zero cannot carry hard authorization")
    _validate_gate_receipt_for_gate_v1(gate.provenance, gate=gate)


def _spatial_dilate(value: Any, radius: int) -> Any:
    torch = _torch()
    if type(radius) is not int or radius < 0:
        raise ActionRegenerationRouterError("dilation radius must be a non-negative integer")
    if radius == 0:
        return value
    batch, channels, phases, height, width = map(int, value.shape)
    flat = value.permute(0, 2, 1, 3, 4).reshape(batch * phases, channels, height, width)
    kernel = 2 * radius + 1
    flat = torch.nn.functional.max_pool2d(
        flat, kernel_size=kernel, stride=1, padding=radius
    )
    return flat.reshape(batch, phases, channels, height, width).permute(0, 2, 1, 3, 4)


def _derive_teacher_gate_fields_v1(
    *,
    source_occupancy: Any,
    target_occupancy_in_source_coordinates: Any,
    contact_corridor: Optional[Any],
    valid: Optional[Any],
    dilation: int,
) -> Tuple[Any, Any, Any, Any, Any, str]:
    torch = _torch()
    _validate_probability_volume(source_occupancy, label="source_occupancy")
    _validate_probability_volume(
        target_occupancy_in_source_coordinates,
        label="target_occupancy_in_source_coordinates",
    )
    if tuple(source_occupancy.shape) != tuple(target_occupancy_in_source_coordinates.shape):
        raise ActionRegenerationRouterError("source/target occupancy geometry differs")
    if (
        source_occupancy.device != target_occupancy_in_source_coordinates.device
        or source_occupancy.dtype != target_occupancy_in_source_coordinates.dtype
    ):
        raise ActionRegenerationRouterError("source/target occupancy dtype/device differs")
    if (
        source_occupancy.requires_grad
        or source_occupancy.grad_fn is not None
        or target_occupancy_in_source_coordinates.requires_grad
        or target_occupancy_in_source_coordinates.grad_fn is not None
    ):
        raise ActionRegenerationRouterError("teacher occupancies must be detached")
    source = source_occupancy.float()
    target = target_occupancy_in_source_coordinates.float()
    if valid is None:
        validity = torch.ones_like(source, dtype=torch.bool)
    else:
        if (
            not isinstance(valid, torch.Tensor)
            or valid.dtype != torch.bool
            or tuple(valid.shape) != tuple(source.shape)
            or valid.device != source.device
        ):
            raise ActionRegenerationRouterError("occupancy validity mask differs")
        validity = valid
    phase_authority = torch.ones_like(validity)
    phase_authority[:, :, 0] = False
    validity_with_phase_authority = validity & phase_authority
    delete = torch.relu(source - target) * validity.float()
    create = torch.relu(target - source) * validity.float()
    if contact_corridor is None:
        contact = torch.zeros_like(source)
        contact_present = False
    else:
        _validate_probability_volume(contact_corridor, label="contact_corridor")
        if (
            tuple(contact_corridor.shape) != tuple(source.shape)
            or contact_corridor.device != source.device
            or contact_corridor.dtype != source_occupancy.dtype
        ):
            raise ActionRegenerationRouterError("contact corridor dtype/device/geometry differs")
        if contact_corridor.requires_grad or contact_corridor.grad_fn is not None:
            raise ActionRegenerationRouterError("contact corridor must be detached")
        contact = contact_corridor.float() * validity.float()
        contact_present = True
    # Dilation may cross an annotation boundary, so every field is re-masked
    # after max-pooling.  Invalid voxels never become no-change negatives or
    # permissions.
    delete = _spatial_dilate(delete, dilation).clamp(0.0, 1.0)
    create = _spatial_dilate(create, dilation).clamp(0.0, 1.0)
    contact = _spatial_dilate(contact, dilation).clamp(0.0, 1.0)
    mask = validity_with_phase_authority.float()
    delete = delete * mask
    create = create * mask
    contact = contact * mask
    hard_support = (
        torch.maximum(torch.maximum(delete, create), contact) > 0.0
    ) & validity_with_phase_authority
    state_payload_sha256 = _canonical_sha256(
        {
            "source_occupancy": _tensor_sha256(source_occupancy, label="source_occupancy"),
            "target_occupancy": _tensor_sha256(
                target_occupancy_in_source_coordinates,
                label="target_occupancy_in_source_coordinates",
            ),
            "contact_corridor": _tensor_sha256(contact, label="derived_contact_permission"),
            "contact_corridor_present": contact_present,
            "valid": _tensor_sha256(validity, label="valid"),
            "dilation": dilation,
        }
    )
    return delete, create, contact, validity, hard_support, state_payload_sha256


def _gate_receipt_payload_v1(receipt: GateProvenanceReceiptV1) -> Dict[str, Any]:
    return {
        "schema_version": receipt.schema_version,
        "role": receipt.role,
        "origin": receipt.origin,
        "sample_ids": list(receipt.sample_ids),
        "state_payload_sha256": receipt.state_payload_sha256,
        "gate_payload_sha256": receipt.gate_payload_sha256,
        "valid_sha256": receipt.valid_sha256,
        "hard_support_sha256": receipt.hard_support_sha256,
        "producer_artifact_sha256": receipt.producer_artifact_sha256,
        "split_manifest_sha256": receipt.split_manifest_sha256,
        "dilation": receipt.dilation,
        "valid_voxels": receipt.valid_voxels,
        "hard_support_voxels": receipt.hard_support_voxels,
        "valid_per_sample_phase": [list(row) for row in receipt.valid_per_sample_phase],
        "hard_support_per_sample_phase": [
            list(row) for row in receipt.hard_support_per_sample_phase
        ],
    }


def _make_gate_receipt_v1(
    *,
    role: str,
    origin: str,
    sample_ids: Tuple[str, ...],
    state_payload_sha256: str,
    delete: Any,
    create: Any,
    contact_permission: Any,
    valid: Any,
    hard_support: Any,
    producer_artifact_sha256: str,
    split_manifest_sha256: str,
    dilation: int,
) -> GateProvenanceReceiptV1:
    samples = _validated_ids(sample_ids, label="gate sample_ids", expected=int(valid.shape[0]))
    support_per_sample_phase = tuple(
        tuple(int(value) for value in row)
        for row in hard_support.sum(dim=(1, 3, 4)).detach().cpu().tolist()
    )
    valid_per_sample_phase = tuple(
        tuple(int(value) for value in row)
        for row in valid.sum(dim=(1, 3, 4)).detach().cpu().tolist()
    )
    receipt_without_digest = GateProvenanceReceiptV1(
        schema_version=GATE_RECEIPT_SCHEMA_VERSION,
        role=role,
        origin=origin,
        sample_ids=samples,
        state_payload_sha256=_require_sha256(
            state_payload_sha256, label="state_payload_sha256"
        ),
        gate_payload_sha256=_canonical_sha256(
            {
                "delete": _tensor_sha256(delete, label="delete"),
                "create": _tensor_sha256(create, label="create"),
                "contact_permission": _tensor_sha256(
                    contact_permission, label="contact_permission"
                ),
            }
        ),
        valid_sha256=_tensor_sha256(valid, label="valid"),
        hard_support_sha256=_tensor_sha256(hard_support, label="hard_support"),
        producer_artifact_sha256=_require_sha256(
            producer_artifact_sha256, label="producer_artifact_sha256"
        ),
        split_manifest_sha256=_require_sha256(
            split_manifest_sha256, label="split_manifest_sha256"
        ),
        dilation=dilation,
        valid_voxels=int(valid.sum().item()),
        hard_support_voxels=int(hard_support.sum().item()),
        valid_per_sample_phase=valid_per_sample_phase,
        hard_support_per_sample_phase=support_per_sample_phase,
        receipt_sha256="",
    )
    payload = _gate_receipt_payload_v1(receipt_without_digest)
    return GateProvenanceReceiptV1(
        **{
            name: getattr(receipt_without_digest, name)
            for name in payload
            if name
            not in (
                "sample_ids",
                "valid_per_sample_phase",
                "hard_support_per_sample_phase",
            )
        },
        sample_ids=samples,
        valid_per_sample_phase=valid_per_sample_phase,
        hard_support_per_sample_phase=support_per_sample_phase,
        receipt_sha256=_canonical_sha256(payload),
    )


def _validate_gate_receipt_for_gate_v1(
    receipt: GateProvenanceReceiptV1,
    *,
    gate: StateChangeGateV1,
) -> None:
    if not isinstance(receipt, GateProvenanceReceiptV1):
        raise ActionRegenerationRouterError(
            "gate provenance must be a structured content-bound receipt"
        )
    _validated_ids(receipt.sample_ids, label="gate sample_ids", expected=int(gate.delete.shape[0]))
    if receipt.schema_version != GATE_RECEIPT_SCHEMA_VERSION:
        raise ActionRegenerationRouterError("gate receipt schema differs")
    if receipt.role != "q_y_state_change_teacher":
        raise ActionRegenerationRouterError(
            "predicted gate execution is disabled until an external support authority exists"
        )
    allowed_origins = {"q_y_state_change_teacher": "clean_source_target_pair"}
    if receipt.origin != allowed_origins[receipt.role]:
        raise ActionRegenerationRouterError("anchor-derived gate provenance is forbidden")
    if type(receipt.dilation) is not int or receipt.dilation < 0:
        raise ActionRegenerationRouterError("gate receipt dilation differs")
    if type(receipt.valid_voxels) is not int or type(receipt.hard_support_voxels) is not int:
        raise ActionRegenerationRouterError("gate receipt counts differ")
    if (
        type(receipt.valid_per_sample_phase) is not tuple
        or len(receipt.valid_per_sample_phase) != int(gate.valid.shape[0])
        or type(receipt.hard_support_per_sample_phase) is not tuple
        or len(receipt.hard_support_per_sample_phase) != int(gate.valid.shape[0])
    ):
        raise ActionRegenerationRouterError("gate receipt topology differs")
    for valid_row, support_row in zip(
        receipt.valid_per_sample_phase, receipt.hard_support_per_sample_phase
    ):
        if (
            type(valid_row) is not tuple
            or type(support_row) is not tuple
            or len(valid_row) != PHASE_COUNT
            or len(support_row) != PHASE_COUNT
            or any(type(value) is not int or value < 0 for value in valid_row)
            or any(type(value) is not int or value < 0 for value in support_row)
        ):
            raise ActionRegenerationRouterError("gate receipt sample/phase topology differs")
        for valid_count, support_count in zip(valid_row, support_row):
            if valid_count == 0:
                if support_count != 0:
                    raise ActionRegenerationRouterError("support exists in an invalid phase")
            elif support_count / float(valid_count) > MAX_HARD_AUTHORIZATION_FRACTION_PER_PHASE:
                raise ActionRegenerationRouterError(
                    "hard authorization support is not sparse per sample/phase"
                )
    for name in (
        "state_payload_sha256",
        "gate_payload_sha256",
        "valid_sha256",
        "hard_support_sha256",
        "producer_artifact_sha256",
        "split_manifest_sha256",
        "receipt_sha256",
    ):
        _require_sha256(getattr(receipt, name), label=name)
    if receipt.valid_sha256 != _tensor_sha256(gate.valid, label="gate.valid"):
        raise ActionRegenerationRouterError("gate valid bytes differ from receipt")
    if receipt.hard_support_sha256 != _tensor_sha256(
        gate.hard_authorization_support, label="gate.hard_authorization_support"
    ):
        raise ActionRegenerationRouterError("gate hard support bytes differ from receipt")
    observed_gate_payload = _canonical_sha256(
        {
            "delete": _tensor_sha256(gate.delete, label="gate.delete"),
            "create": _tensor_sha256(gate.create, label="gate.create"),
            "contact_permission": _tensor_sha256(
                gate.contact_permission, label="gate.contact_permission"
            ),
        }
    )
    if receipt.gate_payload_sha256 != observed_gate_payload:
        raise ActionRegenerationRouterError("gate permission values differ from receipt")
    observed_support_per_sample_phase = tuple(
        tuple(int(value) for value in row)
        for row in gate.hard_authorization_support.sum(dim=(1, 3, 4))
        .detach().cpu().tolist()
    )
    observed_valid_per_sample_phase = tuple(
        tuple(int(value) for value in row)
        for row in gate.valid.sum(dim=(1, 3, 4)).detach().cpu().tolist()
    )
    if (
        receipt.valid_voxels != int(gate.valid.sum().item())
        or receipt.hard_support_voxels != int(gate.hard_authorization_support.sum().item())
        or receipt.valid_per_sample_phase != observed_valid_per_sample_phase
        or receipt.hard_support_per_sample_phase != observed_support_per_sample_phase
    ):
        raise ActionRegenerationRouterError("gate support mass/topology differs from receipt")
    if receipt.receipt_sha256 != _canonical_sha256(_gate_receipt_payload_v1(receipt)):
        raise ActionRegenerationRouterError("gate receipt digest differs")


def build_teacher_gate_receipt_v1(
    *,
    source_occupancy: Any,
    target_occupancy_in_source_coordinates: Any,
    contact_corridor: Optional[Any] = None,
    valid: Optional[Any] = None,
    dilation: int = 1,
    sample_ids: Tuple[str, ...],
    producer_artifact_sha256: str,
    split_manifest_sha256: str,
) -> GateProvenanceReceiptV1:
    delete, create, contact, validity, hard_support, state_payload = _derive_teacher_gate_fields_v1(
        source_occupancy=source_occupancy,
        target_occupancy_in_source_coordinates=target_occupancy_in_source_coordinates,
        contact_corridor=contact_corridor,
        valid=valid,
        dilation=dilation,
    )
    return _make_gate_receipt_v1(
        role="q_y_state_change_teacher",
        origin="clean_source_target_pair",
        sample_ids=sample_ids,
        state_payload_sha256=state_payload,
        delete=delete,
        create=create,
        contact_permission=contact,
        valid=validity,
        hard_support=hard_support,
        producer_artifact_sha256=producer_artifact_sha256,
        split_manifest_sha256=split_manifest_sha256,
        dilation=dilation,
    )


def build_teacher_state_change_gate_v1(
    *,
    source_occupancy: Any,
    target_occupancy_in_source_coordinates: Any,
    contact_corridor: Optional[Any] = None,
    valid: Optional[Any] = None,
    dilation: int = 1,
    teacher_receipt: GateProvenanceReceiptV1,
) -> StateChangeGateV1:
    """Build a training-only delete/create gate from aligned occupancy states.

    ``target_occupancy_in_source_coordinates`` must be camera/correspondence
    aligned by an upstream qualified teacher.  Raw source/anchor latent
    difference is intentionally not accepted by this API.
    """

    torch = _torch()
    delete, create, contact, validity, hard_support, state_payload = (
        _derive_teacher_gate_fields_v1(
            source_occupancy=source_occupancy,
            target_occupancy_in_source_coordinates=target_occupancy_in_source_coordinates,
            contact_corridor=contact_corridor,
            valid=valid,
            dilation=dilation,
        )
    )
    if not isinstance(teacher_receipt, GateProvenanceReceiptV1):
        raise ActionRegenerationRouterError("teacher gate requires a structured receipt")
    if teacher_receipt.role != "q_y_state_change_teacher" or teacher_receipt.origin != "clean_source_target_pair":
        raise ActionRegenerationRouterError("anchor-derived gate cannot point-supervise")
    if teacher_receipt.state_payload_sha256 != state_payload or teacher_receipt.dilation != dilation:
        raise ActionRegenerationRouterError("teacher gate inputs differ from receipt")
    regenerate = torch.maximum(torch.maximum(delete, create), contact)
    gate = StateChangeGateV1(
        delete=delete,
        create=create,
        contact_permission=contact,
        regenerate=regenerate,
        preserve=1.0 - regenerate,
        valid=validity,
        hard_authorization_support=hard_support,
        provenance=teacher_receipt,
    )
    validate_state_change_gate_v1(gate)
    return gate


class ActionRegenerationGatePredictorV1(_nn().Module):
    """Predict dense delete/create permission from source features and q_pred."""

    def __init__(self, config: Optional[RegenerationGatePredictorConfigV1] = None) -> None:
        super().__init__()
        torch = _torch()
        nn = _nn()
        self.config = config or RegenerationGatePredictorConfigV1()
        self.config.validate()
        hidden = self.config.hidden_channels
        self.source_projection = nn.Conv3d(
            self.config.source_channels, hidden, kernel_size=1
        )
        self.phase_projection = nn.Linear(self.config.action_width, hidden)
        self.global_projection = nn.Linear(self.config.action_width, hidden)
        groups = next(
            candidate
            for candidate in range(min(32, hidden), 0, -1)
            if hidden % candidate == 0
        )
        self.trunk = nn.Sequential(
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.output = nn.Conv3d(hidden, 3, kernel_size=1)
        probability = self.config.initial_change_probability
        bias = math.log(probability / (1.0 - probability))
        nn.init.normal_(self.output.weight, mean=0.0, std=self.config.output_weight_std)
        nn.init.constant_(self.output.bias, bias)
        self.register_buffer(
            "_abi",
            torch.tensor(
                (
                    1,
                    self.config.phase_count,
                    self.config.source_channels,
                    self.config.action_width,
                    self.config.hidden_channels,
                ),
                dtype=torch.int64,
            ),
            persistent=True,
        )

    def forward(self, source_features: Any, plan: Any) -> Tuple[Any, Any, Any]:
        torch = _torch()
        _validate_volume(
            source_features,
            label="source_features",
            channels=self.config.source_channels,
        )
        phase = getattr(plan, "phase_tokens", None)
        global_token = getattr(plan, "global_token", None)
        if (
            not isinstance(phase, torch.Tensor)
            or not isinstance(global_token, torch.Tensor)
            or tuple(phase.shape)
            != (
                int(source_features.shape[0]),
                self.config.phase_count,
                self.config.action_width,
            )
            or tuple(global_token.shape)
            != (int(source_features.shape[0]), self.config.action_width)
        ):
            raise ActionRegenerationRouterError("q_pred geometry differs")
        if phase.device != source_features.device or global_token.device != source_features.device:
            raise ActionRegenerationRouterError("source/q_pred devices differ")
        if (
            not phase.is_floating_point()
            or not global_token.is_floating_point()
            or phase.dtype != global_token.dtype
            or source_features.dtype != phase.dtype
        ):
            raise ActionRegenerationRouterError(
                "source/q_pred must share a floating-point dtype"
            )
        if not bool(torch.isfinite(phase).all()) or not bool(torch.isfinite(global_token).all()):
            raise ActionRegenerationRouterError("q_pred contains non-finite values")
        parameter = next(self.parameters())
        if parameter.device != source_features.device:
            raise ActionRegenerationRouterError("predictor/source devices differ")
        compute_dtype = parameter.dtype
        source = self.source_projection(source_features.to(dtype=compute_dtype))
        phase_condition = self.phase_projection(phase.to(dtype=compute_dtype)).permute(0, 2, 1)
        phase_condition = phase_condition.unsqueeze(-1).unsqueeze(-1)
        global_condition = self.global_projection(global_token.to(dtype=compute_dtype)).unsqueeze(-1)
        global_condition = global_condition.unsqueeze(-1).unsqueeze(-1)
        hidden = source + phase_condition + global_condition
        logits = self.output(self.trunk(hidden))
        delete_logits = logits[:, 0:1]
        create_logits = logits[:, 1:2]
        contact_logits = logits[:, 2:3]
        # Initial-state authority is structural, not learned from a large
        # negative logit.  Consumers must apply the same zero after sigmoid.
        return delete_logits, create_logits, contact_logits


def build_predicted_gate_authorization_receipt_v1(
    *,
    delete_logits: Any,
    create_logits: Any,
    contact_logits: Any,
    hard_authorization_support: Any,
    valid: Any,
    sample_ids: Tuple[str, ...],
    source_instruction_condition_sha256: str,
    predictor_artifact_sha256: str,
    split_manifest_sha256: str,
) -> GateProvenanceReceiptV1:
    raise ActionRegenerationRouterError(
        "predicted support cannot self-authorize; external support authority is not implemented"
    )
    torch = _torch()
    _validate_volume(delete_logits, label="delete_logits", channels=1)
    _validate_volume(create_logits, label="create_logits", channels=1)
    _validate_volume(contact_logits, label="contact_logits", channels=1)
    if (
        tuple(delete_logits.shape) != tuple(create_logits.shape)
        or tuple(delete_logits.shape) != tuple(contact_logits.shape)
        or delete_logits.dtype != create_logits.dtype
        or delete_logits.dtype != contact_logits.dtype
        or delete_logits.device != create_logits.device
        or delete_logits.device != contact_logits.device
    ):
        raise ActionRegenerationRouterError("delete/create/contact logits dtype/device/geometry differs")
    if (
        not isinstance(hard_authorization_support, torch.Tensor)
        or hard_authorization_support.dtype != torch.bool
        or hard_authorization_support.ndim != 5
        or int(hard_authorization_support.shape[1]) != 1
        or int(hard_authorization_support.shape[2]) != PHASE_COUNT
        or tuple(hard_authorization_support.shape) != tuple(delete_logits.shape)
        or hard_authorization_support.device != delete_logits.device
    ):
        raise ActionRegenerationRouterError("hard authorization must be bool [B,1,21,H,W]")
    if (
        not isinstance(valid, torch.Tensor)
        or valid.dtype != torch.bool
        or tuple(valid.shape) != tuple(hard_authorization_support.shape)
        or valid.device != hard_authorization_support.device
    ):
        raise ActionRegenerationRouterError("predicted authorization valid mask differs")
    support = hard_authorization_support & valid
    phase_authority = torch.ones_like(support)
    phase_authority[:, :, 0] = False
    support = support & phase_authority
    if not torch.equal(support, hard_authorization_support):
        raise ActionRegenerationRouterError(
            "hard authorization must already be valid-masked and phase-zero-free"
        )
    delete = torch.sigmoid(delete_logits.float()) * support.float()
    create = torch.sigmoid(create_logits.float()) * support.float()
    contact = torch.sigmoid(contact_logits.float()) * support.float()
    return _make_gate_receipt_v1(
        role="q_pred_hard_authorization",
        origin="source_instruction_support_predictor",
        sample_ids=sample_ids,
        state_payload_sha256=source_instruction_condition_sha256,
        delete=delete,
        create=create,
        contact_permission=contact,
        valid=valid,
        hard_support=support,
        producer_artifact_sha256=predictor_artifact_sha256,
        split_manifest_sha256=split_manifest_sha256,
        dilation=0,
    )


def predicted_state_change_gate_v1(
    delete_logits: Any,
    create_logits: Any,
    contact_logits: Any,
    *,
    hard_authorization_support: Any,
    valid: Any,
    authorization_receipt: GateProvenanceReceiptV1,
) -> StateChangeGateV1:
    raise ActionRegenerationRouterError(
        "predicted gate execution is fail-closed until an external support authority exists"
    )
    torch = _torch()
    _validate_volume(delete_logits, label="delete_logits", channels=1)
    _validate_volume(create_logits, label="create_logits", channels=1)
    _validate_volume(contact_logits, label="contact_logits", channels=1)
    if (
        tuple(delete_logits.shape) != tuple(create_logits.shape)
        or tuple(delete_logits.shape) != tuple(contact_logits.shape)
    ):
        raise ActionRegenerationRouterError("delete/create/contact logits differ")
    if (
        delete_logits.device != create_logits.device
        or delete_logits.device != contact_logits.device
        or delete_logits.dtype != create_logits.dtype
        or delete_logits.dtype != contact_logits.dtype
    ):
        raise ActionRegenerationRouterError("delete/create/contact logits dtype/device differs")
    if (
        not isinstance(hard_authorization_support, torch.Tensor)
        or hard_authorization_support.dtype != torch.bool
        or tuple(hard_authorization_support.shape) != tuple(delete_logits.shape)
        or hard_authorization_support.device != delete_logits.device
        or not isinstance(valid, torch.Tensor)
        or valid.dtype != torch.bool
        or tuple(valid.shape) != tuple(delete_logits.shape)
        or valid.device != delete_logits.device
    ):
        raise ActionRegenerationRouterError("predicted gate support/valid geometry differs")
    if not isinstance(authorization_receipt, GateProvenanceReceiptV1):
        raise ActionRegenerationRouterError("predicted gate requires an authorization receipt")
    if (
        authorization_receipt.role != "q_pred_hard_authorization"
        or authorization_receipt.origin != "source_instruction_support_predictor"
    ):
        raise ActionRegenerationRouterError("anchor-derived support cannot authorize regeneration")
    delete = torch.sigmoid(delete_logits.float())
    create = torch.sigmoid(create_logits.float())
    contact = torch.sigmoid(contact_logits.float())
    support = hard_authorization_support & valid
    delete = delete * support.float()
    create = create * support.float()
    contact = contact * support.float()
    regenerate = torch.maximum(torch.maximum(delete, create), contact)
    gate = StateChangeGateV1(
        delete=delete,
        create=create,
        contact_permission=contact,
        regenerate=regenerate,
        preserve=1.0 - regenerate,
        valid=valid,
        hard_authorization_support=support,
        provenance=authorization_receipt,
    )
    validate_state_change_gate_v1(gate)
    return gate


def _dice_loss(prediction: Any, target: Any, valid: Any, epsilon: float = 1.0e-6) -> Any:
    prediction = prediction.float()[valid]
    target = target.float()[valid]
    if prediction.numel() == 0:
        return prediction.sum() * 0.0
    numerator = 2.0 * (prediction * target).sum()
    denominator = prediction.sum() + target.sum()
    return 1.0 - (numerator + epsilon) / (denominator + epsilon)


def _masked_bce_with_logits(logits: Any, target: Any, valid: Any) -> Any:
    torch = _torch()
    if not bool(valid.any()):
        return logits.float().sum() * 0.0
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.float(), target.float(), reduction="none"
    )
    weights = valid.float()
    return (loss * weights).sum() / weights.sum()


def state_change_gate_loss_v1(
    *,
    delete_logits: Any,
    create_logits: Any,
    contact_logits: Any,
    target: StateChangeGateV1,
    config: Optional[RegenerationGateLossConfigV1] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Distill only a clean-target q_y gate; anchors cannot point-supervise it."""

    torch = _torch()
    if not isinstance(target, StateChangeGateV1) or not isinstance(
        target.provenance, GateProvenanceReceiptV1
    ):
        raise ActionRegenerationRouterError("point supervision requires a structured q_y receipt")
    if (
        target.provenance.role != "q_y_state_change_teacher"
        or target.provenance.origin != "clean_source_target_pair"
    ):
        raise ActionRegenerationRouterError(
            "anchor-derived gate cannot enter q_y point supervision"
        )
    cfg = config or RegenerationGateLossConfigV1()
    cfg.validate()
    validate_state_change_gate_v1(target)
    _validate_volume(delete_logits, label="delete_logits", channels=1)
    _validate_volume(create_logits, label="create_logits", channels=1)
    _validate_volume(contact_logits, label="contact_logits", channels=1)
    if (
        tuple(delete_logits.shape) != tuple(target.delete.shape)
        or tuple(create_logits.shape) != tuple(target.create.shape)
        or tuple(contact_logits.shape) != tuple(target.contact_permission.shape)
    ):
        raise ActionRegenerationRouterError("gate logits/target geometry differs")
    delete_target = target.delete.detach().float()
    create_target = target.create.detach().float()
    contact_target = target.contact_permission.detach().float()
    if (
        delete_logits.device != target.delete.device
        or create_logits.device != target.create.device
        or contact_logits.device != target.contact_permission.device
    ):
        raise ActionRegenerationRouterError("gate logits/target devices differ")
    valid = target.valid
    delete_bce = _masked_bce_with_logits(delete_logits, delete_target, valid)
    create_bce = _masked_bce_with_logits(create_logits, create_target, valid)
    contact_bce = _masked_bce_with_logits(contact_logits, contact_target, valid)
    predicted_delete = torch.sigmoid(delete_logits.float()) * valid.float()
    predicted_create = torch.sigmoid(create_logits.float()) * valid.float()
    predicted_contact = torch.sigmoid(contact_logits.float()) * valid.float()
    predicted_union = torch.maximum(
        torch.maximum(predicted_delete, predicted_create), predicted_contact
    )
    union_dice = _dice_loss(
        predicted_union, target.regenerate.detach(), valid
    )
    valid_mass = valid.float().sum(dim=(-2, -1))
    phase_estimable = valid_mass > 0.0
    predicted_mass = (predicted_union * valid.float()).sum(dim=(-2, -1)) / valid_mass.clamp_min(1.0)
    target_mass = (target.regenerate.detach().float() * valid.float()).sum(dim=(-2, -1)) / valid_mass.clamp_min(1.0)
    if bool(phase_estimable.any()):
        phase_mass = torch.nn.functional.l1_loss(
            predicted_mass[phase_estimable], target_mass[phase_estimable]
        )
    else:
        phase_mass = predicted_mass.sum() * 0.0
    total = (
        cfg.delete_bce_weight * delete_bce
        + cfg.create_bce_weight * create_bce
        + cfg.contact_bce_weight * contact_bce
        + cfg.union_dice_weight * union_dice
        + cfg.phase_mass_weight * phase_mass
    )
    return total, {
        "delete_bce": delete_bce,
        "create_bce": create_bce,
        "contact_bce": contact_bce,
        "union_dice": union_dice,
        "phase_mass": phase_mass,
        "point_teacher_receipt_role": target.provenance.role,
        "teacher_receipt_sha256": target.provenance.receipt_sha256,
        "valid_voxels": int(valid.sum().item()),
        "invalid_voxels_used_as_negatives": False,
        "q_anchor_point_gate_used": False,
        "local_mechanical_loss_only": True,
        "optimizer_authorized": False,
    }


_TENSOR_ARTIFACT_ROLES = (
    "source_correlated_noise",
    "independent_regeneration_noise",
    "source_aware_clean",
    "high_r2v_clean",
)


def _storage_nbytes(value: Any) -> int:
    if hasattr(value, "untyped_storage"):
        return int(value.untyped_storage().nbytes())
    return int(value.storage().size()) * int(value.element_size())


def _validate_detached_owned_tensor_v1(value: Any, *, label: str) -> None:
    _validate_volume(value, label=label)
    if value.requires_grad or value.grad_fn is not None or not value.is_leaf:
        raise ActionRegenerationRouterError(
            "%s must be a detached leaf with no autograd provenance" % label
        )
    if not value.is_contiguous() or int(value.storage_offset()) != 0:
        raise ActionRegenerationRouterError("%s must own a contiguous zero-offset buffer" % label)
    expected_bytes = int(value.numel()) * int(value.element_size())
    if _storage_nbytes(value) != expected_bytes:
        raise ActionRegenerationRouterError("%s must own its complete storage" % label)


def build_regeneration_tensor_artifact_receipt_v1(
    *,
    tensor: Any,
    role: str,
    sample_ids: Tuple[str, ...],
    producer_artifact_sha256: str,
    producer_checkpoint_sha256: str,
    producer_config_sha256: str,
    producer_frozen: bool,
    input_payload_sha256: str,
    source_identity_sha256: str,
    external_manifest_sha256: str,
    derivation_key_sha256: str,
    solver_sigma: float,
    solver_step: int,
) -> RegenerationTensorArtifactReceiptV1:
    raise ActionRegenerationRouterError(
        "tensor artifacts cannot self-authorize; external frozen-expert authority is not implemented"
    )
    _validate_detached_owned_tensor_v1(
        tensor, label=role if type(role) is str else "tensor_artifact"
    )
    if role not in _TENSOR_ARTIFACT_ROLES:
        raise ActionRegenerationRouterError("tensor artifact role differs")
    samples = _validated_ids(sample_ids, label="tensor artifact sample_ids", expected=int(tensor.shape[0]))
    if producer_frozen is not True:
        raise ActionRegenerationRouterError("regeneration tensor producer must be frozen")
    sigma = float(solver_sigma)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ActionRegenerationRouterError("solver_sigma must be finite and non-negative")
    if type(solver_step) is not int or solver_step < 0:
        raise ActionRegenerationRouterError("solver_step must be a non-negative integer")
    payload = {
        "schema_version": TENSOR_RECEIPT_SCHEMA_VERSION,
        "role": role,
        "sample_ids": list(samples),
        "tensor_sha256": _tensor_sha256(tensor, label=role),
        "producer_artifact_sha256": _require_sha256(
            producer_artifact_sha256, label="producer_artifact_sha256"
        ),
        "producer_checkpoint_sha256": _require_sha256(
            producer_checkpoint_sha256, label="producer_checkpoint_sha256"
        ),
        "producer_config_sha256": _require_sha256(
            producer_config_sha256, label="producer_config_sha256"
        ),
        "producer_frozen": True,
        "input_payload_sha256": _require_sha256(
            input_payload_sha256, label="input_payload_sha256"
        ),
        "source_identity_sha256": _require_sha256(
            source_identity_sha256, label="source_identity_sha256"
        ),
        "external_manifest_sha256": _require_sha256(
            external_manifest_sha256, label="external_manifest_sha256"
        ),
        "derivation_key_sha256": _require_sha256(
            derivation_key_sha256, label="derivation_key_sha256"
        ),
        "solver_sigma": repr(sigma),
        "solver_step": solver_step,
    }
    return RegenerationTensorArtifactReceiptV1(
        schema_version=payload["schema_version"],
        role=payload["role"],
        sample_ids=samples,
        tensor_sha256=payload["tensor_sha256"],
        producer_artifact_sha256=payload["producer_artifact_sha256"],
        producer_checkpoint_sha256=payload["producer_checkpoint_sha256"],
        producer_config_sha256=payload["producer_config_sha256"],
        producer_frozen=True,
        input_payload_sha256=payload["input_payload_sha256"],
        source_identity_sha256=payload["source_identity_sha256"],
        external_manifest_sha256=payload["external_manifest_sha256"],
        derivation_key_sha256=payload["derivation_key_sha256"],
        solver_sigma=sigma,
        solver_step=solver_step,
        receipt_sha256=_canonical_sha256(payload),
    )


def _validate_tensor_artifact_receipt_v1(
    receipt: RegenerationTensorArtifactReceiptV1,
    *,
    tensor: Any,
    expected_role: str,
    expected_sample_ids: Tuple[str, ...],
) -> None:
    if not isinstance(receipt, RegenerationTensorArtifactReceiptV1):
        raise ActionRegenerationRouterError("regeneration tensor requires an artifact receipt")
    _validate_detached_owned_tensor_v1(tensor, label=expected_role)
    samples = _validated_ids(
        receipt.sample_ids,
        label="tensor artifact sample_ids",
        expected=int(tensor.shape[0]),
    )
    payload = {
        "schema_version": receipt.schema_version,
        "role": receipt.role,
        "sample_ids": list(samples),
        "tensor_sha256": receipt.tensor_sha256,
        "producer_artifact_sha256": receipt.producer_artifact_sha256,
        "producer_checkpoint_sha256": receipt.producer_checkpoint_sha256,
        "producer_config_sha256": receipt.producer_config_sha256,
        "producer_frozen": receipt.producer_frozen,
        "input_payload_sha256": receipt.input_payload_sha256,
        "source_identity_sha256": receipt.source_identity_sha256,
        "external_manifest_sha256": receipt.external_manifest_sha256,
        "derivation_key_sha256": receipt.derivation_key_sha256,
        "solver_sigma": repr(float(receipt.solver_sigma)),
        "solver_step": receipt.solver_step,
    }
    if receipt.schema_version != TENSOR_RECEIPT_SCHEMA_VERSION or receipt.role != expected_role:
        raise ActionRegenerationRouterError("regeneration tensor receipt role/schema differs")
    if receipt.producer_frozen is not True:
        raise ActionRegenerationRouterError("regeneration tensor producer is not frozen")
    if not math.isfinite(float(receipt.solver_sigma)) or float(receipt.solver_sigma) < 0.0:
        raise ActionRegenerationRouterError("regeneration tensor solver sigma differs")
    if type(receipt.solver_step) is not int or receipt.solver_step < 0:
        raise ActionRegenerationRouterError("regeneration tensor solver step differs")
    if samples != expected_sample_ids:
        raise ActionRegenerationRouterError("tensor/gate sample order differs")
    for name in (
        "tensor_sha256",
        "producer_artifact_sha256",
        "producer_checkpoint_sha256",
        "producer_config_sha256",
        "input_payload_sha256",
        "source_identity_sha256",
        "external_manifest_sha256",
        "derivation_key_sha256",
        "receipt_sha256",
    ):
        _require_sha256(getattr(receipt, name), label=name)
    if receipt.tensor_sha256 != _tensor_sha256(tensor, label=expected_role):
        raise ActionRegenerationRouterError("regeneration tensor bytes differ from receipt")
    if receipt.receipt_sha256 != _canonical_sha256(payload):
        raise ActionRegenerationRouterError("regeneration tensor receipt digest differs")


def _tensor_byte_interval(value: Any) -> Tuple[int, int]:
    _validate_volume(value, label="byte_interval_tensor")
    element_size = int(value.element_size())
    minimum_offset = 0
    maximum_offset = 0
    for size, stride in zip(value.shape, value.stride()):
        extent = (int(size) - 1) * int(stride)
        minimum_offset += min(0, extent)
        maximum_offset += max(0, extent)
    first_address = int(value.data_ptr()) + minimum_offset * element_size
    exclusive_end = int(value.data_ptr()) + (maximum_offset + 1) * element_size
    return first_address, exclusive_end


def tensor_byte_ranges_overlap_v1(first: Any, second: Any) -> bool:
    """Detect real address-range overlap, including distinct Storage wrappers."""

    _validate_volume(first, label="first_overlap_tensor")
    _validate_volume(second, label="second_overlap_tensor")
    if first.device != second.device:
        return False
    first_start, first_end = _tensor_byte_interval(first)
    second_start, second_end = _tensor_byte_interval(second)
    return bool(max(first_start, second_start) < min(first_end, second_end))


def _shares_storage(first: Any, second: Any) -> bool:
    return tensor_byte_ranges_overlap_v1(first, second)


def mix_regeneration_noise_v1(
    *,
    source_correlated_noise: Any,
    independent_regeneration_noise: Any,
    gate: StateChangeGateV1,
    source_receipt: RegenerationTensorArtifactReceiptV1,
    independent_receipt: RegenerationTensorArtifactReceiptV1,
) -> Any:
    """Return source identity for G=0; reject every active route fail-closed."""

    torch = _torch()
    _validate_volume(source_correlated_noise, label="source_correlated_noise")
    validate_state_change_gate_v1(gate, reference=source_correlated_noise)
    if not bool((gate.regenerate != 0.0).any()):
        return source_correlated_noise
    raise ActionRegenerationRouterError(
        "active noise routing is fail-closed until external artifact authority is implemented"
    )
    _validate_volume(
        independent_regeneration_noise, label="independent_regeneration_noise"
    )
    if tuple(source_correlated_noise.shape) != tuple(independent_regeneration_noise.shape):
        raise ActionRegenerationRouterError("source/regeneration noise geometry differs")
    if (
        source_correlated_noise.dtype != independent_regeneration_noise.dtype
        or source_correlated_noise.device != independent_regeneration_noise.device
    ):
        raise ActionRegenerationRouterError("source/regeneration noise dtype/device differs")
    validate_state_change_gate_v1(gate, reference=source_correlated_noise)
    _validate_tensor_artifact_receipt_v1(
        source_receipt,
        tensor=source_correlated_noise,
        expected_role="source_correlated_noise",
        expected_sample_ids=gate.provenance.sample_ids,
    )
    _validate_tensor_artifact_receipt_v1(
        independent_receipt,
        tensor=independent_regeneration_noise,
        expected_role="independent_regeneration_noise",
        expected_sample_ids=gate.provenance.sample_ids,
    )
    if (
        source_receipt.solver_sigma != independent_receipt.solver_sigma
        or source_receipt.solver_step != independent_receipt.solver_step
    ):
        raise ActionRegenerationRouterError("noise receipts refer to different solver states")
    if source_receipt.derivation_key_sha256 == independent_receipt.derivation_key_sha256:
        raise ActionRegenerationRouterError("independent noise reuses the source derivation key")
    if source_receipt.tensor_sha256 == independent_receipt.tensor_sha256:
        raise ActionRegenerationRouterError("independent noise duplicates source-noise bytes")
    if (
        source_receipt.source_identity_sha256 != independent_receipt.source_identity_sha256
        or source_receipt.external_manifest_sha256
        != independent_receipt.external_manifest_sha256
    ):
        raise ActionRegenerationRouterError("noise receipts do not share source/manifest identity")
    if source_receipt.external_manifest_sha256 != gate.provenance.split_manifest_sha256:
        raise ActionRegenerationRouterError("noise receipts are outside the gate's pinned manifest")
    if _shares_storage(source_correlated_noise, independent_regeneration_noise):
        raise ActionRegenerationRouterError("regeneration noise storage overlaps source noise")
    original_mask = gate.regenerate
    if not bool((original_mask != 0.0).any()):
        return source_correlated_noise
    support = gate.hard_authorization_support.expand_as(source_correlated_noise)
    hard_one = support & (original_mask.expand_as(source_correlated_noise) == 1.0)
    compute_dtype = (
        torch.float64
        if source_correlated_noise.dtype == torch.float64
        else torch.float32
    )
    mask = original_mask.to(dtype=compute_dtype)
    blended = (
        (1.0 - mask) * source_correlated_noise.to(dtype=compute_dtype)
        + mask * independent_regeneration_noise.to(dtype=compute_dtype)
    ).to(dtype=source_correlated_noise.dtype)
    mixed = torch.where(
        hard_one,
        independent_regeneration_noise,
        torch.where(support, blended, source_correlated_noise),
    )
    outside = ~support
    if not torch.equal(mixed[outside], source_correlated_noise[outside]):
        raise ActionRegenerationRouterError("noise changed outside hard authorization")
    return mixed


def route_high_r2v_regeneration_v1(
    *,
    source_aware_clean: Any,
    high_r2v_clean: Any,
    gate: StateChangeGateV1,
    source_receipt: RegenerationTensorArtifactReceiptV1,
    high_r2v_receipt: RegenerationTensorArtifactReceiptV1,
) -> RegenerationRouteResultV1:
    """Return source identity for G=0; reject active high-R2V routing."""

    torch = _torch()
    _validate_volume(source_aware_clean, label="source_aware_clean")
    validate_state_change_gate_v1(gate, reference=source_aware_clean)
    if not bool((gate.regenerate != 0.0).any()):
        return RegenerationRouteResultV1(
            clean=source_aware_clean,
            gate=gate,
            diagnostics={
                "semantic_noop": True,
                "same_object": True,
                "same_bytes": True,
                "active_execution_authorized": False,
            },
        )
    raise ActionRegenerationRouterError(
        "active high-R2V routing is fail-closed until external artifact authority is implemented"
    )
    _validate_volume(high_r2v_clean, label="high_r2v_clean")
    if tuple(source_aware_clean.shape) != tuple(high_r2v_clean.shape):
        raise ActionRegenerationRouterError("R2V/V2V clean geometry differs")
    if source_aware_clean.dtype != high_r2v_clean.dtype or source_aware_clean.device != high_r2v_clean.device:
        raise ActionRegenerationRouterError("R2V/V2V clean dtype/device differs")
    validate_state_change_gate_v1(gate, reference=source_aware_clean)
    _validate_tensor_artifact_receipt_v1(
        source_receipt,
        tensor=source_aware_clean,
        expected_role="source_aware_clean",
        expected_sample_ids=gate.provenance.sample_ids,
    )
    _validate_tensor_artifact_receipt_v1(
        high_r2v_receipt,
        tensor=high_r2v_clean,
        expected_role="high_r2v_clean",
        expected_sample_ids=gate.provenance.sample_ids,
    )
    if (
        source_receipt.solver_sigma != high_r2v_receipt.solver_sigma
        or source_receipt.solver_step != high_r2v_receipt.solver_step
    ):
        raise ActionRegenerationRouterError("R2V/V2V receipts refer to different solver states")
    if source_receipt.derivation_key_sha256 == high_r2v_receipt.derivation_key_sha256:
        raise ActionRegenerationRouterError("high-R2V branch reuses the source derivation key")
    if source_receipt.tensor_sha256 == high_r2v_receipt.tensor_sha256:
        raise ActionRegenerationRouterError("high-R2V output duplicates source-aware bytes")
    if (
        source_receipt.source_identity_sha256 != high_r2v_receipt.source_identity_sha256
        or source_receipt.external_manifest_sha256
        != high_r2v_receipt.external_manifest_sha256
    ):
        raise ActionRegenerationRouterError("R2V/V2V receipts do not share source/manifest identity")
    if source_receipt.external_manifest_sha256 != gate.provenance.split_manifest_sha256:
        raise ActionRegenerationRouterError("R2V/V2V receipts are outside the gate's pinned manifest")
    if _shares_storage(source_aware_clean, high_r2v_clean):
        raise ActionRegenerationRouterError("R2V/V2V clean storage overlaps")
    original_mask = gate.regenerate
    hard_one_voxels = 0
    soft_active_voxels = 0
    if not bool((original_mask != 0.0).any()):
        clean = source_aware_clean
    else:
        support = gate.hard_authorization_support.expand_as(source_aware_clean)
        hard_one = support & (original_mask.expand_as(source_aware_clean) == 1.0)
        soft_active = (
            support
            & (original_mask.expand_as(source_aware_clean) != 0.0)
            & ~hard_one
        )
        hard_one_voxels = int(hard_one.sum().detach().cpu().item())
        soft_active_voxels = int(soft_active.sum().detach().cpu().item())
        compute_dtype = (
            torch.float64 if source_aware_clean.dtype == torch.float64 else torch.float32
        )
        mask = original_mask.to(dtype=compute_dtype)
        blended = (
            (1.0 - mask) * source_aware_clean.to(dtype=compute_dtype)
            + mask * high_r2v_clean.to(dtype=compute_dtype)
        ).to(dtype=source_aware_clean.dtype)
        clean = torch.where(
            hard_one,
            high_r2v_clean,
            torch.where(support, blended, source_aware_clean),
        )
    outside = (~gate.hard_authorization_support).expand_as(clean)
    if not torch.equal(clean[outside], source_aware_clean[outside]):
        raise ActionRegenerationRouterError("clean route changed outside gate")
    if not torch.equal(clean[:, :, 0], source_aware_clean[:, :, 0]):
        raise ActionRegenerationRouterError("clean route changed phase zero")
    diagnostics = {
        "delete_fraction": float(gate.delete.float().mean().detach().cpu().item()),
        "create_fraction": float(gate.create.float().mean().detach().cpu().item()),
        "regenerate_fraction": float(gate.regenerate.float().mean().detach().cpu().item()),
        "phase0_max_abs_delta": float(
            (clean[:, :, 0].float() - source_aware_clean[:, :, 0].float())
            .abs()
            .amax()
            .detach()
            .cpu()
            .item()
        ),
        "outside_hard_authorization_exact": True,
        "hard_one_is_exact_tensor_selection": True,
        "soft_gate_is_numeric_blend_only": True,
        "hard_one_channel_voxels": hard_one_voxels,
        "soft_active_channel_voxels": soft_active_voxels,
        "high_r2v_is_execution_expert_not_action_code": True,
    }
    return RegenerationRouteResultV1(clean=clean, gate=gate, diagnostics=diagnostics)


def state_change_phase_plan_v1(gate: StateChangeGateV1) -> Any:
    """Build a receipt-bound plan; only its exact no-op is executable here."""

    torch = _torch()
    validate_state_change_gate_v1(gate)
    try:
        from methods.bernini_action_editing.spt_v2 import phase_transport as spt
    except ImportError:
        try:
            from spt_v2 import phase_transport as spt  # type: ignore
        except ImportError as error:  # pragma: no cover - deployment layout
            raise ActionRegenerationRouterError(
                "audited spt_v2.phase_transport is unavailable"
            ) from error
    batch, _, phases, height, width = map(int, gate.regenerate.shape)
    offsets = torch.zeros(
        batch, 3, phases, height, width,
        device=gate.regenerate.device,
        dtype=torch.float32,
    )
    gates = torch.zeros_like(offsets)
    gates[:, spt.GATE_PRESERVE] = gate.preserve[:, 0].float()
    gates[:, spt.GATE_GENERATE] = gate.regenerate[:, 0].float()
    semantic_noop = not bool((gate.regenerate != 0.0).any())
    receipt_payload = {
        "schema_version": PHASE_PLAN_RECEIPT_SCHEMA_VERSION,
        "sample_ids": list(gate.provenance.sample_ids),
        "gate_receipt_sha256": gate.provenance.receipt_sha256,
        "offsets_sha256": _tensor_sha256(offsets, label="phase_plan.offsets"),
        "gate_probs_sha256": _tensor_sha256(gates, label="phase_plan.gate_probs"),
    }
    execution_receipt = PhasePlanExecutionReceiptV1(
        schema_version=receipt_payload["schema_version"],
        sample_ids=gate.provenance.sample_ids,
        gate_receipt_sha256=receipt_payload["gate_receipt_sha256"],
        offsets_sha256=receipt_payload["offsets_sha256"],
        gate_probs_sha256=receipt_payload["gate_probs_sha256"],
        receipt_sha256=_canonical_sha256(receipt_payload),
    )
    plan = spt.PhasePlan(
        offsets=offsets,
        gate_probs=gates,
        provenance="student",
        diagnostics={
            "schema_version": SCHEMA_VERSION,
            "delete": gate.delete,
            "create": gate.create,
            "contact_permission": gate.contact_permission,
            "regeneration_union": gate.regenerate,
            "valid": gate.valid,
            "hard_authorization_support": gate.hard_authorization_support,
            "gate_receipt_sha256": gate.provenance.receipt_sha256,
            "execution_receipt": execution_receipt,
            "semantic_noop": semantic_noop,
            "transport_disabled_in_v1_canary": True,
        },
    )
    reference = torch.zeros(
        batch, phases, height, width, 1,
        device=gate.regenerate.device,
        dtype=torch.float32,
    )
    plan.validate(reference)
    return plan


def execute_state_change_phase_plan_clean_v1(
    *,
    source_clean: Any,
    generated_clean: Any,
    plan: Any,
    detach_source_bank: bool = True,
) -> Any:
    """Execute the bridge while preserving semantic G=0 object/bytes identity."""

    try:
        from methods.bernini_action_editing.spt_v2 import phase_transport as spt
    except ImportError:
        try:
            from spt_v2 import phase_transport as spt  # type: ignore
        except ImportError as error:  # pragma: no cover - deployment layout
            raise ActionRegenerationRouterError(
                "audited spt_v2.phase_transport is unavailable"
            ) from error
    torch = _torch()
    diagnostics = getattr(plan, "diagnostics", None)
    if not isinstance(diagnostics, Mapping) or diagnostics.get("schema_version") != SCHEMA_VERSION:
        raise ActionRegenerationRouterError("phase plan is not a regeneration-router bridge")
    receipt = diagnostics.get("execution_receipt")
    if not isinstance(receipt, PhasePlanExecutionReceiptV1):
        raise ActionRegenerationRouterError("phase plan execution receipt is missing")
    samples = _validated_ids(
        receipt.sample_ids,
        label="phase plan sample_ids",
        expected=int(source_clean.shape[0]),
    )
    payload = {
        "schema_version": receipt.schema_version,
        "sample_ids": list(samples),
        "gate_receipt_sha256": receipt.gate_receipt_sha256,
        "offsets_sha256": receipt.offsets_sha256,
        "gate_probs_sha256": receipt.gate_probs_sha256,
    }
    if receipt.schema_version != PHASE_PLAN_RECEIPT_SCHEMA_VERSION:
        raise ActionRegenerationRouterError("phase plan receipt schema differs")
    for name in (
        "gate_receipt_sha256",
        "offsets_sha256",
        "gate_probs_sha256",
        "receipt_sha256",
    ):
        _require_sha256(getattr(receipt, name), label=name)
    if receipt.offsets_sha256 != _tensor_sha256(plan.offsets, label="phase_plan.offsets"):
        raise ActionRegenerationRouterError("phase plan offsets changed after receipt")
    if receipt.gate_probs_sha256 != _tensor_sha256(
        plan.gate_probs, label="phase_plan.gate_probs"
    ):
        raise ActionRegenerationRouterError("phase plan gates changed after receipt")
    if receipt.receipt_sha256 != _canonical_sha256(payload):
        raise ActionRegenerationRouterError("phase plan receipt digest differs")
    if source_clean.device != generated_clean.device or source_clean.dtype != generated_clean.dtype:
        raise ActionRegenerationRouterError("SPT source/generated dtype or device differs")
    plan.validate(source_clean)
    preserve = plan.gate_probs[:, 0]
    transport = plan.gate_probs[:, 1]
    generate = plan.gate_probs[:, 2]
    semantic_noop = bool(
        torch.equal(preserve, torch.ones_like(preserve))
        and torch.equal(transport, torch.zeros_like(transport))
        and torch.equal(generate, torch.zeros_like(generate))
        and torch.equal(plan.offsets, torch.zeros_like(plan.offsets))
    )
    if semantic_noop:
        # Passing noop=True is mandatory: SPT's arithmetic preserve path changes
        # signed-zero bits and allocates a new object even when G is all zero.
        return spt.execute_clean_plan(
            source_clean,
            generated_clean,
            plan,
            noop=True,
            detach_source_bank=detach_source_bank,
        )
    raise ActionRegenerationRouterError(
        "active SPT generation is fail-closed until external artifact authority is implemented"
    )


def contract_v1() -> Mapping[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_count": PHASE_COUNT,
        "gate_axes": [
            "delete",
            "create",
            "contact_permission",
            "regenerate",
            "preserve",
            "valid",
            "hard_authorization_support",
        ],
        "phase_zero_source_authority": True,
        "point_gate_teacher": "receipt_bound_clean_source_target_q_y_local_mechanical_only",
        "point_gate_teacher_receipt_required": True,
        "external_receipt_authenticity_verifier_implemented_here": False,
        "external_receipt_authenticity_hard_blocker_for_training": True,
        "free_form_teacher_role_accepted": False,
        "q_anchor_point_gate_forbidden": True,
        "contact_permission_is_not_create_label": True,
        "dilation_remasked_by_valid": True,
        "invalid_voxels_used_as_negatives": False,
        "soft_probability_separate_from_hard_authorization": True,
        "outside_defined_by_hard_authorization": True,
        "predicted_gate_execution_authorized": False,
        "predicted_gate_external_support_authority_implemented": False,
        "maximum_hard_authorization_fraction_per_phase": MAX_HARD_AUTHORIZATION_FRACTION_PER_PHASE,
        "high_r2v_role": "planned_local_regeneration_execution_expert_fail_closed",
        "high_r2v_is_action_representation": False,
        "noise_reset_scope": "regeneration_gate_only_step0_canary",
        "noise_mixer_replaces_flowedit_algebra": False,
        "outside_gate_clean_identity": "source_aware_clean_exact",
        "spt_executor_reused": True,
        "spt_semantic_noop_wrapper_required": True,
        "zero_gate_returns_same_object_and_bytes": True,
        "tensor_artifact_receipts_required": True,
        "external_artifact_authority_implemented": False,
        "active_noise_routing_authorized": False,
        "active_high_r2v_routing_authorized": False,
        "local_tensor_receipt_builder_enabled": False,
        "tensor_outputs_must_be_detached_owned_leaves": True,
        "tensor_overlap_check": "real_address_interval",
        "phase_plan_execution_receipt_required": True,
        "phase_plan_semantic_noop_recomputed_at_execution": True,
        "active_spt_execution_authorized": False,
        "training_authorized": False,
        "optimizer_authorized": False,
        "gpu_launch_authorized": False,
    }


__all__ = [
    "ACTION_WIDTH",
    "PHASE_COUNT",
    "SCHEMA_VERSION",
    "MAX_HARD_AUTHORIZATION_FRACTION_PER_PHASE",
    "GATE_RECEIPT_SCHEMA_VERSION",
    "TENSOR_RECEIPT_SCHEMA_VERSION",
    "PHASE_PLAN_RECEIPT_SCHEMA_VERSION",
    "ActionRegenerationGatePredictorV1",
    "ActionRegenerationRouterError",
    "RegenerationGateLossConfigV1",
    "RegenerationGatePredictorConfigV1",
    "RegenerationRouteResultV1",
    "RegenerationTensorArtifactReceiptV1",
    "PhasePlanExecutionReceiptV1",
    "GateProvenanceReceiptV1",
    "StateChangeGateV1",
    "build_predicted_gate_authorization_receipt_v1",
    "build_regeneration_tensor_artifact_receipt_v1",
    "build_teacher_gate_receipt_v1",
    "build_teacher_state_change_gate_v1",
    "contract_v1",
    "mix_regeneration_noise_v1",
    "predicted_state_change_gate_v1",
    "route_high_r2v_regeneration_v1",
    "execute_state_change_phase_plan_clean_v1",
    "state_change_gate_loss_v1",
    "state_change_phase_plan_v1",
    "tensor_byte_ranges_overlap_v1",
    "validate_state_change_gate_v1",
]
