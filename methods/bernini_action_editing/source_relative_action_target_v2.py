"""Typed source-relative action target ABI (V2).

This module is deliberately narrower than a representation learner.  It turns
locally digest-bound source/target annotation candidates into one explicit
draft schema and provides a deterministic, lossless transport codec for that
schema.  Neither the local receipt nor a successful round trip proves a clean
pair, an action representation, held-out performance, or permission to train.

The camera frame is asymmetric by construction.  Source and target media may
each be camera-stabilized, but the final origin, scale, and orientation are
derived once from the source phase-0 primary actor and then applied to both
media.  There is no target-side normalization input that could erase action
amplitude.

There is exactly one reserved point-target type,
``QYSourceRelativeActionTargetV2``, but local code cannot mint it without the
future external clean-pair authority.  No anchor-target type, reconstruction
loss, gate, optimizer, or qualification entry point is exposed here.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Tuple


SCHEMA_VERSION = "bernini-source-relative-action-target-v2"
CAMERA_RECEIPT_SCHEMA_VERSION = (
    "bernini-source-relative-camera-canonicalization-receipt-v2"
)
DRAFT_RECEIPT_SCHEMA_VERSION = (
    "bernini-local-source-relative-action-target-draft-receipt-v2"
)
TRANSPORT_RECEIPT_SCHEMA_VERSION = (
    "bernini-frozen-source-relative-schema-transport-receipt-v2"
)
TRANSPORT_SCHEMA_VERSION = "bernini-local-source-relative-schema-transport-v2"
EVALUATION_SCHEMA_VERSION = "bernini-local-source-relative-schema-evaluation-v2"

PHASE_COUNT = 21
ACTOR_SLOT_COUNT = 2
OBJECT_SLOT_COUNT = 4
ENTITY_SLOT_COUNT = ACTOR_SLOT_COUNT + OBJECT_SLOT_COUNT

ACTOR_ROLE_NAMES = ("primary_actor", "co_actor")
OBJECT_ROLE_NAMES = (
    "primary_patient",
    "secondary_patient",
    "instrument",
    "goal_container",
)
OWNERSHIP_NAMES = (
    "free",
    "primary_actor",
    "co_actor",
    "environment",
    "goal_container",
)
PHASE_CHANNEL_NAMES = ("onset", "transition", "terminal", "hold")
ABSTAIN_REASON_NAMES = (
    "camera_reference_invalid",
    "slot_overflow",
    "role_assignment_ambiguous",
    "ownership_ambiguous",
    "primary_actor_missing",
    "phase_or_terminal_hold_invalid",
    "track_coverage_insufficient",
    "lifecycle_or_relation_inconsistent",
    "nonfinite_valid_evidence",
    "terminal_hold_evidence_missing",
    "pre_onset_or_transition_not_action_like",
)

ONSET_CHANNEL = 0
TRANSITION_CHANNEL = 1
TERMINAL_CHANNEL = 2
HOLD_CHANNEL = 3

OWNER_FREE = 0
# Compatibility alias only.  Code zero means known-free iff ownership_valid is
# true; when validity is false the stored zero is padding and means unknown.
OWNER_NONE = OWNER_FREE
OWNER_PRIMARY_ACTOR = 1
OWNER_CO_ACTOR = 2
OWNER_ENVIRONMENT = 3
OWNER_GOAL_CONTAINER = 4
TERMINAL_HOLD_DELTA_TOLERANCE = 1.0e-5
PRE_ONSET_DELTA_TOLERANCE = 1.0e-5
MIN_ACTION_STATE_CHANGE = 1.0e-4


class SourceRelativeActionTargetError(ValueError):
    """Raised when the V2 typed-target contract is violated."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for the source-relative V2 ABI") from error
    return torch


def _nn() -> Any:
    return _torch().nn


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise SourceRelativeActionTargetError(
            "%s must be a non-null SHA-256 hex digest" % label
        )
    lowered = value.lower()
    if any(character not in "0123456789abcdef" for character in lowered):
        raise SourceRelativeActionTargetError(
            "%s must be a non-null SHA-256 hex digest" % label
        )
    if lowered == "0" * 64:
        raise SourceRelativeActionTargetError("%s must not be the null digest" % label)
    return lowered


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: Any, *, label: str) -> str:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise SourceRelativeActionTargetError("%s must be a tensor" % label)
    detached = value.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(detached.dtype), "shape": list(detached.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    raw = bytes(detached.view(torch.uint8).reshape(-1).tolist())
    return hashlib.sha256(header + b"\0" + raw).hexdigest()


def _require_ids(values: Any, *, label: str, expected: Optional[int] = None) -> Tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise SourceRelativeActionTargetError("%s must be a non-empty tuple" % label)
    if expected is not None and len(values) != expected:
        raise SourceRelativeActionTargetError("%s length differs from batch" % label)
    if any(type(value) is not str or not value for value in values):
        raise SourceRelativeActionTargetError("%s entries must be non-empty strings" % label)
    if len(set(values)) != len(values):
        raise SourceRelativeActionTargetError("%s entries must be unique" % label)
    return values


def _require_digest_tuple(
    values: Any, *, label: str, expected: int
) -> Tuple[str, ...]:
    if not isinstance(values, tuple) or len(values) != expected:
        raise SourceRelativeActionTargetError("%s must match the batch" % label)
    return tuple(
        _require_sha256(value, label="%s[%d]" % (label, index))
        for index, value in enumerate(values)
    )


def _require_tensor(
    value: Any,
    *,
    label: str,
    shape: Tuple[int, ...],
    dtype: Any,
    detached_leaf: bool = True,
) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise SourceRelativeActionTargetError("%s must be a tensor" % label)
    if tuple(value.shape) != tuple(shape):
        raise SourceRelativeActionTargetError(
            "%s shape must be %r, got %r" % (label, shape, tuple(value.shape))
        )
    if value.dtype != dtype:
        raise SourceRelativeActionTargetError(
            "%s dtype must be %s" % (label, str(dtype))
        )
    if detached_leaf and (value.requires_grad or value.grad_fn is not None):
        raise SourceRelativeActionTargetError("%s must be a detached leaf" % label)
    return value


def _owned(value: Any) -> Any:
    return value.detach().clone().contiguous()


@dataclass(frozen=True)
class SourceRelativeCameraEvidenceV2:
    """Raw pair coordinates plus per-media camera stabilization.

    ``source_actor_scale`` and ``source_actor_orientation`` are source-media
    annotations.  Only ``[:, 0, 0]`` is used for the final shared coordinate
    frame.  Target scale/orientation are intentionally absent.
    """

    sample_ids: Tuple[str, ...]
    source_media_sha256: Tuple[str, ...]
    target_media_sha256: Tuple[str, ...]
    source_actor_xy: Any
    target_actor_xy: Any
    source_object_xy: Any
    target_object_xy: Any
    source_actor_track_valid: Any
    target_actor_track_valid: Any
    source_object_track_valid: Any
    target_object_track_valid: Any
    source_camera_to_stabilized: Any
    target_camera_to_stabilized: Any
    source_actor_scale: Any
    source_actor_orientation: Any


@dataclass(frozen=True)
class CanonicalizedSourceRelativeCoordinatesV2:
    source_actor_xy: Any
    target_actor_xy: Any
    source_object_xy: Any
    target_object_xy: Any
    source_actor_track_valid: Any
    target_actor_track_valid: Any
    source_object_track_valid: Any
    target_object_track_valid: Any
    actor_track_valid: Any
    object_track_valid: Any
    source_phase0_origin: Any
    source_phase0_scale: Any
    source_phase0_orientation: Any
    camera_reference_valid: Any
    finite_valid_evidence: Any


@dataclass(frozen=True)
class SourceRelativeCameraCanonicalizationReceiptV2:
    schema_version: str
    sample_ids: Tuple[str, ...]
    source_media_sha256: Tuple[str, ...]
    target_media_sha256: Tuple[str, ...]
    evidence_payload_sha256: str
    source_phase0_reference_sha256: str
    canonical_coordinates_sha256: str
    canonicalizer_artifact_sha256: str
    annotation_manifest_sha256: str
    external_authority_verified: bool
    receipt_sha256: str


@dataclass(frozen=True)
class SourceRelativeCameraBundleV2:
    evidence: SourceRelativeCameraEvidenceV2
    coordinates: CanonicalizedSourceRelativeCoordinatesV2
    receipt: SourceRelativeCameraCanonicalizationReceiptV2


@dataclass(frozen=True)
class SourceRelativeActionAnnotationsV2:
    """Target-side semantic annotations; visibility is not presence."""

    actor_slot_valid: Any
    object_slot_valid: Any
    observed_actor_count: Any
    observed_object_count: Any
    role_assignment_unique: Any
    ownership_unambiguous: Any
    source_actor_presence: Any
    target_actor_presence: Any
    source_object_presence: Any
    target_object_presence: Any
    source_presence_valid: Any
    target_presence_valid: Any
    contact: Any
    contact_valid: Any
    ownership: Any
    ownership_valid: Any
    phase_channels: Any
    phase_valid: Any


@dataclass(frozen=True)
class SourceRelativeActionTargetV2:
    """Complete typed payload carried by the local non-teacher draft."""

    actor_roles: Any
    actor_role_valid: Any
    object_roles: Any
    object_role_valid: Any
    actor_delta: Any
    actor_delta_valid: Any
    actor_target_position: Any
    actor_target_position_valid: Any
    object_delta: Any
    object_delta_valid: Any
    object_target_position: Any
    object_target_position_valid: Any
    relative_delta: Any
    relative_delta_valid: Any
    actor_presence: Any
    actor_presence_valid: Any
    object_presence: Any
    object_presence_valid: Any
    source_presence: Any
    source_presence_valid: Any
    initial_source_presence: Any
    initial_source_presence_valid: Any
    entity_create: Any
    entity_delete: Any
    lifecycle_valid: Any
    contact: Any
    contact_valid: Any
    ownership: Any
    ownership_valid: Any
    phase_channels: Any
    phase_valid: Any
    action_start_phase: Any
    action_end_phase: Any
    terminal_hold_start_phase: Any
    duration_phases: Any
    phase_summary_valid: Any
    actor_amplitude: Any
    actor_amplitude_valid: Any
    object_amplitude: Any
    object_amplitude_valid: Any
    relative_amplitude: Any
    relative_amplitude_valid: Any
    mean_speed: Any
    peak_speed: Any
    speed_valid: Any
    sample_valid: Any
    abstain: Any
    abstain_reasons: Any


TARGET_FIELD_NAMES = tuple(field.name for field in fields(SourceRelativeActionTargetV2))


@dataclass(frozen=True)
class LocalSourceRelativeActionTargetDraftReceiptV2:
    """Local integrity binding for a non-teacher schema draft.

    This is not an external qualification token.  Its boolean authority field
    is required to remain false in this module.
    """

    schema_version: str
    sample_ids: Tuple[str, ...]
    source_media_sha256: Tuple[str, ...]
    target_media_sha256: Tuple[str, ...]
    target_payload_sha256: str
    camera_receipt_sha256: str
    annotation_payload_sha256: str
    annotation_artifact_sha256: str
    split_manifest_sha256: str
    external_authority_verified: bool
    receipt_sha256: str


@dataclass(frozen=True)
class LocalSourceRelativeActionTargetDraftV2:
    target: SourceRelativeActionTargetV2
    receipt: LocalSourceRelativeActionTargetDraftReceiptV2


class QYSourceRelativeActionTargetV2:
    """Reserved external-authority point-teacher capability.

    The local V2 module intentionally cannot construct this type.  A future
    external clean-pair authority must define an authenticated promotion
    adapter; until then every attempted construction fails closed.
    """

    __slots__ = ()

    def __new__(cls, *_: Any, **__: Any) -> "QYSourceRelativeActionTargetV2":
        raise SourceRelativeActionTargetError(
            "q_y promotion requires external clean-pair authority; not implemented"
        )


@dataclass(frozen=True)
class FrozenSchemaTransportReceiptV2:
    schema_version: str
    encoder_state_sha256: str
    decoder_state_sha256: str
    evaluator_state_sha256: str
    layout_sha256: str
    implementation_artifact_sha256: str
    external_authority_verified: bool
    receipt_sha256: str


@dataclass(frozen=True)
class LocalSchemaTransportV2:
    schema_version: str
    sample_ids: Tuple[str, ...]
    phase_code: Any
    global_code: Any
    draft_receipt_sha256: str
    codec_receipt_sha256: str
    transport_sha256: str


@dataclass(frozen=True)
class LocalSchemaTransportEvaluationV2:
    schema_version: str
    sample_ids: Tuple[str, ...]
    field_exact: Tuple[Tuple[str, bool], ...]
    field_valid_counts: Tuple[Tuple[str, int], ...]
    eligible_sample_count: int
    abstained_sample_count: int
    exact_roundtrip: bool
    local_checks_passed: bool
    representation_qualification_evidence: bool
    r2_evidence: bool
    formally_qualified: bool
    training_authorized: bool
    optimizer_authorized: bool
    selection_authorized: bool
    gate_authorized: bool
    original_payload_sha256: str
    decoded_payload_sha256: str
    codec_receipt_sha256: str
    receipt_sha256: str


def _validate_camera_evidence_v2(evidence: SourceRelativeCameraEvidenceV2) -> int:
    torch = _torch()
    if not isinstance(evidence, SourceRelativeCameraEvidenceV2):
        raise SourceRelativeActionTargetError("camera evidence type differs")
    batch = len(_require_ids(evidence.sample_ids, label="sample_ids"))
    _require_digest_tuple(
        evidence.source_media_sha256, label="source_media_sha256", expected=batch
    )
    _require_digest_tuple(
        evidence.target_media_sha256, label="target_media_sha256", expected=batch
    )
    float_shapes = {
        "source_actor_xy": (batch, PHASE_COUNT, ACTOR_SLOT_COUNT, 2),
        "target_actor_xy": (batch, PHASE_COUNT, ACTOR_SLOT_COUNT, 2),
        "source_object_xy": (batch, PHASE_COUNT, OBJECT_SLOT_COUNT, 2),
        "target_object_xy": (batch, PHASE_COUNT, OBJECT_SLOT_COUNT, 2),
        "source_camera_to_stabilized": (batch, PHASE_COUNT, 3, 3),
        "target_camera_to_stabilized": (batch, PHASE_COUNT, 3, 3),
        "source_actor_scale": (batch, PHASE_COUNT, ACTOR_SLOT_COUNT),
        "source_actor_orientation": (batch, PHASE_COUNT, ACTOR_SLOT_COUNT, 2),
    }
    bool_shapes = {
        "source_actor_track_valid": (batch, PHASE_COUNT, ACTOR_SLOT_COUNT),
        "target_actor_track_valid": (batch, PHASE_COUNT, ACTOR_SLOT_COUNT),
        "source_object_track_valid": (batch, PHASE_COUNT, OBJECT_SLOT_COUNT),
        "target_object_track_valid": (batch, PHASE_COUNT, OBJECT_SLOT_COUNT),
    }
    devices = set()
    for name, shape in float_shapes.items():
        tensor = _require_tensor(
            getattr(evidence, name),
            label="evidence.%s" % name,
            shape=shape,
            dtype=torch.float32,
        )
        devices.add(tensor.device)
    for name, shape in bool_shapes.items():
        tensor = _require_tensor(
            getattr(evidence, name),
            label="evidence.%s" % name,
            shape=shape,
            dtype=torch.bool,
        )
        devices.add(tensor.device)
    if len(devices) != 1:
        raise SourceRelativeActionTargetError(
            "all camera evidence tensors must share one device"
        )
    return batch


def _camera_evidence_payload_sha256(evidence: SourceRelativeCameraEvidenceV2) -> str:
    _validate_camera_evidence_v2(evidence)
    tensor_names = (
        "source_actor_xy",
        "target_actor_xy",
        "source_object_xy",
        "target_object_xy",
        "source_actor_track_valid",
        "target_actor_track_valid",
        "source_object_track_valid",
        "target_object_track_valid",
        "source_camera_to_stabilized",
        "target_camera_to_stabilized",
        "source_actor_scale",
        "source_actor_orientation",
    )
    return _canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "sample_ids": list(evidence.sample_ids),
            "source_media_sha256": list(evidence.source_media_sha256),
            "target_media_sha256": list(evidence.target_media_sha256),
            "tensors": {
                name: _tensor_sha256(getattr(evidence, name), label="evidence.%s" % name)
                for name in tensor_names
            },
        }
    )


def _apply_homography(points: Any, transforms: Any) -> Any:
    torch = _torch()
    ones = torch.ones(
        (*points.shape[:-1], 1), dtype=points.dtype, device=points.device
    )
    homogeneous = torch.cat((points, ones), dim=-1)
    transformed = torch.einsum("btij,btkj->btki", transforms, homogeneous)
    denominator = transformed[..., 2:3]
    safe_denominator = torch.where(
        denominator.abs() > 1.0e-8,
        denominator,
        torch.ones_like(denominator),
    )
    return transformed[..., :2] / safe_denominator


def _canonicalize_camera_evidence_v2(
    evidence: SourceRelativeCameraEvidenceV2,
) -> CanonicalizedSourceRelativeCoordinatesV2:
    torch = _torch()
    batch = _validate_camera_evidence_v2(evidence)
    device = evidence.source_actor_xy.device

    source_h = evidence.source_camera_to_stabilized
    target_h = evidence.target_camera_to_stabilized
    affine_tail = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32, device=device)
    source_affine = (source_h[..., 2, :] == affine_tail).all(dim=-1)
    target_affine = (target_h[..., 2, :] == affine_tail).all(dim=-1)
    source_det = torch.linalg.det(source_h[..., :2, :2])
    target_det = torch.linalg.det(target_h[..., :2, :2])
    identity2 = torch.eye(2, dtype=torch.float32, device=device)
    source_gram = torch.matmul(
        source_h[..., :2, :2].transpose(-1, -2), source_h[..., :2, :2]
    )
    target_gram = torch.matmul(
        target_h[..., :2, :2].transpose(-1, -2), target_h[..., :2, :2]
    )
    source_rigid = (
        (source_gram - identity2).abs().amax(dim=-1).amax(dim=-1) <= 1.0e-6
    ) & ((source_det - 1.0).abs() <= 1.0e-6)
    target_rigid = (
        (target_gram - identity2).abs().amax(dim=-1).amax(dim=-1) <= 1.0e-6
    ) & ((target_det - 1.0).abs() <= 1.0e-6)
    shared_stabilization = (
        (source_h == target_h).all(dim=3).all(dim=2).all(dim=1)
    )
    matrix_ok = (
        torch.isfinite(source_h).all(dim=-1).all(dim=-1)
        & torch.isfinite(target_h).all(dim=-1).all(dim=-1)
        & source_affine
        & target_affine
        & torch.isfinite(source_det)
        & torch.isfinite(target_det)
        & (source_det.abs() > 1.0e-8)
        & (target_det.abs() > 1.0e-8)
        & source_rigid
        & target_rigid
    ).all(dim=1)
    matrix_ok = matrix_ok & shared_stabilization

    source_primary_track = evidence.source_actor_track_valid[:, 0, 0]
    raw_scale = evidence.source_actor_scale[:, 0, 0]
    raw_orientation = evidence.source_actor_orientation[:, 0, 0]
    reference_finite = (
        torch.isfinite(evidence.source_actor_xy[:, 0, 0]).all(dim=-1)
        & torch.isfinite(raw_scale)
        & torch.isfinite(raw_orientation).all(dim=-1)
    )

    identity = torch.eye(3, dtype=torch.float32, device=device).reshape(1, 1, 3, 3)
    safe_source_h = torch.where(
        matrix_ok[:, None, None, None], source_h, identity
    )
    safe_target_h = torch.where(
        matrix_ok[:, None, None, None], target_h, identity
    )
    safe_source_actor = torch.nan_to_num(evidence.source_actor_xy)
    safe_target_actor = torch.nan_to_num(evidence.target_actor_xy)
    safe_source_object = torch.nan_to_num(evidence.source_object_xy)
    safe_target_object = torch.nan_to_num(evidence.target_object_xy)

    source_actor_stabilized = _apply_homography(safe_source_actor, safe_source_h)
    target_actor_stabilized = _apply_homography(safe_target_actor, safe_target_h)
    source_object_stabilized = _apply_homography(safe_source_object, safe_source_h)
    target_object_stabilized = _apply_homography(safe_target_object, safe_target_h)

    origin = source_actor_stabilized[:, 0, 0]
    source_linear0 = safe_source_h[:, 0, :2, :2]
    oriented = torch.einsum(
        "bij,bj->bi", source_linear0, torch.nan_to_num(raw_orientation)
    )
    orientation_norm = torch.linalg.vector_norm(oriented, dim=-1)
    orientation_ok = torch.isfinite(orientation_norm) & (orientation_norm > 1.0e-8)
    unit_orientation = oriented / orientation_norm.clamp_min(1.0e-8).unsqueeze(-1)
    unit_orientation = torch.where(
        orientation_ok[:, None],
        unit_orientation,
        torch.tensor((1.0, 0.0), dtype=torch.float32, device=device),
    )
    stabilized_scale = torch.nan_to_num(raw_scale) * source_det[:, 0].abs().sqrt()
    scale_ok = torch.isfinite(stabilized_scale) & (stabilized_scale > 1.0e-8)
    safe_scale = torch.where(scale_ok, stabilized_scale, torch.ones_like(stabilized_scale))
    camera_reference_valid = (
        matrix_ok
        & source_primary_track
        & reference_finite
        & orientation_ok
        & scale_ok
    )

    perpendicular = torch.stack((-unit_orientation[:, 1], unit_orientation[:, 0]), dim=-1)

    def shared_frame(points: Any) -> Any:
        centered = points - origin[:, None, None, :]
        x_value = (centered * unit_orientation[:, None, None, :]).sum(dim=-1)
        y_value = (centered * perpendicular[:, None, None, :]).sum(dim=-1)
        return torch.stack((x_value, y_value), dim=-1) / safe_scale[:, None, None, None]

    source_actor = shared_frame(source_actor_stabilized)
    target_actor = shared_frame(target_actor_stabilized)
    source_object = shared_frame(source_object_stabilized)
    target_object = shared_frame(target_object_stabilized)

    source_actor_track_valid = evidence.source_actor_track_valid
    target_actor_track_valid = evidence.target_actor_track_valid
    source_object_track_valid = evidence.source_object_track_valid
    target_object_track_valid = evidence.target_object_track_valid
    actor_track_valid = source_actor_track_valid & target_actor_track_valid
    object_track_valid = source_object_track_valid & target_object_track_valid
    finite_valid_evidence = (
        (
            (~source_actor_track_valid)
            | torch.isfinite(evidence.source_actor_xy).all(dim=-1)
        ).all(dim=2).all(dim=1)
        & (
            (~target_actor_track_valid)
            | torch.isfinite(evidence.target_actor_xy).all(dim=-1)
        ).all(dim=2).all(dim=1)
        & (
            (~source_object_track_valid)
            | torch.isfinite(evidence.source_object_xy).all(dim=-1)
        ).all(dim=2).all(dim=1)
        & (
            (~target_object_track_valid)
            | torch.isfinite(evidence.target_object_xy).all(dim=-1)
        ).all(dim=2).all(dim=1)
    )

    source_actor_track_valid = (
        source_actor_track_valid & camera_reference_valid[:, None, None]
    )
    target_actor_track_valid = (
        target_actor_track_valid & camera_reference_valid[:, None, None]
    )
    source_object_track_valid = (
        source_object_track_valid & camera_reference_valid[:, None, None]
    )
    target_object_track_valid = (
        target_object_track_valid & camera_reference_valid[:, None, None]
    )
    actor_track_valid = actor_track_valid & camera_reference_valid[:, None, None]
    object_track_valid = object_track_valid & camera_reference_valid[:, None, None]
    source_actor = torch.where(
        source_actor_track_valid[..., None],
        source_actor,
        torch.zeros_like(source_actor),
    )
    target_actor = torch.where(
        target_actor_track_valid[..., None],
        target_actor,
        torch.zeros_like(target_actor),
    )
    source_object = torch.where(
        source_object_track_valid[..., None],
        source_object,
        torch.zeros_like(source_object),
    )
    target_object = torch.where(
        target_object_track_valid[..., None],
        target_object,
        torch.zeros_like(target_object),
    )

    return CanonicalizedSourceRelativeCoordinatesV2(
        source_actor_xy=_owned(source_actor),
        target_actor_xy=_owned(target_actor),
        source_object_xy=_owned(source_object),
        target_object_xy=_owned(target_object),
        source_actor_track_valid=_owned(source_actor_track_valid),
        target_actor_track_valid=_owned(target_actor_track_valid),
        source_object_track_valid=_owned(source_object_track_valid),
        target_object_track_valid=_owned(target_object_track_valid),
        actor_track_valid=_owned(actor_track_valid),
        object_track_valid=_owned(object_track_valid),
        source_phase0_origin=_owned(origin),
        source_phase0_scale=_owned(safe_scale),
        source_phase0_orientation=_owned(unit_orientation),
        camera_reference_valid=_owned(camera_reference_valid),
        finite_valid_evidence=_owned(finite_valid_evidence),
    )


def _canonical_coordinates_payload_sha256(
    coordinates: CanonicalizedSourceRelativeCoordinatesV2,
) -> str:
    if not isinstance(coordinates, CanonicalizedSourceRelativeCoordinatesV2):
        raise SourceRelativeActionTargetError("canonical coordinates type differs")
    return _canonical_sha256(
        {
            field.name: _tensor_sha256(
                getattr(coordinates, field.name), label="coordinates.%s" % field.name
            )
            for field in fields(CanonicalizedSourceRelativeCoordinatesV2)
        }
    )


def _source_reference_sha256(
    coordinates: CanonicalizedSourceRelativeCoordinatesV2,
) -> str:
    return _canonical_sha256(
        {
            name: _tensor_sha256(
                getattr(coordinates, name), label="coordinates.%s" % name
            )
            for name in (
                "source_phase0_origin",
                "source_phase0_scale",
                "source_phase0_orientation",
                "camera_reference_valid",
            )
        }
    )


def build_source_relative_camera_bundle_v2(
    evidence: SourceRelativeCameraEvidenceV2,
    *,
    canonicalizer_artifact_sha256: str,
    annotation_manifest_sha256: str,
) -> SourceRelativeCameraBundleV2:
    """Build a locally tamper-evident, source-framed camera bundle."""

    coordinates = _canonicalize_camera_evidence_v2(evidence)
    payload = {
        "schema_version": CAMERA_RECEIPT_SCHEMA_VERSION,
        "sample_ids": list(evidence.sample_ids),
        "source_media_sha256": list(evidence.source_media_sha256),
        "target_media_sha256": list(evidence.target_media_sha256),
        "evidence_payload_sha256": _camera_evidence_payload_sha256(evidence),
        "source_phase0_reference_sha256": _source_reference_sha256(coordinates),
        "canonical_coordinates_sha256": _canonical_coordinates_payload_sha256(
            coordinates
        ),
        "canonicalizer_artifact_sha256": _require_sha256(
            canonicalizer_artifact_sha256, label="canonicalizer_artifact_sha256"
        ),
        "annotation_manifest_sha256": _require_sha256(
            annotation_manifest_sha256, label="annotation_manifest_sha256"
        ),
        "external_authority_verified": False,
    }
    receipt = SourceRelativeCameraCanonicalizationReceiptV2(
        schema_version=CAMERA_RECEIPT_SCHEMA_VERSION,
        sample_ids=evidence.sample_ids,
        source_media_sha256=evidence.source_media_sha256,
        target_media_sha256=evidence.target_media_sha256,
        evidence_payload_sha256=payload["evidence_payload_sha256"],
        source_phase0_reference_sha256=payload["source_phase0_reference_sha256"],
        canonical_coordinates_sha256=payload["canonical_coordinates_sha256"],
        canonicalizer_artifact_sha256=payload["canonicalizer_artifact_sha256"],
        annotation_manifest_sha256=payload["annotation_manifest_sha256"],
        external_authority_verified=False,
        receipt_sha256=_canonical_sha256(payload),
    )
    return SourceRelativeCameraBundleV2(
        evidence=evidence, coordinates=coordinates, receipt=receipt
    )


def validate_source_relative_camera_bundle_v2(
    bundle: SourceRelativeCameraBundleV2,
) -> None:
    if not isinstance(bundle, SourceRelativeCameraBundleV2):
        raise SourceRelativeActionTargetError("camera bundle type differs")
    _validate_camera_evidence_v2(bundle.evidence)
    receipt = bundle.receipt
    if not isinstance(receipt, SourceRelativeCameraCanonicalizationReceiptV2):
        raise SourceRelativeActionTargetError("camera receipt type differs")
    if receipt.schema_version != CAMERA_RECEIPT_SCHEMA_VERSION:
        raise SourceRelativeActionTargetError("camera receipt schema differs")
    if receipt.external_authority_verified is not False:
        raise SourceRelativeActionTargetError(
            "local camera receipt cannot claim external authority"
        )
    for name in (
        "evidence_payload_sha256",
        "source_phase0_reference_sha256",
        "canonical_coordinates_sha256",
        "canonicalizer_artifact_sha256",
        "annotation_manifest_sha256",
        "receipt_sha256",
    ):
        _require_sha256(getattr(receipt, name), label="camera_receipt.%s" % name)
    if receipt.sample_ids != bundle.evidence.sample_ids:
        raise SourceRelativeActionTargetError("camera receipt sample order differs")
    if receipt.source_media_sha256 != bundle.evidence.source_media_sha256:
        raise SourceRelativeActionTargetError("camera receipt source media differs")
    if receipt.target_media_sha256 != bundle.evidence.target_media_sha256:
        raise SourceRelativeActionTargetError("camera receipt target media differs")
    recomputed_coordinates = _canonicalize_camera_evidence_v2(bundle.evidence)
    if (
        receipt.evidence_payload_sha256
        != _camera_evidence_payload_sha256(bundle.evidence)
    ):
        raise SourceRelativeActionTargetError("camera evidence bytes differ from receipt")
    if (
        receipt.canonical_coordinates_sha256
        != _canonical_coordinates_payload_sha256(bundle.coordinates)
        or receipt.canonical_coordinates_sha256
        != _canonical_coordinates_payload_sha256(recomputed_coordinates)
    ):
        raise SourceRelativeActionTargetError(
            "canonical coordinate bytes differ from receipt"
        )
    if (
        receipt.source_phase0_reference_sha256
        != _source_reference_sha256(bundle.coordinates)
        or receipt.source_phase0_reference_sha256
        != _source_reference_sha256(recomputed_coordinates)
    ):
        raise SourceRelativeActionTargetError(
            "source phase-0 reference differs from receipt"
        )
    payload = {
        "schema_version": receipt.schema_version,
        "sample_ids": list(receipt.sample_ids),
        "source_media_sha256": list(receipt.source_media_sha256),
        "target_media_sha256": list(receipt.target_media_sha256),
        "evidence_payload_sha256": receipt.evidence_payload_sha256,
        "source_phase0_reference_sha256": receipt.source_phase0_reference_sha256,
        "canonical_coordinates_sha256": receipt.canonical_coordinates_sha256,
        "canonicalizer_artifact_sha256": receipt.canonicalizer_artifact_sha256,
        "annotation_manifest_sha256": receipt.annotation_manifest_sha256,
        "external_authority_verified": False,
    }
    if receipt.receipt_sha256 != _canonical_sha256(payload):
        raise SourceRelativeActionTargetError("camera receipt digest differs")


def _validate_annotations_v2(
    annotations: SourceRelativeActionAnnotationsV2,
    *,
    batch: int,
    device: Any,
) -> None:
    torch = _torch()
    if not isinstance(annotations, SourceRelativeActionAnnotationsV2):
        raise SourceRelativeActionTargetError("action annotations type differs")
    bool_shapes = {
        "actor_slot_valid": (batch, ACTOR_SLOT_COUNT),
        "object_slot_valid": (batch, OBJECT_SLOT_COUNT),
        "role_assignment_unique": (batch,),
        "ownership_unambiguous": (batch,),
        "source_actor_presence": (batch, PHASE_COUNT, ACTOR_SLOT_COUNT),
        "target_actor_presence": (batch, PHASE_COUNT, ACTOR_SLOT_COUNT),
        "source_object_presence": (batch, PHASE_COUNT, OBJECT_SLOT_COUNT),
        "target_object_presence": (batch, PHASE_COUNT, OBJECT_SLOT_COUNT),
        "source_presence_valid": (batch, PHASE_COUNT, ENTITY_SLOT_COUNT),
        "target_presence_valid": (batch, PHASE_COUNT, ENTITY_SLOT_COUNT),
        "contact": (batch, PHASE_COUNT, ACTOR_SLOT_COUNT, OBJECT_SLOT_COUNT),
        "contact_valid": (
            batch,
            PHASE_COUNT,
            ACTOR_SLOT_COUNT,
            OBJECT_SLOT_COUNT,
        ),
        "ownership_unambiguous": (batch,),
        "ownership_valid": (batch, PHASE_COUNT, OBJECT_SLOT_COUNT),
        "phase_channels": (batch, PHASE_COUNT, len(PHASE_CHANNEL_NAMES)),
        "phase_valid": (batch, PHASE_COUNT),
    }
    for name, shape in bool_shapes.items():
        tensor = _require_tensor(
            getattr(annotations, name),
            label="annotations.%s" % name,
            shape=shape,
            dtype=torch.bool,
        )
        if tensor.device != device:
            raise SourceRelativeActionTargetError(
                "annotations.%s device differs from camera evidence" % name
            )
    for name in ("observed_actor_count", "observed_object_count"):
        tensor = _require_tensor(
            getattr(annotations, name),
            label="annotations.%s" % name,
            shape=(batch,),
            dtype=torch.int64,
        )
        if tensor.device != device:
            raise SourceRelativeActionTargetError(
                "annotations.%s device differs from camera evidence" % name
            )
    ownership = _require_tensor(
        annotations.ownership,
        label="annotations.ownership",
        shape=(batch, PHASE_COUNT, OBJECT_SLOT_COUNT),
        dtype=torch.int64,
    )
    if ownership.device != device:
        raise SourceRelativeActionTargetError(
            "annotations.ownership device differs from camera evidence"
        )


def _annotations_payload_sha256(
    annotations: SourceRelativeActionAnnotationsV2,
    *,
    batch: int,
    device: Any,
) -> str:
    _validate_annotations_v2(annotations, batch=batch, device=device)
    return _canonical_sha256(
        {
            field.name: _tensor_sha256(
                getattr(annotations, field.name),
                label="annotations.%s" % field.name,
            )
            for field in fields(SourceRelativeActionAnnotationsV2)
        }
    )


def _phase_summary(
    phase_channels: Any, phase_valid: Any
) -> Tuple[Any, Any, Any, Any, Any]:
    """Return start/end/hold-start/duration plus per-sample structure validity."""

    torch = _torch()
    batch = phase_channels.shape[0]
    device = phase_channels.device
    start = torch.zeros(batch, dtype=torch.int64, device=device)
    end = torch.zeros(batch, dtype=torch.int64, device=device)
    hold_start = torch.zeros(batch, dtype=torch.int64, device=device)
    duration = torch.zeros(batch, dtype=torch.float32, device=device)
    structure_valid = torch.zeros(batch, dtype=torch.bool, device=device)
    for index in range(batch):
        if not bool(phase_valid[index].all().item()):
            continue
        channels = phase_channels[index]
        if bool((channels.sum(dim=-1) > 1).any().item()):
            continue
        onset_indices = torch.nonzero(
            channels[:, ONSET_CHANNEL], as_tuple=False
        ).reshape(-1)
        terminal_indices = torch.nonzero(
            channels[:, TERMINAL_CHANNEL], as_tuple=False
        ).reshape(-1)
        if onset_indices.numel() != 1 or terminal_indices.numel() != 1:
            continue
        onset = int(onset_indices[0].item())
        terminal = int(terminal_indices[0].item())
        if onset >= terminal or terminal >= PHASE_COUNT - 1:
            continue
        expected_transition = torch.zeros(PHASE_COUNT, dtype=torch.bool, device=device)
        expected_transition[onset + 1 : terminal] = True
        expected_hold = torch.zeros(PHASE_COUNT, dtype=torch.bool, device=device)
        expected_hold[terminal + 1 :] = True
        if not torch.equal(channels[:, TRANSITION_CHANNEL], expected_transition):
            continue
        if not torch.equal(channels[:, HOLD_CHANNEL], expected_hold):
            continue
        if bool(channels[:onset].any().item()):
            continue
        start[index] = onset
        end[index] = terminal
        hold_start[index] = terminal + 1
        duration[index] = float(terminal - onset)
        structure_valid[index] = True
    return start, end, hold_start, duration, structure_valid


def _masked_amplitude(values: Any, valid: Any) -> Tuple[Any, Any]:
    torch = _torch()
    magnitudes = torch.linalg.vector_norm(values, dim=-1)
    masked = torch.where(valid, magnitudes, torch.zeros_like(magnitudes))
    return masked.amax(dim=1), valid.any(dim=1)


def _masked_speed(values: Any, valid: Any) -> Tuple[Any, Any, Any]:
    torch = _torch()
    interval_valid = valid[:, 1:] & valid[:, :-1]
    speed = torch.linalg.vector_norm(values[:, 1:] - values[:, :-1], dim=-1)
    speed = torch.where(interval_valid, speed, torch.zeros_like(speed))
    count = interval_valid.sum(dim=1)
    speed_valid = count > 0
    mean = speed.sum(dim=1) / count.clamp_min(1).to(speed.dtype)
    peak = speed.amax(dim=1)
    mean = torch.where(speed_valid, mean, torch.zeros_like(mean))
    peak = torch.where(speed_valid, peak, torch.zeros_like(peak))
    return mean, peak, speed_valid


def _zero_unless(value: Any, valid: Any) -> Any:
    torch = _torch()
    expanded = valid
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    return torch.where(expanded, value, torch.zeros_like(value))


def _target_payload_sha256(target: SourceRelativeActionTargetV2) -> str:
    validate_source_relative_action_target_v2(target)
    return _canonical_sha256(
        {
            name: _tensor_sha256(getattr(target, name), label="target.%s" % name)
            for name in TARGET_FIELD_NAMES
        }
    )


def build_local_source_relative_action_target_draft_v2(
    camera_bundle: SourceRelativeCameraBundleV2,
    annotations: SourceRelativeActionAnnotationsV2,
    *,
    annotation_artifact_sha256: str,
    split_manifest_sha256: str,
) -> LocalSourceRelativeActionTargetDraftV2:
    """Construct a local schema draft, never a point-teacher capability."""

    torch = _torch()
    validate_source_relative_camera_bundle_v2(camera_bundle)
    evidence = camera_bundle.evidence
    coordinates = camera_bundle.coordinates
    batch = len(evidence.sample_ids)
    device = evidence.source_actor_xy.device
    _validate_annotations_v2(annotations, batch=batch, device=device)

    with torch.no_grad():
        actor_roles = torch.arange(
            ACTOR_SLOT_COUNT, dtype=torch.int64, device=device
        ).reshape(1, ACTOR_SLOT_COUNT).expand(batch, -1).clone()
        object_roles = torch.arange(
            OBJECT_SLOT_COUNT, dtype=torch.int64, device=device
        ).reshape(1, OBJECT_SLOT_COUNT).expand(batch, -1).clone()

        source_presence = torch.cat(
            (annotations.source_actor_presence, annotations.source_object_presence),
            dim=-1,
        )
        target_presence = torch.cat(
            (annotations.target_actor_presence, annotations.target_object_presence),
            dim=-1,
        )
        initial_source_presence = source_presence[:, 0]
        initial_source_presence_valid = annotations.source_presence_valid[:, 0]
        previous_presence = torch.cat(
            (initial_source_presence[:, None], target_presence[:, :-1]), dim=1
        )
        previous_valid = torch.cat(
            (
                initial_source_presence_valid[:, None],
                annotations.target_presence_valid[:, :-1],
            ),
            dim=1,
        )
        lifecycle_valid = previous_valid & annotations.target_presence_valid
        entity_create = (~previous_presence) & target_presence & lifecycle_valid
        entity_delete = previous_presence & (~target_presence) & lifecycle_valid

        source_actor_presence_valid = annotations.source_presence_valid[
            ..., :ACTOR_SLOT_COUNT
        ]
        target_actor_presence_valid = annotations.target_presence_valid[
            ..., :ACTOR_SLOT_COUNT
        ]
        source_object_presence_valid = annotations.source_presence_valid[
            ..., ACTOR_SLOT_COUNT:
        ]
        target_object_presence_valid = annotations.target_presence_valid[
            ..., ACTOR_SLOT_COUNT:
        ]

        actor_both = (
            coordinates.source_actor_track_valid
            & coordinates.target_actor_track_valid
            & annotations.source_actor_presence
            & annotations.target_actor_presence
            & source_actor_presence_valid
            & target_actor_presence_valid
        )
        actor_delta_valid = actor_both
        actor_delta = torch.where(
            actor_delta_valid[..., None],
            coordinates.target_actor_xy - coordinates.source_actor_xy,
            torch.zeros_like(coordinates.source_actor_xy),
        )
        actor_target_position_valid = (
            coordinates.target_actor_track_valid
            & annotations.target_actor_presence
            & target_actor_presence_valid
        )
        actor_target_position = torch.where(
            actor_target_position_valid[..., None],
            coordinates.target_actor_xy,
            torch.zeros_like(coordinates.target_actor_xy),
        )

        object_both = (
            coordinates.source_object_track_valid
            & coordinates.target_object_track_valid
            & annotations.source_object_presence
            & annotations.target_object_presence
            & source_object_presence_valid
            & target_object_presence_valid
        )
        object_delta_valid = object_both
        object_delta = torch.where(
            object_delta_valid[..., None],
            coordinates.target_object_xy - coordinates.source_object_xy,
            torch.zeros_like(coordinates.source_object_xy),
        )
        object_target_position_valid = (
            coordinates.target_object_track_valid
            & annotations.target_object_presence
            & target_object_presence_valid
        )
        object_target_position = torch.where(
            object_target_position_valid[..., None],
            coordinates.target_object_xy,
            torch.zeros_like(coordinates.target_object_xy),
        )
        actor_delta = _zero_unless(actor_delta, actor_delta_valid)
        object_delta = _zero_unless(object_delta, object_delta_valid)
        relative_delta = (
            object_delta[:, :, None, :, :] - actor_delta[:, :, :, None, :]
        )
        relative_delta_valid = (
            actor_delta_valid[:, :, :, None] & object_delta_valid[:, :, None, :]
        )
        relative_delta = _zero_unless(relative_delta, relative_delta_valid)

        actor_amplitude, actor_amplitude_valid = _masked_amplitude(
            actor_delta, actor_delta_valid
        )
        object_amplitude, object_amplitude_valid = _masked_amplitude(
            object_delta, object_delta_valid
        )
        flat_relative = relative_delta.reshape(
            batch, PHASE_COUNT, ACTOR_SLOT_COUNT * OBJECT_SLOT_COUNT, 2
        )
        flat_relative_valid = relative_delta_valid.reshape(
            batch, PHASE_COUNT, ACTOR_SLOT_COUNT * OBJECT_SLOT_COUNT
        )
        relative_amplitude, relative_amplitude_valid = _masked_amplitude(
            flat_relative, flat_relative_valid
        )
        entity_delta = torch.cat((actor_delta, object_delta), dim=2)
        entity_delta_valid = torch.cat((actor_delta_valid, object_delta_valid), dim=2)
        mean_speed, peak_speed, speed_valid = _masked_speed(
            entity_delta, entity_delta_valid
        )

        start, end, hold_start, duration, phase_structure_valid = _phase_summary(
            annotations.phase_channels, annotations.phase_valid
        )
        safe_ownership = annotations.ownership.clamp(
            min=OWNER_NONE, max=OWNER_GOAL_CONTAINER
        )

        terminal_hold_inconsistent = torch.zeros(
            batch, dtype=torch.bool, device=device
        )
        terminal_hold_evidence_missing = torch.zeros(
            batch, dtype=torch.bool, device=device
        )
        phase_action_invalid = torch.zeros(
            batch, dtype=torch.bool, device=device
        )
        entity_delta_for_hold = torch.cat((actor_delta, object_delta), dim=2)
        entity_delta_valid_for_hold = torch.cat(
            (actor_delta_valid, object_delta_valid), dim=2
        )
        entity_target_position = torch.cat(
            (actor_target_position, object_target_position), dim=2
        )
        entity_target_position_valid = torch.cat(
            (actor_target_position_valid, object_target_position_valid), dim=2
        )
        known_absent_state = (
            (~target_presence) & annotations.target_presence_valid
        )
        entity_state_valid = (
            entity_delta_valid_for_hold
            | entity_target_position_valid
            | known_absent_state
        )
        entity_state_value = torch.where(
            entity_delta_valid_for_hold[..., None],
            entity_delta_for_hold,
            torch.where(
                entity_target_position_valid[..., None],
                entity_target_position,
                torch.zeros_like(entity_delta_for_hold),
            ),
        )
        for sample_index in range(batch):
            if not bool(phase_structure_valid[sample_index].item()):
                continue
            terminal_index = int(end[sample_index].item())
            hold_index = int(hold_start[sample_index].item())
            terminal_presence = target_presence[sample_index, terminal_index]
            if not torch.equal(
                target_presence[sample_index, hold_index:],
                terminal_presence[None].expand(PHASE_COUNT - hold_index, -1),
            ):
                terminal_hold_inconsistent[sample_index] = True
            terminal_contact = annotations.contact[sample_index, terminal_index]
            terminal_contact_valid = annotations.contact_valid[
                sample_index, terminal_index
            ]
            if not torch.equal(
                annotations.contact_valid[sample_index, hold_index:],
                terminal_contact_valid[None].expand(
                    PHASE_COUNT - hold_index, -1, -1
                ),
            ) or not torch.equal(
                annotations.contact[sample_index, hold_index:]
                & annotations.contact_valid[sample_index, hold_index:],
                (terminal_contact & terminal_contact_valid)[None].expand(
                    PHASE_COUNT - hold_index, -1, -1
                ),
            ):
                terminal_hold_inconsistent[sample_index] = True
            terminal_ownership = safe_ownership[sample_index, terminal_index]
            terminal_ownership_valid = annotations.ownership_valid[
                sample_index, terminal_index
            ]
            hold_ownership = annotations.ownership[
                sample_index, hold_index:
            ].clamp(min=OWNER_NONE, max=OWNER_GOAL_CONTAINER)
            if not torch.equal(
                annotations.ownership_valid[sample_index, hold_index:],
                terminal_ownership_valid[None].expand(
                    PHASE_COUNT - hold_index, -1
                ),
            ) or not torch.equal(
                torch.where(
                    annotations.ownership_valid[sample_index, hold_index:],
                    hold_ownership,
                    torch.zeros_like(hold_ownership),
                ),
                torch.where(
                    terminal_ownership_valid,
                    terminal_ownership,
                    torch.zeros_like(terminal_ownership),
                )[None].expand(PHASE_COUNT - hold_index, -1),
            ):
                terminal_hold_inconsistent[sample_index] = True
            hold_values = entity_state_value[
                sample_index, terminal_index:
            ]
            hold_valid = entity_state_valid[
                sample_index, terminal_index:
            ]
            interval_valid = hold_valid[1:] & hold_valid[:-1]
            hold_speed = torch.linalg.vector_norm(
                hold_values[1:] - hold_values[:-1], dim=-1
            )
            if bool(
                (
                    interval_valid
                    & (hold_speed > TERMINAL_HOLD_DELTA_TOLERANCE)
                ).any().item()
            ):
                terminal_hold_inconsistent[sample_index] = True

            active_actor = annotations.actor_slot_valid[sample_index]
            active_object = annotations.object_slot_valid[sample_index]
            active_entity = torch.cat((active_actor, active_object), dim=0)
            active_pair = active_actor[:, None] & active_object[None, :]
            terminal_delta_valid = entity_state_valid[sample_index, terminal_index]
            full_hold_state_valid = entity_state_valid[
                sample_index, terminal_index:
            ]
            full_hold_actor_present = annotations.target_actor_presence[
                sample_index, terminal_index:
            ]
            full_hold_object_present = annotations.target_object_presence[
                sample_index, terminal_index:
            ]
            full_hold_pairs = (
                active_pair[None, :, :]
                & full_hold_actor_present[:, :, None]
                & full_hold_object_present[:, None, :]
            )
            if bool(
                (
                    active_entity[None, :]
                    & (~full_hold_state_valid)
                ).any().item()
            ):
                terminal_hold_evidence_missing[sample_index] = True
            if bool(
                (
                    full_hold_pairs
                    & (~annotations.contact_valid[sample_index, terminal_index:])
                ).any().item()
            ):
                terminal_hold_evidence_missing[sample_index] = True
            if bool(
                (
                    active_object[None, :]
                    & full_hold_object_present
                    & (~annotations.ownership_valid[sample_index, terminal_index:])
                ).any().item()
            ):
                terminal_hold_evidence_missing[sample_index] = True

            baseline_index = max(int(start[sample_index].item()) - 1, 0)
            baseline_delta_valid = entity_state_valid[
                sample_index, : baseline_index + 1
            ]
            baseline_delta = entity_delta_for_hold[
                sample_index, : baseline_index + 1
            ]
            if bool(
                (
                    active_entity[None, :]
                    & (~baseline_delta_valid)
                ).any().item()
            ):
                phase_action_invalid[sample_index] = True
            baseline_norm = torch.linalg.vector_norm(baseline_delta, dim=-1)
            if bool(
                (
                    active_entity[None, :]
                    & baseline_delta_valid
                    & (baseline_norm > PRE_ONSET_DELTA_TOLERANCE)
                ).any().item()
            ):
                phase_action_invalid[sample_index] = True
            pre_source_presence = source_presence[
                sample_index, : baseline_index + 1
            ]
            pre_target_presence = target_presence[
                sample_index, : baseline_index + 1
            ]
            pre_source_presence_valid = annotations.source_presence_valid[
                sample_index, : baseline_index + 1
            ]
            pre_target_presence_valid = annotations.target_presence_valid[
                sample_index, : baseline_index + 1
            ]
            if bool(
                (
                    active_entity[None, :]
                    & (
                        (~pre_source_presence_valid)
                        | (~pre_target_presence_valid)
                        | (pre_source_presence != pre_target_presence)
                    )
                ).any().item()
            ):
                phase_action_invalid[sample_index] = True
            if bool(
                (
                    active_entity[None, :]
                    & pre_target_presence
                    & (~entity_delta_valid_for_hold[
                        sample_index, : baseline_index + 1
                    ])
                ).any().item()
            ):
                phase_action_invalid[sample_index] = True
            if bool(
                entity_create[sample_index, : baseline_index + 1].any().item()
            ) or bool(
                entity_delete[sample_index, : baseline_index + 1].any().item()
            ):
                phase_action_invalid[sample_index] = True
            if bool(
                (
                    (
                        active_pair
                        & annotations.target_actor_presence[
                            sample_index, baseline_index, :, None
                        ]
                        & annotations.target_object_presence[
                            sample_index, baseline_index, None, :
                        ]
                    )
                    & (~annotations.contact_valid[sample_index, baseline_index])
                ).any().item()
            ) or bool(
                (
                    (
                        active_object
                        & annotations.target_object_presence[
                            sample_index, baseline_index
                        ]
                    )
                    & (~annotations.ownership_valid[sample_index, baseline_index])
                ).any().item()
            ):
                phase_action_invalid[sample_index] = True
            pre_event_presence = target_presence[
                sample_index, : baseline_index + 1
            ]
            if not torch.equal(
                pre_event_presence,
                pre_event_presence[-1:].expand_as(pre_event_presence),
            ):
                phase_action_invalid[sample_index] = True
            pre_event_contact_valid = annotations.contact_valid[
                sample_index, : baseline_index + 1
            ]
            pre_event_contact = annotations.contact[
                sample_index, : baseline_index + 1
            ] & pre_event_contact_valid
            if not torch.equal(
                pre_event_contact_valid,
                pre_event_contact_valid[-1:].expand_as(pre_event_contact_valid),
            ) or not torch.equal(
                pre_event_contact,
                pre_event_contact[-1:].expand_as(pre_event_contact),
            ):
                phase_action_invalid[sample_index] = True
            pre_event_owner_valid = annotations.ownership_valid[
                sample_index, : baseline_index + 1
            ]
            pre_event_owner = torch.where(
                pre_event_owner_valid,
                safe_ownership[sample_index, : baseline_index + 1],
                torch.zeros_like(
                    safe_ownership[sample_index, : baseline_index + 1]
                ),
            )
            if not torch.equal(
                pre_event_owner_valid,
                pre_event_owner_valid[-1:].expand_as(pre_event_owner_valid),
            ) or not torch.equal(
                pre_event_owner,
                pre_event_owner[-1:].expand_as(pre_event_owner),
            ):
                phase_action_invalid[sample_index] = True
            baseline_state = entity_delta_for_hold[sample_index, baseline_index]
            terminal_state = entity_delta_for_hold[sample_index, terminal_index]
            kinematic_change = bool(
                (
                    active_entity
                    & entity_delta_valid_for_hold[sample_index, terminal_index]
                    & entity_delta_valid_for_hold[sample_index, baseline_index]
                    & (
                        torch.linalg.vector_norm(
                            terminal_state - baseline_state, dim=-1
                        )
                        >= MIN_ACTION_STATE_CHANGE
                    )
                ).any().item()
            )
            contact_change = bool(
                (
                    active_pair
                    & annotations.contact_valid[sample_index, baseline_index]
                    & annotations.contact_valid[sample_index, terminal_index]
                    & (
                        annotations.contact[sample_index, baseline_index]
                        != annotations.contact[sample_index, terminal_index]
                    )
                ).any().item()
            )
            ownership_change = bool(
                (
                    active_object
                    & annotations.ownership_valid[sample_index, baseline_index]
                    & annotations.ownership_valid[sample_index, terminal_index]
                    & (
                        safe_ownership[sample_index, baseline_index]
                        != safe_ownership[sample_index, terminal_index]
                    )
                ).any().item()
            )
            presence_change = bool(
                (
                    active_entity
                    & (
                        target_presence[sample_index, baseline_index]
                        != target_presence[sample_index, terminal_index]
                    )
                ).any().item()
            )
            transition_start = int(start[sample_index].item())
            left_delta = entity_delta_for_hold[
                sample_index, transition_start:terminal_index
            ]
            right_delta = entity_delta_for_hold[
                sample_index, transition_start + 1 : terminal_index + 1
            ]
            left_valid = entity_delta_valid_for_hold[
                sample_index, transition_start:terminal_index
            ]
            right_valid = entity_delta_valid_for_hold[
                sample_index, transition_start + 1 : terminal_index + 1
            ]
            transition_kinematic_change = bool(
                (
                    active_entity[None, :]
                    & left_valid
                    & right_valid
                    & (
                        torch.linalg.vector_norm(
                            right_delta - left_delta, dim=-1
                        )
                        >= MIN_ACTION_STATE_CHANGE
                    )
                ).any().item()
            )
            transition_presence_change = bool(
                (
                    active_entity[None, :]
                    & (
                        target_presence[
                            sample_index,
                            transition_start:terminal_index,
                        ]
                        != target_presence[
                            sample_index,
                            transition_start + 1 : terminal_index + 1,
                        ]
                    )
                ).any().item()
            )
            left_contact_valid = annotations.contact_valid[
                sample_index, transition_start:terminal_index
            ]
            right_contact_valid = annotations.contact_valid[
                sample_index, transition_start + 1 : terminal_index + 1
            ]
            transition_contact_change = bool(
                (
                    active_pair[None, :, :]
                    & left_contact_valid
                    & right_contact_valid
                    & (
                        annotations.contact[
                            sample_index, transition_start:terminal_index
                        ]
                        != annotations.contact[
                            sample_index,
                            transition_start + 1 : terminal_index + 1,
                        ]
                    )
                ).any().item()
            )
            left_owner_valid = annotations.ownership_valid[
                sample_index, transition_start:terminal_index
            ]
            right_owner_valid = annotations.ownership_valid[
                sample_index, transition_start + 1 : terminal_index + 1
            ]
            transition_owner_change = bool(
                (
                    active_object[None, :]
                    & left_owner_valid
                    & right_owner_valid
                    & (
                        safe_ownership[
                            sample_index, transition_start:terminal_index
                        ]
                        != safe_ownership[
                            sample_index,
                            transition_start + 1 : terminal_index + 1,
                        ]
                    )
                ).any().item()
            )
            if not (
                kinematic_change
                or contact_change
                or ownership_change
                or presence_change
            ):
                phase_action_invalid[sample_index] = True
            if not (
                transition_kinematic_change
                or transition_presence_change
                or transition_contact_change
                or transition_owner_change
            ):
                phase_action_invalid[sample_index] = True

        reasons = torch.zeros(
            (batch, len(ABSTAIN_REASON_NAMES)), dtype=torch.bool, device=device
        )
        reasons[:, 0] = ~coordinates.camera_reference_valid
        reasons[:, 1] = (
            (annotations.observed_actor_count > ACTOR_SLOT_COUNT)
            | (annotations.observed_object_count > OBJECT_SLOT_COUNT)
            | (annotations.observed_actor_count < 1)
            | (annotations.observed_object_count < 0)
        )
        actor_expected = annotations.observed_actor_count.clamp(
            min=0, max=ACTOR_SLOT_COUNT
        )
        object_expected = annotations.observed_object_count.clamp(
            min=0, max=OBJECT_SLOT_COUNT
        )
        slot_count_mismatch = (
            annotations.actor_slot_valid.sum(dim=1) != actor_expected
        ) | (annotations.object_slot_valid.sum(dim=1) != object_expected)
        inactive_actor_has_evidence = (
            (
                annotations.source_actor_presence.any(dim=1)
                | annotations.target_actor_presence.any(dim=1)
                | evidence.source_actor_track_valid.any(dim=1)
                | evidence.target_actor_track_valid.any(dim=1)
            )
            & (~annotations.actor_slot_valid)
        ).any(dim=1)
        inactive_object_has_evidence = (
            (
                annotations.source_object_presence.any(dim=1)
                | annotations.target_object_presence.any(dim=1)
                | evidence.source_object_track_valid.any(dim=1)
                | evidence.target_object_track_valid.any(dim=1)
            )
            & (~annotations.object_slot_valid)
        ).any(dim=1)
        reasons[:, 2] = (
            (~annotations.role_assignment_unique)
            | (~annotations.actor_slot_valid[:, 0])
            | slot_count_mismatch
            | inactive_actor_has_evidence
            | inactive_object_has_evidence
        )

        ownership_in_range = (
            (annotations.ownership >= 0)
            & (annotations.ownership < len(OWNERSHIP_NAMES))
        )
        reasons[:, 3] = (
            (~annotations.ownership_unambiguous)
            | ((~ownership_in_range) & annotations.ownership_valid).any(dim=2).any(dim=1)
            | (
                annotations.ownership_valid[:, :, OBJECT_SLOT_COUNT - 1]
                & (
                    safe_ownership[:, :, OBJECT_SLOT_COUNT - 1]
                    == OWNER_GOAL_CONTAINER
                )
            ).any(dim=1)
        )
        primary_present = (
            annotations.source_presence_valid[:, 0, 0]
            & annotations.target_presence_valid[:, 0, 0]
            & annotations.source_actor_presence[:, 0, 0]
            & annotations.target_actor_presence[:, 0, 0]
        )
        reasons[:, 4] = ~primary_present
        reasons[:, 5] = ~phase_structure_valid
        reasons[:, 6] = actor_delta_valid[:, :, 0].sum(dim=1) < 2

        actor_present = annotations.target_actor_presence
        object_present = annotations.target_object_presence
        contact_presence_ok = (
            actor_present[:, :, :, None] & object_present[:, :, None, :]
        )
        bad_contact = (
            annotations.contact_valid & (~contact_presence_ok)
        ).any(dim=3).any(dim=2).any(dim=1)
        bad_owner_presence = annotations.ownership_valid & (~object_present)
        bad_owner_presence = bad_owner_presence | (
            annotations.ownership_valid
            & (safe_ownership == OWNER_PRIMARY_ACTOR)
            & (~actor_present[:, :, 0, None])
        )
        bad_owner_presence = bad_owner_presence | (
            annotations.ownership_valid
            & (safe_ownership == OWNER_CO_ACTOR)
            & (~actor_present[:, :, 1, None])
        )
        goal_container_present = object_present[:, :, OBJECT_SLOT_COUNT - 1, None]
        bad_owner_presence = bad_owner_presence | (
            annotations.ownership_valid
            & (safe_ownership == OWNER_GOAL_CONTAINER)
            & (~goal_container_present)
        )
        missing_presence_coverage = (
            annotations.actor_slot_valid
            & (~annotations.target_presence_valid[..., :ACTOR_SLOT_COUNT].all(dim=1))
        ).any(dim=1) | (
            annotations.object_slot_valid
            & (~annotations.target_presence_valid[..., ACTOR_SLOT_COUNT:].all(dim=1))
        ).any(dim=1)
        reasons[:, 7] = (
            bad_contact
            | bad_owner_presence.any(dim=2).any(dim=1)
            | missing_presence_coverage
            | terminal_hold_inconsistent
        )
        reasons[:, 8] = ~coordinates.finite_valid_evidence
        reasons[:, 9] = terminal_hold_evidence_missing
        reasons[:, 10] = phase_action_invalid

        abstain = reasons.any(dim=1)
        sample_valid = ~abstain
        sample_mask = sample_valid[:, None]
        phase_sample_mask = sample_valid[:, None, None]

        actor_role_valid = annotations.actor_slot_valid & sample_mask
        object_role_valid = annotations.object_slot_valid & sample_mask
        actor_presence_valid = (
            annotations.target_presence_valid[..., :ACTOR_SLOT_COUNT]
            & actor_role_valid[:, None, :]
            & sample_valid[:, None, None]
        )
        object_presence_valid = (
            annotations.target_presence_valid[..., ACTOR_SLOT_COUNT:]
            & object_role_valid[:, None, :]
            & sample_valid[:, None, None]
        )
        actor_presence_value = annotations.target_actor_presence & actor_presence_valid
        object_presence_value = annotations.target_object_presence & object_presence_valid
        entity_role_valid = torch.cat((actor_role_valid, object_role_valid), dim=1)
        source_presence_valid_value = (
            annotations.source_presence_valid
            & entity_role_valid[:, None, :]
            & sample_valid[:, None, None]
        )
        source_presence_value = source_presence & source_presence_valid_value
        actor_delta_valid = actor_delta_valid & actor_role_valid[:, None, :]
        actor_target_position_valid = (
            actor_target_position_valid & actor_role_valid[:, None, :]
        )
        object_delta_valid = object_delta_valid & object_role_valid[:, None, :]
        object_target_position_valid = (
            object_target_position_valid & object_role_valid[:, None, :]
        )
        relative_delta_valid = (
            relative_delta_valid
            & actor_role_valid[:, None, :, None]
            & object_role_valid[:, None, None, :]
        )
        actor_delta = _zero_unless(actor_delta, actor_delta_valid)
        actor_target_position = _zero_unless(
            actor_target_position, actor_target_position_valid
        )
        object_delta = _zero_unless(object_delta, object_delta_valid)
        object_target_position = _zero_unless(
            object_target_position, object_target_position_valid
        )
        relative_delta = _zero_unless(relative_delta, relative_delta_valid)

        initial_source_presence_valid = (
            initial_source_presence_valid
            & entity_role_valid
        )
        initial_source_presence = initial_source_presence & initial_source_presence_valid
        lifecycle_valid = (
            lifecycle_valid
            & torch.cat((actor_role_valid, object_role_valid), dim=1)[:, None, :]
        )
        entity_create = entity_create & lifecycle_valid
        entity_delete = entity_delete & lifecycle_valid
        contact_valid = (
            annotations.contact_valid
            & contact_presence_ok
            & actor_role_valid[:, None, :, None]
            & object_role_valid[:, None, None, :]
            & sample_valid[:, None, None, None]
        )
        contact = annotations.contact & contact_valid
        ownership_valid = (
            annotations.ownership_valid
            & object_role_valid[:, None, :]
            & sample_valid[:, None, None]
        )
        ownership = torch.where(
            ownership_valid, safe_ownership, torch.zeros_like(safe_ownership)
        )
        phase_valid = annotations.phase_valid & sample_valid[:, None]
        phase_channels = annotations.phase_channels & phase_valid[..., None]
        phase_summary_valid = phase_structure_valid & sample_valid
        start = torch.where(phase_summary_valid, start, torch.zeros_like(start))
        end = torch.where(phase_summary_valid, end, torch.zeros_like(end))
        hold_start = torch.where(
            phase_summary_valid, hold_start, torch.zeros_like(hold_start)
        )
        duration = torch.where(
            phase_summary_valid, duration, torch.zeros_like(duration)
        )

        actor_amplitude_valid = actor_amplitude_valid & actor_role_valid
        object_amplitude_valid = object_amplitude_valid & object_role_valid
        relative_amplitude_valid = (
            relative_amplitude_valid.reshape(
                batch, ACTOR_SLOT_COUNT, OBJECT_SLOT_COUNT
            )
            & actor_role_valid[:, :, None]
            & object_role_valid[:, None, :]
        ).reshape(batch, ACTOR_SLOT_COUNT * OBJECT_SLOT_COUNT)
        speed_valid = (
            speed_valid
            & torch.cat((actor_role_valid, object_role_valid), dim=1)
        )
        actor_amplitude = _zero_unless(actor_amplitude, actor_amplitude_valid)
        object_amplitude = _zero_unless(object_amplitude, object_amplitude_valid)
        relative_amplitude = _zero_unless(
            relative_amplitude, relative_amplitude_valid
        )
        mean_speed = _zero_unless(mean_speed, speed_valid)
        peak_speed = _zero_unless(peak_speed, speed_valid)

        target = SourceRelativeActionTargetV2(
            actor_roles=_owned(actor_roles),
            actor_role_valid=_owned(actor_role_valid),
            object_roles=_owned(object_roles),
            object_role_valid=_owned(object_role_valid),
            actor_delta=_owned(actor_delta),
            actor_delta_valid=_owned(actor_delta_valid),
            actor_target_position=_owned(actor_target_position),
            actor_target_position_valid=_owned(actor_target_position_valid),
            object_delta=_owned(object_delta),
            object_delta_valid=_owned(object_delta_valid),
            object_target_position=_owned(object_target_position),
            object_target_position_valid=_owned(object_target_position_valid),
            relative_delta=_owned(relative_delta),
            relative_delta_valid=_owned(relative_delta_valid),
            actor_presence=_owned(actor_presence_value),
            actor_presence_valid=_owned(actor_presence_valid),
            object_presence=_owned(object_presence_value),
            object_presence_valid=_owned(object_presence_valid),
            source_presence=_owned(source_presence_value),
            source_presence_valid=_owned(source_presence_valid_value),
            initial_source_presence=_owned(initial_source_presence),
            initial_source_presence_valid=_owned(initial_source_presence_valid),
            entity_create=_owned(entity_create),
            entity_delete=_owned(entity_delete),
            lifecycle_valid=_owned(lifecycle_valid),
            contact=_owned(contact),
            contact_valid=_owned(contact_valid),
            ownership=_owned(ownership),
            ownership_valid=_owned(ownership_valid),
            phase_channels=_owned(phase_channels),
            phase_valid=_owned(phase_valid),
            action_start_phase=_owned(start),
            action_end_phase=_owned(end),
            terminal_hold_start_phase=_owned(hold_start),
            duration_phases=_owned(duration),
            phase_summary_valid=_owned(phase_summary_valid),
            actor_amplitude=_owned(actor_amplitude),
            actor_amplitude_valid=_owned(actor_amplitude_valid),
            object_amplitude=_owned(object_amplitude),
            object_amplitude_valid=_owned(object_amplitude_valid),
            relative_amplitude=_owned(relative_amplitude),
            relative_amplitude_valid=_owned(relative_amplitude_valid),
            mean_speed=_owned(mean_speed),
            peak_speed=_owned(peak_speed),
            speed_valid=_owned(speed_valid),
            sample_valid=_owned(sample_valid),
            abstain=_owned(abstain),
            abstain_reasons=_owned(reasons),
        )

    validate_source_relative_action_target_v2(target)
    annotation_payload_sha256 = _annotations_payload_sha256(
        annotations, batch=batch, device=device
    )
    receipt_payload = {
        "schema_version": DRAFT_RECEIPT_SCHEMA_VERSION,
        "sample_ids": list(evidence.sample_ids),
        "source_media_sha256": list(evidence.source_media_sha256),
        "target_media_sha256": list(evidence.target_media_sha256),
        "target_payload_sha256": _target_payload_sha256(target),
        "camera_receipt_sha256": camera_bundle.receipt.receipt_sha256,
        "annotation_payload_sha256": annotation_payload_sha256,
        "annotation_artifact_sha256": _require_sha256(
            annotation_artifact_sha256, label="annotation_artifact_sha256"
        ),
        "split_manifest_sha256": _require_sha256(
            split_manifest_sha256, label="split_manifest_sha256"
        ),
        "external_authority_verified": False,
    }
    receipt = LocalSourceRelativeActionTargetDraftReceiptV2(
        schema_version=DRAFT_RECEIPT_SCHEMA_VERSION,
        sample_ids=evidence.sample_ids,
        source_media_sha256=evidence.source_media_sha256,
        target_media_sha256=evidence.target_media_sha256,
        target_payload_sha256=receipt_payload["target_payload_sha256"],
        camera_receipt_sha256=receipt_payload["camera_receipt_sha256"],
        annotation_payload_sha256=receipt_payload["annotation_payload_sha256"],
        annotation_artifact_sha256=receipt_payload["annotation_artifact_sha256"],
        split_manifest_sha256=receipt_payload["split_manifest_sha256"],
        external_authority_verified=False,
        receipt_sha256=_canonical_sha256(receipt_payload),
    )
    return LocalSourceRelativeActionTargetDraftV2(target=target, receipt=receipt)


def validate_source_relative_action_target_v2(
    target: SourceRelativeActionTargetV2,
) -> None:
    torch = _torch()
    if not isinstance(target, SourceRelativeActionTargetV2):
        raise SourceRelativeActionTargetError("source-relative target type differs")
    if not isinstance(target.sample_valid, torch.Tensor) or target.sample_valid.ndim != 1:
        raise SourceRelativeActionTargetError("target.sample_valid must be rank one")
    batch = target.sample_valid.shape[0]
    phase_shapes = {
        "actor_delta": ((batch, PHASE_COUNT, ACTOR_SLOT_COUNT, 2), torch.float32),
        "actor_delta_valid": ((batch, PHASE_COUNT, ACTOR_SLOT_COUNT), torch.bool),
        "actor_target_position": (
            (batch, PHASE_COUNT, ACTOR_SLOT_COUNT, 2),
            torch.float32,
        ),
        "actor_target_position_valid": (
            (batch, PHASE_COUNT, ACTOR_SLOT_COUNT),
            torch.bool,
        ),
        "object_delta": ((batch, PHASE_COUNT, OBJECT_SLOT_COUNT, 2), torch.float32),
        "object_delta_valid": ((batch, PHASE_COUNT, OBJECT_SLOT_COUNT), torch.bool),
        "object_target_position": (
            (batch, PHASE_COUNT, OBJECT_SLOT_COUNT, 2),
            torch.float32,
        ),
        "object_target_position_valid": (
            (batch, PHASE_COUNT, OBJECT_SLOT_COUNT),
            torch.bool,
        ),
        "relative_delta": (
            (batch, PHASE_COUNT, ACTOR_SLOT_COUNT, OBJECT_SLOT_COUNT, 2),
            torch.float32,
        ),
        "relative_delta_valid": (
            (batch, PHASE_COUNT, ACTOR_SLOT_COUNT, OBJECT_SLOT_COUNT),
            torch.bool,
        ),
        "actor_presence": ((batch, PHASE_COUNT, ACTOR_SLOT_COUNT), torch.bool),
        "actor_presence_valid": (
            (batch, PHASE_COUNT, ACTOR_SLOT_COUNT),
            torch.bool,
        ),
        "object_presence": ((batch, PHASE_COUNT, OBJECT_SLOT_COUNT), torch.bool),
        "object_presence_valid": (
            (batch, PHASE_COUNT, OBJECT_SLOT_COUNT),
            torch.bool,
        ),
        "source_presence": ((batch, PHASE_COUNT, ENTITY_SLOT_COUNT), torch.bool),
        "source_presence_valid": (
            (batch, PHASE_COUNT, ENTITY_SLOT_COUNT),
            torch.bool,
        ),
        "entity_create": ((batch, PHASE_COUNT, ENTITY_SLOT_COUNT), torch.bool),
        "entity_delete": ((batch, PHASE_COUNT, ENTITY_SLOT_COUNT), torch.bool),
        "lifecycle_valid": ((batch, PHASE_COUNT, ENTITY_SLOT_COUNT), torch.bool),
        "contact": (
            (batch, PHASE_COUNT, ACTOR_SLOT_COUNT, OBJECT_SLOT_COUNT),
            torch.bool,
        ),
        "contact_valid": (
            (batch, PHASE_COUNT, ACTOR_SLOT_COUNT, OBJECT_SLOT_COUNT),
            torch.bool,
        ),
        "ownership": ((batch, PHASE_COUNT, OBJECT_SLOT_COUNT), torch.int64),
        "ownership_valid": ((batch, PHASE_COUNT, OBJECT_SLOT_COUNT), torch.bool),
        "phase_channels": (
            (batch, PHASE_COUNT, len(PHASE_CHANNEL_NAMES)),
            torch.bool,
        ),
        "phase_valid": ((batch, PHASE_COUNT), torch.bool),
    }
    global_shapes = {
        "actor_roles": ((batch, ACTOR_SLOT_COUNT), torch.int64),
        "actor_role_valid": ((batch, ACTOR_SLOT_COUNT), torch.bool),
        "object_roles": ((batch, OBJECT_SLOT_COUNT), torch.int64),
        "object_role_valid": ((batch, OBJECT_SLOT_COUNT), torch.bool),
        "initial_source_presence": ((batch, ENTITY_SLOT_COUNT), torch.bool),
        "initial_source_presence_valid": ((batch, ENTITY_SLOT_COUNT), torch.bool),
        "action_start_phase": ((batch,), torch.int64),
        "action_end_phase": ((batch,), torch.int64),
        "terminal_hold_start_phase": ((batch,), torch.int64),
        "duration_phases": ((batch,), torch.float32),
        "phase_summary_valid": ((batch,), torch.bool),
        "actor_amplitude": ((batch, ACTOR_SLOT_COUNT), torch.float32),
        "actor_amplitude_valid": ((batch, ACTOR_SLOT_COUNT), torch.bool),
        "object_amplitude": ((batch, OBJECT_SLOT_COUNT), torch.float32),
        "object_amplitude_valid": ((batch, OBJECT_SLOT_COUNT), torch.bool),
        "relative_amplitude": (
            (batch, ACTOR_SLOT_COUNT * OBJECT_SLOT_COUNT),
            torch.float32,
        ),
        "relative_amplitude_valid": (
            (batch, ACTOR_SLOT_COUNT * OBJECT_SLOT_COUNT),
            torch.bool,
        ),
        "mean_speed": ((batch, ENTITY_SLOT_COUNT), torch.float32),
        "peak_speed": ((batch, ENTITY_SLOT_COUNT), torch.float32),
        "speed_valid": ((batch, ENTITY_SLOT_COUNT), torch.bool),
        "sample_valid": ((batch,), torch.bool),
        "abstain": ((batch,), torch.bool),
        "abstain_reasons": ((batch, len(ABSTAIN_REASON_NAMES)), torch.bool),
    }
    devices = set()
    for name, (shape, dtype) in {**phase_shapes, **global_shapes}.items():
        value = _require_tensor(
            getattr(target, name),
            label="target.%s" % name,
            shape=shape,
            dtype=dtype,
        )
        devices.add(value.device)
        if dtype == torch.float32 and not bool(torch.isfinite(value).all().item()):
            raise SourceRelativeActionTargetError("target.%s must be finite" % name)
    if len(devices) != 1:
        raise SourceRelativeActionTargetError("all target tensors must share one device")

    expected_actor_roles = torch.arange(
        ACTOR_SLOT_COUNT, dtype=torch.int64, device=target.actor_roles.device
    ).reshape(1, -1).expand(batch, -1)
    expected_object_roles = torch.arange(
        OBJECT_SLOT_COUNT, dtype=torch.int64, device=target.object_roles.device
    ).reshape(1, -1).expand(batch, -1)
    if not torch.equal(target.actor_roles, expected_actor_roles):
        raise SourceRelativeActionTargetError("actor role slots are not canonical")
    if not torch.equal(target.object_roles, expected_object_roles):
        raise SourceRelativeActionTargetError("object role slots are not canonical")
    if bool(((target.ownership < 0) | (target.ownership >= len(OWNERSHIP_NAMES))).any().item()):
        raise SourceRelativeActionTargetError("ownership code is outside the closed set")
    if not torch.equal(target.sample_valid, ~target.abstain):
        raise SourceRelativeActionTargetError("sample_valid and abstain disagree")
    if not torch.equal(target.abstain, target.abstain_reasons.any(dim=1)):
        raise SourceRelativeActionTargetError("abstain reasons do not bind abstention")

    invalid_sample = target.abstain
    point_valid_names = (
        "actor_role_valid",
        "object_role_valid",
        "actor_delta_valid",
        "actor_target_position_valid",
        "object_delta_valid",
        "object_target_position_valid",
        "relative_delta_valid",
        "actor_presence_valid",
        "object_presence_valid",
        "source_presence_valid",
        "initial_source_presence_valid",
        "lifecycle_valid",
        "contact_valid",
        "ownership_valid",
        "phase_valid",
        "phase_summary_valid",
        "actor_amplitude_valid",
        "object_amplitude_valid",
        "relative_amplitude_valid",
        "speed_valid",
    )
    for name in point_valid_names:
        valid = getattr(target, name)
        if bool(valid[invalid_sample].any().item()):
            raise SourceRelativeActionTargetError(
                "abstained samples retain valid supervision in %s" % name
            )

    value_valid_pairs = (
        ("actor_delta", "actor_delta_valid"),
        ("actor_target_position", "actor_target_position_valid"),
        ("object_delta", "object_delta_valid"),
        ("object_target_position", "object_target_position_valid"),
        ("relative_delta", "relative_delta_valid"),
        ("actor_presence", "actor_presence_valid"),
        ("object_presence", "object_presence_valid"),
        ("source_presence", "source_presence_valid"),
        ("initial_source_presence", "initial_source_presence_valid"),
        ("contact", "contact_valid"),
        ("ownership", "ownership_valid"),
        ("phase_channels", "phase_valid"),
        ("actor_amplitude", "actor_amplitude_valid"),
        ("object_amplitude", "object_amplitude_valid"),
        ("relative_amplitude", "relative_amplitude_valid"),
        ("mean_speed", "speed_valid"),
        ("peak_speed", "speed_valid"),
    )
    for value_name, valid_name in value_valid_pairs:
        value = getattr(target, value_name)
        valid = getattr(target, valid_name)
        expanded = valid
        while expanded.ndim < value.ndim:
            expanded = expanded.unsqueeze(-1)
        invalid_values = value[~expanded.expand_as(value)]
        if invalid_values.numel() and bool((invalid_values != 0).any().item()):
            raise SourceRelativeActionTargetError(
                "invalid %s entries must be zero" % value_name
            )
    for name in ("entity_create", "entity_delete"):
        value = getattr(target, name)
        if bool((value & (~target.lifecycle_valid)).any().item()):
            raise SourceRelativeActionTargetError(
                "%s is active outside lifecycle_valid" % name
            )

    target_presence = torch.cat(
        (target.actor_presence, target.object_presence), dim=-1
    )
    previous_presence = torch.cat(
        (target.initial_source_presence[:, None], target_presence[:, :-1]), dim=1
    )
    expected_create = (~previous_presence) & target_presence & target.lifecycle_valid
    expected_delete = previous_presence & (~target_presence) & target.lifecycle_valid
    target_presence_valid = torch.cat(
        (target.actor_presence_valid, target.object_presence_valid), dim=-1
    )
    previous_presence_valid = torch.cat(
        (
            target.initial_source_presence_valid[:, None],
            target_presence_valid[:, :-1],
        ),
        dim=1,
    )
    expected_lifecycle_valid = previous_presence_valid & target_presence_valid
    if not torch.equal(target.lifecycle_valid, expected_lifecycle_valid):
        raise SourceRelativeActionTargetError(
            "lifecycle validity differs from adjacent presence validity"
        )
    if not torch.equal(target.entity_create, expected_create):
        raise SourceRelativeActionTargetError("create bits disagree with presence transitions")
    if not torch.equal(target.entity_delete, expected_delete):
        raise SourceRelativeActionTargetError("delete bits disagree with presence transitions")

    expected_relative_valid = (
        target.actor_delta_valid[:, :, :, None]
        & target.object_delta_valid[:, :, None, :]
    )
    if not torch.equal(target.relative_delta_valid, expected_relative_valid):
        raise SourceRelativeActionTargetError("relative validity differs from its endpoints")
    expected_relative = (
        target.object_delta[:, :, None, :, :] - target.actor_delta[:, :, :, None, :]
    )
    expected_relative = _zero_unless(expected_relative, expected_relative_valid)
    if not torch.equal(target.relative_delta, expected_relative):
        raise SourceRelativeActionTargetError("relative delta differs from endpoint deltas")

    if bool(
        (
            target.actor_presence_valid
            & (~target.actor_role_valid[:, None, :])
        ).any().item()
    ) or bool(
        (
            target.actor_delta_valid
            & (~target.actor_role_valid[:, None, :])
        ).any().item()
    ) or bool(
        (
            target.actor_target_position_valid
            & (~target.actor_role_valid[:, None, :])
        ).any().item()
    ):
        raise SourceRelativeActionTargetError("actor validity escapes its role slot")
    if bool(
        (
            target.object_presence_valid
            & (~target.object_role_valid[:, None, :])
        ).any().item()
    ) or bool(
        (
            target.object_delta_valid
            & (~target.object_role_valid[:, None, :])
        ).any().item()
    ) or bool(
        (
            target.object_target_position_valid
            & (~target.object_role_valid[:, None, :])
        ).any().item()
    ):
        raise SourceRelativeActionTargetError("object validity escapes its role slot")
    entity_role_valid = torch.cat(
        (target.actor_role_valid, target.object_role_valid), dim=1
    )
    if bool(
        (
            target.source_presence_valid
            & (~entity_role_valid[:, None, :])
        ).any().item()
    ):
        raise SourceRelativeActionTargetError(
            "source presence validity escapes its role slot"
        )
    if not torch.equal(
        target.initial_source_presence, target.source_presence[:, 0]
    ) or not torch.equal(
        target.initial_source_presence_valid, target.source_presence_valid[:, 0]
    ):
        raise SourceRelativeActionTargetError(
            "initial source presence differs from phase-0 source presence"
        )

    source_actor_presence = target.source_presence[..., :ACTOR_SLOT_COUNT]
    source_actor_presence_valid = target.source_presence_valid[
        ..., :ACTOR_SLOT_COUNT
    ]
    source_object_presence = target.source_presence[..., ACTOR_SLOT_COUNT:]
    source_object_presence_valid = target.source_presence_valid[
        ..., ACTOR_SLOT_COUNT:
    ]
    actor_position_allowed = target.actor_presence_valid & target.actor_presence
    object_position_allowed = target.object_presence_valid & target.object_presence
    if bool(
        (
            target.actor_target_position_valid & (~actor_position_allowed)
        ).any().item()
    ) or bool(
        (
            target.object_target_position_valid & (~object_position_allowed)
        ).any().item()
    ):
        raise SourceRelativeActionTargetError(
            "target-position validity requires known target presence"
        )
    actor_delta_allowed = (
        source_actor_presence_valid
        & source_actor_presence
        & actor_position_allowed
        & target.actor_target_position_valid
    )
    object_delta_allowed = (
        source_object_presence_valid
        & source_object_presence
        & object_position_allowed
        & target.object_target_position_valid
    )
    if bool((target.actor_delta_valid & (~actor_delta_allowed)).any().item()) or bool(
        (target.object_delta_valid & (~object_delta_allowed)).any().item()
    ):
        raise SourceRelativeActionTargetError(
            "displacement validity requires both endpoints to be present"
        )

    contact_presence = (
        target.actor_presence[:, :, :, None]
        & target.object_presence[:, :, None, :]
    )
    if bool((target.contact_valid & (~contact_presence)).any().item()):
        raise SourceRelativeActionTargetError(
            "contact-valid cells require both participants to be present"
        )
    if bool((target.contact & target.contact_valid & (~contact_presence)).any().item()):
        raise SourceRelativeActionTargetError("contact is active without both participants")
    if bool((target.ownership_valid & (~target.object_presence)).any().item()):
        raise SourceRelativeActionTargetError("ownership is valid for an absent object")
    if bool(
        (
            target.ownership_valid
            & (target.ownership == OWNER_PRIMARY_ACTOR)
            & (~target.actor_presence[:, :, 0, None])
        ).any().item()
    ):
        raise SourceRelativeActionTargetError("primary ownership lacks its actor")
    if bool(
        (
            target.ownership_valid
            & (target.ownership == OWNER_CO_ACTOR)
            & (~target.actor_presence[:, :, 1, None])
        ).any().item()
    ):
        raise SourceRelativeActionTargetError("co-actor ownership lacks its actor")
    if bool(
        (
            target.ownership_valid
            & (target.ownership == OWNER_GOAL_CONTAINER)
            & (~target.object_presence[:, :, OBJECT_SLOT_COUNT - 1, None])
        ).any().item()
    ):
        raise SourceRelativeActionTargetError(
            "goal-container ownership lacks the goal-container slot"
        )
    if bool(
        (
            target.ownership_valid[:, :, OBJECT_SLOT_COUNT - 1]
            & (
                target.ownership[:, :, OBJECT_SLOT_COUNT - 1]
                == OWNER_GOAL_CONTAINER
            )
        ).any().item()
    ):
        raise SourceRelativeActionTargetError(
            "goal-container slot cannot own itself"
        )

    start, end, hold_start, duration, phase_structure_valid = _phase_summary(
        target.phase_channels, target.phase_valid
    )
    if not torch.equal(target.phase_summary_valid, phase_structure_valid):
        raise SourceRelativeActionTargetError("phase summary validity differs")
    if not torch.equal(target.action_start_phase, start):
        raise SourceRelativeActionTargetError("action onset summary differs")
    if not torch.equal(target.action_end_phase, end):
        raise SourceRelativeActionTargetError("action terminal summary differs")
    if not torch.equal(target.terminal_hold_start_phase, hold_start):
        raise SourceRelativeActionTargetError("terminal-hold summary differs")
    if not torch.equal(target.duration_phases, duration):
        raise SourceRelativeActionTargetError("duration summary differs")
    if not torch.equal(target.phase_summary_valid, target.sample_valid):
        raise SourceRelativeActionTargetError(
            "every eligible draft requires a complete terminal-hold phase plan"
        )

    actor_amplitude, actor_amplitude_valid = _masked_amplitude(
        target.actor_delta, target.actor_delta_valid
    )
    object_amplitude, object_amplitude_valid = _masked_amplitude(
        target.object_delta, target.object_delta_valid
    )
    relative_amplitude, relative_amplitude_valid = _masked_amplitude(
        target.relative_delta.reshape(
            batch,
            PHASE_COUNT,
            ACTOR_SLOT_COUNT * OBJECT_SLOT_COUNT,
            2,
        ),
        target.relative_delta_valid.reshape(
            batch, PHASE_COUNT, ACTOR_SLOT_COUNT * OBJECT_SLOT_COUNT
        ),
    )
    if not torch.equal(target.actor_amplitude_valid, actor_amplitude_valid) or not torch.equal(
        target.actor_amplitude, actor_amplitude
    ):
        raise SourceRelativeActionTargetError("actor amplitude summary differs")
    if not torch.equal(target.object_amplitude_valid, object_amplitude_valid) or not torch.equal(
        target.object_amplitude, object_amplitude
    ):
        raise SourceRelativeActionTargetError("object amplitude summary differs")
    if not torch.equal(
        target.relative_amplitude_valid, relative_amplitude_valid
    ) or not torch.equal(target.relative_amplitude, relative_amplitude):
        raise SourceRelativeActionTargetError("relative amplitude summary differs")
    mean_speed, peak_speed, speed_valid = _masked_speed(
        torch.cat((target.actor_delta, target.object_delta), dim=2),
        torch.cat((target.actor_delta_valid, target.object_delta_valid), dim=2),
    )
    if not torch.equal(target.speed_valid, speed_valid):
        raise SourceRelativeActionTargetError("speed validity summary differs")
    if not torch.equal(target.mean_speed, mean_speed) or not torch.equal(
        target.peak_speed, peak_speed
    ):
        raise SourceRelativeActionTargetError("speed summary differs")

    for sample_index in range(batch):
        if not bool(target.sample_valid[sample_index].item()):
            continue
        terminal_index = int(target.action_end_phase[sample_index].item())
        hold_index = int(target.terminal_hold_start_phase[sample_index].item())
        baseline_index = max(
            int(target.action_start_phase[sample_index].item()) - 1, 0
        )
        active_actor = target.actor_role_valid[sample_index]
        active_object = target.object_role_valid[sample_index]
        active_entity = torch.cat((active_actor, active_object), dim=0)
        active_pair = active_actor[:, None] & active_object[None, :]
        entity_all_values = torch.cat(
            (target.actor_delta, target.object_delta), dim=2
        )[sample_index]
        entity_all_valid = torch.cat(
            (target.actor_delta_valid, target.object_delta_valid), dim=2
        )[sample_index]
        entity_position = torch.cat(
            (target.actor_target_position, target.object_target_position), dim=2
        )[sample_index]
        entity_position_valid = torch.cat(
            (
                target.actor_target_position_valid,
                target.object_target_position_valid,
            ),
            dim=2,
        )[sample_index]
        presence_all = target_presence[sample_index]
        presence_valid_all = target_presence_valid[sample_index]
        entity_state_valid = entity_all_valid | entity_position_valid | (
            (~presence_all) & presence_valid_all
        )
        entity_state_values = torch.where(
            entity_all_valid[..., None],
            entity_all_values,
            torch.where(
                entity_position_valid[..., None],
                entity_position,
                torch.zeros_like(entity_all_values),
            ),
        )
        if bool(
            (
                active_entity[None, :]
                & (~entity_state_valid[terminal_index:])
            ).any().item()
        ):
            raise SourceRelativeActionTargetError(
                "terminal hold lacks active-slot state evidence"
            )
        hold_actor_present = target.actor_presence[
            sample_index, terminal_index:
        ]
        hold_object_present = target.object_presence[
            sample_index, terminal_index:
        ]
        required_hold_pairs = (
            active_pair[None, :, :]
            & hold_actor_present[:, :, None]
            & hold_object_present[:, None, :]
        )
        if bool(
            (
                required_hold_pairs
                & (~target.contact_valid[sample_index, terminal_index:])
            ).any().item()
        ) or bool(
            (
                active_object[None, :]
                & hold_object_present
                & (~target.ownership_valid[sample_index, terminal_index:])
            ).any().item()
        ):
            raise SourceRelativeActionTargetError(
                "terminal hold lacks relation-state evidence"
            )

        pre_state_valid = entity_state_valid[: baseline_index + 1]
        if bool(
            (active_entity[None, :] & (~pre_state_valid)).any().item()
        ):
            raise SourceRelativeActionTargetError(
                "pre-onset state evidence is incomplete"
            )
        if bool(
            (
                active_entity[None, :]
                & (
                    (~target.source_presence_valid[
                        sample_index, : baseline_index + 1
                    ])
                    | (~presence_valid_all[: baseline_index + 1])
                    | (
                        target.source_presence[
                            sample_index, : baseline_index + 1
                        ]
                        != presence_all[: baseline_index + 1]
                    )
                )
            ).any().item()
        ):
            raise SourceRelativeActionTargetError(
                "pre-onset source/target presence differs"
            )
        if bool(
            (
                active_entity[None, :]
                & presence_all[: baseline_index + 1]
                & (~entity_all_valid[: baseline_index + 1])
            ).any().item()
        ):
            raise SourceRelativeActionTargetError(
                "present pre-onset slots lack strict pair delta evidence"
            )
        if bool(
            target.entity_create[
                sample_index, : baseline_index + 1
            ].any().item()
        ) or bool(
            target.entity_delete[
                sample_index, : baseline_index + 1
            ].any().item()
        ):
            raise SourceRelativeActionTargetError(
                "lifecycle changes before onset"
            )
        pre_norm = torch.linalg.vector_norm(
            entity_all_values[: baseline_index + 1], dim=-1
        )
        if bool(
            (
                active_entity[None, :]
                & entity_all_valid[: baseline_index + 1]
                & (pre_norm > PRE_ONSET_DELTA_TOLERANCE)
            ).any().item()
        ):
            raise SourceRelativeActionTargetError(
                "pre-onset state is not the source baseline"
            )
        if not torch.equal(
            presence_all[: baseline_index + 1],
            presence_all[baseline_index : baseline_index + 1].expand(
                baseline_index + 1, -1
            ),
        ):
            raise SourceRelativeActionTargetError(
                "presence changes before onset"
            )
        pre_contact_valid = target.contact_valid[
            sample_index, : baseline_index + 1
        ]
        pre_contact = target.contact[sample_index, : baseline_index + 1]
        required_baseline_pairs = (
            active_pair
            & target.actor_presence[sample_index, baseline_index, :, None]
            & target.object_presence[sample_index, baseline_index, None, :]
        )
        required_baseline_owners = (
            active_object
            & target.object_presence[sample_index, baseline_index]
        )
        if bool(
            (
                required_baseline_pairs
                & (~pre_contact_valid[baseline_index])
            ).any().item()
        ) or bool(
            (
                required_baseline_owners
                & (~target.ownership_valid[sample_index, baseline_index])
            ).any().item()
        ):
            raise SourceRelativeActionTargetError(
                "pre-onset relation-state evidence is incomplete"
            )
        if not torch.equal(
            pre_contact_valid,
            pre_contact_valid[-1:].expand_as(pre_contact_valid),
        ) or not torch.equal(
            pre_contact,
            pre_contact[-1:].expand_as(pre_contact),
        ):
            raise SourceRelativeActionTargetError(
                "contact changes before onset"
            )
        pre_owner_valid = target.ownership_valid[
            sample_index, : baseline_index + 1
        ]
        pre_owner = target.ownership[sample_index, : baseline_index + 1]
        if not torch.equal(
            pre_owner_valid,
            pre_owner_valid[-1:].expand_as(pre_owner_valid),
        ) or not torch.equal(
            pre_owner,
            pre_owner[-1:].expand_as(pre_owner),
        ):
            raise SourceRelativeActionTargetError(
                "ownership changes before onset"
            )

        transition_start = int(target.action_start_phase[sample_index].item())
        left_values = entity_all_values[transition_start:terminal_index]
        right_values = entity_all_values[transition_start + 1 : terminal_index + 1]
        left_valid = entity_all_valid[transition_start:terminal_index]
        right_valid = entity_all_valid[
            transition_start + 1 : terminal_index + 1
        ]
        transition_changed = bool(
            (
                active_entity[None, :]
                & left_valid
                & right_valid
                & (
                    torch.linalg.vector_norm(right_values - left_values, dim=-1)
                    >= MIN_ACTION_STATE_CHANGE
                )
            ).any().item()
        )
        transition_changed = transition_changed or bool(
            (
                active_entity[None, :]
                & (
                    presence_all[transition_start:terminal_index]
                    != presence_all[transition_start + 1 : terminal_index + 1]
                )
            ).any().item()
        )
        transition_changed = transition_changed or bool(
            (
                active_pair[None, :, :]
                & target.contact_valid[
                    sample_index, transition_start:terminal_index
                ]
                & target.contact_valid[
                    sample_index, transition_start + 1 : terminal_index + 1
                ]
                & (
                    target.contact[
                        sample_index, transition_start:terminal_index
                    ]
                    != target.contact[
                        sample_index, transition_start + 1 : terminal_index + 1
                    ]
                )
            ).any().item()
        )
        transition_changed = transition_changed or bool(
            (
                active_object[None, :]
                & target.ownership_valid[
                    sample_index, transition_start:terminal_index
                ]
                & target.ownership_valid[
                    sample_index, transition_start + 1 : terminal_index + 1
                ]
                & (
                    target.ownership[
                        sample_index, transition_start:terminal_index
                    ]
                    != target.ownership[
                        sample_index, transition_start + 1 : terminal_index + 1
                    ]
                )
            ).any().item()
        )
        if not transition_changed:
            raise SourceRelativeActionTargetError(
                "transition interval contains no state change"
            )
        terminal_changed = bool(
            (
                active_entity
                & entity_all_valid[baseline_index]
                & entity_all_valid[terminal_index]
                & (
                    torch.linalg.vector_norm(
                        entity_all_values[terminal_index]
                        - entity_all_values[baseline_index],
                        dim=-1,
                    )
                    >= MIN_ACTION_STATE_CHANGE
                )
            ).any().item()
        )
        terminal_changed = terminal_changed or bool(
            (
                active_entity
                & (presence_all[baseline_index] != presence_all[terminal_index])
            ).any().item()
        )
        terminal_changed = terminal_changed or bool(
            (
                active_pair
                & target.contact_valid[sample_index, baseline_index]
                & target.contact_valid[sample_index, terminal_index]
                & (
                    target.contact[sample_index, baseline_index]
                    != target.contact[sample_index, terminal_index]
                )
            ).any().item()
        )
        terminal_changed = terminal_changed or bool(
            (
                active_object
                & target.ownership_valid[sample_index, baseline_index]
                & target.ownership_valid[sample_index, terminal_index]
                & (
                    target.ownership[sample_index, baseline_index]
                    != target.ownership[sample_index, terminal_index]
                )
            ).any().item()
        )
        if not terminal_changed:
            raise SourceRelativeActionTargetError(
                "terminal state does not differ from the source baseline"
            )
        terminal_presence = target_presence[sample_index, terminal_index]
        if not torch.equal(
            target_presence[sample_index, hold_index:],
            terminal_presence[None].expand(PHASE_COUNT - hold_index, -1),
        ):
            raise SourceRelativeActionTargetError(
                "presence changes during terminal hold"
            )
        terminal_contact_valid = target.contact_valid[sample_index, terminal_index]
        terminal_contact = target.contact[sample_index, terminal_index]
        if not torch.equal(
            target.contact_valid[sample_index, hold_index:],
            terminal_contact_valid[None].expand(PHASE_COUNT - hold_index, -1, -1),
        ) or not torch.equal(
            target.contact[sample_index, hold_index:],
            terminal_contact[None].expand(PHASE_COUNT - hold_index, -1, -1),
        ):
            raise SourceRelativeActionTargetError(
                "contact state changes during terminal hold"
            )
        terminal_owner_valid = target.ownership_valid[sample_index, terminal_index]
        terminal_owner = target.ownership[sample_index, terminal_index]
        if not torch.equal(
            target.ownership_valid[sample_index, hold_index:],
            terminal_owner_valid[None].expand(PHASE_COUNT - hold_index, -1),
        ) or not torch.equal(
            target.ownership[sample_index, hold_index:],
            terminal_owner[None].expand(PHASE_COUNT - hold_index, -1),
        ):
            raise SourceRelativeActionTargetError(
                "ownership state changes during terminal hold"
            )
        entity_values = entity_state_values[terminal_index:]
        entity_valid = entity_state_valid[terminal_index:]
        interval_valid = entity_valid[1:] & entity_valid[:-1]
        hold_speed = torch.linalg.vector_norm(
            entity_values[1:] - entity_values[:-1], dim=-1
        )
        if bool(
            (
                interval_valid
                & (hold_speed > TERMINAL_HOLD_DELTA_TOLERANCE)
            ).any().item()
        ):
            raise SourceRelativeActionTargetError(
                "source-relative state moves during terminal hold"
            )


def validate_local_source_relative_action_target_draft_v2(
    draft: LocalSourceRelativeActionTargetDraftV2,
    *,
    camera_bundle: Optional[SourceRelativeCameraBundleV2] = None,
    annotations: Optional[SourceRelativeActionAnnotationsV2] = None,
) -> None:
    if not isinstance(draft, LocalSourceRelativeActionTargetDraftV2):
        raise SourceRelativeActionTargetError(
            "local schema value must be LocalSourceRelativeActionTargetDraftV2"
        )
    validate_source_relative_action_target_v2(draft.target)
    receipt = draft.receipt
    if not isinstance(receipt, LocalSourceRelativeActionTargetDraftReceiptV2):
        raise SourceRelativeActionTargetError("local draft receipt type differs")
    if receipt.schema_version != DRAFT_RECEIPT_SCHEMA_VERSION:
        raise SourceRelativeActionTargetError("local draft receipt schema differs")
    if receipt.external_authority_verified is not False:
        raise SourceRelativeActionTargetError(
            "local draft receipt cannot claim external authority"
        )
    batch = draft.target.sample_valid.shape[0]
    _require_ids(receipt.sample_ids, label="draft_receipt.sample_ids", expected=batch)
    _require_digest_tuple(
        receipt.source_media_sha256,
        label="draft_receipt.source_media_sha256",
        expected=batch,
    )
    _require_digest_tuple(
        receipt.target_media_sha256,
        label="draft_receipt.target_media_sha256",
        expected=batch,
    )
    for name in (
        "target_payload_sha256",
        "camera_receipt_sha256",
        "annotation_payload_sha256",
        "annotation_artifact_sha256",
        "split_manifest_sha256",
        "receipt_sha256",
    ):
        _require_sha256(getattr(receipt, name), label="draft_receipt.%s" % name)
    if receipt.target_payload_sha256 != _target_payload_sha256(draft.target):
        raise SourceRelativeActionTargetError("local draft bytes differ from receipt")
    payload = {
        "schema_version": receipt.schema_version,
        "sample_ids": list(receipt.sample_ids),
        "source_media_sha256": list(receipt.source_media_sha256),
        "target_media_sha256": list(receipt.target_media_sha256),
        "target_payload_sha256": receipt.target_payload_sha256,
        "camera_receipt_sha256": receipt.camera_receipt_sha256,
        "annotation_payload_sha256": receipt.annotation_payload_sha256,
        "annotation_artifact_sha256": receipt.annotation_artifact_sha256,
        "split_manifest_sha256": receipt.split_manifest_sha256,
        "external_authority_verified": False,
    }
    if receipt.receipt_sha256 != _canonical_sha256(payload):
        raise SourceRelativeActionTargetError("local draft receipt digest differs")
    if camera_bundle is not None:
        validate_source_relative_camera_bundle_v2(camera_bundle)
        if receipt.sample_ids != camera_bundle.evidence.sample_ids:
            raise SourceRelativeActionTargetError("draft/camera sample order differs")
        if receipt.source_media_sha256 != camera_bundle.evidence.source_media_sha256:
            raise SourceRelativeActionTargetError("draft/camera source media differs")
        if receipt.target_media_sha256 != camera_bundle.evidence.target_media_sha256:
            raise SourceRelativeActionTargetError("draft/camera target media differs")
        if receipt.camera_receipt_sha256 != camera_bundle.receipt.receipt_sha256:
            raise SourceRelativeActionTargetError("draft camera receipt differs")
        if annotations is not None:
            annotation_hash = _annotations_payload_sha256(
                annotations,
                batch=batch,
                device=camera_bundle.evidence.source_actor_xy.device,
            )
            if receipt.annotation_payload_sha256 != annotation_hash:
                raise SourceRelativeActionTargetError(
                    "draft annotation bytes differ from receipt"
                )
    elif annotations is not None:
        raise SourceRelativeActionTargetError(
            "annotations cannot be checked without their camera bundle/device binding"
        )


def promote_local_draft_to_qy_source_relative_action_target_v2(
    draft: LocalSourceRelativeActionTargetDraftV2,
    *,
    external_clean_pair_authority: Any,
) -> QYSourceRelativeActionTargetV2:
    del draft, external_clean_pair_authority
    raise SourceRelativeActionTargetError(
        "q_y promotion requires external clean-pair authority; not implemented"
    )


def validate_qy_source_relative_action_target_v2(
    qy: QYSourceRelativeActionTargetV2,
) -> None:
    del qy
    raise SourceRelativeActionTargetError(
        "q_y validation requires external clean-pair authority; not implemented"
    )


PHASE_TRANSPORT_LAYOUT = (
    ("actor_delta", ACTOR_SLOT_COUNT * 2),
    ("actor_delta_valid", ACTOR_SLOT_COUNT),
    ("actor_target_position", ACTOR_SLOT_COUNT * 2),
    ("actor_target_position_valid", ACTOR_SLOT_COUNT),
    ("object_delta", OBJECT_SLOT_COUNT * 2),
    ("object_delta_valid", OBJECT_SLOT_COUNT),
    ("object_target_position", OBJECT_SLOT_COUNT * 2),
    ("object_target_position_valid", OBJECT_SLOT_COUNT),
    ("relative_delta", ACTOR_SLOT_COUNT * OBJECT_SLOT_COUNT * 2),
    ("relative_delta_valid", ACTOR_SLOT_COUNT * OBJECT_SLOT_COUNT),
    ("actor_presence", ACTOR_SLOT_COUNT),
    ("actor_presence_valid", ACTOR_SLOT_COUNT),
    ("object_presence", OBJECT_SLOT_COUNT),
    ("object_presence_valid", OBJECT_SLOT_COUNT),
    ("source_presence", ENTITY_SLOT_COUNT),
    ("source_presence_valid", ENTITY_SLOT_COUNT),
    ("entity_create", ENTITY_SLOT_COUNT),
    ("entity_delete", ENTITY_SLOT_COUNT),
    ("lifecycle_valid", ENTITY_SLOT_COUNT),
    ("contact", ACTOR_SLOT_COUNT * OBJECT_SLOT_COUNT),
    ("contact_valid", ACTOR_SLOT_COUNT * OBJECT_SLOT_COUNT),
    ("ownership", OBJECT_SLOT_COUNT),
    ("ownership_valid", OBJECT_SLOT_COUNT),
    ("phase_channels", len(PHASE_CHANNEL_NAMES)),
    ("phase_valid", 1),
)
GLOBAL_TRANSPORT_LAYOUT = (
    ("actor_roles", ACTOR_SLOT_COUNT),
    ("actor_role_valid", ACTOR_SLOT_COUNT),
    ("object_roles", OBJECT_SLOT_COUNT),
    ("object_role_valid", OBJECT_SLOT_COUNT),
    ("initial_source_presence", ENTITY_SLOT_COUNT),
    ("initial_source_presence_valid", ENTITY_SLOT_COUNT),
    ("action_start_phase", 1),
    ("action_end_phase", 1),
    ("terminal_hold_start_phase", 1),
    ("duration_phases", 1),
    ("phase_summary_valid", 1),
    ("actor_amplitude", ACTOR_SLOT_COUNT),
    ("actor_amplitude_valid", ACTOR_SLOT_COUNT),
    ("object_amplitude", OBJECT_SLOT_COUNT),
    ("object_amplitude_valid", OBJECT_SLOT_COUNT),
    ("relative_amplitude", ACTOR_SLOT_COUNT * OBJECT_SLOT_COUNT),
    ("relative_amplitude_valid", ACTOR_SLOT_COUNT * OBJECT_SLOT_COUNT),
    ("mean_speed", ENTITY_SLOT_COUNT),
    ("peak_speed", ENTITY_SLOT_COUNT),
    ("speed_valid", ENTITY_SLOT_COUNT),
    ("sample_valid", 1),
    ("abstain", 1),
    ("abstain_reasons", len(ABSTAIN_REASON_NAMES)),
)
PHASE_TRANSPORT_WIDTH = sum(width for _, width in PHASE_TRANSPORT_LAYOUT)
GLOBAL_TRANSPORT_WIDTH = sum(width for _, width in GLOBAL_TRANSPORT_LAYOUT)


def _transport_layout_sha256() -> str:
    return _canonical_sha256(
        {
            "schema_version": TRANSPORT_SCHEMA_VERSION,
            "phase_count": PHASE_COUNT,
            "actor_slots": list(ACTOR_ROLE_NAMES),
            "object_slots": list(OBJECT_ROLE_NAMES),
            "ownership": list(OWNERSHIP_NAMES),
            "phase_channels": list(PHASE_CHANNEL_NAMES),
            "abstain_reasons": list(ABSTAIN_REASON_NAMES),
            "phase_layout": [list(item) for item in PHASE_TRANSPORT_LAYOUT],
            "global_layout": [list(item) for item in GLOBAL_TRANSPORT_LAYOUT],
            "storage_dtype": "torch.float32",
            "terminal_hold_delta_tolerance": TERMINAL_HOLD_DELTA_TOLERANCE,
            "pre_onset_delta_tolerance": PRE_ONSET_DELTA_TOLERANCE,
            "minimum_action_state_change": MIN_ACTION_STATE_CHANGE,
            "role": "schema_transport_roundtrip_only",
        }
    )


class _FrozenTransportModuleV2(_nn().Module):
    def __init__(self, module_code: int) -> None:
        super().__init__()
        torch = _torch()
        self.register_buffer(
            "_abi",
            torch.tensor(
                (
                    2,
                    module_code,
                    PHASE_COUNT,
                    ACTOR_SLOT_COUNT,
                    OBJECT_SLOT_COUNT,
                    PHASE_TRANSPORT_WIDTH,
                    GLOBAL_TRANSPORT_WIDTH,
                ),
                dtype=torch.int64,
            ),
            persistent=True,
        )
        super().train(False)

    def train(self, mode: bool = True) -> "_FrozenTransportModuleV2":
        if mode:
            raise SourceRelativeActionTargetError(
                "schema transport modules are permanently frozen"
            )
        super().train(False)
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Validate and then bypass nn.Module's hook machinery completely.
        # Instance hooks are rejected as non-canonical state, while process-wide
        # global hooks are deliberately inert for this lossless transport ABI.
        # A check only inside forward would be too late: a pre-hook could remove
        # itself and replace inputs, while a forward hook could replace output.
        _validate_frozen_module(
            self,
            expected_type=type(self),
            label="schema transport module",
        )
        return self.forward(*args, **kwargs)


class FrozenSourceRelativeSchemaEncoderV2(_FrozenTransportModuleV2):
    """Deterministic field packer; not a learned representation encoder."""

    def __init__(self) -> None:
        super().__init__(module_code=1)

    def forward(
        self,
        draft: LocalSourceRelativeActionTargetDraftV2,
        transport_receipt: FrozenSchemaTransportReceiptV2,
    ) -> LocalSchemaTransportV2:
        validate_frozen_schema_transport_receipt_v2(
            transport_receipt, encoder=self
        )
        validate_local_source_relative_action_target_draft_v2(draft)
        torch = _torch()
        target = draft.target
        batch = target.sample_valid.shape[0]
        phase_parts = (
            target.actor_delta.reshape(batch, PHASE_COUNT, -1),
            target.actor_delta_valid.reshape(batch, PHASE_COUNT, -1),
            target.actor_target_position.reshape(batch, PHASE_COUNT, -1),
            target.actor_target_position_valid.reshape(batch, PHASE_COUNT, -1),
            target.object_delta.reshape(batch, PHASE_COUNT, -1),
            target.object_delta_valid.reshape(batch, PHASE_COUNT, -1),
            target.object_target_position.reshape(batch, PHASE_COUNT, -1),
            target.object_target_position_valid.reshape(batch, PHASE_COUNT, -1),
            target.relative_delta.reshape(batch, PHASE_COUNT, -1),
            target.relative_delta_valid.reshape(batch, PHASE_COUNT, -1),
            target.actor_presence.reshape(batch, PHASE_COUNT, -1),
            target.actor_presence_valid.reshape(batch, PHASE_COUNT, -1),
            target.object_presence.reshape(batch, PHASE_COUNT, -1),
            target.object_presence_valid.reshape(batch, PHASE_COUNT, -1),
            target.source_presence.reshape(batch, PHASE_COUNT, -1),
            target.source_presence_valid.reshape(batch, PHASE_COUNT, -1),
            target.entity_create.reshape(batch, PHASE_COUNT, -1),
            target.entity_delete.reshape(batch, PHASE_COUNT, -1),
            target.lifecycle_valid.reshape(batch, PHASE_COUNT, -1),
            target.contact.reshape(batch, PHASE_COUNT, -1),
            target.contact_valid.reshape(batch, PHASE_COUNT, -1),
            target.ownership.reshape(batch, PHASE_COUNT, -1),
            target.ownership_valid.reshape(batch, PHASE_COUNT, -1),
            target.phase_channels.reshape(batch, PHASE_COUNT, -1),
            target.phase_valid.reshape(batch, PHASE_COUNT, 1),
        )
        global_parts = (
            target.actor_roles,
            target.actor_role_valid,
            target.object_roles,
            target.object_role_valid,
            target.initial_source_presence,
            target.initial_source_presence_valid,
            target.action_start_phase[:, None],
            target.action_end_phase[:, None],
            target.terminal_hold_start_phase[:, None],
            target.duration_phases[:, None],
            target.phase_summary_valid[:, None],
            target.actor_amplitude,
            target.actor_amplitude_valid,
            target.object_amplitude,
            target.object_amplitude_valid,
            target.relative_amplitude,
            target.relative_amplitude_valid,
            target.mean_speed,
            target.peak_speed,
            target.speed_valid,
            target.sample_valid[:, None],
            target.abstain[:, None],
            target.abstain_reasons,
        )
        with torch.no_grad():
            phase_code = torch.cat(
                tuple(part.to(dtype=torch.float32) for part in phase_parts), dim=-1
            ).detach().clone().contiguous()
            global_code = torch.cat(
                tuple(part.to(dtype=torch.float32) for part in global_parts), dim=-1
            ).detach().clone().contiguous()
        if tuple(phase_code.shape) != (
            batch,
            PHASE_COUNT,
            PHASE_TRANSPORT_WIDTH,
        ):
            raise SourceRelativeActionTargetError("phase transport layout drifted")
        if tuple(global_code.shape) != (batch, GLOBAL_TRANSPORT_WIDTH):
            raise SourceRelativeActionTargetError("global transport layout drifted")
        payload = {
            "schema_version": TRANSPORT_SCHEMA_VERSION,
            "sample_ids": list(draft.receipt.sample_ids),
            "phase_code_sha256": _tensor_sha256(
                phase_code, label="transport.phase_code"
            ),
            "global_code_sha256": _tensor_sha256(
                global_code, label="transport.global_code"
            ),
            "draft_receipt_sha256": draft.receipt.receipt_sha256,
            "codec_receipt_sha256": transport_receipt.receipt_sha256,
        }
        return LocalSchemaTransportV2(
            schema_version=TRANSPORT_SCHEMA_VERSION,
            sample_ids=draft.receipt.sample_ids,
            phase_code=phase_code,
            global_code=global_code,
            draft_receipt_sha256=draft.receipt.receipt_sha256,
            codec_receipt_sha256=transport_receipt.receipt_sha256,
            transport_sha256=_canonical_sha256(payload),
        )


def _require_binary_storage(value: Any, *, label: str) -> Any:
    if not bool(((value == 0.0) | (value == 1.0)).all().item()):
        raise SourceRelativeActionTargetError("%s must contain exact binary values" % label)
    return value.to(dtype=_torch().bool)


def _require_integer_storage(
    value: Any, *, label: str, minimum: int, maximum: int
) -> Any:
    rounded = value.round()
    if not bool((value == rounded).all().item()):
        raise SourceRelativeActionTargetError("%s must contain exact integers" % label)
    integer = rounded.to(dtype=_torch().int64)
    if bool(((integer < minimum) | (integer > maximum)).any().item()):
        raise SourceRelativeActionTargetError("%s is outside the closed range" % label)
    return integer


class FrozenSourceRelativeSchemaDecoderV2(_FrozenTransportModuleV2):
    """Exact inverse of the deterministic schema transport packer."""

    def __init__(self) -> None:
        super().__init__(module_code=2)

    def forward(
        self,
        transport: LocalSchemaTransportV2,
        target_receipt: LocalSourceRelativeActionTargetDraftReceiptV2,
        transport_receipt: FrozenSchemaTransportReceiptV2,
    ) -> LocalSourceRelativeActionTargetDraftV2:
        validate_frozen_schema_transport_receipt_v2(
            transport_receipt, decoder=self
        )
        validate_local_schema_transport_v2(
            transport, transport_receipt=transport_receipt
        )
        if not isinstance(
            target_receipt, LocalSourceRelativeActionTargetDraftReceiptV2
        ):
            raise SourceRelativeActionTargetError(
                "decoder requires the local draft receipt"
            )
        if transport.draft_receipt_sha256 != target_receipt.receipt_sha256:
            raise SourceRelativeActionTargetError("transport/draft receipt binding differs")
        if transport.sample_ids != target_receipt.sample_ids:
            raise SourceRelativeActionTargetError(
                "transport/draft ordered sample IDs differ"
            )
        torch = _torch()
        phase = transport.phase_code
        global_value = transport.global_code
        batch = phase.shape[0]

        phase_cursor = 0

        def phase_take(width: int) -> Any:
            nonlocal phase_cursor
            result = phase[..., phase_cursor : phase_cursor + width]
            phase_cursor += width
            return result

        actor_delta = phase_take(ACTOR_SLOT_COUNT * 2).reshape(
            batch, PHASE_COUNT, ACTOR_SLOT_COUNT, 2
        )
        actor_delta_valid = _require_binary_storage(
            phase_take(ACTOR_SLOT_COUNT), label="transport.actor_delta_valid"
        )
        actor_target_position = phase_take(ACTOR_SLOT_COUNT * 2).reshape(
            batch, PHASE_COUNT, ACTOR_SLOT_COUNT, 2
        )
        actor_target_position_valid = _require_binary_storage(
            phase_take(ACTOR_SLOT_COUNT),
            label="transport.actor_target_position_valid",
        )
        object_delta = phase_take(OBJECT_SLOT_COUNT * 2).reshape(
            batch, PHASE_COUNT, OBJECT_SLOT_COUNT, 2
        )
        object_delta_valid = _require_binary_storage(
            phase_take(OBJECT_SLOT_COUNT), label="transport.object_delta_valid"
        )
        object_target_position = phase_take(OBJECT_SLOT_COUNT * 2).reshape(
            batch, PHASE_COUNT, OBJECT_SLOT_COUNT, 2
        )
        object_target_position_valid = _require_binary_storage(
            phase_take(OBJECT_SLOT_COUNT),
            label="transport.object_target_position_valid",
        )
        relative_delta = phase_take(ACTOR_SLOT_COUNT * OBJECT_SLOT_COUNT * 2).reshape(
            batch, PHASE_COUNT, ACTOR_SLOT_COUNT, OBJECT_SLOT_COUNT, 2
        )
        relative_delta_valid = _require_binary_storage(
            phase_take(ACTOR_SLOT_COUNT * OBJECT_SLOT_COUNT),
            label="transport.relative_delta_valid",
        ).reshape(batch, PHASE_COUNT, ACTOR_SLOT_COUNT, OBJECT_SLOT_COUNT)
        actor_presence = _require_binary_storage(
            phase_take(ACTOR_SLOT_COUNT), label="transport.actor_presence"
        )
        actor_presence_valid = _require_binary_storage(
            phase_take(ACTOR_SLOT_COUNT), label="transport.actor_presence_valid"
        )
        object_presence = _require_binary_storage(
            phase_take(OBJECT_SLOT_COUNT), label="transport.object_presence"
        )
        object_presence_valid = _require_binary_storage(
            phase_take(OBJECT_SLOT_COUNT), label="transport.object_presence_valid"
        )
        source_presence = _require_binary_storage(
            phase_take(ENTITY_SLOT_COUNT), label="transport.source_presence"
        )
        source_presence_valid = _require_binary_storage(
            phase_take(ENTITY_SLOT_COUNT),
            label="transport.source_presence_valid",
        )
        entity_create = _require_binary_storage(
            phase_take(ENTITY_SLOT_COUNT), label="transport.entity_create"
        )
        entity_delete = _require_binary_storage(
            phase_take(ENTITY_SLOT_COUNT), label="transport.entity_delete"
        )
        lifecycle_valid = _require_binary_storage(
            phase_take(ENTITY_SLOT_COUNT), label="transport.lifecycle_valid"
        )
        contact = _require_binary_storage(
            phase_take(ACTOR_SLOT_COUNT * OBJECT_SLOT_COUNT),
            label="transport.contact",
        ).reshape(batch, PHASE_COUNT, ACTOR_SLOT_COUNT, OBJECT_SLOT_COUNT)
        contact_valid = _require_binary_storage(
            phase_take(ACTOR_SLOT_COUNT * OBJECT_SLOT_COUNT),
            label="transport.contact_valid",
        ).reshape(batch, PHASE_COUNT, ACTOR_SLOT_COUNT, OBJECT_SLOT_COUNT)
        ownership = _require_integer_storage(
            phase_take(OBJECT_SLOT_COUNT),
            label="transport.ownership",
            minimum=OWNER_NONE,
            maximum=OWNER_GOAL_CONTAINER,
        )
        ownership_valid = _require_binary_storage(
            phase_take(OBJECT_SLOT_COUNT), label="transport.ownership_valid"
        )
        phase_channels = _require_binary_storage(
            phase_take(len(PHASE_CHANNEL_NAMES)), label="transport.phase_channels"
        )
        phase_valid = _require_binary_storage(
            phase_take(1), label="transport.phase_valid"
        ).squeeze(-1)
        if phase_cursor != PHASE_TRANSPORT_WIDTH:
            raise SourceRelativeActionTargetError("phase transport cursor drifted")

        global_cursor = 0

        def global_take(width: int) -> Any:
            nonlocal global_cursor
            result = global_value[:, global_cursor : global_cursor + width]
            global_cursor += width
            return result

        actor_roles = _require_integer_storage(
            global_take(ACTOR_SLOT_COUNT),
            label="transport.actor_roles",
            minimum=0,
            maximum=ACTOR_SLOT_COUNT - 1,
        )
        actor_role_valid = _require_binary_storage(
            global_take(ACTOR_SLOT_COUNT), label="transport.actor_role_valid"
        )
        object_roles = _require_integer_storage(
            global_take(OBJECT_SLOT_COUNT),
            label="transport.object_roles",
            minimum=0,
            maximum=OBJECT_SLOT_COUNT - 1,
        )
        object_role_valid = _require_binary_storage(
            global_take(OBJECT_SLOT_COUNT), label="transport.object_role_valid"
        )
        initial_source_presence = _require_binary_storage(
            global_take(ENTITY_SLOT_COUNT),
            label="transport.initial_source_presence",
        )
        initial_source_presence_valid = _require_binary_storage(
            global_take(ENTITY_SLOT_COUNT),
            label="transport.initial_source_presence_valid",
        )
        action_start_phase = _require_integer_storage(
            global_take(1),
            label="transport.action_start_phase",
            minimum=0,
            maximum=PHASE_COUNT - 1,
        ).squeeze(-1)
        action_end_phase = _require_integer_storage(
            global_take(1),
            label="transport.action_end_phase",
            minimum=0,
            maximum=PHASE_COUNT - 1,
        ).squeeze(-1)
        terminal_hold_start_phase = _require_integer_storage(
            global_take(1),
            label="transport.terminal_hold_start_phase",
            minimum=0,
            maximum=PHASE_COUNT - 1,
        ).squeeze(-1)
        duration_phases = global_take(1).squeeze(-1)
        phase_summary_valid = _require_binary_storage(
            global_take(1), label="transport.phase_summary_valid"
        ).squeeze(-1)
        actor_amplitude = global_take(ACTOR_SLOT_COUNT)
        actor_amplitude_valid = _require_binary_storage(
            global_take(ACTOR_SLOT_COUNT), label="transport.actor_amplitude_valid"
        )
        object_amplitude = global_take(OBJECT_SLOT_COUNT)
        object_amplitude_valid = _require_binary_storage(
            global_take(OBJECT_SLOT_COUNT), label="transport.object_amplitude_valid"
        )
        relative_amplitude = global_take(ACTOR_SLOT_COUNT * OBJECT_SLOT_COUNT)
        relative_amplitude_valid = _require_binary_storage(
            global_take(ACTOR_SLOT_COUNT * OBJECT_SLOT_COUNT),
            label="transport.relative_amplitude_valid",
        )
        mean_speed = global_take(ENTITY_SLOT_COUNT)
        peak_speed = global_take(ENTITY_SLOT_COUNT)
        speed_valid = _require_binary_storage(
            global_take(ENTITY_SLOT_COUNT), label="transport.speed_valid"
        )
        sample_valid = _require_binary_storage(
            global_take(1), label="transport.sample_valid"
        ).squeeze(-1)
        abstain = _require_binary_storage(
            global_take(1), label="transport.abstain"
        ).squeeze(-1)
        abstain_reasons = _require_binary_storage(
            global_take(len(ABSTAIN_REASON_NAMES)),
            label="transport.abstain_reasons",
        )
        if global_cursor != GLOBAL_TRANSPORT_WIDTH:
            raise SourceRelativeActionTargetError("global transport cursor drifted")

        values = locals()
        decoded_fields: Dict[str, Any] = {}
        with torch.no_grad():
            for name in TARGET_FIELD_NAMES:
                decoded_fields[name] = _owned(values[name])
        target = SourceRelativeActionTargetV2(**decoded_fields)
        draft = LocalSourceRelativeActionTargetDraftV2(
            target=target, receipt=target_receipt
        )
        validate_local_source_relative_action_target_draft_v2(draft)
        return draft


class FrozenSourceRelativeSchemaEvaluatorV2(_FrozenTransportModuleV2):
    """Exact schema round-trip checker; never a representation evaluator."""

    def __init__(self) -> None:
        super().__init__(module_code=3)

    def forward(
        self,
        original: LocalSourceRelativeActionTargetDraftV2,
        decoded: LocalSourceRelativeActionTargetDraftV2,
        transport_receipt: FrozenSchemaTransportReceiptV2,
    ) -> LocalSchemaTransportEvaluationV2:
        validate_frozen_schema_transport_receipt_v2(
            transport_receipt, evaluator=self
        )
        validate_local_source_relative_action_target_draft_v2(original)
        validate_local_source_relative_action_target_draft_v2(decoded)
        if original.receipt.receipt_sha256 != decoded.receipt.receipt_sha256:
            raise SourceRelativeActionTargetError("round-trip draft receipts differ")
        torch = _torch()
        field_exact = tuple(
            (
                name,
                bool(
                    torch.equal(
                        getattr(original.target, name), getattr(decoded.target, name)
                    )
                    and _tensor_sha256(
                        getattr(original.target, name), label="original.%s" % name
                    )
                    == _tensor_sha256(
                        getattr(decoded.target, name), label="decoded.%s" % name
                    )
                ),
            )
            for name in TARGET_FIELD_NAMES
        )
        valid_counts = tuple(
            (name, int(getattr(original.target, name).sum().item()))
            for name in TARGET_FIELD_NAMES
            if name.endswith("_valid") or name == "sample_valid"
        )
        exact = all(value for _, value in field_exact)
        payload = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "sample_ids": list(original.receipt.sample_ids),
            "field_exact": [list(item) for item in field_exact],
            "field_valid_counts": [list(item) for item in valid_counts],
            "eligible_sample_count": int(original.target.sample_valid.sum().item()),
            "abstained_sample_count": int(original.target.abstain.sum().item()),
            "exact_roundtrip": exact,
            "local_checks_passed": exact,
            "representation_qualification_evidence": False,
            "r2_evidence": False,
            "formally_qualified": False,
            "training_authorized": False,
            "optimizer_authorized": False,
            "selection_authorized": False,
            "gate_authorized": False,
            "original_payload_sha256": _target_payload_sha256(original.target),
            "decoded_payload_sha256": _target_payload_sha256(decoded.target),
            "codec_receipt_sha256": transport_receipt.receipt_sha256,
        }
        return LocalSchemaTransportEvaluationV2(
            schema_version=EVALUATION_SCHEMA_VERSION,
            sample_ids=original.receipt.sample_ids,
            field_exact=field_exact,
            field_valid_counts=valid_counts,
            eligible_sample_count=payload["eligible_sample_count"],
            abstained_sample_count=payload["abstained_sample_count"],
            exact_roundtrip=exact,
            local_checks_passed=exact,
            representation_qualification_evidence=False,
            r2_evidence=False,
            formally_qualified=False,
            training_authorized=False,
            optimizer_authorized=False,
            selection_authorized=False,
            gate_authorized=False,
            original_payload_sha256=payload["original_payload_sha256"],
            decoded_payload_sha256=payload["decoded_payload_sha256"],
            codec_receipt_sha256=transport_receipt.receipt_sha256,
            receipt_sha256=_canonical_sha256(payload),
        )


def validate_local_schema_transport_evaluation_v2(
    report: LocalSchemaTransportEvaluationV2,
) -> None:
    if not isinstance(report, LocalSchemaTransportEvaluationV2):
        raise SourceRelativeActionTargetError("schema transport evaluation type differs")
    if report.schema_version != EVALUATION_SCHEMA_VERSION:
        raise SourceRelativeActionTargetError("schema transport evaluation schema differs")
    _require_ids(report.sample_ids, label="evaluation.sample_ids")
    exact_names = tuple(name for name, _ in report.field_exact)
    if exact_names != TARGET_FIELD_NAMES:
        raise SourceRelativeActionTargetError("evaluation field order differs")
    if any(type(value) is not bool for _, value in report.field_exact):
        raise SourceRelativeActionTargetError("evaluation exact verdicts must be booleans")
    valid_names = tuple(name for name, _ in report.field_valid_counts)
    expected_valid_names = tuple(
        name
        for name in TARGET_FIELD_NAMES
        if name.endswith("_valid") or name == "sample_valid"
    )
    if valid_names != expected_valid_names or any(
        type(value) is not int or value < 0
        for _, value in report.field_valid_counts
    ):
        raise SourceRelativeActionTargetError("evaluation valid counts differ")
    for name in (
        "eligible_sample_count",
        "abstained_sample_count",
    ):
        if type(getattr(report, name)) is not int or getattr(report, name) < 0:
            raise SourceRelativeActionTargetError("evaluation sample counts differ")
    if report.eligible_sample_count + report.abstained_sample_count != len(
        report.sample_ids
    ):
        raise SourceRelativeActionTargetError("evaluation sample counts do not close")
    if report.exact_roundtrip is not all(value for _, value in report.field_exact):
        raise SourceRelativeActionTargetError("evaluation exact summary differs")
    if report.local_checks_passed is not report.exact_roundtrip:
        raise SourceRelativeActionTargetError("evaluation local check summary differs")
    for name in (
        "representation_qualification_evidence",
        "r2_evidence",
        "formally_qualified",
        "training_authorized",
        "optimizer_authorized",
        "selection_authorized",
        "gate_authorized",
    ):
        if getattr(report, name) is not False:
            raise SourceRelativeActionTargetError(
                "local schema transport evaluation cannot authorize %s" % name
            )
    for name in (
        "original_payload_sha256",
        "decoded_payload_sha256",
        "codec_receipt_sha256",
        "receipt_sha256",
    ):
        _require_sha256(getattr(report, name), label="evaluation.%s" % name)
    payload = {
        "schema_version": report.schema_version,
        "sample_ids": list(report.sample_ids),
        "field_exact": [list(item) for item in report.field_exact],
        "field_valid_counts": [list(item) for item in report.field_valid_counts],
        "eligible_sample_count": report.eligible_sample_count,
        "abstained_sample_count": report.abstained_sample_count,
        "exact_roundtrip": report.exact_roundtrip,
        "local_checks_passed": report.local_checks_passed,
        "representation_qualification_evidence": False,
        "r2_evidence": False,
        "formally_qualified": False,
        "training_authorized": False,
        "optimizer_authorized": False,
        "selection_authorized": False,
        "gate_authorized": False,
        "original_payload_sha256": report.original_payload_sha256,
        "decoded_payload_sha256": report.decoded_payload_sha256,
        "codec_receipt_sha256": report.codec_receipt_sha256,
    }
    if report.receipt_sha256 != _canonical_sha256(payload):
        raise SourceRelativeActionTargetError("evaluation receipt digest differs")


def _module_state_sha256(module: Any) -> str:
    if not isinstance(module, _FrozenTransportModuleV2):
        raise SourceRelativeActionTargetError("schema transport module type differs")
    return _canonical_sha256(
        {
            name: _tensor_sha256(value, label="module.%s" % name)
            for name, value in sorted(module.state_dict().items())
        }
    )


def _validate_frozen_module(module: Any, *, expected_type: Any, label: str) -> None:
    torch = _torch()
    module_codes = {
        FrozenSourceRelativeSchemaEncoderV2: 1,
        FrozenSourceRelativeSchemaDecoderV2: 2,
        FrozenSourceRelativeSchemaEvaluatorV2: 3,
    }
    if expected_type not in module_codes or type(module) is not expected_type:
        raise SourceRelativeActionTargetError("%s must have its exact frozen type" % label)
    if module.training:
        raise SourceRelativeActionTargetError("%s must remain in eval mode" % label)
    if "forward" in module.__dict__ or getattr(
        module.forward, "__func__", None
    ) is not expected_type.forward:
        raise SourceRelativeActionTargetError("%s forward is shadowed" % label)
    for name in (
        "__call__",
        "_call_impl",
        "_wrapped_call_impl",
        "_slow_forward",
    ):
        if name in module.__dict__:
            raise SourceRelativeActionTargetError(
                "%s execution path is shadowed" % label
            )
    if getattr(module, "_compiled_call_impl", None) is not None:
        raise SourceRelativeActionTargetError(
            "%s compiled execution path is not part of the frozen ABI" % label
        )
    hook_names = (
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_forward_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_hooks_always_called",
        "_backward_pre_hooks",
        "_backward_hooks",
        "_state_dict_pre_hooks",
        "_state_dict_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    )
    for name in hook_names:
        hooks = getattr(module, name, None)
        if hooks is not None and len(hooks) != 0:
            raise SourceRelativeActionTargetError(
                "%s must not carry execution or state hooks" % label
            )
    if getattr(module, "_is_full_backward_hook", None) is not None:
        raise SourceRelativeActionTargetError(
            "%s backward-hook mode differs from the frozen ABI" % label
        )
    parameters = tuple(module.parameters())
    if parameters:
        raise SourceRelativeActionTargetError(
            "%s must have zero parameters; it is schema transport only" % label
        )
    if any(parameter.requires_grad for parameter in parameters):
        raise SourceRelativeActionTargetError("%s parameters require gradients" % label)
    if tuple(module._modules.keys()) or tuple(module._parameters.keys()):
        raise SourceRelativeActionTargetError(
            "%s registered module/parameter topology differs" % label
        )
    if tuple(module._buffers.keys()) != ("_abi",) or tuple(
        name for name, _ in module.named_buffers()
    ) != ("_abi",):
        raise SourceRelativeActionTargetError(
            "%s must contain only the canonical ABI buffer" % label
        )
    if module._non_persistent_buffers_set:
        raise SourceRelativeActionTargetError(
            "%s ABI buffer must remain persistent" % label
        )
    abi = _require_tensor(
        module._abi,
        label="%s._abi" % label,
        shape=(7,),
        dtype=torch.int64,
    )
    expected_abi = torch.tensor(
        (
            2,
            module_codes[expected_type],
            PHASE_COUNT,
            ACTOR_SLOT_COUNT,
            OBJECT_SLOT_COUNT,
            PHASE_TRANSPORT_WIDTH,
            GLOBAL_TRANSPORT_WIDTH,
        ),
        dtype=torch.int64,
    )
    if abi.device.type != "cpu" or not abi.is_contiguous() or not torch.equal(
        abi, expected_abi
    ):
        raise SourceRelativeActionTargetError(
            "%s canonical ABI buffer differs" % label
        )
    if tuple(module.state_dict().keys()) != ("_abi",):
        raise SourceRelativeActionTargetError(
            "%s persistent state topology differs" % label
        )


def bind_frozen_source_relative_schema_transport_v2(
    encoder: FrozenSourceRelativeSchemaEncoderV2,
    decoder: FrozenSourceRelativeSchemaDecoderV2,
    evaluator: FrozenSourceRelativeSchemaEvaluatorV2,
    *,
    implementation_artifact_sha256: str,
) -> FrozenSchemaTransportReceiptV2:
    _validate_frozen_module(
        encoder, expected_type=FrozenSourceRelativeSchemaEncoderV2, label="encoder"
    )
    _validate_frozen_module(
        decoder, expected_type=FrozenSourceRelativeSchemaDecoderV2, label="decoder"
    )
    _validate_frozen_module(
        evaluator,
        expected_type=FrozenSourceRelativeSchemaEvaluatorV2,
        label="evaluator",
    )
    payload = {
        "schema_version": TRANSPORT_RECEIPT_SCHEMA_VERSION,
        "encoder_state_sha256": _module_state_sha256(encoder),
        "decoder_state_sha256": _module_state_sha256(decoder),
        "evaluator_state_sha256": _module_state_sha256(evaluator),
        "layout_sha256": _transport_layout_sha256(),
        "implementation_artifact_sha256": _require_sha256(
            implementation_artifact_sha256,
            label="implementation_artifact_sha256",
        ),
        "external_authority_verified": False,
    }
    return FrozenSchemaTransportReceiptV2(
        schema_version=TRANSPORT_RECEIPT_SCHEMA_VERSION,
        encoder_state_sha256=payload["encoder_state_sha256"],
        decoder_state_sha256=payload["decoder_state_sha256"],
        evaluator_state_sha256=payload["evaluator_state_sha256"],
        layout_sha256=payload["layout_sha256"],
        implementation_artifact_sha256=payload["implementation_artifact_sha256"],
        external_authority_verified=False,
        receipt_sha256=_canonical_sha256(payload),
    )


def validate_frozen_schema_transport_receipt_v2(
    receipt: FrozenSchemaTransportReceiptV2,
    *,
    encoder: Optional[FrozenSourceRelativeSchemaEncoderV2] = None,
    decoder: Optional[FrozenSourceRelativeSchemaDecoderV2] = None,
    evaluator: Optional[FrozenSourceRelativeSchemaEvaluatorV2] = None,
) -> None:
    if not isinstance(receipt, FrozenSchemaTransportReceiptV2):
        raise SourceRelativeActionTargetError("schema transport receipt type differs")
    if receipt.schema_version != TRANSPORT_RECEIPT_SCHEMA_VERSION:
        raise SourceRelativeActionTargetError("schema transport receipt schema differs")
    if receipt.external_authority_verified is not False:
        raise SourceRelativeActionTargetError(
            "schema transport receipt cannot claim external authority"
        )
    for name in (
        "encoder_state_sha256",
        "decoder_state_sha256",
        "evaluator_state_sha256",
        "layout_sha256",
        "implementation_artifact_sha256",
        "receipt_sha256",
    ):
        _require_sha256(getattr(receipt, name), label="transport_receipt.%s" % name)
    payload = {
        "schema_version": receipt.schema_version,
        "encoder_state_sha256": receipt.encoder_state_sha256,
        "decoder_state_sha256": receipt.decoder_state_sha256,
        "evaluator_state_sha256": receipt.evaluator_state_sha256,
        "layout_sha256": receipt.layout_sha256,
        "implementation_artifact_sha256": receipt.implementation_artifact_sha256,
        "external_authority_verified": False,
    }
    if receipt.receipt_sha256 != _canonical_sha256(payload):
        raise SourceRelativeActionTargetError("schema transport receipt digest differs")
    if receipt.layout_sha256 != _transport_layout_sha256():
        raise SourceRelativeActionTargetError("schema transport layout differs")
    if encoder is not None:
        _validate_frozen_module(
            encoder,
            expected_type=FrozenSourceRelativeSchemaEncoderV2,
            label="encoder",
        )
        if receipt.encoder_state_sha256 != _module_state_sha256(encoder):
            raise SourceRelativeActionTargetError("encoder state differs from receipt")
    if decoder is not None:
        _validate_frozen_module(
            decoder,
            expected_type=FrozenSourceRelativeSchemaDecoderV2,
            label="decoder",
        )
        if receipt.decoder_state_sha256 != _module_state_sha256(decoder):
            raise SourceRelativeActionTargetError("decoder state differs from receipt")
    if evaluator is not None:
        _validate_frozen_module(
            evaluator,
            expected_type=FrozenSourceRelativeSchemaEvaluatorV2,
            label="evaluator",
        )
        if receipt.evaluator_state_sha256 != _module_state_sha256(evaluator):
            raise SourceRelativeActionTargetError("evaluator state differs from receipt")


def validate_local_schema_transport_v2(
    transport: LocalSchemaTransportV2,
    *,
    transport_receipt: FrozenSchemaTransportReceiptV2,
) -> None:
    torch = _torch()
    validate_frozen_schema_transport_receipt_v2(transport_receipt)
    if not isinstance(transport, LocalSchemaTransportV2):
        raise SourceRelativeActionTargetError("local schema transport type differs")
    if transport.schema_version != TRANSPORT_SCHEMA_VERSION:
        raise SourceRelativeActionTargetError("local schema transport schema differs")
    batch = len(_require_ids(transport.sample_ids, label="transport.sample_ids"))
    phase = _require_tensor(
        transport.phase_code,
        label="transport.phase_code",
        shape=(batch, PHASE_COUNT, PHASE_TRANSPORT_WIDTH),
        dtype=torch.float32,
    )
    global_value = _require_tensor(
        transport.global_code,
        label="transport.global_code",
        shape=(batch, GLOBAL_TRANSPORT_WIDTH),
        dtype=torch.float32,
    )
    if phase.device != global_value.device:
        raise SourceRelativeActionTargetError("transport tensors must share one device")
    if not bool(torch.isfinite(phase).all().item()) or not bool(
        torch.isfinite(global_value).all().item()
    ):
        raise SourceRelativeActionTargetError("transport values must be finite")
    _require_sha256(
        transport.draft_receipt_sha256,
        label="transport.draft_receipt_sha256",
    )
    _require_sha256(
        transport.codec_receipt_sha256,
        label="transport.codec_receipt_sha256",
    )
    _require_sha256(transport.transport_sha256, label="transport.transport_sha256")
    if transport.codec_receipt_sha256 != transport_receipt.receipt_sha256:
        raise SourceRelativeActionTargetError("transport codec receipt differs")
    payload = {
        "schema_version": transport.schema_version,
        "sample_ids": list(transport.sample_ids),
        "phase_code_sha256": _tensor_sha256(
            transport.phase_code, label="transport.phase_code"
        ),
        "global_code_sha256": _tensor_sha256(
            transport.global_code, label="transport.global_code"
        ),
        "draft_receipt_sha256": transport.draft_receipt_sha256,
        "codec_receipt_sha256": transport.codec_receipt_sha256,
    }
    if transport.transport_sha256 != _canonical_sha256(payload):
        raise SourceRelativeActionTargetError("transport bytes differ from receipt")


def source_relative_action_target_v2_contract() -> Mapping[str, Any]:
    """Return the fail-closed capability statement for this local ABI."""

    return {
        "schema_version": SCHEMA_VERSION,
        "phase_count": PHASE_COUNT,
        "actor_role_slots": ACTOR_ROLE_NAMES,
        "object_role_slots": OBJECT_ROLE_NAMES,
        "ownership_classes": OWNERSHIP_NAMES,
        "ownership_code_zero": "known_free_only_when_ownership_valid_is_true",
        "ownership_unknown_encoding": (
            "ownership_valid_false; stored_zero_is_padding_not_free"
        ),
        "phase_channels": PHASE_CHANNEL_NAMES,
        "abstain_reasons": ABSTAIN_REASON_NAMES,
        "camera_frame": "source_phase0_primary_actor_shared_for_source_and_target",
        "local_camera_stabilization": "exact_shared_rigid_transform_only",
        "independent_media_stabilization_requires_external_authority": True,
        "target_self_normalization_forbidden": True,
        "presence_is_visibility": False,
        "source_and_target_presence_validity_are_separate": True,
        "delta_valid_requires_both_present_endpoints": True,
        "target_position_is_state_not_displacement": True,
        "abstained_samples_zero_all_supervision": True,
        "qy_is_only_point_teacher_type": True,
        "qy_promotion_implemented": False,
        "local_draft_is_point_teacher": False,
        "legacy_r7_teacher_accepted": False,
        "random_lift_teacher_accepted": False,
        "q_anchor_type_exposed": False,
        "q_anchor_reconstruction_api_exists": False,
        "q_anchor_gate_api_exists": False,
        "reconstruction_loss_api_exists": False,
        "gate_api_exists": False,
        "optimizer_api_exists": False,
        "codec_role": "deterministic_lossless_schema_transport_only",
        "schema_transport_roundtrip_is_representation_evidence": False,
        "terminal_hold_delta_tolerance": TERMINAL_HOLD_DELTA_TOLERANCE,
        "pre_onset_delta_tolerance": PRE_ONSET_DELTA_TOLERANCE,
        "minimum_action_state_change": MIN_ACTION_STATE_CHANGE,
        "representation_qualification_evidence": False,
        "r2_evidence": False,
        "formally_qualified": False,
        "training_authorized": False,
        "optimizer_authorized": False,
        "selection_authorized": False,
        "gate_authorized": False,
        "external_authority_implemented": False,
        "camera_receipt_is_local_integrity_only": True,
        "local_draft_receipt_is_teacher_authority": False,
        "external_decoded_evaluator_required": True,
        "hard_blockers": (
            "external typed annotation and clean-pair authority receipt is not implemented",
            "background-only camera-stabilization provenance is not externally authenticated",
            "fit-only low-capacity encoder is not implemented",
            "frozen learned probes are not implemented",
            "locked held-out representation qualification is not implemented",
        ),
    }


__all__ = (
    "ABSTAIN_REASON_NAMES",
    "ACTOR_ROLE_NAMES",
    "ACTOR_SLOT_COUNT",
    "CanonicalizedSourceRelativeCoordinatesV2",
    "FrozenSchemaTransportReceiptV2",
    "FrozenSourceRelativeSchemaDecoderV2",
    "FrozenSourceRelativeSchemaEncoderV2",
    "FrozenSourceRelativeSchemaEvaluatorV2",
    "GLOBAL_TRANSPORT_WIDTH",
    "LocalSchemaTransportEvaluationV2",
    "OBJECT_ROLE_NAMES",
    "OBJECT_SLOT_COUNT",
    "ENTITY_SLOT_COUNT",
    "OWNER_CO_ACTOR",
    "OWNER_ENVIRONMENT",
    "OWNER_FREE",
    "OWNER_GOAL_CONTAINER",
    "OWNER_NONE",
    "OWNER_PRIMARY_ACTOR",
    "OWNERSHIP_NAMES",
    "PHASE_CHANNEL_NAMES",
    "PHASE_COUNT",
    "PHASE_TRANSPORT_WIDTH",
    "LocalSchemaTransportV2",
    "LocalSourceRelativeActionTargetDraftReceiptV2",
    "LocalSourceRelativeActionTargetDraftV2",
    "QYSourceRelativeActionTargetV2",
    "SourceRelativeActionAnnotationsV2",
    "SourceRelativeActionTargetError",
    "SourceRelativeActionTargetV2",
    "SourceRelativeCameraBundleV2",
    "SourceRelativeCameraCanonicalizationReceiptV2",
    "SourceRelativeCameraEvidenceV2",
    "bind_frozen_source_relative_schema_transport_v2",
    "build_local_source_relative_action_target_draft_v2",
    "build_source_relative_camera_bundle_v2",
    "source_relative_action_target_v2_contract",
    "validate_frozen_schema_transport_receipt_v2",
    "validate_local_schema_transport_evaluation_v2",
    "promote_local_draft_to_qy_source_relative_action_target_v2",
    "validate_local_schema_transport_v2",
    "validate_local_source_relative_action_target_draft_v2",
    "validate_qy_source_relative_action_target_v2",
    "validate_source_relative_action_target_v2",
    "validate_source_relative_camera_bundle_v2",
)
