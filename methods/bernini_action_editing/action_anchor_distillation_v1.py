#!/usr/bin/env python3
"""Local-only action-anchor distillation contract for Bernini.

This module binds externally materialized ``q_y`` and ``q_anchor`` codes to
the exact ``ActionPlanPredictorV1`` output ABI: 21 phase tokens of width 256
plus one 256-wide global token.  It binds ``q_pred`` to the same ABI, makes
``q_y`` the only point-distillation teacher, and routes reviewed anchors only
as contrastive positive/negative prototypes or exclusions.  It also supplies
an optional-PyTorch loss implementation.

It deliberately does *not* read media, extract optical flow or tracks, load a
visual action encoder, qualify an anchor, or authorize training.  Teacher
codes and their producer/classification authority hashes must be supplied by
an external, frozen, separately audited process.  There is no filesystem,
network, GPU, launcher, optimizer, or renderer integration in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
import struct
from typing import Any, Mapping, Sequence
import unicodedata

if __package__:
    from .action_plan_predictor_v1 import (
        ACTION_WIDTH,
        ARCHITECTURE_NAME,
        PHASE_COUNT,
        PREDICTOR_ABI_SCHEMA,
        ActionPlanOutput,
    )
else:  # Direct import from methods/bernini_action_editing.
    from action_plan_predictor_v1 import (  # type: ignore[no-redef]
        ACTION_WIDTH,
        ARCHITECTURE_NAME,
        PHASE_COUNT,
        PREDICTOR_ABI_SCHEMA,
        ActionPlanOutput,
    )


SCHEMA_VERSION = "bernini-action-anchor-distillation-contract-v1"
Q_RECEIPT_SCHEMA = "bernini-action-anchor-q-receipt-v1"
COMPATIBILITY_RECEIPT_SCHEMA = (
    "bernini-action-anchor-compatibility-receipt-v1"
)
TENSOR_HASH_SCHEMA = "bernini-action-plan-fp32-tensor-sha256-v1"
LOSS_SCHEMA = "bernini-action-anchor-distillation-loss-v1"
INTERVENTION_SCHEMA = "bernini-action-anchor-intervention-audit-v1"
MATERIALIZATION_RECEIPT_SCHEMA = "bernini-action-feature-teacher-v1"
MATERIALIZATION_SOURCE_SCHEMA = "motive-r7-event-temporal-teacher-v2"
MATERIALIZATION_PROJECTION_SCHEMA = "sha256-signed-random-lift-v1"
TEACHER_QUALIFICATION_RECEIPT_SCHEMA = (
    "bernini-action-teacher-external-qualification-receipt-v1"
)
TEACHER_QUALIFICATION_AUTHORITY_SCHEMA = (
    "bernini-action-teacher-qualification-authority-v1"
)

Q_KINDS = ("q_y", "q_anchor", "q_pred")
TEACHER_Q_KINDS = ("q_y", "q_anchor")
SEMANTIC_AXES = (
    "actor",
    "action",
    "object",
    "direction",
    "speed",
    "amplitude",
    "outcome",
)
HARD_NEGATIVE_KINDS = (
    "noop",
    "reverse",
    "incomplete",
    "wrong-actor",
    "wrong-object",
    "camera",
    "appearance",
)
CANDIDATE_KINDS = ("compatible",) + HARD_NEGATIVE_KINDS + ("unqualified",)
QUALIFICATION_VERDICTS = ("accept", "reject", "abstain")
TRAINING_USES = ("point-distill", "contrastive-only", "excluded")
DISTILLATION_ROLES = {
    "q_y": "unique-point-teacher",
    "q_anchor": "contrastive-prototype-only",
    "q_pred": "student-prediction",
}

EXTERNAL_TEACHER_PRODUCER = "external-frozen-action-feature-encoder"
PREDICTOR_PRODUCER = ARCHITECTURE_NAME
CANONICAL_DTYPE = "float32"
LOCAL_ONLY = True
IMPLEMENTS_VISUAL_TEACHER_EXTRACTION = False

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_Q_RECEIPT_FIELDS = {
    "schema_version",
    "q_kind",
    "plan_abi_schema",
    "layout",
    "items",
    "producer",
    "teacher_authority",
    "distillation_role",
    "teacher_stop_gradient",
    "phase_tensor_sha256",
    "global_tensor_sha256",
    "receipt_digest",
}
_LAYOUT_FIELDS = {
    "phase_count",
    "action_width",
    "batch_size",
    "phase_shape",
    "global_shape",
    "canonical_dtype",
    "tensor_hash_schema",
}
_Q_ITEM_INPUT_FIELDS = {
    "row_id",
    "source_sha256",
    "instruction_sha256",
    "endpoint_sha256",
    "semantics",
    "teacher_evidence",
}
_Q_ITEM_FIELDS = _Q_ITEM_INPUT_FIELDS | {
    "batch_index",
    "semantics_sha256",
}
_PRODUCER_FIELDS = {"kind", "artifact_sha256", "frozen"}
_TEACHER_EVIDENCE_FIELDS = {
    "materialization_receipt",
    "qualification_receipt",
}
_MATERIALIZATION_RECEIPT_FIELDS = {
    "schema_version",
    "role",
    "source_teacher_schema",
    "input_phases",
    "output_phases",
    "action_width",
    "phase_features",
    "global_features",
    "phase_weights",
    "projection",
    "action_embedding_sha256",
    "action_camera_sha256_audit_only",
    "action_upstream_authority_sha256",
    "baseline_mode",
    "baseline_embedding_sha256",
    "baseline_camera_sha256_audit_only",
    "baseline_upstream_authority_sha256",
    "action_event_duration",
    "action_event_normalized_start",
    "action_event_normalized_end",
    "baseline_event_duration",
    "baseline_event_normalized_start",
    "baseline_event_normalized_end",
    "delta_feature_sha256",
    "delta_feature_l2",
    "phase_tokens_sha256",
    "global_token_sha256",
    "camera_trajectory_excluded_from_tokens",
    "camera_invariance_claimed",
    "direct_rgb_or_latent_feature_input",
    "appearance_invariance_claimed",
    "actor_object_contact_geometry_in_tokens",
    "training_only_not_inference_input",
    "teacher_qualification_status",
    "point_distillation_authorized",
    "action_following_claimed",
    "receipt_sha256",
}
_QUALIFICATION_RECEIPT_FIELDS = {
    "schema_version",
    "materialization_receipt_sha256",
    "materialization_role",
    "phase_tokens_sha256",
    "global_token_sha256",
    "row_id",
    "source_sha256",
    "instruction_sha256",
    "endpoint_sha256",
    "semantics_sha256",
    "teacher_producer_sha256",
    "upstream_authority_manifest_sha256",
    "qualification_split_manifest_sha256",
    "qualification_protocol_sha256",
    "qualification_evaluator_sha256",
    "qualification_metrics_sha256",
    "qualification_authority_sha256",
    "independent_evaluator",
    "content_disjoint_holdout",
    "qualification_status",
    "point_distillation_authorized",
    "contrastive_authorized",
    "receipt_digest",
}
_TEACHER_AUTHORITY_FIELDS = {
    "schema_version",
    "teacher_producer_sha256",
    "upstream_authority_manifest_sha256",
    "qualification_split_manifest_sha256",
    "qualification_protocol_sha256",
    "qualification_evaluator_sha256",
    "qualification_metrics_sha256",
    "qualification_authority_sha256",
    "independent_evaluator",
    "content_disjoint_holdout",
    "authority_digest",
}
_MATERIALIZATION_PHASE_WEIGHTS = [
    1.0,
    1.0,
    0.25,
    0.25,
    0.019999999552965164,
    0.019999999552965164,
    0.10000000149011612,
    0.50,
    0.50,
    0.50,
    0.50,
    0.25,
]
_MATERIALIZATION_PHASE_PROJECTION_SHA256 = (
    "30ef308aefe27c77520e53c0a7e164a122f684c437c9214286508ebffdd2883a"
)
_MATERIALIZATION_GLOBAL_PROJECTION_SHA256 = (
    "040d69b405bafaa637e845ee03c60d54c0c14e0017fb9f91b3816ab25a6b301d"
)
_COMPATIBILITY_FIELDS = {
    "schema_version",
    "policy_sha256",
    "q_y_receipt_digest",
    "q_anchor_receipt_digest",
    "classification_authority_sha256",
    "items",
    "receipt_digest",
}
_COMPATIBILITY_ITEM_FIELDS = {
    "batch_index",
    "row_id",
    "candidate_kind",
    "qualification_verdict",
    "axis_matches",
    "mismatch_axes",
    "training_use",
    "contrastive_role",
}


class ActionAnchorDistillationError(RuntimeError):
    """Raised before an ambiguous action code can affect a loss."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the closed deterministic JSON encoding used by all receipts."""

    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ActionAnchorDistillationError(
            f"value is not canonical finite JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _exact_dict(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ActionAnchorDistillationError(f"{label} must be an exact dict")
    return value


def _closed_dict(
    value: Any, fields: set[str], *, label: str
) -> dict[str, Any]:
    row = _exact_dict(value, label=label)
    if set(row) != fields:
        raise ActionAnchorDistillationError(
            f"{label} field closure differs: "
            f"missing={sorted(fields - set(row))} extra={sorted(set(row) - fields)}"
        )
    return row


def _exact_list(value: Any, *, label: str) -> list[Any]:
    if type(value) is not list:
        raise ActionAnchorDistillationError(f"{label} must be an exact list")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ActionAnchorDistillationError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _authority_sha256(value: Any, *, label: str) -> str:
    digest = _sha256(value, label=label)
    if digest == "0" * 64:
        raise ActionAnchorDistillationError(
            f"{label} may not be an all-zero placeholder"
        )
    return digest


def _positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ActionAnchorDistillationError(
            f"{label} must be a positive exact integer"
        )
    return value


def _semantic_text(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or len(value.encode("utf-8")) > 1024
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise ActionAnchorDistillationError(
            f"{label} must be non-empty, NFC, boundary-trimmed text without control characters"
        )
    return value


def validate_action_semantics(value: Any) -> dict[str, str]:
    row = _closed_dict(value, set(SEMANTIC_AXES), label="action semantics")
    return {
        axis: _semantic_text(row[axis], label=f"action semantics {axis}")
        for axis in SEMANTIC_AXES
    }


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - host dependent
        raise ActionAnchorDistillationError(
            "PyTorch is required only for tensor hashing/loss/intervention operations"
        ) from error
    return torch


def _validate_plan(
    value: Any,
    *,
    label: str,
    batch_size: int | None = None,
) -> ActionPlanOutput:
    torch = _torch()
    if type(value) is not ActionPlanOutput:
        raise ActionAnchorDistillationError(
            f"{label} must be the exact ActionPlanOutput type"
        )
    phase = value.phase_tokens
    global_token = value.global_token
    if type(phase) is not torch.Tensor or type(global_token) is not torch.Tensor:
        raise ActionAnchorDistillationError(
            f"{label} members must be exact torch.Tensor objects"
        )
    if (
        phase.dtype != torch.float32
        or global_token.dtype != torch.float32
        or phase.layout != torch.strided
        or global_token.layout != torch.strided
        or not phase.is_contiguous()
        or not global_token.is_contiguous()
    ):
        raise ActionAnchorDistillationError(
            f"{label} must use contiguous strided FP32 tensors"
        )
    if (
        phase.ndim != 3
        or tuple(map(int, phase.shape[1:])) != (PHASE_COUNT, ACTION_WIDTH)
        or global_token.ndim != 2
        or tuple(map(int, global_token.shape[1:])) != (ACTION_WIDTH,)
        or int(phase.shape[0]) <= 0
        or int(global_token.shape[0]) != int(phase.shape[0])
    ):
        raise ActionAnchorDistillationError(
            f"{label} must be phase [B,{PHASE_COUNT},{ACTION_WIDTH}] plus global [B,{ACTION_WIDTH}]"
        )
    if batch_size is not None and int(phase.shape[0]) != batch_size:
        raise ActionAnchorDistillationError(f"{label} batch size differs")
    if phase.device != global_token.device:
        raise ActionAnchorDistillationError(
            f"{label} phase/global tensors must share one device"
        )
    if not bool(torch.isfinite(phase.detach()).all().item()) or not bool(
        torch.isfinite(global_token.detach()).all().item()
    ):
        raise ActionAnchorDistillationError(f"{label} contains non-finite values")
    return value


def tensor_sha256_v1(value: Any) -> str:
    """Hash one finite contiguous FP32 tensor in a platform-stable format."""

    torch = _torch()
    if (
        type(value) is not torch.Tensor
        or value.dtype != torch.float32
        or value.layout != torch.strided
        or not value.is_contiguous()
        or value.ndim <= 0
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise ActionAnchorDistillationError(
            "tensor hash input must be a finite contiguous strided FP32 tensor"
        )
    cpu = value.detach().cpu().contiguous().reshape(-1)
    digest = hashlib.sha256()
    header = {
        "schema_version": TENSOR_HASH_SCHEMA,
        "dtype": CANONICAL_DTYPE,
        "shape": list(map(int, value.shape)),
    }
    digest.update(canonical_json_bytes(header))
    digest.update(b"\x00")
    # Explicit little-endian packing avoids NumPy and host-endian dependence.
    chunk_size = 4096
    for start in range(0, int(cpu.numel()), chunk_size):
        values = cpu[start : start + chunk_size].tolist()
        digest.update(struct.pack(f"<{len(values)}f", *values))
    return digest.hexdigest()


def _raw_fp32_tensor_sha256(value: Any) -> str:
    """Hash only little-endian FP32 payload bytes (materializer-compatible)."""

    torch = _torch()
    if (
        type(value) is not torch.Tensor
        or value.dtype != torch.float32
        or value.layout != torch.strided
        or not value.is_contiguous()
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise ActionAnchorDistillationError(
            "raw tensor hash input must be finite contiguous strided FP32"
        )
    cpu = value.detach().cpu().contiguous().reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, int(cpu.numel()), 4096):
        values = cpu[start : start + 4096].tolist()
        digest.update(struct.pack(f"<{len(values)}f", *values))
    return digest.hexdigest()


def _validate_materialization_receipt(
    value: Any, *, q_kind: str
) -> dict[str, Any]:
    receipt = _closed_dict(
        value,
        _MATERIALIZATION_RECEIPT_FIELDS,
        label="teacher materialization receipt",
    )
    expected_role = "target" if q_kind == "q_y" else "anchor"
    if (
        receipt["schema_version"] != MATERIALIZATION_RECEIPT_SCHEMA
        or receipt["role"] != expected_role
        or receipt["source_teacher_schema"] != MATERIALIZATION_SOURCE_SCHEMA
        or type(receipt["input_phases"]) is not int
        or receipt["input_phases"] != 32
        or type(receipt["output_phases"]) is not int
        or receipt["output_phases"] != PHASE_COUNT
        or type(receipt["action_width"]) is not int
        or receipt["action_width"] != ACTION_WIDTH
        or type(receipt["phase_features"]) is not int
        or receipt["phase_features"] != 12
        or type(receipt["global_features"]) is not int
        or receipt["global_features"] != 37
    ):
        raise ActionAnchorDistillationError(
            "teacher materialization schema/role/geometry differs"
        )
    if (
        type(receipt["phase_weights"]) is not list
        or any(type(item) is not float for item in receipt["phase_weights"])
        or receipt["phase_weights"] != _MATERIALIZATION_PHASE_WEIGHTS
    ):
        raise ActionAnchorDistillationError(
            "teacher materialization phase weights differ"
        )
    projection = _closed_dict(
        receipt["projection"],
        {"schema", "phase_sha256", "global_sha256"},
        label="teacher materialization projection",
    )
    if (
        projection["schema"] != MATERIALIZATION_PROJECTION_SCHEMA
        or projection["phase_sha256"]
        != _MATERIALIZATION_PHASE_PROJECTION_SHA256
        or projection["global_sha256"]
        != _MATERIALIZATION_GLOBAL_PROJECTION_SHA256
    ):
        raise ActionAnchorDistillationError(
            "teacher materialization projection differs"
        )
    hash_fields = (
        "phase_tokens_sha256",
        "global_token_sha256",
        "receipt_sha256",
    )
    for name in hash_fields:
        _sha256(receipt[name], label=f"teacher materialization {name}")
    for name in (
        "action_embedding_sha256",
        "action_camera_sha256_audit_only",
        "action_upstream_authority_sha256",
        "baseline_upstream_authority_sha256",
        "delta_feature_sha256",
    ):
        _authority_sha256(
            receipt[name], label=f"teacher materialization {name}"
        )
    mode = receipt["baseline_mode"]
    if mode == "externally_verified_static_noop":
        if (
            receipt["baseline_embedding_sha256"] is not None
            or receipt["baseline_camera_sha256_audit_only"] is not None
            or receipt["baseline_event_normalized_start"] is not None
            or receipt["baseline_event_normalized_end"] is not None
        ):
            raise ActionAnchorDistillationError(
                "static materialization baseline fields differ"
            )
    elif mode == "explicit_temporal_teacher":
        for name in (
            "baseline_embedding_sha256",
            "baseline_camera_sha256_audit_only",
        ):
            _authority_sha256(
                receipt[name], label=f"teacher materialization {name}"
            )
        for name in (
            "baseline_event_normalized_start",
            "baseline_event_normalized_end",
        ):
            if type(receipt[name]) is not float or not math.isfinite(receipt[name]):
                raise ActionAnchorDistillationError(
                    "explicit materialization baseline event differs"
                )
    else:
        raise ActionAnchorDistillationError(
            "teacher materialization baseline mode differs"
        )
    for name in (
        "action_event_duration",
        "baseline_event_duration",
        "delta_feature_l2",
    ):
        if (
            type(receipt[name]) is not float
            or not math.isfinite(receipt[name])
            or receipt[name] <= 0.0
        ):
            raise ActionAnchorDistillationError(
                f"teacher materialization {name} differs"
            )
    for name in (
        "action_event_normalized_start",
        "action_event_normalized_end",
    ):
        if type(receipt[name]) is not float or not math.isfinite(receipt[name]):
            raise ActionAnchorDistillationError(
                "teacher materialization action event differs"
            )
    if not (
        0.0
        <= receipt["action_event_normalized_start"]
        < receipt["action_event_normalized_end"]
        <= 1.0
    ):
        raise ActionAnchorDistillationError(
            "teacher materialization action event window differs"
        )
    if mode == "explicit_temporal_teacher" and not (
        0.0
        <= receipt["baseline_event_normalized_start"]
        < receipt["baseline_event_normalized_end"]
        <= 1.0
    ):
        raise ActionAnchorDistillationError(
            "teacher materialization baseline event window differs"
        )
    exact_flags = {
        "camera_trajectory_excluded_from_tokens": True,
        "camera_invariance_claimed": False,
        "direct_rgb_or_latent_feature_input": False,
        "appearance_invariance_claimed": False,
        "actor_object_contact_geometry_in_tokens": False,
        "training_only_not_inference_input": True,
        "point_distillation_authorized": False,
        "action_following_claimed": False,
    }
    if any(receipt[name] is not expected for name, expected in exact_flags.items()):
        raise ActionAnchorDistillationError(
            "teacher materialization safety flags differ"
        )
    if receipt["teacher_qualification_status"] != "candidate_unqualified":
        raise ActionAnchorDistillationError(
            "teacher materialization may not self-authorize qualification"
        )
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256")
    if object_sha256(unsigned) != claimed:
        raise ActionAnchorDistillationError(
            "teacher materialization receipt digest differs"
        )
    return json.loads(canonical_json_bytes(receipt).decode("ascii"))


def _validate_teacher_authority(
    value: Any, *, expected_sha256: str
) -> dict[str, Any]:
    """Validate the caller-pinned trust root for teacher qualification.

    The authority is intentionally external to every materialization and
    qualification receipt.  A self-consistent rewritten receipt tree is not
    sufficient: its canonical authority digest must equal the value pinned by
    the row/launch manifest and supplied by the caller.
    """

    expected = _authority_sha256(
        expected_sha256, label="expected teacher qualification authority SHA-256"
    )
    authority = _closed_dict(
        value,
        _TEACHER_AUTHORITY_FIELDS,
        label="teacher qualification authority",
    )
    if authority["schema_version"] != TEACHER_QUALIFICATION_AUTHORITY_SCHEMA:
        raise ActionAnchorDistillationError(
            "teacher qualification authority schema differs"
        )
    for name in (
        "teacher_producer_sha256",
        "upstream_authority_manifest_sha256",
        "qualification_split_manifest_sha256",
        "qualification_protocol_sha256",
        "qualification_evaluator_sha256",
        "qualification_metrics_sha256",
        "qualification_authority_sha256",
        "authority_digest",
    ):
        _authority_sha256(authority[name], label=f"teacher authority {name}")
    if (
        authority["qualification_evaluator_sha256"]
        == authority["teacher_producer_sha256"]
    ):
        raise ActionAnchorDistillationError(
            "teacher authority evaluator is not independent"
        )
    for name in ("independent_evaluator", "content_disjoint_holdout"):
        if type(authority[name]) is not bool or authority[name] is not True:
            raise ActionAnchorDistillationError(
                f"teacher authority {name} gate differs"
            )
    unsigned = dict(authority)
    declared = unsigned.pop("authority_digest")
    observed = object_sha256(unsigned)
    if declared != observed or observed != expected:
        raise ActionAnchorDistillationError(
            "teacher qualification authority is not the externally pinned authority"
        )
    return json.loads(canonical_json_bytes(authority).decode("ascii"))


def _validate_qualification_receipt(
    value: Any,
    *,
    q_kind: str,
    materialization: Mapping[str, Any],
    binding: Mapping[str, Any],
    authority: Mapping[str, Any],
    expected_receipt_digest: str,
) -> dict[str, Any]:
    receipt = _closed_dict(
        value,
        _QUALIFICATION_RECEIPT_FIELDS,
        label="teacher qualification receipt",
    )
    expected_role = "target" if q_kind == "q_y" else "anchor"
    if (
        receipt["schema_version"] != TEACHER_QUALIFICATION_RECEIPT_SCHEMA
        or receipt["materialization_role"] != expected_role
        or receipt["materialization_receipt_sha256"]
        != materialization["receipt_sha256"]
        or receipt["phase_tokens_sha256"]
        != materialization["phase_tokens_sha256"]
        or receipt["global_token_sha256"]
        != materialization["global_token_sha256"]
    ):
        raise ActionAnchorDistillationError(
            "teacher qualification/materialization binding differs"
        )
    for name in (
        "row_id",
        "source_sha256",
        "instruction_sha256",
        "endpoint_sha256",
        "semantics_sha256",
    ):
        if receipt[name] != binding[name]:
            raise ActionAnchorDistillationError(
                f"teacher qualification {name} binding differs"
            )
    for name in (
        "teacher_producer_sha256",
        "upstream_authority_manifest_sha256",
        "qualification_split_manifest_sha256",
        "qualification_protocol_sha256",
        "qualification_evaluator_sha256",
        "qualification_metrics_sha256",
        "qualification_authority_sha256",
        "receipt_digest",
    ):
        _sha256(receipt[name], label=f"teacher qualification {name}")
    authority_bindings = (
        "teacher_producer_sha256",
        "upstream_authority_manifest_sha256",
        "qualification_split_manifest_sha256",
        "qualification_protocol_sha256",
        "qualification_evaluator_sha256",
        "qualification_metrics_sha256",
        "qualification_authority_sha256",
    )
    for name in authority_bindings:
        if receipt[name] != authority[name]:
            raise ActionAnchorDistillationError(
                f"teacher qualification {name} differs from external authority"
            )
    if receipt["qualification_evaluator_sha256"] == receipt["teacher_producer_sha256"]:
        raise ActionAnchorDistillationError(
            "teacher qualification evaluator is not independent"
        )
    for name in ("independent_evaluator", "content_disjoint_holdout"):
        if type(receipt[name]) is not bool or receipt[name] is not True:
            raise ActionAnchorDistillationError(
                f"teacher qualification {name} gate differs"
            )
    point = receipt["point_distillation_authorized"]
    contrastive = receipt["contrastive_authorized"]
    if type(point) is not bool or type(contrastive) is not bool:
        raise ActionAnchorDistillationError(
            "teacher qualification authorization flags differ"
        )
    status = receipt["qualification_status"]
    if q_kind == "q_y":
        if status != "qualified" or point is not True or contrastive is not True:
            raise ActionAnchorDistillationError(
                "q_y requires external qualified point-distillation authority"
            )
    elif q_kind == "q_anchor":
        if point is not False or status not in (
            "qualified",
            "candidate_unqualified",
            "rejected",
        ):
            raise ActionAnchorDistillationError(
                "q_anchor qualification status/point policy differs"
            )
        if contrastive is not (status == "qualified"):
            raise ActionAnchorDistillationError(
                "q_anchor contrastive authority differs from qualification status"
            )
    else:  # pragma: no cover - internal caller contract
        raise ActionAnchorDistillationError("teacher qualification q kind differs")
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest")
    if object_sha256(unsigned) != declared:
        raise ActionAnchorDistillationError(
            "teacher qualification receipt digest differs"
        )
    expected = _authority_sha256(
        expected_receipt_digest,
        label="externally pinned teacher qualification receipt digest",
    )
    if declared != expected:
        raise ActionAnchorDistillationError(
            "teacher qualification receipt is not pinned by the row authority"
        )
    return json.loads(canonical_json_bytes(receipt).decode("ascii"))


def _validate_teacher_evidence(
    value: Any,
    *,
    q_kind: str,
    binding: Mapping[str, Any],
    authority: Mapping[str, Any],
    expected_qualification_receipt_digest: str,
) -> dict[str, Any]:
    evidence = _closed_dict(
        value,
        _TEACHER_EVIDENCE_FIELDS,
        label="teacher evidence",
    )
    materialization = _validate_materialization_receipt(
        evidence["materialization_receipt"], q_kind=q_kind
    )
    qualification = _validate_qualification_receipt(
        evidence["qualification_receipt"],
        q_kind=q_kind,
        materialization=materialization,
        binding=binding,
        authority=authority,
        expected_receipt_digest=expected_qualification_receipt_digest,
    )
    return {
        "materialization_receipt": materialization,
        "qualification_receipt": qualification,
    }


def _validate_binding_input(
    value: Any,
    *,
    q_kind: str,
    batch_index: int,
    teacher_authority: Mapping[str, Any] | None,
    expected_qualification_receipt_digest: str | None,
) -> dict[str, Any]:
    row = _closed_dict(
        value, _Q_ITEM_INPUT_FIELDS, label=f"q binding[{batch_index}]"
    )
    endpoint = row["endpoint_sha256"]
    if q_kind == "q_pred":
        if endpoint is not None:
            raise ActionAnchorDistillationError(
                "q_pred binding must not claim a target/anchor endpoint"
            )
    else:
        endpoint = _authority_sha256(
            endpoint, label=f"q binding[{batch_index}] endpoint SHA-256"
        )
    semantics = validate_action_semantics(row["semantics"])
    normalized = {
        "batch_index": batch_index,
        "row_id": _authority_sha256(
            row["row_id"], label=f"q binding[{batch_index}] row ID"
        ),
        "source_sha256": _authority_sha256(
            row["source_sha256"], label=f"q binding[{batch_index}] source SHA-256"
        ),
        "instruction_sha256": _authority_sha256(
            row["instruction_sha256"],
            label=f"q binding[{batch_index}] instruction SHA-256",
        ),
        "endpoint_sha256": endpoint,
        "semantics": semantics,
        "semantics_sha256": object_sha256(semantics),
    }
    evidence = row["teacher_evidence"]
    if q_kind == "q_pred":
        if evidence is not None or expected_qualification_receipt_digest is not None:
            raise ActionAnchorDistillationError(
                "q_pred binding must not carry teacher evidence or qualification pins"
            )
        normalized["teacher_evidence"] = None
    else:
        if (
            teacher_authority is None
            or expected_qualification_receipt_digest is None
        ):
            raise ActionAnchorDistillationError(
                "teacher binding requires authority and a pinned qualification receipt"
            )
        normalized["teacher_evidence"] = _validate_teacher_evidence(
            evidence,
            q_kind=q_kind,
            binding=normalized,
            authority=teacher_authority,
            expected_qualification_receipt_digest=
            expected_qualification_receipt_digest,
        )
    return normalized


def _validate_materialized_tensor_bindings(
    items: Sequence[Mapping[str, Any]], plan: ActionPlanOutput
) -> None:
    """Bind each qualified item to the exact materialized tensor payload."""

    for index, item in enumerate(items):
        materialization = item["teacher_evidence"]["materialization_receipt"]
        if (
            _raw_fp32_tensor_sha256(plan.phase_tokens[index])
            != materialization["phase_tokens_sha256"]
            or _raw_fp32_tensor_sha256(plan.global_token[index])
            != materialization["global_token_sha256"]
        ):
            raise ActionAnchorDistillationError(
                f"teacher item[{index}] tensors differ from materialization receipt"
            )


def build_q_receipt_v1(
    *,
    q_kind: str,
    plan: ActionPlanOutput,
    bindings: Sequence[Mapping[str, Any]],
    producer_artifact_sha256: str,
    teacher_authority: Mapping[str, Any] | None = None,
    expected_teacher_authority_sha256: str | None = None,
    expected_qualification_receipt_digests: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a hash-bound batch receipt for q_y, q_anchor, or q_pred."""

    if type(q_kind) is not str or q_kind not in Q_KINDS:
        raise ActionAnchorDistillationError("q kind differs")
    checked_plan = _validate_plan(plan, label=q_kind)
    if type(bindings) not in (list, tuple):
        raise ActionAnchorDistillationError("q bindings must be an exact list or tuple")
    batch_size = int(checked_plan.phase_tokens.shape[0])
    if len(bindings) != batch_size:
        raise ActionAnchorDistillationError("q binding count differs from tensor batch")
    is_teacher = q_kind in TEACHER_Q_KINDS
    if is_teacher:
        if expected_teacher_authority_sha256 is None:
            raise ActionAnchorDistillationError(
                "teacher q build requires an externally pinned authority SHA-256"
            )
        checked_authority: dict[str, Any] | None = _validate_teacher_authority(
            teacher_authority,
            expected_sha256=expected_teacher_authority_sha256,
        )
        expected_qualification_digests = _exact_list(
            expected_qualification_receipt_digests,
            label="expected qualification receipt digests",
        )
        if len(expected_qualification_digests) != batch_size:
            raise ActionAnchorDistillationError(
                "expected qualification receipt coverage differs"
            )
        for index, digest in enumerate(expected_qualification_digests):
            _sha256(digest, label=f"expected qualification receipt[{index}] digest")
    else:
        if (
            teacher_authority is not None
            or expected_teacher_authority_sha256 is not None
            or expected_qualification_receipt_digests is not None
        ):
            raise ActionAnchorDistillationError(
                "q_pred build must not carry teacher qualification authority"
            )
        checked_authority = None
        expected_qualification_digests = [None] * batch_size
    items = [
        _validate_binding_input(
            item,
            q_kind=q_kind,
            batch_index=index,
            teacher_authority=checked_authority,
            expected_qualification_receipt_digest=
            expected_qualification_digests[index],
        )
        for index, item in enumerate(bindings)
    ]
    producer_sha = _authority_sha256(
        producer_artifact_sha256, label="q producer artifact SHA-256"
    )
    if is_teacher:
        if producer_sha != checked_authority["teacher_producer_sha256"]:
            raise ActionAnchorDistillationError(
                "teacher q producer differs from external authority"
            )
        _validate_materialized_tensor_bindings(items, checked_plan)
    unsigned = {
        "schema_version": Q_RECEIPT_SCHEMA,
        "q_kind": q_kind,
        "plan_abi_schema": PREDICTOR_ABI_SCHEMA,
        "layout": {
            "phase_count": PHASE_COUNT,
            "action_width": ACTION_WIDTH,
            "batch_size": batch_size,
            "phase_shape": list(map(int, checked_plan.phase_tokens.shape)),
            "global_shape": list(map(int, checked_plan.global_token.shape)),
            "canonical_dtype": CANONICAL_DTYPE,
            "tensor_hash_schema": TENSOR_HASH_SCHEMA,
        },
        "items": items,
        "producer": {
            "kind": EXTERNAL_TEACHER_PRODUCER if is_teacher else PREDICTOR_PRODUCER,
            "artifact_sha256": producer_sha,
            "frozen": is_teacher,
        },
        "teacher_authority": checked_authority,
        "distillation_role": DISTILLATION_ROLES[q_kind],
        "teacher_stop_gradient": is_teacher,
        "phase_tensor_sha256": tensor_sha256_v1(checked_plan.phase_tokens),
        "global_tensor_sha256": tensor_sha256_v1(checked_plan.global_token),
    }
    receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    return validate_q_receipt_v1(
        receipt,
        plan=checked_plan,
        expected_teacher_authority_sha256=expected_teacher_authority_sha256,
        expected_qualification_receipt_digests=(
            expected_qualification_digests if is_teacher else None
        ),
    )


def _validate_q_item(
    value: Any,
    *,
    q_kind: str,
    batch_index: int,
    teacher_authority: Mapping[str, Any] | None,
    expected_qualification_receipt_digest: str | None,
) -> dict[str, Any]:
    row = _closed_dict(value, _Q_ITEM_FIELDS, label=f"q receipt item[{batch_index}]")
    if type(row["batch_index"]) is not int or row["batch_index"] != batch_index:
        raise ActionAnchorDistillationError("q receipt batch indices are not exact")
    binding = {
        key: row[key]
        for key in _Q_ITEM_INPUT_FIELDS
    }
    normalized = _validate_binding_input(
        binding,
        q_kind=q_kind,
        batch_index=batch_index,
        teacher_authority=teacher_authority,
        expected_qualification_receipt_digest=
        expected_qualification_receipt_digest,
    )
    declared_semantics_sha = _sha256(
        row["semantics_sha256"],
        label=f"q receipt item[{batch_index}] semantics SHA-256",
    )
    if declared_semantics_sha != normalized["semantics_sha256"]:
        raise ActionAnchorDistillationError("q semantics SHA-256 differs")
    return normalized


def validate_q_receipt_v1(
    value: Any,
    *,
    plan: ActionPlanOutput | None = None,
    expected_teacher_authority_sha256: str | None = None,
    expected_qualification_receipt_digests: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate field closure, roles, self-digest, and optionally tensor hashes."""

    receipt = _closed_dict(value, _Q_RECEIPT_FIELDS, label="q receipt")
    if receipt["schema_version"] != Q_RECEIPT_SCHEMA:
        raise ActionAnchorDistillationError("q receipt schema differs")
    q_kind = receipt["q_kind"]
    if type(q_kind) is not str or q_kind not in Q_KINDS:
        raise ActionAnchorDistillationError("q receipt kind differs")
    if receipt["plan_abi_schema"] != PREDICTOR_ABI_SCHEMA:
        raise ActionAnchorDistillationError("q receipt plan ABI differs")
    layout = _closed_dict(receipt["layout"], _LAYOUT_FIELDS, label="q layout")
    if (
        type(layout["phase_count"]) is not int
        or layout["phase_count"] != PHASE_COUNT
        or type(layout["action_width"]) is not int
        or layout["action_width"] != ACTION_WIDTH
        or layout["canonical_dtype"] != CANONICAL_DTYPE
        or layout["tensor_hash_schema"] != TENSOR_HASH_SCHEMA
    ):
        raise ActionAnchorDistillationError("q receipt layout ABI differs")
    batch_size = _positive_int(layout["batch_size"], label="q layout batch size")
    phase_shape = _exact_list(layout["phase_shape"], label="q phase shape")
    global_shape = _exact_list(layout["global_shape"], label="q global shape")
    if (
        any(type(item) is not int for item in phase_shape + global_shape)
        or phase_shape != [batch_size, PHASE_COUNT, ACTION_WIDTH]
        or global_shape != [batch_size, ACTION_WIDTH]
    ):
        raise ActionAnchorDistillationError("q receipt tensor shapes differ")
    is_teacher = q_kind in TEACHER_Q_KINDS
    raw_authority = receipt["teacher_authority"]
    if is_teacher:
        if expected_teacher_authority_sha256 is None:
            raise ActionAnchorDistillationError(
                "teacher q validation requires an externally pinned authority SHA-256"
            )
        teacher_authority: dict[str, Any] | None = _validate_teacher_authority(
            raw_authority,
            expected_sha256=expected_teacher_authority_sha256,
        )
        raw_expected_qualification_digests = _exact_list(
            expected_qualification_receipt_digests,
            label="expected qualification receipt digests",
        )
        if len(raw_expected_qualification_digests) != batch_size:
            raise ActionAnchorDistillationError(
                "expected qualification receipt coverage differs"
            )
        expected_qualification_digests: list[str | None] = []
        for index, digest in enumerate(raw_expected_qualification_digests):
            expected_qualification_digests.append(
                _sha256(
                    digest,
                    label=f"expected qualification receipt[{index}] digest",
                )
            )
    else:
        if raw_authority is not None or expected_qualification_receipt_digests is not None:
            raise ActionAnchorDistillationError(
                "q_pred receipt must not carry teacher authority or qualification pins"
            )
        teacher_authority = None
        expected_qualification_digests = [None] * batch_size
    raw_items = _exact_list(receipt["items"], label="q receipt items")
    if len(raw_items) != batch_size:
        raise ActionAnchorDistillationError("q receipt item count differs")
    items = [
        _validate_q_item(
            item,
            q_kind=q_kind,
            batch_index=index,
            teacher_authority=teacher_authority,
            expected_qualification_receipt_digest=
            expected_qualification_digests[index],
        )
        for index, item in enumerate(raw_items)
    ]
    producer = _closed_dict(receipt["producer"], _PRODUCER_FIELDS, label="q producer")
    _authority_sha256(
        producer["artifact_sha256"], label="q producer artifact SHA-256"
    )
    expected_producer = EXTERNAL_TEACHER_PRODUCER if is_teacher else PREDICTOR_PRODUCER
    if (
        producer["kind"] != expected_producer
        or type(producer["frozen"]) is not bool
        or producer["frozen"] is not is_teacher
        or type(receipt["teacher_stop_gradient"]) is not bool
        or receipt["teacher_stop_gradient"] is not is_teacher
        or receipt["distillation_role"] != DISTILLATION_ROLES[q_kind]
    ):
        raise ActionAnchorDistillationError("q producer/autograd role differs")
    if is_teacher and (
        producer["artifact_sha256"]
        != teacher_authority["teacher_producer_sha256"]
    ):
        raise ActionAnchorDistillationError(
            "teacher q producer differs from external authority"
        )
    _sha256(receipt["phase_tensor_sha256"], label="q phase tensor SHA-256")
    _sha256(receipt["global_tensor_sha256"], label="q global tensor SHA-256")
    declared_digest = _sha256(receipt["receipt_digest"], label="q receipt digest")
    unsigned = dict(receipt)
    del unsigned["receipt_digest"]
    if object_sha256(unsigned) != declared_digest:
        raise ActionAnchorDistillationError("q receipt self-digest differs")
    if plan is not None:
        checked_plan = _validate_plan(plan, label=q_kind, batch_size=batch_size)
        if (
            tensor_sha256_v1(checked_plan.phase_tokens)
            != receipt["phase_tensor_sha256"]
            or tensor_sha256_v1(checked_plan.global_token)
            != receipt["global_tensor_sha256"]
        ):
            raise ActionAnchorDistillationError("q receipt tensor hash differs")
        if is_teacher:
            _validate_materialized_tensor_bindings(items, checked_plan)
    # Return a fresh JSON-shaped value, never caller-owned nested containers.
    return json.loads(canonical_json_bytes(receipt).decode("ascii"))


def _axis_matches(
    desired: Mapping[str, str], candidate: Mapping[str, str]
) -> dict[str, bool]:
    return {axis: candidate[axis] == desired[axis] for axis in SEMANTIC_AXES}


def _validate_candidate_relation(
    *,
    candidate_kind: str,
    desired: Mapping[str, str],
    candidate: Mapping[str, str],
    matches: Mapping[str, bool],
) -> None:
    mismatch = tuple(axis for axis in SEMANTIC_AXES if not matches[axis])
    expected_mismatch = {
        "compatible": (),
        "noop": ("action",),
        "reverse": ("direction",),
        "incomplete": ("outcome",),
        "wrong-actor": ("actor",),
        "wrong-object": ("object",),
        "camera": (),
        "appearance": (),
    }[candidate_kind]
    if mismatch != expected_mismatch:
        raise ActionAnchorDistillationError(
            f"{candidate_kind} semantic mismatch axes differ: {mismatch} != {expected_mismatch}"
        )
    if candidate_kind == "noop" and candidate["action"] != "noop":
        raise ActionAnchorDistillationError("noop hard negative must use action=noop")
    if candidate_kind == "incomplete" and candidate["outcome"] != "incomplete":
        raise ActionAnchorDistillationError(
            "incomplete hard negative must use outcome=incomplete"
        )
    if candidate_kind == "noop" and desired["action"] == "noop":
        raise ActionAnchorDistillationError("noop cannot be negative for a noop target")


def _compatibility_policy_body() -> dict[str, Any]:
    return {
        "schema_version": "bernini-action-anchor-compatibility-policy-v1",
        "semantic_axes": list(SEMANTIC_AXES),
        "candidate_kinds": list(CANDIDATE_KINDS),
        "hard_negative_kinds": list(HARD_NEGATIVE_KINDS),
        "qualification_verdicts": list(QUALIFICATION_VERDICTS),
        "training_uses": list(TRAINING_USES),
        "q_y_default_use": "point-distill",
        "accepted_compatible_anchor_use": "contrastive-only/positive",
        "accepted_hard_negative_use": "contrastive-only",
        "unqualified_use": "excluded",
        "q_anchor_point_distillation_enabled": False,
        "future_phase_alignment_authority_implemented": False,
        "camera_and_appearance_are_authority_classified_not_inferred": True,
    }


# Pinned below after canonical hashing; the import-time assertion prevents a
# silent policy change without an explicit version/hash update.
COMPATIBILITY_POLICY_SHA256 = (
    "80a8e4c84c93ce8b1c5a65177246273bd4325c34ad10d8750dd26cb7872747ce"
)


def build_compatibility_receipt_v1(
    *,
    q_y_receipt: Mapping[str, Any],
    q_anchor_receipt: Mapping[str, Any],
    candidate_kinds: Sequence[str],
    qualification_verdicts: Sequence[str],
    classification_authority_sha256: str,
    expected_teacher_authority_sha256: str,
    expected_q_y_qualification_receipt_digests: Sequence[str],
    expected_q_anchor_qualification_receipt_digests: Sequence[str],
    expected_decision_receipt_digest: str,
) -> dict[str, Any]:
    """Bind semantic compatibility and a fail-closed training route."""

    q_y = validate_q_receipt_v1(
        q_y_receipt,
        expected_teacher_authority_sha256=expected_teacher_authority_sha256,
        expected_qualification_receipt_digests=
        expected_q_y_qualification_receipt_digests,
    )
    q_anchor = validate_q_receipt_v1(
        q_anchor_receipt,
        expected_teacher_authority_sha256=expected_teacher_authority_sha256,
        expected_qualification_receipt_digests=
        expected_q_anchor_qualification_receipt_digests,
    )
    if q_y["q_kind"] != "q_y" or q_anchor["q_kind"] != "q_anchor":
        raise ActionAnchorDistillationError(
            "compatibility requires q_y and q_anchor receipts"
        )
    if q_y["layout"] != q_anchor["layout"]:
        raise ActionAnchorDistillationError("q_y/q_anchor layouts differ")
    if q_y["producer"] != q_anchor["producer"]:
        raise ActionAnchorDistillationError(
            "q_y/q_anchor must share one frozen teacher artifact"
        )
    if type(candidate_kinds) not in (list, tuple) or type(
        qualification_verdicts
    ) not in (list, tuple):
        raise ActionAnchorDistillationError(
            "candidate kinds/verdicts must be exact lists or tuples"
        )
    authority = _authority_sha256(
        classification_authority_sha256,
        label="compatibility classification authority SHA-256",
    )
    receipt = _build_compatibility_without_recursive_validation(
        q_y=q_y,
        q_anchor=q_anchor,
        candidate_kinds=candidate_kinds,
        qualification_verdicts=qualification_verdicts,
        authority=authority,
    )
    expected_decision = _authority_sha256(
        expected_decision_receipt_digest,
        label="externally pinned compatibility decision receipt digest",
    )
    if receipt["receipt_digest"] != expected_decision:
        raise ActionAnchorDistillationError(
            "compatibility decision is not pinned by the row authority"
        )
    return validate_compatibility_receipt_v1(
        receipt,
        q_y_receipt=q_y,
        q_anchor_receipt=q_anchor,
        expected_teacher_authority_sha256=expected_teacher_authority_sha256,
        expected_classification_authority_sha256=authority,
        expected_q_y_qualification_receipt_digests=
        expected_q_y_qualification_receipt_digests,
        expected_q_anchor_qualification_receipt_digests=
        expected_q_anchor_qualification_receipt_digests,
        expected_decision_receipt_digest=expected_decision,
    )


def validate_compatibility_receipt_v1(
    value: Any,
    *,
    q_y_receipt: Mapping[str, Any],
    q_anchor_receipt: Mapping[str, Any],
    expected_teacher_authority_sha256: str,
    expected_classification_authority_sha256: str,
    expected_q_y_qualification_receipt_digests: Sequence[str],
    expected_q_anchor_qualification_receipt_digests: Sequence[str],
    expected_decision_receipt_digest: str,
) -> dict[str, Any]:
    receipt = _closed_dict(
        value, _COMPATIBILITY_FIELDS, label="compatibility receipt"
    )
    if receipt["schema_version"] != COMPATIBILITY_RECEIPT_SCHEMA:
        raise ActionAnchorDistillationError(
            "compatibility receipt schema differs"
        )
    if receipt["policy_sha256"] != COMPATIBILITY_POLICY_SHA256:
        raise ActionAnchorDistillationError("compatibility policy hash differs")
    q_y = validate_q_receipt_v1(
        q_y_receipt,
        expected_teacher_authority_sha256=expected_teacher_authority_sha256,
        expected_qualification_receipt_digests=
        expected_q_y_qualification_receipt_digests,
    )
    q_anchor = validate_q_receipt_v1(
        q_anchor_receipt,
        expected_teacher_authority_sha256=expected_teacher_authority_sha256,
        expected_qualification_receipt_digests=
        expected_q_anchor_qualification_receipt_digests,
    )
    if (
        receipt["q_y_receipt_digest"] != q_y["receipt_digest"]
        or receipt["q_anchor_receipt_digest"] != q_anchor["receipt_digest"]
    ):
        raise ActionAnchorDistillationError(
            "compatibility receipt q binding differs"
        )
    authority = _authority_sha256(
        receipt["classification_authority_sha256"],
        label="compatibility classification authority SHA-256",
    )
    expected_classification = _authority_sha256(
        expected_classification_authority_sha256,
        label="expected compatibility classification authority SHA-256",
    )
    if authority != expected_classification:
        raise ActionAnchorDistillationError(
            "compatibility classification authority is not externally pinned"
        )
    expected_decision = _authority_sha256(
        expected_decision_receipt_digest,
        label="externally pinned compatibility decision receipt digest",
    )
    if receipt["receipt_digest"] != expected_decision:
        raise ActionAnchorDistillationError(
            "compatibility decision receipt is not externally pinned"
        )
    raw_items = _exact_list(
        receipt["items"], label="compatibility receipt items"
    )
    candidate_kinds: list[str] = []
    verdicts: list[str] = []
    for index, item_value in enumerate(raw_items):
        item = _closed_dict(
            item_value,
            _COMPATIBILITY_ITEM_FIELDS,
            label=f"compatibility item[{index}]",
        )
        if type(item["batch_index"]) is not int or item["batch_index"] != index:
            raise ActionAnchorDistillationError(
                "compatibility batch indices differ"
            )
        _sha256(item["row_id"], label="compatibility row ID")
        axis_matches = _closed_dict(
            item["axis_matches"], set(SEMANTIC_AXES), label="axis matches"
        )
        if any(type(axis_matches[axis]) is not bool for axis in SEMANTIC_AXES):
            raise ActionAnchorDistillationError(
                "compatibility axis matches must be exact booleans"
            )
        mismatch_axes = _exact_list(
            item["mismatch_axes"], label="compatibility mismatch axes"
        )
        if any(type(axis) is not str for axis in mismatch_axes):
            raise ActionAnchorDistillationError(
                "compatibility mismatch axes must be strings"
            )
        candidate_kinds.append(item["candidate_kind"])
        verdicts.append(item["qualification_verdict"])
    # Rebuild from the bound q receipts.  This simultaneously checks every
    # derived axis, route, row, ordering, and candidate relation.
    expected = _build_compatibility_without_recursive_validation(
        q_y=q_y,
        q_anchor=q_anchor,
        candidate_kinds=candidate_kinds,
        qualification_verdicts=verdicts,
        authority=authority,
    )
    if receipt != expected:
        raise ActionAnchorDistillationError(
            "compatibility receipt differs from the derived policy"
        )
    return json.loads(canonical_json_bytes(receipt).decode("ascii"))


def _build_compatibility_without_recursive_validation(
    *,
    q_y: dict[str, Any],
    q_anchor: dict[str, Any],
    candidate_kinds: Sequence[str],
    qualification_verdicts: Sequence[str],
    authority: str,
) -> dict[str, Any]:
    """Internal equivalent of build_compatibility_receipt_v1 after q checks."""

    if q_y["q_kind"] != "q_y" or q_anchor["q_kind"] != "q_anchor":
        raise ActionAnchorDistillationError("compatibility q kinds differ")
    if q_y["layout"] != q_anchor["layout"] or q_y["producer"] != q_anchor["producer"]:
        raise ActionAnchorDistillationError("compatibility teacher space differs")
    batch_size = q_y["layout"]["batch_size"]
    if len(candidate_kinds) != batch_size or len(qualification_verdicts) != batch_size:
        raise ActionAnchorDistillationError("compatibility batch coverage differs")
    items: list[dict[str, Any]] = []
    for index, (desired_item, anchor_item) in enumerate(
        zip(q_y["items"], q_anchor["items"])
    ):
        for name in ("row_id", "source_sha256", "instruction_sha256"):
            if desired_item[name] != anchor_item[name]:
                raise ActionAnchorDistillationError(
                    f"compatibility {name} binding differs"
                )
        if desired_item["endpoint_sha256"] == anchor_item["endpoint_sha256"]:
            raise ActionAnchorDistillationError(
                "q_anchor endpoint must differ from the q_y target endpoint"
            )
        kind = candidate_kinds[index]
        verdict = qualification_verdicts[index]
        if type(kind) is not str or kind not in CANDIDATE_KINDS:
            raise ActionAnchorDistillationError("anchor candidate kind differs")
        if type(verdict) is not str or verdict not in QUALIFICATION_VERDICTS:
            raise ActionAnchorDistillationError("anchor verdict differs")
        matches = _axis_matches(desired_item["semantics"], anchor_item["semantics"])
        if kind == "unqualified":
            if verdict == "accept":
                raise ActionAnchorDistillationError(
                    "unqualified anchor cannot be accepted"
                )
            training_use = "excluded"
            contrastive_role = "none"
        else:
            if verdict != "accept":
                raise ActionAnchorDistillationError(
                    "non-accepted anchor must be unqualified"
                )
            qualification = anchor_item["teacher_evidence"][
                "qualification_receipt"
            ]
            if (
                qualification["qualification_status"] != "qualified"
                or qualification["contrastive_authorized"] is not True
                or qualification["point_distillation_authorized"] is not False
            ):
                raise ActionAnchorDistillationError(
                    "active q_anchor lacks externally qualified contrastive authority"
                )
            _validate_candidate_relation(
                candidate_kind=kind,
                desired=desired_item["semantics"],
                candidate=anchor_item["semantics"],
                matches=matches,
            )
            training_use = "contrastive-only"
            contrastive_role = "positive" if kind == "compatible" else "negative"
        items.append(
            {
                "batch_index": index,
                "row_id": desired_item["row_id"],
                "candidate_kind": kind,
                "qualification_verdict": verdict,
                "axis_matches": matches,
                "mismatch_axes": [axis for axis in SEMANTIC_AXES if not matches[axis]],
                "training_use": training_use,
                "contrastive_role": contrastive_role,
            }
        )
    unsigned = {
        "schema_version": COMPATIBILITY_RECEIPT_SCHEMA,
        "policy_sha256": COMPATIBILITY_POLICY_SHA256,
        "q_y_receipt_digest": q_y["receipt_digest"],
        "q_anchor_receipt_digest": q_anchor["receipt_digest"],
        "classification_authority_sha256": authority,
        "items": items,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


@dataclass(frozen=True)
class RoutedAnchorV1:
    """One batched anchor plus its q and compatibility receipts."""

    plan: ActionPlanOutput
    q_receipt: Mapping[str, Any]
    compatibility_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class DistillationLossConfigV1:
    smooth_l1_weight: float = 1.0
    cosine_weight: float = 1.0
    infonce_weight: float = 1.0
    preservation_weight: float = 0.25
    smooth_l1_beta: float = 1.0
    temperature: float = 0.07

    def validate(self) -> None:
        for name in (
            "smooth_l1_weight",
            "cosine_weight",
            "infonce_weight",
            "preservation_weight",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
                raise ActionAnchorDistillationError(
                    f"{name} must be a finite non-negative number"
                )
        for name in ("smooth_l1_beta", "temperature"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(float(value)) or value <= 0:
                raise ActionAnchorDistillationError(
                    f"{name} must be a finite positive number"
                )
        if float(self.smooth_l1_weight) + float(self.cosine_weight) <= 0.0:
            raise ActionAnchorDistillationError(
                "q_y point-distillation cannot disable both SmoothL1 and cosine"
            )


@dataclass(frozen=True)
class DistillationLossV1:
    schema_version: str
    total: Any
    smooth_l1: Any
    cosine: Any
    infonce: Any
    preservation: Any
    point_pair_count: int
    contrastive_positive_pair_count: int
    contrastive_negative_pair_count: int
    excluded_pair_count: int


def _flatten_plan(plan: ActionPlanOutput) -> Any:
    # Equalize phase-block and global-block energy before cosine/InfoNCE.  A
    # raw concatenation would give the 21 phase tokens roughly 21x the weight
    # of the one global action token solely because of dimensionality.
    return _torch().cat(
        (
            plan.phase_tokens.reshape(int(plan.phase_tokens.shape[0]), -1)
            / math.sqrt(float(PHASE_COUNT)),
            plan.global_token,
        ),
        dim=1,
    )


def _tensor_storage_span(value: Any) -> tuple[str, int, int]:
    """Return an exact byte span for overlap checks on contiguous tensors."""

    device = f"{value.device.type}:{value.device.index}"
    start = int(value.untyped_storage().data_ptr()) + (
        int(value.storage_offset()) * int(value.element_size())
    )
    return device, start, start + int(value.numel()) * int(value.element_size())


def _assert_cross_role_storage_disjoint(
    labelled_plans: Sequence[tuple[str, ActionPlanOutput]]
) -> None:
    spans: list[tuple[str, str, int, int]] = []
    for role, plan in labelled_plans:
        for member, tensor in (
            ("phase", plan.phase_tokens),
            ("global", plan.global_token),
        ):
            device, start, end = _tensor_storage_span(tensor)
            spans.append((f"{role}.{member}", device, start, end))
    for left_index, (left_name, left_device, left_start, left_end) in enumerate(spans):
        for right_name, right_device, right_start, right_end in spans[left_index + 1 :]:
            if left_name.split(".", 1)[0] == right_name.split(".", 1)[0]:
                continue
            if left_device == right_device and max(left_start, right_start) < min(
                left_end, right_end
            ):
                raise ActionAnchorDistillationError(
                    f"action-code tensor storage overlaps across {left_name} and {right_name}"
                )


def _validate_pred_target_bindings(
    q_pred_receipt: Mapping[str, Any],
    q_y_receipt: Mapping[str, Any],
    *,
    expected_teacher_authority_sha256: str,
    expected_q_y_qualification_receipt_digests: Sequence[str],
) -> None:
    q_pred = validate_q_receipt_v1(q_pred_receipt)
    q_y = validate_q_receipt_v1(
        q_y_receipt,
        expected_teacher_authority_sha256=expected_teacher_authority_sha256,
        expected_qualification_receipt_digests=
        expected_q_y_qualification_receipt_digests,
    )
    if q_pred["q_kind"] != "q_pred" or q_y["q_kind"] != "q_y":
        raise ActionAnchorDistillationError("loss requires q_pred and q_y receipts")
    if q_pred["layout"] != q_y["layout"]:
        raise ActionAnchorDistillationError("q_pred/q_y layouts differ")
    for pred_item, target_item in zip(q_pred["items"], q_y["items"]):
        for name in (
            "batch_index",
            "row_id",
            "source_sha256",
            "instruction_sha256",
            "semantics_sha256",
        ):
            if pred_item[name] != target_item[name]:
                raise ActionAnchorDistillationError(
                    f"q_pred/q_y {name} binding differs"
                )


def action_anchor_distillation_loss_v1(
    *,
    q_pred: ActionPlanOutput,
    q_y: ActionPlanOutput,
    q_pred_receipt: Mapping[str, Any],
    q_y_receipt: Mapping[str, Any],
    expected_teacher_authority_sha256: str,
    expected_classification_authority_sha256: str,
    expected_q_y_qualification_receipt_digests: Sequence[str],
    expected_anchor_qualification_receipt_digests: Sequence[Sequence[str]] = (),
    expected_compatibility_decision_receipt_digests: Sequence[str] = (),
    anchors: Sequence[RoutedAnchorV1] = (),
    preservation_loss: Any | None = None,
    config: DistillationLossConfigV1 | None = None,
) -> DistillationLossV1:
    """Compute SmoothL1 + cosine + multi-positive InfoNCE + preservation.

    All q_y and q_anchor tensors are detached inside this function regardless
    of their incoming ``requires_grad`` state.  Only q_pred and an optional
    caller-supplied preservation scalar can receive gradients.
    """

    torch = _torch()
    functional = torch.nn.functional
    cfg = config or DistillationLossConfigV1()
    if type(cfg) is not DistillationLossConfigV1:
        raise ActionAnchorDistillationError(
            "loss config must be exact DistillationLossConfigV1"
        )
    cfg.validate()
    pred_plan = _validate_plan(q_pred, label="q_pred")
    if not pred_plan.phase_tokens.requires_grad or not pred_plan.global_token.requires_grad:
        raise ActionAnchorDistillationError(
            "training q_pred phase/global tensors must both retain an autograd path"
        )
    target_plan = _validate_plan(
        q_y, label="q_y", batch_size=int(pred_plan.phase_tokens.shape[0])
    )
    if pred_plan.phase_tokens.device != target_plan.phase_tokens.device:
        raise ActionAnchorDistillationError("q_pred/q_y devices differ")
    teacher_authority_sha = _authority_sha256(
        expected_teacher_authority_sha256,
        label="loss expected teacher authority SHA-256",
    )
    classification_authority_sha = _authority_sha256(
        expected_classification_authority_sha256,
        label="loss expected classification authority SHA-256",
    )
    pred_receipt = validate_q_receipt_v1(q_pred_receipt, plan=pred_plan)
    target_receipt = validate_q_receipt_v1(
        q_y_receipt,
        plan=target_plan,
        expected_teacher_authority_sha256=teacher_authority_sha,
        expected_qualification_receipt_digests=
        expected_q_y_qualification_receipt_digests,
    )
    _validate_pred_target_bindings(
        pred_receipt,
        target_receipt,
        expected_teacher_authority_sha256=teacher_authority_sha,
        expected_q_y_qualification_receipt_digests=
        expected_q_y_qualification_receipt_digests,
    )
    if type(anchors) not in (list, tuple):
        raise ActionAnchorDistillationError("anchors must be an exact list or tuple")
    if (
        type(expected_anchor_qualification_receipt_digests) not in (list, tuple)
        or type(expected_compatibility_decision_receipt_digests) not in (list, tuple)
        or len(expected_anchor_qualification_receipt_digests) != len(anchors)
        or len(expected_compatibility_decision_receipt_digests) != len(anchors)
    ):
        raise ActionAnchorDistillationError(
            "external anchor qualification/decision pin coverage differs"
        )

    batch_size = int(pred_plan.phase_tokens.shape[0])
    pred_flat = _flatten_plan(pred_plan)
    target_flat = _flatten_plan(target_plan).detach()
    contrastive_positives: list[list[Any]] = [
        [target_flat[index]] for index in range(batch_size)
    ]
    negative_targets: list[list[Any]] = [[] for _ in range(batch_size)]
    point_pair_count = batch_size
    contrastive_positive_pair_count = 0
    contrastive_negative_pair_count = 0
    excluded_pair_count = 0
    labelled_plans: list[tuple[str, ActionPlanOutput]] = [
        ("q_pred", pred_plan),
        ("q_y", target_plan),
    ]
    teacher_tensors: list[Any] = [
        target_plan.phase_tokens,
        target_plan.global_token,
    ]
    seen_receipts: set[str] = set()
    seen_endpoints: list[set[str]] = [set() for _ in range(batch_size)]
    seen_materializations: list[set[str]] = [set() for _ in range(batch_size)]
    seen_qualifications: list[set[str]] = [set() for _ in range(batch_size)]

    for anchor_index, routed in enumerate(anchors):
        if type(routed) is not RoutedAnchorV1:
            raise ActionAnchorDistillationError(
                f"anchor[{anchor_index}] must be exact RoutedAnchorV1"
            )
        anchor_plan = _validate_plan(
            routed.plan, label=f"anchor[{anchor_index}]", batch_size=batch_size
        )
        if anchor_plan.phase_tokens.device != pred_plan.phase_tokens.device:
            raise ActionAnchorDistillationError("anchor device differs")
        anchor_receipt = validate_q_receipt_v1(
            routed.q_receipt,
            plan=anchor_plan,
            expected_teacher_authority_sha256=teacher_authority_sha,
            expected_qualification_receipt_digests=
            expected_anchor_qualification_receipt_digests[anchor_index],
        )
        receipt_digest = anchor_receipt["receipt_digest"]
        if receipt_digest in seen_receipts:
            raise ActionAnchorDistillationError(
                "duplicate q_anchor receipt cannot change contrastive weight"
            )
        seen_receipts.add(receipt_digest)
        compatibility = validate_compatibility_receipt_v1(
            routed.compatibility_receipt,
            q_y_receipt=target_receipt,
            q_anchor_receipt=anchor_receipt,
            expected_teacher_authority_sha256=teacher_authority_sha,
            expected_classification_authority_sha256=classification_authority_sha,
            expected_q_y_qualification_receipt_digests=
            expected_q_y_qualification_receipt_digests,
            expected_q_anchor_qualification_receipt_digests=
            expected_anchor_qualification_receipt_digests[anchor_index],
            expected_decision_receipt_digest=
            expected_compatibility_decision_receipt_digests[anchor_index],
        )
        labelled_plans.append((f"q_anchor[{anchor_index}]", anchor_plan))
        teacher_tensors.extend((anchor_plan.phase_tokens, anchor_plan.global_token))
        anchor_flat = _flatten_plan(anchor_plan).detach()
        for index, item in enumerate(compatibility["items"]):
            anchor_item = anchor_receipt["items"][index]
            evidence = anchor_item["teacher_evidence"]
            endpoint = anchor_item["endpoint_sha256"]
            materialization_digest = evidence["materialization_receipt"][
                "receipt_sha256"
            ]
            qualification_digest = evidence["qualification_receipt"][
                "receipt_digest"
            ]
            if (
                endpoint in seen_endpoints[index]
                or materialization_digest in seen_materializations[index]
                or qualification_digest in seen_qualifications[index]
            ):
                raise ActionAnchorDistillationError(
                    "duplicate q_anchor endpoint/materialization/qualification"
                )
            seen_endpoints[index].add(endpoint)
            seen_materializations[index].add(materialization_digest)
            seen_qualifications[index].add(qualification_digest)
            if (
                item["training_use"] == "contrastive-only"
                and item["contrastive_role"] == "positive"
            ):
                contrastive_positives[index].append(anchor_flat[index])
                contrastive_positive_pair_count += 1
            elif (
                item["training_use"] == "contrastive-only"
                and item["contrastive_role"] == "negative"
            ):
                negative_targets[index].append(anchor_flat[index])
                contrastive_negative_pair_count += 1
            elif (
                item["training_use"] == "excluded"
                and item["contrastive_role"] == "none"
            ):
                excluded_pair_count += 1
            else:  # pragma: no cover - protected by receipt validation
                raise ActionAnchorDistillationError("unknown anchor training use")

    _assert_cross_role_storage_disjoint(labelled_plans)

    # q_y is the only point-distillation teacher in V1.  Anchors never enter
    # either term: no per-row phase-alignment authority exists in this module.
    if any(
        len(contrastive_positives[index]) > 1 and not negative_targets[index]
        for index in range(batch_size)
    ):
        raise ActionAnchorDistillationError(
            "a compatible q_anchor needs at least one hard negative in the same record"
        )
    if (
        (contrastive_positive_pair_count or contrastive_negative_pair_count)
        and float(cfg.infonce_weight) <= 0.0
    ):
        raise ActionAnchorDistillationError(
            "routed contrastive anchors require a positive InfoNCE weight"
        )
    smooth_phase = functional.smooth_l1_loss(
        pred_plan.phase_tokens,
        target_plan.phase_tokens.detach(),
        reduction="none",
        beta=float(cfg.smooth_l1_beta),
    ).mean(dim=(1, 2))
    smooth_global = functional.smooth_l1_loss(
        pred_plan.global_token,
        target_plan.global_token.detach(),
        reduction="none",
        beta=float(cfg.smooth_l1_beta),
    ).mean(dim=1)
    smooth = (0.5 * (smooth_phase + smooth_global)).mean()
    cosine_phase = 1.0 - functional.cosine_similarity(
        pred_plan.phase_tokens.reshape(batch_size, -1),
        target_plan.phase_tokens.detach().reshape(batch_size, -1),
        dim=1,
        eps=1.0e-8,
    )
    cosine_global = 1.0 - functional.cosine_similarity(
        pred_plan.global_token,
        target_plan.global_token.detach(),
        dim=1,
        eps=1.0e-8,
    )
    cosine = (
        0.5 * (cosine_phase + cosine_global)
    ).mean()
    infonce_terms: list[Any] = []
    for index in range(batch_size):
        prediction = pred_flat[index]
        positives = contrastive_positives[index]
        negatives = negative_targets[index]
        if negatives:
            prediction_unit = functional.normalize(
                prediction.unsqueeze(0), dim=1, eps=1.0e-8
            )[0]
            positive_logits = torch.stack(
                [
                    torch.dot(
                        prediction_unit,
                        functional.normalize(
                            positive.unsqueeze(0), dim=1, eps=1.0e-8
                        )[0],
                    )
                    / float(cfg.temperature)
                    for positive in positives
                ]
            )
            negative_logits = torch.stack(
                [
                    torch.dot(
                        prediction_unit,
                        functional.normalize(
                            negative.unsqueeze(0), dim=1, eps=1.0e-8
                        )[0],
                    )
                    / float(cfg.temperature)
                    for negative in negatives
                ]
            )
            infonce_terms.append(
                (
                    torch.logsumexp(
                        torch.cat((positive_logits, negative_logits)), dim=0
                    )
                    - torch.logsumexp(positive_logits, dim=0)
                ).reshape(1)
            )

    infonce = (
        torch.cat(infonce_terms).mean()
        if infonce_terms
        else pred_flat.sum() * 0.0
    )
    if preservation_loss is None:
        preservation = pred_flat.sum() * 0.0
    else:
        if (
            type(preservation_loss) is not torch.Tensor
            or not preservation_loss.is_floating_point()
            or preservation_loss.numel() != 1
            or preservation_loss.device != pred_plan.phase_tokens.device
            or not bool(torch.isfinite(preservation_loss.detach()).all().item())
        ):
            raise ActionAnchorDistillationError(
                "preservation loss must be one finite floating tensor on the q_pred device"
            )
        connected_teachers = [tensor for tensor in teacher_tensors if tensor.requires_grad]
        if preservation_loss.requires_grad and connected_teachers:
            gradients = torch.autograd.grad(
                preservation_loss,
                connected_teachers,
                allow_unused=True,
                retain_graph=True,
            )
            if any(gradient is not None for gradient in gradients):
                raise ActionAnchorDistillationError(
                    "preservation loss may not carry a q_y/q_anchor teacher gradient path"
                )
        preservation = preservation_loss.float().reshape(())
    total = (
        float(cfg.smooth_l1_weight) * smooth
        + float(cfg.cosine_weight) * cosine
        + float(cfg.infonce_weight) * infonce
        + float(cfg.preservation_weight) * preservation
    )
    if not bool(torch.isfinite(total.detach()).item()):
        raise ActionAnchorDistillationError("distillation loss is non-finite")
    return DistillationLossV1(
        schema_version=LOSS_SCHEMA,
        total=total,
        smooth_l1=smooth,
        cosine=cosine,
        infonce=infonce,
        preservation=preservation,
        point_pair_count=point_pair_count,
        contrastive_positive_pair_count=contrastive_positive_pair_count,
        contrastive_negative_pair_count=contrastive_negative_pair_count,
        excluded_pair_count=excluded_pair_count,
    )


@dataclass(frozen=True)
class InterventionAuditV1:
    schema_version: str
    q_pred_receipt_digest: str
    q_y_receipt_digest: str
    reverse_compatibility_receipt_digest: str
    correct_score: Any
    shuffled_score: Any
    zero_score: Any
    reverse_score: Any
    correct_minus_shuffled: Any
    correct_minus_zero: Any
    correct_minus_reverse: Any
    minimum_margin: float
    passed: bool


def audit_action_plan_interventions_v1(
    *,
    q_pred: ActionPlanOutput,
    q_y: ActionPlanOutput,
    q_pred_receipt: Mapping[str, Any],
    q_y_receipt: Mapping[str, Any],
    reverse_anchor: RoutedAnchorV1,
    expected_teacher_authority_sha256: str,
    expected_classification_authority_sha256: str,
    expected_q_y_qualification_receipt_digests: Sequence[str],
    expected_reverse_qualification_receipt_digests: Sequence[str],
    expected_reverse_decision_receipt_digest: str,
    minimum_margin: float = 0.0,
) -> InterventionAuditV1:
    """Audit that the prediction prefers correct over shuffled/zero/reverse q."""

    torch = _torch()
    functional = torch.nn.functional
    if (
        type(minimum_margin) not in (int, float)
        or not math.isfinite(float(minimum_margin))
        or minimum_margin < 0
    ):
        raise ActionAnchorDistillationError(
            "minimum intervention margin must be finite and non-negative"
        )
    pred = _validate_plan(q_pred, label="intervention q_pred")
    batch_size = int(pred.phase_tokens.shape[0])
    if batch_size < 2:
        raise ActionAnchorDistillationError(
            "shuffled intervention requires batch size at least two"
        )
    target = _validate_plan(q_y, label="intervention q_y", batch_size=batch_size)
    teacher_authority_sha = _authority_sha256(
        expected_teacher_authority_sha256,
        label="intervention expected teacher authority SHA-256",
    )
    classification_authority_sha = _authority_sha256(
        expected_classification_authority_sha256,
        label="intervention expected classification authority SHA-256",
    )
    pred_receipt = validate_q_receipt_v1(q_pred_receipt, plan=pred)
    target_receipt = validate_q_receipt_v1(
        q_y_receipt,
        plan=target,
        expected_teacher_authority_sha256=teacher_authority_sha,
        expected_qualification_receipt_digests=
        expected_q_y_qualification_receipt_digests,
    )
    _validate_pred_target_bindings(
        pred_receipt,
        target_receipt,
        expected_teacher_authority_sha256=teacher_authority_sha,
        expected_q_y_qualification_receipt_digests=
        expected_q_y_qualification_receipt_digests,
    )
    if len({item["row_id"] for item in target_receipt["items"]}) != batch_size:
        raise ActionAnchorDistillationError(
            "shuffled intervention requires distinct row IDs"
        )
    if type(reverse_anchor) is not RoutedAnchorV1:
        raise ActionAnchorDistillationError(
            "reverse intervention requires an exact RoutedAnchorV1"
        )
    reverse = _validate_plan(
        reverse_anchor.plan,
        label="intervention q_reverse",
        batch_size=batch_size,
    )
    reverse_receipt = validate_q_receipt_v1(
        reverse_anchor.q_receipt,
        plan=reverse,
        expected_teacher_authority_sha256=teacher_authority_sha,
        expected_qualification_receipt_digests=
        expected_reverse_qualification_receipt_digests,
    )
    reverse_compatibility = validate_compatibility_receipt_v1(
        reverse_anchor.compatibility_receipt,
        q_y_receipt=target_receipt,
        q_anchor_receipt=reverse_receipt,
        expected_teacher_authority_sha256=teacher_authority_sha,
        expected_classification_authority_sha256=classification_authority_sha,
        expected_q_y_qualification_receipt_digests=
        expected_q_y_qualification_receipt_digests,
        expected_q_anchor_qualification_receipt_digests=
        expected_reverse_qualification_receipt_digests,
        expected_decision_receipt_digest=
        expected_reverse_decision_receipt_digest,
    )
    if any(
        item["candidate_kind"] != "reverse"
        or item["training_use"] != "contrastive-only"
        or item["contrastive_role"] != "negative"
        for item in reverse_compatibility["items"]
    ):
        raise ActionAnchorDistillationError(
            "reverse intervention is not authority-routed as a reverse hard negative"
        )
    if not (
        pred.phase_tokens.device
        == target.phase_tokens.device
        == reverse.phase_tokens.device
    ):
        raise ActionAnchorDistillationError("intervention devices differ")
    _assert_cross_role_storage_disjoint(
        (("q_pred", pred), ("q_y", target), ("q_reverse", reverse))
    )
    pred_flat = _flatten_plan(pred).detach()
    target_flat = _flatten_plan(target).detach()
    reverse_flat = _flatten_plan(reverse).detach()
    shuffled_flat = torch.roll(target_flat, shifts=-1, dims=0)
    zero_flat = torch.zeros_like(target_flat)

    def score(reference: Any) -> Any:
        return functional.cosine_similarity(
            pred_flat, reference, dim=1, eps=1.0e-8
        )

    correct_score = score(target_flat)
    shuffled_score = score(shuffled_flat)
    zero_score = score(zero_flat)
    reverse_score = score(reverse_flat)
    correct_minus_shuffled = correct_score - shuffled_score
    correct_minus_zero = correct_score - zero_score
    correct_minus_reverse = correct_score - reverse_score
    threshold = float(minimum_margin)
    passed = bool(
        (correct_minus_shuffled > threshold).all().item()
        and (correct_minus_zero > threshold).all().item()
        and (correct_minus_reverse > threshold).all().item()
    )
    return InterventionAuditV1(
        schema_version=INTERVENTION_SCHEMA,
        q_pred_receipt_digest=pred_receipt["receipt_digest"],
        q_y_receipt_digest=target_receipt["receipt_digest"],
        reverse_compatibility_receipt_digest=reverse_compatibility[
            "receipt_digest"
        ],
        correct_score=correct_score,
        shuffled_score=shuffled_score,
        zero_score=zero_score,
        reverse_score=reverse_score,
        correct_minus_shuffled=correct_minus_shuffled,
        correct_minus_zero=correct_minus_zero,
        correct_minus_reverse=correct_minus_reverse,
        minimum_margin=threshold,
        passed=passed,
    )


def require_action_plan_interventions_v1(**kwargs: Any) -> InterventionAuditV1:
    report = audit_action_plan_interventions_v1(**kwargs)
    if not report.passed:
        raise ActionAnchorDistillationError(
            "q_pred did not beat shuffled, zero, and reverse interventions"
        )
    return report


def _contract_body() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "local_only": LOCAL_ONLY,
        "implements_visual_teacher_extraction": IMPLEMENTS_VISUAL_TEACHER_EXTRACTION,
        "plan_abi": {
            "predictor_schema": PREDICTOR_ABI_SCHEMA,
            "phase_count": PHASE_COUNT,
            "action_width": ACTION_WIDTH,
            "global_width": ACTION_WIDTH,
            "canonical_dtype": CANONICAL_DTYPE,
        },
        "q_receipt_schema": Q_RECEIPT_SCHEMA,
        "materialization_receipt_schema": MATERIALIZATION_RECEIPT_SCHEMA,
        "teacher_qualification_receipt_schema": TEACHER_QUALIFICATION_RECEIPT_SCHEMA,
        "teacher_qualification_authority_schema": TEACHER_QUALIFICATION_AUTHORITY_SCHEMA,
        "teacher_authority_requires_external_expected_sha256": True,
        "each_teacher_qualification_receipt_requires_external_expected_digest": True,
        "classification_authority_requires_external_expected_sha256": True,
        "each_compatibility_decision_requires_external_expected_digest": True,
        "authority_digests_reject_all_zero_placeholders": True,
        "provenance_and_producer_digests_reject_all_zero_placeholders": True,
        "materialization_tensor_payload_rehashed_per_item": True,
        "q_kinds": list(Q_KINDS),
        "teacher_q_kinds": list(TEACHER_Q_KINDS),
        "teacher_gradient_policy": "always-stop-gradient-inside-loss",
        "teacher_and_prediction_storage_must_be_pairwise_disjoint": True,
        "teacher_connected_preservation_graph_rejected": True,
        "duplicate_anchor_receipts_endpoints_materializations_qualifications_rejected": True,
        "distillation_roles": dict(DISTILLATION_ROLES),
        "compatibility_receipt_schema": COMPATIBILITY_RECEIPT_SCHEMA,
        "compatibility_policy_sha256": COMPATIBILITY_POLICY_SHA256,
        "semantic_axes": list(SEMANTIC_AXES),
        "hard_negative_kinds": list(HARD_NEGATIVE_KINDS),
        "training_uses": list(TRAINING_USES),
        "q_y_is_unique_point_teacher": True,
        "q_anchor_point_distillation_enabled": False,
        "q_anchor_positive_aggregation": "multi-positive-logsumexp-not-anchor-mean",
        "point_loss_cannot_be_fully_disabled": True,
        "phase_and_global_point_blocks_are_equal_weight": True,
        "phase_and_global_contrastive_energy_is_dimension_equalized": True,
        "compatible_anchor_requires_same-record_hard_negative": True,
        "routed_anchor_requires_positive_infonce_weight": True,
        "q_pred_autograd_required_by_loss": True,
        "loss_schema": LOSS_SCHEMA,
        "loss_components": [
            "smooth-l1",
            "cosine-distance",
            "multi-positive-infonce",
            "caller-supplied-preservation",
        ],
        "interventions": ["shuffled", "zero", "reverse"],
        "intervention_inputs_are_receipt_bound": True,
        "internal_reverse_direction_ontology_implemented": False,
        "reverse_relation_requires_external_per_item_decision_authority": True,
    }


CONTRACT_SHA256 = (
    "6e2159102c712c57b35037679eaf31768eebaa2554ef0efe0ac1553fecca8a5b"
)


def contract_receipt_v1() -> dict[str, Any]:
    body = _contract_body()
    return {**body, "receipt_digest": object_sha256(body)}


def _assert_pinned_contract_hashes() -> None:
    observed_policy = object_sha256(_compatibility_policy_body())
    if observed_policy != COMPATIBILITY_POLICY_SHA256:
        raise RuntimeError(
            "action-anchor compatibility policy hash is not pinned to its body"
        )
    observed_contract = object_sha256(_contract_body())
    if observed_contract != CONTRACT_SHA256:
        raise RuntimeError(
            "action-anchor distillation contract hash is not pinned to its body"
        )


_assert_pinned_contract_hashes()


__all__ = [
    "ACTION_WIDTH",
    "CANDIDATE_KINDS",
    "CANONICAL_DTYPE",
    "COMPATIBILITY_POLICY_SHA256",
    "COMPATIBILITY_RECEIPT_SCHEMA",
    "CONTRACT_SHA256",
    "DISTILLATION_ROLES",
    "EXTERNAL_TEACHER_PRODUCER",
    "HARD_NEGATIVE_KINDS",
    "IMPLEMENTS_VISUAL_TEACHER_EXTRACTION",
    "INTERVENTION_SCHEMA",
    "LOCAL_ONLY",
    "LOSS_SCHEMA",
    "PHASE_COUNT",
    "Q_KINDS",
    "Q_RECEIPT_SCHEMA",
    "QUALIFICATION_VERDICTS",
    "SCHEMA_VERSION",
    "SEMANTIC_AXES",
    "TENSOR_HASH_SCHEMA",
    "TEACHER_QUALIFICATION_AUTHORITY_SCHEMA",
    "TEACHER_QUALIFICATION_RECEIPT_SCHEMA",
    "MATERIALIZATION_PROJECTION_SCHEMA",
    "MATERIALIZATION_RECEIPT_SCHEMA",
    "MATERIALIZATION_SOURCE_SCHEMA",
    "TEACHER_Q_KINDS",
    "TRAINING_USES",
    "ActionAnchorDistillationError",
    "DistillationLossConfigV1",
    "DistillationLossV1",
    "InterventionAuditV1",
    "RoutedAnchorV1",
    "action_anchor_distillation_loss_v1",
    "audit_action_plan_interventions_v1",
    "build_compatibility_receipt_v1",
    "build_q_receipt_v1",
    "canonical_json_bytes",
    "contract_receipt_v1",
    "object_sha256",
    "require_action_plan_interventions_v1",
    "tensor_sha256_v1",
    "validate_action_semantics",
    "validate_compatibility_receipt_v1",
    "validate_q_receipt_v1",
]
