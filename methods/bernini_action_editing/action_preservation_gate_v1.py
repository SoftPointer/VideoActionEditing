#!/usr/bin/env python3
"""Fail-closed six-axis gate for decoded action-editing candidates.

The gate keeps action/order, onset, source identity, background, camera, and
quality as separate axes.  Terminal hold is an explicit, calibrated component
of action/order.  It never forms a weighted score.  Missing calibration or an
unsupported sample yields ``abstain`` rather than silently dropping an axis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence


CALIBRATION_SCHEMA = "bernini-action-preservation-calibration-v1"
MEASUREMENT_SCHEMA = "bernini-action-preservation-measurement-v1"
DECISION_SCHEMA = "bernini-action-preservation-gate-decision-v1"
BLIND_REVIEW_SCHEMA = "bernini-action-preservation-blind-review-v1"
BLIND_BALLOT_SCHEMA = "bernini-action-preservation-blind-ballot-v1"
EVALUATION_AGGREGATE_SCHEMA = "bernini-action-preservation-decoded-eval-aggregate-v3"
PUBLIC_PACKET_SCHEMA = "blind-full-video-review-packet-v1"
PRIVATE_PACKET_SCHEMA = "bernini-action-preservation-blind-private-map-v1"
ACTION_REVIEW_CONTRACT_SCHEMA = "bernini-action-preservation-action-review-contract-v1"
PROMOTION_SCHEMA = "bernini-action-preservation-promotion-v1"
WORK_ROOT_BINDING_ENV = "APV2_EVAL_WORK_ROOT_AUTHORITY"
TASK_FD_BINDING_ENV = "APV2_EVAL_INHERITED_AUTHORITY_FDS"
GATE_RUNTIME_TARGET = "action_preservation_gate_v1.py"
LOOP_RUNTIME_TARGET = "action_preservation_loop_controller_v1.py"

AXES = (
    "action_order",
    "onset",
    "source_identity",
    "background",
    "camera",
    "quality",
)
PRESERVATION_AXES = ("source_identity", "background", "camera", "quality")
STATES = ("pass", "fail", "undetermined")
PRIVATE_REVIEW_TOKENS = (
    "v2_onset_all", "v2_noop020_all", "v2_func010_all",
    "v2_func025_all", "v2_func050_all", "v2_onset_cross_qo",
    "v2_func010_cross_qo", "v2_func025_cross_qo",
    "hard1_every_step", "checkpoint-", "checkpoint_step", "onset_policy",
    "adapter_sha256", "lora_safe_merge",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")


class ActionPreservationGateError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ActionPreservationGateError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _closed(value: Any, fields: set[str] | frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ActionPreservationGateError(f"{label} field closure differs")
    return value


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ActionPreservationGateError(f"{label} is not a lowercase SHA-256")
    return value


def _identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ActionPreservationGateError(f"{label} is invalid")
    return value


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionPreservationGateError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ActionPreservationGateError(f"{label} is non-finite")
    return result


def _unit(value: Any, *, label: str) -> float:
    result = _finite(value, label=label)
    if not 0.0 <= result <= 1.0:
        raise ActionPreservationGateError(f"{label} lies outside [0,1]")
    return result


def _verify_digest(value: Mapping[str, Any], *, field: str, label: str) -> str:
    digest = _sha(value.get(field), label=f"{label} digest")
    payload = dict(value)
    payload.pop(field)
    if object_sha256(payload) != digest:
        raise ActionPreservationGateError(f"{label} digest differs")
    return digest


_THRESHOLD_FIELDS = frozenset(
    {
        "face_similarity_min",
        "face_coverage_min",
        "background_similarity_min",
        "background_coverage_min",
        "camera_translation_max",
        "camera_log_scale_abs_max",
        "camera_rotation_degrees_max",
        "camera_reprojection_error_max",
        "quality_min",
        "action_order_min",
        "onset_score_min",
        "onset_timing_error_frames_max",
        "terminal_hold_score_min",
        "terminal_hold_frames_min",
    }
)
_NEGATIVE_FIELDS = frozenset(
    {
        "identity_swap",
        "same_clothes_different_person",
        "mid_video_identity_swap",
        "face_occlusion_positive",
        "background_delete_or_add",
        "scene_swap",
        "shadow_or_reflection",
        "camera_pan_zoom_rotation",
        "background_occlusion_reward_hack",
        "reverse_action",
        "truncated_action",
        "static_terminal_pose",
        "early_onset",
        "late_onset",
        "missing_terminal_hold",
    }
)

_FACE_FIELDS = frozenset(
    {"available", "similarity", "coverage", "source_face_pool_size", "receipt_sha256"}
)
_BACKGROUND_FIELDS = frozenset(
    {
        "available",
        "similarity",
        "valid_coverage",
        "source_and_output_masks_independent",
        "union_exclusion_used",
        "receipt_sha256",
    }
)
_CAMERA_FIELDS = frozenset(
    {
        "available",
        "translation",
        "log_scale_abs",
        "rotation_degrees_abs",
        "reprojection_error",
        "background_registration_used",
        "receipt_sha256",
    }
)
_QUALITY_FIELDS = frozenset({"available", "score", "receipt_sha256"})
_ONSET_FIELDS = frozenset(
    {
        "available",
        "anchor_frame",
        "candidate_frame",
        "timing_error_frames",
        "score",
        "receipt_sha256",
    }
)
_ACTION_FIELDS = frozenset(
    {
        "available",
        "score",
        "reverse_rejected",
        "truncation_rejected",
        "terminal_hold_score",
        "terminal_hold_start_frame",
        "terminal_hold_end_frame",
        "terminal_hold_frames",
        "receipt_sha256",
    }
)


def validate_calibration(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version",
        "calibration_id",
        "heldout_manifest_sha256",
        "human_labels_sha256",
        "thresholds",
        "validation",
        "controlled_negatives",
        "thresholds_frozen_before_candidate_generation",
        "calibration_digest",
    }
    row = dict(_closed(value, fields, label="calibration"))
    if row["schema_version"] != CALIBRATION_SCHEMA:
        raise ActionPreservationGateError("calibration schema differs")
    _identifier(row["calibration_id"], label="calibration id")
    _sha(row["heldout_manifest_sha256"], label="heldout manifest")
    _sha(row["human_labels_sha256"], label="human labels")
    thresholds = dict(_closed(row["thresholds"], _THRESHOLD_FIELDS, label="thresholds"))
    for key in (
        "face_similarity_min",
        "face_coverage_min",
        "background_similarity_min",
        "background_coverage_min",
        "quality_min",
        "action_order_min",
        "onset_score_min",
        "terminal_hold_score_min",
    ):
        thresholds[key] = _unit(thresholds[key], label=key)
    for key in (
        "camera_translation_max",
        "camera_log_scale_abs_max",
        "camera_rotation_degrees_max",
        "camera_reprojection_error_max",
    ):
        thresholds[key] = _finite(thresholds[key], label=key)
        if thresholds[key] < 0.0:
            raise ActionPreservationGateError(f"{key} is negative")
    if (
        type(thresholds["onset_timing_error_frames_max"]) is not int
        or thresholds["onset_timing_error_frames_max"] < 0
    ):
        raise ActionPreservationGateError(
            "onset_timing_error_frames_max must be a non-negative integer"
        )
    if (
        type(thresholds["terminal_hold_frames_min"]) is not int
        or thresholds["terminal_hold_frames_min"] <= 0
    ):
        raise ActionPreservationGateError(
            "terminal_hold_frames_min must be a positive integer"
        )
    validation = dict(
        _closed(
            row["validation"],
            {
                "positive_count",
                "negative_count",
                "face_fixed_fpr_recall",
                "background_fixed_fpr_recall",
                "camera_fixed_fpr_recall",
                "onset_fixed_fpr_recall",
                "terminal_hold_fixed_fpr_recall",
                "human_agreement_report_sha256",
                "worst_group_reported",
                "domain_gap_reported",
                "hacking_failures_reported",
            },
            label="calibration validation",
        )
    )
    for key in ("positive_count", "negative_count"):
        if type(validation[key]) is not int or validation[key] <= 0:
            raise ActionPreservationGateError(f"{key} must be positive")
    for key in (
        "face_fixed_fpr_recall",
        "background_fixed_fpr_recall",
        "camera_fixed_fpr_recall",
        "onset_fixed_fpr_recall",
        "terminal_hold_fixed_fpr_recall",
    ):
        validation[key] = _unit(validation[key], label=key)
    _sha(validation["human_agreement_report_sha256"], label="human agreement report")
    for key in ("worst_group_reported", "domain_gap_reported", "hacking_failures_reported"):
        if validation[key] is not True:
            raise ActionPreservationGateError(f"{key} must be true")
    negatives = dict(
        _closed(row["controlled_negatives"], _NEGATIVE_FIELDS, label="controlled negatives")
    )
    if any(value is not True for value in negatives.values()):
        raise ActionPreservationGateError("controlled-negative coverage is incomplete")
    if row["thresholds_frozen_before_candidate_generation"] is not True:
        raise ActionPreservationGateError("calibration thresholds were not preregistered")
    _verify_digest(row, field="calibration_digest", label="calibration")
    row["thresholds"] = thresholds
    row["validation"] = validation
    row["controlled_negatives"] = negatives
    return row


def _optional_unit(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    return _unit(value, label=label)


def _optional_finite(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    return _finite(value, label=label)


def _optional_nonnegative_int(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ActionPreservationGateError(f"{label} is not a non-negative integer")
    return value


def validate_measurement(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version",
        "candidate_id",
        "candidate_video_sha256",
        "source_video_sha256",
        "scope",
        "face",
        "background",
        "camera",
        "quality",
        "onset",
        "action_order",
        "input_closure",
        "measurement_digest",
    }
    row = dict(_closed(value, fields, label="measurement"))
    if row["schema_version"] != MEASUREMENT_SCHEMA:
        raise ActionPreservationGateError("measurement schema differs")
    _identifier(row["candidate_id"], label="candidate id")
    _sha(row["candidate_video_sha256"], label="candidate video")
    _sha(row["source_video_sha256"], label="source video")
    scope = dict(
        _closed(
            row["scope"],
            {
                "single_subject",
                "human_subject",
                "source_face_visible",
                "output_face_visible",
                "static_or_weak_camera",
                "no_shot_cut",
                "background_expected_unchanged",
            },
            label="scope",
        )
    )
    if any(type(value) is not bool for value in scope.values()):
        raise ActionPreservationGateError("scope flags must be booleans")
    face = dict(
        _closed(
            row["face"],
            _FACE_FIELDS,
            label="face measurement",
        )
    )
    if type(face["available"]) is not bool:
        raise ActionPreservationGateError("face availability is not boolean")
    face["similarity"] = _optional_unit(face["similarity"], label="face similarity")
    face["coverage"] = _optional_unit(face["coverage"], label="face coverage")
    if type(face["source_face_pool_size"]) is not int or face["source_face_pool_size"] < 0:
        raise ActionPreservationGateError("source face pool size is invalid")
    if face["available"] is not (
        face["similarity"] is not None
        and face["coverage"] is not None
        and face["source_face_pool_size"] > 0
    ):
        raise ActionPreservationGateError("face availability/value closure differs")
    background = dict(
        _closed(
            row["background"],
            _BACKGROUND_FIELDS,
            label="background measurement",
        )
    )
    if type(background["available"]) is not bool:
        raise ActionPreservationGateError("background availability is not boolean")
    background["similarity"] = _optional_unit(
        background["similarity"], label="background similarity"
    )
    background["valid_coverage"] = _optional_unit(
        background["valid_coverage"], label="background coverage"
    )
    for key in ("source_and_output_masks_independent", "union_exclusion_used"):
        if type(background[key]) is not bool:
            raise ActionPreservationGateError(f"background {key} is not boolean")
    if background["available"] is not (
        background["similarity"] is not None
        and background["valid_coverage"] is not None
    ):
        raise ActionPreservationGateError("background availability/value closure differs")
    camera = dict(
        _closed(
            row["camera"],
            _CAMERA_FIELDS,
            label="camera measurement",
        )
    )
    if type(camera["available"]) is not bool:
        raise ActionPreservationGateError("camera availability is not boolean")
    if type(camera["background_registration_used"]) is not bool:
        raise ActionPreservationGateError("camera registration flag is not boolean")
    for key in ("translation", "log_scale_abs", "rotation_degrees_abs", "reprojection_error"):
        camera[key] = _optional_finite(camera[key], label=f"camera {key}")
        if camera[key] is not None and camera[key] < 0.0:
            raise ActionPreservationGateError(f"camera {key} is negative")
    if camera["available"] is not all(
        camera[key] is not None
        for key in ("translation", "log_scale_abs", "rotation_degrees_abs", "reprojection_error")
    ):
        raise ActionPreservationGateError("camera availability/value closure differs")
    quality = dict(_closed(row["quality"], _QUALITY_FIELDS, label="quality"))
    onset = dict(_closed(row["onset"], _ONSET_FIELDS, label="onset"))
    action = dict(
        _closed(
            row["action_order"],
            _ACTION_FIELDS,
            label="action/order",
        )
    )
    for label, item in (("quality", quality), ("onset", onset), ("action/order", action)):
        if type(item["available"]) is not bool:
            raise ActionPreservationGateError(f"{label} availability is not boolean")
        item["score"] = _optional_unit(item["score"], label=f"{label} score")
    for key in ("anchor_frame", "candidate_frame", "timing_error_frames"):
        onset[key] = _optional_nonnegative_int(onset[key], label=f"onset {key}")
    onset_values_available = all(
        onset[key] is not None
        for key in ("score", "anchor_frame", "candidate_frame", "timing_error_frames")
    )
    if onset["available"] is not onset_values_available:
        raise ActionPreservationGateError("onset availability/value closure differs")
    if onset["available"] and onset["timing_error_frames"] != abs(
        onset["candidate_frame"] - onset["anchor_frame"]
    ):
        raise ActionPreservationGateError("onset timing error is not frame-auditable")
    if quality["available"] is not (quality["score"] is not None):
        raise ActionPreservationGateError("quality availability/value closure differs")
    action["terminal_hold_score"] = _optional_unit(
        action["terminal_hold_score"], label="terminal hold score"
    )
    for key in (
        "terminal_hold_start_frame",
        "terminal_hold_end_frame",
        "terminal_hold_frames",
    ):
        action[key] = _optional_nonnegative_int(
            action[key], label=key.replace("_", " ")
        )
    action_values_available = all(
        action[key] is not None
        for key in (
            "score",
            "terminal_hold_score",
            "terminal_hold_start_frame",
            "terminal_hold_end_frame",
            "terminal_hold_frames",
        )
    )
    if action["available"] is not action_values_available:
        raise ActionPreservationGateError("action/order availability/value closure differs")
    if action["available"] and (
        action["terminal_hold_end_frame"] < action["terminal_hold_start_frame"]
        or action["terminal_hold_frames"]
        != action["terminal_hold_end_frame"] - action["terminal_hold_start_frame"] + 1
    ):
        raise ActionPreservationGateError("terminal hold is not frame-auditable")
    for key in ("reverse_rejected", "truncation_rejected"):
        if type(action[key]) is not bool:
            raise ActionPreservationGateError(f"{key} is not boolean")
    for label, item in (
        ("face", face),
        ("background", background),
        ("camera", camera),
        ("quality", quality),
        ("onset", onset),
        ("action/order", action),
    ):
        receipt = item["receipt_sha256"]
        if item["available"]:
            _sha(receipt, label=f"{label} receipt")
        elif receipt is not None:
            raise ActionPreservationGateError(f"unavailable {label} has a receipt")
    closure = dict(
        _closed(
            row["input_closure"],
            {
                "target_video_read",
                "anchor_appearance_used_for_preservation",
                "whole_frame_dino_used_as_identity_gate",
                "fixed_source_mask_used_as_background_gate",
                "training_loss_used_as_decoded_gate",
            },
            label="input closure",
        )
    )
    if any(value is not False for value in closure.values()):
        raise ActionPreservationGateError("forbidden preservation shortcut was used")
    _verify_digest(row, field="measurement_digest", label="measurement")
    row.update(
        scope=scope,
        face=face,
        background=background,
        camera=camera,
        quality=quality,
        onset=onset,
        action_order=action,
        input_closure=closure,
    )
    return row


def _axis(state: str, reasons: list[str], values: Mapping[str, Any]) -> dict[str, Any]:
    if state not in STATES:
        raise ActionPreservationGateError("axis state differs")
    return {"state": state, "reasons": reasons, "values": dict(values)}


def decide(measurement: Any, calibration: Any | None = None) -> dict[str, Any]:
    row = validate_measurement(measurement)
    axes: dict[str, dict[str, Any]] = {}
    if calibration is None:
        for name in AXES:
            axes[name] = _axis("undetermined", ["calibration_missing"], {})
        status = "abstain"
        calibration_ref = None
    else:
        authority = validate_calibration(calibration)
        thresholds = authority["thresholds"]
        calibration_ref = {
            "calibration_id": authority["calibration_id"],
            "calibration_digest": authority["calibration_digest"],
        }
        scope = row["scope"]
        face = row["face"]
        if not (scope["single_subject"] and scope["human_subject"]):
            axes["source_identity"] = _axis(
                "undetermined", ["source_identity_scope_unsupported"], face
            )
        elif not (scope["source_face_visible"] and scope["output_face_visible"]):
            axes["source_identity"] = _axis(
                "undetermined", ["source_identity_visibility_insufficient"], face
            )
        elif not face["available"] or face["source_face_pool_size"] <= 0:
            axes["source_identity"] = _axis(
                "undetermined", ["source_identity_measurement_unavailable"], face
            )
        elif face["coverage"] < thresholds["face_coverage_min"]:
            axes["source_identity"] = _axis(
                "undetermined", ["source_identity_coverage_below_calibrated_minimum"], face
            )
        elif face["similarity"] < thresholds["face_similarity_min"]:
            axes["source_identity"] = _axis(
                "fail", ["source_identity_similarity_below_calibrated_threshold"], face
            )
        else:
            axes["source_identity"] = _axis("pass", [], face)

        background = row["background"]
        if not (
            scope["single_subject"]
            and scope["static_or_weak_camera"]
            and scope["no_shot_cut"]
            and scope["background_expected_unchanged"]
        ):
            axes["background"] = _axis(
                "undetermined", ["safe_background_scope_unsupported"], background
            )
        elif not (
            background["available"]
            and background["source_and_output_masks_independent"]
            and background["union_exclusion_used"]
        ):
            axes["background"] = _axis(
                "undetermined", ["safe_background_measurement_unavailable"], background
            )
        elif background["valid_coverage"] < thresholds["background_coverage_min"]:
            axes["background"] = _axis(
                "undetermined", ["background_coverage_below_calibrated_minimum"], background
            )
        elif background["similarity"] < thresholds["background_similarity_min"]:
            axes["background"] = _axis(
                "fail", ["background_similarity_below_calibrated_threshold"], background
            )
        else:
            axes["background"] = _axis("pass", [], background)

        camera = row["camera"]
        if not (scope["static_or_weak_camera"] and scope["no_shot_cut"]):
            axes["camera"] = _axis(
                "undetermined", ["camera_scope_unsupported"], camera
            )
        elif not (camera["available"] and camera["background_registration_used"]):
            axes["camera"] = _axis(
                "undetermined", ["camera_measurement_unavailable"], camera
            )
        else:
            failures = [
                key
                for key, threshold_key in (
                    ("translation", "camera_translation_max"),
                    ("log_scale_abs", "camera_log_scale_abs_max"),
                    ("rotation_degrees_abs", "camera_rotation_degrees_max"),
                    ("reprojection_error", "camera_reprojection_error_max"),
                )
                if camera[key] > thresholds[threshold_key]
            ]
            axes["camera"] = _axis(
                "fail" if failures else "pass",
                [f"camera_{key}_above_calibrated_maximum" for key in failures],
                camera,
            )

        quality = row["quality"]
        if not quality["available"]:
            axes["quality"] = _axis("undetermined", ["quality_unavailable"], quality)
        elif quality["score"] < thresholds["quality_min"]:
            axes["quality"] = _axis("fail", ["quality_below_calibrated_threshold"], quality)
        else:
            axes["quality"] = _axis("pass", [], quality)

        onset = row["onset"]
        if not onset["available"]:
            axes["onset"] = _axis("undetermined", ["onset_unavailable"], onset)
        else:
            onset_failures = []
            if onset["score"] < thresholds["onset_score_min"]:
                onset_failures.append("onset_score_below_calibrated_threshold")
            if (
                onset["timing_error_frames"]
                > thresholds["onset_timing_error_frames_max"]
            ):
                onset_failures.append("onset_timing_error_above_calibrated_maximum")
            axes["onset"] = _axis(
                "fail" if onset_failures else "pass", onset_failures, onset
            )

        action = row["action_order"]
        if not action["available"]:
            axes["action_order"] = _axis(
                "undetermined", ["action_order_unavailable"], action
            )
        else:
            action_failures = []
            if not action["reverse_rejected"] or not action["truncation_rejected"]:
                action_failures.append("reverse_or_truncation_gate_failed")
            if action["score"] < thresholds["action_order_min"]:
                action_failures.append("action_order_below_calibrated_threshold")
            if action["terminal_hold_score"] < thresholds["terminal_hold_score_min"]:
                action_failures.append("terminal_hold_below_calibrated_threshold")
            if action["terminal_hold_frames"] < thresholds["terminal_hold_frames_min"]:
                action_failures.append("terminal_hold_frames_below_calibrated_minimum")
            axes["action_order"] = _axis(
                "fail" if action_failures else "pass", action_failures, action
            )

        axes = {name: axes[name] for name in AXES}
        states = [axes[name]["state"] for name in AXES]
        if "fail" in states:
            status = "reject"
        elif "undetermined" in states:
            status = "abstain"
        else:
            status = "eligible_for_motion_ranking"
    result = {
        "schema_version": DECISION_SCHEMA,
        "candidate_id": row["candidate_id"],
        "candidate_video_sha256": row["candidate_video_sha256"],
        "source_video_sha256": row["source_video_sha256"],
        "measurement_digest": row["measurement_digest"],
        "calibration": calibration_ref,
        "axes": axes,
        "status": status,
        "weighted_score": None,
        "motion_ranking_allowed": status == "eligible_for_motion_ranking",
        "training_promotion_authorized": False,
    }
    result["decision_digest"] = object_sha256(result)
    return validate_decision(result)


_DECISION_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "candidate_video_sha256",
        "source_video_sha256",
        "measurement_digest",
        "calibration",
        "axes",
        "status",
        "weighted_score",
        "motion_ranking_allowed",
        "training_promotion_authorized",
        "decision_digest",
    }
)
_AXIS_VALUE_FIELDS = {
    "action_order": _ACTION_FIELDS,
    "onset": _ONSET_FIELDS,
    "source_identity": _FACE_FIELDS,
    "background": _BACKGROUND_FIELDS,
    "camera": _CAMERA_FIELDS,
    "quality": _QUALITY_FIELDS,
}
_UNDETERMINED_REASONS = {
    "action_order": {"action_order_unavailable"},
    "onset": {"onset_unavailable"},
    "source_identity": {
        "source_identity_scope_unsupported",
        "source_identity_visibility_insufficient",
        "source_identity_measurement_unavailable",
        "source_identity_coverage_below_calibrated_minimum",
    },
    "background": {
        "safe_background_scope_unsupported",
        "safe_background_measurement_unavailable",
        "background_coverage_below_calibrated_minimum",
    },
    "camera": {"camera_scope_unsupported", "camera_measurement_unavailable"},
    "quality": {"quality_unavailable"},
}
_FAIL_REASONS = {
    "action_order": {
        "reverse_or_truncation_gate_failed",
        "action_order_below_calibrated_threshold",
        "terminal_hold_below_calibrated_threshold",
        "terminal_hold_frames_below_calibrated_minimum",
    },
    "onset": {
        "onset_score_below_calibrated_threshold",
        "onset_timing_error_above_calibrated_maximum",
    },
    "source_identity": {"source_identity_similarity_below_calibrated_threshold"},
    "background": {"background_similarity_below_calibrated_threshold"},
    "camera": {
        "camera_translation_above_calibrated_maximum",
        "camera_log_scale_abs_above_calibrated_maximum",
        "camera_rotation_degrees_abs_above_calibrated_maximum",
        "camera_reprojection_error_above_calibrated_maximum",
    },
    "quality": {"quality_below_calibrated_threshold"},
}


def validate_decision(value: Any) -> dict[str, Any]:
    """Validate the complete, digest-bound machine-decision closure."""

    row = dict(_closed(value, _DECISION_FIELDS, label="gate decision"))
    if row["schema_version"] != DECISION_SCHEMA:
        raise ActionPreservationGateError("gate decision schema differs")
    _identifier(row["candidate_id"], label="decision candidate id")
    _sha(row["candidate_video_sha256"], label="decision candidate video")
    _sha(row["source_video_sha256"], label="decision source video")
    _sha(row["measurement_digest"], label="decision measurement")
    _verify_digest(row, field="decision_digest", label="gate decision")

    calibration = row["calibration"]
    if calibration is not None:
        calibration = dict(
            _closed(
                calibration,
                {"calibration_id", "calibration_digest"},
                label="decision calibration reference",
            )
        )
        _identifier(calibration["calibration_id"], label="decision calibration id")
        _sha(calibration["calibration_digest"], label="decision calibration")

    axes_value = row["axes"]
    axes = dict(_closed(axes_value, set(AXES), label="decision axes"))
    normalized_axes: dict[str, dict[str, Any]] = {}
    for axis in AXES:
        item = dict(
            _closed(
                axes[axis],
                {"state", "reasons", "values"},
                label=f"decision axis {axis}",
            )
        )
        state = item["state"]
        if state not in STATES:
            raise ActionPreservationGateError(f"decision axis state differs: {axis}")
        reasons = item["reasons"]
        if (
            not isinstance(reasons, list)
            or any(not isinstance(reason, str) for reason in reasons)
            or len(reasons) != len(set(reasons))
        ):
            raise ActionPreservationGateError(f"decision axis reasons differ: {axis}")
        values = item["values"]
        if calibration is None:
            if state != "undetermined" or reasons != ["calibration_missing"] or values != {}:
                raise ActionPreservationGateError(
                    f"uncalibrated decision axis closure differs: {axis}"
                )
        else:
            if state == "pass":
                if reasons:
                    raise ActionPreservationGateError(
                        f"passing decision axis has reasons: {axis}"
                    )
            else:
                allowed = (
                    _UNDETERMINED_REASONS[axis]
                    if state == "undetermined"
                    else _FAIL_REASONS[axis]
                )
                if not reasons or not set(reasons).issubset(allowed):
                    raise ActionPreservationGateError(
                        f"decision axis reason/state closure differs: {axis}"
                    )
            _closed(values, _AXIS_VALUE_FIELDS[axis], label=f"decision axis values {axis}")
        normalized_axes[axis] = {"state": state, "reasons": reasons, "values": values}

    states = [normalized_axes[axis]["state"] for axis in AXES]
    expected_status = (
        "reject"
        if "fail" in states
        else "abstain"
        if "undetermined" in states
        else "eligible_for_motion_ranking"
    )
    if row["status"] != expected_status:
        raise ActionPreservationGateError("decision status/axis closure differs")
    if row["weighted_score"] is not None:
        raise ActionPreservationGateError("decision uses a weighted score")
    if row["motion_ranking_allowed"] is not (
        expected_status == "eligible_for_motion_ranking"
    ):
        raise ActionPreservationGateError("decision motion-ranking closure differs")
    if row["training_promotion_authorized"] is not False:
        raise ActionPreservationGateError("decision directly authorizes training promotion")
    row["calibration"] = calibration
    row["axes"] = normalized_axes
    return row


def aggregate_reviewer_ballots(reviewers: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Resolve each axis only when all independent ballots exactly agree."""

    return {
        axis: (
            next(iter(states)) if len(states := {item["labels"][axis] for item in reviewers}) == 1
            else "undetermined"
        )
        for axis in AXES
    }


def _validate_action_review_contract(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version", "action_order_description",
        "action_order_description_sha256", "expected_onset_frame_min",
        "expected_onset_frame_max", "terminal_hold_start_frame_min",
        "terminal_hold_end_frame", "full_video_frame_count", "fps_num",
        "fps_den", "contract_digest",
    }
    row = dict(_closed(value, fields, label="action review contract"))
    if row["schema_version"] != ACTION_REVIEW_CONTRACT_SCHEMA:
        raise ActionPreservationGateError("action review contract schema differs")
    description = row["action_order_description"]
    if (
        not isinstance(description, str)
        or not description.strip()
        or description != description.strip()
        or "\x00" in description
        or hashlib.sha256(description.encode("utf-8")).hexdigest()
        != _sha(
            row["action_order_description_sha256"],
            label="action order description",
        )
    ):
        raise ActionPreservationGateError("action order authority differs")
    lowered_description = description.lower()
    if any(token.lower() in lowered_description for token in PRIVATE_REVIEW_TOKENS):
        raise ActionPreservationGateError(
            "action order authority leaks method/arm/checkpoint/policy"
        )
    integer_fields = (
        "expected_onset_frame_min", "expected_onset_frame_max",
        "terminal_hold_start_frame_min", "terminal_hold_end_frame",
        "full_video_frame_count", "fps_num", "fps_den",
    )
    if any(type(row[key]) is not int for key in integer_fields):
        raise ActionPreservationGateError("action timing authority is not integral")
    if not (
        0 <= row["expected_onset_frame_min"]
        <= row["expected_onset_frame_max"]
        < row["terminal_hold_start_frame_min"]
        <= row["terminal_hold_end_frame"] == 80
        and row["full_video_frame_count"] == 81
        and row["fps_num"] == 25
        and row["fps_den"] == 1
    ):
        raise ActionPreservationGateError("action timing authority differs")
    _verify_digest(row, field="contract_digest", label="action review contract")
    return row


def validate_evaluation_aggregate(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version", "evaluation_id", "evaluation_manifest_digest",
        "physical_bindings_digest", "holder_summaries", "holder_count",
        "holder_authority_set_digest",
        "ordered_task_authority_chain_set_digest",
        "candidate_output_count", "matched_base_output_count",
        "total_output_count", "exact_full81_at_25fps_pts_verified",
        "all_native_inference_receipts_verified",
        "all_model_and_adapter_consumption_authority_verified_offline",
        "all_fd_inheritance_evidence_verified",
        "all_consumption_publication_gates_verified",
        "all_outputs_create_only_and_sealed",
        "aggregate_verified_release_capture", "automatic_retry_count",
        "training_loss_read_or_used", "checkpoint_loss_ranking",
        "private_mapping_digest", "public_packet_digest",
        "blinding_key_sha256", "machine_calibration_digest", "machine_status",
        "blind_review_status", "next_action",
        "scientific_promotion_authorized", "aggregate_digest",
    }
    row = dict(_closed(value, fields, label="evaluation aggregate"))
    if row["schema_version"] != EVALUATION_AGGREGATE_SCHEMA:
        raise ActionPreservationGateError("evaluation aggregate schema differs")
    _identifier(row["evaluation_id"], label="evaluation id")
    for key in (
        "evaluation_manifest_digest", "physical_bindings_digest",
        "private_mapping_digest", "public_packet_digest", "blinding_key_sha256",
        "holder_authority_set_digest",
        "ordered_task_authority_chain_set_digest",
    ):
        _sha(row[key], label=key)
    if row["machine_calibration_digest"] is not None:
        _sha(row["machine_calibration_digest"], label="machine calibration")
    aggregate_capture = dict(
        _closed(
            row["aggregate_verified_release_capture"],
            {
                "receipt_path", "receipt_sha256", "capture_digest", "target",
                "target_arguments_sha256",
            },
            label="aggregate verified release capture",
        )
    )
    if (
        not isinstance(aggregate_capture["receipt_path"], str)
        or not Path(aggregate_capture["receipt_path"]).is_absolute()
        or aggregate_capture["target"]
        != "action_preservation_decoded_eval_aggregate_v2.py"
    ):
        raise ActionPreservationGateError(
            "aggregate verified release capture differs"
        )
    for key in ("receipt_sha256", "capture_digest", "target_arguments_sha256"):
        _sha(aggregate_capture[key], label=f"aggregate capture {key}")
    if not isinstance(row["holder_summaries"], list) or len(row["holder_summaries"]) != 4:
        raise ActionPreservationGateError("evaluation aggregate holder closure differs")
    holder_fields = {
        "job_id", "node", "summary_path", "summary_sha256",
        "summary_digest", "holder_execution_digest",
        "executor_verified_release_capture",
        "model_capture_path", "model_capture_sha256",
        "model_capture_digest", "model_final_path", "model_final_sha256",
        "model_final_digest", "task_consumption_set_digest",
        "ordered_chain_digests_digest", "holder_authority_digest",
        "all_task_fd_inheritance_evidence_verified",
    }
    capture_fields = {
        "receipt_path", "receipt_sha256", "capture_digest", "target",
        "target_arguments_sha256",
    }
    normalized_holders = []
    for item in row["holder_summaries"]:
        holder = dict(_closed(item, holder_fields, label="evaluation holder summary"))
        _identifier(holder["job_id"], label="evaluation holder job id")
        _identifier(holder["node"], label="evaluation holder node")
        for key in (
            "summary_path", "model_capture_path", "model_final_path"
        ):
            if not isinstance(holder[key], str) or not Path(holder[key]).is_absolute():
                raise ActionPreservationGateError(
                    "evaluation holder summary path differs"
                )
        for key in (
            "summary_sha256", "summary_digest", "holder_execution_digest",
            "model_capture_sha256", "model_capture_digest",
            "model_final_sha256", "model_final_digest",
            "task_consumption_set_digest", "ordered_chain_digests_digest",
            "holder_authority_digest",
        ):
            _sha(holder[key], label=f"evaluation holder {key}")
        capture = dict(
            _closed(
                holder["executor_verified_release_capture"], capture_fields,
                label="evaluation holder executor capture",
            )
        )
        if (
            not isinstance(capture["receipt_path"], str)
            or not Path(capture["receipt_path"]).is_absolute()
            or capture["target"]
            != "action_preservation_decoded_eval_executor_v2.py"
        ):
            raise ActionPreservationGateError(
                "evaluation holder executor capture differs"
            )
        for key in (
            "receipt_sha256", "capture_digest", "target_arguments_sha256"
        ):
            _sha(capture[key], label=f"evaluation holder executor {key}")
        authority = {
            "job_id": holder["job_id"],
            "model_capture_digest": holder["model_capture_digest"],
            "model_final_digest": holder["model_final_digest"],
            "task_consumption_set_digest": holder[
                "task_consumption_set_digest"
            ],
            "ordered_chain_digests_digest": holder[
                "ordered_chain_digests_digest"
            ],
        }
        if (
            holder["holder_authority_digest"] != object_sha256(authority)
            or holder["all_task_fd_inheritance_evidence_verified"] is not True
        ):
            raise ActionPreservationGateError(
                "evaluation holder consumption authority differs"
            )
        holder["executor_verified_release_capture"] = capture
        normalized_holders.append(holder)
    if (
        len({item["job_id"] for item in normalized_holders}) != 4
        or len({item["node"] for item in normalized_holders}) != 4
        or len({item["summary_path"] for item in normalized_holders}) != 4
        or len(
            {
                item["executor_verified_release_capture"]["capture_digest"]
                for item in normalized_holders
            }
        ) != 4
    ):
        raise ActionPreservationGateError(
            "evaluation holder identities or captures are not unique"
        )
    holder_authorities = [
        {
            "job_id": item["job_id"],
            "model_capture_digest": item["model_capture_digest"],
            "model_final_digest": item["model_final_digest"],
            "task_consumption_set_digest": item[
                "task_consumption_set_digest"
            ],
            "ordered_chain_digests_digest": item[
                "ordered_chain_digests_digest"
            ],
        }
        for item in normalized_holders
    ]
    if row["holder_authority_set_digest"] != object_sha256(
        holder_authorities
    ):
        raise ActionPreservationGateError(
            "evaluation holder authority-set digest differs"
        )
    if (
        row["holder_count"] != 4
        or row["candidate_output_count"] != 256
        or row["matched_base_output_count"] != 8
        or row["total_output_count"] != 264
        or row["exact_full81_at_25fps_pts_verified"] is not True
        or row["all_native_inference_receipts_verified"] is not True
        or row[
            "all_model_and_adapter_consumption_authority_verified_offline"
        ] is not True
        or row["all_fd_inheritance_evidence_verified"] is not True
        or row["all_consumption_publication_gates_verified"] is not True
        or row["all_outputs_create_only_and_sealed"] is not True
        or row["automatic_retry_count"] != 0
        or row["training_loss_read_or_used"] is not False
        or row["checkpoint_loss_ranking"] is not False
        or row["blind_review_status"] != "WAIT_FOR_BLIND_REVIEW"
        or row["next_action"] != "WAIT_FOR_BLIND_REVIEW"
        or row["scientific_promotion_authorized"] is not False
    ):
        raise ActionPreservationGateError("evaluation aggregate authority differs")
    expected_machine = (
        "ABSTAIN_CALIBRATION_MISSING"
        if row["machine_calibration_digest"] is None
        else "WAIT_FOR_MACHINE_MEASUREMENT"
    )
    if row["machine_status"] != expected_machine:
        raise ActionPreservationGateError("evaluation machine-status closure differs")
    _verify_digest(row, field="aggregate_digest", label="evaluation aggregate")
    row["aggregate_verified_release_capture"] = aggregate_capture
    row["holder_summaries"] = normalized_holders
    return row


_PUBLIC_ROW_FIELDS = frozenset(
    {
        "blind_candidate_id", "source_media_sha256", "source_receipt_sha256",
        "source_media_relpath", "review_media_sha256", "review_media_relpath",
        "review_output_digest", "full_video_receipt_sha256",
        "matched_base_media_sha256", "matched_base_media_relpath",
        "matched_base_output_digest", "matched_base_full_video_receipt_sha256",
        "instruction", "instruction_sha256", "action_review_contract",
        "action_review_contract_digest", "required_axes",
        "minimum_independent_reviewer_count", "full_81_frame_video_required",
        "blind_row_digest",
    }
)


def _validate_public_row(value: Any, *, index: int) -> dict[str, Any]:
    row = dict(_closed(value, _PUBLIC_ROW_FIELDS, label=f"public blind row {index}"))
    _identifier(row["blind_candidate_id"], label="blind candidate id")
    for key in (
        "source_media_sha256", "source_receipt_sha256", "review_media_sha256",
        "review_output_digest", "full_video_receipt_sha256",
        "matched_base_media_sha256", "matched_base_output_digest",
        "matched_base_full_video_receipt_sha256", "instruction_sha256",
        "action_review_contract_digest",
    ):
        _sha(row[key], label=f"public blind row {key}")
    instruction = row["instruction"]
    if (
        not isinstance(instruction, str)
        or not instruction.strip()
        or instruction != instruction.strip()
        or "\x00" in instruction
        or hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        != row["instruction_sha256"]
    ):
        raise ActionPreservationGateError("public blind instruction differs")
    if any(token.lower() in instruction.lower() for token in PRIVATE_REVIEW_TOKENS):
        raise ActionPreservationGateError(
            "public blind instruction leaks method/arm/checkpoint/policy"
        )
    contract = _validate_action_review_contract(row["action_review_contract"])
    if contract["contract_digest"] != row["action_review_contract_digest"]:
        raise ActionPreservationGateError("public blind action contract differs")
    if (
        row["source_media_relpath"]
        != f"media/{row['source_media_sha256']}.mp4"
        or row["review_media_relpath"]
        != f"media/{row['review_media_sha256']}.mp4"
        or row["matched_base_media_relpath"]
        != f"media/{row['matched_base_media_sha256']}.mp4"
        or row["required_axes"] != list(AXES)
        or row["minimum_independent_reviewer_count"] != 2
        or row["full_81_frame_video_required"] is not True
    ):
        raise ActionPreservationGateError("public blind row review closure differs")
    _verify_digest(row, field="blind_row_digest", label="public blind row")
    row["action_review_contract"] = contract
    return row


def validate_public_packet(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version", "packet_id", "review_contract_digest",
        "private_mapping_digest", "rows", "row_count", "method_hidden",
        "arm_hidden", "checkpoint_hidden", "onset_policy_hidden",
        "private_key_in_public_packet", "training_loss_present",
        "public_packet_digest",
    }
    row = dict(_closed(value, fields, label="public blind packet"))
    if row["schema_version"] != PUBLIC_PACKET_SCHEMA:
        raise ActionPreservationGateError("public blind packet schema differs")
    _identifier(row["packet_id"], label="public packet id")
    _sha(row["review_contract_digest"], label="review contract")
    _sha(row["private_mapping_digest"], label="private mapping")
    if not isinstance(row["rows"], list) or len(row["rows"]) != 256:
        raise ActionPreservationGateError("public blind packet row count differs")
    rows = [_validate_public_row(item, index=index) for index, item in enumerate(row["rows"])]
    ids = [item["blind_candidate_id"] for item in rows]
    digests = [item["blind_row_digest"] for item in rows]
    if len(set(ids)) != 256 or len(set(digests)) != 256:
        raise ActionPreservationGateError("public blind row identifiers are not unique")
    if (
        row["row_count"] != 256
        or row["method_hidden"] is not True
        or row["arm_hidden"] is not True
        or row["checkpoint_hidden"] is not True
        or row["onset_policy_hidden"] is not True
        or row["private_key_in_public_packet"] is not False
        or row["training_loss_present"] is not False
    ):
        raise ActionPreservationGateError("public blind packet authority differs")
    _verify_digest(row, field="public_packet_digest", label="public blind packet")
    row["rows"] = rows
    return row


def _public_media_sha256_set(value: Any) -> frozenset[str]:
    """Return the exact content-addressed media namespace named by a packet.

    Aggregate publication de-duplicates media by SHA-256.  In particular, the
    four source videos and eight matched-base controls are referenced by many
    public rows, so ``total_output_count`` is not the media-directory count.
    """
    public = validate_public_packet(value)
    return frozenset(
        row[field]
        for row in public["rows"]
        for field in (
            "source_media_sha256",
            "review_media_sha256",
            "matched_base_media_sha256",
        )
    )


_PRIVATE_ROW_FIELDS = frozenset(
    {
        "blind_candidate_id", "blind_row_digest", "order_digest", "candidate_id",
        "arm", "checkpoint_step", "iid", "onset_policy", "matched_control_id",
        "candidate_output_path", "candidate_output_receipt_path",
        "candidate_output_receipt_sha256", "candidate_output_digest",
        "matched_base_output_receipt_path",
        "matched_base_output_receipt_sha256", "matched_base_output_digest",
        "instruction_sha256", "action_review_contract_digest",
        "private_row_digest",
    }
)


def _validate_private_row(value: Any, *, index: int) -> dict[str, Any]:
    row = dict(_closed(value, _PRIVATE_ROW_FIELDS, label=f"private blind row {index}"))
    for key in (
        "blind_candidate_id", "candidate_id", "arm", "iid", "onset_policy",
        "matched_control_id",
    ):
        _identifier(row[key], label=f"private blind row {key}")
    if type(row["checkpoint_step"]) is not int or row["checkpoint_step"] < 0:
        raise ActionPreservationGateError("private checkpoint step differs")
    for key in (
        "blind_row_digest", "order_digest", "candidate_output_receipt_sha256",
        "candidate_output_digest", "matched_base_output_receipt_sha256",
        "matched_base_output_digest", "instruction_sha256",
        "action_review_contract_digest",
    ):
        _sha(row[key], label=f"private blind row {key}")
    for key in (
        "candidate_output_path", "candidate_output_receipt_path",
        "matched_base_output_receipt_path",
    ):
        if not isinstance(row[key], str) or not Path(row[key]).is_absolute():
            raise ActionPreservationGateError(f"private blind row {key} differs")
    _verify_digest(row, field="private_row_digest", label="private blind row")
    return row


def validate_private_packet(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version", "evaluation_id", "evaluation_manifest_digest",
        "blinding_key_sha256", "rows", "row_count",
        "method_arm_checkpoint_policy_private", "private_mapping_digest",
    }
    row = dict(_closed(value, fields, label="private blind mapping"))
    if row["schema_version"] != PRIVATE_PACKET_SCHEMA:
        raise ActionPreservationGateError("private blind mapping schema differs")
    _identifier(row["evaluation_id"], label="private evaluation id")
    _sha(row["evaluation_manifest_digest"], label="private evaluation manifest")
    _sha(row["blinding_key_sha256"], label="private blinding key")
    if not isinstance(row["rows"], list) or len(row["rows"]) != 256:
        raise ActionPreservationGateError("private blind mapping row count differs")
    rows = [_validate_private_row(item, index=index) for index, item in enumerate(row["rows"])]
    if (
        len({item["blind_candidate_id"] for item in rows}) != 256
        or len({item["candidate_id"] for item in rows}) != 256
        or len({item["order_digest"] for item in rows}) != 256
        or row["row_count"] != 256
        or row["method_arm_checkpoint_policy_private"] is not True
    ):
        raise ActionPreservationGateError("private blind mapping closure differs")
    _verify_digest(row, field="private_mapping_digest", label="private blind mapping")
    row["rows"] = rows
    return row


_BALLOT_BINDING_FIELDS = (
    "source_video_sha256", "candidate_video_sha256", "matched_base_video_sha256",
    "candidate_output_digest", "matched_base_output_digest",
    "full_video_receipt_sha256", "matched_base_full_video_receipt_sha256",
    "instruction_sha256", "action_review_contract_digest",
)


def validate_blind_ballot(
    value: Any, *, public_packet: Mapping[str, Any]
) -> dict[str, Any]:
    packet = validate_public_packet(public_packet)
    fields = {
        "schema_version", "public_packet_digest", "blind_candidate_id",
        "blind_row_digest", "reviewer_id", "independent_review",
        "full_video_reviewed", "labels", "ballot_digest",
        *_BALLOT_BINDING_FIELDS,
    }
    row = dict(_closed(value, fields, label="blind reviewer ballot"))
    if row["schema_version"] != BLIND_BALLOT_SCHEMA:
        raise ActionPreservationGateError("blind reviewer ballot schema differs")
    if row["public_packet_digest"] != packet["public_packet_digest"]:
        raise ActionPreservationGateError("ballot public packet binding differs")
    _identifier(row["blind_candidate_id"], label="ballot blind candidate id")
    _identifier(row["reviewer_id"], label="ballot reviewer id")
    if row["independent_review"] is not True:
        raise ActionPreservationGateError("blind reviewer was not independent")
    if row["full_video_reviewed"] is not True:
        raise ActionPreservationGateError("blind reviewer did not review the full video")
    labels = dict(_closed(row["labels"], set(AXES), label="blind ballot labels"))
    for axis, state in labels.items():
        if state not in STATES:
            raise ActionPreservationGateError(f"blind label differs: {axis}")
    matching = [
        item for item in packet["rows"]
        if item["blind_candidate_id"] == row["blind_candidate_id"]
    ]
    if len(matching) != 1:
        raise ActionPreservationGateError("ballot blind candidate is not in public packet")
    public_row = matching[0]
    expected_bindings = {
        "source_video_sha256": public_row["source_media_sha256"],
        "candidate_video_sha256": public_row["review_media_sha256"],
        "matched_base_video_sha256": public_row["matched_base_media_sha256"],
        "candidate_output_digest": public_row["review_output_digest"],
        "matched_base_output_digest": public_row["matched_base_output_digest"],
        "full_video_receipt_sha256": public_row["full_video_receipt_sha256"],
        "matched_base_full_video_receipt_sha256": public_row[
            "matched_base_full_video_receipt_sha256"
        ],
        "instruction_sha256": public_row["instruction_sha256"],
        "action_review_contract_digest": public_row[
            "action_review_contract_digest"
        ],
    }
    if (
        row["blind_row_digest"] != public_row["blind_row_digest"]
        or any(row[key] != expected for key, expected in expected_bindings.items())
    ):
        raise ActionPreservationGateError("ballot public blind-row binding differs")
    _verify_digest(row, field="ballot_digest", label="blind reviewer ballot")
    row["labels"] = labels
    return row


def build_blind_ballot(
    *, public_packet: Mapping[str, Any], blind_candidate_id: str,
    reviewer_id: str, labels: Mapping[str, str]
) -> dict[str, Any]:
    packet = validate_public_packet(public_packet)
    matches = [
        item for item in packet["rows"]
        if item["blind_candidate_id"] == blind_candidate_id
    ]
    if len(matches) != 1:
        raise ActionPreservationGateError("blind candidate is not in public packet")
    public_row = matches[0]
    ballot: dict[str, Any] = {
        "schema_version": BLIND_BALLOT_SCHEMA,
        "public_packet_digest": packet["public_packet_digest"],
        "blind_candidate_id": public_row["blind_candidate_id"],
        "blind_row_digest": public_row["blind_row_digest"],
        "source_video_sha256": public_row["source_media_sha256"],
        "candidate_video_sha256": public_row["review_media_sha256"],
        "matched_base_video_sha256": public_row["matched_base_media_sha256"],
        "candidate_output_digest": public_row["review_output_digest"],
        "matched_base_output_digest": public_row["matched_base_output_digest"],
        "full_video_receipt_sha256": public_row["full_video_receipt_sha256"],
        "matched_base_full_video_receipt_sha256": public_row[
            "matched_base_full_video_receipt_sha256"
        ],
        "instruction_sha256": public_row["instruction_sha256"],
        "action_review_contract_digest": public_row[
            "action_review_contract_digest"
        ],
        "reviewer_id": reviewer_id,
        "independent_review": True,
        "full_video_reviewed": True,
        "labels": dict(labels),
    }
    ballot["ballot_digest"] = object_sha256(ballot)
    return validate_blind_ballot(ballot, public_packet=packet)


def _closed_packet_authority(
    *, evaluation_aggregate: Mapping[str, Any], public_packet: Mapping[str, Any],
    private_mapping: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    aggregate = validate_evaluation_aggregate(evaluation_aggregate)
    public = validate_public_packet(public_packet)
    private = validate_private_packet(private_mapping)
    if (
        aggregate["public_packet_digest"] != public["public_packet_digest"]
        or aggregate["private_mapping_digest"] != private["private_mapping_digest"]
        or public["private_mapping_digest"] != private["private_mapping_digest"]
        or aggregate["evaluation_id"] != private["evaluation_id"]
        or aggregate["evaluation_manifest_digest"]
        != private["evaluation_manifest_digest"]
        or aggregate["blinding_key_sha256"] != private["blinding_key_sha256"]
    ):
        raise ActionPreservationGateError("aggregate/public/private packet binding differs")
    public_by_id = {item["blind_candidate_id"]: item for item in public["rows"]}
    for private_row in private["rows"]:
        public_row = public_by_id.get(private_row["blind_candidate_id"])
        if public_row is None:
            raise ActionPreservationGateError("private row is absent from public packet")
        if (
            private_row["blind_row_digest"] != public_row["blind_row_digest"]
            or private_row["candidate_output_receipt_sha256"]
            != public_row["full_video_receipt_sha256"]
            or private_row["candidate_output_digest"]
            != public_row["review_output_digest"]
            or private_row["matched_base_output_receipt_sha256"]
            != public_row["matched_base_full_video_receipt_sha256"]
            or private_row["matched_base_output_digest"]
            != public_row["matched_base_output_digest"]
            or private_row["instruction_sha256"] != public_row["instruction_sha256"]
            or private_row["action_review_contract_digest"]
            != public_row["action_review_contract_digest"]
        ):
            raise ActionPreservationGateError("public/private blind-row binding differs")
    return aggregate, public, private


def build_blind_review(
    *, decision: Mapping[str, Any], evaluation_aggregate: Mapping[str, Any],
    public_packet: Mapping[str, Any], private_mapping: Mapping[str, Any],
    ballots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    authority = validate_decision(decision)
    aggregate, public, private = _closed_packet_authority(
        evaluation_aggregate=evaluation_aggregate,
        public_packet=public_packet,
        private_mapping=private_mapping,
    )
    decision_calibration_digest = (
        None
        if authority["calibration"] is None
        else authority["calibration"]["calibration_digest"]
    )
    if aggregate["machine_calibration_digest"] != decision_calibration_digest:
        raise ActionPreservationGateError(
            "machine decision calibration differs from evaluation aggregate"
        )
    private_matches = [
        item for item in private["rows"]
        if item["candidate_id"] == authority["candidate_id"]
    ]
    if len(private_matches) != 1:
        raise ActionPreservationGateError("machine candidate is absent from private mapping")
    private_row = private_matches[0]
    public_matches = [
        item for item in public["rows"]
        if item["blind_candidate_id"] == private_row["blind_candidate_id"]
    ]
    if len(public_matches) != 1:
        raise ActionPreservationGateError("mapped candidate is absent from public packet")
    public_row = public_matches[0]
    if (
        authority["source_video_sha256"] != public_row["source_media_sha256"]
        or authority["candidate_video_sha256"] != public_row["review_media_sha256"]
    ):
        raise ActionPreservationGateError("machine decision/public row binding differs")
    if not isinstance(ballots, Sequence) or isinstance(ballots, (str, bytes)):
        raise ActionPreservationGateError("review ballots are not a sequence")
    reviewers = [validate_blind_ballot(item, public_packet=public) for item in ballots]
    if len(reviewers) < 2:
        raise ActionPreservationGateError("blind review requires at least two reviewers")
    reviewers.sort(key=lambda item: item["reviewer_id"])
    reviewer_ids = [item["reviewer_id"] for item in reviewers]
    if len(reviewer_ids) != len(set(reviewer_ids)):
        raise ActionPreservationGateError("blind reviewer ids are not unique")
    if any(
        item["blind_candidate_id"] != public_row["blind_candidate_id"]
        for item in reviewers
    ):
        raise ActionPreservationGateError("review ballots span multiple candidates")
    resolution = aggregate_reviewer_ballots(reviewers)
    ballot_closure = {
        "reviewer_ids": reviewer_ids,
        "ballot_digests": [item["ballot_digest"] for item in reviewers],
    }
    review: dict[str, Any] = {
        "schema_version": BLIND_REVIEW_SCHEMA,
        "evaluation_aggregate_digest": aggregate["aggregate_digest"],
        "evaluation_manifest_digest": aggregate["evaluation_manifest_digest"],
        "public_packet_digest": public["public_packet_digest"],
        "private_mapping_digest": private["private_mapping_digest"],
        "blind_candidate_id": public_row["blind_candidate_id"],
        "blind_row_digest": public_row["blind_row_digest"],
        "private_row_digest": private_row["private_row_digest"],
        "source_video_sha256": public_row["source_media_sha256"],
        "source_receipt_sha256": public_row["source_receipt_sha256"],
        "candidate_video_sha256": public_row["review_media_sha256"],
        "matched_base_video_sha256": public_row["matched_base_media_sha256"],
        "candidate_output_digest": public_row["review_output_digest"],
        "matched_base_output_digest": public_row["matched_base_output_digest"],
        "full_video_receipt_sha256": public_row["full_video_receipt_sha256"],
        "matched_base_full_video_receipt_sha256": public_row[
            "matched_base_full_video_receipt_sha256"
        ],
        "instruction_sha256": public_row["instruction_sha256"],
        "action_review_contract_digest": public_row[
            "action_review_contract_digest"
        ],
        "decision_digest": authority["decision_digest"],
        "method_hidden": True,
        "reviewers": reviewers,
        "axis_resolution": resolution,
        "ballot_closure_digest": object_sha256(ballot_closure),
    }
    review["review_digest"] = object_sha256(review)
    return review


def validate_blind_review(
    value: Any, *, decision: Mapping[str, Any],
    evaluation_aggregate: Mapping[str, Any], public_packet: Mapping[str, Any],
    private_mapping: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema_version", "evaluation_aggregate_digest",
        "evaluation_manifest_digest", "public_packet_digest",
        "private_mapping_digest", "blind_candidate_id", "blind_row_digest",
        "private_row_digest", "source_video_sha256", "source_receipt_sha256",
        "candidate_video_sha256", "matched_base_video_sha256",
        "candidate_output_digest", "matched_base_output_digest",
        "full_video_receipt_sha256", "matched_base_full_video_receipt_sha256",
        "instruction_sha256", "action_review_contract_digest", "decision_digest",
        "method_hidden", "reviewers", "axis_resolution",
        "ballot_closure_digest", "review_digest",
    }
    row = dict(_closed(value, fields, label="blind review"))
    if row["schema_version"] != BLIND_REVIEW_SCHEMA:
        raise ActionPreservationGateError("blind review schema differs")
    if row["method_hidden"] is not True:
        raise ActionPreservationGateError("review method was not blinded")
    _verify_digest(row, field="review_digest", label="blind review")
    expected = build_blind_review(
        decision=decision,
        evaluation_aggregate=evaluation_aggregate,
        public_packet=public_packet,
        private_mapping=private_mapping,
        ballots=row["reviewers"],
    )
    if row != expected:
        raise ActionPreservationGateError(
            "blind review differs from aggregate/public/private ballot authority"
        )
    return row


def promotion_decision(
    decision: Any, blind_review: Any, *, evaluation_aggregate: Mapping[str, Any],
    public_packet: Mapping[str, Any], private_mapping: Mapping[str, Any],
) -> dict[str, Any]:
    authority = validate_decision(decision)
    review = validate_blind_review(
        blind_review,
        decision=authority,
        evaluation_aggregate=evaluation_aggregate,
        public_packet=public_packet,
        private_mapping=private_mapping,
    )
    labels = review["axis_resolution"]
    any_fail = any(labels[axis] == "fail" for axis in AXES)
    any_unknown = any(labels[axis] == "undetermined" for axis in AXES)
    machine_status = authority["status"]
    if any_fail or machine_status == "reject":
        status = "stop_and_rollback"
    elif any_unknown or machine_status == "abstain":
        status = "hold_for_more_evidence"
    elif machine_status == "eligible_for_motion_ranking" and all(
        labels[axis] == "pass" for axis in AXES
    ):
        status = "eligible_for_next_20_update_stage"
    else:
        status = "hold_for_more_evidence"
    result = {
        "schema_version": PROMOTION_SCHEMA,
        "candidate_id": authority["candidate_id"],
        "candidate_video_sha256": authority["candidate_video_sha256"],
        "decision_digest": authority["decision_digest"],
        "evaluation_aggregate_digest": review["evaluation_aggregate_digest"],
        "public_packet_digest": review["public_packet_digest"],
        "private_mapping_digest": review["private_mapping_digest"],
        "review_digest": review["review_digest"],
        "status": status,
        "automatic_model_update": False,
        "requires_fresh_create_only_training_stage": status
        == "eligible_for_next_20_update_stage",
    }
    result["promotion_digest"] = object_sha256(result)
    return result


def _load(path: str, *, label: str) -> Any:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ActionPreservationGateError(f"{label} root is not an object")
    return value


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_uid),
        int(value.st_gid), int(value.st_mode), int(value.st_nlink),
        int(value.st_rdev), int(value.st_size),
        int(getattr(value, "st_blocks", 0)), int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _file_identity_value(value: os.stat_result) -> dict[str, int]:
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "uid": int(value.st_uid),
        "gid": int(value.st_gid),
        "mode": int(value.st_mode),
        "nlink": int(value.st_nlink),
        "rdev": int(value.st_rdev),
        "size": int(value.st_size),
        "blocks": int(getattr(value, "st_blocks", 0)),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _read_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _stable_expected_bytes(
    path: str, expected_sha256: str, *, label: str,
) -> bytes:
    artifact = Path(path)
    expected = _sha(expected_sha256, label=f"{label} SHA")
    if (
        not artifact.is_absolute()
        or os.path.normpath(str(artifact)) != str(artifact)
        or artifact.name in ("", ".", "..")
    ):
        raise ActionPreservationGateError(f"{label} path is not canonical absolute")
    if not hasattr(os, "O_NOFOLLOW"):
        raise ActionPreservationGateError(
            f"{label} safe descriptor capture is unavailable"
        )
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        parent_fd = os.open(str(artifact.parent), directory_flags)
    except OSError as error:
        raise ActionPreservationGateError(
            f"cannot open held parent for {label}"
        ) from error
    descriptor: int | None = None
    try:
        os.set_inheritable(parent_fd, False)
        parent_before = os.fstat(parent_fd)
        descriptor = os.open(
            artifact.name,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        os.set_inheritable(descriptor, False)
        before = os.fstat(descriptor)
        first = _read_fd(descriptor)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor)
        after = os.fstat(descriptor)
        named = os.stat(
            artifact.name, dir_fd=parent_fd, follow_symlinks=False
        )
        parent_after = os.fstat(parent_fd)
        named_parent = artifact.parent.lstat()
    except OSError as error:
        raise ActionPreservationGateError(
            f"cannot capture {label} through held descriptors"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or _file_identity(parent_before) != _file_identity(parent_after)
        or _file_identity(parent_before) != _file_identity(named_parent)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_ISLNK(named.st_mode)
        or _file_identity(before) != _file_identity(middle)
        or _file_identity(before) != _file_identity(after)
        or _file_identity(before) != _file_identity(named)
        or first != second
        or len(first) != before.st_size
    ):
        raise ActionPreservationGateError(
            f"{label} changed during held same-FD double read or has a hard link"
        )
    if hashlib.sha256(first).hexdigest() != expected:
        raise ActionPreservationGateError(f"{label} SHA differs")
    return first


def _load_expected(path: str, expected_sha256: str, *, label: str) -> dict[str, Any]:
    raw = _stable_expected_bytes(path, expected_sha256, label=label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ActionPreservationGateError(f"cannot decode {label}: {error}") from error
    if not isinstance(value, Mapping):
        raise ActionPreservationGateError(f"{label} root is not an object")
    if raw != canonical_json_bytes(value) + b"\n":
        raise ActionPreservationGateError(f"{label} is not canonical newline JSON")
    return dict(value)


def _load_gate_work_root_authority() -> dict[str, Any]:
    has_work_root = os.environ.get(WORK_ROOT_BINDING_ENV) is not None
    has_task_fds = os.environ.get(TASK_FD_BINDING_ENV) is not None
    if not has_work_root:
        raise ActionPreservationGateError(
            "inherited WORK_ROOT A authority is absent"
        )
    if has_task_fds:
        raise ActionPreservationGateError(
            "mixed WORK_ROOT A and task-FD B authorities are forbidden"
        )
    import action_preservation_decoded_eval_verified_release_v1 as verified_release

    try:
        row = verified_release.load_inherited_work_root_environment(
            verify_open_fds=True,
            expected_inheritable=False,
            verify_entries=False,
            allow_root_metadata_change=True,
        )
    except verified_release.DecodedEvalVerifiedReleaseError as error:
        raise ActionPreservationGateError(str(error)) from error
    if row["target"] != GATE_RUNTIME_TARGET:
        raise ActionPreservationGateError(
            "inherited WORK_ROOT A target is not the gate"
        )
    return row


def _replay_gate_work_root_authority(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    return _replay_eval_work_root_authority(
        value, expected_target=GATE_RUNTIME_TARGET
    )


def _replay_eval_work_root_authority(
    value: Mapping[str, Any], *, expected_target: str,
) -> dict[str, Any]:
    import action_preservation_decoded_eval_verified_release_v1 as verified_release

    if expected_target not in {GATE_RUNTIME_TARGET, LOOP_RUNTIME_TARGET}:
        raise ActionPreservationGateError(
            "aggregate replay WORK_ROOT target differs"
        )
    try:
        row = verified_release.validate_inherited_work_root_binding(
            value,
            verify_open_fds=True,
            expected_inheritable=False,
            verify_entries=False,
            allow_root_metadata_change=True,
        )
    except verified_release.DecodedEvalVerifiedReleaseError as error:
        raise ActionPreservationGateError(str(error)) from error
    if row["target"] != expected_target:
        raise ActionPreservationGateError(
            "inherited WORK_ROOT A target differs during aggregate replay"
        )
    return row


def _read_aggregate_anchor_member(
    *, root_fd: int, root_path: Path, binding: Mapping[str, Any],
    supplied_path: str, supplied_sha256: str,
    supplied_value: Mapping[str, Any], label: str,
) -> bytes:
    relative = binding["relative_path"]
    expected_path = root_path / relative
    if (
        Path(supplied_path) != expected_path
        or supplied_sha256 != binding["sha256"]
    ):
        raise ActionPreservationGateError(
            f"{label} path/SHA differs from dynamic completion anchor"
        )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            relative,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=root_fd,
        )
        os.set_inheritable(descriptor, False)
        before = os.fstat(descriptor)
        first = _read_fd(descriptor)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor)
        after = os.fstat(descriptor)
        named = os.stat(relative, dir_fd=root_fd, follow_symlinks=False)
    except OSError as error:
        raise ActionPreservationGateError(
            f"cannot replay {label} through aggregate root FD"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    expected_raw = canonical_json_bytes(supplied_value) + b"\n"
    identity = binding["identity"]
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != binding["mode"]
        or _file_identity_value(before) != identity
        or _file_identity_value(middle) != identity
        or _file_identity_value(after) != identity
        or _file_identity_value(named) != identity
        or first != second
        or first != expected_raw
        or len(first) != binding["size"]
        or hashlib.sha256(first).hexdigest() != binding["sha256"]
    ):
        raise ActionPreservationGateError(
            f"{label} differs from dynamic completion anchor inode"
        )
    return first


def _replay_aggregate_completion_publication(
    *, completion_anchor: Mapping[str, Any],
    work_root: Mapping[str, Any],
    aggregate_path: str, aggregate_sha256: str,
    aggregate: Mapping[str, Any],
    public_path: str, public_sha256: str, public: Mapping[str, Any],
    private_path: str, private_sha256: str, private: Mapping[str, Any],
    expected_media_sha256: frozenset[str],
    expected_work_root_target: str,
) -> None:
    """Replay the exact online-anchored aggregate tree through held A.

    The controller's dynamic anchor is useful only if the later gate proves
    that the names it consumes still resolve to those exact inodes.  All
    aggregate files and media are therefore opened relative to the inherited
    WORK_ROOT descriptor and rehashed while their directory descriptors stay
    live.
    """

    root_path = Path(completion_anchor["aggregate_root"])
    work_path = Path(work_root["path"])
    if root_path.parent != work_path or root_path.name in ("", ".", ".."):
        raise ActionPreservationGateError(
            "aggregate root is not a direct inherited WORK_ROOT child"
        )
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ActionPreservationGateError(
            "safe aggregate descriptor replay is unavailable"
        )
    _replay_eval_work_root_authority(
        work_root, expected_target=expected_work_root_target
    )
    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    root_fd: int | None = None
    media_fd: int | None = None
    try:
        root_fd = os.open(
            root_path.name, directory_flags, dir_fd=work_root["root_fd"]
        )
        os.set_inheritable(root_fd, False)
        root_identity = completion_anchor["aggregate_root_identity"]
        named_root = os.stat(
            root_path.name,
            dir_fd=work_root["root_fd"],
            follow_symlinks=False,
        )
        if (
            _file_identity_value(os.fstat(root_fd)) != root_identity
            or _file_identity_value(named_root) != root_identity
            or set(os.listdir(root_fd))
            != {
                "evaluation_complete.json",
                "private_blind_mapping.json",
                "blind_review_packet.json",
                "media",
            }
        ):
            raise ActionPreservationGateError(
                "aggregate root differs from dynamic completion anchor"
            )
        _read_aggregate_anchor_member(
            root_fd=root_fd,
            root_path=root_path,
            binding=completion_anchor["aggregate_file"],
            supplied_path=aggregate_path,
            supplied_sha256=aggregate_sha256,
            supplied_value=aggregate,
            label="evaluation aggregate",
        )
        _read_aggregate_anchor_member(
            root_fd=root_fd,
            root_path=root_path,
            binding=completion_anchor["public_file"],
            supplied_path=public_path,
            supplied_sha256=public_sha256,
            supplied_value=public,
            label="public blind packet",
        )
        _read_aggregate_anchor_member(
            root_fd=root_fd,
            root_path=root_path,
            binding=completion_anchor["private_file"],
            supplied_path=private_path,
            supplied_sha256=private_sha256,
            supplied_value=private,
            label="private blind mapping",
        )
        media_fd = os.open("media", directory_flags, dir_fd=root_fd)
        os.set_inheritable(media_fd, False)
        media_identity = completion_anchor["media_directory_identity"]
        named_media = os.stat("media", dir_fd=root_fd, follow_symlinks=False)
        expected_names = sorted(
            f"{digest}.mp4" for digest in expected_media_sha256
        )
        if (
            _file_identity_value(os.fstat(media_fd)) != media_identity
            or _file_identity_value(named_media) != media_identity
            or sorted(os.listdir(media_fd)) != expected_names
        ):
            raise ActionPreservationGateError(
                "aggregate media namespace differs from dynamic completion anchor"
            )
        media_rows: list[dict[str, Any]] = []
        for name in expected_names:
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=media_fd,
                )
                os.set_inheritable(descriptor, False)
                before = os.fstat(descriptor)
                first = _read_fd(descriptor)
                middle = os.fstat(descriptor)
                second = _read_fd(descriptor)
                after = os.fstat(descriptor)
                named = os.stat(name, dir_fd=media_fd, follow_symlinks=False)
            except OSError as error:
                raise ActionPreservationGateError(
                    "cannot replay aggregate media through held directory FD"
                ) from error
            finally:
                if descriptor is not None:
                    os.close(descriptor)
            identity = _file_identity_value(before)
            expected_sha = name[:-4]
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != 0o444
                or identity != _file_identity_value(middle)
                or identity != _file_identity_value(after)
                or identity != _file_identity_value(named)
                or first != second
                or hashlib.sha256(first).hexdigest() != expected_sha
            ):
                raise ActionPreservationGateError(
                    "aggregate media differs from dynamic completion anchor"
                )
            media_rows.append(
                {
                    "relative_path": f"media/{name}",
                    "sha256": expected_sha,
                    "size": len(first),
                    "mode": 0o444,
                    "identity": identity,
                }
            )
        if (
            object_sha256(media_rows)
            != completion_anchor["media_rows_digest"]
            or len(media_rows) != completion_anchor["media_file_count"]
            or object_sha256(
                {
                    "media_directory_identity": media_identity,
                    "media_file_count": len(media_rows),
                    "media_rows_digest": object_sha256(media_rows),
                }
            ) != completion_anchor["media_tree_digest"]
            or _file_identity_value(os.fstat(media_fd)) != media_identity
            or _file_identity_value(
                os.stat("media", dir_fd=root_fd, follow_symlinks=False)
            ) != media_identity
            or _file_identity_value(os.fstat(root_fd)) != root_identity
            or _file_identity_value(
                os.stat(
                    root_path.name,
                    dir_fd=work_root["root_fd"],
                    follow_symlinks=False,
                )
            ) != root_identity
        ):
            raise ActionPreservationGateError(
                "aggregate tree final replay differs from dynamic completion anchor"
            )
    finally:
        if media_fd is not None:
            os.close(media_fd)
        if root_fd is not None:
            os.close(root_fd)
        _replay_eval_work_root_authority(
            work_root, expected_target=expected_work_root_target
        )


def _verify_cli_eval_release(
    *, physical_bindings_path: str, physical_bindings_sha256: str,
    evaluation_aggregate: Mapping[str, Any],
    public_packet: Mapping[str, Any],
    private_mapping: Mapping[str, Any],
    evaluation_aggregate_path: str,
    evaluation_aggregate_sha256: str,
    public_packet_path: str,
    public_packet_sha256: str,
    private_mapping_path: str,
    private_mapping_sha256: str,
    aggregate_completion_anchor_raw: str,
    work_root: Mapping[str, Any],
    expected_work_root_target: str,
    running_target_arguments: Sequence[str] | None = None,
) -> dict[str, Any]:
    # Delayed import avoids the plan -> gate import cycle during library use.
    import action_preservation_decoded_eval_bridge_v1 as eval_bridge

    try:
        bindings = eval_bridge.load_physical_bindings(
            physical_bindings_path,
            expected_sha256=physical_bindings_sha256,
            verify_files=True,
        )
        aggregate = validate_evaluation_aggregate(evaluation_aggregate)
        public = validate_public_packet(public_packet)
        private = validate_private_mapping(private_mapping)
        expected_media_sha256 = _public_media_sha256_set(public)
        import action_preservation_decoded_eval_verified_release_v1 as runtime

        def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in items:
                if key in result:
                    raise ActionPreservationGateError(
                        "aggregate completion anchor contains a duplicate key"
                    )
                result[key] = item
            return result

        try:
            anchor_value = json.loads(
                aggregate_completion_anchor_raw,
                object_pairs_hook=pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(token)
                ),
            )
        except (TypeError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise ActionPreservationGateError(
                "aggregate completion anchor literal is not JSON"
            ) from error
        if (
            not isinstance(aggregate_completion_anchor_raw, str)
            or not isinstance(anchor_value, Mapping)
            or runtime.canonical_json_bytes(anchor_value).decode("utf-8")
            != aggregate_completion_anchor_raw
        ):
            raise ActionPreservationGateError(
                "aggregate completion anchor literal is not canonical"
            )
        try:
            completion_anchor = runtime.validate_aggregate_completion_anchor(
                anchor_value
            )
        except runtime.DecodedEvalVerifiedReleaseError as error:
            raise ActionPreservationGateError(str(error)) from error
        aggregate_path = Path(evaluation_aggregate_path)
        expected_aggregate_sha = _sha(
            evaluation_aggregate_sha256,
            label="aggregate completion file",
        )
        if (
            aggregate["physical_bindings_digest"]
            != bindings["physical_bindings_digest"]
            or aggregate["evaluation_manifest_digest"] != bindings["manifest_digest"]
            or completion_anchor["evaluation_id"]
            != aggregate["evaluation_id"]
            or Path(completion_anchor["aggregate_root"])
            != aggregate_path.parent
            or completion_anchor["aggregate_file"]["relative_path"]
            != aggregate_path.name
            or completion_anchor["aggregate_file"]["sha256"]
            != expected_aggregate_sha
            or completion_anchor["aggregate_file"]["object_digest"]
            != aggregate["aggregate_digest"]
            or completion_anchor["private_file"]["object_digest"]
            != aggregate["private_mapping_digest"]
            or completion_anchor["public_file"]["object_digest"]
            != aggregate["public_packet_digest"]
            or private["private_mapping_digest"]
            != aggregate["private_mapping_digest"]
            or public["private_mapping_digest"]
            != private["private_mapping_digest"]
            or public["public_packet_digest"]
            != aggregate["public_packet_digest"]
            or completion_anchor["media_file_count"]
            != len(expected_media_sha256)
        ):
            raise ActionPreservationGateError(
                "evaluation aggregate differs from dynamic completion anchor"
            )
        _replay_aggregate_completion_publication(
            completion_anchor=completion_anchor,
            work_root=work_root,
            aggregate_path=evaluation_aggregate_path,
            aggregate_sha256=evaluation_aggregate_sha256,
            aggregate=aggregate,
            public_path=public_packet_path,
            public_sha256=public_packet_sha256,
            public=public,
            private_path=private_mapping_path,
            private_sha256=private_mapping_sha256,
            private=private,
            expected_media_sha256=expected_media_sha256,
            expected_work_root_target=expected_work_root_target,
        )
        for relative_path, module_path in (
            ("action_preservation_gate_v1.py", __file__),
            (
                "action_preservation_decoded_eval_bridge_v1.py",
                eval_bridge.__file__,
            ),
            (
                "action_preservation_decoded_eval_plan_v1.py",
                eval_bridge.plan.__file__,
            ),
        ):
            eval_bridge.require_running_eval_release_member(
                bindings["eval_release"],
                relative_path=relative_path,
                running_path=module_path,
            )
        if running_target_arguments is not None:
            eval_bridge.validate_running_verified_capture(
                bindings,
                target="action_preservation_gate_v1.py",
                expected_arguments=list(running_target_arguments),
                verify_file=True,
            )
        return bindings
    except eval_bridge.DecodedEvaluationBridgeError as error:
        raise ActionPreservationGateError(str(error)) from error


def _write_create_only_json(
    path: Path, value: Mapping[str, Any], *, work_root: Mapping[str, Any],
) -> None:
    """Publish one JSON artifact through the inherited held WORK_ROOT FD."""

    row = _replay_gate_work_root_authority(work_root)
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.parent != Path(row["path"])
        or path.name in ("", ".", "..")
    ):
        raise ActionPreservationGateError(
            "output must be a canonical direct child of inherited WORK_ROOT A"
        )
    if not hasattr(os, "O_NOFOLLOW"):
        raise ActionPreservationGateError(
            "safe held-root gate output publication is unavailable"
        )
    payload = canonical_json_bytes(value) + b"\n"
    flags = (
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path.name, flags, 0o400, dir_fd=row["root_fd"])
    except FileExistsError as error:
        raise ActionPreservationGateError(f"refusing to overwrite: {path}") from error
    except OSError as error:
        raise ActionPreservationGateError(
            "cannot create gate output through held WORK_ROOT A"
        ) from error
    try:
        os.set_inheritable(descriptor, False)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ActionPreservationGateError("create-only gate output made no progress")
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.fsync(row["root_fd"])
        before = os.fstat(descriptor)
        first = _read_fd(descriptor)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor)
        after = os.fstat(descriptor)
        named = os.stat(
            path.name, dir_fd=row["root_fd"], follow_symlinks=False
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_nlink != 1
            or _file_identity(before) != _file_identity(middle)
            or _file_identity(before) != _file_identity(after)
            or _file_identity(before) != _file_identity(named)
            or first != payload
            or second != payload
            or os.get_inheritable(descriptor)
        ):
            raise ActionPreservationGateError(
                "gate output held same-FD write replay differs"
            )
    except OSError as error:
        raise ActionPreservationGateError(
            "gate output held same-FD write replay is unavailable"
        ) from error
    finally:
        os.close(descriptor)
    try:
        replay_fd = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=row["root_fd"],
        )
    except OSError as error:
        raise ActionPreservationGateError(
            "gate output held post-close replay is unavailable"
        ) from error
    try:
        os.set_inheritable(replay_fd, False)
        replay_before = os.fstat(replay_fd)
        replay_first = _read_fd(replay_fd)
        replay_middle = os.fstat(replay_fd)
        replay_second = _read_fd(replay_fd)
        replay_after = os.fstat(replay_fd)
        # Replay A while the output remains pinned, then re-check the leaf name.
        row = _replay_gate_work_root_authority(row)
        replay_named = os.stat(
            path.name, dir_fd=row["root_fd"], follow_symlinks=False
        )
        replay_final = os.fstat(replay_fd)
    except OSError as error:
        raise ActionPreservationGateError(
            "gate output held post-close relative replay is unavailable"
        ) from error
    finally:
        os.close(replay_fd)
    if (
        _file_identity(replay_before) != _file_identity(before)
        or _file_identity(replay_middle) != _file_identity(before)
        or _file_identity(replay_after) != _file_identity(before)
        or _file_identity(replay_final) != _file_identity(before)
        or _file_identity(replay_named) != _file_identity(before)
        or replay_first != payload
        or replay_second != payload
    ):
        raise ActionPreservationGateError(
            "gate output held post-close relative replay differs"
        )
    try:
        os.fsync(row["root_fd"])
    except OSError as error:
        raise ActionPreservationGateError(
            "gate output held-root final fsync is unavailable"
        ) from error
    _replay_gate_work_root_authority(row)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    gate = sub.add_parser("gate")
    gate.add_argument("--measurement", required=True)
    gate.add_argument("--measurement-sha256", required=True)
    gate.add_argument("--calibration")
    gate.add_argument("--calibration-sha256")
    gate.add_argument("--output", required=True)
    ballot = sub.add_parser("ballot")
    ballot.add_argument("--public-packet", required=True)
    ballot.add_argument("--public-packet-sha256", required=True)
    ballot.add_argument("--blind-candidate-id", required=True)
    ballot.add_argument("--reviewer-id", required=True)
    ballot.add_argument("--labels", required=True)
    ballot.add_argument("--labels-sha256", required=True)
    ballot.add_argument("--output", required=True)
    aggregate_review = sub.add_parser("aggregate-review")
    aggregate_review.add_argument("--decision", required=True)
    aggregate_review.add_argument("--decision-sha256", required=True)
    aggregate_review.add_argument("--evaluation-complete", required=True)
    aggregate_review.add_argument("--evaluation-complete-sha256", required=True)
    aggregate_review.add_argument("--public-packet", required=True)
    aggregate_review.add_argument("--public-packet-sha256", required=True)
    aggregate_review.add_argument("--private-mapping", required=True)
    aggregate_review.add_argument("--private-mapping-sha256", required=True)
    aggregate_review.add_argument("--physical-bindings", required=True)
    aggregate_review.add_argument("--physical-bindings-sha256", required=True)
    aggregate_review.add_argument(
        "--aggregate-completion-anchor", required=True
    )
    aggregate_review.add_argument("--ballot", action="append", required=True)
    aggregate_review.add_argument("--ballot-sha256", action="append", required=True)
    aggregate_review.add_argument("--output", required=True)
    promote = sub.add_parser("promote")
    promote.add_argument("--decision", required=True)
    promote.add_argument("--decision-sha256", required=True)
    promote.add_argument("--blind-review", required=True)
    promote.add_argument("--blind-review-sha256", required=True)
    promote.add_argument("--evaluation-complete", required=True)
    promote.add_argument("--evaluation-complete-sha256", required=True)
    promote.add_argument("--public-packet", required=True)
    promote.add_argument("--public-packet-sha256", required=True)
    promote.add_argument("--private-mapping", required=True)
    promote.add_argument("--private-mapping-sha256", required=True)
    promote.add_argument("--physical-bindings", required=True)
    promote.add_argument("--physical-bindings-sha256", required=True)
    promote.add_argument("--aggregate-completion-anchor", required=True)
    promote.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    work_root = _load_gate_work_root_authority()
    output = Path(args.output)
    if args.command == "gate":
        if (args.calibration is None) is not (args.calibration_sha256 is None):
            raise ActionPreservationGateError(
                "calibration path and literal SHA must be supplied together"
            )
        calibration = (
            _load_expected(
                args.calibration, args.calibration_sha256,
                label="calibration",
            )
            if args.calibration is not None else None
        )
        result = decide(
            _load_expected(
                args.measurement, args.measurement_sha256,
                label="measurement",
            ),
            calibration,
        )
    elif args.command == "ballot":
        packet = _load_expected(
            args.public_packet, args.public_packet_sha256,
            label="public blind packet",
        )
        labels = _load_expected(
            args.labels, args.labels_sha256, label="review labels"
        )
        result = build_blind_ballot(
            public_packet=packet,
            blind_candidate_id=args.blind_candidate_id,
            reviewer_id=args.reviewer_id,
            labels=labels,
        )
    elif args.command == "aggregate-review":
        if len(args.ballot) != len(args.ballot_sha256):
            raise ActionPreservationGateError(
                "ballot paths and expected SHA lists differ"
            )
        decision = _load_expected(
            args.decision, args.decision_sha256, label="machine decision"
        )
        aggregate = _load_expected(
            args.evaluation_complete, args.evaluation_complete_sha256,
            label="evaluation complete",
        )
        public = _load_expected(
            args.public_packet, args.public_packet_sha256,
            label="public blind packet",
        )
        private = _load_expected(
            args.private_mapping, args.private_mapping_sha256,
            label="private blind mapping",
        )
        _verify_cli_eval_release(
            physical_bindings_path=args.physical_bindings,
            physical_bindings_sha256=args.physical_bindings_sha256,
            evaluation_aggregate=aggregate,
            public_packet=public,
            private_mapping=private,
            evaluation_aggregate_path=args.evaluation_complete,
            evaluation_aggregate_sha256=args.evaluation_complete_sha256,
            public_packet_path=args.public_packet,
            public_packet_sha256=args.public_packet_sha256,
            private_mapping_path=args.private_mapping,
            private_mapping_sha256=args.private_mapping_sha256,
            aggregate_completion_anchor_raw=(
                args.aggregate_completion_anchor
            ),
            work_root=work_root,
            expected_work_root_target=GATE_RUNTIME_TARGET,
            running_target_arguments=list(
                sys.argv[1:] if argv is None else argv
            ),
        )
        ballots = [
            _load_expected(path, digest, label=f"reviewer ballot {index}")
            for index, (path, digest) in enumerate(
                zip(args.ballot, args.ballot_sha256)
            )
        ]
        result = build_blind_review(
            decision=decision,
            evaluation_aggregate=aggregate,
            public_packet=public,
            private_mapping=private,
            ballots=ballots,
        )
    else:
        aggregate = _load_expected(
            args.evaluation_complete, args.evaluation_complete_sha256,
            label="evaluation complete",
        )
        public = _load_expected(
            args.public_packet, args.public_packet_sha256,
            label="public blind packet",
        )
        private = _load_expected(
            args.private_mapping, args.private_mapping_sha256,
            label="private blind mapping",
        )
        _verify_cli_eval_release(
            physical_bindings_path=args.physical_bindings,
            physical_bindings_sha256=args.physical_bindings_sha256,
            evaluation_aggregate=aggregate,
            public_packet=public,
            private_mapping=private,
            evaluation_aggregate_path=args.evaluation_complete,
            evaluation_aggregate_sha256=args.evaluation_complete_sha256,
            public_packet_path=args.public_packet,
            public_packet_sha256=args.public_packet_sha256,
            private_mapping_path=args.private_mapping,
            private_mapping_sha256=args.private_mapping_sha256,
            aggregate_completion_anchor_raw=(
                args.aggregate_completion_anchor
            ),
            work_root=work_root,
            expected_work_root_target=GATE_RUNTIME_TARGET,
            running_target_arguments=list(
                sys.argv[1:] if argv is None else argv
            ),
        )
        result = promotion_decision(
            _load_expected(
                args.decision, args.decision_sha256, label="machine decision"
            ),
            _load_expected(
                args.blind_review, args.blind_review_sha256,
                label="blind review",
            ),
            evaluation_aggregate=aggregate,
            public_packet=public,
            private_mapping=private,
        )
    _write_create_only_json(output, result, work_root=work_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
