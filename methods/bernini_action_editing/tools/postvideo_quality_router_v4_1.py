#!/usr/bin/env python3
"""Fail-closed canonical16 router for v3.1 artifact and Qwen-r6 evidence.

V4.1 is a new evidence schema and create-only release.  It never rewrites v4
records.  Artifact quality is authoritative from gate-v3.1; Qwen-r6 supplies
semantic/action/camera evidence and cannot clear a gate hard failure.  A v3.1
``unresolved`` result never auto-promotes: its scientific route remains
``REVIEW`` while its operational action is automatic adjudication.  No route
depends on per-sample human review, and automatic training eligibility is true
only when all four axes are ``PASS``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import checkpoint_visual_quality_gate_v3 as gate_v3  # noqa: E402
import checkpoint_visual_quality_gate_v3_1 as gate_v31  # noqa: E402
from tools import build_postvideo_quality_routing as qwen_builder  # noqa: E402
from tools import postvideo_quality_router_v4 as v4  # noqa: E402


GATE_INPUT_SCHEMA = "bernini-canonical16-gate-v3.1-input-manifest-v1"
GATE_MANIFEST_SCHEMA = "bernini-canonical16-gate-v3.1-report-manifest-v1"
RECORD_SCHEMA = "bernini-postvideo-quality-router-v4.1-record-v1"
SUMMARY_SCHEMA = "bernini-postvideo-quality-router-v4.1-summary-v1"
RECEIPT_SCHEMA = "bernini-postvideo-quality-router-v4.1-receipt-v1"
POLICY_SCHEMA = "bernini-postvideo-quality-router-v4.1-fixed-policy-v1"
JSON_SCHEMA_PATH = (
    METHOD_ROOT / "schemas" / "bernini_postvideo_quality_router_v4_1.schema.json"
).resolve(strict=True)

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_IID_RE = re.compile(r"^s([0-9]{8})-case([0-9]{2})$")
_FAMILIES = ("NOISE", "BLUR", "ROUTEOFF_STRUCTURE", "FREEZE")
_SCALES = {f"{width}x{height}" for width, height in gate_v31.ANALYSIS_SCALES}
_AXES = (
    "artifact_quality",
    "identity_content_preservation",
    "action_alignment",
    "camera_flicker",
)

FIXED_POLICY = {
    "schema_version": POLICY_SCHEMA,
    "artifact_authority": gate_v31.SCHEMA_VERSION,
    "qwen_semantic_schema": qwen_builder.QA_SCHEMA,
    "hard_artifact_non_compensating": True,
    "qwen_can_override_hard_artifact": False,
    "qwen_technical_labels_are_artifact_authority": False,
    "artifact_unresolved_can_auto_promote": False,
    "all_four_axes_must_pass_for_training": True,
    "routes": {
        "any_axis_fail": "REJECT",
        "no_fail_and_any_axis_unresolved": "REVIEW",
        "all_axes_pass": "PROMOTE",
    },
    "execution_actions": {
        "REJECT": "AUTO_RETRY",
        "REVIEW": "AUTO_ADJUDICATE",
        "PROMOTE": "ACCEPT_FOR_TRAINING",
    },
    "human_review_dependency": False,
}


class RouterV41Error(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return v4.canonical_json_bytes(value)


def object_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    return v4.file_sha256(path)


def required_sha(value: Any, *, context: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise RouterV41Error(f"{context} must be one lowercase SHA-256")
    return value


def plain_file(path: Path, *, context: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise RouterV41Error(f"missing {context}: {path}") from error
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise RouterV41Error(f"{context} must be a non-symlink file: {path}")
    return path.resolve(strict=True)


def read_object(path: Path, *, context: str) -> dict[str, Any]:
    path = plain_file(path, context=context)
    value = v4._decode_json(path.read_bytes(), context=context)
    if not isinstance(value, dict):
        raise RouterV41Error(f"{context} is not an object")
    return value


def read_jsonl(path: Path, *, context: str) -> list[dict[str, Any]]:
    payload = plain_file(path, context=context).read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise RouterV41Error(f"{context} must be newline-terminated JSONL")
    rows = []
    for ordinal, line in enumerate(payload.splitlines()):
        value = v4._decode_json(line, context=f"{context} row {ordinal}")
        if not isinstance(value, dict):
            raise RouterV41Error(f"{context} row {ordinal} is not an object")
        rows.append(value)
    return rows


def _bool_map(value: Any, *, keys: set[str], context: str) -> dict[str, bool]:
    if (
        not isinstance(value, Mapping)
        or set(value) != keys
        or any(type(item) is not bool for item in value.values())
    ):
        raise RouterV41Error(f"{context} boolean evidence differs")
    return {str(key): bool(item) for key, item in value.items()}


def _validate_scale_family(
    family: str, value: Any, *, context: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RouterV41Error(f"{context} is not an object")
    for field in ("triggered", "raw_candidate_triggered", "unresolved"):
        if type(value.get(field)) is not bool:
            raise RouterV41Error(f"{context}.{field} differs")
    count = value.get("independent_evidence_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise RouterV41Error(f"{context}.independent_evidence_count differs")
    if family == "NOISE":
        evidence = _bool_map(
            value.get("artifact_evidence"),
            keys={
                "distributional_luma_noise",
                "chroma_highpass_excess",
                "temporal_residual_with_strong_luma_excess",
            },
            context=context,
        )
        raw_conditions = value.get("raw_conditions")
        if not isinstance(raw_conditions, Mapping):
            raise RouterV41Error(f"{context}.raw_conditions differs")
        expected_raw = bool(
            any(evidence.values())
            or raw_conditions.get(
                "motion_compensated_residual_ratio_above_1p80"
            )
            is True
        )
        expected_count = sum(evidence.values())
        expected_trigger = expected_count >= 2
        evidence_field = "artifact_evidence"
    elif family == "BLUR":
        evidence = _bool_map(
            value.get("artifact_evidence"),
            keys={"global_frequency_loss", "salient_edge_retention_loss"},
            context=context,
        )
        expected_raw = bool(any(evidence.values()))
        expected_count = sum(evidence.values())
        expected_trigger = expected_count >= 2
        evidence_field = "artifact_evidence"
    elif family == "ROUTEOFF_STRUCTURE":
        raw_evidence = _bool_map(
            value.get("raw_structure_evidence"),
            keys={"low_windowed_ssim", "low_global_ssim_and_edge_correlation"},
            context=context,
        )
        evidence = _bool_map(
            value.get("artifact_support_evidence"),
            keys={
                "salient_structure_retention_loss",
                "confirmed_noise_artifact",
                "confirmed_blur_artifact",
            },
            context=context,
        )
        expected_raw = bool(any(raw_evidence.values()))
        expected_count = sum(evidence.values())
        expected_trigger = bool(expected_raw and expected_count >= 2)
        evidence_field = "artifact_support_evidence"
    else:
        evidence = _bool_map(
            value.get("artifact_evidence"),
            keys={
                "near_duplicate_prevalence",
                "near_duplicate_excess_over_base",
            },
            context=context,
        )
        expected_raw = bool(any(evidence.values()))
        expected_count = 1 if all(evidence.values()) else 0
        expected_trigger = False
        evidence_field = "artifact_evidence"
    if (
        value["raw_candidate_triggered"] is not expected_raw
        or value["independent_evidence_count"] != expected_count
        or value["triggered"] is not expected_trigger
        or value["unresolved"] is not (expected_raw and not expected_trigger)
    ):
        raise RouterV41Error(f"{context} decision recomputation differs")
    return {
        "triggered": expected_trigger,
        "raw": expected_raw,
        "unresolved": expected_raw and not expected_trigger,
        "evidence": evidence,
        "evidence_field": evidence_field,
    }


def _validate_families(value: Any, *, iid: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(_FAMILIES):
        raise RouterV41Error(f"v3.1 family set differs for {iid}")
    result = {}
    for family in _FAMILIES:
        aggregate = value[family]
        if not isinstance(aggregate, Mapping):
            raise RouterV41Error(f"v3.1 {family} differs for {iid}")
        per_scale = aggregate.get("per_scale")
        if not isinstance(per_scale, Mapping) or set(per_scale) != _SCALES:
            raise RouterV41Error(f"v3.1 {family} scales differ for {iid}")
        scales = {
            scale: _validate_scale_family(
                family,
                row,
                context=f"v3.1 {iid}.{family}.{scale}",
            )
            for scale, row in per_scale.items()
        }
        same_scale = sorted(
            scale for scale, row in scales.items() if row["triggered"]
        )
        if family in {"NOISE", "BLUR"}:
            evidence_names = set.intersection(
                *(set(row["evidence"]) for row in scales.values())
            )
            cross = sorted(
                name
                for name in evidence_names
                if all(row["evidence"][name] for row in scales.values())
            )
        elif family == "ROUTEOFF_STRUCTURE":
            raw_both = all(row["raw"] for row in scales.values())
            evidence_names = set.intersection(
                *(set(row["evidence"]) for row in scales.values())
            )
            cross = (
                sorted(
                    name
                    for name in evidence_names
                    if all(row["evidence"][name] for row in scales.values())
                )
                if raw_both
                else []
            )
        else:
            cross = (
                ["near_duplicate_prevalence_and_excess_over_base"]
                if all(all(row["evidence"].values()) for row in scales.values())
                else []
            )
        triggered = bool(same_scale or cross)
        any_raw = any(row["raw"] for row in scales.values())
        unresolved = bool(any_raw and not triggered)
        unresolved_scales = (
            sorted(
                scale
                for scale, row in scales.items()
                if row["raw"] and not row["triggered"]
            )
            if not triggered
            else []
        )
        if (
            aggregate.get("triggered") is not triggered
            or sorted(aggregate.get("triggered_scales", [])) != same_scale
            or sorted(aggregate.get("cross_scale_confirmed_evidence", []))
            != cross
            or aggregate.get("unresolved") is not unresolved
            or sorted(aggregate.get("unresolved_scales", []))
            != unresolved_scales
        ):
            raise RouterV41Error(f"v3.1 {family} aggregate differs for {iid}")
        result[family] = {
            "triggered": triggered,
            "unresolved": unresolved,
            "triggered_scales": same_scale,
            "cross_scale_confirmed_evidence": cross,
            "unresolved_scales": unresolved_scales,
        }
    return result


def validate_gate_report(
    report: Any,
    *,
    iid: str,
    expected_media: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        raise RouterV41Error(f"v3.1 report is not an object for {iid}")
    if (
        report.get("schema_version") != gate_v31.SCHEMA_VERSION
        or report.get("tool_sha256") != gate_v31.TOOL_SHA256
        or report.get("feature_extractor_schema_version")
        != gate_v31.FEATURE_EXTRACTOR_SCHEMA_VERSION
        or report.get("feature_extractor_tool_sha256")
        != gate_v31.FEATURE_EXTRACTOR_TOOL_SHA256
        or report.get("fail_closed") is not True
    ):
        raise RouterV41Error(f"v3.1 source/schema closure differs for {iid}")
    policy = report.get("confirmation_policy")
    if policy != {
        "cross_scale_or_two_independent_evidence_required": True,
        "single_scale_chroma_hp_alone_hard_fail_forbidden": True,
        "single_scale_salient_retention_alone_hard_fail_forbidden": True,
        "unsupported_base_relative_structure_is_unresolved": True,
    }:
        raise RouterV41Error(f"v3.1 confirmation policy differs for {iid}")
    status = report.get("status")
    if status not in {"pass", "fail", "unresolved", "error"}:
        raise RouterV41Error(f"v3.1 status differs for {iid}")
    metadata = report.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("sample_id") != iid:
        raise RouterV41Error(f"v3.1 IID differs for {iid}")
    if status == "error":
        if (
            report.get("passed") is not False
            or report.get("publishable") is not False
            or report.get("input_contract_passed") is not False
        ):
            raise RouterV41Error(f"v3.1 error closure differs for {iid}")
        return {
            "status": "error",
            "hard": True,
            "unresolved": False,
            "failure_codes": list(report.get("failure_codes", [])),
            "unresolved_codes": [],
            "families": {},
            "media": expected_media,
        }
    inputs = metadata.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "source",
        "candidate",
        "frozen_base",
    }:
        raise RouterV41Error(f"v3.1 media set differs for {iid}")
    for name in inputs:
        if (
            inputs[name].get("sha256") != expected_media[name]["sha256"]
            or inputs[name].get("size_bytes") != expected_media[name]["bytes"]
        ):
            raise RouterV41Error(f"v3.1 media binding differs for {iid}.{name}")
    decision = report.get("decision")
    if not isinstance(decision, Mapping) or decision.get("outcome") != status:
        raise RouterV41Error(f"v3.1 decision differs for {iid}")
    if (
        decision.get("single_scale_chroma_hp_alone_can_hard_fail") is not False
        or decision.get(
            "single_scale_salient_retention_alone_can_hard_fail"
        )
        is not False
    ):
        raise RouterV41Error(f"v3.1 single-scale hard-fail guard differs for {iid}")
    families = _validate_families(decision.get("evidence_families"), iid=iid)
    failures = [
        f"quality_{name.lower()}"
        for name, row in families.items()
        if row["triggered"]
    ]
    unresolved_codes = [
        f"quality_{name.lower()}_requires_external_verifier"
        for name, row in families.items()
        if row["unresolved"] and not row["triggered"]
    ]
    recomputed = "fail" if failures else "unresolved" if unresolved_codes else "pass"
    if (
        recomputed != status
        or report.get("passed") is not (status == "pass")
        or report.get("publishable") is not (status == "pass")
        or report.get("hard_artifact_failure") is not (status == "fail")
        or report.get("unresolved") is not (status == "unresolved")
        or report.get("failure_codes") != failures
        or report.get("unresolved_codes") != unresolved_codes
    ):
        raise RouterV41Error(f"v3.1 outcome recomputation differs for {iid}")
    features = report.get("features")
    if (
        not isinstance(features, Mapping)
        or features.get("all_frames_evaluated") is not True
        or features.get("all_transitions_evaluated") is not True
        or features.get("evaluated_frame_count_per_scale") != 81
        or features.get("evaluated_transition_count_per_scale") != 80
        or set(features.get("scales", {})) != _SCALES
    ):
        raise RouterV41Error(f"v3.1 full-frame feature closure differs for {iid}")
    return {
        "status": status,
        "hard": status == "fail",
        "unresolved": status == "unresolved",
        "failure_codes": failures,
        "unresolved_codes": unresolved_codes,
        "families": families,
        "media": expected_media,
    }


def load_gate_audit(
    directory: Path, *, expected_manifest_sha256: str
) -> dict[str, Any]:
    root = directory.expanduser().resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise RouterV41Error("gate audit must be a non-symlink directory")
    manifest_path = plain_file(root / "gate-manifest.json", context="gate manifest")
    if file_sha(manifest_path) != required_sha(
        expected_manifest_sha256, context="caller-pinned gate manifest hash"
    ):
        raise RouterV41Error("gate manifest differs from caller-pinned hash")
    manifest = read_object(manifest_path, context="gate manifest")
    input_path = plain_file(root / "input-manifest.json", context="gate input manifest")
    if (
        manifest.get("schema_version") != GATE_MANIFEST_SCHEMA
        or manifest.get("complete") is not True
        or manifest.get("input_manifest_sha256") != file_sha(input_path)
    ):
        raise RouterV41Error("gate report manifest closure differs")
    input_manifest = read_object(input_path, context="gate input manifest")
    input_rows = input_manifest.get("rows")
    if (
        input_manifest.get("schema_version") != GATE_INPUT_SCHEMA
        or input_manifest.get("complete") is not True
        or not isinstance(input_rows, list)
        or len(input_rows) != 16
        or input_manifest.get("row_count") != 16
        or input_manifest.get("rows_digest") != object_sha(input_rows)
        or input_manifest.get("gate_tool", {}).get("sha256")
        != gate_v31.TOOL_SHA256
    ):
        raise RouterV41Error("gate input manifest closure differs")
    expected_by_iid = {row.get("iid"): row for row in input_rows}
    if len(expected_by_iid) != 16 or any(
        _IID_RE.fullmatch(str(iid)) is None for iid in expected_by_iid
    ):
        raise RouterV41Error("gate input IID set differs")
    report_rows = manifest.get("rows")
    if (
        not isinstance(report_rows, list)
        or len(report_rows) != 16
        or manifest.get("row_count") != 16
        or manifest.get("rows_digest") != object_sha(report_rows)
    ):
        raise RouterV41Error("gate report row closure differs")
    loaded = []
    for ordinal, row in enumerate(report_rows):
        iid = row.get("iid")
        if iid not in expected_by_iid or row.get("ordinal") != ordinal:
            raise RouterV41Error("gate report ordering differs")
        report_path = plain_file(
            root / "reports" / f"{iid}.quality-v3_1.json",
            context=f"gate report for {iid}",
        )
        if file_sha(report_path) != row.get("sha256"):
            raise RouterV41Error(f"gate report SHA differs for {iid}")
        expected = expected_by_iid[iid]
        gate = validate_gate_report(
            read_object(report_path, context=f"gate report for {iid}"),
            iid=iid,
            expected_media=expected["media"],
        )
        if (
            row.get("status") != gate["status"]
            or row.get("hard_artifact_failure") is not gate["hard"]
            or expected.get("qwen_record_digest") is None
        ):
            raise RouterV41Error(f"gate manifest decision binding differs for {iid}")
        loaded.append(
            {
                "iid": iid,
                "report_path": str(report_path),
                "report_sha256": row["sha256"],
                "qwen_record_digest": expected["qwen_record_digest"],
                "gate": gate,
            }
        )
    if [row["iid"] for row in loaded] != sorted(expected_by_iid):
        raise RouterV41Error("gate rows are not canonical IID order")
    return {
        "root": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha(manifest_path),
        "input_manifest_path": str(input_path),
        "input_manifest_sha256": file_sha(input_path),
        "qwen_records_sha256": input_manifest["qwen_records"]["sha256"],
        "rows": loaded,
    }


def load_qwen_r6(
    directory: Path, *, expected_final_sha256: str
) -> dict[str, Any]:
    root = directory.expanduser().resolve(strict=True)
    expected_names = {
        "canonical16-r1.final.json",
        "done.json",
        "summary.json",
        "records.jsonl",
    }
    members = list(root.iterdir())
    if (
        root.is_symlink()
        or not root.is_dir()
        or {path.name for path in members} != expected_names
        or any(path.is_symlink() or not path.is_file() for path in members)
    ):
        raise RouterV41Error("Qwen-r6 detached four-file closure differs")
    final_path = root / "canonical16-r1.final.json"
    if file_sha(final_path) != required_sha(
        expected_final_sha256, context="caller-pinned Qwen-r6 final hash"
    ):
        raise RouterV41Error("Qwen-r6 final differs from caller-pinned hash")
    final = read_object(final_path, context="Qwen-r6 final")
    done_path, summary_path, records_path = (
        root / "done.json",
        root / "summary.json",
        root / "records.jsonl",
    )
    done = read_object(done_path, context="Qwen-r6 done")
    summary = read_object(summary_path, context="Qwen-r6 summary")
    records = read_jsonl(records_path, context="Qwen-r6 records")
    audit = final.get("audit")
    if (
        final.get("schema_version")
        != "bernini-v16r5-multisample-qwen-audit-final-v1"
        or final.get("status") != "complete"
        or final.get("qwen_evidence_is_non_promotional") is not True
        or final.get("training_invoked") is not False
        or not isinstance(audit, Mapping)
        or audit.get("rows") != 16
        or audit.get("records_sha256") != file_sha(records_path)
        or audit.get("summary_sha256") != file_sha(summary_path)
        or audit.get("done_sha256") != file_sha(done_path)
    ):
        raise RouterV41Error("Qwen-r6 final release closure differs")
    if (
        done.get("schema_version") != qwen_builder.AUDIT_DONE_SCHEMA
        or done.get("status") != "complete"
        or done.get("rows") != 16
        or done.get("outcome_counts") != {"success": 16}
        or done.get("files", {}).get("records.jsonl", {}).get("sha256")
        != file_sha(records_path)
        or done.get("files", {}).get("summary.json", {}).get("sha256")
        != file_sha(summary_path)
    ):
        raise RouterV41Error("Qwen-r6 done closure differs")
    backend = summary.get("runtime", {}).get("backend_execution")
    if (
        summary.get("schema_version") != qwen_builder.AUDIT_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("rows") != 16
        or summary.get("outcome_counts") != {"success": 16}
        or summary.get("outputs", {}).get("records.jsonl", {}).get("sha256")
        != file_sha(records_path)
        or summary.get("model_identity_sha256")
        != done.get("model_identity_sha256")
        or not isinstance(backend, Mapping)
        or backend.get("mode") != "production_local_qwen"
        or backend.get("production_backend") is not True
        or backend.get("cuda_only") is not True
        or backend.get("cpu_offload_detected") is not False
        or backend.get("disk_offload_detected") is not False
        or backend.get("meta_offload_detected") is not False
    ):
        raise RouterV41Error("Qwen-r6 production summary closure differs")
    validated = []
    for ordinal, record in enumerate(records):
        iid = record.get("iid")
        candidate = dict(record)
        digest = candidate.pop("record_digest", None)
        if (
            _IID_RE.fullmatch(str(iid)) is None
            or record.get("schema_version") != qwen_builder.AUDIT_RECORD_SCHEMA
            or record.get("ordinal") != ordinal
            or record.get("audit_outcome") != "success"
            or required_sha(digest, context=f"Qwen record digest for {iid}")
            != object_sha(candidate)
            or record.get("model_identity_sha256")
            != done.get("model_identity_sha256")
            or record.get("prompt_contract_sha256")
            != done.get("prompt_contract_sha256")
        ):
            raise RouterV41Error(f"Qwen-r6 record closure differs for {iid}")
        quality = qwen_builder.validate_quality_observation(
            record.get("quality"), nframes=12
        )
        if record.get("quality_sha256") != object_sha(quality):
            raise RouterV41Error(f"Qwen-r6 quality digest differs for {iid}")
        validated.append(record)
    if [row["iid"] for row in validated] != sorted(row["iid"] for row in validated):
        raise RouterV41Error("Qwen-r6 records are not canonical IID order")
    return {
        "root": str(root),
        "final_path": str(final_path),
        "final_sha256": file_sha(final_path),
        "done_sha256": file_sha(done_path),
        "summary_sha256": file_sha(summary_path),
        "records_sha256": file_sha(records_path),
        "model_identity_sha256": done["model_identity_sha256"],
        "prompt_contract_sha256": done["prompt_contract_sha256"],
        "records": validated,
    }


def combine_one(
    gate_row: Mapping[str, Any], qwen_record: Mapping[str, Any]
) -> dict[str, Any]:
    iid = gate_row["iid"]
    if qwen_record.get("iid") != iid:
        raise RouterV41Error(f"gate/Qwen IID differs for {iid}")
    if qwen_record.get("record_digest") != gate_row["qwen_record_digest"]:
        raise RouterV41Error(f"gate/Qwen record digest differs for {iid}")
    gate = gate_row["gate"]
    qwen_input = qwen_record.get("input", {})
    if (
        qwen_input.get("source_video", {}).get("sha256")
        != gate["media"]["source"]["sha256"]
        or qwen_input.get("target_video", {}).get("sha256")
        != gate["media"]["candidate"]["sha256"]
    ):
        raise RouterV41Error(f"gate/Qwen media SHA differs for {iid}")
    qwen_state = v4._classify_qwen(qwen_record)
    artifact_state = {
        "pass": "PASS",
        "fail": "FAIL",
        "error": "FAIL",
        "unresolved": "UNRESOLVED",
    }[gate["status"]]
    quality = qwen_record["quality"]
    axes = {
        "artifact_quality": {
            "status": artifact_state,
            "authority": gate_v31.SCHEMA_VERSION,
            "gate_status": gate["status"],
            "hard_artifact_failure": gate["hard"],
            "failure_codes": gate["failure_codes"],
            "unresolved_codes": gate["unresolved_codes"],
            "families": gate["families"],
            "qwen_blur_level_non_authoritative": quality["blur_level"],
            "qwen_artifact_level_non_authoritative": quality["artifact_level"],
            "qwen_can_override_hard_artifact": False,
        },
        "identity_content_preservation": {
            "status": qwen_state["identity_content"],
            "identity_preserved": quality["identity_preserved"],
            "species_preserved": quality["species_preserved"],
            "clothing_preserved": quality["clothing_preserved"],
            "non_edited_content_preserved": quality[
                "non_edited_content_preserved"
            ],
            "evidence": {
                "identity": quality["evidence"]["identity"],
                "preservation": quality["evidence"]["preservation"],
            },
        },
        "action_alignment": {
            "status": qwen_state["action"],
            "action_implemented": quality["action_implemented"],
            "evidence": quality["evidence"]["action"],
        },
        "camera_flicker": {
            "status": qwen_state["camera_flicker"],
            "camera_preserved": quality["camera_preserved"],
            "flicker_level": quality["flicker_level"],
            "evidence": {
                "preservation": quality["evidence"]["preservation"],
                "technical": quality["evidence"]["technical"],
            },
        },
    }
    states = [axes[name]["status"] for name in _AXES]
    if "FAIL" in states:
        route, overall = "REJECT", "FAIL"
    elif "UNRESOLVED" in states:
        route, overall = "REVIEW", "UNRESOLVED"
    else:
        route, overall = "PROMOTE", "PASS"
    execution_action = FIXED_POLICY["execution_actions"][route]
    reasons = []
    for name in _AXES:
        if axes[name]["status"] != "PASS":
            reasons.append(f"axis_{name}_{axes[name]['status'].lower()}")
    if route == "PROMOTE":
        reasons.append("all_four_axes_pass")
    record = {
        "schema_version": RECORD_SCHEMA,
        "iid": iid,
        "checkpoint_step": int(_IID_RE.fullmatch(iid).group(1)),
        "case_index": int(_IID_RE.fullmatch(iid).group(2)),
        "evidence_binding": {
            "gate_report_path": gate_row["report_path"],
            "gate_report_sha256": gate_row["report_sha256"],
            "qwen_record_digest": qwen_record["record_digest"],
            "qwen_quality_sha256": qwen_record["quality_sha256"],
            "source_video_sha256": gate["media"]["source"]["sha256"],
            "candidate_video_sha256": gate["media"]["candidate"]["sha256"],
            "frozen_base_video_sha256": gate["media"]["frozen_base"]["sha256"],
        },
        "axes": axes,
        "decision": {
            "route": route,
            "overall_status": overall,
            "training_eligible": route == "PROMOTE",
            "execution_action": execution_action,
            "manual_review_required": False,
            "human_review_dependency": False,
            "fail_closed": True,
            "reason_codes": reasons,
        },
        "policy_sha256": object_sha(FIXED_POLICY),
    }
    record["record_digest"] = object_sha(record)
    return record


def pretty(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical(dict(row)) + b"\n" for row in rows)


def build_release(
    *,
    gate_audit_dir: Path,
    expected_gate_manifest_sha256: str,
    qwen_evidence_dir: Path,
    expected_qwen_final_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    gate = load_gate_audit(
        gate_audit_dir,
        expected_manifest_sha256=expected_gate_manifest_sha256,
    )
    qwen = load_qwen_r6(
        qwen_evidence_dir,
        expected_final_sha256=expected_qwen_final_sha256,
    )
    if gate["qwen_records_sha256"] != qwen["records_sha256"]:
        raise RouterV41Error("gate input/Qwen-r6 records SHA differs")
    qwen_by_iid = {record["iid"]: record for record in qwen["records"]}
    if set(qwen_by_iid) != {row["iid"] for row in gate["rows"]}:
        raise RouterV41Error("gate/Qwen canonical16 IID sets differ")
    rows = [combine_one(row, qwen_by_iid[row["iid"]]) for row in gate["rows"]]
    route_counts = dict(sorted(Counter(row["decision"]["route"] for row in rows).items()))
    execution_action_counts = dict(
        sorted(Counter(row["decision"]["execution_action"] for row in rows).items())
    )
    axis_counts = {
        axis: dict(sorted(Counter(row["axes"][axis]["status"] for row in rows).items()))
        for axis in _AXES
    }
    checkpoint_route_counts = {}
    for step in (32, 359):
        checkpoint_route_counts[str(step)] = dict(
            sorted(
                Counter(
                    row["decision"]["route"]
                    for row in rows
                    if row["checkpoint_step"] == step
                ).items()
            )
        )
    records_payload = jsonl(rows)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "rows": len(rows),
        "route_counts": route_counts,
        "execution_action_counts": execution_action_counts,
        "axis_counts": axis_counts,
        "checkpoint_route_counts": checkpoint_route_counts,
        "training_eligible_count": route_counts.get("PROMOTE", 0),
        "fail_closed": True,
        "policy": FIXED_POLICY,
        "policy_sha256": object_sha(FIXED_POLICY),
    }
    summary_payload = pretty(summary)
    implementation = {
        "router_v4_1_path": str(Path(__file__).resolve(strict=True)),
        "router_v4_1_sha256": file_sha(Path(__file__).resolve(strict=True)),
        "schema_path": str(JSON_SCHEMA_PATH),
        "schema_sha256": file_sha(JSON_SCHEMA_PATH),
        "gate_v3_1_path": str(Path(gate_v31.__file__).resolve(strict=True)),
        "gate_v3_1_sha256": file_sha(Path(gate_v31.__file__).resolve(strict=True)),
        "gate_v3_path": str(Path(gate_v3.__file__).resolve(strict=True)),
        "gate_v3_sha256": file_sha(Path(gate_v3.__file__).resolve(strict=True)),
        "qwen_tool_path": str(Path(qwen_builder.__file__).resolve(strict=True)),
        "qwen_tool_sha256": file_sha(Path(qwen_builder.__file__).resolve(strict=True)),
        "router_v4_path": str(Path(v4.__file__).resolve(strict=True)),
        "router_v4_sha256": file_sha(Path(v4.__file__).resolve(strict=True)),
    }
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "complete": True,
        "fail_closed": True,
        "rows": len(rows),
        "route_counts": route_counts,
        "execution_action_counts": execution_action_counts,
        "axis_counts": axis_counts,
        "training_eligible_count": route_counts.get("PROMOTE", 0),
        "gate_audit": {
            key: gate[key]
            for key in (
                "root",
                "manifest_path",
                "manifest_sha256",
                "input_manifest_path",
                "input_manifest_sha256",
            )
        },
        "qwen_r6": {
            key: qwen[key]
            for key in (
                "root",
                "final_path",
                "final_sha256",
                "done_sha256",
                "summary_sha256",
                "records_sha256",
                "model_identity_sha256",
                "prompt_contract_sha256",
            )
        },
        "policy_sha256": object_sha(FIXED_POLICY),
        "implementation": implementation,
        "files": {
            "records.jsonl": {
                "sha256": hashlib.sha256(records_payload).hexdigest(),
                "bytes": len(records_payload),
                "rows": len(rows),
            },
            "summary.json": {
                "sha256": hashlib.sha256(summary_payload).hexdigest(),
                "bytes": len(summary_payload),
            },
        },
        "old_v4_evidence_overwritten": False,
        "training_or_inference_started": False,
    }
    receipt["receipt_digest"] = object_sha(receipt)
    receipt_payload = pretty(receipt)
    sums_payload = (
        f"{hashlib.sha256(records_payload).hexdigest()}  records.jsonl\n"
        f"{hashlib.sha256(summary_payload).hexdigest()}  summary.json\n"
        f"{hashlib.sha256(receipt_payload).hexdigest()}  receipt.json\n"
    ).encode("ascii")
    output = output_dir.expanduser().resolve(strict=False)
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise RouterV41Error("output must be one absent absolute directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    )
    try:
        for name, payload in {
            "records.jsonl": records_payload,
            "summary.json": summary_payload,
            "receipt.json": receipt_payload,
            "SHA256SUMS": sums_payload,
        }.items():
            with (staging / name).open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(staging, output)
    except BaseException:
        for path in staging.iterdir() if staging.exists() else []:
            path.unlink(missing_ok=True)
        if staging.exists():
            staging.rmdir()
        raise
    return {**summary, "output_dir": str(output), "receipt_sha256": file_sha(output / "receipt.json")}


def validate_release(output_dir: Path, *, expected_receipt_sha256: str) -> dict[str, Any]:
    root = output_dir.expanduser().resolve(strict=True)
    expected = {"records.jsonl", "summary.json", "receipt.json", "SHA256SUMS"}
    members = list(root.iterdir())
    if (
        root.is_symlink()
        or not root.is_dir()
        or {path.name for path in members} != expected
        or any(path.is_symlink() or not path.is_file() for path in members)
    ):
        raise RouterV41Error("v4.1 four-file release closure differs")
    receipt_path = root / "receipt.json"
    if file_sha(receipt_path) != required_sha(
        expected_receipt_sha256, context="caller-pinned v4.1 receipt hash"
    ):
        raise RouterV41Error("v4.1 receipt differs from caller pin")
    receipt = read_object(receipt_path, context="v4.1 receipt")
    candidate = dict(receipt)
    digest = candidate.pop("receipt_digest", None)
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("complete") is not True
        or receipt.get("fail_closed") is not True
        or required_sha(digest, context="v4.1 receipt digest")
        != object_sha(candidate)
    ):
        raise RouterV41Error("v4.1 receipt closure differs")
    records_payload = (root / "records.jsonl").read_bytes()
    summary_payload = (root / "summary.json").read_bytes()
    if (
        receipt.get("files", {}).get("records.jsonl", {}).get("sha256")
        != hashlib.sha256(records_payload).hexdigest()
        or receipt.get("files", {}).get("summary.json", {}).get("sha256")
        != hashlib.sha256(summary_payload).hexdigest()
    ):
        raise RouterV41Error("v4.1 output hashes differ")
    rows = read_jsonl(root / "records.jsonl", context="v4.1 records")
    for row in rows:
        candidate = dict(row)
        digest = candidate.pop("record_digest", None)
        if (
            row.get("schema_version") != RECORD_SCHEMA
            or required_sha(digest, context="v4.1 record digest")
            != object_sha(candidate)
            or row.get("decision", {}).get("training_eligible")
            is not (row.get("decision", {}).get("route") == "PROMOTE")
            or row.get("decision", {}).get("manual_review_required") is not False
            or row.get("decision", {}).get("human_review_dependency") is not False
            or row.get("decision", {}).get("execution_action")
            != FIXED_POLICY["execution_actions"].get(
                row.get("decision", {}).get("route")
            )
        ):
            raise RouterV41Error("v4.1 record closure differs")
    return {
        "status": "VALID",
        "rows": len(rows),
        "route_counts": receipt["route_counts"],
        "execution_action_counts": receipt["execution_action_counts"],
        "axis_counts": receipt["axis_counts"],
        "training_eligible_count": receipt["training_eligible_count"],
        "receipt_sha256": file_sha(receipt_path),
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    route = commands.add_parser("route")
    route.add_argument("--gate-audit-dir", type=Path, required=True)
    route.add_argument("--expected-gate-manifest-sha256", required=True)
    route.add_argument("--qwen-evidence-dir", type=Path, required=True)
    route.add_argument("--expected-qwen-final-sha256", required=True)
    route.add_argument("--output-dir", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--expected-receipt-sha256", required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "route":
        result = build_release(
            gate_audit_dir=args.gate_audit_dir,
            expected_gate_manifest_sha256=args.expected_gate_manifest_sha256,
            qwen_evidence_dir=args.qwen_evidence_dir,
            expected_qwen_final_sha256=args.expected_qwen_final_sha256,
            output_dir=args.output_dir,
        )
    else:
        result = validate_release(
            args.output_dir,
            expected_receipt_sha256=args.expected_receipt_sha256,
        )
    print(canonical(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
