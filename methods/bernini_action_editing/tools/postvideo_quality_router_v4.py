#!/usr/bin/env python3
"""Combine quality-gate-v3 and Qwen evidence into a fail-closed route.

This router is intentionally independent of the older v2/v3 implementations:
it consumes their immutable evidence but never rewrites it.  Gate-v3 hard
artifact failures are non-compensating and cannot be overridden by Qwen.
Conversely, base-relative low SSIM is never inspected as a standalone failure
signal.  A gate-v3 ``unresolved`` result remains review-only even when Qwen is
positive, so automatic training selection is fail-closed.

The production CLI consumes a caller-pinned gate manifest and a caller-pinned,
validated Qwen audit ``done.json``.  It emits create-only JSONL, receipt, and
SHA-256 sidecars.  Source/candidate hashes from both evidence producers must
agree before any route is published.
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
from typing import Any


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import checkpoint_visual_quality_gate_v3 as gate_v3  # noqa: E402
from tools import build_postvideo_quality_routing as qwen_builder  # noqa: E402


MANIFEST_SCHEMA = "bernini-postvideo-quality-router-v4-input-manifest-v1"
RECORD_SCHEMA = "bernini-postvideo-quality-router-v4-record-v1"
RECEIPT_SCHEMA = "bernini-postvideo-quality-router-v4-receipt-v1"
POLICY_SCHEMA = "bernini-postvideo-quality-router-v4-fixed-policy-v1"
JSON_SCHEMA_PATH = (
    METHOD_ROOT
    / "schemas"
    / "bernini_postvideo_quality_router_v4.schema.json"
).resolve(strict=True)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_AXIS_STATES = frozenset(("PASS", "FAIL", "UNRESOLVED"))
_ROUTES = frozenset(("PROMOTE", "REVIEW", "REJECT"))
_FAMILIES = ("NOISE", "BLUR", "ROUTEOFF_STRUCTURE", "FREEZE")
_SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


FIXED_POLICY: dict[str, Any] = {
    "schema_version": POLICY_SCHEMA,
    "gate_v3": {
        "accepted_schema_version": gate_v3.SCHEMA_VERSION,
        "hard_artifact_failure_route": "REJECT",
        "hard_artifact_failure_is_non_compensating": True,
        "qwen_may_override_hard_artifact_failure": False,
        "unresolved_route_ceiling": "REVIEW",
        "unresolved_may_auto_promote": False,
        "low_ssim_is_standalone_reject_signal": False,
        "routeoff_structure_requires_independent_spatial_artifact_support": True,
    },
    "qwen": {
        "accepted_quality_schema_version": qwen_builder.QA_SCHEMA,
        "required_audit_outcome_for_promotion": "success",
        "required_confidence_for_promotion": "high",
        "minimum_confidence_for_qwen_only_rejection": "medium",
        "required_uncertainty_codes_for_promotion": [],
        "maximum_promotable_blur": "low",
        "maximum_promotable_flicker": "low",
        "maximum_promotable_artifact": "low",
        "required_action_implemented": "yes",
        "required_identity_preserved": "yes",
        "allowed_species_preserved": ["yes", "not_applicable"],
        "allowed_clothing_preserved": ["yes", "not_applicable"],
        "required_non_edited_content_preserved": "yes",
        "required_camera_preserved": "yes",
    },
    "automatic_training_selection": {
        "eligible_route": "PROMOTE",
        "review_is_eligible": False,
        "reject_is_eligible": False,
        "fail_closed": True,
    },
}


class QualityRouterV4Error(RuntimeError):
    """One input, evidence binding, or publication contract is invalid."""


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
        raise QualityRouterV4Error(f"value is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_sha256(value: Any, *, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise QualityRouterV4Error(f"{context} must be one lowercase SHA-256")
    return value


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QualityRouterV4Error(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise QualityRouterV4Error(f"non-finite JSON constant: {value}")


def _decode_json(payload: bytes | str, *, context: str) -> Any:
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except QualityRouterV4Error:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualityRouterV4Error(f"invalid {context}: {error}") from error


def _plain_file(path: Path, *, context: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise QualityRouterV4Error(f"missing {context}: {path}") from error
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise QualityRouterV4Error(f"{context} must be one non-symlink file: {path}")
    return path.resolve(strict=True)


def _absolute_file(value: str | Path, *, context: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise QualityRouterV4Error(f"{context} must be an absolute path")
    return _plain_file(path, context=context)


def _absolute_output(value: str | Path, *, context: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path == Path("/"):
        raise QualityRouterV4Error(f"{context} must be a non-root absolute path")
    # Canonicalise macOS' /var -> /private/var alias before the path is bound
    # into a receipt.  The destination need not exist yet.
    return path.resolve(strict=False)


def _stable_file_binding(
    value: str | Path,
    declared_sha256: Any,
    *,
    context: str,
) -> dict[str, Any]:
    path = _absolute_file(value, context=context)
    expected = _required_sha256(declared_sha256, context=f"{context} hash")
    before = path.stat()
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise QualityRouterV4Error(f"cannot read {context}: {path}: {error}") from error
    signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if signature != (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
    ) or signature != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise QualityRouterV4Error(f"{context} changed while hashing: {path}")
    observed = digest.hexdigest()
    if observed != expected:
        raise QualityRouterV4Error(f"{context} differs from declared hash: {path}")
    return {"path": str(path), "sha256": observed, "bytes": before.st_size}


def _load_json_object(path: Path, *, context: str) -> dict[str, Any]:
    value = _decode_json(_plain_file(path, context=context).read_bytes(), context=context)
    if not isinstance(value, dict):
        raise QualityRouterV4Error(f"{context} must be one JSON object")
    return value


def _normal_gate_family(
    name: str,
    value: Any,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QualityRouterV4Error(f"gate-v3 {name} family differs")
    triggered = value.get("triggered")
    unresolved = value.get("unresolved")
    per_scale = value.get("per_scale")
    if (
        type(triggered) is not bool
        or type(unresolved) is not bool
        or not isinstance(per_scale, Mapping)
    ):
        raise QualityRouterV4Error(f"gate-v3 {name} family state differs")
    expected_scales = {f"{width}x{height}" for width, height in gate_v3.ANALYSIS_SCALES}
    if set(per_scale) != expected_scales:
        raise QualityRouterV4Error(f"gate-v3 {name} scale set differs")
    scale_states: dict[str, dict[str, bool]] = {}
    for scale, raw in per_scale.items():
        if not isinstance(raw, Mapping) or type(raw.get("triggered")) is not bool:
            raise QualityRouterV4Error(f"gate-v3 {name}[{scale}] differs")
        row = {
            "triggered": raw["triggered"],
            "unresolved": bool(raw.get("unresolved", False)),
        }
        if name == "ROUTEOFF_STRUCTURE":
            raw_candidate = raw.get("raw_candidate_triggered")
            support = raw.get("independent_spatial_artifact_support")
            if type(raw_candidate) is not bool or type(support) is not bool:
                raise QualityRouterV4Error(
                    f"gate-v3 ROUTEOFF_STRUCTURE[{scale}] support state differs"
                )
            if row["triggered"] and not support:
                raise QualityRouterV4Error(
                    "gate-v3 route-off structure cannot hard-fail from low SSIM alone"
                )
            if raw_candidate and not support and row["triggered"]:
                raise QualityRouterV4Error(
                    "unsupported base-relative structure was promoted to a hard failure"
                )
            row.update(
                {
                    "raw_candidate_triggered": raw_candidate,
                    "independent_spatial_artifact_support": support,
                }
            )
        scale_states[str(scale)] = row
    observed_triggered = sorted(
        scale for scale, row in scale_states.items() if row["triggered"]
    )
    declared_triggered = value.get("triggered_scales")
    if (
        triggered != bool(observed_triggered)
        or not isinstance(declared_triggered, list)
        or sorted(declared_triggered) != observed_triggered
    ):
        raise QualityRouterV4Error(f"gate-v3 {name} triggered scale closure differs")
    observed_unresolved = sorted(
        scale for scale, row in scale_states.items() if row["unresolved"]
    )
    declared_unresolved = value.get("unresolved_scales")
    if (
        unresolved != bool(observed_unresolved)
        or not isinstance(declared_unresolved, list)
        or sorted(declared_unresolved) != observed_unresolved
    ):
        raise QualityRouterV4Error(f"gate-v3 {name} unresolved scale closure differs")
    return {
        "triggered": triggered,
        "unresolved": unresolved,
        "triggered_scales": observed_triggered,
        "unresolved_scales": observed_unresolved,
        "per_scale": scale_states,
    }


def validate_gate_report(value: Any, *, iid: str) -> dict[str, Any]:
    """Validate the decision-level v3 contract and re-bind all three videos."""

    if not isinstance(value, Mapping):
        raise QualityRouterV4Error(f"gate-v3 report for {iid} is not an object")
    if (
        value.get("schema_version") != gate_v3.SCHEMA_VERSION
        or value.get("fail_closed") is not True
    ):
        raise QualityRouterV4Error(f"gate-v3 schema/fail-closed flag differs for {iid}")
    status = value.get("status")
    if status not in {"pass", "fail", "unresolved", "error"}:
        raise QualityRouterV4Error(f"gate-v3 status differs for {iid}")
    if value.get("passed") is not (status == "pass"):
        raise QualityRouterV4Error(f"gate-v3 passed flag differs for {iid}")
    if value.get("publishable") is not (status == "pass"):
        raise QualityRouterV4Error(f"gate-v3 publishable flag differs for {iid}")
    metadata = value.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("sample_id") != iid:
        raise QualityRouterV4Error(f"gate-v3 sample binding differs for {iid}")

    report: dict[str, Any] = {
        "status": status,
        "hard_artifact_failure": status in {"fail", "error"},
        "failure_codes": list(value.get("failure_codes", [])),
        "unresolved_codes": list(value.get("unresolved_codes", [])),
        "families": {},
        "media": None,
    }
    for field in ("failure_codes", "unresolved_codes"):
        codes = report[field]
        if any(type(code) is not str or not code for code in codes) or len(
            codes
        ) != len(set(codes)):
            raise QualityRouterV4Error(f"gate-v3 {field} differs for {iid}")

    # A runtime/input error is already a fail-closed REJECT.  Such an error can
    # legitimately lack decoded media or evidence-family payloads.
    if status == "error":
        if value.get("input_contract_passed") is not False or not report["failure_codes"]:
            raise QualityRouterV4Error(f"gate-v3 error closure differs for {iid}")
        return report

    if value.get("input_contract_passed") is not True:
        raise QualityRouterV4Error(f"gate-v3 input contract differs for {iid}")
    hard = value.get("hard_artifact_failure")
    unresolved = value.get("unresolved")
    if type(hard) is not bool or type(unresolved) is not bool:
        raise QualityRouterV4Error(f"gate-v3 decision flags differ for {iid}")
    decision = value.get("decision")
    if not isinstance(decision, Mapping) or decision.get("outcome") != status:
        raise QualityRouterV4Error(f"gate-v3 decision outcome differs for {iid}")
    families_raw = decision.get("evidence_families")
    if not isinstance(families_raw, Mapping) or set(families_raw) != set(_FAMILIES):
        raise QualityRouterV4Error(f"gate-v3 family set differs for {iid}")
    families = {
        name: _normal_gate_family(name, families_raw[name]) for name in _FAMILIES
    }
    any_triggered = any(row["triggered"] for row in families.values())
    any_unresolved = any(row["unresolved"] for row in families.values())
    if status == "fail":
        if (
            hard is not True
            or unresolved is not False
            or not any_triggered
            or not report["failure_codes"]
        ):
            raise QualityRouterV4Error(f"gate-v3 hard-failure closure differs for {iid}")
    elif status == "unresolved":
        if (
            hard is not False
            or unresolved is not True
            or any_triggered
            or not any_unresolved
            or not report["unresolved_codes"]
        ):
            raise QualityRouterV4Error(f"gate-v3 unresolved closure differs for {iid}")
    else:
        if (
            hard is not False
            or unresolved is not False
            or any_triggered
            or any_unresolved
            or report["failure_codes"]
            or report["unresolved_codes"]
        ):
            raise QualityRouterV4Error(f"gate-v3 pass closure differs for {iid}")
    if (
        decision.get("hard_artifact_failure") is not hard
        or decision.get("unresolved") is not unresolved
    ):
        raise QualityRouterV4Error(f"gate-v3 top-level/decision flags differ for {iid}")

    inputs = metadata.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {"source", "candidate", "frozen_base"}:
        raise QualityRouterV4Error(f"gate-v3 media input set differs for {iid}")
    media: dict[str, Any] = {}
    for name in ("source", "candidate", "frozen_base"):
        identity = inputs[name]
        if not isinstance(identity, Mapping):
            raise QualityRouterV4Error(f"gate-v3 {name} identity differs for {iid}")
        binding = _stable_file_binding(
            str(identity.get("path", "")),
            identity.get("sha256"),
            context=f"gate-v3 {name} media for {iid}",
        )
        declared_size = identity.get("size_bytes")
        if declared_size is not None and declared_size != binding["bytes"]:
            raise QualityRouterV4Error(f"gate-v3 {name} byte count differs for {iid}")
        media[name] = binding
    report.update(
        {
            "hard_artifact_failure": hard,
            "unresolved": unresolved,
            "families": families,
            "media": media,
        }
    )
    return report


def load_gate_manifest(
    manifest_path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    binding = _stable_file_binding(
        manifest_path,
        expected_sha256,
        context="gate-v3 input manifest",
    )
    manifest = _load_json_object(Path(binding["path"]), context="gate-v3 input manifest")
    expected_fields = {"schema_version", "complete", "row_count", "rows", "rows_digest"}
    if (
        set(manifest) != expected_fields
        or manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("complete") is not True
    ):
        raise QualityRouterV4Error("gate-v3 input manifest schema differs")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows or manifest.get("row_count") != len(rows):
        raise QualityRouterV4Error("gate-v3 input manifest row count differs")
    if manifest.get("rows_digest") != object_sha256(rows):
        raise QualityRouterV4Error("gate-v3 input manifest row digest differs")
    loaded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "iid",
            "gate_report_path",
            "gate_report_sha256",
        }:
            raise QualityRouterV4Error(f"gate-v3 manifest row {index} differs")
        iid = row.get("iid")
        if type(iid) is not str or not iid or "/" in iid or "\x00" in iid or iid in seen:
            raise QualityRouterV4Error(f"gate-v3 manifest IID differs at row {index}")
        seen.add(iid)
        report_binding = _stable_file_binding(
            str(row.get("gate_report_path", "")),
            row.get("gate_report_sha256"),
            context=f"gate-v3 report for {iid}",
        )
        report_value = _load_json_object(
            Path(report_binding["path"]), context=f"gate-v3 report for {iid}"
        )
        loaded.append(
            {
                "iid": iid,
                "gate_report": report_binding,
                "gate": validate_gate_report(report_value, iid=iid),
            }
        )
    if [row["iid"] for row in loaded] != sorted(seen):
        raise QualityRouterV4Error("gate-v3 manifest rows must be sorted by IID")
    return {
        "manifest": binding,
        "rows_digest": manifest["rows_digest"],
        "rows": loaded,
    }


def _qwen_quality_axis_state(
    *,
    explicit_failure: bool,
    failure_evidence_usable: bool,
    promotable: bool,
    audit_success: bool,
    evidence_strong: bool,
) -> str:
    if explicit_failure and failure_evidence_usable:
        return "FAIL"
    if audit_success and evidence_strong and promotable:
        return "PASS"
    return "UNRESOLVED"


def _classify_qwen(record: Mapping[str, Any]) -> dict[str, Any]:
    outcome = record.get("audit_outcome")
    quality = record.get("quality")
    if not isinstance(quality, Mapping):
        raise QualityRouterV4Error("validated Qwen audit record lacks quality object")
    success = outcome == "success"
    strong = bool(
        success
        and quality.get("confidence") == "high"
        and quality.get("uncertainty_codes") == []
    )
    failure_evidence_usable = bool(
        success
        and quality.get("confidence") in {"medium", "high"}
        and quality.get("uncertainty_codes") == []
    )

    def severity(field: str) -> tuple[bool, bool]:
        value = quality.get(field)
        return (
            value in _SEVERITY_RANK and _SEVERITY_RANK[value] >= _SEVERITY_RANK["medium"],
            value in {"none", "low"},
        )

    blur_fail, blur_ok = severity("blur_level")
    artifact_fail, artifact_ok = severity("artifact_level")
    flicker_fail, flicker_ok = severity("flicker_level")

    identity_values = [
        quality.get("identity_preserved"),
        quality.get("species_preserved"),
        quality.get("clothing_preserved"),
        quality.get("non_edited_content_preserved"),
    ]
    identity_failure = any(value == "no" for value in identity_values)
    identity_ok = bool(
        quality.get("identity_preserved") == "yes"
        and quality.get("species_preserved") in {"yes", "not_applicable"}
        and quality.get("clothing_preserved") in {"yes", "not_applicable"}
        and quality.get("non_edited_content_preserved") == "yes"
    )

    action_failure = quality.get("action_implemented") == "no"
    action_ok = quality.get("action_implemented") == "yes"
    camera_failure = quality.get("camera_preserved") == "no" or flicker_fail
    camera_ok = quality.get("camera_preserved") == "yes" and flicker_ok

    return {
        "audit_success": success,
        "strong_evidence": strong,
        "failure_evidence_usable": failure_evidence_usable,
        "artifact": _qwen_quality_axis_state(
            explicit_failure=blur_fail or artifact_fail,
            failure_evidence_usable=failure_evidence_usable,
            promotable=blur_ok and artifact_ok,
            audit_success=success,
            evidence_strong=strong,
        ),
        "identity_content": _qwen_quality_axis_state(
            explicit_failure=identity_failure,
            failure_evidence_usable=failure_evidence_usable,
            promotable=identity_ok,
            audit_success=success,
            evidence_strong=strong,
        ),
        "action": _qwen_quality_axis_state(
            explicit_failure=action_failure,
            failure_evidence_usable=failure_evidence_usable,
            promotable=action_ok,
            audit_success=success,
            evidence_strong=strong,
        ),
        "camera_flicker": _qwen_quality_axis_state(
            explicit_failure=camera_failure,
            failure_evidence_usable=failure_evidence_usable,
            promotable=camera_ok,
            audit_success=success,
            evidence_strong=strong,
        ),
    }


def _qwen_media_binding(record: Mapping[str, Any], *, iid: str) -> dict[str, str]:
    public_input = record.get("input")
    if not isinstance(public_input, Mapping):
        raise QualityRouterV4Error(f"Qwen input binding differs for {iid}")
    result: dict[str, str] = {}
    for name, qwen_name in (("source", "source_video"), ("candidate", "target_video")):
        identity = public_input.get(qwen_name)
        if not isinstance(identity, Mapping):
            raise QualityRouterV4Error(f"Qwen {qwen_name} binding differs for {iid}")
        result[name] = _required_sha256(
            identity.get("sha256"), context=f"Qwen {qwen_name} hash for {iid}"
        )
    return result


def route_one(
    *,
    iid: str,
    gate: Mapping[str, Any],
    gate_report_binding: Mapping[str, Any],
    qwen_record: Mapping[str, Any],
    qwen_audit_done_sha256: str,
) -> dict[str, Any]:
    """Pure decision combiner after both producer-specific validators run."""

    if qwen_record.get("iid") != iid:
        raise QualityRouterV4Error(f"Qwen IID differs for {iid}")
    qwen_record_digest = _required_sha256(
        qwen_record.get("record_digest"), context=f"Qwen record digest for {iid}"
    )
    qwen_record_candidate = dict(qwen_record)
    qwen_record_candidate.pop("record_digest", None)
    if object_sha256(qwen_record_candidate) != qwen_record_digest:
        raise QualityRouterV4Error(f"Qwen record digest differs for {iid}")
    qwen_quality = qwen_record.get("quality")
    if (
        not isinstance(qwen_quality, Mapping)
        or qwen_record.get("quality_sha256") != object_sha256(qwen_quality)
    ):
        raise QualityRouterV4Error(f"Qwen quality digest differs for {iid}")
    gate_status = gate.get("status")
    if gate_status not in {"pass", "fail", "unresolved", "error"}:
        raise QualityRouterV4Error(f"validated gate-v3 status differs for {iid}")
    gate_media = gate.get("media")
    qwen_media = _qwen_media_binding(qwen_record, iid=iid)
    if gate_status != "error":
        if not isinstance(gate_media, Mapping):
            raise QualityRouterV4Error(f"gate-v3 media binding is missing for {iid}")
        for name in ("source", "candidate"):
            if gate_media[name]["sha256"] != qwen_media[name]:
                raise QualityRouterV4Error(
                    f"gate-v3/Qwen {name} SHA-256 binding differs for {iid}"
                )

    qwen = _classify_qwen(qwen_record)
    gate_hard = gate_status in {"fail", "error"}
    gate_unresolved = gate_status == "unresolved"
    if gate_hard:
        artifact_state = "FAIL"
    elif gate_unresolved:
        artifact_state = "FAIL" if qwen["artifact"] == "FAIL" else "UNRESOLVED"
    else:
        artifact_state = qwen["artifact"]

    family_summary = {
        name: {
            "triggered": bool(row.get("triggered")),
            "unresolved": bool(row.get("unresolved")),
            "triggered_scales": list(row.get("triggered_scales", [])),
            "unresolved_scales": list(row.get("unresolved_scales", [])),
        }
        for name, row in gate.get("families", {}).items()
    }
    quality = qwen_record["quality"]
    axes = {
        "artifact_quality": {
            "status": artifact_state,
            "gate_v3_status": gate_status,
            "gate_v3_hard_artifact_failure": gate_hard,
            "gate_v3_failure_codes": list(gate.get("failure_codes", [])),
            "gate_v3_unresolved_codes": list(gate.get("unresolved_codes", [])),
            "gate_v3_families": family_summary,
            "qwen_blur_level": quality.get("blur_level"),
            "qwen_artifact_level": quality.get("artifact_level"),
            "qwen_status": qwen["artifact"],
            "qwen_technical_evidence": quality.get("evidence", {}).get("technical", []),
            "low_ssim_used_as_standalone_reject_signal": False,
            "qwen_may_override_gate_v3_hard_failure": False,
        },
        "identity_content_preservation": {
            "status": qwen["identity_content"],
            "identity_preserved": quality.get("identity_preserved"),
            "species_preserved": quality.get("species_preserved"),
            "clothing_preserved": quality.get("clothing_preserved"),
            "non_edited_content_preserved": quality.get("non_edited_content_preserved"),
            "identity_evidence": quality.get("evidence", {}).get("identity", []),
            "preservation_evidence": quality.get("evidence", {}).get("preservation", []),
        },
        "action_alignment": {
            "status": qwen["action"],
            "action_implemented": quality.get("action_implemented"),
            "evidence": quality.get("evidence", {}).get("action", []),
        },
        "camera_flicker": {
            "status": qwen["camera_flicker"],
            "camera_preserved": quality.get("camera_preserved"),
            "flicker_level": quality.get("flicker_level"),
            "preservation_evidence": quality.get("evidence", {}).get("preservation", []),
            "technical_evidence": quality.get("evidence", {}).get("technical", []),
        },
    }
    states = {name: axis["status"] for name, axis in axes.items()}
    if any(state not in _AXIS_STATES for state in states.values()):
        raise QualityRouterV4Error(f"internal axis state differs for {iid}")

    reasons: list[str] = []
    if gate_hard:
        reasons.append("gate_v3_hard_artifact_non_compensating")
    for name, state in states.items():
        if state == "FAIL":
            reasons.append(f"axis_{name}_failed")
    if gate_unresolved:
        reasons.append("gate_v3_unresolved_review_ceiling")
    if not qwen["audit_success"]:
        reasons.append(f"qwen_{qwen_record.get('audit_outcome', 'unknown')}")
    elif not qwen["strong_evidence"]:
        reasons.append("qwen_evidence_not_high_confidence_and_clear")
    for name, state in states.items():
        if state == "UNRESOLVED":
            reasons.append(f"axis_{name}_unresolved")
    reasons = list(dict.fromkeys(reasons))

    if gate_hard or any(state == "FAIL" for state in states.values()):
        route, overall = "REJECT", "FAIL"
    elif gate_unresolved or any(state == "UNRESOLVED" for state in states.values()):
        route, overall = "REVIEW", "UNRESOLVED"
    else:
        route, overall = "PROMOTE", "PASS"
        reasons.append("all_non_compensating_axes_passed")
    record: dict[str, Any] = {
        "schema_version": RECORD_SCHEMA,
        "iid": iid,
        "evidence_binding": {
            "gate_report_path": gate_report_binding["path"],
            "gate_report_sha256": gate_report_binding["sha256"],
            "qwen_audit_done_sha256": qwen_audit_done_sha256,
            "qwen_record_digest": qwen_record_digest,
            "qwen_quality_sha256": qwen_record.get("quality_sha256"),
            "source_video_sha256": qwen_media["source"],
            "candidate_video_sha256": qwen_media["candidate"],
            "frozen_base_video_sha256": (
                gate_media["frozen_base"]["sha256"]
                if isinstance(gate_media, Mapping)
                else None
            ),
            "pair_binding": "gate-v3 and Qwen source/candidate content SHA-256 equality",
        },
        "qwen_audit": {
            "audit_outcome": qwen_record.get("audit_outcome"),
            "confidence": quality.get("confidence"),
            "uncertainty_codes": list(quality.get("uncertainty_codes", [])),
            "model_identity_sha256": qwen_record.get("model_identity_sha256"),
            "prompt_contract_sha256": qwen_record.get("prompt_contract_sha256"),
        },
        "axes": axes,
        "decision": {
            "route": route,
            "overall_status": overall,
            "training_eligible": route == "PROMOTE",
            "promotion_eligible": route == "PROMOTE",
            "manual_review_required": route == "REVIEW",
            "fail_closed": True,
            "reason_codes": reasons,
        },
        "policy_sha256": object_sha256(FIXED_POLICY),
    }
    record["record_digest"] = object_sha256(record)
    return record


def _route_records(
    gate_manifest: Mapping[str, Any],
    qwen_audit: Mapping[str, Any],
    *,
    expected_qwen_done_sha256: str,
) -> list[dict[str, Any]]:
    records = qwen_audit.get("records")
    if not isinstance(records, list):
        raise QualityRouterV4Error("validated Qwen audit lacks records")
    by_iid: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or type(record.get("iid")) is not str:
            raise QualityRouterV4Error("validated Qwen audit record shape differs")
        iid = record["iid"]
        if iid in by_iid:
            raise QualityRouterV4Error(f"duplicate Qwen IID: {iid}")
        by_iid[iid] = record
    gate_iids = [row["iid"] for row in gate_manifest["rows"]]
    if set(by_iid) != set(gate_iids):
        raise QualityRouterV4Error("gate-v3 and Qwen audit IID sets differ")
    return [
        route_one(
            iid=row["iid"],
            gate=row["gate"],
            gate_report_binding=row["gate_report"],
            qwen_record=by_iid[row["iid"]],
            qwen_audit_done_sha256=expected_qwen_done_sha256,
        )
        for row in gate_manifest["rows"]
    ]


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _pretty_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise QualityRouterV4Error(f"cannot encode JSON: {error}") from error


def _publish(
    output: Path,
    *,
    routing_payload: bytes,
    receipt_payload: bytes,
    hash_payload: bytes,
) -> None:
    destinations = (output, Path(f"{output}.receipt.json"), Path(f"{output}.sha256"))
    if len(set(destinations)) != 3 or any(
        path.exists() or path.is_symlink() for path in destinations
    ):
        raise QualityRouterV4Error("create-only v4 output exists or aliases")
    output.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        for path, payload in zip(destinations, (routing_payload, receipt_payload, hash_payload)):
            with path.open("xb") as handle:
                created.append(path)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise


def build_routing(
    *,
    gate_manifest_path: str | Path,
    expected_gate_manifest_sha256: str,
    qwen_audit_dir: str | Path,
    expected_qwen_audit_done_sha256: str,
    output_jsonl: str | Path,
) -> dict[str, Any]:
    expected_done = _required_sha256(
        expected_qwen_audit_done_sha256,
        context="caller-pinned Qwen audit done hash",
    )
    gate_manifest = load_gate_manifest(
        gate_manifest_path, expected_sha256=expected_gate_manifest_sha256
    )
    try:
        qwen_audit = qwen_builder.validate_published_audit(
            qwen_audit_dir,
            expected_done_sha256=expected_done,
            require_production=True,
        )
    except qwen_builder.PostVideoQualityError as error:
        raise QualityRouterV4Error(f"Qwen audit validation failed: {error}") from error
    if qwen_audit.get("production_backend") is not True:
        raise QualityRouterV4Error("v4 routing requires a production Qwen audit")
    rows = _route_records(
        gate_manifest,
        qwen_audit,
        expected_qwen_done_sha256=expected_done,
    )
    output = _absolute_output(output_jsonl, context="v4 routing JSONL")
    routing_payload = _jsonl_bytes(rows)
    route_counts = dict(sorted(Counter(row["decision"]["route"] for row in rows).items()))
    axis_counts = {
        axis: dict(sorted(Counter(row["axes"][axis]["status"] for row in rows).items()))
        for axis in (
            "artifact_quality",
            "identity_content_preservation",
            "action_alignment",
            "camera_flicker",
        )
    }
    receipt_path = Path(f"{output}.receipt.json")
    hash_path = Path(f"{output}.sha256")
    implementation = {
        "router_path": str(Path(__file__).resolve(strict=True)),
        "router_sha256": file_sha256(Path(__file__).resolve(strict=True)),
        "schema_path": str(JSON_SCHEMA_PATH),
        "schema_sha256": file_sha256(JSON_SCHEMA_PATH),
        "gate_v3_path": str(Path(gate_v3.__file__).resolve(strict=True)),
        "gate_v3_sha256": file_sha256(Path(gate_v3.__file__).resolve(strict=True)),
        "qwen_tool_path": str(Path(qwen_builder.__file__).resolve(strict=True)),
        "qwen_tool_sha256": file_sha256(Path(qwen_builder.__file__).resolve(strict=True)),
    }
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "complete": True,
        "fail_closed": True,
        "record_schema_version": RECORD_SCHEMA,
        "row_count": len(rows),
        "route_counts": route_counts,
        "axis_counts": axis_counts,
        "training_eligible_count": route_counts.get("PROMOTE", 0),
        "gate_manifest_path": gate_manifest["manifest"]["path"],
        "gate_manifest_sha256": gate_manifest["manifest"]["sha256"],
        "gate_manifest_rows_digest": gate_manifest["rows_digest"],
        "qwen_audit_dir": str(Path(qwen_audit["output_dir"]).resolve(strict=True)),
        "qwen_audit_done_sha256": expected_done,
        "qwen_audit_input_sha256": qwen_audit.get("input_sha256"),
        "qwen_model_identity_sha256": qwen_audit.get("model_identity_sha256"),
        "qwen_method_source_revision": qwen_audit.get("method_source_revision"),
        "policy": FIXED_POLICY,
        "policy_sha256": object_sha256(FIXED_POLICY),
        "implementation": implementation,
        "routing_jsonl_path": str(output),
        "routing_jsonl_sha256": hashlib.sha256(routing_payload).hexdigest(),
        "routing_jsonl_bytes": len(routing_payload),
        "routing_rows_digest": object_sha256(rows),
        "publication_contract": "create_only_jsonl_receipt_sha256_sidecar",
        "receipt_path": str(receipt_path),
        "sha256_sidecar_path": str(hash_path),
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    receipt_payload = _pretty_bytes(receipt)
    hash_payload = (
        f"{hashlib.sha256(routing_payload).hexdigest()}  {output.name}\n"
        f"{hashlib.sha256(receipt_payload).hexdigest()}  {receipt_path.name}\n"
    ).encode("ascii")
    _publish(
        output,
        routing_payload=routing_payload,
        receipt_payload=receipt_payload,
        hash_payload=hash_payload,
    )
    return receipt


def _decode_jsonl(payload: bytes, *, context: str) -> list[dict[str, Any]]:
    if not payload or not payload.endswith(b"\n"):
        raise QualityRouterV4Error(f"{context} must be non-empty newline-terminated JSONL")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(payload.splitlines(), 1):
        if not line:
            raise QualityRouterV4Error(f"blank {context} row at {index}")
        value = _decode_json(line, context=f"{context} row {index}")
        if not isinstance(value, dict):
            raise QualityRouterV4Error(f"{context} row {index} is not an object")
        rows.append(value)
    return rows


def validate_release(
    output_jsonl: str | Path,
    *,
    expected_receipt_sha256: str,
) -> dict[str, Any]:
    output = _absolute_file(output_jsonl, context="v4 routing JSONL")
    receipt_path = _plain_file(Path(f"{output}.receipt.json"), context="v4 receipt")
    hash_path = _plain_file(Path(f"{output}.sha256"), context="v4 SHA sidecar")
    expected_receipt = _required_sha256(
        expected_receipt_sha256, context="caller-pinned v4 receipt hash"
    )
    if file_sha256(receipt_path) != expected_receipt:
        raise QualityRouterV4Error("v4 receipt differs from caller-pinned hash")
    routing_payload = output.read_bytes()
    rows = _decode_jsonl(routing_payload, context="v4 routing")
    receipt = _load_json_object(receipt_path, context="v4 receipt")
    candidate = dict(receipt)
    declared_digest = candidate.pop("receipt_digest", None)
    _required_sha256(declared_digest, context="v4 receipt digest")
    if object_sha256(candidate) != declared_digest:
        raise QualityRouterV4Error("v4 receipt digest differs")
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("complete") is not True
        or receipt.get("fail_closed") is not True
        or receipt.get("record_schema_version") != RECORD_SCHEMA
        or receipt.get("row_count") != len(rows)
        or receipt.get("routing_jsonl_path") != str(output)
        or receipt.get("routing_jsonl_sha256") != hashlib.sha256(routing_payload).hexdigest()
        or receipt.get("routing_jsonl_bytes") != len(routing_payload)
        or receipt.get("routing_rows_digest") != object_sha256(rows)
        or receipt.get("policy") != FIXED_POLICY
        or receipt.get("policy_sha256") != object_sha256(FIXED_POLICY)
    ):
        raise QualityRouterV4Error("v4 receipt output/policy closure differs")
    for row in rows:
        row_candidate = dict(row)
        row_digest = row_candidate.pop("record_digest", None)
        _required_sha256(row_digest, context="v4 record digest")
        decision = row.get("decision")
        axes = row.get("axes")
        expected_axis_names = {
            "artifact_quality",
            "identity_content_preservation",
            "action_alignment",
            "camera_flicker",
        }
        valid_axes = bool(
            isinstance(axes, Mapping)
            and set(axes) == expected_axis_names
            and all(
                isinstance(axis, Mapping) and axis.get("status") in _AXIS_STATES
                for axis in axes.values()
            )
        )
        route = decision.get("route") if isinstance(decision, Mapping) else None
        states = (
            [axis["status"] for axis in axes.values()]
            if valid_axes
            else []
        )
        route_flags_valid = bool(
            isinstance(decision, Mapping)
            and route in _ROUTES
            and decision.get("training_eligible") is (route == "PROMOTE")
            and decision.get("promotion_eligible") is (route == "PROMOTE")
            and decision.get("manual_review_required") is (route == "REVIEW")
            and decision.get("fail_closed") is True
            and (
                (
                    route == "PROMOTE"
                    and decision.get("overall_status") == "PASS"
                    and all(state == "PASS" for state in states)
                )
                or (
                    route == "REVIEW"
                    and decision.get("overall_status") == "UNRESOLVED"
                    and "FAIL" not in states
                    and "UNRESOLVED" in states
                )
                or (
                    route == "REJECT"
                    and decision.get("overall_status") == "FAIL"
                    and "FAIL" in states
                )
            )
        )
        if (
            object_sha256(row_candidate) != row_digest
            or row.get("schema_version") != RECORD_SCHEMA
            or not valid_axes
            or not route_flags_valid
        ):
            raise QualityRouterV4Error("v4 record closure differs")
    expected_sidecar = (
        f"{hashlib.sha256(routing_payload).hexdigest()}  {output.name}\n"
        f"{hashlib.sha256(receipt_path.read_bytes()).hexdigest()}  {receipt_path.name}\n"
    ).encode("ascii")
    if hash_path.read_bytes() != expected_sidecar:
        raise QualityRouterV4Error("v4 SHA sidecar differs")
    implementation = receipt.get("implementation")
    expected_sources = {
        "router_path": Path(__file__).resolve(strict=True),
        "schema_path": JSON_SCHEMA_PATH,
        "gate_v3_path": Path(gate_v3.__file__).resolve(strict=True),
        "qwen_tool_path": Path(qwen_builder.__file__).resolve(strict=True),
    }
    if not isinstance(implementation, Mapping):
        raise QualityRouterV4Error("v4 implementation closure differs")
    for field, source in expected_sources.items():
        digest_field = field.replace("_path", "_sha256")
        if implementation.get(field) != str(source) or implementation.get(
            digest_field
        ) != file_sha256(source):
            raise QualityRouterV4Error(f"v4 implementation binding differs: {field}")

    # Validation replays both producer-specific validators and the fixed v4
    # decision function.  A receipt is therefore not accepted merely because
    # its output hashes are self-consistent after an input disappeared or was
    # replaced.
    gate_manifest = load_gate_manifest(
        str(receipt.get("gate_manifest_path", "")),
        expected_sha256=str(receipt.get("gate_manifest_sha256", "")),
    )
    if gate_manifest["rows_digest"] != receipt.get("gate_manifest_rows_digest"):
        raise QualityRouterV4Error("v4 gate manifest receipt binding differs")
    expected_done = _required_sha256(
        receipt.get("qwen_audit_done_sha256"),
        context="v4 receipt Qwen done hash",
    )
    try:
        qwen_audit = qwen_builder.validate_published_audit(
            str(receipt.get("qwen_audit_dir", "")),
            expected_done_sha256=expected_done,
            require_production=True,
        )
    except qwen_builder.PostVideoQualityError as error:
        raise QualityRouterV4Error(
            f"v4 receipt Qwen audit validation failed: {error}"
        ) from error
    if (
        qwen_audit.get("production_backend") is not True
        or qwen_audit.get("input_sha256") != receipt.get("qwen_audit_input_sha256")
        or qwen_audit.get("model_identity_sha256")
        != receipt.get("qwen_model_identity_sha256")
        or qwen_audit.get("method_source_revision")
        != receipt.get("qwen_method_source_revision")
    ):
        raise QualityRouterV4Error("v4 receipt Qwen provenance differs")
    replayed_rows = _route_records(
        gate_manifest,
        qwen_audit,
        expected_qwen_done_sha256=expected_done,
    )
    if replayed_rows != rows:
        raise QualityRouterV4Error("v4 decision replay differs from published rows")
    route_counts = dict(sorted(Counter(row["decision"]["route"] for row in rows).items()))
    if receipt.get("route_counts") != route_counts or receipt.get(
        "training_eligible_count"
    ) != route_counts.get("PROMOTE", 0):
        raise QualityRouterV4Error("v4 route count closure differs")
    return {
        "status": "VALID",
        "rows": len(rows),
        "route_counts": route_counts,
        "training_eligible_count": route_counts.get("PROMOTE", 0),
        "routing_jsonl_sha256": hashlib.sha256(routing_payload).hexdigest(),
        "receipt_sha256": expected_receipt,
        "fail_closed": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    route = commands.add_parser("route", help="merge pinned gate-v3 and Qwen evidence")
    route.add_argument("--gate-manifest", type=Path, required=True)
    route.add_argument("--expected-gate-manifest-sha256", required=True)
    route.add_argument("--qwen-audit-dir", type=Path, required=True)
    route.add_argument("--expected-qwen-audit-done-sha256", required=True)
    route.add_argument("--output-jsonl", type=Path, required=True)
    validate = commands.add_parser("validate", help="validate a create-only v4 release")
    validate.add_argument("--output-jsonl", type=Path, required=True)
    validate.add_argument("--expected-receipt-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "route":
        result = build_routing(
            gate_manifest_path=args.gate_manifest,
            expected_gate_manifest_sha256=args.expected_gate_manifest_sha256,
            qwen_audit_dir=args.qwen_audit_dir,
            expected_qwen_audit_done_sha256=args.expected_qwen_audit_done_sha256,
            output_jsonl=args.output_jsonl,
        )
    else:
        result = validate_release(
            args.output_jsonl,
            expected_receipt_sha256=args.expected_receipt_sha256,
        )
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


__all__ = [
    "FIXED_POLICY",
    "JSON_SCHEMA_PATH",
    "MANIFEST_SCHEMA",
    "POLICY_SCHEMA",
    "QualityRouterV4Error",
    "RECEIPT_SCHEMA",
    "RECORD_SCHEMA",
    "build_routing",
    "canonical_json_bytes",
    "file_sha256",
    "load_gate_manifest",
    "object_sha256",
    "route_one",
    "validate_gate_report",
    "validate_release",
]


if __name__ == "__main__":
    raise SystemExit(main())
