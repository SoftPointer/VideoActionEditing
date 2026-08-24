"""Finalize eight Qwen full-motion shards into pending generation proposals.

This stage is intentionally authorization-free.  It revalidates every Qwen
record and terminal shard receipt, selects a deterministic diversity-bounded
primary/reserve pool, and publishes the exact contract consumed by the
full-motion postcheck.  No row is human reviewed or generation authorized.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


FINALIZE_SCHEMA = "motive-goku-full-motion-finalize-v1"
DONE_SCHEMA = "motive-goku-full-motion-finalize-done-v1"
GENERATION_SCHEMA = "motive-goku-full-motion-generation-v6"
MOTION_SPEC_SCHEMA = "motive-goku-full-motion-generation-spec-v6"
QWEN_EVIDENCE_SCHEMA = "motive-goku-full-motion-qwen-evidence-v6"
TEMPORAL_GEOMETRY_SCHEMA = "motive-goku-full-motion-temporal-geometry-v1"
FINALIZATION_ROW_SCHEMA = "motive-goku-full-motion-finalization-row-v1"
POLICY_VERSION = "full-motion-all-source-dynamics-v1"

DEFAULT_PRIMARY_SIZE = 256
DEFAULT_RESERVE_SIZE = 64
DEFAULT_MIN_PRIMARY_MULTI_DYNAMIC = 64
DEFAULT_TARGET_SIGNATURE_CAP = 32
DEFAULT_FAMILY_CAP = 32
DEFAULT_CANARY_IID = "1dbe39537c984690"
QWEN_SHARD_COUNT = 8

REVIEW_NAME = "review_candidates.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_IID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_GENERATION_KEYS = {
    "schema_version",
    "iid",
    "group_id",
    "family",
    "source_video",
    "resolved_source_video",
    "anchor_image",
    "resolved_anchor_image",
    "source_video_sha256",
    "anchor_sha256",
    "selected_media_evidence",
    "selected_media_evidence_sha256",
    "strict_temporal_geometry",
    "edit_instruction",
    "edit_instruction_sha256",
    "motion_spec",
    "motion_spec_sha256",
    "qwen_evidence",
    "full_motion_finalization",
    "action_change_substantive",
    "manifest_role",
    "human_review_status",
    "generation_authorized",
    "production_eligible",
    "approval",
    "authorization_interface_available",
    "annotation_source",
    "human_reviewed",
}
_MOTION_SPEC_KEYS = {
    "schema_version",
    "change_region_proposals",
    "coverage_authority",
    "i0_grounding",
    "source_census",
    "secondary_source_census",
    "source_inventory_alignment",
    "coverage_authority_alignment",
    "target_plan",
    "compiled_instruction",
    "coverage_critic",
    "full_motion_contract",
    "qwen_result_digest",
    "qwen_provenance_digest",
}
_TEMPORAL_KEYS = {
    "schema_version",
    "source_frame_count",
    "source_frame_rate",
    "source_timeline_span_seconds",
    "target_frame_count",
    "target_frame_rate",
    "target_timeline_span_seconds",
    "requires_exact_frame_count_and_rate_match",
}
_QWEN_EVIDENCE_KEYS = {
    "schema_version",
    "record_schema_version",
    "input_digest",
    "result_digest",
    "provenance_digest",
    "config_digest",
    "run_config_digest",
    "implementation_digest",
    "visual_input_digest",
    "media_verification",
    "hard_gate",
    "change_region_proposals_digest",
    "coverage_authority_inventory_prompt_digest",
    "coverage_authority_inventory_visual_input_digest",
    "coverage_authority_inventory_digest",
    "coverage_authority_assignments_prompt_digest",
    "coverage_authority_assignments_visual_input_digest",
    "coverage_authority_assignments_digest",
    "coverage_authority_digest",
    "coverage_authority_alignment_digest",
    "i0_grounding_digest",
    "source_census_canonicalization",
    "source_census_canonicalization_digest",
    "source_census_digest",
    "secondary_source_census_canonicalization",
    "secondary_source_census_canonicalization_digest",
    "secondary_source_census_digest",
    "source_inventory_alignment_digest",
    "target_plan_canonicalization",
    "target_plan_canonicalization_digest",
    "target_plan_digest",
    "compiled_instruction_digest",
    "full_motion_contract_digest",
    "coverage_critic_digest",
    "shard_index",
    "num_shards",
    "receipt_digest",
    "receipt_sha256",
    "output_sha256",
    "model_path",
    "model_revision",
    "transformers_version",
    "qwen_record_payload",
}
_FINALIZATION_KEYS = {
    "schema_version",
    "policy_version",
    "candidate_rank",
    "review_rank",
    "selection_bucket",
    "dynamic_unit_count",
    "target_action_signatures",
    "family",
    "required_canary",
    "qwen_shard_index",
    "qwen_receipt_digest",
}


class GokuFullMotionFinalizeError(RuntimeError):
    """An input, quota, receipt, or output closure is invalid."""


def _reject_constant(value: str) -> None:
    raise GokuFullMotionFinalizeError(f"non-finite JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise GokuFullMotionFinalizeError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _parse_json(raw: bytes, *, context: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GokuFullMotionFinalizeError(f"{context} is not UTF-8") from error
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        if isinstance(error, GokuFullMotionFinalizeError):
            raise
        raise GokuFullMotionFinalizeError(
            f"{context} is not strict JSON: {error}"
        ) from error


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(dict(row)) + b"\n" for row in rows)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _object_digest(value: Any) -> str:
    return _sha256(_canonical_bytes(value))


def _stable_read(path: Path, *, context: str) -> bytes:
    unresolved = Path(os.path.abspath(path.expanduser()))
    if unresolved.is_symlink() or not unresolved.is_file():
        raise GokuFullMotionFinalizeError(
            f"{context} must be a regular non-symlink file: {unresolved}"
        )
    before = unresolved.stat()
    raw = unresolved.read_bytes()
    after = unresolved.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or len(raw) != after.st_size:
        raise GokuFullMotionFinalizeError(
            f"{context} changed while it was read: {unresolved}"
        )
    return raw


def _strict_object(path: Path, *, context: str) -> tuple[dict[str, Any], bytes]:
    raw = _stable_read(path, context=context)
    value = _parse_json(raw, context=context)
    if not isinstance(value, dict):
        raise GokuFullMotionFinalizeError(f"{context} must contain one object")
    return value, raw


def _strict_jsonl(
    path: Path,
    *,
    context: str,
    allow_empty: bool = False,
) -> tuple[list[dict[str, Any]], bytes]:
    raw = _stable_read(path, context=context)
    if not raw:
        if allow_empty:
            return [], raw
        raise GokuFullMotionFinalizeError(f"{context} is empty")
    if not raw.endswith(b"\n"):
        raise GokuFullMotionFinalizeError(
            f"{context} must be newline terminated"
        )
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise GokuFullMotionFinalizeError(
                f"{context} has a blank line at {line_number}"
            )
        value = _parse_json(line, context=f"{context}:{line_number}")
        if not isinstance(value, dict):
            raise GokuFullMotionFinalizeError(
                f"{context}:{line_number} is not an object"
            )
        rows.append(value)
    return rows, raw


def _text(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise GokuFullMotionFinalizeError(f"{context} must be non-empty text")
    return value


def _digest(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise GokuFullMotionFinalizeError(f"{context} is not a lowercase SHA-256")
    return value


def _validate_canonicalization_receipt(
    value: Any,
    *,
    artifact_kind: str,
    canonical: Mapping[str, Any],
    expected_context: Mapping[str, Any],
    context: str,
) -> dict[str, Any]:
    """Validate the projected, digest-bound part of one Qwen v6 receipt."""

    from .goku_full_motion_contract import (
        MODEL_OUTPUT_CANONICALIZATION_POLICY,
        MODEL_OUTPUT_CANONICALIZATION_RECEIPT_SCHEMA,
    )

    keys = {
        "schema_version",
        "artifact_kind",
        "policy",
        "semantic_repair",
        "context",
        "raw_sha256",
        "canonical_sha256",
        "normalized_field_paths",
        "changed_field_paths",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise GokuFullMotionFinalizeError(f"{context} is not a closed schema")
    receipt = dict(value)
    if (
        receipt.get("schema_version")
        != MODEL_OUTPUT_CANONICALIZATION_RECEIPT_SCHEMA
        or receipt.get("artifact_kind") != artifact_kind
        or receipt.get("policy") != MODEL_OUTPUT_CANONICALIZATION_POLICY
        or receipt.get("semantic_repair") is not False
        or receipt.get("context") != dict(expected_context)
        or receipt.get("canonical_sha256") != _object_digest(canonical)
    ):
        raise GokuFullMotionFinalizeError(
            f"{context} canonical artifact binding differs"
        )
    _digest(receipt.get("raw_sha256"), context=f"{context}.raw_sha256")
    _digest(
        receipt.get("canonical_sha256"),
        context=f"{context}.canonical_sha256",
    )
    normalized = receipt.get("normalized_field_paths")
    changed = receipt.get("changed_field_paths")
    if (
        not isinstance(normalized, list)
        or not normalized
        or any(
            not isinstance(path, str) or not path or path != path.strip()
            for path in normalized
        )
        or len(set(normalized)) != len(normalized)
        or not isinstance(changed, list)
        or any(
            not isinstance(path, str) or not path or path != path.strip()
            for path in changed
        )
        or len(set(changed)) != len(changed)
        or not set(changed).issubset(normalized)
    ):
        raise GokuFullMotionFinalizeError(
            f"{context} normalized/changed path closure differs"
        )
    receipt_payload = dict(receipt)
    receipt_sha = receipt_payload.pop("receipt_sha256")
    if (
        _digest(receipt_sha, context=f"{context}.receipt_sha256")
        != _object_digest(receipt_payload)
    ):
        raise GokuFullMotionFinalizeError(f"{context} receipt SHA differs")
    return receipt


def _expected_qwen_result_payload(
    *,
    change_region_proposals: Mapping[str, Any],
    coverage_authority: Mapping[str, Any],
    i0_grounding: Mapping[str, Any],
    primary_source: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    secondary_source: Mapping[str, Any],
    secondary_receipt: Mapping[str, Any],
    inventory_alignment: Mapping[str, Any],
    coverage_authority_alignment: Mapping[str, Any],
    target_plan: Mapping[str, Any],
    target_receipt: Mapping[str, Any],
    compiled_instruction: Mapping[str, Any],
    full_motion_contract: Mapping[str, Any],
    coverage_critic: Mapping[str, Any],
    hard_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild the canonical successful-Qwen result without trusting a SHA."""

    return {
        "change_region_proposals": dict(change_region_proposals),
        "coverage_authority": dict(coverage_authority),
        "i0_grounding": dict(i0_grounding),
        "source_census": dict(primary_source),
        "source_census_canonicalization": dict(source_receipt),
        "secondary_source_census": dict(secondary_source),
        "secondary_source_census_canonicalization": dict(
            secondary_receipt
        ),
        "source_inventory_alignment": dict(inventory_alignment),
        "coverage_authority_alignment": dict(
            coverage_authority_alignment
        ),
        "target_plan": dict(target_plan),
        "target_plan_canonicalization": dict(target_receipt),
        "compiled_instruction": dict(compiled_instruction),
        "full_motion_contract": dict(full_motion_contract),
        "coverage_critic": dict(coverage_critic),
        "hard_gate": dict(hard_gate),
        "pipeline_stage": "coverage_critic",
        "pipeline_decision": "pass",
    }


def _validate_qwen_record_payload(
    value: Any,
    *,
    row: Mapping[str, Any],
    evidence: Mapping[str, Any],
    expected_result_payload: Mapping[str, Any],
    semantic_objects: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Close the full projected record and independently recompute both SHAs."""

    from . import goku_full_motion_qwen as qwen_module

    if not isinstance(value, Mapping) or set(value) != qwen_module._RECORD_KEYS:
        raise GokuFullMotionFinalizeError(
            "qwen_record_payload is not the closed Qwen v6 record"
        )
    record = dict(value)
    scalar_bindings = {
        "schema_version": evidence.get("record_schema_version"),
        "iid": row.get("iid"),
        "group_id": row.get("group_id"),
        "family": row.get("family"),
        "status": "ok",
        "error_type": None,
        "error": None,
        "input_digest": evidence.get("input_digest"),
        "config_digest": evidence.get("config_digest"),
        "run_config_digest": evidence.get("run_config_digest"),
        "implementation_digest": evidence.get("implementation_digest"),
        "model_path": evidence.get("model_path"),
        "model_revision": evidence.get("model_revision"),
        "transformers_version": evidence.get("transformers_version"),
        "shard_index": evidence.get("shard_index"),
        "num_shards": evidence.get("num_shards"),
        "resolved_src_video": row.get("resolved_source_video"),
        "resolved_anchor_image": row.get("resolved_anchor_image"),
        "visual_input_digest": evidence.get("visual_input_digest"),
        "failure_stage": None,
        "pipeline_stage": "coverage_critic",
        "pipeline_decision": "pass",
        "result_digest": evidence.get("result_digest"),
        "provenance_digest": evidence.get("provenance_digest"),
    }
    if any(
        record.get(field) != expected
        for field, expected in scalar_bindings.items()
    ):
        raise GokuFullMotionFinalizeError(
            "qwen_record_payload scalar provenance binding differs"
        )
    object_bindings = {
        "media_verification": evidence.get("media_verification"),
        **semantic_objects,
    }
    if any(
        record.get(field) != expected
        for field, expected in object_bindings.items()
    ):
        raise GokuFullMotionFinalizeError(
            "qwen_record_payload semantic artifact binding differs"
        )
    digest_fields = (
        "change_region_proposals_digest",
        "coverage_authority_inventory_prompt_digest",
        "coverage_authority_inventory_visual_input_digest",
        "coverage_authority_inventory_digest",
        "coverage_authority_assignments_prompt_digest",
        "coverage_authority_assignments_visual_input_digest",
        "coverage_authority_assignments_digest",
        "coverage_authority_digest",
        "coverage_authority_alignment_digest",
        "i0_grounding_digest",
        "source_census_canonicalization_digest",
        "source_census_digest",
        "secondary_source_census_canonicalization_digest",
        "secondary_source_census_digest",
        "source_inventory_alignment_digest",
        "target_plan_canonicalization_digest",
        "target_plan_digest",
        "compiled_instruction_digest",
        "full_motion_contract_digest",
        "coverage_critic_digest",
    )
    if any(
        record.get(field) != evidence.get(field) for field in digest_fields
    ):
        raise GokuFullMotionFinalizeError(
            "qwen_record_payload artifact digest binding differs"
        )
    authority = semantic_objects.get("coverage_authority")
    if not isinstance(authority, Mapping):
        raise GokuFullMotionFinalizeError(
            "qwen_record_payload coverage authority is malformed"
        )
    inventory = authority.get("inventory")
    assignments = authority.get("assignments")
    try:
        inventory_raw = qwen_module.coverage_authority_validated_raw(
            record, stage="coverage_authority_inventory"
        )
        validated_inventory, inventory_validated_from = (
            qwen_module._validate_original_a0_output(
                stage="coverage_authority_inventory",
                original_raw=inventory_raw,
                validator=lambda value: (
                    qwen_module.validate_coverage_authority_inventory(
                        value, expected_iid=str(row.get("iid"))
                    )
                ),
                canonicalizer=lambda value: (
                    qwen_module.canonicalize_coverage_authority_inventory_model_output(
                        value, expected_iid=str(row.get("iid"))
                    )
                ),
            )
        )
        assignments_raw = qwen_module.coverage_authority_validated_raw(
            record, stage="coverage_authority_assignments"
        )
        validated_assignments, assignments_validated_from = (
            qwen_module._validate_original_a0_output(
                stage="coverage_authority_assignments",
                original_raw=assignments_raw,
                validator=lambda value: (
                    qwen_module.validate_coverage_authority_assignments(
                        value,
                        expected_iid=str(row.get("iid")),
                        coverage_authority_inventory=validated_inventory,
                        change_region_proposals=semantic_objects[
                            "change_region_proposals"
                        ],
                    )
                ),
                canonicalizer=lambda value: (
                    qwen_module.canonicalize_coverage_authority_assignments_model_output(
                        value,
                        expected_iid=str(row.get("iid")),
                        coverage_authority_inventory=validated_inventory,
                        change_region_proposals=semantic_objects[
                            "change_region_proposals"
                        ],
                    )
                ),
            )
        )
        rebuilt_authority = qwen_module.build_coverage_authority(
            coverage_authority_inventory=validated_inventory,
            coverage_authority_assignments=validated_assignments,
            change_region_proposals=semantic_objects["change_region_proposals"],
        )
    except Exception as error:
        raise GokuFullMotionFinalizeError(
            f"qwen_record_payload two-stage A0 raw closure differs: {error}"
        ) from error
    if (
        record.get("coverage_authority_inventory_validated_from")
        != inventory_validated_from
        or record.get("coverage_authority_assignments_validated_from")
        != assignments_validated_from
        or record.get("coverage_authority_inventory_digest")
        != qwen_module.object_sha256(validated_inventory)
        or record.get("coverage_authority_assignments_digest")
        != qwen_module.object_sha256(validated_assignments)
        or
        validated_inventory != inventory
        or validated_assignments != assignments
        or rebuilt_authority != authority
    ):
        raise GokuFullMotionFinalizeError(
            "qwen_record_payload two-stage A0 raw/object binding differs"
        )
    try:
        selected_target_raw = qwen_module.target_plan_validated_raw(
            record,
            source_census=semantic_objects["source_census"],
        )
        (
            _parsed_target_raw,
            validated_target_plan,
            validated_target_receipt,
        ) = qwen_module._canonicalize_target_plan_raw(
            selected_target_raw,
            stage="stored selected PASS_B target plan",
            source_census=semantic_objects["source_census"],
        )
    except Exception as error:
        raise GokuFullMotionFinalizeError(
            f"qwen_record_payload PASS_B selected raw closure differs: {error}"
        ) from error
    if (
        validated_target_plan != semantic_objects["target_plan"]
        or validated_target_receipt
        != record.get("target_plan_canonicalization")
        or record.get("target_plan_digest")
        != qwen_module.object_sha256(validated_target_plan)
        or record.get("target_plan_canonicalization_digest")
        != qwen_module.object_sha256(validated_target_receipt)
    ):
        raise GokuFullMotionFinalizeError(
            "qwen_record_payload PASS_B selected raw/object binding differs"
        )
    actual_result_payload = qwen_module.qwen_result_payload(record)
    if actual_result_payload != dict(expected_result_payload):
        raise GokuFullMotionFinalizeError(
            "qwen_record_payload canonical result projection differs"
        )
    result_digest = _object_digest(expected_result_payload)
    if (
        record.get("result_digest") != result_digest
        or evidence.get("result_digest") != result_digest
    ):
        raise GokuFullMotionFinalizeError(
            "Qwen result digest is not recomputed from the canonical payload"
        )
    provenance_digest = qwen_module.qwen_provenance_digest(record)
    if (
        record.get("provenance_digest") != provenance_digest
        or evidence.get("provenance_digest") != provenance_digest
    ):
        raise GokuFullMotionFinalizeError(
            "Qwen provenance digest is not recomputed from the full record"
        )
    return record


def _safe_iid(value: Any, *, context: str) -> str:
    iid = _text(value, context=context)
    if _SAFE_IID_RE.fullmatch(iid) is None or iid in {".", ".."}:
        raise GokuFullMotionFinalizeError(f"{context} is unsafe")
    return iid


def _load_qwen_api() -> Any:
    from . import goku_full_motion_qwen

    return goku_full_motion_qwen


def _validate_candidate_rows(
    path: Path,
) -> tuple[list[dict[str, Any]], bytes, dict[str, dict[str, Any]]]:
    rows, raw = _strict_jsonl(path, context="candidate manifest")
    by_iid: dict[str, dict[str, Any]] = {}
    groups: set[str] = set()
    for index, row in enumerate(rows, start=1):
        iid = _safe_iid(row.get("iid"), context=f"candidate[{index}].iid")
        if iid in by_iid:
            raise GokuFullMotionFinalizeError(f"duplicate candidate IID: {iid}")
        group = _text(
            row.get("group_id"), context=f"candidate iid={iid} group_id"
        )
        if group in groups:
            raise GokuFullMotionFinalizeError(
                f"duplicate candidate group_id: {group}"
            )
        _text(row.get("family"), context=f"candidate iid={iid} family")
        for field in (
            "src_video",
            "resolved_src_video",
            "anchor_image",
            "resolved_anchor_image",
            "prompt",
        ):
            _text(row.get(field), context=f"candidate iid={iid} {field}")
        for field in ("source_video_sha256", "anchor_sha256"):
            _digest(row.get(field), context=f"candidate iid={iid} {field}")
        media = row.get("media")
        if not isinstance(media, Mapping):
            raise GokuFullMotionFinalizeError(
                f"candidate iid={iid} media must be an object"
            )
        if media.get("frame_count") != 81:
            raise GokuFullMotionFinalizeError(
                f"candidate iid={iid} must have exactly 81 frames"
            )
        fps = media.get("fps")
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not math.isfinite(float(fps))
            or not math.isclose(float(fps), 25.0, rel_tol=0.0, abs_tol=1e-9)
        ):
            raise GokuFullMotionFinalizeError(
                f"candidate iid={iid} must have exactly 25 FPS"
            )
        by_iid[iid] = row
        groups.add(group)
    return rows, raw, by_iid


def _receipt_backend(receipt: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        model_path=receipt.get("model_path"),
        model_revision=receipt.get("model_revision"),
        transformers_version=receipt.get("transformers_version"),
    )


def _load_qwen_shards(
    *,
    candidate_path: Path,
    candidate_rows: Sequence[Mapping[str, Any]],
    candidate_raw: bytes,
    selected_by_iid: Mapping[str, Mapping[str, Any]],
    qwen_outputs: Sequence[Path],
    api: Any,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    if len(qwen_outputs) != QWEN_SHARD_COUNT:
        raise GokuFullMotionFinalizeError(
            f"exactly {QWEN_SHARD_COUNT} Qwen outputs are required"
        )
    candidate_sha = _sha256(candidate_raw)
    records_by_iid: dict[str, dict[str, Any]] = {}
    receipt_by_iid: dict[str, dict[str, Any]] = {}
    artifacts: list[dict[str, Any]] = []
    seen_shards: set[int] = set()
    common: dict[str, Any] | None = None

    for supplied in qwen_outputs:
        output = Path(os.path.abspath(supplied.expanduser()))
        output_rows, output_raw = _strict_jsonl(
            output,
            context=f"Qwen output {output}",
            allow_empty=True,
        )
        receipt_path = api.shard_receipt_path(output)
        receipt, receipt_raw = _strict_object(
            receipt_path, context=f"Qwen terminal receipt {receipt_path}"
        )
        shard_index = receipt.get("shard_index")
        num_shards = receipt.get("num_shards")
        if type(shard_index) is not int or not 0 <= shard_index < QWEN_SHARD_COUNT:
            raise GokuFullMotionFinalizeError("invalid Qwen receipt shard_index")
        if num_shards != QWEN_SHARD_COUNT:
            raise GokuFullMotionFinalizeError("Qwen receipt is not an 8-shard run")
        if shard_index in seen_shards:
            raise GokuFullMotionFinalizeError(
                f"duplicate Qwen shard receipt: {shard_index}"
            )
        seen_shards.add(shard_index)
        run_config = receipt.get("run_config")
        if not isinstance(run_config, Mapping):
            raise GokuFullMotionFinalizeError("Qwen receipt run_config is malformed")
        max_samples = run_config.get("max_samples")
        if max_samples is not None and type(max_samples) is not int:
            raise GokuFullMotionFinalizeError("Qwen max_samples binding is malformed")
        assigned_iids = api.assigned_iids_for_shard(
            candidate_rows,
            shard_index=shard_index,
            num_shards=QWEN_SHARD_COUNT,
            max_samples=max_samples,
        )
        root_value = receipt.get("root")
        if not isinstance(root_value, str) or not root_value:
            raise GokuFullMotionFinalizeError("Qwen receipt root is malformed")
        backend = _receipt_backend(receipt)
        try:
            validated_receipt = api.validate_shard_receipt(
                receipt,
                output=output,
                input_path=candidate_path,
                input_sha256=candidate_sha,
                root=Path(root_value),
                assigned_iids=assigned_iids,
                selected_by_iid=selected_by_iid,
                shard_index=shard_index,
                num_shards=QWEN_SHARD_COUNT,
                implementation_digest=receipt.get("implementation_digest"),
                config_digest=receipt.get("config_digest"),
                run_config_digest=receipt.get("run_config_digest"),
                run_config=run_config,
                backend=backend,
            )
        except Exception as error:
            raise GokuFullMotionFinalizeError(
                f"Qwen shard {shard_index} receipt validation failed: {error}"
            ) from error

        if [str(row.get("iid") or "") for row in output_rows] != assigned_iids:
            raise GokuFullMotionFinalizeError(
                f"Qwen shard {shard_index} output order/coverage differs"
            )
        receipt_digest = _digest(
            validated_receipt.get("receipt_digest"),
            context=f"Qwen shard {shard_index} receipt_digest",
        )
        receipt_sha = _sha256(receipt_raw)
        output_sha = _sha256(output_raw)
        if validated_receipt.get("output", {}).get("sha256") != output_sha:
            raise GokuFullMotionFinalizeError(
                f"Qwen shard {shard_index} output hash differs"
            )
        expected_common = {
            "execution_manifest": str(candidate_path),
            "execution_manifest_sha256": candidate_sha,
            "root": root_value,
            "implementation_digest": receipt.get("implementation_digest"),
            "run_config_digest": receipt.get("run_config_digest"),
            "run_config": dict(run_config),
            "model_path": receipt.get("model_path"),
            "model_revision": receipt.get("model_revision"),
            "transformers_version": receipt.get("transformers_version"),
        }
        if common is None:
            common = expected_common
        elif expected_common != common:
            raise GokuFullMotionFinalizeError(
                "Qwen shard receipts do not describe one common run"
            )

        expected_bindings = {
            "execution_manifest": str(candidate_path),
            "execution_manifest_sha256": candidate_sha,
            "shard_index": shard_index,
            "num_shards": QWEN_SHARD_COUNT,
            "implementation_digest": receipt.get("implementation_digest"),
            "config_digest": receipt.get("config_digest"),
            "run_config_digest": receipt.get("run_config_digest"),
            "model_path": receipt.get("model_path"),
            "model_revision": receipt.get("model_revision"),
            "transformers_version": receipt.get("transformers_version"),
        }
        for record in output_rows:
            iid = str(record.get("iid") or "")
            if iid not in selected_by_iid or iid in records_by_iid:
                raise GokuFullMotionFinalizeError(
                    f"duplicate or unknown Qwen output IID: {iid!r}"
                )
            try:
                validated = api.validate_output_record(
                    record,
                    selected_row=selected_by_iid[iid],
                    expected_bindings=expected_bindings,
                )
                if validated.get("status") == "ok":
                    payload = api.qwen_result_payload(validated)
                    if _object_digest(payload) != validated.get("result_digest"):
                        raise GokuFullMotionFinalizeError(
                            f"Qwen result payload digest differs for iid={iid}"
                        )
                    if api.qwen_provenance_digest(validated) != validated.get(
                        "provenance_digest"
                    ):
                        raise GokuFullMotionFinalizeError(
                            f"Qwen provenance digest differs for iid={iid}"
                        )
            except Exception as error:
                if isinstance(error, GokuFullMotionFinalizeError):
                    raise
                raise GokuFullMotionFinalizeError(
                    f"Qwen output validation failed for iid={iid}: {error}"
                ) from error
            records_by_iid[iid] = validated
            receipt_by_iid[iid] = {
                "shard_index": shard_index,
                "receipt_digest": receipt_digest,
                "receipt_sha256": receipt_sha,
                "output_sha256": output_sha,
            }
        if _stable_read(output, context=f"Qwen output {output}") != output_raw:
            raise GokuFullMotionFinalizeError(
                f"Qwen shard {shard_index} changed during validation"
            )
        artifacts.append(
            {
                "shard_index": shard_index,
                "output_path": str(output.resolve(strict=True)),
                "output_sha256": output_sha,
                "output_bytes": len(output_raw),
                "output_rows": len(output_rows),
                "receipt_path": str(receipt_path.resolve(strict=True)),
                "receipt_sha256": receipt_sha,
                "receipt_bytes": len(receipt_raw),
                "receipt_digest": receipt_digest,
                "assigned_iids": assigned_iids,
            }
        )

    if seen_shards != set(range(QWEN_SHARD_COUNT)):
        raise GokuFullMotionFinalizeError("Qwen terminal shard set is incomplete")
    expected_iids = [str(row["iid"]) for row in candidate_rows]
    if set(records_by_iid) != set(expected_iids):
        missing = sorted(set(expected_iids) - set(records_by_iid))
        extra = sorted(set(records_by_iid) - set(expected_iids))
        raise GokuFullMotionFinalizeError(
            f"Qwen IID closure differs: missing={missing} extra={extra}"
        )
    return records_by_iid, receipt_by_iid, sorted(
        artifacts, key=lambda item: int(item["shard_index"])
    )


def _target_signatures(record: Mapping[str, Any]) -> list[str]:
    plan = record.get("target_plan")
    if not isinstance(plan, Mapping):
        raise GokuFullMotionFinalizeError("hard-pass record lacks target_plan")
    targets = plan.get("dynamic_unit_targets")
    if not isinstance(targets, list) or not targets:
        raise GokuFullMotionFinalizeError(
            "hard-pass target_plan has no dynamic targets"
        )
    signatures: list[str] = []
    for index, target in enumerate(targets):
        if not isinstance(target, Mapping):
            raise GokuFullMotionFinalizeError(
                f"target dynamic unit {index} is malformed"
            )
        signature = _text(
            target.get("target_action_signature"),
            context=f"target dynamic unit {index} signature",
        )
        signatures.append(signature)
    return signatures


def _dynamic_count(record: Mapping[str, Any]) -> int:
    census = record.get("source_census")
    if not isinstance(census, Mapping):
        raise GokuFullMotionFinalizeError("hard-pass record lacks source_census")
    units = census.get("dynamic_units")
    if not isinstance(units, list) or not units:
        raise GokuFullMotionFinalizeError("source census has no dynamic units")
    return len(units)


def _can_add(
    entry: Mapping[str, Any],
    *,
    signature_counts: Mapping[str, int],
    family_counts: Mapping[str, int],
    target_signature_cap: int,
    family_cap: int,
) -> bool:
    if family_counts.get(str(entry["family_key"]), 0) >= family_cap:
        return False
    return all(
        signature_counts.get(signature, 0) < target_signature_cap
        for signature in entry["signature_keys"]
    )


def _increment_caps(
    entry: Mapping[str, Any],
    *,
    signature_counts: dict[str, int],
    family_counts: dict[str, int],
) -> None:
    family = str(entry["family_key"])
    family_counts[family] = family_counts.get(family, 0) + 1
    for signature in entry["signature_keys"]:
        signature_counts[str(signature)] = (
            signature_counts.get(str(signature), 0) + 1
        )


def _select_entries(
    entries: Sequence[dict[str, Any]],
    *,
    primary_size: int,
    reserve_size: int,
    min_primary_multi_dynamic: int,
    target_signature_cap: int,
    family_cap: int,
    required_iids: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_iid = {str(entry["iid"]): entry for entry in entries}
    missing = [iid for iid in required_iids if iid not in by_iid]
    if missing:
        raise GokuFullMotionFinalizeError(
            f"required canary did not hard-pass Qwen: {missing}"
        )
    if len(required_iids) > primary_size:
        raise GokuFullMotionFinalizeError(
            "required canaries exceed the primary size"
        )

    signature_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    primary_iids: set[str] = set()
    reserve_iids: set[str] = set()

    def add_primary(entry: dict[str, Any], *, required: bool = False) -> None:
        if not _can_add(
            entry,
            signature_counts=signature_counts,
            family_counts=family_counts,
            target_signature_cap=target_signature_cap,
            family_cap=family_cap,
        ):
            reason = "required canary violates diversity caps" if required else (
                "internal primary cap violation"
            )
            raise GokuFullMotionFinalizeError(
                f"{reason}: iid={entry['iid']}"
            )
        primary_iids.add(str(entry["iid"]))
        _increment_caps(
            entry,
            signature_counts=signature_counts,
            family_counts=family_counts,
        )

    for iid in required_iids:
        add_primary(by_iid[iid], required=True)

    multi_count = sum(
        int(by_iid[iid]["dynamic_unit_count"]) >= 2 for iid in primary_iids
    )
    for entry in entries:
        if (
            multi_count >= min_primary_multi_dynamic
            or len(primary_iids) >= primary_size
        ):
            break
        iid = str(entry["iid"])
        if iid in primary_iids or int(entry["dynamic_unit_count"]) < 2:
            continue
        if not _can_add(
            entry,
            signature_counts=signature_counts,
            family_counts=family_counts,
            target_signature_cap=target_signature_cap,
            family_cap=family_cap,
        ):
            continue
        add_primary(entry)
        multi_count += 1
    if multi_count < min_primary_multi_dynamic:
        raise GokuFullMotionFinalizeError(
            "primary multi-dynamic quota cannot be satisfied: "
            f"required={min_primary_multi_dynamic} available={multi_count}"
        )

    for entry in entries:
        if len(primary_iids) >= primary_size:
            break
        iid = str(entry["iid"])
        if iid in primary_iids:
            continue
        if not _can_add(
            entry,
            signature_counts=signature_counts,
            family_counts=family_counts,
            target_signature_cap=target_signature_cap,
            family_cap=family_cap,
        ):
            continue
        add_primary(entry)
    if len(primary_iids) != primary_size:
        raise GokuFullMotionFinalizeError(
            "insufficient diverse hard-pass rows for primary: "
            f"required={primary_size} selected={len(primary_iids)}"
        )

    for entry in entries:
        if len(reserve_iids) >= reserve_size:
            break
        iid = str(entry["iid"])
        if iid in primary_iids:
            continue
        if not _can_add(
            entry,
            signature_counts=signature_counts,
            family_counts=family_counts,
            target_signature_cap=target_signature_cap,
            family_cap=family_cap,
        ):
            continue
        reserve_iids.add(iid)
        _increment_caps(
            entry,
            signature_counts=signature_counts,
            family_counts=family_counts,
        )
    if len(reserve_iids) != reserve_size:
        raise GokuFullMotionFinalizeError(
            "insufficient diverse hard-pass rows for reserve: "
            f"required={reserve_size} selected={len(reserve_iids)}"
        )

    primary = [entry for entry in entries if entry["iid"] in primary_iids]
    reserve = [entry for entry in entries if entry["iid"] in reserve_iids]
    evidence = {
        "primary_multi_dynamic_rows": sum(
            int(entry["dynamic_unit_count"]) >= 2 for entry in primary
        ),
        "target_signature_counts": dict(sorted(signature_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
    }
    return primary, reserve, evidence


def validate_generation_row(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the closed pending row shared with release and postcheck."""

    if not isinstance(value, Mapping) or set(value) != _GENERATION_KEYS:
        raise GokuFullMotionFinalizeError(
            "full-motion generation row is not a closed schema"
        )
    row = dict(value)
    if row.get("schema_version") != GENERATION_SCHEMA:
        raise GokuFullMotionFinalizeError("generation row schema differs")
    iid = _safe_iid(row.get("iid"), context="generation iid")
    for field in (
        "group_id",
        "family",
        "source_video",
        "resolved_source_video",
        "anchor_image",
        "resolved_anchor_image",
        "edit_instruction",
    ):
        _text(row.get(field), context=f"generation iid={iid} {field}")
    for field in (
        "source_video_sha256",
        "anchor_sha256",
        "selected_media_evidence_sha256",
        "edit_instruction_sha256",
        "motion_spec_sha256",
    ):
        _digest(row.get(field), context=f"generation iid={iid} {field}")
    if _sha256(row["edit_instruction"].encode("utf-8")) != row[
        "edit_instruction_sha256"
    ]:
        raise GokuFullMotionFinalizeError("generation instruction SHA differs")

    media = row.get("selected_media_evidence")
    if not isinstance(media, Mapping):
        raise GokuFullMotionFinalizeError("selected media evidence is malformed")
    if _object_digest(media) != row["selected_media_evidence_sha256"]:
        raise GokuFullMotionFinalizeError("selected media evidence SHA differs")
    fps = media.get("fps")
    if media.get("frame_count") != 81 or (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isclose(float(fps), 25.0, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise GokuFullMotionFinalizeError("generation source geometry differs")

    temporal = row.get("strict_temporal_geometry")
    expected_temporal = {
        "schema_version": TEMPORAL_GEOMETRY_SCHEMA,
        "source_frame_count": 81,
        "source_frame_rate": "25/1",
        "source_timeline_span_seconds": 3.2,
        "target_frame_count": 81,
        "target_frame_rate": "25/1",
        "target_timeline_span_seconds": 3.2,
        "requires_exact_frame_count_and_rate_match": True,
    }
    if not isinstance(temporal, Mapping) or set(temporal) != _TEMPORAL_KEYS:
        raise GokuFullMotionFinalizeError("temporal geometry is not closed")
    if dict(temporal) != expected_temporal:
        raise GokuFullMotionFinalizeError("temporal geometry differs")

    spec = row.get("motion_spec")
    if not isinstance(spec, Mapping) or set(spec) != _MOTION_SPEC_KEYS:
        raise GokuFullMotionFinalizeError("motion_spec is not a closed schema")
    if spec.get("schema_version") != MOTION_SPEC_SCHEMA:
        raise GokuFullMotionFinalizeError("motion_spec schema differs")
    if _object_digest(spec) != row["motion_spec_sha256"]:
        raise GokuFullMotionFinalizeError("motion_spec SHA differs")
    compiled = spec.get("compiled_instruction")
    if not isinstance(compiled, Mapping):
        raise GokuFullMotionFinalizeError("compiled_instruction is malformed")
    if (
        compiled.get("edit_instruction") != row["edit_instruction"]
        or compiled.get("instruction_sha256")
        != row["edit_instruction_sha256"]
    ):
        raise GokuFullMotionFinalizeError(
            "compiled_instruction differs from executable instruction"
        )

    evidence = row.get("qwen_evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != _QWEN_EVIDENCE_KEYS:
        raise GokuFullMotionFinalizeError("qwen_evidence is not a closed schema")
    if evidence.get("schema_version") != QWEN_EVIDENCE_SCHEMA:
        raise GokuFullMotionFinalizeError("qwen_evidence schema differs")
    for field in (
        "input_digest",
        "result_digest",
        "provenance_digest",
        "config_digest",
        "run_config_digest",
        "implementation_digest",
        "visual_input_digest",
        "change_region_proposals_digest",
        "coverage_authority_inventory_prompt_digest",
        "coverage_authority_inventory_visual_input_digest",
        "coverage_authority_inventory_digest",
        "coverage_authority_assignments_prompt_digest",
        "coverage_authority_assignments_visual_input_digest",
        "coverage_authority_assignments_digest",
        "coverage_authority_digest",
        "coverage_authority_alignment_digest",
        "i0_grounding_digest",
        "source_census_canonicalization_digest",
        "source_census_digest",
        "secondary_source_census_canonicalization_digest",
        "secondary_source_census_digest",
        "source_inventory_alignment_digest",
        "target_plan_canonicalization_digest",
        "target_plan_digest",
        "compiled_instruction_digest",
        "full_motion_contract_digest",
        "coverage_critic_digest",
        "receipt_digest",
        "receipt_sha256",
        "output_sha256",
    ):
        _digest(evidence.get(field), context=f"qwen_evidence.{field}")

    # The finalizer is the projection boundary between the complete Qwen record
    # and the compact generation manifest.  Preserve enough independent source
    # inventory and canonicalization evidence to re-run the deterministic v6
    # closure; accepting only canonical objects would discard how safe redundant
    # fields were mechanically derived from the raw visual-model response.
    from .goku_full_motion_contract import (
        build_contract,
        object_sha256 as contract_object_sha256,
        validate_contract_binding,
        validate_coverage_critic,
        validate_source_census,
        validate_source_inventory_alignment,
        validate_target_plan,
    )
    from .goku_full_motion_instruction import validate_compiled_instruction
    from .goku_full_motion_qwen import (
        build_hard_gate,
        validate_change_region_proposals,
        validate_coverage_authority,
        validate_coverage_authority_alignment,
        validate_i0_grounding,
        validate_source_census_i0_binding,
    )

    try:
        i0_grounding = validate_i0_grounding(
            spec.get("i0_grounding"), expected_iid=iid
        )
        primary_source = validate_source_census(spec.get("source_census"))
        primary_source = validate_source_census_i0_binding(
            primary_source, i0_grounding
        )
        secondary_source = validate_source_census(
            spec.get("secondary_source_census")
        )
        secondary_source = validate_source_census_i0_binding(
            secondary_source, i0_grounding
        )
        inventory_alignment = validate_source_inventory_alignment(
            spec.get("source_inventory_alignment"),
            primary=primary_source,
            secondary=secondary_source,
        )
        change_region_proposals = validate_change_region_proposals(
            spec.get("change_region_proposals"), expected_iid=iid
        )
        coverage_authority = validate_coverage_authority(
            spec.get("coverage_authority"),
            expected_iid=iid,
            change_region_proposals=change_region_proposals,
        )
        coverage_authority_alignment = validate_coverage_authority_alignment(
            spec.get("coverage_authority_alignment"),
            coverage_authority=coverage_authority,
            change_region_proposals=change_region_proposals,
            i0_grounding=i0_grounding,
            primary=primary_source,
            secondary=secondary_source,
            source_inventory_alignment=inventory_alignment,
        )
        target_plan = validate_target_plan(
            spec.get("target_plan"), source_census=primary_source
        )
        compiled_instruction = validate_compiled_instruction(
            spec.get("compiled_instruction"),
            source_census=primary_source,
            target_plan=target_plan,
        )
        coverage_critic = validate_coverage_critic(
            spec.get("coverage_critic"),
            source_census=primary_source,
            target_plan=target_plan,
            compiled_instruction=compiled_instruction,
        )
        full_motion_contract = validate_contract_binding(
            spec.get("full_motion_contract"),
            source_census=primary_source,
            target_plan=target_plan,
        )
        if full_motion_contract != build_contract(
            source_census=primary_source, target_plan=target_plan
        ):
            raise ValueError("full-motion contract is not deterministic")
    except Exception as error:
        raise GokuFullMotionFinalizeError(
            f"Qwen v6 coverage-authority/exact-I0 closure differs: {error}"
        ) from error

    proposals_sha = contract_object_sha256(change_region_proposals)
    authority_inventory_sha = contract_object_sha256(
        coverage_authority["inventory"]
    )
    authority_assignments_sha = contract_object_sha256(
        coverage_authority["assignments"]
    )
    authority_sha = contract_object_sha256(coverage_authority)
    authority_alignment_sha = contract_object_sha256(
        coverage_authority_alignment
    )
    i0_grounding_sha = contract_object_sha256(i0_grounding)
    primary_sha = contract_object_sha256(primary_source)
    secondary_sha = contract_object_sha256(secondary_source)
    alignment_sha = contract_object_sha256(inventory_alignment)
    source_receipt = _validate_canonicalization_receipt(
        evidence.get("source_census_canonicalization"),
        artifact_kind="source_census",
        canonical=primary_source,
        expected_context={"expected_iid": primary_source["iid"]},
        context="qwen_evidence.source_census_canonicalization",
    )
    secondary_receipt = _validate_canonicalization_receipt(
        evidence.get("secondary_source_census_canonicalization"),
        artifact_kind="source_census",
        canonical=secondary_source,
        expected_context={"expected_iid": primary_source["iid"]},
        context="qwen_evidence.secondary_source_census_canonicalization",
    )
    target_receipt = _validate_canonicalization_receipt(
        evidence.get("target_plan_canonicalization"),
        artifact_kind="target_plan",
        canonical=target_plan,
        expected_context={
            "iid": primary_source["iid"],
            "source_census_sha256": primary_sha,
        },
        context="qwen_evidence.target_plan_canonicalization",
    )
    source_receipt_sha = contract_object_sha256(source_receipt)
    secondary_receipt_sha = contract_object_sha256(secondary_receipt)
    target_receipt_sha = contract_object_sha256(target_receipt)
    hard_gate = evidence.get("hard_gate")
    try:
        expected_hard_gate = build_hard_gate(
            change_region_proposals=change_region_proposals,
            coverage_authority=coverage_authority,
            coverage_authority_alignment=coverage_authority_alignment,
            i0_grounding=i0_grounding,
            source_census=primary_source,
            source_census_canonicalization=source_receipt,
            secondary_source_census=secondary_source,
            secondary_source_census_canonicalization=secondary_receipt,
            source_inventory_alignment=inventory_alignment,
            target_plan=target_plan,
            target_plan_canonicalization=target_receipt,
            compiled_instruction=compiled_instruction,
            coverage_critic=coverage_critic,
        )
    except Exception as error:
        raise GokuFullMotionFinalizeError(
            f"Qwen v6 hard-gate reconstruction failed: {error}"
        ) from error
    expected_result_payload = _expected_qwen_result_payload(
        change_region_proposals=change_region_proposals,
        coverage_authority=coverage_authority,
        i0_grounding=i0_grounding,
        primary_source=primary_source,
        source_receipt=source_receipt,
        secondary_source=secondary_source,
        secondary_receipt=secondary_receipt,
        inventory_alignment=inventory_alignment,
        coverage_authority_alignment=coverage_authority_alignment,
        target_plan=target_plan,
        target_receipt=target_receipt,
        compiled_instruction=compiled_instruction,
        full_motion_contract=full_motion_contract,
        coverage_critic=coverage_critic,
        hard_gate=expected_hard_gate,
    )
    _validate_qwen_record_payload(
        evidence.get("qwen_record_payload"),
        row=row,
        evidence=evidence,
        expected_result_payload=expected_result_payload,
        semantic_objects={
            "change_region_proposals": change_region_proposals,
            "coverage_authority": coverage_authority,
            "i0_grounding": i0_grounding,
            "source_census": primary_source,
            "source_census_canonicalization": source_receipt,
            "secondary_source_census": secondary_source,
            "secondary_source_census_canonicalization": secondary_receipt,
            "source_inventory_alignment": inventory_alignment,
            "coverage_authority_alignment": coverage_authority_alignment,
            "target_plan": target_plan,
            "target_plan_canonicalization": target_receipt,
            "compiled_instruction": compiled_instruction,
            "full_motion_contract": full_motion_contract,
            "coverage_critic": coverage_critic,
            "hard_gate": expected_hard_gate,
        },
    )
    if (
        evidence.get("record_schema_version")
        != "goku-full-motion-qwen-record-v6"
        or evidence.get("result_digest") != spec.get("qwen_result_digest")
        or evidence.get("provenance_digest")
        != spec.get("qwen_provenance_digest")
        or evidence.get("change_region_proposals_digest") != proposals_sha
        or evidence.get("coverage_authority_inventory_digest")
        != authority_inventory_sha
        or evidence.get("coverage_authority_assignments_digest")
        != authority_assignments_sha
        or evidence.get("coverage_authority_digest") != authority_sha
        or evidence.get("coverage_authority_alignment_digest")
        != authority_alignment_sha
        or evidence.get("i0_grounding_digest") != i0_grounding_sha
        or evidence.get("source_census_canonicalization_digest")
        != source_receipt_sha
        or evidence.get("source_census_digest") != primary_sha
        or evidence.get("secondary_source_census_canonicalization_digest")
        != secondary_receipt_sha
        or evidence.get("secondary_source_census_digest") != secondary_sha
        or evidence.get("source_inventory_alignment_digest") != alignment_sha
        or evidence.get("target_plan_canonicalization_digest")
        != target_receipt_sha
        or evidence.get("target_plan_digest")
        != contract_object_sha256(target_plan)
        or evidence.get("compiled_instruction_digest")
        != contract_object_sha256(compiled_instruction)
        or evidence.get("full_motion_contract_digest")
        != contract_object_sha256(full_motion_contract)
        or evidence.get("coverage_critic_digest")
        != contract_object_sha256(coverage_critic)
        or hard_gate != expected_hard_gate
        or expected_hard_gate.get("decision") != "pass"
        or expected_hard_gate.get("risk_codes") != []
    ):
        raise GokuFullMotionFinalizeError(
            "Qwen v6 coverage-authority/evidence/spec hard-pass binding differs"
        )

    finalization = row.get("full_motion_finalization")
    if (
        not isinstance(finalization, Mapping)
        or set(finalization) != _FINALIZATION_KEYS
    ):
        raise GokuFullMotionFinalizeError("finalization row is not closed")
    if (
        finalization.get("schema_version") != FINALIZATION_ROW_SCHEMA
        or finalization.get("policy_version") != POLICY_VERSION
        or finalization.get("selection_bucket")
        not in {"primary", "reserve", "review_only"}
        or finalization.get("family") != row["family"]
        or finalization.get("qwen_receipt_digest")
        != evidence.get("receipt_digest")
    ):
        raise GokuFullMotionFinalizeError("finalization binding differs")
    for field in ("candidate_rank", "review_rank", "dynamic_unit_count"):
        if type(finalization.get(field)) is not int or finalization[field] <= 0:
            raise GokuFullMotionFinalizeError(f"finalization {field} is invalid")
    if type(finalization.get("required_canary")) is not bool:
        raise GokuFullMotionFinalizeError("finalization canary flag is invalid")
    signatures = finalization.get("target_action_signatures")
    if (
        not isinstance(signatures, list)
        or not signatures
        or any(not isinstance(item, str) or not item for item in signatures)
    ):
        raise GokuFullMotionFinalizeError("target action signatures are invalid")
    targets = target_plan.get("dynamic_unit_targets")
    source = spec.get("source_census")
    if not isinstance(targets, list) or not isinstance(source, Mapping):
        raise GokuFullMotionFinalizeError("motion dynamic units are malformed")
    source_units = source.get("dynamic_units")
    if (
        not isinstance(source_units, list)
        or len(source_units) != finalization["dynamic_unit_count"]
        or any(not isinstance(item, Mapping) for item in targets)
        or [
            item.get("target_action_signature")
            for item in targets
            if isinstance(item, Mapping)
        ]
        != signatures
    ):
        raise GokuFullMotionFinalizeError("dynamic unit/signature binding differs")

    fixed = {
        "action_change_substantive": "yes",
        "manifest_role": "review_proposal",
        "human_review_status": "pending",
        "generation_authorized": False,
        "production_eligible": False,
        "approval": None,
        "authorization_interface_available": False,
        "annotation_source": "qwen3-vl-32b",
        "human_reviewed": False,
    }
    for field, expected in fixed.items():
        if row.get(field) != expected:
            raise GokuFullMotionFinalizeError(
                f"generation pending semantic {field} differs"
            )
    return row


def _generation_row(
    entry: Mapping[str, Any],
    *,
    selection_bucket: str,
    required_iids: set[str],
) -> dict[str, Any]:
    candidate = entry["candidate"]
    record = entry["record"]
    receipt = entry["receipt"]
    compiled = record["compiled_instruction"]
    instruction = _text(
        compiled.get("edit_instruction"), context="compiled edit_instruction"
    )
    instruction_sha = _digest(
        compiled.get("instruction_sha256"),
        context="compiled instruction_sha256",
    )
    if _sha256(instruction.encode("utf-8")) != instruction_sha:
        raise GokuFullMotionFinalizeError(
            f"compiled instruction text/hash differs for iid={entry['iid']}"
        )
    motion_spec = {
        "schema_version": MOTION_SPEC_SCHEMA,
        "change_region_proposals": record["change_region_proposals"],
        "coverage_authority": record["coverage_authority"],
        "i0_grounding": record["i0_grounding"],
        "source_census": record["source_census"],
        "secondary_source_census": record["secondary_source_census"],
        "source_inventory_alignment": record["source_inventory_alignment"],
        "coverage_authority_alignment": record[
            "coverage_authority_alignment"
        ],
        "target_plan": record["target_plan"],
        "compiled_instruction": compiled,
        "coverage_critic": record["coverage_critic"],
        "full_motion_contract": record["full_motion_contract"],
        "qwen_result_digest": record["result_digest"],
        "qwen_provenance_digest": record["provenance_digest"],
    }
    media = dict(candidate["media"])
    temporal = {
        "schema_version": TEMPORAL_GEOMETRY_SCHEMA,
        "source_frame_count": 81,
        "source_frame_rate": "25/1",
        "source_timeline_span_seconds": 3.2,
        "target_frame_count": 81,
        "target_frame_rate": "25/1",
        "target_timeline_span_seconds": 3.2,
        "requires_exact_frame_count_and_rate_match": True,
    }
    qwen_evidence = {
        "schema_version": QWEN_EVIDENCE_SCHEMA,
        "record_schema_version": record["schema_version"],
        "input_digest": record["input_digest"],
        "result_digest": record["result_digest"],
        "provenance_digest": record["provenance_digest"],
        "config_digest": record["config_digest"],
        "run_config_digest": record["run_config_digest"],
        "implementation_digest": record["implementation_digest"],
        "visual_input_digest": record["visual_input_digest"],
        "media_verification": record["media_verification"],
        "hard_gate": record["hard_gate"],
        "change_region_proposals_digest": record[
            "change_region_proposals_digest"
        ],
        "coverage_authority_inventory_prompt_digest": record[
            "coverage_authority_inventory_prompt_digest"
        ],
        "coverage_authority_inventory_visual_input_digest": record[
            "coverage_authority_inventory_visual_input_digest"
        ],
        "coverage_authority_inventory_digest": record[
            "coverage_authority_inventory_digest"
        ],
        "coverage_authority_assignments_prompt_digest": record[
            "coverage_authority_assignments_prompt_digest"
        ],
        "coverage_authority_assignments_visual_input_digest": record[
            "coverage_authority_assignments_visual_input_digest"
        ],
        "coverage_authority_assignments_digest": record[
            "coverage_authority_assignments_digest"
        ],
        "coverage_authority_digest": record["coverage_authority_digest"],
        "coverage_authority_alignment_digest": record[
            "coverage_authority_alignment_digest"
        ],
        "i0_grounding_digest": record["i0_grounding_digest"],
        "source_census_canonicalization": record[
            "source_census_canonicalization"
        ],
        "source_census_canonicalization_digest": record[
            "source_census_canonicalization_digest"
        ],
        "source_census_digest": record["source_census_digest"],
        "secondary_source_census_canonicalization": record[
            "secondary_source_census_canonicalization"
        ],
        "secondary_source_census_canonicalization_digest": record[
            "secondary_source_census_canonicalization_digest"
        ],
        "secondary_source_census_digest": record[
            "secondary_source_census_digest"
        ],
        "source_inventory_alignment_digest": record[
            "source_inventory_alignment_digest"
        ],
        "target_plan_canonicalization": record[
            "target_plan_canonicalization"
        ],
        "target_plan_canonicalization_digest": record[
            "target_plan_canonicalization_digest"
        ],
        "target_plan_digest": record["target_plan_digest"],
        "compiled_instruction_digest": record["compiled_instruction_digest"],
        "full_motion_contract_digest": record[
            "full_motion_contract_digest"
        ],
        "coverage_critic_digest": record["coverage_critic_digest"],
        "shard_index": record["shard_index"],
        "num_shards": record["num_shards"],
        "receipt_digest": receipt["receipt_digest"],
        "receipt_sha256": receipt["receipt_sha256"],
        "output_sha256": receipt["output_sha256"],
        "model_path": record["model_path"],
        "model_revision": record["model_revision"],
        "transformers_version": record["transformers_version"],
        # Preserve the complete already-validated Qwen record so downstream
        # release verification can independently recompute both the canonical
        # result digest and the full provenance digest instead of trusting the
        # two projected strings above.
        "qwen_record_payload": dict(record),
    }
    finalization = {
        "schema_version": FINALIZATION_ROW_SCHEMA,
        "policy_version": POLICY_VERSION,
        "candidate_rank": entry["candidate_rank"],
        "review_rank": entry["review_rank"],
        "selection_bucket": selection_bucket,
        "dynamic_unit_count": entry["dynamic_unit_count"],
        "target_action_signatures": list(entry["target_signatures"]),
        "family": candidate["family"],
        "required_canary": entry["iid"] in required_iids,
        "qwen_shard_index": record["shard_index"],
        "qwen_receipt_digest": receipt["receipt_digest"],
    }
    result = {
        "schema_version": GENERATION_SCHEMA,
        "iid": candidate["iid"],
        "group_id": candidate["group_id"],
        "family": candidate["family"],
        "source_video": candidate["src_video"],
        "resolved_source_video": record["resolved_src_video"],
        "anchor_image": candidate["anchor_image"],
        "resolved_anchor_image": record["resolved_anchor_image"],
        "source_video_sha256": candidate["source_video_sha256"],
        "anchor_sha256": candidate["anchor_sha256"],
        "selected_media_evidence": media,
        "selected_media_evidence_sha256": _object_digest(media),
        "strict_temporal_geometry": temporal,
        "edit_instruction": instruction,
        "edit_instruction_sha256": instruction_sha,
        "motion_spec": motion_spec,
        "motion_spec_sha256": _object_digest(motion_spec),
        "qwen_evidence": qwen_evidence,
        "full_motion_finalization": finalization,
        "action_change_substantive": "yes",
        "manifest_role": "review_proposal",
        "human_review_status": "pending",
        "generation_authorized": False,
        "production_eligible": False,
        "approval": None,
        "authorization_interface_available": False,
        "annotation_source": "qwen3-vl-32b",
        "human_reviewed": False,
    }
    return validate_generation_row(result)


def _implementation_bundle() -> dict[str, str]:
    directory = Path(__file__).resolve(strict=True).parent
    names = {
        "finalizer": "goku_full_motion_finalize.py",
        "qwen": "goku_full_motion_qwen.py",
        "contract": "goku_full_motion_contract.py",
        "instruction": "goku_full_motion_instruction.py",
        "postcheck": "goku_full_motion_postcheck.py",
    }
    return {
        key: _sha256(_stable_read(directory / name, context=f"{key} implementation"))
        for key, name in names.items()
    }


def _write_new(path: Path, raw: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_directory_noreplace(staging: Path, output: Path) -> None:
    source = os.fsencode(staging)
    destination = os.fsencode(output)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source, -100, destination, 1)
    elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source, destination, 0x00000004)
    else:  # pragma: no cover - supported production/test platforms use above
        raise GokuFullMotionFinalizeError(
            "platform lacks atomic no-replace directory rename"
        )
    if result == 0:
        return
    number = ctypes.get_errno()
    if number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(output)
    raise OSError(number, os.strerror(number), str(output))


def finalize_full_motion(
    *,
    candidate_manifest: str | Path,
    qwen_outputs: Sequence[str | Path],
    output_dir: str | Path,
    primary_size: int = DEFAULT_PRIMARY_SIZE,
    reserve_size: int = DEFAULT_RESERVE_SIZE,
    min_primary_multi_dynamic: int = DEFAULT_MIN_PRIMARY_MULTI_DYNAMIC,
    target_signature_cap: int = DEFAULT_TARGET_SIGNATURE_CAP,
    family_cap: int = DEFAULT_FAMILY_CAP,
    required_iids: Sequence[str] = (DEFAULT_CANARY_IID,),
) -> dict[str, Any]:
    """Validate, select, and atomically publish one pending full-motion pool."""

    for name, value, allow_zero in (
        ("primary_size", primary_size, False),
        ("reserve_size", reserve_size, True),
        ("min_primary_multi_dynamic", min_primary_multi_dynamic, True),
        ("target_signature_cap", target_signature_cap, False),
        ("family_cap", family_cap, False),
    ):
        if type(value) is not int or value < (0 if allow_zero else 1):
            raise GokuFullMotionFinalizeError(f"{name} is invalid")
    if min_primary_multi_dynamic > primary_size:
        raise GokuFullMotionFinalizeError(
            "min_primary_multi_dynamic exceeds primary_size"
        )
    required = list(required_iids)
    if len(set(required)) != len(required):
        raise GokuFullMotionFinalizeError("required_iids contains duplicates")
    for iid in required:
        _safe_iid(iid, context="required canary IID")

    candidate_unresolved = Path(
        os.path.abspath(Path(candidate_manifest).expanduser())
    )
    if candidate_unresolved.is_symlink() or not candidate_unresolved.is_file():
        raise GokuFullMotionFinalizeError(
            "candidate manifest must be a regular non-symlink file"
        )
    candidate_path = candidate_unresolved.resolve(strict=True)
    candidate_rows, candidate_raw, selected_by_iid = _validate_candidate_rows(
        candidate_path
    )
    output = Path(output_dir).expanduser().resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise GokuFullMotionFinalizeError("output parent is not a plain directory")

    api = _load_qwen_api()
    records_by_iid, receipt_by_iid, shard_artifacts = _load_qwen_shards(
        candidate_path=candidate_path,
        candidate_rows=candidate_rows,
        candidate_raw=candidate_raw,
        selected_by_iid=selected_by_iid,
        qwen_outputs=[Path(value) for value in qwen_outputs],
        api=api,
    )

    entries: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    for candidate_rank, candidate in enumerate(candidate_rows, start=1):
        iid = str(candidate["iid"])
        record = records_by_iid[iid]
        if record.get("status") != "ok":
            key = f"qwen_error:{record.get('failure_stage') or 'unknown'}"
            rejection_counts[key] = rejection_counts.get(key, 0) + 1
            continue
        if record.get("pipeline_decision") != "pass" or record.get(
            "hard_gate", {}
        ).get("decision") != "pass":
            rejection_counts["qwen_hard_gate_reject"] = (
                rejection_counts.get("qwen_hard_gate_reject", 0) + 1
            )
            continue
        signatures = _target_signatures(record)
        entries.append(
            {
                "iid": iid,
                "candidate": candidate,
                "record": record,
                "receipt": receipt_by_iid[iid],
                "candidate_rank": candidate_rank,
                "review_rank": len(entries) + 1,
                "dynamic_unit_count": _dynamic_count(record),
                "target_signatures": signatures,
                "signature_keys": list(
                    dict.fromkeys(value.casefold() for value in signatures)
                ),
                "family_key": str(candidate["family"]).casefold(),
            }
        )

    primary_entries, reserve_entries, diversity = _select_entries(
        entries,
        primary_size=primary_size,
        reserve_size=reserve_size,
        min_primary_multi_dynamic=min_primary_multi_dynamic,
        target_signature_cap=target_signature_cap,
        family_cap=family_cap,
        required_iids=required,
    )
    primary_iids = {str(entry["iid"]) for entry in primary_entries}
    reserve_iids = {str(entry["iid"]) for entry in reserve_entries}
    required_set = set(required)

    review_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    reserve_rows: list[dict[str, Any]] = []
    for entry in entries:
        iid = str(entry["iid"])
        bucket = (
            "primary"
            if iid in primary_iids
            else "reserve"
            if iid in reserve_iids
            else "review_only"
        )
        row = _generation_row(
            entry, selection_bucket=bucket, required_iids=required_set
        )
        review_rows.append(row)
        if bucket == "primary":
            primary_rows.append(row)
        elif bucket == "reserve":
            reserve_rows.append(row)

    primary_name = f"primary_{primary_size}.jsonl"
    reserve_name = f"reserve_{reserve_size}.jsonl"
    raw_outputs = {
        primary_name: _jsonl_bytes(primary_rows),
        reserve_name: _jsonl_bytes(reserve_rows),
        REVIEW_NAME: _jsonl_bytes(review_rows),
    }
    implementation = _implementation_bundle()
    implementation_digest = _object_digest(implementation)
    input_closure = {
        "candidate_manifest": {
            "path": str(candidate_path),
            "sha256": _sha256(candidate_raw),
            "bytes": len(candidate_raw),
            "rows": len(candidate_rows),
        },
        "qwen_shards": shard_artifacts,
    }
    input_digest = _object_digest(input_closure)
    artifact_metadata = {
        name: {
            "sha256": _sha256(raw),
            "bytes": len(raw),
            "rows": (
                len(primary_rows)
                if name == primary_name
                else len(reserve_rows)
                if name == reserve_name
                else len(review_rows)
            ),
        }
        for name, raw in raw_outputs.items()
    }
    summary = {
        "schema_version": FINALIZE_SCHEMA,
        "status": "complete",
        "policy_version": POLICY_VERSION,
        "semantics": {
            "all_source_dynamic_units_targeted": True,
            "camera_clause_explicit": True,
            "initial_frame_only_generation": True,
            "pseudo_label_only": True,
            "human_reviewed": False,
            "generation_authorized": False,
        },
        "config": {
            "qwen_shards": QWEN_SHARD_COUNT,
            "primary_size": primary_size,
            "reserve_size": reserve_size,
            "min_primary_multi_dynamic": min_primary_multi_dynamic,
            "target_signature_cap": target_signature_cap,
            "family_cap": family_cap,
            "required_iids": required,
            "selection_order": "candidate_manifest_order_with_multi_unit_quota_first",
            "caps_apply_to": "primary_plus_reserve",
        },
        "counts": {
            "candidate_rows": len(candidate_rows),
            "qwen_hard_pass_rows": len(entries),
            "qwen_rejected_rows": len(candidate_rows) - len(entries),
            "review_rows": len(review_rows),
            "primary_rows": len(primary_rows),
            "reserve_rows": len(reserve_rows),
            "primary_multi_dynamic_rows": diversity[
                "primary_multi_dynamic_rows"
            ],
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "diversity": diversity,
        "selection": {
            "primary_iids": [row["iid"] for row in primary_rows],
            "reserve_iids": [row["iid"] for row in reserve_rows],
            "required_iids_in_primary": [
                iid for iid in required if iid in primary_iids
            ],
        },
        "implementation": implementation,
        "implementation_digest": implementation_digest,
        "inputs": input_closure,
        "input_digest": input_digest,
        "artifacts": dict(artifact_metadata),
    }
    summary_raw = _pretty_bytes(summary)
    raw_outputs[SUMMARY_NAME] = summary_raw
    artifact_metadata[SUMMARY_NAME] = {
        "sha256": _sha256(summary_raw),
        "bytes": len(summary_raw),
        "rows": 1,
    }
    done_payload = {
        "schema_version": DONE_SCHEMA,
        "status": "complete",
        "policy_version": POLICY_VERSION,
        "implementation": implementation,
        "implementation_digest": implementation_digest,
        "inputs": input_closure,
        "input_digest": input_digest,
        "config": summary["config"],
        "artifacts": artifact_metadata,
        "artifact_digest": _object_digest(artifact_metadata),
    }
    done = dict(done_payload)
    done["done_digest"] = _object_digest(done_payload)
    done_raw = _pretty_bytes(done)

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
    )
    try:
        for name in (primary_name, reserve_name, REVIEW_NAME, SUMMARY_NAME):
            _write_new(staging / name, raw_outputs[name])
        _write_new(staging / DONE_NAME, done_raw)
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        _publish_directory_noreplace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def _outputs_from_dir(path: Path) -> list[Path]:
    root = path.expanduser().resolve(strict=True)
    if path.is_symlink() or not root.is_dir():
        raise GokuFullMotionFinalizeError("qwen_dir is not a plain directory")
    return [root / f"qwen_shard_{index:03d}.jsonl" for index in range(8)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-manifest", "--candidates", dest="candidate_manifest",
        required=True, type=Path
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--qwen-output", action="append", type=Path)
    source.add_argument("--qwen-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--primary-size", type=int, default=DEFAULT_PRIMARY_SIZE)
    parser.add_argument("--reserve-size", type=int, default=DEFAULT_RESERVE_SIZE)
    parser.add_argument(
        "--min-primary-multi-dynamic",
        type=int,
        default=DEFAULT_MIN_PRIMARY_MULTI_DYNAMIC,
    )
    parser.add_argument(
        "--target-signature-cap", type=int, default=DEFAULT_TARGET_SIGNATURE_CAP
    )
    parser.add_argument("--family-cap", type=int, default=DEFAULT_FAMILY_CAP)
    parser.add_argument("--require-iid", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = (
        args.qwen_output
        if args.qwen_output is not None
        else _outputs_from_dir(args.qwen_dir)
    )
    summary = finalize_full_motion(
        candidate_manifest=args.candidate_manifest,
        qwen_outputs=outputs,
        output_dir=args.output_dir,
        primary_size=args.primary_size,
        reserve_size=args.reserve_size,
        min_primary_multi_dynamic=args.min_primary_multi_dynamic,
        target_signature_cap=args.target_signature_cap,
        family_cap=args.family_cap,
        required_iids=[DEFAULT_CANARY_IID]
        + [iid for iid in args.require_iid if iid != DEFAULT_CANARY_IID],
    )
    print(
        "[goku-full-motion-finalize] "
        f"primary={summary['counts']['primary_rows']} "
        f"reserve={summary['counts']['reserve_rows']} "
        f"review={summary['counts']['review_rows']} "
        f"output={args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
